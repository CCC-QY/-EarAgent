#!/usr/bin/env python3
"""One-shot speech emotion inference: audio -> emotion2vec_base embedding -> distilled head -> 5-class probabilities.

Pipeline (matches the distillation training exactly):
  1. emotion2vec_base (frozen) utterance embedding (768-d) via FunASR generate(extract_embedding=True)
  2. standardize with the head's train mean/std
  3. MLP head (768 -> 256 -> 5) -> softmax over [angry, happy, neutral, sad, surprise]

Usage:
  python inference.py path/to/audio.wav
  python inference.py path/to/folder --jsonl out.jsonl
  python inference.py a.wav b.wav --threshold 0.4
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf

ROOT = Path(__file__).resolve().parent
CLASSES = ["angry", "happy", "neutral", "sad", "surprise"]
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 5)
        )

    def forward(self, x):
        return self.net(x)


def load_head(path: Path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    head = MLPHead()
    head.load_state_dict(ckpt["state_dict"])
    head.eval().to(device)
    mu = np.asarray(ckpt["mu"], dtype=np.float32)
    sd = np.asarray(ckpt["sd"], dtype=np.float32)
    return head, mu, sd


def collect_inputs(items):
    files = []
    for item in items:
        p = Path(item).expanduser()
        if p.is_dir():
            files += sorted([q for q in p.rglob("*") if q.suffix.lower() in AUDIO_EXTS])
        elif p.is_file():
            files.append(p)
        else:
            print(f"[skip] not found: {p}", file=sys.stderr)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="audio file(s) or directory(ies)")
    ap.add_argument("--backbone", default=str(ROOT / "models/e2v_plus_base"), help="emotion2vec_base model dir")
    ap.add_argument("--head", default=str(ROOT / "checkpoints/e2vbase_head.pt"), help="distilled head checkpoint")
    ap.add_argument("--threshold", type=float, default=0.40,
                    help="below this top-probability the label is reported as 'uncertain'")
    ap.add_argument("--int8", action="store_true", help="use the int8-quantized backbone (slower on Apple Silicon; smaller)")
    ap.add_argument("--jsonl", default="", help="optional path to also write one JSON per line")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    files = collect_inputs(args.inputs)
    if not files:
        print("error: no audio files found", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    head, mu, sd = load_head(Path(args.head), device)

    from funasr import AutoModel
    with contextlib.redirect_stdout(sys.stderr):  # keep stdout clean for JSON results
        am = AutoModel(model=str(Path(args.backbone).resolve()), disable_update=True, device=args.device)
    if args.int8:
        torch.backends.quantized.engine = "qnnpack"
        q = torch.ao.quantization.quantize_dynamic(am.model, {nn.Linear}, dtype=torch.qint8).eval()
        qpath = ROOT / "checkpoints/e2vbase_int8_state.pt"
        if qpath.exists():
            q.load_state_dict(torch.load(qpath, map_location="cpu", weights_only=False))
        am.model = q

    writer = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None
    try:
        for f in files:
            rec = {"file": str(f)}
            try:
                info = sf.info(str(f))
                rec["duration_s"] = round(info.duration, 3)
                rec["sample_rate"] = info.samplerate
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(sys.stderr):
                    out = am.generate(str(f), granularity="utterance", extract_embedding=True)
                emb = np.asarray(out[0]["feats"], dtype=np.float32)
                z = (emb - mu) / (sd + 1e-5)
                with torch.no_grad():
                    logits = head(torch.from_numpy(z).unsqueeze(0).to(device))
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                rec["processing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                top = int(np.argmax(probs))
                rec["emotion"] = CLASSES[top]
                rec["confidence"] = round(float(probs[top]), 4)
                rec["label"] = CLASSES[top] if probs[top] >= args.threshold else "uncertain"
                rec["probabilities"] = {c: round(float(p), 4) for c, p in zip(CLASSES, probs)}
                rec["model"] = {"backbone": "emotion2vec_base", "quantized": bool(args.int8),
                                "head": Path(args.head).name, "version": "distilled-1"}
                rec["note"] = "Acoustic emotion estimate, not a fact about the speaker's mental state."
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            line = json.dumps(rec, ensure_ascii=False)
            print(line)
            if writer:
                writer.write(line + "\n")
    finally:
        if writer:
            writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
