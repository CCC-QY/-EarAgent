#!/usr/bin/env python3
"""DSH audio-perception MCP server (stdio) — v0.2.

Exposes speech tools to any MCP-capable agent (dsh via @deepseek-ai/dsh-mcp-client):
  - analyze_audio        : transcript + language + emotion (5-class OR CNSCED multi-label) + signal stats
  - transcribe_audio     : transcript + language (SenseVoiceSmall)
  - detect_emotion       : emotion only (5-class OR multi-label)
  - analyze_batch        : run analyze over a directory of audio files

Handles common audio formats (wav/m4a/mp3/flac/ogg/…) by transcoding to 16k mono via ffmpeg.
Protocol: MCP JSON-RPC 2.0 over stdio, one JSON message per line; all logs go to stderr.
"""
from __future__ import annotations

import contextlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CLASSES5 = ["angry", "happy", "neutral", "sad", "surprise"]
LABELS7 = ["angry", "happy", "neutral", "sad", "surprise", "fear", "aroused"]
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".ogg", ".oga", ".opus", ".aac", ".aiff", ".aif"}
EMOTION_TOKEN = re.compile(r"<\|(ANGRY|HAPPY|SAD|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED|EMO_UNK)\|>")
LANGUAGE_TOKEN = re.compile(r"<\|(zh|en|ja|ko|yue|nospeech|unk)\|>")
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

_m = {"asr": None, "emo": None, "head5": None, "mu5": None, "sd5": None,
      "head7": None, "mu7": None, "sd7": None, "thr7": None, "loaded": False}


def log(msg: str):
    print(f"[audio-mcp] {msg}", file=sys.stderr, flush=True)


def _load_models():
    if _m["loaded"]:
        return
    import torch
    import torch.nn as nn

    with contextlib.redirect_stdout(sys.stderr):
        from funasr import AutoModel
        _m["asr"] = AutoModel(model=str(ROOT / "models/SenseVoiceSmall"), device="cpu",
                              disable_update=True, trust_remote_code=True)
        _m["emo"] = AutoModel(model=str(ROOT / "models/e2v_plus_base"), device="cpu", disable_update=True)

    class Head(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, n))

        def forward(self, x):
            return self.net(x)

    def _load_ckpt(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            import pickle
            return pickle.load(open(path, "rb"))

    def load_head(path, n):
        ck = _load_ckpt(path)
        h = Head(n)
        h.load_state_dict(ck["state_dict"])
        h.eval()
        return h, np.asarray(ck["mu"], np.float32), np.asarray(ck["sd"], np.float32)

    _m["head5"], _m["mu5"], _m["sd5"] = load_head(ROOT / "checkpoints/e2vbase_head.pt", 5)
    h7path = ROOT / "checkpoints/cnscd_multilabel_head.pt"
    th7 = ROOT / "config/cnscd_thresholds.json"
    if h7path.exists():
        _m["head7"], _m["mu7"], _m["sd7"] = load_head(h7path, 7)
        _m["thr7"] = json.load(open(th7, encoding="utf-8"))["thresholds"] if th7.exists() else {l: 0.5 for l in LABELS7}
    _m["loaded"] = True
    log("models loaded")


def _prepare_wav(path: Path) -> tuple[Path, Path | None]:
    """Return (wav_path, temp_dir_or_None); transcode non-16k-mono/wav via ffmpeg."""
    import soundfile as sf
    if path.suffix.lower() == ".wav":
        try:
            info = sf.info(str(path))
            if info.samplerate == 16000 and info.channels == 1:
                return path, None
        except Exception:
            pass
    if not Path(FFMPEG).exists():
        raise RuntimeError("ffmpeg not found; cannot transcode non-16k wav")
    tmpdir = Path(tempfile.mkdtemp(prefix="audio-mcp-"))
    tmp = tmpdir / "in16k.wav"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(path),
                    "-ar", "16000", "-ac", "1", str(tmp)], check=True, capture_output=True)
    return tmp, tmpdir


def _transcribe(wav: Path) -> dict:
    with contextlib.redirect_stdout(sys.stderr):
        raw = _m["asr"].generate(input=str(wav), disable_pbar=True)
    item = raw[0] if raw else {}
    tagged = str(item.get("text", ""))
    emo = EMOTION_TOKEN.search(tagged)
    lang = LANGUAGE_TOKEN.search(tagged)
    clean = re.sub(r"<\|[^|]+\|>", "", tagged).strip()
    return {"text": clean or None, "language": lang.group(1) if lang else None,
            "sensevoice_emotion_tag": (emo.group(1).lower() if emo else None)}


def _embedding(wav: Path) -> np.ndarray:
    with contextlib.redirect_stdout(sys.stderr):
        out = _m["emo"].generate(str(wav), granularity="utterance", extract_embedding=True)
    return np.asarray(out[0]["feats"], np.float32)


def _emotion(wav: Path, scheme: str, threshold: float) -> dict:
    import torch
    emb = _embedding(wav)
    if scheme == "multilabel":
        if _m["head7"] is None:
            raise RuntimeError("multilabel head not available")
        z = (emb - _m["mu7"]) / (_m["sd7"] + 1e-5)
        with torch.no_grad():
            probs = torch.sigmoid(_m["head7"](torch.from_numpy(z).unsqueeze(0)))[0].numpy()
        thr = _m["thr7"]
        hits = [l for l, p in zip(LABELS7, probs) if p >= thr.get(l, 0.5)]
        return {"scheme": "multilabel", "labels": hits,
                "probabilities": {l: round(float(p), 4) for l, p in zip(LABELS7, probs)},
                "thresholds": thr}
    z = (emb - _m["mu5"]) / (_m["sd5"] + 1e-5)
    with torch.no_grad():
        probs = torch.softmax(_m["head5"](torch.from_numpy(z).unsqueeze(0)), dim=1)[0].numpy()
    top = int(np.argmax(probs))
    return {"scheme": "5class", "emotion": CLASSES5[top], "confidence": round(float(probs[top]), 4),
            "label": CLASSES5[top] if float(probs[top]) >= threshold else "uncertain",
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES5, probs)}}


def _quality(wav: Path) -> dict:
    import soundfile as sf
    a, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(a, dtype=np.float64)))) if len(a) else 0.0
    return {"duration_s": round(len(a) / sr if sr else 0.0, 3), "sample_rate": sr,
            "rms_dbfs": round(20 * math.log10(max(rms, 1e-10)), 2),
            "clipped_sample_ratio": round(float(np.mean(np.abs(a) >= 0.999)) if len(a) else 0.0, 4)}


def _analyze_one(path: Path, scheme: str, threshold: float, want_transcript: bool) -> dict:
    _load_models()
    wav, tmpdir = _prepare_wav(path)
    t0 = time.perf_counter()
    try:
        rec = {"file": str(path)}
        if want_transcript:
            tr = _transcribe(wav)
            rec["transcript"] = tr["text"]
            rec["language"] = tr["language"]
            rec["sensevoice_emotion_tag"] = tr["sensevoice_emotion_tag"]
        rec.update(_emotion(wav, scheme, threshold))
        rec["quality"] = _quality(wav)
        rec["processing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        rec["note"] = "Acoustic emotion estimate, not a fact about the speaker's mental state."
        return rec
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _expand(path: Path):
    if path.is_dir():
        return sorted([p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXTS])
    return [path]


TOOLS = [
    {"name": "analyze_audio",
     "description": "Analyze one local audio file or a directory: spoken transcript, language, and emotion. emotion_scheme='5class' (angry/happy/neutral/sad/surprise) or 'multilabel' (CNSCED: +fear/aroused, multi-label).",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "File or directory path."},
         "emotion_scheme": {"type": "string", "enum": ["5class", "multilabel"], "default": "5class"},
         "emotion_threshold": {"type": "number", "default": 0.4, "description": "5-class uncertain threshold."}},
         "required": ["path"]}},
    {"name": "transcribe_audio",
     "description": "Transcribe one audio file (or directory) to text + language using SenseVoiceSmall.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "detect_emotion",
     "description": "Detect emotion only (5class or multilabel scheme) for one file or directory.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string"},
         "emotion_scheme": {"type": "string", "enum": ["5class", "multilabel"], "default": "5class"},
         "emotion_threshold": {"type": "number", "default": 0.4}},
         "required": ["path"]}},
    {"name": "analyze_batch",
     "description": "Run analyze_audio over every audio file in a directory; returns a list of results.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string"},
         "emotion_scheme": {"type": "string", "enum": ["5class", "multilabel"], "default": "5class"},
         "limit": {"type": "integer", "default": 0, "description": "Max files; 0 = all."}},
         "required": ["path"]}},
]


def call_tool(name: str, args: dict) -> dict:
    if name == "transcribe_audio":
        p = Path(args["path"]).expanduser()
        files = _expand(p)
        if not files:
            raise FileNotFoundError(f"no audio found: {p}")
        out = []
        for f in files:
            if not f.is_file():
                raise FileNotFoundError(f"not a file: {f}")
            wav, tmpdir = _prepare_wav(f)
            try:
                out.append({"file": str(f), **_transcribe(wav)})
            finally:
                if tmpdir is not None:
                    shutil.rmtree(tmpdir, ignore_errors=True)
        return out[0] if p.is_file() else {"count": len(out), "results": out}

    if name == "detect_emotion":
        p = Path(args["path"]).expanduser()
        scheme = args.get("emotion_scheme", "5class")
        thr = float(args.get("emotion_threshold", 0.4))
        files = _expand(p)
        if not files:
            raise FileNotFoundError(f"no audio found: {p}")
        out = [_analyze_one(f, scheme, thr, want_transcript=False) for f in files]
        return out[0] if p.is_file() else {"count": len(out), "results": out}

    if name in ("analyze_audio", "analyze_batch"):
        p = Path(args["path"]).expanduser()
        scheme = args.get("emotion_scheme", "5class")
        thr = float(args.get("emotion_threshold", 0.4))
        files = _expand(p)
        if not files:
            raise FileNotFoundError(f"no audio found: {p}")
        if name == "analyze_batch" or p.is_dir():
            lim = int(args.get("limit", 0))
            if lim:
                files = files[:lim]
            results = [_analyze_one(f, scheme, thr, True) for f in files]
            return {"count": len(results), "results": results}
        return _analyze_one(p, scheme, thr, True)

    raise ValueError(f"unknown tool: {name}")


def handle(msg: dict):
    method = msg.get("method"); mid = msg.get("id", None); params = msg.get("params") or {}
    if method == "initialize":
        return {"protocolVersion": params.get("protocolVersion") or "2024-11-05",
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "dsh-audio-perception", "version": "0.2.0"}}
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        try:
            result = call_tool(params.get("name"), params.get("arguments") or {})
            return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}], "isError": False}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
    if method == "resources/list":
        return {"resources": []}
    if method == "prompts/list":
        return {"prompts": []}
    if mid is None:
        return None
    raise KeyError(f"method not found: {method}")


def main():
    log("audio-mcp v0.2 started; waiting for MCP messages on stdio")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            log(f"bad json: {e}")
            continue
        mid = msg.get("id", None)
        try:
            result = handle(msg)
        except KeyError as e:
            if mid is not None:
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": str(e)}}) + "\n"); sys.stdout.flush()
            continue
        except Exception as e:
            if mid is not None:
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}}) + "\n"); sys.stdout.flush()
            continue
        if mid is None:
            continue
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": result}, ensure_ascii=False) + "\n"); sys.stdout.flush()


if __name__ == "__main__":
    main()
