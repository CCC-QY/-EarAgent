#!/usr/bin/env python3
"""Download the two backbone models EarAgent needs (ModelScope).

  models/e2v_plus_base  -> iic/emotion2vec_plus_base   (frozen emotion encoder, ~90M)
  models/SenseVoiceSmall-> iic/SenseVoiceSmall         (ASR + rich transcript)
"""
from modelscope import snapshot_download
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pairs = [("iic/emotion2vec_plus_base", "models/e2v_plus_base"),
         ("iic/SenseVoiceSmall", "models/SenseVoiceSmall")]
for mid, local in pairs:
    dst = ROOT / local
    dst.mkdir(parents=True, exist_ok=True)
    print(f"downloading {mid} -> {dst}")
    snapshot_download(mid, local_dir=str(dst))
print("done")
