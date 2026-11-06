# Running EarAgent on Windows (i7, CPU)

EarAgent is pure Python + PyTorch/ONNX-free at runtime; it runs fine on Windows CPU.
The repo is platform-neutral (paths use pathlib); the only OS-specific piece is `ffmpeg`.

## 1. Prerequisites

- Python **3.11 or 3.12** (64-bit) — https://www.python.org/downloads/windows/
  - During install tick **"Add python.exe to PATH"**.
- **FFmpeg** (needed to decode wav/m4a/mp3/… to 16 kHz mono):
  ```powershell
  winget install --id Gyan.FFmpeg -e
  # or: choco install ffmpeg
  ```
  Open a new terminal and check `ffmpeg -version`.
- (Optional) NVIDIA GPU: install CUDA build of torch for faster inference.

## 2. Clone + install

```powershell
git clone https://github.com/CCC-QY/-EarAgent.git
Set-Location -Path '.\-EarAgent'      # repo name starts with '-' ; use -Path
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## 3. Download the backbone models (~1.9 GB)

```powershell
python scripts\download_models.py
```

This downloads to `models\e2v_plus_base` and `models\SenseVoiceSmall`.
If ModelScope is slow/unavailable, use the Hugging Face mirrors instead:

```powershell
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download as d; d('emotion2vec/emotion2vec_plus_base', local_dir='models/e2v_plus_base')"
python -c "from huggingface_hub import snapshot_download as d; d('FunAudioLLM/SenseVoiceSmall', local_dir='models/SenseVoiceSmall')"
```
(SenseVoice from HF needs `funasr` to load a local dir; keep the same layout.)

## 4. Test

```powershell
python dsh_plugin\selftest.py C:\path\to\audio.wav
python inference.py C:\path\to\audio.wav
```

## Notes / known Windows quirks

- **Paths**: all handled by `pathlib`. Quote paths with spaces.
- **ffmpeg detection**: resolved from `PATH` first, then `C:\ffmpeg\bin\ffmpeg.exe`.
- **Long paths**: enable `git config --system core.longpaths true` if clone hits MAX_PATH.
- **Multiprocessing/spawn**: funasr/torch on Windows use `spawn`; run from a normal console (not Jupyter) first.
- **Quantization**: the shipped compact option is PyTorch dynamic int8. On x86 it is generally *faster* than on Apple Silicon (fbgemm backend). `inference.py --int8` needs `checkpoints/e2vbase_int8_state.pt` (not bundled; generate from the fp32 model if needed).
- **dsh profile installer** (`dsh_plugin/install_to_dsh.py`) targets `~/.dsh/profiles/...` — only relevant if you run the dsh app on this Windows box. For a Windows-only dev/test machine you don't need it; just run `audio_mcp_server.py` as an MCP server or the selftest.

## Performance (approx, CPU)

- First call loads both models: ~5–10 s (lazy, once per process).
- Per clip: ~0.1–0.5 s depending on length (i7 desktop class similar to M-series laptop here).
- RAM: ~2–3 GB with both backbones loaded.
