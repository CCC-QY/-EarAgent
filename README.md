# EarAgent

**Give your agent ears** — speech transcription + emotion recognition exposed as a local **MCP server**.

EarAgent turns any audio file into structured, machine-readable perception:

- **Speech content** — transcript + language (SenseVoiceSmall, multilingual incl. Chinese).
- **Emotion (acoustic)** — either a simple 5-class label or a richer CNSCED multi-label scheme with arousal and fear.
- **Signal stats** — duration, sample rate, RMS/loudness, clipping.

It speaks the **Model Context Protocol (MCP)** over stdio, so it plugs into any MCP-capable agent — [the agent harness (dsh)](https://github.com/deepseek-ai/deepseek-harness), [AstrBot](https://docs.astrbot.app/use/mcp.html), SnowLuma, Claude Desktop, etc.

```
audio ──> SenseVoiceSmall ──> transcript / language
      └─> emotion2vec_base (frozen) ──> head ──> 5-class / multi-label emotion
                          └── exposed as MCP tools ──> your agent
```

## Features

- **Two emotion schemes**
  - `5class`: `angry / happy / neutral / sad / surprise` + confidence + `uncertain` fallback.
  - `multilabel` (CNSCED): `angry, happy, neutral, sad, surprise, fear, aroused` — multi-label, per-class thresholds calibrated on a held-out speaker split.
- **Any common audio format** — `wav/flac/mp3/m4a/mp4/ogg/opus/aiff`; non-16k-mono is auto-transcoded with `ffmpeg`.
- **File or directory** — single file, or batch a whole folder.
- **Composable tools** — transcribe only, emotion only, or full analysis.
- **CPU friendly** — no GPU required (frozen ~90M encoder + tiny MLP head).

## Requirements

- Python 3.10–3.12
- `ffmpeg` on PATH (for non-wav inputs)
- ~2–3 GB RAM at runtime (both backbones loaded)
- The two backbone models (downloaded separately, see below)

## Install on Windows (i7 / CPU)

See [`docs/WINDOWS.md`](docs/WINDOWS.md). Quick version:

```powershell
winget install --id Gyan.FFmpeg -e
git clone https://github.com/CCC-QY/-EarAgent.git
Set-Location -Path '.\-EarAgent'
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\download_models.py
python dsh_plugin\selftest.py C:\path\to\audio.wav
```

## Install

```bash
git clone https://github.com/CCC-QY/-EarAgent.git
cd -- -EarAgent 2>/dev/null || cd EarAgent
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py        # -> models/e2v_plus_base, models/SenseVoiceSmall
```

`models/` is git-ignored; only the small head checkpoints under `checkpoints/` ship with the repo.

## Quick test

```bash
python dsh_plugin/selftest.py path/to/audio.wav
# tools: ['analyze_audio','transcribe_audio','detect_emotion','analyze_batch']
# 5class     -> "..." | surprise
# multilabel -> "..." | ['happy','surprise','aroused']
```

Or plain Python (emotion only):

```bash
python inference.py path/to/audio.wav
```

## Tools

| tool (`mcp__audio__*` in dsh) | description | arguments |
| --- | --- | --- |
| `analyze_audio` | transcript + language + emotion + signal stats | `path`, `emotion_scheme?="5class"\|"multilabel"`, `emotion_threshold?` |
| `transcribe_audio` | transcript + language only | `path` |
| `detect_emotion` | emotion only | `path`, `emotion_scheme?`, `emotion_threshold?` |
| `analyze_batch` | analyze every audio file in a directory | `path`, `emotion_scheme?`, `limit?` |

Example result:

```json
{
  "file": "sample.wav",
  "transcript": "哇你的牌打的也忒好了吧",
  "language": "zh",
  "scheme": "5class",
  "emotion": "surprise",
  "confidence": 0.9964,
  "probabilities": {"angry":0.0,"happy":0.0035,"neutral":0.0001,"sad":0.0,"surprise":0.9964},
  "quality": {"duration_s":4.587,"sample_rate":16000,"rms_dbfs":-26.02},
  "sensevoice_emotion_tag": "neutral"
}
```

## Integrate with an agent

### dsh / the agent harness

```bash
python dsh_plugin/install_to_dsh.py --profile desktop
# then restart dsh; tools appear as mcp__audio__analyze_audio etc.
```

This appends a `@deepseek-ai/dsh-mcp-client` entry to `~/.dsh/profiles/<profile>/cordis.patch.yml` (with a timestamped backup). See `dsh_plugin/README.md`.

### AstrBot / SnowLuma / other hosts

Run the server as a stdio MCP server and register it in the host's MCP config; or spawn `python audio_mcp_server.py` directly:

```bash
python audio_mcp_server.py
```

> **QQ voice note:** QQ voice messages are Tencent **SILK** (often `\x02`-prefixed), which plain `ffmpeg` cannot decode. A SILK→wav pre-step is required on the host side (or a future EarAgent codec hook). This is currently a known limitation.

## Models & credits

| component | model | license |
| --- | --- | --- |
| ASR / transcript | SenseVoiceSmall (FunAudioLLM / FunASR) | see upstream (model license) |
| Emotion encoder | emotion2vec_base (FunAudioLLM) | see upstream (model license) |
| Emotion heads | distilled MLP heads (this repo) | MIT |

Training pipeline (distillation from `emotion2vec+ large` into a small head over `emotion2vec_base`, and CNSCED multi-label adaptation) is described in `docs/DESIGN.md`.

## Limitations & disclaimer

- Emotion output is an **acoustic estimate of the way something is said**, not a fact about a person's feelings, mental state, or health. Do not use it for diagnosis, employment, or any high-stakes decision.
- Real-world Chinese accuracy is still modest (5-class held-out ≈ 60%; multi-label macro-F1 ≈ 0.54). `surprise` and `angry` are the weakest classes; probabilities are **not calibrated** and tend to be overconfident.
- Trained/evaluated on acted speech corpora; natural, noisy, multi-speaker conversation is harder.

## Roadmap

- [ ] SILK/AMR decoding for QQ voice
- [ ] `streamable-http` transport for remote hosts
- [ ] more real-world Chinese data + calibration
- [ ] multi-label intensities (weak/medium/strong)

## License

Code: MIT (see `LICENSE`). Model weights: follow the upstream model licenses.
