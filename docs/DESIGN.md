# Design notes

## Pipeline

1. **Audio in** — any format; transcoded to 16 kHz mono via `ffmpeg`.
2. **Transcript** — SenseVoiceSmall (FunASR) returns rich text with language + event/emotion tags; EarAgent strips tags and keeps `text`, `language`, and `sensevoice_emotion_tag` as a cross-check.
3. **Emotion** — `emotion2vec_base` (frozen, ~90M) produces a 768-d utterance embedding; a small MLP head maps it to class probabilities.
   - 5-class head: `angry/happy/neutral/sad/surprise`, softmax, `uncertain` below threshold.
   - Multi-label head (CNSCED): `angry/happy/neutral/sad/surprise/fear/aroused`, sigmoid + per-class calibrated thresholds.
4. **MCP tools** — the above exposed over stdio JSON-RPC.

## How the heads were made

- Teacher: `emotion2vec+ large` (~300M, 9-class).
- Student head: MLP over **frozen `emotion2vec_base`** embeddings (768→256→N).
- Distillation: CE on hard labels + KL on teacher soft labels (T=2).
- 5-class head is a general baseline; the multi-label head was adapted on the CNSCED corpus with a **speaker-disjoint** train/val/test split, and thresholds calibrated on the val speakers.

## Quantization

- Dynamic int8 (PyTorch `quantize_dynamic`) of the encoder: ~376 MB → ~113 MB, accuracy essentially unchanged. This is the recommended compact option.
- Static int8 (ONNX QDQ PTQ) collapsed accuracy; QAT could not be converted cleanly (qnnpack lacks Conv1d / quantized layer_norm on this stack). Not shipped.

## Known weaknesses

- `surprise` and `angry` recall are low; probabilities are overconfident → use thresholds / `uncertain`.
- CNSCED is multi-dimensional (emotion + arousal + intensity). The shipped multi-label head predicts presence (binary), not intensity.
- ESD/CNSCED are acted corpora; natural conversation performance is lower.
