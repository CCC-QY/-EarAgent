# DSH 音频感知插件（MCP）

把本地「语音转写 + 情绪识别」做成一个 **MCP server**（[audio_mcp_server.py](../audio_mcp_server.py)），任何支持 MCP 的 agent（包括 dsh，通过 `@deepseek-ai/dsh-mcp-client`）都能像调用原生工具一样调用。这是"通用型"接口：dsh、Claude Desktop、Cursor 等都能直接接。

## 提供的工具

| 工具名（dsh 里为 `mcp__audio__<name>`） | 作用 | 输入 |
| --- | --- | --- |
| `analyze_audio` | 一次返回：转写文本、语言、情绪、基础信号统计（文件或目录） | `{path, emotion_scheme?="5class"|"multilabel", emotion_threshold?}` |
| `transcribe_audio` | 只用 SenseVoiceSmall 做语音转写 | `{path}` |
| `detect_emotion` | 只做情绪（5class 或 multilabel） | `{path, emotion_scheme?, emotion_threshold?}` |
| `analyze_batch` | 对目录下所有音频批量分析 | `{path, emotion_scheme?, limit?}` |

**两种情绪方案：**
- `5class`（默认）：angry / happy / neutral / sad / surprise，单标签 + 置信度。
- `multilabel`：CNSCED 语义，7 维（上面 5 类 + **fear** + **aroused 唤醒**），可多标签同时命中，阈值按验证集校准。

**音频格式**：`wav/flac/mp3/m4a/mp4/ogg/opus/aiff` 等都支持——非 16k 单声道 wav 会经 ffmpeg 自动转成 16k 单声道再分析（无需手动转换）。

### analyze_audio 返回示例（中文音频）

```json
{
  "file": "Data/chiSER-5/angry/angry_001.wav",
  "transcript": "我跟住所检察官反映了他们不管啊他们说我变态还说我什么妄想症难道说蔡成功",
  "language": "zh",
  "emotion": "sad",
  "confidence": 0.972,
  "label": "sad",
  "probabilities": {"angry": 0.0135, "happy": 0.0002, "neutral": 0.0136, "sad": 0.9722, "surprise": 0.0005},
  "quality": {"duration_s": 12.841, "sample_rate": 16000, "rms_dbfs": -24.08, "clipped_sample_ratio": 0.0},
  "sensevoice_emotion_tag": "angry"
}
```

- `label`：最高概率低于 `emotion_threshold`（默认 0.4）时为 `uncertain`。
- `emotion`/`probabilities`：声学情绪估计（偏自信、未校准），不代表真实心理状态。
- `sensevoice_emotion_tag`：SenseVoice 自带的情绪标签，可作为交叉参考。
- `scheme`：返回里会标明当前用的是 `5class` 还是 `multilabel`。multilabel 返回 `labels`（命中的标签列表）+ `probabilities`（7 维）+ `thresholds`。

## 在 dsh 中启用

dsh 通过 `@deepseek-ai/dsh-mcp-client` 连接 MCP server，每个 server 一条 profile patch。用安装脚本（幂等、自动备份、YAML 校验）：

```bash
# 默认安装到 desktop profile（GUI 正在用的那个）
.venv/bin/python dsh_plugin/install_to_dsh.py --profile desktop
# 若你的 GUI 跑的是 web profile：
.venv/bin/python dsh_plugin/install_to_dsh.py --profile web
```

它会往 `~/.dsh/profiles/<profile>/cordis.patch.yml` 追加：

```yaml
- id: dsh-audio-perception
  name: "@deepseek-ai/dsh-mcp-client"
  config:
    serverName: audio
    transport: stdio
    command: /Users/table/DSH_Audio_Perception/.venv/bin/python
    args: ["/Users/table/DSH_Audio_Perception/audio_mcp_server.py"]
    cwd: /Users/table/DSH_Audio_Perception
```

> 安装脚本写的是工作区之外的 `~/.dsh`，需要沙箱授权；安装后**重启 dsh app** 才会加载。

启用后 agent 会看到工具：`mcp__audio__analyze_audio`、`mcp__audio__transcribe_audio`、`mcp__audio__detect_emotion`。

## 不接 dsh 也能用

任何 MCP 客户端都可直接以 stdio 启动：

```bash
.venv/bin/python audio_mcp_server.py
```

或命令行单文件推理（只做情绪）：`.venv/bin/python inference.py audio.wav`。

## 依赖

- `models/SenseVoiceSmall/`（语音转写，MIT/模型许可）
- `models/e2v_plus_base/`（冻结情绪骨干，~90M）
- `distillation_experiment/e2vbase_head.pt`（蒸馏头 + μ/σ）
- `.venv` 需含 `funasr torch transformers soundfile librosa numpy`

模型在**首次工具调用时懒加载**（约 5–8 秒），之后单条 ~60–200ms（取决于时长）。

## 已知限制

- 情绪模型跨语料（chiSER 真实中文）约 56–59%；训练用的 ESD 是英语说话人（0011–0020），其 86% 不能代表中文；ESD 普通话重跑虽 99.7% 但疑似训练污染。自然口语会更低。
- 概率未校准，偏自信；必要时用 `emotion_threshold` 兜底为 `uncertain`。
- 情绪标签是"声学表达估计"，不得作为对说话人心理状态的事实判断。
