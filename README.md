<div align="center">

![FunSpeech](./docs/images/banner.png)

  <h3>开箱即用的本地私有化部署语音服务</h3>

基于 FunASR 和 CosyVoice 的语音处理 API 服务,提供语音识别(ASR)和语音合成(TTS)功能,与阿里云语音 API 完全兼容,且支持 Websocket 流式 ASR/TTS 协议。

---

![Static Badge](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Static Badge](https://img.shields.io/badge/Torch-2.3.1-%23EE4C2C?logo=pytorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/CUDA-12.1+-%2376B900?logo=nvidia&logoColor=white)

  <div style="margin: 30px 0;">
    <h3>强劲动力来自</h3>
    <a href="https://cnb.cool" target="_blank">
      <img src="https://docs.cnb.cool/images/logo/svg/LogoCnColorfulIcon.svg" alt="云原生构建" width="120" height="40">
    </a>
  </div>
</div>

## ✨ 主要特性

- **🚀 多模型支持** - 集成 FunASR、Dolphin、CosyVoice 等多种高质量模型
- **🌐 完全 API 兼容** - 支持阿里云语音 API 和 OpenAI TTS API 格式,及 Websocket 流式 ASR/TTS 协议
- **🎭 智能音色管理** - 支持预训练音色和零样本克隆音色
- **🧩 Voice Cloner 对接** - 提供音色设计、音色同步和实时 ASR->TTS 变声 WebSocket 接口
- **🕒 异步任务** - 支持长文本异步 TTS 和长录音异步 ASR,可轮询结果或配置回调通知
- **🔧 灵活配置** - 统一的配置系统,支持环境变量和文件配置
- **🛡️ 安全鉴权** - 完善的身份认证和权限控制
- **💾 性能优化** - 智能模型缓存和动态加载机制
- **🎯 智能过滤与控制** - 支持 ASR 热词/语气词过滤/情感标签,以及 TTS 音量、语调、情感控制

## 📦 快速部署

### Docker 部署(推荐)

```bash
# 下载配置文件
curl -sSL https://cnb.cool/nexa/FunSpeech/-/git/raw/main/docker-compose.yml -o docker-compose.yml

# 启动服务
docker-compose up -d
```

服务将在 `http://localhost:8000` 启动

**GPU 部署**请将 docker-compose.yml 文件中的 image 替换为 **docker.cnb.cool/nexa/funspeech:gpu-latest**

> 💡 详细部署说明(包括 CPU/GPU 版本区别、环境变量配置)请查看 [部署指南](./docs/deployment.md)

### 服务器本地构建/打包

在服务器源码目录中可直接用脚本构建镜像并导出 `tar.gz` 包:

```bash
# CPU 镜像: funspeech:latest -> dist/funspeech-latest.tar.gz
bash scripts/build-docker.sh cpu

# GPU 镜像: funspeech:gpu-latest -> dist/funspeech-gpu-latest.tar.gz
bash scripts/build-docker.sh gpu

# 同时构建 CPU + GPU
bash scripts/build-docker.sh all
```

常用参数:

```bash
# 指定镜像名/版本并推送到镜像仓库
bash scripts/build-docker.sh all --image docker.cnb.cool/nexa/funspeech --tag v1.0.0 --gpu-tag gpu-v1.0.0 --push

# 只构建镜像,不导出 tar.gz
bash scripts/build-docker.sh cpu --no-save
```

### 数据持久化

FunSpeech 会在以下目录存储持久化数据:

- **`./data`** - 数据库文件(异步 TTS 任务记录等)
- **`./temp`** - 临时文件(音频缓存等)
- **`./logs`** - 日志文件
- **`./voices`** - 零样本音色文件

Docker Compose 已自动配置数据卷映射,确保容器重启后数据不丢失。

对于要使用和下载的模型,您可以在运行中动态下载,也可以提前从 ModelScope 下载后映射,需要的模型在 [支持的模型](#-支持的模型),同时注意提前规划好存储空间以免存储空间不足无法下载～

### 并发配置

FunSpeech 支持多路并发处理,通过以下环境变量配置:

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `WORKERS` | `1` | Worker进程数,每个进程加载独立模型(内存×N) |
| `INFERENCE_THREAD_POOL_SIZE` | `auto` | 推理线程池大小,确保事件循环不阻塞 |
| `TTS_GPUS` | `""` | TTS GPU配置: `""` (自动), `cpu`, `0` (单卡), `0,1` (多卡) |
| `ASR_GPUS` | `""` | ASR GPU配置: `""` (自动), `cpu`, `0` (单卡), `0,1` (多卡) |
| `REALTIME_TTS_GLOBAL_MAX_INFLIGHT` | `2` | Realtime Voice 全局同时合成任务数 |
| `REALTIME_TTS_GLOBAL_QUEUE_SIZE` | `16` | Realtime Voice 全局 TTS 排队长度 |
| `REALTIME_AUDIO_INPUT_MAX_MS` | `1800` | Realtime Voice 单会话音频输入缓冲上限 |

**配置示例:**

```yaml
# docker-compose.yml
environment:
  - WORKERS=2                        # 2个Worker进程
  - INFERENCE_THREAD_POOL_SIZE=4     # 每个Worker 4个推理线程
  - TTS_GPUS=0,1                     # TTS在GPU 0和1上各创建副本
  - ASR_GPUS=0,1                     # ASR在GPU 0和1上各创建副本
```

> 💡 详细配置说明和资源规划请查看 [部署指南 - 并发配置](./docs/deployment.md#并发配置)

### 本地开发

**系统要求:**

- Python 3.10+
- CUDA 12.1+(可选,用于 GPU 加速)
- FFmpeg(音频格式转换)

**安装步骤:**

```bash
# 克隆项目
cd FunSpeech
git submodule update --init --recursive

# 安装依赖
pip install -r dependencies/requirements.txt

# 根据运行环境安装 CosyVoice 依赖
pip install -r dependencies/CosyVoice/requirements-cpu.txt
# 或 GPU 环境:
# pip install -r dependencies/CosyVoice/requirements-gpu.txt

# 启动服务
python main.py
```

## 🛠️ 脚本工具

脚本依赖可按需安装:

```bash
pip install numpy matplotlib soundfile websockets tqdm
```

### Docker 构建脚本

`scripts/build-docker.sh` 用于在服务器源码目录构建 CPU/GPU 镜像并可选导出离线包:

```bash
bash scripts/build-docker.sh cpu
bash scripts/build-docker.sh gpu
bash scripts/build-docker.sh all --image docker.cnb.cool/nexa/funspeech --tag v1.0.0 --gpu-tag gpu-v1.0.0 --push
```

### RMS 音频分析

`scripts/analyze_audio_rms.py` 用于分析录音 RMS 能量时序,辅助调优流式 ASR 远场过滤阈值:

```bash
# 基础分析
python scripts/analyze_audio_rms.py audio.wav

# 指定声道、阈值并保存图表
python scripts/analyze_audio_rms.py recording.wav \
  --channel right \
  --threshold 0.015 \
  --output rms_analysis.png

# 仅输出统计信息
python scripts/analyze_audio_rms.py recording.wav --no-plot
```

常用参数: `--channel stereo|left|right`、`--threshold 0.01`、`--chunk-size 240`、`--output`、`--no-plot`。详细说明见 [scripts/README.md](./scripts/README.md)。

### ASR/TTS 并发基准测试

`scripts.benchmark.run` 用于压测 `/ws/v1/asr` 和 `/ws/v1/tts` 的并发性能,生成 Markdown 报告和图表:

```bash
# 完整测试(ASR + TTS)
python -m scripts.benchmark.run --audio-file test.wav

# 仅测试 TTS
python -m scripts.benchmark.run --test-type tts --voice 中文女

# 自定义并发级别和远程地址
python -m scripts.benchmark.run \
  --host 192.168.1.100 \
  --port 8000 \
  --audio-file test.wav \
  --concurrency 5 10 20 50 100
```

常用参数: `--host`、`--port`、`--audio-file`、`--test-type asr|tts|both`、`--concurrency`、`--output`、`--timeout`、`--voice`。详细说明见 [benchmark 文档](./scripts/benchmark/README.md)。

### Realtime Voice 压测

`scripts/benchmark/realtime_voice_pressure.py` 用于压测 `/ws/v1/realtime/voice`,输入本地 WAV 文件并输出每个并发档位的延迟、事件数、背压统计:

```bash
python scripts/benchmark/realtime_voice_pressure.py \
  --base-url http://localhost:8000 \
  --audio test.wav \
  --voice desktop_voice \
  --levels 1,2,4 \
  --duration 12 \
  --chunk-ms 100 \
  --output benchmark_results/realtime_voice.json
```

输入音频会被脚本转换为 16kHz PCM16 mono。常用参数: `--base-url`、`--audio`、`--voice`、`--levels`、`--duration`、`--chunk-ms`、`--realtime-factor`、`--drain-seconds`、`--timeout`、`--output`。

## 📚 API 接口

### ASR(语音识别)

| 端点                          | 方法      | 功能                             |
| ----------------------------- | --------- | -------------------------------- |
| `/stream/v1/asr`              | POST      | 一句话语音识别                   |
| **`/rest/v1/asr/async`**      | **POST**  | **提交长录音异步识别任务** 🆕    |
| **`/rest/v1/asr/async`**      | **GET**   | **查询长录音异步识别结果** 🆕    |
| **`/ws/v1/asr`**              | WebSocket | **双向流式语音识别**             |
| `/ws/v1/asr/test`             | GET       | WebSocket ASR 测试页面           |
| `/stream/v1/asr/models`       | GET       | 模型列表                         |
| `/stream/v1/asr/health`       | GET       | 健康检查                         |

**完整接口文档:**

- 一句话 ASR：[阿里云一句话语音识别 API](https://help.aliyun.com/zh/isi/developer-reference/restful-api-2)
- 流式 ASR：[Websocket 协议说明](https://help.aliyun.com/zh/isi/developer-reference/websocket)

**特殊说明:**

- 一句话识别限制音频时长 60 秒
- 一句话和异步 ASR 均支持 `hotwords` 临时热词、`vocabulary_id` 热词表、`disfluency` 语气词过滤
- SenseVoice 可通过 `enable_emotion=true` 返回情感标签,通过 `return_rich_text=true` 返回模型原始 rich transcription 文本
- 异步 ASR 支持 `audio_address`(HTTP/HTTPS) 或 `audio_bytes`(0-255 整数数组)作为输入,长录音默认启用 VAD

**流式ASR高级功能:**

- **远场声音过滤** 🆕 - 自动过滤远场声音和环境音，减少误触发
  - 基于RMS能量阈值检测
  - 零性能开销（<0.1ms），完全可配置
  - 默认启用，详见 [远场过滤文档](./docs/nearfield_filter.md)
- **Voice Cloner 字段** - WebSocket 识别结果会携带 `task_id`、`text`、`is_final`、`confidence`、`duration_ms` 等稳定字段,便于桌面端消费

### TTS(语音合成)

| 端点                            | 方法      | 功能                        |
| ------------------------------- | --------- | --------------------------- |
| `/stream/v1/tts`                | POST      | 语音合成                    |
| `/openai/v1/audio/speech`       | POST      | OpenAI 兼容接口             |
| **`/rest/v1/tts/async`**        | **POST**  | **提交异步语音合成任务** 🚀 |
| **`/rest/v1/tts/async`**        | **GET**   | **查询异步语音合成结果** 🚀 |
| `/stream/v1/tts/voices`         | GET       | 音色列表                    |
| `/stream/v1/tts/voices/info`    | GET       | 音色详细信息                |
| `/stream/v1/tts/voices/refresh` | POST      | 刷新音色配置                |
| **`/stream/v1/tts/emotions`**   | **GET**   | **情感控制标签列表** 🆕     |
| `/stream/v1/tts/health`         | GET       | 健康检查                    |
| **`/ws/v1/tts`**                | WebSocket | **双向流式语音合成** 🚀     |
| `/ws/v1/tts/test`               | GET       | WebSocket 测试页面          |

**完整接口文档:**

- 基础 TTS: [语音合成 RESTful API](https://help.aliyun.com/zh/isi/developer-reference/restful-api-3)
- 流式 TTS: [Websocket 协议说明](https://help.aliyun.com/zh/isi/developer-reference/websocket-protocol-description)
- 异步 TTS: [阿里云异步长文本语音合成 RESTful API](https://help.aliyun.com/zh/isi/developer-reference/restful-api)

**特殊说明:**

- 合成传入采样率中，CosyVoice1 采样率固定（默认）为 22050，CosyVoice2 采样率固定（默认）为 24000
- `/stream/v1/tts` 支持 `volume`(0-100)、`pitch_rate`(-500~500)、`prompt`、`emotion`、`emotion_intensity` 等控制参数
- `/openai/v1/audio/speech` 兼容 OpenAI 请求格式,`instructions` 会映射为本项目的音色指导 `prompt`

### Voice Cloner / 实时变声

| 端点                              | 方法      | 功能                                      |
| --------------------------------- | --------- | ----------------------------------------- |
| **`/voices/v1/voice-design`**     | **POST**  | **根据音色设计指令生成参考音频** 🆕       |
| **`/voices/v1/list`**             | **GET**   | **首次启动全量同步 voice_manager 音色**   |
| **`/voices/v1/register`**         | **POST**  | **增量注册桌面端自定义音色**              |
| **`/voices/v1/update`**           | **POST**  | **增量更新桌面端自定义音色**              |
| **`/voices/v1/delete`**           | **POST**  | **增量删除桌面端自定义音色**              |
| **`/voices/v1/refresh`**          | **POST**  | **重新扫描并加载 voice_manager 音色**     |
| **`/ws/v1/realtime/voice`**       | WebSocket | **实时 ASR->TTS 变声会话** 🆕             |

**对接约定:**

- 桌面端以唯一 `voice_name` 作为音色关联键;本项目不引入 `voice_id`
- `/voices/v1/voice-design` 默认不静默降级,需要设置 `VOICE_DESIGN_PROVIDER=module.submodule:function` 注入 VoxCPM 参考音频生成函数
- `register` / `update` 支持 JSON 的 `reference_audio_url`,也支持 multipart 的 `reference_audio` 文件上传
- 预置音色名称不可被注册、覆盖或删除;找不到 `voice_name` 时会返回明确错误
- Realtime Voice 连接后先收 `session_started`,再发送 `configure`,后续可发送 PCM 二进制音频块和 `update` 参数事件

## 🎯 快速开始

**ASR 语音识别:**

```bash
curl -X POST "http://localhost:8000/stream/v1/asr?format=wav&sample_rate=16000" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

**WebSocket 流式识别测试:** 访问 `http://localhost:8000/ws/v1/asr/test`

**长录音异步识别:**

```bash
# 提交任务
curl -X POST "http://localhost:8000/rest/v1/asr/async" \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "asr_request": {
        "audio_address": "https://example.com/long.wav",
        "format": "wav",
        "sample_rate": 16000,
        "hotwords": "FunSpeech SenseVoice",
        "disfluency": true
      },
      "enable_notify": false
    },
    "header": {"appkey": "your_appkey", "token": "your_token"}
  }'

# 查询结果
curl "http://localhost:8000/rest/v1/asr/async?appkey=your_appkey&token=your_token&task_id=<task_id>"
```

**TTS 语音合成:**

```bash
curl -X POST "http://localhost:8000/stream/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是语音合成测试。",
    "voice": "中文女",
    "volume": 60,
    "pitch_rate": 0,
    "emotion": "happy",
    "emotion_intensity": 0.8
  }' \
  --output speech.wav
```

**WebSocket 流式合成测试:** 访问 `http://localhost:8000/ws/v1/tts/test`

**异步 TTS:**

```bash
# 提交长文本合成任务
curl -X POST "http://localhost:8000/rest/v1/tts/async" \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "tts_request": {
        "text": "这是一段需要异步合成的长文本。",
        "voice": "中文女",
        "format": "wav",
        "sample_rate": 22050,
        "enable_subtitle": true
      },
      "enable_notify": false
    },
    "header": {"appkey": "your_appkey", "token": "your_token"}
  }'

# 查询结果
curl "http://localhost:8000/rest/v1/tts/async?appkey=your_appkey&token=your_token&task_id=<task_id>"
```

**音色设计与同步:**

```bash
# 音色设计接口需要先注入 VoxCPM 生成函数
export VOICE_DESIGN_PROVIDER="your_module.voice_design:generate_reference_audio"

# 生成参考音频
curl -X POST "http://localhost:8000/voices/v1/voice-design" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_name": "desktop_voice",
    "voice_instruction": "清亮、少年感、自然口语",
    "reference_text": "今天的天气很好。",
    "format": "wav",
    "sample_rate": 24000
  }'

# 注册桌面端自定义音色
curl -X POST "http://localhost:8000/voices/v1/register" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_name": "desktop_voice",
    "reference_text": "今天的天气很好。",
    "voice_instruction": "清亮、少年感、自然口语",
    "reference_audio_url": "https://example.com/desktop_voice.wav"
  }'
```

**Realtime Voice WebSocket 协议示例:**

```json
{"event":"configure","voice_name":"desktop_voice","format":"pcm","sample_rate":16000,"pipeline":"asr_tts","parameters":{"volume":50,"speech_rate":0,"pitch_rate":0}}
{"event":"update","parameters":{"pitch_rate":120,"emotion_control":"asr"}}
```

发送 `configure` 后即可持续发送 16kHz PCM16 mono 二进制音频块;服务端会返回 `asr.hypothesis`、`asr.text_committed`、`tts.first_audio`、音频二进制帧、`session_completed` 等事件。

> 💡 更多示例请查看 `tests/` 目录或访问 `http://localhost:8000/docs`(开发模式)

## 🎵 音色系统

### 智能音色列表

音色列表 API (`/stream/v1/tts/voices`) 会根据当前的模型模式智能返回对应的音色:

- **sft 模式**: 仅返回预设音色列表(7 个)
- **clone 模式**: 仅返回零样本克隆音色列表(允许为空)
- **all 模式**: 返回所有音色列表(预设+零样本克隆)

### 预训练音色

- **中文女** - 温柔甜美的女性音色
- **中文男** - 深沉稳重的男性音色
- **英文女** - 清晰自然的英文女性音色
- **英文男** - 低沉磁性的英文男性音色
- **日语男** - 标准的日语男性音色
- **韩语女** - 清新可爱的韩语女性音色
- **粤语女** - 地道的粤语女性音色

### 零样本克隆音色

**准备音色文件:**

克隆音色需要准备一对文件:

- **音频文件** (`*.wav`): 3-30 秒,清晰无噪音,建议 16kHz+ 采样率
- **文本文件** (`*.txt`): 音频对应的文字内容,需完全匹配

文件命名必须一致,例如: `张三.wav` 和 `张三.txt`

**添加新音色:**

```bash
# 1. 将音频和文本文件放入 voices 目录
mkdir -p ./voices
cp 张三.wav 张三.txt ./voices/

# 2. 运行音色管理工具添加
python -m app.services.tts.clone.voice_manager --add

# 3. 验证音色
curl "http://localhost:8000/stream/v1/tts/voices"
```

**音色管理命令:**

```bash
python -m app.services.tts.clone.voice_manager --list           # 列出所有音色
python -m app.services.tts.clone.voice_manager --remove <名称>  # 删除音色
python -m app.services.tts.clone.voice_manager --info <名称>    # 查看音色信息
python -m app.services.tts.clone.voice_manager --refresh        # 刷新音色列表
```

**使用克隆音色:**

```bash
curl -X POST "http://localhost:8000/stream/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是使用克隆音色的测试。",
    "voice": "张三"
  }' \
  --output cloned_voice.wav
```

**音色指导功能:**

对于零样本克隆音色,可以使用 `prompt` 参数进行音色指导:

```bash
curl -X POST "http://localhost:8000/stream/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "欢迎使用语音服务",
    "voice": "张三",
    "prompt": "说话温柔一些，像客服一样亲切"
  }' \
  --output guided_voice.wav
```

> ⚠️ 注意: 音色指导功能仅适用于零样本克隆音色(CosyVoice2 模型)

## 🤖 支持的模型

服务会在首次使用时自动从 ModelScope 下载模型,也可以提前手动下载以加快启动速度。

### TTS 模型 (语音合成)

通过环境变量 `TTS_MODEL_MODE` 控制加载模式。

| 模型名称               | 加载模式             | 大小  | 说明                              | ModelScope 链接                                         |
| ---------------------- | -------------------- | ----- | --------------------------------- | ------------------------------------------------------- |
| **CosyVoice-300M-SFT** | `sft` / `all` | 5.4GB | 预训练音色模型,支持 7 种预设音色  | https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT |
| **CosyVoice2-0.5B**    | `clone` / `all` | 5.5GB | 零样本克隆模型,支持音色克隆和指导 | https://www.modelscope.cn/models/iic/CosyVoice2-0.5B    |
| **Fun-CosyVoice3-0.5B** | `clone` / `all` | 5.5GB | CosyVoice3 零样本克隆模型 | https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 |

**模式说明:**

- `TTS_MODEL_MODE=sft` - 仅加载预设音色模型 (~5.4GB)
- `TTS_MODEL_MODE=clone` - 仅加载音色克隆模型 (~5.5GB)
- `TTS_MODEL_MODE=all` - 加载全部模型 (~11GB,默认)

**克隆模型版本选择:**

通过 `CLONE_MODEL_VERSION` 环境变量选择克隆模型版本:
- `CLONE_MODEL_VERSION=cosyvoice3` - 使用 Fun-CosyVoice3-0.5B-2512 (默认)
- `CLONE_MODEL_VERSION=cosyvoice2` - 使用 CosyVoice2-0.5B

### ASR 模型 (语音识别)

通过环境变量 `ASR_MODEL_MODE` 控制加载模式。

| 模型名称                    | 加载模式           | 大小  | 说明                               | ModelScope 链接                                                                                         |
| --------------------------- | ------------------ | ----- | ---------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **SenseVoice Small**        | `offline` / `realtime` / `all` | 897MB | 默认模型；离线识别 + VAD 驱动窗口化伪流式实时识别 | https://www.modelscope.cn/models/iic/SenseVoiceSmall                                                    |
| **Paraformer Large (离线)** | 按需加载  | 848MB | 兼容的中文离线识别模型        | https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch |
| **Paraformer Large (流式)** | 按需加载 | 848MB | 兼容的中文实时流式识别             | https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online  |
| **Dolphin Small**           | 按需加载           | 600MB | 轻量级多语言识别模型               | https://www.modelscope.cn/models/DataoceanAI/dolphin-small                                              |

**模式说明:**

- `ASR_MODEL_MODE=realtime` - 默认加载 SenseVoiceSmall 离线模型，并通过 `ASR_STREAMING_STRATEGY=windowed_offline` 提供 WebSocket 实时协议
- `ASR_MODEL_MODE=offline` - 仅加载离线模型 (~897MB,默认 SenseVoiceSmall)
- `ASR_MODEL_MODE=all` - 加载默认模型可用形态；SenseVoiceSmall 没有独立 realtime 权重

**自定义模型预加载:**

默认情况下，SenseVoiceSmall 会在启动时自动加载。如果需要在启动时预加载其他模型（如 Paraformer、Dolphin），可以使用 `AUTO_LOAD_CUSTOM_ASR_MODELS` 环境变量：

```bash
# 预加载单个自定义模型
export AUTO_LOAD_CUSTOM_ASR_MODELS="paraformer-large"

# 预加载多个自定义模型（逗号分隔）
export AUTO_LOAD_CUSTOM_ASR_MODELS="sensevoice-small,dolphin-small"
```

这样在启动时就会自动下载并加载指定的模型，避免首次调用时的等待时间。模型配置详见 `app/services/asr/models.json`。

### 辅助模型

| 模型名称             | 类型       | 大小  | 说明                                                                    | ModelScope 链接                                                                                |
| -------------------- | ---------- | ----- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **PUNC Transformer** | 标点预测   | 283MB | 为离线识别结果添加标点符号                                              | https://www.modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch      |
| **PUNC Realtime**    | 实时标点   | 279MB | 为实时识别中间结果添加标点(可选,需设置 `ASR_ENABLE_REALTIME_PUNC=true`) | https://www.modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727 |
| **FSMN VAD**         | 语音检测   | 3.9MB | 检测语音片段,过滤静音和噪音                                             | https://www.modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch                  |
| **CAM++ Speaker**    | 说话人识别 | 28MB  | 说话人特征提取(未启用)                                                  | https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common                       |

### 提前下载模型

**安装 ModelScope CLI:**

```bash
pip install modelscope
```

**下载 TTS 模型:**

```bash
# 预设音色模型 (TTS_MODEL_MODE=sft 或 all)
modelscope download --model iic/CosyVoice-300M-SFT

# 音色克隆模型 - CosyVoice3 (CLONE_MODEL_VERSION=cosyvoice3,默认)
modelscope download --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512

# 音色克隆模型 - CosyVoice2 (CLONE_MODEL_VERSION=cosyvoice2)
modelscope download --model iic/CosyVoice2-0.5B
```

**下载 ASR 模型:**

```bash
# 默认离线/实时协议模型
modelscope download --model iic/SenseVoiceSmall

# 可选兼容模型
modelscope download --model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
modelscope download --model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
```

**下载辅助模型(按需):**

```bash
# 标点预测模型(离线识别使用)
modelscope download --model iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch

# 实时标点模型(实时识别使用,可选)
modelscope download --model iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727

# VAD 模型
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
```

> 💡 **提示**: 模型默认下载到 `~/.cache/modelscope/hub`,Docker 部署时需映射此目录以复用模型文件。

### 存储空间规划

根据使用场景规划所需存储空间:

| 场景           | 环境变量配置                                             | 所需模型                   | 总大小 |
| -------------- | -------------------------------------------------------- | -------------------------- | ------ |
| **最小部署**   | `TTS_MODEL_MODE=sft`<br>`ASR_MODEL_MODE=offline`  | 1 个 TTS + SenseVoiceSmall + 辅助 | ~7GB   |
| **实时流式**   | `TTS_MODEL_MODE=sft`<br>`ASR_MODEL_MODE=realtime` | 1 个 TTS + SenseVoiceSmall + FSMN VAD + 辅助 | ~7GB   |
| **完整 TTS**   | `TTS_MODEL_MODE=all`<br>`ASR_MODEL_MODE=offline`         | 2 个 TTS + 离线 ASR + 辅助 | ~12GB  |
| **全功能部署** | `TTS_MODEL_MODE=all`<br>`ASR_MODEL_MODE=all`             | 全部模型                   | ~14GB  |

### API 文档

- **开发模式**: 访问 `http://localhost:8000/docs` 查看完整 API 文档
- **生产模式**: API 文档自动隐藏

## 🌐 相关链接

- **部署指南**: [详细文档](./docs/deployment.md)
- **远场过滤配置**: [配置指南](./docs/nearfield_filter.md)
- **Voice Cloner 对接说明**: [一期功能增补清单](./docs/voice-cloner-api-additions.md)
- **CosyVoice 模型**: [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- **Dolphin 模型**: [DataoceanAI/Dolphin](https://github.com/DataoceanAI/Dolphin)
- **FunASR**: [FunASR GitHub](https://github.com/alibaba-damo-academy/FunASR)

## 📋 TODO

- [x] 实现 ASR 热词功能 (`vocabulary_id` / `hotwords`)
- [x] 实现过滤语气词功能 (`disfluency`)
- [x] 实现 TTS 语调控制 (`pitch_rate`)
- [x] 实现长录音文件异步识别接口
- [x] 实现 Voice Cloner 音色设计、音色同步与 Realtime Voice 对接接口

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目!
