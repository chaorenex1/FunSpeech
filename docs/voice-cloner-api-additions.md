# Voice Cloner 对 FunSpeech 的一期功能增补清单

## 1. 当前边界

结合 `voice-cloner` 当前架构，一期边界确定为：

- ASR 云端项目：`FunSpeech`
- TTS 云端项目：`FunSpeech`
- LLM 服务：桌面端直接调用本地模型服务
- 实时变声 backend：`FunSpeech`
- 自定义音色注册表：放在桌面端本地
- 音色管理需要与 `FunSpeech voice_manager` 同步
- `FunSpeech` 与桌面端通过“唯一音色名称 `voice_name`”关联

因此，`FunSpeech` 一期负责：

- REST / WebSocket ASR
- TTS
- Realtime Voice
- 音色设计接口
- `voice_manager`
- 基于唯一音色名称消费桌面端定义的音色资产
- 首次启动全量同步
- 后续新增 / 修改 / 删除增量同步

`FunSpeech` 一期不负责：

- 统一音色注册表 API
- `voice_id` 体系
- 能力发现接口
- 自定义音色 preview 接口
- 桌面端本地 LLM 调度

## 2. 当前已有能力

从当前仓库可以确认，`FunSpeech` 已有这些能力：

- REST ASR：`app/api/v1/asr.py`
- WebSocket ASR：`app/api/v1/websocket_asr.py`
- REST TTS：`app/api/v1/tts.py`
- OpenAI 兼容 TTS：`app/api/v1/openai.py`
- WebSocket TTS：`app/api/v1/websocket_tts.py`
- 零样本音色管理脚本：`app/services/tts/clone/voice_manager.py`

这意味着 `FunSpeech` 已具备 ASR / TTS 基础与 `voice_manager` 运行时基础，但还缺少“面向 Voice Cloner 的 Realtime Voice、音色设计接口，以及明确的全量/增量同步约定”。

## 3. 和 Voice Cloner 的对接关系

桌面端音色设计流程应为：

1. 用户在桌面端录音或输入文本描述
2. 若输入为语音，调用 `FunSpeech ASR`
3. 桌面端调用本地 LLM 服务，生成：
   - `voice_instruction`
   - `reference_text`
   - `voice_name`
4. 桌面端调用 `FunSpeech` 音色设计接口
5. `FunSpeech` 使用 `VoxCPM` 基于 `voice_instruction` 生成 `reference_audio`
6. 桌面端把：
   - `voice_name`
   - `reference_text`
   - `voice_instruction`
   - `reference_audio`
   保存到本地音色注册表
7. 首次启动时，桌面端从 `FunSpeech voice_manager` 全量同步音色
8. 后续新增 / 修改 / 删除时，桌面端与 `voice_manager` 做增量同步
9. 桌面端在后续 TTS / Realtime Voice 调用中，用唯一 `voice_name` 与 `FunSpeech` 关联

关键结论：

- `FunSpeech` 不负责 LLM 指令生成
- `FunSpeech` 不作为音色注册表权威
- 桌面端本地音色注册表才是 source of truth
- `voice_manager` 是 `FunSpeech` 里的运行时音色源

## 4. 必须新增的功能

### 4.1 新增 WebSocket ASR 对接约定

虽然仓库里已经有 `app/api/v1/websocket_asr.py`，但对 `Voice Cloner` 来说，一期需要把它作为正式对接面固定下来。

建议对接接口：

- `WS /ws/v1/asr`

一期要求：

- 支持桌面端持续发送 PCM 音频块
- 支持中间结果与最终结果区分
- 返回稳定结构化字段，至少包括：
  - `task_id`
  - `text`
  - `is_final`
  - `confidence`（若可提供）
  - `duration_ms`（若可提供）

### 4.2 新增音色设计接口

这是本轮新增的核心接口。

接口目标：

- 由桌面端提供 LLM 生成的音色设计指令
- 由 `FunSpeech` 内部使用 `VoxCPM` 生成参考音频

建议接口：

- `POST /voices/v1/voice-design`

请求字段建议：

- `voice_name`
- `voice_instruction`
- `reference_text`
- `format`
- `sample_rate`

返回字段建议：

- `voice_name`
- `reference_audio_url` 或二进制音频结果
- `reference_text`
- `status`
- `task_id`

说明：

- `voice_instruction` 由桌面端本地 LLM 生成
- `FunSpeech` 负责消费该指令并调用 `VoxCPM`
- `reference_text` 仍由桌面端传入，避免把文本生成责任重新塞回 `FunSpeech`

### 4.3 新增 Realtime Voice 接口

这是实时变声主链路，现阶段应直接放在 `FunSpeech` 内，而不是独立服务。

建议接口：

- `WS /ws/v1/realtime/voice`

一期要求：

- 建立实时变声会话
- 接收连续音频输入流
- 返回连续变声音频输出流
- 支持运行中参数更新
- 支持运行中角色切换

### 4.4 TTS / Realtime 按唯一音色名称消费音色

由于你明确不要 `voice_id`，一期应继续按 `voice` / `voice_name` 对接。

建议约束：

- 桌面端负责保证 `voice_name` 全局唯一
- `FunSpeech` 按 `voice_name` 查找可用音色
- 当 `voice_name` 找不到时，返回明确错误

需要加强的地方：

- `POST /stream/v1/tts`
- `POST /openai/v1/audio/speech`
- `WS /ws/v1/realtime/voice`

建议补充能力：

- 明确区分预置音色与桌面端注册音色
- 保证 `voice_name` 冲突时返回稳定错误码

### 4.5 voice_manager 全量 / 增量同步接口

由于音色注册表在桌面端，本期 `FunSpeech` 不需要提供统一的“列表 / 详情 / 删除”注册表 API。

但它仍需要一组围绕 `voice_manager` 的同步接口。

首次启动全量同步：

- `GET /voices/v1/list`

后续增量同步：

- `POST /voices/v1/register`
- `POST /voices/v1/update`
- `POST /voices/v1/delete`
- `POST /voices/v1/refresh`

请求字段建议：

- `voice_name`
- `reference_text`
- `reference_audio`（multipart file 或 URL）
- `voice_instruction`（可选，供调试或追踪）

说明：

- `list` 用于桌面端首次启动时从 `voice_manager` 拉取全量音色
- `register` 用于新增音色时写入 `voice_manager`
- `update` 用于修改音色时覆盖 `voice_manager` 中的同名音色
- `delete` 用于删除音色时移除 `voice_manager` 中的同名音色
- `refresh` 用于服务重启或音色变更后重新加载
- 这不是统一音色注册表 API，只是运行时装载接口

### 4.6 ASR 响应结构增强

为了让桌面端本地 LLM 更容易消费 ASR 结果，建议保持以下字段稳定：

- `task_id`
- `text`
- `language`（若可提供）
- `duration_ms`（若可提供）
- `confidence`（若可提供）

## 5. 明确删除的方向

以下方向按你的要求不进入一期：

- `voice_id`
- `GET /stream/v1/capabilities`
- 自定义音色 preview 接口
- 统一音色注册表 API
- `FunSpeech` 作为音色注册表 source of truth

## 6. 建议接口清单

### 已有可复用

- `POST /stream/v1/asr`
- `WS /ws/v1/asr`
- `POST /stream/v1/tts`
- `POST /openai/v1/audio/speech`
- `WS /ws/v1/tts`

### 建议新增

- `WS /ws/v1/realtime/voice`
- `POST /voices/v1/voice-design`
- `GET /voices/v1/list`
- `POST /voices/v1/register`
- `POST /voices/v1/update`
- `POST /voices/v1/delete`
- `POST /voices/v1/refresh`

## 7. 一期实施优先级

### P0

- WebSocket ASR 对接结构稳定化
- Realtime Voice 接口 `WS /ws/v1/realtime/voice`
- 音色设计接口 `POST /voices/v1/voice-design`
- 首次全量同步接口 `GET /voices/v1/list`
- 按唯一音色名称装载音色的 `POST /voices/v1/register`

### P1

- `POST /voices/v1/update`
- `POST /voices/v1/delete`
- `POST /voices/v1/refresh`
- TTS 的 `voice_name` 冲突与缺失错误标准化
- ASR 响应结构增强

### P2

- 更好的音色装载诊断信息
- 更完整的音色工件清理策略

## 8. 一句话结论

对 `Voice Cloner` 来说，`FunSpeech` 一期应进化成：

- `ASR/TTS 云端能力提供者`
- `Realtime Voice 执行器`
- `VoxCPM 驱动的音色设计执行器`
- `带 voice_manager 全量/增量同步的运行时服务`

而不是：

- `统一音色注册表`
- `桌面端 LLM 编排器`
- `独立于主语音服务之外的第二套实时后端`
