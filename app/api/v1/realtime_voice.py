# -*- coding: utf-8 -*-
"""Voice Cloner实时变声WebSocket接口。"""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...core.config import settings
from ...utils.common import generate_task_id
from ...services.tts.engine import get_tts_engine


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws/v1/realtime", tags=["Realtime Voice"])


def _voice_available(voice_name: str) -> bool:
    if voice_name in settings.PRESET_VOICES:
        return True
    tts_engine = get_tts_engine()
    voices = tts_engine.get_voices() if hasattr(tts_engine, "get_voices") else []
    return voice_name in voices


async def _send_error(websocket: WebSocket, task_id: str, message: str):
    await websocket.send_json(
        {
            "event": "error",
            "task_id": task_id,
            "status": 40000003,
            "message": message,
        }
    )


@router.websocket("/voice")
async def realtime_voice_endpoint(websocket: WebSocket):
    """实时变声会话。

    当前落地稳定WebSocket会话、运行中参数更新和连续音频流返回；实际变声DSP/模型可在
    transform_audio_chunk处替换，默认保持字节流透传，避免伪造不可用的模型能力。
    """
    await websocket.accept()
    task_id = generate_task_id("realtime_voice")
    voice_name = ""
    parameters = {}

    await websocket.send_json(
        {
            "event": "session_started",
            "task_id": task_id,
            "status": 20000000,
            "audio_mode": "passthrough",
        }
    )

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                if not voice_name:
                    await _send_error(websocket, task_id, "请先发送configure事件设置voice_name")
                    continue
                await websocket.send_bytes(transform_audio_chunk(message["bytes"], voice_name, parameters))
                continue

            if "text" not in message or message["text"] is None:
                continue

            try:
                data = json.loads(message["text"])
            except json.JSONDecodeError:
                await _send_error(websocket, task_id, "消息必须是JSON")
                continue

            event = data.get("event")
            if event in {"configure", "switch_voice"}:
                next_voice = (data.get("voice_name") or "").strip()
                if not next_voice:
                    await _send_error(websocket, task_id, "voice_name不能为空")
                    continue
                if not _voice_available(next_voice):
                    await _send_error(websocket, task_id, f"voice_name不存在: {next_voice}")
                    continue

                voice_name = next_voice
                parameters.update(data.get("parameters") or {})
                await websocket.send_json(
                    {
                        "event": "configured" if event == "configure" else "voice_switched",
                        "task_id": task_id,
                        "voice_name": voice_name,
                        "format": data.get("format", "pcm"),
                        "sample_rate": data.get("sample_rate", 16000),
                        "status": 20000000,
                    }
                )
            elif event == "update":
                parameters.update(data.get("parameters") or {})
                await websocket.send_json(
                    {
                        "event": "parameters_updated",
                        "task_id": task_id,
                        "voice_name": voice_name,
                        "parameters": parameters,
                        "status": 20000000,
                    }
                )
            elif event in {"close", "stop"}:
                await websocket.send_json(
                    {"event": "session_completed", "task_id": task_id, "status": 20000000}
                )
                break
            else:
                await _send_error(websocket, task_id, f"不支持的事件: {event}")
    except WebSocketDisconnect:
        logger.info("[%s] 实时变声客户端断开", task_id)
    except Exception as exc:
        logger.error("[%s] 实时变声处理异常: %s", task_id, exc)
        try:
            await _send_error(websocket, task_id, f"实时变声处理失败: {exc}")
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def transform_audio_chunk(audio: bytes, voice_name: str, parameters: dict) -> bytes:
    """实时音频转换钩子。

    仓库当前没有独立voice-conversion模型，默认透传以固定协议和背压路径。
    """
    return audio
