# -*- coding: utf-8 -*-
"""Voice Cloner实时变声WebSocket接口。"""

import json
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...core.config import settings
from ...utils.common import convert_speech_rate_to_speed
from ...utils.common import generate_task_id
from ...services.tts.engine import get_tts_engine
from ...services.websocket_asr import get_aliyun_websocket_asr_service
from ...services.websocket_tts import get_aliyun_websocket_tts_service


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws/v1/realtime", tags=["Realtime Voice"])


async def _send_error(websocket: WebSocket, task_id: str, message: str):
    await websocket.send_json(
        {
            "event": "error",
            "task_id": task_id,
            "status": 40000003,
            "message": message,
        }
    )


class RealtimeVoiceAsrTtsSession:
    """Internal realtime voice chain: incoming audio -> ASR text -> TTS audio."""

    def __init__(
        self,
        voice_name: str,
        audio_format: str = "pcm",
        sample_rate: int = 16000,
        parameters: dict | None = None,
    ):
        self.voice_name = voice_name
        self.audio_format = (audio_format or "pcm").lower()
        self.sample_rate = int(sample_rate or 16000)
        self.parameters = parameters or {}
        self.asr_service = get_aliyun_websocket_asr_service()
        self.tts_service = get_aliyun_websocket_tts_service()
        self.asr_params = {
            "format": "pcm",
            "sample_rate": self.sample_rate,
            "enable_punctuation_prediction": True,
            "enable_inverse_text_normalization": True,
            "enable_voice_detection": True,
        }
        asr_engine = self.asr_service._ensure_asr_engine()
        if hasattr(asr_engine, "create_streaming_session"):
            self.streaming_session = asr_engine.create_streaming_session(self.asr_params)
        else:
            self.streaming_session = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_cache = {}
        self.punc_cache = {}
        self.audio_time = 0
        self.last_text = ""

    async def process_audio(self, websocket: WebSocket, task_id: str, audio: bytes) -> bool:
        """Process one audio chunk and emit ASR/TTS events.

        Returns True when a TTS audio chunk was emitted. Short ASR buffers may return
        False because FunASR needs at least a standard streaming window.
        """
        await websocket.send_json(
            {"event": "pipeline_stage", "stage": "asr_receiving_audio", "task_id": task_id}
        )
        text = await self._transcribe(audio, task_id)
        if not text:
            return False

        await websocket.send_json(
            {
                "event": "asr_result",
                "stage": "asr_text_received",
                "task_id": task_id,
                "text": text,
                "is_final": False,
            }
        )
        if text == self.last_text:
            return False
        self.last_text = text

        await websocket.send_json(
            {
                "event": "pipeline_stage",
                "stage": "tts_synthesizing",
                "task_id": task_id,
                "text": text,
            }
        )
        emitted = False
        async for chunk in self._synthesize(text, websocket, task_id):
            if chunk:
                await websocket.send_bytes(chunk)
                emitted = True
        if emitted:
            await websocket.send_json(
                {"event": "tts_completed", "stage": "tts_audio_sent", "task_id": task_id}
            )
        return emitted

    async def _transcribe(self, audio: bytes, task_id: str) -> str:
        incoming = self.asr_service._convert_audio_bytes_to_array(
            audio,
            self.audio_format,
            self.sample_rate,
            task_id,
        )
        if self.streaming_session is not None:
            events = await self.streaming_session.accept_audio(incoming, is_final=False)
            for event in reversed(events):
                if event.kind in {"end", "partial"} and event.text:
                    return event.text.strip()
            return ""

        self.audio_buffer = np.concatenate([self.audio_buffer, incoming])
        standard_chunk_sizes = [3840, 9600]
        selected_chunk_size = next(
            (size for size in sorted(standard_chunk_sizes, reverse=True) if len(self.audio_buffer) >= size),
            None,
        )
        if selected_chunk_size is None:
            logger.debug(
                "[%s] realtime voice ASR buffer waiting: %s samples",
                task_id,
                len(self.audio_buffer),
            )
            return ""

        audio_chunk = self.audio_buffer[:selected_chunk_size]
        self.audio_buffer = self.audio_buffer[selected_chunk_size:]
        audio_bytes = (audio_chunk * 32768.0).astype(np.int16).tobytes()
        result_text, _, _, _, self.audio_cache, self.audio_time = await self.asr_service._process_audio_chunk(
            audio_bytes,
            self.audio_cache,
            self.punc_cache,
            self.asr_params,
            self.audio_time,
            task_id,
            is_final=False,
        )
        return (result_text or "").strip()

    async def _synthesize(self, text: str, websocket: WebSocket, task_id: str):
        speech_rate = self.parameters.get("speech_rate", self.parameters.get("speechRate", 0))
        speed = convert_speech_rate_to_speed(speech_rate)
        async for chunk in self.tts_service._synthesize_streaming_audio(
            text,
            self.voice_name,
            speed,
            "PCM",
            self.sample_rate,
            int(self.parameters.get("volume", 50)),
            int(self.parameters.get("pitch_rate", self.parameters.get("pitchRate", self.parameters.get("pitch", 0)))),
            task_id,
            websocket,
            self.parameters.get("prompt", ""),
        ):
            yield chunk


@router.websocket("/voice")
async def realtime_voice_endpoint(websocket: WebSocket):
    """实时变声会话。

    在 FunSpeech 内部执行 ASR -> TTS。
    """
    await websocket.accept()
    task_id = generate_task_id("realtime_voice")
    voice_name = ""
    parameters = {}
    pipeline = "passthrough"
    asr_tts_session = None

    await websocket.send_json(
        {
            "event": "session_started",
            "task_id": task_id,
            "status": 20000000,
            "audio_mode": "asr_tts_pipeline",
            "supported_pipelines": ["asr_tts"],
            "pipeline_aliases": {"passthrough": "asr_tts"},
        }
    )

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            if "bytes" in message and message["bytes"] is not None:
                if not voice_name:
                    await _send_error(websocket, task_id, "请先发送configure事件设置voice_name")
                    continue
                if asr_tts_session is None:
                    await _send_error(websocket, task_id, "ASR->TTS会话未初始化")
                    continue
                await asr_tts_session.process_audio(websocket, task_id, message["bytes"])
                continue

            if "text" not in message or message["text"] is None:
                continue

            try:
                data = json.loads(message["text"])
            except json.JSONDecodeError:
                await _send_error(websocket, task_id, "消息必须是JSON")
                continue

            event = data.get("event") or data.get("type")
            if event == "start":
                event = "configure"
            elif event == "update_params":
                event = "update"
            elif event == "update_voice":
                event = "switch_voice"

            if event in {"configure", "switch_voice"}:
                next_voice = (data.get("voice_name") or data.get("voiceName") or "").strip()
                if not next_voice:
                    await _send_error(websocket, task_id, "voice_name不能为空")
                    continue
                tts_engine = get_tts_engine()
                voices = tts_engine.get_voices() if hasattr(tts_engine, "get_voices") else []
                if next_voice not in voices:
                    await _send_error(websocket, task_id, f"voice_name不存在: {next_voice}")
                    continue

                voice_name = next_voice
                parameters.update(data.get("parameters") or data.get("params") or {})
                pipeline = data.get("pipeline") or data.get("mode") or "asr_tts"
                if pipeline == "passthrough":
                    pipeline = "asr_tts"
                asr_tts_session = RealtimeVoiceAsrTtsSession(
                    voice_name=voice_name,
                    audio_format=data.get("format", "pcm"),
                    sample_rate=data.get("sample_rate", data.get("sampleRate", 16000)),
                    parameters=parameters,
                )
                await websocket.send_json(
                    {
                        "event": "configured" if event == "configure" else "voice_switched",
                        "task_id": task_id,
                        "voice_name": voice_name,
                        "format": data.get("format", "pcm"),
                        "sample_rate": data.get("sample_rate", data.get("sampleRate", 16000)),
                        "pipeline": pipeline,
                        "audio_mode": "asr_tts_pipeline",
                        "status": 20000000,
                    }
                )
            elif event == "update":
                parameters.update(data.get("parameters") or data.get("params") or {})
                if asr_tts_session is not None:
                    asr_tts_session.parameters.update(data.get("parameters") or data.get("params") or {})
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
