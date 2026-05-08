# -*- coding: utf-8 -*-
"""Voice Cloner实时变声WebSocket接口。"""

import json
import logging
import asyncio
import contextlib

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...core.config import settings
from ...utils.common import convert_speech_rate_to_speed
from ...utils.common import generate_task_id
from ...services.tts.engine import get_tts_engine
from ...services.websocket_asr import get_aliyun_websocket_asr_service
from ...services.websocket_tts import get_aliyun_websocket_tts_service
from ...services.realtime_voice.audio_pacer import AudioPacer
from ...services.realtime_voice.backpressure import BoundedAudioQueue, TtsJobQueue
from ...services.realtime_voice.text_commit import StableTextCommitter
from ...services.realtime_voice.types import AudioFrame, AsrHypothesis, TtsJob


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
        self.asr_engine = asr_engine
        self.asr_engine_index = None
        if hasattr(asr_engine, "get_engine_for_session") and hasattr(
            asr_engine, "release_session_engine"
        ):
            self.asr_engine_index, self.session_asr_engine = (
                asr_engine.get_engine_for_session()
            )
        else:
            self.session_asr_engine = asr_engine

        if hasattr(self.session_asr_engine, "create_streaming_session"):
            self.streaming_session = self.session_asr_engine.create_streaming_session(
                self.asr_params
            )
        else:
            self.streaming_session = None
        self._closed = False
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_cache = {}
        self.punc_cache = {}
        self.audio_time = 0
        self.last_text = ""
        self.audio_queue = BoundedAudioQueue(
            settings.REALTIME_AUDIO_INPUT_HIGH_WATERMARK_MS,
            settings.REALTIME_AUDIO_INPUT_MAX_MS,
        )
        self.tts_jobs = TtsJobQueue(settings.REALTIME_TTS_JOB_QUEUE_SIZE)
        self.text_committer = StableTextCommitter(
            stable_hypotheses=settings.REALTIME_TEXT_STABLE_HYPOTHESES,
            min_commit_chars=settings.REALTIME_TEXT_MIN_COMMIT_CHARS,
            max_commit_wait_ms=settings.REALTIME_TEXT_MAX_COMMIT_WAIT_MS,
        )
        self.audio_pacer = AudioPacer(
            self.sample_rate, frame_ms=settings.REALTIME_PACER_FRAME_MS
        )
        self._tasks: list[asyncio.Task] = []
        self._websocket: WebSocket | None = None
        self._task_id: str | None = None
        self._send_lock = asyncio.Lock()
        self._started = False

    def close(self):
        """Cancel workers and release a session-pinned ASR engine."""
        if self._closed:
            return
        self._closed = True
        for task in self._tasks:
            task.cancel()
        if self.asr_engine_index is not None and hasattr(
            self.asr_engine, "release_session_engine"
        ):
            self.asr_engine.release_session_engine(self.asr_engine_index)

    async def aclose(self):
        """Async close variant that waits for worker cancellation."""
        self.close()
        if self._tasks:
            with contextlib.suppress(Exception):
                await asyncio.gather(*self._tasks, return_exceptions=True)

    def start(self, websocket: WebSocket, task_id: str) -> None:
        """Start background actors for non-blocking audio ingestion."""
        if self._started:
            return
        self._started = True
        self._websocket = websocket
        self._task_id = task_id
        self._tasks = [
            asyncio.create_task(self._audio_worker(), name=f"{task_id}:realtime-asr"),
            asyncio.create_task(self._tts_worker(), name=f"{task_id}:realtime-tts"),
        ]

    async def accept_audio(self, audio: bytes) -> bool:
        """Queue audio for async ASR/TTS processing without blocking reception."""
        if not self._started or self._websocket is None or self._task_id is None:
            raise RuntimeError("RealtimeVoiceAsrTtsSession.start() must be called first")

        frame = self._build_audio_frame(audio)
        for event in await self.audio_queue.put(frame):
            await self._send_json(
                {
                    "event": "backpressure",
                    "type": event.type,
                    "task_id": self._task_id,
                    "queue_ms": event.queue_ms,
                    "dropped_ms": event.dropped_ms,
                    "message": event.message,
                }
            )
        return True

    async def process_audio(self, websocket: WebSocket, task_id: str, audio: bytes) -> bool:
        """Backward-compatible entry point used by older tests/callers.

        The actual pipeline is actor-based: this method only starts workers and
        queues the input so the WebSocket receive loop is not held by ASR/TTS.
        """
        self.start(websocket, task_id)
        return await self.accept_audio(audio)

    async def _audio_worker(self) -> None:
        assert self._websocket is not None
        assert self._task_id is not None
        while not self._closed:
            frame = await self.audio_queue.get()
            await self._send_json(
                {
                    "event": "pipeline_stage",
                    "stage": "asr_receiving_audio",
                    "task_id": self._task_id,
                    "queue_ms": self.audio_queue.queued_ms,
                }
            )
            events = await self._transcribe_events(frame.payload, self._task_id)
            for asr_event in events:
                await self._handle_asr_hypothesis(asr_event)

    async def _handle_asr_hypothesis(self, hypothesis: AsrHypothesis) -> None:
        assert self._task_id is not None
        if not hypothesis.text:
            return
        await self._send_json(
            {
                "event": "asr_result",
                "stage": "asr_text_received",
                "task_id": self._task_id,
                "text": hypothesis.text,
                "is_final": hypothesis.is_final,
            }
        )
        committed = self.text_committer.update(hypothesis)
        if committed:
            job = TtsJob(
                revision_id=committed.revision_id,
                text=committed.text,
                voice_name=self.voice_name,
                parameters=dict(self.parameters),
                priority="final" if committed.is_final else "stable",
            )
            for event in await self.tts_jobs.put(job):
                await self._send_json(
                    {
                        "event": "backpressure",
                        "type": event.type,
                        "task_id": self._task_id,
                        "message": event.message,
                    }
                )
        if hypothesis.is_final:
            self.text_committer.reset_sentence()

    async def _tts_worker(self) -> None:
        assert self._websocket is not None
        assert self._task_id is not None
        while not self._closed:
            job = await self.tts_jobs.get()
            await self._send_json(
                {
                    "event": "pipeline_stage",
                    "stage": "tts_synthesizing",
                    "task_id": self._task_id,
                    "text": job.text,
                    "revision_id": job.revision_id,
                }
            )
            emitted = False
            async for chunk in self._synthesize(
                job.text,
                self._websocket,
                self._task_id,
                voice_name=job.voice_name,
                parameters=job.parameters,
            ):
                if not chunk:
                    continue
                async for frame in self.audio_pacer.iter_frames(chunk):
                    await self._send_bytes(frame)
                    emitted = True
            async for frame in self.audio_pacer.flush():
                await self._send_bytes(frame)
                emitted = True
            if emitted:
                await self._send_json(
                    {
                        "event": "tts_completed",
                        "stage": "tts_audio_sent",
                        "task_id": self._task_id,
                        "revision_id": job.revision_id,
                    }
                )

    async def _transcribe(self, audio: bytes, task_id: str) -> str:
        events = await self._transcribe_events(audio, task_id)
        for event in reversed(events):
            if event.text:
                return event.text
        return ""

    async def _transcribe_events(self, audio: bytes, task_id: str) -> list[AsrHypothesis]:
        incoming = self.asr_service._convert_audio_bytes_to_array(
            audio,
            self.audio_format,
            self.sample_rate,
            task_id,
        )
        if self.streaming_session is not None:
            events = await self.streaming_session.accept_audio(incoming, is_final=False)
            return [
                AsrHypothesis(
                    text=event.text.strip(),
                    is_final=event.kind == "end",
                    time_ms=getattr(event, "time_ms", 0),
                )
                for event in events
                if event.kind in {"end", "partial"} and event.text
            ]

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
            return []

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
            session_engine=self.session_asr_engine,
        )
        text = (result_text or "").strip()
        return [AsrHypothesis(text=text, is_final=False)] if text else []

    async def _synthesize(
        self,
        text: str,
        websocket: WebSocket,
        task_id: str,
        voice_name: str | None = None,
        parameters: dict | None = None,
    ):
        parameters = parameters or self.parameters
        speech_rate = parameters.get("speech_rate", parameters.get("speechRate", 0))
        speed = convert_speech_rate_to_speed(speech_rate)
        async for chunk in self.tts_service._synthesize_streaming_audio(
            text,
            voice_name or self.voice_name,
            speed,
            "PCM",
            self.sample_rate,
            int(parameters.get("volume", 50)),
            int(parameters.get("pitch_rate", parameters.get("pitchRate", parameters.get("pitch", 0)))),
            task_id,
            websocket,
            parameters.get("prompt", ""),
        ):
            yield chunk

    def _build_audio_frame(self, audio: bytes) -> AudioFrame:
        duration_ms = self._estimate_duration_ms(audio)
        return AudioFrame(
            payload=audio,
            duration_ms=duration_ms,
            is_silence=self._is_silence(audio),
        )

    def _estimate_duration_ms(self, audio: bytes) -> int:
        if self.audio_format != "pcm" or self.sample_rate <= 0:
            return 20
        samples = max(1, len(audio) // 2)
        return max(1, int(samples / self.sample_rate * 1000))

    def _is_silence(self, audio: bytes) -> bool:
        if self.audio_format != "pcm" or len(audio) < 2:
            return False
        pcm = np.frombuffer(audio, dtype=np.int16)
        if pcm.size == 0:
            return True
        audio_array = pcm.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(audio_array))))
        return rms < settings.ASR_NEARFIELD_RMS_THRESHOLD

    async def _send_json(self, payload: dict) -> None:
        if self._websocket is None:
            return
        async with self._send_lock:
            await self._websocket.send_json(payload)

    async def _send_bytes(self, payload: bytes) -> None:
        if self._websocket is None:
            return
        async with self._send_lock:
            await self._websocket.send_bytes(payload)


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
                if hasattr(asr_tts_session, "accept_audio"):
                    if hasattr(asr_tts_session, "start"):
                        asr_tts_session.start(websocket, task_id)
                    await asr_tts_session.accept_audio(message["bytes"])
                else:
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
                if asr_tts_session is not None and hasattr(asr_tts_session, "aclose"):
                    await asr_tts_session.aclose()
                elif asr_tts_session is not None and hasattr(asr_tts_session, "close"):
                    asr_tts_session.close()
                asr_tts_session = RealtimeVoiceAsrTtsSession(
                    voice_name=voice_name,
                    audio_format=data.get("format", "pcm"),
                    sample_rate=data.get("sample_rate", data.get("sampleRate", 16000)),
                    parameters=parameters,
                )
                if hasattr(asr_tts_session, "start"):
                    asr_tts_session.start(websocket, task_id)
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
        if asr_tts_session is not None and hasattr(asr_tts_session, "aclose"):
            await asr_tts_session.aclose()
        elif asr_tts_session is not None and hasattr(asr_tts_session, "close"):
            asr_tts_session.close()
        try:
            await websocket.close()
        except Exception:
            pass
