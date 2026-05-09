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
from ...services.realtime_voice.events import RealtimeVoiceEventBuilder
from ...services.realtime_voice.text_commit import StableTextCommitter
from ...services.realtime_voice.tts_dispatcher import get_realtime_tts_dispatcher
from ...services.realtime_voice.types import AudioFrame, AsrHypothesis, TtsJob


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws/v1/realtime", tags=["Realtime Voice"])


async def _send_error(
    websocket: WebSocket,
    task_id: str,
    message: str,
    event_builder: RealtimeVoiceEventBuilder | None = None,
):
    if event_builder is not None:
        await websocket.send_json(
            event_builder.build(
                "session.error",
                status=40000003,
                payload={"message": message},
                message=message,
            )
        )
        return
    await websocket.send_json(
        {"event": "error", "task_id": task_id, "status": 40000003, "message": message}
    )


class RealtimeVoiceAsrTtsSession:
    """Internal realtime voice chain: incoming audio -> ASR text -> TTS audio."""

    def __init__(
        self,
        voice_name: str,
        audio_format: str = "pcm",
        sample_rate: int = 16000,
        parameters: dict | None = None,
        event_builder: RealtimeVoiceEventBuilder | None = None,
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
            "enable_emotion": bool(self.parameters.get("enable_emotion_detection", True)),
            "return_rich_text": bool(self.parameters.get("return_rich_text", False)),
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
        self._event_builder = event_builder
        self._send_lock = asyncio.Lock()
        self._started = False
        self._utterance_index = 1
        self._hypothesis_index = 0
        self._audio_chunk_index = 0
        self._input_frame_index = 0
        self._speech_active = False
        self.config_version = 1

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
        if self._event_builder is None:
            self._event_builder = RealtimeVoiceEventBuilder(task_id)
        self._tasks = [
            asyncio.create_task(self._audio_worker(), name=f"{task_id}:realtime-asr"),
            asyncio.create_task(self._tts_worker(), name=f"{task_id}:realtime-tts"),
        ]

    def update_parameters(self, parameters: dict) -> int:
        """Update runtime parameters and return the new config version."""
        self.parameters.update(parameters)
        if "enable_emotion_detection" in parameters:
            self.asr_params["enable_emotion"] = bool(parameters["enable_emotion_detection"])
        if "return_rich_text" in parameters:
            self.asr_params["return_rich_text"] = bool(parameters["return_rich_text"])
        self.config_version += 1
        return self.config_version

    async def accept_audio(self, audio: bytes) -> bool:
        """Queue audio for async ASR/TTS processing without blocking reception."""
        if not self._started or self._websocket is None or self._task_id is None:
            raise RuntimeError("RealtimeVoiceAsrTtsSession.start() must be called first")

        frame = self._build_audio_frame(audio)
        for event in await self.audio_queue.put(frame):
            await self._send_json(
                self._event(
                    "backpressure.applied",
                    payload={
                        "scope": "input",
                        "level": "warning",
                        "reason": event.type,
                        "queue_ms": event.queue_ms,
                        "dropped_ms": event.dropped_ms,
                        "action": event.type,
                        "message": event.message,
                        "vad_state": frame.vad_state,
                        "speech_active": frame.speech_active,
                    },
                    type=event.type,
                    queue_ms=event.queue_ms,
                    dropped_ms=event.dropped_ms,
                    message=event.message,
                )
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
                self._event(
                    "input.audio_dequeued",
                    payload={
                        "queue_ms": self.audio_queue.queued_ms,
                        "duration_ms": frame.duration_ms,
                        "is_silence": frame.is_silence,
                        "input_frame_index": frame.sequence,
                        "vad_state": frame.vad_state,
                        "speech_active": frame.speech_active,
                    },
                    stage="asr_receiving_audio",
                    queue_ms=self.audio_queue.queued_ms,
                )
            )
            if frame.speech_active:
                await self._send_json(
                    self._event(
                        "vad.speech_frame",
                        payload={
                            "utterance_id": self._current_utterance_id(),
                            "input_frame_index": frame.sequence,
                            "duration_ms": frame.duration_ms,
                            "vad_state": frame.vad_state,
                            "source": "vad",
                        },
                    )
                )
            events = await self._transcribe_events(frame.payload, self._task_id)
            for asr_event in events:
                await self._handle_asr_hypothesis(asr_event)

    async def _handle_asr_hypothesis(self, hypothesis: AsrHypothesis) -> None:
        assert self._task_id is not None
        utterance_id = self._current_utterance_id()
        if hypothesis.kind == "begin":
            self._speech_active = True
            await self._send_json(
                self._event(
                    "vad.speech_started",
                    payload={
                        "utterance_id": utterance_id,
                        "speech_begin_ms": hypothesis.speech_begin_ms
                        if hypothesis.speech_begin_ms is not None
                        else hypothesis.begin_time_ms,
                        "time_ms": hypothesis.time_ms,
                        "source": hypothesis.vad_source or "vad",
                    },
                )
            )
            return
        if hypothesis.kind == "end":
            await self._send_json(
                self._event(
                    "vad.speech_ended",
                    payload={
                        "utterance_id": utterance_id,
                        "speech_end_ms": hypothesis.speech_end_ms
                        if hypothesis.speech_end_ms is not None
                        else hypothesis.time_ms,
                        "time_ms": hypothesis.time_ms,
                        "source": hypothesis.vad_source or "vad",
                    },
                )
            )
            self._speech_active = False
        if not hypothesis.text:
            return
        self._hypothesis_index += 1
        hypothesis_id = f"{utterance_id}_hyp_{self._hypothesis_index}"
        await self._send_json(
            self._event(
                "asr.hypothesis",
                payload={
                    "utterance_id": utterance_id,
                    "hypothesis_id": hypothesis_id,
                    "text": hypothesis.text,
                    "is_final": hypothesis.is_final,
                    "time_ms": hypothesis.time_ms,
                    "begin_time_ms": hypothesis.begin_time_ms,
                    "speech_active": self._speech_active,
                    "emotion": hypothesis.emotion,
                    "emotion_confidence": hypothesis.emotion_confidence,
                    "raw_rich_text": hypothesis.raw_rich_text,
                },
            )
        )
        await self._send_json(
            self._event(
                "asr_result",
                payload={
                    "protocol_event": "asr.hypothesis",
                    "utterance_id": utterance_id,
                    "hypothesis_id": hypothesis_id,
                    "text": hypothesis.text,
                    "is_final": hypothesis.is_final,
                    "speech_active": self._speech_active,
                    "emotion": hypothesis.emotion,
                    "emotion_confidence": hypothesis.emotion_confidence,
                    "raw_rich_text": hypothesis.raw_rich_text,
                },
                stage="asr_text_received",
                text=hypothesis.text,
                is_final=hypothesis.is_final,
            )
        )
        committed = self.text_committer.update(hypothesis)
        if committed:
            tts_job_id = f"tts_{committed.revision_id}"
            await self._send_json(
                self._event(
                    "asr.text_committed",
                    payload={
                        "utterance_id": utterance_id,
                        "revision_id": committed.revision_id,
                        "delta_text": committed.text,
                        "full_text": committed.full_text,
                        "is_final": committed.is_final,
                        "tts_job_id": tts_job_id,
                    },
                )
            )
            job_parameters = dict(self.parameters)
            if (
                hypothesis.emotion
                and job_parameters.get("emotion_control", "asr") != "off"
                and not job_parameters.get("emotion")
            ):
                job_parameters["emotion"] = hypothesis.emotion
                job_parameters["emotion_intensity"] = job_parameters.get("emotion_intensity")
                job_parameters["emotion_source"] = "asr"

            job = TtsJob(
                revision_id=committed.revision_id,
                text=committed.text,
                voice_name=self.voice_name,
                parameters=job_parameters,
                priority="final" if committed.is_final else "stable",
            )
            for event in await self.tts_jobs.put(job):
                await self._send_json(
                    self._event(
                        "backpressure.applied",
                        payload={
                            "scope": "tts",
                            "level": "warning",
                            "reason": event.type,
                            "action": event.type,
                            "message": event.message,
                        },
                        type=event.type,
                        message=event.message,
                    )
                )
            await self._send_json(
                self._event(
                    "tts.job_queued",
                    payload={
                        "tts_job_id": tts_job_id,
                        "revision_id": committed.revision_id,
                        "utterance_id": utterance_id,
                        "text": committed.text,
                        "voice_name": self.voice_name,
                        "config_version": self.config_version,
                        "priority": job.priority,
                    },
                )
            )
        if hypothesis.is_final:
            await self._send_json(
                self._event(
                    "asr.sentence_finalized",
                    payload={"utterance_id": utterance_id, "text": hypothesis.text},
                )
            )
            self.text_committer.reset_sentence()
            self._utterance_index += 1

    async def _tts_worker(self) -> None:
        assert self._websocket is not None
        assert self._task_id is not None
        dispatcher = get_realtime_tts_dispatcher()
        while not self._closed:
            job = await self.tts_jobs.get()
            tts_job_id = f"tts_{job.revision_id}"
            await self._send_json(
                self._event(
                    "tts.job_waiting",
                    payload={
                        "tts_job_id": tts_job_id,
                        "revision_id": job.revision_id,
                        "text": job.text,
                        "voice_name": job.voice_name,
                        "config_version": self.config_version,
                        "priority": job.priority,
                        "global_active": dispatcher.active,
                        "global_waiting": dispatcher.waiting,
                    },
                )
            )
            lease_or_rejection = await dispatcher.acquire()
            if not getattr(lease_or_rejection, "admission", lease_or_rejection).accepted:
                rejection = lease_or_rejection
                await self._send_json(
                    self._event(
                        "tts.job_dropped",
                        payload={
                            "tts_job_id": tts_job_id,
                            "revision_id": job.revision_id,
                            "reason": rejection.reason,
                            "queue_wait_ms": rejection.queue_wait_ms,
                            "global_active": rejection.active,
                            "global_waiting": rejection.waiting,
                            "priority": job.priority,
                        },
                    )
                )
                await self._send_json(
                    self._event(
                        "backpressure.applied",
                        payload={
                            "scope": "tts",
                            "level": "warning",
                            "reason": rejection.reason,
                            "action": "drop_tts_job",
                            "message": f"revision={job.revision_id}",
                        },
                        type=rejection.reason,
                        message=f"revision={job.revision_id}",
                    )
                )
                continue

            lease = lease_or_rejection
            admission = lease.admission
            await self._send_json(
                self._event(
                    "tts.job_admitted",
                    payload={
                        "tts_job_id": tts_job_id,
                        "revision_id": job.revision_id,
                        "queue_wait_ms": admission.queue_wait_ms,
                        "global_active": admission.active,
                        "global_waiting": admission.waiting,
                    },
                )
            )
            await self._send_json(
                self._event(
                    "tts.job_started",
                    payload={
                        "tts_job_id": tts_job_id,
                        "revision_id": job.revision_id,
                        "text": job.text,
                        "voice_name": job.voice_name,
                        "config_version": self.config_version,
                        "priority": job.priority,
                    },
                    stage="tts_synthesizing",
                    text=job.text,
                    revision_id=job.revision_id,
                )
            )
            emitted = False
            first_audio_sent = False
            tts_audio_queue: asyncio.Queue = asyncio.Queue(
                maxsize=max(1, settings.REALTIME_TTS_AUDIO_QUEUE_SIZE)
            )
            synth_done = object()

            async def synthesize_job_audio() -> None:
                try:
                    async with lease:
                        async for chunk in self._synthesize(
                            job.text,
                            self._websocket,
                            self._task_id,
                            voice_name=job.voice_name,
                            parameters=job.parameters,
                        ):
                            if chunk:
                                await tts_audio_queue.put(chunk)
                except BaseException as exc:
                    await tts_audio_queue.put(exc)
                finally:
                    await tts_audio_queue.put(synth_done)

            synth_task = asyncio.create_task(
                synthesize_job_audio(),
                name=f"{self._task_id}:tts-synth:{job.revision_id}",
            )
            try:
                while True:
                    chunk = await tts_audio_queue.get()
                    if chunk is synth_done:
                        break
                    if isinstance(chunk, BaseException):
                        raise chunk
                    async for frame in self.audio_pacer.iter_frames(chunk):
                        self._audio_chunk_index += 1
                        if not first_audio_sent:
                            await self._send_json(
                                self._event(
                                    "tts.first_audio",
                                    payload={
                                        "tts_job_id": tts_job_id,
                                        "revision_id": job.revision_id,
                                        "audio_chunk_index": self._audio_chunk_index,
                                    },
                                )
                            )
                            first_audio_sent = True
                        await self._send_audio_frame(
                            tts_job_id=tts_job_id,
                            revision_id=job.revision_id,
                            frame=frame,
                        )
                        emitted = True
            finally:
                if not synth_task.done():
                    synth_task.cancel()
                with contextlib.suppress(BaseException):
                    await synth_task
            async for frame in self.audio_pacer.flush():
                self._audio_chunk_index += 1
                if not first_audio_sent:
                    await self._send_json(
                        self._event(
                            "tts.first_audio",
                            payload={
                                "tts_job_id": tts_job_id,
                                "revision_id": job.revision_id,
                                "audio_chunk_index": self._audio_chunk_index,
                            },
                        )
                    )
                    first_audio_sent = True
                await self._send_audio_frame(
                    tts_job_id=tts_job_id,
                    revision_id=job.revision_id,
                    frame=frame,
                )
                emitted = True
            if emitted:
                await self._send_json(
                    self._event(
                        "tts.job_completed",
                        payload={
                            "tts_job_id": tts_job_id,
                            "revision_id": job.revision_id,
                        },
                    )
                )
                await self._send_json(
                    self._event(
                        "tts_completed",
                        payload={
                            "protocol_event": "tts.job_completed",
                            "tts_job_id": tts_job_id,
                            "revision_id": job.revision_id,
                        },
                        stage="tts_audio_sent",
                        revision_id=job.revision_id,
                    )
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
                    kind=event.kind,
                    time_ms=getattr(event, "time_ms", 0),
                    begin_time_ms=getattr(event, "begin_time_ms", 0),
                    speech_begin_ms=getattr(event, "speech_begin_ms", None),
                    speech_end_ms=getattr(event, "speech_end_ms", None),
                    vad_source=getattr(event, "vad_source", None),
                    speech_active=getattr(event, "speech_active", False),
                    emotion=getattr(event, "emotion", None),
                    emotion_confidence=getattr(event, "emotion_confidence", None),
                    raw_rich_text=getattr(event, "raw_rich_text", None),
                )
                for event in events
                if event.kind in {"begin", "end", "partial"}
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
            parameters.get("emotion"),
            parameters.get("emotion_intensity"),
        ):
            yield chunk

    def _build_audio_frame(self, audio: bytes) -> AudioFrame:
        duration_ms = self._estimate_duration_ms(audio)
        is_silence = self._is_silence(audio)
        self._input_frame_index += 1
        return AudioFrame(
            payload=audio,
            duration_ms=duration_ms,
            is_silence=is_silence,
            sequence=self._input_frame_index,
            vad_state="speech" if self._speech_active else ("silence" if is_silence else "speech_like"),
            speech_active=self._speech_active,
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

    async def _send_audio_frame(
        self,
        *,
        tts_job_id: str,
        revision_id: int,
        frame: bytes,
    ) -> None:
        """Send audio metadata and binary bytes without interleaving events."""
        if self._websocket is None:
            return
        async with self._send_lock:
            await self._websocket.send_json(
                self._event(
                    "tts.audio_chunk",
                    payload={
                        "tts_job_id": tts_job_id,
                        "revision_id": revision_id,
                        "audio_chunk_index": self._audio_chunk_index,
                        "bytes": len(frame),
                        "sample_rate": self.sample_rate,
                        "format": "PCM",
                    },
                )
            )
            await self._websocket.send_bytes(frame)

    def _event(self, event: str, *, payload: dict | None = None, **fields) -> dict:
        if self._event_builder is None:
            assert self._task_id is not None
            self._event_builder = RealtimeVoiceEventBuilder(self._task_id)
        return self._event_builder.build(event, payload=payload, **fields)

    def _current_utterance_id(self) -> str:
        return f"utt_{self._utterance_index}"


@router.websocket("/voice")
async def realtime_voice_endpoint(websocket: WebSocket):
    """实时变声会话。

    在 FunSpeech 内部执行 ASR -> TTS。
    """
    await websocket.accept()
    task_id = generate_task_id("realtime_voice")
    event_builder = RealtimeVoiceEventBuilder(task_id)
    voice_name = ""
    parameters = {}
    pipeline = "passthrough"
    asr_tts_session = None

    await websocket.send_json(
        event_builder.build(
            "session_started",
            payload={
                "protocol_event": "session.started",
                "audio_mode": "asr_tts_pipeline",
                "supported_pipelines": ["asr_tts"],
                "pipeline_aliases": {"passthrough": "asr_tts"},
            },
            protocol_event="session.started",
            audio_mode="asr_tts_pipeline",
            supported_pipelines=["asr_tts"],
            pipeline_aliases={"passthrough": "asr_tts"},
        )
    )

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            if "bytes" in message and message["bytes"] is not None:
                if not voice_name:
                    await _send_error(
                        websocket, task_id, "请先发送configure事件设置voice_name", event_builder
                    )
                    continue
                if asr_tts_session is None:
                    await _send_error(websocket, task_id, "ASR->TTS会话未初始化", event_builder)
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
                await _send_error(websocket, task_id, "消息必须是JSON", event_builder)
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
                    await _send_error(websocket, task_id, "voice_name不能为空", event_builder)
                    continue
                tts_engine = get_tts_engine()
                voices = tts_engine.get_voices() if hasattr(tts_engine, "get_voices") else []
                if next_voice not in voices:
                    await _send_error(
                        websocket, task_id, f"voice_name不存在: {next_voice}", event_builder
                    )
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
                session_kwargs = {
                    "voice_name": voice_name,
                    "audio_format": data.get("format", "pcm"),
                    "sample_rate": data.get("sample_rate", data.get("sampleRate", 16000)),
                    "parameters": parameters,
                }
                try:
                    asr_tts_session = RealtimeVoiceAsrTtsSession(
                        **session_kwargs, event_builder=event_builder
                    )
                except TypeError as exc:
                    if "event_builder" not in str(exc):
                        raise
                    asr_tts_session = RealtimeVoiceAsrTtsSession(**session_kwargs)
                if hasattr(asr_tts_session, "start"):
                    asr_tts_session.start(websocket, task_id)
                config_version = getattr(asr_tts_session, "config_version", None)
                await websocket.send_json(
                    event_builder.build(
                        "configured" if event == "configure" else "voice_switched",
                        payload={
                            "protocol_event": "session.configured"
                            if event == "configure"
                            else "session.voice_switched",
                            "voice_name": voice_name,
                            "format": data.get("format", "pcm"),
                            "sample_rate": data.get("sample_rate", data.get("sampleRate", 16000)),
                            "pipeline": pipeline,
                            "audio_mode": "asr_tts_pipeline",
                            "config_version": config_version,
                        },
                        protocol_event="session.configured"
                        if event == "configure"
                        else "session.voice_switched",
                        voice_name=voice_name,
                        format=data.get("format", "pcm"),
                        sample_rate=data.get("sample_rate", data.get("sampleRate", 16000)),
                        pipeline=pipeline,
                        audio_mode="asr_tts_pipeline",
                        config_version=config_version,
                    )
                )
            elif event == "update":
                next_parameters = data.get("parameters") or data.get("params") or {}
                parameters.update(next_parameters)
                config_version = None
                if asr_tts_session is not None:
                    if hasattr(asr_tts_session, "update_parameters"):
                        config_version = asr_tts_session.update_parameters(next_parameters)
                    else:
                        asr_tts_session.parameters.update(next_parameters)
                await websocket.send_json(
                    event_builder.build(
                        "parameters_updated",
                        payload={
                            "protocol_event": "session.parameters_updated",
                            "voice_name": voice_name,
                            "parameters": parameters,
                            "config_version": config_version,
                        },
                        protocol_event="session.parameters_updated",
                        voice_name=voice_name,
                        parameters=parameters,
                        config_version=config_version,
                    )
                )
            elif event in {"close", "stop"}:
                await websocket.send_json(
                    event_builder.build(
                        "session_completed",
                        payload={"protocol_event": "session.completed"},
                        protocol_event="session.completed",
                    )
                )
                break
            else:
                await _send_error(websocket, task_id, f"不支持的事件: {event}", event_builder)
    except WebSocketDisconnect:
        logger.info("[%s] 实时变声客户端断开", task_id)
    except Exception as exc:
        logger.error("[%s] 实时变声处理异常: %s", task_id, exc)
        try:
            await _send_error(websocket, task_id, f"实时变声处理失败: {exc}", event_builder)
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
