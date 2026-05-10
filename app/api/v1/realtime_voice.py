# -*- coding: utf-8 -*-
"""Voice Cloner实时变声WebSocket接口。"""

import json
import logging
import asyncio
import contextlib
from dataclasses import dataclass, replace

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...core.config import settings
from ...core.executor import run_sync
from ...utils.common import convert_speech_rate_to_speed
from ...utils.common import generate_task_id
from ...services.tts.engine import get_tts_engine
from ...services.websocket_asr import get_aliyun_websocket_asr_service
from ...services.websocket_tts import get_aliyun_websocket_tts_service
from ...services.asr.vad import StreamingVADEndpointDetector
from ...services.realtime_voice.backpressure import AsrSegmentQueue, BoundedAudioQueue, TtsJobQueue
from ...services.realtime_voice.events import RealtimeVoiceEventBuilder
from ...services.realtime_voice.playback_queue import PlaybackChunk, TtsPlaybackQueue
from ...services.realtime_voice.tts_dispatcher import get_realtime_tts_dispatcher
from ...services.realtime_voice.types import AsrSegment, AudioFrame, AsrHypothesis, TtsJob
from ...services.realtime_voice.vad_segmenter import SlidingVadSegmenter


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws/v1/realtime", tags=["Realtime Voice"])
MAX_ASR_SEGMENT_MS = 60_000
PCM_SAMPLE_BYTES = 2


@dataclass
class PrefetchedTtsJob:
    job: TtsJob
    tts_job_id: str
    audio_queue: asyncio.Queue
    done_sentinel: object
    task: asyncio.Task


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
        self.realtime_vad_detector = StreamingVADEndpointDetector(
            self.session_asr_engine,
            self.sample_rate,
        )

        self._closed = False
        self.audio_queue = BoundedAudioQueue(
            settings.REALTIME_AUDIO_INPUT_HIGH_WATERMARK_MS,
            settings.REALTIME_AUDIO_INPUT_MAX_MS,
            sample_rate=self.sample_rate,
            frame_ms=settings.REALTIME_VAD_FRAME_MS,
            frame_factory=self._build_audio_frame,
        )
        self.asr_queue = AsrSegmentQueue(
            settings.REALTIME_AUDIO_INPUT_HIGH_WATERMARK_MS,
            settings.REALTIME_AUDIO_INPUT_MAX_MS,
            preserve_speech=settings.REALTIME_PRESERVE_SPEECH_UNDER_PRESSURE,
        )
        self.vad_segmenter = SlidingVadSegmenter(
            window_ms=settings.REALTIME_VAD_FRAME_MS,
            pre_roll_ms=settings.ASR_VAD_SPEECH_PAD_MS,
            end_silence_ms=settings.ASR_VAD_END_FALLBACK_MS,
            post_pad_ms=settings.REALTIME_VAD_POST_PAD_MS,
            smooth_window_frames=settings.REALTIME_VAD_SMOOTH_WINDOW_FRAMES,
            smooth_speech_frames=settings.REALTIME_VAD_SMOOTH_SPEECH_FRAMES,
            start_speech_frames=settings.REALTIME_VAD_START_SPEECH_FRAMES,
            end_silence_frames=settings.REALTIME_VAD_END_SILENCE_FRAMES,
        )
        self.tts_jobs = TtsJobQueue(
            settings.REALTIME_TTS_JOB_QUEUE_SIZE,
            drop_on_overload=settings.REALTIME_TTS_DROP_ON_OVERLOAD,
        )
        self.playback_queue = TtsPlaybackQueue(
            settings.REALTIME_PLAYBACK_QUEUE_SIZE,
            settings.REALTIME_PLAYBACK_MAX_INFLIGHT,
            backpressure_sleep_ms=settings.REALTIME_PLAYBACK_BACKPRESSURE_SLEEP_MS,
        )
        self._tasks: list[asyncio.Task] = []
        self._websocket: WebSocket | None = None
        self._task_id: str | None = None
        self._event_builder = event_builder
        self._send_lock = asyncio.Lock()
        self._started = False
        self._utterance_index = 1
        self._hypothesis_index = 0
        self._tts_revision_id = 0
        self._audio_chunk_index = 0
        self._first_audio_sent_jobs: set[str] = set()
        self._playback_job_chunks: dict[str, set[str]] = {}
        self._playback_job_meta: dict[str, TtsJob] = {}
        self._playback_jobs_done_queueing: set[str] = set()
        self._playback_flush_lock = asyncio.Lock()
        self._input_frame_index = 0
        self._speech_active = False
        self.config_version = 1

    def realtime_config_snapshot(self) -> dict:
        return {
            "sensevoice_partial_decode_interval_ms": settings.SENSEVOICE_PARTIAL_DECODE_INTERVAL_MS,
            "sensevoice_min_decode_window_ms": settings.SENSEVOICE_MIN_DECODE_WINDOW_MS,
            "sensevoice_max_partial_window_ms": settings.SENSEVOICE_MAX_PARTIAL_WINDOW_MS,
            "preserve_speech_under_pressure": settings.REALTIME_PRESERVE_SPEECH_UNDER_PRESSURE,
            "tts_job_queue_size": settings.REALTIME_TTS_JOB_QUEUE_SIZE,
            "tts_prefetch_jobs": settings.REALTIME_TTS_PREFETCH_JOBS,
            "tts_drop_on_overload": settings.REALTIME_TTS_DROP_ON_OVERLOAD,
            "vad_frame_ms": settings.REALTIME_VAD_FRAME_MS,
            "vad_smooth_window_frames": settings.REALTIME_VAD_SMOOTH_WINDOW_FRAMES,
            "vad_smooth_speech_frames": settings.REALTIME_VAD_SMOOTH_SPEECH_FRAMES,
            "vad_start_speech_frames": settings.REALTIME_VAD_START_SPEECH_FRAMES,
            "vad_end_silence_frames": settings.REALTIME_VAD_END_SILENCE_FRAMES,
            "vad_post_pad_ms": settings.REALTIME_VAD_POST_PAD_MS,
            "tts_global_max_inflight": settings.REALTIME_TTS_GLOBAL_MAX_INFLIGHT,
            "tts_audio_queue_size": settings.REALTIME_TTS_AUDIO_QUEUE_SIZE,
            "playback_queue_size": settings.REALTIME_PLAYBACK_QUEUE_SIZE,
            "playback_max_inflight": settings.REALTIME_PLAYBACK_MAX_INFLIGHT,
            "playback_backpressure_sleep_ms": settings.REALTIME_PLAYBACK_BACKPRESSURE_SLEEP_MS,
        }

    async def update_client_audio_backpressure(self, payload: dict) -> None:
        playback_queue_ms = payload.get("playback_queue_ms")
        level = payload.get("level")
        if not level:
            level = "high" if isinstance(playback_queue_ms, (int, float)) and playback_queue_ms > 0 else "normal"
        self.playback_queue.set_backpressure(
            str(level),
            int(playback_queue_ms) if isinstance(playback_queue_ms, (int, float)) else None,
        )
        await self._flush_playback_queue()

    async def mark_client_audio_played(self, payload: dict) -> None:
        chunk_id = str(payload.get("chunk_id") or "")
        if not chunk_id:
            return
        chunk = await self.playback_queue.mark_played(chunk_id)
        if chunk is None:
            logger.debug("[%s] unknown played audio chunk: %s", self._task_id, chunk_id)
            return
        job_chunks = self._playback_job_chunks.get(chunk.tts_job_id)
        if job_chunks is not None:
            job_chunks.discard(chunk.chunk_id)
        await self._maybe_complete_playback_job(chunk.tts_job_id)
        await self._flush_playback_queue()

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
            asyncio.create_task(self._audio_worker(), name=f"{task_id}:realtime-vad"),
            asyncio.create_task(self._asr_worker(), name=f"{task_id}:realtime-asr"),
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

        for event in await self.audio_queue.put_audio(audio):
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
                        "drop_pre_class": event.pre_class,
                        "drop_vad_state": event.vad_state,
                        "utterance_id": event.utterance_id,
                        "first_dropped_seq": event.first_dropped_seq,
                        "last_dropped_seq": event.last_dropped_seq,
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

    async def _send_backpressure_event(self, scope: str, event) -> None:
        await self._send_json(
            self._event(
                "backpressure.applied",
                payload={
                    "scope": scope,
                    "level": "warning",
                    "reason": event.type,
                    "queue_ms": event.queue_ms,
                    "dropped_ms": event.dropped_ms,
                    "action": event.type,
                    "message": event.message,
                    "drop_pre_class": event.pre_class,
                    "drop_vad_state": event.vad_state,
                    "utterance_id": event.utterance_id,
                    "first_dropped_seq": event.first_dropped_seq,
                    "last_dropped_seq": event.last_dropped_seq,
                },
                type=event.type,
                queue_ms=event.queue_ms,
                dropped_ms=event.dropped_ms,
                message=event.message,
            )
        )

    async def _audio_worker(self) -> None:
        assert self._websocket is not None
        assert self._task_id is not None
        while not self._closed:
            frame = await self.audio_queue.get()
            frame = await self._classify_audio_frame(frame)
            if frame.sequence == 1 or frame.sequence % 10 == 0 or self.audio_queue.queued_ms:
                await self._send_json(
                    self._event(
                        "input.audio_dequeued",
                        payload={
                            "queue_ms": self.audio_queue.queued_ms,
                            "duration_ms": frame.duration_ms,
                            "is_silence": frame.is_silence,
                            "input_frame_index": frame.sequence,
                            "pre_class": frame.pre_class,
                            "vad_state": frame.vad_state,
                            "speech_active": frame.speech_active,
                        },
                        stage="asr_receiving_audio",
                        queue_ms=self.audio_queue.queued_ms,
                    )
                )
            segments = self.vad_segmenter.accept(frame, self._current_utterance_id())
            if self.vad_segmenter.consume_speech_started() and not self._speech_active:
                self._speech_active = True
                await self._send_json(
                    self._event(
                        "vad.speech_started",
                        payload={
                            "utterance_id": self._current_utterance_id(),
                            "speech_begin_ms": 0,
                            "source": "frame_smoothed_vad_rms",
                        },
                    )
                )
            for segment in segments:
                for queue_segment in self._split_asr_segment_for_queue(segment):
                    await self._queue_asr_segment(queue_segment)
                if segment.is_final:
                    self._utterance_index += 1

    async def _queue_asr_segment(self, segment: AsrSegment) -> None:
        await self._send_json(
            self._event(
                "asr.segment_committed",
                payload={
                    "segment_id": segment.segment_id,
                    "utterance_id": segment.utterance_id,
                    "first_input_frame_index": segment.first_frame_seq,
                    "last_input_frame_index": segment.last_frame_seq,
                    "duration_ms": segment.duration_ms,
                    "frame_count": segment.frame_count,
                    "is_final": segment.is_final,
                    "commit_reason": segment.commit_reason,
                    "queue_ms": self.asr_queue.queued_ms,
                },
            )
        )
        for event in await self.asr_queue.put(segment):
            await self._send_backpressure_event("asr", event)

    def _split_asr_segment_for_queue(self, segment: AsrSegment) -> list[AsrSegment]:
        if (
            segment.duration_ms <= MAX_ASR_SEGMENT_MS
            or self.audio_format != "pcm"
            or self.sample_rate <= 0
            or not segment.payload
        ):
            return [segment]

        max_payload_bytes = int(
            self.sample_rate * MAX_ASR_SEGMENT_MS / 1000
        ) * PCM_SAMPLE_BYTES
        max_payload_bytes -= max_payload_bytes % PCM_SAMPLE_BYTES
        if max_payload_bytes <= 0 or len(segment.payload) <= max_payload_bytes:
            return [segment]

        chunks = [
            segment.payload[index : index + max_payload_bytes]
            for index in range(0, len(segment.payload), max_payload_bytes)
        ]
        if len(chunks) <= 1:
            return [segment]

        split_segments: list[AsrSegment] = []
        base_segment_id = segment.segment_id or f"{segment.utterance_id}_seg"
        next_frame_seq = segment.first_frame_seq
        for index, payload in enumerate(chunks, start=1):
            duration_ms = self._estimate_duration_ms(payload)
            frame_count = max(
                1,
                int(round(duration_ms / max(1, settings.REALTIME_VAD_FRAME_MS))),
            )
            last_frame_seq = (
                segment.last_frame_seq
                if index == len(chunks)
                else next_frame_seq + frame_count - 1
            )
            split_segments.append(
                AsrSegment(
                    payload=payload,
                    duration_ms=duration_ms,
                    frame_count=frame_count,
                    utterance_id=segment.utterance_id,
                    first_frame_seq=next_frame_seq,
                    last_frame_seq=last_frame_seq,
                    segment_id=f"{base_segment_id}_part_{index}",
                    is_final=segment.is_final,
                    vad_source=segment.vad_source,
                    commit_reason=f"{segment.commit_reason}_slice",
                )
            )
            next_frame_seq = last_frame_seq + 1
        return split_segments

    async def _asr_worker(self) -> None:
        assert self._task_id is not None
        while not self._closed:
            segment = await self.asr_queue.get()
            events = await self._transcribe_events(
                segment.payload,
                self._task_id,
                is_final=segment.is_final,
            )
            if segment.is_final:
                if self._speech_active:
                    await self._send_json(
                        self._event(
                            "vad.speech_ended",
                            payload={
                                "utterance_id": segment.utterance_id,
                                "speech_end_ms": 0,
                                "source": segment.vad_source,
                            },
                        )
                    )
                self._speech_active = False
            for asr_event in events:
                await self._handle_asr_hypothesis(
                    asr_event,
                    utterance_id=segment.utterance_id,
                )

    async def _handle_asr_hypothesis(
        self,
        hypothesis: AsrHypothesis,
        *,
        utterance_id: str | None = None,
    ) -> None:
        assert self._task_id is not None
        utterance_id = utterance_id or self._current_utterance_id()

        if not hypothesis.is_final:
            return

        text = (hypothesis.text or "").strip()
        if not text:
            return

        self._hypothesis_index += 1
        hypothesis_id = f"{utterance_id}_final_{self._hypothesis_index}"
        self._tts_revision_id += 1
        tts_job_id = f"tts_{self._tts_revision_id}"

        await self._send_json(
            self._event(
                "asr_result",
                payload={
                    "protocol_event": "asr.utterance_final",
                    "utterance_id": utterance_id,
                    "hypothesis_id": hypothesis_id,
                    "text": text,
                    "is_final": True,
                    "speech_active": False,
                    "emotion": hypothesis.emotion,
                    "emotion_confidence": hypothesis.emotion_confidence,
                    "raw_rich_text": hypothesis.raw_rich_text,
                },
                stage="asr_text_received",
                text=text,
                is_final=True,
            )
        )
        await self._send_json(
            self._event(
                "asr.utterance_final",
                payload={
                    "utterance_id": utterance_id,
                    "revision_id": self._tts_revision_id,
                    "text": text,
                    "is_final": True,
                    "tts_job_id": tts_job_id,
                    "emotion": hypothesis.emotion,
                    "emotion_confidence": hypothesis.emotion_confidence,
                    "raw_rich_text": hypothesis.raw_rich_text,
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
            revision_id=self._tts_revision_id,
            text=text,
            voice_name=self.voice_name,
            parameters=job_parameters,
            priority="final",
        )
        queued_job, queue_events = await self.tts_jobs.put_with_result(job)
        queued_tts_job_id = f"tts_{queued_job.revision_id}"
        for event in queue_events:
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
                    "tts_job_id": queued_tts_job_id,
                    "revision_id": queued_job.revision_id,
                    "utterance_id": utterance_id,
                    "text": queued_job.text,
                    "text_chars": len(queued_job.text),
                    "voice_name": self.voice_name,
                    "config_version": self.config_version,
                    "priority": queued_job.priority,
                },
            )
        )
        await self._send_json(
            self._event(
                "asr.sentence_finalized",
                payload={"utterance_id": utterance_id, "text": text},
            )
        )

    async def _tts_worker(self) -> None:
        assert self._websocket is not None
        assert self._task_id is not None
        dispatcher = get_realtime_tts_dispatcher()
        prefetch_limit = max(1, int(settings.REALTIME_TTS_PREFETCH_JOBS))
        prefetched_jobs: list[PrefetchedTtsJob] = []
        try:
            while not self._closed:
                while len(prefetched_jobs) < prefetch_limit and not self._closed:
                    job = await self.tts_jobs.get() if not prefetched_jobs else self.tts_jobs.get_nowait()
                    if job is None:
                        break
                    prefetched_jobs.append(self._start_tts_prefetch(job, dispatcher))

                if not prefetched_jobs:
                    continue

                prefetched = prefetched_jobs.pop(0)
                await self._drain_prefetched_tts(prefetched)
        finally:
            for prefetched in prefetched_jobs:
                if not prefetched.task.done():
                    prefetched.task.cancel()
                with contextlib.suppress(BaseException):
                    await prefetched.task

    def _start_tts_prefetch(self, job: TtsJob, dispatcher) -> PrefetchedTtsJob:
        assert self._websocket is not None
        assert self._task_id is not None
        tts_job_id = f"tts_{job.revision_id}"
        audio_queue: asyncio.Queue = asyncio.Queue(
            maxsize=max(1, settings.REALTIME_TTS_AUDIO_QUEUE_SIZE)
        )
        synth_done = object()

        async def synthesize_job_audio() -> None:
            try:
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
                            "prefetch_jobs": settings.REALTIME_TTS_PREFETCH_JOBS,
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
                            "text": job.text,
                            "text_chars": len(job.text),
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
                    return

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
                            "prefetch_jobs": settings.REALTIME_TTS_PREFETCH_JOBS,
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
                            "prefetch_jobs": settings.REALTIME_TTS_PREFETCH_JOBS,
                        },
                        stage="tts_synthesizing",
                        text=job.text,
                        revision_id=job.revision_id,
                    )
                )
                async with lease:
                    async for chunk in self._synthesize(
                        job.text,
                        self._websocket,
                        self._task_id,
                        voice_name=job.voice_name,
                        parameters=job.parameters,
                    ):
                        if chunk:
                            await audio_queue.put(chunk)
            except BaseException as exc:
                if self._closed:
                    with contextlib.suppress(asyncio.QueueFull):
                        audio_queue.put_nowait(exc)
                else:
                    await audio_queue.put(exc)
            finally:
                if self._closed:
                    with contextlib.suppress(asyncio.QueueFull):
                        audio_queue.put_nowait(synth_done)
                else:
                    await audio_queue.put(synth_done)

        task = asyncio.create_task(
            synthesize_job_audio(),
            name=f"{self._task_id}:tts-prefetch:{job.revision_id}",
        )
        return PrefetchedTtsJob(
            job=job,
            tts_job_id=tts_job_id,
            audio_queue=audio_queue,
            done_sentinel=synth_done,
            task=task,
        )

    async def _drain_prefetched_tts(self, prefetched: PrefetchedTtsJob) -> None:
        job = prefetched.job
        emitted = False
        self._playback_job_meta[prefetched.tts_job_id] = job
        self._playback_job_chunks.setdefault(prefetched.tts_job_id, set())
        try:
            while True:
                chunk = await prefetched.audio_queue.get()
                if chunk is prefetched.done_sentinel:
                    break
                if isinstance(chunk, BaseException):
                    raise chunk
                if chunk:
                    await self._enqueue_playback_chunk(prefetched, chunk)
                    emitted = True
        finally:
            if not prefetched.task.done():
                prefetched.task.cancel()
            with contextlib.suppress(BaseException):
                await prefetched.task

        if emitted:
            self._playback_jobs_done_queueing.add(prefetched.tts_job_id)
            await self._maybe_complete_playback_job(prefetched.tts_job_id)

    async def _enqueue_playback_chunk(
        self,
        prefetched: PrefetchedTtsJob,
        payload: bytes,
    ) -> PlaybackChunk:
        self._audio_chunk_index += 1
        chunk = PlaybackChunk(
            chunk_id=f"{prefetched.tts_job_id}_chunk_{self._audio_chunk_index}",
            tts_job_id=prefetched.tts_job_id,
            revision_id=prefetched.job.revision_id,
            audio_chunk_index=self._audio_chunk_index,
            payload=payload,
            sample_rate=self.sample_rate,
        )
        self._playback_job_chunks.setdefault(prefetched.tts_job_id, set()).add(chunk.chunk_id)
        await self.playback_queue.put(chunk)
        await self._flush_playback_queue()
        return chunk

    async def _flush_playback_queue(self) -> None:
        async with self._playback_flush_lock:
            for chunk in await self.playback_queue.ready_chunks():
                await self._send_playback_chunk(chunk)

    async def _send_playback_chunk(self, chunk: PlaybackChunk) -> None:
        if chunk.tts_job_id not in self._first_audio_sent_jobs:
            await self._send_json(
                self._event(
                    "tts.first_audio",
                    payload={
                        "tts_job_id": chunk.tts_job_id,
                        "revision_id": chunk.revision_id,
                        "audio_chunk_index": chunk.audio_chunk_index,
                        "chunk_id": chunk.chunk_id,
                        "prefetched": True,
                    },
                )
            )
            self._first_audio_sent_jobs.add(chunk.tts_job_id)
        await self._send_audio_frame(chunk)

    async def _maybe_complete_playback_job(self, tts_job_id: str) -> None:
        if tts_job_id not in self._playback_jobs_done_queueing:
            return
        if self._playback_job_chunks.get(tts_job_id):
            return
        job = self._playback_job_meta.pop(tts_job_id, None)
        self._playback_jobs_done_queueing.discard(tts_job_id)
        self._playback_job_chunks.pop(tts_job_id, None)
        self._first_audio_sent_jobs.discard(tts_job_id)
        if job is None:
            return
        await self._send_json(
            self._event(
                "tts.job_completed",
                payload={
                    "tts_job_id": tts_job_id,
                    "revision_id": job.revision_id,
                    "text": job.text,
                    "text_chars": len(job.text),
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
                    "text": job.text,
                    "text_chars": len(job.text),
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

    async def _transcribe_events(self, audio: bytes, task_id: str, is_final: bool = False) -> list[AsrHypothesis]:
        if not is_final or not audio:
            return []

        incoming = self.asr_service._convert_audio_bytes_to_array(
            audio,
            self.audio_format,
            self.sample_rate,
            task_id,
        )
        if incoming.size == 0:
            return []

        duration_ms = int(len(incoming) / self.sample_rate * 1000)
        hypothesis = await self._transcribe_final_utterance(incoming, audio, task_id)
        if hypothesis is None:
            return []
        return [
            replace(
                hypothesis,
                is_final=True,
                kind="end",
                time_ms=duration_ms,
                begin_time_ms=0,
                speech_end_ms=duration_ms,
                speech_active=False,
                vad_source=hypothesis.vad_source or "utterance_vad",
            )
        ]

    async def _transcribe_final_utterance(
        self,
        audio_array: np.ndarray,
        audio_bytes: bytes,
        task_id: str,
    ) -> AsrHypothesis | None:
        if hasattr(self.session_asr_engine, "transcribe_array_with_metadata"):
            result = await run_sync(
                self.session_asr_engine.transcribe_array_with_metadata,
                audio_array,
                sample_rate=self.sample_rate,
                enable_itn=self.asr_params.get("enable_inverse_text_normalization", True),
                enable_vad=False,
                enable_emotion=self.asr_params.get("enable_emotion", False),
                return_rich_text=self.asr_params.get("return_rich_text", False),
            )
            text = (getattr(result, "text", "") or "").strip()
            if not text:
                return None
            return AsrHypothesis(
                text=text,
                is_final=True,
                kind="end",
                emotion=getattr(result, "emotion", None),
                emotion_confidence=getattr(result, "emotion_confidence", None),
                raw_rich_text=getattr(result, "raw_rich_text", None),
            )

        if hasattr(self.session_asr_engine, "transcribe_array"):
            text = (
                await run_sync(
                    self.session_asr_engine.transcribe_array,
                    audio_array,
                    sample_rate=self.sample_rate,
                    enable_itn=self.asr_params.get("enable_inverse_text_normalization", True),
                    enable_vad=False,
                )
                or ""
            ).strip()
            return AsrHypothesis(text=text, is_final=True, kind="end") if text else None

        result_text, _, _, _, _, _ = await self.asr_service._process_audio_chunk(
            audio_bytes,
            {},
            {},
            self.asr_params,
            0,
            task_id,
            is_final=True,
            session_engine=self.session_asr_engine,
        )
        text = (result_text or "").strip()
        return AsrHypothesis(text=text, is_final=True, kind="end") if text else None

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
        rms_voice = self._is_voice_by_rms(self._pcm_bytes_to_float_array(audio))
        self._input_frame_index += 1
        return AudioFrame(
            payload=audio,
            duration_ms=duration_ms,
            is_silence=not rms_voice,
            sequence=self._input_frame_index,
            pre_class="rms_voice" if rms_voice else "rms_silence",
            vad_state="pending",
            speech_active=False,
        )

    async def _classify_audio_frame(self, frame: AudioFrame) -> AudioFrame:
        audio_array = self._pcm_bytes_to_float_array(frame.payload)
        rms_voice = self._is_voice_by_rms(audio_array)
        vad_event = await self.realtime_vad_detector.accept_audio(audio_array)
        detector_started = vad_event.speech_begin_ms is not None
        detector_active = bool(vad_event.is_speech_active)
        detector_ended = (
            vad_event.speech_end_ms is not None
            and not detector_started
            and not detector_active
        )
        is_voice = (
            rms_voice or detector_started or detector_active
        ) and not detector_ended
        return replace(
            frame,
            is_silence=not is_voice,
            pre_class="rms_voice" if rms_voice else "rms_silence",
            vad_state="speech" if is_voice else "silence",
            speech_active=detector_started or detector_active,
        )

    def _estimate_duration_ms(self, audio: bytes) -> int:
        if self.audio_format != "pcm" or self.sample_rate <= 0:
            return 20
        samples = max(1, len(audio) // 2)
        return max(1, int(samples / self.sample_rate * 1000))

    def _pcm_bytes_to_float_array(self, audio: bytes) -> np.ndarray:
        if self.audio_format != "pcm" or len(audio) < 2:
            return np.array([], dtype=np.float32)
        pcm = np.frombuffer(audio, dtype=np.int16)
        if pcm.size == 0:
            return np.array([], dtype=np.float32)
        return pcm.astype(np.float32) / 32768.0

    def _is_voice_by_rms(self, audio_array: np.ndarray) -> bool:
        if audio_array.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(audio_array))))
        return rms >= settings.ASR_NEARFIELD_RMS_THRESHOLD

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

    async def _send_audio_frame(self, chunk: PlaybackChunk) -> None:
        """Send audio metadata and binary bytes without interleaving events."""
        if self._websocket is None:
            return
        stats = self.playback_queue.stats()
        async with self._send_lock:
            await self._websocket.send_json(
                self._event(
                    "tts.audio_chunk",
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "tts_job_id": chunk.tts_job_id,
                        "revision_id": chunk.revision_id,
                        "audio_chunk_index": chunk.audio_chunk_index,
                        "bytes": len(chunk.payload),
                        "sample_rate": chunk.sample_rate,
                        "format": chunk.format,
                        "pending": stats["pending"],
                        "in_flight": stats["in_flight"],
                    },
                )
            )
            await self._websocket.send_bytes(chunk.payload)

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
                realtime_config = (
                    asr_tts_session.realtime_config_snapshot()
                    if hasattr(asr_tts_session, "realtime_config_snapshot")
                    else None
                )
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
                            "realtime_config": realtime_config,
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
                        realtime_config=realtime_config,
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
            elif event == "client.audio_backpressure":
                payload = data.get("payload") or {}
                if asr_tts_session is not None and hasattr(asr_tts_session, "update_client_audio_backpressure"):
                    await asr_tts_session.update_client_audio_backpressure(payload)
                logger.debug(
                    "[%s] client audio backpressure received: level=%s queue_ms=%s",
                    task_id,
                    payload.get("level"),
                    payload.get("playback_queue_ms"),
                )
            elif event == "client.audio_played":
                payload = data.get("payload") or {}
                if asr_tts_session is not None and hasattr(asr_tts_session, "mark_client_audio_played"):
                    await asr_tts_session.mark_client_audio_played(payload)
                logger.debug(
                    "[%s] client audio played received: chunk_id=%s",
                    task_id,
                    payload.get("chunk_id"),
                )
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
