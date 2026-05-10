# -*- coding: utf-8 -*-
"""Bounded queues and drop policies for realtime voice audio input."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Deque, Optional

from .types import AsrSegment, AudioFrame, BackpressureEvent


class BoundedAudioQueue:
    """Audio queue budgeted by duration instead of item count.

    Realtime audio can arrive faster than ASR/TTS can consume it. The queue
    keeps latency bounded with a layered policy: VAD silence, RMS pre-silence,
    already-covered speech, speech-like pending frames, then oldest speech.
    """

    def __init__(self, high_watermark_ms: int, max_ms: int, preserve_speech: bool = True):
        self.high_watermark_ms = high_watermark_ms
        self.max_ms = max_ms
        self.preserve_speech = preserve_speech
        self._frames: Deque[AudioFrame] = deque()
        self._queued_ms = 0
        self._condition = asyncio.Condition()

    @property
    def queued_ms(self) -> int:
        return self._queued_ms

    async def put(self, frame: AudioFrame) -> list[BackpressureEvent]:
        events: list[BackpressureEvent] = []
        async with self._condition:
            incoming_drop_reason = _incoming_drop_reason(frame)
            if self._queued_ms >= self.high_watermark_ms and incoming_drop_reason is not None:
                events.append(
                    _drop_event(incoming_drop_reason, self._queued_ms, frame)
                )
                return events

            self._frames.append(frame)
            self._queued_ms += frame.duration_ms

            if self._queued_ms >= self.high_watermark_ms:
                events.append(
                    BackpressureEvent(type="input_throttle", queue_ms=self._queued_ms)
                )

            events.extend(self._drop_until_within_budget())

            self._condition.notify()
        return events

    async def get(self) -> AudioFrame:
        async with self._condition:
            while not self._frames:
                await self._condition.wait()
            frame = self._frames.popleft()
            self._queued_ms = max(0, self._queued_ms - frame.duration_ms)
            return frame

    def get_nowait(self) -> Optional[AudioFrame]:
        if not self._frames:
            return None
        frame = self._frames.popleft()
        self._queued_ms = max(0, self._queued_ms - frame.duration_ms)
        return frame

    def clear(self) -> None:
        self._frames.clear()
        self._queued_ms = 0

    def _drop_until_within_budget(self) -> list[BackpressureEvent]:
        events: list[BackpressureEvent] = []
        while self._queued_ms > self.max_ms and self._frames:
            index, reason = self._find_best_drop_candidate()
            if self.preserve_speech and reason == "drop_oldest_speech":
                frame = self._frames[index]
                events.append(
                    BackpressureEvent(
                        type="input_preserve_speech_backpressure",
                        queue_ms=self._queued_ms,
                        message=f"preserve speech seq={frame.sequence}",
                        vad_state=frame.vad_state,
                        pre_class=frame.pre_class,
                        utterance_id=frame.utterance_id,
                        first_dropped_seq=frame.sequence or None,
                        last_dropped_seq=frame.sequence or None,
                    )
                )
                break
            frame = self._remove_at(index)
            self._queued_ms = max(0, self._queued_ms - frame.duration_ms)
            events.append(_drop_event(reason, self._queued_ms, frame))
        return events

    def _find_best_drop_candidate(self) -> tuple[int, str]:
        best_index = 0
        best_priority = 10_000
        best_reason = "drop_oldest_speech"
        for index, frame in enumerate(self._frames):
            priority, reason = _drop_priority(frame)
            if priority < best_priority:
                best_index = index
                best_priority = priority
                best_reason = reason
                if priority == 0:
                    break
        return best_index, best_reason

    def _remove_at(self, index: int) -> AudioFrame:
        if index == 0:
            return self._frames.popleft()
        self._frames.rotate(-index)
        frame = self._frames.popleft()
        self._frames.rotate(index)
        return frame


def _incoming_drop_reason(frame: AudioFrame) -> Optional[str]:
    """Only drop incoming frames early when they are silence-like."""
    priority, reason = _drop_priority(frame)
    return reason if priority <= 1 else None


def _drop_priority(frame: AudioFrame) -> tuple[int, str]:
    """Lower priority values are safer to drop under input backpressure."""
    if frame.vad_state == "silence":
        return 0, "drop_vad_silence"
    if _is_pre_silence(frame):
        return 1, "drop_pre_silence"
    if frame.covered_by_asr:
        return 2, "drop_covered_speech"
    if _is_speech_like(frame):
        return 3, "drop_speech_like"
    return 4, "drop_oldest_speech"


def _is_pre_silence(frame: AudioFrame) -> bool:
    if frame.speech_active or frame.vad_state in {"speech", "active"}:
        return False
    return frame.pre_class == "rms_silence" or (
        frame.vad_state in {"pending", "unknown"} and frame.is_silence
    )


def _is_speech_like(frame: AudioFrame) -> bool:
    if frame.speech_active or frame.vad_state in {"speech", "active"}:
        return False
    return frame.vad_state == "speech_like" or frame.pre_class == "rms_voice"


def _drop_event(reason: str, queue_ms: int, frame: AudioFrame) -> BackpressureEvent:
    return BackpressureEvent(
        type=reason,
        queue_ms=queue_ms,
        dropped_ms=frame.duration_ms,
        message=f"{reason} seq={frame.sequence}" if frame.sequence else reason,
        vad_state=frame.vad_state,
        pre_class=frame.pre_class,
        utterance_id=frame.utterance_id,
        first_dropped_seq=frame.sequence or None,
        last_dropped_seq=frame.sequence or None,
    )


class TtsJobQueue:
    """Small revision-aware FIFO TTS job queue.

    Synthesis may prefetch later jobs, but queue admission must preserve text
    revision order. Final revisions are not allowed to jump ahead of stable
    revisions that were committed earlier.
    """

    def __init__(self, maxsize: int, drop_on_overload: bool = False, hard_limit: int | None = None):
        self.maxsize = max(1, maxsize)
        self.hard_limit = max(self.maxsize, int(hard_limit or self.maxsize * 2))
        self.drop_on_overload = drop_on_overload
        self._jobs: Deque = deque()
        self._condition = asyncio.Condition()

    async def put(self, job) -> list[BackpressureEvent]:
        _, events = await self.put_with_result(job)
        return events

    async def put_with_result(self, job) -> tuple[object, list[BackpressureEvent]]:
        events: list[BackpressureEvent] = []
        async with self._condition:
            if job.priority == "stable" and len(self._jobs) >= self.maxsize:
                stable_tail = []
                while self._jobs and self._jobs[-1].priority == "stable":
                    stable_tail.append(self._jobs.pop())
                if stable_tail:
                    stable_tail.reverse()
                    merged_text = "".join(existing.text for existing in stable_tail) + job.text
                    job = type(job)(
                        revision_id=job.revision_id,
                        text=merged_text,
                        voice_name=job.voice_name,
                        parameters=job.parameters,
                        priority=job.priority,
                    )
                    events.append(
                        BackpressureEvent(
                            type="tts_jobs_coalesced",
                            message=f"merged={len(stable_tail) + 1} revision={job.revision_id}",
                        )
                    )
            while self.drop_on_overload and len(self._jobs) >= self.maxsize:
                dropped = self._drop_lowest_priority()
                events.append(
                    BackpressureEvent(
                        type="tts_job_dropped",
                        message=f"revision={dropped.revision_id}",
                    )
                )
            if not self.drop_on_overload and len(self._jobs) >= self.maxsize:
                events.append(
                    BackpressureEvent(
                        type="tts_queue_preserved",
                        message=f"queued={len(self._jobs) + 1} revision={job.revision_id}",
                    )
                )
            self._jobs.append(job)
            while len(self._jobs) > self.hard_limit:
                dropped = self._drop_lowest_priority()
                events.append(
                    BackpressureEvent(
                        type="tts_job_dropped_hard_limit",
                        message=f"revision={dropped.revision_id}",
                    )
                )
            self._condition.notify()
        return job, events

    async def get(self):
        async with self._condition:
            while not self._jobs:
                await self._condition.wait()
            return self._jobs.popleft()

    def get_nowait(self):
        if not self._jobs:
            return None
        return self._jobs.popleft()

    def clear(self) -> None:
        self._jobs.clear()

    def _drop_lowest_priority(self):
        for index, job in enumerate(self._jobs):
            if job.priority == "stable":
                return self._remove_at(index)
        return self._jobs.popleft()

    def _remove_at(self, index: int):
        if index == 0:
            return self._jobs.popleft()
        self._jobs.rotate(-index)
        job = self._jobs.popleft()
        self._jobs.rotate(index)
        return job


class AsrSegmentQueue:
    """Bounded ASR queue that coalesces speech before dropping it."""

    def __init__(
        self,
        high_watermark_ms: int,
        max_ms: int,
        preserve_speech: bool = True,
        hard_max_ms: int | None = None,
    ):
        self.high_watermark_ms = max(1, high_watermark_ms)
        self.max_ms = max(self.high_watermark_ms, max_ms)
        self.hard_max_ms = max(self.max_ms, int(hard_max_ms or self.max_ms * 2))
        self.preserve_speech = preserve_speech
        self._segments: Deque[AsrSegment] = deque()
        self._queued_ms = 0
        self._condition = asyncio.Condition()

    @property
    def queued_ms(self) -> int:
        return self._queued_ms

    async def put(self, segment: AsrSegment) -> list[BackpressureEvent]:
        events: list[BackpressureEvent] = []
        async with self._condition:
            if self._queued_ms >= self.high_watermark_ms and not segment.is_final:
                if self._segments and not self._segments[-1].is_final:
                    previous = self._segments.pop()
                    self._queued_ms = max(0, self._queued_ms - previous.duration_ms)
                    segment = AsrSegment(
                        payload=previous.payload + segment.payload,
                        duration_ms=previous.duration_ms + segment.duration_ms,
                        frame_count=previous.frame_count + segment.frame_count,
                        utterance_id=segment.utterance_id,
                        first_frame_seq=previous.first_frame_seq,
                        last_frame_seq=segment.last_frame_seq,
                        segment_id=f"{previous.segment_id}+{segment.segment_id}".strip("+"),
                        is_final=False,
                        vad_source=segment.vad_source,
                        commit_reason="pressure_coalesced",
                    )
                    events.append(
                        BackpressureEvent(
                            type="asr_segments_coalesced",
                            queue_ms=self._queued_ms,
                            message=f"frames={segment.first_frame_seq}-{segment.last_frame_seq}",
                            utterance_id=segment.utterance_id,
                            first_dropped_seq=segment.first_frame_seq,
                            last_dropped_seq=segment.last_frame_seq,
                        )
                    )
                else:
                    events.append(BackpressureEvent(type="asr_input_throttle", queue_ms=self._queued_ms))

            self._segments.append(segment)
            self._queued_ms += segment.duration_ms
            events.extend(self._drop_until_within_budget())
            self._condition.notify()
        return events

    async def get(self) -> AsrSegment:
        async with self._condition:
            while not self._segments:
                await self._condition.wait()
            segment = self._segments.popleft()
            self._queued_ms = max(0, self._queued_ms - segment.duration_ms)
            return segment

    def clear(self) -> None:
        self._segments.clear()
        self._queued_ms = 0

    def _drop_until_within_budget(self) -> list[BackpressureEvent]:
        events: list[BackpressureEvent] = []
        while self._queued_ms > self.max_ms and self._segments:
            if self.preserve_speech and self._queued_ms <= self.hard_max_ms:
                segment = self._segments[0]
                events.append(
                    BackpressureEvent(
                        type="asr_preserve_speech_backpressure",
                        queue_ms=self._queued_ms,
                        message=f"frames={segment.first_frame_seq}-{segment.last_frame_seq}",
                        utterance_id=segment.utterance_id,
                        first_dropped_seq=segment.first_frame_seq,
                        last_dropped_seq=segment.last_frame_seq,
                    )
                )
                break
            index = self._find_drop_index()
            segment = self._remove_at(index)
            self._queued_ms = max(0, self._queued_ms - segment.duration_ms)
            events.append(
                BackpressureEvent(
                    type="drop_asr_speech_hard_limit" if self.preserve_speech else "drop_asr_speech",
                    queue_ms=self._queued_ms,
                    dropped_ms=segment.duration_ms,
                    message=f"frames={segment.first_frame_seq}-{segment.last_frame_seq}",
                    utterance_id=segment.utterance_id,
                    first_dropped_seq=segment.first_frame_seq,
                    last_dropped_seq=segment.last_frame_seq,
                )
            )
        return events

    def _find_drop_index(self) -> int:
        for index, segment in enumerate(self._segments):
            if not segment.is_final:
                return index
        return 0

    def _remove_at(self, index: int) -> AsrSegment:
        if index == 0:
            return self._segments.popleft()
        self._segments.rotate(-index)
        segment = self._segments.popleft()
        self._segments.rotate(index)
        return segment
