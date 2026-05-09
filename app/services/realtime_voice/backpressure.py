# -*- coding: utf-8 -*-
"""Bounded queues and drop policies for realtime voice audio input."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Deque, Optional

from .types import AudioFrame, BackpressureEvent


class BoundedAudioQueue:
    """Audio queue budgeted by duration instead of item count.

    Realtime audio can arrive faster than ASR/TTS can consume it. This queue
    keeps latency bounded by preferring to drop silence and old uncommitted
    frames before allowing unbounded backlog.
    """

    def __init__(self, high_watermark_ms: int, max_ms: int):
        self.high_watermark_ms = high_watermark_ms
        self.max_ms = max_ms
        self._frames: Deque[AudioFrame] = deque()
        self._queued_ms = 0
        self._condition = asyncio.Condition()

    @property
    def queued_ms(self) -> int:
        return self._queued_ms

    async def put(self, frame: AudioFrame) -> list[BackpressureEvent]:
        events: list[BackpressureEvent] = []
        async with self._condition:
            if self._queued_ms >= self.high_watermark_ms and _is_backpressure_silence(frame):
                events.append(
                    BackpressureEvent(
                        type="dropped_silence",
                        queue_ms=self._queued_ms,
                        dropped_ms=frame.duration_ms,
                    )
                )
                return events

            self._frames.append(frame)
            self._queued_ms += frame.duration_ms

            if self._queued_ms >= self.high_watermark_ms:
                events.append(
                    BackpressureEvent(type="input_throttle", queue_ms=self._queued_ms)
                )

            dropped_ms = self._drop_until_within_budget()
            if dropped_ms:
                events.append(
                    BackpressureEvent(
                        type="dropped_audio",
                        queue_ms=self._queued_ms,
                        dropped_ms=dropped_ms,
                    )
                )

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

    def _drop_until_within_budget(self) -> int:
        dropped_ms = 0
        while self._queued_ms > self.max_ms and self._frames:
            index = self._find_oldest_silence_index()
            if index is None:
                index = 0
            frame = self._remove_at(index)
            dropped_ms += frame.duration_ms
            self._queued_ms = max(0, self._queued_ms - frame.duration_ms)
        return dropped_ms

    def _find_oldest_silence_index(self) -> Optional[int]:
        for index, frame in enumerate(self._frames):
            if _is_backpressure_silence(frame):
                return index
        return None

    def _remove_at(self, index: int) -> AudioFrame:
        if index == 0:
            return self._frames.popleft()
        self._frames.rotate(-index)
        frame = self._frames.popleft()
        self._frames.rotate(index)
        return frame


def _is_backpressure_silence(frame: AudioFrame) -> bool:
    """Use VAD state when available, with RMS silence as a fallback."""
    if frame.vad_state == "silence":
        return True
    if frame.vad_state in {"speech", "speech_like", "active"} or frame.speech_active:
        return False
    return frame.is_silence


class TtsJobQueue:
    """Small revision-aware TTS job queue.

    It keeps the newest stable/final jobs and drops stale speculative work when
    synthesis cannot keep up.
    """

    def __init__(self, maxsize: int):
        self.maxsize = max(1, maxsize)
        self._jobs: Deque = deque()
        self._condition = asyncio.Condition()

    async def put(self, job) -> list[BackpressureEvent]:
        events: list[BackpressureEvent] = []
        async with self._condition:
            if job.priority == "stable":
                self._jobs = deque(
                    existing
                    for existing in self._jobs
                    if not (
                        existing.priority == "stable"
                        and existing.revision_id < job.revision_id
                    )
                )
            while len(self._jobs) >= self.maxsize:
                dropped = self._drop_lowest_priority()
                events.append(
                    BackpressureEvent(
                        type="tts_job_dropped",
                        message=f"revision={dropped.revision_id}",
                    )
                )
            self._jobs.append(job)
            self._condition.notify()
        return events

    async def get(self):
        async with self._condition:
            while not self._jobs:
                await self._condition.wait()
            final_jobs = [i for i, job in enumerate(self._jobs) if job.priority == "final"]
            if final_jobs:
                return self._remove_at(final_jobs[0])
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
