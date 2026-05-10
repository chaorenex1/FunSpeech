# -*- coding: utf-8 -*-
"""Client playback cache for realtime TTS audio chunks."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from time import monotonic
from typing import Deque


@dataclass(frozen=True)
class PlaybackChunk:
    """A raw TTS PCM chunk waiting for client playback."""

    chunk_id: str
    tts_job_id: str
    revision_id: int
    audio_chunk_index: int
    payload: bytes
    sample_rate: int
    format: str = "PCM"
    created_at: float = field(default_factory=monotonic)


class TtsPlaybackQueue:
    """Bounded cache with a multi-chunk in-flight send window."""

    def __init__(
        self,
        maxsize: int,
        max_inflight: int,
        *,
        backpressure_sleep_ms: int = 0,
    ):
        self.maxsize = max(1, int(maxsize))
        self.max_inflight = max(1, int(max_inflight))
        self.backpressure_sleep_ms = max(0, int(backpressure_sleep_ms))
        self._pending: Deque[PlaybackChunk] = deque()
        self._in_flight: dict[str, PlaybackChunk] = {}
        self._completed: set[str] = set()
        self._backpressure_level = "normal"
        self._playback_queue_ms = 0
        self._condition = asyncio.Condition()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def in_flight_count(self) -> int:
        return len(self._in_flight)

    @property
    def backpressure_level(self) -> str:
        return self._backpressure_level

    async def put(self, chunk: PlaybackChunk) -> None:
        await self.wait_if_backpressured()
        async with self._condition:
            while len(self._pending) >= self.maxsize:
                await self._condition.wait()
            self._pending.append(chunk)
            self._condition.notify_all()

    async def ready_chunks(self) -> list[PlaybackChunk]:
        async with self._condition:
            chunks: list[PlaybackChunk] = []
            while self._pending and len(self._in_flight) < self.max_inflight:
                chunk = self._pending.popleft()
                self._in_flight[chunk.chunk_id] = chunk
                chunks.append(chunk)
            if chunks:
                self._condition.notify_all()
            return chunks

    async def mark_played(self, chunk_id: str) -> PlaybackChunk | None:
        async with self._condition:
            chunk = self._in_flight.pop(chunk_id, None)
            if chunk is not None:
                self._completed.add(chunk_id)
                self._condition.notify_all()
            return chunk

    def set_backpressure(self, level: str, playback_queue_ms: int | None = None) -> None:
        normalized = (level or "normal").lower()
        if normalized not in {"normal", "high"}:
            normalized = "high"
        self._backpressure_level = normalized
        if playback_queue_ms is not None:
            self._playback_queue_ms = max(0, int(playback_queue_ms))

    async def wait_if_backpressured(self) -> None:
        if self._backpressure_level == "high" and self.backpressure_sleep_ms > 0:
            await asyncio.sleep(self.backpressure_sleep_ms / 1000)

    def stats(self) -> dict:
        return {
            "pending": len(self._pending),
            "in_flight": len(self._in_flight),
            "completed": len(self._completed),
            "backpressure_level": self._backpressure_level,
            "playback_queue_ms": self._playback_queue_ms,
        }

    def clear(self) -> None:
        self._pending.clear()
        self._in_flight.clear()
        self._completed.clear()
