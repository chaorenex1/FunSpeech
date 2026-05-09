# -*- coding: utf-8 -*-
"""PCM frame pacing for realtime voice output."""

from __future__ import annotations

import asyncio
from time import monotonic
from typing import AsyncIterator

from ...core.config import settings


class AudioPacer:
    """Split PCM chunks into fixed-duration frames and send them by audio clock."""

    def __init__(
        self,
        sample_rate: int,
        frame_ms: int = 20,
        burst_ms: int | None = None,
        target_queue_ms: int | None = None,
        high_queue_ms: int | None = None,
        max_backpressure_sleep_ms: int | None = None,
    ):
        self.sample_rate = int(sample_rate or 16000)
        self.frame_ms = max(10, int(frame_ms or 20))
        self.burst_ms = max(0, int(settings.REALTIME_PACER_BURST_MS if burst_ms is None else burst_ms))
        self.target_queue_ms = max(
            0,
            int(
                settings.REALTIME_OUTPUT_TARGET_QUEUE_MS
                if target_queue_ms is None
                else target_queue_ms
            ),
        )
        self.high_queue_ms = max(
            self.target_queue_ms,
            int(settings.REALTIME_OUTPUT_HIGH_QUEUE_MS if high_queue_ms is None else high_queue_ms),
        )
        self.max_backpressure_sleep_ms = max(
            0,
            int(
                settings.REALTIME_OUTPUT_BACKPRESSURE_MAX_SLEEP_MS
                if max_backpressure_sleep_ms is None
                else max_backpressure_sleep_ms
            ),
        )
        self.bytes_per_frame = max(2, int(self.sample_rate * self.frame_ms / 1000) * 2)
        self._pending = b""
        self._next_send_at: float | None = None
        self._client_queue_ms = 0

    def update_client_queue_ms(self, queue_ms: int | None) -> None:
        if queue_ms is None:
            return
        self._client_queue_ms = max(0, int(queue_ms))

    async def iter_frames(self, chunk: bytes) -> AsyncIterator[bytes]:
        if not chunk:
            return
        self._pending += chunk
        while len(self._pending) >= self.bytes_per_frame:
            frame = self._pending[: self.bytes_per_frame]
            self._pending = self._pending[self.bytes_per_frame :]
            await self._pace()
            yield frame

    async def flush(self) -> AsyncIterator[bytes]:
        if self._pending:
            await self._pace()
            frame = self._pending
            self._pending = b""
            yield frame

    async def _pace(self) -> None:
        if self._client_queue_ms > self.target_queue_ms and self.max_backpressure_sleep_ms > 0:
            pressure_ratio = min(
                1.0,
                (self._client_queue_ms - self.target_queue_ms)
                / max(1, self.high_queue_ms - self.target_queue_ms),
            )
            await asyncio.sleep((self.max_backpressure_sleep_ms * pressure_ratio) / 1000)

        now = monotonic()
        burst_seconds = self.burst_ms / 1000
        if self._next_send_at is None or self._next_send_at < now - burst_seconds:
            self._next_send_at = now - burst_seconds
        delay = self._next_send_at - now
        if delay > 0:
            await asyncio.sleep(delay)
        self._next_send_at += self.frame_ms / 1000
