# -*- coding: utf-8 -*-
"""PCM frame pacing for realtime voice output."""

from __future__ import annotations

import asyncio
from time import monotonic
from typing import AsyncIterator


class AudioPacer:
    """Split PCM chunks into fixed-duration frames and send them by audio clock."""

    def __init__(self, sample_rate: int, frame_ms: int = 20):
        self.sample_rate = int(sample_rate or 16000)
        self.frame_ms = max(10, int(frame_ms or 20))
        self.bytes_per_frame = max(2, int(self.sample_rate * self.frame_ms / 1000) * 2)
        self._pending = b""
        self._next_send_at: float | None = None

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
        now = monotonic()
        if self._next_send_at is None or self._next_send_at < now - 0.2:
            self._next_send_at = now
        delay = self._next_send_at - now
        if delay > 0:
            await asyncio.sleep(delay)
        self._next_send_at += self.frame_ms / 1000

