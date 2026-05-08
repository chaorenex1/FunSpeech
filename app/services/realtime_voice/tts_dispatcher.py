# -*- coding: utf-8 -*-
"""Global admission control for realtime TTS synthesis."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter

from ...core.config import settings


@dataclass(frozen=True)
class TtsAdmission:
    """Result of trying to enter the global realtime TTS synthesis pool."""

    accepted: bool
    reason: str = ""
    queue_wait_ms: int = 0
    active: int = 0
    waiting: int = 0


class TtsLease:
    """A held realtime TTS synthesis slot."""

    def __init__(self, dispatcher: "RealtimeTTSDispatcher", admission: TtsAdmission):
        self.dispatcher = dispatcher
        self.admission = admission
        self._released = False

    async def __aenter__(self) -> "TtsLease":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self.dispatcher.release()


class RealtimeTTSDispatcher:
    """Limit global realtime TTS concurrency and expose queue state."""

    def __init__(
        self,
        max_inflight: int = 1,
        max_queue_size: int = 8,
        queue_timeout_ms: int = 3000,
    ):
        self.max_inflight = max(1, int(max_inflight))
        self.max_queue_size = max(0, int(max_queue_size))
        self.queue_timeout_ms = max(1, int(queue_timeout_ms))
        self._semaphore = asyncio.Semaphore(self.max_inflight)
        self._lock = asyncio.Lock()
        self._active = 0
        self._waiting = 0

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiting

    async def acquire(self) -> TtsLease | TtsAdmission:
        """Acquire a global TTS slot or return a rejected admission."""
        started = perf_counter()
        async with self._lock:
            if self._active >= self.max_inflight and self._waiting >= self.max_queue_size:
                return TtsAdmission(
                    accepted=False,
                    reason="global_tts_queue_full",
                    active=self._active,
                    waiting=self._waiting,
                )
            self._waiting += 1

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.queue_timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            async with self._lock:
                self._waiting = max(0, self._waiting - 1)
                active = self._active
                waiting = self._waiting
            return TtsAdmission(
                accepted=False,
                reason="global_tts_queue_timeout",
                queue_wait_ms=int((perf_counter() - started) * 1000),
                active=active,
                waiting=waiting,
            )

        async with self._lock:
            self._waiting = max(0, self._waiting - 1)
            self._active += 1
            admission = TtsAdmission(
                accepted=True,
                queue_wait_ms=int((perf_counter() - started) * 1000),
                active=self._active,
                waiting=self._waiting,
            )
        return TtsLease(self, admission)

    async def release(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)
        self._semaphore.release()


_dispatcher: RealtimeTTSDispatcher | None = None


def get_realtime_tts_dispatcher() -> RealtimeTTSDispatcher:
    """Return the process-wide realtime TTS dispatcher."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = RealtimeTTSDispatcher(
            max_inflight=settings.REALTIME_TTS_GLOBAL_MAX_INFLIGHT,
            max_queue_size=settings.REALTIME_TTS_GLOBAL_QUEUE_SIZE,
            queue_timeout_ms=settings.REALTIME_TTS_QUEUE_TIMEOUT_MS,
        )
    return _dispatcher


def reset_realtime_tts_dispatcher() -> None:
    """Reset the global dispatcher, primarily for tests."""
    global _dispatcher
    _dispatcher = None

