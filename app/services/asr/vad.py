# -*- coding: utf-8 -*-
"""Streaming VAD endpoint detection for pseudo-streaming ASR.

网关不再持有本地 FSMN-VAD 模型(推理已迁移到 services/funasr 子服务),
因此这里只保留基于 RMS/静音时长的端点检测, 阈值仍由 ASR_VAD_* 配置控制。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List, Optional

import numpy as np

from ...core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class VADEvent:
    """Normalized VAD boundary event."""

    speech_begin_ms: Optional[int] = None
    speech_end_ms: Optional[int] = None
    is_speech_active: bool = False
    raw_segments: Optional[List[List[int]]] = None
    source: str = "vad"


class StreamingVADEndpointDetector:
    """RMS/silence based endpoint detector."""

    def __init__(self, engine, sample_rate: int):
        self.engine = engine
        self.sample_rate = sample_rate
        self.audio_time_ms = 0
        self.speech_active = False
        self.nearfield_ms = 0
        self.silence_ms = 0

    async def accept_audio(
        self,
        audio_array: np.ndarray,
        is_final: bool = False,
    ) -> VADEvent:
        chunk_duration_ms = int(len(audio_array) / self.sample_rate * 1000)
        event = VADEvent(is_speech_active=self.speech_active, raw_segments=[])

        fallback_event = self._fallback_event(audio_array, chunk_duration_ms)
        event = self._merge_events(event, fallback_event)
        self.audio_time_ms += chunk_duration_ms
        event.is_speech_active = self.speech_active
        return event

    def _fallback_event(
        self, audio_array: np.ndarray, chunk_duration_ms: int
    ) -> VADEvent:
        if len(audio_array) == 0:
            return VADEvent(is_speech_active=self.speech_active, source="fallback")

        rms = float(np.sqrt(np.mean(np.square(audio_array)))) if len(audio_array) else 0.0
        is_voice = rms >= settings.ASR_NEARFIELD_RMS_THRESHOLD
        event = VADEvent(is_speech_active=self.speech_active, source="fallback")

        if is_voice:
            self.nearfield_ms += chunk_duration_ms
            self.silence_ms = 0
            if (
                not self.speech_active
                and self.nearfield_ms >= settings.ASR_VAD_MIN_SPEECH_MS
            ):
                begin_ms = max(0, self.audio_time_ms - self.nearfield_ms)
                event.speech_begin_ms = begin_ms
                event.source = "rms_fallback"
        else:
            self.silence_ms += chunk_duration_ms
            self.nearfield_ms = 0
            if (
                self.speech_active
                and self.silence_ms >= settings.ASR_VAD_END_FALLBACK_MS
            ):
                event.speech_end_ms = max(0, self.audio_time_ms - self.silence_ms)
                event.source = "silence_fallback"

        return event

    def _merge_events(self, primary: VADEvent, fallback: VADEvent) -> VADEvent:
        if primary.speech_begin_ms is None and fallback.speech_begin_ms is not None:
            primary.speech_begin_ms = fallback.speech_begin_ms
            primary.source = fallback.source
            self.speech_active = True
        if primary.speech_end_ms is None and fallback.speech_end_ms is not None:
            primary.speech_end_ms = fallback.speech_end_ms
            primary.source = fallback.source
            self.speech_active = False
        return primary
