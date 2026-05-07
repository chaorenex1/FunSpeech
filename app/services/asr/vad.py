# -*- coding: utf-8 -*-
"""Streaming VAD endpoint detection for pseudo-streaming ASR."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, List, Optional

import numpy as np

from ...core.config import settings
from ...core.executor import run_sync

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
    """FSMN-VAD based endpoint detector with RMS/silence fallback."""

    def __init__(self, engine, sample_rate: int):
        self.engine = engine
        self.sample_rate = sample_rate
        self.cache: dict = {}
        self.audio_time_ms = 0
        self.speech_active = False
        self.nearfield_ms = 0
        self.silence_ms = 0
        self._vad_model = None

    def _get_vad_model(self):
        if self._vad_model is None:
            from .engine import get_global_vad_model

            self._vad_model = get_global_vad_model(self.engine.device)
        return self._vad_model

    async def accept_audio(
        self,
        audio_array: np.ndarray,
        is_final: bool = False,
    ) -> VADEvent:
        chunk_duration_ms = int(len(audio_array) / self.sample_rate * 1000)
        event = VADEvent(is_speech_active=self.speech_active, raw_segments=[])

        if settings.ASR_ENABLE_VAD_ENDPOINT and len(audio_array) > 0:
            try:
                vad_model = self._get_vad_model()
                result = await run_sync(
                    vad_model.generate,
                    input=audio_array,
                    cache=self.cache,
                    is_final=is_final,
                    chunk_size=settings.ASR_VAD_CHUNK_SIZE_MS,
                )
                event = self._parse_vad_result(result)
            except Exception as e:
                logger.warning(f"VAD推理失败，使用RMS兜底: {e}")
                event = VADEvent(source="rms_fallback")

        fallback_event = self._fallback_event(audio_array, chunk_duration_ms)
        event = self._merge_events(event, fallback_event)
        self.audio_time_ms += chunk_duration_ms
        event.is_speech_active = self.speech_active
        return event

    def _parse_vad_result(self, result: Any) -> VADEvent:
        segments = []
        if result and isinstance(result, list):
            first = result[0] if result else {}
            if isinstance(first, dict):
                segments = first.get("value", []) or []
            elif isinstance(first, list):
                segments = first

        event = VADEvent(is_speech_active=self.speech_active, raw_segments=segments)
        for segment in segments:
            if not isinstance(segment, (list, tuple)) or len(segment) < 2:
                continue
            begin_ms, end_ms = int(segment[0]), int(segment[1])
            if begin_ms >= 0 and not self.speech_active:
                event.speech_begin_ms = begin_ms
                self.speech_active = True
            if end_ms >= 0:
                event.speech_end_ms = end_ms
                self.speech_active = False
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
