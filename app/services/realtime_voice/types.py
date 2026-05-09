# -*- coding: utf-8 -*-
"""Typed payloads for the realtime voice pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Literal


@dataclass(frozen=True)
class AudioFrame:
    """A client audio chunk with enough metadata for queue budgeting."""

    payload: bytes
    duration_ms: int
    is_silence: bool
    sequence: int = 0
    vad_state: str = "unknown"
    speech_active: bool = False
    created_at: float = field(default_factory=monotonic)


@dataclass(frozen=True)
class BackpressureEvent:
    """Observable backpressure decision."""

    type: str
    queue_ms: int = 0
    dropped_ms: int = 0
    message: str = ""


@dataclass(frozen=True)
class AsrHypothesis:
    """ASR text hypothesis emitted by the ASR stage."""

    text: str
    is_final: bool = False
    kind: str = "partial"
    time_ms: int = 0
    begin_time_ms: int = 0
    speech_begin_ms: int | None = None
    speech_end_ms: int | None = None
    vad_source: str | None = None
    speech_active: bool = False
    emotion: str | None = None
    emotion_confidence: float | None = None
    raw_rich_text: str | None = None


@dataclass(frozen=True)
class CommittedText:
    """Text that is stable enough to synthesize."""

    revision_id: int
    text: str
    full_text: str
    is_final: bool = False


@dataclass(frozen=True)
class TtsJob:
    """A cancellable TTS unit."""

    revision_id: int
    text: str
    voice_name: str
    parameters: dict
    priority: Literal["stable", "final"] = "stable"
