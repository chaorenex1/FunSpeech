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
    pre_class: str = "unknown"
    vad_state: str = "unknown"
    utterance_id: str | None = None
    speech_active: bool = False
    created_at: float = field(default_factory=monotonic)


@dataclass(frozen=True)
class AsrSegment:
    """A VAD-approved speech segment queued for ASR."""

    payload: bytes
    duration_ms: int
    frame_count: int
    utterance_id: str
    first_frame_seq: int
    last_frame_seq: int
    segment_id: str = ""
    is_final: bool = False
    vad_source: str = "vad"
    commit_reason: str = "partial"
    created_at: float = field(default_factory=monotonic)


@dataclass(frozen=True)
class BackpressureEvent:
    """Observable backpressure decision."""

    type: str
    queue_ms: int = 0
    dropped_ms: int = 0
    message: str = ""
    vad_state: str | None = None
    pre_class: str | None = None
    utterance_id: str | None = None
    first_dropped_seq: int | None = None
    last_dropped_seq: int | None = None


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
class TtsJob:
    """A cancellable TTS unit."""

    revision_id: int
    text: str
    voice_name: str
    parameters: dict
    priority: Literal["final"] = "final"
