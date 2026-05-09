# -*- coding: utf-8 -*-
"""Shared ASR emotion parsing and TTS emotion-control helpers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Optional


SUPPORTED_EMOTIONS = {
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgusted",
    "surprised",
}

SENSEVOICE_EMOTION_TAGS = {
    "NEUTRAL": "neutral",
    "HAPPY": "happy",
    "SAD": "sad",
    "ANGRY": "angry",
    "FEARFUL": "fearful",
    "DISGUSTED": "disgusted",
    "SURPRISED": "surprised",
}

EMOTION_PROMPTS = {
    "happy": "请用开心、愉悦的语气说这句话。",
    "sad": "请用伤心、低落的语气说这句话。",
    "angry": "请用生气、强烈的语气说这句话。",
    "fearful": "请用害怕、紧张的语气说这句话。",
    "disgusted": "请用厌恶、不满的语气说这句话。",
    "surprised": "请用惊讶、意外的语气说这句话。",
    "neutral": "请用自然平静的语气说这句话。",
}

_RICH_TAG_RE = re.compile(r"<\|([^|<>]+)\|>")


@dataclass(frozen=True)
class ASRTranscriptionResult:
    """Structured ASR result that preserves optional SenseVoice emotion metadata."""

    text: str
    raw_text: str = ""
    raw_rich_text: Optional[str] = None
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def normalize_emotion(value: Optional[str]) -> Optional[str]:
    """Normalize incoming emotion labels from clients or model tags."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    aliases = {
        "calm": "neutral",
        "normal": "neutral",
        "fear": "fearful",
        "disgust": "disgusted",
        "surprise": "surprised",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_EMOTIONS:
        raise ValueError(
            f"unsupported emotion: {value}. supported: {', '.join(sorted(SUPPORTED_EMOTIONS))}"
        )
    return normalized


def extract_sensevoice_emotion(raw_text: str) -> Optional[str]:
    """Extract canonical emotion from SenseVoice rich transcription tags."""
    for tag in _RICH_TAG_RE.findall(raw_text or ""):
        emotion = SENSEVOICE_EMOTION_TAGS.get(tag.upper())
        if emotion:
            return emotion
    return None


def strip_rich_tags(raw_text: str) -> str:
    """Remove rich transcription tags without depending on FunASR postprocess utils."""
    return _RICH_TAG_RE.sub("", raw_text or "").strip()


def build_asr_result(
    text: str,
    *,
    raw_text: Optional[str] = None,
    raw_rich_text: Optional[str] = None,
    enable_emotion: bool = False,
    return_rich_text: bool = False,
) -> ASRTranscriptionResult:
    """Build an ASR result while keeping emotion optional for compatibility."""
    source = raw_rich_text if raw_rich_text is not None else raw_text or text
    emotion = extract_sensevoice_emotion(source) if enable_emotion else None
    return ASRTranscriptionResult(
        text=text or "",
        raw_text=raw_text or text or "",
        raw_rich_text=source if return_rich_text else None,
        emotion=emotion,
        emotion_confidence=None,
    )


def compose_emotion_prompt(
    prompt: Optional[str] = "",
    emotion: Optional[str] = None,
    emotion_intensity: Optional[float] = None,
) -> str:
    """Merge a structured emotion tag into the natural-language CosyVoice prompt."""
    normalized = normalize_emotion(emotion)
    user_prompt = (prompt or "").strip()
    if not normalized:
        return user_prompt

    emotion_prompt = EMOTION_PROMPTS[normalized]
    if emotion_intensity is not None:
        intensity = max(0.0, min(1.0, float(emotion_intensity)))
        if intensity >= 0.75 and normalized != "neutral":
            emotion_prompt = emotion_prompt.replace("请用", "请用非常")
        elif intensity <= 0.35 and normalized != "neutral":
            emotion_prompt = emotion_prompt.replace("请用", "请用稍微")

    return f"{emotion_prompt} {user_prompt}".strip()
