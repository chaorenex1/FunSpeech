# -*- coding: utf-8 -*-
"""Stable-text commit logic for ASR partial hypotheses."""

from __future__ import annotations

from collections import deque
from time import monotonic

from .types import AsrHypothesis, CommittedText


class StableTextCommitter:
    """Convert noisy ASR partials into TTS-safe text deltas."""

    def __init__(
        self,
        stable_hypotheses: int = 2,
        min_commit_chars: int = 8,
        max_commit_wait_ms: int = 400,
    ):
        self.stable_hypotheses = max(1, stable_hypotheses)
        self.min_commit_chars = max(1, min_commit_chars)
        self.max_commit_wait_ms = max(1, max_commit_wait_ms)
        self._recent: deque[str] = deque(maxlen=self.stable_hypotheses)
        self._committed_text = ""
        self._last_commit_at = monotonic()
        self._revision_id = 0

    @property
    def committed_text(self) -> str:
        return self._committed_text

    def update(self, hypothesis: AsrHypothesis) -> CommittedText | None:
        text = (hypothesis.text or "").strip()
        if not text:
            return None

        if hypothesis.is_final:
            stable_text = text
        else:
            self._recent.append(text)
            if len(self._recent) < self.stable_hypotheses:
                return None
            stable_text = self._stable_prefix()

        if not stable_text.startswith(self._committed_text):
            stable_text = self._trim_to_committed_boundary(stable_text)

        delta = stable_text[len(self._committed_text) :].strip()
        if not delta:
            return None

        if not hypothesis.is_final and not self._should_commit(delta):
            return None

        self._revision_id += 1
        self._committed_text = f"{self._committed_text}{delta}"
        self._last_commit_at = monotonic()
        return CommittedText(
            revision_id=self._revision_id,
            text=delta,
            full_text=self._committed_text,
            is_final=hypothesis.is_final,
        )

    def reset_sentence(self) -> None:
        self._recent.clear()
        self._committed_text = ""
        self._last_commit_at = monotonic()

    def _stable_prefix(self) -> str:
        if not self._recent:
            return ""
        prefix = self._recent[0]
        for text in list(self._recent)[1:]:
            prefix = _longest_common_prefix(prefix, text)
        return prefix.strip()

    def _trim_to_committed_boundary(self, text: str) -> str:
        if not self._committed_text:
            return text
        prefix = _longest_common_prefix(self._committed_text, text)
        return prefix if len(prefix) >= len(self._committed_text) else self._committed_text

    def _should_commit(self, delta: str) -> bool:
        if len(delta) >= self.min_commit_chars:
            return True
        if any(mark in delta for mark in "，。！？；,.!?;"):
            return True
        elapsed_ms = int((monotonic() - self._last_commit_at) * 1000)
        return elapsed_ms >= self.max_commit_wait_ms


def _longest_common_prefix(left: str, right: str) -> str:
    index = 0
    for lch, rch in zip(left, right):
        if lch != rch:
            break
        index += 1
    return left[:index]
