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
        max_commit_chars: int | None = None,
        max_commit_wait_ms: int = 400,
    ):
        self.stable_hypotheses = max(1, stable_hypotheses)
        self.min_commit_chars = max(1, min_commit_chars)
        self.max_commit_chars = (
            max(self.min_commit_chars, max_commit_chars)
            if max_commit_chars is not None and max_commit_chars > 0
            else 0
        )
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
            stable_text = self._merge_rolling_window_text(stable_text)

        delta = stable_text[len(self._committed_text) :].strip()
        if not delta:
            return None

        if not hypothesis.is_final and not self._should_commit(delta):
            return None

        if not hypothesis.is_final and self.max_commit_chars > 0:
            delta = self._bounded_delta(delta)
            if not delta:
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

    def _merge_rolling_window_text(self, text: str) -> str:
        """Merge ASR rolling-window text with already committed text.

        SenseVoice partial decoding may use only the latest audio window for
        latency control. In that mode a new hypothesis can be a suffix window
        rather than the whole sentence, so strict prefix matching would stop all
        commits until VAD final. We keep prefix behavior when possible, otherwise
        append only the non-overlapping suffix of the rolling hypothesis.
        """
        if not self._committed_text:
            return text
        prefix = _longest_common_prefix(self._committed_text, text)
        if len(prefix) >= len(self._committed_text):
            return prefix
        if text in self._committed_text:
            return self._committed_text
        overlap = _longest_suffix_prefix_overlap(self._committed_text, text)
        if overlap > 0:
            return f"{self._committed_text}{text[overlap:]}"
        return f"{self._committed_text}{text}"

    def _should_commit(self, delta: str) -> bool:
        if len(delta) >= self.min_commit_chars:
            return True
        if any(mark in delta for mark in "，。！？；,.!?;"):
            return True
        elapsed_ms = int((monotonic() - self._last_commit_at) * 1000)
        return elapsed_ms >= self.max_commit_wait_ms

    def _bounded_delta(self, delta: str) -> str:
        """Emit short speculative TTS chunks while keeping punctuation intact."""
        if len(delta) <= self.max_commit_chars:
            return delta
        window = delta[: self.max_commit_chars]
        for index in range(len(window) - 1, self.min_commit_chars - 2, -1):
            if window[index] in "，。！？；,.!?;":
                return window[: index + 1].strip()
        return window.strip()


def _longest_common_prefix(left: str, right: str) -> str:
    index = 0
    for lch, rch in zip(left, right):
        if lch != rch:
            break
        index += 1
    return left[:index]


def _longest_suffix_prefix_overlap(left: str, right: str) -> int:
    max_len = min(len(left), len(right))
    for size in range(max_len, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0
