# -*- coding: utf-8 -*-
"""Sliding-window VAD and ordered ASR segment commits."""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from ...core.config import settings
from .types import AsrSegment, AudioFrame


class SlidingVadSegmenter:
    """Detect speech with overlapping windows while committing contiguous audio.

    The detection window is allowed to slide and overlap. ASR segments are cut
    from an utterance accumulator, so the ASR queue receives ordered frame ranges
    instead of raw VAD windows.
    """

    def __init__(
        self,
        window_ms: int,
        pre_roll_ms: int,
        end_silence_ms: int,
        *,
        hop_ms: int | None = None,
        partial_commit_ms: int | None = None,
        active_ratio: float = 0.6,
        silence_ratio: float = 0.2,
        post_pad_ms: int = 0,
    ):
        self.window_ms = max(20, int(window_ms))
        self.hop_ms = max(20, int(hop_ms or max(20, self.window_ms // 2)))
        self.pre_roll_ms = max(0, int(pre_roll_ms))
        self.end_silence_ms = max(20, int(end_silence_ms))
        self.partial_commit_ms = max(20, int(partial_commit_ms or max(800, self.window_ms)))
        self.active_ratio = min(1.0, max(0.0, float(active_ratio)))
        self.silence_ratio = min(self.active_ratio, max(0.0, float(silence_ratio)))
        self.post_pad_ms = max(0, int(post_pad_ms))

        self.active = False
        self.silence_ms = 0
        self._pressure_queued_ms = 0
        self._pressure_high_ms = 0
        self._pressure_max_ms = 0
        self._since_eval_ms = 0
        self._analysis_frames: Deque[AudioFrame] = deque()
        self._pre_roll: Deque[AudioFrame] = deque()
        self._utterance_frames: list[AudioFrame] = []
        self._commit_cursor = 0
        self._last_voice_frame_index = 0
        self._segment_index = 0

    def set_asr_pressure(self, queued_ms: int, high_watermark_ms: int, max_ms: int) -> None:
        """Slow partial commits when ASR queue pressure rises."""
        self._pressure_queued_ms = max(0, int(queued_ms))
        self._pressure_high_ms = max(0, int(high_watermark_ms))
        self._pressure_max_ms = max(self._pressure_high_ms, int(max_ms))

    def accept(self, frame: AudioFrame, utterance_id: str) -> list[AsrSegment]:
        self._analysis_frames.append(frame)
        self._trim_analysis_frames()
        self._since_eval_ms += frame.duration_ms

        if self.active:
            self._utterance_frames.append(frame)
            if self._is_voice_frame(frame):
                self._last_voice_frame_index = len(self._utterance_frames)
        else:
            self._remember_pre_roll(frame)

        segments: list[AsrSegment] = []
        if self._since_eval_ms < self.hop_ms or self._window_duration_ms() < self.window_ms:
            return segments

        self._since_eval_ms = 0
        speech_ratio = self._speech_ratio()
        is_voice = speech_ratio >= self.active_ratio
        if is_voice:
            self.silence_ms = 0
            if not self.active:
                self.active = True
                self._utterance_frames = list(self._pre_roll)
                self._pre_roll.clear()
                self._commit_cursor = 0
                self._last_voice_frame_index = self._last_voice_index(self._utterance_frames)
            segments.extend(self._commit_ready_segments(utterance_id))
            return segments

        if self.active:
            if speech_ratio <= self.silence_ratio:
                self.silence_ms += self.hop_ms
            else:
                self.silence_ms = max(0, self.silence_ms - self.hop_ms)
            if self.silence_ms >= self.end_silence_ms:
                segments.extend(self._commit_final_segment(utterance_id))
                self._reset_utterance()
        return segments

    def _commit_ready_segments(self, utterance_id: str) -> list[AsrSegment]:
        if self._suppress_partial_commits():
            return []
        pending = self._utterance_frames[self._commit_cursor :]
        if _duration_ms(pending) < self._effective_partial_commit_ms():
            return []
        if not self._contains_voice(pending):
            return []
        segment = self._segment(pending, utterance_id, is_final=False, commit_reason="partial")
        self._commit_cursor = len(self._utterance_frames)
        return [segment]

    def _commit_final_segment(self, utterance_id: str) -> list[AsrSegment]:
        final_end_index = self._final_commit_end_index()
        pending = self._utterance_frames[self._commit_cursor : final_end_index]
        if not pending:
            return [self._final_marker(utterance_id)]
        if not self._contains_voice(pending):
            return [self._final_marker(utterance_id)]
        self._commit_cursor = len(self._utterance_frames)
        return [self._segment(pending, utterance_id, is_final=True, commit_reason="vad_end")]

    def _segment(
        self,
        frames: list[AudioFrame],
        utterance_id: str,
        *,
        is_final: bool,
        commit_reason: str,
    ) -> AsrSegment:
        self._segment_index += 1
        payload = b"".join(frame.payload for frame in frames)
        return AsrSegment(
            payload=payload,
            duration_ms=_duration_ms(frames),
            frame_count=len(frames),
            utterance_id=utterance_id,
            first_frame_seq=frames[0].sequence if frames else 0,
            last_frame_seq=frames[-1].sequence if frames else 0,
            segment_id=f"{utterance_id}_seg_{self._segment_index}",
            is_final=is_final,
            vad_source="sliding_rms",
            commit_reason=commit_reason,
        )

    def _speech_ratio(self) -> float:
        window = list(self._analysis_frames)
        total_ms = _duration_ms(window)
        if total_ms <= 0:
            return 0.0
        voice_ms = sum(frame.duration_ms for frame in window if self._is_voice_frame(frame))
        return voice_ms / total_ms

    def _is_voice_frame(self, frame: AudioFrame) -> bool:
        if frame.pre_class == "rms_voice" or frame.vad_state in {"speech", "active"}:
            return True
        samples = _pcm_bytes_to_float(frame.payload)
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(samples))))
        return rms >= settings.ASR_NEARFIELD_RMS_THRESHOLD

    def _remember_pre_roll(self, frame: AudioFrame) -> None:
        self._pre_roll.append(frame)
        while _duration_ms(self._pre_roll) > self.pre_roll_ms and self._pre_roll:
            self._pre_roll.popleft()

    def _trim_analysis_frames(self) -> None:
        while _duration_ms(self._analysis_frames) > self.window_ms and self._analysis_frames:
            self._analysis_frames.popleft()

    def _window_duration_ms(self) -> int:
        return _duration_ms(self._analysis_frames)

    def _effective_partial_commit_ms(self) -> int:
        if self._pressure_high_ms and self._pressure_queued_ms >= self._pressure_high_ms:
            return self.partial_commit_ms * 2
        return self.partial_commit_ms

    def _suppress_partial_commits(self) -> bool:
        return bool(self._pressure_max_ms and self._pressure_queued_ms >= self._pressure_max_ms)

    def _final_commit_end_index(self) -> int:
        end_index = min(self._last_voice_frame_index, len(self._utterance_frames))
        if self.post_pad_ms <= 0:
            return end_index
        total = 0
        padded_end = end_index
        while padded_end < len(self._utterance_frames) and total < self.post_pad_ms:
            total += self._utterance_frames[padded_end].duration_ms
            padded_end += 1
        return padded_end

    def _final_marker(self, utterance_id: str) -> AsrSegment:
        self._segment_index += 1
        anchor = self._last_voice_frame() or (self._utterance_frames[-1] if self._utterance_frames else None)
        sequence = anchor.sequence if anchor is not None else 0
        return AsrSegment(
            payload=b"",
            duration_ms=0,
            frame_count=0,
            utterance_id=utterance_id,
            first_frame_seq=sequence,
            last_frame_seq=sequence,
            segment_id=f"{utterance_id}_seg_{self._segment_index}",
            is_final=True,
            vad_source="sliding_rms",
            commit_reason="vad_end_marker",
        )

    def _last_voice_frame(self) -> AudioFrame | None:
        if self._last_voice_frame_index <= 0:
            return None
        return self._utterance_frames[self._last_voice_frame_index - 1]

    def _last_voice_index(self, frames: list[AudioFrame]) -> int:
        for index in range(len(frames), 0, -1):
            if self._is_voice_frame(frames[index - 1]):
                return index
        return 0

    def _contains_voice(self, frames: list[AudioFrame]) -> bool:
        return any(self._is_voice_frame(frame) for frame in frames)

    def _reset_utterance(self) -> None:
        self.active = False
        self.silence_ms = 0
        self._utterance_frames = []
        self._commit_cursor = 0
        self._last_voice_frame_index = 0
        self._since_eval_ms = 0


def _duration_ms(frames) -> int:
    return sum(frame.duration_ms for frame in frames)


def _pcm_bytes_to_float(audio: bytes) -> np.ndarray:
    if len(audio) < 2:
        return np.array([], dtype=np.float32)
    pcm = np.frombuffer(audio, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0
