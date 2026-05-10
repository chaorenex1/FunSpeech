# -*- coding: utf-8 -*-
"""Frame-smoothed VAD and ordered ASR utterance commits."""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque

from ...core.config import settings
from .types import AsrSegment, AudioFrame


logger = logging.getLogger(__name__)


class SlidingVadSegmenter:
    """Detect utterance boundaries from fixed-size frames.

    Raw frame decisions are majority-smoothed before feeding an explicit
    IDLE/SPEAKING/END_OF_UTTERANCE state machine. Long speech is emitted as
    ASR-ready chunks while the VAD utterance remains active.
    """

    def __init__(
        self,
        window_ms: int,
        pre_roll_ms: int,
        end_silence_ms: int,
        *,
        post_pad_ms: int = 0,
        smooth_window_frames: int = 5,
        smooth_speech_frames: int = 3,
        start_speech_frames: int | None = None,
        end_silence_frames: int | None = None,
        max_segment_ms: int = 60_000,
    ):
        self.window_ms = max(20, int(window_ms))
        self.pre_roll_ms = max(0, int(pre_roll_ms))
        end_silence_ms = max(20, int(end_silence_ms))
        self.post_pad_ms = max(0, int(post_pad_ms))
        self.smooth_window_frames = max(1, int(smooth_window_frames))
        self.smooth_speech_frames = min(
            self.smooth_window_frames,
            max(1, int(smooth_speech_frames)),
        )
        self.start_speech_frames = (
            max(1, int(start_speech_frames))
            if start_speech_frames is not None
            else max(1, _ceil_div(settings.ASR_VAD_MIN_SPEECH_MS, self.window_ms))
        )
        self.end_silence_frames = (
            max(1, int(end_silence_frames))
            if end_silence_frames is not None
            else max(1, _ceil_div(end_silence_ms, self.window_ms))
        )
        self.max_segment_ms = max(self.window_ms, int(max_segment_ms))

        self.state = "IDLE"
        self.active = False
        self.silence_ms = 0
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._speech_started = False
        self._decision_window: Deque[bool] = deque(maxlen=self.smooth_window_frames)
        self._pre_roll: Deque[AudioFrame] = deque()
        self._utterance_frames: list[AudioFrame] = []
        self._last_voice_frame_index = 0
        self._segment_body_ms = 0
        self._segment_index = 0

    def accept(self, frame: AudioFrame, utterance_id: str) -> list[AsrSegment]:
        smoothed_voice = self._smoothed_voice_decision(frame)

        if self.active:
            self._utterance_frames.append(frame)
            if smoothed_voice:
                self._last_voice_frame_index = len(self._utterance_frames)
        else:
            self._remember_pre_roll(frame)

        segments: list[AsrSegment] = []
        if smoothed_voice:
            self.silence_ms = 0
            self._consecutive_silence_frames = 0
            self._consecutive_speech_frames += 1
            if not self.active:
                if self._consecutive_speech_frames < self.start_speech_frames:
                    return segments
                self.active = True
                self._transition("SPEAKING", utterance_id, frame)
                self._speech_started = True
                self._utterance_frames = list(self._pre_roll)
                if (
                    not self._utterance_frames
                    or self._utterance_frames[-1].sequence != frame.sequence
                ):
                    self._utterance_frames.append(frame)
                self._pre_roll.clear()
                self._last_voice_frame_index = self._last_voice_index(self._utterance_frames)
            self._segment_body_ms += frame.duration_ms
            if self._segment_body_ms >= self.max_segment_ms:
                segments.extend(self._commit_max_duration_segment(utterance_id))
            return segments

        if self.active:
            self._consecutive_speech_frames = 0
            self._consecutive_silence_frames += 1
            self.silence_ms = self._consecutive_silence_frames * frame.duration_ms
            if self._consecutive_silence_frames >= self.end_silence_frames:
                self._transition("END_OF_UTTERANCE", utterance_id, frame)
                segments.extend(self._commit_final_segment(utterance_id))
                self._reset_utterance()
        else:
            self._consecutive_speech_frames = 0
        return segments

    def consume_speech_started(self) -> bool:
        started = self._speech_started
        self._speech_started = False
        return started

    def _commit_final_segment(self, utterance_id: str) -> list[AsrSegment]:
        final_end_index = self._final_commit_end_index()
        pending = self._utterance_frames[:final_end_index]
        if not pending:
            return [self._final_marker(utterance_id)]
        if not self._contains_voice(pending):
            return [self._final_marker(utterance_id)]
        return [self._segment(pending, utterance_id, commit_reason="vad_end")]

    def _commit_max_duration_segment(self, utterance_id: str) -> list[AsrSegment]:
        pending = list(self._utterance_frames)
        if not pending or not self._contains_voice(pending):
            self._utterance_frames = []
            self._last_voice_frame_index = 0
            self._segment_body_ms = 0
            return []

        segment = self._segment(
            pending,
            utterance_id,
            commit_reason="max_duration",
            body_ms=self._segment_body_ms,
        )
        self._utterance_frames = []
        self._last_voice_frame_index = 0
        self._segment_body_ms = 0
        return [segment]

    def _segment(
        self,
        frames: list[AudioFrame],
        utterance_id: str,
        *,
        commit_reason: str,
        body_ms: int | None = None,
    ) -> AsrSegment:
        self._segment_index += 1
        payload = b"".join(frame.payload for frame in frames)
        segment = AsrSegment(
            payload=payload,
            duration_ms=_duration_ms(frames),
            frame_count=len(frames),
            utterance_id=utterance_id,
            first_frame_seq=frames[0].sequence if frames else 0,
            last_frame_seq=frames[-1].sequence if frames else 0,
            segment_id=f"{utterance_id}_seg_{self._segment_index}",
            vad_source="frame_smoothed_vad_rms",
            commit_reason=commit_reason,
        )
        logger.debug(
            "vad.segment_committed utterance_id=%s segment_id=%s commit_reason=%s "
            "duration_ms=%s body_ms=%s frames=%s-%s frame_count=%s",
            utterance_id,
            segment.segment_id,
            commit_reason,
            segment.duration_ms,
            body_ms if body_ms is not None else segment.duration_ms,
            segment.first_frame_seq,
            segment.last_frame_seq,
            segment.frame_count,
        )
        return segment

    def _smoothed_voice_decision(self, frame: AudioFrame) -> bool:
        self._decision_window.append(self._is_voice_frame(frame))
        return sum(1 for is_voice in self._decision_window if is_voice) >= self.smooth_speech_frames

    def _is_voice_frame(self, frame: AudioFrame) -> bool:
        return _is_marked_voice(frame)

    def _remember_pre_roll(self, frame: AudioFrame) -> None:
        self._pre_roll.append(frame)
        while _duration_ms(self._pre_roll) > self.pre_roll_ms and self._pre_roll:
            self._pre_roll.popleft()

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
        segment = AsrSegment(
            payload=b"",
            duration_ms=0,
            frame_count=0,
            utterance_id=utterance_id,
            first_frame_seq=sequence,
            last_frame_seq=sequence,
            segment_id=f"{utterance_id}_seg_{self._segment_index}",
            vad_source="frame_smoothed_vad_rms",
            commit_reason="vad_end_marker",
        )
        logger.debug(
            "vad.segment_committed utterance_id=%s segment_id=%s "
            "commit_reason=vad_end_marker frames=%s-%s",
            utterance_id,
            segment.segment_id,
            segment.first_frame_seq,
            segment.last_frame_seq,
        )
        return segment

    def _last_voice_frame(self) -> AudioFrame | None:
        if self._last_voice_frame_index <= 0:
            return None
        return self._utterance_frames[self._last_voice_frame_index - 1]

    def _last_voice_index(self, frames: list[AudioFrame]) -> int:
        for index in range(len(frames), 0, -1):
            if _is_marked_voice(frames[index - 1]):
                return index
        return 0

    def _contains_voice(self, frames: list[AudioFrame]) -> bool:
        return any(_is_marked_voice(frame) for frame in frames)

    def _reset_utterance(self) -> None:
        previous_state = self.state
        self.state = "IDLE"
        self.active = False
        self.silence_ms = 0
        self._utterance_frames = []
        self._last_voice_frame_index = 0
        self._segment_body_ms = 0
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        logger.debug("vad.state_transition %s -> IDLE", previous_state)

    def _transition(self, next_state: str, utterance_id: str, frame: AudioFrame) -> None:
        previous_state = self.state
        self.state = next_state
        logger.debug(
            "vad.state_transition %s -> %s utterance_id=%s frame_seq=%s "
            "speech_frames=%s silence_frames=%s silence_ms=%s",
            previous_state,
            next_state,
            utterance_id,
            frame.sequence,
            self._consecutive_speech_frames,
            self._consecutive_silence_frames,
            self.silence_ms,
        )


def _duration_ms(frames) -> int:
    return sum(frame.duration_ms for frame in frames)


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _is_marked_voice(frame: AudioFrame) -> bool:
    return (
        frame.speech_active
        or frame.pre_class == "rms_voice"
        or frame.vad_state in {"speech", "active"}
    )
