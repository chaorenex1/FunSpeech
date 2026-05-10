# -*- coding: utf-8 -*-

import asyncio
import logging

import numpy as np

from app.api.v1 import realtime_voice as realtime_voice_api
from app.services.asr.vad import VADEvent
from app.services.realtime_voice.backpressure import AsrSegmentQueue, BoundedAudioQueue, TtsJobQueue
from app.services.realtime_voice.events import RealtimeVoiceEventBuilder
from app.services.realtime_voice.playback_queue import PlaybackChunk, TtsPlaybackQueue
from app.services.realtime_voice.tts_dispatcher import RealtimeTTSDispatcher
from app.services.realtime_voice.types import AsrSegment, AudioFrame, AsrHypothesis, TtsJob
from app.services.realtime_voice.vad_segmenter import SlidingVadSegmenter


def _pcm_frame(amplitude: int, duration_ms: int = 20, sample_rate: int = 16_000) -> bytes:
    samples = np.full(int(sample_rate * duration_ms / 1000), amplitude, dtype=np.int16)
    return samples.tobytes()


def _audio_frame(sequence: int, amplitude: int, duration_ms: int = 20) -> AudioFrame:
    return AudioFrame(
        payload=_pcm_frame(amplitude, duration_ms),
        duration_ms=duration_ms,
        is_silence=amplitude == 0,
        sequence=sequence,
        pre_class="rms_silence" if amplitude == 0 else "rms_voice",
        vad_state="pending",
    )


def _asr_segment(
    sequence: int,
    duration_ms: int = 40,
    commit_reason: str = "max_duration",
) -> AsrSegment:
    return AsrSegment(
        payload=f"seg-{sequence}".encode(),
        duration_ms=duration_ms,
        frame_count=max(1, duration_ms // 20),
        utterance_id="utt_1",
        first_frame_seq=sequence,
        last_frame_seq=sequence,
        vad_source="test",
        commit_reason=commit_reason,
    )


def _bounded_audio_queue_from_frames(
    frames: list[AudioFrame],
    *,
    high_watermark_ms: int,
    max_ms: int,
) -> BoundedAudioQueue:
    frame_iter = iter(frames)
    return BoundedAudioQueue(
        high_watermark_ms=high_watermark_ms,
        max_ms=max_ms,
        sample_rate=1000,
        frame_ms=20,
        frame_factory=lambda _audio: next(frame_iter),
    )


def _dummy_pcm_frames(frame_count: int) -> bytes:
    return b"\x00\x00" * 20 * frame_count


def test_bounded_audio_queue_splits_arbitrary_pcm_into_20ms_frames():
    async def run():
        queue = BoundedAudioQueue(
            high_watermark_ms=1000,
            max_ms=1000,
            sample_rate=1000,
            frame_ms=20,
            frame_factory=lambda audio: AudioFrame(
                payload=audio,
                duration_ms=20,
                is_silence=False,
            ),
        )

        assert await queue.put_audio(b"\x01\x00" * 25) == []
        assert queue.queued_ms == 20
        assert (await queue.get()).payload == b"\x01\x00" * 20

        assert await queue.put_audio(b"\x02\x00" * 15) == []
        assert queue.queued_ms == 20
        assert (await queue.get()).payload == b"\x01\x00" * 5 + b"\x02\x00" * 15

    asyncio.run(run())


def test_bounded_audio_queue_accepts_only_20ms_or_30ms_windows():
    try:
        BoundedAudioQueue(
            high_watermark_ms=1000,
            max_ms=1000,
            sample_rate=16000,
            frame_ms=25,
        )
    except ValueError as exc:
        assert "20ms or 30ms" in str(exc)
    else:
        raise AssertionError("25ms VAD frames should be rejected")


def test_bounded_audio_queue_clamps_pcm_ring_cache_to_one_to_three_seconds():
    assert (
        BoundedAudioQueue(
            high_watermark_ms=500,
            max_ms=500,
            sample_rate=16000,
            frame_ms=20,
        ).max_ms
        == 1000
    )
    assert (
        BoundedAudioQueue(
            high_watermark_ms=1200,
            max_ms=6000,
            sample_rate=16000,
            frame_ms=20,
        ).max_ms
        == 3000
    )


def test_bounded_audio_queue_ring_buffer_drops_oldest_pcm_frames_when_full():
    async def run():
        queue = BoundedAudioQueue(
            high_watermark_ms=1000,
            max_ms=1000,
            sample_rate=1000,
            frame_ms=20,
            frame_factory=lambda audio: AudioFrame(
                payload=audio,
                duration_ms=20,
                is_silence=False,
            ),
        )

        audio = b"".join(
            value.to_bytes(2, "little", signed=True) * 20
            for value in range(1, 53)
        )
        events = await queue.put_audio(audio)

        dropped = [event for event in events if event.type == "drop_oldest_audio"]
        assert len(dropped) == 2
        assert queue.queued_ms == 1000
        first_remaining = await queue.get()
        assert first_remaining.payload == (3).to_bytes(2, "little", signed=True) * 20

    asyncio.run(run())


def test_sliding_vad_segmenter_smooths_last_five_frames_by_majority():
    segmenter = SlidingVadSegmenter(
        window_ms=20,
        pre_roll_ms=0,
        end_silence_ms=40,
        smooth_window_frames=5,
        smooth_speech_frames=3,
        start_speech_frames=1,
    )

    raw_pattern = [0, 1, 1, 1, 0]
    for sequence, raw_voice in enumerate(raw_pattern, start=1):
        segmenter.accept(_audio_frame(sequence, 2400 if raw_voice else 0), "utt_1")

    assert segmenter.active is True
    assert segmenter.consume_speech_started() is True


def test_sliding_vad_segmenter_uses_frame_vad_metadata_without_recomputing_rms():
    segmenter = SlidingVadSegmenter(
        window_ms=20,
        pre_roll_ms=0,
        end_silence_ms=40,
        smooth_window_frames=1,
        smooth_speech_frames=1,
        start_speech_frames=1,
    )
    loud_but_vad_silence = AudioFrame(
        payload=_pcm_frame(2400),
        duration_ms=20,
        is_silence=True,
        sequence=1,
        pre_class="rms_silence",
        vad_state="silence",
    )

    assert segmenter.accept(loud_but_vad_silence, "utt_1") == []
    assert segmenter.active is False


def test_realtime_voice_frame_classifier_uses_streaming_vad_over_rms_silence():
    class FakeDetector:
        async def accept_audio(self, audio_array, is_final=False):
            assert audio_array.dtype == np.float32
            return VADEvent(is_speech_active=True, source="vad")

    async def run():
        session = realtime_voice_api.RealtimeVoiceAsrTtsSession.__new__(
            realtime_voice_api.RealtimeVoiceAsrTtsSession
        )
        session.audio_format = "pcm"
        session.sample_rate = 1000
        session.realtime_vad_detector = FakeDetector()
        frame = AudioFrame(
            payload=_pcm_frame(0, sample_rate=1000),
            duration_ms=20,
            is_silence=True,
            sequence=1,
            pre_class="rms_silence",
            vad_state="pending",
        )

        classified = await session._classify_audio_frame(frame)

        assert classified.vad_state == "speech"
        assert classified.pre_class == "rms_silence"
        assert classified.speech_active is True
        assert classified.is_silence is False

    asyncio.run(run())


def test_realtime_voice_frame_classifier_uses_rms_when_streaming_vad_is_inactive():
    class FakeDetector:
        async def accept_audio(self, audio_array, is_final=False):
            return VADEvent(is_speech_active=False, source="fallback")

    async def run():
        session = realtime_voice_api.RealtimeVoiceAsrTtsSession.__new__(
            realtime_voice_api.RealtimeVoiceAsrTtsSession
        )
        session.audio_format = "pcm"
        session.sample_rate = 1000
        session.realtime_vad_detector = FakeDetector()
        frame = AudioFrame(
            payload=_pcm_frame(2400, sample_rate=1000),
            duration_ms=20,
            is_silence=True,
            sequence=1,
            pre_class="rms_silence",
            vad_state="pending",
        )

        classified = await session._classify_audio_frame(frame)

        assert classified.vad_state == "speech"
        assert classified.pre_class == "rms_voice"
        assert classified.speech_active is False
        assert classified.is_silence is False

    asyncio.run(run())


def test_realtime_voice_does_not_emit_audio_dequeued_by_default():
    class FakeWebSocket:
        def __init__(self):
            self.json_messages = []

        async def send_json(self, payload):
            self.json_messages.append(payload)

    class OneFrameQueue:
        queued_ms = 0

        def __init__(self, session):
            self.session = session

        async def get(self):
            self.session._closed = True
            return _audio_frame(1, 2400)

    class FakeSegmenter:
        def accept(self, frame, utterance_id):
            return []

        def consume_speech_started(self):
            return False

    async def run(parameters):
        websocket = FakeWebSocket()
        session = realtime_voice_api.RealtimeVoiceAsrTtsSession.__new__(
            realtime_voice_api.RealtimeVoiceAsrTtsSession
        )
        session._closed = False
        session._websocket = websocket
        session._task_id = "task-1"
        session._event_builder = RealtimeVoiceEventBuilder("task-1")
        session._send_lock = asyncio.Lock()
        session._utterance_index = 1
        session._speech_active = False
        session.parameters = parameters
        session.audio_queue = OneFrameQueue(session)
        session.vad_segmenter = FakeSegmenter()

        async def classify(frame):
            return frame

        session._classify_audio_frame = classify

        await session._audio_worker()
        return [message["event"] for message in websocket.json_messages]

    assert asyncio.run(run({})) == []
    assert asyncio.run(run({"emit_input_audio_dequeued": True})) == ["input.audio_dequeued"]


def test_sliding_vad_segmenter_waits_for_consecutive_smoothed_speech_before_speaking():
    segmenter = SlidingVadSegmenter(
        window_ms=20,
        pre_roll_ms=80,
        end_silence_ms=40,
        smooth_window_frames=5,
        smooth_speech_frames=3,
        start_speech_frames=2,
    )

    for sequence, raw_voice in enumerate([0, 1, 1, 0, 0], start=1):
        assert segmenter.accept(_audio_frame(sequence, 2400 if raw_voice else 0), "utt_1") == []

    assert segmenter.active is False

    segmenter.accept(_audio_frame(6, 2400), "utt_1")
    segmenter.accept(_audio_frame(7, 2400), "utt_1")

    assert segmenter.active is True


def test_sliding_vad_segmenter_commits_one_final_utterance_with_pre_and_post_padding():
    segmenter = SlidingVadSegmenter(
        window_ms=20,
        pre_roll_ms=80,
        end_silence_ms=40,
        post_pad_ms=40,
        smooth_window_frames=1,
        smooth_speech_frames=1,
        start_speech_frames=2,
    )

    emitted = []
    for sequence, amplitude in [
        (1, 0),
        (2, 0),
        (3, 2400),
        (4, 2400),
        (5, 2400),
        (6, 0),
        (7, 0),
    ]:
        emitted.extend(segmenter.accept(_audio_frame(sequence, amplitude), "utt_1"))

    assert len(emitted) == 1
    assert emitted[0].first_frame_seq == 1
    assert emitted[0].last_frame_seq == 7
    assert emitted[0].frame_count == 7
    assert (
        emitted[0].payload
        == _pcm_frame(0) * 2 + _pcm_frame(2400) * 3 + _pcm_frame(0) * 2
    )
    assert emitted[0].commit_reason == "vad_end"


def test_sliding_vad_segmenter_auto_commits_when_body_reaches_max_segment_ms():
    segmenter = SlidingVadSegmenter(
        window_ms=20,
        pre_roll_ms=0,
        end_silence_ms=40,
        smooth_window_frames=1,
        smooth_speech_frames=1,
        start_speech_frames=1,
        max_segment_ms=60,
    )

    emitted = []
    for sequence in range(1, 4):
        emitted.extend(segmenter.accept(_audio_frame(sequence, 2400), "utt_1"))

    assert len(emitted) == 1
    assert emitted[0].commit_reason == "max_duration"
    assert emitted[0].duration_ms == 60
    assert emitted[0].frame_count == 3
    assert emitted[0].first_frame_seq == 1
    assert emitted[0].last_frame_seq == 3
    assert segmenter.active is True
    assert realtime_voice_api._is_utterance_end_segment(emitted[0]) is False


def test_sliding_vad_segmenter_emits_tail_after_auto_commit():
    segmenter = SlidingVadSegmenter(
        window_ms=20,
        pre_roll_ms=0,
        end_silence_ms=40,
        post_pad_ms=40,
        smooth_window_frames=1,
        smooth_speech_frames=1,
        start_speech_frames=1,
        max_segment_ms=60,
    )

    emitted = []
    for sequence, amplitude in [
        (1, 2400),
        (2, 2400),
        (3, 2400),
        (4, 2400),
        (5, 0),
        (6, 0),
    ]:
        emitted.extend(segmenter.accept(_audio_frame(sequence, amplitude), "utt_1"))

    assert [segment.commit_reason for segment in emitted] == ["max_duration", "vad_end"]
    assert [(segment.first_frame_seq, segment.last_frame_seq) for segment in emitted] == [
        (1, 3),
        (4, 6),
    ]
    assert segmenter.active is False


def test_sliding_vad_segmenter_logs_state_machine(caplog):
    caplog.set_level(
        logging.DEBUG,
        logger="app.services.realtime_voice.vad_segmenter",
    )
    segmenter = SlidingVadSegmenter(
        window_ms=20,
        pre_roll_ms=0,
        end_silence_ms=40,
        smooth_window_frames=1,
        smooth_speech_frames=1,
        start_speech_frames=1,
        max_segment_ms=60,
    )

    for sequence, amplitude in [(1, 2400), (2, 0), (3, 0)]:
        segmenter.accept(_audio_frame(sequence, amplitude), "utt_1")

    messages = [record.getMessage() for record in caplog.records]
    assert any("vad.state_transition IDLE -> SPEAKING" in message for message in messages)
    assert any(
        "vad.state_transition SPEAKING -> END_OF_UTTERANCE" in message
        for message in messages
    )
    assert any("vad.state_transition END_OF_UTTERANCE -> IDLE" in message for message in messages)
    assert any("commit_reason=vad_end" in message for message in messages)


def test_sliding_vad_segmenter_ignores_short_silence_inside_speech():
    segmenter = SlidingVadSegmenter(
        window_ms=20,
        pre_roll_ms=0,
        end_silence_ms=60,
        post_pad_ms=60,
        smooth_window_frames=1,
        smooth_speech_frames=1,
        start_speech_frames=1,
    )

    emitted = []
    for sequence, amplitude in [
        (1, 2400),
        (2, 2400),
        (3, 0),
        (4, 2400),
        (5, 0),
        (6, 0),
    ]:
        emitted.extend(segmenter.accept(_audio_frame(sequence, amplitude), "utt_1"))

    assert emitted == []
    assert segmenter.active is True

    emitted.extend(segmenter.accept(_audio_frame(7, 0), "utt_1"))

    assert len(emitted) == 1
    assert emitted[0].first_frame_seq == 1
    assert emitted[0].last_frame_seq == 7



def test_realtime_voice_ignores_nonfinal_asr_text_for_tts_and_display():
    class FakeWebSocket:
        def __init__(self):
            self.json_messages = []

        async def send_json(self, payload):
            self.json_messages.append(payload)

    async def run():
        websocket = FakeWebSocket()
        session = realtime_voice_api.RealtimeVoiceAsrTtsSession.__new__(
            realtime_voice_api.RealtimeVoiceAsrTtsSession
        )
        session._task_id = "task-1"
        session._event_builder = RealtimeVoiceEventBuilder("task-1")
        session._websocket = websocket
        session._send_lock = asyncio.Lock()
        session._utterance_index = 1
        session._hypothesis_index = 0
        session._tts_revision_id = 0
        session.tts_jobs = TtsJobQueue(maxsize=3)

        await session._handle_asr_hypothesis(AsrHypothesis("中间结果", is_final=False))

        assert websocket.json_messages == []
        assert session.tts_jobs.get_nowait() is None

    asyncio.run(run())


def test_realtime_voice_queues_one_tts_job_from_final_utterance_text():
    class FakeWebSocket:
        def __init__(self):
            self.json_messages = []

        async def send_json(self, payload):
            self.json_messages.append(payload)

    async def run():
        websocket = FakeWebSocket()
        session = realtime_voice_api.RealtimeVoiceAsrTtsSession.__new__(
            realtime_voice_api.RealtimeVoiceAsrTtsSession
        )
        session._task_id = "task-1"
        session._event_builder = RealtimeVoiceEventBuilder("task-1")
        session._websocket = websocket
        session._send_lock = asyncio.Lock()
        session._utterance_index = 1
        session._hypothesis_index = 0
        session._tts_revision_id = 0
        session._speech_active = False
        session.parameters = {}
        session.voice_name = "voice-1"
        session.config_version = 1
        session.tts_jobs = TtsJobQueue(maxsize=3)

        await session._handle_asr_hypothesis(
            AsrHypothesis(
                "完整的一句话",
                is_final=True,
                kind="end",
                emotion="happy",
                emotion_confidence=0.8,
            )
        )

        events = [message["event"] for message in websocket.json_messages]
        assert "asr.hypothesis" not in events
        assert "asr_result" not in events
        assert events == [
            "asr.utterance_final",
            "tts.job_queued",
            "asr.sentence_finalized",
        ]
        final_event = websocket.json_messages[0]
        assert final_event["text"] == "完整的一句话"
        assert final_event["is_final"] is True
        assert final_event["stage"] == "asr_text_received"
        assert final_event["protocol_event"] == "asr.utterance_final"
        assert final_event["payload"]["protocol_event"] == "asr.utterance_final"
        assert final_event["payload"]["hypothesis_id"] == "utt_1_final_1"
        assert final_event["payload"]["tts_job_id"] == "tts_1"
        assert final_event["payload"]["speech_active"] is False
        queued = session.tts_jobs.get_nowait()
        assert queued.text == "完整的一句话"
        assert queued.priority == "final"
        assert queued.revision_id == 1

    asyncio.run(run())


def test_bounded_audio_queue_drops_oldest_frame_when_over_budget():
    async def run():
        frames = [
            AudioFrame(b"voice-1", duration_ms=20, is_silence=False),
            AudioFrame(b"silence", duration_ms=20, is_silence=True),
            *[
                AudioFrame(b"voice-2", duration_ms=20, is_silence=False)
                for _ in range(49)
            ],
        ]
        queue = _bounded_audio_queue_from_frames(
            frames,
            high_watermark_ms=1000,
            max_ms=1000,
        )

        events = await queue.put_audio(_dummy_pcm_frames(len(frames)))

        assert any(event.type == "drop_oldest_audio" for event in events)
        assert queue.queued_ms == 1000
        assert (await queue.get()).payload == b"silence"
        assert (await queue.get()).payload == b"voice-2"

    asyncio.run(run())


def test_bounded_audio_queue_reports_oldest_drop_metadata():
    async def run():
        frames = [
            AudioFrame(
                b"old",
                duration_ms=20,
                is_silence=True,
                sequence=7,
                vad_state="silence",
            ),
            *[
                AudioFrame(
                    b"new",
                    duration_ms=20,
                    is_silence=False,
                    sequence=8 + index,
                    vad_state="speech",
                )
                for index in range(50)
            ],
        ]
        queue = _bounded_audio_queue_from_frames(
            frames,
            high_watermark_ms=1000,
            max_ms=1000,
        )

        events = await queue.put_audio(_dummy_pcm_frames(len(frames)))

        assert events[0].type == "drop_oldest_audio"
        assert events[0].first_dropped_seq == 7
        assert events[0].last_dropped_seq == 7
        assert queue.queued_ms == 1000

    asyncio.run(run())


def test_bounded_audio_queue_drops_oldest_speech_like_before_newer_active_speech():
    async def run():
        frames = [
            AudioFrame(
                b"active",
                duration_ms=20,
                is_silence=False,
                sequence=1,
                vad_state="speech",
                speech_active=True,
            ),
            AudioFrame(
                b"speech-like",
                duration_ms=20,
                is_silence=False,
                sequence=2,
                pre_class="rms_voice",
                vad_state="pending",
            ),
            *[
                AudioFrame(
                    b"active-2",
                    duration_ms=20,
                    is_silence=False,
                    sequence=3 + index,
                    vad_state="speech",
                    speech_active=True,
                )
                for index in range(49)
            ],
        ]
        queue = _bounded_audio_queue_from_frames(
            frames,
            high_watermark_ms=1000,
            max_ms=1000,
        )
        events = await queue.put_audio(_dummy_pcm_frames(len(frames)))

        assert any(event.type == "drop_oldest_audio" for event in events)
        assert queue.queued_ms == 1000
        assert (await queue.get()).payload == b"speech-like"
        assert (await queue.get()).payload == b"active-2"

    asyncio.run(run())


def test_bounded_audio_queue_drops_oldest_speech_when_no_lower_layer_exists():
    async def run():
        frames = [
            AudioFrame(
                b"one",
                duration_ms=20,
                is_silence=False,
                sequence=1,
                vad_state="speech",
                speech_active=True,
            ),
            AudioFrame(
                b"two",
                duration_ms=20,
                is_silence=False,
                sequence=2,
                vad_state="speech",
                speech_active=True,
            ),
            *[
                AudioFrame(
                    b"three",
                    duration_ms=20,
                    is_silence=False,
                    sequence=3 + index,
                    vad_state="speech",
                    speech_active=True,
                )
                for index in range(49)
            ],
        ]
        queue = _bounded_audio_queue_from_frames(
            frames,
            high_watermark_ms=1000,
            max_ms=1000,
        )
        events = await queue.put_audio(_dummy_pcm_frames(len(frames)))

        dropped = [event for event in events if event.type == "drop_oldest_audio"]
        assert dropped
        assert dropped[0].first_dropped_seq == 1
        assert queue.queued_ms == 1000

    asyncio.run(run())


def test_bounded_audio_queue_never_preserves_old_speech_over_ring_budget():
    async def run():
        frames = [
            AudioFrame(
                b"one",
                duration_ms=20,
                is_silence=False,
                sequence=1,
                vad_state="speech",
                speech_active=True,
            ),
            AudioFrame(
                b"two",
                duration_ms=20,
                is_silence=False,
                sequence=2,
                vad_state="speech",
                speech_active=True,
            ),
            *[
                AudioFrame(
                    b"three",
                    duration_ms=20,
                    is_silence=False,
                    sequence=3 + index,
                    vad_state="speech",
                    speech_active=True,
                )
                for index in range(49)
            ],
        ]
        queue = _bounded_audio_queue_from_frames(
            frames,
            high_watermark_ms=1000,
            max_ms=1000,
        )
        events = await queue.put_audio(_dummy_pcm_frames(len(frames)))

        assert any(event.type == "drop_oldest_audio" for event in events)
        assert queue.queued_ms == 1000
        assert (await queue.get()).payload == b"two"
        assert (await queue.get()).payload == b"three"

    asyncio.run(run())


def test_asr_segment_queue_throttles_when_pressure_is_high():
    async def run():
        queue = AsrSegmentQueue(high_watermark_ms=40, max_ms=160)

        await queue.put(_asr_segment(1, duration_ms=40))
        events = await queue.put(_asr_segment(2, duration_ms=40))

        assert any(event.type == "asr_input_throttle" for event in events)
        assert queue.queued_ms == 80
        segment = await queue.get()
        assert segment.duration_ms == 40
        assert segment.frame_count == 2
        assert segment.payload == b"seg-1"
        assert segment.first_frame_seq == 1
        assert segment.last_frame_seq == 1

    asyncio.run(run())


def test_asr_segment_queue_drops_speech_over_budget_but_keeps_vad_end():
    async def run():
        queue = AsrSegmentQueue(high_watermark_ms=40, max_ms=80, preserve_speech=False)

        await queue.put(_asr_segment(1, duration_ms=40))
        await queue.put(_asr_segment(2, duration_ms=40))
        events = await queue.put(
            _asr_segment(3, duration_ms=40, commit_reason="vad_end")
        )

        assert any(event.type == "drop_asr_speech" for event in events)
        assert queue.queued_ms <= 80
        assert (await queue.get()).commit_reason == "max_duration"
        assert (await queue.get()).commit_reason == "vad_end"

    asyncio.run(run())


def test_asr_segment_queue_preserves_speech_by_default_under_pressure():
    async def run():
        queue = AsrSegmentQueue(high_watermark_ms=40, max_ms=80)

        await queue.put(_asr_segment(1, duration_ms=40))
        await queue.put(_asr_segment(2, duration_ms=40))
        events = await queue.put(_asr_segment(3, duration_ms=40))

        assert any(event.type == "asr_preserve_speech_backpressure" for event in events)
        assert queue.queued_ms == 120

    asyncio.run(run())


def test_asr_segment_queue_has_no_hard_drop_when_preserving_speech():
    async def run():
        queue = AsrSegmentQueue(high_watermark_ms=40, max_ms=80)

        await queue.put(_asr_segment(1, duration_ms=40))
        await queue.put(_asr_segment(2, duration_ms=40))
        events = await queue.put(_asr_segment(3, duration_ms=40))

        assert any(event.type == "asr_preserve_speech_backpressure" for event in events)
        assert not any(event.type == "drop_asr_speech" for event in events)
        assert queue.queued_ms == 120

    asyncio.run(run())



def test_tts_job_queue_drops_oldest_final_job_when_drop_on_overload():
    async def run():
        queue = TtsJobQueue(maxsize=2, drop_on_overload=True)
        await queue.put(TtsJob(1, "甲", "voice", {}, "final"))
        await queue.put(TtsJob(2, "乙", "voice", {}, "final"))
        events = await queue.put(TtsJob(3, "丙", "voice", {}, "final"))

        assert any(event.type == "tts_job_dropped" for event in events)
        assert [queue.get_nowait().revision_id for _ in range(2)] == [2, 3]
        assert queue.get_nowait() is None

    asyncio.run(run())


def test_tts_job_queue_applies_hard_limit_after_soft_preserve():
    async def run():
        queue = TtsJobQueue(maxsize=2, hard_limit=3)
        await queue.put(TtsJob(1, "甲", "voice", {}, "final"))
        await queue.put(TtsJob(2, "乙", "voice", {}, "final"))
        await queue.put(TtsJob(3, "丙", "voice", {}, "final"))
        events = await queue.put(TtsJob(4, "丁", "voice", {}, "final"))

        assert any(event.type == "tts_queue_preserved" for event in events)
        assert any(event.type == "tts_job_dropped_hard_limit" for event in events)
        assert [queue.get_nowait().revision_id for _ in range(3)] == [2, 3, 4]
        assert queue.get_nowait() is None

    asyncio.run(run())


def test_tts_job_queue_preserves_jobs_by_default_when_overloaded():
    async def run():
        queue = TtsJobQueue(maxsize=2)
        await queue.put(TtsJob(1, "甲", "voice", {}, "final"))
        await queue.put(TtsJob(2, "乙", "voice", {}, "final"))
        events = await queue.put(TtsJob(3, "丙", "voice", {}, "final"))

        assert any(event.type == "tts_queue_preserved" for event in events)
        assert (await queue.get()).text == "甲"
        assert (await queue.get()).text == "乙"
        assert (await queue.get()).text == "丙"

    asyncio.run(run())

def test_realtime_tts_dispatcher_rejects_when_global_queue_is_full():
    async def run():
        dispatcher = RealtimeTTSDispatcher(
            max_inflight=1,
            max_queue_size=0,
            queue_timeout_ms=50,
        )
        lease = await dispatcher.acquire()
        assert lease.admission.accepted is True

        rejected = await dispatcher.acquire()
        assert rejected.accepted is False
        assert rejected.reason == "global_tts_queue_full"
        assert rejected.active == 1

        await lease.release()
        next_lease = await dispatcher.acquire()
        assert next_lease.admission.accepted is True
        await next_lease.release()

    asyncio.run(run())


def test_realtime_tts_dispatcher_times_out_waiting_for_slot():
    async def run():
        dispatcher = RealtimeTTSDispatcher(
            max_inflight=1,
            max_queue_size=1,
            queue_timeout_ms=10,
        )
        lease = await dispatcher.acquire()
        rejected = await dispatcher.acquire()

        assert rejected.accepted is False
        assert rejected.reason == "global_tts_queue_timeout"
        assert rejected.queue_wait_ms >= 0
        await lease.release()

    asyncio.run(run())



def test_tts_playback_queue_allows_multiple_inflight_chunks():
    async def run():
        queue = TtsPlaybackQueue(maxsize=4, max_inflight=2)
        chunks = [
            PlaybackChunk(f"chunk-{index}", "tts_1", 1, index, bytes([index]), 1000)
            for index in range(1, 4)
        ]
        for chunk in chunks:
            await queue.put(chunk)

        ready = await queue.ready_chunks()
        assert [chunk.chunk_id for chunk in ready] == ["chunk-1", "chunk-2"]
        assert queue.pending_count == 1
        assert queue.in_flight_count == 2

        played = await queue.mark_played("chunk-1")
        assert played is chunks[0]
        ready = await queue.ready_chunks()
        assert [chunk.chunk_id for chunk in ready] == ["chunk-3"]
        assert queue.pending_count == 0
        assert queue.in_flight_count == 2

    asyncio.run(run())


def test_tts_playback_queue_backpressure_delays_raw_chunk_put():
    async def run():
        queue = TtsPlaybackQueue(
            maxsize=4,
            max_inflight=2,
            backpressure_sleep_ms=10,
        )
        queue.set_backpressure("high", playback_queue_ms=1800)
        chunk = PlaybackChunk("chunk-1", "tts_1", 1, 1, b"a", 1000)

        started = asyncio.get_running_loop().time()
        await queue.put(chunk)
        elapsed_ms = int((asyncio.get_running_loop().time() - started) * 1000)

        assert elapsed_ms >= 8
        assert queue.pending_count == 1
        assert queue.stats()["playback_queue_ms"] == 1800

    asyncio.run(run())


def test_realtime_voice_flushes_playback_queue_by_inflight_window():
    class FakeWebSocket:
        def __init__(self):
            self.json_messages = []
            self.binary_messages = []

        async def send_json(self, payload):
            self.json_messages.append(payload)

        async def send_bytes(self, payload):
            self.binary_messages.append(payload)

    async def run():
        websocket = FakeWebSocket()
        session = realtime_voice_api.RealtimeVoiceAsrTtsSession.__new__(
            realtime_voice_api.RealtimeVoiceAsrTtsSession
        )
        session._task_id = "task-1"
        session._event_builder = RealtimeVoiceEventBuilder("task-1")
        session._websocket = websocket
        session._send_lock = asyncio.Lock()
        session._playback_flush_lock = asyncio.Lock()
        session._first_audio_sent_jobs = set()
        session._playback_job_chunks = {}
        session._playback_job_meta = {}
        session._playback_jobs_done_queueing = set()
        session.playback_queue = TtsPlaybackQueue(maxsize=4, max_inflight=2)

        chunks = [
            PlaybackChunk(f"tts_1_chunk_{index}", "tts_1", 1, index, bytes([index]), 1000)
            for index in range(1, 4)
        ]
        for chunk in chunks:
            await session.playback_queue.put(chunk)

        await session._flush_playback_queue()

        audio_events = [
            message for message in websocket.json_messages if message["event"] == "tts.audio_chunk"
        ]
        assert [event["payload"]["chunk_id"] for event in audio_events] == [
            "tts_1_chunk_1",
            "tts_1_chunk_2",
        ]
        assert websocket.binary_messages == [b"\x01", b"\x02"]

        await session.mark_client_audio_played({"chunk_id": "tts_1_chunk_1"})

        audio_events = [
            message for message in websocket.json_messages if message["event"] == "tts.audio_chunk"
        ]
        assert [event["payload"]["chunk_id"] for event in audio_events] == [
            "tts_1_chunk_1",
            "tts_1_chunk_2",
            "tts_1_chunk_3",
        ]
        assert websocket.binary_messages == [b"\x01", b"\x02", b"\x03"]

    asyncio.run(run())
