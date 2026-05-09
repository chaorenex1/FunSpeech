# -*- coding: utf-8 -*-

import asyncio

import numpy as np

from app.api.v1.realtime_voice import SlidingVadSegmenter
from app.services.realtime_voice.audio_pacer import AudioPacer
from app.services.realtime_voice.backpressure import AsrSegmentQueue, BoundedAudioQueue, TtsJobQueue
from app.services.realtime_voice.text_commit import StableTextCommitter
from app.services.realtime_voice.tts_dispatcher import RealtimeTTSDispatcher
from app.services.realtime_voice.types import AsrSegment, AudioFrame, AsrHypothesis, TtsJob


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


def _asr_segment(sequence: int, duration_ms: int = 40, final: bool = False) -> AsrSegment:
    return AsrSegment(
        payload=f"seg-{sequence}".encode(),
        duration_ms=duration_ms,
        frame_count=max(1, duration_ms // 20),
        utterance_id="utt_1",
        first_frame_seq=sequence,
        last_frame_seq=sequence,
        is_final=final,
        vad_source="test",
    )


def test_sliding_vad_segmenter_emits_preroll_once_before_voice():
    segmenter = SlidingVadSegmenter(window_ms=40, pre_roll_ms=40, end_silence_ms=40)

    assert segmenter.accept(_audio_frame(1, 0), "utt_1") == []
    assert segmenter.accept(_audio_frame(2, 0), "utt_1") == []
    assert segmenter.accept(_audio_frame(3, 2400), "utt_1") == []
    segments = segmenter.accept(_audio_frame(4, 2400), "utt_1")

    assert len(segments) == 1
    assert segments[0].first_frame_seq == 1
    assert segments[0].last_frame_seq == 4
    assert segments[0].frame_count == 4
    assert segments[0].duration_ms == 80
    assert segments[0].is_final is False


def test_sliding_vad_segmenter_keeps_continuous_voice_smooth_and_finalizes_on_silence():
    segmenter = SlidingVadSegmenter(window_ms=40, pre_roll_ms=40, end_silence_ms=40)

    segmenter.accept(_audio_frame(1, 2400), "utt_1")
    first = segmenter.accept(_audio_frame(2, 2400), "utt_1")
    segmenter.accept(_audio_frame(3, 2400), "utt_1")
    second = segmenter.accept(_audio_frame(4, 2400), "utt_1")
    segmenter.accept(_audio_frame(5, 0), "utt_1")
    final = segmenter.accept(_audio_frame(6, 0), "utt_1")

    assert [segment.first_frame_seq for segment in first + second] == [1, 3]
    assert [segment.last_frame_seq for segment in first + second] == [2, 4]
    assert final and final[0].is_final is True
    assert final[0].first_frame_seq == 5
    assert final[0].last_frame_seq == 6


def test_stable_text_committer_emits_incremental_stable_text():
    committer = StableTextCommitter(
        stable_hypotheses=2,
        min_commit_chars=3,
        max_commit_wait_ms=10_000,
    )

    assert committer.update(AsrHypothesis("你好今")) is None

    committed = committer.update(AsrHypothesis("你好今天"))
    assert committed is not None
    assert committed.text == "你好今"
    assert committed.full_text == "你好今"
    assert committed.is_final is False

    assert committer.update(AsrHypothesis("你好")) is None

    final = committer.update(AsrHypothesis("你好今天。", is_final=True))
    assert final is not None
    assert final.text == "天。"
    assert final.is_final is True


def test_stable_text_committer_resets_sentence_after_final():
    committer = StableTextCommitter(stable_hypotheses=1, min_commit_chars=1)

    first = committer.update(AsrHypothesis("你好。", is_final=True))
    assert first is not None
    committer.reset_sentence()

    second = committer.update(AsrHypothesis("再见。", is_final=True))
    assert second is not None
    assert second.text == "再见。"
    assert second.full_text == "再见。"


def test_stable_text_committer_appends_rolling_window_overlap():
    committer = StableTextCommitter(stable_hypotheses=1, min_commit_chars=1)

    first = committer.update(AsrHypothesis("今天我们去公园"))
    assert first is not None
    assert first.text == "今天我们去公园"

    rolling = committer.update(AsrHypothesis("去公园散步聊天"))

    assert rolling is not None
    assert rolling.text == "散步聊天"
    assert rolling.full_text == "今天我们去公园散步聊天"


def test_stable_text_committer_does_not_recommit_rolling_window_subset():
    committer = StableTextCommitter(stable_hypotheses=1, min_commit_chars=1)

    assert committer.update(AsrHypothesis("今天我们去公园")) is not None

    assert committer.update(AsrHypothesis("我们去公园")) is None


def test_stable_text_committer_bounds_speculative_delta_size():
    committer = StableTextCommitter(
        stable_hypotheses=1,
        min_commit_chars=2,
        max_commit_chars=4,
        max_commit_wait_ms=10_000,
    )

    first = committer.update(AsrHypothesis("今天我们一起去公园"))
    second = committer.update(AsrHypothesis("今天我们一起去公园"))

    assert first is not None
    assert first.text == "今天我们"
    assert second is not None
    assert second.text == "一起去公"


def test_bounded_audio_queue_drops_silence_before_speech_when_over_budget():
    async def run():
        queue = BoundedAudioQueue(high_watermark_ms=60, max_ms=100)

        await queue.put(AudioFrame(b"voice-1", duration_ms=50, is_silence=False))
        await queue.put(AudioFrame(b"silence", duration_ms=50, is_silence=True))
        events = await queue.put(AudioFrame(b"voice-2", duration_ms=50, is_silence=False))

        assert any(event.type == "drop_pre_silence" for event in events)
        assert queue.queued_ms == 100
        assert (await queue.get()).payload == b"voice-1"
        assert (await queue.get()).payload == b"voice-2"

    asyncio.run(run())


def test_bounded_audio_queue_preserves_vad_metadata_on_backpressure():
    async def run():
        queue = BoundedAudioQueue(high_watermark_ms=10, max_ms=20)

        await queue.put(AudioFrame(b"voice", duration_ms=10, is_silence=False))
        events = await queue.put(
            AudioFrame(
                b"silence",
                duration_ms=20,
                is_silence=True,
                sequence=7,
                vad_state="silence",
            )
        )

        assert events[0].type == "drop_vad_silence"
        assert queue.queued_ms == 10

    asyncio.run(run())


def test_bounded_audio_queue_drops_speech_like_before_active_speech():
    async def run():
        queue = BoundedAudioQueue(high_watermark_ms=20, max_ms=40)

        await queue.put(
            AudioFrame(
                b"active",
                duration_ms=20,
                is_silence=False,
                sequence=1,
                vad_state="speech",
                speech_active=True,
            )
        )
        await queue.put(
            AudioFrame(
                b"speech-like",
                duration_ms=20,
                is_silence=False,
                sequence=2,
                pre_class="rms_voice",
                vad_state="pending",
            )
        )
        events = await queue.put(
            AudioFrame(
                b"active-2",
                duration_ms=20,
                is_silence=False,
                sequence=3,
                vad_state="speech",
                speech_active=True,
            )
        )

        assert any(event.type == "drop_speech_like" for event in events)
        assert queue.queued_ms == 40
        assert (await queue.get()).payload == b"active"
        assert (await queue.get()).payload == b"active-2"

    asyncio.run(run())


def test_bounded_audio_queue_reports_oldest_speech_when_no_lower_layer_exists():
    async def run():
        queue = BoundedAudioQueue(high_watermark_ms=20, max_ms=40, preserve_speech=False)

        await queue.put(AudioFrame(b"one", duration_ms=20, is_silence=False, sequence=1, vad_state="speech", speech_active=True))
        await queue.put(AudioFrame(b"two", duration_ms=20, is_silence=False, sequence=2, vad_state="speech", speech_active=True))
        events = await queue.put(AudioFrame(b"three", duration_ms=20, is_silence=False, sequence=3, vad_state="speech", speech_active=True))

        dropped = [event for event in events if event.type == "drop_oldest_speech"]
        assert dropped
        assert dropped[0].first_dropped_seq == 1
        assert queue.queued_ms == 40

    asyncio.run(run())


def test_bounded_audio_queue_preserves_active_speech_by_default():
    async def run():
        queue = BoundedAudioQueue(high_watermark_ms=20, max_ms=40)

        await queue.put(AudioFrame(b"one", duration_ms=20, is_silence=False, sequence=1, vad_state="speech", speech_active=True))
        await queue.put(AudioFrame(b"two", duration_ms=20, is_silence=False, sequence=2, vad_state="speech", speech_active=True))
        events = await queue.put(AudioFrame(b"three", duration_ms=20, is_silence=False, sequence=3, vad_state="speech", speech_active=True))

        assert any(event.type == "input_preserve_speech_backpressure" for event in events)
        assert queue.queued_ms == 60

    asyncio.run(run())


def test_asr_segment_queue_coalesces_when_pressure_is_high():
    async def run():
        queue = AsrSegmentQueue(high_watermark_ms=40, max_ms=160)

        await queue.put(_asr_segment(1, duration_ms=40))
        events = await queue.put(_asr_segment(2, duration_ms=40))

        assert any(event.type == "asr_segments_coalesced" for event in events)
        assert queue.queued_ms == 80
        segment = await queue.get()
        assert segment.duration_ms == 80
        assert segment.frame_count == 4
        assert segment.payload == b"seg-1seg-2"
        assert segment.first_frame_seq == 1
        assert segment.last_frame_seq == 2

    asyncio.run(run())


def test_asr_segment_queue_drops_speech_over_high_water_budget_but_keeps_final_marker():
    async def run():
        queue = AsrSegmentQueue(high_watermark_ms=40, max_ms=80, preserve_speech=False)

        await queue.put(_asr_segment(1, duration_ms=40))
        await queue.put(_asr_segment(2, duration_ms=40))
        events = await queue.put(_asr_segment(3, duration_ms=40, final=True))

        assert any(event.type == "drop_asr_speech" for event in events)
        assert queue.queued_ms <= 80
        queued = await queue.get()
        assert queued.is_final is True

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


def test_tts_job_queue_drops_stale_stable_jobs_and_prefers_final():
    async def run():
        queue = TtsJobQueue(maxsize=2, drop_on_overload=True)
        await queue.put(TtsJob(1, "旧", "voice", {}, "stable"))
        await queue.put(TtsJob(2, "新", "voice", {}, "stable"))
        await queue.put(TtsJob(3, "最终", "voice", {}, "final"))

        first = await queue.get()
        second = await queue.get()

        assert first.priority == "final"
        assert first.text == "最终"
        assert second.revision_id == 2

    asyncio.run(run())


def test_tts_job_queue_preserves_jobs_by_default_when_overloaded():
    async def run():
        queue = TtsJobQueue(maxsize=2)
        await queue.put(TtsJob(1, "甲", "voice", {}, "final"))
        await queue.put(TtsJob(2, "乙", "voice", {}, "final"))
        events = await queue.put(TtsJob(3, "丙", "voice", {}, "stable"))

        assert any(event.type == "tts_queue_preserved" for event in events)
        assert (await queue.get()).text == "甲"
        assert (await queue.get()).text == "乙"
        assert (await queue.get()).text == "丙"

    asyncio.run(run())


def test_tts_job_queue_preserves_small_stable_jobs_when_waiting():
    async def run():
        queue = TtsJobQueue(maxsize=1)

        await queue.put(TtsJob(1, "你", "voice", {}, "stable"))
        events = await queue.put(TtsJob(2, "好", "voice", {}, "stable"))

        assert any(event.type == "tts_queue_preserved" for event in events)
        first = await queue.get()
        second = await queue.get()
        assert first.text == "你"
        assert second.text == "好"

    asyncio.run(run())


def test_tts_job_queue_keeps_small_stable_jobs_separate_before_pressure():
    async def run():
        queue = TtsJobQueue(maxsize=3)

        assert await queue.put(TtsJob(1, "你", "voice", {}, "stable")) == []
        assert await queue.put(TtsJob(2, "好", "voice", {}, "stable")) == []

        first = await queue.get()
        second = await queue.get()
        assert first.text == "你"
        assert second.text == "好"

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


def test_audio_pacer_can_burst_initial_frames_to_seed_client_jitter_buffer():
    async def run():
        pacer = AudioPacer(sample_rate=1000, frame_ms=20, burst_ms=100)
        audio = b"\x01\x00" * 100
        started = asyncio.get_running_loop().time()
        frames = [frame async for frame in pacer.iter_frames(audio)]
        elapsed_ms = int((asyncio.get_running_loop().time() - started) * 1000)

        assert len(frames) == 5
        assert elapsed_ms < 50

    asyncio.run(run())


def test_audio_pacer_slows_when_client_playback_queue_is_high():
    async def run():
        pacer = AudioPacer(
            sample_rate=1000,
            frame_ms=20,
            burst_ms=0,
            target_queue_ms=10,
            high_queue_ms=20,
            max_backpressure_sleep_ms=20,
        )
        pacer.update_client_queue_ms(20)
        audio = b"\x01\x00" * 20
        started = asyncio.get_running_loop().time()
        frames = [frame async for frame in pacer.iter_frames(audio)]
        elapsed_ms = int((asyncio.get_running_loop().time() - started) * 1000)

        assert len(frames) == 1
        assert elapsed_ms >= 15

    asyncio.run(run())
