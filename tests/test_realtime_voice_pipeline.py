# -*- coding: utf-8 -*-

import asyncio

from app.services.realtime_voice.backpressure import BoundedAudioQueue, TtsJobQueue
from app.services.realtime_voice.text_commit import StableTextCommitter
from app.services.realtime_voice.tts_dispatcher import RealtimeTTSDispatcher
from app.services.realtime_voice.types import AudioFrame, AsrHypothesis, TtsJob


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
        queue = BoundedAudioQueue(high_watermark_ms=20, max_ms=40)

        await queue.put(AudioFrame(b"one", duration_ms=20, is_silence=False, sequence=1, vad_state="speech", speech_active=True))
        await queue.put(AudioFrame(b"two", duration_ms=20, is_silence=False, sequence=2, vad_state="speech", speech_active=True))
        events = await queue.put(AudioFrame(b"three", duration_ms=20, is_silence=False, sequence=3, vad_state="speech", speech_active=True))

        dropped = [event for event in events if event.type == "drop_oldest_speech"]
        assert dropped
        assert dropped[0].first_dropped_seq == 1
        assert queue.queued_ms == 40

    asyncio.run(run())


def test_tts_job_queue_drops_stale_stable_jobs_and_prefers_final():
    async def run():
        queue = TtsJobQueue(maxsize=2)
        await queue.put(TtsJob(1, "旧", "voice", {}, "stable"))
        await queue.put(TtsJob(2, "新", "voice", {}, "stable"))
        await queue.put(TtsJob(3, "最终", "voice", {}, "final"))

        first = await queue.get()
        second = await queue.get()

        assert first.priority == "final"
        assert first.text == "最终"
        assert second.revision_id == 2

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
