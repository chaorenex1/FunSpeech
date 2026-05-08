# -*- coding: utf-8 -*-

import asyncio

from app.services.realtime_voice.backpressure import BoundedAudioQueue, TtsJobQueue
from app.services.realtime_voice.text_commit import StableTextCommitter
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

        assert any(event.type == "dropped_audio" for event in events)
        assert queue.queued_ms == 100
        assert (await queue.get()).payload == b"voice-1"
        assert (await queue.get()).payload == b"voice-2"

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
