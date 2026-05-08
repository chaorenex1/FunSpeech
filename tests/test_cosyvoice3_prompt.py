# -*- coding: utf-8 -*-

import asyncio

import numpy as np

from app.services.websocket_tts import AliyunWebSocketTTSService


class _FakeSpeech:
    def numpy(self):
        return np.array([[0.0, 0.1]], dtype=np.float32)


class _FakeWebSocket:
    class State:
        name = "CONNECTED"

    client_state = State()


def test_websocket_tts_cosyvoice3_empty_prompt_uses_default_instruct_prompt():
    service = AliyunWebSocketTTSService()

    class FakeEngine:
        _clone_model_version = "cosyvoice3"
        cosyvoice_clone = type("Clone", (), {"sample_rate": 16000})()

        def __init__(self):
            self.instruct_calls = []
            self.zero_shot_calls = []

        def inference_instruct2(self, *args, **kwargs):
            self.instruct_calls.append((args, kwargs))
            yield {"tts_speech": _FakeSpeech()}

        def inference_zero_shot(self, *args, **kwargs):
            self.zero_shot_calls.append((args, kwargs))
            yield {"tts_speech": _FakeSpeech()}

    engine = FakeEngine()

    async def run():
        chunks = []
        async for chunk in service._stream_clone_voice_with_engine(
            "你好",
            "desktop_voice",
            1.0,
            "PCM",
            16000,
            50,
            0,
            "task-1",
            _FakeWebSocket(),
            engine,
            prompt="",
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())

    assert chunks
    assert len(engine.instruct_calls) == 1
    assert engine.zero_shot_calls == []
    instruct_args, instruct_kwargs = engine.instruct_calls[0]
    assert instruct_args[1] == "You are a helpful assistant.<|endofprompt|>"
    assert instruct_kwargs["zero_shot_spk_id"] == "desktop_voice"

