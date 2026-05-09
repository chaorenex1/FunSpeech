# -*- coding: utf-8 -*-

import asyncio

import numpy as np

import app.services.tts.engine as tts_engine_module
from app.services.tts.engine import CosyVoiceTTSEngine
from app.services.websocket_tts import AliyunWebSocketTTSService


class _FakeSpeech:
    def numpy(self):
        return np.array([[0.0, 0.1]], dtype=np.float32)


class _FakeWebSocket:
    class State:
        name = "CONNECTED"

    client_state = State()


class _FakeVoiceManager:
    def __init__(self, prompt_wav="C:/voices/desktop_voice.wav"):
        self.prompt_wav = prompt_wav

    def get_voice_audio_path(self, voice_name):
        return self.prompt_wav


def _make_saved_voice_engine(prompt_wav="C:/voices/desktop_voice.wav"):
    engine = CosyVoiceTTSEngine.__new__(CosyVoiceTTSEngine)
    engine.cosyvoice_clone = type("Clone", (), {"sample_rate": 16000})()
    engine._clone_model_version = "cosyvoice3"
    engine._voice_manager = _FakeVoiceManager(prompt_wav)
    engine.instruct_calls = []
    engine.zero_shot_calls = []

    def inference_instruct2(*args, **kwargs):
        engine.instruct_calls.append((args, kwargs))
        yield {"tts_speech": _FakeSpeech()}

    def inference_zero_shot(*args, **kwargs):
        engine.zero_shot_calls.append((args, kwargs))
        yield {"tts_speech": _FakeSpeech()}

    engine.inference_instruct2 = inference_instruct2
    engine.inference_zero_shot = inference_zero_shot
    return engine


def test_saved_voice_cosyvoice3_empty_prompt_uses_zero_shot(monkeypatch, tmp_path):
    engine = _make_saved_voice_engine()
    output_path = tmp_path / "out.wav"
    monkeypatch.setattr(
        tts_engine_module,
        "generate_temp_audio_path",
        lambda *_: str(output_path),
    )
    monkeypatch.setattr(tts_engine_module, "save_audio_array", lambda *_, **__: None)

    result = engine._synthesize_with_saved_voice(
        "你好",
        "desktop_voice",
        prompt="",
    )

    assert result == str(output_path)
    assert engine.instruct_calls == []
    assert len(engine.zero_shot_calls) == 1
    zero_shot_args, zero_shot_kwargs = engine.zero_shot_calls[0]
    assert zero_shot_args == ("你好", "", None)
    assert zero_shot_kwargs["zero_shot_spk_id"] == "desktop_voice"


def test_saved_voice_cosyvoice3_prompt_uses_prompt_wav_without_spk_id(
    monkeypatch,
    tmp_path,
):
    engine = _make_saved_voice_engine("C:/voices/desktop_voice.wav")
    output_path = tmp_path / "out.wav"
    monkeypatch.setattr(
        tts_engine_module,
        "generate_temp_audio_path",
        lambda *_: str(output_path),
    )
    monkeypatch.setattr(tts_engine_module, "save_audio_array", lambda *_, **__: None)

    result = engine._synthesize_with_saved_voice(
        "你好",
        "desktop_voice",
        prompt="像客服一样亲切",
    )

    assert result == str(output_path)
    assert engine.zero_shot_calls == []
    assert len(engine.instruct_calls) == 1
    instruct_args, instruct_kwargs = engine.instruct_calls[0]
    assert instruct_args[0] == "你好"
    assert "像客服一样亲切" in instruct_args[1]
    assert instruct_args[2] == "C:/voices/desktop_voice.wav"
    assert "zero_shot_spk_id" not in instruct_kwargs


def test_websocket_tts_cosyvoice3_empty_prompt_uses_zero_shot():
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

        def _get_saved_voice_prompt_wav(self, voice_name):
            return "C:/voices/desktop_voice.wav"

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
    assert engine.instruct_calls == []
    assert len(engine.zero_shot_calls) == 1
    zero_shot_args, zero_shot_kwargs = engine.zero_shot_calls[0]
    assert zero_shot_args == ("你好", "", None)
    assert zero_shot_kwargs["zero_shot_spk_id"] == "desktop_voice"


def test_websocket_tts_emotion_tag_maps_to_instruct_prompt():
    service = AliyunWebSocketTTSService()

    class FakeEngine:
        _clone_model_version = "cosyvoice3"
        cosyvoice_clone = type("Clone", (), {"sample_rate": 16000})()

        def __init__(self):
            self.instruct_calls = []

        def inference_instruct2(self, *args, **kwargs):
            self.instruct_calls.append((args, kwargs))
            yield {"tts_speech": _FakeSpeech()}

        def _get_saved_voice_prompt_wav(self, voice_name):
            return "C:/voices/desktop_voice.wav"

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
            prompt="像客服一样亲切",
            emotion="happy",
            emotion_intensity=0.9,
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())

    assert chunks
    instruct_args, _ = engine.instruct_calls[0]
    assert "开心" in instruct_args[1]
    assert "像客服一样亲切" in instruct_args[1]
    assert instruct_args[2] == "C:/voices/desktop_voice.wav"


def test_websocket_tts_preformatted_prompt_does_not_append_second_endofprompt():
    service = AliyunWebSocketTTSService()

    class FakeEngine:
        _clone_model_version = "cosyvoice3"
        cosyvoice_clone = type("Clone", (), {"sample_rate": 16000})()

        def __init__(self):
            self.instruct_calls = []

        def inference_instruct2(self, *args, **kwargs):
            self.instruct_calls.append((args, kwargs))
            yield {"tts_speech": _FakeSpeech()}

        def _get_saved_voice_prompt_wav(self, voice_name):
            return "C:/voices/desktop_voice.wav"

    engine = FakeEngine()
    prompt = (
        "You are a helpful assistant.<|endofprompt|>"
        "希望你以后能够做的比我还好呦。"
    )

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
            prompt=prompt,
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())

    assert chunks
    instruct_args, instruct_kwargs = engine.instruct_calls[0]
    assert instruct_args[1] == prompt
    assert instruct_args[2] == "C:/voices/desktop_voice.wav"
    assert instruct_args[1].count("<|endofprompt|>") == 1
    assert "zero_shot_spk_id" not in instruct_kwargs
