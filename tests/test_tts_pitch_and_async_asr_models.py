# -*- coding: utf-8 -*-

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.database import DatabaseManager
from app.models.async_asr import AsyncASRRequest
from app.models.tts import TTSRequest
from app.utils.audio import adjust_audio_pitch, resample_audio_array
from app.utils.common import validate_pitch_rate_parameter


def test_tts_request_accepts_pitch_rate():
    request = TTSRequest(text="你好", voice="中文女", pitch_rate=120)

    assert request.pitch_rate == 120


def test_tts_request_rejects_pitch_rate_out_of_range():
    with pytest.raises(ValueError):
        TTSRequest(text="你好", voice="中文女", pitch_rate=501)


def test_validate_pitch_rate_parameter_uses_aliyun_range():
    assert validate_pitch_rate_parameter(-500)[0] is True
    assert validate_pitch_rate_parameter(500)[0] is True
    assert validate_pitch_rate_parameter(501)[0] is False


def test_adjust_audio_pitch_zero_is_noop():
    audio = np.array([0.0, 0.1, -0.1], dtype=np.float32)

    assert adjust_audio_pitch(audio, 22050, 0) is audio


def test_resample_audio_array_uses_long_axis_for_single_channel_model_output(
    monkeypatch,
):
    captured = {}

    def fake_resample(audio, orig_sr, target_sr):
        captured["audio"] = audio
        captured["orig_sr"] = orig_sr
        captured["target_sr"] = target_sr
        return audio

    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(resample=fake_resample))
    audio = np.arange(8, dtype=np.float32).reshape(1, 8)

    result = resample_audio_array(audio, 24000, 16000)

    assert result.shape == (8,)
    assert np.array_equal(captured["audio"], audio[0, :])
    assert captured["orig_sr"] == 24000
    assert captured["target_sr"] == 16000


def test_async_asr_request_shape_matches_long_recording_api():
    request = AsyncASRRequest(
        payload={
            "asr_request": {
                "audio_address": "https://example.com/long.wav",
                "format": "wav",
                "sample_rate": 16000,
                "hotwords": "FunSpeech",
                "disfluency": True,
            },
            "enable_notify": False,
        },
        header={"appkey": "app", "token": "0123456789"},
    )

    assert request.payload.asr_request.audio_address == "https://example.com/long.wav"
    assert request.payload.asr_request.enable_voice_detection is True
    assert request.payload.asr_request.hotwords == "FunSpeech"


def test_async_asr_request_accepts_audio_byte_array_without_url():
    request = AsyncASRRequest(
        payload={
            "asr_request": {
                "audio_bytes": [82, 73, 70, 70],
                "format": "wav",
                "sample_rate": 16000,
            },
            "enable_notify": False,
        },
        header={"appkey": "app", "token": "0123456789"},
    )

    assert request.payload.asr_request.audio_address is None
    assert request.payload.asr_request.audio_bytes == [82, 73, 70, 70]


def test_async_asr_request_requires_url_or_audio_bytes():
    with pytest.raises(ValueError):
        AsyncASRRequest(
            payload={
                "asr_request": {
                    "format": "wav",
                    "sample_rate": 16000,
                },
                "enable_notify": False,
            },
            header={"appkey": "app", "token": "0123456789"},
        )


def test_async_asr_database_persists_audio_bytes(tmp_path, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    DatabaseManager._instance = None
    manager = DatabaseManager()

    created = manager.create_asr_task(
        {
            "task_id": "task-bytes",
            "request_id": "request-bytes",
            "audio_address": None,
            "audio_bytes": bytes([82, 73, 70, 70]),
            "format": "wav",
            "sample_rate": 16000,
            "customization_id": "sensevoice-small",
            "enable_punctuation_prediction": False,
            "enable_inverse_text_normalization": False,
            "enable_voice_detection": True,
            "disfluency": False,
            "dolphin_lang_sym": "zh",
            "dolphin_region_sym": "SHANGHAI",
        }
    )
    task = manager.get_asr_task("task-bytes")

    assert created is True
    assert not task["audio_address"]
    assert bytes(task["audio_bytes"]) == b"RIFF"
