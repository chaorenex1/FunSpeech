# -*- coding: utf-8 -*-

import numpy as np
import pytest

from app.models.async_asr import AsyncASRRequest
from app.models.tts import TTSRequest
from app.utils.audio import adjust_audio_pitch
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
