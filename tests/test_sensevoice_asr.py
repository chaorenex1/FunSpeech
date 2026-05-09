# -*- coding: utf-8 -*-

import asyncio
import json
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.models.asr import ASRQueryParams
from app.services.asr.streaming import SenseVoiceWindowedStreamingSession
from app.services.asr.vad import StreamingVADEndpointDetector


def test_sensevoice_is_default_asr_model():
    config = json.loads(Path("app/services/asr/models.json").read_text(encoding="utf-8"))
    defaults = [
        model_id
        for model_id, model_config in config["models"].items()
        if model_config.get("default")
    ]

    assert defaults == ["sensevoice-small"]
    sensevoice = config["models"]["sensevoice-small"]
    assert sensevoice["models"]["offline"] == "iic/SenseVoiceSmall"
    assert sensevoice["streaming_strategy"] == "windowed_offline"
    assert sensevoice["supports_realtime"] is True


def test_asr_query_params_default_to_sensevoice():
    assert ASRQueryParams().customization_id == "sensevoice-small"


def test_vad_result_parser_normalizes_begin_and_end():
    detector = StreamingVADEndpointDetector(engine=object(), sample_rate=16000)

    begin = detector._parse_vad_result([{"value": [[120, -1]]}])
    assert begin.speech_begin_ms == 120
    assert begin.speech_end_ms is None
    assert detector.speech_active is True

    end = detector._parse_vad_result([{"value": [[-1, 980]]}])
    assert end.speech_begin_ms is None
    assert end.speech_end_ms == 980
    assert detector.speech_active is False


def test_vad_result_parser_handles_complete_segment():
    detector = StreamingVADEndpointDetector(engine=object(), sample_rate=16000)

    event = detector._parse_vad_result([{"value": [[100, 900]]}])

    assert event.speech_begin_ms == 100
    assert event.speech_end_ms == 900
    assert detector.speech_active is False


def test_sensevoice_streaming_session_uses_rms_fallback(monkeypatch):
    monkeypatch.setattr(settings, "ASR_ENABLE_VAD_ENDPOINT", False)
    monkeypatch.setattr(settings, "ASR_NEARFIELD_RMS_THRESHOLD", 0.01)
    monkeypatch.setattr(settings, "ASR_VAD_MIN_SPEECH_MS", 200)
    monkeypatch.setattr(settings, "ASR_VAD_END_FALLBACK_MS", 200)
    monkeypatch.setattr(settings, "ASR_VAD_SPEECH_PAD_MS", 0)
    monkeypatch.setattr(settings, "SENSEVOICE_MIN_DECODE_WINDOW_MS", 200)
    monkeypatch.setattr(settings, "SENSEVOICE_PARTIAL_DECODE_INTERVAL_MS", 0)
    monkeypatch.setattr(settings, "SENSEVOICE_MAX_SENTENCE_MS", 5000)
    monkeypatch.setattr(settings, "ASR_VAD_MAX_SENTENCE_MS", 5000)

    class FakeEngine:
        device = "cpu"

        def transcribe_array(self, audio_array, **kwargs):
            return "你好"

    session = SenseVoiceWindowedStreamingSession(
        FakeEngine(),
        {
            "sample_rate": 1000,
            "enable_inverse_text_normalization": False,
            "enable_voice_detection": True,
        },
    )

    async def run():
        voice_chunk = np.ones(200, dtype=np.float32) * 0.1
        silence_chunk = np.zeros(200, dtype=np.float32)
        first_events = await session.accept_audio(voice_chunk)
        second_events = await session.accept_audio(silence_chunk)
        return first_events + second_events

    events = asyncio.run(run())

    assert [event.kind for event in events] == ["begin", "partial", "end"]
    assert events[0].speech_begin_ms == 0
    assert events[0].speech_active is True
    assert events[-1].speech_end_ms is not None
    assert events[-1].speech_active is False
    assert events[-1].text == "你好"
