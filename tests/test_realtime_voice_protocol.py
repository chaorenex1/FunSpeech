# -*- coding: utf-8 -*-

from fastapi.testclient import TestClient

from app.main import app
from app.services.realtime_voice.events import RealtimeVoiceEventBuilder


client = TestClient(app)


def test_realtime_voice_event_builder_adds_protocol_envelope():
    builder = RealtimeVoiceEventBuilder("task-1", session_id="session-1")

    first = builder.build("session_started", payload={"protocol_event": "session.started"})
    second = builder.build("configured", payload={"protocol_event": "session.configured"})

    assert first["schema_version"] == "realtime_voice.v1"
    assert first["session_id"] == "session-1"
    assert first["seq"] == 1
    assert isinstance(first["server_ts_ms"], int)
    assert second["seq"] == 2


def test_realtime_voice_lifecycle_events_include_precise_payload(monkeypatch):
    import app.api.v1.realtime_voice as realtime_voice_api

    class FakeTTSEngine:
        def get_voices(self):
            return ["desktop_voice"]

    class FakeRealtimeVoiceAsrTtsSession:
        def __init__(
            self,
            voice_name,
            audio_format="pcm",
            sample_rate=16000,
            parameters=None,
            event_builder=None,
        ):
            self.voice_name = voice_name
            self.audio_format = audio_format
            self.sample_rate = sample_rate
            self.parameters = parameters or {}
            self.config_version = 1

        def start(self, websocket, task_id):
            return None

        def update_parameters(self, parameters):
            self.parameters.update(parameters)
            self.config_version += 1
            return self.config_version

        async def aclose(self):
            return None

    monkeypatch.setattr(realtime_voice_api, "get_tts_engine", lambda: FakeTTSEngine())
    monkeypatch.setattr(
        realtime_voice_api,
        "RealtimeVoiceAsrTtsSession",
        FakeRealtimeVoiceAsrTtsSession,
    )

    with client.websocket_connect("/ws/v1/realtime/voice") as websocket:
        started = websocket.receive_json()
        assert started["event"] == "session_started"
        assert started["protocol_event"] == "session.started"
        assert started["schema_version"] == "realtime_voice.v1"
        assert started["seq"] == 1
        assert started["payload"]["supported_pipelines"] == ["asr_tts"]

        websocket.send_json(
            {
                "event": "configure",
                "voice_name": "desktop_voice",
                "format": "pcm",
                "sample_rate": 16000,
            }
        )
        configured = websocket.receive_json()
        assert configured["event"] == "configured"
        assert configured["protocol_event"] == "session.configured"
        assert configured["payload"]["config_version"] == 1
        assert configured["seq"] == 2

        websocket.send_json({"event": "update", "parameters": {"pitch": 1.1}})
        updated = websocket.receive_json()
        assert updated["event"] == "parameters_updated"
        assert updated["protocol_event"] == "session.parameters_updated"
        assert updated["payload"]["config_version"] == 2
        assert updated["seq"] == 3

