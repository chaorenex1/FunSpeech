# -*- coding: utf-8 -*-
"""Voice Cloner一期新增接口契约测试。"""

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services.websocket_asr import AliyunWebSocketASRService
from app.services.websocket_tts import AliyunWebSocketTTSService


client = TestClient(app)


def _patch_voice_manager_paths(monkeypatch, tmp_path):
    import app.services.tts.clone.voice_manager as voice_manager_module

    voices_dir = tmp_path / "voices"
    spk_dir = voices_dir / "spk"
    monkeypatch.setattr(voice_manager_module, "VOICES_DIR", voices_dir)
    monkeypatch.setattr(
        voice_manager_module,
        "VOICE_REGISTRY_CONFIG",
        voices_dir / "voice_registry.json",
    )
    monkeypatch.setattr(voice_manager_module, "SPK_DIR", spk_dir)
    monkeypatch.setattr(voice_manager_module, "SPKINFO_FILE", spk_dir / "spk2info.pt")
    return voices_dir


class FakeCosyVoiceForRegistry:
    def __init__(self, existing_voices=None):
        self.frontend = type("Frontend", (), {"spk2info": dict(existing_voices or {})})()

    def add_zero_shot_spk(self, reference_text, wav_file, voice_name):
        self.frontend.spk2info[voice_name] = {
            "reference_text": reference_text,
            "wav_file": wav_file,
        }
        return True

    def save_spkinfo(self):
        pass


class DummyVoiceManager:
    def __init__(self):
        self.voices = {
            "desktop_voice": {
                "name": "desktop_voice",
                "reference_text": "你好，欢迎使用我的声音。",
                "audio_file": "desktop_voice.wav",
                "voice_instruction": "温暖、清晰",
                "status": "active",
            }
        }
        self.refreshed = False

    def list_clone_voices(self):
        return list(self.voices)

    def get_voice_info(self, voice_name):
        return self.voices.get(voice_name)

    def register_voice_asset(
        self,
        voice_name,
        reference_text,
        audio_bytes=None,
        audio_filename=None,
        reference_audio_url=None,
        voice_instruction=None,
        overwrite=False,
    ):
        if voice_name in self.voices and not overwrite:
            raise ValueError(f"音色已存在: {voice_name}")
        self.voices[voice_name] = {
            "name": voice_name,
            "reference_text": reference_text,
            "audio_file": audio_filename or reference_audio_url or f"{voice_name}.wav",
            "voice_instruction": voice_instruction or "",
            "status": "active",
        }
        return self.voices[voice_name]

    def remove_voice(self, voice_name):
        if voice_name not in self.voices:
            return False
        del self.voices[voice_name]
        return True

    def refresh_voices(self):
        self.refreshed = True
        return len(self.voices), len(self.voices)


class DummyTTSEngine:
    def __init__(self):
        self.voice_manager = DummyVoiceManager()

    def get_voices(self):
        return ["中文女", *self.voice_manager.list_clone_voices()]

    def get_voices_info(self):
        info = {
            "中文女": {
                "name": "中文女",
                "type": "preset",
                "language": "zh-CN",
                "gender": "female",
                "description": "标准中文女声",
                "sample_rate": 22050,
                "available": True,
            }
        }
        for name in self.voice_manager.list_clone_voices():
            voice = self.voice_manager.get_voice_info(name)
            info[name] = {
                "name": name,
                "type": "clone",
                "language": "zh-CN",
                "gender": "unknown",
                "description": f"零样本克隆音色：{name}",
                "sample_rate": 24000,
                "available": True,
                "reference_text": voice["reference_text"],
                "audio_file": voice["audio_file"],
                "voice_instruction": voice.get("voice_instruction", ""),
            }
        return info

    def refresh_voices(self):
        return self.voice_manager.refresh_voices()


def patch_voice_engine(monkeypatch):
    import app.api.v1.voices as voices_api
    import app.api.v1.realtime_voice as realtime_voice_api

    engine = DummyTTSEngine()
    monkeypatch.setattr(voices_api, "get_tts_engine", lambda: engine)
    monkeypatch.setattr(realtime_voice_api, "get_tts_engine", lambda: engine)
    return engine


def test_voice_manager_refresh_repairs_registry_when_spkinfo_already_has_voice(
    monkeypatch,
    tmp_path,
):
    from app.services.tts.clone import VoiceManager

    voices_dir = _patch_voice_manager_paths(monkeypatch, tmp_path)
    voices_dir.mkdir(parents=True, exist_ok=True)
    (voices_dir / "shenhenjp2.txt").write_text("参考文本", encoding="utf-8")
    (voices_dir / "shenhenjp2.wav").write_bytes(b"fake-wav")
    monkeypatch.setattr(VoiceManager, "_get_audio_duration", lambda *_: 2.0)

    manager = VoiceManager(FakeCosyVoiceForRegistry({"shenhenjp2": {}}))
    success, total = manager.refresh_voices()

    assert (success, total) == (1, 1)
    assert "shenhenjp2" in manager.list_clone_voices()
    registry = json.loads(
        (voices_dir / "voice_registry.json").read_text(encoding="utf-8")
    )
    assert registry["voices"]["shenhenjp2"]["audio_file"] == "shenhenjp2.wav"
    assert registry["voices"]["shenhenjp2"]["text_file"] == "shenhenjp2.txt"


def test_voice_manager_add_voice_persists_registry(monkeypatch, tmp_path):
    from app.services.tts.clone import VoiceManager

    voices_dir = _patch_voice_manager_paths(monkeypatch, tmp_path)
    voices_dir.mkdir(parents=True, exist_ok=True)
    txt_file = voices_dir / "new_voice.txt"
    wav_file = voices_dir / "new_voice.wav"
    txt_file.write_text("新增音色参考文本", encoding="utf-8")
    wav_file.write_bytes(b"fake-wav")
    monkeypatch.setattr(VoiceManager, "_validate_and_prepare_audio", lambda *_: True)
    monkeypatch.setattr(VoiceManager, "_get_audio_duration", lambda *_: 2.0)

    manager = VoiceManager(FakeCosyVoiceForRegistry())
    manager.cosyvoice.save_spkinfo = lambda: None

    assert manager.add_voice("new_voice", txt_file, wav_file) is True
    registry = json.loads(
        (voices_dir / "voice_registry.json").read_text(encoding="utf-8")
    )
    assert "new_voice" in registry["voices"]


def test_voice_manager_sync_endpoints(monkeypatch):
    engine = patch_voice_engine(monkeypatch)

    list_response = client.get("/voices/v1/list")
    assert list_response.status_code == 200
    list_data = list_response.json()
    assert list_data["total"] == 2
    assert any(v["voice_name"] == "desktop_voice" for v in list_data["voices"])

    register_response = client.post(
        "/voices/v1/register",
        json={
            "voice_name": "new_voice",
            "reference_text": "这是一段参考文本。",
            "voice_instruction": "年轻、自然",
            "reference_audio_url": "https://example.test/new_voice.wav",
        },
    )
    assert register_response.status_code == 200
    assert register_response.json()["voice_name"] == "new_voice"
    assert "new_voice" in engine.voice_manager.voices

    conflict_response = client.post(
        "/voices/v1/register",
        json={
            "voice_name": "中文女",
            "reference_text": "不能覆盖预设音色。",
            "reference_audio_url": "https://example.test/preset.wav",
        },
    )
    assert conflict_response.status_code == 400
    assert conflict_response.json()["status"] == 40000003

    update_response = client.post(
        "/voices/v1/update",
        json={
            "voice_name": "new_voice",
            "reference_text": "更新后的参考文本。",
            "voice_instruction": "更沉稳",
            "reference_audio_url": "https://example.test/new_voice.wav",
        },
    )
    assert update_response.status_code == 200
    assert engine.voice_manager.voices["new_voice"]["reference_text"] == "更新后的参考文本。"

    delete_response = client.post("/voices/v1/delete", json={"voice_name": "new_voice"})
    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "deleted"
    assert "new_voice" not in engine.voice_manager.voices

    refresh_response = client.post("/voices/v1/refresh")
    assert refresh_response.status_code == 200
    assert refresh_response.json()["status"] == "refreshed"
    assert engine.voice_manager.refreshed is True


def test_voice_design_endpoint_returns_generated_audio_url(monkeypatch, tmp_path):
    patch_voice_engine(monkeypatch)

    generated = tmp_path / "voice_design.wav"
    generated.write_bytes(b"RIFF....WAVEfmt ")

    def fake_generate_reference_audio(**kwargs):
        assert kwargs["voice_name"] == "designed_voice"
        assert kwargs["voice_instruction"] == "清亮、少年感"
        return str(generated)

    import app.api.v1.voices as voices_api

    monkeypatch.setattr(voices_api, "generate_reference_audio", fake_generate_reference_audio)

    response = client.post(
        "/voices/v1/voice-design",
        json={
            "voice_name": "designed_voice",
            "voice_instruction": "清亮、少年感",
            "reference_text": "今天的天气很好。",
            "format": "wav",
            "sample_rate": 24000,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["voice_name"] == "designed_voice"
    assert data["status"] == "completed"
    assert data["reference_audio_url"].endswith("voice_design.wav")


def test_realtime_voice_websocket_supports_config_update_and_audio_stream(monkeypatch):
    patch_voice_engine(monkeypatch)

    class FakeRealtimeVoiceAsrTtsSession:
        def __init__(self, voice_name, audio_format="pcm", sample_rate=16000, parameters=None):
            self.voice_name = voice_name
            self.audio_format = audio_format
            self.sample_rate = sample_rate
            self.parameters = parameters or {}

        async def process_audio(self, websocket, task_id, audio):
            await websocket.send_bytes(b"tts-audio")
            return True

    import app.api.v1.realtime_voice as realtime_voice_api

    monkeypatch.setattr(
        realtime_voice_api,
        "RealtimeVoiceAsrTtsSession",
        FakeRealtimeVoiceAsrTtsSession,
    )

    with client.websocket_connect("/ws/v1/realtime/voice") as websocket:
        started = websocket.receive_json()
        assert started["event"] == "session_started"
        assert started["task_id"]
        assert started["audio_mode"] == "asr_tts_pipeline"

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
        assert configured["voice_name"] == "desktop_voice"
        assert configured["pipeline"] == "asr_tts"
        assert configured["audio_mode"] == "asr_tts_pipeline"

        websocket.send_json({"event": "update", "parameters": {"pitch": 1.1}})
        updated = websocket.receive_json()
        assert updated["event"] == "parameters_updated"
        assert updated["parameters"]["pitch"] == 1.1

        websocket.send_bytes(b"\x00\x01\x02\x03")
        assert websocket.receive_bytes() == b"tts-audio"


def test_realtime_voice_websocket_supports_internal_asr_tts_pipeline(monkeypatch):
    patch_voice_engine(monkeypatch)

    class FakeRealtimeVoiceAsrTtsSession:
        def __init__(self, voice_name, audio_format="pcm", sample_rate=16000, parameters=None):
            self.voice_name = voice_name
            self.audio_format = audio_format
            self.sample_rate = sample_rate
            self.parameters = parameters or {}

        async def process_audio(self, websocket, task_id, audio):
            await websocket.send_json(
                {
                    "event": "asr_result",
                    "stage": "asr_text_received",
                    "task_id": task_id,
                    "text": "你好",
                    "is_final": False,
                }
            )
            await websocket.send_bytes(b"tts-audio")
            await websocket.send_json(
                {"event": "tts_completed", "stage": "tts_audio_sent", "task_id": task_id}
            )
            return True

    import app.api.v1.realtime_voice as realtime_voice_api

    monkeypatch.setattr(
        realtime_voice_api,
        "RealtimeVoiceAsrTtsSession",
        FakeRealtimeVoiceAsrTtsSession,
    )

    with client.websocket_connect("/ws/v1/realtime/voice") as websocket:
        started = websocket.receive_json()
        assert started["event"] == "session_started"
        assert "asr_tts" in started["supported_pipelines"]

        websocket.send_json(
                {
                    "event": "configure",
                    "voice_name": "desktop_voice",
                    "pipeline": "asr_tts",
                    "format": "pcm",
                    "sample_rate": 16000,
                    "parameters": {"pitch": 1.1},
            }
        )
        configured = websocket.receive_json()
        assert configured["event"] == "configured"
        assert configured["audio_mode"] == "asr_tts_pipeline"

        websocket.send_bytes(b"\x00\x01\x02\x03")
        asr_result = websocket.receive_json()
        assert asr_result["event"] == "asr_result"
        assert asr_result["text"] == "你好"
        assert websocket.receive_bytes() == b"tts-audio"
        completed = websocket.receive_json()
        assert completed["event"] == "tts_completed"


def test_websocket_asr_messages_include_voice_cloner_fields():
    service = AliyunWebSocketASRService()
    sent = []

    class FakeWebSocket:
        async def send_text(self, message):
            sent.append(json.loads(message))

    async def run():
        await service._send_transcription_result_changed(
            FakeWebSocket(), "task-1", 1, 320, "中间结果"
        )
        await service._send_sentence_end(
            FakeWebSocket(), "task-1", 1, 640, "最终结果", begin_time=320
        )

    asyncio.run(run())

    interim = sent[0]["payload"]
    final = sent[1]["payload"]
    assert interim["task_id"] == "task-1"
    assert interim["text"] == "中间结果"
    assert interim["is_final"] is False
    assert interim["duration_ms"] == 320
    assert interim["confidence"] is None
    assert final["task_id"] == "task-1"
    assert final["text"] == "最终结果"
    assert final["is_final"] is True
    assert final["duration_ms"] == 320


def test_websocket_tts_refreshes_registered_clone_voice_before_synthesis(monkeypatch):
    service = AliyunWebSocketTTSService()

    class FakeVoiceManager:
        def __init__(self):
            self.loaded = False
            self.refresh_count = 0

        def list_clone_voices(self):
            return ["shenhenjp2"]

        def is_voice_available(self, voice_name):
            return voice_name == "shenhenjp2" and self.loaded

        def refresh_voices(self):
            self.refresh_count += 1
            self.loaded = True
            return 1, 1

    class FakeEngine:
        def __init__(self):
            self._voice_manager = FakeVoiceManager()

    class FakeWebSocket:
        class State:
            name = "CONNECTED"

        client_state = State()

    async def fake_stream_clone_voice_with_engine(*args, **kwargs):
        yield b"tts-audio"

    fake_engine = FakeEngine()
    service.tts_engine = fake_engine
    monkeypatch.setattr(
        service,
        "_stream_clone_voice_with_engine",
        fake_stream_clone_voice_with_engine,
    )

    async def run():
        chunks = []
        async for chunk in service._synthesize_streaming_audio(
            "你好",
            "shenhenjp2",
            1.0,
            "PCM",
            16000,
            50,
            0,
            "task-1",
            FakeWebSocket(),
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())

    assert chunks == [b"tts-audio"]
    assert fake_engine._voice_manager.refresh_count == 1


def test_websocket_tts_rejects_unknown_non_preset_voice_before_sft_keyerror():
    service = AliyunWebSocketTTSService()

    class FakeEngine:
        _voice_manager = None
        cosyvoice_sft = object()

    class FakeWebSocket:
        class State:
            name = "CONNECTED"

        client_state = State()

    service.tts_engine = FakeEngine()

    async def run():
        chunks = []
        async for chunk in service._synthesize_streaming_audio(
            "你好",
            "missing_voice",
            1.0,
            "PCM",
            16000,
            50,
            0,
            "task-1",
            FakeWebSocket(),
        ):
            chunks.append(chunk)

    try:
        asyncio.run(run())
    except Exception as exc:
        assert "voice_name不存在或未同步到FunSpeech" in str(exc)
    else:
        raise AssertionError("unknown voice should be rejected before SFT inference")
