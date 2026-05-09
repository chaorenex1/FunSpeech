# -*- coding: utf-8 -*-
"""ASR streaming-session adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ...core.config import settings
from ...core.executor import run_sync
from ...utils.emotion import ASRTranscriptionResult, build_asr_result
from .vad import StreamingVADEndpointDetector


@dataclass
class StreamingASREvent:
    """Protocol-neutral streaming ASR event."""

    kind: str  # begin, partial, end
    time_ms: int
    text: str = ""
    raw_text: str = ""
    begin_time_ms: int = 0
    speech_begin_ms: Optional[int] = None
    speech_end_ms: Optional[int] = None
    vad_source: Optional[str] = None
    speech_active: bool = False
    itn_applied: bool = False
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    raw_rich_text: Optional[str] = None


class SenseVoiceWindowedStreamingSession:
    """Pseudo-streaming ASR using VAD endpoints plus SenseVoice offline windows."""

    def __init__(self, engine, params: dict):
        self.engine = engine
        sample_rate = params.get("sample_rate", 16000)
        self.sample_rate = int(sample_rate[0] if isinstance(sample_rate, list) else sample_rate)
        self.enable_itn = bool(params.get("enable_inverse_text_normalization", True))
        self.enable_vad = bool(params.get("enable_voice_detection", True))
        self.enable_emotion = bool(params.get("enable_emotion", False))
        self.return_rich_text = bool(params.get("return_rich_text", False))
        self.vad = StreamingVADEndpointDetector(engine, self.sample_rate)

        self.audio_time_ms = 0
        self.sentence_active = False
        self.sentence_begin_time_ms = 0
        self.current_sentence_audio = np.array([], dtype=np.float32)
        self.pre_speech_audio = np.array([], dtype=np.float32)
        self.last_partial_decode_ms = 0
        self.last_emitted_text = ""

    async def accept_audio(
        self,
        audio_array: np.ndarray,
        is_final: bool = False,
    ) -> List[StreamingASREvent]:
        audio_array = np.asarray(audio_array, dtype=np.float32)
        chunk_duration_ms = int(len(audio_array) / self.sample_rate * 1000)
        chunk_end_ms = self.audio_time_ms + chunk_duration_ms

        vad_event = await self.vad.accept_audio(audio_array, is_final=is_final)
        events: List[StreamingASREvent] = []

        if vad_event.speech_begin_ms is not None and not self.sentence_active:
            self._begin_sentence(vad_event.speech_begin_ms, audio_array)
            events.append(
                StreamingASREvent(
                    kind="begin",
                    time_ms=self.sentence_begin_time_ms,
                    begin_time_ms=self.sentence_begin_time_ms,
                    speech_begin_ms=vad_event.speech_begin_ms,
                    vad_source=vad_event.source,
                    speech_active=True,
                )
            )
        elif self.sentence_active and len(audio_array) > 0:
            self.current_sentence_audio = np.concatenate(
                [self.current_sentence_audio, audio_array]
            )
        else:
            self._remember_pre_speech(audio_array)

        if self.sentence_active:
            if await self._should_emit_partial(chunk_end_ms):
                asr_result = await self._decode_current_sentence_result(partial=True)
                text = asr_result.text
                if self._should_emit_text(text):
                    self.last_emitted_text = text
                    events.append(
                        StreamingASREvent(
                            kind="partial",
                            time_ms=chunk_end_ms,
                            text=text,
                            raw_text=text,
                            raw_rich_text=asr_result.raw_rich_text,
                            emotion=asr_result.emotion,
                            emotion_confidence=asr_result.emotion_confidence,
                            begin_time_ms=self.sentence_begin_time_ms,
                            speech_active=True,
                            vad_source=vad_event.source,
                            itn_applied=self.enable_itn,
                        )
                    )

            max_sentence_ms = min(
                settings.SENSEVOICE_MAX_SENTENCE_MS, settings.ASR_VAD_MAX_SENTENCE_MS
            )
            if chunk_end_ms - self.sentence_begin_time_ms >= max_sentence_ms:
                vad_event.speech_end_ms = chunk_end_ms

        if vad_event.speech_end_ms is not None and self.sentence_active:
            events.append(await self._end_sentence(vad_event.speech_end_ms))

        self.audio_time_ms = chunk_end_ms
        return events

    async def flush(self) -> List[StreamingASREvent]:
        if not self.sentence_active:
            return []
        return [await self._end_sentence(self.audio_time_ms)]

    def _begin_sentence(self, begin_time_ms: int, audio_array: np.ndarray) -> None:
        padded_begin = max(0, begin_time_ms - settings.ASR_VAD_SPEECH_PAD_MS)
        self.sentence_active = True
        self.sentence_begin_time_ms = padded_begin
        self.current_sentence_audio = np.concatenate(
            [self.pre_speech_audio, audio_array]
        )
        self.last_partial_decode_ms = padded_begin
        self.last_emitted_text = ""
        self.pre_speech_audio = np.array([], dtype=np.float32)

    async def _end_sentence(self, end_time_ms: int) -> StreamingASREvent:
        asr_result = await self._decode_current_sentence_result(partial=False)
        text = asr_result.text
        event = StreamingASREvent(
            kind="end",
            time_ms=max(end_time_ms, self.sentence_begin_time_ms),
            text=text,
            raw_text=text,
            raw_rich_text=asr_result.raw_rich_text,
            emotion=asr_result.emotion,
            emotion_confidence=asr_result.emotion_confidence,
            begin_time_ms=self.sentence_begin_time_ms,
            speech_end_ms=end_time_ms,
            speech_active=False,
            vad_source="vad",
            itn_applied=self.enable_itn,
        )
        self._reset_sentence()
        return event

    async def _should_emit_partial(self, chunk_end_ms: int) -> bool:
        window_ms = int(len(self.current_sentence_audio) / self.sample_rate * 1000)
        if window_ms < settings.SENSEVOICE_MIN_DECODE_WINDOW_MS:
            return False
        if chunk_end_ms - self.last_partial_decode_ms < settings.SENSEVOICE_PARTIAL_DECODE_INTERVAL_MS:
            return False
        self.last_partial_decode_ms = chunk_end_ms
        return True

    async def _decode_current_sentence(self, partial: bool = False) -> str:
        return (await self._decode_current_sentence_result(partial=partial)).text

    async def _decode_current_sentence_result(
        self, partial: bool = False
    ) -> ASRTranscriptionResult:
        if len(self.current_sentence_audio) == 0:
            return build_asr_result("")
        audio = self.current_sentence_audio
        if partial:
            max_partial_samples = int(
                self.sample_rate * settings.SENSEVOICE_MAX_PARTIAL_WINDOW_MS / 1000
            )
            if max_partial_samples > 0 and len(audio) > max_partial_samples:
                audio = audio[-max_partial_samples:]
        if hasattr(self.engine, "transcribe_array_with_metadata"):
            return await run_sync(
                self.engine.transcribe_array_with_metadata,
                audio,
                sample_rate=self.sample_rate,
                enable_itn=self.enable_itn,
                enable_vad=False,
                enable_emotion=self.enable_emotion,
                return_rich_text=self.return_rich_text,
            )
        text = await run_sync(
            self.engine.transcribe_array,
            audio,
            sample_rate=self.sample_rate,
            enable_itn=self.enable_itn,
            enable_vad=False,
        )
        return build_asr_result(text)

    def _should_emit_text(self, text: str) -> bool:
        if not text or text == self.last_emitted_text:
            return False
        old = self.last_emitted_text
        if not old:
            return True
        common = 0
        for old_ch, new_ch in zip(old, text):
            if old_ch != new_ch:
                break
            common += 1
        if len(text) + 3 < len(old) and common < int(len(old) * 0.6):
            return False
        return True

    def _remember_pre_speech(self, audio_array: np.ndarray) -> None:
        if len(audio_array) == 0:
            return
        self.pre_speech_audio = np.concatenate([self.pre_speech_audio, audio_array])
        max_samples = int(self.sample_rate * settings.ASR_VAD_SPEECH_PAD_MS / 1000)
        if len(self.pre_speech_audio) > max_samples:
            self.pre_speech_audio = self.pre_speech_audio[-max_samples:]

    def _reset_sentence(self) -> None:
        self.sentence_active = False
        self.sentence_begin_time_ms = 0
        self.current_sentence_audio = np.array([], dtype=np.float32)
        self.last_partial_decode_ms = self.audio_time_ms
        self.last_emitted_text = ""


class ParaformerStreamingSession:
    """Compatibility adapter for native online Paraformer models."""

    def __init__(self, engine, params: dict):
        self.engine = engine
        self.cache = {}
        sample_rate = params.get("sample_rate", 16000)
        self.sample_rate = int(sample_rate[0] if isinstance(sample_rate, list) else sample_rate)

    async def accept_audio(
        self, audio_array: np.ndarray, is_final: bool = False
    ) -> List[StreamingASREvent]:
        num_samples = len(audio_array)
        chunk_stride = 10 if num_samples >= 9600 else 4
        result = await run_sync(
            self.engine.realtime_model.generate,
            input=audio_array,
            cache=self.cache,
            is_final=is_final,
            chunk_size=[0, chunk_stride, 5],
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1,
        )
        text = result[0].get("text", "").strip() if result else ""
        if not text:
            return []
        time_ms = int(num_samples / self.sample_rate * 1000)
        return [StreamingASREvent(kind="partial", time_ms=time_ms, text=text, raw_text=text)]

    async def flush(self) -> List[StreamingASREvent]:
        return []
