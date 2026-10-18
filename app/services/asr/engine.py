# -*- coding: utf-8 -*-
"""ASR 引擎抽象基类

模型推理已迁移到 services/* 子服务, 本模块只保留:
  - BaseASREngine / RealTimeASREngine 抽象基类(给 HTTP 客户端引擎实现)
  - ModelType 枚举
  - get_asr_engine() 全局实例入口

进程内的 FunASR / Dolphin / 多 GPU 调度 / VAD / PUNC 全局缓存已删除,
对应代码现存于 services/funasr/server.py 和 services/dolphin/server.py。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional

from ...utils.emotion import ASRTranscriptionResult

logger = logging.getLogger(__name__)


class ModelType(Enum):
    OFFLINE = "offline"
    REALTIME = "realtime"


class BaseASREngine(ABC):
    """基础 ASR 引擎抽象基类"""

    @abstractmethod
    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        enable_vad: bool = False,
        sample_rate: int = 16000,
        dolphin_lang_sym: str = "zh",
        dolphin_region_sym: str = "SHANGHAI",
    ) -> str:
        """转录音频文件"""

    def transcribe_file_with_metadata(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        enable_vad: bool = False,
        sample_rate: int = 16000,
        dolphin_lang_sym: str = "zh",
        dolphin_region_sym: str = "SHANGHAI",
        enable_emotion: bool = False,
        return_rich_text: bool = False,
    ) -> "ASRTranscriptionResult":
        """带元数据的离线转写。

        子服务只回传纯文本, 因此情感等元数据留空; 需要结构化结果的调用方
        (异步 ASR API / realtime pipeline) 依赖这一统一入口。
        """
        text = self.transcribe_file(
            audio_path=audio_path,
            hotwords=hotwords,
            enable_punctuation=enable_punctuation,
            enable_itn=enable_itn,
            enable_vad=enable_vad,
            sample_rate=sample_rate,
            dolphin_lang_sym=dolphin_lang_sym,
            dolphin_region_sym=dolphin_region_sym,
        )
        return ASRTranscriptionResult(
            text=text,
            raw_rich_text=text if return_rich_text else None,
        )

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载(对 HTTP 客户端 = 检查子服务健康)"""

    @property
    @abstractmethod
    def device(self) -> str:
        """获取设备信息(HTTP 客户端返回 'remote:<urls>')"""

    @property
    @abstractmethod
    def supports_realtime(self) -> bool:
        """是否支持实时识别"""


class RealTimeASREngine(BaseASREngine):
    """实时 ASR 引擎抽象基类"""

    @property
    def supports_realtime(self) -> bool:
        return True

    @abstractmethod
    def transcribe_websocket(
        self,
        audio_chunk: bytes,
        cache: Optional[Dict] = None,
        is_final: bool = False,
        **kwargs,
    ) -> str:
        """WebSocket 流式语音识别"""


# 全局 ASR 引擎实例缓存
_asr_engine: Optional[BaseASREngine] = None


def get_asr_engine() -> BaseASREngine:
    """获取全局默认 ASR 引擎实例(model manager 决定具体使用哪个模型)"""
    global _asr_engine
    if _asr_engine is None:
        from .manager import get_model_manager

        model_manager = get_model_manager()
        _asr_engine = model_manager.get_asr_engine()
    return _asr_engine
