# -*- coding: utf-8 -*-
"""长录音异步ASR数据模型。"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .common import AudioFormat, SampleRate


class AsyncASRRequestData(BaseModel):
    """异步ASR请求数据。"""

    audio_address: str = Field(..., description="长录音HTTP/HTTPS下载地址", max_length=2048)
    format: AudioFormat = Field("wav", description="音频格式")
    sample_rate: SampleRate = Field(16000, description="音频采样率")
    vocabulary_id: Optional[str] = Field(None, description="热词表ID", max_length=64)
    hotwords: Optional[str] = Field(None, description="临时热词列表", max_length=2048)
    customization_id: str = Field("sensevoice-small", description="ASR模型ID", max_length=64)
    enable_punctuation_prediction: bool = Field(False, description="是否启用标点")
    enable_inverse_text_normalization: bool = Field(False, description="是否启用ITN")
    enable_voice_detection: bool = Field(True, description="是否启用VAD，长录音默认启用")
    disfluency: bool = Field(False, description="是否过滤语气词")
    enable_emotion: bool = Field(False, description="是否返回ASR识别到的情感标签")
    return_rich_text: bool = Field(False, description="是否返回模型原始rich transcription文本")
    dolphin_lang_sym: str = Field("zh", description="Dolphin语言符号", max_length=8)
    dolphin_region_sym: str = Field("SHANGHAI", description="Dolphin区域符号", max_length=16)

    @field_validator("audio_address")
    @classmethod
    def validate_audio_address(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("audio_address不能为空")
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("audio_address必须是HTTP/HTTPS URL")
        return value


class AsyncASRPayload(BaseModel):
    """异步ASR载荷。"""

    asr_request: AsyncASRRequestData = Field(..., description="ASR请求数据")
    enable_notify: bool = Field(False, description="是否启用回调通知")
    notify_url: Optional[str] = Field(None, description="回调通知URL")


class AsyncASRHeader(BaseModel):
    """异步ASR请求头。"""

    appkey: str = Field(..., description="应用Appkey")
    token: str = Field(..., description="访问令牌")


class AsyncASRContext(BaseModel):
    """异步ASR上下文。"""

    device_id: Optional[str] = Field(None, description="设备ID")


class AsyncASRRequest(BaseModel):
    """异步ASR完整请求。"""

    payload: AsyncASRPayload = Field(..., description="请求载荷")
    context: Optional[AsyncASRContext] = Field(None, description="请求上下文")
    header: AsyncASRHeader = Field(..., description="请求头")


class AsyncASRTaskData(BaseModel):
    """异步ASR任务响应数据。"""

    task_id: str = Field(..., description="任务ID")
    result: Optional[str] = Field(None, description="识别结果文本")
    duration_ms: Optional[int] = Field(None, description="音频时长毫秒")
    emotion: Optional[str] = Field(None, description="识别到的情感标签")
    emotion_confidence: Optional[float] = Field(None, description="情感识别置信度")
    raw_rich_text: Optional[str] = Field(None, description="模型原始rich transcription文本")
    notify_custom: Optional[str] = Field(None, description="自定义通知数据")


class AsyncASRResponse(BaseModel):
    """异步ASR响应。"""

    status: int = Field(..., description="HTTP状态码")
    error_code: int = Field(..., description="错误码")
    error_message: str = Field(..., description="错误消息")
    request_id: str = Field(..., description="请求ID")
    data: Optional[AsyncASRTaskData] = Field(None, description="响应数据")


class AsyncASRErrorResponse(BaseModel):
    """异步ASR错误响应。"""

    error_message: str = Field(..., description="错误消息")
    error_code: int = Field(..., description="错误码")
    request_id: str = Field(..., description="请求ID")
    url: str = Field(..., description="请求URL")
    status: int = Field(..., description="HTTP状态码")
