# -*- coding: utf-8 -*-
"""Voice Cloner音色设计与voice_manager同步接口。"""

import os
from typing import Optional

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...core.config import settings
from ...core.exceptions import (
    APIException,
    AuthenticationException,
    DefaultServerErrorException,
    InvalidParameterException,
)
from ...core.security import validate_token
from ...models.common import AudioFormat, SampleRate
from ...utils.common import generate_task_id
from ...utils.audio import validate_audio_format, validate_sample_rate
from ...services.tts.engine import get_tts_engine
from ...services.tts.voice_design import generate_reference_audio


router = APIRouter(prefix="/voices/v1", tags=["Voice Cloner Voices"])


class VoiceDeleteRequest(BaseModel):
    voice_name: str = Field(..., min_length=1, max_length=64, description="唯一音色名称")


class VoiceDesignRequest(BaseModel):
    voice_name: str = Field(..., min_length=1, max_length=64, description="唯一音色名称")
    voice_instruction: str = Field(..., min_length=1, max_length=1000, description="桌面端LLM生成的音色设计指令")
    reference_text: str = Field(..., min_length=1, max_length=1000, description="参考音频对应文本")
    format: AudioFormat = Field(AudioFormat.WAV, description="参考音频格式")
    sample_rate: SampleRate = Field(SampleRate.RATE_24000, description="参考音频采样率")


def _error_response(exc: APIException, task_id: str = "") -> JSONResponse:
    exc.task_id = exc.task_id or task_id
    http_status = 500 if exc.status_code >= 50000000 else 400
    return JSONResponse(
        status_code=http_status,
        content={
            "task_id": exc.task_id,
            "result": "",
            "status": exc.status_code,
            "message": exc.message,
        },
        headers={"task_id": exc.task_id} if exc.task_id else {},
    )


def _validate_auth(request: Request, task_id: str = "") -> Optional[JSONResponse]:
    result, content = validate_token(request, task_id)
    if not result:
        return _error_response(AuthenticationException(content, task_id), task_id)
    return None


def _get_voice_manager():
    tts_engine = get_tts_engine()
    voice_manager = getattr(tts_engine, "voice_manager", None)
    if not voice_manager:
        raise DefaultServerErrorException("voice_manager未初始化，无法同步自定义音色")
    return tts_engine, voice_manager


def _voice_exists(tts_engine, voice_name: str) -> bool:
    voices = tts_engine.get_voices() if hasattr(tts_engine, "get_voices") else []
    return voice_name in voices


def _format_voice(voice_name: str, info: dict) -> dict:
    return {
        "voice_name": voice_name,
        "type": info.get("type", "clone"),
        "reference_text": info.get("reference_text", ""),
        "reference_audio": info.get("audio_file", ""),
        "voice_instruction": info.get("voice_instruction", ""),
        "status": info.get("status", "active"),
        "updated_at": info.get("updated_at") or info.get("added_at", ""),
    }


@router.get("/list", summary="Voice Cloner首次启动全量同步")
async def list_voices(request: Request):
    auth_error = _validate_auth(request, "voices_list")
    if auth_error:
        return auth_error

    try:
        tts_engine = get_tts_engine()
        voices_info = tts_engine.get_voices_info()
        voices = [_format_voice(name, info) for name, info in voices_info.items()]
        return {"voices": voices, "total": len(voices)}
    except APIException as exc:
        return _error_response(exc, "voices_list")
    except Exception as exc:
        return _error_response(DefaultServerErrorException(f"获取音色列表失败: {exc}"), "voices_list")


async def _parse_voice_sync_payload(request: Request) -> dict:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("reference_audio")
        audio_bytes = await upload.read() if upload is not None else None
        audio_filename = getattr(upload, "filename", None) if upload is not None else None
        return {
            "voice_name": form.get("voice_name"),
            "reference_text": form.get("reference_text"),
            "reference_audio_url": form.get("reference_audio_url"),
            "voice_instruction": form.get("voice_instruction"),
            "audio_bytes": audio_bytes,
            "audio_filename": audio_filename,
        }
    payload = await request.json()
    payload.setdefault("audio_bytes", None)
    payload.setdefault("audio_filename", None)
    return payload


async def _register_or_update(
    *,
    request: Request,
    voice_name: str,
    reference_text: str,
    audio_bytes: Optional[bytes],
    audio_filename: Optional[str],
    reference_audio_url: Optional[str],
    voice_instruction: Optional[str],
    overwrite: bool,
):
    task_id = generate_task_id("voice_sync")
    auth_error = _validate_auth(request, task_id)
    if auth_error:
        return auth_error

    try:
        voice_name = voice_name.strip()
        if voice_name in settings.PRESET_VOICES:
            raise InvalidParameterException(f"voice_name与预置音色冲突: {voice_name}", task_id)

        tts_engine, voice_manager = _get_voice_manager()
        exists = _voice_exists(tts_engine, voice_name)
        if exists and not overwrite:
            raise InvalidParameterException(f"voice_name已存在: {voice_name}", task_id)
        if overwrite and not exists:
            raise InvalidParameterException(f"voice_name不存在，无法更新: {voice_name}", task_id)

        info = voice_manager.register_voice_asset(
            voice_name=voice_name,
            reference_text=reference_text,
            audio_bytes=audio_bytes,
            audio_filename=audio_filename,
            reference_audio_url=reference_audio_url,
            voice_instruction=voice_instruction,
            overwrite=overwrite,
        )
        if hasattr(tts_engine, "refresh_voices"):
            tts_engine.refresh_voices()

        return JSONResponse(
            headers={"task_id": task_id},
            content={
                "task_id": task_id,
                "voice_name": voice_name,
                "status": "updated" if overwrite else "registered",
                "voice": _format_voice(voice_name, info),
            },
        )
    except APIException as exc:
        return _error_response(exc, task_id)
    except ValueError as exc:
        return _error_response(InvalidParameterException(str(exc), task_id), task_id)
    except Exception as exc:
        return _error_response(DefaultServerErrorException(f"音色同步失败: {exc}", task_id), task_id)


@router.post("/register", summary="增量注册桌面端自定义音色")
async def register_voice(request: Request):
    payload = await _parse_voice_sync_payload(request)
    return await _register_or_update(
        request=request,
        voice_name=payload.get("voice_name") or "",
        reference_text=payload.get("reference_text") or "",
        audio_bytes=payload.get("audio_bytes"),
        audio_filename=payload.get("audio_filename"),
        reference_audio_url=payload.get("reference_audio_url"),
        voice_instruction=payload.get("voice_instruction"),
        overwrite=False,
    )


@router.post("/update", summary="增量更新桌面端自定义音色")
async def update_voice(request: Request):
    payload = await _parse_voice_sync_payload(request)
    return await _register_or_update(
        request=request,
        voice_name=payload.get("voice_name") or "",
        reference_text=payload.get("reference_text") or "",
        audio_bytes=payload.get("audio_bytes"),
        audio_filename=payload.get("audio_filename"),
        reference_audio_url=payload.get("reference_audio_url"),
        voice_instruction=payload.get("voice_instruction"),
        overwrite=True,
    )


@router.post("/delete", summary="增量删除桌面端自定义音色")
async def delete_voice(request: Request, body: VoiceDeleteRequest = Body(...)):
    task_id = generate_task_id("voice_delete")
    auth_error = _validate_auth(request, task_id)
    if auth_error:
        return auth_error

    try:
        if body.voice_name in settings.PRESET_VOICES:
            raise InvalidParameterException(f"不能删除预置音色: {body.voice_name}", task_id)

        tts_engine, voice_manager = _get_voice_manager()
        removed = voice_manager.remove_voice(body.voice_name)
        if not removed:
            raise InvalidParameterException(f"voice_name不存在: {body.voice_name}", task_id)
        if hasattr(tts_engine, "refresh_voices"):
            tts_engine.refresh_voices()

        return JSONResponse(
            headers={"task_id": task_id},
            content={
                "task_id": task_id,
                "voice_name": body.voice_name,
                "status": "deleted",
            },
        )
    except APIException as exc:
        return _error_response(exc, task_id)
    except Exception as exc:
        return _error_response(DefaultServerErrorException(f"删除音色失败: {exc}", task_id), task_id)


@router.post("/refresh", summary="重新扫描并加载voice_manager音色")
async def refresh_voices(request: Request):
    task_id = generate_task_id("voice_refresh")
    auth_error = _validate_auth(request, task_id)
    if auth_error:
        return auth_error

    try:
        tts_engine, voice_manager = _get_voice_manager()
        success, total = voice_manager.refresh_voices()
        if hasattr(tts_engine, "refresh_voices"):
            tts_engine.refresh_voices()
        voices = tts_engine.get_voices() if hasattr(tts_engine, "get_voices") else []

        return JSONResponse(
            headers={"task_id": task_id},
            content={
                "task_id": task_id,
                "status": "refreshed",
                "success": success,
                "scanned": total,
                "voices": voices,
                "total": len(voices),
            },
        )
    except APIException as exc:
        return _error_response(exc, task_id)
    except Exception as exc:
        return _error_response(DefaultServerErrorException(f"刷新音色失败: {exc}", task_id), task_id)


@router.post("/voice-design", summary="VoxCPM音色设计参考音频生成")
async def voice_design(request: Request, body: VoiceDesignRequest):
    task_id = generate_task_id("voice_design")
    auth_error = _validate_auth(request, task_id)
    if auth_error:
        return auth_error

    try:
        audio_format = body.format.value if isinstance(body.format, AudioFormat) else str(body.format)
        sample_rate = int(body.sample_rate.value if isinstance(body.sample_rate, SampleRate) else body.sample_rate)

        if not validate_audio_format(audio_format):
            raise InvalidParameterException(f"不支持的音频格式: {body.format}", task_id)
        if not validate_sample_rate(sample_rate):
            raise InvalidParameterException(f"不支持的采样率: {body.sample_rate}", task_id)

        audio_path = generate_reference_audio(
            voice_name=body.voice_name,
            voice_instruction=body.voice_instruction,
            reference_text=body.reference_text,
            format=audio_format,
            sample_rate=sample_rate,
            task_id=task_id,
        )
        return JSONResponse(
            headers={"task_id": task_id},
            content={
                "task_id": task_id,
                "voice_name": body.voice_name,
                "reference_audio_url": f"/tmp/{os.path.basename(audio_path)}",
                "reference_text": body.reference_text,
                "status": "completed",
            },
        )
    except APIException as exc:
        return _error_response(exc, task_id)
    except Exception as exc:
        return _error_response(DefaultServerErrorException(f"音色设计失败: {exc}", task_id), task_id)
