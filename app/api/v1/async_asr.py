# -*- coding: utf-8 -*-
"""长录音异步ASR API路由。"""

import asyncio
import logging
import threading
import uuid
import time

import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from ...core.config import settings
from ...core.database import db_manager
from ...core.exceptions import (
    AuthenticationException,
    DefaultServerErrorException,
    InvalidParameterException,
)
from ...core.security import validate_request_appkey, validate_token_value
from ...models.async_asr import (
    AsyncASRErrorResponse,
    AsyncASRRequest,
    AsyncASRResponse,
    AsyncASRTaskData,
)
from ...services.asr.hotwords import resolve_hotwords
from ...services.asr.manager import get_model_manager
from ...utils.audio import (
    cleanup_temp_file,
    download_audio_from_url,
    get_audio_duration,
    get_audio_file_suffix,
    normalize_audio_for_asr,
    save_audio_to_temp_file,
)
from ...utils.text_processing import filter_disfluencies

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rest/v1/asr", tags=["Async ASR"])

_background_worker_started = False
_worker_lock = threading.Lock()


async def _send_notify_callback(notify_url: str, response_data: dict) -> bool:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                notify_url,
                json=response_data,
                headers={"Content-Type": "application/json"},
            )
            logger.info("异步ASR回调通知发送成功: %s, 状态码: %s", notify_url, response.status_code)
            return True
    except Exception as e:
        logger.error("异步ASR回调通知发送失败: %s, 错误: %s", notify_url, e)
        return False


def _send_notify_sync(notify_url: str, response_data: dict) -> bool:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_send_notify_callback(notify_url, response_data))
        finally:
            loop.close()
    except Exception as e:
        logger.error("异步ASR回调通知异常: %s, 错误: %s", notify_url, e)
        return False


def _process_async_asr_tasks() -> None:
    logger.info("异步ASR后台处理线程启动")

    while True:
        try:
            pending_tasks = db_manager.get_pending_asr_tasks(limit=2)

            for task in pending_tasks:
                task_id = task["task_id"]
                audio_path = None
                normalized_audio_path = None

                try:
                    logger.info("处理异步ASR任务: %s", task_id)

                    if task.get("audio_bytes"):
                        audio_data = bytes(task["audio_bytes"])
                        file_suffix = task["format"]
                    else:
                        audio_data = download_audio_from_url(
                            task["audio_address"],
                            max_size=settings.MAX_ASYNC_ASR_AUDIO_SIZE,
                        )
                        file_suffix = get_audio_file_suffix(task["audio_address"], task["format"])
                    audio_path = save_audio_to_temp_file(audio_data, file_suffix)
                    normalized_audio_path = normalize_audio_for_asr(
                        audio_path,
                        task["sample_rate"],
                    )
                    audio_duration = get_audio_duration(normalized_audio_path)

                    model_manager = get_model_manager()
                    asr_engine = model_manager.get_asr_engine(task["customization_id"])
                    hotwords = resolve_hotwords(task.get("vocabulary_id"), task.get("hotwords"))

                    asr_result = asr_engine.transcribe_file_with_metadata(
                        audio_path=normalized_audio_path,
                        hotwords=hotwords,
                        enable_punctuation=bool(task["enable_punctuation_prediction"]),
                        enable_itn=bool(task["enable_inverse_text_normalization"]),
                        enable_vad=bool(task["enable_voice_detection"]),
                        sample_rate=task["sample_rate"],
                        dolphin_lang_sym=task["dolphin_lang_sym"],
                        dolphin_region_sym=task["dolphin_region_sym"],
                        enable_emotion=bool(task.get("enable_emotion")),
                        return_rich_text=bool(task.get("return_rich_text")),
                    )
                    result_text = asr_result.text

                    if task["disfluency"]:
                        result_text = filter_disfluencies(result_text)

                    duration_ms = int(audio_duration * 1000)
                    db_manager.update_asr_task_status(
                        task_id,
                        "SUCCESS",
                        result=result_text,
                        duration_ms=duration_ms,
                        emotion=asr_result.emotion,
                        emotion_confidence=asr_result.emotion_confidence,
                        raw_rich_text=asr_result.raw_rich_text,
                        error_code=20000000,
                        error_message="SUCCESS",
                    )

                    if task.get("enable_notify") and task.get("notify_url"):
                        success_response = AsyncASRResponse(
                            status=200,
                            error_code=20000000,
                            error_message="SUCCESS",
                            request_id=str(uuid.uuid4()).replace("-", ""),
                            data=AsyncASRTaskData(
                                task_id=task_id,
                                result=result_text,
                                duration_ms=duration_ms,
                                emotion=asr_result.emotion,
                                emotion_confidence=asr_result.emotion_confidence,
                                raw_rich_text=asr_result.raw_rich_text,
                                notify_custom=task["notify_url"],
                            ),
                        )
                        _send_notify_sync(task["notify_url"], success_response.model_dump())

                except Exception as e:
                    logger.error("处理异步ASR任务失败: %s, 错误: %s", task_id, e)
                    db_manager.update_asr_task_status(
                        task_id,
                        "FAILED",
                        error_code=getattr(e, "status_code", 50000000),
                        error_message=str(e),
                    )

                    if task.get("enable_notify") and task.get("notify_url"):
                        error_response = AsyncASRErrorResponse(
                            error_message=str(e),
                            error_code=getattr(e, "status_code", 50000000),
                            request_id=str(uuid.uuid4()).replace("-", ""),
                            url="/rest/v1/asr/async",
                            status=500,
                        )
                        _send_notify_sync(task["notify_url"], error_response.model_dump())

                finally:
                    if audio_path:
                        cleanup_temp_file(audio_path)
                    if normalized_audio_path and normalized_audio_path != audio_path:
                        cleanup_temp_file(normalized_audio_path)

            time.sleep(2)

        except Exception as e:
            logger.error("异步ASR后台处理异常: %s", e)
            time.sleep(5)


def _start_background_worker() -> None:
    global _background_worker_started

    with _worker_lock:
        if not _background_worker_started:
            worker_thread = threading.Thread(target=_process_async_asr_tasks, daemon=True)
            worker_thread.start()
            _background_worker_started = True
            logger.info("异步ASR后台工作线程已启动")


@router.post(
    "/async",
    summary="提交长录音异步识别任务",
    description="提交长录音URL异步识别任务，返回task_id用于查询结果。",
)
async def submit_async_asr(request: Request, asr_request: AsyncASRRequest):
    _start_background_worker()

    request_id = str(uuid.uuid4()).replace("-", "")
    task_id = str(uuid.uuid4()).replace("-", "")

    try:
        if not asr_request.header.token:
            raise AuthenticationException("缺少访问令牌", task_id)

        if not validate_token_value(asr_request.header.token, settings.APPTOKEN):
            raise AuthenticationException("访问令牌无效", task_id)

        result, content = validate_request_appkey(asr_request.header.appkey, task_id)
        if not result:
            raise AuthenticationException(content, task_id)

        payload = asr_request.payload.asr_request
        if payload.audio_bytes is not None and len(payload.audio_bytes) > settings.MAX_ASYNC_ASR_AUDIO_SIZE:
            raise InvalidParameterException("audio_bytes超过长录音异步识别大小限制", task_id)

        if asr_request.payload.enable_notify:
            notify_url = asr_request.payload.notify_url
            if not notify_url:
                raise DefaultServerErrorException("启用回调通知时必须设置notify_url", task_id)
            if not notify_url.startswith(("http://", "https://")):
                raise InvalidParameterException("notify_url必须是有效的HTTP/HTTPS URL", task_id)

        task_data = {
            "task_id": task_id,
            "request_id": request_id,
            "audio_address": payload.audio_address,
            "audio_bytes": bytes(payload.audio_bytes) if payload.audio_bytes is not None else None,
            "format": payload.format,
            "sample_rate": payload.sample_rate,
            "vocabulary_id": payload.vocabulary_id,
            "hotwords": payload.hotwords,
            "customization_id": payload.customization_id,
            "enable_punctuation_prediction": payload.enable_punctuation_prediction,
            "enable_inverse_text_normalization": payload.enable_inverse_text_normalization,
            "enable_voice_detection": payload.enable_voice_detection,
            "disfluency": payload.disfluency,
            "enable_emotion": payload.enable_emotion,
            "return_rich_text": payload.return_rich_text,
            "dolphin_lang_sym": payload.dolphin_lang_sym,
            "dolphin_region_sym": payload.dolphin_region_sym,
            "enable_notify": asr_request.payload.enable_notify,
            "notify_url": asr_request.payload.notify_url if asr_request.payload.enable_notify else None,
        }

        if not db_manager.create_asr_task(task_data):
            raise DefaultServerErrorException("创建异步ASR任务失败", task_id)

        response_data = AsyncASRResponse(
            status=200,
            error_code=20000000,
            error_message="SUCCESS",
            request_id=request_id,
            data=AsyncASRTaskData(task_id=task_id),
        )
        return JSONResponse(content=response_data.model_dump())

    except (InvalidParameterException, AuthenticationException, DefaultServerErrorException) as e:
        logger.error("异步ASR提交失败: %s", e)
        error_response = AsyncASRErrorResponse(
            error_message=str(e),
            error_code=getattr(e, "status_code", 40000000),
            request_id=request_id,
            url="/rest/v1/asr/async",
            status=400 if isinstance(e, (InvalidParameterException, AuthenticationException)) else 500,
        )
        return JSONResponse(content=error_response.model_dump(), status_code=error_response.status)

    except Exception as e:
        logger.error("异步ASR未知异常: %s", e)
        error_response = AsyncASRErrorResponse(
            error_message=f"内部服务错误: {str(e)}",
            error_code=50000000,
            request_id=request_id,
            url="/rest/v1/asr/async",
            status=500,
        )
        return JSONResponse(content=error_response.model_dump(), status_code=500)


@router.get(
    "/async",
    summary="查询长录音异步识别结果",
    description="根据task_id查询异步ASR任务状态和最终识别文本。",
)
async def get_async_asr_result(
    request: Request,
    appkey: str = Query(..., description="应用Appkey"),
    token: str = Query(..., description="访问令牌"),
    task_id: str = Query(..., description="任务ID"),
):
    request_id = str(uuid.uuid4()).replace("-", "")

    try:
        if not validate_token_value(token, settings.APPTOKEN):
            raise AuthenticationException("访问令牌无效", task_id)

        result, content = validate_request_appkey(appkey, task_id)
        if not result:
            raise AuthenticationException(content, task_id)

        task = db_manager.get_asr_task(task_id)
        if not task:
            raise InvalidParameterException("任务不存在", task_id)

        data = AsyncASRTaskData(
            task_id=task_id,
            result=task.get("result") if task["status"] == "SUCCESS" else None,
            duration_ms=task.get("duration_ms") if task["status"] == "SUCCESS" else None,
            emotion=task.get("emotion") if task["status"] == "SUCCESS" else None,
            emotion_confidence=task.get("emotion_confidence") if task["status"] == "SUCCESS" else None,
            raw_rich_text=task.get("raw_rich_text") if task["status"] == "SUCCESS" else None,
            notify_custom=task.get("notify_url") if task.get("enable_notify") else None,
        )
        response_data = AsyncASRResponse(
            status=200,
            error_code=task["error_code"],
            error_message=task["error_message"],
            request_id=request_id,
            data=data,
        )
        return JSONResponse(content=response_data.model_dump())

    except (InvalidParameterException, AuthenticationException) as e:
        logger.error("查询异步ASR失败: %s", e)
        error_response = AsyncASRErrorResponse(
            error_message=str(e),
            error_code=getattr(e, "status_code", 40000000),
            request_id=request_id,
            url="/rest/v1/asr/async",
            status=400,
        )
        return JSONResponse(content=error_response.model_dump(), status_code=400)

    except Exception as e:
        logger.error("查询异步ASR未知异常: %s", e)
        error_response = AsyncASRErrorResponse(
            error_message=f"内部服务错误: {str(e)}",
            error_code=50000000,
            request_id=request_id,
            url="/rest/v1/asr/async",
            status=500,
        )
        return JSONResponse(content=error_response.model_dump(), status_code=500)
