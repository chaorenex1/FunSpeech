# -*- coding: utf-8 -*-
"""异步执行器

把同步阻塞调用(主要是 HTTP 客户端 / 同步 WS session)派发到线程池,
避免阻塞 FastAPI 的事件循环; 同时提供同步生成器 → 异步生成器的桥接。

线程数由 INFERENCE_THREAD_POOL_SIZE 控制, 默认 max(32, CPU 核数 * 8)。
"""

import os
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, TypeVar, Generator, AsyncGenerator
from functools import partial

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar("T")

# 全局线程池执行器
# 线程大部分时间阻塞在同步 WS recv() 等网络 I/O 上 (I/O-bound),
# 默认线程数从 cpu_count 调整为 cpu_count*8 (最低 32),
# 避免多并发 ASR 连接时线程池被耗尽导致服务不可用。
_DEFAULT_WORKERS = max(32, (os.cpu_count() or 4) * 8)
_MAX_WORKERS = int(os.getenv("INFERENCE_THREAD_POOL_SIZE", str(_DEFAULT_WORKERS)))

_executor: ThreadPoolExecutor = None


def get_executor() -> ThreadPoolExecutor:
    """获取全局线程池执行器（懒加载）"""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=_MAX_WORKERS,
            thread_name_prefix="inference_worker"
        )
        logger.info(f"推理线程池已创建，最大工作线程数: {_MAX_WORKERS}")
    return _executor


def shutdown_executor():
    """关闭线程池执行器"""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
        logger.info("推理线程池已关闭")


async def run_sync(func: Callable[..., T], *args, **kwargs) -> T:
    """
    在线程池中执行同步函数，不阻塞事件循环

    Args:
        func: 同步函数
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        函数返回值

    Example:
        result = await run_sync(model.generate, input=audio_array, cache=cache)
    """
    loop = asyncio.get_running_loop()
    executor = get_executor()

    # 使用 partial 绑定参数
    if kwargs:
        func_with_args = partial(func, *args, **kwargs)
    else:
        func_with_args = partial(func, *args) if args else func

    return await loop.run_in_executor(executor, func_with_args)


async def run_sync_generator(
    generator_func: Callable[..., Generator[T, None, None]],
    *args,
    max_queue_size: int = 0,
    cancel_event: threading.Event | None = None,
    **kwargs
) -> AsyncGenerator[T, None]:
    """
    将同步生成器转换为异步生成器，在线程池中执行

    用于流式TTS等需要逐步产出结果的场景。

    Args:
        generator_func: 返回生成器的同步函数
        *args: 位置参数
        **kwargs: 关键字参数

    Yields:
        生成器的每个产出值

    Example:
        async for chunk in run_sync_generator(model.inference_sft, text, voice, stream=True):
            await websocket.send_bytes(chunk)
    """
    loop = asyncio.get_running_loop()
    executor = get_executor()
    queue: asyncio.Queue = asyncio.Queue(maxsize=max(0, int(max_queue_size or 0)))
    stop_event = cancel_event or threading.Event()

    # 标记生成器结束的哨兵值
    _SENTINEL = object()

    def producer():
        """在线程中运行生成器，将结果放入队列"""
        gen = None

        def enqueue(item) -> bool:
            if stop_event.is_set():
                return False
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while not stop_event.is_set():
                try:
                    future.result(timeout=0.1)
                    return True
                except FutureTimeoutError:
                    continue
                except RuntimeError:
                    return False
            future.cancel()
            return False

        try:
            gen = generator_func(*args, **kwargs)
            for item in gen:
                if stop_event.is_set() or not enqueue(item):
                    break
        except BaseException as e:
            # 发生异常时，记录日志并将异常放入队列
            logger.error(f"生成器执行异常: {type(e).__name__}: {e}")
            enqueue(e)
        finally:
            if stop_event.is_set() and gen is not None and hasattr(gen, "close"):
                try:
                    gen.close()
                except Exception as e:
                    logger.debug("关闭生成器失败: %s", e)
            # 发送结束标记
            enqueue(_SENTINEL)

    # 在线程池中启动生产者
    future = executor.submit(producer)

    try:
        while True:
            item = await queue.get()

            if item is _SENTINEL:
                break

            # 使用 BaseException 捕获所有异常类型（包括 KeyboardInterrupt 等）
            if isinstance(item, BaseException):
                raise item

            yield item
    finally:
        stop_event.set()
        # 确保检查线程是否有未捕获的异常
        if future.done():
            try:
                # 如果线程已完成，检查是否有异常
                future.result()
            except Exception as e:
                logger.error(f"生成器线程异常: {type(e).__name__}: {e}")
        elif not future.cancelled():
            future.cancel()


class AsyncInferenceWrapper:
    """
    异步推理包装器

    将同步的模型推理方法包装为异步方法，方便复用。

    Example:
        wrapper = AsyncInferenceWrapper(asr_engine.realtime_model)
        result = await wrapper.generate(input=audio_array, cache=cache)
    """

    def __init__(self, model):
        self._model = model

    async def generate(self, *args, **kwargs):
        """异步调用模型的 generate 方法"""
        return await run_sync(self._model.generate, *args, **kwargs)

    async def inference_sft(self, *args, **kwargs):
        """异步流式调用模型的 inference_sft 方法"""
        async for item in run_sync_generator(self._model.inference_sft, *args, **kwargs):
            yield item

    async def inference_zero_shot(self, *args, **kwargs):
        """异步流式调用模型的 inference_zero_shot 方法"""
        async for item in run_sync_generator(self._model.inference_zero_shot, *args, **kwargs):
            yield item
