# -*- coding: utf-8 -*-
"""Voice Cloner音色设计执行器。

真实的VoxCPM适配器通过环境变量 VOICE_DESIGN_PROVIDER 注入，格式为
``module.submodule:function``。函数需接收本模块同名关键字参数并返回音频文件路径。
"""

import importlib
import os
from typing import Optional

from ...core.exceptions import DefaultServerErrorException


def _load_provider():
    provider = os.getenv("VOICE_DESIGN_PROVIDER", "").strip()
    if not provider:
        raise DefaultServerErrorException(
            "音色设计后端未配置，请设置 VOICE_DESIGN_PROVIDER 指向 VoxCPM 参考音频生成函数"
        )

    if ":" not in provider:
        raise DefaultServerErrorException(
            "VOICE_DESIGN_PROVIDER 格式错误，应为 module.submodule:function"
        )

    module_name, function_name = provider.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except Exception as exc:
        raise DefaultServerErrorException(f"加载音色设计后端失败: {exc}")


def generate_reference_audio(
    *,
    voice_name: str,
    voice_instruction: str,
    reference_text: str,
    format: str = "wav",
    sample_rate: int = 24000,
    task_id: Optional[str] = None,
) -> str:
    """根据音色设计指令生成参考音频。

    默认不静默降级：仓库当前没有内置VoxCPM实现，必须显式配置提供方。
    """
    provider = _load_provider()
    try:
        return provider(
            voice_name=voice_name,
            voice_instruction=voice_instruction,
            reference_text=reference_text,
            format=format,
            sample_rate=sample_rate,
            task_id=task_id,
        )
    except Exception as exc:
        raise DefaultServerErrorException(f"音色设计生成失败: {exc}", task_id or "")
