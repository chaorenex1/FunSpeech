# -*- coding: utf-8 -*-
"""ASR热词表解析与加载。"""

import json
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

from ...core.config import settings
from ...core.exceptions import InvalidParameterException

logger = logging.getLogger(__name__)

_VOCABULARY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _normalize_hotword_items(items: Iterable[object]) -> list[str]:
    hotwords: list[str] = []
    seen: set[str] = set()

    for item in items:
        if isinstance(item, dict):
            value = item.get("word") or item.get("text") or item.get("hotword")
        else:
            value = item

        if value is None:
            continue

        word = str(value).strip()
        if not word or word.startswith("#") or word in seen:
            continue

        seen.add(word)
        hotwords.append(word)

    return hotwords


def _split_inline_hotwords(hotwords: Optional[str]) -> list[str]:
    if not hotwords:
        return []

    return _normalize_hotword_items(re.split(r"[\n,，;；\s]+", hotwords))


def _load_hotword_file(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return _normalize_hotword_items(data)
        if isinstance(data, dict):
            for key in ("hotwords", "words", "items"):
                if key in data and isinstance(data[key], list):
                    return _normalize_hotword_items(data[key])
        raise InvalidParameterException(f"热词表格式错误: {path.name}")

    lines = path.read_text(encoding="utf-8").splitlines()
    items: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.extend(re.split(r"[,，;；\s]+", line))
    return _normalize_hotword_items(items)


def resolve_hotwords(
    vocabulary_id: Optional[str] = None,
    inline_hotwords: Optional[str] = None,
) -> str:
    """将vocabulary_id和inline hotwords解析成FunASR hotword字符串。"""
    hotwords = _split_inline_hotwords(inline_hotwords)

    if vocabulary_id:
        if not _VOCABULARY_ID_PATTERN.fullmatch(vocabulary_id):
            raise InvalidParameterException("vocabulary_id只能包含字母、数字、下划线、点号和短横线")

        hotwords_dir = Path(settings.ASR_HOTWORDS_DIR)
        candidates = [
            hotwords_dir / f"{vocabulary_id}.txt",
            hotwords_dir / f"{vocabulary_id}.json",
        ]
        vocab_path = next((path for path in candidates if path.exists()), None)
        if vocab_path is None:
            raise InvalidParameterException(f"热词表不存在: {vocabulary_id}")

        hotwords.extend(_load_hotword_file(vocab_path))

    normalized = _normalize_hotword_items(hotwords)
    resolved = " ".join(normalized)
    if resolved:
        logger.debug("ASR热词解析完成: vocabulary_id=%s, count=%s", vocabulary_id, len(normalized))
    return resolved
