# -*- coding: utf-8 -*-
"""
基于wetext的ITN（逆文本标准化）工具模块
使用wetext库提供高质量的中文ITN处理
"""

import logging
import re

logger = logging.getLogger(__name__)

# wetext导入 - 延迟导入以避免初始化问题
_wetext_normalizer = None


def _get_normalizer():
    """获取wetext标准化器实例（单例模式）"""
    global _wetext_normalizer
    if _wetext_normalizer is None:
        try:
            from wetext import Normalizer
            _wetext_normalizer = Normalizer(lang="zh", operator="itn")
            logger.info("WeText ITN模块初始化成功")
        except ImportError as e:
            logger.error(f"导入wetext失败: {e}")
            raise ImportError("请安装wetext库: pip install wetext")
        except Exception as e:
            logger.error(f"初始化wetext失败: {e}")
            raise
    return _wetext_normalizer


def apply_itn_to_text(text: str) -> str:
    """
    对文本应用逆文本标准化（ITN）
    使用wetext库进行高质量的中文ITN处理

    Args:
        text: 语音识别结果文本

    Returns:
        应用ITN后的文本
    """
    if not text or not text.strip():
        return text

    try:
        normalizer = _get_normalizer()
        result = normalizer.normalize(text)
        logger.debug(f"ITN处理: '{text}' -> '{result}'")
        return result
    except Exception as e:
        logger.warning(f"ITN处理失败: {text}, 错误: {str(e)}")
        return text


_DISFLUENCY_BOUNDARY = r"[\s,，。.!！?？;；:：、]"

# 可在词内直接移除的填充音，主要覆盖ASR常见的“嗯/呃/emm”等。
_DISFLUENCY_TOKEN_PATTERN = re.compile(
    r"(嗯+|呃+|额+|呣+|唔+|呒+|em+|er+|um+|uh+|ah+|eh+|hm+)",
    re.IGNORECASE,
)

# 只在独立成分时移除的语气词/口头填充短语，避免误删“哈佛”等正常词。
_DISFLUENCY_BOUNDARY_TERMS = [
    "怎么说呢",
    "怎么讲呢",
    "就是说",
    "你知道吗",
    "你知道吧",
    "我跟你说",
    "怎么说",
    "怎么讲",
    "对吧",
    "是吧",
    "其实吧",
    "然后呢",
    "那个",
    "这个",
    "就是",
    "哎呀",
    "诶呀",
    "啊",
    "呀",
    "哎",
    "唉",
    "诶",
    "欸",
    "哦",
    "喔",
    "噢",
    "噫",
    "哇",
    "啦",
    "嘛",
    "呢",
    "哈",
    "呵",
    "嘿",
]
_DISFLUENCY_BOUNDARY_PATTERN = re.compile(
    rf"(^|{_DISFLUENCY_BOUNDARY})"
    rf"({'|'.join(f'(?:{re.escape(term)})+' for term in _DISFLUENCY_BOUNDARY_TERMS)})"
    rf"(?=$|{_DISFLUENCY_BOUNDARY})",
    re.IGNORECASE,
)
_DISFLUENCY_PREFIX_PATTERN = re.compile(
    rf"(^|{_DISFLUENCY_BOUNDARY})(就是说|就是|那个|这个)(?=[\u4e00-\u9fffA-Za-z0-9])",
    re.IGNORECASE,
)


def filter_disfluencies(text: str) -> str:
    """过滤常见语气词/口吃填充词，保留正常句子结构。"""
    if not text or not text.strip():
        return text

    original_text = text
    cleaned = _DISFLUENCY_TOKEN_PATTERN.sub("", text)
    cleaned = _DISFLUENCY_BOUNDARY_PATTERN.sub(lambda m: m.group(1), cleaned)
    cleaned = _DISFLUENCY_PREFIX_PATTERN.sub(lambda m: m.group(1), cleaned)
    if cleaned == original_text:
        return original_text

    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*([,，。.!！?？;；:：、])\s*", r"\1", cleaned)
    cleaned = re.sub(r"([,，。.!！?？;；:：、]){2,}", r"\1", cleaned)
    cleaned = cleaned.strip(" ,，。、")
    logger.debug("语气词过滤: '%s' -> '%s'", text, cleaned)
    return cleaned
