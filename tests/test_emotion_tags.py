# -*- coding: utf-8 -*-

from app.utils.emotion import (
    build_asr_result,
    compose_emotion_prompt,
    extract_sensevoice_emotion,
)


def test_extract_sensevoice_emotion_from_rich_text():
    assert extract_sensevoice_emotion("<|zh|><|HAPPY|><|Speech|>你好") == "happy"


def test_compose_emotion_prompt_keeps_user_prompt():
    prompt = compose_emotion_prompt("像客服一样亲切", "happy", 0.9)

    assert "开心" in prompt
    assert "像客服一样亲切" in prompt


def test_build_asr_result_preserves_emotion_tag():
    result = build_asr_result(
        "你好",
        raw_rich_text="<|zh|><|HAPPY|><|Speech|>你好",
        enable_emotion=True,
        return_rich_text=True,
    )

    assert result.text == "你好"
    assert result.emotion == "happy"
    assert result.raw_rich_text == "<|zh|><|HAPPY|><|Speech|>你好"
