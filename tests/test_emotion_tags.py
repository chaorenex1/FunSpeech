# -*- coding: utf-8 -*-

from app.utils.emotion import (
    build_asr_result,
    compose_emotion_prompt,
    extract_sensevoice_emotion,
    format_cosyvoice_instruction_prompt,
)


def test_extract_sensevoice_emotion_from_rich_text():
    assert extract_sensevoice_emotion("<|zh|><|HAPPY|><|Speech|>你好") == "happy"


def test_compose_emotion_prompt_keeps_user_prompt():
    prompt = compose_emotion_prompt("像客服一样亲切", "happy", 0.9)

    assert "开心" in prompt
    assert "像客服一样亲切" in prompt


def test_compose_emotion_prompt_inserts_before_existing_cosyvoice_boundary():
    prompt = compose_emotion_prompt(
        "You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
        "happy",
        0.9,
    )

    assert prompt == (
        "You are a helpful assistant. 请用非常开心、愉悦的语气说这句话。 "
        "请用广东话表达。<|endofprompt|>"
    )


def test_format_cosyvoice3_instruction_prompt_keeps_existing_boundary():
    prompt = (
        "You are a helpful assistant.<|endofprompt|>"
        "希望你以后能够做的比我还好呦。"
    )

    assert format_cosyvoice_instruction_prompt(prompt, "cosyvoice3") == prompt


def test_format_cosyvoice3_instruction_prompt_wraps_plain_instruction():
    assert format_cosyvoice_instruction_prompt("请用广东话表达。", "cosyvoice3") == (
        "You are a helpful assistant. 请用广东话表达。<|endofprompt|>"
    )


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
