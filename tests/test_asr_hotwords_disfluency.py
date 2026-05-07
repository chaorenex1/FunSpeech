# -*- coding: utf-8 -*-

import json

import pytest

from app.core.config import settings
from app.core.exceptions import InvalidParameterException
from app.models.asr import ASRQueryParams
from app.services.asr.hotwords import resolve_hotwords
from app.utils.text_processing import filter_disfluencies


def test_asr_query_params_accept_inline_hotwords():
    params = ASRQueryParams(hotwords="FunSpeech, SenseVoice")

    assert params.hotwords == "FunSpeech, SenseVoice"


def test_resolve_hotwords_merges_inline_and_vocab_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ASR_HOTWORDS_DIR", str(tmp_path))
    (tmp_path / "product.txt").write_text("FunSpeech\n# comment\nSenseVoice CosyVoice\n", encoding="utf-8")

    hotwords = resolve_hotwords("product", "自定义词,FunSpeech")

    assert hotwords == "自定义词 FunSpeech SenseVoice CosyVoice"


def test_resolve_hotwords_reads_json_word_items(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ASR_HOTWORDS_DIR", str(tmp_path))
    (tmp_path / "names.json").write_text(
        json.dumps({"hotwords": [{"word": "张三"}, {"text": "李四"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    assert resolve_hotwords("names") == "张三 李四"


def test_resolve_hotwords_rejects_missing_vocab(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ASR_HOTWORDS_DIR", str(tmp_path))

    with pytest.raises(InvalidParameterException):
        resolve_hotwords("missing")


def test_filter_disfluencies_removes_common_fillers():
    text = "嗯，今天呃我们测试一下，那个，FunSpeech。"

    assert filter_disfluencies(text) == "今天我们测试一下，FunSpeech"


def test_filter_disfluencies_removes_expanded_particles_and_phrases():
    text = "额，怎么说呢，今天我们，哎呀，就是测试一下，em，对吧。"

    assert filter_disfluencies(text) == "今天我们，测试一下"


def test_filter_disfluencies_keeps_particle_inside_normal_words():
    text = "哈佛大学的项目很重要，先测试，然后发布。"

    assert filter_disfluencies(text) == text
