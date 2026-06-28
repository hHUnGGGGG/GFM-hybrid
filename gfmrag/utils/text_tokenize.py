"""Tách từ (tokenize) đa ngôn ngữ cho khâu tiền xử lý.

Dùng chung cho splitter (stage0) và chunk_process (stage1):
- ``vi`` -> pyvi ``ViTokenizer`` (nối từ ghép bằng dấu '_', vd "điều_trị").
- ``en`` -> tách theo khoảng trắng/dấu câu (giữ hyphen như PEEK-SP và % như 61%).
"""

from __future__ import annotations

import re

__all__ = ["tokenize", "simple_tokenize", "SUPPORTED_LANGUAGES"]

SUPPORTED_LANGUAGES = ("vi", "en")

# Nạp pyvi lười (chỉ khi xử lý tiếng Việt) để tránh phụ thuộc cứng.
_vi_tokenizer = None


def _get_vi_tokenizer():
    global _vi_tokenizer
    if _vi_tokenizer is None:
        from pyvi import ViTokenizer  # import lười

        _vi_tokenizer = ViTokenizer
    return _vi_tokenizer


def simple_tokenize(text: str) -> str:
    """Tách dấu câu phổ biến ra khỏi từ (dùng cho tiếng Anh).

    Giữ nguyên ký tự nối (hyphen) như PEEK-SP và % như 61%.
    """
    text = re.sub(r'([.,!?()\[\]{}":;<>±/])', r" \1 ", text)
    return " ".join(text.split())


def tokenize(text: str, language: str = "vi") -> str:
    """Tách từ theo ngôn ngữ.

    Args:
        text: Văn bản đầu vào.
        language: ``vi`` hoặc ``en``.

    Returns:
        Chuỗi đã tách từ (tokenized_text).
    """
    if not text:
        return ""
    if language == "vi":
        return _get_vi_tokenizer().tokenize(text)
    if language == "en":
        return simple_tokenize(text)
    raise ValueError(
        f"Ngôn ngữ không hỗ trợ: {language!r}. Chọn một trong {SUPPORTED_LANGUAGES}."
    )
