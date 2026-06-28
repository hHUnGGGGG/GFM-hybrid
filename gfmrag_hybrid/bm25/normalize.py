"""
Chuẩn hóa tên thực thể trước khi đưa vào câu truy vấn BM25.
"""

import re


def normalize_entity(entity: str) -> str:
    e = entity.strip()
    e = re.sub(r'^\[(.+)\]$', r'\1', e).strip()
    return e if e else entity.strip()


def normalize_entities(entities: list) -> list:
    seen = set()
    result = []
    for raw in entities:
        cleaned = normalize_entity(str(raw))
        key = cleaned.lower()
        if key not in seen and cleaned:
            seen.add(key)
            result.append(cleaned)
    return result
