import json
import os
import re
import unicodedata

KG_DELIMITER = ","


def processing_phrases(phrase: str) -> str:
    if isinstance(phrase, int):
        return str(phrase)

    # 1. Chuẩn hóa unicode NFC (quan trọng với tiếng Việt)
    phrase = unicodedata.normalize("NFC", phrase)

    # 2. Lowercase
    phrase = phrase.lower()

    # 3. Chỉ giữ: chữ cái (bao gồm có dấu), số, khoảng trắng
    #    Xóa: dấu câu, ký tự đặc biệt, nhưng GIỮ dấu tiếng Việt
    phrase = re.sub(r"[^\w\s]", " ", phrase)
    #               ↑ \w match cả ký tự Unicode có dấu

    # 4. Xóa khoảng trắng thừa
    phrase = re.sub(r"\s+", " ", phrase).strip()

    return phrase

def directory_exists(path: str) -> None:
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def extract_json_dict(text: str) -> str | dict:
    pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}"
    match = re.search(pattern, text)

    if match:
        json_string = match.group()
        try:
            json_dict = json.loads(json_string)
            return json_dict
        except json.JSONDecodeError:
            return ""
    else:
        return ""
