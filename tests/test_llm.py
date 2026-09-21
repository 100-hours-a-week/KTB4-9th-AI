import pytest

from src.shared.llm import LLMOutputParseError, parse_json, strip_code_fence


def test_코드펜스를_벗긴다():
    text = "```python\nprint(1)\n```"
    assert strip_code_fence(text) == "print(1)"


def test_코드펜스가_없으면_그대로_둔다():
    assert strip_code_fence("  print(1)  ") == "print(1)"


def test_순수_JSON을_해석한다():
    assert parse_json('{"score": 90}') == {"score": 90}


def test_코드펜스로_감싼_JSON을_해석한다():
    text = '```json\n{"score": 90}\n```'
    assert parse_json(text) == {"score": 90}


def test_앞뒤에_설명이_붙은_JSON을_해석한다():
    text = '결과는 다음과 같습니다.\n{"score": 90}\n이상입니다.'
    assert parse_json(text) == {"score": 90}


def test_JSON이_없으면_파싱_에러를_낸다():
    with pytest.raises(LLMOutputParseError):
        parse_json("죄송합니다. 답변할 수 없습니다.")


def test_JSON_배열이면_파싱_에러를_낸다():
    with pytest.raises(LLMOutputParseError):
        parse_json("[1, 2, 3]")
