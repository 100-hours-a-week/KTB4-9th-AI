import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

from src.client import llm as module
from src.client.llm import LLMConfig, call_llm_structured, strip_code_fence
from src.core.exception import LLMOutputParseError


class Score(BaseModel):
    score: int


class FakeRunnable:
    """with_structured_output이 돌려주는 객체를 흉내 낸다."""

    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error

    async def ainvoke(self, prompt: str):
        if self.error:
            raise self.error
        return self.result


class FakeChatModel:
    def __init__(self, runnable: FakeRunnable):
        self.runnable = runnable

    def with_structured_output(self, schema):
        return self.runnable


def fix_chat_model(monkeypatch: pytest.MonkeyPatch, runnable: FakeRunnable) -> None:
    """Gemini 모델 대신 정해진 결과를 돌려주는 가짜 모델을 끼운다."""
    monkeypatch.setattr(module, "build_chat_model", lambda cfg: FakeChatModel(runnable))


CFG = LLMConfig(model_name="test-model")


def test_코드펜스를_벗긴다():
    text = "```python\nprint(1)\n```"
    assert strip_code_fence(text) == "print(1)"


def test_코드펜스가_없으면_그대로_둔다():
    assert strip_code_fence("  print(1)  ") == "print(1)"


def test_인라인_백틱으로_감싼_코드를_벗긴다():
    assert strip_code_fence("`print(1)`") == "print(1)"


@pytest.mark.asyncio
async def test_구조화_응답을_지정한_형식으로_돌려준다(monkeypatch):
    fix_chat_model(monkeypatch, FakeRunnable(result=Score(score=90)))
    result = await call_llm_structured("prompt", CFG, Score)

    assert result == Score(score=90)


@pytest.mark.asyncio
async def test_구조화_응답_해석에_실패하면_파싱_에러를_낸다(monkeypatch):
    error = OutputParserException("형식 불일치")
    fix_chat_model(monkeypatch, FakeRunnable(error=error))

    with pytest.raises(LLMOutputParseError):
        await call_llm_structured("prompt", CFG, Score)


@pytest.mark.asyncio
async def test_구조화_응답이_비어_있으면_파싱_에러를_낸다(monkeypatch):
    fix_chat_model(monkeypatch, FakeRunnable(result=None))

    with pytest.raises(LLMOutputParseError):
        await call_llm_structured("prompt", CFG, Score)
