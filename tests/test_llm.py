import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

from src.client import llm as module
from src.client.llm import (
    LLMConfig,
    LocalModel,
    call_llm_structured,
    current_local_model,
    strip_code_fence,
    use_local_model,
)
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


# ── 로컬 모델(vLLM) 경로 ───────────────────────────────────────────────


class FakeCompletions:
    """AsyncOpenAI.chat.completions를 흉내 낸다. 받은 요청을 기록한다."""

    def __init__(self, content: str, finish_reason: str = "stop"):
        self.content = content
        self.finish_reason = finish_reason
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message, finish_reason=self.finish_reason)
        return SimpleNamespace(choices=[choice])


def fix_local_client(
    monkeypatch: pytest.MonkeyPatch, content: str, finish_reason: str = "stop"
) -> FakeCompletions:
    completions = FakeCompletions(content, finish_reason)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(module, "get_local_client", lambda: client)
    return completions


def forbid_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(cfg):
        raise AssertionError("로컬 모델 블록 안에서 Gemini를 불렀다")

    monkeypatch.setattr(module, "build_chat_model", fail)


LOCAL = LocalModel(
    model_name="qwen3-4b-ft",
    chat_template_kwargs={"enable_thinking": False},
    temperature=0.7,
)


@pytest.mark.asyncio
async def test_로컬_모델_블록_안에서는_vLLM으로_보낸다(monkeypatch):
    forbid_gemini(monkeypatch)
    completions = fix_local_client(monkeypatch, '{"score": 90}')

    with use_local_model(LOCAL):
        result = await call_llm_structured("prompt", CFG, Score)

    assert result == Score(score=90)
    request = completions.requests[0]
    assert request["model"] == "qwen3-4b-ft"
    assert request["messages"] == [{"role": "user", "content": "prompt"}]
    assert request["response_format"]["json_schema"] == {
        "name": "Score",
        "schema": Score.model_json_schema(),
    }
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert request["temperature"] == 0.7


@pytest.mark.asyncio
async def test_샘플링을_비워_두면_서버_기본값을_쓴다(monkeypatch):
    completions = fix_local_client(monkeypatch, '{"score": 90}')

    with use_local_model(LocalModel(model_name="base")):
        await call_llm_structured("prompt", CFG, Score)

    request = completions.requests[0]
    assert "temperature" not in request
    assert "extra_body" not in request


@pytest.mark.asyncio
async def test_블록을_벗어나면_다시_Gemini로_보낸다(monkeypatch):
    fix_local_client(monkeypatch, '{"score": 1}')
    fix_chat_model(monkeypatch, FakeRunnable(result=Score(score=90)))

    with use_local_model(LOCAL):
        assert current_local_model() == LOCAL
    result = await call_llm_structured("prompt", CFG, Score)

    assert current_local_model() is None
    assert result == Score(score=90), "블록 밖인데 로컬 모델로 갔다"


@pytest.mark.asyncio
async def test_동시에_도는_다른_요청은_Gemini로_간다(monkeypatch):
    """평가에서 노드 하나만 바꿔도 다른 노드·요청은 Gemini를 써야 한다."""
    fix_local_client(monkeypatch, '{"score": 1}')
    fix_chat_model(monkeypatch, FakeRunnable(result=Score(score=90)))

    async def local_call() -> Score:
        with use_local_model(LOCAL):
            await asyncio.sleep(0)  # 다른 task가 끼어들 틈을 준다
            return await call_llm_structured("prompt", CFG, Score)

    async def gemini_call() -> Score:
        await asyncio.sleep(0)
        return await call_llm_structured("prompt", CFG, Score)

    local, gemini = await asyncio.gather(local_call(), gemini_call())

    assert local == Score(score=1)
    assert gemini == Score(score=90)


@pytest.mark.asyncio
async def test_로컬_응답이_잘리면_파싱_에러에_finish_reason을_남긴다(monkeypatch):
    fix_local_client(monkeypatch, '{"score": ', finish_reason="length")

    with use_local_model(LOCAL), pytest.raises(LLMOutputParseError, match="length"):
        await call_llm_structured("prompt", CFG, Score)


@pytest.mark.asyncio
async def test_로컬_응답이_비어_있으면_파싱_에러를_낸다(monkeypatch):
    fix_local_client(monkeypatch, "")

    with use_local_model(LOCAL), pytest.raises(LLMOutputParseError):
        await call_llm_structured("prompt", CFG, Score)


def test_vLLM_주소가_없으면_클라이언트를_만들지_않는다(monkeypatch):
    settings = SimpleNamespace(vllm_url=None, vllm_api_key="key")
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    module.get_local_client.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="VLLM_URL"):
            module.get_local_client()
    finally:
        module.get_local_client.cache_clear()
