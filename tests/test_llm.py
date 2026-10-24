import asyncio
import logging
from types import SimpleNamespace

import httpx
import openai
import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

from src.client import llm as module
from src.client.llm import (
    LLMConfig,
    LocalModel,
    batch_local_model,
    call_llm_structured,
    call_llm_with_fallback,
    called_config,
    current_local_model,
    strip_code_fence,
    use_local_model,
)
from src.core.config import get_settings
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


def test_gemini_모델이면_temperature를_버린다():
    cfg = LLMConfig(model_name="gemini-3.5-flash-lite", temperature=0.3)
    assert cfg.temperature is None


def test_gemini_모델이_아니면_temperature를_유지한다():
    cfg = LLMConfig(model_name="gpt-6.1-sol", temperature=0.3)
    assert cfg.temperature == 0.3


def test_모델명이_없으면_temperature를_유지한다():
    cfg = LLMConfig(model_name=None, temperature=0.3)
    assert cfg.temperature == 0.3


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


CALLED_CFG = LLMConfig(
    model_name="gemini-test", prompt_version="node/v1", temperature=0.9, top_k=2
)


def test_블록_밖이면_넘긴_설정을_그대로_기록한다():
    assert called_config(CALLED_CFG) is CALLED_CFG


def test_블록_안이면_로컬_모델의_설정으로_바꾼다():
    with use_local_model(LOCAL):
        recorded = called_config(CALLED_CFG)

    assert recorded == LLMConfig(
        model_name="qwen3-4b-ft",
        prompt_version="node/v1",
        temperature=0.7,
        top_k=2,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert CALLED_CFG.model_name == "gemini-test", "원본 설정이 바뀌었다"


def test_로컬_모델의_temperature가_비어_있으면_비워서_기록한다():
    with use_local_model(LocalModel(model_name="base")):
        recorded = called_config(CALLED_CFG)

    assert recorded.temperature is None, "Gemini 설정의 temperature가 새어 나왔다"


def test_로컬_모델의_chat_template_kwargs가_비어_있으면_비워서_기록한다():
    cfg = CALLED_CFG.model_copy(
        update={"chat_template_kwargs": {"enable_thinking": True}}
    )

    with use_local_model(LocalModel(model_name="base")):
        recorded = called_config(cfg)

    assert recorded.chat_template_kwargs is None, (
        "cfg의 chat_template_kwargs가 새어 나왔다"
    )


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


def set_batch(monkeypatch, *, model, thinking=None, timeout_s=180):
    """배치 로컬 모델 설정을 테스트 값으로 바꾼다."""
    settings = get_settings()
    monkeypatch.setattr(settings, "batch_local_model", model)
    monkeypatch.setattr(settings, "batch_local_thinking", thinking)
    monkeypatch.setattr(settings, "batch_local_timeout_s", timeout_s)


@pytest.mark.parametrize("model", [None, ""])
def test_배치_모델_이름이_없으면_로컬_모델을_쓰지_않는다(monkeypatch, model):
    set_batch(monkeypatch, model=model)

    assert batch_local_model() is None


@pytest.mark.parametrize(
    ("thinking", "expected"),
    [
        (None, None),
        (True, {"enable_thinking": True}),
        (False, {"enable_thinking": False}),
    ],
)
def test_thinking_설정을_chat_template_kwargs로_옮긴다(monkeypatch, thinking, expected):
    set_batch(monkeypatch, model="qwen", thinking=thinking)

    assert batch_local_model().chat_template_kwargs == expected


def test_배치_로컬_모델은_설정의_이름과_timeout을_쓰고_샘플링은_비운다(monkeypatch):
    set_batch(monkeypatch, model="qwen", timeout_s=42)

    model = batch_local_model()

    assert model.model_name == "qwen"
    assert model.timeout_s == 42
    assert model.temperature is None, "평가와 달리 temperature가 들어갔다"


# ── 재시도·폴백 (call_llm_with_fallback) ───────────────────────────────

BATCH_LOCAL = LocalModel(
    model_name="qwen-batch", chat_template_kwargs={"enable_thinking": False}
)
VLLM_REQUEST = httpx.Request("POST", "http://vllm.test/v1/chat/completions")


def status_error(cls: type[openai.APIStatusError], code: int) -> openai.APIStatusError:
    """vLLM 서버가 오류 코드로 응답한 경우의 예외."""
    response = httpx.Response(code, request=VLLM_REQUEST)
    return cls(f"{code}", response=response, body=None)


class FakeLocal:
    """call_local_structured 대신 정해 둔 결과를 차례로 낸다. 예외면 raise한다."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.models: list[LocalModel] = []

    async def __call__(self, prompt, model, schema):
        self.models.append(model)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def fix_local(monkeypatch, *outcomes, fallback=True) -> FakeLocal:
    """로컬 모델 호출을 가짜로 바꾸고, 재시도 대기를 없앤다."""
    fake = FakeLocal(*outcomes)
    monkeypatch.setattr(module, "call_local_structured", fake)
    monkeypatch.setattr(module, "LOCAL_RETRY_DELAYS_S", (0, 0))
    monkeypatch.setattr(get_settings(), "batch_local_fallback", fallback)
    return fake


def count_gemini(monkeypatch, result) -> list[LLMConfig]:
    """Gemini 호출을 가짜로 바꾼다. 부를 때 받은 설정을 기록한다."""
    calls: list[LLMConfig] = []

    def build(cfg):
        calls.append(cfg)
        return FakeChatModel(FakeRunnable(result=result))

    monkeypatch.setattr(module, "build_chat_model", build)
    return calls


@pytest.mark.asyncio
async def test_로컬_모델이_없으면_Gemini만_부른다(monkeypatch):
    local = fix_local(monkeypatch)
    gemini = count_gemini(monkeypatch, Score(score=1))

    result, called = await call_llm_with_fallback("p", CALLED_CFG, Score, None)

    assert result == Score(score=1)
    assert called is CALLED_CFG
    assert len(gemini) == 1
    assert local.models == [], "로컬 모델이 없는데 vLLM을 불렀다"


@pytest.mark.asyncio
async def test_로컬_모델이_성공하면_로컬_모델로_기록한다(monkeypatch):
    local = fix_local(monkeypatch, Score(score=90))
    forbid_gemini(monkeypatch)

    result, called = await call_llm_with_fallback("p", CALLED_CFG, Score, BATCH_LOCAL)

    assert result == Score(score=90)
    assert local.models == [BATCH_LOCAL]
    assert called.model_name == "qwen-batch"
    assert called.chat_template_kwargs == {"enable_thinking": False}
    assert current_local_model() is None, "끝난 뒤에도 로컬 모델 스위치가 켜져 있다"


@pytest.mark.asyncio
async def test_재시도_중에_성공하면_Gemini를_부르지_않는다(monkeypatch):
    local = fix_local(
        monkeypatch, openai.APIConnectionError(request=VLLM_REQUEST), Score(score=90)
    )
    forbid_gemini(monkeypatch)

    result, called = await call_llm_with_fallback("p", CALLED_CFG, Score, BATCH_LOCAL)

    assert result == Score(score=90)
    assert len(local.models) == 2
    assert called.model_name == "qwen-batch"


@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(request=VLLM_REQUEST),
        openai.APITimeoutError(request=VLLM_REQUEST),
        status_error(openai.InternalServerError, 503),
        status_error(openai.RateLimitError, 429),
        LLMOutputParseError("형식 불일치"),
    ],
    ids=["연결", "타임아웃", "503", "429", "형식"],
)
@pytest.mark.asyncio
async def test_일시_오류는_3번_시도한_뒤_Gemini로_폴백한다(monkeypatch, error):
    local = fix_local(monkeypatch, error, error, error)
    gemini = count_gemini(monkeypatch, Score(score=1))

    result, called = await call_llm_with_fallback("p", CALLED_CFG, Score, BATCH_LOCAL)

    assert len(local.models) == 3
    assert len(gemini) == 1
    assert result == Score(score=1)
    assert called is CALLED_CFG, "폴백했는데 로컬 모델로 기록했다"
    assert current_local_model() is None


@pytest.mark.parametrize(
    "error",
    [
        status_error(openai.NotFoundError, 404),
        status_error(openai.AuthenticationError, 401),
        RuntimeError("VLLM_URL, VLLM_API_KEY가 설정되지 않았습니다."),
    ],
    ids=["모델_이름", "인증", "설정_누락"],
)
@pytest.mark.asyncio
async def test_다시_불러도_같은_오류는_재시도_없이_폴백한다(monkeypatch, error):
    local = fix_local(monkeypatch, error)
    gemini = count_gemini(monkeypatch, Score(score=1))

    _, called = await call_llm_with_fallback("p", CALLED_CFG, Score, BATCH_LOCAL)

    assert len(local.models) == 1, "재시도해도 소용없는 오류를 다시 불렀다"
    assert len(gemini) == 1
    assert called is CALLED_CFG


@pytest.mark.asyncio
async def test_폴백을_끄면_마지막_오류를_올린다(monkeypatch):
    last = status_error(openai.InternalServerError, 503)
    local = fix_local(
        monkeypatch,
        openai.APIConnectionError(request=VLLM_REQUEST),
        openai.APIConnectionError(request=VLLM_REQUEST),
        last,
        fallback=False,
    )
    forbid_gemini(monkeypatch)

    with pytest.raises(openai.InternalServerError) as raised:
        await call_llm_with_fallback("p", CALLED_CFG, Score, BATCH_LOCAL)

    assert raised.value is last
    assert len(local.models) == 3


@pytest.mark.asyncio
async def test_폴백하면_warning을_남긴다(monkeypatch, caplog):
    fix_local(monkeypatch, status_error(openai.NotFoundError, 404))
    count_gemini(monkeypatch, Score(score=1))

    with caplog.at_level(logging.WARNING, logger=module.__name__):
        await call_llm_with_fallback("p", CALLED_CFG, Score, BATCH_LOCAL)

    assert "Gemini로 폴백" in caplog.text, "폴백 횟수를 셀 로그가 없다"
