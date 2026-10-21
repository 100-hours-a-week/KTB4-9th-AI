import pytest

from src.core.exception import LLMOutputParseError
from src.problem import evaluation as module
from src.problem.evaluation import (
    KEYWORD_MAX,
    Evaluation,
    KeywordVerdict,
    align_keywords,
    build_prompt,
    evaluate,
    keyword_score,
    score_evaluation,
)
from src.schema.evaluation import EvaluationRequest, Keyword

KEYWORDS = ["해시맵", "O(N)", "누적 카운트"]


def make_request(**overrides) -> EvaluationRequest:
    base = {
        "problem_title": "합이 M인 쌍의 개수",
        "problem_content": "두 원소의 합이 M이 되는 쌍의 개수를 구하시오.",
        "category": "HASH",
        "category_select_reason": "M - x 존재 여부를 O(1)에 조회하는 것이 핵심",
        "solution_keywords": KEYWORDS,
        "natural_solution": (
            "해시맵에 지금까지 본 수의 개수를 누적하면서 M - x를 찾는다."
        ),
    }
    return EvaluationRequest(**{**base, **overrides})


def make_evaluation(**overrides) -> Evaluation:
    base = {
        "approach_score": 35,
        "category_score": 20,
        "order_score": 10,
        "feedback": "접근이 정확합니다. 시간복잡도 언급이 있으면 더 좋겠습니다.",
        "keywords": [KeywordVerdict(keyword=k, is_included=True) for k in KEYWORDS],
    }
    return Evaluation(**{**base, **overrides})


def make_keywords(*included: bool) -> list[Keyword]:
    return [
        Keyword(keyword=keyword, is_included=flag)
        for keyword, flag in zip(KEYWORDS, included, strict=True)
    ]


def fix_structured_response(
    monkeypatch: pytest.MonkeyPatch,
    result: Evaluation | None = None,
    error: Exception | None = None,
) -> list[str]:
    """구조화 호출 대신 정해진 결과를 돌려주도록 바꿔 끼운다."""
    prompts: list[str] = []

    async def fake(prompt: str, cfg, schema):
        prompts.append(prompt)
        if error:
            raise error
        return result

    monkeypatch.setattr(module, "call_llm_structured", fake)
    return prompts


@pytest.mark.asyncio
async def test_항목별_점수를_더해_총점을_만든다(monkeypatch):
    fix_structured_response(monkeypatch, result=make_evaluation())
    result = await evaluate(make_request())

    assert result.keyword_score == KEYWORD_MAX
    assert result.score == 35 + 20 + KEYWORD_MAX + 10
    assert "접근이 정확" in result.feedback
    assert len(result.keywords) == 3


@pytest.mark.asyncio
async def test_내용_없는_풀이는_총점이_0이_된다(monkeypatch):
    empty = make_evaluation(
        approach_score=0,
        category_score=0,
        order_score=0,
        feedback="구체적인 방법이 없어 판단할 근거가 없습니다.",
        keywords=[KeywordVerdict(keyword=k, is_included=False) for k in KEYWORDS],
    )
    fix_structured_response(monkeypatch, result=empty)
    result = await evaluate(make_request(natural_solution="잘 풉니다"))

    assert result.score == 0


@pytest.mark.asyncio
async def test_구조화_호출이_실패하면_파싱_에러가_전달된다(monkeypatch):
    fix_structured_response(monkeypatch, error=LLMOutputParseError("형식 불일치"))

    with pytest.raises(LLMOutputParseError):
        await evaluate(make_request())


def test_항목별_점수가_배점을_넘으면_거부한다():
    with pytest.raises(ValueError):
        make_evaluation(approach_score=41)

    with pytest.raises(ValueError):
        make_evaluation(category_score=21)

    with pytest.raises(ValueError):
        make_evaluation(order_score=11)

    with pytest.raises(ValueError):
        make_evaluation(approach_score=-1)


def test_키워드_점수는_포함_비율로_계산한다():
    assert keyword_score(make_keywords(True, True, True)) == KEYWORD_MAX
    assert keyword_score(make_keywords(False, False, False)) == 0
    assert keyword_score(make_keywords(True, False, False)) == 10
    assert keyword_score([]) == 0


def test_모델이_만든_없는_키워드는_점수에_들어가지_않는다():
    evaluation = make_evaluation(
        keywords=[KeywordVerdict(keyword=k, is_included=False) for k in KEYWORDS]
        + [KeywordVerdict(keyword="투 포인터", is_included=True)],
    )
    result = score_evaluation(evaluation, KEYWORDS)

    assert result.keyword_score == 0
    assert [k.keyword for k in result.keywords] == KEYWORDS


def test_프롬프트에_문제와_키워드와_풀이가_들어간다():
    request = make_request()
    prompt = build_prompt(request)

    assert request.problem_title in prompt
    assert request.natural_solution in prompt
    for keyword in KEYWORDS:
        assert keyword in prompt


def test_프롬프트에_0점_규칙이_들어간다():
    prompt = build_prompt(make_request())

    assert "잘 풉니다" in prompt
    assert "0점" in prompt


def test_프롬프트는_풀이가_없어도_만들어진다():
    prompt = build_prompt(make_request(natural_solution=None))

    assert "(없음)" in prompt


def test_판정_결과를_요청한_키워드_순서로_맞춘다():
    verdicts = [
        KeywordVerdict(keyword="O(N)", is_included=False),
        KeywordVerdict(keyword="해시맵", is_included=True),
        KeywordVerdict(keyword="누적 카운트", is_included=True),
    ]
    result = align_keywords(KEYWORDS, verdicts)

    assert [k.keyword for k in result] == KEYWORDS
    assert [k.is_included for k in result] == [True, False, True]


def test_모델이_빠뜨린_키워드는_미포함으로_채운다():
    verdicts = [KeywordVerdict(keyword="해시맵", is_included=True)]
    result = align_keywords(KEYWORDS, verdicts)

    assert len(result) == 3
    assert result[0].is_included is True
    assert result[1].is_included is False
    assert result[2].is_included is False


def test_요청에_없는_키워드는_버린다():
    verdicts = [
        KeywordVerdict(keyword="해시맵", is_included=True),
        KeywordVerdict(keyword="투 포인터", is_included=True),
    ]
    result = align_keywords(KEYWORDS, verdicts)

    assert [k.keyword for k in result] == KEYWORDS
