import pytest

from src.core.config import Settings


def test_설정하지_않으면_기본_모델을_쓴다(monkeypatch: pytest.MonkeyPatch):
    # 준비
    monkeypatch.delenv("LLM_MODELS__GENERATE_PROBLEM", raising=False)

    # 실행
    settings = Settings(_env_file=None)

    # 확인
    assert settings.llm_models.generate_problem == "gemini-3.8-flash"


def test_환경변수로_노드_하나의_모델만_바꾼다(monkeypatch: pytest.MonkeyPatch):
    # 준비
    monkeypatch.delenv("LLM_MODELS__GENERATE_PROBLEM", raising=False)
    monkeypatch.setenv("LLM_MODELS__GENERATE_TESTCASE", "test-model")

    # 실행
    settings = Settings(_env_file=None)

    # 확인: 바꾼 노드는 바뀌고
    assert settings.llm_models.generate_testcase == "test-model"
    # 다른 노드는 기본값 그대로
    assert settings.llm_models.generate_problem == "gemini-3.8-flash"
