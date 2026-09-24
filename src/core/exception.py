from src.core.enums import ErrorCode


class CosmosError(Exception):
    """이 서비스가 의도적으로 발생시키는 예외의 공통 조상.

    핸들러가 code와 status를 그대로 응답에 옮기므로,
    새 실패 상황이 생기면 이 클래스를 상속해 값만 지정한다.
    """

    code: ErrorCode = ErrorCode.INTERNAL_SERVER_ERROR
    status: int = 500

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.__doc__ or self.code.value)
        self.message = message or self.code.value


class ProblemGenerationError(CosmosError):
    """문제 생성 파이프라인이 결과를 내지 못했다."""

    code = ErrorCode.PROBLEM_GENERATION_FAILED
    status = 502


class EvaluationError(CosmosError):
    """자연어 풀이 평가에 실패했다."""

    code = ErrorCode.EVALUATION_FAILED
    status = 502


class LLMOutputParseError(CosmosError):
    """모델 응답을 기대한 형식으로 해석할 수 없다.

    그래프 안에서는 노드가 잡아 DiscardReason.LLM_ERROR로 폐기시킨다.
    여기까지 올라왔다면 그래프 밖에서 난 호출이라는 뜻이다.
    """

    code = ErrorCode.PROBLEM_GENERATION_FAILED
    status = 502
