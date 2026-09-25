from enum import StrEnum


class Difficulty(StrEnum):
    LV1 = "LV1"
    LV2 = "LV2"
    LV3 = "LV3"
    LV4 = "LV4"
    LV5 = "LV5"


class Trigger(StrEnum):
    BATCH = "BATCH"
    ON_DEMAND = "ON_DEMAND"


class Language(StrEnum):
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    CPP = "cpp"


class ConstraintScope(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class ConstraintDataType(StrEnum):
    INT = "int"
    LONG = "long"
    DOUBLE = "double"
    STRING = "str"
    CHAR = "char"
    BOOLEAN = "bool"


class Category(StrEnum):
    """문제 카테고리 (요청용). RANDOM은 카테고리를 지정하지 않음을 뜻한다."""

    RANDOM = "RANDOM"
    ARRAY = "ARRAY"
    STRING = "STRING"
    DP = "DP"
    GRAPH = "GRAPH"
    TREE = "TREE"
    STACK_QUEUE = "STACK_QUEUE"
    BINARY_SEARCH = "BINARY_SEARCH"
    GREEDY = "GREEDY"
    BACKTRACKING = "BACKTRACKING"
    TWO_POINTER = "TWO_POINTER"
    HASH_TABLE = "HASH_TABLE"
    HEAP = "HEAP"
    SORTING = "SORTING"
    IMPLEMENTATION = "IMPLEMENTATION"
    BRUTE_FORCE = "BRUTE_FORCE"
    MATH = "MATH"


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    DISCARDED = "DISCARDED"
    FAILED = "FAILED"  # 예외로 중단 (정상 폐기와 구분)


class ExecutionStatus(StrEnum):
    """코드 한 번 실행의 결과. judge0 상태와 짝이 맞게 둔다."""

    SUCCEEDED = "SUCCEEDED"  # 정상 종료 (exit code 0)
    TIMED_OUT = "TIMED_OUT"  # 실행 제한 시간 초과
    RUNTIME_ERROR = "RUNTIME_ERROR"  # 0이 아닌 코드로 종료, 시그널로 죽음
    COMPILE_ERROR = "COMPILE_ERROR"  # 컴파일 실패 (컴파일 언어용)
    INTERNAL_ERROR = "INTERNAL_ERROR"  # 채점 서버 장애 또는 알 수 없는 상태


class DiscardReason(StrEnum):
    """문제가 폐기된 사유. DiscardedProblem.discard_reason에 그대로 저장된다."""

    # 정적 검증 (static_validate)
    EMPTY_FIELD = "EMPTY_FIELD"  # 필수 필드 또는 언어별 실행 제한 누락
    MISMATCH_LABEL = "MISMATCH_LABEL"  # 요청 난이도 != 생성 난이도
    MISMATCH_TYPE = "MISMATCH_TYPE"  # 예시 출력이 선언한 자료형과 다름
    CONSTRAINT_CONFLICT = "CONSTRAINT_CONFLICT"  # 제약끼리 모순 (min > max 등)
    EXAMPLE_OUT_OF_RANGE = "EXAMPLE_OUT_OF_RANGE"  # 예시 값이 제약 범위를 벗어남

    # 중복 검사 (check_duplicate)
    DUPLICATE = "DUPLICATE"  # 기존 문제와 유사도가 기준 이상

    # 의미 검증 (semantic_validate)
    PROBLEM_CONTRADICTION = "PROBLEM_CONTRADICTION"  # 지문 자체가 모순
    CONSTRAINT_CONTRADICTION = "CONSTRAINT_CONTRADICTION"  # 지문과 제약이 모순
    EXAMPLE_MISMATCH = "EXAMPLE_MISMATCH"  # 레퍼런스 코드 실행 결과 != 예시 출력
    REFERENCE_CODE_FAILED = "REFERENCE_CODE_FAILED"  # 레퍼런스 코드 실행 자체가 실패
    MAX_ATTEMPT_EXCEEDED = "MAX_ATTEMPT_EXCEEDED"  # 재시도 한도 초과

    # 공통
    LLM_ERROR = "LLM_ERROR"  # LLM 호출 또는 구조화 응답 해석 실패


class SeedSource(StrEnum):
    """시드의 출처. 어느 쪽이든 사람이 직접 넣는다."""

    MANUAL = "MANUAL"  # 직접 작성했거나 외부에서 가져온 문제
    PROMOTED = "PROMOTED"  # 생성 결과 중 쓸 만한 것을 골라 넣음


class ErrorCode(StrEnum):
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_DIFFICULTY = "INVALID_DIFFICULTY"
    INVALID_CATEGORY = "INVALID_CATEGORY"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"

    PROBLEM_GENERATION_FAILED = "PROBLEM_GENERATION_FAILED"
    EVALUATION_FAILED = "EVALUATION_FAILED"

    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
