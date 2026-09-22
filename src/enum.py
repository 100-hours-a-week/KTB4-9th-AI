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


class DiscardReason(StrEnum):
    STATIC_VALIDATION = "STATIC_VALIDATION"  # 규칙 기반 정적 검증 실패
    DIFFICULTY_MISMATCH = "DIFFICULTY_MISMATCH"  # 요청 난이도 != 반환 난이도
    DUPLICATE = "DUPLICATE"  # 중복 검사 탈락
    EXAMPLE_MISMATCH = "EXAMPLE_MISMATCH"  # 정답 코드 실행 결과 != 예시 출력
    REFERENCE_CODE_FAILED = "REFERENCE_CODE_FAILED"
    LLM_ERROR = "LLM_ERROR"


class SeedSource(StrEnum):
    MANUAL = "MANUAL"  # 직접 작성
    PROMOTED = "PROMOTED"  # 생성 결과에서 승격


class ErrorCode(StrEnum):
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_DIFFICULTY = "INVALID_DIFFICULTY"
    INVALID_CATEGORY = "INVALID_CATEGORY"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"

    PROBLEM_GENERATION_FAILED = "PROBLEM_GENERATION_FAILED"
    EVALUATION_FAILED = "EVALUATION_FAILED"

    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
