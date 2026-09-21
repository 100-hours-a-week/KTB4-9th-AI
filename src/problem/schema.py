from enum import Enum

from pydantic import BaseModel


class Difficulty(str, Enum):
    """문제 난이도."""

    LV1 = "LV1"
    LV2 = "LV2"
    LV3 = "LV3"
    LV4 = "LV4"
    LV5 = "LV5"


class Category(str, Enum):
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


class ProblemRequest(BaseModel):
    """문제 생성 요청 본문."""

    difficulty: Difficulty
    category: Category


class ProblemResponse(BaseModel):
    """문제 생성 응답 본문."""

    success: bool
    message: str
