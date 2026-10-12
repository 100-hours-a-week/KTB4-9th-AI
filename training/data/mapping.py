"""외부 데이터셋의 난이도·카테고리를 우리 enum으로 옮기는 규칙.

code_contests는 Codeforces 레이팅과 태그를 쓴다.
데이터셋 자체 difficulty 필드는 0이 1,333건이고 그 행들은
레이팅도 태그도 비어 있어 정보 누락으로 보고 쓰지 않는다.
"""

from src.core.enums import Category, Difficulty

# cf_rating 상한(미만)과 우리 난이도
RATING_BANDS: list[tuple[int, Difficulty]] = [
    (1200, Difficulty.LV1),
    (1600, Difficulty.LV2),
    (2000, Difficulty.LV3),
    (2400, Difficulty.LV4),
]
HIGHEST_DIFFICULTY = Difficulty.LV5

# 태그가 여러 개 붙은 문제가 80%라 대표 하나를 골라야 한다.
# 구체적인 알고리즘을 위에, 포괄적인 태그를 아래에 둔다.
TAG_PRIORITY: list[tuple[Category, set[str]]] = [
    (Category.TREE, {"trees"}),
    (Category.GRAPH, {"graphs", "shortest paths", "flows", "dsu", "graph matchings"}),
    (Category.DP, {"dp"}),
    (Category.BINARY_SEARCH, {"binary search", "ternary search"}),
    (Category.TWO_POINTER, {"two pointers"}),
    (Category.BACKTRACKING, {"dfs and similar"}),
    (Category.HASH, {"hashing"}),
    (Category.STRING, {"strings", "string suffix structures"}),
    (Category.SORTING, {"sortings"}),
    (Category.GREEDY, {"greedy"}),
    (Category.BRUTE_FORCE, {"brute force"}),
    (
        Category.MATH,
        {"math", "number theory", "combinatorics", "probabilities", "geometry"},
    ),
    (Category.IMPLEMENTATION, {"implementation", "constructive algorithms"}),
]


def to_difficulty(cf_rating: int) -> Difficulty:
    """Codeforces 레이팅을 우리 난이도로 옮긴다."""
    for upper, difficulty in RATING_BANDS:
        if cf_rating < upper:
            return difficulty
    return HIGHEST_DIFFICULTY


def to_category(cf_tags: list[str]) -> Category | None:
    """Codeforces 태그를 우리 카테고리로 옮긴다. 맞는 것이 없으면 None."""
    tags = set(cf_tags)
    for category, keys in TAG_PRIORITY:
        if tags & keys:
            return category
    return None
