"""온디맨드 생성·자연어 채점 엔드포인트 부하 측정.

실행:
    locust -f loadtest/locustfile.py ProblemUser --host http://localhost:8000
    locust -f loadtest/locustfile.py EvaluationUser --host http://localhost:8000
"""

import random

from locust import HttpUser, between, task

# 생성 요청에 쓸 값. RANDOM은 그래프가 직접 고르므로 제외한다.
CATEGORIES = [
    "ARRAY",
    "STRING",
    "DP",
    "GRAPH",
    "TREE",
    "STACK_QUEUE",
    "BINARY_SEARCH",
    "GREEDY",
    "BACKTRACKING",
    "TWO_POINTER",
    "HASH",
    "HEAP",
    "SORTING",
    "IMPLEMENTATION",
    "BRUTE_FORCE",
    "MATH",
]
DIFFICULTIES = ["LV1", "LV2", "LV3", "LV4", "LV5"]

# 응답을 기다리는 상한. 서버에 상한이 없어 클라이언트에서 끊어야 한다.
REQUEST_TIMEOUT_S = 180

# 채점 요청 본문. 필드명은 서버가 camelCase를 받는다.
EVALUATION_PAYLOAD = {
    "problemTitle": "두 수의 합",
    "problemContent": (
        "정수 배열과 목표값이 주어진다. 배열에서 두 수를 골라 "
        "합이 목표값이 되는 두 수의 인덱스를 출력하라."
    ),
    "category": "HASH",
    "categorySelectReason": "값을 키로 저장해 한 번의 순회로 짝을 찾는다.",
    "solutionKeywords": ["해시맵", "한 번 순회", "여집합"],
    "naturalSolution": (
        "배열을 앞에서부터 돌면서 목표값에서 현재 값을 뺀 수가 "
        "해시맵에 있는지 확인한다. 있으면 그 인덱스와 현재 인덱스를 "
        "답으로 내고, 없으면 현재 값과 인덱스를 해시맵에 넣는다."
    ),
}


class ProblemUser(HttpUser):
    """온디맨드 문제 생성을 호출한다."""

    wait_time = between(1, 3)

    @task
    def generate_problem(self):
        payload = {
            "difficulty": random.choice(DIFFICULTIES),
            "category": random.choice(CATEGORIES),
        }
        with self.client.post(
            "/api/llm/problem",
            json=payload,
            timeout=REQUEST_TIMEOUT_S,
            name="POST /api/llm/problem",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 502:
                # 검증에서 폐기된 경우. 서버 장애와 구분해서 집계한다.
                response.failure("폐기 (502)")
            else:
                response.failure(f"HTTP {response.status_code}")


class EvaluationUser(HttpUser):
    """자연어 풀이 채점을 호출한다."""

    wait_time = between(1, 3)

    @task
    def evaluate(self):
        with self.client.post(
            "/api/llm/evaluation",
            json=EVALUATION_PAYLOAD,
            timeout=REQUEST_TIMEOUT_S,
            name="POST /api/llm/evaluation",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
