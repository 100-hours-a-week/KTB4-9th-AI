"""COSMOS 프롬프트, 커스텀 모델 단가, 채점 지표를 Langfuse에 동기화한다.

사용법:
    uv run python -m src.client.langfuse_setup
"""

import logging
from datetime import UTC, datetime

from src.client.langfuse import get_langfuse_client
from src.problem.battle.nodes import (
    GENERATE_PROMPT as BATTLE_PROMPT,
)
from src.problem.battle.nodes import (
    PROMPT_VERSION as V_BATTLE,
)
from src.problem.evaluation import (
    EVALUATE_PROMPT,
)
from src.problem.evaluation import (
    PROMPT_VERSION as V_EVAL,
)
from src.problem.nodes.check_contradiction import (
    LOGIC_PROMPT,
)
from src.problem.nodes.check_contradiction import (
    PROMPT_VERSION as V_CONTRA,
)
from src.problem.nodes.generate_nl_keyword import (
    GENERATE_NL_KEYWORD_PROMPT,
)
from src.problem.nodes.generate_nl_keyword import (
    PROMPT_VERSION as V_KW,
)
from src.problem.nodes.generate_problem import (
    GENERATE_PROBLEM_PROMPT,
)
from src.problem.nodes.generate_problem import (
    PROMPT_VERSION as V_GEN,
)
from src.problem.nodes.generate_ref_code import (
    GENERATE_REF_CODE_PROMPT,
)
from src.problem.nodes.generate_ref_code import (
    PROMPT_VERSION as V_REF,
)
from src.problem.nodes.generate_solution_code import (
    HINT_PROMPT,
    TRANSLATE_PROMPT,
)
from src.problem.nodes.generate_solution_code import (
    PROMPT_VERSION as V_SOL,
)
from src.problem.nodes.generate_testcase import (
    GENERATE_TESTCASE_PROMPT,
)
from src.problem.nodes.generate_testcase import (
    PROMPT_VERSION as V_TC,
)

logger = logging.getLogger(__name__)

COSMOS_PROMPTS: list[tuple[str, str, str, str]] = [
    (
        "generate_problem",
        V_GEN,
        GENERATE_PROBLEM_PROMPT,
        "코딩 테스트 문제 출제 프롬프트 (난이도·카테고리·Few-shot 기반)",
    ),
    (
        "check_contradiction",
        V_CONTRA,
        LOGIC_PROMPT,
        "생성된 문제 지문·제약·예시 간 논리적 모순 검수 프롬프트",
    ),
    (
        "generate_testcase",
        V_TC,
        GENERATE_TESTCASE_PROMPT,
        "히든 테스트 케이스 생성 프롬프트",
    ),
    (
        "generate_ref_code",
        V_REF,
        GENERATE_REF_CODE_PROMPT,
        "의미 검증용 레퍼런스 정답 코드 생성 프롬프트",
    ),
    (
        "generate_solution_code",
        V_SOL,
        TRANSLATE_PROMPT,
        "다국어(Java/C++/JS) 정답 코드 변환 프롬프트",
    ),
    (
        "generate_solution_code_translate",
        V_SOL,
        TRANSLATE_PROMPT,
        "다국어(Java/C++/JS) 정답 코드 변환 프롬프트",
    ),
    (
        "generate_solution_code_hint",
        V_SOL,
        HINT_PROMPT,
        "단계별 풀이 힌트 생성 프롬프트",
    ),
    (
        "generate_nl_keyword",
        V_KW,
        GENERATE_NL_KEYWORD_PROMPT,
        "자연어 풀이 채점용 핵심 키워드 추출 프롬프트",
    ),
    (
        "generate_battle_problem",
        V_BATTLE,
        BATTLE_PROMPT,
        "실시간 배틀 문제 출제 프롬프트",
    ),
    (
        "evaluate_nl_solution",
        V_EVAL,
        EVALUATE_PROMPT,
        "사용자 자연어 풀이 채점 프롬프트",
    ),
]

COSMOS_SCORE_CONFIGS: list[tuple[str, str, float, float, str]] = [
    (
        "passed",
        "NUMERIC",
        0.0,
        1.0,
        "문제 검증 최종 통과 여부 (1=통과, 0=폐기)",
    ),
    (
        "usable",
        "NUMERIC",
        0.0,
        1.0,
        "실질 통과 여부 (검증 통과 + 메모리/시간 제한 정상)",
    ),
    (
        "generation_ok",
        "NUMERIC",
        0.0,
        1.0,
        "1단계 문제 JSON 스키마 생성 성공 여부",
    ),
    (
        "gpu_cost_usd",
        "NUMERIC",
        0.0,
        50.0,
        "문제 1건 생성에 소요된 클라우드 GPU(Modal/RunPod) 과금액 (USD)",
    ),
    (
        "usable_pass_rate",
        "NUMERIC",
        0.0,
        1.0,
        "평가 세션 전체 실질 통과율",
    ),
    (
        "raw_pass_rate",
        "NUMERIC",
        0.0,
        1.0,
        "평가 세션 전체 검증 통과율",
    ),
    (
        "total_gpu_cost_usd",
        "NUMERIC",
        0.0,
        500.0,
        "평가 세션 전체 총 GPU 과금액 (USD)",
    ),
]


def sync_langfuse_resources() -> None:
    """프롬프트, 모델 단가, 채점 지표 설정을 Langfuse 서버에 동기화한다."""
    client = get_langfuse_client()
    if client is None:
        logger.info("Langfuse 키가 설정되지 않아 리소스 동기화를 건너뜁니다.")
        return

    # 1. 프롬프트 동기화 (변경된 경우에만 새 버전 생성)
    for name, code_ver, template, desc in COSMOS_PROMPTS:
        ver_tag = code_ver.split("/")[-1]
        try:
            existing = client.get_prompt(name, label="latest")
            if getattr(existing, "prompt", None) == template:
                continue
        except Exception:
            pass

        try:
            p = client.create_prompt(
                name=name,
                prompt=template,
                labels=["production", "latest", ver_tag],
                tags=["cosmos", name, code_ver],
                config={"code_version": code_ver, "description": desc},
                commit_message=f"Sync {code_ver}: {desc}",
            )
            logger.info("Langfuse 프롬프트 등록 완료: %s (v%s)", p.name, p.version)
        except Exception as error:
            logger.warning("Langfuse 프롬프트 등록 실패 (%s): %s", name, error)

    # 2. Qwen 오픈소스 모델 단가 정의 동기화
    try:
        models = client.api.models.list(limit=50).data
        custom_names = {m.model_name for m in models if not m.is_langfuse_managed}
        if "Qwen/Qwen3.5-122B-A10B-FP8" not in custom_names:
            client.api.models.create(
                model_name="Qwen/Qwen3.5-122B-A10B-FP8",
                match_pattern="(?i)^(Qwen/.*|.*qwen.*)$",
                start_date=datetime(2025, 1, 1, tzinfo=UTC),
                unit="TOKENS",
                input_price=0.0000008,
                output_price=0.0000024,
            )
            logger.info("Langfuse 커스텀 모델 등록 완료: Qwen/Qwen3.5-122B-A10B-FP8")
    except Exception as error:
        logger.warning("Langfuse 모델 등록 중 오류 (무시): %s", error)

    # 3. 채점 지표(Score Config) 동기화
    try:
        existing_sc = {sc.name for sc in client.api.score_configs.get().data}
        for name, dtype, min_v, max_v, desc in COSMOS_SCORE_CONFIGS:
            if name not in existing_sc:
                client.api.score_configs.create(
                    name=name,
                    data_type=dtype,
                    min_value=min_v,
                    max_value=max_v,
                    description=desc,
                )
                logger.info("Langfuse ScoreConfig 등록 완료: %s", name)
    except Exception as error:
        logger.warning("Langfuse ScoreConfig 등록 중 오류 (무시): %s", error)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    sync_langfuse_resources()
