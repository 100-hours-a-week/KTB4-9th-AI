"""평가 결과를 MLflow, Langfuse, S3, GitHub Step Summary에 기록하는 스크립트.

AI 팀의 evals.summarize 실행 후 생성된 결과 파일들을 읽어,
1. 평가 메트릭(통과율, 실질 통과율, 폐기 사유, GPU 비용)을 summary.json으로 저장한다.
2. 중앙 MLflow Tracking 서버에 평가 Run을 생성하고 메트릭과 아티팩트를 기록한다.
3. Langfuse 세션에 전체 요약 점수(실질 통과율, 총 GPU 비용 등)를 기록한다.
4. S3 버킷에 결과 원본을 불변 업로드한다.
5. GitHub Actions 요약 화면(GITHUB_STEP_SUMMARY)에 모델·GPU 비용·점수표를 출력한다.
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evals.summarize import collect, stage_of  # noqa: E402
from src.client.langfuse import (  # noqa: E402
    flush_langfuse,
    get_gpu_hourly_rate,
    get_langfuse_client,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Log evaluation results to MLflow, Langfuse, S3, and GitHub Summary"
    )
    parser.add_argument(
        "--label", required=True, help="evals/results/<label> 디렉터리 이름"
    )
    parser.add_argument(
        "--track-type", default="base_model", choices=["base_model", "trained_model"]
    )
    parser.add_argument("--model-id", required=True, help="평가 대상 모델 식별자")
    parser.add_argument(
        "--prompt-version", default="generate_problem/v11", help="프롬프트 버전"
    )
    parser.add_argument(
        "--gpu-type",
        default=os.environ.get("MODAL_GPU_TYPE", "H200"),
        help="추론에 사용한 GPU 기종",
    )
    parser.add_argument(
        "--s3-bucket",
        default="cosmos-v2-stage-mlflow-artifacts-352635536602",
        help="S3 버킷 이름",
    )
    return parser.parse_args()


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def compute_summary_metrics(label: str, gpu_type: str) -> dict:
    """생성·채점 결과에서 통과율, 단계별 폐기율, GPU 가동 시간 및 비용을 계산한다."""
    gpu_name, hourly_rate = get_gpu_hourly_rate(gpu_type)
    label_result = collect(label)
    determined = label_result.determined
    total = len(determined)
    passed = sum(o.passed for o in determined)
    usable = sum(o.usable for o in determined)
    by_stage = Counter(stage_of(o.result) for o in determined)

    gen_rows = [o.generation for o in label_result.outcomes if o.generation is not None]
    if gen_rows:
        min_start = min(g.started_at for g in gen_rows)
        max_end = max(g.started_at + timedelta(seconds=g.seconds) for g in gen_rows)
        wall_seconds = max(round((max_end - min_start).total_seconds(), 2), 0.0)
        sum_gen_seconds = round(sum(g.seconds for g in gen_rows), 2)
    else:
        wall_seconds = 0.0
        sum_gen_seconds = 0.0

    # 콜드 스타트 및 유휴 대기(60초 scaledown)를 포함한 예상 GPU 과금액
    billable_seconds = wall_seconds + 60.0 if wall_seconds > 0 else 0.0
    estimated_gpu_cost_usd = round((billable_seconds / 3600.0) * hourly_rate, 4)

    return {
        "label": label,
        "model": label_result.model,
        "gpu_type": gpu_name,
        "gpu_hourly_rate_usd": hourly_rate,
        "total_requests": len(label_result.outcomes),
        "determined_requests": total,
        "passed_count": passed,
        "usable_count": usable,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "usable_rate": round(usable / total, 4) if total else 0.0,
        "format_fail_rate": round(by_stage["generate_problem"] / total, 4)
        if total
        else 0.0,
        "static_discard_rate": round(by_stage["static_validate"] / total, 4)
        if total
        else 0.0,
        "contradiction_discard_rate": round(
            by_stage["check_contradiction"] / total, 4
        )
        if total
        else 0.0,
        "semantic_discard_rate": round(by_stage["semantic_validate"] / total, 4)
        if total
        else 0.0,
        "generation_wall_seconds": wall_seconds,
        "generation_sum_seconds": sum_gen_seconds,
        "billable_gpu_seconds": round(billable_seconds, 2),
        "estimated_gpu_cost_usd": estimated_gpu_cost_usd,
    }


def format_duration(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    if mins > 0:
        return f"{mins}분 {secs}초 ({seconds:.1f}s)"
    return f"{seconds:.1f}초"


def main():
    args = parse_args()
    results_dir = Path("evals/results") / args.label

    if not results_dir.exists():
        logger.error("평가 결과 디렉터리가 존재하지 않습니다: %s", results_dir)
        sys.exit(1)

    gen_count = count_lines(results_dir / "generations.jsonl")
    score_count = count_lines(results_dir / "scores.jsonl")

    summary_data = compute_summary_metrics(args.label, args.gpu_type)
    summary_file = results_dir / "summary.json"
    summary_file.write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        "집계 완료: 생성 %d건, 채점 %d건, 실질 통과율 %.1f%%, 예상 비용 $%.4f",
        gen_count,
        score_count,
        summary_data["usable_rate"] * 100,
        summary_data["estimated_gpu_cost_usd"],
    )

    # 1. Langfuse 세션 레벨 요약 점수 기록
    lf_client = get_langfuse_client()
    if lf_client is not None:
        try:
            usable_cnt = summary_data["usable_count"]
            passed_cnt = summary_data["passed_count"]
            det_cnt = summary_data["determined_requests"]
            gpu_name = summary_data["gpu_type"]
            gpu_rate = summary_data["gpu_hourly_rate_usd"]
            bill_sec = summary_data["billable_gpu_seconds"]

            lf_client.create_score(
                name="usable_pass_rate",
                value=summary_data["usable_rate"],
                session_id=args.label,
                comment=f"실질 통과 {usable_cnt}/{det_cnt} ({args.model_id})",
            )
            lf_client.create_score(
                name="raw_pass_rate",
                value=summary_data["pass_rate"],
                session_id=args.label,
                comment=f"검증 통과 {passed_cnt}/{det_cnt}",
            )
            lf_client.create_score(
                name="total_gpu_cost_usd",
                value=summary_data["estimated_gpu_cost_usd"],
                session_id=args.label,
                comment=f"{gpu_name} (${gpu_rate}/h, 가동 {bill_sec}s)",
            )
            flush_langfuse()
            logger.info("Langfuse 세션(%s) 요약 점수 기록 완료", args.label)
        except Exception as error:  # noqa: BLE001
            logger.warning("Langfuse 세션 점수 기록 중 오류 발생: %s", error)

    # 2. MLflow 기록 (MLFLOW_TRACKING_URI가 있을 때만 실행)
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("cosmos-model-eval")

            run_name = f"{args.track_type}-{args.label}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(
                    {
                        "track_type": args.track_type,
                        "model_id": args.model_id,
                        "prompt_version": args.prompt_version,
                        "gpu_type": summary_data["gpu_type"],
                        "label": args.label,
                        "git_sha": os.environ.get("GITHUB_SHA", "local"),
                    }
                )

                mlflow.log_metrics(
                    {
                        "generated_requests": gen_count,
                        "scored_requests": score_count,
                        "pass_rate": summary_data["pass_rate"],
                        "usable_rate": summary_data["usable_rate"],
                        "format_fail_rate": summary_data["format_fail_rate"],
                        "static_discard_rate": summary_data["static_discard_rate"],
                        "contradiction_discard_rate": summary_data[
                            "contradiction_discard_rate"
                        ],
                        "semantic_discard_rate": summary_data["semantic_discard_rate"],
                        "generation_wall_seconds": summary_data[
                            "generation_wall_seconds"
                        ],
                        "estimated_gpu_cost_usd": summary_data[
                            "estimated_gpu_cost_usd"
                        ],
                    }
                )

                mlflow.log_artifacts(str(results_dir), artifact_path="eval_results")
                logger.info("MLflow에 평가 결과 기록 완료 (Run: %s)", run_name)
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "MLflow 기록 중 오류 발생 (평가 파이프라인은 계속 진행됩니다): %s",
                error,
            )
    else:
        logger.info("MLFLOW_TRACKING_URI가 설정되지 않아 MLflow 기록을 건너뜁니다.")

    # 3. S3 백업 업로드 (AWS 자격 증명 또는 S3_BUCKET이 있을 때)
    s3_bucket = os.environ.get("S3_BUCKET", args.s3_bucket)
    run_id = f"eval-{args.label}"
    s3_prefix = f"evals/{run_id}"

    try:
        import boto3

        s3 = boto3.client("s3")
        for file_path in results_dir.glob("*"):
            if file_path.is_file():
                s3_key = f"{s3_prefix}/{file_path.name}"
                s3.upload_file(str(file_path), s3_bucket, s3_key)
                logger.info("S3 업로드 완료: s3://%s/%s", s3_bucket, s3_key)
        logger.info("S3 평가 아티팩트 보관 완료 (s3://%s/%s/)", s3_bucket, s3_prefix)
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "S3 업로드 중 오류 발생 (선택적 단계이므로 계속 진행): %s", error
        )

    # 4. GitHub Actions Step Summary 출력 (메타데이터 + GPU 비용 + 성적표)
    summary_md_file = results_dir / "summary.md"
    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_path and summary_md_file.exists():
        langfuse_host = os.environ.get(
            "LANGFUSE_HOST", "https://jp.cloud.langfuse.com"
        ).rstrip("/")
        gpu_name = summary_data["gpu_type"]
        gpu_rate = summary_data["gpu_hourly_rate_usd"]
        wall_dur = format_duration(summary_data["generation_wall_seconds"])
        bill_dur = format_duration(summary_data["billable_gpu_seconds"])
        gpu_cost = summary_data["estimated_gpu_cost_usd"]
        u_rate = summary_data["usable_rate"]
        u_cnt = summary_data["usable_count"]
        d_cnt = summary_data["determined_requests"]

        with (
            open(step_summary_path, "a", encoding="utf-8") as out,
            summary_md_file.open(encoding="utf-8") as src,
        ):
            out.write(f"\n# 🚀 LLMOps 모델 평가 리포트 (`{args.label}`)\n\n")
            out.write("### 1. 실행 환경 및 GPU 인프라 비용\n\n")
            out.write("| 항목 | 값 |\n|---|---|\n")
            out.write(f"| **평가 모델 (`model_id`)** | `{args.model_id}` |\n")
            out.write(
                "| **트랙 유형 / 프롬프트** | "
                f"`{args.track_type}` / `{args.prompt_version}` |\n"
            )
            out.write(
                "| **추론 GPU 기종 (단가)** | "
                f"`{gpu_name}` (`${gpu_rate:.2f} / hr`) |\n"
            )
            out.write(
                "| **1단계 생성 소요 시간** | "
                f"{wall_dur} (유휴 대기 60초 포함 과금 기준: {bill_dur}) |\n"
            )
            out.write(
                "| **💰 예상 GPU 과금액** | "
                f"**`${gpu_cost:.4f} USD`** |\n"
            )
            out.write(
                "| **🎯 실질 통과율** | "
                f"**`{u_rate:.0%}`** (`{u_cnt} / {d_cnt}건`) |\n"
            )
            out.write(
                "| **🔎 Langfuse 대시보드** | "
                f"[Langfuse 열기 (`session_id = {args.label}`)]({langfuse_host}) |\n\n"
            )
            out.write("---\n\n")
            out.write(src.read())
            out.write("\n")


if __name__ == "__main__":
    main()
