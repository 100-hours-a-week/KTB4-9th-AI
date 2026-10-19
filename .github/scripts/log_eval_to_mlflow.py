"""평가 결과 JSONL과 요약 메트릭을 MLflow와 S3에 기록하는 스크립트.

AI 팀의 evals.summarize 실행 후 생성된 결과 파일들을 읽어,
1. 중앙 MLflow Tracking 서버에 평가 Run을 생성하고 메트릭(통과율, 폐기율)을 기록한다.
2. S3 버킷(cosmos-mlops-artifacts/evals/<run_id>/)에 결과 원본을 불변 업로드한다.

사용법:
    python scripts/log_eval_to_mlflow.py --label qwen3-122b-eval --track-type base_model --model-id Qwen/Qwen3.5-122B-A10B-FP8
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Log evaluation results to MLflow and S3")
    parser.add_argument("--label", required=True, help="evals/results/<label> 디렉터리 이름")
    parser.add_argument("--track-type", default="base_model", choices=["base_model", "trained_model"])
    parser.add_argument("--model-id", required=True, help="평가 대상 모델 식별자")
    parser.add_argument("--prompt-version", default="generate_problem/v11", help="프롬프트 버전")
    parser.add_argument("--s3-bucket", default="cosmos-v2-stage-mlflow-artifacts-352635536602", help="S3 버킷 이름")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    args = parse_args()
    results_dir = Path("evals/results") / args.label

    if not results_dir.exists():
        logger.error("평가 결과 디렉터리가 존재하지 않습니다: %s", results_dir)
        sys.exit(1)

    gen_meta = load_json(results_dir / "generate_meta.json")
    score_meta = load_json(results_dir / "score_meta.json")
    summary_file = results_dir / "summary.json"
    summary_data = load_json(summary_file) if summary_file.exists() else {}

    gen_count = count_lines(results_dir / "generations.jsonl")
    score_count = count_lines(results_dir / "scores.jsonl")

    logger.info("평가 결과 수집 완료: 생성 %d건, 채점 %d건", gen_count, score_count)

    # 1. MLflow 기록 (MLFLOW_TRACKING_URI가 있을 때만 실행)
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("cosmos-model-eval")

            run_name = f"{args.track_type}-{args.label}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "track_type": args.track_type,
                    "model_id": args.model_id,
                    "prompt_version": args.prompt_version,
                    "label": args.label,
                    "git_sha": os.environ.get("GITHUB_SHA", "local"),
                })

                # 메트릭 기록
                mlflow.log_metric("generated_requests", gen_count)
                mlflow.log_metric("scored_requests", score_count)

                if "pass_rate" in summary_data:
                    mlflow.log_metric("pass_rate", float(summary_data["pass_rate"]))
                if "static_pass_rate" in summary_data:
                    mlflow.log_metric("static_pass_rate", float(summary_data["static_pass_rate"]))
                if "semantic_pass_rate" in summary_data:
                    mlflow.log_metric("semantic_pass_rate", float(summary_data["semantic_pass_rate"]))

                # 결과 파일 아티팩트 업로드
                mlflow.log_artifacts(str(results_dir), artifact_path="eval_results")
                logger.info("MLflow에 평가 결과 기록 완료 (Run: %s)", run_name)
        except Exception as error:  # noqa: BLE001
            logger.warning("MLflow 기록 중 오류 발생 (평가 파이프라인은 계속 진행됩니다): %s", error)
    else:
        logger.info("MLFLOW_TRACKING_URI가 설정되지 않아 MLflow 기록을 건너뜁니다.")

    # 2. S3 백업 업로드 (AWS 자격 증명 또는 S3_BUCKET이 있을 때)
    s3_bucket = os.environ.get("S3_BUCKET", args.s3_bucket)
    run_id = f"eval-{args.label}"
    s3_prefix = f"evals/{run_id}"

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError

        s3 = boto3.client("s3")
        for file_path in results_dir.glob("*"):
            if file_path.is_file():
                s3_key = f"{s3_prefix}/{file_path.name}"
                s3.upload_file(str(file_path), s3_bucket, s3_key)
                logger.info("S3 업로드 완료: s3://%s/%s", s3_bucket, s3_key)
        logger.info("S3 평가 아티팩트 보관 완료 (s3://%s/%s/)", s3_bucket, s3_prefix)
    except Exception as error:  # noqa: BLE001
        logger.warning("S3 업로드 중 오류 발생 (선택적 단계이므로 계속 진행): %s", error)

    # 3. GitHub Actions Step Summary 출력 (있으면 요약 테이블 표시)
    summary_md_file = results_dir / "summary.md"
    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_path and summary_md_file.exists():
        with open(step_summary_path, "a", encoding="utf-8") as out, summary_md_file.open(encoding="utf-8") as src:
            out.write(f"\n## 📊 LLMOps 평가 결과 요약 ({args.model_id})\n\n")
            out.write(src.read())
            out.write("\n")


if __name__ == "__main__":
    main()
