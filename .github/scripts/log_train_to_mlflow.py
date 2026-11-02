"""학습 결과(run_config.json 및 어댑터)를 HF Hub, MLflow, S3, GitHub에 기록한다.

training/train_sft.py 실행 후 생성된 출력 디렉터리(--run-dir)를 읽어:
1. --hf-repo-id 지정 시 adapter(또는 merged)와 run_config.json을 HF Hub에 업로드한다.
2. 중앙 MLflow 서버(cosmos-model-train 실험)에 파라미터와 지표를 기록한다.
3. S3 아티팩트 버킷에 run_config.json 원본을 백업한다.
4. GitHub Actions 요약 화면(GITHUB_STEP_SUMMARY)에 학습 리포트 표를 출력한다.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload trained LoRA adapter and log metrics to MLflow/S3"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="train_sft.py의 --output 디렉터리 경로 (--from-hf-repo 사용 시 생략 가능)",
    )
    parser.add_argument(
        "--from-hf-repo",
        default="",
        help="HF Hub 어댑터 레포지토리에서 메타데이터를 내려받아 MLflow에 기록",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="MLflow 및 리포트에 표시할 실행 이름 (비우면 폴더명 사용)",
    )
    parser.add_argument(
        "--hf-repo-id",
        default="",
        help="어댑터를 업로드할 HF 레포지토리 ID (예: org/qwen3.5-27b-lora-v1)",
    )
    parser.add_argument(
        "--upload-merged",
        action="store_true",
        help="adapter/ 대신 merged/ 병합 모델을 HF Hub에 업로드",
    )
    parser.add_argument(
        "--s3-bucket",
        default="cosmos-v2-stage-mlflow-artifacts-352635536602",
        help="S3 아티팩트 백업 버킷 이름",
    )
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}시간 {mins}분 {secs}초 ({seconds:.0f}s)"
    if mins > 0:
        return f"{mins}분 {secs}초 ({seconds:.0f}s)"
    return f"{seconds:.1f}초"


def upload_to_huggingface(
    run_dir: Path, hf_repo_id: str, upload_merged: bool
) -> str | None:
    """학습 산출물과 run_config.json을 Hugging Face Hub에 업로드한다."""
    hf_token = (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    ).strip()
    if not hf_token:
        logger.warning("HF_TOKEN이 설정되지 않아 Hugging Face 업로드를 건너뜁니다.")
        return None

    target_dir = run_dir / ("merged" if upload_merged else "adapter")
    if not target_dir.exists():
        logger.error("업로드할 대상 폴더가 존재하지 않습니다: %s", target_dir)
        return None

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
        logger.info("Hugging Face 업로드 시작: %s -> %s", target_dir, hf_repo_id)
        artifact_kind = "merged model" if upload_merged else "LoRA adapter"
        msg = f"Upload trained {artifact_kind} from COSMOS LLMOps pipeline"
        api.upload_folder(
            repo_id=hf_repo_id,
            folder_path=str(target_dir),
            repo_type="model",
            commit_message=msg,
        )
        run_cfg_path = run_dir / "run_config.json"
        if run_cfg_path.exists():
            api.upload_file(
                path_or_fileobj=str(run_cfg_path),
                path_in_repo="run_config.json",
                repo_id=hf_repo_id,
                repo_type="model",
                commit_message="Add training run_config.json metadata",
            )
        hf_url = f"https://huggingface.co/{hf_repo_id}"
        logger.info("Hugging Face 업로드 완료: %s", hf_url)
        return hf_url
    except Exception as error:  # noqa: BLE001
        logger.warning("Hugging Face 업로드 중 오류 발생: %s", error)
        return None


def log_to_mlflow(
    config: dict, run_cfg_path: Path, run_name: str, hf_repo_id: str
) -> None:
    """run_config.json의 파라미터와 스텝별 손실값, 최종 지표를 MLflow에 기록한다."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not mlflow_uri:
        logger.info("MLFLOW_TRACKING_URI가 설정되지 않아 MLflow 기록을 건너뜁니다.")
        return

    try:
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("cosmos-model-train")

        args_data = config.get("args", {})
        model_info = config.get("model", {})
        result_data = config.get("result", {})
        data_info = config.get("data", {})
        train_data = data_info.get("train", {})
        val_data = data_info.get("val", {})

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "base_model": model_info.get("id", args_data.get("model", "")),
                    "lora_rank": args_data.get("rank", 16),
                    "lora_alpha": args_data.get("alpha", 16),
                    "epochs": args_data.get("epochs", 2),
                    "learning_rate": args_data.get("lr", 2e-4),
                    "batch_size": args_data.get("batch_size", 1),
                    "grad_accum": args_data.get("grad_accum", 8),
                    "max_seq_length": args_data.get("max_seq_length", 4096),
                    "gpu": config.get("gpu", "unknown"),
                    "train_count": train_data.get("count", 0),
                    "val_count": val_data.get("count", 0),
                    "train_sha256": str(train_data.get("sha256", ""))[:16],
                    "hf_repo_id": hf_repo_id or "local_only",
                    "git_sha": os.environ.get("GITHUB_SHA", "local"),
                }
            )

            for entry in config.get("log_history", []):
                step = int(entry.get("step", 0))
                if "loss" in entry:
                    mlflow.log_metric("train_loss", float(entry["loss"]), step=step)
                if "eval_loss" in entry:
                    val_loss = float(entry["eval_loss"])
                    mlflow.log_metric("val_eval_loss", val_loss, step=step)
                if "learning_rate" in entry:
                    lr_val = float(entry["learning_rate"])
                    mlflow.log_metric("learning_rate", lr_val, step=step)

            summary_metrics = {
                "train_seconds": float(result_data.get("seconds", 0)),
                "total_steps": float(result_data.get("steps", 0)),
                "peak_vram_gb": float(result_data.get("peak_vram_gb", 0.0)),
            }
            if result_data.get("final_train_loss") is not None:
                summary_metrics["final_train_loss"] = float(
                    result_data["final_train_loss"]
                )
            if result_data.get("eval_loss") is not None:
                summary_metrics["final_eval_loss"] = float(result_data["eval_loss"])

            mlflow.log_metrics(summary_metrics)
            mlflow.log_artifact(str(run_cfg_path), artifact_path="training_metadata")
            logger.info("MLflow 학습 기록 완료 (Run: %s)", run_name)
    except Exception as error:  # noqa: BLE001
        logger.warning("MLflow 기록 중 오류 발생 (계속 진행): %s", error)


def backup_to_s3(run_cfg_path: Path, run_name: str, s3_bucket: str) -> None:
    """S3 버킷에 run_config.json을 백업한다."""
    bucket = os.environ.get("S3_BUCKET", s3_bucket)
    s3_key = f"training/{run_name}/run_config.json"
    try:
        import boto3

        s3 = boto3.client("s3")
        s3.upload_file(str(run_cfg_path), bucket, s3_key)
        logger.info("S3 학습 메타데이터 백업 완료: s3://%s/%s", bucket, s3_key)
    except Exception as error:  # noqa: BLE001
        logger.warning("S3 업로드 생략 또는 실패 (계속 진행): %s", error)


def write_github_summary(
    config: dict, run_name: str, hf_repo_id: str, hf_url: str | None
) -> None:
    """GitHub Actions Step Summary에 마크다운 학습 리포트를 작성한다."""
    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not step_summary_path:
        return

    args_data = config.get("args", {})
    model_info = config.get("model", {})
    result_data = config.get("result", {})
    data_info = config.get("data", {})
    train_data = data_info.get("train", {})
    val_data = data_info.get("val", {})

    base_model = model_info.get("id", args_data.get("model", "unknown"))
    gpu_name = config.get("gpu", "unknown")
    duration_str = format_duration(float(result_data.get("seconds", 0)))
    steps = result_data.get("steps", 0)
    final_train_loss = result_data.get("final_train_loss")
    eval_loss = result_data.get("eval_loss")
    peak_vram = result_data.get("peak_vram_gb", 0.0)

    train_loss_str = (
        f"`{final_train_loss:.4f}`" if final_train_loss is not None else "N/A"
    )
    eval_loss_str = f"**`{eval_loss:.4f}`**" if eval_loss is not None else "N/A"
    hf_link_str = (
        f"[{hf_repo_id}]({hf_url})" if hf_url else (hf_repo_id or "볼륨 로컬 저장")
    )

    with open(step_summary_path, "a", encoding="utf-8") as out:
        out.write(f"\n# 🛠️ LLMOps LoRA 파인튜닝 학습 리포트 (`{run_name}`)\n\n")
        out.write("### 1. 학습 요약 및 하드웨어 지표\n\n")
        out.write("| 항목 | 값 |\n|---|---|\n")
        out.write(f"| **베이스 모델 (`base_model`)** | `{base_model}` |\n")
        out.write(f"| **어댑터 배포 경로 (`hf_repo_id`)** | {hf_link_str} |\n")
        out.write(
            f"| **학습 GPU 기종 / 피크 VRAM** | `{gpu_name}` / **`{peak_vram} GB`** |\n"
        )
        out.write(
            f"| **LoRA 설정 (`r` / `alpha`)** | "
            f"`r={args_data.get('rank', 16)}`, "
            f"`alpha={args_data.get('alpha', 16)}` |\n"
        )
        bs = args_data.get("batch_size", 1)
        ga = args_data.get("grad_accum", 8)
        out.write(
            f"| **학습 하이퍼파라미터** | "
            f"`epochs={args_data.get('epochs', 2)}`, "
            f"`lr={args_data.get('lr', 2e-4)}`, `batch={bs}×{ga}` |\n"
        )
        out.write(
            f"| **총 학습 스텝 / 소요 시간** | `{steps} steps` / {duration_str} |\n"
        )
        out.write(f"| **📉 최종 Train Loss** | {train_loss_str} |\n")
        out.write(f"| **🎯 최종 Validation Eval Loss** | {eval_loss_str} |\n\n")

        out.write("### 2. 데이터셋 검증 통계\n\n")
        out.write("| 구분 | 샘플 수 | 토큰 P50 | 토큰 P95 | 토큰 Max | SHA-256 |\n")
        out.write("|---|---:|---:|---:|---:|---|\n")
        splits = (("Train", train_data), ("Validation", val_data))
        for split_label, split_dict in splits:
            tok = split_dict.get("total_tokens", {})
            sha_short = str(split_dict.get("sha256", ""))[:12]
            out.write(
                f"| **{split_label}** | {split_dict.get('count', 0)}건 | "
                f"{tok.get('p50', 0)} | {tok.get('p95', 0)} | "
                f"{tok.get('max', 0)} | `{sha_short}` |\n"
            )
        out.write("\n")


def fetch_metadata_from_hf(hf_repo_id: str, cache_dir: Path) -> Path | None:
    """HF 레포지토리에서 run_config.json 또는 adapter_config.json을 수집한다."""
    hf_token = (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_cfg_path = cache_dir / "run_config.json"

    try:
        from huggingface_hub import hf_hub_download

        try:
            downloaded = hf_hub_download(
                repo_id=hf_repo_id,
                filename="run_config.json",
                token=hf_token,
            )
            out_cfg_path.write_text(
                Path(downloaded).read_text(encoding="utf-8"), encoding="utf-8"
            )
            logger.info("HF Hub에서 run_config.json 수집 완료: %s", hf_repo_id)
            return out_cfg_path
        except Exception:  # noqa: BLE001
            logger.info(
                "HF 레포지토리(%s)에 run_config.json이 없어 "
                "adapter_config.json에서 메타데이터를 추출합니다.",
                hf_repo_id,
            )

        adapter_file = hf_hub_download(
            repo_id=hf_repo_id,
            filename="adapter_config.json",
            token=hf_token,
        )
        adapter_cfg = json.loads(Path(adapter_file).read_text(encoding="utf-8"))

        log_history = []
        steps = 0
        final_train_loss = None
        eval_loss = None
        try:
            state_file = hf_hub_download(
                repo_id=hf_repo_id,
                filename="trainer_state.json",
                token=hf_token,
            )
            state_cfg = json.loads(Path(state_file).read_text(encoding="utf-8"))
            log_history = state_cfg.get("log_history", [])
            steps = int(state_cfg.get("global_step", 0))
            train_losses = [x["loss"] for x in log_history if "loss" in x]
            eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]
            if train_losses:
                final_train_loss = train_losses[-1]
            if eval_losses:
                eval_loss = eval_losses[-1]
        except Exception:  # noqa: BLE001
            pass

        base_model = adapter_cfg.get("base_model_name_or_path", "unknown")
        synthetic_cfg = {
            "model": {"id": base_model, "revision": "hf-adapter"},
            "gpu": "RunPod (External)",
            "args": {
                "model": base_model,
                "rank": adapter_cfg.get("r", 16),
                "alpha": adapter_cfg.get("lora_alpha", 16),
                "epochs": 0,
                "lr": 0.0,
                "batch_size": 1,
                "grad_accum": 8,
                "max_seq_length": 4096,
            },
            "data": {"train": {"count": 0}, "val": {"count": 0}},
            "result": {
                "seconds": 0.0,
                "steps": steps,
                "final_train_loss": final_train_loss,
                "eval_loss": eval_loss,
                "peak_vram_gb": 0.0,
            },
            "log_history": log_history,
        }
        out_cfg_path.write_text(
            json.dumps(synthetic_cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out_cfg_path
    except Exception as error:  # noqa: BLE001
        logger.warning("HF Hub 어댑터 메타데이터 조회 실패: %s", error)
        return None


def main() -> None:
    args = parse_args()
    hf_repo_id = (args.from_hf_repo or args.hf_repo_id or "").strip()

    if args.from_hf_repo:
        target_dir = args.run_dir or Path("/tmp/hf_adapter_meta")
        run_cfg_path = fetch_metadata_from_hf(args.from_hf_repo, target_dir)
        if run_cfg_path is None or not run_cfg_path.exists():
            logger.warning("HF 어댑터 메타데이터를 찾을 수 없어 기록을 종료합니다.")
            return
        config = json.loads(run_cfg_path.read_text(encoding="utf-8"))
        run_name = args.run_name or args.from_hf_repo.replace("/", "-")
        hf_url = f"https://huggingface.co/{args.from_hf_repo}"
        log_to_mlflow(config, run_cfg_path, run_name, hf_repo_id)
        backup_to_s3(run_cfg_path, run_name, args.s3_bucket)
        write_github_summary(config, run_name, hf_repo_id, hf_url)
        return

    if args.run_dir is None:
        logger.error("--run-dir 또는 --from-hf-repo 중 하나는 반드시 지정해야 합니다.")
        sys.exit(1)

    run_cfg_path = args.run_dir / "run_config.json"
    if not run_cfg_path.exists():
        logger.error(
            "학습 결과 파일(run_config.json)을 찾을 수 없습니다: %s", run_cfg_path
        )
        sys.exit(1)

    config = json.loads(run_cfg_path.read_text(encoding="utf-8"))
    run_name = args.run_name or args.run_dir.name

    hf_url = None
    if args.hf_repo_id:
        hf_url = upload_to_huggingface(
            args.run_dir, args.hf_repo_id, args.upload_merged
        )

    log_to_mlflow(config, run_cfg_path, run_name, args.hf_repo_id)
    backup_to_s3(run_cfg_path, run_name, args.s3_bucket)
    write_github_summary(config, run_name, args.hf_repo_id, hf_url)


if __name__ == "__main__":
    main()
