"""Modal 서버리스 LoRA 파인튜닝 실행 스크립트 (training/train_sft.py 연동).

RunPod Pod가 떠 있지 않은 상태에서도 GitHub Actions에서 즉시 파인튜닝을
실행하고 검증할 수 있도록, AI 팀의 `training/train_sft.py`를 Modal GPU 위에서
구동하고 결과(run_config.json 및 Hugging Face Hub 업로드)를 반환합니다.

사용법:
    uv run --with modal modal run modal_train.py \
        --model Qwen/Qwen3.5-27B \
        --run-name qwen35-27b-lora-v1 \
        --max-steps 5
"""

import json
import os
import subprocess
from pathlib import Path

import modal

APP_NAME = os.environ.get("MODAL_TRAIN_APP_NAME", "cosmos-lora-train")
VOLUME_NAME = "cosmos-model-cache"
CACHE_DIR = "/root/.cache/huggingface"
RUNS_DIR = "/root/.cache/huggingface/cosmos_runs"
MINUTES = 60

SELECTED_GPU = os.environ.get("MODAL_GPU_TYPE", "A100-80GB")
if SELECTED_GPU.upper() == "A100":
    SELECTED_GPU = "A100-80GB"
elif SELECTED_GPU.upper() in ("2XA100", "2XA100-80GB"):
    SELECTED_GPU = "A100-80GB:2"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.6.0",
        "unsloth",
        "unsloth_zoo",
        "transformers>=5.0.0",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "safetensors",
        "huggingface_hub>=0.26.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
    )
    .add_local_dir("training", remote_path="/workspace/training")
    .add_local_dir("src", remote_path="/workspace/src")
    .add_local_dir("sql", remote_path="/workspace/sql")
)


def _prepare_fallback_dataset_if_needed(work_dir: Path) -> tuple[Path, Path]:
    """train.jsonl, val.jsonl이 없으면 fewshot_seeds.jsonl을 변환해 생성한다."""
    train_path = work_dir / "training/data/train.jsonl"
    val_path = work_dir / "training/data/val.jsonl"
    if train_path.exists() and val_path.exists():
        return train_path, val_path

    converted_path = work_dir / "training/data/converted.jsonl"
    seeds_path = work_dir / "sql/fewshot_seeds.jsonl"

    if not converted_path.exists() and seeds_path.exists():
        print("[Modal-Train] sql/fewshot_seeds.jsonl 기반으로 학습 원본을 구성합니다.")
        converted_path.parent.mkdir(parents=True, exist_ok=True)
        lines = seeds_path.read_text(encoding="utf-8").strip().splitlines()
        with converted_path.open("w", encoding="utf-8") as out:
            for line in lines:
                if not line.strip():
                    continue
                row = json.loads(line)
                cat = row["category"]
                converted_row = {
                    "difficulty": row["difficulty"],
                    "category": cat,
                    "category_select_reason": row.get(
                        "category_select_reason",
                        f"{cat} 알고리즘의 핵심 원리를 평가하는 문제이다.",
                    ),
                    "algorithm_core": row.get(
                        "algorithm_core",
                        f"주어진 입력 조건에서 {cat} 기법으로 최적해를 구한다.",
                    ),
                    "problem_title": row["problem_title"],
                    "problem_content": row["problem_content"],
                    "input_format": row["input_format"],
                    "output_format": row["output_format"],
                    "input_constraints": row["input_constraints"],
                    "execution_limits": row["execution_limits"],
                    "problem_examples": row["problem_examples"],
                }
                out.write(json.dumps(converted_row, ensure_ascii=False) + "\n")

    print("[Modal-Train] training.data.build_dataset 실행 중...")
    subprocess.run(
        [
            "python",
            "-m",
            "training.data.build_dataset",
            "--input",
            str(converted_path),
            "--train",
            str(train_path),
            "--val",
            str(val_path),
        ],
        cwd=str(work_dir),
        check=True,
    )
    return train_path, val_path


@app.function(
    image=train_image,
    gpu=SELECTED_GPU,
    volumes={CACHE_DIR: volume},
    timeout=120 * MINUTES,
    secrets=[modal.Secret.from_name("cosmos-modal-secret", required_keys=[])],
)
def run_sft_on_modal(
    model_id: str,
    run_name: str,
    epochs: float = 2.0,
    lr: float = 2e-4,
    rank: int = 16,
    alpha: int = 16,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_seq_length: int = 4096,
    max_steps: int = -1,
    merge: bool = False,
    hf_repo_id: str = "",
) -> dict:
    """Modal GPU에서 스냅샷 준비 후 training/train_sft.py를 실행한다."""
    from huggingface_hub import HfApi, snapshot_download

    volume.reload()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

    os.environ["HF_HOME"] = CACHE_DIR

    print(f"[Modal-Train] 베이스 모델 스냅샷 확인/다운로드: {model_id}")
    snapshot_download(model_id, token=hf_token)
    volume.commit()

    work_dir = Path("/workspace")
    train_jsonl, val_jsonl = _prepare_fallback_dataset_if_needed(work_dir)
    output_dir = Path(RUNS_DIR) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "training/train_sft.py",
        "--model",
        model_id,
        "--train",
        str(train_jsonl),
        "--val",
        str(val_jsonl),
        "--output",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--rank",
        str(rank),
        "--alpha",
        str(alpha),
        "--batch-size",
        str(batch_size),
        "--grad-accum",
        str(grad_accum),
        "--max-seq-length",
        str(max_seq_length),
        "--max-steps",
        str(max_steps),
    ]
    if merge:
        cmd.append("--merge")

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    print(f"[Modal-Train] 학습 실행 명령어: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(work_dir), env=env, check=True)
    volume.commit()

    run_cfg_path = output_dir / "run_config.json"
    run_config = json.loads(run_cfg_path.read_text(encoding="utf-8"))

    if hf_repo_id and hf_token:
        target_dir = output_dir / ("merged" if merge else "adapter")
        print(f"[Modal-Train] Hugging Face Hub 업로드: {target_dir} -> {hf_repo_id}")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
        msg = f"Upload trained LoRA ({run_name}) from COSMOS Modal"
        api.upload_folder(
            repo_id=hf_repo_id,
            folder_path=str(target_dir),
            repo_type="model",
            commit_message=msg,
        )
        api.upload_file(
            path_or_fileobj=str(run_cfg_path),
            path_in_repo="run_config.json",
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message="Add run_config.json metadata",
        )

    return run_config


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3.5-27B",
    run_name: str = "qwen35-27b-lora-v1",
    epochs: float = 2.0,
    lr: float = 2e-4,
    rank: int = 16,
    alpha: int = 16,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_seq_length: int = 4096,
    max_steps: int = -1,
    merge: bool = False,
    hf_repo_id: str = "",
    local_out_dir: str = "training/runs",
):
    run_config = run_sft_on_modal.remote(
        model_id=model,
        run_name=run_name,
        epochs=epochs,
        lr=lr,
        rank=rank,
        alpha=alpha,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_seq_length=max_seq_length,
        max_steps=max_steps,
        merge=merge,
        hf_repo_id=hf_repo_id,
    )
    out_dir = Path(local_out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = out_dir / "run_config.json"
    cfg_file.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[Modal-Train] 로컬 메타데이터 저장 완료: {cfg_file}")
