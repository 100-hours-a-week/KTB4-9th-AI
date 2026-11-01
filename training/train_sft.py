"""Qwen3.5 모델을 문제 생성 데이터로 LoRA 학습한다.

RunPod Pod에서 돌리는 독립 스크립트다. src/를 import하지 않으므로
RunPod 볼륨에 저장한 학습 venv(unsloth, transformers v5)와 build_dataset.py가 만든
train.jsonl, val.jsonl만 있으면 된다.

모델은 HF 캐시(HF_HOME)에 미리 받아둔 스냅샷에서 읽는다.
실행할 때 HF_HUB_OFFLINE=1을 주면 실수로 다시 받는 일이 없다.

실행:
    source /workspace/env.sh
    HF_HUB_OFFLINE=1 python train_sft.py --model Qwen/Qwen3.5-27B \
        --train /workspace/data/train.jsonl --val /workspace/data/val.jsonl \
        --output /workspace/runs/27b-v1
"""

import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5 LoRA 학습")

    paths = parser.add_argument_group("경로")
    paths.add_argument("--model", required=True, help="HF 모델 ID 또는 로컬 폴더")
    paths.add_argument("--train", type=Path, required=True, help="학습 jsonl")
    paths.add_argument("--val", type=Path, required=True, help="검증 jsonl")
    paths.add_argument("--output", type=Path, required=True, help="결과 폴더")

    lora = parser.add_argument_group("LoRA")
    lora.add_argument("--rank", type=int, default=16)
    lora.add_argument("--alpha", type=int, default=16)

    train = parser.add_argument_group("학습")
    train.add_argument("--epochs", type=float, default=2)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--batch-size", type=int, default=1, help="GPU당 배치")
    train.add_argument("--grad-accum", type=int, default=8, help="그래디언트 누적")
    train.add_argument(
        "--warmup", type=float, default=0.05, help="1 미만이면 비율, 이상이면 스텝"
    )
    train.add_argument("--max-seq-length", type=int, default=4096)
    train.add_argument(
        "--max-steps", type=int, default=-1, help="스모크용. 양수면 에폭 대신 쓴다"
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--resume", action="store_true", help="output의 마지막 체크포인트부터 이어서"
    )

    merge = parser.add_argument_group("병합")
    merge.add_argument("--merge", action="store_true", help="학습 후 bf16 병합 저장")
    merge.add_argument(
        "--merge-dir", type=Path, help="병합 저장 폴더. 없으면 output/merged"
    )

    return parser.parse_args()


def resolve_snapshot(model: str) -> Path:
    """
    모델 ID를 HF 캐시의 스냅샷 폴더로 바꾼다. 로컬 폴더면 그대로 쓴다.

    폴더 경로로 로드하면 병합할 때 unsloth가 Hub에 묻지 않고
    이 폴더의 원본 가중치를 읽는다.

    Raises:
        FileNotFoundError: 캐시에 없거나 가중치 파일이 빠진 경우
    """
    path = Path(model)
    if not path.is_dir():
        # 네트워크 없이 캐시에서만 찾는다. 없으면 받지 않고 예외를 낸다
        path = Path(snapshot_download(model, local_files_only=True))

    index = path / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        missing = sorted(name for name in shards if not (path / name).exists())
        if missing:
            raise FileNotFoundError(f"{path}에 가중치 파일이 없습니다: {missing}")
    elif not list(path.glob("*.safetensors")):
        raise FileNotFoundError(f"{path}에 safetensors 파일이 없습니다.")

    return path


def main() -> None:
    args = parse_args()
    snapshot = resolve_snapshot(args.model)
    logger.info("모델 %s → %s", args.model, snapshot)


if __name__ == "__main__":
    main()
