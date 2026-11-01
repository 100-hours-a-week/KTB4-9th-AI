"""학습한 LoRA 어댑터를 베이스 모델과 같이 vLLM으로 서빙한다.

평가용이다. 한 서버에서 베이스(--model 이름)와 어댑터(--adapter 이름)를
둘 다 요청할 수 있어서, evals.generate --model만 바꿔 같은 조건으로 비교한다.
병합 모델은 bf16 반올림으로 LoRA 변화량 일부를 잃으므로 어댑터로 서빙한다.

vLLM venv(학습 venv와 torch 버전이 다르다)의 파이썬으로 실행한다.
Mac에서는 SSH 터널(-L 8000:localhost:8000)로 붙는다.

실행:
    source /workspace/env.sh
    /workspace/vllm-venv/bin/python serve_vllm.py --model Qwen/Qwen3.5-27B \
        --adapter 27b-v1=/workspace/runs/27b-v1/adapter
"""

import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM으로 베이스 + LoRA 어댑터 서빙")
    parser.add_argument("--model", required=True, help="베이스 HF 모델 ID")
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="요청할 이름=어댑터 폴더. 여러 번 줄 수 있다",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="프롬프트 + 생성 최대 토큰. 정답이 최대 2,300토큰쯤이라 넉넉히 둔다",
    )
    parser.add_argument(
        "--api-key-file", type=Path, default=Path("/workspace/.vllm_api_key")
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="GPU 수")
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="CUDA 그래프를 끈다. 기동은 빨라지고 생성은 느려진다",
    )
    return parser.parse_args()


def parse_adapters(specs: list[str], model: str) -> dict[str, Path]:
    """
    NAME=PATH 목록을 확인해서 이름 → 어댑터 폴더로 바꾼다.

    Raises:
        ValueError: 형식이 틀렸거나, 이름이 겹치거나, 어댑터 폴더가 아닌 경우
    """
    adapters: dict[str, Path] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--adapter는 NAME=PATH 형식이어야 합니다: {spec}")
        if name in adapters or name == model:
            raise ValueError(f"요청 이름이 겹칩니다: {name}")
        if not (Path(path) / "adapter_config.json").exists():
            raise ValueError(
                f"어댑터 폴더가 아닙니다 (adapter_config.json 없음): {path}"
            )
        adapters[name] = Path(path)
    return adapters


def max_lora_rank(adapters: dict[str, Path]) -> int:
    """vLLM은 rank 상한만큼 LoRA 메모리를 미리 잡는다. 어댑터 중 가장 큰 r을 쓴다."""
    ranks = [
        json.loads((path / "adapter_config.json").read_text())["r"]
        for path in adapters.values()
    ]
    return max(ranks)


def main() -> None:
    args = parse_args()
    adapters = parse_adapters(args.adapter, args.model)
    # 네트워크 없이 캐시에서만 찾는다. 없으면 받지 않고 예외를 낸다
    snapshot = Path(snapshot_download(args.model, local_files_only=True))

    logger.info("베이스 %s → %s", args.model, snapshot)
    for name, path in adapters.items():
        logger.info("어댑터 %s → %s", name, path)
    if adapters:
        logger.info("max_lora_rank %d", max_lora_rank(adapters))


if __name__ == "__main__":
    main()
