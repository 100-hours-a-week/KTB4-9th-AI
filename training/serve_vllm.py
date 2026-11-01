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
import os
import secrets
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# 2B 스모크에서 확인한 서빙 환경. 이미 설정된 값이 있으면 그 값을 쓴다
DEFAULT_ENV = {
    "HF_HUB_OFFLINE": "1",  # 볼륨에 받은 모델만 쓴다
    "VLLM_USE_FLASHINFER_SAMPLER": "0",  # JIT 빌드에 ninja, nvcc가 필요해 끈다
    "VLLM_CACHE_ROOT": "/workspace/vllm-cache",  # 컴파일 결과를 볼륨에 남긴다
}


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


def load_api_key(path: Path) -> str:
    """API 키를 읽는다. 없으면 만들어 나만 읽을 수 있게 저장한다."""
    if path.exists() and (key := path.read_text().strip()):
        return key
    # token_urlsafe는 '-'로 시작할 수 있어 vLLM 인자 해석이 깨진다
    key = secrets.token_hex(16)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key + "\n")
    path.chmod(0o600)
    logger.info("API 키를 새로 만들었습니다: %s", path)
    return key


def build_command(
    args: argparse.Namespace, snapshot: Path, adapters: dict[str, Path], api_key: str
) -> list[str]:
    """vllm serve 명령. 요청 이름은 베이스 HF ID와 어댑터 이름이다."""
    command = [
        "vllm",
        "serve",
        str(snapshot),
        "--served-model-name",
        args.model,
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--port",
        str(args.port),
        "--api-key",
        api_key,
    ]
    if adapters:
        command += ["--enable-lora", "--max-lora-rank", str(max_lora_rank(adapters))]
        command += ["--lora-modules"]
        command += [f"{name}={path}" for name, path in adapters.items()]
    if args.gpu_memory_utilization is not None:
        command += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    if args.enforce_eager:
        command.append("--enforce-eager")
    return command


def main() -> None:
    args = parse_args()
    adapters = parse_adapters(args.adapter, args.model)
    # 네트워크 없이 캐시에서만 찾는다. 없으면 받지 않고 예외를 낸다
    snapshot = Path(snapshot_download(args.model, local_files_only=True))

    logger.info("베이스 %s → %s", args.model, snapshot)
    for name, path in adapters.items():
        logger.info("어댑터 %s → %s", name, path)

    # 이 스크립트를 실행한 파이썬과 같은 venv의 vllm을 쓴다 (PATH에 없어도 된다)
    venv_bin = Path(sys.executable).parent
    vllm = venv_bin / "vllm"
    if not vllm.exists():
        raise FileNotFoundError(
            f"{vllm}이 없습니다. vLLM venv의 파이썬으로 실행하세요."
        )

    env = os.environ.copy()
    for key, value in DEFAULT_ENV.items():
        env.setdefault(key, value)
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

    api_key = load_api_key(args.api_key_file)
    command = build_command(args, snapshot, adapters, api_key)
    logger.info("환경 %s", {key: env[key] for key in DEFAULT_ENV})
    logger.info("실행 %s", " ".join(command).replace(api_key, "***"))

    # 이 프로세스를 vLLM으로 바꾼다. Ctrl+C가 vLLM에 바로 가고 남는 프로세스가 없다
    os.execve(vllm, command, env)


if __name__ == "__main__":
    main()
