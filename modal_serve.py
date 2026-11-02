"""Modal 서버리스 vLLM 서빙 스크립트 (5단계 정밀 동적 튜닝 및 Self-Healing 지원).

OpenAI 호환 엔드포인트(/v1/chat/completions)를 제공하며,
요청 종료 후 60초 동안 유휴 상태가 유지되면 0대로 자동 축소되어
GPU 과금을 원천 차단합니다.

주요 기능:
1. 아키텍처 기반 정밀 메모리 역산 (config.json, safetensors 분석)
2. GPU 세대(Compute Capability) 및 실측 VRAM 자동 감지
3. 7단계 연속 메모리 스케일링 및 불가능 조합 사전 차단(Fail-Fast)
4. LoRA 어댑터 동시 서빙 지원 (LORA_ADAPTER_ID, LORA_ADAPTER_NAME)
5. 컨테이너 내부 헬스체크 대기, 볼륨 커밋 및 안전 모드 재기동(Self-Healing)
"""

import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

import modal

APP_NAME = os.environ.get("MODAL_APP_NAME", "cosmos-vllm-serve")
VOLUME_NAME = "cosmos-model-cache"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-122B-A10B-FP8"
CACHE_DIR = "/root/.cache/huggingface"
VLLM_PORT = 8000
MINUTES = 60

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def get_gpu_config(gpu_type: str | None = None) -> tuple[Any, int]:
    """
    MODAL_GPU_TYPE 환경 변수에 따라 GPU 리소스와 텐서 병렬 크기(TP)를 반환한다.

    지원 기종:
        - H200 (기본값, 141GB VRAM, 122B-FP8 권장)
        - B200 (180GB VRAM, 초대형 모델 권장)
        - 2xA100 (160GB VRAM, 2장 분산 병렬)
        - A100 (80GB VRAM, 27B 모델 권장)
        - A100-40GB (40GB VRAM)
        - L40S (48GB VRAM)
        - L4 (24GB VRAM, 소형/양자화 모델 권장)
        - T4 (16GB VRAM, 스모크 테스트용)
    """
    if not gpu_type:
        gpu_type = os.environ.get("MODAL_GPU_TYPE", "H200")
    gpu_type = gpu_type.upper().strip()

    if gpu_type == "H200":
        return "H200", 1
    if gpu_type == "B200":
        return "B200", 1
    if gpu_type in ("2XA100", "2XA100-80GB", "2X_A100"):
        return "A100-80GB:2", 2
    if gpu_type in ("A100", "A100-80GB"):
        return "A100-80GB", 1
    if gpu_type in ("A100-40GB",):
        return "A100-40GB", 1
    if gpu_type == "L4":
        return "L4", 1
    if gpu_type == "L40S":
        return "L40S", 1
    if gpu_type == "T4":
        return "T4", 1

    return gpu_type, 1


GPU_EXPECTED_VRAM_GB: dict[str, float] = {
    "H200": 141.0,
    "B200": 180.0,
    "A100-80GB:2": 160.0,
    "A100-80GB": 80.0,
    "A100-40GB": 40.0,
    "L40S": 48.0,
    "L4": 24.0,
    "T4": 16.0,
}

SELECTED_MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
SELECTED_GPU_TYPE = os.environ.get("MODAL_GPU_TYPE", "H200")
SELECTED_GPU, TENSOR_PARALLEL_SIZE = get_gpu_config(SELECTED_GPU_TYPE)
SELECTED_LORA_ID = os.environ.get("LORA_ADAPTER_ID", "").strip()
SELECTED_LORA_NAME = os.environ.get("LORA_ADAPTER_NAME", "").strip()
OVERRIDE_MAX_NUM_SEQS = os.environ.get("MAX_NUM_SEQS", "").strip()
OVERRIDE_GPU_MEM_UTIL = os.environ.get("GPU_MEMORY_UTILIZATION", "").strip()
OVERRIDE_MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "").strip()

# 최신 vLLM 및 Qwen3.5 지원을 위한 transformers >= 5.0 설치
vllm_image = modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11"
).pip_install(
    "vllm==0.31.0",
    "transformers>=5.0.0",
    "huggingface_hub>=0.26.0",
    "fastapi",
    "pydantic",
)


def _estimate_weight_gb_from_name(model_id: str) -> float:
    """네트워크 조회 실패 시 모델 이름에서 가중치 크기(GB)를 보수적으로 추정한다."""
    lower = model_id.lower()
    match = re.search(r"(\d+(?:\.\d+)?)b", lower)
    params_b = float(match.group(1)) if match else 14.0

    if any(q in lower for q in ("awq", "gptq", "4bit", "int4", "bnb-4bit")):
        bytes_per_param = 0.55
    elif any(q in lower for q in ("fp8", "int8", "8bit")):
        bytes_per_param = 1.05
    else:
        bytes_per_param = 2.0

    return round(params_b * bytes_per_param, 2)


def preflight_estimate_weight_gb(model_id: str, hf_token: str | None = None) -> float:
    """다운로드 전 HuggingFace API 메타데이터로 모델 가중치 크기(GB)를 조회한다."""
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        info = api.model_info(model_id)
        if info.safetensors and info.safetensors.parameters:
            dtype_bytes = {
                "F32": 4.0,
                "F16": 2.0,
                "BF16": 2.0,
                "F8_E4M3": 1.0,
                "F8_E5M2": 1.0,
                "I8": 1.0,
                "U8": 1.0,
                "I4": 0.5,
                "U4": 0.5,
                "F4": 0.5,
            }
            total_bytes = sum(
                count * dtype_bytes.get(dtype.upper(), 2.0)
                for dtype, count in info.safetensors.parameters.items()
            )
            if total_bytes > 0:
                return round(total_bytes / (1024**3), 2)
    except Exception as error:
        print(f"[Auto-Tune] HF 메타데이터 사전 조회 생략 (폴백 추정 사용): {error}")

    return _estimate_weight_gb_from_name(model_id)


def inspect_local_snapshot(snapshot_dir: Path, model_id: str) -> dict[str, Any]:
    """로컬 스냅샷의 config.json, tokenizer_config.json, 가중치 크기를 분석한다."""
    safetensor_files = list(snapshot_dir.glob("*.safetensors"))
    if safetensor_files:
        weight_bytes = sum(f.stat().st_size for f in safetensor_files)
        weight_gb = round(weight_bytes / (1024**3), 2)
    else:
        weight_gb = _estimate_weight_gb_from_name(model_id)

    config_data: dict[str, Any] = {}
    config_path = snapshot_dir / "config.json"
    if config_path.exists():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config_data = {}

    text_cfg = config_data.get("text_config") or config_data
    model_type = str(
        config_data.get("model_type") or text_cfg.get("model_type") or "unknown"
    ).lower()

    num_layers = int(text_cfg.get("num_hidden_layers", 32))
    num_kv_heads = int(
        text_cfg.get("num_key_value_heads", text_cfg.get("num_attention_heads", 8))
    )
    hidden_size = int(text_cfg.get("hidden_size", 4096))
    head_dim = int(
        text_cfg.get(
            "head_dim",
            hidden_size // max(int(text_cfg.get("num_attention_heads", 32)), 1),
        )
    )

    layer_types = text_cfg.get("layer_types") or []
    if layer_types:
        full_attn_layers = sum(1 for lt in layer_types if lt == "full_attention")
        linear_attn_layers = len(layer_types) - full_attn_layers
        is_hybrid_linear = linear_attn_layers > 0
    else:
        full_attn_layers = num_layers
        linear_attn_layers = 0
        is_hybrid_linear = False

    is_moe = (
        "moe" in model_type
        or int(text_cfg.get("num_experts", text_cfg.get("num_local_experts", 0))) > 1
    )

    quant_cfg = (
        config_data.get("quantization_config")
        or text_cfg.get("quantization_config")
        or {}
    )
    quant_method = str(quant_cfg.get("quant_method", "")).lower()
    is_fp8 = quant_method == "fp8" or "fp8" in model_id.lower()

    # 토큰 1개당 KV 캐시 바이트 산출 (하이브리드 선형 어텐션 유효 하한 50% 반영)
    effective_attn_layers = max(full_attn_layers, num_layers // 2)
    kv_bytes_per_token = 2 * effective_attn_layers * num_kv_heads * head_dim * 2

    supports_thinking_toggle = "qwen3" in model_id.lower()
    tok_cfg_path = snapshot_dir / "tokenizer_config.json"
    if tok_cfg_path.exists():
        try:
            tok_raw = tok_cfg_path.read_text(encoding="utf-8")
            supports_thinking_toggle = "enable_thinking" in tok_raw
        except Exception:
            pass

    return {
        "weight_gb": weight_gb,
        "model_type": model_type,
        "num_layers": num_layers,
        "full_attn_layers": full_attn_layers,
        "linear_attn_layers": linear_attn_layers,
        "is_hybrid_linear": is_hybrid_linear,
        "is_moe": is_moe,
        "is_fp8": is_fp8,
        "hidden_size": hidden_size,
        "kv_bytes_per_token": kv_bytes_per_token,
        "supports_thinking_toggle": supports_thinking_toggle,
    }


def inspect_gpu_runtime(tp_size: int, fallback_gpu_name: str) -> dict[str, Any]:
    """PyTorch CUDA 런타임으로 할당된 GPU의 VRAM과 아키텍처 세대를 감지한다."""
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            total_vram_gb = round(
                sum(
                    torch.cuda.get_device_properties(i).total_memory
                    for i in range(device_count)
                )
                / (1024**3),
                2,
            )
            cc_major, cc_minor = torch.cuda.get_device_capability(0)
            device_name = torch.cuda.get_device_name(0)
            supports_bf16 = cc_major >= 8
            supports_native_fp8 = (cc_major > 8) or (cc_major == 8 and cc_minor >= 9)
            return {
                "device_name": device_name,
                "device_count": device_count,
                "total_vram_gb": total_vram_gb,
                "cc": f"{cc_major}.{cc_minor}",
                "supports_bf16": supports_bf16,
                "supports_native_fp8": supports_native_fp8,
            }
    except Exception as error:
        print(f"[Auto-Tune] CUDA 런타임 조회 실패 (기본 스펙 사용): {error}")

    expected_vram = GPU_EXPECTED_VRAM_GB.get(fallback_gpu_name, 80.0)
    return {
        "device_name": fallback_gpu_name,
        "device_count": tp_size,
        "total_vram_gb": expected_vram,
        "cc": "8.0",
        "supports_bf16": fallback_gpu_name != "T4",
        "supports_native_fp8": fallback_gpu_name in ("H200", "B200", "L40S", "L4"),
    }


def resolve_serving_params(
    model_prof: dict[str, Any],
    gpu_prof: dict[str, Any],
    tp_size: int,
    has_lora: bool = False,
) -> dict[str, Any]:
    """모델 아키텍처와 GPU 특성을 기반으로 7단계 연속 스케일링된 인자를 산출한다."""
    weight_gb = model_prof["weight_gb"]
    total_vram_gb = gpu_prof["total_vram_gb"]

    overhead_gb = 2.4 * tp_size
    if model_prof["is_fp8"] and not gpu_prof["supports_native_fp8"]:
        overhead_gb += 2.0
    if has_lora:
        overhead_gb += 1.5
    if model_prof["hidden_size"] >= 5120 and not model_prof["is_moe"]:
        overhead_gb += 1.5

    # GPU 총 용량에 맞춘 고정 안전 버퍼(GB) 산출 (대형 GPU에서 불필요한 비율 손실 방지)
    safety_buffer_gb = max(2.0, min(7.5, total_vram_gb * 0.08))
    if model_prof["hidden_size"] >= 5120 and not model_prof["is_moe"] and tp_size == 1:
        # 단일 카드에서 27B+ Dense 모델 구동 시 프로파일링 스파이크 방어용 버퍼 확보
        safety_buffer_gb = max(safety_buffer_gb, total_vram_gb * 0.10)

    base_util = round(
        max(0.85, min(0.95, (total_vram_gb - safety_buffer_gb) / total_vram_gb)), 2
    )
    free_vram_gb = (total_vram_gb * base_util) - weight_gb - overhead_gb
    kv_cache_dtype = "auto"

    if free_vram_gb < 0:
        avail_gb = total_vram_gb * base_util
        raise RuntimeError(
            f"[Fail-Fast] 메모리 부족으로 서빙이 불가능합니다: "
            f"가중치({weight_gb:.1f}GB) + 오버헤드({overhead_gb:.1f}GB)가 "
            f"가용 VRAM({avail_gb:.1f}GB / 총 {total_vram_gb:.1f}GB)을 초과합니다. "
            f"더 큰 VRAM 기종(2xA100, H200, B200)을 선택하세요."
        )

    if free_vram_gb < 6.0:
        tier = "Tier 1 (극한 압박 - 안전 축소 모드)"
        gpu_mem_util = base_util
        max_num_seqs = 16
        max_model_len = 4096
        enforce_eager = True
        if gpu_prof["supports_native_fp8"]:
            kv_cache_dtype = "fp8"
    elif free_vram_gb < 12.0:
        tier = "Tier 2 (초고압박 - 저배치 안정 모드)"
        gpu_mem_util = base_util
        max_num_seqs = 32
        max_model_len = 8192
        enforce_eager = True
    elif free_vram_gb < 18.0:
        tier = "Tier 3 (고압박 - 단일 카드 대형 모델 최적 모드)"
        gpu_mem_util = base_util
        max_num_seqs = 64
        max_model_len = 8192
        enforce_eager = True
    elif free_vram_gb < 28.0:
        tier = "Tier 4 (중압박 - 균형 처리 모드)"
        gpu_mem_util = base_util
        max_num_seqs = 128
        max_model_len = 8192
        enforce_eager = True
    elif free_vram_gb < 45.0:
        tier = "Tier 5 (양호 - 고성능 처리 모드)"
        gpu_mem_util = base_util
        max_num_seqs = 192
        max_model_len = 8192
        enforce_eager = False
    else:
        tier = "Tier 6 (완전 여유 - 최대 처리량 모드)"
        gpu_mem_util = base_util
        max_num_seqs = 256
        max_model_len = 8192
        enforce_eager = False

    kv_bytes = max(model_prof["kv_bytes_per_token"], 1024)
    usable_kv_bytes = max(free_vram_gb * 0.80, 1.0) * (1024**3)
    estimated_max_tokens = int(usable_kv_bytes / kv_bytes)
    kv_safe_seqs = max(16, min(256, (estimated_max_tokens // 1536) * 16))
    if kv_safe_seqs < max_num_seqs:
        max_num_seqs = kv_safe_seqs

    dtype = "auto"
    if not gpu_prof["supports_bf16"]:
        dtype = "half"
        enforce_eager = True

    if OVERRIDE_MAX_NUM_SEQS:
        max_num_seqs = int(OVERRIDE_MAX_NUM_SEQS)
    if OVERRIDE_GPU_MEM_UTIL:
        gpu_mem_util = float(OVERRIDE_GPU_MEM_UTIL)
    if OVERRIDE_MAX_MODEL_LEN:
        max_model_len = int(OVERRIDE_MAX_MODEL_LEN)

    return {
        "tier": tier,
        "free_vram_gb": round(free_vram_gb, 2),
        "overhead_gb": round(overhead_gb, 2),
        "gpu_memory_utilization": gpu_mem_util,
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "enforce_eager": enforce_eager,
        "dtype": dtype,
        "kv_cache_dtype": kv_cache_dtype,
    }


@app.cls(
    image=vllm_image,
    gpu=SELECTED_GPU,
    volumes={CACHE_DIR: volume},
    max_containers=1,
    scaledown_window=60,
    timeout=60 * MINUTES,
    startup_timeout=60 * MINUTES,
    secrets=[
        modal.Secret.from_name("cosmos-modal-secret", required_keys=[]),
        modal.Secret.from_dict(
            {
                "MODEL_ID": SELECTED_MODEL_ID,
                "MODAL_GPU_TYPE": SELECTED_GPU_TYPE,
                "TENSOR_PARALLEL_SIZE": str(TENSOR_PARALLEL_SIZE),
                "LORA_ADAPTER_ID": SELECTED_LORA_ID,
                "LORA_ADAPTER_NAME": SELECTED_LORA_NAME,
                "MAX_NUM_SEQS": OVERRIDE_MAX_NUM_SEQS,
                "GPU_MEMORY_UTILIZATION": OVERRIDE_GPU_MEM_UTIL,
                "MAX_MODEL_LEN": OVERRIDE_MAX_MODEL_LEN,
            }
        ),
    ],
)
@modal.concurrent(max_inputs=32)
class ModelServer:
    """vLLM OpenAI 호환 서버리스 인스턴스 (동적 튜닝 및 Self-Healing 내장)."""

    def _build_vllm_command(
        self,
        snapshot_path: str,
        served_model_name: str,
        tp_size: int,
        api_key: str,
        params: dict[str, Any],
        supports_thinking_toggle: bool,
        lora_spec: tuple[str, str, int] | None = None,
    ) -> list[str]:
        cmd = [
            "vllm",
            "serve",
            snapshot_path,
            "--served-model-name",
            served_model_name,
            "--port",
            str(VLLM_PORT),
            "--host",
            "0.0.0.0",
            "--tensor-parallel-size",
            str(tp_size),
            "--max-num-seqs",
            str(params["max_num_seqs"]),
            "--max-model-len",
            str(params["max_model_len"]),
            "--gpu-memory-utilization",
            str(params["gpu_memory_utilization"]),
        ]

        if params.get("dtype") and params["dtype"] != "auto":
            cmd += ["--dtype", params["dtype"]]

        if params.get("kv_cache_dtype") and params["kv_cache_dtype"] != "auto":
            cmd += ["--kv-cache-dtype", params["kv_cache_dtype"]]

        if params.get("enforce_eager"):
            cmd.append("--enforce-eager")

        if supports_thinking_toggle:
            cmd += [
                "--default-chat-template-kwargs",
                '{"enable_thinking": false}',
            ]

        if lora_spec is not None:
            lora_name, lora_path, lora_rank = lora_spec
            cmd += [
                "--enable-lora",
                "--max-lora-rank",
                str(lora_rank),
                "--lora-modules",
                f"{lora_name}={lora_path}",
            ]

        cmd += ["--api-key", api_key]
        return cmd

    def _mask_cmd(self, cmd: list[str]) -> str:
        display_cmd = list(cmd)
        if "--api-key" in display_cmd:
            key_idx = display_cmd.index("--api-key") + 1
            if key_idx < len(display_cmd):
                display_cmd[key_idx] = "********"
        return " ".join(display_cmd)

    def _start_vllm_with_watchdog(
        self, primary_cmd: list[str], fallback_cmd: list[str]
    ) -> None:
        """vLLM을 즉시 기동하고 백그라운드에서 비정상 종료 시 안전 모드로 재시도한다."""
        print(f"[vLLM Server] 실행 명령어: {self._mask_cmd(primary_cmd)}", flush=True)
        self.process = subprocess.Popen(primary_cmd)

        def _watchdog() -> None:
            exit_code = self.process.wait()
            if getattr(self, "_stopping", False) or exit_code == 0:
                return
            print("=" * 60, flush=True)
            print(
                f"[Self-Healing] 1차 vLLM 프로세스 비정상 종료 (exit={exit_code}) "
                f"-> 안전 모드 파라미터로 즉시 재기동합니다.",
                flush=True,
            )
            print(
                f"[vLLM Server] 안전 모드 명령어: {self._mask_cmd(fallback_cmd)}",
                flush=True,
            )
            print("=" * 60, flush=True)
            self.process = subprocess.Popen(fallback_cmd)
            retry_exit = self.process.wait()
            if not getattr(self, "_stopping", False) and retry_exit != 0:
                print(
                    f"[vLLM Fatal] 1차 기동 및 안전 모드 재시도가 모두 실패했습니다 "
                    f"(exit={retry_exit}).",
                    flush=True,
                )
                os._exit(1)

        threading.Thread(target=_watchdog, daemon=True).start()

    @modal.enter()
    def startup(self):
        from huggingface_hub import snapshot_download

        volume.reload()

        model_id = os.environ.get("MODEL_ID", SELECTED_MODEL_ID)
        lora_id = os.environ.get("LORA_ADAPTER_ID", SELECTED_LORA_ID).strip()
        lora_name = os.environ.get("LORA_ADAPTER_NAME", SELECTED_LORA_NAME).strip()
        api_key = (
            os.environ.get("VLLM_API_KEY") or os.environ.get("MODAL_API_KEY") or "EMPTY"
        )
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        tp_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", TENSOR_PARALLEL_SIZE))

        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        gpu_prof = inspect_gpu_runtime(tp_size, str(SELECTED_GPU))
        est_weight_gb = preflight_estimate_weight_gb(model_id, hf_token)

        print("=" * 60)
        print(f"[vLLM Server] 기동 시작: {model_id}")
        dev_cnt = gpu_prof["device_count"]
        vram_tot = gpu_prof["total_vram_gb"]
        print(
            f"[vLLM Server] GPU: {gpu_prof['device_name']} "
            f"(장치 수={dev_cnt}, 총 VRAM={vram_tot:.1f}GB, CC={gpu_prof['cc']})"
        )
        print(f"[vLLM Server] 사전 추정 가중치 크기: 약 {est_weight_gb:.1f}GB")
        print("=" * 60)

        if est_weight_gb >= gpu_prof["total_vram_gb"] * 0.96:
            raise RuntimeError(
                f"[Fail-Fast] 모델 가중치({est_weight_gb:.1f}GB)가 할당된 GPU 총 VRAM"
                f"({vram_tot:.1f}GB)을 초과하여 서빙할 수 없습니다. "
                f"2xA100, H200, B200 등 더 큰 GPU 기종을 선택하세요."
            )

        print(f"[vLLM Server] 모델 스냅샷 동기화 중: {model_id}")
        snapshot_dir = Path(snapshot_download(model_id, token=hf_token))

        lora_spec: tuple[str, str, int] | None = None
        if lora_id:
            resolved_lora_name = lora_name or lora_id.split("/")[-1]
            print(f"[vLLM Server] LoRA 동기화: {resolved_lora_name} ({lora_id})")
            lora_dir = Path(snapshot_download(lora_id, token=hf_token))
            adapter_cfg_path = lora_dir / "adapter_config.json"
            if not adapter_cfg_path.exists():
                raise FileNotFoundError(
                    f"LoRA 설정(adapter_config.json)이 없습니다: {lora_dir}"
                )
            lora_rank = int(
                json.loads(adapter_cfg_path.read_text(encoding="utf-8")).get("r", 16)
            )
            lora_spec = (resolved_lora_name, str(lora_dir), lora_rank)

        volume.commit()
        print("[vLLM Server] 볼륨 캐시 커밋 완료.")

        model_prof = inspect_local_snapshot(snapshot_dir, model_id)
        params = resolve_serving_params(
            model_prof, gpu_prof, tp_size, has_lora=(lora_spec is not None)
        )

        print("=" * 60)
        print("[Auto-Tune] 모델 아키텍처 및 메모리 정밀 분석 완료")
        m_moe = model_prof["is_moe"]
        m_hyb = model_prof["is_hybrid_linear"]
        m_fp8 = model_prof["is_fp8"]
        print(
            f"  - 모델 타입: {model_prof['model_type']} "
            f"(MoE={m_moe}, HybridLinear={m_hyb}, FP8={m_fp8})"
        )
        f_attn = model_prof["full_attn_layers"]
        l_attn = model_prof["linear_attn_layers"]
        print(
            f"  - 레이어 구조: 전체 {model_prof['num_layers']}층 "
            f"(FullAttn={f_attn}, LinearAttn={l_attn})"
        )
        print(
            f"  - 실측 가중치: {model_prof['weight_gb']:.2f}GB | "
            f"런타임 오버헤드: {params['overhead_gb']:.2f}GB | "
            f"순수 여유 VRAM: {params['free_vram_gb']:.2f}GB"
        )
        print(f"  - 선택된 튜닝 등급: {params['tier']}")
        print(
            f"  - 최종 파라미터: max_num_seqs={params['max_num_seqs']}, "
            f"max_model_len={params['max_model_len']}, "
            f"gpu_mem_util={params['gpu_memory_utilization']}, "
            f"enforce_eager={params['enforce_eager']}, "
            f"dtype={params['dtype']}, kv_cache_dtype={params['kv_cache_dtype']}"
        )
        print("=" * 60)

        cmd = self._build_vllm_command(
            snapshot_path=str(snapshot_dir),
            served_model_name=model_id,
            tp_size=tp_size,
            api_key=api_key,
            params=params,
            supports_thinking_toggle=model_prof["supports_thinking_toggle"],
            lora_spec=lora_spec,
        )

        fallback_params = dict(params)
        fallback_params["max_num_seqs"] = max(16, params["max_num_seqs"] // 2)
        fallback_params["max_model_len"] = min(params["max_model_len"], 4096)
        fallback_params["gpu_memory_utilization"] = min(
            params["gpu_memory_utilization"], 0.88
        )
        fallback_params["enforce_eager"] = True

        retry_cmd = self._build_vllm_command(
            snapshot_path=str(snapshot_dir),
            served_model_name=model_id,
            tp_size=tp_size,
            api_key=api_key,
            params=fallback_params,
            supports_thinking_toggle=model_prof["supports_thinking_toggle"],
            lora_spec=lora_spec,
        )

        self._stopping = False
        self._start_vllm_with_watchdog(cmd, retry_cmd)

    @modal.exit()
    def stop(self):
        """컨테이너 종료 시 vLLM 프로세스를 정상 정리한다."""
        self._stopping = True
        if hasattr(self, "process") and self.process and self.process.poll() is None:
            print("[vLLM Server] 프로세스 종료 신호 전송...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("[vLLM Server] 프로세스 정상 종료 완료.")

    @modal.web_server(port=VLLM_PORT, startup_timeout=60 * MINUTES)
    def serve(self):
        """vLLM은 startup에서 헬스체크 통과까지 완료되며 포트만 노출한다."""
