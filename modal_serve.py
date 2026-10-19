"""Modal 서버리스 vLLM 서빙 스크립트.

OpenAI 호환 엔드포인트(/v1/chat/completions)를 제공하며,
요청 종료 후 60초 동안 유휴 상태가 유지되면 0대로 자동 축소되어
GPU 과금을 원천 차단합니다.

사용법:
    # 1. 개발 및 디버깅용 임시 서빙 (코드가 변경되면 자동 핫 리로드)
    modal serve modal_serve.py

    # 2. 운영 환경 배포 (고정 HTTPS 웹 엔드포인트 발행)
    modal deploy modal_serve.py

    # 3. 모델 ID 변경 배포 (예: 파인튜닝 병합 모델)
    MODEL_ID="your-org/cosmos-problemgen-merged" modal deploy modal_serve.py
"""

import os
import subprocess

import modal

APP_NAME = "cosmos-vllm-serve"
VOLUME_NAME = "cosmos-model-cache"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-122B-A10B-FP8"
CACHE_DIR = "/root/.cache/huggingface"
VLLM_PORT = 8000
MINUTES = 60

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def get_gpu_config() -> tuple[any, int]:
    """
    MODAL_GPU_TYPE 환경 변수에 따라 적절한 GPU 리소스와 텐서 병렬 크기(TP)를 반환한다.

    지원 기종:
        - H200 (기본값, 141GB VRAM, 122B-FP8 권장)
        - B200 (180GB VRAM, 초대형 모델 권장)
        - 2xA100 (160GB VRAM, 2장 분산 병렬)
        - A100 (80GB VRAM, 27B 병합 모델 권장)
        - L4 (24GB VRAM, W4A16 양자화 모델 권장)
        - L40S (48GB VRAM)
    """
    gpu_type = os.environ.get("MODAL_GPU_TYPE", "H200").upper().strip()

    if gpu_type == "H200":
        return modal.gpu.H200(), 1
    if gpu_type == "B200":
        return modal.gpu.B200(), 1
    if gpu_type in ("2XA100", "2XA100-80GB", "2X_A100"):
        return modal.gpu.A100(count=2, size="80GB"), 2
    if gpu_type in ("A100", "A100-80GB"):
        return modal.gpu.A100(size="80GB"), 1
    if gpu_type in ("A100-40GB",):
        return modal.gpu.A100(size="40GB"), 1
    if gpu_type == "L4":
        return modal.gpu.L4(), 1
    if gpu_type == "L40S":
        return modal.gpu.L40S(), 1

    # 기타 사용자 정의 기종
    return gpu_type, 1


SELECTED_GPU, TENSOR_PARALLEL_SIZE = get_gpu_config()

# 최신 vLLM 및 Qwen3.5 지원을 위한 transformers >= 5.0 설치
vllm_image = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11"
).pip_install(
    "vllm>=0.7.0",
    "transformers>=5.0.0",
    "huggingface_hub>=0.26.0",
    "fastapi",
    "pydantic",
)


@app.cls(
    image=vllm_image,
    # 환경 변수 MODAL_GPU_TYPE으로 동적 선택 (기본값: H200 141GB)
    gpu=SELECTED_GPU,
    volumes={CACHE_DIR: volume},
    # 60초 유휴 시 컨테이너를 0대로 자동 축소하여 유휴 비용 방어
    scaledown_window=60,
    timeout=10 * MINUTES,
    secrets=[
        # MODAL_API_KEY (또는 VLLM_API_KEY) 및 HF_TOKEN 주입
        modal.Secret.from_name("cosmos-modal-secret", required_keys=[]),
    ],
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
class ModelServer:
    """vLLM OpenAI 호환 서버리스 인스턴스."""

    @modal.enter()
    def startup(self):
        # 최신 캐싱된 가중치를 컨테이너 파일시스템에 즉시 동기화
        volume.reload()

        model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
        api_key = (
            os.environ.get("VLLM_API_KEY") or os.environ.get("MODAL_API_KEY") or "EMPTY"
        )
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )

        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        print("=" * 60)
        print(f"[vLLM Server] 기동 시작: {model_id}")
        print(f"[vLLM Server] 마운트 볼륨: {VOLUME_NAME} -> {CACHE_DIR}")
        auth_desc = "설정됨" if api_key != "EMPTY" else "비활성화(EMPTY)"
        print(f"[vLLM Server] API 인증: {auth_desc}")
        print("=" * 60)

        cmd = [
            "vllm",
            "serve",
            model_id,
            "--port",
            str(VLLM_PORT),
            "--host",
            "0.0.0.0",
            "--tensor-parallel-size",
            str(TENSOR_PARALLEL_SIZE),
            "--max-model-len",
            "8192",
            # Qwen3.5 계열 추론 토큰(<think>) 노출을 서버 레벨에서 원천 차단
            "--default-chat-template-kwargs",
            '{"enable_thinking": false}',
            "--api-key",
            api_key,
            "--disable-log-requests",
        ]

        print(f"[vLLM Server] 실행 명령어: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        """컨테이너 종료 시 vLLM 프로세스를 정상 정리한다."""
        if hasattr(self, "process") and self.process:
            print("[vLLM Server] 프로세스 종료 신호 전송...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("[vLLM Server] 프로세스 정상 종료 완료.")
