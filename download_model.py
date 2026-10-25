"""Modal Volume에 Hugging Face 대용량 모델 가중치를 사전 다운로드(캐싱)하는 스크립트.

사용법:
    # 기본 모델 (Qwen/Qwen3.5-122B-A10B-FP8) 사전 캐싱
    modal run download_model.py

    # 특정 모델 지정 캐싱 (예: 파인튜닝 병합 모델)
    modal run download_model.py --model-id "your-org/cosmos-problemgen-qwen3-merged"
"""

import os
import time

import modal

APP_NAME = "cosmos-model-download"
VOLUME_NAME = "cosmos-model-cache"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-122B-A10B-FP8"
CACHE_DIR = "/root/.cache/huggingface"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# hf_transfer를 활성화하여 고속 대역폭 다운로드 지원
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub[hf_transfer]>=0.26.0",
        "humanize>=4.9.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=download_image,
    volumes={CACHE_DIR: volume},
    timeout=7200,  # 120GB 대용량 다운로드를 위한 타임아웃 2시간
    secrets=[
        # Hugging Face 비공개 저장소 접근 및 다운로드 속도 제한 방지용 토큰
        modal.Secret.from_name("cosmos-modal-secret", required_keys=[]),
    ],
)
def download_model(model_id: str = DEFAULT_MODEL_ID) -> dict:
    """지정된 Hugging Face 모델을 Modal Volume에 다운로드하고 커밋한다."""
    import humanize
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("=" * 60)
    print(f"[Modal Cache] 모델 다운로드 시작: {model_id}")
    print(f"[Modal Cache] 대상 볼륨: {VOLUME_NAME} -> {CACHE_DIR}")
    print("=" * 60)

    start_time = time.time()
    download_path = snapshot_download(
        repo_id=model_id,
        token=hf_token,
        max_workers=8,
    )
    elapsed_time = time.time() - start_time

    # 다운로드된 파일 총 크기 계산
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(download_path)
        for filename in filenames
    )
    readable_size = humanize.naturalsize(total_size, binary=True)

    print("=" * 60)
    print(f"[Modal Cache] 다운로드 완료 경로: {download_path}")
    print(f"[Modal Cache] 총 용량: {readable_size} ({total_size:,} bytes)")
    print(f"[Modal Cache] 소요 시간: {elapsed_time:.1f}초")
    print("[Modal Cache] Modal Volume 변경 사항 커밋 중...")
    print("=" * 60)

    volume.commit()
    print(
        "[Modal Cache] 볼륨 커밋 완료. modal_serve.py에서 1분 이내로 "
        "즉시 마운트 가능합니다."
    )

    return {
        "model_id": model_id,
        "download_path": download_path,
        "total_bytes": total_size,
        "readable_size": readable_size,
        "elapsed_seconds": round(elapsed_time, 2),
    }


@app.local_entrypoint()
def main(model_id: str = DEFAULT_MODEL_ID):
    """CLI 진입점."""
    result = download_model.remote(model_id=model_id)
    print(f"\n[성공] 모델 캐싱 결과 요약: {result}")
