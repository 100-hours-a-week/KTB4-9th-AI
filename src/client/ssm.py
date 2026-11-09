"""AWS SSM Parameter Store 연동 모듈.

승인된 운영 모델 식별자(/cosmos/prod/model_version)를 조회하며,
배치 실행 중 빈번한 AWS API 호출을 방지하기 위해 5분 단위 메모리 TTL 캐시를 적용한다.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

__all__ = ["get_active_model_version", "clear_ssm_cache"]

# 인메모리 TTL 캐시 (5분)
_CACHE_TTL_SECONDS = 300.0
_cached_model_version: str | None = None
_cached_at: float = 0.0


def clear_ssm_cache() -> None:
    """메모리 캐시를 초기화한다 (테스트 및 즉시 갱신용)."""
    global _cached_model_version, _cached_at
    _cached_model_version = None
    _cached_at = 0.0


def get_active_model_version(
    default_model: str | None = None,
    force_refresh: bool = False,
) -> str | None:
    """
    AWS SSM Parameter Store에서 운영 승인된 모델 버전을 조회한다.

    Parameters:
        default_model (str | None): SSM 조회 실패 시 반환할 기본 모델명
            (예: 'Qwen/Qwen3.5-122B-A10B-FP8').
        force_refresh (bool): True이면 캐시를 무시하고 SSM을 직접 조회한다.

    Returns:
        str | None: 승인된 모델 식별자 문자열. 조회 실패 시 default_model 반환.
    """
    global _cached_model_version, _cached_at

    now = time.time()
    if (
        not force_refresh
        and _cached_model_version
        and (now - _cached_at < _CACHE_TTL_SECONDS)
    ):
        return _cached_model_version

    aws_region = os.getenv("AWS_REGION", "ap-northeast-2")
    default_param = (
        "/cosmos/dev/model_version"
        if aws_region == "us-west-2"
        else "/cosmos/prod/model_version"
    )
    param_name = os.getenv("SSM_MODEL_VERSION_PARAM", default_param)

    try:
        from src.core.config import get_settings

        settings = get_settings()
        if getattr(settings, "ssm_model_version_param", None):
            param_name = settings.ssm_model_version_param
        if getattr(settings, "aws_region", None):
            aws_region = settings.aws_region
    except Exception:
        pass

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError

        ssm_client = boto3.client("ssm", region_name=aws_region)
        response = ssm_client.get_parameter(Name=param_name, WithDecryption=False)
        val = response.get("Parameter", {}).get("Value")
        if val:
            val_clean = val.strip()
            if val_clean.upper() == "NONE":
                logger.info(
                    "AWS SSM Parameter Store 모델 버전이 NONE으로 설정되어 있습니다.",
                )
                return None
            _cached_model_version = val_clean
            _cached_at = now
            logger.info(
                "AWS SSM Parameter Store에서 운영 모델 버전 조회 성공: %s",
                _cached_model_version,
            )
            return _cached_model_version
    except (BotoCoreError, ClientError) as error:
        logger.warning(
            "AWS SSM Parameter Store 조회 실패 (파라미터: %s): %s. "
            "기본 설정을 사용합니다.",
            param_name,
            error,
        )
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "AWS SSM 클라이언트 초기화 실패 (파라미터: %s): %s",
            param_name,
            error,
        )

    return default_model
