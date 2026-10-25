from types import SimpleNamespace

from src.client.ssm import clear_ssm_cache, get_active_model_version


def test_캐시가_유효하면_SSM_API를_다시_부르지_않는다(monkeypatch):
    clear_ssm_cache()
    api_calls = 0

    class FakeSSMClient:
        def get_parameter(self, Name, WithDecryption):
            nonlocal api_calls
            api_calls += 1
            return {"Parameter": {"Value": "Qwen/Qwen3.5-122B-A10B-FP8"}}

    fake_boto3 = SimpleNamespace(client=lambda service, region_name: FakeSSMClient())
    monkeypatch.setattr("boto3.client", fake_boto3.client)

    v1 = get_active_model_version()
    v2 = get_active_model_version()

    assert v1 == "Qwen/Qwen3.5-122B-A10B-FP8"
    assert v2 == "Qwen/Qwen3.5-122B-A10B-FP8"
    assert api_calls == 1, "5분 캐시가 유효하면 추가 AWS API 호출이 발생하지 않아야 함"


def test_force_refresh면_캐시를_무시하고_다시_조회한다(monkeypatch):
    clear_ssm_cache()
    api_calls = 0

    class FakeSSMClient:
        def get_parameter(self, Name, WithDecryption):
            nonlocal api_calls
            api_calls += 1
            return {"Parameter": {"Value": f"model-v{api_calls}"}}

    fake_boto3 = SimpleNamespace(client=lambda service, region_name: FakeSSMClient())
    monkeypatch.setattr("boto3.client", fake_boto3.client)

    v1 = get_active_model_version()
    v2 = get_active_model_version(force_refresh=True)

    assert v1 == "model-v1"
    assert v2 == "model-v2"
    assert api_calls == 2


def test_SSM_에러_발생_시_None을_반환한다(monkeypatch):
    clear_ssm_cache()

    class ErrorSSMClient:
        def get_parameter(self, Name, WithDecryption):
            from botocore.exceptions import ClientError

            raise ClientError(
                {"Error": {"Code": "ParameterNotFound", "Message": "Not found"}},
                "GetParameter",
            )

    fake_boto3 = SimpleNamespace(client=lambda service, region_name: ErrorSSMClient())
    monkeypatch.setattr("boto3.client", fake_boto3.client)

    assert get_active_model_version() is None


def test_SSM_조회_실패_시_default_model을_반환한다(monkeypatch):
    clear_ssm_cache()

    class ErrorSSMClient:
        def get_parameter(self, Name, WithDecryption):
            from botocore.exceptions import ClientError

            raise ClientError(
                {"Error": {"Code": "ParameterNotFound", "Message": "Not found"}},
                "GetParameter",
            )

    fake_boto3 = SimpleNamespace(client=lambda service, region_name: ErrorSSMClient())
    monkeypatch.setattr("boto3.client", fake_boto3.client)

    result = get_active_model_version(default_model="fallback-model-name")
    assert result == "fallback-model-name"
