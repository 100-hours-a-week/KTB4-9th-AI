import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from src.core.telemetry import (
    LLMUsage,
    TelemetryRuntime,
    estimate_llm_cost_usd,
    extract_llm_usage,
    init_telemetry,
    metrics_response,
    observe_batch_job,
    observe_llm_call,
    record_llm_fallback,
    record_llm_retry,
    safe_llm_attributes,
)


def test_safe_llm_attributes_only_exposes_bounded_dimensions() -> None:
    secret = "user code and credential must never appear"

    attributes = safe_llm_attributes(
        provider="google",
        model="gemini-3.8-flash",
        operation="structured.generate",
        result="success",
        finish_reason="stop",
        fallback=False,
    )

    assert set(attributes) == {
        "provider",
        "model",
        "operation",
        "result",
        "finish_reason",
        "fallback",
    }
    assert secret not in json.dumps(attributes)


def test_unknown_or_malformed_usage_is_omitted_instead_of_recorded_as_zero() -> None:
    assert extract_llm_usage(object()) == LLMUsage()
    assert extract_llm_usage({"input_tokens": "not-a-number"}) == LLMUsage()


def test_usage_is_extracted_from_mapping_or_sdk_object() -> None:
    mapping = {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }

    assert extract_llm_usage(mapping) == LLMUsage(
        input_tokens=11, output_tokens=7, total_tokens=18
    )


def test_cost_is_only_estimated_when_exact_provider_model_pricing_exists() -> None:
    pricing = json.dumps({"google/gemini-3.8-flash": {"input": 0.10, "output": 0.40}})
    usage = LLMUsage(input_tokens=1_000_000, output_tokens=500_000)

    assert estimate_llm_cost_usd("google", "gemini-3.8-flash", usage, pricing) == 0.3
    assert estimate_llm_cost_usd("google", "unknown", usage, pricing) is None
    assert estimate_llm_cost_usd("google", "gemini-3.8-flash", usage, "bad") is None


class RecordingProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.flush_timeout: int | None = None
        self.shutdown_timeout: int | None = None

    def force_flush(self, timeout_millis: int) -> bool:
        self.flush_timeout = timeout_millis
        if self.fail:
            raise RuntimeError("collector unavailable")
        return True

    def shutdown(self, timeout_millis: int) -> None:
        self.shutdown_timeout = timeout_millis
        if self.fail:
            raise RuntimeError("collector unavailable")


def test_shutdown_is_bounded_and_exporter_failure_is_not_propagated() -> None:
    trace_provider = RecordingProvider(fail=True)
    meter_provider = RecordingProvider()
    runtime = TelemetryRuntime()
    runtime.tracer_provider = trace_provider
    runtime.meter_provider = meter_provider

    runtime.shutdown(timeout_millis=25)

    assert trace_provider.flush_timeout == 25
    assert meter_provider.flush_timeout == 25
    assert meter_provider.shutdown_timeout == 25


def test_disabled_runtime_initialization_is_safe_and_idempotent() -> None:
    runtime = TelemetryRuntime()
    environment = {"OTEL_SDK_DISABLED": "true"}

    assert runtime.initialize(environ=environment) is False
    assert runtime.initialize(environ=environment) is False


def test_prometheus_metric_names_match_dashboard_contract(monkeypatch) -> None:
    monkeypatch.setenv(
        "LLM_PRICING_USD_PER_MILLION",
        json.dumps({"google/gemini-test": {"input": 1, "output": 2}}),
    )
    monkeypatch.setenv("LLM_PRICING_VERSION", "reviewed-2026-11-13")
    init_telemetry()
    with observe_llm_call(provider="google", model="gemini-test") as observation:
        observation.usage = LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        observation.record_first_token()
    record_llm_retry(provider="google", model="gemini-test")
    record_llm_fallback(provider="google", model="gemini-test")

    body = bytes(metrics_response().body).decode()

    for metric in (
        "cosmos_llm_requests_total",
        "cosmos_llm_request_duration_seconds",
        "cosmos_llm_tokens_total",
        "cosmos_llm_estimated_cost_usd_total",
        "cosmos_llm_retries_total",
        "cosmos_llm_fallbacks_total",
        "cosmos_llm_time_to_first_token_seconds",
    ):
        assert metric in body
    assert 'pricing_version="reviewed-2026-11-13"' in body


def test_llm_error_span_never_exports_raw_exception_content(monkeypatch) -> None:
    secret = "prompt-and-api-key-must-not-leave-process"
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        "src.core.telemetry.trace.get_tracer",
        lambda _name: provider.get_tracer("privacy-test"),
    )

    with pytest.raises(RuntimeError, match=secret):
        with observe_llm_call(provider="google", model="gemini-test"):
            raise RuntimeError(secret)

    exported = json.dumps(
        [
            {
                "attributes": dict(span.attributes or {}),
                "events": [dict(event.attributes or {}) for event in span.events],
                "status": span.status.description,
            }
            for span in exporter.get_finished_spans()
        ],
        default=str,
    )
    assert secret not in exported
    assert "exception.message" not in exported
    assert "exception.stacktrace" not in exported


def test_batch_root_span_records_identity_status_without_exception_content(
    monkeypatch,
) -> None:
    secret = "batch-payload-must-not-leave-process"
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        "src.core.telemetry.trace.get_tracer",
        lambda _name: provider.get_tracer("batch-test"),
    )

    with pytest.raises(RuntimeError, match=secret):
        with observe_batch_job(job="generate", only="battle"):
            raise RuntimeError(secret)

    span = exporter.get_finished_spans()[0]
    assert span.name == "batch.generate"
    assert span.attributes["batch.job"] == "generate"
    assert span.attributes["batch.step"] == "battle"
    assert span.attributes["error.type"] == "RuntimeError"
    assert secret not in json.dumps(
        [dict(event.attributes or {}) for event in span.events], default=str
    )
