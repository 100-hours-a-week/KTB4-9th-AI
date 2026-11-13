import inspect
import json
import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Response
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

logger = logging.getLogger(__name__)

_SAFE_VALUE = re.compile(r"[^a-zA-Z0-9._:/-]+")
_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter", "unknown"}


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except TypeError, ValueError:
        return None
    return parsed if parsed >= 0 else None


def _mapping_or_attributes(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return {}
    return {
        key: getattr(value, key, None)
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
        )
    }


def extract_llm_usage(value: Any) -> LLMUsage:
    """Return token counts only when a provider supplied valid usage metadata."""
    if not isinstance(value, Mapping):
        usage_metadata = getattr(value, "usage_metadata", None)
        usage = getattr(value, "usage", None)
        value = usage_metadata or usage or value
    data = _mapping_or_attributes(value)
    input_tokens = _positive_int(data.get("input_tokens", data.get("prompt_tokens")))
    output_tokens = _positive_int(
        data.get("output_tokens", data.get("completion_tokens"))
    )
    total_tokens = _positive_int(data.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return LLMUsage(input_tokens, output_tokens, total_tokens)


def _bounded(value: str | None, *, default: str = "unknown") -> str:
    if not value:
        return default
    normalized = _SAFE_VALUE.sub("_", value.strip())[:80]
    return normalized or default


def _finish_reason(value: str | None) -> str:
    normalized = _bounded(value)
    return normalized if normalized in _FINISH_REASONS else "other"


def safe_llm_attributes(
    *,
    provider: str,
    model: str,
    operation: str,
    result: str,
    finish_reason: str | None,
    fallback: bool,
) -> dict[str, str | bool]:
    """Build the only dimensions allowed on LLM metrics."""
    return {
        "provider": _bounded(provider),
        "model": _bounded(model),
        "operation": _bounded(operation),
        "result": "success" if result == "success" else "error",
        "finish_reason": _finish_reason(finish_reason),
        "fallback": bool(fallback),
    }


def estimate_llm_cost_usd(
    provider: str,
    model: str,
    usage: LLMUsage,
    pricing_json: str | None,
) -> float | None:
    """Estimate cost only from an exact provider/model price configured by operators."""
    if usage.input_tokens is None or usage.output_tokens is None or not pricing_json:
        return None
    try:
        pricing = json.loads(pricing_json)
        rate = pricing[f"{provider}/{model}"]
        input_rate = float(rate["input"])
        output_rate = float(rate["output"])
    except KeyError, TypeError, ValueError, json.JSONDecodeError:
        return None
    token_cost = usage.input_tokens * input_rate + usage.output_tokens * output_rate
    return token_cost / 1_000_000


@dataclass
class _LLMInstruments:
    requests: Any
    duration: Any
    tokens: Any
    cost: Any
    retries: Any
    fallbacks: Any
    first_token: Any


class TelemetryRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.initialized = False
        self.tracer_provider: Any | None = None
        self.meter_provider: Any | None = None
        self.instruments: _LLMInstruments | None = None
        self._instrumented_apps: set[int] = set()
        self._instrumented_engines: set[int] = set()
        self._httpx_instrumented = False

    def initialize(
        self,
        *,
        app: FastAPI | None = None,
        engine: Any | None = None,
        environ: Mapping[str, str] = os.environ,
    ) -> bool:
        if environ.get("OTEL_SDK_DISABLED", "false").lower() == "true":
            return False

        with self._lock:
            if not self.initialized:
                resource = Resource.create(
                    {"service.name": environ.get("OTEL_SERVICE_NAME", "cosmos-ai")}
                )
                reader = PrometheusMetricReader()
                self.meter_provider = MeterProvider(
                    resource=resource, metric_readers=[reader]
                )

                endpoint = environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
                traces_exporter = environ.get("OTEL_TRACES_EXPORTER", "otlp")
                if endpoint and traces_exporter != "none":
                    provider = TracerProvider(resource=resource)
                    provider.add_span_processor(
                        BatchSpanProcessor(
                            OTLPSpanExporter(
                                endpoint=endpoint,
                                insecure=endpoint.startswith("http://"),
                                timeout=float(
                                    environ.get("OTEL_EXPORTER_OTLP_TIMEOUT", "10")
                                ),
                            )
                        )
                    )
                    trace.set_tracer_provider(provider)
                    self.tracer_provider = provider

                meter = self.meter_provider.get_meter("cosmos.ai.llm")
                self.instruments = _LLMInstruments(
                    requests=meter.create_counter("cosmos.llm.requests"),
                    duration=meter.create_histogram(
                        "cosmos.llm.request.duration", unit="s"
                    ),
                    tokens=meter.create_counter("cosmos.llm.tokens"),
                    cost=meter.create_counter("cosmos.llm.estimated.cost.usd"),
                    retries=meter.create_counter("cosmos.llm.retries"),
                    fallbacks=meter.create_counter("cosmos.llm.fallbacks"),
                    first_token=meter.create_histogram(
                        "cosmos.llm.time.to.first.token", unit="s"
                    ),
                )
                self.initialized = True

            self._instrument(app=app, engine=engine)
            return True

    def _instrument(self, *, app: FastAPI | None, engine: Any | None) -> None:
        if app is not None and id(app) not in self._instrumented_apps:
            FastAPIInstrumentor.instrument_app(
                app,
                tracer_provider=self.tracer_provider,
                meter_provider=self.meter_provider,
                excluded_urls="health,metrics",
                http_capture_headers_server_request=[],
                http_capture_headers_server_response=[],
            )
            self._instrumented_apps.add(id(app))
        if engine is not None and id(engine) not in self._instrumented_engines:
            SQLAlchemyInstrumentor().instrument(
                engine=engine.sync_engine,
                tracer_provider=self.tracer_provider,
            )
            self._instrumented_engines.add(id(engine))
        if not self._httpx_instrumented:
            HTTPXClientInstrumentor().instrument(tracer_provider=self.tracer_provider)
            self._httpx_instrumented = True

    def shutdown(self, timeout_millis: int = 5000) -> None:
        providers = [self.tracer_provider, self.meter_provider]
        for provider in providers:
            if provider is None:
                continue
            try:
                provider.force_flush(timeout_millis=timeout_millis)
            except Exception:
                logger.warning("telemetry force flush failed", exc_info=True)
            self._bounded_shutdown(provider, timeout_millis)
        with self._lock:
            self.initialized = False
            self.tracer_provider = None
            self.meter_provider = None
            self.instruments = None

    @staticmethod
    def _bounded_shutdown(provider: Any, timeout_millis: int) -> None:
        def close() -> None:
            try:
                parameters = inspect.signature(provider.shutdown).parameters
                if "timeout_millis" in parameters:
                    provider.shutdown(timeout_millis=timeout_millis)
                else:
                    provider.shutdown()
            except Exception:
                logger.warning("telemetry shutdown failed", exc_info=True)

        thread = threading.Thread(target=close, daemon=True)
        thread.start()
        thread.join(timeout_millis / 1000)
        if thread.is_alive():
            logger.warning("telemetry shutdown exceeded %dms", timeout_millis)


_runtime = TelemetryRuntime()


def init_telemetry(*, app: FastAPI | None = None, engine: Any | None = None) -> bool:
    return _runtime.initialize(app=app, engine=engine)


def shutdown_telemetry(timeout_millis: int = 5000) -> None:
    _runtime.shutdown(timeout_millis)


def install_metrics_endpoint(app: FastAPI) -> None:
    if not any(getattr(route, "path", None) == "/metrics" for route in app.routes):
        app.add_api_route(
            "/metrics",
            metrics_response,
            methods=["GET"],
            include_in_schema=False,
        )


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class LLMObservation:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        operation: str,
        fallback: bool,
    ) -> None:
        self.provider = _bounded(provider)
        self.model = _bounded(model)
        self.operation = _bounded(operation)
        self.fallback = fallback
        self.usage = LLMUsage()
        self.finish_reason: str | None = None
        self.first_token_seconds: float | None = None
        self.started_at = time.perf_counter()

    def record_response(self, response: Any) -> None:
        self.usage = extract_llm_usage(response)
        self.finish_reason = _extract_finish_reason(response)

    def record_first_token(self) -> None:
        self.first_token_seconds = time.perf_counter() - self.started_at


def _extract_finish_reason(response: Any) -> str | None:
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, Mapping):
        reason = metadata.get("finish_reason") or metadata.get("finishReason")
        if reason:
            return str(reason)
    choices = getattr(response, "choices", None)
    if choices:
        return str(getattr(choices[0], "finish_reason", None) or "unknown")
    return None


@contextmanager
def observe_llm_call(
    *,
    provider: str,
    model: str,
    operation: str = "structured.generate",
    fallback: bool = False,
):
    observation = LLMObservation(
        provider=provider,
        model=model,
        operation=operation,
        fallback=fallback,
    )
    tracer = trace.get_tracer("cosmos.ai.llm")
    with tracer.start_as_current_span(f"gen_ai.{operation}") as span:
        span.set_attribute("gen_ai.operation.name", observation.operation)
        span.set_attribute("gen_ai.provider.name", observation.provider)
        span.set_attribute("gen_ai.request.model", observation.model)
        try:
            yield observation
        except BaseException as error:
            span.set_attribute("error.type", type(error).__name__)
            span.set_status(Status(StatusCode.ERROR, type(error).__name__))
            _record_llm_observation(observation, result="error")
            raise
        else:
            span.set_status(Status(StatusCode.OK))
            if observation.usage.input_tokens is not None:
                span.set_attribute(
                    "gen_ai.usage.input_tokens", observation.usage.input_tokens
                )
            if observation.usage.output_tokens is not None:
                span.set_attribute(
                    "gen_ai.usage.output_tokens", observation.usage.output_tokens
                )
            span.set_attribute(
                "gen_ai.response.finish_reasons",
                [_finish_reason(observation.finish_reason)],
            )
            _record_llm_observation(observation, result="success")


def _record_llm_observation(observation: LLMObservation, *, result: str) -> None:
    instruments = _runtime.instruments
    if instruments is None:
        return
    attributes = safe_llm_attributes(
        provider=observation.provider,
        model=observation.model,
        operation=observation.operation,
        result=result,
        finish_reason=observation.finish_reason,
        fallback=observation.fallback,
    )
    instruments.requests.add(1, attributes)
    instruments.duration.record(
        time.perf_counter() - observation.started_at, attributes
    )
    for token_type, value in (
        ("input", observation.usage.input_tokens),
        ("output", observation.usage.output_tokens),
        ("total", observation.usage.total_tokens),
    ):
        if value is not None:
            instruments.tokens.add(value, {**attributes, "token_type": token_type})
    cost = estimate_llm_cost_usd(
        observation.provider,
        observation.model,
        observation.usage,
        os.getenv("LLM_PRICING_USD_PER_MILLION"),
    )
    if cost is not None:
        instruments.cost.add(cost, attributes)
    if observation.first_token_seconds is not None:
        instruments.first_token.record(observation.first_token_seconds, attributes)


def record_llm_retry(*, provider: str, model: str) -> None:
    if _runtime.instruments is not None:
        _runtime.instruments.retries.add(
            1, {"provider": _bounded(provider), "model": _bounded(model)}
        )


def record_llm_fallback(*, provider: str, model: str) -> None:
    if _runtime.instruments is not None:
        _runtime.instruments.fallbacks.add(
            1, {"provider": _bounded(provider), "model": _bounded(model)}
        )
