"""Reusable Ollama adapter for any configured local model."""

from __future__ import annotations

import random
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.config import Settings
from promptlab.errors import (
    PermanentProviderError,
    TransientProviderError,
    TruncatedResponseError,
    UnknownModelError,
)
from promptlab.usage import CallRecord, append_record, compute_cost

MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.25
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class OllamaAdapter:
    """One adapter class; Mistral and Qwen differ only by configured model_id."""

    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        timeout_seconds: float = 180.0,
    ) -> None:
        settings = Settings.from_env()
        self.provider = "ollama"
        self.model_id = model_id
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._timeout_seconds = timeout_seconds

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        self._require_configured_model()
        records: list[CallRecord] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            record = self._one_attempt(request, run_id, attempt)
            append_record(record, run_id)
            records.append(record)
            if record.error_type is None:
                return CompletionResult(
                    succeeded=True,
                    text=record.response_text,
                    error_type=None,
                    records=records,
                )
            if record.error_type != TransientProviderError.__name__ or attempt == MAX_ATTEMPTS:
                return CompletionResult(
                    succeeded=False,
                    text=record.response_text,
                    error_type=record.error_type,
                    records=records,
                )
            time.sleep(_backoff_seconds(attempt))
        last = records[-1]
        return CompletionResult(
            succeeded=False,
            text=last.response_text,
            error_type=last.error_type,
            records=records,
        )

    def _require_configured_model(self) -> None:
        settings = Settings.from_env()
        if all(config.model_id != self.model_id for config in settings.models.values()):
            raise UnknownModelError(self.model_id)

    def _one_attempt(self, request: CompletionRequest, run_id: str, attempt: int) -> CallRecord:
        payload: dict[str, Any] = {}
        error_type: str | None = None
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json=_generate_body(self.model_id, request),
                timeout=self._timeout_seconds,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            status_code = int(getattr(response, "status_code", 200))
            payload = _json_object(response)
            if status_code >= 400:
                error_type = _classify_status(status_code)
            elif payload.get("done_reason") == "length":
                error_type = TruncatedResponseError.__name__
        except (httpx.TimeoutException, httpx.NetworkError):
            latency_ms = int((time.perf_counter() - started) * 1000)
            error_type = TransientProviderError.__name__
        except httpx.HTTPError:
            latency_ms = int((time.perf_counter() - started) * 1000)
            error_type = PermanentProviderError.__name__

        input_tokens = _optional_int(payload.get("prompt_eval_count"))
        output_tokens = _optional_int(payload.get("eval_count"))
        stop_reason = payload.get("done_reason")
        response_text = _completion_text(payload)
        return CallRecord(
            record_id=str(uuid4()),
            run_id=run_id,
            timestamp=datetime.now(UTC),
            provider="ollama",
            model_id=self.model_id,
            task=request.task,
            case_id=request.case_id,
            prompt_id=request.prompt_id,
            prompt_version=request.prompt_version,
            attempt=attempt,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=None,
            latency_ms=latency_ms,
            cost_usd=compute_cost(self.model_id, input_tokens, output_tokens),
            stop_reason=stop_reason if isinstance(stop_reason, str) else None,
            error_type=error_type,
            response_text=response_text if isinstance(response_text, str) else None,
        )


def _generate_body(model_id: str, request: CompletionRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_id,
        "prompt": request.user_content,
        "stream": False,
        "think": False,
        "options": {
            "temperature": request.temperature,
            "num_predict": request.max_output_tokens,
        },
    }
    if request.system:
        body["system"] = request.system
    return body


def _backoff_seconds(failed_attempt: int) -> float:
    multiplier = 2 ** (failed_attempt - 1)
    ceiling: float = BACKOFF_BASE_SECONDS * multiplier
    jitter: float = float(random.uniform(0, ceiling))
    return ceiling + jitter


def _classify_status(status_code: int) -> str:
    if status_code in TRANSIENT_STATUS_CODES:
        return TransientProviderError.__name__
    return PermanentProviderError.__name__


def _json_object(response: object) -> dict[str, Any]:
    json_method = getattr(response, "json", None)
    if not callable(json_method):
        return {}
    try:
        payload: Any = json_method()
    except ValueError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _completion_text(payload: dict[str, Any]) -> str | None:
    response_text = payload.get("response")
    if isinstance(response_text, str):
        return response_text
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return None


def _optional_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value
