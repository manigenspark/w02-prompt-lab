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
            if (
                record.error_type != TransientProviderError.__name__
                or attempt == MAX_ATTEMPTS
            ):
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

    def _one_attempt(
        self, request: CompletionRequest, run_id: str, attempt: int
    ) -> CallRecord:
        payload: dict[str, Any] = {}
        error_type: str | None = None
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self.model_id,
                    "prompt": request.user_content,
                    "stream": False,
                    "think": False,
                    "system": request.system,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_output_tokens,
                    },
                },
                timeout=self._timeout_seconds,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            try:
                raw = response.json()
            except ValueError:
                raw = {}
            payload = raw if isinstance(raw, dict) else {}
            status_code = int(getattr(response, "status_code", 200))
            if status_code >= 400:
                if status_code in TRANSIENT_STATUS_CODES:
                    error_type = TransientProviderError.__name__
                else:
                    error_type = PermanentProviderError.__name__
            elif payload.get("done_reason") == "length":
                error_type = TruncatedResponseError.__name__
        except (httpx.TimeoutException, httpx.NetworkError):
            latency_ms = int((time.perf_counter() - started) * 1000)
            error_type = TransientProviderError.__name__
        except httpx.HTTPError:
            latency_ms = int((time.perf_counter() - started) * 1000)
            error_type = PermanentProviderError.__name__

        input_tokens = payload.get("prompt_eval_count")
        output_tokens = payload.get("eval_count")
        if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
            input_tokens = 0
        if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
            output_tokens = 0

        stop_reason = payload.get("done_reason")
        response_text = payload.get("response")
        if not isinstance(response_text, str):
            message = payload.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                response_text = message["content"]
            else:
                response_text = None

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
            response_text=response_text,
        )


def _backoff_seconds(failed_attempt: int) -> float:
    multiplier = 2 ** (failed_attempt - 1)
    ceiling: float = BACKOFF_BASE_SECONDS * multiplier
    jitter: float = float(random.uniform(0, ceiling))
    return ceiling + jitter
