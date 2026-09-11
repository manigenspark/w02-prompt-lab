"""Adapter contract and retry tests. Does not make a live Ollama call."""

from __future__ import annotations

from typing import Any, get_args, get_type_hints
from unittest.mock import MagicMock

import httpx
import pytest

from promptlab.adapters.base import CompletionRequest, CompletionResult, ModelAdapter
from promptlab.adapters.ollama import MAX_ATTEMPTS, OllamaAdapter
from promptlab.config import Settings
from promptlab.errors import (
    PermanentProviderError,
    TransientProviderError,
    TruncatedResponseError,
    UnknownModelError,
)
from promptlab.usage import CallRecord

REQUEST_FIELDS = {
    "task",
    "case_id",
    "prompt_id",
    "prompt_version",
    "system",
    "user_content",
    "temperature",
    "max_output_tokens",
}

RESULT_FIELDS = {"succeeded", "text", "error_type", "records"}


def _request() -> CompletionRequest:
    return CompletionRequest(
        task="summarization",
        case_id="S01",
        prompt_id="baseline",
        prompt_version="v0",
        system="Summarize the document.",
        user_content="A short policy document.",
        temperature=0.0,
        max_output_tokens=256,
    )


def _adapter() -> OllamaAdapter:
    settings = Settings.from_env()
    model = next(iter(settings.models.values()))
    return OllamaAdapter(model_id=model.model_id, base_url="http://adapter.test")


def _ok_payload() -> dict[str, Any]:
    return {
        "response": "summary text",
        "prompt_eval_count": 40,
        "eval_count": 12,
        "done_reason": "stop",
    }


def _response(status_code: int, payload: dict[str, Any] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "http://adapter.test/api/generate")
    return httpx.Response(status_code, json=payload or {}, request=request)


def _mock_post(payload: dict[str, Any]) -> MagicMock:
    mock = MagicMock(return_value=_response(200, payload))
    return mock


def test_completion_models_match_contract() -> None:
    assert set(CompletionRequest.model_fields) == REQUEST_FIELDS
    assert set(CompletionResult.model_fields) == RESULT_FIELDS
    task_annotation = CompletionRequest.model_fields["task"].annotation
    assert set(get_args(task_annotation)) == {"triage", "summarization", "extraction"}


def test_model_adapter_protocol_surface() -> None:
    hints = get_type_hints(ModelAdapter.complete)
    assert hints["request"] is CompletionRequest
    assert hints["return"] is CompletionResult
    adapter = _adapter()
    assert isinstance(adapter, ModelAdapter)
    assert adapter.provider == "ollama"


def test_error_types_exist() -> None:
    assert issubclass(TransientProviderError, Exception)
    assert issubclass(PermanentProviderError, Exception)
    assert issubclass(TruncatedResponseError, Exception)
    assert issubclass(UnknownModelError, ValueError)


def test_unknown_model_is_not_attempted(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OllamaAdapter(model_id="not-a-configured-model", base_url="http://adapter.test")
    monkeypatch.setattr(httpx, "post", MagicMock())
    with pytest.raises(UnknownModelError):
        adapter.complete(_request(), "run-unknown")
    assert httpx.post.call_count == 0  # type: ignore[attr-defined]


def test_success_records_one_attempt(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(httpx, "post", _mock_post(_ok_payload()))
    result = _adapter().complete(_request(), "run-ok")
    assert result.succeeded is True
    assert result.text == "summary text"
    assert result.error_type is None
    assert len(result.records) == 1
    record = result.records[0]
    assert record.attempt == 1
    assert record.input_tokens == 40
    assert record.output_tokens == 12
    assert record.stop_reason == "stop"
    assert record.cost_usd == pytest.approx(0.0)
    CallRecord.model_validate(record.model_dump())


def test_transient_failure_retries_then_succeeds(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sleeps: list[float] = []
    monkeypatch.setattr("promptlab.adapters.ollama.time.sleep", lambda delay: sleeps.append(delay))
    monkeypatch.setattr("promptlab.adapters.ollama.random.uniform", lambda _a, b: b)

    calls: list[httpx.Response | BaseException] = [
        httpx.TimeoutException("timeout"),
        _response(200, _ok_payload()),
    ]

    def fake_post(*_args: object, **_kwargs: object) -> httpx.Response:
        outcome = calls.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _adapter().complete(_request(), "run-retry")

    assert result.succeeded is True
    assert [record.attempt for record in result.records] == [1, 2]
    assert result.records[0].error_type == TransientProviderError.__name__
    assert result.records[1].error_type is None
    assert len(sleeps) == 1
    assert sleeps[0] > 0


def test_permanent_failure_is_not_retried(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sleep = MagicMock()
    monkeypatch.setattr("promptlab.adapters.ollama.time.sleep", sleep)
    monkeypatch.setattr(httpx, "post", MagicMock(return_value=_response(400, {"error": "bad"})))
    result = _adapter().complete(_request(), "run-perm")
    assert result.succeeded is False
    assert result.error_type == PermanentProviderError.__name__
    assert len(result.records) == 1
    assert result.records[0].attempt == 1
    sleep.assert_not_called()
    assert httpx.post.call_count == 1  # type: ignore[attr-defined]


def test_truncation_is_not_retried(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sleep = MagicMock()
    monkeypatch.setattr("promptlab.adapters.ollama.time.sleep", sleep)
    payload = {
        "response": "partial",
        "prompt_eval_count": 40,
        "eval_count": 256,
        "done_reason": "length",
    }
    monkeypatch.setattr(httpx, "post", _mock_post(payload))
    result = _adapter().complete(_request(), "run-trunc")
    assert result.succeeded is False
    assert result.error_type == TruncatedResponseError.__name__
    assert len(result.records) == 1
    assert result.records[0].stop_reason == "length"
    sleep.assert_not_called()


def test_transient_failures_stop_at_three_attempts(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("promptlab.adapters.ollama.time.sleep", lambda _delay: None)
    monkeypatch.setattr(
        httpx,
        "post",
        MagicMock(side_effect=httpx.ConnectError("offline")),
    )
    result = _adapter().complete(_request(), "run-max")
    assert result.succeeded is False
    assert result.error_type == TransientProviderError.__name__
    assert len(result.records) == MAX_ATTEMPTS
    assert [record.attempt for record in result.records] == [1, 2, 3]
    assert httpx.post.call_count == MAX_ATTEMPTS  # type: ignore[attr-defined]


def test_server_error_is_retryable(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("promptlab.adapters.ollama.time.sleep", lambda _delay: None)
    monkeypatch.setattr(
        httpx,
        "post",
        MagicMock(
            side_effect=[
                _response(503, {"error": "busy"}),
                _response(200, _ok_payload()),
            ]
        ),
    )
    result = _adapter().complete(_request(), "run-503")
    assert result.succeeded is True
    assert result.records[0].error_type == TransientProviderError.__name__
    assert len(result.records) == 2
