"""Extra Day 1 checks. Leaves tests/test_usage_contract.py unmodified."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import httpx
import pytest

from promptlab import day1
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.usage import CallRecord, compute_cost


def _payload(
    *, prompt_eval_count: int, eval_count: int, done_reason: str, text: str
) -> dict[str, Any]:
    return {
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "done_reason": done_reason,
        "response": text,
    }


def _fake_response(payload: dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


def test_selected_cases_are_short_middle_and_long() -> None:
    cases = day1.load_selected_cases(day1.CASES_PATH, day1.CASE_IDS)

    assert [case["id"] for case in cases] == ["E12", "E07", "E11"]
    lengths = [len(case["source"]) for case in cases]
    assert lengths == sorted(lengths)


def test_missing_case_raises() -> None:
    with pytest.raises(KeyError, match="missing extraction cases"):
        day1.load_selected_cases(day1.CASES_PATH, ("E12", "MISSING"))


def test_record_from_payload_maps_ollama_fields() -> None:
    model_id = Settings.from_env().models["mistral"].model_id
    record = day1.record_from_payload(
        payload=_payload(
            prompt_eval_count=228,
            eval_count=50,
            done_reason="stop",
            text="ok",
        ),
        run_id="run-1",
        model_id=model_id,
        case_id="E12",
        attempt=1,
        temperature=0.0,
        max_output_tokens=256,
        latency_ms=3921,
        error_type=None,
    )

    UUID(record.record_id, version=4)
    assert record.timestamp.tzinfo is not None
    assert record.timestamp.utcoffset() is not None
    assert record.provider == "ollama"
    assert record.model_id == model_id
    assert record.task == "extraction"
    assert record.prompt_id == "baseline"
    assert record.prompt_version == "v0"
    assert record.input_tokens == 228
    assert record.output_tokens == 50
    assert record.stop_reason == "stop"
    assert record.cached_input_tokens is None
    assert record.cost_usd == pytest.approx(0.0)
    assert record.error_type is None
    assert record.response_text == "ok"


def test_length_stop_reason_is_recorded_as_truncation() -> None:
    model_id = Settings.from_env().models["mistral"].model_id
    record = day1.record_from_payload(
        payload=_payload(
            prompt_eval_count=274,
            eval_count=8,
            done_reason="length",
            text="truncated",
        ),
        run_id="run-1",
        model_id=model_id,
        case_id="E11",
        attempt=2,
        temperature=0.0,
        max_output_tokens=8,
        latency_ms=346,
        error_type="TruncatedResponseError",
    )

    assert record.stop_reason == "length"
    assert record.error_type == "TruncatedResponseError"
    assert record.max_output_tokens == day1.TRUNCATION_NUM_PREDICT


def test_qwen_also_has_zero_provider_charge() -> None:
    model_id = Settings.from_env().models["qwen"].model_id
    assert compute_cost(model_id, input_tokens=100, output_tokens=20) == pytest.approx(0.0)


def test_evidence_file_has_exactly_three_successful_records() -> None:
    path = PROJECT_ROOT / "docs" / "day1-run.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 3
    assert [row["case_id"] for row in rows] == ["E12", "E07", "E11"]
    run_ids = {row["run_id"] for row in rows}
    assert len(run_ids) == 1

    settings = Settings.from_env()
    for row in rows:
        timestamp = CallRecord.model_validate(row).timestamp
        assert timestamp.tzinfo is not None
        assert timestamp.utcoffset() == UTC.utcoffset(timestamp)
        assert row["provider"] == "ollama"
        assert row["model_id"] == settings.models["mistral"].model_id
        assert row["task"] == "extraction"
        assert row["prompt_id"] == "baseline"
        assert row["prompt_version"] == "v0"
        assert row["attempt"] == 1
        assert row["error_type"] is None
        assert isinstance(row["input_tokens"], int)
        assert isinstance(row["output_tokens"], int)
        assert row["input_tokens"] > 0
        assert row["output_tokens"] > 0
        assert row["cost_usd"] == pytest.approx(0.0)
        assert row["stop_reason"] == "stop"
        assert row["response_text"]


def test_write_evidence_does_not_include_truncation(tmp_path: Path) -> None:
    model_id = Settings.from_env().models["mistral"].model_id
    success = day1.record_from_payload(
        payload=_payload(prompt_eval_count=10, eval_count=4, done_reason="stop", text="ok"),
        run_id="run-1",
        model_id=model_id,
        case_id="E12",
        attempt=1,
        temperature=0.0,
        max_output_tokens=256,
        latency_ms=10,
        error_type=None,
    )
    truncated = day1.record_from_payload(
        payload=_payload(prompt_eval_count=10, eval_count=8, done_reason="length", text="cut"),
        run_id="run-1",
        model_id=model_id,
        case_id="E11",
        attempt=2,
        temperature=0.0,
        max_output_tokens=8,
        latency_ms=10,
        error_type="TruncatedResponseError",
    )
    path = tmp_path / "day1-run.jsonl"
    day1.write_evidence([success], path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert rows[0]["case_id"] == "E12"
    assert truncated.error_type == "TruncatedResponseError"


def test_main_uses_config_model_and_keeps_truncation_out_of_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path = tmp_path / "docs" / "day1-run.jsonl"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(day1, "EVIDENCE_PATH", evidence_path)

    settings = Settings.from_env()
    model_id = settings.models["mistral"].model_id
    requests: list[dict[str, Any]] = []

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> MagicMock:
        requests.append(json)
        num_predict = json["options"]["num_predict"]
        if num_predict == day1.TRUNCATION_NUM_PREDICT:
            payload = _payload(
                prompt_eval_count=274, eval_count=8, done_reason="length", text="cut"
            )
        else:
            payload = _payload(prompt_eval_count=200, eval_count=40, done_reason="stop", text="ok")
        return _fake_response(payload)

    monkeypatch.setattr(httpx, "post", fake_post)
    day1.main()

    assert len(requests) == 4
    assert all(request["model"] == model_id for request in requests)
    assert [request["options"]["num_predict"] for request in requests] == [
        day1.MAX_OUTPUT_TOKENS,
        day1.MAX_OUTPUT_TOKENS,
        day1.MAX_OUTPUT_TOKENS,
        day1.TRUNCATION_NUM_PREDICT,
    ]
    assert all(request["options"]["temperature"] == 0.0 for request in requests)
    assert "{document_text}" not in requests[0]["prompt"]

    evidence = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()]
    assert [row["case_id"] for row in evidence] == ["E12", "E07", "E11"]
    assert all(row["error_type"] is None for row in evidence)

    run_files = list((tmp_path / "runs").glob("*.jsonl"))
    assert len(run_files) == 1
    run_rows = [json.loads(line) for line in run_files[0].read_text(encoding="utf-8").splitlines()]
    assert len(run_rows) == 4
    assert run_rows[-1]["error_type"] == "TruncatedResponseError"
    assert run_rows[-1]["stop_reason"] == "length"
    assert run_rows[-1]["case_id"] == "E11"
