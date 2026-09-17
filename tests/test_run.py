from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

import promptlab.run as run_mod
from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import Settings
from promptlab.usage import CallRecord


def _stub_record(request: CompletionRequest, run_id: str, model_id: str) -> CallRecord:
    return CallRecord(
        record_id="00000000-0000-4000-8000-000000000001",
        run_id=run_id,
        timestamp=datetime.now(UTC),
        provider="ollama",
        model_id=model_id,
        task=request.task,
        case_id=request.case_id,
        prompt_id=request.prompt_id,
        prompt_version=request.prompt_version,
        attempt=1,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
        input_tokens=1,
        output_tokens=1,
        cached_input_tokens=None,
        latency_ms=1,
        cost_usd=0.0,
        stop_reason="stop",
        error_type=None,
        response_text="ok",
    )


def _absent() -> dict[str, object]:
    return {"value": None, "status": "absent", "citation": None}


def _present(value: str) -> dict[str, object]:
    return {"value": value, "status": "present", "citation": "1. Document Control"}


def _payload_for(task: str) -> str:
    if task == "triage":
        return json.dumps(
            {
                "queue": "card_dispute",
                "escalation_required": False,
                "confidence": 0.8,
                "rationale": "Recognized duplicate charge.",
                "draft_reply": "We will review the duplicate charge.",
                "human_review_required": True,
                "customer_outcome": None,
            }
        )
    if task == "summarization":
        return json.dumps(
            {
                "document_status": "valid",
                "title": _present("Example"),
                "version": _present("1.0"),
                "effective_date": _present("2025-01-01"),
                "purpose": _absent(),
                "required_steps": _absent(),
                "exceptions": _absent(),
            }
        )
    return json.dumps(
        {
            "document_status": "valid",
            "policy_name": _present("Example"),
            "version": _present("1.0"),
            "effective_date": _present("2025-01-01"),
            "jurisdictions": _absent(),
            "beneficial_ownership_threshold": _absent(),
            "review_frequency": _absent(),
            "required_documents": _absent(),
        }
    )


def test_unfiltered_run_covers_three_tasks_and_both_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_mod, "OUTPUT_PATH", tmp_path / "day5-run.jsonl")
    monkeypatch.setattr(run_mod, "SCORES_PATH", tmp_path / "day5-scores.jsonl")
    monkeypatch.setattr(run_mod, "CALLS_PATH", tmp_path / "day5-calls.jsonl")
    monkeypatch.setattr(run_mod, "REPORT_PATH", tmp_path / "comparison.md")
    monkeypatch.setattr(run_mod, "DECISION_PATH", tmp_path / "model-decision.md")
    calls: list[tuple[str, str, str]] = []

    def fake_complete(
        self: OllamaAdapter, request: CompletionRequest, run_id: str
    ) -> CompletionResult:
        calls.append((self.model_id, request.task, request.case_id))
        assert "{document_text}" not in request.user_content
        assert "JSON instance" in request.system
        record = _stub_record(request, run_id, self.model_id)
        return CompletionResult(
            succeeded=True,
            text=_payload_for(request.task),
            error_type=None,
            records=[record],
        )

    monkeypatch.setattr(OllamaAdapter, "complete", fake_complete)
    run_mod.main(["--run-id", "00000000-0000-4000-8000-000000000099"])

    settings = Settings.from_env()
    expected_models = [settings.models["mistral"].model_id, settings.models["qwen"].model_id]
    tasks = ["summarization", "extraction", "triage"]
    assert len(calls) == 72
    assert [model for model, _task, _case in calls[:36]] == [expected_models[0]] * 36
    assert [model for model, _task, _case in calls[36:]] == [expected_models[1]] * 36
    assert {task for _model, task, _case in calls} == set(tasks)

    outputs = [
        json.loads(line)
        for line in (tmp_path / "day5-run.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(outputs) == 72
    UUID(outputs[0]["run_id"])
    assert {row["run_id"] for row in outputs} == {"00000000-0000-4000-8000-000000000099"}
    assert {row["task"] for row in outputs} == set(tasks)
    assert {row["model_name"] for row in outputs} == {"mistral", "qwen"}
    assert all(row["prompt_id"] for row in outputs)
    assert (tmp_path / "comparison.md").exists()
    assert (tmp_path / "model-decision.md").exists()
    comparison = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    decision = (tmp_path / "model-decision.md").read_text(encoding="utf-8")
    assert "$0.00" in comparison
    assert "transfer" in comparison
    assert "HTTP calls" in comparison
    assert "Do nothing" in decision
    assert all(row["model_id"] for row in outputs)
    scores = [
        json.loads(line)
        for line in (tmp_path / "day5-scores.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert scores
    assert all(row.get("model_id") and row.get("prompt_id") for row in scores)
