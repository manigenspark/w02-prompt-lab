from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from promptlab.records import OutputRecord, ScoreRecord
from promptlab.report import write_reports
from promptlab.usage import CallRecord


def _call(*, model_id: str, case_id: str, prompt_version: str = "v1") -> CallRecord:
    return CallRecord(
        record_id="00000000-0000-4000-8000-000000000001",
        run_id="report-test",
        timestamp=datetime.now(UTC),
        provider="ollama",
        model_id=model_id,
        task="triage",
        case_id=case_id,
        prompt_id="triage",
        prompt_version=prompt_version,
        attempt=1,
        temperature=0.0,
        max_output_tokens=128,
        input_tokens=10,
        output_tokens=5,
        cached_input_tokens=None,
        latency_ms=100,
        cost_usd=0.0,
        stop_reason="stop",
        error_type=None,
        response_text="ok",
    )


def _output(*, model_name: str, model_id: str, case_id: str) -> OutputRecord:
    return OutputRecord(
        run_id="report-test",
        task="triage",
        case_id=case_id,
        model_name=model_name,
        model_id=model_id,
        prompt_id="triage",
        prompt_version="v1",
        succeeded=True,
        repairs=0,
        output={
            "queue": "card_dispute",
            "customer_outcome": None,
            "draft_reply": "We will review.",
        },
        error=None,
    )


def _score(
    *,
    model_name: str,
    model_id: str,
    case_id: str,
    metric: str,
    numerator: int,
    lower_is_better: bool = False,
) -> ScoreRecord:
    return ScoreRecord(
        run_id="report-test",
        task="triage",
        case_id=case_id,
        model_name=model_name,
        model_id=model_id,
        prompt_id="triage",
        prompt_version="v1",
        scorer_version="v2",
        metric=metric,
        numerator=numerator,
        denominator=1,
        lower_is_better=lower_is_better,
        detail=None,
    )


def test_comparison_labels_transfer_and_avoids_percentages(tmp_path: Path) -> None:
    report_path = tmp_path / "comparison.md"
    decision_path = tmp_path / "model-decision.md"
    write_reports(
        run_id="report-test",
        models=["mistral", "qwen"],
        usage=[
            _call(model_id="mistral:7b", case_id="T01"),
            _call(model_id="qwen3:8b", case_id="T01"),
        ],
        outputs=[
            _output(model_name="mistral", model_id="mistral:7b", case_id="T01"),
            _output(model_name="qwen", model_id="qwen3:8b", case_id="T01"),
        ],
        scores=[
            _score(
                model_name="mistral",
                model_id="mistral:7b",
                case_id="T01",
                metric="missed_escalation",
                numerator=1,
                lower_is_better=True,
            ),
            _score(
                model_name="qwen",
                model_id="qwen3:8b",
                case_id="T01",
                metric="missed_escalation",
                numerator=0,
                lower_is_better=True,
            ),
            _score(
                model_name="mistral",
                model_id="mistral:7b",
                case_id="T01",
                metric="human_boundary",
                numerator=1,
            ),
            _score(
                model_name="qwen",
                model_id="qwen3:8b",
                case_id="T01",
                metric="human_boundary",
                numerator=1,
            ),
            _score(
                model_name="mistral",
                model_id="mistral:7b",
                case_id="T01",
                metric="queue",
                numerator=0,
            ),
            _score(
                model_name="qwen",
                model_id="qwen3:8b",
                case_id="T01",
                metric="queue",
                numerator=1,
            ),
        ],
        report_path=report_path,
        decision_path=decision_path,
    )
    comparison = report_path.read_text(encoding="utf-8")
    decision = decision_path.read_text(encoding="utf-8")
    assert "$0.00" in comparison
    assert "triage.v1 transfer" in comparison
    assert "HTTP calls" in comparison
    assert "%" not in comparison
    assert "Twelve cases per task" in comparison
    assert "human_boundary" in comparison
    assert "`mistral`" in comparison
    assert "`qwen`" in comparison
    assert "selected model: `qwen`" in decision
    assert "missed_escalation" in decision
    assert "Do nothing" in decision
    assert "not because of latency" in decision
    assert "reopen if:" in decision


def test_report_module_does_not_call_a_model() -> None:
    import inspect

    import promptlab.report as report

    source = inspect.getsource(report)
    assert "OllamaAdapter" not in source
    assert "httpx" not in source
