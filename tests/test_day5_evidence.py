from __future__ import annotations

import json
from pathlib import Path

from promptlab.config import PROJECT_ROOT
from promptlab.records import OutputRecord, ScoreRecord
from promptlab.usage import CallRecord

RUN_ID = "c9e2f0a1-7d44-4b1c-8e6f-2a91b0c5d387"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "day5-run.jsonl"
SCORES_PATH = PROJECT_ROOT / "docs" / "day5-scores.jsonl"
CALLS_PATH = PROJECT_ROOT / "docs" / "day5-calls.jsonl"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "comparison.md"
DECISION_PATH = PROJECT_ROOT / "docs" / "model-decision.md"
JOIN_KEYS = ("run_id", "case_id", "task", "model_id", "prompt_id", "prompt_version")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_day5_outputs_cover_72_cells() -> None:
    rows = [OutputRecord.model_validate(row) for row in _load_jsonl(OUTPUT_PATH)]
    assert len(rows) == 72
    assert {row.run_id for row in rows} == {RUN_ID}
    assert {row.task for row in rows} == {"summarization", "extraction", "triage"}
    assert {row.model_name for row in rows} == {"mistral", "qwen"}
    assert all(row.prompt_id and row.model_id for row in rows)


def test_day5_scores_join_to_outputs_and_calls() -> None:
    outputs = [OutputRecord.model_validate(row) for row in _load_jsonl(OUTPUT_PATH)]
    scores = [ScoreRecord.model_validate(row) for row in _load_jsonl(SCORES_PATH)]
    calls = [CallRecord.model_validate(row) for row in _load_jsonl(CALLS_PATH)]
    assert {row.run_id for row in scores} == {RUN_ID}
    assert {row.run_id for row in calls} == {RUN_ID}
    assert all(row.scorer_version == "v2" for row in scores)

    output_keys = {
        (row.run_id, row.case_id, row.task, row.model_id, row.prompt_id, row.prompt_version)
        for row in outputs
    }
    call_keys = {
        (row.run_id, row.case_id, row.task, row.model_id, row.prompt_id, row.prompt_version)
        for row in calls
    }
    for row in scores:
        if row.metric == "current_version":
            continue
        key = (row.run_id, row.case_id, row.task, row.model_id, row.prompt_id, row.prompt_version)
        assert key in output_keys
        assert key in call_keys
    assert all(getattr(row, field) for row in scores for field in JOIN_KEYS)


def test_triage_human_boundary_passes_for_both_models() -> None:
    scores = [ScoreRecord.model_validate(row) for row in _load_jsonl(SCORES_PATH)]
    boundary = [
        row
        for row in scores
        if row.task == "triage" and row.metric == "human_boundary"
    ]
    models = {row.model_name for row in boundary}
    assert models == {"mistral", "qwen"}
    for model_name in models:
        rows = [row for row in boundary if row.model_name == model_name]
        assert sum(row.numerator for row in rows) == 12
        assert sum(row.denominator for row in rows) == 12


def test_comparison_and_decision_state_limits_and_cost() -> None:
    comparison = COMPARISON_PATH.read_text(encoding="utf-8")
    decision = DECISION_PATH.read_text(encoding="utf-8")
    assert RUN_ID in comparison
    assert RUN_ID in decision
    assert "$0.00" in comparison
    assert "transfer" in comparison
    assert "Twelve cases per task" in comparison
    assert "human_boundary" in comparison
    assert "Do nothing" in decision
    assert "%" not in comparison.split("## Limits")[0]
