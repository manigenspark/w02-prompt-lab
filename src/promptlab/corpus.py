"""Corpus and gold-label loading for the Week 2 prompt lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from promptlab.config import PROJECT_ROOT

Task = Literal["triage", "summarization", "extraction"]

_CASES_DIR = PROJECT_ROOT / "cases"
_GOLD_DIR = _CASES_DIR / "gold"


class Case(BaseModel):
    """One scored Week 2 case."""

    model_config = ConfigDict(extra="allow")

    id: str
    task: Task
    document_text: str

    @model_validator(mode="before")
    @classmethod
    def normalize_common_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        row = dict(value)
        if "id" not in row and "case_id" in row:
            row["id"] = row["case_id"]
        if "document_text" not in row:
            for key in (
                "source",
                "text",
                "document",
                "content",
                "customer_message",
                "message",
            ):
                if key in row:
                    row["document_text"] = row[key]
                    break
        return row


class GoldLabel(BaseModel):
    """Gold fields consumed by the current deterministic Week 2 scorers."""

    model_config = ConfigDict(extra="allow")

    id: str
    task: Task
    expected_status: str | None = None
    recoverable_fields: list[str] = Field(default_factory=list)
    expected_queue: str | None = None
    expected_escalation: bool | None = None
    version_group: str | None = None
    expected_current_case_id: str | None = None
    as_of: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_id(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        row = dict(value)
        if "id" not in row and "case_id" in row:
            row["id"] = row["case_id"]
        return row


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


def _load_case_rows(task: Task) -> list[dict[str, Any]]:
    path = _CASES_DIR / f"{task}.jsonl"
    rows = _read_jsonl(path)
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row.setdefault("task", task)
        normalized.append(row)
    return normalized


def _load_gold_rows(task: Task) -> list[dict[str, Any]]:
    jsonl_path = _GOLD_DIR / f"{task}.jsonl"
    if jsonl_path.exists():
        rows = _read_jsonl(jsonl_path)
        for row in rows:
            row.setdefault("task", task)
        return rows
    raise FileNotFoundError(jsonl_path)


def load_cases(task: Task) -> list[tuple[Case, GoldLabel]]:
    """Return each case paired with its matching gold label."""
    cases = [Case.model_validate(row) for row in _load_case_rows(task)]
    labels = [GoldLabel.model_validate(row) for row in _load_gold_rows(task)]
    gold_by_id = {label.id: label for label in labels}
    if len(gold_by_id) != len(labels):
        raise ValueError(f"Duplicate gold-label id for task {task!r}")

    pairs: list[tuple[Case, GoldLabel]] = []
    for case in cases:
        gold = gold_by_id.get(case.id)
        if gold is None:
            raise ValueError(f"Missing gold label for task {task!r}, case {case.id!r}")
        pairs.append((case, gold))

    unmatched_gold = set(gold_by_id) - {case.id for case in cases}
    if unmatched_gold:
        raise ValueError(
            f"Gold labels without matching {task!r} cases: {sorted(unmatched_gold)}"
        )
    return pairs


def validate_corpus() -> dict[str, int]:
    """Validate the three fixed Week 2 case sets and return their counts."""
    tasks: tuple[Task, ...] = ("triage", "summarization", "extraction")
    counts: dict[str, int] = {task: len(load_cases(task)) for task in tasks}
    all_ids = [case.id for task in tasks for case, _gold in load_cases(task)]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Case ids must be unique across the Week 2 corpus")
    return counts
