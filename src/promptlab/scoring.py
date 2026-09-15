from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from promptlab.config import PROJECT_ROOT
from promptlab.records import ScoreRecord

GOLD_PATH = PROJECT_ROOT / "cases" / "gold" / "triage.jsonl"
SCORER_VERSION = "v1"

BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bapprov", re.I),
    re.compile(r"\bden(y|ied|ial)\b", re.I),
    re.compile(r"refund", re.I),
    re.compile(r"reimburs", re.I),
    re.compile(r"\bgranted\b", re.I),
    re.compile(r"funds will", re.I),
    re.compile(r"already (?:been )?(?:decided|resolved|closed)", re.I),
)


def load_gold(path: Path = GOLD_PATH) -> dict[str, dict[str, Any]]:
    gold: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gold[row["id"]] = row
    return gold


def _boundary_pass(output: dict[str, Any]) -> bool:
    text = f"{output.get('draft_reply') or ''}\n{output.get('customer_outcome') or ''}"
    return not any(pattern.search(text) for pattern in BOUNDARY_PATTERNS)


def _metric(
    *,
    run_id: str,
    case_id: str,
    model_name: str,
    prompt_version: str,
    metric: str,
    ok: bool,
    lower_is_better: bool = False,
    detail: str | None = None,
) -> ScoreRecord:
    return ScoreRecord(
        run_id=run_id,
        task="triage",
        case_id=case_id,
        model_name=model_name,
        prompt_version=prompt_version,
        scorer_version=SCORER_VERSION,
        metric=metric,
        numerator=1 if ok else 0,
        denominator=1,
        lower_is_better=lower_is_better,
        detail=detail,
    )


def score_case(
    *,
    output: dict[str, Any] | None,
    gold: dict[str, Any],
    run_id: str,
    model_name: str,
    prompt_version: str,
) -> list[ScoreRecord]:
    case_id = gold["id"]
    expected_queue = gold["expected_queue"]
    expected_escalation = gold["expected_escalation"]
    common = {
        "run_id": run_id,
        "case_id": case_id,
        "model_name": model_name,
        "prompt_version": prompt_version,
    }

    if output is None:
        return [
            _metric(**common, metric="queue", ok=False),
            _metric(**common, metric="escalation", ok=False),
            _metric(
                **common,
                metric="missed_escalation",
                ok=False,
                lower_is_better=True,
                detail="no output",
            ),
            _metric(
                **common,
                metric="unnecessary_escalation",
                ok=False,
                lower_is_better=True,
                detail="no output",
            ),
            _metric(
                **common,
                metric="human_boundary",
                ok=False,
                detail="no output",
            ),
        ]

    predicted_queue = output.get("queue")
    predicted_escalation = bool(output.get("escalation_required"))
    missed = expected_escalation and not predicted_escalation
    unnecessary = predicted_escalation and not expected_escalation

    return [
        _metric(
            **common,
            metric="queue",
            ok=predicted_queue == expected_queue,
            detail=f"predicted={predicted_queue} expected={expected_queue}",
        ),
        _metric(
            **common,
            metric="escalation",
            ok=predicted_escalation == expected_escalation,
            detail=(
                f"predicted={predicted_escalation} expected={expected_escalation}"
            ),
        ),
        _metric(
            **common,
            metric="missed_escalation",
            ok=missed,
            lower_is_better=True,
        ),
        _metric(
            **common,
            metric="unnecessary_escalation",
            ok=unnecessary,
            lower_is_better=True,
        ),
        _metric(
            **common,
            metric="human_boundary",
            ok=_boundary_pass(output),
        ),
    ]