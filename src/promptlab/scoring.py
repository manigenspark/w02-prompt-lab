"""Deterministic gold scoring. No LLM.

Required-evidence recall — summarization and extraction. Compare
gold.recoverable_fields with fields returned as status="present".

Citation correctness — summarization and extraction. For each field
returned as status="present", the citation must name a section that
actually appears in the source.

Personal-data leakage — applicable free-text outputs only
(draft_reply, rationale, evidence values). Scan with PII_PATTERNS.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, cast

from promptlab.config import PII_PATTERNS, PROJECT_ROOT
from promptlab.records import ScoreRecord
from promptlab.rules import VersionCandidate, select_current_version
from promptlab.schemas import TaskName

GOLD_DIR = PROJECT_ROOT / "cases" / "gold"
GOLD_PATH = GOLD_DIR / "triage.jsonl"
SCORER_VERSION = "v2"

BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bapprov", re.I),
    re.compile(r"\bden(y|ied|ial)\b", re.I),
    re.compile(r"refund", re.I),
    re.compile(r"reimburs", re.I),
    re.compile(r"\bgranted\b", re.I),
    re.compile(r"funds will", re.I),
    re.compile(r"already (?:been )?(?:decided|resolved|closed)", re.I),
)


def _load_gold_file(path: Path) -> dict[str, dict[str, Any]]:
    gold: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gold[row["id"]] = row
    return gold


def load_gold(path: Path | None = None) -> dict[str, dict[str, Any]]:
    if path is not None:
        return _load_gold_file(path)
    gold: dict[str, dict[str, Any]] = {}
    for name in ("triage", "summarization", "extraction"):
        gold.update(_load_gold_file(GOLD_DIR / f"{name}.jsonl"))
    return gold


def _outcome_is_null(output: dict[str, Any]) -> bool:
    return output.get("customer_outcome") is None


def _draft_respects_boundary(output: dict[str, Any]) -> bool:
    text = str(output.get("draft_reply") or "")
    return not any(pattern.search(text) for pattern in BOUNDARY_PATTERNS)


_HEADING_LINE = re.compile(r"^(?:#{1,6}\s+|\d+\.\s+)\S")
_HEADING_PREFIX = re.compile(r"^(?:#{1,6}\s+|\d+\.\s+)")


def _evidence_fields(output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for key, raw in output.items():
        if isinstance(raw, dict) and "status" in raw:
            fields[key] = raw
    return fields


def _is_present(field: dict[str, Any]) -> bool:
    """Recall/citation/avoidance all key off status='present', not value."""
    return field.get("status") == "present"


def _source_headings(source: str) -> list[str]:
    """Numbered or markdown headings that a citation may name."""
    headings: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if _HEADING_LINE.match(stripped):
            headings.append(stripped)
    return headings


def _citation_names_section(citation: str, source: str) -> bool:
    """True when the citation names a section heading that appears in the source."""
    cite = citation.strip()
    if not cite:
        return False
    headings = _source_headings(source)
    if not headings:
        return cite in source
    for heading in headings:
        title = _HEADING_PREFIX.sub("", heading).strip()
        if cite in {heading, title}:
            return True
        if heading.startswith(cite) or title.startswith(cite):
            return True
        if cite.startswith(heading) or (title and cite.startswith(title)):
            return True
    return False


def _citation_ok(field: dict[str, Any], source: str) -> bool:
    citation = field.get("citation")
    if not isinstance(citation, str):
        return False
    return _citation_names_section(citation, source)


def _value_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _free_text_strings(output: dict[str, Any]) -> list[str]:
    """Draft/rationale plus evidence values. Citations and enums are not free text."""
    texts: list[str] = []
    for key in ("draft_reply", "rationale"):
        raw = output.get(key)
        if isinstance(raw, str):
            texts.append(raw)
    for field in _evidence_fields(output).values():
        texts.extend(_value_strings(field.get("value")))
    return texts


def _pii_leak(output: dict[str, Any]) -> bool:
    """Deterministic PII_PATTERNS scan. Does not call a model."""
    return any(
        pattern.search(text)
        for text in _free_text_strings(output)
        for pattern in PII_PATTERNS
    )


def _ratio(recovered: int, total: int) -> tuple[int, int]:
    if total == 0:
        return 1, 1
    return recovered, total


def _metric(
    *,
    run_id: str,
    case_id: str,
    task: TaskName,
    model_name: str,
    model_id: str,
    prompt_id: str,
    prompt_version: str,
    metric: str,
    ok: bool | None = None,
    numerator: int | None = None,
    denominator: int | None = None,
    lower_is_better: bool = False,
    detail: str | None = None,
) -> ScoreRecord:
    if numerator is None or denominator is None:
        if ok is None:
            raise ValueError("ok or numerator/denominator required")
        numerator = 1 if ok else 0
        denominator = 1
    return ScoreRecord(
        run_id=run_id,
        task=task,
        case_id=case_id,
        model_name=model_name,
        model_id=model_id,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        scorer_version=SCORER_VERSION,
        metric=metric,
        numerator=numerator,
        denominator=denominator,
        lower_is_better=lower_is_better,
        detail=detail,
    )


def _score_triage(
    *,
    output: dict[str, Any] | None,
    gold: dict[str, Any],
    common: dict[str, Any],
) -> list[ScoreRecord]:
    expected_queue = gold["expected_queue"]
    expected_escalation = bool(gold["expected_escalation"])

    if output is None:
        return [
            _metric(**common, metric="queue", ok=False),
            _metric(**common, metric="escalation", ok=False),
            _metric(
                **common,
                metric="missed_escalation",
                ok=expected_escalation,
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
                metric="human_boundary.customer_outcome",
                ok=False,
                detail="no output",
            ),
            _metric(
                **common,
                metric="human_boundary.draft_reply",
                ok=False,
                detail="no output",
            ),
            _metric(
                **common,
                metric="human_boundary",
                ok=False,
                detail="no output",
            ),
            _metric(
                **common,
                metric="pii_leakage",
                ok=False,
                lower_is_better=True,
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
            detail=(f"predicted={predicted_escalation} expected={expected_escalation}"),
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
            metric="human_boundary.customer_outcome",
            ok=_outcome_is_null(output),
        ),
        _metric(
            **common,
            metric="human_boundary.draft_reply",
            ok=_draft_respects_boundary(output),
        ),
        _metric(
            **common,
            metric="human_boundary",
            ok=_outcome_is_null(output) and _draft_respects_boundary(output),
        ),
        _metric(
            **common,
            metric="pii_leakage",
            ok=_pii_leak(output),
            lower_is_better=True,
        ),
    ]


def _score_evidence(
    *,
    output: dict[str, Any] | None,
    gold: dict[str, Any],
    source: str,
    common: dict[str, Any],
) -> list[ScoreRecord]:
    recoverable = list(gold.get("recoverable_fields") or [])
    expected_status = gold.get("expected_status")

    if output is None:
        recall_n, recall_d = _ratio(0, len(recoverable))
        return [
            _metric(**common, metric="document_status", ok=False, detail="no output"),
            _metric(
                **common,
                metric="required_evidence_recall",
                numerator=recall_n,
                denominator=recall_d,
                detail="no output",
            ),
            _metric(
                **common,
                metric="citation_correctness",
                numerator=0,
                denominator=1,
                detail="no output",
            ),
            _metric(
                **common,
                metric="unsupported_field_avoidance",
                ok=False,
                detail="no output",
            ),
            _metric(
                **common,
                metric="pii_leakage",
                ok=False,
                lower_is_better=True,
                detail="no output",
            ),
        ]

    fields = _evidence_fields(output)
    # Required-evidence recall: gold.recoverable_fields vs status="present".
    recovered = sum(1 for name in recoverable if name in fields and _is_present(fields[name]))
    recall_n, recall_d = _ratio(recovered, len(recoverable))

    # Citation correctness: each status="present" field must name a source section.
    present_fields = [field for field in fields.values() if _is_present(field)]
    cited = sum(1 for field in present_fields if _citation_ok(field, source))
    cite_n, cite_d = _ratio(cited, len(present_fields))

    extra_present = [
        name
        for name, field in fields.items()
        if name not in recoverable and _is_present(field)
    ]

    return [
        _metric(
            **common,
            metric="document_status",
            ok=output.get("document_status") == expected_status,
            detail=(
                f"predicted={output.get('document_status')} expected={expected_status}"
            ),
        ),
        _metric(
            **common,
            metric="required_evidence_recall",
            numerator=recall_n,
            denominator=recall_d,
            detail=f"recovered={recovered} recoverable={len(recoverable)}",
        ),
        _metric(
            **common,
            metric="citation_correctness",
            numerator=cite_n,
            denominator=cite_d,
            detail=f"cited={cited} present={len(present_fields)}",
        ),
        _metric(
            **common,
            metric="unsupported_field_avoidance",
            ok=not extra_present,
            detail=None if not extra_present else f"invented={extra_present}",
        ),
        _metric(
            **common,
            metric="pii_leakage",
            ok=_pii_leak(output),
            lower_is_better=True,
        ),
    ]


def score_case(
    *,
    output: dict[str, Any] | None,
    gold: dict[str, Any],
    run_id: str,
    model_name: str,
    model_id: str,
    prompt_id: str,
    prompt_version: str,
    source: str = "",
) -> list[ScoreRecord]:
    task = cast(TaskName, gold["task"])
    common = {
        "run_id": run_id,
        "case_id": gold["id"],
        "task": task,
        "model_name": model_name,
        "model_id": model_id,
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,
    }
    if task == "triage":
        return _score_triage(output=output, gold=gold, common=common)
    if task in ("summarization", "extraction"):
        return _score_evidence(output=output, gold=gold, source=source, common=common)
    raise ValueError(f"unsupported task: {task}")


def _parse_iso_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    text = value.strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def score_current_version(
    *,
    outputs_by_case: dict[str, dict[str, Any] | None],
    gold_rows: list[dict[str, Any]],
    run_id: str,
    model_name: str,
    model_id: str,
    prompt_id: str,
    prompt_version: str,
) -> list[ScoreRecord]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in gold_rows:
        group = row.get("version_group")
        if not group:
            continue
        grouped.setdefault(str(group), []).append(row)

    scores: list[ScoreRecord] = []
    for group, rows in grouped.items():
        as_of_raw = rows[0].get("as_of")
        expected = rows[0].get("expected_current_case_id")
        as_of = _parse_iso_date(as_of_raw)
        task = cast(TaskName, rows[0]["task"])
        candidates: list[VersionCandidate] = []
        if as_of is not None:
            for row in rows:
                case_id = str(row["id"])
                output = outputs_by_case.get(case_id)
                if not output:
                    continue
                fields = _evidence_fields(output)
                effective = _parse_iso_date((fields.get("effective_date") or {}).get("value"))
                version_value = (fields.get("version") or {}).get("value")
                version = version_value if isinstance(version_value, str) else ""
                if effective is None:
                    continue
                candidates.append(
                    VersionCandidate(
                        case_id=case_id,
                        version=version,
                        effective_date=effective,
                    )
                )
        selected = select_current_version(candidates, as_of) if as_of is not None else None
        predicted = selected.case_id if selected is not None else None
        scores.append(
            _metric(
                run_id=run_id,
                case_id=group,
                task=task,
                model_name=model_name,
                model_id=model_id,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                metric="current_version",
                ok=predicted == expected,
                detail=f"predicted={predicted} expected={expected}",
            )
        )
    return scores
