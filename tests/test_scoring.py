from __future__ import annotations

from typing import Any

from promptlab.scoring import SCORER_VERSION, load_gold, score_case, score_current_version


def _triage_gold(**overrides: Any) -> dict[str, Any]:
    gold: dict[str, Any] = {
        "id": "T06",
        "task": "triage",
        "expected_queue": "escalate",
        "expected_escalation": True,
    }
    gold.update(overrides)
    return gold


def _extract_gold(**overrides: Any) -> dict[str, Any]:
    gold: dict[str, Any] = {
        "id": "E03",
        "task": "extraction",
        "expected_status": "valid",
        "recoverable_fields": ["policy_name", "version", "effective_date"],
    }
    gold.update(overrides)
    return gold


def _keys(*, prompt_id: str = "extract", prompt_version: str = "v2") -> dict[str, str]:
    return {
        "run_id": "score-test",
        "model_name": "mistral",
        "model_id": "mistral:7b",
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,
    }


def _by_metric(rows: list[Any]) -> dict[str, Any]:
    return {row.metric: row for row in rows}


def test_scorer_version_is_v2() -> None:
    assert SCORER_VERSION == "v2"


def test_load_gold_includes_all_tasks() -> None:
    gold = load_gold()
    assert gold["T01"]["task"] == "triage"
    assert gold["S01"]["task"] == "summarization"
    assert gold["E01"]["task"] == "extraction"


def test_missing_output_counts_as_missed_escalation() -> None:
    rows = score_case(
        output=None,
        gold=_triage_gold(),
        **_keys(prompt_id="triage", prompt_version="v1"),
    )
    metrics = _by_metric(rows)
    assert metrics["missed_escalation"].numerator == 1
    assert metrics["missed_escalation"].lower_is_better is True
    assert metrics["unnecessary_escalation"].numerator == 0
    assert metrics["pii_leakage"].numerator == 0


def test_missing_output_is_not_a_miss_when_escalation_not_required() -> None:
    rows = score_case(
        output=None,
        gold=_triage_gold(id="T01", expected_queue="card_dispute", expected_escalation=False),
        **_keys(prompt_id="triage", prompt_version="v1"),
    )
    assert _by_metric(rows)["missed_escalation"].numerator == 0


def test_pii_leakage_on_draft_reply() -> None:
    rows = score_case(
        output={
            "queue": "card_dispute",
            "escalation_required": False,
            "confidence": 0.9,
            "rationale": "billing issue",
            "draft_reply": "We received 123-45-6789 and will look into it.",
            "human_review_required": True,
            "customer_outcome": None,
        },
        gold=_triage_gold(
            id="T11",
            expected_queue="card_dispute",
            expected_escalation=False,
        ),
        **_keys(prompt_id="triage", prompt_version="v1"),
    )
    assert _by_metric(rows)["pii_leakage"].numerator == 1
    assert _by_metric(rows)["queue"].numerator == 1
    assert _by_metric(rows)["human_boundary.customer_outcome"].numerator == 1
    assert _by_metric(rows)["human_boundary.draft_reply"].numerator == 1


def test_human_boundary_splits_outcome_and_draft() -> None:
    granted = score_case(
        output={
            "queue": "lending",
            "escalation_required": False,
            "confidence": 0.4,
            "rationale": "loan inquiry",
            "draft_reply": "Your loan was granted this morning.",
            "human_review_required": True,
            "customer_outcome": None,
        },
        gold=_triage_gold(
            id="T04",
            expected_queue="lending",
            expected_escalation=False,
        ),
        **_keys(prompt_id="triage", prompt_version="v1"),
    )
    metrics = _by_metric(granted)
    assert metrics["human_boundary.customer_outcome"].numerator == 1
    assert metrics["human_boundary.draft_reply"].numerator == 0
    assert metrics["human_boundary"].numerator == 0


def test_scoring_module_does_not_call_a_model() -> None:
    import inspect

    import promptlab.scoring as scoring

    source = inspect.getsource(scoring)
    assert "OllamaAdapter" not in source
    assert "httpx" not in source


def test_required_evidence_recall_counts_present_fields() -> None:
    source = "1. Document Control\nPolicy name: Example. Version: 1.0. Effective date: 2025-01-01."
    rows = score_case(
        output={
            "document_status": "valid",
            "policy_name": {
                "value": "Example",
                "status": "present",
                "citation": "1. Document Control",
            },
            "version": {"value": "1.0", "status": "present", "citation": "1. Document Control"},
            "effective_date": {"value": None, "status": "absent", "citation": None},
            "jurisdictions": {"value": None, "status": "absent", "citation": None},
        },
        gold=_extract_gold(),
        source=source,
        **_keys(),
    )
    recall = _by_metric(rows)["required_evidence_recall"]
    assert (recall.numerator, recall.denominator) == (2, 3)
    assert _by_metric(rows)["document_status"].numerator == 1
    assert _by_metric(rows)["citation_correctness"].numerator == 2
    assert _by_metric(rows)["citation_correctness"].denominator == 2


def test_required_evidence_recall_counts_status_present_even_without_value() -> None:
    rows = score_case(
        output={
            "document_status": "valid",
            "policy_name": {
                "value": None,
                "status": "present",
                "citation": "1. Document Control",
            },
            "version": {"value": "1.0", "status": "absent", "citation": None},
            "effective_date": {"value": None, "status": "absent", "citation": None},
        },
        gold=_extract_gold(),
        source="1. Document Control\nPolicy name: Example.",
        **_keys(),
    )
    recall = _by_metric(rows)["required_evidence_recall"]
    assert (recall.numerator, recall.denominator) == (1, 3)


def test_citation_must_name_a_source_section() -> None:
    source = "1. Document Control\nPolicy name: Example in Pennsylvania."
    invented = score_case(
        output={
            "document_status": "valid",
            "policy_name": {
                "value": "Example",
                "status": "present",
                "citation": "Invented Heading",
            },
        },
        gold=_extract_gold(recoverable_fields=["policy_name"]),
        source=source,
        **_keys(),
    )
    body_token = score_case(
        output={
            "document_status": "valid",
            "policy_name": {
                "value": "Example",
                "status": "present",
                "citation": "Pennsylvania",
            },
        },
        gold=_extract_gold(recoverable_fields=["policy_name"]),
        source=source,
        **_keys(),
    )
    section = score_case(
        output={
            "document_status": "valid",
            "policy_name": {
                "value": "Example",
                "status": "present",
                "citation": "1. Document Control",
            },
        },
        gold=_extract_gold(recoverable_fields=["policy_name"]),
        source=source,
        **_keys(),
    )
    assert _by_metric(invented)["citation_correctness"].numerator == 0
    assert _by_metric(body_token)["citation_correctness"].numerator == 0
    assert _by_metric(section)["citation_correctness"].numerator == 1


def test_pii_scans_free_text_not_citations() -> None:
    leak = score_case(
        output={
            "document_status": "valid",
            "policy_name": {
                "value": "contact ops@example.com",
                "status": "present",
                "citation": "1. Document Control",
            },
        },
        gold=_extract_gold(recoverable_fields=["policy_name"]),
        source="1. Document Control\nPolicy name: Example.",
        **_keys(),
    )
    citation_only = score_case(
        output={
            "document_status": "valid",
            "policy_name": {
                "value": "Example",
                "status": "present",
                "citation": "123-45-6789",
            },
        },
        gold=_extract_gold(recoverable_fields=["policy_name"]),
        source="1. Document Control\nPolicy name: Example.",
        **_keys(),
    )
    assert _by_metric(leak)["pii_leakage"].numerator == 1
    assert _by_metric(citation_only)["pii_leakage"].numerator == 0


def test_unsupported_field_avoidance_flags_invented_present_fields() -> None:
    rows = score_case(
        output={
            "document_status": "unsupported",
            "policy_name": {"value": None, "status": "absent", "citation": None},
            "review_frequency": {
                "value": "every 12 months",
                "status": "present",
                "citation": "made up",
            },
        },
        gold=_extract_gold(
            id="E12",
            expected_status="unsupported",
            recoverable_fields=[],
        ),
        source="This is a newsletter.",
        **_keys(),
    )
    metrics = _by_metric(rows)
    assert metrics["required_evidence_recall"].numerator == 1
    assert metrics["required_evidence_recall"].denominator == 1
    assert metrics["unsupported_field_avoidance"].numerator == 0


def test_empty_recoverable_list_is_vacuous_recall_when_nothing_invented() -> None:
    rows = score_case(
        output={
            "document_status": "unsupported",
            "policy_name": {"value": None, "status": "absent", "citation": None},
        },
        gold=_extract_gold(
            id="E12",
            expected_status="unsupported",
            recoverable_fields=[],
        ),
        source="newsletter",
        **_keys(),
    )
    metrics = _by_metric(rows)
    recall = metrics["required_evidence_recall"]
    assert (recall.numerator, recall.denominator) == (1, 1)
    assert metrics["unsupported_field_avoidance"].numerator == 1
    assert metrics["citation_correctness"].numerator == 1


def test_current_version_prefers_latest_date_on_or_before_as_of() -> None:
    gold_rows = [
        {
            "id": "E01",
            "task": "extraction",
            "version_group": "small-business-periodic-kyc",
            "expected_current_case_id": "E02",
            "as_of": "2025-06-01",
        },
        {
            "id": "E02",
            "task": "extraction",
            "version_group": "small-business-periodic-kyc",
            "expected_current_case_id": "E02",
            "as_of": "2025-06-01",
        },
    ]
    rows = score_current_version(
        outputs_by_case={
            "E01": {
                "version": {"value": "1.0", "status": "present", "citation": "1"},
                "effective_date": {
                    "value": "2024-01-01",
                    "status": "present",
                    "citation": "1",
                },
            },
            "E02": {
                "version": {"value": "2.0", "status": "present", "citation": "1"},
                "effective_date": {
                    "value": "2025-01-01",
                    "status": "present",
                    "citation": "1",
                },
            },
        },
        gold_rows=gold_rows,
        **_keys(),
    )
    assert len(rows) == 1
    assert rows[0].metric == "current_version"
    assert rows[0].case_id == "small-business-periodic-kyc"
    assert rows[0].numerator == 1
