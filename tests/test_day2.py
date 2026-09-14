"""Day 2 request-construction checks. No live Ollama calls."""

from __future__ import annotations

from promptlab.adapters.base import CompletionRequest
from promptlab.config import Settings
from promptlab.day2 import (
    CASES_PATH,
    COMPARISON_PATH,
    EVIDENCE_PATH,
    MAX_OUTPUT_TOKENS,
    PROMPT_ID,
    PROMPT_PATH,
    PROMPT_VERSION,
    load_cases,
    split_baseline_prompt,
)
from promptlab.usage import CallRecord


def test_summarization_corpus_has_twelve_cases() -> None:
    cases = load_cases(CASES_PATH)
    assert len(cases) == 12
    assert [case["id"] for case in cases] == [f"S{index:02d}" for index in range(1, 13)]


def test_shared_request_fields_match_across_configured_models() -> None:
    settings = Settings.from_env()
    template = PROMPT_PATH.read_text(encoding="utf-8")
    case = load_cases(CASES_PATH)[0]
    system, user_content = split_baseline_prompt(template, case["source"])
    requests = [
        CompletionRequest(
            task="summarization",
            case_id=case["id"],
            prompt_id=PROMPT_ID,
            prompt_version=PROMPT_VERSION,
            system=system,
            user_content=user_content,
            temperature=settings.temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        for _config in settings.models.values()
    ]
    assert len(requests) == 2
    left, right = requests
    assert left.task == right.task == "summarization"
    assert left.case_id == right.case_id
    assert left.prompt_id == right.prompt_id
    assert left.prompt_version == right.prompt_version
    assert left.temperature == right.temperature
    assert left.max_output_tokens == right.max_output_tokens
    assert left.system == right.system
    assert left.user_content == right.user_content
    assert "{document_text}" not in left.user_content


def test_day2_evidence_covers_both_models() -> None:
    rows = [
        CallRecord.model_validate_json(line)
        for line in EVIDENCE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    settings = Settings.from_env()
    assert len(rows) == 24
    assert len({row.run_id for row in rows}) == 1
    assert {row.provider for row in rows} == {"ollama"}
    assert all(row.timestamp.tzinfo is not None for row in rows)
    assert all(row.error_type is None for row in rows)
    model_ids = {config.model_id for config in settings.models.values()}
    assert {row.model_id for row in rows} == model_ids
    for config in settings.models.values():
        cases = [row.case_id for row in rows if row.model_id == config.model_id]
        assert cases == [f"S{index:02d}" for index in range(1, 13)]
    comparison = COMPARISON_PATH.read_text(encoding="utf-8").lower()
    assert "successful completions" in comparison
    assert "total input tokens" in comparison
    assert "median latency" in comparison
    assert "max latency" in comparison
