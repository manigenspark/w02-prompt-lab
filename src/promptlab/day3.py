from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.records import OutputRecord
from promptlab.schemas import (
    PolicyExtraction,
    SummarizationOutput,
    schema_description,
)
from promptlab.structured import complete_structured

SUMMARIZATION_CASES = PROJECT_ROOT / "cases" / "summarization.jsonl"
EXTRACTION_CASES = PROJECT_ROOT / "cases" / "extraction.jsonl"
SUMMARIZE_PROMPT = PROJECT_ROOT / "src" / "prompts" / "summarize.v1.md"
EXTRACT_PROMPT = PROJECT_ROOT / "src" / "prompts" / "extract.v2.md"
EVIDENCE_PATH = PROJECT_ROOT / "docs" / "day3-run.jsonl"
MAX_OUTPUT_TOKENS = 2048


class LabCase(TypedDict):
    id: str
    task: str
    source: str


class CountingAdapter:
    def __init__(self, inner: OllamaAdapter) -> None:
        self._inner = inner
        self.provider = inner.provider
        self.model_id = inner.model_id
        self.calls = 0

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        self.calls += 1
        return self._inner.complete(request, run_id)


def load_cases(path: Path, expected: int) -> list[LabCase]:
    cases: list[LabCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw: Any = json.loads(line)
        if not isinstance(raw, dict):
            raise TypeError("each case must be a JSON object")
        case_id = raw.get("id")
        task = raw.get("task")
        source = raw.get("source")
        if not isinstance(case_id, str) or not isinstance(task, str) or not isinstance(source, str):
            raise TypeError("case is missing required string fields")
        cases.append(LabCase(id=case_id, task=task, source=source))
    if len(cases) != expected:
        raise ValueError(f"expected {expected} cases in {path}, found {len(cases)}")
    return cases


def render_prompt(template: str, document_text: str, schema: type[BaseModel]) -> tuple[str, str]:
    filled = template.replace("{schema}", schema_description(schema))
    filled = filled.replace("{document_text}", document_text)
    marker = "<document>"
    if schema is PolicyExtraction:
        system = filled.replace(document_text, "[The source document is in the user message.]")
        system += (
            "\n\nEach EvidenceField must be "
            '{"value": string or list or null, "status": "present"|"absent"|"ambiguous", '
            '"citation": string or null}. Never use keys named text, field, or field_name. '
            'document_status must be exactly valid, contradictory, superseded, or unsupported.'
        )
        user_content = f"<document>\n{document_text}\n</document>"
        return system.strip(), user_content
    if marker not in filled:
        return filled.strip(), document_text
    system, rest = filled.split(marker, 1)
    system += (
        "\n\nEach EvidenceField must be "
        '{"value": string or list or null, "status": "present"|"absent"|"ambiguous", '
        '"citation": string or null}. Never use the key text. '
        "document_status is a string: valid, contradictory, superseded, or unsupported. "
        "required_steps and exceptions are one EvidenceField each, not a list of objects. "
        'When status is absent, still include "value": null.'
    )
    return system.strip(), marker + rest


def run_case(
    *,
    adapter: CountingAdapter,
    template: str,
    schema: type[BaseModel],
    case: LabCase,
    task: str,
    prompt_id: str,
    prompt_version: str,
    run_id: str,
    model_name: str,
    temperature: float,
) -> OutputRecord:
    adapter.calls = 0
    system, user_content = render_prompt(template, case["source"], schema)
    request = CompletionRequest(
        task=task,  # type: ignore[arg-type]
        case_id=case["id"],
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        system=system,
        user_content=user_content,
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    try:
        parsed = complete_structured(adapter, request, schema, run_id, max_repairs=1)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        return OutputRecord(
            run_id=run_id,
            task=task,  # type: ignore[arg-type]
            case_id=case["id"],
            model_name=model_name,
            model_id=adapter.model_id,
            prompt_version=prompt_version,
            succeeded=False,
            repairs=max(0, adapter.calls - 1),
            output=None,
            error=str(exc),
        )
    return OutputRecord(
        run_id=run_id,
        task=task,  # type: ignore[arg-type]
        case_id=case["id"],
        model_name=model_name,
        model_id=adapter.model_id,
        prompt_version=prompt_version,
        succeeded=True,
        repairs=max(0, adapter.calls - 1),
        output=parsed.model_dump(),
        error=None,
    )


def main() -> None:
    settings = Settings.from_env()
    config = settings.models["mistral"]
    inner = OllamaAdapter(model_id=config.model_id, base_url=settings.ollama_base_url)
    adapter = CountingAdapter(inner)
    run_id = str(uuid4())
    records: list[OutputRecord] = []

    summarize_template = SUMMARIZE_PROMPT.read_text(encoding="utf-8")
    extract_template = EXTRACT_PROMPT.read_text(encoding="utf-8")

    for case in load_cases(SUMMARIZATION_CASES, 12):
        record = run_case(
            adapter=adapter,
            template=summarize_template,
            schema=SummarizationOutput,
            case=case,
            task="summarization",
            prompt_id="summarize",
            prompt_version="v1",
            run_id=run_id,
            model_name=config.logical_name,
            temperature=settings.temperature,
        )
        records.append(record)
        print(
            f"summarization {case['id']}: "
            f"{'ok' if record.succeeded else record.error} repairs={record.repairs}"
        )

    for case in load_cases(EXTRACTION_CASES, 12):
        record = run_case(
            adapter=adapter,
            template=extract_template,
            schema=PolicyExtraction,
            case=case,
            task="extraction",
            prompt_id="extract",
            prompt_version="v2",
            run_id=run_id,
            model_name=config.logical_name,
            temperature=settings.temperature,
        )
        records.append(record)
        print(
            f"extraction {case['id']}: "
            f"{'ok' if record.succeeded else record.error} repairs={record.repairs}"
        )

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVIDENCE_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")

    print(f"run_id={run_id}")
    print(f"wrote {len(records)} records to {EVIDENCE_PATH}")


if __name__ == "__main__":
    main()