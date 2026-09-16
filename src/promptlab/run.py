from __future__ import annotations

import argparse
import json
from typing import Any

from pydantic import BaseModel, ValidationError

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.corpus import load_cases, validate_corpus
from promptlab.prompts import load, render_user
from promptlab.records import OutputRecord, ScoreRecord, append_record, load_records
from promptlab.report import write_reports
from promptlab.schemas import (
    PolicyExtraction,
    SummarizationOutput,
    TaskName,
    TriageOutput,
    schema_description,
)
from promptlab.scoring import score_case, score_current_version
from promptlab.structured import complete_structured
from promptlab.usage import CallRecord

RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "day5-run.jsonl"
SCORES_PATH = PROJECT_ROOT / "docs" / "day5-scores.jsonl"
CALLS_PATH = PROJECT_ROOT / "docs" / "day5-calls.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports" / "comparison.md"
DECISION_PATH = PROJECT_ROOT / "docs" / "model-decision.md"

TASK_SPECS: dict[TaskName, dict[str, Any]] = {
    "summarization": {
        "prompt_id": "summarize",
        "prompt_version": "v1",
        "schema": SummarizationOutput,
        "max_output_tokens": 2048,
        "schema_variable": False,
    },
    "extraction": {
        "prompt_id": "extract",
        "prompt_version": "v2",
        "schema": PolicyExtraction,
        "max_output_tokens": 2048,
        "schema_variable": True,
    },
    "triage": {
        "prompt_id": "triage",
        "prompt_version": "v1",
        "schema": TriageOutput,
        "max_output_tokens": 1024,
        "schema_variable": False,
    },
}

JSON_INSTANCE_CONTRACT = (
    "Return only a JSON instance of the requested schema. "
    "Do not return JSON Schema. Do not include $defs, properties, type, "
    "required, title, or additionalProperties. Do not use Markdown fences. "
    "Do not add keys that are not in the schema."
)

EVIDENCE_CONTRACT = (
    "Each EvidenceField must be "
    '{"value": string or list or null, "status": "present"|"absent"|"ambiguous", '
    '"citation": string or null}. Never use keys named text, field, or field_name. '
    "document_status must be exactly valid, contradictory, superseded, or unsupported. "
    "required_steps and exceptions are one EvidenceField each, not a list of objects. "
    'When status is absent, still include "value": null. '
    'A present citation must be the full heading line copied from the source '
    '(for example "1. Document Control"), not a number, label, or invented heading.'
)

TRIAGE_INSTANCE_CONTRACT = (
    "confidence must be a number from 0.0 to 1.0, never a word such as high or medium. "
    "human_review_required must be true. customer_outcome must be null. "
    "draft_reply must not approve, deny, refund, reimburse, grant, or say an outcome "
    "is already decided."
)


def _system_text(template_system: str, task: TaskName, schema: type[BaseModel]) -> str:
    parts = [template_system.strip(), JSON_INSTANCE_CONTRACT]
    if task in ("summarization", "extraction"):
        parts.append(schema_description(schema))
        parts.append(EVIDENCE_CONTRACT)
    if task == "triage":
        parts.append(TRIAGE_INSTANCE_CONTRACT)
    return "\n\n".join(part for part in parts if part)


class CountingAdapter:
    def __init__(self, inner: OllamaAdapter) -> None:
        self._inner = inner
        self.provider = inner.provider
        self.model_id = inner.model_id
        self.calls = 0

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        self.calls += 1
        return self._inner.complete(request, run_id)


def _run_case(
    *,
    adapter: CountingAdapter,
    spec: dict[str, Any],
    case_id: str,
    task: TaskName,
    document_text: str,
    run_id: str,
    model_name: str,
    temperature: float,
) -> OutputRecord:
    adapter.calls = 0
    prompt_id = str(spec["prompt_id"])
    prompt_version = str(spec["prompt_version"])
    schema: type[BaseModel] = spec["schema"]
    template = load(prompt_id, prompt_version)
    variables: dict[str, str] = {}
    if spec["schema_variable"]:
        variables["schema"] = schema_description(schema)
    request = CompletionRequest(
        task=task,
        case_id=case_id,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        system=_system_text(template.system, task, schema),
        user_content=render_user(template, variables, document_text),
        temperature=temperature,
        max_output_tokens=int(spec["max_output_tokens"]),
    )
    try:
        parsed = complete_structured(adapter, request, schema, run_id, max_repairs=1)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        return OutputRecord(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            model_id=adapter.model_id,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            succeeded=False,
            repairs=max(0, adapter.calls - 1),
            output=None,
            error=str(exc),
        )
    return OutputRecord(
        run_id=run_id,
        task=task,
        case_id=case_id,
        model_name=model_name,
        model_id=adapter.model_id,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        succeeded=True,
        repairs=max(0, adapter.calls - 1),
        output=parsed.model_dump(),
        error=None,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 2 local model comparison harness")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", choices=["triage", "summarization", "extraction"])
    parser.add_argument("--model", choices=["mistral", "qwen"])
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild comparison.md and model-decision.md from existing Day 5 files",
    )
    return parser.parse_args(argv)


def _selected_tasks(task: str | None) -> list[TaskName]:
    mapping: dict[str, TaskName] = {
        "summarization": "summarization",
        "extraction": "extraction",
        "triage": "triage",
    }
    if task is None:
        return ["summarization", "extraction", "triage"]
    return [mapping[task]]


def _copy_calls(run_id: str) -> list[CallRecord]:
    usage_path = RUNS_DIR / f"{run_id}.jsonl"
    usage = load_records(usage_path, CallRecord) if usage_path.exists() else []
    if usage:
        CALLS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALLS_PATH.write_text(
            "\n".join(record.model_dump_json() for record in usage) + "\n",
            encoding="utf-8",
        )
    return usage


def _write_from_records(run_id: str, model_names: list[str]) -> None:
    outputs = load_records(OUTPUT_PATH, OutputRecord)
    scores = load_records(SCORES_PATH, ScoreRecord)
    usage = _copy_calls(run_id)
    write_reports(
        run_id=run_id,
        models=model_names,
        usage=usage,
        outputs=outputs,
        scores=scores,
        report_path=REPORT_PATH,
        decision_path=DECISION_PATH,
    )
    print(f"run_id={run_id}")
    print(f"wrote {len(outputs)} outputs to {OUTPUT_PATH}")
    print(f"wrote {len(scores)} scores to {SCORES_PATH}")
    if CALLS_PATH.exists():
        print(f"wrote {CALLS_PATH}")
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {DECISION_PATH}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    validate_corpus()
    settings = Settings.from_env()
    tasks = _selected_tasks(args.task)
    model_names = [args.model] if args.model else ["mistral", "qwen"]
    run_id = str(args.run_id)
    if args.report_only:
        _write_from_records(run_id, model_names)
        return
    for path in (OUTPUT_PATH, SCORES_PATH, CALLS_PATH):
        if path.exists():
            path.unlink()

    outputs: list[OutputRecord] = []
    scores: list[ScoreRecord] = []

    for model_name in model_names:
        config = settings.models[model_name]
        inner = OllamaAdapter(model_id=config.model_id, base_url=settings.ollama_base_url)
        adapter = CountingAdapter(inner)
        for task in tasks:
            spec = TASK_SPECS[task]
            pairs = load_cases(task)
            gold_rows: list[dict[str, Any]] = []
            outputs_by_case: dict[str, dict[str, Any] | None] = {}
            for case, gold in pairs:
                record = _run_case(
                    adapter=adapter,
                    spec=spec,
                    case_id=case.id,
                    task=task,
                    document_text=case.document_text,
                    run_id=run_id,
                    model_name=config.logical_name,
                    temperature=settings.temperature,
                )
                outputs.append(record)
                append_record(OUTPUT_PATH, record)
                outputs_by_case[case.id] = record.output
                gold_row = gold.model_dump()
                gold_rows.append(gold_row)
                case_scores = score_case(
                    output=record.output,
                    gold=gold_row,
                    run_id=run_id,
                    model_name=config.logical_name,
                    model_id=adapter.model_id,
                    prompt_id=str(spec["prompt_id"]),
                    prompt_version=str(spec["prompt_version"]),
                    source=case.document_text,
                )
                scores.extend(case_scores)
                for score in case_scores:
                    append_record(SCORES_PATH, score)
                print(
                    f"{task} {model_name} {case.id}: "
                    f"{'ok' if record.succeeded else record.error} repairs={record.repairs}"
                )

            version_scores = score_current_version(
                outputs_by_case=outputs_by_case,
                gold_rows=gold_rows,
                run_id=run_id,
                model_name=config.logical_name,
                model_id=adapter.model_id,
                prompt_id=str(spec["prompt_id"]),
                prompt_version=str(spec["prompt_version"]),
            )
            scores.extend(version_scores)
            for score in version_scores:
                append_record(SCORES_PATH, score)

    usage = _copy_calls(run_id)
    write_reports(
        run_id=run_id,
        models=model_names,
        usage=usage,
        outputs=outputs,
        scores=scores,
        report_path=REPORT_PATH,
        decision_path=DECISION_PATH,
    )
    print(f"run_id={run_id}")
    print(f"wrote {len(outputs)} outputs to {OUTPUT_PATH}")
    print(f"wrote {len(scores)} scores to {SCORES_PATH}")
    if CALLS_PATH.exists():
        print(f"wrote {CALLS_PATH}")
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {DECISION_PATH}")


if __name__ == "__main__":
    main()
