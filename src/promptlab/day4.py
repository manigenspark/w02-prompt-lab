from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.prompts import load, render_user
from promptlab.records import OutputRecord, ScoreRecord
from promptlab.schemas import TriageOutput, TriageOutputWithAnalysis
from promptlab.scoring import load_gold, score_case
from promptlab.structured import complete_structured

CASES_PATH = PROJECT_ROOT / "cases" / "triage.jsonl"
RUN_PATH = PROJECT_ROOT / "docs" / "day4-run.jsonl"
SCORES_PATH = PROJECT_ROOT / "docs" / "day4-scores.jsonl"
MAX_OUTPUT_TOKENS = 1024


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
        cases.append(LabCase(id=raw["id"], task=raw["task"], source=raw["source"]))
    if len(cases) != expected:
        raise ValueError(f"expected {expected} cases in {path}, found {len(cases)}")
    return cases


def run_case(
    *,
    adapter: CountingAdapter,
    prompt_version: str,
    schema: type[BaseModel],
    case: LabCase,
    run_id: str,
    model_name: str,
    temperature: float,
) -> OutputRecord:
    adapter.calls = 0
    template = load("triage", prompt_version)
    request = CompletionRequest(
        task="triage",
        case_id=case["id"],
        prompt_id="triage",
        prompt_version=prompt_version,
        system=template.system,
        user_content=render_user(template, {}, case["source"]),
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    try:
        parsed = complete_structured(adapter, request, schema, run_id, max_repairs=1)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        return OutputRecord(
            run_id=run_id,
            task="triage",
            case_id=case["id"],
            model_name=model_name,
            model_id=adapter.model_id,
            prompt_id="triage",
            prompt_version=prompt_version,
            succeeded=False,
            repairs=max(0, adapter.calls - 1),
            output=None,
            error=str(exc),
        )
    return OutputRecord(
        run_id=run_id,
        task="triage",
        case_id=case["id"],
        model_name=model_name,
        model_id=adapter.model_id,
        prompt_version=prompt_version,
        prompt_id="triage",
        succeeded=True,
        repairs=max(0, adapter.calls - 1),
        output=parsed.model_dump(),
        error=None,
    )


def write_jsonl(path: Path, records: list[OutputRecord] | list[ScoreRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def main() -> None:
    settings = Settings.from_env()
    config = settings.models["mistral"]
    inner = OllamaAdapter(model_id=config.model_id, base_url=settings.ollama_base_url)
    adapter = CountingAdapter(inner)
    run_id = str(uuid4())
    gold = load_gold()
    cases = load_cases(CASES_PATH, 12)

    outputs: list[OutputRecord] = []
    scores: list[ScoreRecord] = []
    versions: list[tuple[str, type[BaseModel]]] = [
        ("v1", TriageOutput),
        ("v2", TriageOutputWithAnalysis),
    ]

    for prompt_version, schema in versions:
        for case in cases:
            record = run_case(
                adapter=adapter,
                prompt_version=prompt_version,
                schema=schema,
                case=case,
                run_id=run_id,
                model_name=config.logical_name,
                temperature=settings.temperature,
            )
            outputs.append(record)
            scores.extend(
                score_case(
                    output=record.output,
                    gold=gold[case["id"]],
                    run_id=run_id,
                    model_name=config.logical_name,
                    prompt_version=prompt_version,
                    model_id=record.model_id,
                    prompt_id="triage",
                    source=case["source"],
                )
            )
            print(
                f"triage {prompt_version} {case['id']}: "
                f"{'ok' if record.succeeded else record.error} repairs={record.repairs}"
            )

    write_jsonl(RUN_PATH, outputs)
    write_jsonl(SCORES_PATH, scores)
    print(f"run_id={run_id}")
    print(f"wrote {len(outputs)} outputs to {RUN_PATH}")
    print(f"wrote {len(scores)} scores to {SCORES_PATH}")


if __name__ == "__main__":
    main()