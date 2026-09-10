"""Day 1: instrument three local Mistral extraction calls and record usage."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

import httpx

from promptlab.config import PROJECT_ROOT, Settings
from promptlab.usage import CallRecord, append_record, compute_cost

CASE_IDS = ("E12", "E07", "E11")
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 256
TRUNCATION_NUM_PREDICT = 8
CASES_PATH = PROJECT_ROOT / "cases" / "extraction.jsonl"
PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "baseline.v0.md"
EVIDENCE_PATH = PROJECT_ROOT / "docs" / "day1-run.jsonl"


class ExtractionCase(TypedDict):
    id: str
    task: str
    source: str


def load_selected_cases(path: Path, case_ids: tuple[str, ...]) -> list[ExtractionCase]:
    by_id: dict[str, ExtractionCase] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw: Any = json.loads(line)
        if not isinstance(raw, dict):
            raise TypeError("each extraction case must be a JSON object")
        case_id = raw.get("id")
        task = raw.get("task")
        source = raw.get("source")
        if not isinstance(case_id, str) or not isinstance(task, str) or not isinstance(source, str):
            raise TypeError("extraction case is missing required string fields")
        by_id[case_id] = ExtractionCase(id=case_id, task=task, source=source)
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise KeyError(f"missing extraction cases: {missing}")
    return [by_id[case_id] for case_id in case_ids]


def _as_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int for {field_name}, got {type(value)!r}")
    return value


def generate(
    *,
    base_url: str,
    model_id: str,
    prompt: str,
    temperature: float,
    num_predict: int,
) -> tuple[dict[str, Any], int]:
    started = time.perf_counter()
    response = httpx.post(
        f"{base_url}/api/generate",
        json={
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        },
        timeout=180.0,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    response.raise_for_status()
    payload: Any = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Ollama response must be a JSON object")
    return payload, latency_ms


def record_from_payload(
    *,
    payload: dict[str, Any],
    run_id: str,
    model_id: str,
    case_id: str,
    attempt: int,
    temperature: float,
    max_output_tokens: int,
    latency_ms: int,
    error_type: str | None,
) -> CallRecord:
    input_tokens = _as_int(payload.get("prompt_eval_count"), "prompt_eval_count")
    output_tokens = _as_int(payload.get("eval_count"), "eval_count")
    stop_reason = payload.get("done_reason")
    response_text = payload.get("response")
    return CallRecord(
        record_id=str(uuid4()),
        run_id=run_id,
        timestamp=datetime.now(UTC),
        provider="ollama",
        model_id=model_id,
        task="extraction",
        case_id=case_id,
        prompt_id="baseline",
        prompt_version="v0",
        attempt=attempt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=None,
        latency_ms=latency_ms,
        cost_usd=compute_cost(model_id, input_tokens, output_tokens),
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
        error_type=error_type,
        response_text=response_text if isinstance(response_text, str) else None,
    )


def write_evidence(records: list[CallRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def main() -> None:
    settings = Settings.from_env()
    model_id = settings.models["mistral"].model_id
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    cases = load_selected_cases(CASES_PATH, CASE_IDS)
    run_id = str(uuid4())
    successful: list[CallRecord] = []

    for case in cases:
        prompt = prompt_template.replace("{document_text}", case["source"])
        payload, latency_ms = generate(
            base_url=settings.ollama_base_url,
            model_id=model_id,
            prompt=prompt,
            temperature=TEMPERATURE,
            num_predict=MAX_OUTPUT_TOKENS,
        )
        record = record_from_payload(
            payload=payload,
            run_id=run_id,
            model_id=model_id,
            case_id=case["id"],
            attempt=1,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            latency_ms=latency_ms,
            error_type=None,
        )
        append_record(record, run_id)
        successful.append(record)
        print(
            f"{case['id']}: input_tokens={record.input_tokens} "
            f"output_tokens={record.output_tokens} latency_ms={record.latency_ms} "
            f"stop_reason={record.stop_reason}"
        )

    truncation_case = next(case for case in cases if case["id"] == "E11")
    truncation_prompt = prompt_template.replace("{document_text}", truncation_case["source"])
    truncation_payload, truncation_latency_ms = generate(
        base_url=settings.ollama_base_url,
        model_id=model_id,
        prompt=truncation_prompt,
        temperature=TEMPERATURE,
        num_predict=TRUNCATION_NUM_PREDICT,
    )
    truncation_stop = truncation_payload.get("done_reason")
    if truncation_stop == "length":
        truncation_record = record_from_payload(
            payload=truncation_payload,
            run_id=run_id,
            model_id=model_id,
            case_id=truncation_case["id"],
            attempt=2,
            temperature=TEMPERATURE,
            max_output_tokens=TRUNCATION_NUM_PREDICT,
            latency_ms=truncation_latency_ms,
            error_type="TruncatedResponseError",
        )
        append_record(truncation_record, run_id)
        print(
            f"E11 truncation demo: stop_reason={truncation_record.stop_reason} "
            f"error_type={truncation_record.error_type} "
            f"output_tokens={truncation_record.output_tokens}"
        )
    else:
        print(
            f"E11 truncation demo did not hit the output ceiling "
            f"(done_reason={truncation_stop!r})"
        )

    write_evidence(successful, EVIDENCE_PATH)
    print(f"run_id={run_id}")
    print(f"wrote {len(successful)} successful records to {EVIDENCE_PATH}")


if __name__ == "__main__":
    main()
