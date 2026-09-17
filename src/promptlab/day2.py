"""Day 2: run the baseline prompt on summarization cases across both local models."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

from promptlab.adapters.base import CompletionRequest
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.usage import CallRecord

CASES_PATH = PROJECT_ROOT / "cases" / "summarization.jsonl"
PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "baseline.v0.md"
EVIDENCE_PATH = PROJECT_ROOT / "docs" / "day2-run.jsonl"
COMPARISON_PATH = PROJECT_ROOT / "docs" / "day2-comparison.md"
PROMPT_ID = "baseline"
PROMPT_VERSION = "v0"
MAX_OUTPUT_TOKENS = 512


class SummarizationCase(TypedDict):
    id: str
    task: str
    source: str


def load_cases(path: Path) -> list[SummarizationCase]:
    cases: list[SummarizationCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw: Any = json.loads(line)
        if not isinstance(raw, dict):
            raise TypeError("each summarization case must be a JSON object")
        case_id = raw.get("id")
        task = raw.get("task")
        source = raw.get("source")
        if not isinstance(case_id, str) or not isinstance(task, str) or not isinstance(source, str):
            raise TypeError("summarization case is missing required string fields")
        cases.append(SummarizationCase(id=case_id, task=task, source=source))
    if len(cases) != 12:
        raise ValueError(f"expected 12 summarization cases, found {len(cases)}")
    return cases


def split_baseline_prompt(template: str, document_text: str) -> tuple[str, str]:
    marker = "<document>"
    if marker not in template:
        return template.strip(), document_text
    system = template.split(marker, 1)[0].strip()
    return system, document_text


def write_jsonl(records: list[CallRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


TRUNCATION_APPENDIX = """
## Truncation evidence from the reasoning-enabled run

An earlier run (`run_id=2c7d725b-84b6-4806-9b39-5cb32b3c15cc`) used the identical
prompt, temperature, and 512-token ceiling but left Qwen's native reasoning
output enabled. Under that configuration Qwen truncated on two of twelve cases:

| Case | Model | input_tokens | output_tokens | stop_reason | error_type |
|---|---|---|---|---|---|
| S01 | `qwen3:8b` | 238 | 512 | `length` | `TruncatedResponseError` |
| S02 | `qwen3:8b` | 225 | 512 | `length` | `TruncatedResponseError` |

Both attempts consumed the entire output budget; S01 returned no answer text at
all and S02 returned a partial field list. Mistral completed the same S01
document in 169 output tokens, so document length was not the cause. Reasoning
tokens are billed against the same `num_predict` ceiling as the answer, which is
why only the hybrid-reasoning model hit the limit.

Two consequences worth recording. First, truncation was classified as
`TruncatedResponseError` and was not retried, which is correct: a second
identical attempt would exhaust the same budget. Second, with reasoning enabled
Qwen's median successful latency was 17604 ms against 5152 ms for Mistral, and
its successful cases averaged roughly 400 output tokens. Those totals are not
comparable to Mistral's answer-only tokens, which is why the adapter disables
reasoning output uniformly for the run reported above.
"""


def render_comparison(records: list[CallRecord], settings: Settings) -> str:
    lines = [
        "# Day 2 model comparison",
        "",
        (
            "Both models ran locally through Ollama on the same twelve summarization cases, "
            f"baseline prompt, temperature, and {MAX_OUTPUT_TOKENS}-token output ceiling. The "
            "adapter disables native reasoning output for every model, so the token counts "
            "below measure answer generation rather than hidden reasoning. Provider charge is "
            "$0.00 for both; this comparison uses success counts, tokens, and latency only."
        ),
        "",
        "| Metric | Mistral | Qwen |",
        "| --- | ---: | ---: |",
    ]
    by_name: dict[str, list[CallRecord]] = {}
    for config in settings.models.values():
        by_name[config.logical_name] = [
            record for record in records if record.model_id == config.model_id
        ]

    mistral_rows = by_name.get("mistral", [])
    qwen_rows = by_name.get("qwen", [])
    mistral_ok = [row for row in mistral_rows if row.error_type is None]
    qwen_ok = [row for row in qwen_rows if row.error_type is None]
    lines.extend(
        [
            "| Cases | 12 | 12 |",
            f"| Successful completions | {len(mistral_ok)} / 12 | {len(qwen_ok)} / 12 |",
            "| Truncated (`stop_reason=length`) | "
            f"{sum(1 for row in mistral_rows if row.stop_reason == 'length')} | "
            f"{sum(1 for row in qwen_rows if row.stop_reason == 'length')} |",
            "| Input tokens (successful, sum) | "
            f"{sum(row.input_tokens for row in mistral_ok):,} | "
            f"{sum(row.input_tokens for row in qwen_ok):,} |",
            "| Output tokens (successful, sum) | "
            f"{sum(row.output_tokens for row in mistral_ok):,} | "
            f"{sum(row.output_tokens for row in qwen_ok):,} |",
            "| Median latency (successful, ms) | "
            f"{_median([row.latency_ms for row in mistral_ok]):,.0f} | "
            f"{_median([row.latency_ms for row in qwen_ok]):,.0f} |",
            "| Max latency (successful, ms) | "
            f"{max((row.latency_ms for row in mistral_ok), default=0):,} | "
            f"{max((row.latency_ms for row in qwen_ok), default=0):,} |",
            f"| Observation count | {len(mistral_rows)} | {len(qwen_rows)} |",
            "",
        ]
    )
    for config in settings.models.values():
        model_records = [record for record in records if record.model_id == config.model_id]
        successes = [record for record in model_records if record.error_type is None]
        truncated = [record for record in model_records if record.stop_reason == "length"]
        latencies = [record.latency_ms for record in successes]
        input_tokens = sum(record.input_tokens for record in successes)
        output_tokens = sum(record.output_tokens for record in successes)
        lines.extend(
            [
                f"## {config.logical_name} (`{config.model_id}`)",
                "",
                f"- Successful completions: {len(successes)} / 12",
                f"- Attempts recorded: {len(model_records)}",
                f"- Truncated attempts (output ceiling reached): {len(truncated)}",
                f"- Total input tokens (successful): {input_tokens}",
                f"- Total output tokens (successful): {output_tokens}",
                f"- Median latency (successful): {_median(latencies):.0f} ms",
                f"- Max latency (successful): {max(latencies) if latencies else 0} ms",
                "",
            ]
        )

    configs = list(settings.models.values())
    observation = (
        "Both configured models completed the same twelve cases, so workload differences "
        "come from tokenizer and generation behavior, not different prompts or settings."
    )
    if len(configs) >= 2:
        first = [
            record
            for record in records
            if record.model_id == configs[0].model_id and record.error_type is None
        ]
        second = [
            record
            for record in records
            if record.model_id == configs[1].model_id and record.error_type is None
        ]
        if first and second:
            first_lat = _median([record.latency_ms for record in first])
            second_lat = _median([record.latency_ms for record in second])
            faster = configs[0] if first_lat <= second_lat else configs[1]
            slower = configs[1] if faster is configs[0] else configs[0]
            faster_lat = min(first_lat, second_lat)
            slower_lat = max(first_lat, second_lat)
            observation = (
                f"{faster.logical_name.capitalize()} had the lower median successful latency "
                f"({faster_lat:.0f} ms vs {slower_lat:.0f} ms for {slower.logical_name}). "
                "Token totals also differ across models for identical inputs, so a short sample "
                "from only one model would not estimate the other's workload."
            )
    lines.extend(["## Observation", "", observation, "", TRUNCATION_APPENDIX.strip(), ""])
    return "\n".join(lines)


def main() -> None:
    settings = Settings.from_env()
    template = PROMPT_PATH.read_text(encoding="utf-8")
    cases = load_cases(CASES_PATH)
    run_id = str(uuid4())
    records: list[CallRecord] = []

    for config in settings.models.values():
        adapter = OllamaAdapter(model_id=config.model_id, base_url=settings.ollama_base_url)
        for case in cases:
            system, user_content = split_baseline_prompt(template, case["source"])
            request = CompletionRequest(
                task="summarization",
                case_id=case["id"],
                prompt_id=PROMPT_ID,
                prompt_version=PROMPT_VERSION,
                system=system,
                user_content=user_content,
                temperature=settings.temperature,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            result = adapter.complete(request, run_id)
            records.extend(result.records)
            status = "ok" if result.succeeded else result.error_type
            print(
                f"{config.logical_name} {case['id']}: {status} "
                f"attempts={len(result.records)} "
                f"latency_ms={result.records[-1].latency_ms}"
            )

    write_jsonl(records, EVIDENCE_PATH)
    COMPARISON_PATH.write_text(render_comparison(records, settings), encoding="utf-8")
    print(f"run_id={run_id}")
    print(f"wrote {len(records)} records to {EVIDENCE_PATH}")
    print(f"wrote comparison to {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
