"""Reporting for the Week 2 model-comparison lab.

Consumes CallRecord, OutputRecord, and ScoreRecord. Does not rescore or call a model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

from promptlab.config import Settings
from promptlab.records import OutputRecord, ScoreRecord
from promptlab.usage import CallRecord

_ConfigKey = tuple[str, str, str]

TASK_QUALITY: dict[str, tuple[str, ...]] = {
    "extraction": (
        "document_status",
        "required_evidence_recall",
        "citation_correctness",
        "unsupported_field_avoidance",
        "current_version",
        "pii_leakage",
    ),
    "summarization": (
        "document_status",
        "required_evidence_recall",
        "citation_correctness",
        "unsupported_field_avoidance",
        "current_version",
        "pii_leakage",
    ),
    "triage": (
        "queue",
        "escalation",
        "missed_escalation",
        "unnecessary_escalation",
        "human_boundary",
        "human_boundary.customer_outcome",
        "human_boundary.draft_reply",
        "pii_leakage",
    ),
}

REOPEN_IF: dict[str, str] = {
    "extraction": (
        "a held-out set larger than 12 cases, or an adapted Qwen prompt, beats this "
        "row on document_status or unsupported_field_avoidance without dropping "
        "required-evidence recall or citation correctness, and without adding PII leakage"
    ),
    "summarization": (
        "a held-out set larger than 12 cases, or an adapted prompt, matches 12/12 valid "
        "outputs and evidence recall/citations without adding PII leakage or invented fields"
    ),
    "triage": (
        "a held-out set larger than 12 cases, or an adapted prompt, keeps missed_escalation "
        "at 0/12 and human_boundary at 12/12 while reducing unnecessary escalations without "
        "dropping queue accuracy or adding PII leakage"
    ),
}


def _logical_name(model_id: str, mapping: dict[str, str]) -> str:
    return mapping.get(model_id, model_id)


def _model_id_map() -> dict[str, str]:
    settings = Settings.from_env()
    return {config.model_id: config.logical_name for config in settings.models.values()}


def _key_from_output(record: OutputRecord) -> _ConfigKey:
    return (str(record.task), str(record.model_name), str(record.prompt_version))


def _key_from_score(record: ScoreRecord) -> _ConfigKey:
    return (str(record.task), str(record.model_name), str(record.prompt_version))


def _key_from_call(record: CallRecord, mapping: dict[str, str]) -> _ConfigKey:
    task = "summarization" if record.task == "summarize" else str(record.task)
    return (task, _logical_name(record.model_id, mapping), str(record.prompt_version))


def _for_run(records: Sequence[Any], run_id: str) -> list[Any]:
    return [record for record in records if str(record.run_id) == run_id]


def _fmt_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _prompt_label(
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    prompt_version: str,
    transfer: bool,
) -> str:
    prompt_id = ""
    for output in outputs:
        if output.prompt_id:
            prompt_id = output.prompt_id
            break
    if not prompt_id:
        for score in scores:
            if score.prompt_id:
                prompt_id = score.prompt_id
                break
    label = f"{prompt_id}.{prompt_version}" if prompt_id else prompt_version
    return f"{label} transfer" if transfer else label


def _aggregate_scores(
    records: Sequence[ScoreRecord],
) -> dict[str, tuple[int, int, bool | None]]:
    grouped: dict[str, list[ScoreRecord]] = defaultdict(list)
    for record in records:
        grouped[str(record.metric)].append(record)

    result: dict[str, tuple[int, int, bool | None]] = {}
    for metric, rows in sorted(grouped.items()):
        numerator = sum(int(row.numerator) for row in rows)
        denominator = sum(int(row.denominator) for row in rows)
        directions = {bool(row.lower_is_better) for row in rows}
        lower_is_better = next(iter(directions)) if len(directions) == 1 else None
        result[metric] = (numerator, denominator, lower_is_better)
    return result


def _metric_text(records: Sequence[ScoreRecord]) -> str:
    metrics = _aggregate_scores(records)
    if not metrics:
        return "—"
    rendered: list[str] = []
    for metric, (numerator, denominator, lower_is_better) in metrics.items():
        suffix = " ↓" if lower_is_better else ""
        rendered.append(f"{metric}: {numerator}/{denominator}{suffix}")
    return " ".join(rendered)


def _ratio_text(records: Sequence[ScoreRecord], metric: str) -> str:
    numerator, denominator = _metric_tuple(records, metric)
    if denominator == 0:
        return "—"
    return f"{numerator}/{denominator}"


def _usage_summary(records: Sequence[CallRecord]) -> tuple[int, int, str, str, int, int]:
    if not records:
        return 0, 0, "—", "—", 0, 0
    input_tokens = sum(int(row.input_tokens or 0) for row in records)
    output_tokens = sum(int(row.output_tokens or 0) for row in records)
    latencies = [float(row.latency_ms) for row in records]
    median_latency = f"{_fmt_number(float(median(latencies)))} ms" if latencies else "—"
    max_latency = f"{_fmt_number(float(max(latencies)))} ms" if latencies else "—"
    retries = sum(1 for row in records if int(row.attempt or 1) > 1)
    return input_tokens, output_tokens, median_latency, max_latency, len(latencies), retries


def _output_summary(records: Sequence[OutputRecord]) -> tuple[str, str, str]:
    if not records:
        return "0/0", "0/0", "0"
    total = len(records)
    succeeded = sum(1 for row in records if bool(row.succeeded))
    repairs_needed = sum(1 for row in records if int(row.repairs or 0) > 0)
    return f"{succeeded}/{total}", f"{repairs_needed}/{total}", str(total - succeeded)


def _all_config_keys(
    usage: Sequence[CallRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    mapping: dict[str, str],
) -> list[_ConfigKey]:
    keys = {_key_from_call(row, mapping) for row in usage}
    keys.update(_key_from_output(row) for row in outputs)
    keys.update(_key_from_score(row) for row in scores)
    return sorted(keys)


def _rows_for_key(
    key: _ConfigKey,
    usage: Sequence[CallRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    mapping: dict[str, str],
) -> tuple[list[CallRecord], list[OutputRecord], list[ScoreRecord]]:
    return (
        [row for row in usage if _key_from_call(row, mapping) == key],
        [row for row in outputs if _key_from_output(row) == key],
        [row for row in scores if _key_from_score(row) == key],
    )


def _write_report(
    *,
    run_id: str,
    usage: Sequence[CallRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    report_path: Path,
    mapping: dict[str, str],
) -> None:
    lines: list[str] = [
        "# Model Comparison",
        "",
        f"Run ID: `{run_id}`",
        "",
        "Counts are reported with their denominators, not percentages. "
        "Latency uses median and maximum rather than mean. HTTP-call count is the "
        "observation count for latency (primary attempts plus repairs and transport retries).",
        "",
        "Qwen rows use the same frozen prompt files as Mistral (prompt transfer). "
        "They measure that transferred configuration, not an adapted Qwen prompt.",
        "",
        "Local Ollama provider/API charge is `$0.00`. Token usage and latency still "
        "represent real operational work.",
        "",
    ]

    keys = _all_config_keys(usage, outputs, scores, mapping)
    tasks = sorted({task for task, _model, _prompt in keys})
    if not tasks:
        lines.extend(["No records were supplied for this run.", ""])

    for task in tasks:
        task_keys = [key for key in keys if key[0] == task]
        lines.extend(
            [
                f"## {task.title()}",
                "",
                "| Model | Prompt | Transfer | Valid outputs | Input tokens | "
                "Output tokens | Input tokens/case | Output tokens/case | "
                "Median latency | Max latency | HTTP calls | Repairs | "
                "Retries | Final failures |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for key in task_keys:
            _task, model_name, prompt_version = key
            u, o, s = _rows_for_key(key, usage, outputs, scores, mapping)
            (
                input_tokens,
                output_tokens,
                median_latency,
                max_latency,
                http_calls,
                retries,
            ) = _usage_summary(u)
            valid_outputs, repairs, failures = _output_summary(o)
            cases = len(o) or 1
            transfer = model_name == "qwen"
            prompt = _prompt_label(o, s, prompt_version, transfer)
            lines.append(
                "| "
                f"{model_name} | {prompt} | {'yes' if transfer else 'no'} | "
                f"{valid_outputs} | {input_tokens} | {output_tokens} | "
                f"{_fmt_number(input_tokens / cases)} | {_fmt_number(output_tokens / cases)} | "
                f"{median_latency} | {max_latency} | {http_calls} | {repairs} | "
                f"{retries} | {failures} |"
            )

        quality_metrics = TASK_QUALITY.get(task, ())
        if quality_metrics:
            header = "| Model | Prompt | " + " | ".join(quality_metrics) + " |"
            align = "| --- | --- | " + " | ".join("---:" for _ in quality_metrics) + " |"
            lines.extend(["", header, align])
            for key in task_keys:
                _task, model_name, prompt_version = key
                _u, o, s = _rows_for_key(key, usage, outputs, scores, mapping)
                transfer = model_name == "qwen"
                prompt = _prompt_label(o, s, prompt_version, transfer)
                cells = " | ".join(_ratio_text(s, metric) for metric in quality_metrics)
                lines.append(f"| {model_name} | {prompt} | {cells} |")
        lines.append("")

    lines.extend(_human_boundary_section(outputs, scores))
    lines.extend(
        [
            "## Limits",
            "",
            "- Twelve cases per task. Counts are directional, not a production ranking.",
            "- A row measures the model together with the prompt version shown in that row.",
            "- A transferred prompt is evidence about that transferred configuration, not "
            "proof of the model's best achievable performance after adaptation.",
            "- Untested combinations include adapted Qwen prompts, `triage.v2`, `extract.v1`, "
            "and `baseline.v0` as a Day 5 task prompt.",
            "- No production-volume reliability claim is being made.",
            "- Local Ollama latency depends on lab hardware, not a cloud SLA.",
            "- Latency is per HTTP POST, not a sum of retries onto one row.",
            "- Do not turn 11/12 versus 10/12 into a universal model ranking.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _human_boundary_section(
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
) -> list[str]:
    triage_scores = [row for row in scores if row.task == "triage"]
    if not triage_scores:
        return []
    models = sorted({row.model_name for row in triage_scores})
    lines = [
        "## Human-boundary re-check",
        "",
        "Triage `draft_reply` must not approve, deny, refund, or imply a final customer "
        "outcome. `customer_outcome` must stay null. Both configured local models were tested.",
        "",
        "| Model | Prompt | human_boundary | customer_outcome | draft_reply |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    grouped: dict[tuple[str, str], list[ScoreRecord]] = defaultdict(list)
    for score in triage_scores:
        grouped[(score.model_name, score.prompt_version)].append(score)
    outputs_by: dict[tuple[str, str], list[OutputRecord]] = defaultdict(list)
    for output in outputs:
        if output.task == "triage":
            outputs_by[(output.model_name, output.prompt_version)].append(output)
    for (model_name, prompt_version), rows in sorted(grouped.items()):
        prompt = _prompt_label(
            outputs_by[(model_name, prompt_version)],
            rows,
            prompt_version,
            model_name == "qwen",
        )
        lines.append(
            "| "
            f"{model_name} | {prompt} | "
            f"{_ratio_text(rows, 'human_boundary')} | "
            f"{_ratio_text(rows, 'human_boundary.customer_outcome')} | "
            f"{_ratio_text(rows, 'human_boundary.draft_reply')} |"
        )
    lines.extend(
        [
            "",
            "Models tested: " + ", ".join(f"`{name}`" for name in models) + ".",
            "",
        ]
    )
    return lines


def _metric_tuple(
    scores: Sequence[ScoreRecord], metric: str
) -> tuple[int, int]:
    rows = [row for row in scores if row.metric == metric]
    return sum(row.numerator for row in rows), sum(row.denominator for row in rows)


def _prefer_higher(left: tuple[int, int], right: tuple[int, int]) -> int:
    left_n, left_d = left
    right_n, right_d = right
    if left_d == 0 and right_d == 0:
        return 0
    if left_d == 0:
        return -1
    if right_d == 0:
        return 1
    left_score = left_n / left_d
    right_score = right_n / right_d
    if left_score > right_score:
        return 1
    if left_score < right_score:
        return -1
    return 0


def _prefer_lower(left: tuple[int, int], right: tuple[int, int]) -> int:
    return -_prefer_higher(left, right)


def _comparisons_for(task: str) -> list[tuple[str, bool]]:
    if task == "triage":
        return [
            ("missed_escalation", True),
            ("pii_leakage", True),
            ("human_boundary", False),
            ("queue", False),
            ("escalation", False),
            ("unnecessary_escalation", True),
        ]
    return [
        ("required_evidence_recall", False),
        ("citation_correctness", False),
        ("unsupported_field_avoidance", False),
        ("document_status", False),
        ("current_version", False),
        ("pii_leakage", True),
    ]


def _choose_config(
    task: str,
    configs: list[_ConfigKey],
    scores: Sequence[ScoreRecord],
) -> _ConfigKey:
    def score_for(key: _ConfigKey) -> list[ScoreRecord]:
        return [row for row in scores if _key_from_score(row) == key]

    def better(left: _ConfigKey, right: _ConfigKey) -> _ConfigKey:
        left_s = score_for(left)
        right_s = score_for(right)
        for metric, lower_is_better in _comparisons_for(task):
            prefer = _prefer_lower if lower_is_better else _prefer_higher
            comparison = prefer(_metric_tuple(left_s, metric), _metric_tuple(right_s, metric))
            if comparison > 0:
                return left
            if comparison < 0:
                return right
        return left

    chosen = configs[0]
    for candidate in configs[1:]:
        chosen = better(chosen, candidate)
    return chosen


def _reject_reason(
    task: str,
    selected: _ConfigKey,
    rejected: _ConfigKey,
    scores: Sequence[ScoreRecord],
) -> str:
    selected_s = [row for row in scores if _key_from_score(row) == selected]
    rejected_s = [row for row in scores if _key_from_score(row) == rejected]
    for metric, lower_is_better in _comparisons_for(task):
        prefer = _prefer_lower if lower_is_better else _prefer_higher
        comparison = prefer(
            _metric_tuple(selected_s, metric),
            _metric_tuple(rejected_s, metric),
        )
        if comparison != 0:
            selected_ratio = _ratio_text(selected_s, metric)
            rejected_ratio = _ratio_text(rejected_s, metric)
            direction = "lower is better" if lower_is_better else "higher is better"
            return (
                f"`{rejected[1]}` `{rejected[2]}` lost on `{metric}` "
                f"({rejected_ratio} vs {selected_ratio}; {direction}), "
                "not because of latency or token volume"
            )
    return "no quality metric separated the rows; the first listed configuration was kept"


def _write_decision(
    *,
    run_id: str,
    models: Sequence[str],
    usage: Sequence[CallRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    decision_path: Path,
    mapping: dict[str, str],
) -> None:
    keys = _all_config_keys(usage, outputs, scores, mapping)
    lines: list[str] = [
        "# Model Decision Record",
        "",
        f"Run ID: `{run_id}`",
        "",
        "This is a routing recommendation, not a claim that either model resolves "
        "customer outcomes. The desk still needs a human for approval, denial, refund, "
        "and investigation. Qwen rows used transferred Day 3/4 prompts, not adapted prompts.",
        "",
        "## Customer problem",
        "",
        "Intake must assign a queue, escalate mixed or unsafe cases, extract policy "
        "fields with citations, and never leak PII or decide the customer's outcome.",
        "",
        "## Evaluated models",
        "",
    ]
    evaluated_models = sorted({model for _task, model, _prompt in keys} | {str(m) for m in models})
    lines.extend(f"- {model}" for model in evaluated_models or ["None"])
    lines.extend(["", "## Evaluated configurations", ""])
    if keys:
        for task, model, prompt in keys:
            transfer = "transfer" if model == "qwen" else "native"
            _u, o, s = _rows_for_key((task, model, prompt), usage, outputs, scores, mapping)
            label = _prompt_label(o, s, prompt, model == "qwen")
            lines.append(f"- `{task}` — {model} — `{label}` ({transfer})")
    else:
        lines.append("- No configurations supplied.")

    lines.extend(["", "## Task decisions", ""])
    tasks = sorted({task for task, _model, _prompt in keys})
    for task in tasks:
        configs = [key for key in keys if key[0] == task]
        selected = _choose_config(task, configs, scores)
        selected_scores = [row for row in scores if _key_from_score(row) == selected]
        rejected = [key for key in configs if key != selected]
        _u, selected_outputs, _s = _rows_for_key(selected, usage, outputs, scores, mapping)
        selected_prompt = _prompt_label(
            selected_outputs, selected_scores, selected[2], selected[1] == "qwen"
        )
        reject_text = (
            "; ".join(_reject_reason(task, selected, key, scores) for key in rejected)
            or "none"
        )
        lines.extend(
            [
                f"### {task}",
                "",
                f"- selected model: `{selected[1]}`",
                f"- prompt version: `{selected_prompt}`",
                f"- measured reason: `{_metric_text(selected_scores)}`",
                "- rejected alternative(s): "
                + (
                    ", ".join(f"`{model}` `{prompt}`" for _task, model, prompt in rejected)
                    or "none"
                ),
                f"- rejected because: {reject_text}.",
                f"- reopen if: {REOPEN_IF.get(task, 'a larger held-out set changes the ranking')}.",
                "",
            ]
        )

    lines.extend(
        [
            "## Options",
            "",
            "| ID | Name | What it does | Risk | Reversible |",
            "| --- | --- | --- | --- | --- |",
            "| 0 | Do nothing | Send every item to a human with no router | "
            "No missed auto-route; high loaded-hour cost | n/a |",
            "| 1 | Mistral native prompts | Route with the frozen Day 3/4 Mistral files | "
            "Keeps the measured Mistral error modes | yes |",
            "| 2 | Qwen prompt transfer | Run the same frozen files on Qwen | "
            "Measures transfer, not an adapted Qwen prompt | yes |",
            "| 3 | Adapt Qwen prompts and shadow | New prompt versions plus dual-run | "
            "Delivery cost; still needs a held-out set | yes, while shadowed |",
            "",
            "Per-task selected rows are listed above. Latency and token volume were not "
            "used to choose a winner. Option 0 remains the fallback if missed escalation "
            "or PII leakage rises on a larger set.",
            "",
            "## Value model",
            "",
            "This lab has no book-of-business volume or indemnity table. Dollar impact is "
            "therefore **not claimed**. The money-adjacent metrics are:",
            "",
            "| Field | Value | Source |",
            "| --- | --- | --- |",
            "| Lever | risk avoided and capacity (unnecessary escalation) | lab design |",
            "| Volume | unknown | no customer file in this repository |",
            "| Baseline | all items to a human (option 0) | do-nothing |",
            "| Target | missed_escalation 0/n and pii_leakage 0/n on a held-out set | safety bar |",
            "| $ per unit | not estimated | would invent a number |",
            "| Owner of the number | intake operations lead, if productionized | assumption |",
            "",
            "Conservative case: keep option 0 if missed escalation or PII appears on a "
            "larger set. Base case: option 2 as a router, not a resolver. Upside: option 3 "
            "only after a held-out set confirms the safety bar.",
            "",
            "## 90-day metric",
            "",
            "- Name: missed_escalation count and pii_leakage count on held-out intake.",
            "- Baseline: human-only routing (option 0).",
            "- Target: 0 missed escalations and 0 PII leaks on the held-out set.",
            "- Owner: intake operations lead.",
            "- Kill: missed_escalation or PII above 0, or a human-boundary failure.",
            "",
            "## Limits",
            "",
            "- 12 cases per task; 11/12 vs 10/12 is not a production ranking.",
            "- Provider cost is `$0.00`. Latency reflects local hardware, not a cloud SLA.",
            "- Transfer rows are labeled; they do not claim Qwen was optimized.",
            "- Do-nothing (keep sending everything to a human with no router) remains "
            "the fallback if missed escalation or PII leakage rises on a larger set.",
            "- Production routing would need shadow/dual-run, an audit trail, and human "
            "sign-off before changing customer-facing outcomes.",
            "",
        ]
    )
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    *,
    run_id: str,
    models: Sequence[str],
    usage: Sequence[CallRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    report_path: Path,
    decision_path: Path,
) -> None:
    mapping = _model_id_map()
    run_usage = _for_run(usage, run_id)
    run_outputs = _for_run(outputs, run_id)
    run_scores = _for_run(scores, run_id)
    _write_report(
        run_id=run_id,
        usage=run_usage,
        outputs=run_outputs,
        scores=run_scores,
        report_path=Path(report_path),
        mapping=mapping,
    )
    _write_decision(
        run_id=run_id,
        models=models,
        usage=run_usage,
        outputs=run_outputs,
        scores=run_scores,
        decision_path=Path(decision_path),
        mapping=mapping,
    )
