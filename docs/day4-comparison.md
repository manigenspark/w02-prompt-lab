# Day 4 prompt comparison: triage.v1 vs triage.v2

Source: `docs/day4-run.jsonl`, `docs/day4-scores.jsonl` (`run_id` `b7a3b6a8-4bdf-499f-a933-677d1e5ffb75`).

Both prompt versions ran on the same local model (`mistral:7b`) at `temperature=0.0` over the same 12 rows in `cases/triage.jsonl`. `triage.v1.md` and `triage.v2.md` are frozen against this run. The variable under test is `prompt_version`. v1 validated against `TriageOutput`; v2 against `TriageOutputWithAnalysis`. Provider/API cost is **$0.00**.

Escalation is scored on `escalation_required` vs gold, not on `human_review_required`. Human-boundary passes require a valid object whose `customer_outcome` is null and whose `draft_reply` has no approve/deny/refund language. Schema failures count as boundary misses.

| Metric | v1 | v2 |
| --- | ---: | ---: |
| Queue correct | 6/12 | 5/12 |
| Escalation correct | 7/12 | 9/12 |
| Missed escalations (scored) | 1 | 1 |
| Unnecessary escalations | 0 | 0 |
| Human-boundary passes | 8/12 | 10/12 |

Changed-queue count: **3/12** (T02, T09, T12). None of those changes matched gold.

| Metric | v1 | v2 |
| --- | ---: | ---: |
| Output tokens (sum, all attempts) | 2,314 | 2,446 |
| Attempts recorded | 16 | 14 |
| Median latency (ms) | 5,796 | 7,124 |
| Maximum latency (ms) | 9,461 | 9,052 |
| Observation count | 16 | 14 |

The extra `analysis` field did not earn its overhead: queue fell from 6/12 to 5/12 while output tokens and median latency rose. A one-or-two-case gap on 12 rows is not proof that either prompt is universally better.

The dominant failure was `confidence` as a label like `"high"` instead of a 0.0–1.0 float, including after one repair. Gold escalate cases T06 and T08 also produced no valid object, so the missed-escalation file count of 1 understates those schema failures. Day 5's runner-side JSON-instance contract (not a prompt-file edit) exists so later runs ask for a number, not a word.
