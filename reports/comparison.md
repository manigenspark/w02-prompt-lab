# Model Comparison

Run ID: `c9e2f0a1-7d44-4b1c-8e6f-2a91b0c5d387`

Counts are reported with their denominators, not percentages. Latency uses median and maximum rather than mean. HTTP-call count is the observation count for latency (primary attempts plus repairs and transport retries).

Qwen rows use the same frozen prompt files as Mistral (prompt transfer). They measure that transferred configuration, not an adapted Qwen prompt.

Local Ollama provider/API charge is `$0.00`. Token usage and latency still represent real operational work.

## Extraction

| Model | Prompt | Transfer | Valid outputs | Input tokens | Output tokens | Input tokens/case | Output tokens/case | Median latency | Max latency | HTTP calls | Repairs | Retries | Final failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | extract.v2 | no | 12/12 | 20327 | 5022 | 1693.9 | 418.5 | 18498.5 ms | 19714 ms | 12 | 0/12 | 0 | 0 |
| qwen | extract.v2 transfer | yes | 12/12 | 17003 | 2122 | 1416.9 | 176.8 | 9810 ms | 10723 ms | 12 | 0/12 | 0 | 0 |

| Model | Prompt | document_status | required_evidence_recall | citation_correctness | unsupported_field_avoidance | current_version | pii_leakage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | extract.v2 | 9/12 | 73/73 | 77/77 | 8/12 | 1/1 | 0/12 |
| qwen | extract.v2 transfer | 11/12 | 73/73 | 74/74 | 11/12 | 1/1 | 0/12 |

## Summarization

| Model | Prompt | Transfer | Valid outputs | Input tokens | Output tokens | Input tokens/case | Output tokens/case | Median latency | Max latency | HTTP calls | Repairs | Retries | Final failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | summarize.v1 | no | 10/12 | 25273 | 6328 | 2106.1 | 527.3 | 10995.5 ms | 15698 ms | 24 | 12/12 | 0 | 2 |
| qwen | summarize.v1 transfer | yes | 12/12 | 10590 | 3189 | 882.5 | 265.8 | 11815 ms | 13060 ms | 13 | 1/12 | 0 | 0 |

| Model | Prompt | document_status | required_evidence_recall | citation_correctness | unsupported_field_avoidance | current_version | pii_leakage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | summarize.v1 | 9/12 | 56/60 | 58/60 | 8/12 | 1/1 | 0/12 |
| qwen | summarize.v1 transfer | 12/12 | 60/60 | 63/63 | 9/12 | 1/1 | 0/12 |

## Triage

| Model | Prompt | Transfer | Valid outputs | Input tokens | Output tokens | Input tokens/case | Output tokens/case | Median latency | Max latency | HTTP calls | Repairs | Retries | Final failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | triage.v1 | no | 12/12 | 7400 | 1938 | 616.7 | 161.5 | 5863 ms | 7239 ms | 13 | 1/12 | 0 | 0 |
| qwen | triage.v1 transfer | yes | 12/12 | 5947 | 1517 | 495.6 | 126.4 | 5405 ms | 7218 ms | 12 | 0/12 | 0 | 0 |

| Model | Prompt | queue | escalation | missed_escalation | unnecessary_escalation | human_boundary | human_boundary.customer_outcome | human_boundary.draft_reply | pii_leakage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | triage.v1 | 6/12 | 11/12 | 1/12 | 0/12 | 12/12 | 12/12 | 12/12 | 0/12 |
| qwen | triage.v1 transfer | 10/12 | 10/12 | 0/12 | 2/12 | 12/12 | 12/12 | 12/12 | 0/12 |

## Human-boundary re-check

Triage `draft_reply` must not approve, deny, refund, or imply a final customer outcome. `customer_outcome` must stay null. Both configured local models were tested.

| Model | Prompt | human_boundary | customer_outcome | draft_reply |
| --- | --- | ---: | ---: | ---: |
| mistral | triage.v1 | 12/12 | 12/12 | 12/12 |
| qwen | triage.v1 transfer | 12/12 | 12/12 | 12/12 |

Models tested: `mistral`, `qwen`.

## Limits

- Twelve cases per task. Counts are directional, not a production ranking.
- A row measures the model together with the prompt version shown in that row.
- A transferred prompt is evidence about that transferred configuration, not proof of the model's best achievable performance after adaptation.
- Untested combinations include adapted Qwen prompts, `triage.v2`, `extract.v1`, and `baseline.v0` as a Day 5 task prompt.
- No production-volume reliability claim is being made.
- Local Ollama latency depends on lab hardware, not a cloud SLA.
- Latency is per HTTP POST, not a sum of retries onto one row.
- Do not turn 11/12 versus 10/12 into a universal model ranking.
