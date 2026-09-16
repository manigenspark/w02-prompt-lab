# Model Decision Record

Run ID: `c9e2f0a1-7d44-4b1c-8e6f-2a91b0c5d387`

This is a routing recommendation, not a claim that either model resolves customer outcomes. The desk still needs a human for approval, denial, refund, and investigation. Qwen rows used transferred Day 3/4 prompts, not adapted prompts.

## Customer problem

Intake must assign a queue, escalate mixed or unsafe cases, extract policy fields with citations, and never leak PII or decide the customer's outcome.

## Evaluated models

- mistral
- qwen

## Evaluated configurations

- `extraction` — mistral — `extract.v2` (native)
- `extraction` — qwen — `extract.v2 transfer` (transfer)
- `summarization` — mistral — `summarize.v1` (native)
- `summarization` — qwen — `summarize.v1 transfer` (transfer)
- `triage` — mistral — `triage.v1` (native)
- `triage` — qwen — `triage.v1 transfer` (transfer)

## Task decisions

### extraction

- selected model: `qwen`
- prompt version: `extract.v2 transfer`
- measured reason: `citation_correctness: 74/74 current_version: 1/1 document_status: 11/12 pii_leakage: 0/12 ↓ required_evidence_recall: 73/73 unsupported_field_avoidance: 11/12`
- rejected alternative(s): `mistral` `v2`
- rejected because: `mistral` `v2` lost on `unsupported_field_avoidance` (8/12 vs 11/12; higher is better), not because of latency or token volume.
- reopen if: a held-out set larger than 12 cases, or an adapted Qwen prompt, beats this row on document_status or unsupported_field_avoidance without dropping required-evidence recall or citation correctness, and without adding PII leakage.

### summarization

- selected model: `qwen`
- prompt version: `summarize.v1 transfer`
- measured reason: `citation_correctness: 63/63 current_version: 1/1 document_status: 12/12 pii_leakage: 0/12 ↓ required_evidence_recall: 60/60 unsupported_field_avoidance: 9/12`
- rejected alternative(s): `mistral` `v1`
- rejected because: `mistral` `v1` lost on `required_evidence_recall` (56/60 vs 60/60; higher is better), not because of latency or token volume.
- reopen if: a held-out set larger than 12 cases, or an adapted prompt, matches 12/12 valid outputs and evidence recall/citations without adding PII leakage or invented fields.

### triage

- selected model: `qwen`
- prompt version: `triage.v1 transfer`
- measured reason: `escalation: 10/12 human_boundary: 12/12 human_boundary.customer_outcome: 12/12 human_boundary.draft_reply: 12/12 missed_escalation: 0/12 ↓ pii_leakage: 0/12 ↓ queue: 10/12 unnecessary_escalation: 2/12 ↓`
- rejected alternative(s): `mistral` `v1`
- rejected because: `mistral` `v1` lost on `missed_escalation` (1/12 vs 0/12; lower is better), not because of latency or token volume.
- reopen if: a held-out set larger than 12 cases, or an adapted prompt, keeps missed_escalation at 0/12 and human_boundary at 12/12 while reducing unnecessary escalations without dropping queue accuracy or adding PII leakage.

## Options

| ID | Name | What it does | Risk | Reversible |
| --- | --- | --- | --- | --- |
| 0 | Do nothing | Send every item to a human with no router | No missed auto-route; high loaded-hour cost | n/a |
| 1 | Mistral native prompts | Route with the frozen Day 3/4 Mistral files | Keeps the measured Mistral error modes | yes |
| 2 | Qwen prompt transfer | Run the same frozen files on Qwen | Measures transfer, not an adapted Qwen prompt | yes |
| 3 | Adapt Qwen prompts and shadow | New prompt versions plus dual-run | Delivery cost; still needs a held-out set | yes, while shadowed |

Per-task selected rows are listed above. Latency and token volume were not used to choose a winner. Option 0 remains the fallback if missed escalation or PII leakage rises on a larger set.

## Value model

This lab has no book-of-business volume or indemnity table. Dollar impact is therefore **not claimed**. The money-adjacent metrics are:

| Field | Value | Source |
| --- | --- | --- |
| Lever | risk avoided and capacity (unnecessary escalation) | lab design |
| Volume | unknown | no customer file in this repository |
| Baseline | all items to a human (option 0) | do-nothing |
| Target | missed_escalation 0/n and pii_leakage 0/n on a held-out set | safety bar |
| $ per unit | not estimated | would invent a number |
| Owner of the number | intake operations lead, if productionized | assumption |

Conservative case: keep option 0 if missed escalation or PII appears on a larger set. Base case: option 2 as a router, not a resolver. Upside: option 3 only after a held-out set confirms the safety bar.

## 90-day metric

- Name: missed_escalation count and pii_leakage count on held-out intake.
- Baseline: human-only routing (option 0).
- Target: 0 missed escalations and 0 PII leaks on the held-out set.
- Owner: intake operations lead.
- Kill: missed_escalation or PII above 0, or a human-boundary failure.

## Limits

- 12 cases per task; 11/12 vs 10/12 is not a production ranking.
- Provider cost is `$0.00`. Latency reflects local hardware, not a cloud SLA.
- Transfer rows are labeled; they do not claim Qwen was optimized.
- Do-nothing (keep sending everything to a human with no router) remains the fallback if missed escalation or PII leakage rises on a larger set.
- Production routing would need shadow/dual-run, an audit trail, and human sign-off before changing customer-facing outcomes.
