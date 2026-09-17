# Day 5 notes

Run `c9e2f0a1-7d44-4b1c-8e6f-2a91b0c5d387` at temperature `0.0` on local Ollama.
Prompts were frozen from Days 3–4: `summarize.v1`, `extract.v2`, `triage.v1`.
Qwen rows are prompt-transfer results, not adapted prompts. Provider/API cost is `$0.00`.

JSON-instance and confidence-as-number rules live in the Day 5 runner system message, not in rewritten prompt files.

Mistral summarization finished 10/12 valid. S05 and S12 failed after one repair because the frozen `summarize.v1` file still says “out-of-scope,” which is not a legal `document_status`. Qwen mapped those cases onto the schema and finished 12/12.

Triage human-boundary passed 12/12 on both Mistral and Qwen. Qwen missed 0/12 gold escalations and routed 10/12 queues correctly; Mistral missed 1/12 escalations and routed 6/12. Qwen also had 2/12 unnecessary escalations. The decision ranking treats missed escalation and PII as safety metrics before queue accuracy or latency.

Do-nothing (send every item to a human) remains the fallback if missed escalation or PII leakage rises on a held-out set. This router does not approve, deny, or refund.
