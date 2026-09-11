# Day 2 model comparison

Both models ran locally through Ollama on the same twelve summarization cases, baseline prompt, temperature, and output ceiling. Provider charge is $0.00 for both; this comparison uses success counts, tokens, and latency only.

## mistral (`mistral:7b`)

- Successful completions: 12 / 12
- Attempts recorded: 12
- Total input tokens (successful): 2679
- Total output tokens (successful): 1539
- Median latency (successful): 5206 ms
- Max latency (successful): 8507 ms

## qwen (`qwen3:8b`)

- Successful completions: 12 / 12
- Attempts recorded: 12
- Total input tokens (successful): 2487
- Total output tokens (successful): 774
- Median latency (successful): 2896 ms
- Max latency (successful): 7544 ms

## Observation

Qwen had the lower median successful latency (2896 ms vs 5206 ms for mistral). Token totals also differ across models for identical inputs, so a short sample from only one model would not estimate the other's workload.
