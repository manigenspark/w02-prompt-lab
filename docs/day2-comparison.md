# Day 2 model comparison

Both models ran locally through Ollama on the same twelve summarization cases, baseline prompt, temperature, and 512-token output ceiling. The adapter disables native reasoning output for every model, so the token counts below measure answer generation rather than hidden reasoning. Provider charge is $0.00 for both; this comparison uses success counts, tokens, and latency only.

| Metric | Mistral | Qwen |
| --- | ---: | ---: |
| Cases | 12 | 12 |
| Successful completions | 12 / 12 | 12 / 12 |
| Truncated (`stop_reason=length`) | 0 | 0 |
| Input tokens (successful, sum) | 2,679 | 2,487 |
| Output tokens (successful, sum) | 1,602 | 774 |
| Median latency (successful, ms) | 5,152 | 2,882 |
| Max latency (successful, ms) | 8,720 | 6,971 |
| Observation count | 12 | 12 |

## mistral (`mistral:7b`)

- Successful completions: 12 / 12
- Attempts recorded: 12
- Truncated attempts (output ceiling reached): 0
- Total input tokens (successful): 2679
- Total output tokens (successful): 1602
- Median latency (successful): 5152 ms
- Max latency (successful): 8720 ms

## qwen (`qwen3:8b`)

- Successful completions: 12 / 12
- Attempts recorded: 12
- Truncated attempts (output ceiling reached): 0
- Total input tokens (successful): 2487
- Total output tokens (successful): 774
- Median latency (successful): 2882 ms
- Max latency (successful): 6971 ms

## Observation

Qwen had the lower median successful latency (2882 ms vs 5152 ms for mistral). Token totals also differ across models for identical inputs, so a short sample from only one model would not estimate the other's workload.

## Truncation evidence from the reasoning-enabled run

An earlier run (`run_id=2c7d725b-84b6-4806-9b39-5cb32b3c15cc`) used the identical prompt, temperature, and 512-token ceiling but left Qwen's native reasoning output enabled. Under that configuration Qwen truncated on two of twelve cases:

| Case | Model | input_tokens | output_tokens | stop_reason | error_type |
|---|---|---|---|---|---|
| S01 | `qwen3:8b` | 238 | 512 | `length` | `TruncatedResponseError` |
| S02 | `qwen3:8b` | 225 | 512 | `length` | `TruncatedResponseError` |

Both attempts consumed the entire output budget; S01 returned no answer text at all and S02 returned a partial field list. Mistral completed the same S01 document in 169 output tokens, so document length was not the cause. Reasoning tokens are billed against the same `num_predict` ceiling as the answer, which is why only the hybrid-reasoning model hit the limit.

Two consequences worth recording. First, truncation was classified as `TruncatedResponseError` and was not retried, which is correct: a second identical attempt would exhaust the same budget. Second, with reasoning enabled Qwen's median successful latency was 17604 ms against 5152 ms for Mistral, and its successful cases averaged roughly 400 output tokens. Those totals are not comparable to Mistral's answer-only tokens, which is why the adapter disables reasoning output uniformly for the run reported above.
