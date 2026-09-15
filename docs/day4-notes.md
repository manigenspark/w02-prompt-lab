# Day 4 notes

Run `b7a3b6a8-4bdf-499f-a933-677d1e5ffb75` on `mistral:7b` at temperature `0.0`.
Both prompt versions used the same 12 triage cases. Provider/API cost is $0.00.

triage.v1
queue correct: 6/12
escalation correct: 7/12
missed escalations: 1
unnecessary escalations: 0
human-boundary passes: 8/12

triage.v2
queue correct: 5/12
escalation correct: 9/12
missed escalations: 1
unnecessary escalations: 0
human-boundary passes: 10/12

Changed-queue count: 3/12 (T02, T09, T12). None of those changes matched gold.

Output tokens: v1 2314 across 16 attempts; v2 2446 across 14 attempts; difference +132.
Median latency: v1 5796 ms, v2 7124 ms. Max latency: v1 9461 ms, v2 9052 ms.
Observation count: 30.

The extra analysis field did not earn its overhead: queue fell from 6/12 to 5/12 while output tokens and median latency rose. A one-or-two-case gap on 12 rows is not proof that either prompt is universally better. The dominant failure was `confidence` as a label like "high" instead of a 0.0-1.0 float, including after one repair; gold escalate cases T06 and T08 also produced no valid object, so the missed-escalation file count of 1 understates those schema failures.