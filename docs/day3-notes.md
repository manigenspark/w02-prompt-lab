# Day 3 notes

Run `3d843d2a-dfb1-4dea-b626-76b3861ff5f0` on `mistral:7b` at temperature `0.0`.
Summarization used `summarize.v1.md` (12 cases). Extraction used `extract.v2.md` (12 cases).

- Summarization repair rate: 12/12 (9/12 validated after the one allowed repair)
- Extraction repair rate: 12/12 (11/12 validated after the one allowed repair)
- Example leakage count: 0
- Citation-existence failure count: 0 missing-from-source (`EvidenceField.citation` for every `status: "present"` appears in that case's source). Several successful citations were abbreviated (`"1"` rather than `"1. Document Control"`).

The most common first-pass error was invalid JSON or the wrong EvidenceField shape (`text` instead of `value`, missing `status`). Compact `schema_description()`, a document-only extraction user message, and coerce-before-validate recovered most cases within `max_repairs=1`. Remaining failures: S01 and E02 (invalid JSON after repair), S05 (missing `version`), S09 (missing `required_steps.status`).

Standing JSON-instance rules belong in the runner system message, not in a rewritten `summarize.v1.md`. The shipped prompt file for this run is frozen.
