from promptlab.corpus import load_cases, validate_corpus


def test_corpus_has_twelve_cases_per_task() -> None:
    counts = validate_corpus()
    assert counts == {"triage": 12, "summarization": 12, "extraction": 12}


def test_cases_pair_with_gold_ids() -> None:
    for task in ("triage", "summarization", "extraction"):
        pairs = load_cases(task)
        assert [case.id for case, gold in pairs] == [gold.id for case, gold in pairs]
