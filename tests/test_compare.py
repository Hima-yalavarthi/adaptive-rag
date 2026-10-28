from evaluation.compare import compare, paired_bootstrap


def test_paired_bootstrap_interval_is_ordered():
    lo, hi = paired_bootstrap([0.0, 1.0] * 50, iterations=200, seed=0)
    assert lo <= hi


def test_compare_requires_same_question_ids():
    a = [{"id": "1", "em": 1, "f1": 1, "retrieval_recall": 1, "all_gold_retrieved": 1, "citation_precision": 1, "prediction": "yes"}]
    b = [{"id": "2", "em": 0, "f1": 0, "retrieval_recall": 0, "all_gold_retrieved": 0, "citation_precision": 0, "prediction": "no"}]
    try:
        compare(a, b)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
