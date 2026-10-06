import pytest

from evaluation.metrics import exact_match, f1_score, normalize_answer, set_precision_recall


def test_normalize_strips_case_punctuation_and_articles():
    assert normalize_answer("The  Arthur's Magazine!") == "arthurs magazine"


def test_exact_match_ignores_formatting():
    assert exact_match("Tim Burton.", "tim burton") == 1.0
    assert exact_match("Burton", "Tim Burton") == 0.0


def test_f1_gives_partial_credit():
    assert f1_score("Burton", "Tim Burton") == pytest.approx(2 / 3)


def test_f1_yes_no_must_match_exactly():
    assert f1_score("yes", "no") == 0.0
    assert f1_score("yes", "yes") == 1.0
    assert f1_score("yes it is", "yes") == 0.0


def test_set_precision_recall():
    assert set_precision_recall({1, 2, 3}, {2, 3, 4, 5}) == (2 / 3, 2 / 4)
    assert set_precision_recall(set(), {1}) == (0.0, 0.0)
