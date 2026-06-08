"""Judge-reliability tooling: Cohen's kappa, confusion, agreement reports."""

import pytest

from eval_harness_evaluator_bbbarky_framework.reliability import (
    agreement_report,
    cohens_kappa,
    confusion_matrix,
    inter_judge_agreement,
)


def test_kappa_perfect_agreement_is_one():
    assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)


def test_kappa_known_value():
    # po=0.75, pe=0.5 -> kappa=0.5
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)


def test_kappa_all_same_category_defined():
    # No variance and full agreement -> kappa 1.0 (avoid 0/0).
    assert cohens_kappa(["a", "a"], ["a", "a"]) == pytest.approx(1.0)


def test_kappa_requires_equal_length():
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])


def test_confusion_matrix_counts():
    cm = confusion_matrix(["a", "a", "b"], ["a", "b", "b"])
    assert cm["a"]["a"] == 1
    assert cm["a"]["b"] == 1
    assert cm["b"]["b"] == 1


def test_agreement_report_fields():
    report = agreement_report(["a", "b", "a", "b"], ["a", "b", "a", "a"])
    assert report["n"] == 4
    assert report["accuracy"] == pytest.approx(0.75)
    assert "kappa" in report
    assert "confusion" in report


def test_inter_judge_agreement_pairwise():
    out = inter_judge_agreement(
        {"j1": ["a", "b", "a"], "j2": ["a", "b", "a"], "j3": ["b", "b", "a"]}
    )
    assert out["j1|j2"] == pytest.approx(1.0)
    assert "j1|j3" in out and "j2|j3" in out
