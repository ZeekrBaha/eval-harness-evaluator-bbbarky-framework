"""Judge-reliability statistics: agreement before you let a judge gate CI.

An LLM judge is itself a noisy classifier. Before its scores block a pipeline,
measure how well it agrees with human labels (Cohen's kappa, accuracy, a
confusion matrix) and how well judges agree with each other.
"""

from __future__ import annotations

from collections import Counter


def cohens_kappa(labels_a: list, labels_b: list) -> float:
    """Cohen's kappa for two raters labeling the same items.

    Returns 1.0 when raters fully agree (even with a single category), and a
    chance-corrected score in [-1, 1] otherwise.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("label lists must be the same length")
    n = len(labels_a)
    if n == 0:
        raise ValueError("need at least one item")

    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n

    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    categories = set(count_a) | set(count_b)
    p_e = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)

    if p_e == 1.0:
        # No variance: agreement is meaningful only if observed agreement is full.
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def confusion_matrix(reference: list, predicted: list) -> dict[str, dict[str, int]]:
    """Nested ``{reference_label: {predicted_label: count}}`` matrix."""
    if len(reference) != len(predicted):
        raise ValueError("label lists must be the same length")
    matrix: dict[str, dict[str, int]] = {}
    for ref, pred in zip(reference, predicted):
        matrix.setdefault(str(ref), {}).setdefault(str(pred), 0)
        matrix[str(ref)][str(pred)] += 1
    return matrix


def agreement_report(human_labels: list, judge_labels: list) -> dict:
    """Compare judge labels against human labels: n, accuracy, kappa, confusion."""
    if len(human_labels) != len(judge_labels):
        raise ValueError("label lists must be the same length")
    n = len(human_labels)
    correct = sum(1 for h, j in zip(human_labels, judge_labels) if h == j)
    return {
        "n": n,
        "accuracy": round(correct / n, 3) if n else 0.0,
        "kappa": round(cohens_kappa(human_labels, judge_labels), 3) if n else 0.0,
        "confusion": confusion_matrix(human_labels, judge_labels),
    }


def inter_judge_agreement(label_lists: dict[str, list]) -> dict[str, float]:
    """Pairwise Cohen's kappa across judges. Keys are ``"<a>|<b>"`` (a<b)."""
    names = sorted(label_lists)
    out: dict[str, float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out[f"{a}|{b}"] = round(cohens_kappa(label_lists[a], label_lists[b]), 3)
    return out
