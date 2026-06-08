"""Deterministic text scorers and an evaluator that applies one."""

from __future__ import annotations

from typing import Protocol

from ..models.core import EvalResult, Invocation, PerInvocationResult
from .base import Evaluator


def _tokens(text: str) -> list[str]:
    return text.lower().split()


class Scorer(Protocol):
    """A scorer maps (prediction, reference) to a score in [0, 1]."""

    def score(self, prediction: str, reference: object | None) -> float: ...


class ExactMatchScorer:
    """1.0 when prediction equals reference (case-insensitive, trimmed)."""

    def score(self, prediction: str, reference: object | None) -> float:
        return 1.0 if prediction.strip().lower() == str(reference).strip().lower() else 0.0


class ContainsKeywordsScorer:
    """Fraction of required keywords present in the prediction."""

    def __init__(self, keywords: list[str]) -> None:
        self.keywords = [k.lower() for k in keywords]

    def score(self, prediction: str, reference: object | None) -> float:
        if not self.keywords:
            return 1.0
        text = prediction.lower()
        present = sum(1 for k in self.keywords if k in text)
        return present / len(self.keywords)


class FuzzyF1Scorer:
    """Token-overlap F1 over unigrams (simple, dependency-free)."""

    def score(self, prediction: str, reference: object | None) -> float:
        pred = _tokens(prediction)
        ref = _tokens(str(reference))
        if not pred or not ref:
            return 0.0
        pred_set, ref_set = set(pred), set(ref)
        overlap = pred_set & ref_set
        if not overlap:
            return 0.0
        precision = len(overlap) / len(pred_set)
        recall = len(overlap) / len(ref_set)
        return 2 * precision * recall / (precision + recall)


class ScorerEvaluator(Evaluator):
    """Apply a :class:`Scorer` to each invocation's final response."""

    metric_name = "scorer"

    def __init__(self, scorer: Scorer, threshold: float = 0.5) -> None:
        self.scorer = scorer
        self.threshold = threshold

    async def evaluate_invocations(
        self, invocations: list[Invocation], expected: object | None
    ) -> EvalResult:
        per: list[PerInvocationResult] = []
        for invocation in invocations:
            score = self.scorer.score(invocation.final_response, expected)
            per.append(PerInvocationResult(score=score, passed=score >= self.threshold))
        return self._aggregate(per)
