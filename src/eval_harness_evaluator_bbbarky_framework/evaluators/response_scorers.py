"""Deterministic text scorers and an evaluator that applies one."""

from __future__ import annotations

import re
from collections import Counter
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
    """Fraction of required keywords present in the prediction (whole-word match)."""

    def __init__(self, keywords: list[str]) -> None:
        self.keywords = [k.lower() for k in keywords]
        self._patterns = [
            re.compile(r"\b" + re.escape(k) + r"\b")
            for k in self.keywords
        ]

    def score(self, prediction: str, reference: object | None) -> float:
        if not self._patterns:
            return 1.0
        text = prediction.lower()
        present = sum(1 for p in self._patterns if p.search(text))
        return present / len(self._patterns)


class FuzzyF1Scorer:
    """Token-overlap F1 over unigrams using multiset (SQuAD-style)."""

    def score(self, prediction: str, reference: object | None) -> float:
        pred = _tokens(prediction)
        ref = _tokens(str(reference))
        if not pred or not ref:
            return 0.0
        pred_counts = Counter(pred)
        ref_counts = Counter(ref)
        overlap = sum((pred_counts & ref_counts).values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred)
        recall = overlap / len(ref)
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
