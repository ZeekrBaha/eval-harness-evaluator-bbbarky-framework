"""eval-harness-evaluator-bbbarky-framework: provider-agnostic agent evaluation."""

from __future__ import annotations

from .evaluators.base import Evaluator
from .evaluators.composite import CompositeEvaluator
from .evaluators.json_schema import JsonSchemaEvaluator
from .evaluators.label_match import LabelMatchEvaluator
from .evaluators.llm_judge import LlmJudgeEvaluator
from .evaluators.response_scorers import ScorerEvaluator, ExactMatchScorer, ContainsKeywordsScorer, FuzzyF1Scorer
from .facade import HarnessEvaluator
from .reliability.kappa import agreement_report, cohens_kappa, inter_judge_agreement
from .models.core import (
    EvalCase,
    EvalResult,
    EvalSet,
    Invocation,
    JudgeResult,
    PerInvocationResult,
)
from .runner.core import run_suite

__all__ = [
    "HarnessEvaluator",
    "Evaluator",
    "CompositeEvaluator",
    "JsonSchemaEvaluator",
    "LabelMatchEvaluator",
    "LlmJudgeEvaluator",
    "ScorerEvaluator",
    "ExactMatchScorer",
    "ContainsKeywordsScorer",
    "FuzzyF1Scorer",
    "run_suite",
    "EvalSet",
    "EvalCase",
    "Invocation",
    "EvalResult",
    "PerInvocationResult",
    "JudgeResult",
    "agreement_report",
    "cohens_kappa",
    "inter_judge_agreement",
]
