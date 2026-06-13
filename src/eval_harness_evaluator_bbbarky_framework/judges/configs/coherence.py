"""Generic coherence judge: is the response internally consistent and clear?"""

from __future__ import annotations

from ..base_judge import BaseJudge


def make_coherence_judge() -> BaseJudge:
    return BaseJudge(
        name="coherence",
        system_prompt=(
            "You are an expert evaluator of response coherence.\n\n"
            "TASK: Score how coherent, well-structured, and logically consistent the "
            "response is on a 1-5 scale.\n\n"
            "RUBRIC:\n"
            "5 - Perfectly coherent, well-structured, logical flow throughout.\n"
            "4 - Mostly coherent; one minor logical gap or structural issue.\n"
            "3 - Partially coherent; follows in places but has confusing sections.\n"
            "2 - Mostly incoherent; hard to follow overall.\n"
            "1 - Completely incoherent or self-contradictory.\n\n"
            "EDGE CASES:\n"
            "- Single-sentence responses with no logical structure score 2.\n"
            "- Responses correct in conclusion but with broken reasoning score 3.\n\n"
            "INSTRUCTIONS:\n"
            "1. Identify any logical gaps, contradictions, or structural problems.\n"
            "2. Assign a score from the rubric.\n"
            "3. Respond ONLY with JSON — no prose outside the JSON.\n"
            "   Required keys: 'score' (int 1-5), 'label' (string '1' through '5'), "
            "'issues' (list of strings, empty if none), 'rationale' (string), "
            "'confidence' (float 0.0-1.0)."
        ),
        user_prompt_template="Response:\n{answer}",
        passing_labels=["4", "5"],
        rubric_version="v2",
    )
