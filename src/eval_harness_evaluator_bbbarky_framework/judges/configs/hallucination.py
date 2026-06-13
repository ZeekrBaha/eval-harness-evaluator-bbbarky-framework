"""Hallucination judge: does the generated answer introduce fabricated claims?"""

from __future__ import annotations

from ..base_judge import BaseJudge


def make_hallucination_judge() -> BaseJudge:
    return BaseJudge(
        name="hallucination",
        system_prompt=(
            "You are an expert fact-checker evaluating whether a generated answer "
            "introduces hallucinated claims not supported by the reference answer.\n\n"
            "TASK: Score the generated answer on a 1-5 hallucination-free scale.\n\n"
            "RUBRIC:\n"
            "5 - No hallucinations: all claims match or are consistent with the reference.\n"
            "4 - Trivial inference: one minor claim goes slightly beyond the reference but is plausible.\n"
            "3 - Partial hallucination: some unsupported claims; core facts are correct.\n"
            "2 - Significant hallucination: key claims fabricated or contradict the reference.\n"
            "1 - Complete hallucination: answer contradicts or ignores the reference entirely.\n\n"
            "EDGE CASES:\n"
            "- An answer that says 'I don't know' when reference has an answer scores 3.\n"
            "- Paraphrasing that preserves meaning scores 5.\n"
            "- Adding correct background context not in reference scores 4.\n\n"
            "INSTRUCTIONS:\n"
            "1. List any hallucinated claims you find (empty list if none).\n"
            "2. Assign a score from the rubric.\n"
            "3. Respond ONLY with JSON — no prose outside the JSON.\n"
            "   Required keys: 'score' (int 1-5), 'label' (string '1'-'5'), "
            "'hallucinated_claims' (list of strings, use key 'issues' for framework compat), "
            "'rationale' (string), 'confidence' (float 0.0-1.0)."
        ),
        user_prompt_template="Reference answer:\n{reference}\n\nGenerated answer:\n{answer}",
        passing_labels=["4", "5"],
        rubric_version="v1",
    )
