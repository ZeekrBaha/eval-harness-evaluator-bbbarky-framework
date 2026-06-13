"""Generic relevance judge: is the answer on-topic for the question?"""

from __future__ import annotations

from ..base_judge import BaseJudge


def make_relevance_judge() -> BaseJudge:
    return BaseJudge(
        name="relevance",
        system_prompt=(
            "You are an expert evaluator of answer relevance.\n\n"
            "TASK: Score how well the answer addresses the user's question on a 1-5 scale.\n\n"
            "RUBRIC:\n"
            "5 - Fully relevant: directly addresses all parts of the question.\n"
            "4 - Mostly relevant: addresses the main intent; minor aspect omitted.\n"
            "3 - Partially relevant: addresses some parts but misses key aspects.\n"
            "2 - Marginally relevant: touches the topic but does not answer.\n"
            "1 - Irrelevant: off-topic or completely unhelpful.\n\n"
            "EDGE CASES:\n"
            "- A clarifying question back to the user scores <= 2.\n"
            "- A correct answer to a different question scores <= 2.\n"
            "- Multi-part questions: score reflects the worst-addressed sub-question.\n\n"
            "INSTRUCTIONS:\n"
            "1. State which parts of the question are addressed and which are not.\n"
            "2. Assign a score from the rubric.\n"
            "3. Respond ONLY with JSON — no prose outside the JSON.\n"
            "   Required keys: 'score' (int 1-5), 'label' (string '1' through '5'), "
            "'issues' (list of strings, empty if none), 'rationale' (string), "
            "'confidence' (float 0.0-1.0)."
        ),
        user_prompt_template="Question:\n{question}\n\nAnswer:\n{answer}",
        passing_labels=["4", "5"],
        rubric_version="v2",
    )
