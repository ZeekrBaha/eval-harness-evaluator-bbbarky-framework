"""Generic relevance judge: is the answer on-topic for the question?"""

from __future__ import annotations

from ..base_judge import BaseJudge
from ..registry import register_judge

relevance_judge = register_judge(
    BaseJudge(
        name="relevance",
        system_prompt=(
            "You are a strict evaluator of answer relevance. Decide whether the "
            "answer actually addresses the user's question. Respond with a single "
            "JSON object containing the keys 'label', 'issues' (a list), and "
            "'rationale'. Use the label 'A - Good' when the answer is on-topic and "
            "'B - Bad' otherwise."
        ),
        user_prompt_template="Question:\n{question}\n\nAnswer:\n{answer}",
        passing_labels=["A - Good"],
    )
)
