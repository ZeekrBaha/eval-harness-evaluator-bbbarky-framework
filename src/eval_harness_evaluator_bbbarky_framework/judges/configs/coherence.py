"""Generic coherence judge: is the response internally consistent and clear?"""

from __future__ import annotations

from ..base_judge import BaseJudge
from ..registry import register_judge

coherence_judge = register_judge(
    BaseJudge(
        name="coherence",
        system_prompt=(
            "You are a strict evaluator of response coherence. Decide whether the "
            "response is clear, internally consistent, and free of contradictions. "
            "Respond with a single JSON object containing 'label', 'issues' (a "
            "list), and 'rationale'. Use 'A - Good' for coherent responses and "
            "'B - Bad' otherwise."
        ),
        user_prompt_template="Response:\n{answer}",
        passing_labels=["A - Good"],
    )
)
