"""Generic groundedness judge: is the answer supported by the provided context?

Designed for RAG / retrieval agents. Reads the retrieved ``context`` an
invocation carries (see Invocation.context).
"""

from __future__ import annotations

from ..base_judge import BaseJudge
from ..registry import register_judge

groundedness_judge = register_judge(
    BaseJudge(
        name="groundedness",
        system_prompt=(
            "You are a strict evaluator of factual grounding. Decide whether the "
            "answer is fully supported by the provided context, with no claims "
            "that go beyond it. Respond with a single JSON object containing "
            "'label', 'issues' (a list of unsupported claims), 'rationale', and "
            "an optional 'confidence' between 0 and 1. Use 'A - Good' when every "
            "claim is supported and 'B - Bad' otherwise."
        ),
        user_prompt_template="Context:\n{context}\n\nAnswer:\n{answer}",
        passing_labels=["A - Good"],
        rubric_version="v1",
    )
)
