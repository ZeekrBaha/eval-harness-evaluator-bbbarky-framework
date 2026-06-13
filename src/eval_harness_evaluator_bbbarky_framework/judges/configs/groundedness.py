"""Generic groundedness judge: is the answer supported by the provided context?

Designed for RAG / retrieval agents. Reads the retrieved ``context`` an
invocation carries (see Invocation.context).
"""

from __future__ import annotations

from ..base_judge import BaseJudge


def make_groundedness_judge() -> BaseJudge:
    return BaseJudge(
        name="groundedness",
        system_prompt=(
            "You are an expert evaluator of factual grounding.\n\n"
            "TASK: Score how well the answer is supported by the provided context "
            "on a 1-5 scale.\n\n"
            "RUBRIC:\n"
            "5 - Every claim directly supported by the provided context.\n"
            "4 - Nearly all claims supported; one minor unsupported detail.\n"
            "3 - Core claims supported but notable unsupported assertions present.\n"
            "2 - Some claims supported but significant fabrications present.\n"
            "1 - Answer contradicts or ignores the context entirely.\n\n"
            "EDGE CASES:\n"
            "- Claims consistent with but not explicitly in context score 3-4 "
            "(note as inference).\n"
            "- If context is empty, score 1.\n\n"
            "INSTRUCTIONS:\n"
            "1. List any claims not directly supported by the context.\n"
            "2. Assign a score from the rubric.\n"
            "3. Respond ONLY with JSON — no prose outside the JSON.\n"
            "   Required keys: 'score' (int 1-5), 'label' (string '1' through '5'), "
            "'issues' (list of strings, empty if none), 'rationale' (string), "
            "'confidence' (float 0.0-1.0)."
        ),
        user_prompt_template="Context:\n{context}\n\nAnswer:\n{answer}",
        passing_labels=["4", "5"],
        rubric_version="v2",
    )
