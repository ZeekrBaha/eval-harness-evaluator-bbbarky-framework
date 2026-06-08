"""Decide if an LLM judge is trustworthy enough to gate CI.

Loads sampled human labels vs judge labels, prints accuracy, Cohen's kappa, and
a confusion matrix, then applies a simple gate policy.

    uv run python examples/reliability/reliability_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

from eval_harness_evaluator_bbbarky_framework.reliability import agreement_report

KAPPA_GATE = 0.6  # below this, keep the judged metric report-only

HERE = Path(__file__).parent


def main() -> None:
    data = json.loads((HERE / "human_vs_judge.json").read_text())
    human = [row["human"] for row in data["items"]]
    judge = [row["judge"] for row in data["items"]]

    report = agreement_report(human, judge)
    print(f"n            : {report['n']}")
    print(f"accuracy     : {report['accuracy']}")
    print(f"cohen kappa  : {report['kappa']}")
    print(f"confusion    : {json.dumps(report['confusion'])}")

    if report["kappa"] >= KAPPA_GATE:
        print(f"VERDICT: kappa >= {KAPPA_GATE} -> judged metric MAY block CI.")
    else:
        print(f"VERDICT: kappa < {KAPPA_GATE} -> keep judged metric REPORT-ONLY.")


if __name__ == "__main__":
    main()
