"""The reliability example data must produce a valid agreement report."""

import json
from pathlib import Path

from eval_harness_evaluator_bbbarky_framework.reliability import agreement_report

DATA = Path(__file__).parents[1] / "examples" / "reliability" / "human_vs_judge.json"


def test_reliability_example_produces_agreement_report():
    items = json.loads(DATA.read_text())["items"]
    human = [r["human"] for r in items]
    judge = [r["judge"] for r in items]
    report = agreement_report(human, judge)
    assert report["n"] == len(items)
    # 6 of 8 agree
    assert report["accuracy"] == 0.75
    assert -1.0 <= report["kappa"] <= 1.0
    assert "A - Good" in report["confusion"]
