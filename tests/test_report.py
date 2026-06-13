"""Tests for aggregation (Wilson CI), report building, and reporters."""

import json

import pytest

from eval_harness_evaluator_bbbarky_framework.report.aggregate import summarize, wilson_interval
from eval_harness_evaluator_bbbarky_framework.report.generator import build_report
from eval_harness_evaluator_bbbarky_framework.report.reporters import to_json, to_markdown


def _rows():
    return [
        {
            "id": "c1",
            "metric": "label_match",
            "kind": "routing",
            "lang": "en",
            "success": True,
            "score": 1.0,
        },
        {
            "id": "c2",
            "metric": "label_match",
            "kind": "routing",
            "lang": "en",
            "success": False,
            "score": 0.0,
        },
        {
            "id": "c3",
            "metric": "label_match",
            "kind": "faq",
            "lang": "en",
            "success": True,
            "score": 1.0,
        },
    ]


# --- wilson -------------------------------------------------------------


def test_wilson_interval_within_unit_range():
    lo, hi = wilson_interval(8, 10)
    assert 0.0 <= lo <= 0.8 <= hi <= 1.0


def test_wilson_interval_matches_known_value():
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.490, abs=0.01)
    assert hi == pytest.approx(0.943, abs=0.01)


def test_wilson_interval_handles_zero_n():
    assert wilson_interval(0, 0) == (0.0, 0.0)


# --- summarize ----------------------------------------------------------


def test_summarize_overall_counts():
    summary = summarize(_rows())
    assert summary["n"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["pass_rate"] == pytest.approx(2 / 3, abs=0.001)


def test_summarize_by_metric_slice_has_ci():
    summary = summarize(_rows())
    lm = summary["by_metric"]["label_match"]
    assert lm["n"] == 3
    assert lm["passed"] == 2
    assert "ci_low" in lm and "ci_high" in lm
    assert lm["ci_low"] <= lm["pass_rate"] <= lm["ci_high"]


def test_summarize_by_kind_slice():
    summary = summarize(_rows())
    assert summary["by_kind"]["routing"]["n"] == 2
    assert summary["by_kind"]["faq"]["passed"] == 1


# --- report + reporters -------------------------------------------------


def test_build_report_includes_suite_and_summary():
    report = build_report("demo", _rows())
    assert report["suite"] == "demo"
    assert report["summary"]["n"] == 3


def test_to_json_is_parseable():
    report = build_report("demo", _rows())
    parsed = json.loads(to_json(report))
    assert parsed["suite"] == "demo"


def test_to_markdown_mentions_suite_and_pass_rate():
    report = build_report("demo", _rows())
    md = to_markdown(report)
    assert "demo" in md
    assert "Pass rate" in md or "pass rate" in md


def test_summarize_uses_custom_z_value():
    rows = _rows()
    summary_95 = summarize(rows)
    summary_99 = summarize(rows, z=2.576)
    assert summary_99["ci_z"] == 2.576
    assert summary_99["ci_low"] < summary_95["ci_low"]
    assert summary_99["ci_high"] > summary_95["ci_high"]


def test_build_report_includes_metadata():
    result = build_report("my_suite", [])
    assert result["schema_version"] == "1.1"
    assert "generated_at" in result and (
        result["generated_at"].endswith("Z") or "+00:00" in result["generated_at"]
    )
    assert "run_id" in result and len(result["run_id"]) == 36
