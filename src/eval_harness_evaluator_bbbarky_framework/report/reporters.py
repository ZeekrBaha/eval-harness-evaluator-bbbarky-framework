"""Render a report to JSON or Markdown."""

from __future__ import annotations

import json


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def to_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        f"# Eval report: {report['suite']}",
        "",
        f"- Cases evaluated (rows): {summary['n']}",
        f"- Pass rate: {summary['pass_rate']:.1%} "
        f"({summary['passed']}/{summary['n']}), "
        f"95% CI [{summary['ci_low']:.3f}, {summary['ci_high']:.3f}]",
        "",
        "## By metric",
        "",
        "| metric | n | pass_rate | 95% CI |",
        "| --- | --- | --- | --- |",
    ]
    for metric, stats in sorted(report["summary"]["by_metric"].items()):
        lines.append(
            f"| {metric} | {stats['n']} | {stats['pass_rate']:.3f} | "
            f"[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}] |"
        )
    return "\n".join(lines) + "\n"
