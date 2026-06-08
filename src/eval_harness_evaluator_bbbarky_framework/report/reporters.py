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

    lines += _failure_lines(report.get("rows", []))
    return "\n".join(lines) + "\n"


def _failure_lines(rows: list[dict], limit: int = 10) -> list[str]:
    """Render up to ``limit`` failing rows with judge rationale/issues, if any."""
    failures = [r for r in rows if not r.get("success", True)]
    if not failures:
        return []
    out = ["", "## Failure examples", ""]
    for row in failures[:limit]:
        details = row.get("details") or {}
        bits = [f"**{row['id']}** ({row['metric']})"]
        if details.get("rationale"):
            bits.append(f"— {details['rationale']}")
        if details.get("issues"):
            bits.append(f"[issues: {', '.join(map(str, details['issues']))}]")
        out.append("- " + " ".join(bits))
    if len(failures) > limit:
        out.append(f"- …and {len(failures) - limit} more failures.")
    return out
