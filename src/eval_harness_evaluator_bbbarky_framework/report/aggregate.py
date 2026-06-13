"""Pure aggregation: collapse per-(case, metric) rows into one summary.

No external dependencies. Pass rates carry a Wilson 95% confidence interval so
small-n slices can be read as directional vs strong.
"""

from __future__ import annotations

import math

def wilson_interval(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = passed / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _slice_stats(rows: list[dict], key: str, z: float = 1.96) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row.get(key, "all"), []).append(row)
    out: dict[str, dict] = {}
    for value, bucket in buckets.items():
        out[value] = _counts(bucket, z=z)
    return out


def _counts(rows: list[dict], z: float = 1.96) -> dict:
    n = len(rows)
    passed = sum(1 for r in rows if r["success"])
    failed = n - passed
    lo, hi = wilson_interval(passed, n, z=z)
    return {
        "n": n,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / n, 3) if n else 0.0,
        "ci_low": round(lo, 3),
        "ci_high": round(hi, 3),
    }


def summarize(rows: list[dict], z: float = 1.96) -> dict:
    """Aggregate rows into overall counts plus by-metric/kind/lang slices."""
    summary = _counts(rows, z=z)
    summary["by_metric"] = _slice_stats(rows, "metric", z=z)
    summary["by_kind"] = _slice_stats(rows, "kind", z=z)
    summary["by_lang"] = _slice_stats(rows, "lang", z=z)
    summary["ci_z"] = z
    return summary
