"""Tests for config loading and the suite runner."""

import pytest

from eval_harness_evaluator_bbbarky_framework.evaluators.label_match import LabelMatchEvaluator
from eval_harness_evaluator_bbbarky_framework.models.core import EvalCase, EvalSet, Invocation
from eval_harness_evaluator_bbbarky_framework.runner.config import (
    RunConfig,
    load_class_from_path,
    load_config,
)
from eval_harness_evaluator_bbbarky_framework.runner.core import run_suite


def _case(cid, response, expected, **metadata):
    return EvalCase(
        id=cid,
        invocations=[Invocation(user_input="q", final_response=response)],
        expected=expected,
        metadata=metadata,
    )


# --- config -------------------------------------------------------------


def test_load_config_parses_required_fields(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "suite: demo\nevalsets:\n  - data/regression.evalset.json\nevaluators:\n  - mod:Cls\n"
    )
    config = load_config(str(cfg))
    assert isinstance(config, RunConfig)
    assert config.suite == "demo"
    assert config.evalsets == ["data/regression.evalset.json"]
    assert config.evaluators == ["mod:Cls"]


def test_load_config_rejects_empty_evalsets(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("suite: demo\nevalsets: []\nevaluators:\n  - mod:Cls\n")
    with pytest.raises(ValueError):
        load_config(str(cfg))


def test_load_class_from_path_imports_evaluator():
    cls = load_class_from_path(
        "eval_harness_evaluator_bbbarky_framework.evaluators.label_match:LabelMatchEvaluator"
    )
    assert cls is LabelMatchEvaluator


# --- run_suite ----------------------------------------------------------


async def test_run_suite_produces_one_row_per_case_and_evaluator():
    evalset = EvalSet(
        name="demo",
        cases=[
            _case("c1", "Sales", "Sales", kind="routing"),
            _case("c2", "Refund", "Sales", kind="routing"),
        ],
    )
    rows = await run_suite(evalset, {"label_match": LabelMatchEvaluator()})
    assert len(rows) == 2
    by_id = {r["id"]: r for r in rows}
    assert by_id["c1"]["success"] is True
    assert by_id["c1"]["metric"] == "label_match"
    assert by_id["c1"]["kind"] == "routing"
    assert by_id["c2"]["success"] is False


async def test_run_suite_runs_every_evaluator_over_every_case():
    evalset = EvalSet(name="demo", cases=[_case("c1", "Sales", "Sales")])
    rows = await run_suite(
        evalset,
        {"a": LabelMatchEvaluator(), "b": LabelMatchEvaluator()},
    )
    assert {r["metric"] for r in rows} == {"a", "b"}
    assert len(rows) == 2


# --- concurrency ------------------------------------------------------------

import asyncio
import time

from eval_harness_evaluator_bbbarky_framework.evaluators.base import Evaluator
from eval_harness_evaluator_bbbarky_framework.models.core import EvalResult


class SlowEvaluator(Evaluator):
    """Sleeps 0.05 s per call to make sequential vs concurrent timing obvious."""

    metric_name = "slow"

    async def evaluate_invocations(self, invocations, expected):
        await asyncio.sleep(0.05)
        return EvalResult(
            score=1.0,
            passed=True,
            per_invocation=[],
        )


async def test_run_suite_executes_concurrently():
    """4 cases × 1 evaluator (each 0.05 s) must finish in < 0.15 s total.

    Sequential execution would take ~0.20 s; concurrent takes ~0.05 s.
    """
    evalset = EvalSet(
        name="perf",
        cases=[
            _case("c1", "a", "a"),
            _case("c2", "b", "b"),
            _case("c3", "c", "c"),
            _case("c4", "d", "d"),
        ],
    )
    start = time.monotonic()
    rows = await run_suite(evalset, {"slow": SlowEvaluator()})
    elapsed = time.monotonic() - start

    assert len(rows) == 4
    assert elapsed < 0.15, (
        f"run_suite took {elapsed:.3f}s — expected concurrent execution (~0.05s), "
        f"got sequential-like timing (~0.20s)"
    )
