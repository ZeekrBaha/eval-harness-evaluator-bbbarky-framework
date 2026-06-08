"""``eval-harness`` command-line interface.

Subcommands:
  run      Evaluate evalsets named by a YAML config; write JSON + Markdown reports.
  convert  Convert a CSV file into an EvalSet JSON file.
  report   Re-render an existing report JSON as Markdown.

The ``run`` command instantiates evaluators from their ``module:Class``
references with no arguments, so it supports deterministic evaluators out of
the box. LLM-judge evaluators that need a model client are driven via the
library API rather than the CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ..converters.converter import csv_to_evalset
from ..models.core import EvalSet
from ..report.generator import build_report
from ..report.reporters import to_json, to_markdown
from ..runner.config import instantiate_evaluators, load_config
from ..runner.core import run_suite


def _load_evalset(path: str) -> EvalSet:
    return EvalSet.from_dict(json.loads(Path(path).read_text()))


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    evaluators = instantiate_evaluators(config.evaluators)

    rows: list[dict] = []
    for evalset_path in config.evalsets:
        evalset = _load_evalset(evalset_path)
        rows.extend(asyncio.run(run_suite(evalset, evaluators)))

    report = build_report(config.suite, rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{config.suite}_report.json").write_text(to_json(report))
    (out_dir / f"{config.suite}_report.md").write_text(to_markdown(report))
    print(
        f"{config.suite}: {report['summary']['passed']}/{report['summary']['n']} passed "
        f"({report['summary']['pass_rate']:.1%})"
    )
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    evalset = csv_to_evalset(args.input, suite=args.suite)
    Path(args.output).write_text(json.dumps(evalset.to_dict(), indent=2))
    print(f"wrote {len(evalset.cases)} cases to {args.output}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.input).read_text())
    print(to_markdown(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval-harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a suite from a YAML config")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", default="results")
    run.set_defaults(func=_cmd_run)

    convert = sub.add_parser("convert", help="convert a CSV file into an EvalSet")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--suite", required=True)
    convert.set_defaults(func=_cmd_convert)

    report = sub.add_parser("report", help="render a report JSON as Markdown")
    report.add_argument("--input", required=True)
    report.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
