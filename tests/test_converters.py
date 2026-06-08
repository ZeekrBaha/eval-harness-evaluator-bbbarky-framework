"""Tests for format converters (CSV / legacy rows -> EvalSet)."""

from eval_harness_evaluator_bbbarky_framework.converters.converter import (
    csv_to_evalset,
    rows_to_evalset,
)


def test_rows_to_evalset_builds_cases():
    rows = [
        {"id": "c1", "input": "hi", "response": "hello", "expected": "greet", "kind": "smalltalk"},
        {"id": "c2", "input": "bye", "response": "goodbye", "expected": "farewell"},
    ]
    evalset = rows_to_evalset("demo", rows)
    assert evalset.name == "demo"
    assert len(evalset.cases) == 2
    assert evalset.cases[0].invocations[0].final_response == "hello"
    assert evalset.cases[0].expected == "greet"
    assert evalset.cases[0].metadata["kind"] == "smalltalk"


def test_csv_to_evalset_reads_columns(tmp_path):
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(
        "id,input,response,expected,kind\n"
        "c1,hi,hello,greet,smalltalk\n"
        "c2,bye,goodbye,farewell,smalltalk\n"
    )
    evalset = csv_to_evalset(str(csv_path), suite="demo")
    assert evalset.name == "demo"
    assert len(evalset.cases) == 2
    assert evalset.cases[1].id == "c2"
    assert evalset.cases[1].expected == "farewell"
