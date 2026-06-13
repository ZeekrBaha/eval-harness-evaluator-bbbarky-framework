"""Tests for format converters (CSV / legacy rows -> EvalSet)."""

import io

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


def test_converter_handles_none_input_as_empty_string():
    rows = [{"id": "c1", "input": None, "response": None, "expected": "greet"}]
    evalset = rows_to_evalset("demo", rows)
    inv = evalset.cases[0].invocations[0]
    assert inv.user_input == "", f"expected '' but got {inv.user_input!r}"
    assert inv.final_response == "", f"expected '' but got {inv.final_response!r}"


def test_converter_maps_latency_ms_column(tmp_path):
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(
        "id,input,response,expected,latency_ms\n"
        "c1,hi,hello,greet,150.5\n"
    )
    evalset = csv_to_evalset(str(csv_path), suite="demo")
    inv = evalset.cases[0].invocations[0]
    assert inv.latency_ms == 150.5, f"expected 150.5 but got {inv.latency_ms!r}"


def test_converter_maps_trace_id_column(tmp_path):
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(
        "id,input,response,expected,trace_id\n"
        "c1,hi,hello,greet,abc-123\n"
    )
    evalset = csv_to_evalset(str(csv_path), suite="demo")
    inv = evalset.cases[0].invocations[0]
    assert inv.trace_id == "abc-123", f"expected 'abc-123' but got {inv.trace_id!r}"
