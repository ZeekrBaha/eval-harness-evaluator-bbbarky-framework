"""Nested evaluator specs: recursive object construction from config."""


from eval_harness_evaluator_bbbarky_framework.evaluators.composite import CompositeEvaluator
from eval_harness_evaluator_bbbarky_framework.evaluators.response_scorers import (
    ContainsKeywordsScorer,
    ScorerEvaluator,
)
from eval_harness_evaluator_bbbarky_framework.models.core import Invocation
from eval_harness_evaluator_bbbarky_framework.runner.config import (
    build_object,
    instantiate_evaluators,
)

P = "eval_harness_evaluator_bbbarky_framework.evaluators"
S = "eval_harness_evaluator_bbbarky_framework.evaluators.response_scorers"


def test_build_object_constructs_scorer_with_params():
    obj = build_object({"type": f"{S}:ContainsKeywordsScorer", "params": {"keywords": ["refund"]}})
    assert isinstance(obj, ContainsKeywordsScorer)
    assert obj.keywords == ["refund"]


def test_build_object_builds_nested_scorer_evaluator():
    spec = {
        "type": f"{S}:ScorerEvaluator",
        "params": {
            "scorer": {"type": f"{S}:ContainsKeywordsScorer", "params": {"keywords": ["refund"]}},
            "threshold": 1.0,
        },
    }
    ev = build_object(spec)
    assert isinstance(ev, ScorerEvaluator)
    assert isinstance(ev.scorer, ContainsKeywordsScorer)
    assert ev.threshold == 1.0


async def test_nested_composite_evaluator_from_config():
    specs = [
        {
            "type": f"{P}.composite:CompositeEvaluator",
            "name": "safety_gate",
            "params": {
                "evaluators": [
                    {
                        "type": f"{P}.json_schema:JsonSchemaEvaluator",
                        "params": {"required_keys": ["intent"]},
                    },
                    {
                        "type": f"{S}:ScorerEvaluator",
                        "params": {
                            "scorer": {
                                "type": f"{S}:ContainsKeywordsScorer",
                                "params": {"keywords": ["intent"]},
                            },
                            "threshold": 1.0,
                        },
                    },
                ]
            },
        }
    ]
    evaluators = instantiate_evaluators(specs)
    gate = evaluators["safety_gate"]
    assert isinstance(gate, CompositeEvaluator)
    # response satisfies both sub-evaluators -> passes
    result = await gate.evaluate_invocations(
        [Invocation(user_input="q", final_response='{"intent": "buy"}')], expected=None
    )
    assert result.passed is True


def test_string_and_flat_specs_still_work():
    evaluators = instantiate_evaluators(
        [
            f"{P}.label_match:LabelMatchEvaluator",
            {
                "type": f"{P}.json_schema:JsonSchemaEvaluator",
                "name": "schema",
                "params": {"required_keys": ["a"]},
            },
        ]
    )
    assert "label_match" in evaluators
    assert evaluators["schema"].required_keys == ["a"]


def test_lists_of_plain_values_are_not_treated_as_specs():
    # required_keys is a list of strings, must pass through untouched
    ev = build_object(
        {"type": f"{P}.json_schema:JsonSchemaEvaluator", "params": {"required_keys": ["x", "y"]}}
    )
    assert ev.required_keys == ["x", "y"]
