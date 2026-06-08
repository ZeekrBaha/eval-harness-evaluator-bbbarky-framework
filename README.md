# eval-harness-evaluator-bbbarky-framework

Provider-agnostic evaluation framework for LLM agents. It combines an
LLM-as-judge library, a pattern/hybrid **evaluator** hierarchy, a YAML-driven
**runner**, **reporting** with Wilson confidence intervals, format
**converters**, and a CLI (`eval-harness`).

The core is dependency-light and provider-agnostic (litellm). ADK and Phoenix
integrations are optional extras (`[adk]`, `[phoenix]`).

## Install

```bash
uv sync --extra dev
```

## Quick start — library

```python
import asyncio
from eval_harness_evaluator_bbbarky_framework import HarnessEvaluator, Invocation
from eval_harness_evaluator_bbbarky_framework.model_clients.litellm_client import LiteLLMClient

async def main():
    client = LiteLLMClient(model="gpt-4o-mini", temperature=0.0)  # any litellm model
    harness = HarnessEvaluator(client)
    results = await harness.evaluate(
        [Invocation(user_input="Why is the sky blue?", final_response="Rayleigh scattering.")],
        ["relevance", "coherence"],
    )
    for name, result in results.items():
        print(name, result.passed, result.score)

asyncio.run(main())
```

## Quick start — CLI (offline, no API key)

```bash
uv run eval-harness run --config examples/config/faq_router.yaml --output-dir results
# -> results/faq_router_report.json and results/faq_router_report.md
```

Convert a CSV of cases into an EvalSet:

```bash
uv run eval-harness convert --input cases.csv --output suite.evalset.json --suite my_suite
```

## Concepts

- **Models** (`models/core.py`) — `EvalSet`, `EvalCase`, `Invocation`, `EvalResult`, `JudgeResult`.
- **Judges** (`judges/`) — `BaseJudge` builds a prompt, calls a `ModelClient`, parses a verdict. Generic judges (`relevance`, `coherence`) self-register.
- **Evaluators** (`evaluators/`) — `LabelMatchEvaluator`, `LlmJudgeEvaluator`, `JsonSchemaEvaluator`, scorers (`ExactMatch`, `ContainsKeywords`, `FuzzyF1`), and `CompositeEvaluator`.
- **Model clients** (`model_clients/`) — a `ModelClient` Protocol + the litellm-backed default; bring your own.
- **Runner** (`runner/`) — `load_config()` + `run_suite()` produce flat result rows.
- **Report** (`report/`) — `summarize()` with Wilson 95% CIs, plus JSON and Markdown reporters.

## Development

Built strictly test-first (TDD).

```bash
uv run python -m pytest        # tests
uv run ruff check src tests    # lint
uv run ruff format src tests   # format
uv run mypy src                # types
```

## License

MIT.
