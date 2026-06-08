# eval-harness-evaluator-bbbarky-framework

A **provider-agnostic evaluation framework for LLM agents**. You give it cases
(input + what the agent replied + what was expected), it scores each reply with
**evaluators**, aggregates pass-rates with **confidence intervals**, and writes a
report. Think **pytest, but for non-deterministic agent output** — it mixes
deterministic checks with LLM-as-judge grading.

Built strictly test-first (TDD). The core is dependency-light and talks to any
LLM through one seam (litellm); ADK and Phoenix are optional extras.

```bash
uv sync --extra dev
uv run eval-harness run --config examples/config/faq_router.yaml --output-dir results
```

---

## 1. Mental model — two halves

An eval harness is always two independent halves. Keeping them separate is the
whole game:

| Half | Question it answers | Lives in |
|---|---|---|
| **System Under Test (SUT)** | "What did the agent reply?" | **Your code, not this repo.** You run the agent and capture the reply. |
| **Evaluation** | "Was that reply good?" | This framework — evaluators, judges, runner, report. |

The framework **never calls your agent**. You produce `(user_input,
final_response)` pairs however you like, hand them over, and the framework
grades them. That decoupling is why *any* text-in/text-out agent can be the SUT
(see §6).

---

## 2. Where the SUT lives (read this — the #1 source of confusion)

There is no "agent" inside this package. The unit of input is an
**`Invocation`**: one user turn plus the agent's final response to it.

```python
from eval_harness_evaluator_bbbarky_framework import Invocation
Invocation(user_input="I was charged twice", final_response="Billing")
```

You build invocations in one of two ways:

1. **Online** — call your live agent, record each `final_response`, build
   invocations from the transcript.
2. **Offline / golden** — pre-record responses into an `*.evalset.json` file (or
   a CSV you convert), then grade them repeatedly with zero agent cost.

Because the SUT is just data by the time it reaches the framework, evaluation is
**reproducible and cheap to re-run** — the expensive, flaky part (calling the
agent) is done once and frozen.

---

## 3. Architecture — layers, dependencies point down

```
models/core      ← data types. EvalSet → EvalCase → Invocation. EvalResult, JudgeResult.
                   (knows nothing about any other layer)
      ↑
model_clients    ← ModelClient Protocol + LiteLLMClient. THE only seam to an LLM.
                   network is injected → tests run fully offline.
judges           ← BaseJudge: prompt → ModelClient → parse JSON verdict → JudgeResult.
                   registry + bundled judges (relevance, coherence).
      ↑
evaluators       ← Evaluator ABC. evaluate_invocations(invocations, expected) → EvalResult.
                   concrete: label_match, llm_judge, scorers, json_schema, composite.
      ↑
runner           ← load_config(yaml) + run_suite() → flat rows {id,metric,success,score,kind,lang}
report           ← summarize() with Wilson 95% CI → JSON + Markdown
cli              ← eval-harness {run, convert, report}
facade           ← HarnessEvaluator: one-call evaluate(invocations, judge_names)
```

**Key invariant: nothing imports a vendor.** The provider lives behind the
`ModelClient` protocol. Swapping OpenAI → Anthropic → Gemini → local is a single
model string (see §9). `models/` depends on nothing; `model_clients/` is
*injected* into judges and evaluators, never the reverse.

---

## 4. Data flow

1. **Case** = `EvalCase`: an `id`, a list of `Invocation`s (multi-turn allowed),
   an `expected` outcome, and free-form `metadata` (e.g. `kind`, `lang`).
2. **Runner** loops every case × every evaluator → one flat **result row** each.
3. **Evaluator** scores the response → `EvalResult{score, passed, per_invocation}`.
   - Deterministic evaluators (label_match, json_schema, scorers) = pure logic,
     **no LLM, no cost, no key**.
   - `LlmJudgeEvaluator` sends the response to a judge LLM, parses the verdict.
4. **Report** buckets rows → pass-rate per `metric` / `kind` / `lang`, each with a
   **Wilson 95% CI** so `3/4 passed` reads as *wide and uncertain*, not solid.

Result row shape (what the report aggregates):

```json
{"id": "route-billing", "metric": "label_match", "kind": "routing",
 "lang": "all", "success": true, "score": 1.0}
```

---

## 5. The evaluator stack

### Deterministic (no key, no judge, free)

| Evaluator | Checks | Use for |
|---|---|---|
| `LabelMatchEvaluator` | final response == one of the expected labels (case-insensitive) | routing / classification / tool choice |
| `JsonSchemaEvaluator` | response parses as a JSON object containing required keys | structured-output agents |
| `ExactMatchScorer` | exact string match vs reference | precise outputs |
| `ContainsKeywordsScorer` | fraction of required keywords present | key-concept coverage |
| `FuzzyF1Scorer` | token-overlap F1 vs reference | summaries / paraphrase |
| `CompositeEvaluator` | logical AND of several evaluators (mean score) | hybrid gates |

`ScorerEvaluator` wraps any scorer with a pass `threshold`.

### LLM-as-judge (needs a model client + provider key)

| Judge | Grades | Passing label |
|---|---|---|
| `relevance` | is the answer on-topic for the question? | `A - Good` |
| `coherence` | is the response clear and internally consistent? | `A - Good` |

`LlmJudgeEvaluator` bridges any `BaseJudge` to any `ModelClient`. Judges return a
structured `JudgeResult{label, issues, rationale, raw_response, score}` — parsing
is lenient (handles JSON wrapped in prose or ```json fences). Judge temperature
defaults to deterministic when you set it on the client.

---

## 6. What can be the System Under Test (SUT)

**Anything that takes text in and produces text out.**

| SUT | What you grade | Evaluator(s) |
|---|---|---|
| **Customer-support router** | message → queue label | `LabelMatchEvaluator` (shipped `faq_router` example) |
| **RAG / Q&A bot** | answer grounded & on-topic | `LlmJudgeEvaluator` (`relevance`, + a `groundedness` judge you add) |
| **Structured-output agent** | valid JSON intent + slots | `JsonSchemaEvaluator` |
| **Summarizer / rewriter** | faithful to source, good quality | `FuzzyF1Scorer` + a judge |
| **Multi-turn chatbot** | coherent across turns | `coherence` judge over a multi-`Invocation` case |
| **Tool-calling agent** | picked the right tool / final answer | `LabelMatchEvaluator` / `JsonSchemaEvaluator` |

The framework doesn't care what language or framework your agent is written in —
only that you can capture its responses as text.

---

## 7. Typical workflow to test a real SUT

1. **Run your agent** over a set of inputs; record each `final_response`.
2. **Write an evalset** — `*.evalset.json` (or a CSV → `eval-harness convert`).
3. **Pick evaluators** in a YAML config.
4. **`eval-harness run`** → a report with pass-rate + confidence interval.
5. **Wire into CI**:
   - Gate merges on a **pass-rate floor** for judged metrics.
   - **Deterministic safety checks must be 100%** (e.g. "never leaks PII",
     "always valid JSON") — these are cheap, keyless, and should never regress.
   - Keep judge/flaky scores reported but non-blocking until you've validated the
     judge.

The point: **regression-test a non-deterministic agent** the way you unit-test
deterministic code — catch quality drops *before* they ship.

### CI sketch

```yaml
# offline gate — no keys, runs on every PR
- run: uv run python -m pytest                      # framework's own 55 tests
- run: uv run eval-harness run --config suite.yaml --output-dir results
- run: |
    uv run python - <<'PY'
    import json; from pathlib import Path
    s = json.loads(Path("results/my_suite_report.json").read_text())["summary"]
    # deterministic safety gate: must be perfect
    sc = s["by_metric"]["json_schema"]
    assert sc["pass_rate"] == 1.0, f"schema regressions: {sc}"
    # quality floor: pass-rate must clear a bar
    assert s["pass_rate"] >= 0.85, f"overall pass-rate dropped: {s['pass_rate']}"
    PY
```

---

## 8. Reporting — and why Wilson confidence intervals

`summarize()` collapses result rows into overall counts plus `by_metric`,
`by_kind`, and `by_lang` slices. Every pass-rate carries a **Wilson 95%
interval**.

Why: a naive pass-rate hides sample size. `3/3 = 100%` and `300/300 = 100%` look
identical but mean very different things. Wilson gives:

- `3/3` → pass_rate 1.0, CI roughly `[0.44, 1.0]` (wide — directional only)
- `300/300` → pass_rate 1.0, CI roughly `[0.99, 1.0]` (strong)

So small-n slices (a new `kind`, a rare `lang`) read honestly instead of looking
falsely solid. Reporters: `to_json()` and `to_markdown()`.

---

## 9. Providers — swap models with one string

The only LLM seam is the `ModelClient` protocol:

```python
async def generate(self, messages: list[dict], **opts) -> str: ...
```

The default implementation is `LiteLLMClient`, so any litellm-supported model
works by name:

```python
from eval_harness_evaluator_bbbarky_framework.model_clients.litellm_client import LiteLLMClient

LiteLLMClient(model="gpt-4o-mini")                 # OpenAI       (OPENAI_API_KEY)
LiteLLMClient(model="anthropic/claude-3-5-sonnet") # Anthropic    (ANTHROPIC_API_KEY)
LiteLLMClient(model="gemini/gemini-2.5-flash")     # Google       (GEMINI_API_KEY)
LiteLLMClient(model="ollama/llama3")               # local        (no key)
```

It retries transient failures and normalizes provider errors to
`ModelClientError`. Need a native SDK or a custom gateway? Implement `generate()`
on your own object and pass it anywhere a `ModelClient` is expected — no subclass
required (it's a `Protocol`).

---

## 10. How to run

```bash
# install (dev tools: pytest, ruff, mypy)
uv sync --extra dev

# OFFLINE — no keys, CI-safe: the framework's own test suite (55 tests)
uv run python -m pytest

# OFFLINE — run the shipped example (deterministic label-match, no LLM call)
uv run eval-harness run --config examples/config/faq_router.yaml --output-dir results
#   -> results/faq_router_report.json
#   -> results/faq_router_report.md

# Convert a CSV of cases into an EvalSet
uv run eval-harness convert --input cases.csv --output suite.evalset.json --suite my_suite

# Re-render an existing report JSON as Markdown
uv run eval-harness report --input results/faq_router_report.json
```

### Library — one-call facade

```python
import asyncio
from eval_harness_evaluator_bbbarky_framework import HarnessEvaluator, Invocation
from eval_harness_evaluator_bbbarky_framework.model_clients.litellm_client import LiteLLMClient

async def main():
    client = LiteLLMClient(model="gpt-4o-mini", temperature=0.0)
    harness = HarnessEvaluator(client)
    results = await harness.evaluate(
        [Invocation(user_input="Why is the sky blue?", final_response="Rayleigh scattering.")],
        ["relevance", "coherence"],
    )
    for name, result in results.items():
        print(name, result.passed, result.score)

asyncio.run(main())
```

### Library — explicit evaluators + runner

```python
import asyncio
from eval_harness_evaluator_bbbarky_framework import EvalSet, run_suite, LabelMatchEvaluator
from eval_harness_evaluator_bbbarky_framework.report.generator import build_report
from eval_harness_evaluator_bbbarky_framework.report.reporters import to_markdown

evalset = EvalSet.from_dict({...})
rows = asyncio.run(run_suite(evalset, {"label_match": LabelMatchEvaluator()}))
print(to_markdown(build_report("my_suite", rows)))
```

---

## 11. Keys

| Variable | Needed for |
|---|---|
| `OPENAI_API_KEY` | OpenAI models (SUT and/or judge) |
| `ANTHROPIC_API_KEY` | Anthropic models |
| `GEMINI_API_KEY` | Google models |
| *(none)* | All deterministic evaluators + offline tests + the `faq_router` example |

No secrets live in code or config — provide them as environment variables.

---

## 12. Repo map (what every file is)

```
src/eval_harness_evaluator_bbbarky_framework/
├── __init__.py              # public API exports
├── facade.py                # HarnessEvaluator: evaluate(invocations, judge_names)
├── models/
│   └── core.py              # EvalSet, EvalCase, Invocation, EvalResult,
│                            #   PerInvocationResult, JudgeResult
├── model_clients/
│   ├── base.py              # ModelClient Protocol
│   ├── litellm_client.py    # default provider-agnostic client
│   ├── retry.py             # with_retry async decorator
│   └── error.py             # ModelClientError
├── judges/
│   ├── base_judge.py        # build_messages / parse / evaluate
│   ├── registry.py          # JudgeRegistry + default registry
│   └── configs/             # bundled judges (relevance, coherence) — self-register
├── evaluators/
│   ├── base.py              # Evaluator ABC + per-invocation aggregation
│   ├── label_match.py       # classification / routing
│   ├── llm_judge.py         # bridges a BaseJudge to a ModelClient
│   ├── response_scorers.py  # ExactMatch / ContainsKeywords / FuzzyF1 + ScorerEvaluator
│   ├── json_schema.py       # JSON-with-required-keys
│   └── composite.py         # logical AND of evaluators
├── runner/
│   ├── config.py            # RunConfig + load_config(yaml) + load_class_from_path
│   └── core.py              # run_suite() → result rows
├── report/
│   ├── aggregate.py         # wilson_interval + summarize()
│   ├── generator.py         # build_report()
│   └── reporters.py         # to_json / to_markdown
├── converters/
│   └── converter.py         # csv_to_evalset / rows_to_evalset
└── cli/
    └── main.py              # eval-harness {run, convert, report}

examples/
├── config/faq_router.yaml                       # offline example config
└── faq_router/evalsets/regression.evalset.json  # 3 routing cases

tests/                       # 55 tests, one suite per module (TDD)
docs/implementation/design.md
```

---

## 13. Extending

**Add a judge** — register a `BaseJudge`; it's then available by name to
`HarnessEvaluator` and `LlmJudgeEvaluator`:

```python
from eval_harness_evaluator_bbbarky_framework.judges.base_judge import BaseJudge
from eval_harness_evaluator_bbbarky_framework.judges.registry import register_judge

register_judge(BaseJudge(
    name="groundedness",
    system_prompt="Judge whether the answer is supported by the provided context. "
                  "Reply with JSON: label ('A - Good'|'B - Bad'), issues, rationale.",
    user_prompt_template="Context:\n{context}\n\nAnswer:\n{answer}",
    passing_labels=["A - Good"],
))
```

**Add an evaluator** — subclass `Evaluator` and implement the one async method:

```python
from eval_harness_evaluator_bbbarky_framework.evaluators.base import Evaluator
from eval_harness_evaluator_bbbarky_framework.models.core import EvalResult, PerInvocationResult

class MaxLengthEvaluator(Evaluator):
    metric_name = "max_length"
    def __init__(self, limit: int): self.limit = limit
    async def evaluate_invocations(self, invocations, expected) -> EvalResult:
        per = [PerInvocationResult(score=1.0 if len(i.final_response) <= self.limit else 0.0,
                                   passed=len(i.final_response) <= self.limit)
               for i in invocations]
        return self._aggregate(per)
```

---

## 14. Tech stack

- **Python ≥ 3.12**, managed with **uv**.
- **pydantic v2** — data models and validation.
- **litellm** — one interface over every LLM provider.
- **PyYAML** — run configs.
- **rich / tqdm** — CLI output.
- **pytest + pytest-asyncio** — tests. **ruff** — lint/format. **mypy** — types.
- Optional extras: `[adk]` (google-adk), `[phoenix]` (Arize Phoenix tracing).

---

## 15. Quality & development

Built strictly test-first (TDD) — every flow has a test written before the code.

```bash
uv run python -m pytest        # 55 tests, all green
uv run ruff check src tests    # lint (clean)
uv run ruff format src tests   # format
uv run mypy src                # types (no issues)
```

The core imports with **no** google-adk or Phoenix present — extras are isolated.

---

## 16. Limitations / next steps

- Bundled judges are intentionally minimal (`relevance`, `coherence`). Add
  domain judges (groundedness, safety, faithfulness) via the registry.
- The CLI `run` instantiates evaluators with **no-arg constructors**, so it
  drives deterministic evaluators out of the box; LLM-judge runs go through the
  library API (they need a model client).
- No built-in judge-reliability check yet (e.g. Cohen's κ vs human labels) —
  recommended before you let a judged metric block CI.
- ADK runtime, recordings, simulation, and Phoenix tracing are out of the v1
  core (reserved for the `[adk]` / `[phoenix]` extras).

---

## License

MIT.
