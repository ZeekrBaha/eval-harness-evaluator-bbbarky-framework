# Design — eval-harness-evaluator-bbbarky-framework

Clean-room synthesis of two prior internal eval frameworks: a focused
LLM-as-judge library and a fuller ADK eval platform. Combines the judge library
+ provider abstraction of the former with the evaluator hierarchy, YAML runner,
reporting, converters, and CLI of the latter. Provider-agnostic; no proprietary
naming or vendor coupling.

## Positioning (what this is / is not)

This is a **generic, provider-neutral, offline evaluation library**. It is **not**
a live agent-runtime harness: it does not execute the agent, manage a live
session/ADK runtime, manage a proprietary/live runtime gateway, or checkpoint
batch execution. (It *can* make a direct provider completion call via
`LiteLLMClient` for the LLM-judge path — that is one outbound call, not a managed
gateway.) Those responsibilities belong to an upstream live-session/ADK harness.

| Concern | Upstream live-session / ADK harness | This framework |
|---|---|---|
| Live agent execution, session/ADK state | ✅ | ❌ |
| Gateway, auth, rate limits, checkpointed batches | ✅ | ❌ |
| Frozen-transcript scoring (deterministic + judge) | partial | ✅ |
| Reporting with confidence intervals | partial | ✅ |
| Judge reliability stats (κ, confusion, agreement) | — | ✅ |
| Vendor-neutral, pip-installable | — | ✅ |

**Integration path:** capture sessions upstream → export as plain records →
convert to an EvalSet via `formats/session.py` (`session_to_evalcase` for
multi-turn, `agent_session_to_evalcase` for a richer single-turn export that
also preserves history and judge configs as case `artifacts`) → score, report,
and calibrate offline. Live execution stays upstream; evaluation lives here.

## Layering

```
models/core         ← canonical data types (no deps on other layers)
   ↑
model_clients       ← ModelClient Protocol + litellm default (injected network)
judges              ← BaseJudge: prompt → ModelClient → JudgeResult; registry; configs
   ↑
evaluators          ← Evaluator ABC + concrete (label_match, llm_judge, scorers,
                       json_schema, composite). llm_judge bridges judges via ModelClient.
   ↑
runner              ← load_config(yaml) + run_suite() → flat result rows
report              ← summarize() (Wilson CI) + JSON/Markdown reporters
cli                 ← eval-harness {run, convert, report}
facade              ← HarnessEvaluator: one-call "evaluate(invocations, judge_names)"
```

Dependencies point downward only. `model_clients` is injected into judges /
evaluators, never imported by `models`.

## Key contracts

- `Evaluator.evaluate_invocations(invocations, expected) -> EvalResult` — async, so
  deterministic and LLM-backed evaluators are uniform to the runner.
- `ModelClient.generate(messages, **opts) -> str` — the only provider seam.
- `BaseJudge.build_messages(**vars)` / `parse(raw) -> JudgeResult` — lenient JSON
  extraction (handles fenced/prose-wrapped output).
- Runner rows: `{id, metric, kind, lang, success, score}` — plain dicts the report
  layer aggregates without importing evaluator types.

## Decisions

- **Provider-agnostic core via litellm**, plus a `ModelClient` Protocol so native
  or custom clients can be registered. No vendor gateway.
- **ADK / Phoenix are optional extras**, isolated so the core imports without them.
- **Wilson 95% CIs** on every pass-rate slice (overall, by_metric, by_kind, by_lang)
  so small-n deltas read as directional vs strong.
- **Strict TDD** — every flow has a test written first; all commands run via `uv`.

## Non-goals (v1)

- No ADK runtime, recordings, simulation, or Phoenix tracing in the core (extras only).
- No bench/latency CLI subcommands yet (deferred to the `[adk]` surface).
