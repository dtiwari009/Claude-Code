# Agent Eval Harness — Reference Blueprint

A minimal but real agent evaluation harness demonstrating a six-dimensional
scoring framework: outcome correctness, trajectory quality, tool precision,
cost/latency, safety, and **LLM-as-judge response quality with bias mitigations**.

This is a **learning artifact**, not a production framework. For production,
use one of: Inspect (UK AISI), DeepEval, LangSmith, Braintrust.

## Architecture

```
environment.py    Mock customer-support backend (tools + state)
tasks.py          The eval suite — 5 tasks, each surfacing a failure class
agent.py          Minimal tool-use loop, fully instrumented
mock_llm.py       Deterministic mock LLM (for CI / harness self-test)
graders.py        Six-dimensional scorers
judge.py          LLM-as-judge with position/self-preference/verbosity mitigations
trace_utils.py    Helpers for extracting human-readable text from traces
runner.py         Orchestrator: calibrate judge, N runs/task, pass^k, aggregate
```

## Key design decisions

1. **Mocked environment, not real APIs.** Reliability measurement requires
   determinism in the environment so observed variance is attributable to the
   agent, not to flaky third parties.
2. **Pass^k built in from day one.** Single-run pass rate is theater for
   production deployments. The τ-bench paper's central finding: models that
   "succeed" 50% on a single try often fall under 25% on pass^8.
3. **Six dimensions, not one number.** A scorecard, not a leaderboard.
   Cheap-and-wrong is the worst quadrant; you can only see it with a
   multi-dimensional view.
4. **Deterministic graders where possible, LLM-judge only where necessary.**
   Programmatic checks (did refund get issued? was right policy applied?)
   are the gold standard. LLM-as-judge is the fallback for unstructured
   qualities (prose quality, professionalism, completeness of explanation).
5. **Mock-LLM mode for CI.** A real eval harness must be runnable in CI to
   detect regressions in the harness itself.

## LLM-as-judge: the bias mitigations that matter

The naive "ask GPT-4 to score it" pattern is systematically wrong. This harness
implements the patterns the literature actually recommends:

| Bias | Mitigation in this harness |
|------|----------------------------|
| **Position bias** | Every pairwise comparison runs in BOTH orderings. A "win" requires the judge to prefer the same response in both. Disagreements are flagged as position-bias incidents and EXCLUDED from win-rates. |
| **Self-preference bias** | Judge model family is asserted at construction time to differ from agent family. Strict check that throws if you try to use a same-family judge. |
| **Verbosity bias** | Response lengths are logged alongside verdicts so length-correlation can be detected post-hoc. |
| **Rubric sensitivity** | Rubric is fixed in code, not regenerated per task. |
| **Calibration** | Judge is run against a calibration set with known-good and known-bad responses BEFORE any production scoring. Agreement < 0.85 triggers a "do not trust" flag. |

## Running

```bash
# Real API (requires ANTHROPIC_API_KEY env var):
python runner.py claude-sonnet-4-5 5

# With LLM-as-judge enabled:
python runner.py claude-sonnet-4-5 5 --judge

# Harness self-test with mock LLM and mock judge (no API key needed):
python runner.py claude-sonnet-4-5 3 demo --judge
```

## What's deliberately NOT included (and why)

- **Multi-judge ensembles.** Running N different judges and taking majority
  vote is the next-step mitigation. Hook is in place; adding it is ~50 lines.
- **Confidence-weighted scoring.** Currently any non-confident verdict is
  excluded; production systems should weight by judge confidence.
- **Active learning for calibration set refresh.** Calibration sets decay.
  Production systems should periodically add edge cases from production
  traffic and re-label.
- **No retrieval/RAG component.** This is a tool-use agent, not a RAG agent.
  RAG evals (faithfulness, context relevance) belong in a separate harness.
- **No statistical significance testing on pass rates.** For real comparisons
  you'd want Wilson confidence intervals or bootstrap CIs on the success
  rates. Easy to add; left out for clarity.

## Extending it

The most valuable extensions, in order of leverage:
1. **Add real tasks from your domain.** This is the single most important
   thing. Public benchmarks are contaminated; your task suite is your moat.
2. **Add a regression-tracking dashboard.** Store reports over time, alert
   on regressions. Langfuse/Braintrust shine here.
3. **Add scaffolding variations.** Test ReAct vs plan-and-execute vs simple
   tool-loop on the same model, same tasks. The right scaffold can move
   scores 10-20 points.
4. **Add multi-judge ensembles.** The variance across different judge
   families is itself a useful signal about rubric ambiguity.

## The principle behind it all

> **"If your eval can be gamed, it will be. If your judge can be biased, it
> is. The job of an eval harness is not to produce a number — it's to
> produce a number you can defend in a design review."**

Every design choice here flows from that.
