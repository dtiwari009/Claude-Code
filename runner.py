"""
The harness orchestrator.

DESIGN NOTE: This is where the reliability story lives. Notice that we run
each task K times (K=5 by default). This is non-negotiable in a serious
harness. From the τ-bench paper:

    "Even top models score below 50% success and fall under pass^8 of 25%
     on the same retail tasks."

A single run tells you almost nothing about production behavior. The
function `pass_at_k_estimate` and `pass_power_k` compute the two reliability
metrics that matter:

  - pass@k: "in k tries, did the agent succeed at least once?"
    (optimistic — relevant if you have retry logic)

  - pass^k: "in k tries, did the agent succeed EVERY time?"
    (pessimistic — relevant if every customer interaction must succeed)

For production, pass^k is the honest number. Most vendors hide it.
"""
import json
import sys
from pathlib import Path
from statistics import mean, stdev

from agent import run_agent
from graders import grade
from judge import LLMJudge
from tasks import TASKS


def pass_at_1(outcomes: list[bool]) -> float:
    """Plain success rate across runs."""
    return sum(outcomes) / len(outcomes) if outcomes else 0.0


def pass_power_k(outcomes: list[bool], k: int) -> float:
    """Estimate of pass^k from observed pass-rate.

    If single-run success rate is p, then probability of succeeding k
    times in a row (assuming independence) is p^k. This is the metric
    that exposes reliability problems hidden by pass@1.
    """
    p = pass_at_1(outcomes)
    return p ** k


def run_suite(model: str, runs_per_task: int = 5,
              output_dir: str = "results",
              mock_policy: str = None,
              use_judge: bool = False,
              judge_model: str = "claude-opus-4-5",
              agent_family: str = "openai") -> dict:
    """Run the full eval suite and return aggregate report.

    If use_judge=True, instantiates an LLMJudge, runs calibration first,
    and includes the judge_quality dimension. The judge model family MUST
    differ from the agent's family — see judge.py for the rationale.
    """
    all_results = []
    Path(output_dir).mkdir(exist_ok=True)
    label = mock_policy or model

    # ---- Judge setup + calibration (BEFORE running tasks) ---------------
    judge = None
    calibration_report = None
    if use_judge:
        print(f"\n--- Initializing LLM judge ({judge_model}) ---")
        # For mock_policy runs we want strict_family_check off because the
        # 'mock' agent has no real family. For real runs it must be on.
        judge = LLMJudge(
            judge_model=judge_model,
            agent_model_family=agent_family,
            strict_family_check=(mock_policy is None),
        )
        print("Running calibration set...")
        calibration_report = judge.calibrate()
        print(f"  Calibration agreement: {calibration_report['agreement_rate']:.0%}")
        print(f"  Trust judge for production scores: "
              f"{calibration_report['trust_judge']}")
        if not calibration_report["trust_judge"]:
            print("  WARNING: Judge failed calibration. Quality scores will "
                  "be reported but should not be trusted.")

    for task in TASKS:
        print(f"\n=== {task.task_id} ({task.category}) ===")
        task_runs = []
        for i in range(runs_per_task):
            print(f"  Run {i+1}/{runs_per_task}...", end=" ", flush=True)
            trace = run_agent(task.user_message, model, task.max_steps,
                              mock_policy=mock_policy)
            scored = grade(trace, task, judge=judge)
            task_runs.append(scored)
            status = "PASS" if scored["outcome"]["passed"] else "FAIL"
            cost = scored["cost"]["cost_usd"]
            jq = scored["judge_quality"]
            jq_str = ""
            if jq.get("applicable"):
                if jq.get("agent_wins"): jq_str = " judge:WIN"
                elif jq.get("reference_wins"): jq_str = " judge:LOSS"
                elif jq.get("tie"): jq_str = " judge:TIE"
                elif jq.get("position_bias_detected"): jq_str = " judge:BIAS"
            print(f"{status}  (${cost:.4f}, {scored['cost']['steps']} steps){jq_str}")

        # Aggregate across runs for this task
        outcomes = [r["outcome"]["passed"] for r in task_runs]
        costs = [r["cost"]["cost_usd"] for r in task_runs]
        latencies = [r["cost"]["latency_seconds"] for r in task_runs]
        # Judge aggregation (only counts confident verdicts)
        judge_runs = [r["judge_quality"] for r in task_runs
                      if r["judge_quality"].get("applicable")
                      and r["judge_quality"].get("confident")]
        n_judge_wins = sum(1 for r in judge_runs if r.get("agent_wins"))
        n_judge_bias = sum(1 for r in task_runs
                           if r["judge_quality"].get("position_bias_detected"))

        task_report = {
            "task_id": task.task_id,
            "category": task.category,
            "n_runs": runs_per_task,
            "pass@1": round(pass_at_1(outcomes), 3),
            "pass^3": round(pass_power_k(outcomes, 3), 3),
            "pass^5": round(pass_power_k(outcomes, 5), 3),
            "outcomes": outcomes,
            "cost_mean_usd": round(mean(costs), 6),
            "cost_stdev_usd": round(stdev(costs) if len(costs) > 1 else 0, 6),
            "latency_mean_s": round(mean(latencies), 2),
            "trajectory_precision_mean": round(
                mean(r["trajectory"]["precision"] for r in task_runs), 3),
            "trajectory_recall_mean": round(
                mean(r["trajectory"]["recall"] for r in task_runs), 3),
            "judge_win_rate": (round(n_judge_wins / len(judge_runs), 3)
                                if judge_runs else None),
            "judge_position_bias_count": n_judge_bias,
            "individual_runs": task_runs,
        }
        all_results.append(task_report)

    # Suite-level aggregates
    judge_win_rates = [t["judge_win_rate"] for t in all_results
                       if t["judge_win_rate"] is not None]
    total_position_bias = sum(t["judge_position_bias_count"] for t in all_results)
    suite_report = {
        "model": label,
        "runs_per_task": runs_per_task,
        "n_tasks": len(TASKS),
        "task_results": all_results,
        "judge_calibration": calibration_report,
        "overall": {
            "pass@1": round(mean(t["pass@1"] for t in all_results), 3),
            "pass^5": round(mean(t["pass^5"] for t in all_results), 3),
            "total_cost_usd": round(sum(t["cost_mean_usd"] * t["n_runs"]
                                        for t in all_results), 4),
            "mean_latency_s": round(mean(t["latency_mean_s"] for t in all_results), 2),
            "judge_win_rate": (round(mean(judge_win_rates), 3)
                               if judge_win_rates else None),
            "position_bias_incidents": total_position_bias,
        },
    }

    # Persist
    safe = label.replace('/', '_')
    outpath = Path(output_dir) / f"report_{safe}.json"
    # Strip non-serializable env objects before saving
    cleaned = json.loads(json.dumps(suite_report, default=str))
    outpath.write_text(json.dumps(cleaned, indent=2))
    print(f"\nFull report written to: {outpath}")

    return suite_report


def print_scorecard(report: dict):
    """The senior-engineer-in-a-design-review summary."""
    print("\n" + "=" * 84)
    print(f"AGENT EVAL SCORECARD — {report['model']}")
    print("=" * 84)
    print(f"Tasks: {report['n_tasks']}, Runs/task: {report['runs_per_task']}")
    if report.get("judge_calibration"):
        c = report["judge_calibration"]
        print(f"Judge: {c['judge_model']} | calibration agreement: "
              f"{c['agreement_rate']:.0%} | trust: {c['trust_judge']}")
    print()

    has_judge = report["overall"].get("judge_win_rate") is not None
    if has_judge:
        print(f"{'Task':<30} {'Cat':<12} {'pass@1':>7} {'pass^5':>7} "
              f"{'$':>7}  {'lat':>6} {'judge':>6}")
    else:
        print(f"{'Task':<30} {'Cat':<12} {'pass@1':>7} {'pass^5':>7} "
              f"{'$':>7}  {'lat':>6}")
    print("-" * 84)
    for t in report["task_results"]:
        line = (f"{t['task_id']:<30} {t['category']:<12} "
                f"{t['pass@1']:>7.2f} {t['pass^5']:>7.2f} "
                f"{t['cost_mean_usd']:>7.4f}  {t['latency_mean_s']:>6.2f}")
        if has_judge:
            jw = t["judge_win_rate"]
            line += f" {jw:>6.2f}" if jw is not None else f" {'n/a':>6}"
        print(line)
    print("-" * 84)
    o = report["overall"]
    line = (f"{'OVERALL':<30} {'':<12} "
            f"{o['pass@1']:>7.2f} {o['pass^5']:>7.2f} "
            f"{o['total_cost_usd']:>7.4f}  {o['mean_latency_s']:>6.2f}")
    if has_judge:
        line += f" {o['judge_win_rate']:>6.2f}"
    print(line)
    print()
    print("KEY INSIGHTS:")
    print(f"  • Headline pass@1 — what vendors quote: {o['pass@1']:.0%}")
    print(f"  • Reliability pass^5 — what production needs: {o['pass^5']:.0%}")
    print(f"  • The gap is the reliability tax. Bigger gap = less productionizable.")
    if has_judge:
        print(f"  • Judge win-rate vs reference: {o['judge_win_rate']:.0%} "
              f"(higher = agent prose beat the reference)")
        print(f"  • Position-bias incidents flagged: {o['position_bias_incidents']} "
              f"(these were excluded from the win-rate — the right thing to do)")
    print()


if __name__ == "__main__":
    # Usage: python runner.py [model] [runs] [mock_policy] [--judge]
    # mock_policy ∈ {competent, lazy, gullible, demo} or omit for real API
    # --judge enables LLM-as-judge dimension
    args = sys.argv[1:]
    use_judge = "--judge" in args
    args = [a for a in args if a != "--judge"]

    model = args[0] if len(args) > 0 else "claude-sonnet-4-5"
    runs = int(args[1]) if len(args) > 1 else 3
    mock = args[2] if len(args) > 2 else None

    if mock == "demo":
        all_reports = {}
        for policy in ["competent", "lazy", "gullible"]:
            print(f"\n{'#' * 84}\n# AGENT POLICY: {policy.upper()}\n{'#' * 84}")
            r = run_suite(model=model, runs_per_task=runs, mock_policy=policy,
                          use_judge=use_judge)
            print_scorecard(r)
            all_reports[policy] = r
        # Cross-policy comparison
        print("\n" + "=" * 84)
        print("CROSS-AGENT COMPARISON — same task suite, different agents")
        print("=" * 84)
        if use_judge:
            print(f"{'Agent':<14} {'pass@1':>8} {'pass^5':>8} {'cost':>8} "
                  f"{'lat(s)':>8} {'judge':>8} {'bias':>6}")
        else:
            print(f"{'Agent':<14} {'pass@1':>8} {'pass^5':>8} {'cost':>8} {'lat(s)':>8}")
        print("-" * 84)
        for policy, r in all_reports.items():
            o = r["overall"]
            line = (f"{policy:<14} {o['pass@1']:>8.2f} {o['pass^5']:>8.2f} "
                    f"{o['total_cost_usd']:>8.4f} {o['mean_latency_s']:>8.2f}")
            if use_judge:
                jw = o.get("judge_win_rate")
                line += (f" {jw:>8.2f}" if jw is not None else f" {'n/a':>8}")
                line += f" {o.get('position_bias_incidents', 0):>6}"
            print(line)
    else:
        report = run_suite(model=model, runs_per_task=runs, mock_policy=mock,
                           use_judge=use_judge)
        print_scorecard(report)
