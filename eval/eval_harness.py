"""Eval harness CLI (design D.2; evolves the Nov 17 lab skeleton).

One script runs every system on the same tasks with the same grader and
writes the results table with pass@k / pass^k and the flipping-task list,
plus a traced failure write-up.

Modes:
- lab50:   the Week 6 lab — 50 dev tasks, answer grading (standard rubric).
- cluster: the final benchmark matrix — hidden-grader grading over mock
           cluster runs (one canonical point per scenario by default).
- demo:    a single demo_agent pass over the lab tasks for smoke checks.

Examples:
    python -m eval.eval_harness --mode lab50
    python -m eval.eval_harness --mode demo --out runs/demo
    python -m eval.eval_harness --mode cluster --systems template_script,react_agent,planning_agent
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct invocation (`python -m eval.eval_harness`) from the repo
# root: the shared `trainpilot` package lives under src/ (pytest adds it
# via pyproject; a plain interpreter does not).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from . import report as report_mod
from .runner import load_tasks, run_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent

MODE_DEFAULTS = {
    "lab50": {
        "tasks": REPO_ROOT / "benchmark" / "tasks_lab50.jsonl",
        "systems": ["demo_agent", "non_agentic_baseline"],
        "repeats": 3,
        "out": REPO_ROOT / "runs" / "lab50",
    },
    "demo": {
        "tasks": REPO_ROOT / "benchmark" / "tasks_lab50.jsonl",
        "systems": ["demo_agent"],
        "repeats": 1,
        "out": REPO_ROOT / "runs" / "demo",
    },
    "cluster": {
        "tasks": REPO_ROOT / "benchmark" / "datapoints.jsonl",
        "systems": ["template_script", "oneshot_llm", "react_agent",
                    "planning_agent"],
        "repeats": 3,
        "out": REPO_ROOT / "runs" / "cluster",
    },
}


def _canonical_points(tasks):
    """Cluster mode runs one canonical datapoint per scenario (the matrix
    in design D.2.2 is per scenario, not per phrasing variant)."""
    seen: set[str] = set()
    picked = []
    for t in tasks:
        key = t.scenario_id or t.id
        if key not in seen:
            seen.add(key)
            picked.append(t)
    return picked


def write_failure_trace(records: list[dict], out_dir: Path) -> Path | None:
    """Trace one failure in prose (lab deliverable: 'show one failure and
    why it failed')."""
    failed = [r for r in records if not r["passed"] and r["system"] !=
              "non_agentic_baseline"]
    pool = failed or [r for r in records if not r["passed"]]
    if not pool:
        return None
    first = pool[0]
    run_dir = Path(first["run_dir"])
    trace_lines = []
    trace_file = run_dir / "tool_trace.jsonl"
    if trace_file.exists():
        for line in trace_file.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                trace_lines.append(
                    f"- seq {e['seq']}: {e['tool_name']} -> "
                    f"{e['result_status']} (approval: "
                    f"{e['approval'].get('reason_code') or 'n/a'})")
    text = (
        f"# Failure trace: {first['run_id']}\n\n"
        f"- System: {first['system']}\n"
        f"- Task: {first['task_id']} (scenario {first['scenario_id']}, "
        f"family {first['family']}, difficulty {first['difficulty']})\n"
        f"- Attempts: {first['attempts']}, tool calls: {first['tool_calls']}, "
        f"tokens: {first['tokens']}\n"
        f"- Claimed success: {first['claimed_success']}; grader pass: False\n\n"
        f"## Final answer\n\n{first['answer']}\n\n"
        f"## Tool trace\n\n" + "\n".join(trace_lines) + "\n\n"
        "## Why it failed\n\n"
        "The run did not satisfy the grader: either the repair never "
        "reached a verified state (the trace above shows where the loop "
        "stopped), or the final answer missed a required component of the "
        "standard rubric (Correct / Grounded / Complete / In policy). The "
        "trace is the evidence; no step is reconstructed from memory.\n"
    )
    path = out_dir / "failure_trace.md"
    path.write_text(text)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODE_DEFAULTS),
                        default="lab50")
    parser.add_argument("--tasks", default=None,
                        help="tasks JSONL (default per mode)")
    parser.add_argument("--systems", default=None,
                        help="comma-separated system names (default per mode)")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    defaults = MODE_DEFAULTS[args.mode]
    tasks_path = Path(args.tasks) if args.tasks else defaults["tasks"]
    systems = (args.systems.split(",") if args.systems
               else list(defaults["systems"]))
    repeats = args.repeats if args.repeats is not None else defaults["repeats"]
    out_dir = Path(args.out) if args.out else defaults["out"]
    mode = "cluster" if args.mode == "cluster" else "lab50"

    tasks = load_tasks(tasks_path)
    if args.mode == "cluster":
        tasks = _canonical_points(tasks)

    result = run_matrix(tasks, systems, repeats, out_dir, mode=mode,
                        seed=args.seed)
    summary = report_mod.write_report(result["records"], out_dir)
    write_failure_trace(result["records"], out_dir)

    print((out_dir / "results_table.md").read_text())
    print(f"Artifacts in {out_dir} (results.json, summary.json, "
          f"results_table.md, queue.jsonl, per-run directories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
