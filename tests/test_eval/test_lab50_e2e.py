"""End-to-end lab50 CLI run in demo systems mode (the Nov 17 lab shape):
results table + pass@3/pass^3 + flipping list + per-run artifacts, plus a
small cluster-mode matrix through the hidden grader."""
import json

from eval.eval_harness import main as harness_main
from eval.runner import load_tasks, run_matrix

ROOT_RUN_ARTIFACTS = [
    "run_meta.json", "experiment_config.yaml", "input_request.txt",
    "initial_config.yaml", "final_config.yaml", "tool_trace.jsonl",
    "llm_calls.jsonl", "token_ledger.json", "diagnosis.json",
    "agent_final_message.md", "checkpoint_meta.json", "safety_events.jsonl",
    "grader_result.json",
]


def test_lab50_cli_end_to_end(tmp_path):
    out = tmp_path / "lab50"
    rc = harness_main(["--mode", "lab50", "--out", str(out),
                       "--repeats", "3"])
    assert rc == 0
    for name in ("results_table.md", "results.json", "summary.json",
                 "queue.jsonl", "failure_trace.md"):
        assert (out / name).exists(), name

    summary = json.loads((out / "summary.json").read_text())
    demo = summary["systems"]["demo_agent"]
    assert demo["accuracy"] == 1.0
    assert demo["pass_at_k"]["pass@3"] == 1.0
    assert demo["pass_at_k"]["pass^3"] == 1.0
    baseline = summary["systems"]["non_agentic_baseline"]
    assert baseline["accuracy"] < demo["accuracy"]
    # The stub baseline's echo can leak a prompt keyword into one rubric
    # task (observed: 1 of 50); it must stay near zero and far below the
    # agent — that gap is the Week 5/6 baseline contrast.
    assert baseline["accuracy"] <= 0.05
    assert baseline["pass_at_k"]["pass^3"] < demo["pass_at_k"]["pass^3"]

    table = (out / "results_table.md").read_text()
    for marker in ("Table 1", "Table 2", "Table 3", "Table 4", "Table 5",
                   "Table 6", "Flipping tasks", "Price table version",
                   "Noise note"):
        assert marker in table, marker

    run_dirs = list((out / "demo_agent").rglob("grader_result.json"))
    assert len(run_dirs) == 50 * 3
    run_dir = run_dirs[0].parent
    for name in ROOT_RUN_ARTIFACTS:
        assert (run_dir / name).exists(), name


def test_cluster_mode_matrix_with_hidden_grader(tmp_path):
    from pathlib import Path
    tasks = load_tasks(Path(__file__).resolve().parents[2]
                       / "benchmark" / "datapoints.jsonl")
    picked = [t for t in tasks if t.scenario_id in
              ("clean-canonical", "gpu-nccl-backend")]
    canonical = {}
    for t in picked:
        canonical.setdefault(t.scenario_id, t)
    result = run_matrix(list(canonical.values()),
                        ["template_script", "react_agent"], 1,
                        tmp_path / "cluster", mode="cluster")
    by_key = {(r["system"], r["scenario_id"]): r for r in result["records"]}
    # Template script: passes clean, fails the fault (no repair, by design).
    assert by_key[("template_script", "clean-canonical")]["passed"] is True
    assert by_key[("template_script", "gpu-nccl-backend")]["passed"] is False
    # ReAct loop repairs the fault through the mock gateway.
    assert by_key[("react_agent", "clean-canonical")]["passed"] is True
    assert by_key[("react_agent", "gpu-nccl-backend")]["passed"] is True
    ckpt = (tmp_path / "cluster" / "react_agent" / "gpu-nccl-backend"
            / "repeat_1" / "checkpoint-final" / "metrics.json")
    assert ckpt.exists()
