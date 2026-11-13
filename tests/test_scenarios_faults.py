"""Behavior tests: every scenario fault fails with its signature in the
smoke trainer, and its declared correct repair makes the run pass."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "training" / "train.py"
SCENARIOS_DIR = ROOT / "scenarios"

FAULT_SCENARIOS = [
    "gpu-nccl-backend",
    "gpu-hardcoded-cuda",
    "mem-batch32-oom",
    "mem-maxlength512-slow",
    "dep-torch-transformers",
    "dep-numpy2-abi",
    "data-label-invalid",
    "data-lr-too-high",
]
CLEAN_SCENARIOS = ["clean-canonical", "clean-small-lr"]


def load(scenario_id):
    return yaml.safe_load(
        (SCENARIOS_DIR / scenario_id / "scenario.yaml").read_text(
            encoding="utf-8"
        )
    )


def deep_merge(base, patch):
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def run_config(tmp_path, config):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = dict(config)
    cfg["output_dir"] = str(tmp_path)
    cfg.setdefault("scenario_id", "test")
    cfg.setdefault("attempt", 1)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(TRAIN_PY), "--config", str(cfg_path),
         "--backend", "smoke"],
        capture_output=True, text=True, timeout=60,
    )
    results = [
        json.loads(line[len("TRAINPILOT_RESULT "):])
        for line in proc.stdout.splitlines()
        if line.startswith("TRAINPILOT_RESULT ")
    ]
    assert len(results) == 1
    return proc, results[0]


@pytest.mark.parametrize("scenario_id", FAULT_SCENARIOS)
def test_fault_fails_with_signature(tmp_path, scenario_id):
    scenario = load(scenario_id)
    proc, result = run_config(tmp_path, scenario["config_overrides"])
    assert result["status"] == "failed", scenario_id
    assert proc.returncode == 1
    code = scenario["ground_truth"]["root_cause_code"]
    assert result["error_signature"] == code
    signatures = scenario["expected_log_signatures"]
    assert any(sig in proc.stdout for sig in signatures), (
        f"{scenario_id}: none of {signatures} found in logs"
    )


@pytest.mark.parametrize("scenario_id", FAULT_SCENARIOS)
def test_correct_repair_makes_it_pass(tmp_path, scenario_id):
    scenario = load(scenario_id)
    repaired = deep_merge(
        scenario["config_overrides"], scenario["correct_repair"]
    )
    proc, result = run_config(tmp_path, repaired)
    assert proc.returncode == 0, proc.stdout
    assert result["status"] == "succeeded"
    assert result["val_accuracy"] >= scenario["accuracy_threshold"]


@pytest.mark.parametrize("scenario_id", CLEAN_SCENARIOS)
def test_clean_scenarios_pass_unmodified(tmp_path, scenario_id):
    scenario = load(scenario_id)
    proc, result = run_config(tmp_path, scenario["config_overrides"])
    assert proc.returncode == 0
    assert result["status"] == "succeeded"
    assert result["val_accuracy"] >= scenario["accuracy_threshold"]


def test_label_flipped_is_a_successful_run_with_failed_accuracy(tmp_path):
    scenario = load("data-label-flipped")
    proc, result = run_config(tmp_path, scenario["config_overrides"])
    assert proc.returncode == 0
    assert result["status"] == "succeeded"
    assert result["val_accuracy"] < 0.40
    assert result["val_accuracy"] < scenario["accuracy_threshold"]
    assert result["error_signature"] == "DATA_LABEL_FLIPPED_ACC_LOW"


def test_dependency_faults_die_before_any_step(tmp_path):
    for scenario_id in ("dep-torch-transformers", "dep-numpy2-abi"):
        scenario = load(scenario_id)
        proc, result = run_config(
            tmp_path / scenario_id, scenario["config_overrides"]
        )
        assert result["status"] == "failed"
        assert "TRAINPILOT_STEP " not in proc.stdout


def test_demo_two_stage_sequence(tmp_path):
    demo = load("demo-two-stage")
    stage1, stage2 = demo["stages"]

    # Attempt 1: stage 1 fault fails with the NCCL signature.
    _, r1 = run_config(tmp_path / "a1", stage1["config_overrides"])
    assert r1["status"] == "failed"
    assert r1["error_signature"] == "NCCL_ON_CPU"

    # Attempt 2: backend repaired, but stage 2's OOM config arrives.
    stage2_cfg = deep_merge(
        stage2["config_overrides"], stage1["correct_repair"]
    )
    _, r2 = run_config(tmp_path / "a2", stage2_cfg)
    assert r2["status"] == "failed"
    assert r2["error_signature"] == "OOMKILLED_137"

    # Attempt 3: both repairs applied -> success at threshold.
    final_cfg = deep_merge(stage2_cfg, stage2["correct_repair"])
    proc, r3 = run_config(tmp_path / "a3", final_cfg)
    assert proc.returncode == 0
    assert r3["status"] == "succeeded"
    assert r3["val_accuracy"] >= demo["accuracy_threshold"]
