"""Tests for training/local_runner.py (LocalExecutor + RayJob render)."""
from pathlib import Path

import yaml

from trainpilot.schemas import TrainingSpec
from training.local_runner import LocalExecutor, render_rayjob_yaml


def test_render_rayjob_yaml_parses_and_carries_spec_values():
    spec = TrainingSpec()
    text = render_rayjob_yaml(spec, image="trainpilot-train:cpu-test")
    doc = yaml.safe_load(text)
    assert doc["kind"] == "RayJob"
    assert doc["apiVersion"] == "ray.io/v1"
    worker = doc["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]
    assert worker["replicas"] == spec.num_workers
    container = worker["template"]["spec"]["containers"][0]
    assert container["image"] == "trainpilot-train:cpu-test"
    assert container["resources"]["limits"]["memory"] == (
        spec.resources_per_worker_memory
    )
    env = {e["name"]: e["value"] for e in container["env"]}
    assert env["TRAINPILOT_BACKEND"] == spec.backend
    assert env["TRAINPILOT_PER_WORKER_BATCH_SIZE"] == str(
        spec.per_worker_batch_size
    )
    assert env["TRAINPILOT_LEARNING_RATE"] == str(spec.learning_rate)
    assert spec.spec_hash() in text


def test_local_executor_clean_end_to_end(tmp_path):
    executor = LocalExecutor(scenario_id="clean-canonical")
    out = executor.execute(TrainingSpec(), 1, tmp_path / "attempt-1")
    assert out["status"] == "succeeded"
    assert out["metrics"] is not None
    assert 0.78 <= out["metrics"]["eval_accuracy"] <= 0.90
    assert out["checkpoint_dir"] is not None
    assert Path(out["checkpoint_dir"]).is_dir()
    assert out["duration_seconds"] >= 0
    assert (tmp_path / "attempt-1" / "train.log").is_file()


def test_local_executor_fault_then_repair(tmp_path):
    faulty_spec = TrainingSpec(
        per_worker_batch_size=32, gradient_accumulation_steps=1
    )
    executor = LocalExecutor(fault="oom", scenario_id="mem-batch32-oom")
    failed = executor.execute(faulty_spec, 1, tmp_path / "attempt-1")
    assert failed["status"] == "failed"
    assert failed["metrics"] is None
    assert failed["checkpoint_dir"] is None
    assert "OOMKilled" in failed["logs"]
    assert "137" in failed["logs"]

    repaired_spec = TrainingSpec(
        per_worker_batch_size=4, gradient_accumulation_steps=4
    )
    repaired = executor.execute(repaired_spec, 2, tmp_path / "attempt-2")
    assert repaired["status"] == "succeeded"
    assert repaired["metrics"]["eval_accuracy"] >= 0.78
