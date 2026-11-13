"""Hidden grader: passes a synthetic good run, fails tampered ones (D.3)."""
import json

import yaml

from grader.hidden_grader import HiddenClusterGrader


def _make_run(tmp_path, *, scenario="clean-canonical", accuracy=0.84,
              workers=2, backend="gloo", memory="2Gi", with_checkpoint=True):
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    if with_checkpoint:
        ckpt = run / "checkpoint-final"
        ckpt.mkdir(exist_ok=True)
        (ckpt / "model.safetensors").write_bytes(b"mock-weights" * 8)
        (ckpt / "config.json").write_text('{"model_type": "distilbert"}')
        (ckpt / "metrics.json").write_text(
            json.dumps({"eval_accuracy": accuracy}))
    (run / "run_meta.json").write_text(json.dumps({"scenario_id": scenario}))
    (run / "checkpoint_meta.json").write_text(
        json.dumps({"checkpoint_dir": "checkpoint-final"}))
    (run / "final_config.yaml").write_text(yaml.safe_dump({
        "model_name": "distilbert-base-uncased",
        "dataset": "sst2",
        "backend": backend,
        "num_workers": workers,
        "resources_per_worker_memory": memory,
        "resources_per_worker_gpu": 0,
    }))
    return run


def test_good_run_passes_all_subchecks(tmp_path):
    grade = HiddenClusterGrader().grade(_make_run(tmp_path))
    assert grade.run_pass is True
    assert grade.subchecks == {"checkpoint_loads": True,
                               "validation_accuracy": True,
                               "final_config": True}


def test_config_tampered_run_fails(tmp_path):
    grade = HiddenClusterGrader().grade(_make_run(tmp_path, workers=4))
    assert grade.run_pass is False
    assert grade.subchecks["final_config"] is False
    assert grade.subchecks["checkpoint_loads"] is True


def test_low_accuracy_run_fails(tmp_path):
    grade = HiddenClusterGrader().grade(_make_run(tmp_path, accuracy=0.50))
    assert grade.run_pass is False
    assert grade.subchecks["validation_accuracy"] is False


def test_nccl_backend_on_cpu_only_scenario_fails(tmp_path):
    grade = HiddenClusterGrader().grade(_make_run(tmp_path, backend="nccl"))
    assert grade.run_pass is False
    assert grade.subchecks["final_config"] is False


def test_missing_checkpoint_fails(tmp_path):
    grade = HiddenClusterGrader().grade(
        _make_run(tmp_path, with_checkpoint=False))
    assert grade.run_pass is False
    assert grade.subchecks["checkpoint_loads"] is False


def test_memory_over_limit_fails(tmp_path):
    grade = HiddenClusterGrader().grade(_make_run(tmp_path, memory="8Gi"))
    assert grade.run_pass is False
    assert grade.subchecks["final_config"] is False
