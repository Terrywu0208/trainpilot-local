"""Tests for training/train.py (smoke engine + output contract)."""
import ast
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "training" / "train.py"

CANONICAL = {
    "backend": "gloo",
    "per_worker_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_workers": 2,
    "resources_per_worker_memory": "2Gi",
    "epochs": 1,
    "max_steps": 40,
    "learning_rate": 2e-5,
    "train_subsample": 1024,
    "val_subsample": 256,
    "max_length": 64,
    "seed": 42,
    "fault": "none",
    "scenario_id": "clean-canonical",
    "attempt": 1,
}

CHECKPOINT_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "training_args.json",
    "train_config.resolved.yaml",
    "metrics.json",
]


def run_smoke(tmp_path: Path, config: dict):
    cfg = dict(config)
    cfg["output_dir"] = str(tmp_path)
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
    return proc, results


def test_canonical_clean_run_succeeds_with_full_checkpoint(tmp_path):
    proc, results = run_smoke(tmp_path, CANONICAL)
    assert proc.returncode == 0
    assert len(results) == 1, "exactly one TRAINPILOT_RESULT line"
    result = results[0]
    assert result["status"] == "succeeded"
    assert 0.78 <= result["val_accuracy"] <= 0.90
    assert result["global_step"] == 32
    steps = [line for line in proc.stdout.splitlines()
             if line.startswith("TRAINPILOT_STEP ")]
    assert steps, "TRAINPILOT_STEP lines must be emitted"
    ckpt = tmp_path / "checkpoint-final"
    for name in CHECKPOINT_FILES:
        assert (ckpt / name).is_file(), f"missing checkpoint file {name}"
    metrics = json.loads((ckpt / "metrics.json").read_text())
    assert metrics["val_accuracy"] == result["val_accuracy"]
    hf_config = json.loads((ckpt / "config.json").read_text())
    assert hf_config["architectures"] == [
        "DistilBertForSequenceClassification"
    ]
    assert hf_config["num_labels"] == 2


def test_smoke_accuracy_is_deterministic(tmp_path):
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    dir_b = tmp_path / "b"
    dir_b.mkdir()
    _, first = run_smoke(dir_a, CANONICAL)
    _, second = run_smoke(dir_b, CANONICAL)
    assert first[0]["val_accuracy"] == second[0]["val_accuracy"]


def test_effective_batch_invariant_enforced(tmp_path):
    bad = dict(CANONICAL, per_worker_batch_size=8,
               gradient_accumulation_steps=1, num_workers=1)
    proc, results = run_smoke(tmp_path, bad)
    assert proc.returncode == 1
    assert results[0]["status"] == "failed"
    assert results[0]["error_signature"] == "EFFECTIVE_BATCH_OUT_OF_RANGE"


def test_smoke_path_has_no_top_level_heavy_imports():
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
    heavy = {"torch", "transformers", "datasets", "ray", "numpy"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in heavy
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in heavy
    # The smoke engine must work in this venv, which has no torch.
    try:
        import torch  # noqa: F401

        torch_present = True
    except ImportError:
        torch_present = False
    if not torch_present:
        proc, results = run_smoke(
            Path(__import__("tempfile").mkdtemp()), CANONICAL
        )
        assert proc.returncode == 0 and results[0]["status"] == "succeeded"
