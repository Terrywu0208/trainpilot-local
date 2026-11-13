"""Tests for the from-scratch LM pretraining path in training/train.py."""
import json
import struct
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "training" / "train.py"
CORPUS = ROOT / "training" / "corpora" / "tinytales.txt"

CANONICAL_PRETRAIN = {
    "task_type": "pretrain",
    "dataset": "tinytales",
    "backend": "gloo",
    "per_worker_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_workers": 2,
    "epochs": 3,
    "max_steps": 40,
    "learning_rate": 0.1,
    "train_subsample": 100000,
    "val_subsample": 256,
    "seed": 42,
    "fault": "none",
    "scenario_id": "pretrain-canonical",
    "attempt": 1,
}


def run_trainer(tmp_path: Path, config: dict, engine: str = "smoke"):
    cfg = dict(config)
    cfg["output_dir"] = str(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(TRAIN_PY), "--config", str(cfg_path),
         "--backend", engine],
        capture_output=True, text=True, timeout=180,
    )
    results = [
        json.loads(line[len("TRAINPILOT_RESULT "):])
        for line in proc.stdout.splitlines()
        if line.startswith("TRAINPILOT_RESULT ")
    ]
    return proc, results


def test_corpus_meets_size_contract():
    lines = [l for l in CORPUS.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    assert len(lines) >= 200
    total_words = sum(len(line.split()) for line in lines)
    assert total_words >= 3800


def test_canonical_pretrain_succeeds_and_learns(tmp_path):
    proc, results = run_trainer(tmp_path, CANONICAL_PRETRAIN)
    assert proc.returncode == 0
    assert len(results) == 1, "exactly one TRAINPILOT_RESULT line"
    result = results[0]
    assert result["status"] == "succeeded"
    assert result["val_accuracy"] is None
    assert result["eval_perplexity"] < 0.7 * result["baseline_perplexity"]
    assert result["global_step"] == 12  # 3 epochs x 4 chunks
    assert result["corpus_tokens"] >= 1500
    samples = result["samples"]
    assert len(samples) == 3
    assert all(isinstance(s, str) and s.strip() for s in samples)
    sample_lines = [line for line in proc.stdout.splitlines()
                    if line.startswith("SAMPLE: ")]
    assert len(sample_lines) == 3

    ckpt = tmp_path / "checkpoint-final"
    for name in ("config.json", "model.safetensors", "lm_vocab.json",
                 "metrics.json", "training_args.json"):
        assert (ckpt / name).is_file(), f"missing checkpoint file {name}"

    hf_config = json.loads((ckpt / "config.json").read_text())
    assert hf_config["architectures"] == ["TinyWordLM"]
    assert hf_config["model_type"] == "tiny_word_lm"
    assert hf_config["vocab_size"] == result["vocab_size"]
    assert hf_config["embedding_dim"] == 32

    vocab = json.loads((ckpt / "lm_vocab.json").read_text())["vocab"]
    assert len(vocab) == result["vocab_size"]
    assert vocab[:3] == ["<unk>", "<s>", "</s>"]

    metrics = json.loads((ckpt / "metrics.json").read_text())
    assert metrics["eval_perplexity"] == result["eval_perplexity"]
    assert metrics["baseline_perplexity"] == result["baseline_perplexity"]
    assert metrics["global_step"] == result["global_step"]

    # The safetensors file must be a real, parseable safetensors blob:
    # header JSON + raw F32 payload whose lengths match the header.
    raw = (ckpt / "model.safetensors").read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + header_len])
    assert set(header) == {"embedding", "output_weight", "output_bias"}
    v = result["vocab_size"]
    assert header["embedding"]["shape"] == [v, 32]
    assert header["output_weight"]["shape"] == [v, 32]
    assert header["output_bias"]["shape"] == [v]
    body = raw[8 + header_len:]
    for meta in header.values():
        start, end = meta["data_offsets"]
        n_floats = 1
        for dim in meta["shape"]:
            n_floats *= dim
        assert end - start == n_floats * 4
        values = struct.unpack("<%df" % n_floats, body[start:end])
        assert all(isinstance(x, float) for x in values)
    max_end = max(meta["data_offsets"][1] for meta in header.values())
    assert max_end == len(body)


def test_unknown_corpus_fails_honestly(tmp_path):
    cfg = dict(CANONICAL_PRETRAIN, dataset="shakespeare")
    proc, results = run_trainer(tmp_path, cfg)
    assert proc.returncode == 1
    assert results[0]["status"] == "failed"
    assert results[0]["error_signature"] == "CORPUS_NOT_AVAILABLE"
    assert "corpus 'shakespeare' is not available" in proc.stdout


def test_tiny_token_budget_fails(tmp_path):
    cfg = dict(CANONICAL_PRETRAIN, train_subsample=100)
    proc, results = run_trainer(tmp_path, cfg)
    assert proc.returncode == 1
    assert results[0]["status"] == "failed"
    assert results[0]["error_signature"] == "CORPUS_TOO_SMALL"
    assert "corpus too small after split: 100 tokens" in proc.stdout


def test_high_learning_rate_diverges(tmp_path):
    cfg = dict(CANONICAL_PRETRAIN, learning_rate=5.0)
    proc, results = run_trainer(tmp_path, cfg)
    assert proc.returncode == 1
    assert results[0]["status"] == "failed"
    assert results[0]["error_signature"] == "LM_LOSS_NAN"
    assert "lm loss became nan" in proc.stdout
    assert "perplexity" in proc.stdout


def test_pretrain_is_deterministic(tmp_path):
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    dir_b = tmp_path / "b"
    dir_b.mkdir()
    _, first = run_trainer(dir_a, CANONICAL_PRETRAIN)
    _, second = run_trainer(dir_b, CANONICAL_PRETRAIN)
    assert first[0]["eval_perplexity"] == second[0]["eval_perplexity"]
    assert first[0]["samples"] == second[0]["samples"]


def test_torch_backend_routes_to_builtin_pretrain(tmp_path):
    proc, results = run_trainer(tmp_path, CANONICAL_PRETRAIN,
                                 engine="torch")
    assert proc.returncode == 0
    assert "built-in" in proc.stdout
    assert results[0]["status"] == "succeeded"
    assert results[0]["eval_perplexity"] < 0.7 * results[0][
        "baseline_perplexity"]


def test_local_executor_passes_through_perplexity_and_samples(tmp_path):
    from training.local_runner import LocalExecutor

    spec = {
        "model_name": "tiny-word-lm",
        "dataset": "tinytales",
        "task_type": "pretrain",
        "backend": "gloo",
        "per_worker_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "num_workers": 2,
        "resources_per_worker_memory": "2Gi",
        "epochs": 3,
        "max_steps": 40,
        "learning_rate": 0.1,
        "train_subsample": 100000,
        "val_subsample": 256,
        "max_length": 64,
        "seed": 42,
    }
    out = LocalExecutor().execute(spec, 1, tmp_path / "attempt-1")
    assert out["status"] == "succeeded"
    assert out["metrics"]["eval_perplexity"] > 0
    assert out["metrics"]["baseline_perplexity"] > (
        out["metrics"]["eval_perplexity"])
    assert len(out["samples"]) == 3
    assert all(s.strip() for s in out["samples"])
    assert out["checkpoint_dir"] is not None
    assert Path(out["checkpoint_dir"]).is_dir()


def test_lr5_single_epoch_reports_diverged_not_success(tmp_path):
    # The boundary case behind the LM_DIVERGED check: with only 1 epoch
    # (4 steps) the run ends before the NaN step guard can trip, but
    # the final model is far worse than the untrained baseline. The
    # trainer must report divergence, never a successful pretrain.
    cfg = dict(CANONICAL_PRETRAIN, epochs=1, learning_rate=5.0)
    proc, results = run_trainer(tmp_path, cfg)
    assert proc.returncode == 1
    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert results[0]["error_signature"] == "LM_DIVERGED"
    assert "perplexity" in proc.stdout.lower()
