#!/usr/bin/env python3
"""Canonical TrainPilot-Local training script.

Fine-tunes DistilBERT on SST-2 with a native PyTorch loop. Two engines:

- ``--backend torch``: the real path. Hugging Face model/dataset, Ray Train
  multi-worker loop. All heavy imports (torch, transformers, datasets, ray)
  happen lazily inside the torch functions so this module stays importable
  on machines without them.
- ``--backend smoke``: a dependency-free miniature trainer (pure standard
  library) that honors the same config knobs and the same output contract.
  It exists for offline tests and demos, and it simulates the scenario
  fault knob (config key ``fault``) with the exact log signatures the
  scenarios declare.

Output contract (both engines):
- ``TRAINPILOT_STEP {json}`` lines during training.
- Exactly one final ``TRAINPILOT_RESULT {json}`` line.
- On success, ``checkpoint-final/`` with config.json, model.safetensors,
  tokenizer_config.json, vocab.txt, special_tokens_map.json,
  training_args.json, train_config.resolved.yaml and metrics.json.

The five torch-path functions keep fixed signatures (design section C-1.1):
``get_datasets(config)``, ``build_model(config)``,
``train_loop_per_worker(config)``, ``build_trainer(config)``,
``main(config_path)``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import struct
import sys
import time
import zlib
from collections import Counter
from operator import mul
from pathlib import Path

# --------------------------------------------------------------------------
# Config loading (stdlib only; PyYAML used when available, with a small
# fallback parser for the flat configs this project writes)
# --------------------------------------------------------------------------

_DEFAULTS = {
    "model_name": "distilbert-base-uncased",
    "model_revision": "main",
    "dataset": "sst2",
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
    "dep_kind": "torch_transformers",
    "repair_flags": {},
    "scenario_id": "local",
    "attempt": 1,
    "output_dir": ".",
}


def _coerce(value: str):
    value = value.strip()
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if value in ("null", "None", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("'\"")


def _parse_simple_yaml(text: str) -> dict:
    """Parse the flat YAML subset this project emits (one nesting level)."""
    root: dict = {}
    current = root
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, val = raw.strip().partition(":")
        if indent == 0:
            current = root
            if val.strip() == "":
                root[key] = {}
                current = root[key]
            else:
                root[key] = _coerce(val)
        else:
            current[key] = _coerce(val) if val.strip() else {}
            if isinstance(current.get(key), dict):
                current = current[key]
    return root


def load_config(config_path: str) -> dict:
    text = Path(config_path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except ImportError:
        data = _parse_simple_yaml(text)
    config = dict(_DEFAULTS)
    config.update(data)
    if not isinstance(config.get("repair_flags"), dict):
        config["repair_flags"] = {}
    return config


def effective_batch(config: dict) -> int:
    return (
        int(config["per_worker_batch_size"])
        * int(config["gradient_accumulation_steps"])
        * int(config["num_workers"])
    )


# --------------------------------------------------------------------------
# Shared output helpers
# --------------------------------------------------------------------------


def _emit_result(payload: dict) -> None:
    print("TRAINPILOT_RESULT " + json.dumps(payload), flush=True)


def _emit_step(global_step: int, loss: float, lr: float, elapsed: float) -> None:
    print(
        "TRAINPILOT_STEP "
        + json.dumps(
            {
                "global_step": global_step,
                "loss": loss,
                "lr": lr,
                "elapsed_sec": round(elapsed, 2),
            }
        ),
        flush=True,
    )


def _write_checkpoint(config: dict, out_dir: Path, accuracy: float,
                      val_loss: float, global_step: int) -> Path:
    ckpt = out_dir / "checkpoint-final"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DistilBertForSequenceClassification"],
                "model_type": "distilbert",
                "num_labels": 2,
                "vocab_size": 30522,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Deterministic placeholder weights (smoke engine) - real safetensors
    # bytes are written by the torch engine via save_pretrained.
    rng = random.Random(int(config["seed"]))
    header = json.dumps({"metadata": {"format": "pt smoke-placeholder"}})
    body = bytes(rng.getrandbits(8) for _ in range(4096))
    (ckpt / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header.encode() + body
    )
    (ckpt / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "DistilBertTokenizer",
                "model_max_length": int(config["max_length"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [
        f"token_{i}" for i in range(200)
    ]
    (ckpt / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    (ckpt / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
                "pad_token": "[PAD]",
                "sep_token": "[SEP]",
                "unk_token": "[UNK]",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    resolved = {k: config[k] for k in sorted(config)}
    (ckpt / "training_args.json").write_text(
        json.dumps(resolved, indent=2, default=str), encoding="utf-8"
    )
    lines = []
    for key in sorted(config):
        val = config[key]
        if isinstance(val, dict):
            lines.append(f"{key}:")
            for sub_key in sorted(val):
                lines.append(f"  {sub_key}: {val[sub_key]}")
        else:
            lines.append(f"{key}: {val}")
    (ckpt / "train_config.resolved.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (ckpt / "metrics.json").write_text(
        json.dumps(
            {
                "val_accuracy": accuracy,
                "val_loss": val_loss,
                "global_step": global_step,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ckpt


def _failed_result(config: dict, signature: str, global_step: int = 0,
                   val_accuracy=None) -> dict:
    return {
        "scenario_id": config["scenario_id"],
        "attempt": int(config["attempt"]),
        "status": "failed",
        "val_accuracy": val_accuracy,
        "val_loss": None,
        "global_step": global_step,
        "checkpoint_path": None,
        "backend": config["backend"],
        "per_worker_batch_size": int(config["per_worker_batch_size"]),
        "gradient_accumulation_steps": int(
            config["gradient_accumulation_steps"]
        ),
        "error_signature": signature,
    }


# --------------------------------------------------------------------------
# Smoke engine (pure standard library)
# --------------------------------------------------------------------------


def _seeded_accuracy(config: dict) -> float:
    """Deterministic pseudo-accuracy in [0.78, 0.90] for a correct config."""
    key = repr(
        (
            int(config["seed"]),
            int(config["per_worker_batch_size"]),
            int(config["gradient_accumulation_steps"]),
            float(config["learning_rate"]),
            int(config["train_subsample"]),
        )
    ).encode()
    rng = random.Random(zlib.crc32(key))
    return round(0.78 + rng.random() * 0.12, 4)


def _smoke_fail(config: dict, signature: str, lines: list[str],
                global_step: int = 0, val_accuracy=None) -> int:
    for line in lines:
        print(line, flush=True)
    _emit_result(
        _failed_result(config, signature, global_step, val_accuracy)
    )
    return 1


def run_smoke(config: dict) -> int:
    started = time.monotonic()
    # Pretraining (task_type=pretrain) is a separate path: it trains a
    # tiny word-level language model from scratch on a bundled corpus
    # instead of fine-tuning on SST-2.
    if str(config.get("task_type", "")) == "pretrain":
        return _run_pretrain_smoke(config, started)
    out_dir = Path(str(config["output_dir"]))
    fault = str(config.get("fault", "none"))
    flags = config.get("repair_flags") or {}
    comm_backend = str(config["backend"])
    lr = float(config["learning_rate"])
    batch = int(config["per_worker_batch_size"])
    accum = int(config["gradient_accumulation_steps"])
    max_steps = int(config["max_steps"])

    # The local demo bundles exactly one dataset (SST-2). Any other name
    # is an honest failure, never a silent substitution.
    dataset = str(config.get("dataset", "sst2")).lower().replace("-", "")
    if dataset != "sst2":
        return _smoke_fail(
            config,
            "DATASET_NOT_AVAILABLE",
            [
                f"ValueError: dataset '{config.get('dataset')}' is not "
                "available in the local demo (only 'sst2' is bundled)"
            ],
        )

    # Effective-batch invariant (design C-1.2): repairs that leave the
    # invariant are invalid, so the trainer refuses to call them success.
    eff = effective_batch(config)
    if not 16 <= eff <= 64:
        return _smoke_fail(
            config,
            "EFFECTIVE_BATCH_OUT_OF_RANGE",
            [
                f"ValueError: effective_batch {eff} is outside the "
                "allowed range 16-64"
            ],
        )

    # --- Fault: GPU-only communication backend on a CPU-only cluster.
    if fault == "nccl_on_cpu" and comm_backend != "gloo":
        return _smoke_fail(
            config,
            "NCCL_ON_CPU",
            [
                "RuntimeError: Distributed backend 'nccl' is not "
                "supported on CPU-only machines",
                "RuntimeError: NCCL is not available",
            ],
        )

    # --- Fault: hardcoded CUDA device in the training script.
    if fault == "cuda_on_cpu" and comm_backend != "gloo":
        return _smoke_fail(
            config,
            "CUDA_ON_CPU",
            [
                "RuntimeError: Attempting to deserialize object on a CUDA "
                "device but torch.cuda.is_available() is False",
                "CUDA error: no CUDA-capable device is detected",
            ],
        )

    # --- Fault: dependency conflict, dies at import (step 0, no STEP lines).
    if fault == "dep_mismatch" and not flags.get(
        "rebuild_image_with_lockfile"
    ):
        if str(config.get("dep_kind")) == "numpy2_abi":
            return _smoke_fail(
                config,
                "DEP_NUMPY2_ABI",
                [
                    "A module that was compiled using NumPy 1.x cannot "
                    "be run in NumPy 2.x as it may crash",
                    "ImportError: numpy.core.multiarray failed to import",
                ],
            )
        return _smoke_fail(
            config,
            "DEP_TORCH_TRANSFORMERS_MISMATCH",
            [
                "ImportError: transformers requires torch>=2.0 but "
                "torch==1.13.1 is installed",
                "ModuleNotFoundError: No module named "
                "'transformers.models.distilbert.tokenization_distilbert_fast'",
            ],
        )

    # --- Fault: labels lost / invalid in the data pipeline.
    if fault == "label_invalid" and not flags.get("sanitize_labels"):
        return _smoke_fail(
            config,
            "DATA_LABEL_INVALID",
            [
                "ValueError: invalid label values in batch: labels "
                "contain -1 / None (label column dropped by set_format)",
                "IndexError: Target -1 is out of bounds",
            ],
            global_step=1,
        )

    # --- Fault: oversized sequence length + full dataset => timeout.
    if fault == "slow_steps" and (
        max_steps > 40
        or int(config["max_length"]) > 64
        or int(config["train_subsample"]) > 2048
    ):
        for step in (10, 20):
            _emit_step(step, 0.69, lr, step * 17.5)
        return _smoke_fail(
            config,
            "TIMEOUT_STEP_TOO_SLOW",
            [
                "TimeoutError: step time budget exceeded "
                "(max_length=512 padding and full dataset make each "
                "step >15s; harness killed the run at the time limit)",
            ],
            global_step=20,
        )

    # --- Fault: batch too large for the worker memory limit => OOMKill.
    if fault == "oom" and not (batch <= 4 and accum >= 4):
        for step in (1, 2):
            _emit_step(step, 0.69, lr, step * 1.8)
        return _smoke_fail(
            config,
            "OOMKILLED_137",
            [
                "The actor is dead because its worker process has died. "
                "Worker exit type: SYSTEM_ERROR Worker exit detail: "
                "Worker unexpectedly exits with a connection error code 2",
                "Reason: OOMKilled",
                "Exit Code: 137",
            ],
            global_step=2,
        )

    # --- Fault: learning rate explosion => loss diverges to NaN.
    if fault == "loss_nan" and lr > 1e-3:
        _emit_step(1, 0.6931, lr, 1.2)
        _emit_step(2, 12.4700, lr, 2.4)
        _emit_step(3, float("nan"), lr, 3.6)
        return _smoke_fail(
            config,
            "TRAIN_LOSS_NAN",
            [
                "Training diverged: loss became nan after step 3; "
                "validation accuracy collapsed to chance level",
            ],
            global_step=3,
            val_accuracy=0.5,
        )

    # --- Normal run (clean, or a fault neutralized by the correct repair).
    steps_per_epoch = max(
        1,
        int(config["train_subsample"])
        // (batch * int(config["num_workers"]) * accum),
    )
    total_steps = min(max_steps, steps_per_epoch * int(config["epochs"]))
    rng = random.Random(int(config["seed"]))
    loss = 0.6931
    for step in range(1, total_steps + 1):
        loss = max(0.28, loss * (0.985 + rng.random() * 0.01))
        if step % 10 == 0 or step == total_steps:
            _emit_step(step, round(loss, 4), lr, time.monotonic() - started)

    if fault == "label_flipped":
        # Not repairable by config: the run "succeeds" but the model
        # learned the inverted signal, so accuracy lands below 0.40.
        accuracy = round(0.34 + rng.random() * 0.05, 4)
        ckpt = _write_checkpoint(
            config, out_dir, accuracy, 0.91, total_steps
        )
        payload = {
            "scenario_id": config["scenario_id"],
            "attempt": int(config["attempt"]),
            "status": "succeeded",
            "val_accuracy": accuracy,
            "val_loss": 0.91,
            "global_step": total_steps,
            "checkpoint_path": str(ckpt),
            "backend": comm_backend,
            "per_worker_batch_size": batch,
            "gradient_accumulation_steps": accum,
            "error_signature": "DATA_LABEL_FLIPPED_ACC_LOW",
        }
        _emit_result(payload)
        return 0

    accuracy = _seeded_accuracy(config)
    val_loss = round(0.62 - (accuracy - 0.78) * 1.5, 4)
    ckpt = _write_checkpoint(config, out_dir, accuracy, val_loss, total_steps)
    payload = {
        "scenario_id": config["scenario_id"],
        "attempt": int(config["attempt"]),
        "status": "succeeded",
        "val_accuracy": accuracy,
        "val_loss": val_loss,
        "global_step": total_steps,
        "checkpoint_path": str(ckpt),
        "backend": comm_backend,
        "per_worker_batch_size": batch,
        "gradient_accumulation_steps": accum,
    }
    _emit_result(payload)
    return 0


# --------------------------------------------------------------------------
# Pretrain engine (pure standard library): a tiny word-level neural
# bigram language model trained from scratch on a bundled corpus.
# --------------------------------------------------------------------------

_PRETRAIN_CORPORA = {
    "tinytales": Path(__file__).resolve().parent / "corpora"
    / "tinytales.txt",
}
_PRETRAIN_TOKEN_RE = re.compile(r"[a-z']+")
_PRETRAIN_EMBED_DIM = 32
_PRETRAIN_MAX_VOCAB = 384
_PRETRAIN_MIN_TOKENS = 1500
_PRETRAIN_NUM_CHUNKS = 4
_PRETRAIN_MAX_STEPS = 60
_PRETRAIN_INIT_SCALE = 0.5
_LM_UNK = "<unk>"
_LM_BOS = "<s>"
_LM_EOS = "</s>"


def _lm_logits(W: list, b: list, ec: list) -> list:
    """Logits for one context embedding: W . ec + b."""
    return [
        sum(map(mul, wrow, ec)) + bt
        for wrow, bt in zip(W, b)
    ]


def _lm_mean_loss(E: list, W: list, b: list, ctx_counts: dict,
                  total: int) -> float:
    """Mean cross-entropy over all (context, target) positions."""
    nll = 0.0
    for c, targets in ctx_counts.items():
        z = _lm_logits(W, b, E[c])
        m = max(z)
        s = 0.0
        for x in z:
            s += math.exp(x - m)
        lse = m + math.log(s)
        for t, cnt in targets.items():
            nll += cnt * (lse - z[t])
    return nll / max(total, 1)


def _lm_chunk_step(E: list, W: list, b: list, chunk: list,
                   ctx_counts: dict, lr: float) -> None:
    """One full-batch SGD step over one chunk of distinct contexts.

    All positions sharing a context word also share logits, so targets
    are aggregated per context and each distinct context needs only one
    softmax. Gradients are those of the summed per-position loss,
    scaled by 1 / (number of distinct contexts in the chunk).
    """
    vocab_size = len(b)
    dim = _PRETRAIN_EMBED_DIM
    scale = 1.0 / max(len(chunk), 1)
    # Forward pass for each distinct context in the chunk: dl is the
    # gradient of the loss with respect to that context's logits.
    DL = []
    for c in chunk:
        targets = ctx_counts[c]
        n_c = sum(targets.values())
        z = _lm_logits(W, b, E[c])
        m = max(z)
        exps = [math.exp(x - m) for x in z]
        s = sum(exps)
        DL.append(
            [
                (exps[t] / s * n_c - targets.get(t, 0)) * scale
                for t in range(vocab_size)
            ]
        )
    # All gradients are formed from the pre-update parameters, using
    # transposed matrices so each gradient entry is one dot product
    # (summation order matches the naive loops exactly).
    DL_T = [
        [DL[j][t] for j in range(len(chunk))]
        for t in range(vocab_size)
    ]
    E_T = [[E[c][d] for c in chunk] for d in range(dim)]
    W_T = [[W[t][d] for t in range(vocab_size)] for d in range(dim)]
    grad_b = [sum(col) for col in DL_T]
    grad_W = [
        [sum(map(mul, DL_T[t], E_T[d])) for d in range(dim)]
        for t in range(vocab_size)
    ]
    for j, c in enumerate(chunk):
        ge = [sum(map(mul, DL[j], W_T[d])) for d in range(dim)]
        E[c] = [e - lr * g for e, g in zip(E[c], ge)]
    for t in range(vocab_size):
        W[t] = [w - lr * g for w, g in zip(W[t], grad_W[t])]
        b[t] -= lr * grad_b[t]


def _write_safetensors(path: Path, tensors: list) -> None:
    """Write real safetensors: 8-byte header length + JSON + F32 data.

    `tensors` is a list of (name, shape, flat row-major floats).
    """
    header: dict = {}
    data = bytearray()
    for name, shape, flat in tensors:
        start = len(data)
        data += struct.pack("<%df" % len(flat), *flat)
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [start, len(data)],
        }
    header_json = json.dumps(header).encode("utf-8")
    path.write_bytes(
        len(header_json).to_bytes(8, "little") + header_json + bytes(data)
    )


def _read_safetensors(path: Path):
    """Parse a safetensors file back into (header, {name: flat floats})."""
    raw = path.read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + header_len])
    body = raw[8 + header_len:]
    tensors = {}
    for name, meta in header.items():
        start, end = meta["data_offsets"]
        count = (end - start) // 4
        tensors[name] = list(
            struct.unpack("<%df" % count, body[start:end])
        )
    return header, tensors


def _lm_sample(E: list, W: list, b: list, vocab: list, bos: int,
               eos: int, rng: random.Random, max_words: int = 18,
               temperature: float = 0.8) -> str:
    words: list = []
    c = bos
    for _ in range(max_words):
        z = [x / temperature for x in _lm_logits(W, b, E[c])]
        m = max(z)
        exps = [math.exp(x - m) for x in z]
        s = sum(exps)
        r = rng.random() * s
        acc = 0.0
        nxt = len(z) - 1
        for i, e in enumerate(exps):
            acc += e
            if r <= acc:
                nxt = i
                break
        if nxt == eos:
            break
        token = vocab[nxt]
        if token not in (_LM_UNK, _LM_BOS, _LM_EOS):
            words.append(token)
        c = nxt
    return " ".join(words)


def _write_lm_checkpoint(config: dict, out_dir: Path, E: list, W: list,
                         b: list, vocab: list, metrics: dict) -> Path:
    ckpt = out_dir / "checkpoint-final"
    ckpt.mkdir(parents=True, exist_ok=True)
    vocab_size = len(vocab)
    (ckpt / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["TinyWordLM"],
                "model_type": "tiny_word_lm",
                "vocab_size": vocab_size,
                "embedding_dim": _PRETRAIN_EMBED_DIM,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_safetensors(
        ckpt / "model.safetensors",
        [
            (
                "embedding",
                [vocab_size, _PRETRAIN_EMBED_DIM],
                [x for row in E for x in row],
            ),
            (
                "output_weight",
                [vocab_size, _PRETRAIN_EMBED_DIM],
                [x for row in W for x in row],
            ),
            ("output_bias", [vocab_size], list(b)),
        ],
    )
    (ckpt / "lm_vocab.json").write_text(
        json.dumps({"vocab": vocab}, indent=2), encoding="utf-8"
    )
    (ckpt / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    resolved = {k: config[k] for k in sorted(config)}
    (ckpt / "training_args.json").write_text(
        json.dumps(resolved, indent=2, default=str), encoding="utf-8"
    )
    return ckpt


def _run_pretrain_smoke(config: dict, started: float) -> int:
    out_dir = Path(str(config["output_dir"]))
    seed = int(config["seed"])
    lr = float(config["learning_rate"])
    epochs = int(config["epochs"])

    corpus_name = str(config.get("dataset", "tinytales"))
    corpus_path = _PRETRAIN_CORPORA.get(
        corpus_name, _PRETRAIN_CORPORA.get(corpus_name.lower())
    )
    if corpus_path is None or not corpus_path.is_file():
        return _smoke_fail(
            config,
            "CORPUS_NOT_AVAILABLE",
            [
                f"ValueError: corpus '{corpus_name}' is not available "
                "in the local demo (only 'tinytales' is bundled)"
            ],
        )

    lines = [
        line.strip()
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenized = [
        _PRETRAIN_TOKEN_RE.findall(line.lower()) for line in lines
    ]

    counts: Counter = Counter()
    for words in tokenized:
        counts.update(words)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab_words = [
        w for w, c in ranked if c >= 2
    ][:_PRETRAIN_MAX_VOCAB]
    vocab = [_LM_UNK, _LM_BOS, _LM_EOS] + vocab_words
    word_to_id = {w: i for i, w in enumerate(vocab)}
    unk_id = word_to_id[_LM_UNK]
    bos_id = word_to_id[_LM_BOS]
    eos_id = word_to_id[_LM_EOS]
    vocab_size = len(vocab)

    sequences = [
        [bos_id]
        + [word_to_id.get(w, unk_id) for w in words]
        + [eos_id]
        for words in tokenized
    ]

    order = list(range(len(sequences)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(round(0.15 * len(sequences))))
    val_seqs = [sequences[i] for i in order[:n_val]]
    train_seqs = [sequences[i] for i in order[n_val:]]

    val_counts: dict = {}
    val_total = 0
    for seq in val_seqs:
        for i in range(len(seq) - 1):
            val_counts.setdefault(seq[i], Counter())[seq[i + 1]] += 1
            val_total += 1

    train_available = sum(len(seq) for seq in train_seqs)
    tokens_used = min(int(config["train_subsample"]), train_available)
    if tokens_used < _PRETRAIN_MIN_TOKENS:
        return _smoke_fail(
            config,
            "CORPUS_TOO_SMALL",
            [
                "ValueError: corpus too small after split: "
                f"{tokens_used} tokens below the 1500-token minimum "
                "for language-model pretraining"
            ],
        )

    train_counts: dict = {}
    train_total = 0
    consumed = 0
    for seq in train_seqs:
        if consumed >= tokens_used:
            break
        piece = seq[: tokens_used - consumed]
        consumed += len(piece)
        for i in range(len(piece) - 1):
            train_counts.setdefault(piece[i], Counter())[piece[i + 1]] += 1
            train_total += 1

    # Model init: seeded uniform, deterministic.
    init_rng = random.Random(seed)
    E = [
        [init_rng.uniform(-_PRETRAIN_INIT_SCALE, _PRETRAIN_INIT_SCALE)
         for _ in range(_PRETRAIN_EMBED_DIM)]
        for _ in range(vocab_size)
    ]
    W = [
        [init_rng.uniform(-_PRETRAIN_INIT_SCALE, _PRETRAIN_INIT_SCALE)
         for _ in range(_PRETRAIN_EMBED_DIM)]
        for _ in range(vocab_size)
    ]
    b = [0.0] * vocab_size

    baseline_loss = _lm_mean_loss(E, W, b, val_counts, val_total)
    baseline_ppl = math.exp(baseline_loss)

    contexts = sorted(train_counts)
    chunks = [
        contexts[i::_PRETRAIN_NUM_CHUNKS]
        for i in range(_PRETRAIN_NUM_CHUNKS)
    ]
    chunks = [chunk for chunk in chunks if chunk]
    total_steps = min(
        epochs * _PRETRAIN_NUM_CHUNKS, _PRETRAIN_MAX_STEPS
    )
    step = 0
    for step_index in range(total_steps):
        chunk = chunks[step_index % len(chunks)]
        _lm_chunk_step(E, W, b, chunk, train_counts, lr)
        step = step_index + 1
        loss = _lm_mean_loss(E, W, b, train_counts, train_total)
        if not math.isfinite(loss) or loss > 50:
            return _smoke_fail(
                config,
                "LM_LOSS_NAN",
                [
                    "Training diverged: lm loss became nan after step "
                    f"{step}; perplexity=inf"
                ],
                global_step=step,
            )
        _emit_step(step, round(loss, 4), lr, time.monotonic() - started)

    final_loss = _lm_mean_loss(E, W, b, val_counts, val_total)
    if not math.isfinite(final_loss) or final_loss > baseline_loss + 1.0:
        # The loss exploded without ever producing a NaN step (possible
        # when the run ends before the step guard trips): the trained
        # model is far worse than the untrained baseline, which is a
        # diverged run by any honest reading - never report it as a
        # successful pretrain.
        shown_ppl = (
            math.exp(min(final_loss, 700.0))
            if math.isfinite(final_loss) else float("inf")
        )
        return _smoke_fail(
            config,
            "LM_DIVERGED",
            [
                "Training diverged: validation perplexity "
                f"{shown_ppl:.2f} is far worse than the untrained "
                f"baseline {baseline_ppl:.2f} (perplexity exploded; "
                "the learning rate is too high for this tiny model)"
            ],
            global_step=step,
        )
    final_ppl = math.exp(final_loss)

    metrics = {
        "eval_perplexity": round(final_ppl, 2),
        "baseline_perplexity": round(baseline_ppl, 2),
        "val_loss": round(final_loss, 4),
        "global_step": step,
    }
    ckpt = _write_lm_checkpoint(config, out_dir, E, W, b, vocab, metrics)

    # Proof of loadability: rebuild the model from the checkpoint files
    # on disk and generate sample sentences with the reloaded weights.
    _, tensors = _read_safetensors(ckpt / "model.safetensors")
    disk_vocab = json.loads(
        (ckpt / "lm_vocab.json").read_text(encoding="utf-8")
    )["vocab"]
    dim = _PRETRAIN_EMBED_DIM
    E2 = [
        tensors["embedding"][i * dim:(i + 1) * dim]
        for i in range(len(disk_vocab))
    ]
    W2 = [
        tensors["output_weight"][i * dim:(i + 1) * dim]
        for i in range(len(disk_vocab))
    ]
    b2 = list(tensors["output_bias"])
    disk_bos = disk_vocab.index(_LM_BOS)
    disk_eos = disk_vocab.index(_LM_EOS)
    sample_rng = random.Random(seed)
    samples = []
    for _ in range(3):
        text = _lm_sample(
            E2, W2, b2, disk_vocab, disk_bos, disk_eos, sample_rng
        )
        samples.append(text)
        print(f"SAMPLE: {text}", flush=True)

    payload = {
        "scenario_id": config["scenario_id"],
        "attempt": int(config["attempt"]),
        "status": "succeeded",
        "val_accuracy": None,
        "eval_perplexity": round(final_ppl, 2),
        "baseline_perplexity": round(baseline_ppl, 2),
        "val_loss": round(final_loss, 4),
        "global_step": step,
        "checkpoint_path": str(ckpt),
        "backend": config["backend"],
        "per_worker_batch_size": int(config["per_worker_batch_size"]),
        "gradient_accumulation_steps": int(
            config["gradient_accumulation_steps"]
        ),
        "samples": samples,
        "vocab_size": vocab_size,
        "corpus_tokens": tokens_used,
    }
    _emit_result(payload)
    return 0


# --------------------------------------------------------------------------
# Torch engine (real path; heavy imports are lazy, inside the functions)
# --------------------------------------------------------------------------


def get_datasets(config: dict):
    """Load SST-2, subsample with seed 42, tokenize at max_length=64."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    seed = int(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model_name"]), revision=str(config["model_revision"])
    )

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=int(config["max_length"]),
        )

    train_ds = (
        load_dataset("glue", "sst2", split="train")
        .shuffle(seed=seed)
        .select(range(int(config["train_subsample"])))
    )
    val_ds = (
        load_dataset("glue", "sst2", split="validation")
        .shuffle(seed=seed)
        .select(range(int(config["val_subsample"])))
    )
    columns = ["input_ids", "attention_mask", "label"]
    train_ds = train_ds.map(tokenize, batched=True).set_format(
        type="torch", columns=columns
    )
    val_ds = val_ds.map(tokenize, batched=True).set_format(
        type="torch", columns=columns
    )
    return train_ds, val_ds


def build_model(config: dict):
    """Load the pinned DistilBERT sequence-classification model."""
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        str(config["model_name"]),
        revision=str(config["model_revision"]),
        num_labels=2,
    )


def train_loop_per_worker(config: dict):
    """Ray Train worker entry: native PyTorch loop, report checkpoint."""
    import random as _random

    import numpy as np
    import ray.train
    import ray.train.torch
    import torch
    from torch.utils.data import DataLoader, DistributedSampler

    seed = int(config["seed"])
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = ray.train.torch.get_device()
    train_ds, val_ds = get_datasets(config)
    model = ray.train.torch.prepare_model(build_model(config).to(device))

    sampler = DistributedSampler(train_ds, shuffle=True, seed=seed)
    loader = ray.train.torch.prepare_data_loader(
        DataLoader(
            train_ds,
            batch_size=int(config["per_worker_batch_size"]),
            sampler=sampler,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"])
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    accum = int(config["gradient_accumulation_steps"])
    max_steps = int(config["max_steps"])

    global_step = 0
    micro = 0
    started = time.monotonic()
    done = False
    for _epoch in range(int(config["epochs"])):
        if done:
            break
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels) / accum
            loss.backward()
            micro += 1
            if micro % accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    _emit_step(
                        global_step,
                        round(float(loss.item()) * accum, 4),
                        float(config["learning_rate"]),
                        time.monotonic() - started,
                    )
                if global_step >= max_steps:
                    done = True
                    break

    # Validation: full val subsample, argmax vs label.
    model.eval()
    val_loader = DataLoader(val_ds, batch_size=32)
    correct = 0
    total = 0
    val_loss_sum = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            outputs = model(**batch)
            val_loss_sum += float(loss_fn(outputs.logits, labels).item())
            correct += int(
                (outputs.logits.argmax(dim=-1) == labels).sum().item()
            )
            total += int(labels.shape[0])
    accuracy = correct / max(total, 1)
    val_loss = val_loss_sum / max(len(val_loader), 1)

    if ray.train.get_context().get_world_rank() == 0:
        ckpt_dir = Path(str(config["output_dir"])) / "checkpoint-final"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.save_pretrained(
            ckpt_dir, safe_serialization=True
        )
        (ckpt_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "val_accuracy": accuracy,
                    "val_loss": val_loss,
                    "global_step": global_step,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        with ray.train.report(
            metrics={"val_accuracy": accuracy, "val_loss": val_loss},
            checkpoint=ray.train.Checkpoint.from_directory(str(ckpt_dir)),
        ):
            pass


def build_trainer(config: dict):
    """Assemble the Ray Train TorchTrainer (2 CPU workers, gloo/nccl)."""
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchConfig, TorchTrainer

    return TorchTrainer(
        train_loop_per_worker,
        train_loop_config=config,
        scaling_config=ScalingConfig(
            num_workers=int(config["num_workers"]), use_gpu=False
        ),
        torch_config=TorchConfig(backend=str(config["backend"])),
        run_config=RunConfig(
            storage_path=os.path.abspath(str(config["output_dir"]))
        ),
    )


def main(config_path: str, engine: str = "torch") -> int:
    config = load_config(config_path)
    if engine == "smoke":
        return run_smoke(config)
    if str(config.get("task_type", "")) == "pretrain":
        # The local demo pretrains with its built-in standard-library
        # engine; there is no torch/Ray path for from-scratch LM
        # pretraining.
        print(
            "note: task_type=pretrain uses the built-in "
            "standard-library pretraining engine in the local demo",
            flush=True,
        )
        return run_smoke(config)

    seed = int(config["seed"])
    random.seed(seed)
    trainer = build_trainer(config)
    result = trainer.fit()
    metrics = result.metrics or {}
    accuracy = metrics.get("val_accuracy")
    ckpt_path = None
    if result.checkpoint is not None:
        ckpt_path = result.checkpoint.path
    payload = {
        "scenario_id": config["scenario_id"],
        "attempt": int(config["attempt"]),
        "status": "succeeded" if accuracy is not None else "failed",
        "val_accuracy": accuracy,
        "val_loss": metrics.get("val_loss"),
        "global_step": metrics.get("global_step"),
        "checkpoint_path": ckpt_path,
        "backend": config["backend"],
        "per_worker_batch_size": int(config["per_worker_batch_size"]),
        "gradient_accumulation_steps": int(
            config["gradient_accumulation_steps"]
        ),
    }
    _emit_result(payload)
    return 0 if accuracy is not None else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrainPilot-Local trainer")
    parser.add_argument(
        "--config", default="configs/train_config.default.yaml"
    )
    parser.add_argument(
        "--backend", choices=["torch", "smoke"], default="torch",
        dest="engine",
    )
    args = parser.parse_args()
    sys.exit(main(args.config, engine=args.engine))
