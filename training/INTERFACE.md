# Training Interface (Section C) - FROZEN

This document freezes the interface between the training workload
(section C) and everyone else: the agent core (A), the tool layer (B) and
the evaluation layer (D). Changing any string here (result keys, error
signatures, checkpoint file names) requires updating all consumers.

## Entry points

- `training/train.py --config <yaml> --backend {torch,smoke}`
  - `torch`: real path (HF DistilBERT/SST-2, native PyTorch loop, Ray
    Train). Heavy imports are lazy; the module imports cleanly without
    torch installed.
  - `smoke`: dependency-free miniature trainer (pure standard library)
    honoring the same knobs and output contract; used by tests, the
    local executor and the offline demo.
- `training/local_runner.py`
  - `LocalExecutor(backend="smoke", fault="none", repair_flags=None,
    scenario_id=None, extra_config=None)` implements the executor
    protocol from `docs/CONTRACTS.md`:
    `execute(spec, attempt_index, attempt_dir) -> dict` with keys
    `status`, `logs`, `metrics` (`{"eval_accuracy": float}` on success,
    else `None`), `checkpoint_dir`, `duration_seconds`. It writes the
    resolved `train_config.yaml` and `train.log` into `attempt_dir`.
  - `render_rayjob_yaml(spec, image="trainpilot-train:cpu-local",
    command=None, scenario=None) -> str`: Jinja2-rendered KubeRay
    RayJob manifest; parses with `yaml.safe_load`.

## The 9 knobs (config keys, from `trainpilot.schemas.TrainingSpec`)

| key | canonical default | notes |
|---|---|---|
| `backend` | `gloo` | distributed backend; `gloo` or `nccl` |
| `per_worker_batch_size` | 8 | powers of two, 2-32 |
| `gradient_accumulation_steps` | 2 | 1, 2, 4, 8 |
| `num_workers` | 2 | locked to 2 in graded scenarios |
| `resources_per_worker_memory` | `2Gi` | scenario cap is `max_memory_per_worker` |
| `epochs` | 1 | 1-2 |
| `max_steps` | 40 | optimizer-step cap, takes precedence |
| `learning_rate` | 2e-5 | valid range 5e-6 to 1e-4 |
| `train_subsample` / `val_subsample` | 1024 / 256 | drawn with `seed` |

Fixed (not knobs): `model_name` (`distilbert-base-uncased`),
`model_revision`, `dataset` (`sst2`), `max_length` (64; a scenario may
inject 512 as a fault and list restoring 64 as an acceptable repair),
`seed` (42).

Invariant: `effective_batch = per_worker_batch_size *
gradient_accumulation_steps * num_workers` must stay in **16-64**
(canonical 32). A run whose effective batch leaves the range fails with
signature `EFFECTIVE_BATCH_OUT_OF_RANGE`.

Extra config keys consumed by the smoke engine (not agent knobs):
`fault`, `dep_kind` (`torch_transformers` | `numpy2_abi`),
`repair_flags` (`sanitize_labels`, `rebuild_image_with_lockfile`),
`scenario_id`, `attempt`, `output_dir`.

## Fault knob semantics (smoke engine)

| `fault` | fails unless (correct repair) | error_signature |
|---|---|---|
| `none` | - | - |
| `nccl_on_cpu` | `backend == "gloo"` | `NCCL_ON_CPU` |
| `cuda_on_cpu` | `backend == "gloo"` | `CUDA_ON_CPU` |
| `oom` | `per_worker_batch_size <= 4` and `gradient_accumulation_steps >= 4` | `OOMKILLED_137` |
| `slow_steps` | `max_steps <= 40` and `max_length <= 64` and `train_subsample <= 2048` | `TIMEOUT_STEP_TOO_SLOW` |
| `dep_mismatch` | `repair_flags.rebuild_image_with_lockfile == true` | `DEP_TORCH_TRANSFORMERS_MISMATCH` or `DEP_NUMPY2_ABI` (by `dep_kind`) |
| `label_invalid` | `repair_flags.sanitize_labels == true` | `DATA_LABEL_INVALID` |
| `label_flipped` | not repairable by config: run succeeds, accuracy < 0.40 | `DATA_LABEL_FLIPPED_ACC_LOW` |
| `loss_nan` | `learning_rate <= 1e-3` | `TRAIN_LOSS_NAN` |

Dependency faults die at import time: no `TRAINPILOT_STEP` lines.
The OOM fault emits pod-style lines (`Reason: OOMKilled`, `Exit Code:
137`) rather than a Python traceback, by design.

## Output contract

- Progress lines: `TRAINPILOT_STEP {json}` with `global_step`, `loss`,
  `lr`, `elapsed_sec`.
- Exactly one final line: `TRAINPILOT_RESULT {json}` with keys
  `scenario_id`, `attempt`, `status` (`succeeded` | `failed`),
  `val_accuracy` (float 0-1 or null), `val_loss`, `global_step`,
  `checkpoint_path`, `backend`, `per_worker_batch_size`,
  `gradient_accumulation_steps`, and `error_signature` on failure.
- Exit code 0 on success, 1 on failure (the RESULT line is still
  printed).
- For a correct configuration the smoke engine reports a deterministic
  (seeded) pseudo-accuracy in 0.78-0.90.

## Checkpoint layout (`<output_dir>/checkpoint-final/`)

```text
config.json                  # architectures: [DistilBertForSequenceClassification], num_labels: 2
model.safetensors            # real weights (torch) / deterministic placeholder (smoke)
tokenizer_config.json
vocab.txt
special_tokens_map.json
training_args.json           # full resolved knob snapshot
train_config.resolved.yaml   # same snapshot as YAML
metrics.json                 # {"val_accuracy", "val_loss", "global_step"}
```

## Scenarios

`scenarios/<id>/scenario.yaml` carries: `id`, `family`, `injection`,
`log_signature`, `expected_log_signatures`, `ground_truth`
(`fault_family`, `root_cause_code`; a list for `demo-two-stage`),
`config_overrides` (starting spec values including the `fault` knob),
`correct_repair` (config overrides that neutralize the fault),
`wrong_repair` (a plausible but invalid repair, with the reason),
`accuracy_threshold`, `resource_limits`, `max_attempts`.
Instances: clean-canonical, clean-small-lr, gpu-nccl-backend,
gpu-hardcoded-cuda, mem-batch32-oom, mem-maxlength512-slow,
dep-torch-transformers, dep-numpy2-abi, data-label-invalid,
data-label-flipped, data-lr-too-high, demo-two-stage.
