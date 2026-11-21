# Failure trace: planning_agent on L31 (dependency conflict)

**Task L31** (`dep-torch-transformers`, easy, tool-graded): fine-tune DistilBERT on
SST-2 with 2 CPU workers. The training image has a dependency conflict, and the job
fails at import time. To pass, the agent must read the job log, diagnose the
conflict, rebuild the image from the pinned lockfile, and deliver a verified
checkpoint. **Result: the agent failed in all 3 runs (pass@3 = 0).**

## Trace (matrix run `runs/lab50/planning_agent/dep-torch-transformers/L31/repeat_2`)

| Step | Node / tool | What happened |
|---|---|---|
| 1 | intake_profiler (gpt-5-nano) | Parsed the request correctly: distilbert, sst2, 2 workers, CPU only |
| 2 | planner, coder, preflight | Spec: backend=gloo, batch 8, accumulation 2, 2Gi per worker |
| 3 | approval gate → `edit_rayjob_yaml`, `apply_rayjob` | Launch approved by the scripted user and submitted |
| 4 | `get_rayjob_status`, `get_rayjob_logs`, `get_metrics` | Job FAILED. Log: `ImportError: transformers 4.57 is incompatible with the installed torch build (ABI mismatch)` |
| 5 | **critic (gpt-5-mini, high effort)** | **Diagnosed correctly, but the fix was in the wrong shape (see below)** |
| 6 | repair_router | No usable fix → `give_up_reason = no_safe_repair` after 1 of 3 attempts |
| 7 | final_reporter | "Run gave up … Last diagnosis: unknown." Recorded diagnosis: `UNKNOWN / fix NONE` |

The critic's raw reply, captured when L31 was re-run on the same code
(the matrix run stores only the post-validation diagnosis, which is `UNKNOWN`):

```json
{
  "symptom": "ImportError on startup: transformers 4.57 incompatible with the installed torch build (ABI mismatch); job failed.",
  "root_cause": "Dependency/version mismatch: the installed transformers package (4.57) requires a different PyTorch ABI ...",
  "evidence": ["ImportError: transformers 4.57 is incompatible with the installed torch build (ABI mismatch)"],
  "error_class": "DEPENDENCY",
  "proposed_fix": "align_transformers_and_torch_versions",
  "confidence": 0.95
}
```

## What went wrong, and why

The failing step is the hand-off from the **critic** to the **repair router**, not the
diagnosis itself. The critic read the log and identified the dependency conflict
correctly, with 0.95 confidence and a verbatim evidence quote. However, it returned
`proposed_fix` as a free-text string (`"align_transformers_and_torch_versions"`).
The `Diagnosis` schema requires a dict, and the only repair the system can execute
for this fault is the named action `{"rebuild_image_with_lockfile": true}`.
Pydantic validation rejected the reply. The critic node's exception handler then
silently replaced the whole diagnosis with a generic `UNKNOWN` record. As a result,
the repair router had nothing to apply and gave up with `no_safe_repair`, using
1 of its 3 attempts.

The root cause is a contract bug in our critic prompt (`prompts/critic_system.txt`):
- It says `proposed_fix` may be "a dict of TrainingSpec field changes, **or a single
  named repair action**", but never lists the action names.
- Its last line says "Never propose actions outside TrainingSpec fields", which forbids
  exactly the repair this fault needs.

Our offline stub LLM hard-codes the right keys, so every unit test passed. The real
model has to guess the keys, and it can't.

The same mechanism explains why the agent scored **0% on both the dependency and
data-error families**:
- 21 of 27 dependency runs (unanswerable tasks excluded) ended `UNKNOWN` or `DEPENDENCY_UNMAPPED`.
- 16 of 27 data runs were diagnosed correctly as `DATA_SCHEMA` but the fix could not be mapped.

The fix is to:
1. List the allowed repair actions in the critic prompt.
2. Remove the contradictory last line.
3. Map a valid `error_class` to its canonical repair instead of discarding the diagnosis
   when only `proposed_fix` is malformed.

A second, smaller failure mode is visible in the table: `RESOURCE_QUOTA_UNMAPPED`
diagnoses, which come from a preflight memory-limit conflict. We will address it
before the Nov 24 benchmark.
