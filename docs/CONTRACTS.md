# Build Contracts (read this before writing any code)

This repository implements the TrainPilot-Local design in
`~/workspace/your_files/ee-p-500d-architecture/` (sections 00, A–F).
This file fixes the cross-module interfaces so the five slices can be
built in parallel. **Do not change the shared files** (`src/trainpilot/schemas.py`,
`src/trainpilot/llm.py`, `src/trainpilot/config/models.py`, `pyproject.toml`);
if an interface is missing, note it in your final report instead of editing them.

## Global rules

- **English only** everywhere: code, comments, docs, commit messages.
- **Zero cost**: never call a paid API. The default LLM provider is the
  deterministic stub in `trainpilot.llm`. No network calls in tests.
- **No new dependencies**: use only what `pyproject.toml` plus the standard
  library provide. The project venv is `~/workspace/trainpilot-local/.venv`
  (langgraph, fastapi, uvicorn, pydantic, pyyaml, jinja2, pytest, httpx).
- Run your tests with:
  `~/workspace/trainpilot-local/.venv/bin/python -m pytest tests/<your_prefix> -q`
- Every slice ships its own tests under `tests/` with its prefix and they
  must pass before you report back.

## Executor protocol (A defines, C and D consume)

An executor is any object with:

```python
def execute(self, spec: TrainingSpec, attempt_index: int,
            attempt_dir: Path) -> dict:
    ...
```

returning a dict with keys:

- `status`: `"succeeded"` or `"failed"`
- `logs`: full log text of the attempt (str)
- `metrics`: dict with at least `eval_accuracy` on success, else `None`
- `checkpoint_dir`: path string of the produced checkpoint, or `None`
- `duration_seconds`: float

## Public API that section A must expose (`trainpilot.graph`)

- `build_agent_graph(llm_client, executor, approval_handler=None, run_dir=None)`
  returning a compiled LangGraph graph with the 12 nodes from section A.
- `run_agent(request_text: str, *, executor, llm_client=None,
  approval_handler=None, run_dir=None, budget=None,
  session_context: str = "") -> dict` returning at least:
  `success` (bool), `claimed_success` (bool), `attempts` (int),
  `final_message` (str), `verification` (VerificationResult as dict),
  `diagnosis_records` (list of DiagnosisRecord dicts),
  `events` (list of event dicts with `type`, `timestamp`, `payload`),
  `failures` (list of failure-memory entries written this run).
- `approval_handler` is a callable `ProposedAction -> ApprovalDecision`.
  When it is `None`, only the first launch is auto-approved and every
  irreversible action is denied (safe default).
- `StubExecutor(outcomes: list[dict])`: scripted executor for tests/demo;
  each call pops the next outcome dict (same shape as the protocol return).
- Failure memory: append-only `failures.jsonl` in `run_dir`, entries with
  `fault_label`, `root_cause_code`, `evidence`, `fix_code`, `run_id`;
  at most 5 entries injected into the critic prompt, formatted with a
  `[DO NOT REPEAT]` marker for fixes that already failed.

## Public API that section B must expose (`trainpilot.tools`, `trainpilot.mcp`)

- `trainpilot.tools.registry.TOOL_REGISTRY`: dict of the 13 tool specs
  (7 read + 6 action from section B.4), each with `name`, `kind`
  (`read`/`action`), `description`, `dangerous_args` list.
- `trainpilot.tools.validators.validate_tool_call(tool_name, args,
  context) -> ToolResult`: allowlist/blacklist/approved-image checks;
  rejects shell-like input, paths outside the workspace, unapproved images.
- `trainpilot.tools.approval_tokens`: HMAC token issue/verify for
  approval-gated action tools (first launch, image build, purge).
- `trainpilot.tools.purge.purge_attempt_artifacts(...)`: the reserved
  deletion tool; only deletes paths matching an allowlist glob inside the
  run workspace.
- `trainpilot.mcp.read_server` / `trainpilot.mcp.action_server`: modules
  exposing `TOOLS` (the subset specs) and `call_tool(name, args, context)`
  functions implemented on a fake in-memory cluster client
  (`trainpilot.mcp.fake_cluster.FakeClusterClient`) so they are testable
  without Kubernetes. If the `mcp` package is importable, also expose a
  `build_fastmcp_server(...)`; it must not be required for tests.
- `infra/`: Kind cluster YAML (1 control-plane + 2 workers, digest-pinned
  node image), KubeRay operator Helm values (chart 1.3.0 / Ray 2.41.0),
  namespaces, ResourceQuota/LimitRange/NetworkPolicy/PodSecurity YAML,
  two ServiceAccount RBAC manifests (reader/writer, no pods/exec).

## Section C deliverables (`training/`, `scenarios/`)

- `training/train.py`: canonical training script with the five fixed
  function signatures and the 9-knob whitelist from section C; native
  PyTorch loop; emits single-line `TRAINPILOT_RESULT {json}` and
  `TRAINPILOT_STEP` lines; writes `checkpoint-final/` with the five files
  plus `metrics.json`. It must support `--config <yaml>` and a
  `--backend {torch,smoke}` flag: `smoke` is a dependency-free miniature
  trainer (pure Python) that produces the same checkpoint layout and
  result contract for offline tests and demos.
- `training/local_runner.py`: `LocalExecutor` implementing the executor
  protocol by running `train.py` as a subprocess (smoke backend by
  default), plus `render_rayjob_yaml(spec, scenario) -> str` (Jinja2).
- `training/INTERFACE.md`: the frozen interface (knobs, result contract,
  checkpoint layout).
- `scenarios/<id>/scenario.yaml` for the 11 designed instances
  (clean x2, GPU x2, memory x2, dependency x2, data/training x3) plus
  `scenarios/demo-two-stage/` — each with injection description, log
  signature, ground truth, correct repair, plausible wrong repair, and
  the fault knob overrides the smoke trainer must honor
  (`fault: none|nccl_on_cpu|cuda_on_cpu|oom|slow_steps|dep_mismatch|
  label_invalid|label_flipped|loss_nan`).
- The smoke trainer must actually fail with the scenario's log signature
  when a fault knob is set, and succeed (with plausible accuracy) when
  the spec's knobs carry the correct repair.

## Public API that section D must expose (`eval/`, `grader/`, `benchmark/`)

- `eval.eval_harness` CLI with `--mode {lab50, cluster, demo}`,
  `--tasks`, `--systems`, `--repeats`, `--out` (design D.1). Systems:
  `demo_agent`, `non_agentic_baseline`, `template_script`, `oneshot_llm`,
  `react_agent`, `planning_agent`. The planning adapter wraps
  `trainpilot.graph.run_agent` with a scenario executor; lab50 uses the
  mock gateway. Everything must run offline with the stub LLM.
- `eval.schema.TaskSpec` is `trainpilot.schemas.TaskSpec` (re-export);
  `validate_task(row) -> list[str]`.
- `eval.budget.BudgetTracker`: 41st tool call returns
  `BUDGET_EXCEEDED_TOOL_CALLS`; token and wall-clock caps; exceeded runs
  are still graded.
- `eval.scripted_user.ScriptedUser.decide(action, ctx)` implementing the
  policy table in D.2.3, backed by `configs/approval_policy.yaml`.
- `eval.llm_client`: re-export of `trainpilot.llm.LLMClient` plus
  `write_llm_ledger(client, path)`; `configs/prices.yaml` with the price
  table structure (values marked as placeholder, dated, with source note).
- `grader.hidden_grader.HiddenClusterGrader.grade(run_dir) -> GradeResult`
  with the three deterministic subchecks (checkpoint loads, validation
  accuracy vs `grader/thresholds.yaml`, final config honors request).
- `grader.diagnosis_scorer`, `grader.tool_arg_scorer`,
  `grader.trajectory_scorer`, `grader.safety_monitor` (the five counters
  from D.7), `grader/code_tables.yaml`, `grader/JUDGE.md` (judge prompt +
  pass definition + calibration protocol: 50 human labels, agreement
  >= 0.88, judge from a different model family than the agent).
- `eval.metrics.pass_at_k_stats(outcomes)` implementing the professor's
  combinatorial pass@k / pass^k definitions (section F, R8).
- `eval.report` producing Tables 1–6 data (markdown + `summary.json`)
  from run directories, including the parameter block (model,
  temperature, tool set, budget) and the noise note.
- `benchmark/generate_benchmark.py`: deterministic generator producing
  `benchmark/datapoints.jsonl` (200 points: 20 scenarios x 10, difficulty
  buckets 100 easy / 60 hard / 20 edge / 20 unanswerable, variant quotas
  from D.6), `benchmark/safety_suite.jsonl` (60 prompts, six categories
  x 10, plus >= 5 harmless controls), `benchmark/tasks_lab50.jsonl`
  (derived view: 50 tasks, >= 5 edge, >= 4 unanswerable, all four grading
  types), and `benchmark/manifest.json` with SHA-256 hashes. Generated
  files are committed; the generator must be re-runnable with identical
  output (fixed seed).

## Section E deliverables (`webapp/`, `demo/`, `docs/`)

- FastAPI app `webapp/main.py` with the four views in
  `webapp/static/index.html` (Run Console, Failure Memory, Benchmark
  Dashboard, Replay), SSE events, `RunManager` (one run at a time),
  two approval gates wired to the graph's `approval_handler`, session
  follow-up context (the previous run's summary is passed as
  `session_context` to `run_agent`), replay of `events.jsonl` at 8x with
  a REPLAY watermark, and run artifacts persisted under `runs/<run_id>/`
  per E.1.4.
- The webapp runs the agent with the C section's `LocalExecutor`
  (smoke backend) by default so the demo works offline; if the import is
  unavailable it falls back to A's `StubExecutor` demo script
  (fail, fail, succeed).
- `demo/demo_script.md`: the 3-minute video script updated with F/R11
  (baseline named in the first 20 seconds; closing numbers as
  agent-vs-baseline including safety pass rate; one line on how the
  benchmark was built).
- `docs/deployment.md`: how to put the webapp on a public URL
  (cloud VM full-stack route and tunnel route), with the acceptance
  check from E.7. `docs/design-diagram.md`: the design-slide flowchart
  in Mermaid with every box labeled with its model/library/service,
  including short-term memory, long-term memory and guardrails boxes.
- `docs/PROFESSOR_STANDARD.md`: a short English summary of the grading
  standard (section F of the design) and how this repo satisfies it,
  including the presentation checklist mapping.
