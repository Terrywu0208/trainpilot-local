# TrainPilot-Local

**An agent that repairs KubeRay training runs — and proves it with numbers.**

TrainPilot-Local turns a plain-language request ("fine-tune DistilBERT on
SST-2 with two CPU workers") into a *verified* checkpoint on a KubeRay
cluster. It plans the run, generates the training script and RayJob YAML
from a single canonical spec, submits through a sandboxed tool layer,
diagnoses failures from condensed logs with a critic, repairs within a
hard budget, and only claims success after an independent verifier checks
the checkpoint, the accuracy and the final configuration.

Built as the final project for **EE P 500 D — Graduate Seminar: Agentic AI
Deep Dive** (University of Washington, Autumn 2026) by Chia-En Wu and
Andy Lee, implementing the proposal *"Agent for Repairing KubeRay
Training Runs"* (Canvas grade: 100/100).

## Research question

Does explicit planning plus a log-condensing critic beat a ReAct-style
agent with identical tools and budgets at run-time recovery of failed
training jobs? The evaluation layer (section D) exists to answer that
question honestly: same tasks, same grader, same script, four systems —
template script, one-shot LLM, ReAct, and the planning agent.

## How it is graded (the professor's standard)

This project is built to a grading standard specified by the course
instructor in the Week 3/5/6 lecture decks. The full mapping lives in
[docs/PROFESSOR_STANDARD.md](docs/PROFESSOR_STANDARD.md). Headlines:

- **Three axes** — Accuracy (task success over repeated runs; correct
  tool choice and arguments), Safety (unsafe actions blocked, injection
  tests, approval before irreversible steps), Efficiency (tokens, calls,
  latency, cost per task vs. a baseline).
- **Benchmark of 200+ tasks** with a fixed difficulty mix
  (100 easy / 60 hard / 20 edge / 20 unanswerable), every output
  team-validated, one script that runs everything and writes the table.
- **Safety suite of 60 prompts** — six attack categories × 10 — run in
  every evaluation, reported as a pass rate against the baseline.
- **Code-first grading**: deterministic checks where possible; an LLM
  judge only for open answers, from a different model family than the
  agent, calibrated on 50 human labels to agreement ≥ 0.88.
- **pass@k and pass^k reported as a pair**, next to a baseline, with the
  grading method and the noise (±3 points ≈ 1 SE on 200 tasks) stated.
- **A hosted live demo**: public URL, questions typed in live during
  Q&A, follow-up requests remembered across a session.

## Architecture

```
request ──> LangGraph agent (plan → code → preflight → approval gate
            → execute → monitor → critic → repair, max 3 attempts
            → verify → report)
                │  MCP tools only (7 read + 6 action), no shell
                ▼
            Kind + KubeRay Operator (RayJob on 2 CPU workers)
                │  every run leaves a full trace
                ▼
            evaluation layer (hidden grader, safety monitor,
            scripted user, budget enforcement, cost ledger)
                │
                ▼
            web app (run console, failure memory, benchmark
            dashboard, replay) — deployable to a public URL
```

| Directory | Contents |
|---|---|
| `src/trainpilot/graph/` | LangGraph state machine: 12 nodes, failure memory, repair router |
| `src/trainpilot/tools/`, `src/trainpilot/mcp/` | Tool registry, sandbox validators, approval tokens, MCP read/action servers |
| `training/` | Canonical `train.py` (DistilBERT + SST-2) + an offline smoke backend with the same output contract |
| `scenarios/` | 12 fault/clean scenarios with ground truth, signatures and correct repairs |
| `eval/`, `grader/` | Eval harness (lab50 + cluster modes), hidden grader, scorers, safety monitor |
| `benchmark/` | 200 data points, 60-prompt safety suite, 50-task lab set, freeze manifest |
| `webapp/` | FastAPI app: run console, approval gates, SSE, replay, sessions |
| `infra/` | Kind cluster, KubeRay operator values, RBAC, quotas, network policies |
| `docs/` | Professor-standard mapping, deployment guide, design diagram, build contracts, and the [operations & testing guide](docs/OPERATIONS.md) — every flow with real captured outputs |

## Zero-cost by design

The default LLM provider is a **deterministic offline stub**
(`trainpilot.llm`): the full pipeline — agent, evaluation, web app —
runs and is tested with **no API key and no spend**. A real model can be
plugged in by setting `TRAINPILOT_LLM_PROVIDER=openai` with your own
key; nothing in this repository ever requires it.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,webapp]"
.venv/bin/python -m pytest            # full test suite

# Nov 17 lab deliverable pipeline (50 tasks, agent vs baseline, pass@3/pass^3)
.venv/bin/python -m eval.eval_harness --mode lab50 \
    --tasks benchmark/tasks_lab50.jsonl \
    --systems demo_agent,non_agentic_baseline --repeats 3 --out runs/lab50/

# Web app (local)
.venv/bin/uvicorn webapp.main:app --port 8000
# then open http://localhost:8000 — see docs/deployment.md for a public URL
```

## Honest status

- The agent, tool layer, evaluation harness, benchmark data and web app
  are implemented and covered by the test suite in `tests/`.
- Training is exercised end-to-end through the smoke backend, which
  honors the same result contract and the scenario fault knobs. The
  PyTorch/Ray Train path in `training/train.py` is code-complete and
  imports and constructs correctly, but it has **not** been executed
  end-to-end by the authors yet: in our sandbox, Ray cannot complete
  its startup networking (a documented IPv4/gRPC overlay issue that
  also affects other Ray projects here; with an IPv6 shim head the
  driver connects and the trainer starts, but Ray Train's runtime-env
  agent then fails on the same address-family mismatch). On a normal
  Docker/Kubernetes host — including the Kind setup in `infra/` — this
  is the standard Ray Train flow; treat the torch path as
  code-reviewed, not run-verified, until the Dec experiments.
- The Kind/KubeRay manifests in `infra/` are provided and validated as
  YAML; executing them requires a Docker-capable machine.
- Experiment *results* (the Dec 8 headline numbers) are produced by
  `eval/report.py` from run artifacts; this repository ships the
  machinery and the frozen benchmark, not pre-baked results.

## License

MIT
