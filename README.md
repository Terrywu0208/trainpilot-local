# TrainPilot-Local

## Local chat training (Windows + WSL)

The chat interface can run small tabular experiments locally without Ray. In
the project directory inside WSL, install the project and optional Kaggle
support, then start the app:

```sh
python -m pip install -e '.[webapp,classifier,openai,kaggle]'
python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8033
```

Use **＋ 新訓練** to create as many separate training sessions as needed and
the selector to return to an earlier session. Experiments and follow-up state
are scoped to their session; the provider call budget is shared. Conversation
history and the session index are stored in the current browser's local
storage. Clearing that storage or switching browser/origin loses the index;
experiment files remain on the local server. Open the report link from a
completed run for the fixed template, comparison chart, and evaluation notes.
Follow-up requests carry a bounded excerpt of the active session. When the user
explicitly authorizes a previously suggested feature exclusion, the app starts a
new local experiment with those columns removed and preserves the original run.
Version comparisons reuse the test split and require fresh data for final claims.

Set up `OPENAI_API_KEY` in the ignored `.env.local` and, for Kaggle access,
credentials in the ignored `.kaggle.local`; do not commit either file. Upload
a UTF-8 CSV (currently at most 2 MB) or ask the assistant for a named Kaggle
competition or dataset. Kaggle competition rules may require acceptance on
Kaggle itself. The app does not join or submit a competition.

Two independent specialist calls inform the LangGraph planner. The local
trainer compares supported classification, regression, or clustering
baselines; it can create a version with changed input features or linear-model
regularization. It reports validation comparisons and a held-out test, and
warns when versions reuse a test set. New data may be uploaded for another
experiment. Other data formats, model families, and arbitrary training tasks
need additional adapters; this is not a universal training implementation.
For supervised tabular tasks, when likely identifier or high-cardinality text
columns are present and other inputs remain, the trainer fits the same local
models with both original and reduced features on the same split. It selects
using validation scores and evaluates the selected pipeline on the test set;
the report shows feature removals and measured fit times. This is a heuristic,
not proof a field leaks data; explicit requests to keep fields are respected.
When the request explicitly asks for Hugging Face models, the graph performs a
bounded, read-only search of the public Hub and reports model metadata. Search
results are not trained or loaded: no general Hub training adapter is present.
OpenAI receives the request, column metadata, aggregate statistics and, for
Kaggle, public source descriptions; raw CSV rows remain local. Calls are paced
and capped per run, task, and day. The default is `gpt-5-nano` on the Responses
API, with a local cap of 200 requests per day and at least five seconds between
requests. This is not a spending cap. To use the retained Gemini integration,
set `TRAINPILOT_PROVIDER=gemini` and configure its key before starting the app.

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

## Two kinds of training

The agent runs both task types from plain-language requests:

- **Fine-tune** — DistilBERT on SST-2 through the local demo pipeline
  (the smoke backend simulates the run; the torch path is the real
  one, see Honest status).
- **Pretrain** — a tiny word-level language model trained **from
  scratch, for real**: pure-stdlib stochastic gradient descent on the
  bundled `tinytales` corpus (training/corpora/), a genuine
  safetensors checkpoint with the trained weights, and verification
  by held-out perplexity against the untrained baseline. One click in
  the web console ("Pretrain a tiny LM from scratch") or one sentence:
  *"Pretrain a tiny language model from scratch on the tinytales
  corpus with 2 CPU workers."*

Who each flow is designed for — and the four user journeys the
verification drives execute — is in [docs/PERSONAS.md](docs/PERSONAS.md).

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

# Web app UI test (drives a real browser through the whole user flow)
python scripts/ui_walkthrough.py --base http://127.0.0.1:8000 --shots /tmp/ui-shots
```

### Data tasks in the conversation

Install the local training and OpenAI dependencies with `pip install -e ".[dev,webapp,classifier,openai]"`.
In the existing Run Console, choose **＋**, upload a UTF-8 CSV, and describe
the goal. The model proposes a task and fields; LangGraph validates the proposal,
asks for clarification when necessary, then trains and verifies locally. The
current built-in CPU adapters support tabular classification,
numeric regression, and exploratory clustering. They create real scikit-learn
models, reload the saved model, and provide a report and prediction flow.
Text columns are encoded with character TF-IDF, numeric columns are scaled,
and low-cardinality string columns are one-hot encoded. Type inference is
heuristic, so users must check the inputs and exclude identifiers, sensitive
fields and future information. Search public datasets by typing
`搜尋資料集：sentiment`; only the search term is sent to Hugging Face.

OpenAI receives the user's messages, column names and aggregate statistics,
but not uploaded data rows. Provider requests are capped at 12 per planning
cycle, 80 per data task, and 200 per day (America/Los_Angeles). SDK retries
are disabled. Configure `OPENAI_API_KEY` in the environment or ignored
`.env.local`; the service never writes a key into its event logs. OpenAI
account billing and quota policies are separate from these request limits.
Unsupported tasks are explicitly reported and are not silently treated as another task.
This is an extensible starting point, not support for every model modality.
To add a trusted, locally installed adapter, register an object through the
`trainpilot.training_adapters` Python entry-point group. Its entry-point name
is a unique lowercase ID; its `describe()` returns `label`, `needs_target`,
and `model`; `analyze(root, run_id, target, features)`,
`train(root, run_id, target, features)`, `predict(root, run_id, values)`, and
`report(root, run_id)` follow the built-in API return shapes, and
`verify(root, run_id, result)` returns whether the artifact is verified. Only installed
packages are loaded: uploading data never installs or runs an adapter.

## Local KubeRay executor (experimental)

The web app defaults to the offline `LocalExecutor`. To opt into the local
Kind cluster on a host with kubectl configured for `kind-trainpilot`:

```bash
TRAINPILOT_EXECUTOR=kuberay python -m uvicorn webapp.main:app --port 8002
```

This mode creates a RayJob in `trainpilot-jobs`, waits for the trainer result,
and copies its checkpoint from the head pod into the run directory for the
agent verifier. It uses the locally built `trainpilot/training:2.41.0-cpu`
image and supports one or two CPU workers. Failures are reported rather than
silently replaced with the scripted demo. RayJobs and ConfigMaps are retained
for inspection, so the namespace quota must have room for new runs.

The smoke fine-tune remains simulated. The built-in tiny language model does
real single-process gradient descent inside the RayJob; it does not distribute
training across Ray workers. The existing local mode runs without Kubernetes.

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
- Pretraining is **not** simulated: the tiny LM is trained by real
  gradient descent in this sandbox (deterministic, seed 42), and its
  perplexity improvement, divergence repair, and honest corpus
  failures are covered by tests and persona drives. Its scale is
  deliberately tiny — it demonstrates the full agent loop on a model
  a laptop can genuinely train, not a production LM.
- Experiment *results* (the Dec 8 headline numbers) are produced by
  `eval/report.py` from run artifacts; this repository ships the
  machinery and the frozen benchmark, not pre-baked results.

## Local Hugging Face and MCP integrations

Install the optional `hf-tabular` and `mcp` extras for local Hugging Face
model execution and the read-only official Hugging Face MCP connection. For a
CPU-only machine, install CPU PyTorch from the official PyTorch index first.
The agent can search models, datasets and Spaces via MCP; a search result does
not imply its architecture is executable. Remote jobs and repository writes
are not enabled by this integration.

Installed model backends are discovered through the Python entry-point group
`trainpilot.model_backends`. Each trusted installed backend exposes `describe()`
(id, label, tasks, availability, model source, method, and whether weights are
fine-tuned) and `create(task)` (a scikit-learn-compatible local estimator).
Only installed backends can execute; discovering a model or MCP Space does not
install or run its code. The bundled PredictLM adapter uses a pinned checkpoint,
CPU, at most 256 context rows and 32 selected features, and three gradient steps.
These limits trade accuracy for bounded local resource use and are recorded in
the report. Other architectures and MCP integrations need their own adapters.

## License

MIT
