# Lab: eval harness for the project agent (TrainPilot-Local)

**Team:** Chia-En Wu (NetID: chiaen28) · Andy Lee (NetID: andyrf)

## What is in this folder

| File | Requirement |
|---|---|
| `tasks.jsonl` | 50 tasks with known answers or rubrics: 25 easy, 15 hard, **5 edge**, **5 unanswerable**. Each line leads with the lecture format: `id · type · user · setup · pass_if · grader`. The harness fields (`prompt`, `grading_type`, `expected*`, `rubric`) follow and carry the same content. |
| `results_table.md` | Agent vs. non-agentic baseline, side by side: accuracy, avg turns, avg/p95 latency, avg tokens, **pass@3 / pass^3**, and the flipping tasks |
| `summary.json` | The same numbers, machine-readable |
| `failure_analysis.md` | One traced failure: which step went wrong, and why |

**Eval script:** https://github.com/Terrywu0208/trainpilot-local/tree/lab50-eval-real-llm/eval
- `eval/eval_harness.py`: entry point
- `eval/runner.py`: matrix and grading
- `eval/systems.py`: the two systems
- `eval/report.py`: the results table

## How to run (one script, both systems, 3 repeats)

```bash
pip install -e ".[eval]"
export OPENAI_API_KEY=...            # real model; without it a free offline stub is used
TRAINPILOT_LLM_PROVIDER=openai python -m eval.eval_harness --mode lab50 \
    --repeats 3 --workers 16 --max-cost-usd 4 --out runs/lab50
```

## Headline results (150 runs per system, gpt-5-mini / gpt-5-nano)

| System | Accuracy | pass@3 | pass^3 | Avg turns | Avg latency | p95 latency | Avg tokens |
|---|---|---|---|---|---|---|---|
| Non-agentic baseline (1 LLM call, no tools) | 0.587 | 0.660 | 0.500 | 1.0 | 35.9 s | 59.9 s | 3,088 |
| Planning agent (LangGraph + MCP-style tools) | 0.313 | 0.460 | 0.180 | 4.8 | 65.1 s | 142.0 s | 7,886 |

Flipping tasks (passed in some runs and failed in others):
- **Baseline:** L01, L03, L04, L06, L10, L19, L30, L50
- **Agent:** L06, L08, L09, L12, L13, L14, L15, L16, L18, L22, L25, L26, L28, L29

Total API cost for the 300 runs was $2.62.

## Grading method

Grading is code-based, with no LLM judge. A run passes only if all four parts pass:
Correct, Grounded, Complete, and In policy. Each task's grading type decides what
"Correct" means:

- **exact**: the expected value appears in the answer.
- **rubric**: every required keyword appears in the answer.
- **tool**: the expected tool appears in the run's tool trace.
- **unanswerable**: the system must decline (for example "I don't know") instead of
  inventing a number.

One rule applies only to systems that act on the cluster. If such a system's task
expects a succeeded job, it must also deliver a verified checkpoint; a report that
merely contains the right keywords is not enough. The baseline has no tools, so it is
graded on its answer alone. This rule only raises the bar for the agent.

## Reading the result honestly

The agent loses to the baseline on this set, and the failure analysis explains most
of the gap:
- **Unusable fixes:** for two of the five fault families, the agent's critic diagnoses
  correctly but returns a fix in a shape the repair router cannot use. The agent then
  gives up, scoring 0% on both families.
- **No refusal path:** the agent is a training pipeline. For "what accuracy did last
  night's run reach?" it launches a new job instead of saying it does not know
  (0/15 unanswerable runs).
- **Keyword grading favors the baseline:** the baseline's long advice text often
  contains every rubric keyword without ever having to work.

Fixing the critic's output contract is our first change before the Nov 24 benchmark.
