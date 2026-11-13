# The Professor's Grading Standard — and How This Repo Meets It

Source: the EE P 500 D lecture PDFs (Karthik Mohan), verified slide by
slide on 2026-11-12. Canvas carries no rubric for this course; the
standard below is the operative one. Full provenance and the R1–R11
requirement list live in the design set (Section F); this page is the
short English summary plus the mapping to this repository.

## The standard in one paragraph

Measure the agent on three axes — **Accuracy, Safety, Efficiency** —
against a **non-agentic baseline on the same tasks**. The benchmark has
at least **200 tasks** with known answers (mix: 100 easy / 60 hard /
20 edge / 20 unanswerable), graded **code-first** (exact match, tests,
state diffs), with an LLM judge only for open answers — a judge that
uses a rubric (never a 1–10 score), comes from a **different model
family** than the agent, quotes evidence before deciding, and is
calibrated against **50 human labels at agreement ≥ 0.88**. Safety is a
separate suite of **at least 50 attack prompts** (six categories × 10 =
60) run in **every** evaluation. Report **pass@k and pass^k as a pair**,
always next to the baseline, with the grading method stated and the
noise noted (±3 points on 200 tasks is about one standard error). Every
run leaves a **trace**. The results are presented in a 3-minute video
plus live Q&A on a **cloud-hosted demo at a public URL** that accepts
typed questions and remembers follow-ups.

## Requirement mapping (R1–R11)

| Req | Requirement (short form) | Where this repository satisfies it |
|---|---|---|
| R1 | Three axes: accuracy (incl. tool choice/arguments), safety (blocking, injection tests, approval before irreversible steps), efficiency vs baseline | Accuracy + tool-argument scoring in `grader/` (Section D); safety counters in `grader/safety_monitor.py`; efficiency table in `eval/report.py`; approval gates in `webapp/main.py` |
| R2 | Presentation checklist, 9 items: problem, why an agent, baseline, interactive, cloud-hosted public URL, follow-ups, ≥200-task benchmark, ≥50 safety prompts, one design slide | `demo/demo_script.md` (script + signed checklist mapping each item to a timestamp/slide); hosting in `docs/deployment.md`; design slide in `docs/design-diagram.md` |
| R3 | The five numbers: task accuracy (% of 200, grading method stated), avg turns, avg latency, avg tokens, safety pass rate — produced by one script, failures shown | `eval/report.py` generates Tables 1–6 and `reports/final/summary.json` automatically; the webapp dashboard only displays that file |
| R4 | Safety suite: six categories × 10 prompts, in every evaluation run, "safe" = refuse / ask confirmation / answer without harm, harmless controls checked for false refusals | `benchmark/safety_suite.jsonl` (Section D generator: 60 prompts + ≥5 harmless controls); suite runs with every harness invocation |
| R5 | Design slide: flowchart with user, cloud web app, agent loop, answer with sources, tools, short-term memory, long-term memory, guardrails — every box labelled with model/library/service | `docs/design-diagram.md` (Mermaid, all boxes labelled) |
| R6 | Grading methodology: six slices (outcome, trajectory, final state, tool use, message quality, efficiency); code first, rubric judge second, humans calibrate; judge independent, evidence-first JSON, agreement ≥ 0.88 on 50 labels; judge prompt + pass definition in the repo | Six slices scored across `grader/hidden_grader.py`, `grader/trajectory_scorer.py`, `grader/tool_arg_scorer.py`, `grader/diagnosis_scorer.py`; judge protocol in `grader/JUDGE.md` |
| R7 | Benchmark construction: real tasks with known answers, grading tied to the answer, failure-testing mix (100/60/20/20), tasks carry prompt / expected / rubric / expected_state, same tasks+grader+prompt across versions, full re-run after every change | `benchmark/generate_benchmark.py` + `benchmark/datapoints.jsonl`; task schema `src/trainpilot/schemas.py` (`TaskSpec`); freeze manifest with SHA-256 hashes in `benchmark/manifest.json` |
| R8 | Per-run metrics: pass@1 / pass@k / pass^k (combinatorial definition), tokens and latency as mean + p95, mean turns; parameters (model, temperature, tool set, budget) logged; results next to baseline with grading method and noise | `eval/metrics.py` (`pass_at_k_stats`); parameter block emitted by `eval/report.py`; noise note included in every report |
| R9 | Observability: trace every run; a failure becomes a new task (trace → first wrong step → label → add task → fix the cause → re-run everything) | Per-run artifacts (`events.jsonl`, `tool_trace.jsonl`, diagnosis records) persisted by the harness and the webapp under `runs/<run_id>/`; failure intake recorded, held-out set stays frozen |
| R10 | Lab rubric for open answers: Correct + Grounded + Complete + In policy, all four plus the task's own items = pass; agent and baseline in the same harness, grader, tasks, script, and results table | Lab mode of `eval/eval_harness.py` (`--mode lab50`) shares the schema, graders, and metrics with the final benchmark by construction |
| R11 | Presentation/report numbers as agent-vs-baseline pairs, including safety pass rate; benchmark construction explained in the video | Closing table in `demo/demo_script.md`; dashboard KPI cards in `webapp/static/index.html` |

## Honesty notes

- Numbers are produced by running the Section D harness; this document
  maps requirements to mechanisms and does not itself claim any result.
- Two inconsistencies in the source material are recorded, not silently
  resolved: Week 6 slide 12 mentions "accuracy on 100 data points" where
  every other source says ≥200 (we follow ≥200), and the lab slide's
  "50 tasks" sits next to "these 20 are the first tenth of the 200"
  (we build 50). Canvas lists different final deadlines (report Dec 6,
  presentation Dec 11) than the professor's slides (Dec 8); the team
  plans to the slides and has flagged the discrepancy for office hours.
