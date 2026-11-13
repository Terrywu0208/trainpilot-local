# Demo Video Script — TrainPilot-Local (Nov 24, 3 minutes)

Format: 3-minute pre-recorded video played in class + 2 minutes of live
Q&A where the instructor and classmates type questions into the live,
cloud-hosted demo. The video records one real, archived run (recorded
Nov 21–22); waiting time is cut and every sped-up segment is labelled
"sped up 8x" on screen. Total runtime must land in 2:55–3:00; if it runs
long, cut waiting and slide dwell — never cut the Gate 2 segment
(2:10–2:40).

Recording specs: 1920×1080, browser zoom 125%, notifications off, cursor
movements slow, 0.5 s of stillness between segments for clean cuts.

## Timeline

| Time | On screen | Narration points |
|---|---|---|
| 0:00–0:20 | Slide 1: title + one-line problem | Tutorial GPU training scripts fail immediately on a CPU-only Kind cluster; today a human watches logs, edits YAML, and resubmits by hand. TrainPilot-Local automates that loop — but irreversible actions still need a human. **Name the baseline in this segment:** "The same task with a single LLM call — our non-agentic baseline — cannot repair anything; we compare against it at the end." |
| 0:20–0:40 | Run Console: type the demo request, plan appears | Gate 1: the plan is shown to a human first. Nothing launches until "Approve Launch" is clicked. This is not a one-click black box. |
| 0:40–1:10 | Attempt 1 card: red log lines + diagnosis | Failure 1: the script defaults to a GPU communication backend; the cluster has only CPU workers. Point at the evidence lines: the agent diagnoses a backend mismatch and fixes it by switching to the CPU backend (`gloo`). |
| 1:10–1:40 | Timeline advances to attempt 2: OOM card | Failure 2: the run is killed for exceeding the memory limit. Fix: per-worker batch 32 → 4 with gradient accumulation 1 → 4, so the effective batch stays 32. Say this slowly — it is the technical core of the demo. |
| 1:40–2:10 | Attempt 3 succeeds; Verification Panel all green | Walk the four checks one by one: the checkpoint file really exists, the accuracy is a measured number, the config table proves the request was not quietly changed, and every fix has its own card. |
| 2:10–2:40 | Second case (archived segment / Replay) | Disk-full case: the agent's proposed fix is to delete old checkpoints. Gate 2 appears in red, listing exactly which files and how many MB. Narration says it plainly: "Here we press Reject — and the agent stops. It does not delete anything on its own." **This segment is the safety evidence for the whole talk. It is never cut.** |
| 2:40–3:00 | Slide: results + limitations | Closing numbers, agent vs baseline side by side (table below), one line on how the benchmark was built, then honest limitations: small model, local Kind cluster, four fault families. End on time. |

## Closing numbers (fill from `reports/final/summary.json`, never estimate)

| Metric | TrainPilot-Local (agent) | Non-agentic baseline |
|---|---|---|
| Task accuracy | __% (__ / 200) | __% |
| Avg turns | __ | __ |
| Avg latency (p95) | __ s | __ s |
| Avg tokens per task | __ | __ |
| Safety pass rate (60 prompts) | __% | __% |

Benchmark construction, one line for the closing slide: "Our benchmark
expands 20 scenarios into 200 team-validated data points — 100 easy,
60 hard, 20 edge cases, 20 unanswerable — with the held-out half frozen
by hash before any final run."

## Professor's presentation checklist (Week 5, slide 16) — sign-off

Every item must map to a video timestamp or a slide before recording is
frozen (Nov 23, 12:00). Both presenters sign each row.

| # | Checklist item | Where it appears | Wu | Lee |
|---|---|---|---|---|
| 1 | Why the problem matters: who has it, how often, what it costs today | 0:00–0:20 | ☐ | ☐ |
| 2 | Why it needs an agent (one call / fixed pipeline falls short) | 0:00–0:20 | ☐ | ☐ |
| 3 | A non-agentic baseline on the same tasks | Named 0:00–0:20; numbers 2:40–3:00 | ☐ | ☐ |
| 4 | Interactive: ask a question, get a response, live | Live instance during Q&A (typed questions) | ☐ | ☐ |
| 5 | Hosted on the cloud: a public URL anyone in the room can open | `docs/deployment.md` acceptance check passed | ☐ | ☐ |
| 6 | Remembers follow-ups (context carries over) | Q&A drill: a follow-up request typed live reuses the previous run's settings | ☐ | ☐ |
| 7 | Benchmarked on ≥200 tasks: accuracy %, avg turns, avg latency, vs baseline | 2:40–3:00 | ☐ | ☐ |
| 8 | Safety-tested on ≥50 prompts, with handled examples | 2:10–2:40 (Gate 2) + closing safety row | ☐ | ☐ |
| 9 | One design slide: the agent as a flowchart | `docs/design-diagram.md` rendered as a slide | ☐ | ☐ |

## Live Q&A preparation (2 minutes, typed into the live demo)

- Before class: the hosted instance is up at the public URL, one archived
  run is loaded for Replay as backup, and the operator (Wu) opens with:
  "Please type your question into the demo."
- Drill one follow-up out loud if the room is slow: run a request, then
  type "Same settings, but epochs = 3" and show the session summary being
  carried into the second run.
- Six predicted questions and one-line evidence answers live in the design
  document set (Section E.2.3); the short version: diagnoses cite log
  evidence lines (not a scripted sequence), success is only claimed after
  the verification panel passes, deletions stop at Gate 2 (see
  `audit_log.jsonl`), held-out numbers are on the dashboard as measured,
  Kind is the reproducibility choice, and cost/time figures come from the
  dashboard — never from memory.
