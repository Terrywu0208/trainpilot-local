# TrainPilot-Local experiment report

## Parameters (logged with every score, Week 6 slide 43)
- Model: provider=openai; baseline=gpt-5-mini-2025-08-07/medium; agent nodes: intake_profiler=gpt-5-nano-2025-08-07/low, planner=gpt-5-mini-2025-08-07/medium, coder=gpt-5-mini-2025-08-07/medium, critic=gpt-5-mini-2025-08-07/high, final_reporter=gpt-5-nano-2025-08-07/low
- Temperature: provider default (GPT-5 reasoning models take no temperature; runs are not deterministic)
- Tool set: gateway v1: 13 tools (7 read, 6 mutating, no general shell)
- Budget per task: 1500 s wall clock / 40 tool calls / 120000 tokens per task
- Price table version: 2026-11-12-placeholder
- Safety policy: configs/approval_policy.yaml

Noise note: with 50 tasks, +/-6 points is about one standard error (binomial, near p = 0.8). Per-system Wilson 95% intervals are shown in Table 1. Any delta smaller than the noise band is reported as not significant.

## Table 1 — Overall results per system

| System | Runs | Accuracy | Wilson 95% | pass@3 | pass^3 | Avg turns (LLM calls) | Avg tool calls | Avg latency (s) | p95 latency (s) | Avg tokens | Cost/task (USD) | Cost/success (USD) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non_agentic_baseline | 150 | 0.587 | [0.507, 0.662] | 0.660 | 0.500 | 1.0 | 0.0 | 35.94 | 59.90 | 3088 | 0.0058 | 0.0099 |
| planning_agent | 150 | 0.313 | [0.244, 0.391] | 0.460 | 0.180 | 4.8 | 4.2 | 65.12 | 142.03 | 7886 | 0.0117 | 0.0372 |

## Table 2 — Accuracy by fault family

| System | Family | Runs | Accuracy |
|---|---|---|---|
| non_agentic_baseline | data_or_training_error | 30 | 0.667 |
| non_agentic_baseline | dependency_conflict | 30 | 0.600 |
| non_agentic_baseline | gpu_only_code | 30 | 0.667 |
| non_agentic_baseline | memory_limits | 30 | 0.733 |
| non_agentic_baseline | none_clean | 30 | 0.267 |
| planning_agent | data_or_training_error | 30 | 0.000 |
| planning_agent | dependency_conflict | 30 | 0.000 |
| planning_agent | gpu_only_code | 30 | 0.500 |
| planning_agent | memory_limits | 30 | 0.300 |
| planning_agent | none_clean | 30 | 0.767 |

## Table 3 — Repair within k attempts (fault runs only)

| System | Within 1 | Within 2 | Within 3 |
|---|---|---|---|
| non_agentic_baseline | 0.667 | 0.667 | 0.667 |
| planning_agent | 0.175 | 0.200 | 0.200 |

## Table 4 — Safety counters and safety pass rate

| System | Unsafe attempted | Unsafe executed | Planted followed | Unapproved irreversible | Unverified success | Safety pass rate |
|---|---|---|---|---|---|---|
| non_agentic_baseline | 0 | 0 | 0 | 0 | 0 | 1.000 |
| planning_agent | 0 | 0 | 0 | 0 | 5 | 1.000 |

Safety pass rate, non-agentic baseline (non_agentic_baseline): 1.000 — compare each agent row against this baseline (Week 5 slide 17: safety pass rate next to the baseline, same harness).

## Table 5 — Efficiency

| System | Avg tokens | p95 tokens | Avg latency (s) | p95 latency (s) | Total cost (USD) | Cost/task | Cost/successful task |
|---|---|---|---|---|---|---|---|
| non_agentic_baseline | 3088 | 4488 | 35.94 | 59.90 | 0.8718 | 0.0058 | 0.0099 |
| planning_agent | 7886 | 17456 | 65.12 | 142.03 | 1.7508 | 0.0117 | 0.0372 |

## Table 6 — Ablations

No ablation runs in this result set. Ablations (critic log format; replanning on/off) are run as separately named systems and appear here when present; the design labels them exploratory (design D.11).

## Proposal targets check

- Held-out success rate >= 80%
- Repair within 3 attempts >= 70%
- Unsafe executed = 0
- False success reports = 0

## Flipping tasks (pass sometimes, fail sometimes)

- non_agentic_baseline: L01, L03, L04, L06, L10, L19, L30, L50
- planning_agent: L06, L08, L09, L12, L13, L14, L15, L16, L18, L22, L25, L26, L28, L29
