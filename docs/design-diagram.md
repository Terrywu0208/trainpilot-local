# Design Slide — How a Question Becomes an Answer

One flowchart, per the professor's design-slide specification (Week 5,
slide 19): user → cloud web app → agent loop → answer with sources, with
tools, short-term memory, long-term memory, and guardrails as side boxes.
**Every box is labelled with the model, library, or service behind it** —
an unlabelled box counts as a missing item.

```mermaid
flowchart LR
    U["User types a question<br/>Browser — any device on the public URL"]
    W["Web app — cloud-hosted<br/>FastAPI + uvicorn (Python), single-page HTML/JS + Chart.js<br/>SSE event stream, two approval gates"]
    A["Agent loop — up to 40 tool calls / 25 min budget<br/>LangGraph state machine (Python)<br/>Planner + coder: GPT-5 mini, pinned version, OpenAI API<br/>Intake + reporter: GPT-5 nano — offline runs use the built-in stub LLM, $0"]
    ANS["Answer with sources<br/>Final message + verification panel (checkpoint, accuracy, config table)<br/>Produced by the LangGraph final_reporter + verifier nodes"]
    T["Tools<br/>MCP read server + MCP action server (FastMCP, kubernetes Python client)<br/>kubectl get/describe, RayJob status + logs, workspace edit, image build"]
    STM["Short-term memory<br/>Session store in the webapp process (Python)<br/>One-paragraph summary of the previous run, trimmed to the latest run per session"]
    LTM["Long-term memory<br/>Failure memory — failures.jsonl, append-only files<br/>Each entry: fault, evidence, fix, source run id and date"]
    G["Guardrails<br/>Human approval gates in the webapp (Gate 1 launch, Gate 2 irreversible)<br/>Sandbox validators + Kubernetes RBAC / ResourceQuota / Pod Security (restricted)"]
    K["Execution<br/>KubeRay Operator (Helm chart 1.3.0, Ray 2.41.0) on Kind (kindest/node v1.31.2)<br/>Ray Train + PyTorch — DistilBERT on SST-2 (Hugging Face)"]

    U -->|"question / typed follow-up"| W
    W -->|"run request + session summary"| A
    A -->|"verified result"| ANS
    ANS -->|"SSE events"| W
    A <-->|"tool calls via MCP"| T
    T -->|"submit / watch RayJob"| K
    K -->|"logs + metrics"| T
    W <-->|"previous run summary"| STM
    A -->|"read before diagnosing, append after"| LTM
    G -.->|"approvals + allowlist checks"| A
    G -.->|"no shell, no unapproved images"| T
```

## How to read it (speaker notes for the slide)

- The main flow is one sentence: a typed question enters the cloud web
  app, the LangGraph agent loop plans, calls MCP tools against a KubeRay
  cluster, and the answer comes back with its verification evidence —
  never as an unverified claim.
- Short-term memory is what makes follow-ups work: "same settings, but
  epochs = 3" is understood because the session store carries the
  previous run's summary into the next run.
- Long-term memory is the failure memory: every diagnosis is appended
  with the run it came from and its date, and the critic checks it before
  diagnosing — but still needs fresh log evidence before reusing a fix.
- Guardrails are a side box on purpose: the approval gates and the
  sandbox validators constrain the loop from outside it, so a compromised
  or confused agent cannot approve its own irreversible actions.
