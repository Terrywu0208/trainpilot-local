"""The 12 LangGraph nodes of the TrainPilot-Local agent (design A.3).

LLM nodes (intake_profiler, planner, critic, final_reporter) go through
the single shared LLMClient with the routing table from
``trainpilot.config.models``; everything else is deterministic code.
The spec in the state is the single source of truth: the coder only
regenerates files from it, and success can only be declared by the
verifier node.
"""
from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path
from typing import Any

import yaml

from ..config.models import NODE_MODEL_ROUTING
from ..prompts.loader import load_prompt
from ..prompts.untrusted import sanitize_untrusted_text, wrap_untrusted
from ..schemas import (
    ApprovalDecision,
    BudgetState,
    Diagnosis,
    DiagnosisRecord,
    ErrorClass,
    ParsedRequest,
    ProposedAction,
    TrainingSpec,
    VerificationResult,
)
from . import budget as budget_mod
from . import memory as memory_mod
from . import rendering
from .context import AgentContext
from .state import (
    MAX_ATTEMPTS,
    MAX_REPAIR_CYCLES,
    MachineProfile,
    PreflightResult,
)

_ACTION_FIX_KEYS = {"rebuild_image_with_lockfile", "sanitize_labels"}
_BANNED_PATTERNS = ("os.system", "shell=True", "eval(", "rm -rf", "subprocess")

_STRONG_PRETRAIN_MARKERS = ("pretrain", "pre-train", "tiny lm", "train an lm")
_LEARNING_RATE_RE = re.compile(r"learning rate (?:of )?([\d.]+)")


def _pretrain_intent(lowered: str) -> bool:
    """True when the text asks for from-scratch language-model training.

    Deliberately stricter than a bare "from scratch" substring: a
    fine-tune request like "reload the dataset from scratch" (lab50
    task L16) must NOT flip to pretraining. Pretrain intent needs an
    explicit pretrain marker, or a language model being trained.
    """
    if any(marker in lowered for marker in _STRONG_PRETRAIN_MARKERS):
        return True
    if "language model" in lowered and (
            "train" in lowered or "scratch" in lowered):
        return True
    if "from scratch" in lowered and "train" in lowered \
            and "model" in lowered:
        return True
    return False


def _named_corpus(lowered: str) -> str:
    """The corpus named in a pretraining request, normalized to a corpus
    id (lowercase, hyphens stripped). Defaults to the bundled corpus."""
    match = re.search(r"on the ([\w\-]+) corpus", lowered) or re.search(
        r"corpus (?:called |named )?([\w\-]+)", lowered)
    if match:
        return match.group(1).replace("-", "")
    return "tinytales"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _brace_safe(text: str) -> str:
    return text.replace("{", "(").replace("}", ")")


def _chat(ctx: AgentContext, state: dict, node_name: str, system: str,
          user: str) -> tuple[str, dict]:
    """One routed LLM call; charges tokens and wall clock to the budget."""
    model, effort = NODE_MODEL_ROUTING[node_name]
    response = ctx.llm.chat(system=system, user=user, model=model, effort=effort)
    working = budget_mod.budget_from_state(state.get("budget"))
    working = budget_mod.charge_tokens(
        working, response.prompt_tokens, response.completion_tokens)
    working = budget_mod.refresh_wall_clock(working, ctx.started_monotonic)
    return response.text, budget_mod.dump(working)


def _memory_gb(value: str) -> float:
    match = re.match(r"([\d.]+)\s*Gi", str(value))
    return float(match.group(1)) if match else 0.0


def _spec_patch_is_valid(spec: TrainingSpec, patch: dict) -> bool:
    try:
        TrainingSpec(**{**spec.model_dump(), **patch})
    except Exception:
        return False
    return True


def _validate_fix(diagnosis: Diagnosis, spec: TrainingSpec | None) -> dict:
    """Drop proposed-fix keys that are not TrainingSpec fields or known
    repair actions, and drop patches whose values fail validation."""
    fix = dict(diagnosis.proposed_fix or {})
    allowed = set(TrainingSpec.model_fields) | _ACTION_FIX_KEYS
    fix = {key: value for key, value in fix.items() if key in allowed}
    spec_part = {k: v for k, v in fix.items() if k in TrainingSpec.model_fields}
    if spec is not None and spec_part and not _spec_patch_is_valid(spec, spec_part):
        fix = {k: v for k, v in fix.items() if k not in TrainingSpec.model_fields}
    return fix


# ----------------------------------------------------------------------
# 1. intake_profiler
# ----------------------------------------------------------------------

def _extract_hints(text: str) -> dict[str, Any]:
    lowered = text.lower()
    hints: dict[str, Any] = {}
    if "distilbert" in lowered:
        hints["model_name"] = "distilbert-base-uncased"
    if "sst-2" in lowered or "sst2" in lowered:
        hints["dataset"] = "sst2"
    match = re.search(r"(\d+)\s*epochs?", lowered)
    if match:
        hints["epochs"] = int(match.group(1))
    match = re.search(r"(\d+)\s*(?:cpu\s+)?workers?", lowered)
    if match:
        hints["num_workers"] = int(match.group(1))
    elif "two" in lowered and "worker" in lowered:
        hints["num_workers"] = 2
    if _pretrain_intent(lowered):
        hints["task_type"] = "pretrain"
        hints["model_name"] = "tiny-lm"
        hints["dataset"] = _named_corpus(lowered)
        if "epochs" not in hints:
            # A from-scratch LM needs a few passes to visibly learn; the
            # fine-tune default of 1 epoch barely moves perplexity.
            hints["epochs"] = 3
    match = _LEARNING_RATE_RE.search(lowered)
    if match:
        hints["learning_rate"] = float(match.group(1))
    return hints


def _explicit_overrides(text: str) -> dict[str, Any]:
    """Statements the user made explicitly in the request text.

    These override the LLM parse: the demo must never silently turn a
    GPU request into a CPU run, a named dataset into SST-2, or a stated
    worker/epoch count into the defaults. The verifier compares the
    final spec against this parse, so the parse has to reflect what the
    user actually said.
    """
    lowered = text.lower()
    overrides: dict[str, Any] = {}
    if _pretrain_intent(lowered):
        # A pretraining request names a corpus, not a fine-tune dataset:
        # keep the corpus id exactly as named (never substitute the
        # bundled one silently) and pin the tiny-LM model.
        overrides["task_type"] = "pretrain"
        overrides["model_name"] = "tiny-lm"
        overrides["dataset"] = _named_corpus(lowered)
    if re.search(r"\bgpu\b", lowered):
        overrides["cpu_only"] = False
    match = re.search(r"(\d+)\s*(?:cpu\s+|gpu\s+)?workers?", lowered)
    if match:
        overrides["num_workers"] = int(match.group(1))
    match = re.search(r"(\d+)\s*epochs?", lowered)
    if match:
        overrides["epochs"] = int(match.group(1))
    if "dataset" not in overrides:
        match = re.search(r"on the ([\w\-]+) dataset", lowered) or re.search(
            r"dataset (?:called |named )?([\w\-]+)", lowered)
        if match:
            name = match.group(1).replace("-", "")
            if name != "sst2":
                overrides["dataset"] = name
    return overrides


def _detect_irreversible(text: str) -> dict | None:
    lowered = text.lower()
    if "checkpoint" in lowered and re.search(
            r"delete|remove|purge|free up|clean up", lowered):
        action = ProposedAction(
            action_type="delete",
            description="Delete old checkpoints to free up disk space, as asked in the request",
            tool_name="purge_attempt_artifacts",
            args={"target": "old_checkpoints"},
            irreversible=True,
            source="user",
        )
        return action.model_dump()
    return None


def make_intake_profiler(ctx: AgentContext):
    def intake_profiler(state: dict) -> dict:
        raw = state.get("user_request_raw", "")
        hints = _extract_hints(raw)
        session_ctx = state.get("session_context") or ""
        if (session_ctx and "task_type" not in hints
                and any(word in raw.lower()
                        for word in ("same", "as before", "again"))
                and _pretrain_intent(session_ctx.lower())):
            # A referential follow-up ("same corpus, but train for 6
            # epochs") inside a pretraining session continues the
            # pretraining task. Fields the new request states explicitly
            # (epochs, workers) were extracted above and win; only the
            # task identity is carried over from the session context.
            ctx_hints = _extract_hints(session_ctx)
            for key in ("task_type", "model_name", "dataset"):
                if key in ctx_hints:
                    hints.setdefault(key, ctx_hints[key])
        user_prompt = (
            json.dumps(hints)
            + "\n"
            + wrap_untrusted("user_request", _brace_safe(raw))
        )
        text, budget_dump = _chat(
            ctx, state, "intake_profiler",
            load_prompt("intake_system.txt"), user_prompt)
        data = _extract_json(text) or {}
        fields = set(ParsedRequest.model_fields)
        parsed_data = {k: v for k, v in data.items() if k in fields and k != "raw_text"}
        for key, value in hints.items():
            parsed_data.setdefault(key, value)
        parsed_data.update(_explicit_overrides(raw))
        try:
            request = ParsedRequest(raw_text=raw, **parsed_data)
        except Exception:
            request = ParsedRequest(raw_text=raw, **hints)
        profile = MachineProfile()
        return {
            "request": request.model_dump(),
            "machine_profile": profile.model_dump(),
            "pending_irreversible": _detect_irreversible(raw),
            "status": "planning",
            "budget": budget_dump,
        }

    return intake_profiler


# ----------------------------------------------------------------------
# 2. planner
# ----------------------------------------------------------------------

def make_planner(ctx: AgentContext):
    def planner(state: dict) -> dict:
        request = ParsedRequest(**(state.get("request") or {"raw_text": ""}))
        previous_spec = (
            TrainingSpec(**state["spec"]) if state.get("spec") else None)
        diagnosis = state.get("last_diagnosis") or {}
        proposed_fix = dict(diagnosis.get("proposed_fix") or {})

        entries = memory_mod.select_for_prompt(
            state.get("failure_memory", []),
            error_class=diagnosis.get("error_class"))
        lr_match = _LEARNING_RATE_RE.search(request.raw_text.lower())
        merged = {
            "model_name": request.model_name,
            "dataset": request.dataset,
            "task_type": request.task_type,
            "learning_rate": (
                float(lr_match.group(1)) if lr_match else None),
            "num_workers": request.num_workers,
            "epochs": request.epochs,
            "proposed_fix": proposed_fix,
            "machine_cpu_total": (
                state.get("machine_profile") or {}).get("cpu_total", 8),
        }
        parts = [json.dumps(merged)]
        if entries:
            parts.append(wrap_untrusted(
                "failure_memory", memory_mod.format_for_prompt(entries)))
        if state.get("session_context"):
            parts.append(wrap_untrusted(
                "session_context", _brace_safe(state["session_context"])))
        text, budget_dump = _chat(
            ctx, state, "planner",
            load_prompt("planner_system.txt"), "\n".join(parts))
        data = _extract_json(text) or {}
        spec_fields = set(TrainingSpec.model_fields)
        spec_data = {k: v for k, v in data.items() if k in spec_fields}
        try:
            spec = TrainingSpec(**spec_data)
        except Exception:
            if previous_spec is not None:
                spec = previous_spec
            elif request.task_type == "pretrain":
                spec = TrainingSpec(
                    model_name=request.model_name,
                    dataset=request.dataset,
                    num_workers=request.num_workers,
                    epochs=request.epochs,
                    task_type="pretrain",
                    train_subsample=100000,
                    learning_rate=0.1,
                )
            else:
                spec = TrainingSpec(
                    model_name=request.model_name,
                    dataset=request.dataset,
                    num_workers=request.num_workers,
                    epochs=request.epochs,
                )
        # The request wins over the LLM reply on task type: a pretraining
        # request must never come back as a fine-tune spec (or the reverse)
        # just because the reply omitted or garbled the field.
        if spec.task_type != request.task_type:
            spec = TrainingSpec(
                **{**spec.model_dump(), "task_type": request.task_type})
        # Deterministic adoption of the critic's fix: spec fields are
        # applied to the new spec; named repair actions are recorded.
        memory_entries = [dict(e) for e in state.get("failure_memory", [])]
        repair_applied = False
        if proposed_fix:
            valid_fix = _validate_fix(
                Diagnosis(symptom="", root_cause="", proposed_fix=proposed_fix),
                spec)
            spec_part = {k: v for k, v in valid_fix.items() if k in spec_fields}
            if spec_part:
                spec = TrainingSpec(**{**spec.model_dump(), **spec_part})
                repair_applied = True
            if any(k in _ACTION_FIX_KEYS for k in valid_fix):
                repair_applied = True
            if memory_entries:
                memory_entries[-1]["fix_applied"] = valid_fix
                memory_entries[-1]["outcome"] = "applied"
                memory_mod.append_entry(ctx.run_dir, memory_entries[-1])
        # Keep the documented effective-batch invariant (16-64) when the
        # requested worker count would otherwise blow past it: shrink
        # gradient accumulation first (floor 1) instead of emitting a
        # spec the trainer must refuse.
        eff = (spec.per_worker_batch_size
               * spec.gradient_accumulation_steps * spec.num_workers)
        if eff > 64 and spec.gradient_accumulation_steps > 1:
            accum = max(
                1, 64 // (spec.per_worker_batch_size * spec.num_workers))
            spec = TrainingSpec(
                **{**spec.model_dump(),
                   "gradient_accumulation_steps": accum})
        spec_version = int(state.get("spec_version", 0)) + 1
        attempt = int(state.get("attempt", 0))
        plan_steps = [
            {"step_id": "s1", "description": "Parse the request and profile the environment",
             "depends_on": []},
            {"step_id": "s2", "description": f"Render train.py and RayJob YAML from spec {spec.spec_hash()}",
             "depends_on": ["s1"]},
            {"step_id": "s3", "description": f"Launch RayJob attempt {attempt + 1} after approval",
             "depends_on": ["s2"]},
            {"step_id": "s4", "description": "Verify the checkpoint and metrics against the request",
             "depends_on": ["s3"]},
        ]
        if repair_applied:
            plan_steps.append({
                "step_id": "s5",
                "description": f"Repair applied from diagnosis: {json.dumps(proposed_fix)}",
                "depends_on": ["s1"],
            })
        return {
            "spec": spec.model_dump(),
            "spec_version": spec_version,
            "spec_hash": spec.spec_hash(),
            "plan_steps": plan_steps,
            "failure_memory": memory_entries,
            "status": "coding",
            "budget": budget_dump,
        }

    return planner


# ----------------------------------------------------------------------
# 3. coder
# ----------------------------------------------------------------------

def make_coder(ctx: AgentContext):
    def coder(state: dict) -> dict:
        spec = TrainingSpec(**state["spec"])
        attempt = int(state.get("attempt", 0))
        spec_version = int(state.get("spec_version", 1))
        job_name = f"trainpilot-{state.get('run_id', 'run')[:8]}-a{attempt}"
        attempt_dir = ctx.attempt_dir(attempt)
        already_executed = bool(
            attempt_dir and (attempt_dir / "logs.txt").exists())
        contents = {
            "train_script": rendering.render_train_script(spec, attempt, spec_version),
            "rayjob_yaml": rendering.render_rayjob_yaml(spec, job_name, attempt, spec_version),
            "train_config": rendering.render_train_config_yaml(spec, attempt, spec_version),
        }
        filenames = {
            "train_script": "train.py",
            "rayjob_yaml": "rayjob.yaml",
            "train_config": "train_config.yaml",
        }
        generated: dict[str, dict] = {}
        if attempt_dir is not None and not already_executed:
            attempt_dir.mkdir(parents=True, exist_ok=True)
            (attempt_dir / "spec.json").write_text(
                json.dumps(spec.model_dump(), indent=2, sort_keys=True),
                encoding="utf-8")
        for key, content in contents.items():
            path = str(attempt_dir / filenames[key]) if attempt_dir else filenames[key]
            if attempt_dir is not None and not already_executed:
                (attempt_dir / filenames[key]).write_text(content, encoding="utf-8")
            generated[key] = rendering.make_generated_file(
                path, content, spec).model_dump()
        return {
            "generated_files": generated,
            "ray_job_name": job_name,
            "status": "preflight",
        }

    return coder


# ----------------------------------------------------------------------
# 4. preflight_validator (deterministic, no LLM)
# ----------------------------------------------------------------------

def run_preflight_checks(state: dict) -> PreflightResult:
    errors: list[str] = []
    warnings: list[str] = []
    spec = TrainingSpec(**state["spec"])
    generated = state.get("generated_files") or {}
    request = ParsedRequest(**(state.get("request") or {"raw_text": ""}))
    profile = MachineProfile(**(state.get("machine_profile") or {}))

    for key in ("train_script", "rayjob_yaml", "train_config"):
        entry = generated.get(key)
        if not entry:
            errors.append(f"generated file missing: {key}")
        elif entry.get("spec_hash") != spec.spec_hash():
            errors.append(f"spec_hash mismatch in {key}")
    script = (generated.get("train_script") or {}).get("content", "")
    if script:
        try:
            ast.parse(script)
        except SyntaxError as exc:
            errors.append(f"train.py has a syntax error: {exc}")
        for pattern in _BANNED_PATTERNS:
            if pattern in script:
                errors.append(f"banned pattern in train.py: {pattern}")
    rayjob_text = (generated.get("rayjob_yaml") or {}).get("content", "")
    if rayjob_text:
        try:
            parsed_yaml = yaml.safe_load(rayjob_text)
            if not isinstance(parsed_yaml, dict) or "spec" not in parsed_yaml:
                errors.append("rayjob.yaml is missing the spec section")
            elif "entrypoint" not in parsed_yaml["spec"]:
                errors.append("rayjob.yaml is missing spec.entrypoint")
        except yaml.YAMLError as exc:
            errors.append(f"rayjob.yaml does not parse: {exc}")
    if spec.num_workers > profile.cpu_total:
        errors.append("num_workers exceeds the machine profile")
    if _memory_gb(spec.resources_per_worker_memory) > _memory_gb(
            request.max_memory_per_worker):
        errors.append("per-worker memory exceeds the request limit")
    effective = spec.effective_batch()
    if not 16 <= effective <= 64:
        errors.append(f"effective batch {effective} outside the 16..64 invariant")
    if request.cpu_only and spec.backend != "gloo":
        errors.append("cpu_only request requires the gloo backend")
    return PreflightResult(passed=not errors, errors=errors, warnings=warnings)


def make_preflight_validator(ctx: AgentContext):
    def preflight_validator(state: dict) -> dict:
        result = run_preflight_checks(state)
        updates: dict[str, Any] = {
            "preflight_passed": result.passed,
            "preflight_errors": result.errors,
        }
        if result.passed:
            updates["preflight_route"] = "pass"
            updates["status"] = "awaiting_approval"
        elif int(state.get("preflight_fix_count", 0)) < 1:
            updates["preflight_route"] = "fix"
            updates["preflight_fix_count"] = int(state.get("preflight_fix_count", 0)) + 1
            updates["status"] = "coding"
        else:
            updates["preflight_route"] = "fail"
            updates["status"] = "diagnosing"
            updates["ray_job_status"] = "failed"
            updates["logs_summary_sanitized"] = sanitize_untrusted_text(
                "PREFLIGHT FAILURE\n" + "\n".join(result.errors))
            updates["logs_unavailable"] = False
        return updates

    return preflight_validator


# ----------------------------------------------------------------------
# 5. approval_gate
# ----------------------------------------------------------------------

def make_approval_gate(ctx: AgentContext):
    def approval_gate(state: dict) -> dict:
        handler = ctx.handler()
        updates: dict[str, Any] = {"status": "executing"}
        notes: list[str] = []

        pending = state.get("pending_irreversible")
        decided = state.get("approval_irreversible") or {}
        if pending and "delete_old_checkpoints" not in decided:
            action = ProposedAction(**pending)
            ctx.emit("approval_required", {
                "action_type": action.action_type,
                "description": action.description,
                "irreversible": True,
                "attempt": int(state.get("attempt", 0)),
            })
            decision: ApprovalDecision = handler(action)
            decided = dict(decided)
            decided["delete_old_checkpoints"] = (
                "approved" if decision.approve else "rejected")
            updates["approval_irreversible"] = decided
            notes.append(f"irreversible:{decision.reason_code}")
            if not decision.approve:
                updates.update({
                    "approval_route": "rejected",
                    "status": "rejected",
                    "outcome": "rejected",
                    "approval_notes": "; ".join(notes),
                })
                return updates

        attempt = int(state.get("attempt", 0))
        launch = ProposedAction(
            action_type="launch",
            description=f"Launch RayJob {state.get('ray_job_name')} (attempt {attempt + 1})",
            tool_name="apply_rayjob",
            args={"attempt": attempt, "job_name": state.get("ray_job_name")},
            irreversible=False,
            source="agent",
        )
        ctx.emit("approval_required", {
            "action_type": "launch",
            "description": launch.description,
            "irreversible": False,
            "attempt": attempt,
        })
        decision = handler(launch)
        notes.append(f"launch:{decision.reason_code}")
        updates["approval_notes"] = "; ".join(notes)
        if not decision.approve:
            updates.update({
                "approval_route": "rejected",
                "status": "rejected",
                "outcome": "rejected",
            })
            return updates
        updates["approval_route"] = "approved"
        updates["launch_approved_attempt"] = attempt
        if attempt == 0:
            updates["approval_first_launch"] = "approved"
        return updates

    return approval_gate


# ----------------------------------------------------------------------
# 6. executor (deterministic, no LLM)
# ----------------------------------------------------------------------

def make_executor_node(ctx: AgentContext):
    def executor(state: dict) -> dict:
        attempt = int(state.get("attempt", 0))
        if int(state.get("launch_approved_attempt", -1)) != attempt:
            return {
                "status": "failed_fatal",
                "error_fatal": "executor reached without an approved launch",
                "ray_job_status": "failed",
                "executor_result": {
                    "status": "failed",
                    "logs": "executor blocked: launch was not approved",
                    "metrics": None, "checkpoint_dir": None,
                    "duration_seconds": 0.0,
                },
            }
        spec = TrainingSpec(**state["spec"])
        attempt_dir = ctx.attempt_dir(attempt) or Path(".")
        result = ctx.executor.execute(spec, attempt, attempt_dir)
        working = budget_mod.budget_from_state(state.get("budget"))
        working = budget_mod.charge_tool_call(working)
        working = budget_mod.refresh_wall_clock(working, ctx.started_monotonic)
        return {
            "executor_result": result,
            "attempts_started": int(state.get("attempts_started", 0)) + 1,
            "status": "monitoring",
            "budget": budget_mod.dump(working),
        }

    return executor


# ----------------------------------------------------------------------
# 7. monitor_poller (deterministic, no LLM)
# ----------------------------------------------------------------------

def make_monitor_poller(ctx: AgentContext):
    def monitor_poller(state: dict) -> dict:
        result = state.get("executor_result") or {}
        status = "succeeded" if result.get("status") == "succeeded" else "failed"
        working = budget_mod.budget_from_state(state.get("budget"))
        working = budget_mod.charge_tool_call(working)
        working = budget_mod.refresh_wall_clock(working, ctx.started_monotonic)
        return {
            "ray_job_status": status,
            "status": "monitoring",
            "budget": budget_mod.dump(working),
        }

    return monitor_poller


# ----------------------------------------------------------------------
# 8. log_collector (deterministic, no LLM)
# ----------------------------------------------------------------------

def make_log_collector(ctx: AgentContext):
    def log_collector(state: dict) -> dict:
        result = state.get("executor_result") or {}
        logs = result.get("logs") or ""
        attempt = int(state.get("attempt", 0))
        attempt_dir = ctx.attempt_dir(attempt)
        logs_ref = None
        if attempt_dir is not None:
            attempt_dir.mkdir(parents=True, exist_ok=True)
            log_path = attempt_dir / "logs.txt"
            log_path.write_text(logs, encoding="utf-8")
            logs_ref = str(log_path)
        working = budget_mod.budget_from_state(state.get("budget"))
        working = budget_mod.charge_tool_call(working)
        updates: dict[str, Any] = {
            "logs_ref": logs_ref,
            "logs_summary_sanitized": sanitize_untrusted_text(logs),
            "logs_unavailable": not bool(logs),
            "budget": budget_mod.dump(working),
        }
        if state.get("ray_job_status") == "succeeded":
            updates["status"] = "verifying"
        else:
            updates["status"] = "diagnosing"
            ctx.emit("attempt_failed", {
                "attempt": attempt,
                "job_name": state.get("ray_job_name"),
            })
        return updates

    return log_collector


# ----------------------------------------------------------------------
# 9. critic
# ----------------------------------------------------------------------

def make_critic(ctx: AgentContext):
    def critic(state: dict) -> dict:
        # A new diagnosis means the previous entry's fix did not work.
        entries = [dict(e) for e in state.get("failure_memory", [])]
        if entries and entries[-1].get("outcome") in ("pending", "applied"):
            entries[-1]["outcome"] = "failed"
            memory_mod.append_entry(ctx.run_dir, entries[-1])

        logs_summary = state.get("logs_summary_sanitized") or ""
        spec = TrainingSpec(**state["spec"]) if state.get("spec") else None
        history = memory_mod.select_for_prompt(entries)
        parts = [wrap_untrusted("training_log", logs_summary)]
        if history:
            # Per design A.5 the critic receives only the error_class and
            # fix_applied of past entries (for de-duplication), never the
            # free-text causes - those would contaminate the new diagnosis.
            compact = "\n".join(
                "- error_class: {ec} | fix_applied: {fix} | outcome: {oc}".format(
                    ec=entry.get("error_class", ""),
                    fix=json.dumps(entry.get("fix_applied")),
                    oc=entry.get("outcome", "pending"))
                for entry in history)
            parts.append(wrap_untrusted("failure_memory", compact))
        if spec is not None:
            parts.append("Current spec JSON: " + json.dumps(spec.model_dump()))
        text, budget_dump = _chat(
            ctx, state, "critic",
            load_prompt("critic_system.txt"), "\n".join(parts))
        data = _extract_json(text)
        diagnosis: Diagnosis
        if data is not None:
            try:
                diagnosis = Diagnosis(**{
                    k: v for k, v in data.items()
                    if k in Diagnosis.model_fields})
            except Exception:
                diagnosis = Diagnosis(
                    symptom="run failed without a recognized signature",
                    root_cause="unknown", error_class=ErrorClass.UNKNOWN,
                    confidence=0.3)
        else:
            diagnosis = Diagnosis(
                symptom="run failed without a recognized signature",
                root_cause="unknown", error_class=ErrorClass.UNKNOWN,
                confidence=0.3)

        kept_evidence: list[str] = []
        dropped = 0
        lowered_logs = logs_summary.lower()
        for item in list(diagnosis.evidence)[:5]:
            if item.lower() in lowered_logs:
                kept_evidence.append(item[:300])
            else:
                dropped += 1
        confidence = max(0.05, diagnosis.confidence - 0.1 * dropped)
        if state.get("logs_unavailable"):
            confidence = min(confidence, 0.3)
        diagnosis = diagnosis.model_copy(update={
            "evidence": kept_evidence,
            "confidence": confidence,
            "proposed_fix": _validate_fix(diagnosis, spec),
        })

        seq = len(state.get("failures_written", [])) + 1
        entry = memory_mod.make_entry(
            diagnosis.model_dump(mode="json"), state.get("run_id", ctx.run_id),
            int(state.get("attempt", 0)), seq)
        entries.append(entry)
        memory_mod.append_entry(ctx.run_dir, entry)
        fault_family, root_cause_code, fix_code = memory_mod.classify_codes(
            diagnosis.model_dump(mode="json"))
        record = DiagnosisRecord(
            fault_family=fault_family,
            root_cause_code=root_cause_code,
            evidence=kept_evidence,
            proposed_fix_code=fix_code,
        )
        ctx.emit("diagnosis_ready", {
            "attempt": int(state.get("attempt", 0)),
            "error_class": diagnosis.error_class.value,
            "root_cause_code": root_cause_code,
            "fix_code": fix_code,
            "confidence": confidence,
        })
        return {
            "last_diagnosis": diagnosis.model_dump(mode="json"),
            "failure_memory": entries,
            "failures_written": list(state.get("failures_written", [])) + [entry],
            "diagnosis_records": list(state.get("diagnosis_records", [])) + [
                record.model_dump()],
            "status": "repairing",
            "budget": budget_dump,
        }

    return critic


# ----------------------------------------------------------------------
# 10. repair_router (deterministic decision node)
# ----------------------------------------------------------------------

def make_repair_router(ctx: AgentContext):
    def repair_router(state: dict) -> dict:
        working = budget_mod.refresh_wall_clock(
            budget_mod.budget_from_state(state.get("budget")),
            ctx.started_monotonic)
        updates: dict[str, Any] = {"budget": budget_mod.dump(working)}
        entries = [dict(e) for e in state.get("failure_memory", [])]

        def give_up(reason: str, outcome: str = "gave_up") -> dict:
            if entries and entries[-1].get("outcome") == "pending":
                entries[-1]["outcome"] = "not_applied"
                memory_mod.append_entry(ctx.run_dir, entries[-1])
            updates.update({
                "route_decision": "give_up",
                "give_up_reason": reason,
                "outcome": outcome,
                "status": outcome,
                "failure_memory": entries,
            })
            return updates

        if state.get("ray_job_status") == "succeeded":
            # The training run succeeded but verification did not pass.
            # One re-verification is allowed per attempt (a checkpoint
            # flush can lag), but verification is deterministic: routing
            # back to verify a second time would loop until the graph
            # recursion limit. Fall through to the give-up logic instead.
            attempt = int(state.get("attempt", 0))
            if int(state.get("verify_retry_attempt", -1)) != attempt:
                updates.update({
                    "route_decision": "verify",
                    "status": "verifying",
                    "verify_retry_attempt": attempt,
                })
                return updates
        exceeded = working.exceeded()
        if exceeded:
            return give_up(f"budget_exceeded:{exceeded}", outcome="budget_exceeded")
        if int(state.get("attempts_started", 0)) >= int(
                state.get("max_attempts", MAX_ATTEMPTS)):
            return give_up("attempts_exhausted")
        if int(state.get("repair_cycles", 0)) >= MAX_REPAIR_CYCLES:
            return give_up("repair_cycle_limit")
        diagnosis = state.get("last_diagnosis")
        if diagnosis is None:
            updates.update({
                "route_decision": "give_up",
                "give_up_reason": "missing_diagnosis",
                "outcome": "gave_up",
                "status": "failed_fatal",
                "error_fatal": "repair_router reached without a diagnosis",
            })
            return updates
        fix = diagnosis.get("proposed_fix") or {}
        if not fix:
            return give_up("no_safe_repair")
        if float(diagnosis.get("confidence", 0.0)) < 0.40:
            return give_up("low_confidence")
        _, _, fix_code = memory_mod.classify_codes(diagnosis)
        for entry in entries[:-1]:
            if (entry.get("outcome") == "failed"
                    and entry.get("fix_code") == fix_code
                    and entry.get("error_class") == diagnosis.get("error_class")):
                return give_up("repeated_ineffective_fix")
        updates.update({
            "route_decision": "replan",
            "status": "planning",
            "attempt": int(state.get("attempts_started", 0)),
            "repair_cycles": int(state.get("repair_cycles", 0)) + 1,
            "preflight_fix_count": 0,
            "failure_memory": entries,
        })
        return updates

    return repair_router


# ----------------------------------------------------------------------
# 11. verifier (deterministic, no LLM)
# ----------------------------------------------------------------------

def make_verifier(ctx: AgentContext):
    def verifier(state: dict) -> dict:
        result = state.get("executor_result") or {}
        spec = TrainingSpec(**state["spec"]) if state.get("spec") else TrainingSpec()
        request = ParsedRequest(**(state.get("request") or {"raw_text": ""}))
        details: dict[str, Any] = {}

        checkpoint_dir = result.get("checkpoint_dir")
        exists = bool(checkpoint_dir) and Path(str(checkpoint_dir)).exists()
        loadable = False
        if exists:
            ckpt = Path(str(checkpoint_dir))
            config_path = ckpt / "config.json"
            weights = list(ckpt.glob("*.safetensors")) + list(
                ckpt.glob("pytorch_model.bin"))
            try:
                if config_path.exists():
                    json.loads(config_path.read_text(encoding="utf-8"))
                loadable = bool(weights) and all(
                    w.stat().st_size > 0 for w in weights)
            except Exception as exc:  # malformed checkpoint metadata
                details["load_error"] = str(exc)
                loadable = False
        metrics = result.get("metrics") or {}
        accuracy = metrics.get("eval_accuracy")

        honors = True
        if spec.num_workers != request.num_workers:
            honors = False
            details["workers_mismatch"] = (
                f"spec {spec.num_workers} != request {request.num_workers}")
        if request.cpu_only and spec.backend != "gloo":
            honors = False
            details["backend_mismatch"] = "cpu_only request but backend is not gloo"
        if not request.cpu_only and spec.backend == "gloo":
            honors = False
            details["accelerator_mismatch"] = (
                "request asked for GPU but the local demo ran on CPU "
                "(gloo backend)")
        if spec.epochs != request.epochs:
            honors = False
            details["epochs_mismatch"] = (
                f"spec {spec.epochs} != request {request.epochs}")
        if _memory_gb(spec.resources_per_worker_memory) > _memory_gb(
                request.max_memory_per_worker):
            honors = False
            details["memory_mismatch"] = "spec memory exceeds the request limit"
        if spec.model_name != request.model_name or spec.dataset != request.dataset:
            honors = False
            details["model_dataset_mismatch"] = "spec changed the requested model or dataset"

        event: dict[str, Any] = {
            "attempt": int(state.get("attempt", 0)),
        }
        if spec.task_type == "pretrain":
            # A pretrained LM has no accuracy: it passes only if its eval
            # perplexity clearly beats the untrained baseline's.
            eval_perplexity = metrics.get("eval_perplexity")
            baseline_perplexity = metrics.get("baseline_perplexity")
            details["eval_perplexity"] = eval_perplexity
            details["baseline_perplexity"] = baseline_perplexity
            verification = VerificationResult(
                checkpoint_exists=exists,
                checkpoint_loadable=loadable,
                validation_accuracy=None,
                validation_perplexity=eval_perplexity,
                baseline_perplexity=baseline_perplexity,
                config_honors_request=honors,
                details=details,
            )
            event.update({
                "validation_accuracy": None,
                "eval_perplexity": eval_perplexity,
                "baseline_perplexity": baseline_perplexity,
            })
        else:
            verification = VerificationResult(
                checkpoint_exists=exists,
                checkpoint_loadable=loadable,
                validation_accuracy=accuracy,
                config_honors_request=honors,
                details=details,
            )
            event["validation_accuracy"] = accuracy
        dump = verification.model_dump()
        dump["passed"] = verification.passed
        working = budget_mod.budget_from_state(state.get("budget"))
        working = budget_mod.charge_tool_call(working)
        working = budget_mod.refresh_wall_clock(working, ctx.started_monotonic)
        event["passed"] = verification.passed
        ctx.emit("verification_done", event)
        updates: dict[str, Any] = {
            "verification": dump,
            "budget": budget_mod.dump(working),
        }
        entries = [dict(e) for e in state.get("failure_memory", [])]
        if verification.passed:
            if entries and entries[-1].get("fix_applied") is not None:
                entries[-1]["outcome"] = "succeeded"
                memory_mod.append_entry(ctx.run_dir, entries[-1])
            updates.update({
                "success": True,
                "claimed_success": True,
                "outcome": "succeeded",
                "status": "succeeded",
                "failure_memory": entries,
            })
        else:
            updates.update({
                "success": False,
                "claimed_success": False,
                "status": "diagnosing",
            })
        return updates

    return verifier


# ----------------------------------------------------------------------
# 12. final_reporter
# ----------------------------------------------------------------------

def make_final_reporter(ctx: AgentContext):
    def final_reporter(state: dict) -> dict:
        attempts = int(state.get("attempts_started", 0))
        outcome = state.get("outcome") or (
            "succeeded" if state.get("success") else "gave_up")
        verification = state.get("verification") or {}
        diagnosis = state.get("last_diagnosis") or {}
        spec = state.get("spec") or {}
        if outcome == "succeeded":
            if spec.get("task_type") == "pretrain" or (
                    verification.get("validation_perplexity") is not None):
                headline = (
                    f"Pretraining succeeded after {attempts} attempt(s): "
                    f"checkpoint verified with eval_perplexity "
                    f"{verification.get('validation_perplexity')} "
                    f"(untrained baseline "
                    f"{verification.get('baseline_perplexity')}, spec hash "
                    f"{spec.get('spec_hash', state.get('spec_hash', ''))})."
                )
                samples = (state.get("executor_result") or {}).get(
                    "samples") or []
                if samples:
                    headline += f' Sample output: "{samples[0]}"'
            else:
                headline = (
                    f"Training succeeded after {attempts} attempt(s): checkpoint "
                    f"verified with eval_accuracy "
                    f"{verification.get('validation_accuracy')} "
                    f"(spec hash {spec.get('spec_hash', state.get('spec_hash', ''))})."
                )
        elif outcome == "rejected":
            headline = (
                "Run stopped before completion: an approval was denied "
                f"({state.get('approval_notes', 'no reason recorded')}). "
                "No irreversible action was taken."
            )
        elif outcome == "budget_exceeded":
            headline = (
                f"Run stopped: budget exceeded after {attempts} attempt(s). "
                f"Last diagnosis: {diagnosis.get('root_cause', 'none')}."
            )
        else:
            headline = (
                f"Run gave up after {attempts} attempt(s) "
                f"(reason: {state.get('give_up_reason', 'unknown')}). "
                f"Last diagnosis: {diagnosis.get('root_cause', 'none')}."
            )
        # Optional LLM polish (nano, low effort): the stub echoes the
        # headline; accept the rewrite only if it keeps the outcome word
        # consistent with the verified state, else keep the template.
        message = headline
        budget_dump = state.get("budget") or {}
        try:
            text, budget_dump = _chat(
                ctx, state, "final_reporter",
                load_prompt("reporter_system.txt"), headline)
            candidate = text.strip().splitlines()[0][:200] if text.strip() else ""
            if candidate and headline.startswith(candidate):
                # The stub (and some models) echo the headline back,
                # truncated at 200 chars; an echo is not a polish.
                candidate = ""
            if candidate:
                if outcome == "succeeded" and "succeeded" in candidate.lower():
                    message = candidate
                elif outcome != "succeeded" and "succeeded" not in candidate.lower():
                    message = candidate
        except Exception:
            pass

        if ctx.run_dir is not None:
            ctx.run_dir.mkdir(parents=True, exist_ok=True)
            report = {
                "run_id": state.get("run_id", ctx.run_id),
                "outcome": outcome,
                "attempts": attempts,
                "max_attempts": int(state.get("max_attempts", MAX_ATTEMPTS)),
                "claimed_success": bool(state.get("claimed_success", False)),
                "verification": verification,
                "final_message": message,
                "diagnosis_records": state.get("diagnosis_records", []),
                "budget": state.get("budget", {}),
                "models_used": {
                    name: {"model": model, "effort": effort}
                    for name, (model, effort) in NODE_MODEL_ROUTING.items()
                },
            }
            (ctx.run_dir / "final_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            (ctx.run_dir / "final_report.md").write_text(
                rendering.render_final_report({
                    "run_id": report["run_id"],
                    "outcome": outcome,
                    "attempts": attempts,
                    "max_attempts": report["max_attempts"],
                    "claimed_success": report["claimed_success"],
                    "verification_passed": bool(verification.get("passed")),
                    "final_message": message,
                    "diagnosis_records": report["diagnosis_records"],
                }),
                encoding="utf-8")
        return {
            "final_message": message,
            "outcome": outcome,
            "status": outcome,
            "budget": budget_dump,
        }

    return final_reporter
