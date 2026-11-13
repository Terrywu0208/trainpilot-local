"""Graph state for the TrainPilot-Local agent (design section A.2).

Nested values are the shared pydantic models from ``trainpilot.schemas``,
stored in the state as plain dicts (``model_dump()``) so the checkpointer
can serialize them. The spec inside the state is the single source of
truth: files are always regenerated from it, never patched.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from pydantic import BaseModel, Field

MAX_ATTEMPTS = 3  # total launches: initial attempt + up to 2 repairs
MAX_REPAIR_CYCLES = 6  # safety net against pathological routing loops
MAX_INJECTED_FAILURES = 5  # failure-memory entries injected into a prompt


class MachineProfile(BaseModel):
    """Deterministic view of the execution environment (never LLM-made)."""

    cpu_total: int = 8
    gpu_count: int = 0
    ram_gb: int = 16
    namespace: str = "trainpilot-jobs"
    ray_version: str = "2.41.0"
    is_stub_mode: bool = True


class PlanStep(BaseModel):
    step_id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)


class PreflightResult(BaseModel):
    passed: bool = False
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TrainPilotState(TypedDict, total=False):
    # Identity and raw input
    run_id: str
    user_request_raw: str
    session_context: str
    status: str
    # Intake / planning
    request: Optional[dict[str, Any]]  # ParsedRequest dump
    machine_profile: Optional[dict[str, Any]]  # MachineProfile dump
    spec: Optional[dict[str, Any]]  # TrainingSpec dump
    spec_version: int
    spec_hash: str
    plan_steps: list[dict[str, Any]]
    generated_files: dict[str, dict[str, Any]]  # GeneratedFile dumps
    # Attempts and repair loop
    attempt: int  # index of the current attempt (0-based)
    attempts_started: int  # number of launches actually performed
    max_attempts: int
    repair_cycles: int
    route_decision: str  # replan | verify | give_up (set by repair_router)
    give_up_reason: str
    outcome: str  # succeeded | gave_up | rejected | budget_exceeded
    # Failure memory and diagnosis
    failure_memory: list[dict[str, Any]]  # all entries known this run
    failures_written: list[dict[str, Any]]  # entries first written this run
    last_diagnosis: Optional[dict[str, Any]]  # Diagnosis dump
    diagnosis_records: list[dict[str, Any]]  # DiagnosisRecord dumps
    # Approvals
    pending_irreversible: Optional[dict[str, Any]]  # ProposedAction dump
    approval_first_launch: str  # not_required | pending | approved | rejected
    approval_irreversible: dict[str, str]
    approval_notes: str
    launch_approved_attempt: int  # attempt index whose launch was approved
    approval_route: str  # approved | rejected (set by approval_gate)
    # Budget (BudgetState dump)
    budget: dict[str, Any]
    # Execution
    executor_result: Optional[dict[str, Any]]
    ray_job_name: Optional[str]
    ray_job_status: Optional[str]  # succeeded | failed | stopped | unknown
    logs_ref: Optional[str]
    logs_summary_sanitized: Optional[str]
    logs_unavailable: bool
    # Preflight
    preflight_passed: bool
    preflight_errors: list[str]
    preflight_route: str  # pass | fix | fail (set by preflight_validator)
    preflight_fix_count: int
    # Verification and closing
    verification: Optional[dict[str, Any]]  # VerificationResult dump
    claimed_success: bool
    success: bool
    final_message: str
    error_fatal: Optional[str]
