"""Shared schemas for TrainPilot-Local.

Single source of truth for the data contracts between the agent core (A),
the tool layer (B), the training workload (C) and the evaluation layer (D).
Every module imports from here; nobody redefines these shapes locally.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Training request / spec (A section: spec is the single source of truth)
# --------------------------------------------------------------------------

class ParsedRequest(BaseModel):
    raw_text: str
    model_name: str = "distilbert-base-uncased"
    dataset: str = "sst2"
    task_type: str = "fine_tune"  # fine_tune | pretrain | distill
    num_workers: int = 2
    cpu_only: bool = True
    max_memory_per_worker: str = "3Gi"
    epochs: int = 1
    notes: str = ""


class TrainingSpec(BaseModel):
    """The canonical training specification. YAML is always rendered from
    this object (Jinja2), never patched as text."""

    model_name: str = "distilbert-base-uncased"
    model_revision: str = "main"
    dataset: str = "sst2"
    task_type: str = "fine_tune"
    backend: str = "gloo"  # gloo | nccl
    per_worker_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_workers: int = 2
    resources_per_worker_memory: str = "2Gi"
    epochs: int = 1
    max_steps: int = 40
    learning_rate: float = 2e-5
    train_subsample: int = 1024
    val_subsample: int = 256
    max_length: int = 64
    seed: int = 42

    def effective_batch(self) -> int:
        return (
            self.per_worker_batch_size
            * self.gradient_accumulation_steps
            * self.num_workers
        )

    def spec_hash(self) -> str:
        import hashlib
        import json

        payload = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


class GeneratedFile(BaseModel):
    path: str
    content: str
    spec_hash: str


# --------------------------------------------------------------------------
# Budget (A section in-graph copy; D section runner copy is authoritative
# in experiment mode)
# --------------------------------------------------------------------------

class BudgetState(BaseModel):
    max_wall_seconds: int = 1500
    max_tool_calls: int = 40
    max_total_tokens: int = 120000
    wall_seconds_used: float = 0.0
    tool_calls_used: int = 0
    tokens_used: int = 0

    def exceeded(self) -> Optional[str]:
        if self.wall_seconds_used > self.max_wall_seconds:
            return "wall_clock"
        if self.tool_calls_used > self.max_tool_calls:
            return "tool_calls"
        if self.tokens_used > self.max_total_tokens:
            return "tokens"
        return None


# --------------------------------------------------------------------------
# Diagnosis (A critic output + D scorer-side record)
# --------------------------------------------------------------------------

class ErrorClass(str, Enum):
    OOM = "OOM"
    DATA_SCHEMA = "DATA_SCHEMA"
    CODE_BUG = "CODE_BUG"
    RESOURCE_QUOTA = "RESOURCE_QUOTA"
    DEPENDENCY = "DEPENDENCY"
    NETWORK = "NETWORK"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


class Diagnosis(BaseModel):
    """Agent-side diagnosis produced by the critic node."""

    symptom: str
    root_cause: str
    evidence: list[str] = Field(default_factory=list)
    error_class: ErrorClass = ErrorClass.UNKNOWN
    proposed_fix: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5


class DiagnosisRecord(BaseModel):
    """Scorer-side closed-code record (D section). The critic emits this via
    a deterministic mapping when a run ends; scorers never re-guess it."""

    fault_family: str  # gpu_only_code | memory_limits | dependency_conflict |
    # data_or_training_error | none_clean | safety_trap | unknown
    root_cause_code: str
    evidence: list[str] = Field(default_factory=list)
    proposed_fix_code: str


# --------------------------------------------------------------------------
# Verification (A verifier node; run-internal, not the hidden grader)
# --------------------------------------------------------------------------

class VerificationResult(BaseModel):
    checkpoint_exists: bool = False
    checkpoint_loadable: bool = False
    validation_accuracy: Optional[float] = None
    validation_perplexity: Optional[float] = None
    baseline_perplexity: Optional[float] = None
    config_honors_request: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        base = (
            self.checkpoint_exists
            and self.checkpoint_loadable
            and self.config_honors_request
        )
        if self.validation_accuracy is not None:
            return base
        if self.validation_perplexity is not None and self.baseline_perplexity:
            return base and (
                self.validation_perplexity < 0.9 * self.baseline_perplexity)
        return False


# --------------------------------------------------------------------------
# Tools / approvals (B + D shared)
# --------------------------------------------------------------------------

class ToolResult(BaseModel):
    ok: bool
    tool_name: str
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: Optional[str] = None  # e.g. BUDGET_EXCEEDED_TOOL_CALLS
    error_message: str = ""


class ProposedAction(BaseModel):
    action_type: str  # read | patch | launch | delete | build_image | other
    description: str
    tool_name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    irreversible: bool = False
    source: str = "user"  # user | agent | planted


class ApprovalDecision(BaseModel):
    approve: bool
    reason_code: str
    decided_by: str = "scripted_user"  # scripted_user | human


# --------------------------------------------------------------------------
# Evaluation tasks (D section, schema v2 shared by lab50 and the benchmark)
# --------------------------------------------------------------------------

class TaskSpec(BaseModel):
    id: str
    prompt: str
    scenario_id: Optional[str] = None
    split: str = "dev"  # dev | heldout | lab | safety
    grading_type: str = "rubric"  # exact | tool | rubric | unanswerable
    expected: Optional[str] = None
    rubric: list[str] = Field(default_factory=list)
    expected_state: dict[str, Any] = Field(default_factory=dict)
    expected_output: dict[str, Any] = Field(default_factory=dict)
    difficulty: str = "easy"  # easy | hard | edge | unanswerable
    safety_category: Optional[str] = None
    safe_behavior: Optional[str] = None  # refuse | ask_confirmation | answer_without_harm
    notes: str = ""


class AgentRunOutput(BaseModel):
    answer: str = ""
    claimed_success: bool = False
    turns: int = 0
    tokens: int = 0
    latency_seconds: float = 0.0
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    diagnosis: Optional[DiagnosisRecord] = None
    final_state: dict[str, Any] = Field(default_factory=dict)


class GradeResult(BaseModel):
    run_pass: bool
    subchecks: dict[str, bool] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
