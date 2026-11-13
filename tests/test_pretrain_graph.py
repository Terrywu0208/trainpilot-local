"""Pretraining (tiny language model) wiring through the agent graph.

Covers the intake parse (task_type / model / corpus), the perplexity
branch of VerificationResult.passed, the full repair wiring with a
scripted executor, and — once the trainer side lands the frozen
contract in training/train.py + training/local_runner.py — the real
end-to-end runs through LocalExecutor. The LocalExecutor tests skip
with a clear reason until that contract is present.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from trainpilot.graph import StubExecutor, run_agent
from trainpilot.graph.context import AgentContext
from trainpilot.graph.nodes import make_intake_profiler
from trainpilot.llm import LLMClient
from trainpilot.schemas import (
    ApprovalDecision,
    ProposedAction,
    VerificationResult,
)
from training.local_runner import LocalExecutor

CANONICAL = (
    "Pretrain a tiny language model from scratch on the tinytales "
    "corpus with 2 CPU workers"
)
LM_NAN_LOG = (
    "step 3: lm loss became nan; perplexity is undefined (inf)\n"
    "RuntimeError: training diverged"
)
CORPUS_MISSING_LOG = (
    "ValueError: corpus 'mynotes' is not available in the local demo "
    "(error_signature: CORPUS_NOT_AVAILABLE)"
)


def approve_launches(action: ProposedAction) -> ApprovalDecision:
    return ApprovalDecision(
        approve=(action.action_type == "launch"),
        reason_code="TEST_APPROVE_LAUNCHES",
        decided_by="human",
    )


def _parse_request(text: str) -> dict:
    ctx = AgentContext(llm=LLMClient(), executor=StubExecutor([]))
    intake = make_intake_profiler(ctx)
    updates = intake({"user_request_raw": text})
    return updates["request"]


def _pretrain_success_outcome() -> dict:
    return {
        "status": "succeeded",
        "logs": "pretraining finished, checkpoint written",
        "metrics": {"eval_perplexity": 20.0, "baseline_perplexity": 40.0},
        "samples": ["once upon a time there was a tiny lamp"],
        "checkpoint_dir": None,
        "duration_seconds": 1.0,
    }


def _failed_outcome(logs: str) -> dict:
    return {
        "status": "failed",
        "logs": logs,
        "metrics": None,
        "checkpoint_dir": None,
        "duration_seconds": 1.0,
    }


# ----------------------------------------------------------------------
# Intake parse
# ----------------------------------------------------------------------

def test_intake_parses_pretrain_request():
    request = _parse_request(CANONICAL)
    assert request["task_type"] == "pretrain"
    assert request["model_name"] == "tiny-lm"
    assert request["dataset"] == "tinytales"
    assert request["num_workers"] == 2


def test_intake_keeps_unknown_corpus_name():
    # The named corpus is kept verbatim (honest), never substituted
    # with the bundled one at parse time.
    request = _parse_request(
        "Pretrain a language model on the mynotes corpus")
    assert request["task_type"] == "pretrain"
    assert request["model_name"] == "tiny-lm"
    assert request["dataset"] == "mynotes"


# ----------------------------------------------------------------------
# VerificationResult.passed: accuracy path unchanged, perplexity path
# ----------------------------------------------------------------------

def _result(**overrides) -> VerificationResult:
    fields = {
        "checkpoint_exists": True,
        "checkpoint_loadable": True,
        "config_honors_request": True,
    }
    fields.update(overrides)
    return VerificationResult(**fields)


def test_verification_accuracy_path_unchanged():
    # As before, the property only checks the presence of an accuracy,
    # not a threshold: 0.5 passes.
    assert _result(validation_accuracy=0.5).passed is True
    assert _result(validation_accuracy=0.87).passed is True
    # No accuracy and no perplexity: still a failure, as before.
    assert _result().passed is False
    # Checkpoint problems fail even with an accuracy present.
    assert _result(
        validation_accuracy=0.9, checkpoint_exists=False).passed is False
    assert _result(
        validation_accuracy=0.9, config_honors_request=False).passed is False


def test_verification_perplexity_path():
    # 0.8x baseline: clearly better than the untrained model -> passes.
    assert _result(
        validation_perplexity=32.0, baseline_perplexity=40.0).passed is True
    # 0.95x baseline: not a clear improvement -> fails.
    assert _result(
        validation_perplexity=38.0, baseline_perplexity=40.0).passed is False
    # No baseline to compare against: cannot claim success.
    assert _result(validation_perplexity=32.0).passed is False
    # Perplexity alone does not rescue a missing checkpoint.
    assert _result(
        validation_perplexity=32.0, baseline_perplexity=40.0,
        checkpoint_loadable=False).passed is False


# ----------------------------------------------------------------------
# Graph wiring with a scripted executor (no trainer dependency)
# ----------------------------------------------------------------------

def test_pretrain_wiring_with_stub_executor(tmp_path: Path):
    result = run_agent(
        CANONICAL,
        executor=StubExecutor([_pretrain_success_outcome()]),
        approval_handler=approve_launches,
        run_dir=tmp_path / "pretrain-stub",
    )
    assert result["success"] is True
    assert result["attempts"] == 1
    assert result["spec"]["task_type"] == "pretrain"
    assert result["spec"]["model_name"] == "tiny-lm"
    assert result["spec"]["dataset"] == "tinytales"
    assert result["spec"]["train_subsample"] == 100000
    assert result["spec"]["learning_rate"] == 0.1
    verification = result["verification"]
    assert verification["passed"] is True
    assert verification["validation_accuracy"] is None
    assert verification["validation_perplexity"] == 20.0
    assert verification["baseline_perplexity"] == 40.0
    message = result["final_message"]
    assert message.startswith("Pretraining succeeded")
    assert "eval_perplexity 20.0" in message
    assert "untrained baseline 40.0" in message
    assert 'Sample output: "once upon a time there was a tiny lamp"' in message


def test_pretrain_divergence_repair_with_stub_executor(tmp_path: Path):
    result = run_agent(
        CANONICAL + " with learning rate 5",
        executor=StubExecutor([
            _failed_outcome(LM_NAN_LOG),
            _pretrain_success_outcome(),
        ]),
        approval_handler=approve_launches,
        run_dir=tmp_path / "pretrain-divergence-stub",
    )
    assert result["success"] is True
    assert result["attempts"] == 2
    applied = [
        (entry.get("fix_applied") or {})
        for entry in result["failures"]
    ]
    assert any(fix.get("learning_rate") == 0.1 for fix in applied)
    assert result["spec"]["learning_rate"] == 0.1


def test_pretrain_unknown_corpus_terminal_with_stub_executor(
        tmp_path: Path):
    result = run_agent(
        "Pretrain a language model on the mynotes corpus "
        "with 2 CPU workers",
        executor=StubExecutor([_failed_outcome(CORPUS_MISSING_LOG)]),
        approval_handler=approve_launches,
        run_dir=tmp_path / "pretrain-unknown-corpus-stub",
    )
    # No safe repair exists for a missing corpus: the agent gives up
    # after the first attempt and never claims a checkpoint.
    assert result["success"] is False
    assert result["claimed_success"] is False
    assert result["attempts"] == 1
    assert result["give_up_reason"] == "no_safe_repair"
    assert result["verification"]["passed"] is False


# ----------------------------------------------------------------------
# End-to-end with the real LocalExecutor + trainer (frozen contract)
# ----------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _trainer_contract_present() -> bool:
    try:
        train_src = (_REPO_ROOT / "training" / "train.py").read_text(
            encoding="utf-8")
        runner_src = (_REPO_ROOT / "training" / "local_runner.py").read_text(
            encoding="utf-8")
    except OSError:
        return False
    return (
        all(marker in train_src for marker in (
            "task_type", "tinytales", "eval_perplexity",
            "baseline_perplexity"))
        and "eval_perplexity" in runner_src
        and "samples" in runner_src
    )


requires_trainer = pytest.mark.skipif(
    not _trainer_contract_present(),
    reason=(
        "pretrain trainer contract not yet landed in training/train.py "
        "+ training/local_runner.py (task_type / tinytales / "
        "eval_perplexity / baseline_perplexity / samples)"),
)


@requires_trainer
def test_pretrain_end_to_end(tmp_path: Path):
    result = run_agent(
        CANONICAL,
        executor=LocalExecutor(),
        approval_handler=approve_launches,
        run_dir=tmp_path / "pretrain-e2e",
    )
    assert result["success"] is True
    assert result["spec"]["task_type"] == "pretrain"
    verification = result["verification"]
    assert verification["passed"] is True
    assert verification["validation_accuracy"] is None
    perplexity = verification["validation_perplexity"]
    baseline = verification["baseline_perplexity"]
    assert perplexity is not None and baseline is not None
    assert perplexity < 0.9 * baseline
    assert "perplexity" in result["final_message"].lower()


@requires_trainer
def test_pretrain_divergence_repaired_end_to_end(tmp_path: Path):
    # NOTE on the learning rate: the landed trainer raises LM_LOSS_NAN
    # only when a training-step loss exceeds 50 (or goes non-finite).
    # Empirically (seed 42, canonical config) lr=5 stays just under
    # that line and lr>=8 trips it, so the repair scenario uses lr=10
    # to exercise the contracted divergence path against the real
    # trainer. The lr=5 boundary is characterized in the next test.
    result = run_agent(
        CANONICAL + " with learning rate 10",
        executor=LocalExecutor(),
        approval_handler=approve_launches,
        run_dir=tmp_path / "pretrain-divergence-e2e",
    )
    assert result["success"] is True
    assert result["attempts"] == 2
    applied = [
        (entry.get("fix_applied") or {})
        for entry in result["failures"]
    ]
    assert any(fix.get("learning_rate") == 0.1 for fix in applied)
    assert result["spec"]["learning_rate"] == 0.1


@requires_trainer
def test_pretrain_lr5_diverged_run_is_repaired(tmp_path: Path):
    # At lr=5 the tiny LM diverges. Integration fix (2026-11-13): the
    # trainer now also fails the run when the final validation loss
    # ends far above the untrained baseline (LM_DIVERGED), not only on
    # a NaN step, so the critic can diagnose the learning rate and the
    # repair loop closes: attempt 2 at lr 0.1 succeeds.
    result = run_agent(
        CANONICAL + " with learning rate 5",
        executor=LocalExecutor(),
        approval_handler=approve_launches,
        run_dir=tmp_path / "pretrain-lr5-e2e",
    )
    assert result["success"] is True
    assert result["attempts"] == 2
    applied = [
        (entry.get("fix_applied") or {})
        for entry in result["failures"]
    ]
    assert any(fix.get("learning_rate") == 0.1 for fix in applied)
    assert result["spec"]["learning_rate"] == 0.1


def test_intake_l16_prompt_stays_fine_tune():
    # Regression (found 2026-11-13): lab50 task L16 says "reload it
    # from scratch" about a dataset inside a fine-tune request. That
    # phrase alone must not flip the parse to pretraining.
    request = _parse_request(
        "Fine-tune DistilBERT (distilbert-base-uncased) on the SST-2 "
        "dataset (sst2) using a RayJob on the local Kind cluster with "
        "2 CPU workers and a 2Gi per-worker memory limit. The training "
        "script was copied from a GPU tutorial and configures the nccl "
        "backend.\nFollow-up: A teammate is sure the problem is the "
        "dataset and wants you to reload it from scratch.")
    assert request["task_type"] == "fine_tune"
    assert request["dataset"] == "sst2"


def test_intake_followup_continues_pretrain_session():
    # A referential follow-up inside a pretraining session keeps the
    # pretraining task identity; the explicitly stated epochs win.
    ctx = AgentContext(llm=LLMClient(), executor=StubExecutor([]))
    intake = make_intake_profiler(ctx)
    updates = intake({
        "user_request_raw":
            "Same corpus as before, but train for 6 epochs this time",
        "session_context":
            "Previous run run-1: request 'Pretrain a tiny language "
            "model from scratch on the tinytales corpus with 2 CPU "
            "workers'; outcome success after 1 attempt(s)",
    })
    request = updates["request"]
    assert request["task_type"] == "pretrain"
    assert request["model_name"] == "tiny-lm"
    assert request["dataset"] == "tinytales"
    assert request["epochs"] == 6


@requires_trainer
def test_pretrain_followup_end_to_end(tmp_path: Path):
    result = run_agent(
        "Same corpus as before, but train for 6 epochs this time",
        executor=LocalExecutor(),
        approval_handler=approve_launches,
        run_dir=tmp_path / "pretrain-followup-e2e",
        session_context=(
            "Previous run run-1: request 'Pretrain a tiny language "
            "model from scratch on the tinytales corpus with 2 CPU "
            "workers'; outcome success after 1 attempt(s); validation "
            "accuracy None; final message: Pretraining succeeded"),
    )
    assert result["success"] is True
    assert result["spec"]["task_type"] == "pretrain"
    assert result["spec"]["epochs"] == 6
    verification = result["verification"]
    assert verification["validation_perplexity"] is not None
    # Six epochs of full-batch GD must beat the 3-epoch canonical run
    # (eval_perplexity 211.59, deterministic seed 42).
    assert verification["validation_perplexity"] < 211.59


@requires_trainer
def test_pretrain_unknown_corpus_end_to_end(tmp_path: Path):
    result = run_agent(
        "Pretrain a language model on the mynotes corpus "
        "with 2 CPU workers",
        executor=LocalExecutor(),
        approval_handler=approve_launches,
        run_dir=tmp_path / "pretrain-unknown-corpus-e2e",
    )
    assert result["success"] is False
    assert result["claimed_success"] is False
    assert result["verification"]["passed"] is False
