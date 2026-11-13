"""Single LLM entry point for the whole project (D section D.8).

Every LLM call in every system goes through `LLMClient.chat`, which records
usage into an in-memory ledger that the runner persists to llm_calls.jsonl.
The default provider is a deterministic stub: it never touches the network
and never costs money. The OpenAI provider is only constructed when a human
explicitly opts in with TRAINPILOT_LLM_PROVIDER=openai and their own key.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config.models import llm_provider


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@dataclass
class LLMClient:
    provider: str = field(default_factory=llm_provider)
    ledger: list[dict[str, Any]] = field(default_factory=list)
    _stub_handler: Optional[Any] = None  # callable(system, user, model) -> str

    def set_stub_handler(self, handler) -> None:
        """Systems/tests may install a domain-specific deterministic handler.
        The default handler below already covers the agent's node prompts."""
        self._stub_handler = handler

    def chat(self, system: str, user: str, model: str,
             effort: str = "medium") -> LLMResponse:
        started = time.monotonic()
        if self.provider == "openai":
            text, pt, ct = self._openai_chat(system, user, model, effort)
        else:
            text = self._stub_chat(system, user, model)
            pt = max(1, len(system + user) // 4)
            ct = max(1, len(text) // 4)
        latency_ms = (time.monotonic() - started) * 1000.0
        self.ledger.append({
            "model": model, "effort": effort,
            "prompt_tokens": pt, "completion_tokens": ct,
            "latency_ms": round(latency_ms, 2),
        })
        return LLMResponse(text=text, model=model, prompt_tokens=pt,
                           completion_tokens=ct, latency_ms=latency_ms)

    # ------------------------------------------------------------------
    def _openai_chat(self, system: str, user: str, model: str,
                     effort: str) -> tuple[str, int, int]:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "TRAINPILOT_LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Refusing to continue; the stub provider is the default."
            )
        # Imported lazily so the package never requires the SDK (or a key)
        # for offline work.
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=model,
            instructions=system,
            input=user,
            reasoning={"effort": effort},
        )
        usage = getattr(resp, "usage", None)
        return (
            resp.output_text,
            getattr(usage, "input_tokens", 0) if usage else 0,
            getattr(usage, "output_tokens", 0) if usage else 0,
        )

    # ------------------------------------------------------------------
    def _stub_chat(self, system: str, user: str, model: str) -> str:
        if self._stub_handler is not None:
            return self._stub_handler(system, user, model)
        return default_stub_response(system, user)


def _extract_json_block(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def default_stub_response(system: str, user: str) -> str:
    """Deterministic responses for the agent's node prompts.

    The stub is intentionally simple and rule-based: it parses the machine
    readable parts of the prompt (specs, log excerpts, failure memory) and
    answers with the same JSON shapes the real model is asked to produce.
    It exists so the full pipeline is testable offline at zero cost.
    """
    role = system.lower()
    if "intake" in role:
        data = _extract_json_block(user) or {}
        return json.dumps({
            "model_name": data.get("model_name", "distilbert-base-uncased"),
            "dataset": data.get("dataset", "sst2"),
            "task_type": "fine_tune",
            "num_workers": data.get("num_workers", 2),
            "cpu_only": True,
            "max_memory_per_worker": "3Gi",
            "epochs": data.get("epochs", 1),
            "notes": "parsed by stub intake",
        })
    if "planner" in role:
        data = _extract_json_block(user) or {}
        spec = {
            "model_name": data.get("model_name", "distilbert-base-uncased"),
            "dataset": data.get("dataset", "sst2"),
            "backend": "gloo",
            "per_worker_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "num_workers": data.get("num_workers", 2),
            "resources_per_worker_memory": "2Gi",
            "epochs": data.get("epochs", 1),
            "max_steps": 40,
            "learning_rate": 2e-5,
        }
        return json.dumps(spec)
    if "critic" in role or "diagnos" in role:
        return json.dumps(diagnose_log_text(user))
    if "reporter" in role or "report" in role:
        return user.strip().splitlines()[0][:200] if user.strip() else "run finished"
    # Generic fallback: echo a minimal structured answer.
    return json.dumps({"answer": "stub response", "echo": user[:120]})


def diagnose_log_text(text: str) -> dict[str, Any]:
    """Rule-based log diagnosis shared by the stub LLM and the mock tools.

    Recognizes the fault signatures defined in section C of the design.
    """
    lowered = text.lower()
    if "nccl" in lowered or "cuda" in lowered and "unavailable" in lowered \
            or "no cuda" in lowered or "gpu-only" in lowered:
        return {
            "symptom": "training crashed during distributed init",
            "root_cause": "GPU-only communication backend (nccl/CUDA) used on a CPU-only cluster",
            "evidence": ["log line mentions nccl/cuda on a cpu-only run"],
            "error_class": "CODE_BUG",
            "proposed_fix": {"backend": "gloo"},
            "confidence": 0.9,
        }
    if "oomkilled" in lowered or "exit code 137" in lowered or "out of memory" in lowered:
        return {
            "symptom": "worker killed (OOMKilled, exit 137)",
            "root_cause": "per-worker batch too large for the memory limit",
            "evidence": ["log line reports OOMKilled / exit code 137"],
            "error_class": "OOM",
            "proposed_fix": {"per_worker_batch_size": 4,
                             "gradient_accumulation_steps": 4},
            "confidence": 0.9,
        }
    if "modulenotfounderror" in lowered or "importerror" in lowered \
            or "incompatible" in lowered or "abi" in lowered:
        return {
            "symptom": "worker failed at import time",
            "root_cause": "dependency version conflict in the training image",
            "evidence": ["log line reports an import/ABI error"],
            "error_class": "DEPENDENCY",
            "proposed_fix": {"rebuild_image_with_lockfile": True},
            "confidence": 0.8,
        }
    if "label" in lowered and ("invalid" in lowered or "out of range" in lowered):
        return {
            "symptom": "training crashed on a data batch",
            "root_cause": "invalid label value in the dataset split",
            "evidence": ["log line reports an invalid label"],
            "error_class": "DATA_SCHEMA",
            "proposed_fix": {"sanitize_labels": True},
            "confidence": 0.8,
        }
    if "nan" in lowered and "loss" in lowered:
        return {
            "symptom": "training loss became NaN",
            "root_cause": "learning rate far too high for this model",
            "evidence": ["log line reports loss=nan"],
            "error_class": "DATA_SCHEMA",
            "proposed_fix": {"learning_rate": 2e-5},
            "confidence": 0.7,
        }
    return {
        "symptom": "run failed without a recognized signature",
        "root_cause": "unknown",
        "evidence": [],
        "error_class": "UNKNOWN",
        "proposed_fix": {},
        "confidence": 0.3,
    }
