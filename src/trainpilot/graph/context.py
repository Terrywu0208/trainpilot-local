"""Run context shared by the graph nodes.

Nodes are closures over an ``AgentContext`` so the compiled graph stays
a plain LangGraph object while the nodes can still reach the LLM client,
the executor, the approval handler, the run directory and the event
sink. Events use the names the webapp and the replay view consume:
node_started, node_finished, attempt_failed, diagnosis_ready,
approval_required, verification_done.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ..llm import LLMClient
from ..schemas import ApprovalDecision, ProposedAction


def default_approval_handler(action: ProposedAction) -> ApprovalDecision:
    """Safe default when no handler is supplied: approve the first launch
    only; deny every irreversible action and every later launch."""
    if action.irreversible:
        return ApprovalDecision(
            approve=False, reason_code="DEFAULT_DENY_IRREVERSIBLE",
            decided_by="scripted_user")
    if action.action_type == "launch" and int(action.args.get("attempt", 0)) == 0:
        return ApprovalDecision(
            approve=True, reason_code="DEFAULT_APPROVE_FIRST_LAUNCH",
            decided_by="scripted_user")
    return ApprovalDecision(
        approve=False, reason_code="DEFAULT_DENY_NOT_FIRST_LAUNCH",
        decided_by="scripted_user")


@dataclass
class AgentContext:
    llm: LLMClient
    executor: Any
    approval_handler: Optional[Callable[[ProposedAction], ApprovalDecision]] = None
    run_dir: Optional[Path] = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: list[dict[str, Any]] = field(default_factory=list)
    started_monotonic: float = field(default_factory=time.monotonic)

    def handler(self) -> Callable[[ProposedAction], ApprovalDecision]:
        return self.approval_handler or default_approval_handler

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        })

    def attempt_dir(self, attempt: int) -> Optional[Path]:
        if self.run_dir is None:
            return None
        return self.run_dir / "attempts" / f"attempt_{attempt:02d}"
