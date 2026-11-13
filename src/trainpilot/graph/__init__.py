"""Section A: the LangGraph agent core of TrainPilot-Local.

Public API (docs/CONTRACTS.md): ``build_agent_graph``, ``run_agent``
and ``StubExecutor``.
"""
from .executors import StubExecutor
from .graph import build_agent_graph, run_agent

__all__ = ["build_agent_graph", "run_agent", "StubExecutor"]
