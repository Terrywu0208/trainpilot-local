"""The single LLM entry point (design D.8).

Re-exports the shared `trainpilot.llm.LLMClient` so every system in the
eval layer imports the same client, and provides the ledger writer that
persists a client's recorded calls to `llm_calls.jsonl` in a run directory.
No system may call a model API except through this client.
"""
from __future__ import annotations

import json
from pathlib import Path

from trainpilot.llm import LLMClient, LLMResponse

__all__ = ["LLMClient", "LLMResponse", "write_llm_ledger"]


def write_llm_ledger(client: LLMClient, path: str | Path) -> Path:
    """Write the client's in-memory call ledger as JSONL and return the path.

    Each line is one call: model, effort, prompt_tokens, completion_tokens,
    cached_tokens (0 when the provider does not report it), latency_ms, ts
    (a per-ledger sequence number; wall timestamps are not needed to audit
    token sums and keep test output deterministic).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, entry in enumerate(client.ledger):
        row = {
            "model": entry.get("model", ""),
            "effort": entry.get("effort", ""),
            "prompt_tokens": int(entry.get("prompt_tokens", 0)),
            "completion_tokens": int(entry.get("completion_tokens", 0)),
            "cached_tokens": int(entry.get("cached_tokens", 0)),
            "latency_ms": float(entry.get("latency_ms", 0.0)),
            "ts": i,
        }
        lines.append(json.dumps(row, sort_keys=True))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path
