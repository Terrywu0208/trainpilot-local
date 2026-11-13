"""Cost accounting (design D.8).

Costs are recomputed offline from `llm_calls.jsonl` rows and the versioned
price table in `configs/prices.yaml`, so any run's cost can be re-audited
without re-running anything. The headline definitions are the honest ones:
cost per task divides total spend (failed runs included) by the number of
tasks; cost per successful task divides the same total spend by the number
of passing runs only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICES = REPO_ROOT / "configs" / "prices.yaml"


def load_prices(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_PRICES
    return yaml.safe_load(p.read_text())


def _model_key(model: str, table: dict[str, Any]) -> Optional[str]:
    models = table.get("models", {})
    if model in models:
        return model
    for key in models:  # pinned snapshot -> family key, e.g. gpt-5-mini-2025-08-07
        if model.startswith(key):
            return key
    return None


def call_cost(row: dict[str, Any], table: dict[str, Any]) -> float:
    key = _model_key(str(row.get("model", "")), table)
    if key is None:
        return 0.0
    price = table["models"][key]
    prompt = int(row.get("prompt_tokens", 0))
    cached = min(int(row.get("cached_tokens", 0)), prompt)
    fresh = prompt - cached
    completion = int(row.get("completion_tokens", 0))
    return (
        fresh * float(price.get("input_per_1m", 0.0))
        + cached * float(price.get("cached_input_per_1m", 0.0))
        + completion * float(price.get("output_per_1m", 0.0))
    ) / 1_000_000.0


def calls_cost(rows: Iterable[dict[str, Any]], table: dict[str, Any]) -> float:
    return sum(call_cost(r, table) for r in rows)


def load_llm_calls(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "llm_calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_token_ledger(run_dir: str | Path, rows: list[dict[str, Any]],
                       table: dict[str, Any]) -> dict[str, Any]:
    """Write token_ledger.json for one run directory and return it."""
    total_tokens = sum(
        int(r.get("prompt_tokens", 0)) + int(r.get("completion_tokens", 0))
        for r in rows
    )
    ledger = {
        "total_tokens": total_tokens,
        "api_cost_usd": round(calls_cost(rows, table), 6),
        "price_table_version": table.get("price_table_version", "unknown"),
        "n_llm_calls": len(rows),
    }
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "token_ledger.json").write_text(json.dumps(ledger, indent=2))
    return ledger


def rollup_costs(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-run costs up to task-level figures.

    Each record: {cost: float, passed: bool}. Cost per task counts failed
    runs (total / number of records). Cost per successful task is total /
    passing records (None when nothing passed). A third figure, the mean
    cost of passing runs only, is reported alongside so the difference
    between the two definitions is explicit (design D.8).
    """
    total = sum(float(r.get("cost", 0.0)) for r in run_records)
    n = len(run_records)
    passed = [r for r in run_records if r.get("passed")]
    return {
        "total_cost_usd": round(total, 6),
        "n_runs": n,
        "n_passed": len(passed),
        "cost_per_task_usd": round(total / n, 6) if n else 0.0,
        "cost_per_successful_task_usd": (
            round(total / len(passed), 6) if passed else None
        ),
        "mean_cost_passing_runs_only_usd": (
            round(sum(float(r["cost"]) for r in passed) / len(passed), 6)
            if passed else None
        ),
    }
