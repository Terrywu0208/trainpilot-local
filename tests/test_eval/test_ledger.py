"""Cost accounting: price-table math and the honest rollups (D.8)."""
import json

from eval.ledger import call_cost, load_prices, rollup_costs, write_token_ledger

PRICES = load_prices()


def test_call_cost_full_price():
    row = {"model": "gpt-5-mini-2025-08-07", "prompt_tokens": 1_000_000,
           "completion_tokens": 1_000_000, "cached_tokens": 0}
    # 1M fresh input @ 0.25 + 1M output @ 2.00 per 1M.
    assert call_cost(row, PRICES) == 2.25


def test_call_cost_cached_input():
    row = {"model": "gpt-5-mini-2025-08-07", "prompt_tokens": 1_000_000,
           "completion_tokens": 0, "cached_tokens": 1_000_000}
    assert call_cost(row, PRICES) == 0.025


def test_stub_model_is_free():
    row = {"model": "stub", "prompt_tokens": 10**6, "completion_tokens": 10**6}
    assert call_cost(row, PRICES) == 0.0


def test_rollup_counts_failed_runs_in_cost_per_task():
    out = rollup_costs([{"cost": 3.0, "passed": True},
                        {"cost": 1.0, "passed": False}])
    assert out["cost_per_task_usd"] == 2.0            # total 4 / 2 runs
    assert out["cost_per_successful_task_usd"] == 4.0  # total 4 / 1 success
    assert out["mean_cost_passing_runs_only_usd"] == 3.0


def test_rollup_no_successes():
    out = rollup_costs([{"cost": 1.0, "passed": False}])
    assert out["cost_per_successful_task_usd"] is None
    assert out["cost_per_task_usd"] == 1.0


def test_write_token_ledger(tmp_path):
    rows = [{"model": "gpt-5-mini-2025-08-07", "prompt_tokens": 100,
             "completion_tokens": 50, "cached_tokens": 0}]
    ledger = write_token_ledger(tmp_path, rows, PRICES)
    assert ledger["total_tokens"] == 150
    assert ledger["n_llm_calls"] == 1
    on_disk = json.loads((tmp_path / "token_ledger.json").read_text())
    assert on_disk["price_table_version"] == PRICES["price_table_version"]
