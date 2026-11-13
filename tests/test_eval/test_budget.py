"""Budget tracker: the 41st tool call is refused, caps hold (D.2.4)."""
from eval.budget import BudgetTracker


def test_first_40_calls_allowed_41st_refused():
    tracker = BudgetTracker()
    for i in range(40):
        assert tracker.charge_tool_call("get_rayjob_logs") is None, i
    result = tracker.charge_tool_call("get_rayjob_logs")
    assert result is not None
    assert result.ok is False
    assert result.error_code == "BUDGET_EXCEEDED_TOOL_CALLS"
    assert tracker.exceeded() == "tool_calls"


def test_token_cap():
    tracker = BudgetTracker()
    tracker.charge_tokens(120_000)
    assert tracker.exceeded() is None
    tracker.charge_tokens(1)
    assert tracker.exceeded() == "tokens"


def test_warning_at_eighty_percent():
    tracker = BudgetTracker()
    assert tracker.warning() is None
    for _ in range(32):  # exactly 80% of 40
        tracker.charge_tool_call("get_rayjob_logs")
    assert tracker.warning() is not None


def test_snapshot_shape():
    tracker = BudgetTracker()
    snap = tracker.snapshot()
    assert snap["max_tool_calls"] == 40
    assert snap["max_total_tokens"] == 120_000
    assert snap["max_wall_seconds"] == 1500.0
    assert snap["exceeded"] is None
