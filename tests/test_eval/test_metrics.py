"""pass@k / pass^k on known vectors (professor's Week 6 worked example)."""
import pytest

from eval.metrics import flipping_tasks, mean, p95, pass_at_k_stats, pass_at_k_value


def test_professor_example_four_runs():
    # Outcomes [1, 1, 0, 0], k = 2: pass@2 = 5/6, pass^2 = 1/6.
    at, pow_ = pass_at_k_value([1, 1, 0, 0], 2)
    assert at == pytest.approx(5 / 6)
    assert pow_ == pytest.approx(1 / 6)


def test_professor_example_via_stats():
    stats = pass_at_k_stats([[1, 1, 0, 0]])
    assert stats["per_k"][2]["pass@k"] == pytest.approx(5 / 6)
    assert stats["per_k"][2]["pass^k"] == pytest.approx(1 / 6)
    assert stats["pass@1"] == pytest.approx(0.5)
    assert stats["pass^1"] == pytest.approx(0.5)


def test_three_repeat_case_reduces_to_any_all():
    stats = pass_at_k_stats([[1, 0, 1], [1, 1, 1], [0, 0, 0]])
    # pass@3: tasks passing at least once -> 2/3; pass^3: all runs -> 1/3.
    assert stats["pass@3"] == pytest.approx(2 / 3)
    assert stats["pass^3"] == pytest.approx(1 / 3)


def test_flipping_tasks():
    assert flipping_tasks({"a": [1, 0, 1], "b": [1, 1, 1],
                           "c": [0, 0, 0], "d": []}) == ["a"]


def test_mean_and_p95():
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0
    assert p95(list(range(100))) == 94
    assert p95([]) == 0.0
