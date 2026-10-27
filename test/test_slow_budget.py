"""Optional work reduction preserves single-shot and comparison contracts."""
import pytest

from slow_budget import slow_count, work_fraction


def test_default_suite_keeps_full_work(monkeypatch):
    monkeypatch.delenv('RUN_SLOW', raising=False)
    monkeypatch.setenv('RUN_SLOW_FRACTION', '0.5')
    assert slow_count(30) == 30


def test_slow_default_halves_repetitions_and_keeps_single_shots(monkeypatch):
    monkeypatch.setenv('RUN_SLOW', '1')
    monkeypatch.delenv('RUN_SLOW_FRACTION', raising=False)
    assert work_fraction() == 0.5
    assert slow_count(160) == 80
    assert slow_count(32) == 16
    assert slow_count(1) == 1
    assert slow_count(3) == 2


def test_full_budget_is_explicitly_recoverable(monkeypatch):
    monkeypatch.setenv('RUN_SLOW', '1')
    monkeypatch.setenv('RUN_SLOW_FRACTION', '1')
    assert slow_count(160) == 160
    assert slow_count(32) == 32


def test_two_disjoint_comparison_windows_remain_available(monkeypatch):
    monkeypatch.setenv('RUN_SLOW', '1')
    monkeypatch.setenv('RUN_SLOW_FRACTION', '0.1')
    history = list(range(slow_count(30, minimum=10)))
    assert len(history[:5]) == len(history[-5:]) == 5
    assert set(history[:5]).isdisjoint(history[-5:])


@pytest.mark.parametrize('fraction', ['0', '-1', '1.1', 'nan', 'inf', 'invalid'])
def test_invalid_fraction_fails_before_training(monkeypatch, fraction):
    monkeypatch.setenv('RUN_SLOW', '1')
    monkeypatch.setenv('RUN_SLOW_FRACTION', fraction)
    with pytest.raises(ValueError, match='RUN_SLOW_FRACTION'):
        slow_count(30)
