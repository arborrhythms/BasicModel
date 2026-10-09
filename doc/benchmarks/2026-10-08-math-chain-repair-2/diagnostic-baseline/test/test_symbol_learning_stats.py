"""Tests for the symbol-learning Layer's stats half.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 / step 6.

Disabled by default. When enabled, collects DETACHED statistics from
two hook points:
  * zero-order / QE — after conceptual activation forms, before snap
  * higher-order / PMI — after an STM reduce succeeds

Does NOT call ``extend_artifact`` during forward. That happens at an
explicit flush boundary so codebook mutations stay out of autograd.

Migrated from the retired ``symbol_learning`` module to
``Layers.SymbolLearningLayer`` (2026-05-21 SymbolSubSpace / STM Layer
refactor).
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def test_default_disabled():
    """``SymbolLearningLayer()`` defaults to ``enabled=False``."""
    from Layers import SymbolLearningLayer
    sls = SymbolLearningLayer()
    assert sls.enabled is False


def test_observe_qe_noop_when_disabled():
    """When disabled, ``observe_qe`` does not accumulate statistics."""
    from Layers import SymbolLearningLayer
    import torch
    sls = SymbolLearningLayer(enabled=False)
    sls.observe_qe(activation=torch.tensor([0.5, 0.3]),
                   snapped=torch.tensor([0.5, 0.0]))
    assert sls.qe_count == 0
    assert sls.qe_sum_squared == 0.0


def test_observe_qe_accumulates_when_enabled():
    """When enabled, ``observe_qe`` accumulates squared-error running
    totals as detached scalars."""
    from Layers import SymbolLearningLayer
    import torch
    sls = SymbolLearningLayer(enabled=True)
    sls.observe_qe(activation=torch.tensor([0.5, 0.3]),
                   snapped=torch.tensor([0.5, 0.0]))
    # |[0,0.3]|^2 = 0.09
    assert sls.qe_count == 1
    import pytest
    assert sls.qe_sum_squared == pytest.approx(0.09, abs=1e-5)


def test_observe_qe_stats_are_detached():
    """The accumulator must not retain gradient history (symbol learning
    runs outside autograd by contract)."""
    from Layers import SymbolLearningLayer
    import torch
    sls = SymbolLearningLayer(enabled=True)
    act = torch.tensor([0.5, 0.3], requires_grad=True)
    snap = torch.tensor([0.5, 0.0])
    sls.observe_qe(activation=act, snapped=snap)
    # qe_sum_squared is a Python float, no autograd tape attached
    assert isinstance(sls.qe_sum_squared, float)


def test_observe_reduce_noop_when_disabled():
    """When disabled, ``observe_reduce`` does not accumulate adjacency
    counts."""
    from Layers import SymbolLearningLayer
    sls = SymbolLearningLayer(enabled=False)
    sls.observe_reduce(left_ref=1, right_ref=2, parent_ref=10)
    assert sls.pair_counts == {}


def test_observe_reduce_accumulates_when_enabled():
    """When enabled, ``observe_reduce`` accumulates ``(left, right) -> count``
    pair counts (for PMI computation later)."""
    from Layers import SymbolLearningLayer
    sls = SymbolLearningLayer(enabled=True)
    sls.observe_reduce(left_ref=1, right_ref=2, parent_ref=10)
    sls.observe_reduce(left_ref=1, right_ref=2, parent_ref=10)
    sls.observe_reduce(left_ref=3, right_ref=4, parent_ref=11)
    assert sls.pair_counts[(1, 2)] == 2
    assert sls.pair_counts[(3, 4)] == 1


def test_flush_returns_snapshot_and_clears():
    """``flush()`` returns the accumulated stats and resets internal
    state, so a subsequent flush returns fresh counts."""
    from Layers import SymbolLearningLayer
    import torch
    sls = SymbolLearningLayer(enabled=True)
    sls.observe_qe(activation=torch.tensor([0.5, 0.3]),
                   snapped=torch.tensor([0.5, 0.0]))
    sls.observe_reduce(left_ref=1, right_ref=2, parent_ref=10)
    snap = sls.flush()
    assert snap['qe_count'] == 1
    assert (1, 2) in snap['pair_counts']
    # State cleared
    assert sls.qe_count == 0
    assert sls.pair_counts == {}


def test_xml_config_default_disabled():
    """When no ``<symbolLearning>`` section is present in XML config, the
    helper reads ``enabled=False``."""
    from Layers import SymbolLearningLayer
    assert SymbolLearningLayer.enabled_from_config() is False


def test_xml_config_enabled_flag_round_trip(tmp_path):
    """When XML contains ``<symbolLearning enabled="true"/>``, the helper
    returns True."""
    from Layers import SymbolLearningLayer
    import util
    import tempfile
    import textwrap
    xml = textwrap.dedent("""\
        <model>
          <architecture>
            <symbolLearning enabled="true"/>
          </architecture>
        </model>
        """)
    with tempfile.NamedTemporaryFile(
            "w", suffix=".xml", delete=False) as f:
        f.write(xml)
        path = f.name
    saved_root = getattr(util.TheXMLConfig, "_root", None)
    try:
        util.TheXMLConfig.load(path)
        assert SymbolLearningLayer.enabled_from_config() is True
    finally:
        if saved_root is not None:
            util.TheXMLConfig._root = saved_root
