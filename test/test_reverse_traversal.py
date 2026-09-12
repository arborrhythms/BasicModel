"""Compiled reverse-loops plan, slice 1: the bounded tied traversal of the
recorded derivation (doc/plans/2026-09-12-compiled-reverse-loops.md)."""
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data"
for _p in (str(_ROOT / "bin"), str(_ROOT / "test")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_meronomy_ladder import _build_ladder_variant  # noqa: E402
from test_compiled_word_chunk import _stage_fullgraph_tensor_peer  # noqa: E402


def _traversal_model(tmp_path):
    m = _build_ladder_variant(tmp_path, "recon", [
        ("<packSentences>false</packSentences>",
         "<packSentences>false</packSentences>\n      <reconstructInLoop>true</reconstructInLoop>")])
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    return m


def _run(m, samples, fn=None):
    _stage_fullgraph_tensor_peer(m, samples)
    with torch.no_grad():
        out = (fn or m._forward_with_compiled_sentence_state)(None)
    m._publish_compiled_sentence_state(out)
    return out


def test_traversal_recovers_every_active_word_without_underflow(tmp_path):
    m = _traversal_model(tmp_path)
    assert m.reconstruct_in_loop
    out = _run(m, ["12 plus 1", "3 plus 4"])
    assert len(out) == 14
    ideas, cost, truncated = m._recon_ideas, m._recon_cost, m._recon_truncated
    active = m.inputSpace._word_active_mask
    assert tuple(ideas.shape) == (*active.shape, int(m.conceptualSpace.stm.concept_dim))
    assert bool(torch.isfinite(cost).all()) and cost.shape == (2,)
    assert not bool(truncated.any())
    # Every active word received a recovered idea; inactive columns stay zero.
    norms = ideas.norm(dim=-1)
    assert bool((norms[active] > 0).all())
    assert bool((norms[~active] == 0).all())
    m.End(); m.symbolSpace.soft_reset()


def test_tied_binary_reverse_recomposes_the_parent(tmp_path):
    """The lift/lower reverses are the compose weights' own inverses."""
    m = _traversal_model(tmp_path)
    language = m.languageSpace
    binary = language._tree_layer(2)
    names = list(binary.op_names)
    D = int(m.conceptualSpace.stm.concept_dim)
    torch.manual_seed(0)
    for name in ("lift", "lower"):
        if name not in names:
            continue
        idx = names.index(name)
        op = list(binary.ops)[idx]
        op = getattr(op, "gl", op)
        parent = torch.tanh(torch.randn(3, D) * 0.5)
        left, right = language.reverse_binary_step(
            parent, torch.full((3,), idx), torch.ones(3, dtype=torch.bool))
        recomposed = op.compose(left, right)
        cw = int(getattr(op, "_content_width", 0) or D)
        assert torch.allclose(recomposed[:, :cw], parent[:, :cw], atol=1e-3), name
    # An invalid row is left untouched.
    parent = torch.tanh(torch.randn(2, D))
    left, right = language.reverse_binary_step(
        parent, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.bool))
    assert torch.equal(left, parent) and torch.equal(right, parent)


def test_chunk_reverse_is_the_exact_residual_against_the_reference(tmp_path):
    m = _traversal_model(tmp_path)
    language = m.languageSpace
    binary = language._tree_layer(2)
    names = list(binary.op_names)
    if "chunk" not in names:
        pytest.skip("grammar without chunk")
    idx = names.index("chunk")
    D = int(m.conceptualSpace.stm.concept_dim)
    a = torch.randn(2, D); b = torch.randn(2, D)
    left, right = language.reverse_binary_step(
        a + b, torch.full((2,), idx), torch.ones(2, dtype=torch.bool), reference=b)
    assert torch.allclose(right, b) and torch.allclose(left, a, atol=1e-6)


def test_cleared_trace_reports_truncation(tmp_path):
    m = _traversal_model(tmp_path)
    _run(m, ["12 plus 1"])
    assert not bool(m._recon_truncated.any())
    trace = m._reconstruction_stack()
    with torch.no_grad():
        trace._choice_mask.zero_()
    S = m._stm_single_S
    pushed = torch.zeros_like(m._recon_ideas)
    _ideas, _cost, truncated = m._reconstruct_sentence_traversal(S, pushed)
    assert bool(truncated.all())
    m.End(); m.symbolSpace.soft_reset()


def test_traversal_compiles_into_the_one_sentence_graph(tmp_path):
    m = _traversal_model(tmp_path)
    eager = _run(m, ["12 plus 1", "3 plus 4"])
    eager_ideas, eager_cost = m._recon_ideas.clone(), m._recon_cost.clone()
    m.End(); m.symbolSpace.soft_reset()
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(
        lambda _u: m._forward_with_compiled_sentence_state(None),
        backend="eager", fullgraph=True)
    try:
        _run(m, ["12 plus 1", "3 plus 4"], fn=compiled)
        assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) == 1
        assert torch.allclose(m._recon_cost, eager_cost, atol=1e-4)
        assert torch.allclose(m._recon_ideas, eager_ideas, atol=1e-4)
    finally:
        torch._dynamo.reset()
        m.End(); m.symbolSpace.soft_reset()
