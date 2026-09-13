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
    # keep the recovered-idea slab (diagnostic; training carries only sums)
    m._recon_keep_ideas = True
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
    assert len(out) == 21
    ideas, cost, truncated = m._recon_ideas, m._recon_cost, m._recon_truncated
    idea_cost = m._recon_idea_cost
    assert bool(torch.isfinite(idea_cost).all()) and idea_cost.shape == (2,)
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
    _ideas, _idea_cost, _byte_cost, truncated = m._reconstruct_sentence_traversal(S, pushed)
    assert bool(truncated.all())
    m.End(); m.symbolSpace.soft_reset()


def test_byte_fidelity_is_zero_for_the_references_and_positive_when_swapped(tmp_path):
    """The per-word byte cost scores a recovered idea through the
    sentence's own word rows: each retained reference scores (near) zero
    at its own word; a reference at another word's position scores
    positive."""
    m = _traversal_model(tmp_path)
    _run(m, ["12 plus 1"])
    isp = m.inputSpace
    active = isp._word_active_mask
    reference = m._tensor_pushed_ideas                    # the reference slab
    B, W = int(reference.shape[0]), int(reference.shape[1])
    ready, bytes_bwp, valid_bwp = m._byte_tables(B, W)
    assert ready
    ref_n = torch.nn.functional.normalize(reference, dim=-1)
    n_active = int(active[0].sum())
    own, swapped = [], []
    for w in range(n_active):
        idx = torch.tensor(w)
        own.append(float(m._byte_word_cost(
            reference[:, w], idx, ref_n, active, ready, bytes_bwp, valid_bwp)[0]))
        other = (w + 3) % n_active                        # "1" <-> "plus" and the like
        swapped.append(float(m._byte_word_cost(
            reference[:, other], idx, ref_n, active, ready, bytes_bwp, valid_bwp)[0]))
    assert max(own) < 1e-3
    assert min(swapped) > max(own) + 0.5
    m.End(); m.symbolSpace.soft_reset()


def _stage_packed(m, rows):
    """Stage packed rows the way runEpoch's packed cursor does."""
    m._start_spaces_for_forward()
    raw = m.inputSpace.prepPackedInput(rows)
    m._staged_in_sub = m._lex_embed_stem(raw)
    symbol = m.symbolSpace
    if not getattr(symbol, "_per_sentence_initialized", False):
        symbol.soft_reset()
        symbol._per_sentence_initialized = True
    m._stage_reconstruction_teacher()
    slab = m.inputSpace._ar_embedded_N
    m._prepare_reconstruction_choices(int(slab.shape[0]), int(slab.shape[1]), slab.device)
    m.conceptualSpace.stm.begin_forward(int(slab.shape[0]), device=slab.device, dtype=slab.dtype)
    m._stage_fixed_residual_part_capacity()
    m._stage_intersentence_seed()
    return raw


def test_packed_rows_reconstruct_each_sentence_separately(tmp_path):
    """Requirement 3: one traversal per completed sentence; packed rows keep
    separate per-sentence costs; every word of every sentence recovered."""
    m = _build_ladder_variant(tmp_path, "recon16", [
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>"),
        ("<packSentences>false</packSentences>",
         "<packSentences>false</packSentences>\n      <reconstructInLoop>true</reconstructInLoop>")])
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    m._recon_keep_ideas = True
    m._install_unit_span_fn()
    _stage_packed(m, [["12 plus 1", "3 plus 4"], ["ab cd"]])
    with torch.no_grad():
        out = m._forward_with_compiled_sentence_state(None)
    m._publish_compiled_sentence_state(out)
    isp = m.inputSpace
    ids = isp._packed_sentence_ids
    active = isp._word_active_mask
    assert int(ids[0].max()) == 1 and int(ids[1].max()) == 0
    costs = m._recon_sentence_costs                      # [B, slots]
    assert bool(torch.isfinite(costs).all())
    assert costs.shape[0] == 2 and costs.shape[1] >= 2
    assert not bool(m._recon_truncated.any())
    norms = m._recon_ideas.norm(dim=-1)
    assert bool((norms[active] > 0).all()) and bool((norms[~active] == 0).all())
    # Row 1 has one sentence: its second slot carries no cost.
    assert float(costs[1, 1]) == 0.0
    eager_ideas, eager_costs = m._recon_ideas.clone(), costs.clone()
    m.End(); m.symbolSpace.soft_reset()
    # The outer sentence loop nests inside the compiled sentence graph.
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(
        lambda _u: m._forward_with_compiled_sentence_state(None),
        backend="eager", fullgraph=True)
    try:
        _stage_packed(m, [["12 plus 1", "3 plus 4"], ["ab cd"]])
        with torch.no_grad():
            out = compiled(None)
        m._publish_compiled_sentence_state(out)
        assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) == 1
        assert torch.allclose(m._recon_sentence_costs, eager_costs, atol=1e-4)
        assert torch.allclose(m._recon_ideas, eager_ideas, atol=1e-4)
    finally:
        torch._dynamo.reset()
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


def test_tied_traversal_trains_the_fold_parameters_and_owns_none(tmp_path):
    """Tying gate: the reconstruction cost's gradient reaches the compose
    path's fold parameters (the lift/lower inner layers, through their own
    inverses) and the model owns no reverse-student parameter under the
    tied contract; the recorded choices are indices (no gradient path)."""
    m = _traversal_model(tmp_path)
    assert not any(".reverse_chooser." in n or n.startswith("reverse_chooser.")
                   for n, _ in m.named_parameters())
    _stage_fullgraph_tensor_peer(m, ["12 plus 1", "3 plus 4"])
    m.zero_grad(set_to_none=True)
    out = m._forward_with_compiled_sentence_state(None)
    m._publish_compiled_sentence_state(out)
    cost = m._recon_cost
    assert cost.requires_grad and bool(torch.isfinite(cost).all())
    cost.mean().backward()
    language = m.languageSpace
    binary = language._tree_layer(2)
    names = list(binary.op_names)
    touched = 0
    for name in ("lift", "lower"):
        if name not in names:
            continue
        op = list(binary.ops)[names.index(name)]
        gl = getattr(op, "gl", op)
        inner = getattr(gl, "_sigma", None) or getattr(gl, "_pi", None)
        grads = [p.grad for p in inner.layer.parameters() if p.grad is not None]
        assert grads, f"{name}: no gradient reached the tied inverse's weights"
        touched += sum(int(g.abs().sum() > 0) for g in grads)
    assert touched > 0
    trace = m._reconstruction_stack()
    assert not trace._choice_rule_ids.requires_grad
    m.End(); m.symbolSpace.soft_reset()


def test_evaluation_decode_unwinds_from_the_end_state(tmp_path):
    """The retired trace replay's role: in evaluation, reverseReconstruct
    takes the sentence's end state and unwinds the recorded derivation
    into per-word ideas (``_recovered_word_ideas``)."""
    m = _traversal_model(tmp_path)
    _run(m, ["12 plus 1", "3 plus 4"])
    with torch.no_grad():
        recovered = m._recovered_word_ideas(m._stm_single_S)
    active = m.inputSpace._word_active_mask
    assert recovered is not None and tuple(recovered.shape[:2]) == tuple(active.shape)
    assert bool((recovered.norm(dim=-1)[active] > 0).all())
    m.End(); m.symbolSpace.soft_reset()


def test_final_end_state_keeps_three_slots_for_a_relative_row(tmp_path):
    """A relative sentence stops at depth 3: the traversal starts from
    all three slots (newest at 0), not from the root alone."""
    m = _traversal_model(tmp_path)
    _run(m, ["12 plus 1", "3 plus 4"])
    stm = m.conceptualSpace.stm
    B, D = 2, int(stm.concept_dim)
    buf = torch.randn(B, int(stm.capacity), D)
    object.__setattr__(stm, "_live_buffer", buf)
    S = buf[:, 2, :].clone()                          # the oldest of three
    slots, depth = m._final_end_state(S, torch.tensor([3, 1]))
    assert tuple(slots.shape) == (B, 3, D) and depth.tolist() == [3, 1]
    assert torch.equal(slots[0], buf[0, :3])           # row 0: three slots
    assert torch.equal(slots[1, 0], S[1]) and float(slots[1, 1:].abs().sum()) == 0.0
    m.End(); m.symbolSpace.soft_reset()
