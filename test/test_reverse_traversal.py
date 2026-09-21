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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_byte_fidelity_is_zero_on_the_row_and_positive_off_it(tmp_path):
    """The byte decoding snaps a recovered idea to the brick's dictionary
    snapshot: a word's own row scores (near) zero at its position; another
    word's row at that position scores positive."""
    m = _traversal_model(tmp_path)
    _run(m, ["12 plus 1"])
    isp = m.inputSpace
    active = isp._word_active_mask
    reference = m._tensor_pushed_ideas
    B, W = int(reference.shape[0]), int(reference.shape[1])
    ready, bytes_bwp, valid_bwp = m._byte_tables(B, W)
    snap, bank_n, bank_bytes, bank_valid = m._snapshot_tables(reference)
    assert ready and snap and int(bank_n.shape[1]) >= W
    n_active = int(active[0].sum())
    own, swapped = [], []
    for w in range(n_active):
        idx = torch.tensor(w)
        own.append(float(m._byte_word_cost(
            reference[:, w], idx, bank_n, bank_bytes, bank_valid, bytes_bwp, valid_bwp, True)[0]))
        other = (w + 3) % n_active
        swapped.append(float(m._byte_word_cost(
            reference[:, other], idx, bank_n, bank_bytes, bank_valid, bytes_bwp, valid_bwp, True)[0]))
    assert max(own) < 1e-2
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
    return raw


def _replay_operand_rows(m):
    """Replay the real trace's row stack, newest first, before each fold."""
    trace = m._reconstruction_stack()
    _rules, arities, mask = trace.choices()
    active = m.inputSpace._word_active_mask
    ids = m.inputSpace._packed_sentence_ids
    rows = m._word_symbol_rows()
    width = int(active.shape[1])
    seal_width = int(m.conceptualSpace.stm.capacity) - 1
    expected = {}
    for b in range(int(active.shape[0])):
        words = active[b].nonzero().flatten().tolist()
        for sid in sorted(set(ids[b, words].tolist())):
            sentence = [w for w in words if int(ids[b, w]) == sid]
            stack = []

            def fold(slot):
                if not bool(mask[b, slot]):
                    return
                if int(arities[b, slot]) == 2:
                    assert len(stack) >= 2, (b, slot, stack)
                    expected[b, slot] = (stack[1], stack[0])
                    stack[:2] = [-1]
                else:
                    assert int(arities[b, slot]) == 1 and stack
                    stack[0] = -1

            for w in sentence:
                fold(3 * w)
                stack.insert(0, int(rows[b, w]))
                fold(3 * w + 1)
                fold(3 * w + 2)
            hi = sentence[-1]
            base = 3 * width if hi == words[-1] else 3 * width + hi * seal_width
            for k in range(seal_width):
                fold(base + k)
            assert len(stack) == 1, (b, sid, stack)
    return expected


@pytest.mark.slow
def test_packed_trace_records_pre_fold_operand_rows_at_every_binary(tmp_path):
    """Codex item 4: real packed seals retain the operands of the fold,
    including a known leaf beside a composite, before reducing the stack."""
    m = _build_ladder_variant(tmp_path, "operand_rows", [
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>32</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>32</serialWordBuckets>")])
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    m.stm_reduce_tau = 1.0
    m._install_unit_span_fn()
    try:
        _stage_packed(m, [["aa bb cc dd ee", "ff gg hh ii jj"], ["kk ll mm", "nn oo"]])
        with torch.no_grad():
            out = m._forward_with_compiled_sentence_state(None)
        m._publish_compiled_sentence_state(out)
        expected = _replay_operand_rows(m)
        trace = m._reconstruction_stack()
        _rules, arities, mask = trace.choices()
        recorded = set(map(tuple, (mask.bool() & (arities == 2)).nonzero().tolist()))
        assert set(expected) == recorded
        width = int(m.inputSpace._word_active_mask.shape[1])
        seal_width = int(m.conceptualSpace.stm.capacity) - 1
        assert any(slot < 3 * width for _, slot in expected)  # per-word folds
        assert any(3 * width <= slot < 3 * width + seal_width for _, slot in expected)
        intermediate = {key: value for key, value in expected.items()
                        if key[1] >= 3 * width + seal_width}
        assert intermediate and any(min(pair) < 0 <= max(pair)
                                    for pair in intermediate.values())
        for (b, slot), pair in expected.items():
            actual = (int(trace._choice_left_rows[b, slot]),
                      int(trace._choice_right_rows[b, slot]))
            assert actual == pair, (b, slot, actual, pair)
    finally:
        m.End()
        m.symbolSpace.soft_reset()


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_byte_cost_is_positive_for_a_wrong_or_empty_idea_even_with_one_word(tmp_path):
    """The null candidate keeps the byte cost a function of the idea: a
    one-word brick cannot score zero by having nothing to choose between,
    and a zero or wrong idea scores positive with a gradient."""
    m = _traversal_model(tmp_path)
    _run(m, ["ab", "12"])                                   # one unit per row
    reference = m._tensor_pushed_ideas
    B, W = int(reference.shape[0]), int(reference.shape[1])
    ready, bytes_bwp, valid_bwp = m._byte_tables(B, W)
    snap, bank_n, bank_bytes, bank_valid = m._snapshot_tables(reference)
    args = (bank_n, bank_bytes, bank_valid, bytes_bwp, valid_bwp, True)
    own = m._byte_word_cost(reference[:, 0], torch.tensor(0), *args)
    zero = m._byte_word_cost(torch.zeros_like(reference[:, 0]), torch.tensor(0), *args)
    wrong = reference[:, 0].flip(0).clone().requires_grad_(True)   # the other row's word
    wrong_c = m._byte_word_cost(wrong, torch.tensor(0), *args)
    assert float(own.max()) < 1e-2
    # A zero idea is equally near every candidate: its byte distribution
    # is the uniform mixture of the present rows' bytes and the null
    # candidate's uniform bytes, so its cost is exactly the negative log
    # of the target byte's share of that mixture (positive whenever the
    # null candidate is present, even with one surface in the snapshot).
    import math
    expected_rows = []
    for b in range(bank_bytes.shape[0]):
        words = [bank_bytes[b, row][bank_valid[b, row]].tolist() + [0]
                 for row in range(bank_bytes.shape[1]) if bank_valid[b, row].any()]
        target_word = bytes_bwp[b, 0][valid_bwp[b, 0]].tolist() + [0]
        probabilities = [
            (sum(position < len(word) and word[position] == symbol for word in words)
             + 1. / 256) / (len(words) + 1)
            for position, symbol in enumerate(target_word)]
        expected_rows.append(sum(-math.log(p) for p in probabilities) / len(target_word))
    expected = torch.tensor(expected_rows, device=zero.device, dtype=zero.dtype)
    assert torch.allclose(zero, expected, atol=1e-3), (zero.tolist(), expected.tolist())
    assert float(zero.min()) > 0.1 and float(wrong_c.min()) > 0.1
    assert float(wrong_c.min()) > 100.0 * float(own.max())
    assert float(torch.autograd.grad(wrong_c.sum(), [wrong])[0].abs().sum()) > 0
    m.End(); m.symbolSpace.soft_reset()


@pytest.mark.slow
def test_snapshot_rows_absent_from_the_brick_do_not_enter_the_score(tmp_path):
    """Candidates are the staged snapshot's present rows: an absent row,
    however similar to the idea, leaves the word's byte cost unchanged;
    a present row that is nearer changes it."""
    m = _traversal_model(tmp_path)
    _run(m, ["12 plus 1", "3 plus 4"])
    reference = m._tensor_pushed_ideas
    B, W = int(reference.shape[0]), int(reference.shape[1])
    ready, bytes_bwp, valid_bwp = m._byte_tables(B, W)
    snap, bank_n, bank_bytes, bank_valid = m._snapshot_tables(reference)
    idea = reference[:, 1] * 0.5 + reference[:, 3] * 0.5     # between two words
    base = m._byte_word_cost(idea, torch.tensor(1), bank_n, bank_bytes, bank_valid, bytes_bwp, valid_bwp, True)
    absent = bank_valid.clone(); absent[:, 3] = False           # row 3 leaves the snapshot
    off = m._byte_word_cost(idea, torch.tensor(1), bank_n, bank_bytes, absent, bytes_bwp, valid_bwp, True)
    assert not torch.allclose(base, off, atol=1e-4)
    bank2 = bank_n.clone(); bank2[:, 3] = bank_n[:, 1]          # an absent row's content is irrelevant
    same = m._byte_word_cost(idea, torch.tensor(1), bank2, bank_bytes, absent, bytes_bwp, valid_bwp, True)
    assert torch.allclose(off, same, atol=1e-6)
    m.End(); m.symbolSpace.soft_reset()


@pytest.mark.slow
def test_seal_chain_of_chunks_unwinds_to_the_words(tmp_path):
    """Constituent routing (Codex, 2026-09-14): three words a, b, c pushed
    in order and folded by two chunk seals (newest-first: c+b, then
    a+(b+c)) unwind to [a, b, c] at their positions, no truncation."""
    m = _traversal_model(tmp_path)
    _run(m, ["12 plus 1", "3 plus 4"])                      # staging: trace, tables, atoms
    isp = m.inputSpace
    language = m.languageSpace
    names = list(language._tree_layer(2).op_names)
    chunk_id = int(language._cs_binary_rule_ids[names.index("chunk")])
    trace = m._reconstruction_stack()
    active = isp._word_active_mask
    B, W = int(active.shape[0]), int(active.shape[1])
    D = int(m.conceptualSpace.stm.concept_dim)
    torch.manual_seed(3)
    words = torch.randn(B, 3, D)
    a, b, c = words[:, 0], words[:, 1], words[:, 2]
    reference = torch.zeros(B, W, D); reference[:, :3] = words
    with torch.no_grad():
        active.zero_(); active[:, :3] = True
        trace._choice_mask.zero_(); trace._choice_rule_ids.fill_(-1); trace._choice_arities.zero_()
        trace._choice_left_rows.fill_(-1); trace._choice_right_rows.fill_(-1)   # a synthetic trace: no operand rows
        for k in range(2):                                   # the two seals, in the order recorded
            trace._choice_rule_ids[:, 3 * W + k] = chunk_id
            trace._choice_arities[:, 3 * W + k] = 2
            trace._choice_mask[:, 3 * W + k] = True
    S = a + b + c                                            # a + (b + c)
    roots = torch.zeros(B, 1, 3 * D)
    end = torch.cat((S.unsqueeze(1), torch.zeros(B, 2, D)), dim=1)
    m._recon_keep_ideas = True
    rec, idea, byte_c, trunc, _per = m._reconstruct_sentences(
        S, reference, roots, torch.ones(B, 1, dtype=torch.long), end, torch.ones(B, dtype=torch.long))
    assert not bool(trunc.any())
    for w, want in enumerate((a, b, c)):
        assert torch.allclose(rec[:, w], want, atol=1e-4), (w, float((rec[:, w] - want).abs().max()))
    assert float(idea.max()) < 1e-6
    # Compound operand (the reviewer's (a+b)+c): a per-word fold joined a
    # and b when b was pushed, then one seal joined the composite (left)
    # with c (right).  The recorded operand rows route the residuals: the
    # seal's known word is on the right, the per-word fold's on the right.
    rows = torch.arange(3).reshape(1, 3).expand(B, 3).clone() + 100     # symbol rows a, b, c
    object.__setattr__(isp, "_ar_word_concept_rows", torch.full((B, W), -1, dtype=torch.long))
    object.__setattr__(isp, "_ar_word_object_rows", torch.full((B, W), -1, dtype=torch.long))
    isp._ar_word_concept_rows[:, :3] = rows
    with torch.no_grad():
        trace._choice_mask.zero_(); trace._choice_rule_ids.fill_(-1); trace._choice_arities.zero_()
        trace._choice_left_rows.fill_(-1); trace._choice_right_rows.fill_(-1)
        trace._choice_rule_ids[:, 3 * 1 + 1] = chunk_id; trace._choice_arities[:, 3 * 1 + 1] = 2
        trace._choice_mask[:, 3 * 1 + 1] = True
        trace._choice_left_rows[:, 3 * 1 + 1] = rows[:, 0]; trace._choice_right_rows[:, 3 * 1 + 1] = rows[:, 1]
        trace._choice_rule_ids[:, 3 * W] = chunk_id; trace._choice_arities[:, 3 * W] = 2
        trace._choice_mask[:, 3 * W] = True
        trace._choice_left_rows[:, 3 * W] = -1; trace._choice_right_rows[:, 3 * W] = rows[:, 2]
    S2 = (a + b) + c
    end2 = torch.cat((S2.unsqueeze(1), torch.zeros(B, 2, D)), dim=1)
    rec2, idea2, _bc, trunc2, _p = m._reconstruct_sentences(
        S2, reference, roots, torch.ones(B, 1, dtype=torch.long), end2, torch.ones(B, dtype=torch.long))
    assert not bool(trunc2.any())
    for w, want in enumerate((a, b, c)):
        assert torch.allclose(rec2[:, w], want, atol=1e-4), ("compound", w, float((rec2[:, w] - want).abs().max()))
    m.End(); m.symbolSpace.soft_reset()


@pytest.mark.slow
def test_snapshot_bytes_are_staged_on_the_first_brick(tmp_path):
    """The dictionary snapshot is staged after the brick's concept rows,
    so the byte decoder is active from the first brick: the words' rows
    and their object rows carry the words' bytes."""
    m = _traversal_model(tmp_path)
    _run(m, ["ab", "12"])
    isp = m.inputSpace
    bank = isp._ar_concept_lookup_rows
    valid = isp._ar_bank_valid
    assert torch.is_tensor(valid) and int(valid.shape[0]) == int(bank.shape[0])
    L = int(bank.shape[1])
    present = valid.any(-1)                                            # [B, L]
    assert bool(present[:, :L // 2].any(1).all())                      # the words' rows
    assert bool(present[:, L // 2:].any(1).all())                      # their object rows
    snap, _n, _bytes, _valid = m._snapshot_tables(m._tensor_pushed_ideas)
    assert snap
    m.End(); m.symbolSpace.soft_reset()
