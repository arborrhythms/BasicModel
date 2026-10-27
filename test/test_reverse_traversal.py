"""Free reconstruction, tied inverse, and compiled-boundary contracts.

The compose journal is provenance; the free decoder never replays it.
"""
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
    from contextlib import nullcontext
    boundary = nullcontext() if getattr(m, "_sentence_run_active", False) else m._sentence_run()
    with torch.no_grad(), boundary:
        out = (fn or m._forward_with_compiled_sentence_state)(None)
    m._publish_compiled_sentence_state(out)
    return out


@pytest.mark.usefixtures('eager_reading')
def test_free_walk_reports_missing_leaves_without_underflow(tmp_path):
    from test_output_walk import _stop
    m = _traversal_model(tmp_path)
    m.eval()
    _stop(m)
    try:
        out = _run(m, ["12 plus 1", "3 plus 4"])
        assert len(out) == 21
        active = m.inputSpace._word_active_mask
        assert tuple(m._recon_ideas.shape) == (*active.shape, int(m.conceptualSpace.stm.concept_dim))
        assert torch.isfinite(m._recon_cost).all()
        assert torch.isfinite(m._recon_ideas).all()
        assert not m._recon_ideas[~active].any()
        # STOP cannot recover these multiword inputs. Coverage is reported,
        # rather than manufacturing a traversal from the target length.
        assert m._recon_truncated.all()
    finally:
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
        parent = (1 + torch.tanh(torch.randn(3, D) * 0.5)) / 2
        # A composition output has no occurrence band in its content kernel.
        parent = op.content_operand(parent)
        left, right = language.reverse_binary_step(
            parent, torch.full((3,), idx), torch.ones(3, dtype=torch.bool))
        recomposed = op.compose(left, right)
        cw = int(getattr(op, "_content_width", 0) or D)
        assert torch.allclose(recomposed[:, :cw], parent[:, :cw], atol=1e-3), name
    # An invalid row is left untouched.
    parent = (1 + torch.tanh(torch.randn(2, D))) / 2
    left, right = language.reverse_binary_step(
        parent, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.bool))
    assert torch.equal(left, parent) and torch.equal(right, parent)


@pytest.mark.usefixtures('eager_reading')
def test_cleared_journal_cannot_change_the_free_reconstruction(tmp_path):
    m = _traversal_model(tmp_path)
    m.eval()
    try:
        _run(m, ["12 plus 1"])
        record = next(iter(m._sentence_understandings.values()))
        def recover():
            return m._reconstruct_sentences(record.root, record.word_values,
                record.roots, record.depths, record.end_slots, record.end_depth,
                record.sentence, understanding=record, keep_ideas=True)
        with torch.no_grad():
            before = recover()
            m._reconstruction_stack()._choice_mask.zero_()
            after = recover()
        for actual, expected in zip(after, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        m.End(); m.symbolSpace.soft_reset()


@pytest.mark.usefixtures('eager_reading')
def test_reconstruction_uses_held_word_support_after_attention_changes(tmp_path):
    from test_output_walk import _stop
    m = _traversal_model(tmp_path)
    m.eval()
    _stop(m)
    try:
        _run(m, ["12 plus 1"])
        record = next(iter(m._sentence_understandings.values()))
        def recover():
            return m._reconstruct_sentences(record.root, record.word_values,
                record.roots, record.depths, record.end_slots, record.end_depth,
                record.sentence, understanding=record, keep_ideas=True)
        with torch.no_grad():
            before = recover()
            m.inputSpace._ar_grammar_leaf_mask = torch.zeros_like(record.word_valid)
            after = recover()
        assert before[2].gt(0).all()
        for actual, expected in zip(after, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        m.End(); m.symbolSpace.soft_reset()


@pytest.mark.usefixtures('eager_reading')
def test_native_identity_audit_accepts_bank_words_and_rejects_unknown_content(tmp_path):
    m = _traversal_model(tmp_path)
    m.eval()
    try:
        _run(m, ["12 plus 1"])
        record = next(iter(m._sentence_understandings.values()))
        bank = record.primed
        ready, targets, valid = m._byte_tables(*record.word_values.shape[:2])
        assert ready and bank.valid.any()
        index = bank.valid.long().argmax(-1)
        word = bank.codes.gather(1, index[:, None, None].expand(-1,1,bank.codes.shape[-1]))[:,0]
        args = (torch.tensor(0), bank.codes, bank.bytes, bank.byte_valid, targets, valid, True)
        torch.testing.assert_close(m._byte_word_cost(word, *args), torch.zeros(len(word)))
        unknown = torch.zeros_like(word).requires_grad_()
        cost = m._byte_word_cost(unknown, *args)
        torch.testing.assert_close(cost, torch.ones(len(word)))
        # Fixed identity is an audit; continuous byte credit is a separate
        # dictionary contract, covered below and in test_byte_reconstruction.
        torch.testing.assert_close(torch.autograd.grad(cost.sum(), unknown)[0], torch.zeros_like(unknown))
    finally:
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


def _select_completed_binary_path(m):
    """Control the derivation for traversal/provenance assertions, not quality."""
    chooser = m.symbolSpace.languageLayer.operation_layer.chooser

    def binary(x, candidates, *_args, **_kwargs):
        scores = x.new_full(candidates.shape[:-1], -1e6)
        scores[..., 0] = 0.
        return x.new_full((*x.shape[:2], 1), -1e6), scores

    def unary(x, candidates, *_args, **_kwargs):
        return (x.new_full((*x.shape[:2], 1), -1e6),
                x.new_full(candidates.shape[:-1], -1e6))

    chooser.score_binary, chooser.score_unary = binary, unary


def _replay_operand_rows(m):
    """Replay the real trace's row stack, newest first, before each fold."""
    trace = m._reconstruction_stack()
    _rules, arities, mask = trace.choices()
    active = m.inputSpace._word_active_mask
    leaves = m.inputSpace._ar_grammar_leaf_mask
    ids = m.inputSpace._packed_sentence_ids
    rows = m._word_symbol_rows()
    width = int(active.shape[1])
    closing_width = 2 * int(m.conceptualSpace.stm.capacity)
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
                    position = int(trace._choice_positions[b, slot])
                    assert 0 <= position < len(stack)
                    stack[position] = -1

            for w in sentence:
                if leaves is None or bool(leaves[b, w]):
                    stack.insert(0, int(rows[b, w]))
                fold(3 * w)
                fold(3 * w + 1)
                fold(3 * w + 2)
            hi = sentence[-1]
            base = 3 * width if hi == words[-1] else 3 * width + (hi + 1) * closing_width
            for k in range(closing_width):
                fold(base + k)
            assert len(stack) == 1, (b, sid, stack)
    return expected


@pytest.mark.slow
@pytest.mark.usefixtures('eager_reading')
def test_packed_trace_records_pre_fold_operand_rows_at_every_binary(tmp_path, monkeypatch):
    """Codex item 4: real packed endings retain the operands of the fold,
    including a known leaf beside a composite, before reducing the stack."""
    m = _build_ladder_variant(tmp_path, "operand_rows", [
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>32</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>32</serialWordBuckets>")])
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    m._install_unit_span_fn()
    _select_completed_binary_path(m)
    try:
        _stage_packed(m, [["aa bb cc dd ee", "ff gg hh ii jj"], ["kk ll mm", "nn oo"]])
        # Keep the final lexical leaf beside an older unary-rewritten
        # constituent until the closing. Every active round still applies a
        # legal joint choice; there is no per-position unary/no-op layer.
        from Spaces import LanguageOperationChoice
        language = m.languageSpace
        operation = language.language_layer.operation_layer
        ordinary = language.choose_operation
        leaves = m.inputSpace._ar_grammar_leaf_mask.bool()
        sentence_ids = m.inputSpace._packed_sentence_ids
        width = int(leaves.shape[1])
        positions = torch.arange(width, device=leaves.device)
        later = positions[None, :] > positions[:, None]
        same_sentence = sentence_ids[:, :, None] == sentence_ids[:, None, :]
        # A word's accepted leaf can precede its last raw tile. Retain the
        # final accepted leaf of each sentence, not the final raw tile.
        ends = leaves & ~(same_sentence & later & leaves[:, None, :]).any(-1)
        monkeypatch.setattr(m, "_compose_masked_action", lambda slot, depth:
                            torch.as_tensor(slot, device=depth.device).expand_as(depth))

        def retain_final_leaf(state, gate, *, masked_action, **kwargs):
            choice = ordinary(state, gate, **kwargs)
            slot = masked_action
            word = (slot // 3).clamp(0, width - 1)
            retain = (ends.gather(1, word[:, None]).squeeze(1)
                      & (slot >= 0) & (slot < 3 * width)
                      & gate.reshape(-1) & (state[1] > 1))
            candidate = operation._apply_op(
                operation.unary_ops[0], state[0][:, 1:2, :])[:, 0]
            unary = LanguageOperationChoice(
                candidate, torch.full_like(state[1], 2),
                torch.ones_like(state[1]), torch.zeros_like(state[1]),
                retain, candidate.new_ones(state[1].shape),
                torch.full_like(state[1], operation.r_reduce), retain,
                torch.zeros_like(retain))
            return LanguageOperationChoice(*(torch.where(
                retain[:, None] if a.ndim == 2 else retain, a, b) if torch.is_tensor(a) else b
                for a, b in zip(unary, choice)))

        monkeypatch.setattr(language, "choose_operation", retain_final_leaf)
        with torch.no_grad():
            out = m._forward_with_compiled_sentence_state(None)
        m._publish_compiled_sentence_state(out)
        expected = _replay_operand_rows(m)
        trace = m._reconstruction_stack()
        _rules, arities, mask = trace.choices()
        recorded = set(map(tuple, (mask.bool() & (arities == 2)).nonzero().tolist()))
        assert set(expected) == recorded
        width = int(m.inputSpace._word_active_mask.shape[1])
        closing_width = 2 * int(m.conceptualSpace.stm.capacity)
        assert any(slot < 3 * width for _, slot in expected)  # per-word folds
        assert any(3 * width <= slot < 3 * width + closing_width for _, slot in expected)
        intermediate = {key: value for key, value in expected.items()
                        if key[1] >= 3 * width + closing_width}
        assert intermediate and any(min(pair) < 0 <= max(pair)
                                    for pair in intermediate.values())
        for (b, slot), pair in expected.items():
            actual = (int(trace._choice_left_rows[b, slot]),
                      int(trace._choice_right_rows[b, slot]))
            assert actual == pair, (b, slot, actual, pair)
    finally:
        m.End()
        m.symbolSpace.soft_reset()



def _capture_scored_records(model):
    """Hold the exact scored inputs before later thought can bind their roots."""
    from dataclasses import fields, replace
    captured = {}
    original = model._reconstruct_trial
    def reconstruct(record):
        captured[int(record.sentence)] = replace(record, **{
            field.name: getattr(record, field.name).detach().clone()
            for field in fields(record) if torch.is_tensor(getattr(record, field.name))})
        return original(record)
    model._reconstruct_trial = reconstruct
    return captured

def _compiled_record_reconstruction(model, records):
    """Compile the current numerical boundary with identical saved evidence.

    The old test reread the input, promoted new percepts, and bypassed the
    eager sentence boundary in the second arm. It did not compare one state.
    The host sentence/optimizer boundary is now outside the compiled brick.
    """
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    def reconstruct(record):
        return model._reconstruct_sentences(record.root, record.word_values,
            record.roots, record.depths, record.end_slots, record.end_depth,
            record.sentence, understanding=record, keep_ideas=True)
    compiled = torch.compile(reconstruct, backend='eager', fullgraph=True, dynamic=False)
    with torch.no_grad():
        results = [compiled(record) for record in records]
    graphs = int(torch._dynamo.utils.counters['stats']['unique_graphs'])
    # Each held sentence compiles as a full graph. Different bank widths
    # legitimately require distinct static signatures; replay must reuse them.
    assert 1 <= graphs <= len(records)
    with torch.no_grad():
        replay = [compiled(record) for record in records]
    assert int(torch._dynamo.utils.counters['stats']['unique_graphs']) == graphs
    for actual, expected in zip(replay, results):
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
    return torch.stack([r[0] for r in results]).sum(0), torch.stack([r[4] for r in results]).sum(0)


@pytest.mark.slow
def test_packed_rows_reconstruct_each_sentence_separately(tmp_path):
    """Requirement 3: one traversal per completed sentence; packed rows keep
    separate per-sentence costs with identical eager/compiled decoding."""
    m = _build_ladder_variant(tmp_path, "recon16", [
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>"),
        ("<packSentences>false</packSentences>",
         "<packSentences>false</packSentences>\n      <reconstructInLoop>true</reconstructInLoop>")])
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    m._recon_keep_ideas = True
    m._install_unit_span_fn()
    # The compiled and eager traversals must receive the same hard path.
    # Sampling a different path is not a reconstruction parity failure.
    _select_completed_binary_path(m)
    captured = _capture_scored_records(m)
    _stage_packed(m, [["12 plus 1", "3 plus 4"], ["ab cd"]])
    with torch.no_grad(), m._sentence_run():
        out = m._forward_with_compiled_sentence_state(None)
    m._publish_compiled_sentence_state(out)
    isp = m.inputSpace
    ids = isp._packed_sentence_ids
    active = isp._word_active_mask
    assert int(ids[0].max()) == 1 and int(ids[1].max()) == 0
    costs = m._recon_sentence_costs                      # [B, slots]
    assert bool(torch.isfinite(costs).all())
    assert costs.shape[0] == 2 and costs.shape[1] >= 2
    assert torch.isfinite(m._recon_ideas).all()
    assert not m._recon_ideas[~active].any()
    # Untrained free decoding may omit leaves; packed ownership and the
    # compiled/eager numerical boundary must still agree exactly.
    # Row 1 has one sentence: its second slot carries no cost.
    assert float(costs[1, 1]) == 0.0
    eager_ideas, eager_costs = m._recon_ideas.clone(), costs.clone()
    records = list(captured.values())
    try:
        compiled_ideas, compiled_costs = _compiled_record_reconstruction(m, records)
        assert torch.allclose(compiled_costs, eager_costs, atol=1e-4)
        assert torch.allclose(compiled_ideas, eager_ideas, atol=1e-4)
    finally:
        torch._dynamo.reset()
        m.End(); m.symbolSpace.soft_reset()


@pytest.mark.slow
def test_traversal_compiles_into_the_one_sentence_graph(tmp_path):
    m = _traversal_model(tmp_path)
    captured = _capture_scored_records(m)
    with m._sentence_run():
        _run(m, ["12 plus 1", "3 plus 4"])
    eager_ideas, eager_cost = m._recon_ideas.clone(), m._recon_cost.clone()
    records = list(captured.values())
    try:
        compiled_ideas, compiled_costs = _compiled_record_reconstruction(m, records)
        assert torch.allclose(compiled_costs.sum(-1), eager_cost, atol=1e-4)
        assert torch.allclose(compiled_ideas, eager_ideas, atol=1e-4)
    finally:
        torch._dynamo.reset()
        m.End(); m.symbolSpace.soft_reset()


@pytest.mark.slow
@pytest.mark.usefixtures('eager_reading')
def test_final_end_state_keeps_three_slots_for_a_relative_row(tmp_path):
    """A relative sentence stops at depth 3: the traversal starts from
    all three slots (newest at 0), not from the root alone."""
    m = _traversal_model(tmp_path)
    _run(m, ["12 plus 1", "3 plus 4"])
    stm = m.conceptualSpace.stm
    B, D = 2, int(stm.concept_dim)
    buf = torch.randn(B, int(stm.capacity), D)
    from chunk_fixtures import from_stack
    columns = torch.zeros(B, int(stm.capacity), dtype=torch.long)
    stm.set_state(from_stack(buf, torch.tensor([3, 1]), columns, columns,
                             columns-1, torch.ones_like(columns, dtype=buf.dtype)))
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
    from types import SimpleNamespace, MethodType
    from Models import BasicModel
    m = SimpleNamespace(_BYTE_ASSIGNMENT_TAU=.1)
    m._byte_word_cost = MethodType(BasicModel._byte_word_cost, m)
    reference = torch.eye(2).reshape(2,1,2)
    bank_n = torch.eye(2)[None].expand(2,-1,-1)
    bank_bytes = torch.tensor([[[97,98],[49,50]]]).expand(2,-1,-1)
    bank_valid = torch.ones_like(bank_bytes,dtype=torch.bool)
    bytes_bwp = torch.tensor([[[97,98]],[[49,50]]])
    valid_bwp = torch.ones_like(bytes_bwp,dtype=torch.bool)
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


@pytest.mark.slow
def test_snapshot_rows_absent_from_the_brick_do_not_enter_the_score(tmp_path):
    """Candidates are the staged snapshot's present rows: an absent row,
    however similar to the idea, leaves the word's byte cost unchanged;
    a present row that is nearer changes it."""
    from types import SimpleNamespace, MethodType
    from Models import BasicModel
    m = SimpleNamespace(_BYTE_ASSIGNMENT_TAU=.1)
    m._byte_word_cost = MethodType(BasicModel._byte_word_cost, m)
    reference = torch.eye(4)[None]
    bank_n = reference.clone()
    bank_bytes = torch.tensor([[[97],[98],[99],[100]]])
    bank_valid = torch.ones_like(bank_bytes,dtype=torch.bool)
    bytes_bwp, valid_bwp = bank_bytes.clone(), bank_valid.clone()
    idea = reference[:, 1] * 0.5 + reference[:, 3] * 0.5     # between two words
    base = m._byte_word_cost(idea, torch.tensor(1), bank_n, bank_bytes, bank_valid, bytes_bwp, valid_bwp, True)
    absent = bank_valid.clone(); absent[:, 3] = False           # row 3 leaves the snapshot
    off = m._byte_word_cost(idea, torch.tensor(1), bank_n, bank_bytes, absent, bytes_bwp, valid_bwp, True)
    assert not torch.allclose(base, off, atol=1e-4)
    bank2 = bank_n.clone(); bank2[:, 3] = bank_n[:, 1]          # an absent row's content is irrelevant
    same = m._byte_word_cost(idea, torch.tensor(1), bank2, bank_bytes, absent, bytes_bwp, valid_bwp, True)
    assert torch.allclose(off, same, atol=1e-6)


@pytest.mark.slow
@pytest.mark.usefixtures('eager_reading')
def test_snapshot_bytes_are_staged_on_the_first_brick(tmp_path):
    """The dictionary snapshot is staged after the brick's concept rows,
    so the byte decoder is active from the first brick: the words' rows
    and their object rows carry the words' bytes."""
    m = _traversal_model(tmp_path)
    _run(m, ["ab", "12"])
    isp = m.inputSpace
    from PrependedSymbols import lookup_view
    bank = lookup_view(isp, 'lookup_rows')
    valid = isp._ar_bank_valid
    assert tuple(valid.shape[:2]) == tuple(bank.shape)
    assert torch.is_tensor(valid) and int(valid.shape[0]) == int(bank.shape[0])
    L = int(bank.shape[1])
    present = valid.any(-1)                                            # [B, L]
    assert bool(present[:, :L // 2].any(1).all())                      # the words' rows
    assert bool(present[:, L // 2:].any(1).all())                      # their object rows
    snap, _n, _bytes, _valid = m._snapshot_tables(m._tensor_pushed_ideas)
    assert snap
    m.End(); m.symbolSpace.soft_reset()
