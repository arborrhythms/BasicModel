"""Task 6a (doc/plans/2026-05-29-stm-serial-parallel-modes.md §7).

The serial sentence-boundary reduce (``_stm_reduce_to_single_S``)
collapses an ABSOLUTE sentence's STM down to a single idea ``S`` (depth
1 -- the start-symbol root). A RELATIVE sentence (the ``part`` /
``isEqual`` predicate family, ``REL_T`` in ``data/complete.grammar``)
must instead STOP at the depth-3 end-state ``[predicate, idea1, idea2]``
so the binary predicate survives the boundary.

These tests pin three layers of the feature on the REAL default grammar
(``complete.grammar`` via MentalModel.xml; no grammar stubbing): the grammar
``is_relative_rule`` marker (Part A), the conservative
``_sentence_relative_mask`` detection helper (Part B), and the reduce-
site depth-3 preservation (Part C). Only the STM *contents* and the
compose *selection* (``symbolSpace.current_rules``) are hand-set --
exactly the boundary the plan permits -- because driving a real
relative surface string end-to-end through the per-word forward is far
heavier and far more fragile than directly exercising the reduce.

The ABSOLUTE regression (``...still_collapses...``) is the conservatism
guard: it proves the depth-1 collapse is byte-unchanged when no relative
sentence is detected.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
import pytest

import Language
import Models
from util import init_config


_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _reload_config():
    """Reload defaults + MentalModel.xml (loads complete.grammar)."""
    init_config(
        path=os.path.join(_DATA_DIR, 'MentalModel.xml'),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )
    Language.TheGrammar._configured = False


def _build_model():
    _reload_config()
    model, _cfg = Models.BasicModel.from_config(
        os.path.join(_DATA_DIR, 'MentalModel.xml'))
    return model


def _forward_relative_rule_id():
    """A forward relative rule_id, found grammar-agnostically via the relative
    detection set (``REL_T = ...`` in complete.grammar; an ``isEqual``/
    ``isPart`` output-role rule in the role-collapsed default)."""
    g = Language.TheGrammar
    g._ensure_configured()
    rel = g._relative_rule_id_set()
    for rid in sorted(rel):
        r = g.rules[rid]
        if '.reverse' not in (r.canonical or ''):
            return rid
    # No grammar-level relative producer (the is* QUERY cousins never count:
    # queries are not parse structure). complete.grammar regained its
    # producers 2026-07-05 as the COMPOSE relations part/whole/equal heading
    # the relative_truth starts (rule set is lhs-driven), so this skip now
    # fires only on a grammar without relative starts.
    pytest.skip("no relative parse rule in the configured grammar")


def _forward_absolute_rule_id():
    """A forward ABSOLUTE operational rule_id: a forward operator rule NOT in
    the relative set (e.g. ``exist`` / ``conjunction``). Grammar-agnostic."""
    g = Language.TheGrammar
    g._ensure_configured()
    rel = g._relative_rule_id_set()
    for rid, r in enumerate(g.rules):
        if (rid not in rel and r.method_name is not None
                and r.method_name not in g._RELATIVE_OP_NAMES
                and '.reverse' not in (r.canonical or '')):
            return rid
    raise AssertionError("no forward absolute rule found in the grammar")


def _seed_stm(model, depths):
    """Hand-set ``conceptualSpace.stm`` to the given per-row ``depths``.

    Each occupied slot is filled with a distinct non-zero vector so the
    post-reduce non-zero checks are meaningful. Returns ``(B, dim)``.
    """
    stm = model.conceptualSpace.stm
    B = len(depths)
    dim = int(stm.concept_dim)
    stm.ensure_batch(B)
    stm.ensure_capacity(8)
    buf = torch.zeros(B, int(stm.capacity), dim)
    for b, d in enumerate(depths):
        for s in range(d):
            # Distinct, non-zero per (row, slot): base 1.0 + small ramp.
            buf[b, s, :] = 1.0 + 0.1 * (b + 1) + 0.01 * (s + 1)
    stm._buffer = buf
    stm._depth = torch.tensor([int(d) for d in depths], dtype=torch.long)
    return B, dim


def _set_current_rules(model, rule_id, B, per_row=True):
    """Stub ``symbolSpace.current_rules`` SS-space_role with ``rule_id``.

    ``per_row=True`` mirrors the full-router ``LanguageLayer.compose``
    shape (one inner list per batch row); ``per_row=False`` mirrors the
    default-only batch-shared shape (a single inner list).
    """
    if per_row:
        s_rules = [[rule_id] for _ in range(B)]
    else:
        s_rules = [[rule_id]]
    model.symbolSpace.current_rules = {'SS': s_rules}


# --------------------------------------------------------------------------
# Part A -- grammar marker
# --------------------------------------------------------------------------

def test_grammar_marks_relative_rules():
    """``is_relative_rule`` flags REL_T / part / isEqual rules and not
    the absolute / structural ones."""
    _reload_config()
    g = Language.TheGrammar
    g._ensure_configured()

    rel_id = _forward_relative_rule_id()
    abs_id = _forward_absolute_rule_id()

    assert g.is_relative_rule(rel_id), (
        f"forward REL_T rule {rel_id} "
        f"({g.rules[rel_id].canonical!r}) not flagged relative")
    assert not g.is_relative_rule(abs_id), (
        f"absolute rule {abs_id} ({g.rules[abs_id].canonical!r}) "
        f"wrongly flagged relative")

    # Grammar-driven primary signal: a relative start category exists. In
    # complete.grammar that is {'REL_T'}; the role-collapsed default heads its
    # relative rules with operator output roles (isEqual_O1 / isPart_O1).
    rel_starts = g._relative_start_categories()
    assert rel_starts, "expected a non-empty relative start-category set"
    rel_set = g._relative_rule_id_set()
    assert rel_set, "expected a non-empty relative rule set"
    # Every relative-set rule carries a relative signal: it either heads a
    # relative start category (REL_T in complete.grammar; isEqual_O1 / isPart_O1
    # in the role-collapsed default, including the bare output projections) or
    # is itself a relative operator.
    for rid in rel_set:
        r = g.rules[rid]
        assert (r.lhs in rel_starts
                or r.method_name in g._RELATIVE_OP_NAMES), (
            f"rule {rid} in relative set without a relative signal: "
            f"{r.canonical!r}")

    # Out-of-range / junk ids are conservatively non-relative.
    assert not g.is_relative_rule(10 ** 9)
    assert not g.is_relative_rule(None)
    assert not g.is_relative_rule("not-an-id")


# --------------------------------------------------------------------------
# Part B -- conservative detection helper
# --------------------------------------------------------------------------

def test_relative_mask_per_row_and_shared_shapes():
    """``_sentence_relative_mask`` handles per-row and batch-shared
    ``current_rules`` shapes, and stays all-False on no/absolute rules."""
    model = _build_model()
    B = 3
    model.conceptualSpace.stm.ensure_batch(B)
    rel_id = _forward_relative_rule_id()
    abs_id = _forward_absolute_rule_id()

    # No current_rules -> all-False (default {} after construction).
    model.symbolSpace.current_rules = {}
    m = model._sentence_relative_mask(B)
    assert m.dtype == torch.bool and m.shape == (B,)
    assert not bool(m.any())

    # Per-row relative (full-router shape) -> all-True.
    _set_current_rules(model, rel_id, B, per_row=True)
    m = model._sentence_relative_mask(B)
    assert m.shape == (B,) and bool(m.all())

    # Batch-shared relative (default-only shape) -> broadcast all-True.
    _set_current_rules(model, rel_id, B, per_row=False)
    m = model._sentence_relative_mask(B)
    assert m.shape == (B,) and bool(m.all())

    # Absolute rule -> all-False (conservatism).
    _set_current_rules(model, abs_id, B, per_row=True)
    m = model._sentence_relative_mask(B)
    assert m.shape == (B,) and not bool(m.any())

    # Mixed per-row: only the rows whose inner list carries a relative
    # rule_id are flagged.
    model.symbolSpace.current_rules = {
        'SS': [[abs_id], [rel_id], [abs_id]]}
    m = model._sentence_relative_mask(B)
    assert m.tolist() == [False, True, False]


# --------------------------------------------------------------------------
# Part C / Part D -- reduce-site depth-3 preservation + absolute regression
# --------------------------------------------------------------------------

def test_relative_sentence_reduces_to_depth_three():
    """A relative sentence's STM stops at depth 3 with non-zero
    ``[predicate, idea1, idea2]`` slots."""
    model = _build_model()
    B, dim = _seed_stm(model, depths=[5, 5])
    rel_id = _forward_relative_rule_id()
    _set_current_rules(model, rel_id, B, per_row=True)

    S, post_depth = model._stm_reduce_to_single_S()

    assert post_depth.tolist() == [3, 3], (
        f"relative rows must stop at depth 3, got {post_depth.tolist()}")
    buf = model.conceptualSpace.stm._buffer
    for b in range(B):
        for slot in range(3):
            assert buf[b, slot].abs().sum() > 0, (
                f"relative row {b} slot {slot} "
                f"(of the depth-3 relative end-state) is unexpectedly zero")
    # Newest-at-slot-0 convention: the end-state is stored newest-first, so
    # the predicate (oldest constituent) is at the LAST slot ``depth-1``
    # (== slot 2 here). ``S`` is read per-row at slot ``depth-1`` and so
    # carries the predicate for every relative row.
    assert torch.equal(S, buf[:, 2, :])
    assert S.shape == (B, dim)


def test_absolute_sentence_still_collapses_to_depth_one():
    """REGRESSION / conservatism guard: with an absolute rule (or no
    rules) the reduce collapses to depth 1, byte-for-byte as before."""
    model = _build_model()
    abs_id = _forward_absolute_rule_id()

    # (a) absolute rule selected -> depth 1.
    B, _dim = _seed_stm(model, depths=[5, 4])
    _set_current_rules(model, abs_id, B, per_row=True)
    _S, post_depth = model._stm_reduce_to_single_S()
    assert post_depth.tolist() == [1, 1], (
        f"absolute rows must collapse to depth 1, got {post_depth.tolist()}")

    # (b) no current_rules at all -> still depth 1 (the pre-Task-6a
    # default path; nothing is detected relative).
    model2 = _build_model()
    B2, _ = _seed_stm(model2, depths=[6, 3])
    model2.symbolSpace.current_rules = {}
    _S2, post_depth2 = model2._stm_reduce_to_single_S()
    assert post_depth2.tolist() == [1, 1], (
        f"no-rules rows must collapse to depth 1, got {post_depth2.tolist()}")


def test_protect_depth_floor_one_is_byte_identical_to_no_arg():
    """REGRESSION (absolute path byte-identical): on a SINGLE model +
    SINGLE seeded STM, the Task-6a per-row gate with an all-ones
    ``protect_depth`` (the floor the absolute path uses) is BIT-FOR-BIT
    identical to the pre-Task-6a no-arg ``_stm_bounded_reduce_step``.

    This is the precise invariant the change must preserve: the added
    ``depth > protect_depth`` term is implied by the existing
    ``depth >= 2`` term when the floor is 1, so the masked fold -- and
    the whole collapse -- is unchanged. (Comparing two freshly-built
    models would instead measure the reducer's random anchor init, not
    the gate, so we reuse one model and only re-seed the STM.)"""
    model = _build_model()
    stm = model.conceptualSpace.stm
    cap = int(stm.capacity)
    B = 2
    dim = int(stm.concept_dim)

    def _seed():
        stm.ensure_batch(B)
        stm.ensure_capacity(8)
        buf = torch.zeros(B, int(stm.capacity), dim)
        for b in range(B):
            for s in range(5):
                buf[b, s, :] = 1.0 + 0.1 * (b + 1) + 0.01 * (s + 1)
        stm._buffer = buf.clone()
        stm._depth = torch.tensor([5, 5], dtype=torch.long)

    # Old path: no-arg step (protect_depth implicitly None).
    _seed()
    for _ in range(cap - 1):
        model._stm_bounded_reduce_step()
    s_old = stm._buffer[:, 0, :].detach().clone()
    d_old = stm._depth.detach().clone()

    # New path: explicit all-ones floor (what absolute rows get).
    _seed()
    ones = torch.ones(B, dtype=stm._depth.dtype)
    for _ in range(cap - 1):
        model._stm_bounded_reduce_step(protect_depth=ones)
    s_new = stm._buffer[:, 0, :].detach().clone()
    d_new = stm._depth.detach().clone()

    assert torch.equal(d_old, d_new), (
        f"depth diverged: old={d_old.tolist()} new={d_new.tolist()}")
    assert torch.equal(s_old, s_new), (
        "absolute reduce not byte-identical: the floor-1 per-row gate "
        f"perturbed the result (max|diff|={(s_old - s_new).abs().max()})")


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
