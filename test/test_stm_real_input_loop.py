"""Tests for the STM SHIFT/REDUCE loop over real input_vectors.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 deferred — "STM SHIFT/REDUCE loop over real input_vectors".

Today ``_compose_stm`` just constructs the driver and returns an
empty dict. This suite drives the real loop: tokenize ``input_vectors``
via reference-codebook snap, shift each token at order 0, run REDUCE
until either a single S-rooted frame remains or no admissible rule
fires, emit per-tier rule selections compatible with the existing
``current_rules`` consumer.

The output contract mirrors ``SyntacticLayer._collect_rule_selections``:
``dict[tier_name -> list[list[rule_id]]]``.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def _bare_word_space(stm_dim=4, max_depth=12, batch=1):
    """Bare WordSubSpace with manually-allocated typed-STM buffers.

    Post-2026-05-21 (WordSubSpace/STM Layer refactor) the typed STM
    stack lives directly on WordSubSpace, not on
    ``ConceptualSpace._stm_typed``.
    """
    from Language import WordSubSpace
    ws = object.__new__(WordSubSpace)
    nn.Module.__init__(ws)
    ws.batch = int(batch)
    ws._stm_capacity = int(max_depth)
    ws._stm_payload_dim = int(stm_dim)
    ws.max_depth = ws._stm_capacity
    ws.dim = ws._stm_payload_dim
    ws.register_buffer(
        '_buffer', torch.zeros(ws.batch, max_depth, stm_dim),
        persistent=False)
    ws.register_buffer(
        '_category',
        torch.full((ws.batch, max_depth), -1, dtype=torch.long),
        persistent=False)
    ws.register_buffer(
        '_order',
        torch.zeros((ws.batch, max_depth), dtype=torch.long),
        persistent=False)
    ws.register_buffer(
        '_ref_id',
        torch.full((ws.batch, max_depth), -1, dtype=torch.long),
        persistent=False)
    ws.register_buffer(
        '_depth', torch.zeros(ws.batch, dtype=torch.long),
        persistent=False)
    ws._category_names = [
        [None] * max_depth for _ in range(ws.batch)]
    # Idea-stack buffers (Phase E completion of the 2026-05-21 refactor).
    ws._idea_capacity = max_depth
    ws._idea_max_depth_host = 0
    ws.register_buffer(
        '_idea_buffer',
        torch.zeros(ws.batch, max_depth, stm_dim),
        persistent=False)
    ws.register_buffer(
        '_idea_depth',
        torch.zeros(ws.batch, dtype=torch.long),
        persistent=False)
    return ws


def _bare_conceptual_space(stm_dim=4, max_depth=12, batch=1):
    """Bare ConceptualSpace exposing a real ShortTermMemory Layer.

    Post-Phase-E the ShortTermMemory Layer (in bin/Layers.py) holds
    the rule scorer; the typed-STM stack data is on WordSubSpace.
    """
    from Spaces import ConceptualSpace
    from Layers import ShortTermMemory
    cs = object.__new__(ConceptualSpace)
    nn.Module.__init__(cs)
    cs.stm = ShortTermMemory(batch=int(batch), capacity=int(max_depth),
                             concept_dim=int(stm_dim))
    return cs


def _order_zero_grammar():
    """Tiny order-preserving grammar with bare categories:

        NP = conjunction(DET, N)
        S  = disjunction(NP, VP)

    Bare categories bind to order 0, so SHIFT at order 0 (the lexical
    contract) matches the rule's RHS expectation directly. Also points
    ``Language.TheGrammar`` at this instance so ``_tier_for_rule``'s
    canonical lookup hits this grammar — production has the artifact
    and the live grammar in sync; the test fixture mirrors that
    invariant rather than relying on the LHS-prefix fallback alone.
    """
    import Language
    from Language import Grammar
    g = Grammar()
    g.rules = [
        g._parse_rule("NP", "conjunction(DET, N)", tier='S'),
        g._parse_rule("S", "disjunction(NP, VP)", tier='S'),
    ]
    g._configured = True
    Language.TheGrammar = g
    return g


def _view_with_distinct_pos_scalars():
    """Knowledge view whose POS terminals (DET, N, VP) carry distinct
    scalars so the SHIFT snap unambiguously hits the right category.
    Nonterminal and root scalars are pushed out of the input range so
    they never win the nearest-neighbor lookup.
    """
    from embed import (build_knowledge_section, KnowledgeView,
                       build_typed_indexes)
    g = _order_zero_grammar()
    ks = build_knowledge_section(g)
    rc = ks['reference_codebook']
    tax = ks['taxonomy']
    names = tax['taxonomy_names']
    refs = rc['references']
    # Move root + nonterminals out of input range so the snap can't
    # accidentally pick them. POS terminals get distinct in-range
    # scalars matching the tokens we'll feed.
    refs[0] = 99.0  # root
    for nt in ('NP', 'S'):
        if nt in names:
            refs[int(names[nt])] = 99.0
    refs[int(names['DET'])] = 1.0
    refs[int(names['N'])] = 2.0
    refs[int(names['VP'])] = 3.0
    rc['references'] = refs
    # Rebuild typed indexes (order didn't change but for safety).
    ks['typed_indexes'] = build_typed_indexes(tax, rc)
    return KnowledgeView(ks)


def _make_word_space_for_stm(stm_dim=4):
    ws = _bare_word_space(stm_dim=stm_dim, max_depth=16, batch=1)
    ws.parser_backend = 'stm'
    view = _view_with_distinct_pos_scalars()
    ws.attach_knowledge(view)
    cs = _bare_conceptual_space(stm_dim=stm_dim, max_depth=16, batch=1)
    object.__setattr__(ws, 'conceptualSpace', cs)
    return ws, view


def test_compose_stm_none_input_returns_empty_dict():
    """Legacy contract: ``input_vectors=None`` still constructs the
    driver and returns an empty dict (no SHIFT/REDUCE done)."""
    ws, _ = _make_word_space_for_stm()
    rules = ws.compose(input_vectors=None)
    assert rules == {}


def test_compose_stm_real_input_returns_dict():
    """With real input_vectors, compose returns a dict."""
    ws, _ = _make_word_space_for_stm(stm_dim=4)
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    rules = ws.compose(input_vectors=inp)
    assert isinstance(rules, dict)


def test_compose_stm_emits_rule_selections_for_valid_parse():
    """3-token DET-N-VP sequence with the grammar
    ``NP = conjunction(DET, N)``, ``S = disjunction(NP, VP)`` produces
    two REDUCEs: NP-rule then S-rule."""
    ws, _ = _make_word_space_for_stm(stm_dim=4)
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],   # DET → ref scalar 1.0
         [2.0, 0.0, 0.0, 0.0],   # N   → ref scalar 2.0
         [3.0, 0.0, 0.0, 0.0]],  # VP  → ref scalar 3.0
    ])
    rules = ws.compose(input_vectors=inp)
    # Both rules are tier 'S' (per the grammar's tier='S' setting
    # and the LHS-prefix heuristic for fallback derivation)
    assert 'S' in rules
    row0 = rules['S'][0]
    assert len(row0) == 2  # NP then S
    # Each entry is a valid rule index (0 or 1 in our 2-rule grammar)
    assert all(0 <= r < 2 for r in row0)
    # NP comes first (left-corner shift-reduce)
    assert row0[0] == 0  # NP = conjunction(DET, N)
    assert row0[1] == 1  # S = disjunction(NP, VP)


def test_compose_stm_per_row_structure():
    """``current_rules[tier]`` is a list of length ``batch``, each
    entry being a list of rule_ids fired for that row."""
    ws, _ = _make_word_space_for_stm(stm_dim=4)
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    rules = ws.compose(input_vectors=inp)
    for tier, rs in rules.items():
        assert isinstance(rs, list)
        assert len(rs) == 1  # batch = 1
        for r in rs:
            assert isinstance(r, list)
            for rid in r:
                assert isinstance(rid, int)


def test_compose_stm_clears_stack_between_calls():
    """Repeated compose() doesn't accumulate stack state."""
    ws, _ = _make_word_space_for_stm(stm_dim=4)
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    first = ws.compose(input_vectors=inp)
    second = ws.compose(input_vectors=inp)
    assert first == second


def test_generate_stm_returns_dict():
    """``generate()`` under STM mirrors ``compose()`` and returns a
    dict for ``generate_rules``."""
    ws, _ = _make_word_space_for_stm(stm_dim=4)
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    rules = ws.generate(target_vectors=inp)
    assert isinstance(rules, dict)


def test_generate_stm_reverses_compose_order():
    """``_collect_generate_selections`` reverses each row's trace so
    that downward generation pops the last-applied rule first. STM
    mirrors this."""
    ws, _ = _make_word_space_for_stm(stm_dim=4)
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    composed = ws.compose(input_vectors=inp)
    generated = ws.generate(target_vectors=inp)
    assert 'S' in composed and 'S' in generated
    assert generated['S'][0] == list(reversed(composed['S'][0]))


def test_compose_stm_handles_no_admissible_gracefully():
    """When parser gets stuck (no admissible rule), compose returns
    whatever rules fired before getting stuck — no exception."""
    ws, _ = _make_word_space_for_stm(stm_dim=4)
    # Single token in: SHIFT once, no possible REDUCE.
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0]],
    ])
    rules = ws.compose(input_vectors=inp)
    assert isinstance(rules, dict)
    for tier, rs in rules.items():
        assert all(r == [] for r in rs)


def test_compose_stm_unknown_backend_unchanged():
    """STM real-input loop changes don't affect chart backend defaults."""
    ws, _ = _make_word_space_for_stm(stm_dim=4)
    # parser_backend='stm' is set; flip to invalid to confirm dispatch
    # still errors as before
    ws.parser_backend = 'bogus'
    import pytest
    with pytest.raises(ValueError, match='unknown parser_backend'):
        ws.compose(input_vectors=None)
