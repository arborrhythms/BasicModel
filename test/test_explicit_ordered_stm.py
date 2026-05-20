"""Explicit ordered grammar tests for STM.

The production grammar can enumerate order-changing rules directly:

    S4 = lift(NP3, VP1)
    S5 = lift(NP4, MP1)

STM therefore needs exact typed admissibility over ``(category, order)``
rather than Kleene inference.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "bin"))


def _ordered_grammar():
    import Language
    from Language import Grammar
    g = Grammar()
    g.rules = [
        g._parse_rule("S4", "lift(NP3, VP1)", tier='S'),
        g._parse_rule("S5", "lift(NP4, MP1)", tier='S'),
    ]
    g._configured = True
    Language.TheGrammar = g
    return g


def _ordered_view():
    from embed import build_knowledge_section, build_typed_indexes, KnowledgeView
    g = _ordered_grammar()
    ks = build_knowledge_section(g)
    rc = ks['reference_codebook']
    tax = ks['taxonomy']
    ordered = tax['ordered_taxonomy_names']
    refs = rc['references']
    refs[:rc['v_ref_live']] = 99.0
    refs[int(ordered['NP3'])] = 1.0
    refs[int(ordered['VP1'])] = 2.0
    refs[int(ordered['NP4'])] = 3.0
    refs[int(ordered['MP1'])] = 4.0
    rc['references'] = refs
    ks['typed_indexes'] = build_typed_indexes(tax, rc)
    return KnowledgeView(ks)


def _word_space():
    from Language import WordSpace
    from Spaces import ConceptualSpace
    ws = object.__new__(WordSpace)
    nn.Module.__init__(ws)
    ws.parser_backend = 'stm'
    ws.attach_knowledge(_ordered_view())
    cs = object.__new__(ConceptualSpace)
    nn.Module.__init__(cs)
    cs._init_typed_stm(batch=1, max_depth=8, dim=4)
    ws.conceptualSpace = cs
    return ws


def test_ordered_taxonomy_preserves_np3_np4_orders():
    view = _ordered_view()
    ordered = view.ordered_taxonomy_names
    assert view.order_of_ref(ordered['NP3']) == 3
    assert view.order_of_ref(ordered['NP4']) == 4
    assert view.category_of_ref(ordered['NP3']) == 'NP'
    assert view.category_of_ref(ordered['NP4']) == 'NP'
    np_refs = set(view.refs_by_category('NP').tolist())
    assert int(ordered['NP3']) in np_refs
    assert int(ordered['NP4']) in np_refs


def test_stm_parses_explicit_np3_lift_to_s4():
    ws = _word_space()
    inp = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0],  # NP3
        [2.0, 0.0, 0.0, 0.0],  # VP1
    ]])
    rules = ws.compose(inp)
    assert rules['S'][0] == [0]
    ps = ws.parse_state
    root = ps.frames[ps.trace[-1].parent_index]
    assert root.category == 'S'
    assert root.order == 4
    assert ps.trace[-1].probability == 1.0


def test_stm_parses_explicit_np4_lift_to_s5():
    ws = _word_space()
    inp = torch.tensor([[
        [3.0, 0.0, 0.0, 0.0],  # NP4
        [4.0, 0.0, 0.0, 0.0],  # MP1
    ]])
    rules = ws.compose(inp)
    assert rules['S'][0] == [1]
    ps = ws.parse_state
    root = ps.frames[ps.trace[-1].parent_index]
    assert root.category == 'S'
    assert root.order == 5
    assert ps.trace[-1].probability == 1.0
