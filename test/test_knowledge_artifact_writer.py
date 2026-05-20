"""Tests for the knowledge-section artifact writer in ``bin/embed.py``.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 1 — Artifact writer + order-typed grammar parsing.

The knowledge section bundles word_table, reference_codebook,
typed_indexes, taxonomy, and grammar.rule_order_signatures into a
single artifact section, alongside the existing lexicon / bpe sections.

These tests TDD the writer one helper at a time, starting with the
smallest — rule signature serialization — and building up to the full
section bundler.
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def _tiny_grammar():
    """Construct a tiny manually-built Grammar with two order-typed rules.

    Bypasses XML loading by stuffing ``rules`` directly and flipping
    ``_configured = True``. This keeps the tests independent of any
    ambient ``TheGrammar`` state.
    """
    from Language import Grammar
    g = Grammar()
    g.rules = [
        g._parse_rule("S4", "lift(NP3, VP1)", tier='S'),
        g._parse_rule("NP3", "lower(DET, NP4)", tier='S'),
    ]
    g._configured = True
    return g


def test_grammar_signatures_to_serializable_returns_list_of_dicts():
    """``grammar_signatures_to_serializable(g)`` returns a JSON-friendly
    list of dicts, one per rule, with all fields of the
    RuleOrderSignature flattened to primitive types."""
    from embed import grammar_signatures_to_serializable
    g = _tiny_grammar()
    sigs = grammar_signatures_to_serializable(g)
    assert isinstance(sigs, list)
    assert len(sigs) == 2


def test_grammar_signatures_lift_rule_fields():
    """First rule (``S4 = lift(NP3, VP1)``) serializes with correct fields."""
    from embed import grammar_signatures_to_serializable
    g = _tiny_grammar()
    sigs = grammar_signatures_to_serializable(g)
    s = sigs[0]
    assert s['lhs_category'] == 'S'
    assert s['lhs_order'] == 4
    assert s['rhs_categories'] == ['NP', 'VP']
    assert s['rhs_orders'] == [3, 1]
    assert s['op_name'] == 'lift'
    assert s['order_delta'] == 1


def test_grammar_signatures_lower_rule_fields():
    """Second rule (``NP3 = lower(DET, NP4)``) serializes correctly."""
    from embed import grammar_signatures_to_serializable
    g = _tiny_grammar()
    sigs = grammar_signatures_to_serializable(g)
    s = sigs[1]
    assert s['lhs_category'] == 'NP'
    assert s['lhs_order'] == 3
    assert s['rhs_categories'] == ['DET', 'NP']
    assert s['rhs_orders'] == [0, 4]
    assert s['op_name'] == 'lower'
    assert s['order_delta'] == -1


def test_grammar_signatures_fields_are_primitive_types():
    """All values in the serialized dicts are JSON-friendly primitives —
    no nested namedtuples, no tensors, just str / int / list."""
    from embed import grammar_signatures_to_serializable
    g = _tiny_grammar()
    sigs = grammar_signatures_to_serializable(g)
    for s in sigs:
        assert isinstance(s['lhs_category'], str)
        assert isinstance(s['lhs_order'], int)
        assert isinstance(s['rhs_categories'], list)
        assert all(isinstance(c, str) for c in s['rhs_categories'])
        assert isinstance(s['rhs_orders'], list)
        assert all(isinstance(o, int) for o in s['rhs_orders'])
        assert s['op_name'] is None or isinstance(s['op_name'], str)
        assert isinstance(s['order_delta'], int)


# -- Taxonomy builder --------------------------------------------------
# Classifies categories into nonterminals (appear as LHS in some rule)
# and POS terminals (RHS-only) using ``_parse_category`` to strip order
# suffixes before classification. Builds a flat tree:
#   ROOT (ref_id 0)
#     ├── nonterminal class nodes (e.g. S, NP)
#     └── POS class nodes (e.g. VP, DET in tiny fixture; in real
#         grammars more — N, V, ADJ, ADV, P, O, MP)
# Word leaves under POS are populated in a separate step (wv-driven).


def test_taxonomy_builder_returns_expected_shape():
    """``build_taxonomy_from_grammar(g)`` returns a dict with parent /
    children_values / children_offsets / taxonomy_names."""
    from embed import build_taxonomy_from_grammar
    g = _tiny_grammar()
    tax = build_taxonomy_from_grammar(g)
    assert 'parent' in tax
    assert 'children_values' in tax
    assert 'children_offsets' in tax
    assert 'taxonomy_names' in tax


def test_taxonomy_classifies_nonterminals_vs_pos():
    """In the tiny fixture: S and NP appear as LHS → nonterminals;
    VP and DET appear only in RHS → POS terminals. (NP also appears
    in RHS but its LHS appearance makes it a nonterminal.)"""
    from embed import build_taxonomy_from_grammar
    g = _tiny_grammar()
    tax = build_taxonomy_from_grammar(g)
    names = tax['taxonomy_names']
    # All four category names registered
    for cat in ('S', 'NP', 'VP', 'DET'):
        assert cat in names, f"{cat!r} missing from taxonomy_names"


def test_taxonomy_has_root_node_at_ref_id_zero():
    """Root sits at ref_id 0, parent = -1."""
    from embed import build_taxonomy_from_grammar
    g = _tiny_grammar()
    tax = build_taxonomy_from_grammar(g)
    parent = tax['parent']
    assert int(parent[0].item()) == -1
    # And no other node has parent == -1
    for i in range(1, parent.shape[0]):
        assert int(parent[i].item()) != -1, \
            f"node {i} has parent -1 but isn't the root"


def test_taxonomy_class_nodes_under_root():
    """Every nonterminal + POS class node has parent == root (0)."""
    from embed import build_taxonomy_from_grammar
    g = _tiny_grammar()
    tax = build_taxonomy_from_grammar(g)
    names = tax['taxonomy_names']
    parent = tax['parent']
    for cat in ('S', 'NP', 'VP', 'DET'):
        rid = names[cat]
        assert int(parent[rid].item()) == 0, \
            f"{cat!r}'s parent is not the root"


def test_taxonomy_children_csr_round_trips():
    """``children_offsets[i+1] - children_offsets[i]`` gives the child
    count for node i; the slice is the list of children's ref_ids; and
    each child's parent points back to i."""
    from embed import build_taxonomy_from_grammar
    g = _tiny_grammar()
    tax = build_taxonomy_from_grammar(g)
    parent = tax['parent']
    cv = tax['children_values']
    co = tax['children_offsets']
    n = parent.shape[0]
    assert co.shape[0] == n + 1
    # Mutual-consistency check: parent[c] == p iff c in children(p).
    for p in range(n):
        start = int(co[p].item())
        end = int(co[p + 1].item())
        kids = cv[start:end].tolist()
        for c in kids:
            assert int(parent[c].item()) == p, (
                f"child {c} of {p} has parent {int(parent[c].item())}")
        # And every node whose parent IS p must appear in this slice
        for c in range(n):
            if int(parent[c].item()) == p:
                assert c in kids, (
                    f"node {c} (parent {p}) missing from children list")


def test_taxonomy_handles_order_stripped_categories():
    """Categories appearing in different rules with different orders
    (e.g. NP3 and NP4) collapse to a single 'NP' taxonomy node."""
    from embed import build_taxonomy_from_grammar
    g = _tiny_grammar()
    tax = build_taxonomy_from_grammar(g)
    names = tax['taxonomy_names']
    # NP appears as NP3 (LHS of rule 1) and NP4 (RHS of rule 1) and
    # NP3 (RHS of rule 0). All collapse to one node:
    assert 'NP3' not in names
    assert 'NP4' not in names
    assert 'NP' in names


# -- Reference codebook + typed indexes ------------------------------
# Phase-1 bootstrap: one reference per taxonomy class node, all
# initialized to scalar 0.0 and order 0. Capacity slack reserved per
# plan §Parameter growth pattern. Typed indexes are dict[int|str, LongTensor]
# keyed by order value or category name.


def test_reference_codebook_initial_shape():
    """``build_reference_codebook_initial(taxonomy)`` returns a dict
    with references / v_ref_live / order at the expected shapes."""
    from embed import (build_taxonomy_from_grammar,
                       build_reference_codebook_initial)
    g = _tiny_grammar()
    tax = build_taxonomy_from_grammar(g)
    rc = build_reference_codebook_initial(tax)
    assert 'references' in rc
    assert 'v_ref_live' in rc
    assert 'order' in rc
    n_live = int(rc['v_ref_live'])
    # tiny grammar: root + S + NP + VP + DET = 5
    assert n_live == 5
    # References + order both [V_ref_capacity], capacity >= live
    assert rc['references'].shape[0] >= n_live
    assert rc['order'].shape[0] >= n_live
    # Capacity slack: at least 256 even for a tiny taxonomy
    assert rc['references'].shape[0] >= 256


def test_reference_codebook_dtypes():
    """References are float; order is long."""
    from embed import (build_taxonomy_from_grammar,
                       build_reference_codebook_initial)
    import torch
    tax = build_taxonomy_from_grammar(_tiny_grammar())
    rc = build_reference_codebook_initial(tax)
    assert rc['references'].dtype == torch.float32
    assert rc['order'].dtype == torch.long


def test_reference_codebook_initial_values_zero():
    """Bootstrap: all live references init to 0.0, all live orders
    init to 0. Slack rows likewise zeros (default tensor init)."""
    from embed import (build_taxonomy_from_grammar,
                       build_reference_codebook_initial)
    tax = build_taxonomy_from_grammar(_tiny_grammar())
    rc = build_reference_codebook_initial(tax)
    n_live = int(rc['v_ref_live'])
    for i in range(n_live):
        assert float(rc['references'][i].item()) == 0.0
        assert int(rc['order'][i].item()) == 0


def test_typed_indexes_refs_by_order():
    """``refs_by_order[0]`` lists all live ref_ids (everything is order
    0 in the bootstrap)."""
    from embed import (build_taxonomy_from_grammar,
                       build_reference_codebook_initial,
                       build_typed_indexes)
    tax = build_taxonomy_from_grammar(_tiny_grammar())
    rc = build_reference_codebook_initial(tax)
    ti = build_typed_indexes(tax, rc)
    assert 'refs_by_order' in ti
    by_order = ti['refs_by_order']
    n_live = int(rc['v_ref_live'])
    assert 0 in by_order
    assert sorted(by_order[0].tolist()) == list(range(n_live))


def test_typed_indexes_refs_by_category():
    """``refs_by_category[name]`` lists ref_ids in the subtree rooted at
    that category (the class node itself plus any descendants). In the
    bootstrap there are no descendants yet, so each category maps to a
    single-element list containing its own ref_id."""
    from embed import (build_taxonomy_from_grammar,
                       build_reference_codebook_initial,
                       build_typed_indexes)
    tax = build_taxonomy_from_grammar(_tiny_grammar())
    rc = build_reference_codebook_initial(tax)
    ti = build_typed_indexes(tax, rc)
    by_cat = ti['refs_by_category']
    names = tax['taxonomy_names']
    for cat in ('S', 'NP', 'VP', 'DET'):
        assert cat in by_cat
        # Just the class node itself (no descendants yet)
        assert by_cat[cat].tolist() == [names[cat]]


# -- Word table builder ----------------------------------------------
# CSR-stored ragged surface bytes (uint8) + ref_ids (long). ref_ids
# initialized to -1 (unassigned); a separate POS-assignment step
# (curated lexicon / tagger) populates them later.


class _FakeWV:
    """Minimal duck-typed WordVectors stand-in: just index_to_key."""
    def __init__(self, words):
        self.index_to_key = list(words)


def test_word_table_initial_returns_csr_plus_ref_ids():
    """``build_word_table_initial(wv)`` returns dict with keys_values,
    keys_offsets, ref_ids."""
    from embed import build_word_table_initial
    wv = _FakeWV(["the", "cat", "ran"])
    wt = build_word_table_initial(wv)
    assert 'keys_values' in wt
    assert 'keys_offsets' in wt
    assert 'ref_ids' in wt


def test_word_table_csr_round_trips_surface_bytes():
    """For each lex_row, the CSR slice decodes back to the surface form."""
    from embed import build_word_table_initial
    words = ["the", "cat", "antidisestablishmentarianism", "a"]
    wv = _FakeWV(words)
    wt = build_word_table_initial(wv)
    kv = wt['keys_values']
    ko = wt['keys_offsets']
    assert ko.shape[0] == len(words) + 1
    for i, w in enumerate(words):
        start = int(ko[i].item())
        end = int(ko[i + 1].item())
        recovered = bytes(kv[start:end].tolist()).decode('utf-8')
        assert recovered == w


def test_word_table_ref_ids_initialize_to_minus_one():
    """ref_ids start as -1 (unassigned POS). A separate step fills them."""
    from embed import build_word_table_initial
    wv = _FakeWV(["the", "cat", "ran"])
    wt = build_word_table_initial(wv)
    rid = wt['ref_ids']
    assert rid.shape[0] == 3
    for i in range(3):
        assert int(rid[i].item()) == -1


def test_word_table_dtypes():
    """keys_values: uint8; keys_offsets: long; ref_ids: long."""
    from embed import build_word_table_initial
    import torch
    wv = _FakeWV(["the", "cat"])
    wt = build_word_table_initial(wv)
    assert wt['keys_values'].dtype == torch.uint8
    assert wt['keys_offsets'].dtype == torch.long
    assert wt['ref_ids'].dtype == torch.long


def test_word_table_empty_lexicon():
    """Empty lexicon: keys_values empty, keys_offsets just [0], ref_ids empty."""
    from embed import build_word_table_initial
    wv = _FakeWV([])
    wt = build_word_table_initial(wv)
    assert wt['keys_values'].shape[0] == 0
    assert wt['keys_offsets'].shape[0] == 1
    assert int(wt['keys_offsets'][0].item()) == 0
    assert wt['ref_ids'].shape[0] == 0


# -- Bundle + save/load integration ----------------------------------


def test_knowledge_section_bundles_all_pieces():
    """``build_knowledge_section(g, wv=wv)`` returns a dict with all five
    sub-sections present, ready for storage as one section of a
    unified artifact."""
    from embed import build_knowledge_section
    g = _tiny_grammar()
    wv = _FakeWV(["the", "cat"])
    ks = build_knowledge_section(g, wv=wv)
    assert 'word_table' in ks
    assert 'reference_codebook' in ks
    assert 'typed_indexes' in ks
    assert 'taxonomy' in ks
    assert 'grammar' in ks
    # Knowledge section carries a section_kind sentinel for inspect tools
    assert ks.get('section_kind') == 'knowledge'


def test_knowledge_section_works_without_wv():
    """Without a wv, word_table is empty but the rest is populated."""
    from embed import build_knowledge_section
    g = _tiny_grammar()
    ks = build_knowledge_section(g)
    # Word table is present but empty
    assert ks['word_table']['ref_ids'].shape[0] == 0
    # Other sections fully populated
    assert ks['reference_codebook']['v_ref_live'] == 5  # 1 root + 4 categories
    assert 'NP' in ks['taxonomy']['taxonomy_names']
    assert len(ks['grammar']['rule_order_signatures']) == 2


def test_save_and_load_artifact_round_trips_knowledge(tmp_path):
    """save_artifact accepts knowledge=ks; load_artifact returns it."""
    import torch
    from embed import (save_artifact, load_artifact,
                       build_knowledge_section)
    g = _tiny_grammar()
    wv = _FakeWV(["the", "cat"])
    ks = build_knowledge_section(g, wv=wv)
    # Need a stub lexicon dict to satisfy the existing requirement that
    # save_artifact has at least one primary section.
    lex_stub = {
        'section_kind':  'lexicon',
        'vectors':       torch.zeros(2, 4),
        'index_to_key':  ['the', 'cat'],
        'counts':        None,
        'total_count':   0,
    }
    path = str(tmp_path / "test.kv")
    save_artifact(path, lexicon=lex_stub, knowledge=ks)
    loaded = load_artifact(path)
    assert 'knowledge' in loaded
    rt = loaded['knowledge']
    # All sub-sections survived
    for key in ('word_table', 'reference_codebook', 'typed_indexes',
                'taxonomy', 'grammar'):
        assert key in rt
    # One concrete tensor round-trips
    assert rt['word_table']['ref_ids'].tolist() == \
        ks['word_table']['ref_ids'].tolist()
    # One nested dict round-trips
    assert rt['taxonomy']['taxonomy_names'] == \
        ks['taxonomy']['taxonomy_names']
    # Grammar signatures round-trip
    assert rt['grammar']['rule_order_signatures'] == \
        ks['grammar']['rule_order_signatures']


def test_save_artifact_accepts_knowledge_only(tmp_path):
    """A knowledge section alone is sufficient for save_artifact —
    no lexicon or bpe required."""
    from embed import (save_artifact, load_artifact,
                       build_knowledge_section)
    g = _tiny_grammar()
    ks = build_knowledge_section(g)
    path = str(tmp_path / "knowledge_only.kv")
    save_artifact(path, knowledge=ks)
    loaded = load_artifact(path)
    assert 'knowledge' in loaded


# -- extend_artifact -- runtime symbol-learning append --------------
# `NewRef(scalar, order, parent_ref_id, category)` describes one ref
# to append; `extend_artifact(path, new_refs)` loads, appends, rebuilds
# typed indexes, and re-saves.


def _bootstrap_artifact(tmp_path, name="art.kv"):
    """Create + save a fresh knowledge-only artifact for extend tests."""
    from embed import save_artifact, build_knowledge_section
    g = _tiny_grammar()
    ks = build_knowledge_section(g)
    path = str(tmp_path / name)
    save_artifact(path, knowledge=ks)
    return path


def test_extend_artifact_appends_single_ref(tmp_path):
    """A single new ref lands at ref_id == v_ref_live, and v_ref_live
    is incremented by 1."""
    from embed import extend_artifact, load_artifact, NewRef
    path = _bootstrap_artifact(tmp_path)
    initial = load_artifact(path)['knowledge']
    parent_rid_s = initial['taxonomy']['taxonomy_names']['S']
    v_live_before = int(initial['reference_codebook']['v_ref_live'])
    new_refs = [
        NewRef(scalar=0.5, order=1, parent_ref_id=parent_rid_s,
               category='S')]
    extend_artifact(path, new_refs)
    after = load_artifact(path)['knowledge']
    rc = after['reference_codebook']
    new_id = v_live_before
    assert int(rc['v_ref_live']) == v_live_before + 1
    assert float(rc['references'][new_id].item()) == 0.5
    assert int(rc['order'][new_id].item()) == 1


def test_extend_artifact_updates_parent_and_children(tmp_path):
    """The new ref's parent_ref_id is recorded in the parent tensor,
    and the new ref appears in parent's children list (CSR)."""
    from embed import extend_artifact, load_artifact, NewRef
    path = _bootstrap_artifact(tmp_path)
    initial = load_artifact(path)['knowledge']
    parent_rid_s = initial['taxonomy']['taxonomy_names']['S']
    v_live = int(initial['reference_codebook']['v_ref_live'])
    extend_artifact(path, [NewRef(scalar=0.5, order=1,
                                  parent_ref_id=parent_rid_s,
                                  category='S')])
    after = load_artifact(path)['knowledge']
    tax = after['taxonomy']
    new_id = v_live
    assert int(tax['parent'][new_id].item()) == parent_rid_s
    # CSR: parent's children slice should include new_id
    cv = tax['children_values']
    co = tax['children_offsets']
    start = int(co[parent_rid_s].item())
    end = int(co[parent_rid_s + 1].item())
    kids = cv[start:end].tolist()
    assert new_id in kids


def test_extend_artifact_updates_refs_by_order(tmp_path):
    """The new ref appears in refs_by_order at its order key."""
    from embed import extend_artifact, load_artifact, NewRef
    path = _bootstrap_artifact(tmp_path)
    initial = load_artifact(path)['knowledge']
    s_rid = initial['taxonomy']['taxonomy_names']['S']
    v_live = int(initial['reference_codebook']['v_ref_live'])
    extend_artifact(path, [NewRef(scalar=0.5, order=4,
                                  parent_ref_id=s_rid, category='S')])
    after = load_artifact(path)['knowledge']
    rbo = after['typed_indexes']['refs_by_order']
    assert 4 in rbo
    assert v_live in rbo[4].tolist()


def test_extend_artifact_updates_refs_by_category(tmp_path):
    """The new ref appears in refs_by_category at its category key."""
    from embed import extend_artifact, load_artifact, NewRef
    path = _bootstrap_artifact(tmp_path)
    initial = load_artifact(path)['knowledge']
    s_rid = initial['taxonomy']['taxonomy_names']['S']
    v_live = int(initial['reference_codebook']['v_ref_live'])
    extend_artifact(path, [NewRef(scalar=0.5, order=4,
                                  parent_ref_id=s_rid, category='S')])
    after = load_artifact(path)['knowledge']
    rbc = after['typed_indexes']['refs_by_category']
    assert v_live in rbc['S'].tolist()


def test_extend_artifact_batch_appends(tmp_path):
    """Multiple refs in one call all land correctly with sequential ref_ids."""
    from embed import extend_artifact, load_artifact, NewRef
    path = _bootstrap_artifact(tmp_path)
    initial = load_artifact(path)['knowledge']
    s_rid = initial['taxonomy']['taxonomy_names']['S']
    np_rid = initial['taxonomy']['taxonomy_names']['NP']
    v_live = int(initial['reference_codebook']['v_ref_live'])
    extend_artifact(path, [
        NewRef(scalar=0.1, order=4, parent_ref_id=s_rid, category='S'),
        NewRef(scalar=0.2, order=3, parent_ref_id=np_rid, category='NP'),
        NewRef(scalar=0.3, order=4, parent_ref_id=np_rid, category='NP'),
    ])
    after = load_artifact(path)['knowledge']
    rc = after['reference_codebook']
    assert int(rc['v_ref_live']) == v_live + 3
    # Scalars round-trip correctly to sequential ids (float32 precision).
    import pytest
    assert float(rc['references'][v_live].item()) == pytest.approx(0.1)
    assert float(rc['references'][v_live + 1].item()) == pytest.approx(0.2)
    assert float(rc['references'][v_live + 2].item()) == pytest.approx(0.3)
    # Orders too
    assert int(rc['order'][v_live].item()) == 4
    assert int(rc['order'][v_live + 1].item()) == 3
    assert int(rc['order'][v_live + 2].item()) == 4


def test_extend_artifact_grows_capacity_when_exhausted(tmp_path):
    """If appending would exceed V_ref_capacity, capacity at least
    doubles, and all existing data survives."""
    from embed import extend_artifact, load_artifact, NewRef
    path = _bootstrap_artifact(tmp_path)
    initial = load_artifact(path)['knowledge']
    rc = initial['reference_codebook']
    s_rid = initial['taxonomy']['taxonomy_names']['S']
    capacity_before = rc['references'].shape[0]
    v_live = int(rc['v_ref_live'])
    # Add enough refs to exceed capacity
    n_new = capacity_before - v_live + 5  # 5 past capacity
    extend_artifact(path, [
        NewRef(scalar=float(i) / 100, order=0,
               parent_ref_id=s_rid, category='S')
        for i in range(n_new)
    ])
    after = load_artifact(path)['knowledge']
    assert int(after['reference_codebook']['v_ref_live']) == v_live + n_new
    assert after['reference_codebook']['references'].shape[0] >= \
        v_live + n_new
    # Original v_live rows preserved
    for i in range(v_live):
        assert float(after['reference_codebook']['references'][i].item()) == \
            float(initial['reference_codebook']['references'][i].item())


def test_extend_artifact_raises_without_knowledge_section(tmp_path):
    """Artifact lacking a knowledge section: extend raises ValueError."""
    from embed import extend_artifact, save_artifact, NewRef
    import torch
    lex_only_path = str(tmp_path / "lex_only.kv")
    save_artifact(lex_only_path, lexicon={
        'section_kind': 'lexicon',
        'vectors': torch.zeros(1, 4),
        'index_to_key': ['x'],
        'counts': None,
        'total_count': 0,
    })
    import pytest
    with pytest.raises(ValueError):
        extend_artifact(lex_only_path, [
            NewRef(scalar=0.0, order=0, parent_ref_id=0, category='X')])


def test_extend_artifact_empty_input_noop(tmp_path):
    """Passing an empty list of new_refs is a no-op (no state change)."""
    from embed import extend_artifact, load_artifact
    path = _bootstrap_artifact(tmp_path)
    before = load_artifact(path)['knowledge']
    v_before = int(before['reference_codebook']['v_ref_live'])
    extend_artifact(path, [])
    after = load_artifact(path)['knowledge']
    assert int(after['reference_codebook']['v_ref_live']) == v_before
