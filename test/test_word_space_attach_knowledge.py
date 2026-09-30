"""Tests for SymbolSpace.attach_knowledge — wiring a loaded KnowledgeView
into the runtime SymbolSpace.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 — Loaders.

Uses ``object.__new__`` to bypass SymbolSpace's heavy __init__ (which
requires PartSpace / ConceptualSpace / WholeSpace). We only
need to verify the attach mechanics, not the full Space wiring.
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def _tiny_grammar():
    from Language import Grammar
    g = Grammar()
    g.rules = [
        g._parse_rule("S4", "lift(NP3, VP1)", space_role='SS'),
        g._parse_rule("NP3", "lower(DET, NP4)", space_role='SS'),
    ]
    g._configured = True
    return g


def _tiny_view():
    from embed import build_knowledge_section, KnowledgeView
    return KnowledgeView(build_knowledge_section(_tiny_grammar()))


def _bare_word_space():
    """A SymbolSpace instance with __init__ bypassed — just enough for
    attach_knowledge tests."""
    from Language import SymbolSubSpace
    import torch.nn as nn
    ss = object.__new__(SymbolSubSpace)
    nn.Module.__init__(ss)
    return ss


def test_attach_knowledge_stores_view():
    """After ``attach_knowledge(view)``, ``ss.knowledge`` returns it."""
    ss = _bare_word_space()
    view = _tiny_view()
    ss.attach_knowledge(view)
    assert ss.knowledge is view


def test_knowledge_is_none_before_attach():
    """Before any attach_knowledge call, ``ss.knowledge`` is None."""
    ss = _bare_word_space()
    assert ss.knowledge is None


def test_reattach_replaces_previous_view():
    """A second attach replaces the first."""
    ss = _bare_word_space()
    view1 = _tiny_view()
    view2 = _tiny_view()
    ss.attach_knowledge(view1)
    ss.attach_knowledge(view2)
    assert ss.knowledge is view2


def test_knowledge_view_queries_through_word_space():
    """Once attached, all KnowledgeView queries work via ``ss.knowledge``.

    Updated 2026-05-20 for the order-typed taxonomy: NP now subsumes
    base + ordered variants (NP3, NP4) — 3 refs total.
    """
    ss = _bare_word_space()
    ss.attach_knowledge(_tiny_view())
    assert ss.knowledge.ref_id_for('NP') is not None
    assert ss.knowledge.refs_by_category('NP').shape[0] == 3
    assert len(ss.knowledge.rule_order_signatures) == 2


# -- The same attach pattern is inherited by every Space subclass -----
# (PartSpace, WholeSpace, etc.) via the Space base class.


def _bare_space(cls):
    """Bypass __init__ for any Space subclass (saves heavy XML setup
    when all we want is the attach plumbing)."""
    import torch.nn as nn
    inst = object.__new__(cls)
    nn.Module.__init__(inst)
    return inst


def test_perceptual_space_inherits_attach_knowledge():
    """PartSpace inherits attach_knowledge from Space."""
    from Spaces import PartSpace
    ps = _bare_space(PartSpace)
    assert ps.knowledge is None
    view = _tiny_view()
    ps.attach_knowledge(view)
    assert ps.knowledge is view


def test_symbolic_space_inherits_attach_knowledge():
    """WholeSpace inherits attach_knowledge from Space."""
    from Spaces import WholeSpace
    ws = _bare_space(WholeSpace)
    assert ws.knowledge is None
    view = _tiny_view()
    ws.attach_knowledge(view)
    assert ws.knowledge is view


# -- WholeSpace bootstraps trainable references on attach -----------
# Plan §Phase 2 — Loaders + bivector retirement (narrow scope). The
# scalar reference codebook from the artifact lands on WholeSpace
# as an ``nn.Parameter`` (trainable) plus an ``order`` long buffer.












# -- PartSpace attach populates wv.ref_ids -----------------------
# Plan §Phase 2 — Loaders. wv owns surface forms; attach_knowledge
# stamps the artifact's ref_ids (the foreign keys into the reference
# codebook) onto wv so the chart's lexical-lookup step can navigate
# word→reference.


class _FakeWV:
    def __init__(self, words):
        self.index_to_key = list(words)


def test_perceptual_space_attach_sets_wv_ref_ids():
    """After attach, ``ps.wv.ref_ids`` carries the artifact's word_table
    ref_ids (initialized to -1 in the Phase-1 bootstrap)."""
    from Spaces import PartSpace
    from embed import build_knowledge_section, KnowledgeView
    import torch
    ps = _bare_space(PartSpace)
    ps.wv = _FakeWV(['the', 'cat', 'ran'])
    ks = build_knowledge_section(_tiny_grammar(), wv=ps.wv)
    ps.attach_knowledge(KnowledgeView(ks))
    assert hasattr(ps.wv, 'ref_ids')
    assert ps.wv.ref_ids.shape[0] == 3
    # Phase-1 bootstrap: unassigned POS, all -1
    for i in range(3):
        assert int(ps.wv.ref_ids[i].item()) == -1


def test_perceptual_space_attach_without_wv_is_noop():
    """When ps.wv is absent, attach just stores the view — no error."""
    from Spaces import PartSpace
    from embed import build_knowledge_section, KnowledgeView
    ps = _bare_space(PartSpace)
    # No ps.wv attribute
    view = KnowledgeView(build_knowledge_section(_tiny_grammar()))
    ps.attach_knowledge(view)
    # Knowledge is still attached; no error raised
    assert ps.knowledge is view


def test_perceptual_space_reattach_updates_ref_ids():
    """Re-attach overwrites wv.ref_ids with the new artifact's values."""
    from Spaces import PartSpace
    from embed import (build_knowledge_section, KnowledgeView)
    import torch
    ps = _bare_space(PartSpace)
    ps.wv = _FakeWV(['the', 'cat'])
    ks1 = build_knowledge_section(_tiny_grammar(), wv=ps.wv)
    ps.attach_knowledge(KnowledgeView(ks1))
    # Now mutate the underlying ref_ids and re-attach.
    ks2 = build_knowledge_section(_tiny_grammar(), wv=ps.wv)
    ks2['word_table']['ref_ids'][0] = 5  # specific value, not -1
    ps.attach_knowledge(KnowledgeView(ks2))
    assert int(ps.wv.ref_ids[0].item()) == 5
