"""Tests for SymbolicSpace.attach_knowledge — wiring a loaded KnowledgeView
into the runtime SymbolicSpace.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 — Loaders.

Uses ``object.__new__`` to bypass SymbolicSpace's heavy __init__ (which
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
        g._parse_rule("S4", "lift(NP3, VP1)", tier='S'),
        g._parse_rule("NP3", "lower(DET, NP4)", tier='S'),
    ]
    g._configured = True
    return g


def _tiny_view():
    from embed import build_knowledge_section, KnowledgeView
    return KnowledgeView(build_knowledge_section(_tiny_grammar()))


def _bare_word_space():
    """A SymbolicSpace instance with __init__ bypassed — just enough for
    attach_knowledge tests."""
    from Language import SymbolicSubSpace
    import torch.nn as nn
    ss = object.__new__(SymbolicSubSpace)
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


def test_symbolic_space_attach_creates_references_parameter():
    """After attach_knowledge, WholeSpace has a 1-D references
    Parameter at the artifact's capacity, initialized with the live
    rows from the artifact."""
    from Spaces import WholeSpace
    import torch.nn as nn
    ws = _bare_space(WholeSpace)
    view = _tiny_view()
    ws.attach_knowledge(view)
    assert hasattr(ws, 'references')
    assert isinstance(ws.references, nn.Parameter)
    # 1-D, sized to capacity (>= 256 per the bootstrap slack)
    assert ws.references.dim() == 1
    assert ws.references.shape[0] >= 256
    # Live rows match the view's values
    n_live = view.n_refs_live
    for i in range(n_live):
        assert float(ws.references[i].item()) == \
            float(view.references[i].item())


def test_symbolic_space_attach_creates_order_buffer():
    """After attach_knowledge, WholeSpace has an ``order`` long
    buffer with the live rows from the artifact."""
    from Spaces import WholeSpace
    import torch
    ws = _bare_space(WholeSpace)
    view = _tiny_view()
    ws.attach_knowledge(view)
    assert hasattr(ws, 'order')
    assert ws.order.dtype == torch.long
    n_live = view.n_refs_live
    for i in range(n_live):
        assert int(ws.order[i].item()) == int(view.orders[i].item())


def test_symbolic_space_references_in_named_parameters():
    """The references Parameter shows up in ``named_parameters`` so the
    optimizer picks it up for training."""
    from Spaces import WholeSpace
    ws = _bare_space(WholeSpace)
    ws.attach_knowledge(_tiny_view())
    names = [n for n, _ in ws.named_parameters()]
    assert 'references' in names


def test_symbolic_space_order_in_named_buffers():
    """The order tensor shows up in ``named_buffers`` (not as a
    Parameter — it's discrete metadata, not trainable)."""
    from Spaces import WholeSpace
    ws = _bare_space(WholeSpace)
    ws.attach_knowledge(_tiny_view())
    names = [n for n, _ in ws.named_buffers()]
    assert 'order' in names


def test_symbolic_space_reattach_updates_in_place():
    """Re-attaching a view replaces the references / order data
    without breaking Parameter / buffer registration."""
    from Spaces import WholeSpace
    import torch
    ws = _bare_space(WholeSpace)
    view1 = _tiny_view()
    ws.attach_knowledge(view1)
    refs_id_before = id(ws.references)
    # Modify the first ref's value via the underlying section, then
    # build a fresh view to attach.
    view2 = _tiny_view()
    # The bootstrap builds with all zeros — to detect update, write
    # something into view2's underlying section before attaching.
    view2._ks['reference_codebook']['references'][0] = 0.75
    ws.attach_knowledge(view2)
    # New value visible
    assert float(ws.references[0].item()) == 0.75
    # Knowledge field updated too
    assert ws.knowledge is view2


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
