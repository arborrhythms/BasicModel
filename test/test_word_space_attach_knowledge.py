"""Tests for WordSpace.attach_knowledge — wiring a loaded KnowledgeView
into the runtime WordSpace.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 — Loaders.

Uses ``object.__new__`` to bypass WordSpace's heavy __init__ (which
requires PerceptualSpace / ConceptualSpace / SymbolicSpace). We only
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
    """A WordSpace instance with __init__ bypassed — just enough for
    attach_knowledge tests."""
    from Language import WordSpace
    import torch.nn as nn
    ws = object.__new__(WordSpace)
    nn.Module.__init__(ws)
    return ws


def test_attach_knowledge_stores_view():
    """After ``attach_knowledge(view)``, ``ws.knowledge`` returns it."""
    ws = _bare_word_space()
    view = _tiny_view()
    ws.attach_knowledge(view)
    assert ws.knowledge is view


def test_knowledge_is_none_before_attach():
    """Before any attach_knowledge call, ``ws.knowledge`` is None."""
    ws = _bare_word_space()
    assert ws.knowledge is None


def test_reattach_replaces_previous_view():
    """A second attach replaces the first."""
    ws = _bare_word_space()
    view1 = _tiny_view()
    view2 = _tiny_view()
    ws.attach_knowledge(view1)
    ws.attach_knowledge(view2)
    assert ws.knowledge is view2


def test_knowledge_view_queries_through_word_space():
    """Once attached, all KnowledgeView queries work via ``ws.knowledge``.

    Updated 2026-05-20 for the order-typed taxonomy: NP now subsumes
    base + ordered variants (NP3, NP4) — 3 refs total.
    """
    ws = _bare_word_space()
    ws.attach_knowledge(_tiny_view())
    assert ws.knowledge.ref_id_for('NP') is not None
    assert ws.knowledge.refs_by_category('NP').shape[0] == 3
    assert len(ws.knowledge.rule_order_signatures) == 2


# -- The same attach pattern is inherited by every Space subclass -----
# (PerceptualSpace, SymbolicSpace, etc.) via the Space base class.


def _bare_space(cls):
    """Bypass __init__ for any Space subclass (saves heavy XML setup
    when all we want is the attach plumbing)."""
    import torch.nn as nn
    inst = object.__new__(cls)
    nn.Module.__init__(inst)
    return inst


def test_perceptual_space_inherits_attach_knowledge():
    """PerceptualSpace inherits attach_knowledge from Space."""
    from Spaces import PerceptualSpace
    ps = _bare_space(PerceptualSpace)
    assert ps.knowledge is None
    view = _tiny_view()
    ps.attach_knowledge(view)
    assert ps.knowledge is view


def test_symbolic_space_inherits_attach_knowledge():
    """SymbolicSpace inherits attach_knowledge from Space."""
    from Spaces import SymbolicSpace
    ss = _bare_space(SymbolicSpace)
    assert ss.knowledge is None
    view = _tiny_view()
    ss.attach_knowledge(view)
    assert ss.knowledge is view


# -- SymbolicSpace bootstraps trainable references on attach -----------
# Plan §Phase 2 — Loaders + bivector retirement (narrow scope). The
# scalar reference codebook from the artifact lands on SymbolicSpace
# as an ``nn.Parameter`` (trainable) plus an ``order`` long buffer.


def test_symbolic_space_attach_creates_references_parameter():
    """After attach_knowledge, SymbolicSpace has a 1-D references
    Parameter at the artifact's capacity, initialized with the live
    rows from the artifact."""
    from Spaces import SymbolicSpace
    import torch.nn as nn
    ss = _bare_space(SymbolicSpace)
    view = _tiny_view()
    ss.attach_knowledge(view)
    assert hasattr(ss, 'references')
    assert isinstance(ss.references, nn.Parameter)
    # 1-D, sized to capacity (>= 256 per the bootstrap slack)
    assert ss.references.dim() == 1
    assert ss.references.shape[0] >= 256
    # Live rows match the view's values
    n_live = view.n_refs_live
    for i in range(n_live):
        assert float(ss.references[i].item()) == \
            float(view.references[i].item())


def test_symbolic_space_attach_creates_order_buffer():
    """After attach_knowledge, SymbolicSpace has an ``order`` long
    buffer with the live rows from the artifact."""
    from Spaces import SymbolicSpace
    import torch
    ss = _bare_space(SymbolicSpace)
    view = _tiny_view()
    ss.attach_knowledge(view)
    assert hasattr(ss, 'order')
    assert ss.order.dtype == torch.long
    n_live = view.n_refs_live
    for i in range(n_live):
        assert int(ss.order[i].item()) == int(view.orders[i].item())


def test_symbolic_space_references_in_named_parameters():
    """The references Parameter shows up in ``named_parameters`` so the
    optimizer picks it up for training."""
    from Spaces import SymbolicSpace
    ss = _bare_space(SymbolicSpace)
    ss.attach_knowledge(_tiny_view())
    names = [n for n, _ in ss.named_parameters()]
    assert 'references' in names


def test_symbolic_space_order_in_named_buffers():
    """The order tensor shows up in ``named_buffers`` (not as a
    Parameter — it's discrete metadata, not trainable)."""
    from Spaces import SymbolicSpace
    ss = _bare_space(SymbolicSpace)
    ss.attach_knowledge(_tiny_view())
    names = [n for n, _ in ss.named_buffers()]
    assert 'order' in names


def test_symbolic_space_reattach_updates_in_place():
    """Re-attaching a view replaces the references / order data
    without breaking Parameter / buffer registration."""
    from Spaces import SymbolicSpace
    import torch
    ss = _bare_space(SymbolicSpace)
    view1 = _tiny_view()
    ss.attach_knowledge(view1)
    refs_id_before = id(ss.references)
    # Modify the first ref's value via the underlying section, then
    # build a fresh view to attach.
    view2 = _tiny_view()
    # The bootstrap builds with all zeros — to detect update, write
    # something into view2's underlying section before attaching.
    view2._ks['reference_codebook']['references'][0] = 0.75
    ss.attach_knowledge(view2)
    # New value visible
    assert float(ss.references[0].item()) == 0.75
    # Knowledge field updated too
    assert ss.knowledge is view2


# -- PerceptualSpace attach populates wv.ref_ids -----------------------
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
    from Spaces import PerceptualSpace
    from embed import build_knowledge_section, KnowledgeView
    import torch
    ps = _bare_space(PerceptualSpace)
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
    from Spaces import PerceptualSpace
    from embed import build_knowledge_section, KnowledgeView
    ps = _bare_space(PerceptualSpace)
    # No ps.wv attribute
    view = KnowledgeView(build_knowledge_section(_tiny_grammar()))
    ps.attach_knowledge(view)
    # Knowledge is still attached; no error raised
    assert ps.knowledge is view


def test_perceptual_space_reattach_updates_ref_ids():
    """Re-attach overwrites wv.ref_ids with the new artifact's values."""
    from Spaces import PerceptualSpace
    from embed import (build_knowledge_section, KnowledgeView)
    import torch
    ps = _bare_space(PerceptualSpace)
    ps.wv = _FakeWV(['the', 'cat'])
    ks1 = build_knowledge_section(_tiny_grammar(), wv=ps.wv)
    ps.attach_knowledge(KnowledgeView(ks1))
    # Now mutate the underlying ref_ids and re-attach.
    ks2 = build_knowledge_section(_tiny_grammar(), wv=ps.wv)
    ks2['word_table']['ref_ids'][0] = 5  # specific value, not -1
    ps.attach_knowledge(KnowledgeView(ks2))
    assert int(ps.wv.ref_ids[0].item()) == 5
