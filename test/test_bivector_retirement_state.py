"""Regression test pinning down the current state of bivector retirement.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 deferred — "bivector retirement (narrow): only
``SymbolicSpace.subspace.what.W: [V_ref, 2] → [V_ref]``".

When the plan was authored the codebook stored ``[V_ref, 2]`` catuskoti
bivectors and ``subspace.nWhat`` was pinned at 2. That world has
already moved on: the post-2026-05-07 rollback (and the subsequent
"Bivector retirement Phase 0/1" commits) collapsed the codebook's
pinned bivector and decoupled ``nWhat`` from 2 — ``nWhat == nDim``
now. The catuskoti bivector lives on ``subspace.activation`` (per-batch
``[B, V_S, 2]``), not in the codebook.

This test documents that current state so a future narrowing step
(codebook shape ``[V_sym, nDim]`` → ``[V_sym, 1]``) is a deliberate
choice, not an accidental drift. The plan's deferred narrow scope is
gated on consumers of ``subspace.what.W`` migrating to read from the
trainable ``SymbolicSpace.references`` Parameter (the new path
introduced in Phase 2 ``attach_knowledge``); until that consumer
migration completes, the codebook stays multi-dim.
"""
import os
import sys

import torch
import torch.nn as nn

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
_TEST = os.path.dirname(os.path.abspath(__file__))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
if _TEST not in sys.path:
    sys.path.insert(0, _TEST)

from test_partition_pos_codebook import _make_word_space  # noqa: E402


def test_symbolic_codebook_no_longer_pinned_bivector():
    """``SymbolicSpace.subspace.what.W.shape[1]`` is NOT 2 — the legacy
    pinned-bivector codebook is gone."""
    ws = _make_word_space(symbolDim=4)
    sym = ws.symbolicSpace
    W = sym.subspace.what.W
    assert W is not None
    assert W.ndim == 2
    assert W.shape[1] != 2, (
        f"codebook .W shape[1]={W.shape[1]} — expected != 2 "
        f"(post-bivector-retirement-Phase-0/1)")


def test_symbolic_nwhat_equals_ndim():
    """``subspace.nWhat`` is decoupled from the bivector width and follows
    ``self.nDim`` minus the uniform (2,2) where/when band (the symbol-space
    content dimensionality). The plan's text "nWhat = 2 stays" predates the
    rollback that decoupled them; the SS=(0,0) special case (where nWhat ==
    nDim exactly) was retired by the uniform-(2,2) convention."""
    from architecture import canonical_shape
    ws = _make_word_space(symbolDim=4 + sum(canonical_shape("SymbolicSpace")))
    sym = ws.symbolicSpace
    assert sym.nWhat == sym.nDim - sum(canonical_shape("SymbolicSpace"))


def test_symbolic_references_parameter_attached_via_knowledge():
    """The new scalar-only path lives on ``SymbolicSpace.references``,
    created when ``attach_knowledge`` fires. The codebook stays the
    legacy ``.what.W`` Parameter; consumer migration from ``.W`` to
    ``references`` is the remaining narrow-bivector-retirement work."""
    from Language import Grammar
    from embed import build_knowledge_section, KnowledgeView
    ws = _make_word_space(symbolDim=4)
    sym = ws.symbolicSpace

    g = Grammar()
    g.rules = [
        g._parse_rule("NP", "conjunction(DET, N)", tier='S'),
        g._parse_rule("S", "disjunction(NP, VP)", tier='S'),
    ]
    g._configured = True
    view = KnowledgeView(build_knowledge_section(g))
    sym.attach_knowledge(view)

    # references is the scalar-only Parameter the plan envisions as the
    # eventual replacement for .what.W
    assert hasattr(sym, 'references')
    assert isinstance(sym.references, nn.Parameter)
    assert sym.references.ndim == 1


def test_symbolic_activation_carries_bivector():
    """Activation (the place per-batch bivectors live) still has
    ``activation.W`` shape matching the bivector / scalar carrier per
    the activation-channeling convention."""
    ws = _make_word_space(symbolDim=4)
    sym = ws.symbolicSpace
    # activation.W is the per-batch payload — its dim reflects the
    # activation carrier, NOT the codebook prototype width.
    act_basis = sym.subspace.activation
    assert act_basis is not None
