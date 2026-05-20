"""Tests for ``ConceptualSpace.stm_typed`` — the typed-metadata STM.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 / step 2 — ConceptualSpace owns the payload + metadata stack.

The existing ``ConceptualSpace.stm`` (``ShortTermMemory``) stays
untouched for backward compat; the new ``stm_typed`` (``TypedStack``)
carries category / order / ref_id metadata that the STM shift/reduce
driver consumes.

Initialization helper ``_init_typed_stm(batch, max_depth, dim)`` is
callable from real ``__init__`` (with sizes derived from XML config)
and from tests on bare instances.
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def _bare_conceptual_space():
    """Bare ConceptualSpace instance — __init__ bypassed."""
    from Spaces import ConceptualSpace
    import torch.nn as nn
    cs = object.__new__(ConceptualSpace)
    nn.Module.__init__(cs)
    return cs


def test_init_typed_stm_helper_creates_typed_stack():
    """``_init_typed_stm`` allocates ``stm_typed`` as a TypedStack."""
    from typed_stack import TypedStack
    cs = _bare_conceptual_space()
    cs._init_typed_stm(batch=2, max_depth=8, dim=4)
    assert isinstance(cs.stm_typed, TypedStack)
    assert cs.stm_typed.batch == 2
    assert cs.stm_typed.max_depth == 8
    assert cs.stm_typed.dim == 4


def test_stm_typed_is_none_before_init():
    """A ConceptualSpace whose ``_init_typed_stm`` hasn't run yet exposes
    ``stm_typed`` as ``None`` (via the property fallback)."""
    cs = _bare_conceptual_space()
    assert cs.stm_typed is None


def test_stm_typed_reinit_replaces():
    """Re-calling ``_init_typed_stm`` replaces the previous TypedStack."""
    cs = _bare_conceptual_space()
    cs._init_typed_stm(batch=1, max_depth=4, dim=2)
    first = cs.stm_typed
    cs._init_typed_stm(batch=2, max_depth=8, dim=4)
    assert cs.stm_typed is not first
    assert cs.stm_typed.batch == 2
