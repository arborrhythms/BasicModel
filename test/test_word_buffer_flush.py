"""Tensor word buffer round-trip tests (§6c Path B).

The brick-vectorization handoff §6c adds a per-cell tensor buffer
(``word_records`` / ``word_count``) to ``SubSpace`` so the chart
compose can write into it inside the brick (no host sync). The outer
doc-streaming loop then calls ``flush_word_buffer`` once per tick to
materialize the buffer into ``self.word`` (a Python list) for legacy
consumers (``decompose``, ``reconstruct``, the SVO walker).

These tests check the buffer scaffolding directly: same write
sequence → same materialized ``self.word`` between the legacy
scalar ``add_word(int, ...)`` path and the new vector
``add_word(LongTensor, ...)`` path.

Plan reference: doc/plans/2026-04-27-brick-vectorization-and-legacy-removal-handoff.md §6c
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch
import pytest

# The vector ``add_word`` overload requires a TheGrammar with at least
# one rule (encode() validates the rule index). We piggy-back on the
# global default so the test is self-contained.
from data import TheData
from Models import BaseModel

_CONFIG_PATH = str(_project / "data" / "MM_xor.xml")


def _model():
    TheData.load("xor")
    torch.manual_seed(0)
    model, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    return model


def _fresh_subspace(model):
    """Pick a representative SubSpace and clear its scratch state."""
    sub = model.inputSpace.subspace
    sub.word = []
    sub.ensure_word_buffer(4)
    sub.clear_word_buffer()
    return sub


def test_buffer_starts_empty():
    """A fresh SubSpace has zero pending entries; flush is a no-op."""
    model = _model()
    sub = _fresh_subspace(model)
    sub.flush_word_buffer()
    assert sub.word == []
    assert int(sub.word_count.sum().item()) == 0


def test_scalar_and_vector_overloads_agree():
    """Scalar add_word and vector add_word produce identical entries.

    Drive the same sequence of (batch, vector, rule, ...) tuples
    through both APIs and compare the materialized ``self.word`` list
    after flush; they must match entry-for-entry.
    """
    model = _model()

    # Reference: scalar path.
    ref_sub = _fresh_subspace(model)
    seq = [
        (0, 1, 0, 0, -1, -1, -1),
        (0, 2, 0, 1, 5, -1, -1),
        (1, 0, 0, 0, -1, -1, -1),
        (1, 3, 0, 2, 7, 8, -1),
    ]
    for (b, v, r, o, l1, l2, l3) in seq:
        ref_sub.add_word(b, v, r, order=o, leaf1=l1, leaf2=l2, leaf3=l3)
    ref_words = list(ref_sub.word)

    # Candidate: vector path. Group by depth (each cell gets one
    # entry per depth d) so the scatter writes one entry per cell at
    # the cell's current depth, matching the scalar order. Here all
    # entries are at depth 0 except (0,2) at depth 1 and (1,3) at
    # depth 1.
    cand_sub = _fresh_subspace(model)
    # Depth 0 entries: cells 0 and 1 each get one entry.
    cand_sub.add_word(
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([1, 0], dtype=torch.long),
        torch.tensor([0, 0], dtype=torch.long),
        order=torch.tensor([0, 0], dtype=torch.long),
        leaf1=torch.tensor([-1, -1], dtype=torch.long),
        leaf2=torch.tensor([-1, -1], dtype=torch.long),
        leaf3=torch.tensor([-1, -1], dtype=torch.long),
    )
    # Depth 1 entries: cell 0 gets (vec=2, rule=0, leaf1=5); cell 1
    # gets (vec=3, rule=0, leaf1=7, leaf2=8).
    cand_sub.add_word(
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([2, 3], dtype=torch.long),
        torch.tensor([0, 0], dtype=torch.long),
        order=torch.tensor([1, 2], dtype=torch.long),
        leaf1=torch.tensor([5, 7], dtype=torch.long),
        leaf2=torch.tensor([-1, 8], dtype=torch.long),
        leaf3=torch.tensor([-1, -1], dtype=torch.long),
    )
    cand_sub.flush_word_buffer()
    cand_words = list(cand_sub.word)

    # Cell-major flush ordering matches the scalar interleave: cell 0
    # depth 0, cell 0 depth 1, cell 1 depth 0, cell 1 depth 1.
    expected_after_flush = [ref_words[0], ref_words[1], ref_words[2], ref_words[3]]
    assert cand_words == expected_after_flush


def test_flush_resets_count_for_next_tick():
    """After flush, ``word_count`` must be zero so the next tick
    starts fresh; ``self.word`` retains the materialized entries.
    """
    model = _model()
    sub = _fresh_subspace(model)
    sub.add_word(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
    )
    assert int(sub.word_count.sum().item()) == 1
    sub.flush_word_buffer()
    assert int(sub.word_count.sum().item()) == 0
    assert len(sub.word) == 1
    # Next tick: another scatter at cell 0 starts at depth 0 again.
    sub.add_word(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([2], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
    )
    sub.flush_word_buffer()
    assert len(sub.word) == 2
    # Vector slot of the second entry must be 2 (not 1 from tick 1).
    from Spaces import WordEncoding as WE
    assert sub.word[1][WE.VECTOR] == 2


def test_buffer_lazy_resize():
    """A scatter referencing cell index past current size grows the
    buffer rather than failing.
    """
    model = _model()
    sub = _fresh_subspace(model)  # sized to 4
    # Scatter at cell 7 forces a grow.
    sub.add_word(
        torch.tensor([7], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
    )
    assert int(sub.word_count.shape[0]) >= 8
    sub.flush_word_buffer()
    assert len(sub.word) == 1


def test_legacy_scalar_path_unaffected_by_buffer():
    """Pure scalar callers don't interact with the tensor buffer.

    The scalar overload still appends directly to ``self.word`` and
    leaves ``word_count`` at zero, so a flush after legacy use is a
    no-op (no double-counting).
    """
    model = _model()
    sub = _fresh_subspace(model)
    sub.add_word(0, 1, 0)
    sub.add_word(1, 2, 0)
    assert len(sub.word) == 2
    assert int(sub.word_count.sum().item()) == 0
    sub.flush_word_buffer()
    # Length unchanged; flush had nothing to materialize.
    assert len(sub.word) == 2
