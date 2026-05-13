"""Per-word stem + chart-at-C path.

The serial-parser handoff (originally a shift-reduce design) pivoted to
chart-on-LTM: each word does an individual P->C->S->C round trip in the
stem and pushes a C-tier "idea" onto ``ConceptualSpace.stm``; the chart
fires at C-tier over STM in the body, with soft compose preserved.

These tests exercise:
  * Per-word round trip fills STM with one idea per word slot.
  * Chart compose runs at C-tier over the STM buffer.
  * Truth tags can be set/got on STM slots and clear with the buffer.
  * STM and truth tags clear on hard Reset.
  * ``iterations_per_word`` repeats the P->C cycle idempotently.
"""
import sys
from pathlib import Path

import pytest
import torch

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

from data import TheData
from Models import BaseModel
from Spaces import ShortTermMemory


_CONFIG_PATH = str(_project / "data" / "XOR_grammar.xml")


def _xor_input():
    return torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    ).float().unsqueeze(1)


def _build_per_word_stem(stm_capacity=8):
    TheData.load("xor")
    torch.manual_seed(0)
    m, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    m.wordSpace.chart.per_word_stem = True
    m.wordSpace.chart.iterations_per_word = 1
    # Resize STM to a generous capacity that exceeds the XOR sentence
    # length so push() never overflows.
    m.conceptualSpace.stm = ShortTermMemory(
        batch=4, capacity=stm_capacity,
        concept_dim=m.conceptualSpace.stm.concept_dim)
    return m


def test_per_word_stem_round_trip():
    """Each perceptual word slot becomes one idea on STM."""
    m = _build_per_word_stem()
    out = m.forward(_xor_input())
    assert out is not None
    # Every batch row should have at least one idea per perceptual slot.
    for b in range(4):
        assert m.conceptualSpace.stm.size(b) > 0, (
            f"Row {b} STM is empty; per-word stem failed to push.")
    # All rows in the same batch should see the same number of slots
    # (same XOR input shape).
    sizes = [m.conceptualSpace.stm.size(b) for b in range(4)]
    assert len(set(sizes)) == 1, (
        f"STM depth diverged across rows: {sizes}")


def test_chart_fires_at_C_over_stm():
    """After _forward_stem_per_word fills STM, the chart compose should
    have run at C-tier (populating wordSpace.current_rules) at least once.
    """
    m = _build_per_word_stem()
    # Sanity: no rules before forward (or last_composed empty).
    pre_rules = dict(m.wordSpace.current_rules)
    m.forward(_xor_input())
    post_rules = m.wordSpace.current_rules
    # Chart should have run and produced at least one tier's rule list.
    assert isinstance(post_rules, dict)
    assert len(post_rules) > 0, (
        "Chart compose at C did not populate current_rules.")


def test_truth_tag_set_get_on_stm():
    """ShortTermMemory truth-tag accessors round-trip a catuskoti bivector."""
    stm = ShortTermMemory(batch=2, capacity=4, concept_dim=3)
    stm.push(0, torch.tensor([1.0, 2.0, 3.0]))
    stm.set_truth_tag(0, 0, torch.tensor([0.7, 0.3]))
    tag = stm.get_truth_tag(0, 0)
    assert tag is not None
    assert torch.allclose(tag, torch.tensor([0.7, 0.3]), atol=1e-5)
    # Fresh push starts with a zero tag.
    stm.push(0, torch.tensor([4.0, 5.0, 6.0]))
    tag2 = stm.get_truth_tag(0, 0)
    assert tag2 is not None
    assert torch.allclose(tag2, torch.zeros(2), atol=1e-5)


def test_per_word_stem_sentence_boundary():
    """Hard Reset clears both the idea buffer and the truth tags."""
    m = _build_per_word_stem()
    m.forward(_xor_input())
    stm = m.conceptualSpace.stm
    assert stm.size(0) > 0, "STM should be filled after forward."
    # Set a truth tag so we can verify it clears.
    stm.set_truth_tag(0, 0, torch.tensor([0.6, 0.4]))
    assert stm.get_truth_tag(0, 0) is not None
    # Hard Reset clears STM via ConceptualSpace.Reset.
    m.conceptualSpace.Reset(batch=0, hard=True)
    assert stm.size(0) == 0, "Hard Reset did not empty STM."
    # The cleared row's truth-tag slot is zeroed.
    assert torch.allclose(stm._truth_tags[0], torch.zeros_like(stm._truth_tags[0]))


def test_per_word_stem_iterations_per_word_idempotent():
    """iterations_per_word > 1 gives an idempotent C-tier idea under
    the SVD-orthogonal codebook fixed point (the per-word loop reuses
    the same perceptual slot each iteration so the projection should
    not drift between iterations).
    """
    m1 = _build_per_word_stem()
    m1.wordSpace.chart.iterations_per_word = 1
    m1.forward(_xor_input())
    idea_n1 = m1.conceptualSpace.stm.peek(0, 0).clone()

    m2 = _build_per_word_stem()
    m2.wordSpace.chart.iterations_per_word = 3
    m2.forward(_xor_input())
    idea_n3 = m2.conceptualSpace.stm.peek(0, 0).clone()
    assert torch.allclose(idea_n1, idea_n3, atol=1e-4), (
        "iterations_per_word should be idempotent under SVD codebook; "
        f"diff: {(idea_n1 - idea_n3).abs().max().item()}")


def test_per_word_stem_flag_off_path_unchanged():
    """With per_word_stem=False (default), the legacy chart-at-stem path
    runs unmodified; STM is not filled by the stem."""
    TheData.load("xor")
    torch.manual_seed(0)
    m, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    assert m.wordSpace.chart.per_word_stem is False
    m.conceptualSpace.stm = ShortTermMemory(
        batch=4, capacity=8,
        concept_dim=m.conceptualSpace.stm.concept_dim)
    m.forward(_xor_input())
    # In the legacy path the per-word stem does not run; STM stays empty.
    for b in range(4):
        assert m.conceptualSpace.stm.size(b) == 0, (
            f"Row {b} STM should be empty on the flag-off path; "
            f"got size={m.conceptualSpace.stm.size(b)}.")
