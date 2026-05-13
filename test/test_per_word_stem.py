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
    m, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    # Per-word stem is the only path post-2026-05-12; the flag is fixed
    # True at WordSpace construction.  iterations_per_word remains an
    # XML knob; force 1 here so the test asserts the base contract.
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


def test_per_word_stem_quantization_count_matches_word_count():
    """Per-word symbolic quantization invariant: the S
    ``ProjectionBasis``'s ``forward`` and ``reverse`` fire exactly
    once per word slot. This locks the architectural contract that
    words are projected one-by-one as they land on STM.

    Updated 2026-05-13: the per-word loop now calls ProjectionBasis
    (not Codebook.forward(project=True)).  The fixture only fires
    when SymbolicSpace.subspace.what is a ProjectionBasis (bivector
    regime); for other configs the per-word loop's projection
    branch short-circuits and no fwd/rev calls happen.
    """
    m = _build_per_word_stem()
    cb = m.symbolicSpace.subspace.what
    if type(cb).__name__ != 'ProjectionBasis':
        import pytest
        pytest.skip(
            "Config does not use ProjectionBasis on the symbolic "
            "codebook; the per-word projection branch is inactive.")
    call_log = {"fwd": 0, "rev": 0}
    orig_fwd = cb.forward
    orig_rev = cb.reverse

    def trace_fwd(*args, **kwargs):
        call_log["fwd"] += 1
        return orig_fwd(*args, **kwargs)

    def trace_rev(*args, **kwargs):
        call_log["rev"] += 1
        return orig_rev(*args, **kwargs)

    cb.forward = trace_fwd
    cb.reverse = trace_rev
    m.forward(_xor_input())
    stm_depth = m.conceptualSpace.stm.size(0)
    assert stm_depth > 0, "STM should hold at least one idea per word."
    assert call_log["fwd"] == stm_depth, (
        f"ProjectionBasis.forward fired {call_log['fwd']} times; "
        f"expected one per STM slot ({stm_depth}).")
    assert call_log["rev"] == stm_depth, (
        f"ProjectionBasis.reverse fired {call_log['rev']} times; "
        f"expected one per STM slot ({stm_depth}).")


def test_per_word_stem_codebook_distinguishes_distinct_inputs():
    """Per-word quantization invariant: distinct C-tier inputs produce
    distinct bivector snaps via the ProjectionBasis forward.  This
    is the substrate guarantee that word-by-word quantization can
    carry meaningful per-word identity once the C-tier transform
    itself is non-degenerate.

    Updated 2026-05-13: ProjectionBasis is the bivector-regime
    surface (Codebook lost ``project=True``).
    """
    m = _build_per_word_stem()
    cb = m.symbolicSpace.subspace.what
    if type(cb).__name__ != 'ProjectionBasis':
        import pytest
        pytest.skip(
            "Config does not use ProjectionBasis on the symbolic "
            "codebook; bivector snap test is only meaningful there.")
    c1 = torch.randn(1, 1, cb.nDim) * 0.5
    c2 = torch.randn(1, 1, cb.nDim) * 0.5
    snap1 = cb.forward(c1)
    snap2 = cb.forward(c2)
    diff = (snap1 - snap2).abs().max().item()
    assert diff > 1e-3, (
        f"ProjectionBasis is degenerate: distinct C-tier inputs produced "
        f"identical bivector snaps (max-diff={diff:.6f}).")


def test_per_word_stem_iterations_per_word_idempotent():
    """iterations_per_word > 1 gives an idempotent C-tier idea under
    the SVD-orthogonal codebook fixed point: a SINGLE model run with
    iterations=1 vs iterations=3 against the same untrained weights
    must produce the same C-tier idea, because each iteration projects
    the same perceptual slot through the same PiLayer.
    """
    m = _build_per_word_stem()
    m.wordSpace.chart.iterations_per_word = 1
    m.forward(_xor_input())
    idea_n1 = m.conceptualSpace.stm.peek(0, 0).clone()

    # Same model instance, same weights -- only iterations changes.
    m.wordSpace.chart.iterations_per_word = 3
    m.conceptualSpace.stm.clear()
    m.forward(_xor_input())
    idea_n3 = m.conceptualSpace.stm.peek(0, 0).clone()
    assert torch.allclose(idea_n1, idea_n3, atol=1e-4), (
        "iterations_per_word should be idempotent under SVD codebook; "
        f"diff: {(idea_n1 - idea_n3).abs().max().item()}")


def test_per_word_stem_is_default_path():
    """Per-word stem is the only path post-2026-05-12: the chart flag
    defaults to True at WordSpace construction, and the stem fills STM
    on every forward."""
    TheData.load("xor")
    m, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    assert m.wordSpace.chart.per_word_stem is True
    m.conceptualSpace.stm = ShortTermMemory(
        batch=4, capacity=8,
        concept_dim=m.conceptualSpace.stm.concept_dim)
    m.forward(_xor_input())
    # Every row should see at least one idea pushed onto STM.
    for b in range(4):
        assert m.conceptualSpace.stm.size(b) > 0, (
            f"Row {b} STM should be filled by the per-word stem; "
            f"got size={m.conceptualSpace.stm.size(b)}.")
