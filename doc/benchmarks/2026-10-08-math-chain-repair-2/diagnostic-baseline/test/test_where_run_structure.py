"""RunStructureLayer: fixed-shape mereological run / gap / containment over a
SET of ``.where`` brackets (doc/specs/mereological-order-raising.md).

The Layer reports how many CONTIGUOUS RUNS a row's part spans form (``n_runs``
== the part count, the routing signal) and which spans contain which (the
``A isa B`` test). Subsymbolic + compiled-forward-safe: no host-side control
flow, no sort, no variable shapes -- a span starts a new run iff no valid
earlier span (by start, index tie-broken) reaches its start within ``tol``.
"""
import os, sys
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Layers import RunStructureLayer


def _rs(spans, valid=None, tol=0.5):
    """Run the Layer on a [K,2] or [B,K,2] list of (start,end) spans."""
    layer = RunStructureLayer(contiguity_tol=tol)
    return layer(torch.tensor(spans, dtype=torch.float32), valid=valid)


def test_single_span_one_run():
    out = _rs([[10, 50]])
    assert int(out["n_runs"]) == 1
    assert int(out["n_gaps"]) == 0
    assert int(out["ratio"]) == 1


def test_contiguous_siblings_merge_to_one_run():
    out = _rs([[10, 30], [30, 60]])           # touch at 30
    assert int(out["n_runs"]) == 1
    assert int(out["n_gaps"]) == 0


def test_disjoint_two_runs_one_gap():
    out = _rs([[10, 30], [40, 60]])           # gap of 10
    assert int(out["n_runs"]) == 2
    assert int(out["n_gaps"]) == 1


def test_nested_one_run_and_containment():
    # input order: 0 = outer [10,50], 1 = inner [20,30]
    out = _rs([[10, 50], [20, 30]])
    m = out["contained_mask"]
    assert int(out["n_runs"]) == 1            # overlapping spans -> one run
    assert bool(m[1, 0]) is True              # inner within outer (B isa A)
    assert bool(m[0, 1]) is False             # outer not within inner
    assert bool(m[0, 0]) and bool(m[1, 1])    # diagonal: self-containment


def test_single_position_gap_splits():
    # a one-position gap (a skipped position, e.g. a space that is not a part)
    # opens a new run; touching spans do not.
    assert int(_rs([[0, 3], [4, 8]])["n_runs"]) == 2     # gap of 1
    assert int(_rs([[0, 3], [3, 8]])["n_runs"]) == 1     # touch


def test_unsorted_input_handled():
    out = _rs([[40, 60], [10, 30]])           # reverse start order
    assert int(out["n_runs"]) == 2


def test_word_spans_count_as_runs():
    # "ab cd ef" word spans -> (0,2),(3,5),(6,8): three parts, three runs.
    out = _rs([[0, 2], [3, 5], [6, 8]])
    assert int(out["n_runs"]) == 3
    assert int(out["n_gaps"]) == 2


def test_tie_break_same_start_nested():
    # two spans sharing a start (a containment, not two runs).
    out = _rs([[0, 5], [0, 3]])
    assert int(out["n_runs"]) == 1


def test_pad_spans_excluded_by_default_validity():
    # the (0,0) pad is invalid (end == start) -> excluded from the count.
    out = _rs([[0, 2], [3, 5], [0, 0]])
    assert int(out["n_runs"]) == 2


def test_batched_rows():
    spans = torch.tensor([[[0, 2], [3, 5], [6, 8]],     # 3 words
                          [[0, 10], [0, 0], [0, 0]]],   # 1 word + 2 pads
                         dtype=torch.float32)
    out = RunStructureLayer()(spans)
    assert out["n_runs"].tolist() == [3, 1]
    assert out["n_gaps"].tolist() == [2, 0]
    assert out["contained_mask"].shape == (2, 3, 3)


def test_explicit_valid_mask_drops_middle():
    spans = torch.tensor([[[0, 2], [3, 5], [6, 8]]], dtype=torch.float32)
    valid = torch.tensor([[True, False, True]])         # drop the middle word
    out = RunStructureLayer()(spans, valid=valid)
    assert int(out["n_runs"][0]) == 2


def test_is_run_start_flags():
    out = _rs([[0, 2], [3, 5], [6, 8]])
    # every word starts its own run here
    assert out["is_run_start"].tolist() == [True, True, True]
    out2 = _rs([[0, 5], [2, 4]])              # second is nested -> not a start
    assert out2["is_run_start"].tolist() == [True, False]


def test_extent_signals():
    out = _rs([[0, 3], [4, 8]])               # widths 3, 4
    assert out["span_extent"].tolist() == [3.0, 4.0]
    assert float(out["max_extent"]) == 4.0
    assert float(out["total_extent"]) == 7.0


def test_long_singleton_flagged_by_extent():
    # one wide part -> 1 run (the run-count looks "fine") but large max_extent:
    # the signal property-tiling needs to decide to analyse it.
    out = _rs([[0, 28]])
    assert int(out["n_runs"]) == 1
    assert float(out["max_extent"]) == 28.0
    # a basic-level word reads as 1 run with small extent -> no analysis.
    basic = _rs([[0, 4]])
    assert int(basic["n_runs"]) == 1 and float(basic["max_extent"]) == 4.0


def test_extent_excludes_pads():
    out = _rs([[0, 2], [3, 5], [0, 0]])       # third is a pad
    assert out["span_extent"].tolist() == [2.0, 2.0, 0.0]
    assert float(out["max_extent"]) == 2.0


def test_extent_batched():
    spans = torch.tensor([[[0, 3], [4, 8], [0, 0]],     # widths 3,4
                          [[0, 28], [0, 0], [0, 0]]],   # one long part
                         dtype=torch.float32)
    out = RunStructureLayer()(spans)
    assert out["max_extent"].tolist() == [4.0, 28.0]
    assert out["total_extent"].tolist() == [7.0, 28.0]


# -- Force #1: tightest container (the "A isa B" edge, smallest whole) --------

def test_tightest_container_flat_siblings_none():
    # disjoint words, no nesting -> no span contains another -> all -1.
    out = _rs([[0, 3], [4, 7], [8, 10]])
    assert out["tightest_container"].tolist() == [-1, -1, -1]


def test_tightest_container_sentence_over_words():
    # span 0 = the whole "sentence" (0,10); spans 1,2 = words nested in it.
    out = _rs([[0, 10], [0, 3], [4, 7]])
    tc = out["tightest_container"].tolist()
    assert tc[0] == -1            # the sentence has no container
    assert tc[1] == 0 and tc[2] == 0   # each word's tightest whole is span 0


def test_tightest_container_picks_smallest_whole():
    # nested: (1,3) sits in BOTH (0,5) [span1] and (0,10) [span0]; force #1
    # picks the SMALLER whole (span1), not the bigger (span0).
    out = _rs([[0, 10], [0, 5], [1, 3]])
    tc = out["tightest_container"].tolist()
    assert tc[2] == 1             # tightest = the smaller container, not 0
    assert tc[1] == 0             # (0,5) is contained only by (0,10)
    assert tc[0] == -1


def test_tightest_container_batched():
    spans = torch.tensor([[[0, 10], [0, 3], [4, 7]],     # words in a sentence
                          [[0, 3], [4, 7], [0, 0]]],      # flat + a pad
                         dtype=torch.float32)
    out = RunStructureLayer()(spans)
    assert out["tightest_container"].tolist() == [[-1, 0, 0], [-1, -1, -1]]


# -- Three-aspect route hint (null / refine / raise) --------------------------

def test_route_hint_null_when_no_valid_runs():
    out = _rs([[0, 0], [0, 0]])               # all pads -> 0 runs -> NULL
    assert int(out["route_hint"]) == 0


def test_route_hint_refine_one_run():
    out = _rs([[0, 3]])                        # one contiguous run -> REFINE
    assert int(out["route_hint"]) == 1


def test_route_hint_raise_discontiguous():
    out = _rs([[0, 3], [5, 8]])                # two runs (gap) -> RAISE
    assert int(out["route_hint"]) == 2


def test_route_hint_batched():
    spans = torch.tensor([[[0, 3], [0, 0]],     # 1 run -> refine
                          [[0, 3], [5, 8]]],     # 2 runs -> raise
                         dtype=torch.float32)
    out = RunStructureLayer()(spans)
    assert out["route_hint"].tolist() == [1, 2]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
