"""Serial word-loop object/meta tests.

The outer axis is a sentence slab of W words. Inside each word iteration all
of that word's radix constituents are synthesized before the canonical live
PS field enters CS; raw constituent width is neither PS.nOutput nor STM depth.
"""

import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

import Models
import Language
from util import init_config, init_device

_wsw = Models.BasicModel.word_span_window
_DATA = os.path.join(_PROJECT, "data")


def _serial_model_and_batch():
    """Build the serial mereology model (serialObjectMeta on) + one XOR batch."""
    init_device("cpu")
    torch.manual_seed(0)
    cfg_path = os.path.join(_DATA, "MM_mereology_serial.xml")
    init_config(path=cfg_path, defaults_path=os.path.join(_DATA, "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(cfg_path)
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    return m, m.inputSpace.prepInput(items)


def test_word_span_window_sums_active_word_only():
    # two words: word 0 = slots {0,1}, word 1 = slots {2,3}.
    full = torch.tensor([[[1., 1.], [2., 2.], [3., 3.], [4., 4.]]])  # [1,4,2]
    widx = torch.tensor([[0, 0, 1, 1]])
    # center in word 0 -> sum slots 0,1 = [3,3]; center in word 1 -> [7,7].
    assert torch.equal(_wsw(None, full, 0, widx), torch.tensor([[[3., 3.]]]))
    assert torch.equal(_wsw(None, full, 1, widx), torch.tensor([[[3., 3.]]]))
    assert torch.equal(_wsw(None, full, 2, widx), torch.tensor([[[7., 7.]]]))
    assert torch.equal(_wsw(None, full, 3, widx), torch.tensor([[[7., 7.]]]))


def test_word_span_window_is_hard_not_soft():
    # the mask is hard 0/1 -- a slot OUTSIDE the active word contributes zero,
    # never a gaussian-tail fraction.
    full = torch.tensor([[[10., 0.], [0., 0.], [0., 5.]]])           # [1,3,2]
    widx = torch.tensor([[0, 0, 1]])
    # word 0 (slots 0,1): exactly slot-0's value, slot-2 (word 1) excluded.
    assert torch.equal(_wsw(None, full, 0, widx), torch.tensor([[[10., 0.]]]))
    # word 1 (slot 2 only): no bleed from word 0.
    assert torch.equal(_wsw(None, full, 2, widx), torch.tensor([[[0., 5.]]]))


def test_word_span_window_per_row_independent_words():
    # word grouping differs per batch row; the mask is per-row.
    full = torch.tensor([
        [[1., 0.], [1., 0.], [1., 0.]],     # row 0: all one word
        [[2., 0.], [2., 0.], [2., 0.]],     # row 1: slot 0 alone, slots 1-2 a word
    ])
    widx = torch.tensor([[0, 0, 0], [0, 1, 1]])
    out = _wsw(None, full, 0, widx)          # center slot 0 in both rows
    # row 0: whole sequence summed = [3,0]; row 1: slot 0 only = [2,0].
    assert torch.equal(out, torch.tensor([[[3., 0.]], [[2., 0.]]]))


def test_word_span_window_falls_back_to_single_slot_without_index():
    full = torch.tensor([[[1., 1.], [2., 2.], [3., 3.]]])
    # no word index (byte mode) -> single-slot fallback, byte-safe.
    assert torch.equal(_wsw(None, full, 1, None), torch.tensor([[[2., 2.]]]))
    # wrong-shape index -> same fallback (defensive).
    bad = torch.tensor([[0, 0]])                      # T mismatch (2 != 3)
    assert torch.equal(_wsw(None, full, 1, bad), torch.tensor([[[2., 2.]]]))


def test_word_span_window_one_hot_when_each_slot_is_its_own_word():
    # fused-word modes: each slot is its own word -> equals the single slot.
    full = torch.tensor([[[1., 1.], [2., 2.], [3., 3.]]])
    widx = torch.tensor([[0, 1, 2]])
    for k in range(3):
        assert torch.equal(_wsw(None, full, k, widx), full[:, k:k + 1, :])


def test_word_span_window_clamps_center_and_guards_shape():
    full = torch.tensor([[[1., 1.], [2., 2.]]])
    widx = torch.tensor([[0, 0]])
    # out-of-range center clamps into [0, T-1] (both map into the one word).
    assert torch.equal(_wsw(None, full, 99, widx), torch.tensor([[[3., 3.]]]))
    # non-3D input returns unchanged (guard).
    flat = torch.tensor([1., 2.])
    assert torch.equal(_wsw(None, flat, 0, widx), flat)


# -- integration (2b captured-loop wiring) -----------------------------------

def test_serial_object_meta_stamps_and_word_index_device_tensors():
    """serialObjectMeta on: the flag reaches model + InputSpace + CS, and a
    forward populates the capturable per-slot word-index + per-word commit
    boundary device tensors."""
    m, x = _serial_model_and_batch()
    assert m.serial_object_meta is True
    assert getattr(m.inputSpace, "_serial_object_meta", None) is True
    assert getattr(m.conceptualSpace, "_serial_object_meta", None) is True
    with torch.no_grad():
        m.forward(x)
    ie = m.inputSpace
    assert ie._word_index_N is not None and ie._word_index_N.dim() == 2
    assert ie._word_last_slot_mask is not None
    assert ie._word_last_slot_mask.shape == ie._word_index_N.shape
    assert ie._word_last_slot_mask.dtype == torch.bool


def test_serial_radix_has_distinct_word_and_local_part_axes():
    """The sentence loop is [B,W]; raw constituent staging is [B,W,P_raw].

    The XOR rows contain exactly two words. P_raw follows the longest complete
    spelling in this batch and is independent of the eight-wide PS field.
    """
    m, x = _serial_model_and_batch()
    with torch.no_grad():
        m.forward(x)
    ie = m.inputSpace
    assert ie._ar_embedded_N.shape[:2] == (4, 8)       # sentence W
    assert ie._ar_word_part_ids.shape[:2] == (4, 8)
    assert ie._ar_word_part_ids.shape == ie._ar_word_part_mask.shape
    assert ie._ar_word_part_ids.shape[-1] == 6         # longest full spelling
    assert ie._ar_word_part_ids.shape[-1] != m.perceptualSpace.outputShape[0]
    assert ie._word_active_mask.sum(dim=1).tolist() == [2, 2, 2, 2]
    assert ie._word_last_slot_mask.sum(dim=1).tolist() == [2, 2, 2, 2]
    assert torch.equal(
        ie._word_index_N[:, :2],
        torch.tensor([[0, 1]]).expand(4, -1))
    assert bool((ie._word_index_N[:, 2:] == -1).all())


def test_serial_ws_analysis_view_is_fixed_width_and_compact():
    """Full eager cut metadata must not specialize the compiled WS carrier.

    Batch size one is the important stride case: a narrow slice can report
    contiguous while retaining its full parent's batch stride. The live view
    must have a canonical compact stride independent of full cut length.
    """
    m, _ = _serial_model_and_batch()
    # The compact serial fixture keeps WS analysis=raw; engage the same
    # meronymic cut BasicModel uses so this test has full span metadata.
    for stage_ws in m.wholeSpaces:
        stage_ws.analysis_mode = "meronomy"
    one = m.inputSpace.prepInput(["alpha beta gamma"])
    with torch.no_grad():
        m._lex_embed_stem(one)
    ws = m.wholeSpace
    live = ws._staged_analysis_spans
    full = ws._staged_analysis_spans_full
    n_live = int(ws.inputShape[0])
    assert live.shape == (1, n_live, 2)
    assert live.stride() == (n_live * 2, 2, 1)
    assert full is not None and full.dim() == 3
    prefix = min(n_live, int(full.shape[1]))
    assert torch.equal(live[:, :prefix], full[:, :prefix])


def test_partspace_runs_once_per_word_not_once_per_constituent():
    """Two-word rows produce two batch-wide PS calls, each with one whole."""
    m, x = _serial_model_and_batch()
    calls = []
    original = m.perceptualSpace.forward

    def spy(sub, *args, **kwargs):
        event = sub.materialize()
        calls.append(tuple(event.shape))
        return original(sub, *args, **kwargs)

    m.perceptualSpace.forward = spy
    try:
        with torch.no_grad():
            m.forward(x)
    finally:
        m.perceptualSpace.forward = original
    # The optional sentence prelude contributes whole-slab calls; the serial
    # reading loop itself contributes exactly the two one-word calls below.
    local_calls = [shape for shape in calls if shape[1] == 1]
    assert local_calls == [(4, 1, 1024), (4, 1, 1024)]


def test_outer_word_cap_reports_but_complete_word_parts_do_not_truncate():
    """W overflow is reported; an 11-part cold word survives PS width 8."""
    m, _ = _serial_model_and_batch()
    text = "abcdefghijk b c d e f g h i"  # 9 words; first has 11 cold bytes
    x = m.inputSpace.prepInput([text])
    with torch.no_grad():
        m.forward(x)
    ie = m.inputSpace
    assert ie._word_active_mask.sum().item() == 8
    assert ie._sentence_word_truncated_mask.tolist() == [True]
    assert ie._ar_word_part_ids.shape[-1] == 11
    assert int(ie._ar_word_part_mask[0, 0].sum().item()) == 11
    assert not bool(ie._ar_word_truncated_mask.any())


def test_serial_commit_gate_fires_once_per_active_word():
    """The per-word commit boundary is True exactly ONCE per word (at the word's
    last active slot) -- so the STM push fires once per word, not once per slot
    (the radix multi-slot over-push fix)."""
    m, x = _serial_model_and_batch()
    with torch.no_grad():
        m.forward(x)
    ie = m.inputSpace
    widx = ie._word_index_N
    last = ie._word_last_slot_mask
    active = ie._word_active_mask
    for b in range(widx.shape[0]):
        # number of distinct ACTIVE word ids == number of commit-True slots.
        active_ids = {int(widx[b, n]) for n in range(widx.shape[1])
                      if bool(active[b, n])}
        n_commits = int(last[b].sum())
        assert n_commits == len(active_ids), (
            f"row {b}: {n_commits} commits != {len(active_ids)} words")
        # every commit slot is genuinely active (no pad commits).
        for n in range(last.shape[1]):
            if bool(last[b, n]):
                assert bool(active[b, n])


def test_word_prediction_and_physical_push_share_one_commit_boundary():
    """Word mode must neither predict from nor push partial radix spellings.

    ConceptualSpace captures the predictor gate while a spy sums the physical
    STM push gates. The intent-only sentence prelude deliberately leaves STM
    empty, so word 1 has no retained context to predict from; every later word
    has one captured target. Physical commits still occur once per word.
    """
    m, x = _serial_model_and_batch()
    stm = m.conceptualSpace.stm
    pushed = [0]
    original = stm.push_step_masked

    def spy(ideas, row_gate, **metadata):
        pushed[0] += int(row_gate.sum().item())
        return original(ideas, row_gate, **metadata)

    stm.push_step_masked = spy
    try:
        with torch.no_grad(), \
                m.conceptualSpace.capture_intra_predictions() as trace:
            m.forward(x)
    finally:
        stm.push_step_masked = original

    commits = m.inputSpace._word_last_slot_mask.sum(dim=1)
    assert pushed[0] == int(commits.sum().item())
    captured = torch.zeros_like(commits)
    for entry in trace:
        gate = entry["row_gate"]
        if gate is not None:
            captured += gate.reshape(-1).to(captured.dtype)
    assert torch.equal(captured, (commits - 1).clamp_min(0))


def test_serial_object_meta_off_does_not_build_word_index():
    """Flag OFF (forced): the per-word index/commit tensors are NOT built, the
    gaussian path stays in force -- byte-identical plumbing."""
    m, x = _serial_model_and_batch()
    # force the flag off on InputSpace (the build guard reads it) + the model
    # (the window-branch + push-gate read it).
    m.serial_object_meta = False
    m.inputSpace._serial_object_meta = False
    with torch.no_grad():
        m.forward(x)
    assert m.inputSpace._word_index_N is None
    assert m.inputSpace._word_last_slot_mask is None


def _count_unity_calls(m, x):
    """Run a forward, counting WholeSpace._stage0_unity_forward calls with a
    REAL unity (vs None). Returns (real, none)."""
    import Spaces
    calls = {"real": 0, "none": 0}
    orig = Spaces.WholeSpace._stage0_unity_forward

    def _probe(self, IS_concepts):
        calls["real" if IS_concepts is not None else "none"] += 1
        return orig(self, IS_concepts)

    Spaces.WholeSpace._stage0_unity_forward = _probe
    try:
        with torch.no_grad():
            m.forward(x)
    finally:
        Spaces.WholeSpace._stage0_unity_forward = orig
    return calls["real"], calls["none"]


def test_ss_unity_validity_at_stem():
    """Unity VALIDITY law + LIVE delivery (2026-07-12): IS is not the
    lexer -- it delivers the raw surface, so the unity now carries the
    batch's real bytes and the stem stages it (universe routing engaged,
    exactly as the validity law promised once real bytes landed)."""
    import torch as _t
    m, x = _serial_model_and_batch()
    with _t.no_grad():
        m.forward(x)
    u = getattr(m, "_ws_universe", None)
    assert _t.is_tensor(u) and bool((u != 0).any()), (
        "the delivered unity must be staged live (real bytes)")
    real, _none = _count_unity_calls(m, x)
    assert real > 0, "a live unity must feed the universe branch"
