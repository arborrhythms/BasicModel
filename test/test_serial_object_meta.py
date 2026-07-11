"""Serial-mode word-at-a-time (Option A; doc/specs/mereological-order-raising.md
"Serial-mode word-at-a-time loop"). Increment 2 = HARD-MASK-TO-WORD-SPAN: the
``word_span_window`` helper isolates the ACTIVE WORD (sum over the slots sharing
its word index) so PartSpace processes one word at a time -- no part with a
``.where`` outside the word. The pure ``word_span_window`` helper is unit-tested
in isolation (it reads none of ``self``, so it runs with ``self=None``); the
INTEGRATION tests at the bottom exercise the captured-loop wiring (2b): the
``_word_index_N`` / ``_word_last_slot_mask`` device tensors, the per-word commit
gate, and the gaussian->word_span_window branch, gated ``serialObjectMeta``.
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


def test_ss_analyzes_unity_at_prelude_pump_zero():
    """2c dual view: under serialObjectMeta the §6c prelude feeds WS the UNITY
    at pump 0 (CS empty -> the legal _stage0_unity_forward path, no repeated-
    injection NotImplementedError), so WS 'looks at the input' alongside PS."""
    m, x = _serial_model_and_batch()
    real, _none = _count_unity_calls(m, x)
    assert real >= 1, "WS must analyze the unity at prelude pump 0"


def test_ss_unity_feed_unconditional_after_migration():
    """Serial migration (2026-07-11): the unity is OFFERED unconditionally
    (the flag no longer gates the FEED); consumption is decided by the
    typed+liveness routing law -- a dead unity with a live carrier routes
    the carrier body, so non-dual preludes stay behaviorally intact."""
    m, x = _serial_model_and_batch()
    m.serial_object_meta = False
    real, _none = _count_unity_calls(m, x)
    assert real > 0, "unity must be offered every pump post-migration"
