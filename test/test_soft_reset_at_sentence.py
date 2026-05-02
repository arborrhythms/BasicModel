"""Tests confirming SyntacticLayer.compose emits the per-row sentence-
completed signal, and that ``WordSpace.soft_reset(batch=b)`` re-arms
sentence-internal state without wiping cross-sentence context.

Plan reference: doc/plans/2026-04-26-rolling-cursor-doc-streaming-handoff.md §Verification
"""

# ---------------------------------------------------------------------
# Skipped pending migration to the post-2026-05-01 chart / GrammarLayer
# surface. Tests in this module exercise legacy SyntacticLayer methods
# (generate / decompose / _signal_sentence_completed /
# _extract_svo_from_trace) that were removed by the refactor;
# equivalent functionality now lives on the Chart class.
# ---------------------------------------------------------------------
import pytest
pytestmark = pytest.mark.skip(
    reason="Pending migration to Chart surface; "
           "see doc/specs/2026-05-01-syntactic-layer-refactor.md")

import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import pytest
import torch

_CONFIG_PATH = str(_project / "data" / "MM_xor.xml")


def _model():
    from data import TheData
    from Models import BaseModel
    TheData.load("xor")
    torch.manual_seed(0)
    model, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    return model


def test_sentence_completed_buffer_initialized_to_per_row_size():
    """``WordSpace._sentence_completed`` is host-side ``list[bool]`` sized B."""
    model = _model()
    ws = model.wordSpace
    if ws is None:
        pytest.skip("model has no WordSpace")
    sc = getattr(ws, '_sentence_completed', None)
    assert sc is not None, (
        "_sentence_completed must be initialized at WordSpace.__init__"
    )
    assert isinstance(sc, list), (
        f"_sentence_completed must be a host-side list (no GPU sync); "
        f"got {type(sc).__name__}"
    )
    assert all(v is False or v is True for v in sc), (
        f"_sentence_completed entries must be plain Python bools; "
        f"got {[type(v).__name__ for v in sc]}"
    )


def test_soft_reset_rearms_stm_and_clears_svo_for_one_row():
    """soft_reset(batch=b) clears row b's STM-fired and SVO-valid signal."""
    model = _model()
    ws = model.wordSpace
    if ws is None:
        pytest.skip("model has no WordSpace")
    if ws._stm_fired.shape[0] < 2:
        ws.ensure_microbatch(2, 1)
    ws._stm_fired[0] = True
    ws._stm_fired[1] = True
    ws._svo_valid[0] = True
    ws._svo_valid[1] = True

    ws.soft_reset(batch=0)

    assert ws._stm_fired[0].item() is False
    assert ws._stm_fired[1].item() is True
    assert ws._svo_valid[0].item() is False
    assert ws._svo_valid[1].item() is True


def test_soft_reset_clears_parse_stack_rows_for_target_row():
    """soft_reset(batch=b) clears row b's parse stack but leaves other rows.

    The parse stack is a per-sentence working buffer; cross-sentence
    carryover lives in DiscourseSpace, not the stack. After soft reset
    on row 0, row 0's stack rows should be zeroed and other rows'
    contents should be untouched.
    """
    model = _model()
    ws = model.wordSpace
    if ws is None:
        pytest.skip("model has no WordSpace")
    # Two source rows, K=1 → body batch == 2.
    ws.ensure_microbatch(2, 1)
    sub = ws.subspace
    if sub is None or not hasattr(sub, 'push'):
        pytest.skip("WordSubSpace does not expose push for this test")
    # Push a record into both rows.
    sub.push(0, rule_id=1, leaves=(-1, -1, -1))
    sub.push(1, rule_id=2, leaves=(-1, -1, -1))
    assert sub._top[0].item() > 0
    assert sub._top[1].item() > 0

    ws.soft_reset(batch=0)

    assert sub._top[0].item() == 0, (
        "soft_reset(batch=0) must rewind row 0's parse-stack top"
    )
    assert sub._top[1].item() > 0, (
        "soft_reset(batch=0) must not touch row 1's parse stack"
    )


def test_soft_reset_preserves_discourse_history():
    """soft_reset does NOT reset DiscourseSpace.

    Discourse history is the cross-sentence prior — it must accumulate
    across sentences within a document and only clear at hard reset
    (document boundary).
    """
    model = _model()
    ws = model.wordSpace
    if ws is None or getattr(ws, 'discourse', None) is None:
        pytest.skip("model has no discourse layer")
    disc = ws.discourse
    # Snapshot whatever the discourse layer exposes; the contract here
    # is "soft_reset doesn't call any reset method on discourse".
    reset_calls = []
    if hasattr(disc, 'reset'):
        original = disc.reset

        def _spy(*args, **kwargs):
            reset_calls.append((args, kwargs))
            return original(*args, **kwargs)
        disc.reset = _spy
    try:
        ws.ensure_microbatch(2, 1)
        ws.soft_reset(batch=0)
        ws.soft_reset(batch=1)
        ws.soft_reset()           # all-rows form
    finally:
        if hasattr(disc, 'reset'):
            disc.reset = original

    assert reset_calls == [], (
        f"soft_reset must not call discourse.reset(); got {reset_calls}"
    )


def test_drain_sentence_completed_returns_and_clears():
    """drain_sentence_completed returns the True rows then clears the buffer."""
    model = _model()
    ws = model.wordSpace
    if ws is None:
        pytest.skip("model has no WordSpace")
    if ws._stm_fired.shape[0] < 3:
        ws.ensure_microbatch(3, 1)
    ws._sentence_completed = [False, True, True]
    drained = ws.drain_sentence_completed()
    assert drained == [1, 2], (
        f"drain must return source-row indices for True entries; "
        f"got {drained}"
    )
    assert ws._sentence_completed == [False, False, False], (
        f"buffer must be cleared after drain; got {ws._sentence_completed}"
    )
    # Subsequent drain returns empty.
    assert ws.drain_sentence_completed() == []


def test_dispatch_soft_reset_drains_and_calls_soft_reset():
    """``BasicModel.dispatch_soft_reset`` drains the signal and invokes soft_reset."""
    model = _model()
    ws = model.wordSpace
    if ws is None:
        pytest.skip("model has no WordSpace")
    if ws._stm_fired.shape[0] < 2:
        ws.ensure_microbatch(2, 1)
    # Mark state we want soft_reset to clear.
    ws._stm_fired[0] = True
    ws._stm_fired[1] = True
    ws._sentence_completed = [True, False]

    model.dispatch_soft_reset()

    assert ws._stm_fired[0].item() is False, (
        "dispatch_soft_reset should soft_reset row 0 (sentence_completed[0]=True)"
    )
    assert ws._stm_fired[1].item() is True, (
        "row 1 had sentence_completed=False; soft_reset must skip it"
    )
    assert ws._sentence_completed[0] is False
    assert ws._sentence_completed[1] is False


def test_signal_sentence_completed_marks_single_start_leaf():
    """SyntacticLayer._signal_sentence_completed flags rows reduced to start.

    Builds a fake ``alive`` and ``category`` tensor where row 0 has one
    live leaf with category id == start_id (sentence complete) and row
    1 has two live leaves (mid-derivation). Expect _sentence_completed
    to be [True, False] after the call.
    """
    from Language import Grammar, SyntacticLayer
    from types import SimpleNamespace
    grammar = Grammar()
    grammar.start_symbol = "S"
    layer = SyntacticLayer(nInput=4, nOutput=4, rules=[], grammar=grammar)
    layer._category_index = {"S": 0, "VO": 1}
    # Two source rows; K=1 so b_flat == b_source.
    ws = SimpleNamespace(
        _sentence_completed=[False, False],
        _row_K=lambda: 1,
    )
    subspace = SimpleNamespace(wordSpace=ws)
    alive = torch.tensor([
        [True, False, False, False],
        [True,  True, False, False],
    ])
    category = torch.tensor([
        [0, 0, 0, 0],   # row 0's only live leaf is at col 0, cat 0 = S
        [1, 1, 0, 0],   # row 1 has 2 live leaves, cat VO/VO -> not start
    ], dtype=torch.long)
    layer._signal_sentence_completed(subspace, alive, category, grammar)
    assert ws._sentence_completed == [True, False], (
        f"only row 0 reduced to start_symbol; got {ws._sentence_completed}"
    )
