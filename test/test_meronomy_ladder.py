"""Meronomy as the fold ladder (doc/plans/2026-09-10-meronomy-fold-ladder.md).

Phase 0: the Legacy move and the symbolic loop's ``forward`` / ``reverse``
entries, byte-identical to the previous ``compose`` / ``generate`` and to
the previous analysis cuts.
"""
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
_DATA = _ROOT / "data"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from Spaces import WholeSpace  # noqa: E402
import Legacy  # noqa: E402


def _bytes(s):
    return torch.tensor([list(s.encode("ascii"))], dtype=torch.long)


# -- Phase 0: the analysis cuts route through Legacy, byte-identically ----------

def test_legacy_analysis_modes_are_parked_and_meronomy_is_canonical():
    assert Legacy.LEGACY_WHOLE_ANALYSIS_MODES == {
        "byte", "raw", "sentence", "word", "grammatical"}
    assert "meronomy" not in Legacy.LEGACY_WHOLE_ANALYSIS_MODES
    with pytest.raises(ValueError, match="legacy WholeSpace analysis"):
        Legacy.normalize_whole_analysis_mode("meronomy")


@pytest.mark.parametrize("surface", ["abc123", "hi, there", "12 plus 1", "a...b"])
def test_word_and_grammatical_cuts_equal_the_meronomy_cut(surface):
    u = _bytes(surface)
    canonical = WholeSpace.stage_analysis_spans(
        types.SimpleNamespace(analysis_mode="meronomy"), u)
    for mode in ("word", "grammatical"):
        legacy = WholeSpace.stage_analysis_spans(
            types.SimpleNamespace(analysis_mode=mode), u)
        assert torch.equal(legacy, canonical), (mode, legacy, canonical)


@pytest.mark.parametrize("mode", ["byte", "raw", "sentence"])
def test_undivided_legacy_modes_stage_no_spans(mode):
    fake = types.SimpleNamespace(analysis_mode=mode)
    assert WholeSpace.stage_analysis_spans(fake, _bytes("abc def")) is None
    assert fake._staged_property_signatures is None


def test_meronomy_cut_with_no_unity_stages_nothing():
    fake = types.SimpleNamespace(analysis_mode="meronomy")
    assert WholeSpace.stage_analysis_spans(fake, None) is None


# -- Phase 0: the symbolic loop is called as forward / reverse -----------------

def test_language_space_forward_and_reverse_are_the_entries():
    import inspect
    from Language import LanguageSpace
    assert callable(getattr(LanguageSpace, "forward", None))
    assert callable(getattr(LanguageSpace, "reverse", None))
    # The legacy spellings delegate to the entries (no second implementation).
    assert "self.forward(" in inspect.getsource(LanguageSpace.compose)
    assert "self.reverse(" in inspect.getsource(LanguageSpace.generate)


def test_model_calls_the_symbolic_loop_as_forward():
    src = (_BIN / "Models.py").read_text()
    assert "languageSpace.compose(" not in src
    assert "languageSpace.generate(" not in src
    assert "languageSpace.forward(" in src


def test_forward_equals_compose_on_a_live_snapshot():
    """The rename is byte-identical: the plan returned by ``forward`` equals
    the plan returned by the legacy ``compose`` on the same snapshot."""
    import Language
    from util import init_config
    from data import TheData
    import Models

    config = _DATA / "MM_xor.xml"
    init_config(path=str(config), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load("xor")
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(config), data=TheData)
    m = m.to("cpu").eval()
    ls = getattr(m, "languageSpace", None)
    if ls is None:
        pytest.skip("fixture has no LanguageSpace")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=2)
    inputs, _ = next(iter(loader))
    x = m.inputSpace.prepInput(inputs)
    with torch.no_grad():
        m.forward(x)
    snap = getattr(m, "_current_discourse_s", None)
    if not torch.is_tensor(snap):
        pytest.skip("no symbolic snapshot on this fixture")

    def _flat(plan):
        if torch.is_tensor(plan):
            return [plan]
        if isinstance(plan, (tuple, list)):
            return [t for item in plan for t in _flat(item)]
        return []

    with torch.no_grad():
        a = ls.forward(snap.clone())
        b = ls.compose(snap.clone())
    fa, fb = _flat(a), _flat(b)
    assert len(fa) == len(fb)
    for ta, tb in zip(fa, fb):
        assert torch.equal(ta, tb)


# -- Phase 1, step A: the ladder stem (units = staged wholes, atoms = bytes) --

def _build_ladder(dat=None):
    import Language
    from util import init_config
    from data import TheData
    import Models

    config = _DATA / "MM_ladder.xml"
    init_config(path=str(config), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    cfg = Models.BaseModel.load_config(str(config))
    TheData.load("math", dat=dict(cfg["architecture"]["data"], **(dat or {})))
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(config), data=TheData)
    return m.to("cpu")


@pytest.fixture(scope="module")
def ladder():
    return _build_ladder()


def _stage(m, surfaces):
    """Run the eager stem on raw surfaces; return (units, atom bytes per unit)."""
    ps = m.perceptualSpace
    x = m.inputSpace.prepInput(list(surfaces))
    with torch.no_grad():
        m._lex_embed_stem(x)
    units = ps._forward_input["word_texts"]
    ids, mask = m.inputSpace._ar_word_part_ids, m.inputSpace._ar_word_part_mask
    atoms = [[[ps.percept_store.bytes_for(int(i)) for i in ids[b, w][mask[b, w]].tolist()]
              for w in range(ids.shape[1]) if bool(mask[b, w].any())]
             for b in range(ids.shape[0])]
    return units, atoms, ids, mask, m.inputSpace._ar_word_part_offsets


def test_ladder_stem_units_are_the_staged_wholes_and_atoms_are_bytes(ladder):
    m = ladder
    assert m.perceptualSpace._meronomy and m.wholeSpaces[0].digit_wholes
    units, atoms, ids, mask, offsets = _stage(m, ["12 plus 1", "hi, there"])
    assert units[0] == ["1", "2", "plus", "1"]            # digit wholes are units
    assert units[1] == ["hi", ",", "there"]              # punctuation is a unit
    assert atoms[0][2] == [b"p", b"l", b"u", b"s"]       # bytes in surface order
    assert atoms[0][0] == [b"1"] and atoms[0][1] == [b"2"]
    assert offsets[0, :4, 0].tolist() == [0, 1, 3, 8]    # unit starts in bytes
    store = m.perceptualSpace.percept_store
    assert store.get_id(b"12") is None                   # never fused below the grammar


@pytest.mark.parametrize("surface", ["21 plus 1", "11 plus 1", "1 plus 12", "aab ba"])
def test_witness_replay_is_byte_exact(ladder, surface):
    m = ladder
    units, atoms, ids, mask, offsets = _stage(m, [surface])
    replay = b"".join(b"".join(unit) for unit in atoms[0])
    assert replay == surface.replace(" ", "").encode("ascii")
    # Every atom has its exact span; spans are in surface order.
    spans = m.perceptualSpace._forward_input["part_spans"][0]
    live = [tuple(sp) for sp, ok in zip(spans.tolist(), mask[0].reshape(-1).tolist()) if ok]
    assert live == sorted(live) and all(e == s + 1 for s, e in live)


def test_long_unit_presents_every_atom(ladder):
    """The eager stem never truncates a unit: a 20-letter word is 20 atoms
    (the compiled loop's fixed capacity keeps its own fail-loud contract)."""
    m = ladder
    long_word = "abcdefghijklmnopqrst"
    units, atoms, ids, mask, offsets = _stage(m, [long_word + " y"])
    assert units[0] == [long_word, "y"]
    assert len(atoms[0][0]) == 20
    assert b"".join(atoms[0][0]) == long_word.encode("ascii")
    assert not bool(m.inputSpace._ar_word_truncated_mask.any())


def test_rung_zero_is_the_max_over_atoms(ladder):
    from Layers import MeronymicFoldAdapter
    sigma = m_sigma = ladder.perceptualSpace.sigmas[0]
    assert getattr(sigma, "set_law", None) == "max"
    codes = torch.tensor([[[0.2, 0.9, 0.0], [0.7, 0.1, 0.0], [0.5, 0.5, 0.5]]])
    mask = torch.tensor([[True, True, False]])
    out = sigma.compute_aggregate_over_set(codes, mask=mask)
    assert torch.allclose(out, torch.tensor([[0.7, 0.9, 0.0]]))
    # Idempotent: a repeated part does not strengthen the whole.
    twice = sigma.compute_aggregate_over_set(codes[:, [0, 0]], mask=torch.tensor([[True, True]]))
    assert torch.allclose(twice, codes[:, 0])
    # The legacy law is untouched for non-meronomy adapters.
    legacy = MeronymicFoldAdapter.__new__(MeronymicFoldAdapter)
    assert getattr(legacy, "set_law", "union") == "union"


# -- Phase 1, step D-0: rung-0 admission by recurrence --------------------------

def test_recurring_units_are_admitted_at_rung_zero_and_digits_never_fuse():
    m = _build_ladder()                      # fresh store: no prior sightings
    ps = m.perceptualSpace
    store = ps.percept_store
    threshold = max(1, int(getattr(ps, "chunk_promotion_threshold", 2) or 2))
    assert store.get_id(b"plus") is None
    for _ in range(threshold):
        _stage(m, ["12 plus 1"])
        ps.flush_pending_promotions()
    assert store.get_id(b"plus") is not None            # the recurring unit
    assert store.get_id(b"12") is None                  # digits are separate units
    row = store._basis.lookup_rows(torch.tensor([store.get_id(b"plus")]))[0]
    atoms = store._basis.lookup_rows(torch.tensor(
        [store.get_id(bytes([c])) for c in b"plus"])).clamp(0.0, 1.0).amax(dim=0)
    assert torch.allclose(row[: atoms.shape[-1]], atoms)   # seeded with the rung-0 max
    # Evaluation freezes admission.
    object.__setattr__(ps, "_online_learning_frozen", True)
    try:
        for _ in range(threshold + 1):
            _stage(m, ["hello world"])
            ps.flush_pending_promotions()
        assert store.get_id(b"hello") is None
    finally:
        object.__setattr__(ps, "_online_learning_frozen", False)


# -- Phase 2, step 1: the descending tiling ladder (coarse over fine) -----------

def test_tiling_ladder_nests_units_in_space_bounded_wholes():
    fake = types.SimpleNamespace(analysis_mode="meronomy", digit_wholes=True)
    WholeSpace.stage_analysis_spans(fake, _bytes("12 plus 1, ok"))
    coarse, fine = fake._staged_tiling_ladder
    assert [tuple(x) for x in coarse[0].tolist() if x[1] > x[0]] == [(0, 2), (3, 7), (8, 10), (11, 13)]
    assert [tuple(x) for x in fine[0].tolist() if x[1] > x[0]] == \
        [(0, 1), (1, 2), (3, 7), (8, 9), (9, 10), (11, 13)]
    assert fake._staged_unit_parent[0].tolist() == [0, 0, 1, 2, 2, 3]
    assert torch.equal(fake._staged_unit_spans, fine)
    # Letter/digit flips are not unit boundaries (contract 3 priors): one unit.
    fake = types.SimpleNamespace(analysis_mode="meronomy", digit_wholes=False)
    WholeSpace.stage_analysis_spans(fake, _bytes("w0 abc123"))
    assert [tuple(x) for x in fake._staged_unit_spans[0].tolist() if x[1] > x[0]] == [(0, 2), (3, 9)]
    # Without digit wholes the digit run is one unit under one coarse whole.
    fake = types.SimpleNamespace(analysis_mode="meronomy", digit_wholes=False)
    WholeSpace.stage_analysis_spans(fake, _bytes("12 plus 1"))
    assert fake._staged_unit_parent[0].tolist() == [0, 1, 2]
