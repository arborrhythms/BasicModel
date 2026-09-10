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
    clause, coarse, fine = fake._staged_tiling_ladder
    assert [tuple(x) for x in clause[0].tolist() if x[1] > x[0]] == [(0, 9), (10, 13)]   # cut at the comma
    assert fake._staged_unit_clause[0].tolist() == [0, 0, 0, 0, -1, 1]   # the comma bounds, belongs to none
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


# -- Phase 2b: chunk in the grammar, licensed by the tiling, admitted by utility --

def _present(m, surfaces):
    x = m.inputSpace.prepInput(list(surfaces))
    with torch.no_grad():
        m.forward(x)
    return x


def test_ladder_grammar_has_chunk_as_a_reduce_candidate(ladder):
    reducer = ladder._stm_reducer()
    assert reducer is not None and "chunk" in list(reducer.op_names)
    assert ladder._chunk_op_index() == list(reducer.op_names).index("chunk")


def test_chunk_is_licensed_only_inside_one_coarse_whole(ladder):
    m = ladder
    stm = m.conceptualSpace.stm
    stm.wholes_enable(1)
    # Two newest slots from different wholes: chunk forbidden.
    stm.note_whole_masked([True], [0], unit=0, clauses=[0]); stm.note_whole_masked([True], [1], unit=1, clauses=[1])
    window = torch.zeros(1, 2, int(stm.concept_dim))
    prior = m._chunk_structural_prior(stm, 1, window)
    idx = m._chunk_op_index()
    assert prior is not None and float(prior[0, 0, idx]) <= -1e3
    assert float(prior[0, 0].abs().sum()) == float(prior[0, 0, idx].abs())
    # Same whole: licensed with the learned prior (zero at init).
    stm.wholes_enable(1)
    stm.note_whole_masked([True], [3], unit=0, clauses=[0]); stm.note_whole_masked([True], [3], unit=1, clauses=[0])
    prior = m._chunk_structural_prior(stm, 1, window)
    assert float(prior[0, 0, idx]) == float(m._concept_owner().ensure_chunk_prior())
    assert stm.same_whole_rows(1) == [True]
    # A fold keeps the whole when both operands shared it.
    assert stm.newest_units(1) == [(0, 1)]
    stm.note_reduce_wholes([True])
    assert stm._slot_wholes[0] == [(3, -1, 0)]


def test_utility_counts_accrue_once_per_presentation():
    """Counts commit at the training path's sentence boundary, once per
    presentation (the bare per-row reset cascade is not the training path
    on this fixture; see the plan's open defects)."""
    m = _build_ladder()
    cs = m._concept_owner()
    opt = m.getOptimizer(lr=1e-3)
    m.runEpoch(optimizer=opt, batchSize=1, split="train", max_batches=2)
    cu = cs.utility_counts()
    assert cu["n"] == 2 and cu["n_c"] and cu["n_f"]
    assert all(1 <= v <= 2 for v in cu["n_c"].values())
    assert not cs.__dict__.get("_cu_proposals")      # drained at the boundary
    # Below the minimum evidence the utility is withheld; above it, defined.
    cs.utility_min_count = 3
    assert all(cs.category_utility(c) is None for c in cu["n_c"])
    cs.utility_min_count = 1
    defined = [cs.category_utility(c) for c in cu["n_c"]]
    assert any(v is not None for v in defined)       # units with features


def test_recurring_same_whole_pair_is_proposed_and_admitted_as_a_phrase():
    """Under digit wholes a two-digit numeral is a coarse whole over two
    digit units; a chunk chosen on that pair recurring ``admissionCount``
    times is admitted as a concept over the two member concepts.  (On a
    text corpus numerals rarely recur; here the mechanism is under test.)"""
    m = _build_ladder()
    cs = m._concept_owner()
    before = len(cs.__dict__.get("_chunk_admitted", {}))
    # Force the chooser toward chunk on licensed pairs so the mechanism fires
    # without training: a large structural prior.
    with torch.no_grad():
        cs.ensure_chunk_prior().fill_(50.0)
    opt = m.getOptimizer(lr=1e-3)
    m.runEpoch(optimizer=opt, batchSize=8, split="train", max_batches=6)
    admitted = cs.__dict__.get("_chunk_admitted", {})
    assert len(admitted) > before, "no chunked phrase was admitted"
    members, A = next(iter(admitted.items()))
    parts = cs.concept_parts(A)
    assert set(parts) == {("sym", int(mm)) for mm in members}
    with torch.no_grad():
        cs.chunk_prior.zero_()


# -- Phase 2, step 2: learned boundary predicates ---------------------------------

def test_boundary_predicates_reproduce_the_priors_tiling(ladder):
    ws = ladder.wholeSpaces[0]
    assert torch.is_tensor(ws.boundary_weight) and torch.is_tensor(ws.singleton_weight)
    rows = ws._predicate_rows
    on_b = (torch.sigmoid(ws.boundary_weight) > 0.5).tolist()
    on_s = (torch.sigmoid(ws.singleton_weight) > 0.5).tolist()
    from Layers import WHITESPACE, PUNCT, DIGIT, LETTER
    for r, cls in rows.items():
        if r >= len(on_b):
            continue
        if cls & {WHITESPACE, PUNCT}:
            assert on_b[r]
        if LETTER in cls and not (cls & {WHITESPACE, PUNCT}):
            assert not on_b[r]
        if DIGIT in cls:
            assert on_s[r]                           # digit wholes on this fixture
    # The learned-predicate cut equals the priors cut on the fixture surfaces.
    units, atoms, ids, mask, offsets = _stage(ladder, ["12 plus 1", "hi, there", "w0 abc123"])
    # Digit wholes are on for this fixture: every digit stands alone.
    assert units == [["1", "2", "plus", "1"], ["hi", ",", "there"],
                     ["w", "0", "abc", "1", "2", "3"]]


def test_boundary_types_none_starts_without_boundaries(tmp_path):
    import Language
    from util import init_config
    from data import TheData
    import Models
    src = (_DATA / "MM_ladder.xml").read_text()
    assert "<digitWholes>true</digitWholes>" in src
    config = tmp_path / "MM_ladder_none.xml"
    config.write_text(src.replace("<digitWholes>true</digitWholes>",
                                  "<digitWholes>true</digitWholes>\n    <boundaryTypes>none</boundaryTypes>", 1))
    init_config(path=str(config), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    cfg = Models.BaseModel.load_config(str(config))
    TheData.load("math", dat=dict(cfg["architecture"]["data"]))
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(config), data=TheData)
    ws = m.wholeSpaces[0]
    assert ws.boundary_types == "none"
    assert not bool((torch.sigmoid(ws.boundary_weight) > 0.5).any())
    assert not bool((torch.sigmoid(ws.singleton_weight) > 0.5).any())
    units, atoms, ids, mask, offsets = _stage(m, ["12 plus 1"])
    # The cold start is byte-complete (every byte a unit; whitespace stays a
    # discarded boundary-only class until it is demoted to a whole type).
    assert units[0] == ["1", "2", "p", "l", "u", "s", "1"]


# -- Phase 2, step 3: the cold start learns space as the basic boundary ---------

def test_cold_start_learns_space_as_the_basic_boundary(tmp_path):
    """Under <boundaryTypes>none</boundaryTypes> with the learner on, the
    successor corpus (words bounded by spaces, digits, punctuation-free)
    turns the whitespace row's boundary logit on within a few epochs,
    while the letter row's stays off: cutting at space flips yields the
    most recurring wholes at the lowest density."""
    import Language
    from util import init_config
    from data import TheData
    import Models
    from Layers import WHITESPACE, LETTER
    src = (_DATA / "MM_ladder.xml").read_text()
    config = tmp_path / "MM_ladder_learn.xml"
    config.write_text(src.replace(
        "<digitWholes>true</digitWholes>",
        "<digitWholes>true</digitWholes>\n    <boundaryTypes>none</boundaryTypes>\n"
        "    <boundaryLearningRate>8.0</boundaryLearningRate>", 1))
    init_config(path=str(config), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    cfg = Models.BaseModel.load_config(str(config))
    TheData.load("math", dat=dict(cfg["architecture"]["data"]))
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(config), data=TheData)
    ws = m.wholeSpaces[0]
    rows = ws._predicate_rows
    space_rows = [r for r, cl in rows.items() if WHITESPACE in cl and r < ws.boundary_weight.shape[0]]
    letter_rows = [r for r, cl in rows.items() if cl == {LETTER} and r < ws.boundary_weight.shape[0]]
    assert space_rows and letter_rows
    assert not bool((torch.sigmoid(ws.boundary_weight) > 0.5).any())
    opt = m.getOptimizer(lr=1e-3)
    for _ in range(3):
        m.runEpoch(optimizer=opt, batchSize=8, split="train", max_batches=8)
    on = (torch.sigmoid(ws.boundary_weight) > 0.5).tolist()
    # Whitespace turns on (its cut beats the byte-complete start on recurrence
    # and density); a row whose flips add no cut beyond the whitespace
    # boundaries (letters on this corpus) earns nothing and stays off.
    assert all(on[r] for r in space_rows), (ws.boundary_weight.tolist(), rows)
    assert not any(on[r] for r in letter_rows)
    units, atoms, ids, mask, offsets = _stage(m, ["12 plus 1"])
    assert "plus" in units[0]                        # space now bounds words


# -- Phase 1 acceptance (c): digit identity, order, repetition, translation ------

def test_digit_identity_order_and_repetition_are_distinct_occurrences(ladder):
    """``12``, ``21`` and ``11`` share the digit concepts and differ as
    ordered occurrences: the witness (atoms with spans) and the STM units
    differ, while a digit's concept is the same at any position."""
    m = ladder
    units, atoms, ids, mask, offsets = _stage(m, ["12 plus 1", "21 plus 1", "11 plus 1"])
    assert units[0][:2] == ["1", "2"] and units[1][:2] == ["2", "1"] and units[2][:2] == ["1", "1"]
    # Same digit, same atom row (reusable concept), different positions.
    one_rows = {int(ids[b, w][mask[b, w]][0]) for b, row in enumerate(units) for w, u in enumerate(row) if u == "1"}
    assert len(one_rows) == 1
    # The ordered witness distinguishes the three numerals.
    witness = [tuple((u, int(offsets[b, w, 0])) for w, u in enumerate(row) if u in ("1", "2"))
               for b, row in enumerate(units)]
    assert witness[0] != witness[1] != witness[2] and witness[0] != witness[2]
    # A digit's concept identity is the same at another position (translation).
    with torch.no_grad():
        m._lex_embed_stem(m.inputSpace.prepInput(["1 plus 12"]))
    cids = m.inputSpace._ar_word_concept_ids[0].tolist()
    u2 = m.perceptualSpace._forward_input["word_texts"][0]
    ones = [cids[w] for w, u in enumerate(u2) if u == "1"]
    assert len(ones) == 2 and ones[0] == ones[1] and ones[0] >= 0


# -- Phase 2b acceptance (mechanism): phrases on a text corpus ------------------

def test_recurring_phrases_are_admitted_on_a_text_corpus_with_positive_gain():
    """On the inline idiom / literal corpus (``kick the bucket`` vs ``kick
    the ball``, frequency-matched) adjacent words share a clause, so
    ``chunk`` is licensed; recurring pairs are admitted as concepts over
    their member concepts with a positive utility gain.  What is NOT yet
    shown: the admitted idiom's row diverging from the additive
    composition while the literal control stays compositional (that needs
    the phrase row wired into the answer path; plan, Phase 2b)."""
    import Language
    from util import init_config
    from data import TheData
    import Models
    config = _DATA / "MM_ladder_idiom.xml"
    init_config(path=str(config), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    cfg = Models.BaseModel.load_config(str(config))
    TheData.load("inline", dat=dict(cfg["architecture"]["data"]))
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(config), data=TheData)
    cs = m._concept_owner()
    with torch.no_grad():
        cs.ensure_chunk_prior().fill_(50.0)
    opt = m.getOptimizer(lr=1e-3)
    for _ in range(4):
        m.runEpoch(optimizer=opt, batchSize=4, split="train")
    admitted = cs.__dict__.get("_chunk_admitted", {})
    gains = cs.__dict__.get("_chunk_utility_gain", {})
    assert admitted, "no phrase admitted on the text corpus"
    assert all(gains[k] > 0.0 for k in admitted)
    for members, A in admitted.items():
        assert set(cs.concept_parts(A)) == {("sym", int(mm)) for mm in members}
    assert len(set(admitted.values())) == len(admitted)        # distinct concepts
