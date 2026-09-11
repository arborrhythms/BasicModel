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
    # Digit wholes are units; whitespace runs are units (the null operation).
    assert units[0] == ["1", "2", " ", "plus", " ", "1"]
    assert units[1] == ["hi", ",", " ", "there"]         # punctuation is a unit
    assert atoms[0][3] == [b"p", b"l", b"u", b"s"]       # bytes in surface order
    assert atoms[0][0] == [b"1"] and atoms[0][1] == [b"2"]
    assert offsets[0, :6, 0].tolist() == [0, 1, 2, 3, 7, 8]   # unit starts in bytes
    store = m.perceptualSpace.percept_store
    assert store.get_id(b"12") is None                   # never fused below the grammar


@pytest.mark.parametrize("surface", ["21 plus 1", "11 plus 1", "1 plus 12", "aab ba"])
def test_witness_replay_is_byte_exact(ladder, surface):
    m = ladder
    units, atoms, ids, mask, offsets = _stage(m, [surface])
    replay = b"".join(b"".join(unit) for unit in atoms[0])
    assert replay == surface.encode("ascii")             # whitespace units included
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
    assert units[0] == [long_word, " ", "y"]
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
    """The licensing is read from the STM's fixed-shape provenance slab
    ``[B, capacity, 3]`` (whole, unit, clause), the state the compiled
    word loop carries; no host mirror exists."""
    m = ladder
    stm = m.conceptualSpace.stm
    m._ensure_chunk_machinery()
    B = int(stm._buffer.shape[0])
    on = torch.ones(B, dtype=torch.bool)
    # Two newest slots from different wholes: chunk forbidden.
    stm._wholes = torch.full_like(stm.ensure_whole_state(), -1)
    stm.note_whole_masked(on, [0] * B, unit=0, clauses=[0] * B)
    stm.note_whole_masked(on, [1] * B, unit=1, clauses=[1] * B)
    window = torch.zeros(B, 2, int(stm.concept_dim))
    prior = m._chunk_structural_prior(stm.ensure_whole_state(), B, window)
    idx = m._chunk_op_index()
    assert prior is not None and float(prior[0, 0, idx]) <= -1e3
    assert float(prior[0, 0].abs().sum()) == float(prior[0, 0, idx].abs())
    # Same whole: licensed with the learned prior (zero at init).
    stm._wholes = torch.full_like(stm.ensure_whole_state(), -1)
    stm.note_whole_masked(on, [3] * B, unit=0, clauses=[0] * B)
    stm.note_whole_masked(on, [3] * B, unit=1, clauses=[0] * B)
    prior = m._chunk_structural_prior(stm.ensure_whole_state(), B, window)
    assert float(prior[0, 0, idx]) == float(m._concept_owner().ensure_chunk_prior())
    assert bool(stm.same_whole_rows()[0])
    assert stm.newest_units(stm._wholes)[0].tolist() == [0, 1]
    # A fold keeps the whole when both operands shared it; a fold is no unit.
    stm.note_reduce_wholes(on)
    assert stm._wholes[0, 0].tolist() == [3, -1, 0]
    assert stm._wholes[0, 1].tolist() == [-1, -1, -1]
    # The same primitives, ungated, are bit-identical.
    before = stm._wholes.clone()
    stm.note_whole_masked(torch.zeros(B, dtype=torch.bool), [7] * B, unit=2)
    assert torch.equal(before, stm._wholes)


def _chunk_forced_ladder():
    """A fresh ladder model on the tensor word loop whose ``chunk`` prior is
    saturated and whose reduce fires at every word, so a same-whole pair
    (the two digits of ``12``) is chunked deterministically."""
    m = _build_ladder()
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    m.stm_reduce_tau = 0.0
    m._ensure_chunk_machinery()
    with torch.no_grad():
        m._concept_owner().chunk_prior.fill_(20.0)
    return m


def _stage_tensor_peer(m, samples):
    from test_compiled_word_chunk import _stage_fullgraph_tensor_peer
    return _stage_fullgraph_tensor_peer(m, samples)


def test_provenance_and_chunk_proposals_ride_the_fullgraph_word_loop():
    """Alec (2026-09-11): compilation is essential, with fullgraph.  The
    STM provenance slab, the chunk licensing and the phrase proposals are
    loop-carried tensors: one graph, and the compiled forward's explicit
    chunk state equals the eager HOP's; ``Reset`` admits the phrase from
    the compiled proposals and the next compiled call reuses the graph."""
    m = _chunk_forced_ladder()
    cs = m._concept_owner()
    samples = ["12 plus 1", "34 plus 4"]

    def run(fn):
        _stage_tensor_peer(m, samples)
        with torch.no_grad():
            result = fn()
        m._publish_compiled_sentence_state(result)
        slab = cs._chunk_prop_slab.clone()
        count = cs._chunk_prop_count.clone()
        m.End(); m.symbolSpace.soft_reset()
        cs._commit_chunk_admissions()
        return slab, count

    eager_slab, eager_count = run(
        lambda: m._forward_with_compiled_sentence_state(None))
    assert eager_count.tolist() == [1, 1]
    ids = m.inputSpace  # the pair (left unit, right unit, shared whole)
    assert eager_slab[0, 0, 2].item() == 0 and eager_slab[0, 0, 0].item() >= 0

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(
        lambda: m._forward_with_compiled_sentence_state(None),
        backend="eager", fullgraph=True)
    try:
        slab, count = run(compiled)
        assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) == 1
        assert torch.equal(slab, eager_slab) and torch.equal(count, eager_count)
        members = tuple(eager_slab[0, 0, :2].tolist())
        assert members in cs.__dict__["_chunk_admitted"]      # second hit
        assert members in cs.__dict__["_chunk_rows"]
        table = cs._chunk_row_table
        row = cs.lookup_chunk_rows(
            table, torch.tensor([members[0]]), torch.tensor([members[1]]))
        assert int(row) == cs.__dict__["_chunk_rows"][members]
        # An admission changes table values, not the graph.
        slab2, _ = run(compiled)
        assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) == 1
        assert torch.equal(slab2[:, 0], eager_slab[:, 0])
    finally:
        torch._dynamo.reset()


def test_admitted_phrase_row_snaps_at_the_folded_slot():
    """Contract 7 inside the loop primitives: a chunk of an admitted pair
    references the phrase's row (activation one) at slot 0."""
    from Layers import ShortTermMemory
    m = _chunk_forced_ladder()
    cs = m._concept_owner()
    _stage_tensor_peer(m, ["12 plus 1"])
    stm = m.conceptualSpace.stm
    ids = m.inputSpace._ar_word_concept_ids
    members = (int(ids[0, 0]), int(ids[0, 1]))
    cs.__dict__["_chunk_rows"] = {members: 5}
    table = cs.chunk_row_table(ids.device)
    B = 1
    wholes = torch.full((B, int(stm.capacity), 3), -1, dtype=torch.long)
    on = torch.ones(B, dtype=torch.bool)
    for p in (0, 1):
        wholes = ShortTermMemory.functional_wholes_push(
            wholes, on, torch.tensor([0]), torch.tensor([p]), torch.tensor([0]))
    slab, count = cs.ensure_chunk_proposal_state(B, ids.device)
    op = torch.tensor([m._chunk_op_index()])
    wholes2, phrase_row, slab, count = m._chunk_reduce_provenance(
        wholes, op, on, slab, count, table=table)
    assert phrase_row.tolist() == [5] and count.tolist() == [1]
    assert wholes2[0, 0].tolist() == [0, -1, 0]
    state = (torch.zeros(B, int(stm.capacity), int(stm.concept_dim)),
             torch.tensor([1]), torch.full((B, int(stm.capacity)), -1),
             torch.full((B, int(stm.capacity)), -1),
             torch.full((B, int(stm.capacity)), -1),
             torch.zeros(B, int(stm.capacity)))
    snapped = cs.apply_phrase_rows(state, phrase_row, on)
    assert snapped[4][0, 0].item() == 5 and snapped[5][0, 0].item() == 1.0
    # Not applied: untouched.
    kept = cs.apply_phrase_rows(state, phrase_row, torch.zeros(B, dtype=torch.bool))
    assert kept[4][0, 0].item() == -1


# -- Word-unit assurance (Alec 2026-09-11): the tiling is learned, so the
# model reports how many whitespace words it actually stages as one unit.

def test_word_unit_fraction_is_one_on_plain_text(ladder):
    m = ladder
    m.reset_word_unit_stats()
    _stage(m, ["the quick brown fox", "jumps over it"])
    assert m.word_unit_fraction() == 1.0
    assert m.__dict__["_word_unit_stats"] == [7, 7, 7, 7]
    units = m.perceptualSpace._forward_input["word_texts"][0]
    assert [u for u in units if u.strip()] == ["the", "quick", "brown", "fox"]


def test_word_unit_fraction_counts_digit_wholes_as_sub_word_units(ladder):
    """``12`` is two digit units under ``digitWholes``; ``plus`` and ``1``
    are whole words: 2 of 3 words are one unit."""
    m = ladder
    m.reset_word_unit_stats()
    _stage(m, ["12 plus 1"])
    assert m.__dict__["_word_unit_stats"] == [3, 2, 1, 1]
    assert abs(m.word_unit_fraction() - 2.0 / 3.0) < 1e-9


def test_word_unit_fraction_is_zero_on_the_atomic_cold_start(tmp_path):
    m = _build_none_ladder(tmp_path)
    m.reset_word_unit_stats()
    _stage(m, ["the quick brown fox"])
    assert m.word_unit_fraction() == 0.0


def test_epoch_report_carries_the_word_unit_fraction(capsys):
    m = _build_ladder()
    opt = m.getOptimizer(lr=1e-3)
    m.runEpoch(optimizer=opt, batchSize=1, split="train", max_batches=1)
    assert m.word_unit_fraction() is not None
    out = capsys.readouterr().out
    if "Packed training throughput" in out:
        assert "word units" in out


def _build_ladder_variant(tmp_path, name, replacements, dataset="math"):
    import Language
    from util import init_config
    from data import TheData
    import Models
    src = (_DATA / "MM_ladder.xml").read_text()
    for old, new_ in replacements:
        assert old in src, old
        src = src.replace(old, new_, 1)
    config = tmp_path / f"MM_ladder_{name}.xml"
    config.write_text(src)
    init_config(path=str(config), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    cfg = Models.BaseModel.load_config(str(config))
    TheData.load(dataset, dat=dict(cfg["architecture"]["data"]))
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(config), data=TheData)
    return m


def test_packed_rows_are_laid_out_in_units_of_the_ladder(tmp_path):
    """Sentence packing budgets and lays out rows in the loop's own units
    (whitespace and digit units included), so the staged unit mask and the
    packed sentence ids agree and the stem's alignment check passes."""
    m = _build_ladder_variant(tmp_path, "w16", [
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>")])
    isp = m.inputSpace
    m._install_unit_span_fn()
    assert isp._unit_span_fn is not None
    # "12 plus 1" = 1,2,' ',plus,' ',1 (6 units) + the joining space.
    assert isp.sentence_unit_count("12 plus 1") == 6
    assert isp.sentence_unit_count("12 plus 1", trailing_space=True) == 7
    assert isp._unit_counts_by_sentence(["12 plus 1", "3 plus 4"]) == [7, 5]
    x = isp.prepPackedInput([["12 plus 1", "3 plus 4"], ["ab cd"]])
    ids = isp._packed_sentence_ids
    assert ids[0].tolist() == [0] * 7 + [1] * 5 + [-1] * 4
    assert ids[1].tolist() == [0] * 3 + [-1] * 13
    assert isp._packed_sentence_intermediate_end_mask[0, 6].item()
    with torch.no_grad():
        m._lex_embed_stem(x)                     # the alignment check passes
    active = isp._word_active_mask
    assert torch.equal(active.to("cpu"), (ids >= 0).to("cpu"))
    units = m.perceptualSpace._forward_input["word_texts"]
    assert units[0] == ["1", "2", " ", "plus", " ", "1", " ", "3", " ", "plus", " ", "4"]
    # A brick that does not fit in units is refused, not clipped.
    import pytest
    with pytest.raises(ValueError, match="units"):
        isp.prepPackedInput([["12 plus 1", "3 plus 4", "5 plus 6"]])


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
    assert all(torch.is_tensor(getattr(ws, n)) for n in ("begins_weight", "ends_weight", "atom_level"))
    begins_on, ends_on, atom_on, discard_on = ws._predicate_masks()
    from Layers import WHITESPACE, PUNCT, DIGIT, LETTER
    A = ws._ATOMIC_COLUMNS
    for r, cls in ws._predicate_rows.items():
        col = A + r
        if cls & {WHITESPACE, PUNCT}:
            assert bool(begins_on[col]) and bool(ends_on[col])
        if cls == {LETTER}:
            assert not bool(begins_on[col]) and not bool(ends_on[col])
        if DIGIT in cls:
            assert bool(atom_on[col])                # digit wholes on this fixture
    assert not bool(begins_on[:A].any())             # atomic columns start off
    assert not bool(discard_on[1:A].any())           # whitespace units: only the pad is discarded
    units, atoms, ids, mask, offsets = _stage(ladder, ["12 plus 1", "hi, there", "w0 abc123"])
    assert units == [["1", "2", " ", "plus", " ", "1"], ["hi", ",", " ", "there"],
                     ["w", "0", " ", "abc", "1", "2", "3"]]


def _build_none_ladder(tmp_path):
    """The ladder fixture with ``<boundaryTypes>none</boundaryTypes>``: the
    atomic cold start (every byte a whole)."""
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
    return m


def test_boundary_types_none_starts_without_boundaries(tmp_path):
    m = _build_none_ladder(tmp_path)
    ws = m.wholeSpaces[0]
    assert ws.boundary_types == "none"
    b_on, e_on, a_on, _ = ws._predicate_masks()
    assert not bool(b_on.any() or e_on.any() or a_on.any())
    units, atoms, ids, mask, offsets = _stage(m, ["12 plus 1"])
    # The cold start is the atomic tiling: every byte a whole, spaces included.
    assert units[0] == ["1", "2", " ", "p", "l", "u", "s", " ", "1"]


# -- Phase 2, step 3: the cold start learns space as the basic boundary ---------

def test_cold_start_learns_space_as_the_basic_boundary(tmp_path):
    """Under <boundaryTypes>none</boundaryTypes> with the learner on, a
    small varied text corpus (the idiom fixture's sentences) turns a
    whitespace boundary on within a few epochs: cutting where space runs
    begin or end yields the most recurring wholes at the lowest density,
    while a letter byte's own boundaries do not recur across sentences."""
    import Language
    from util import init_config
    from data import TheData
    import Models
    from Layers import WHITESPACE
    import random, re
    src = (_DATA / "MM_ladder_idiom.xml").read_text()
    # A varied corpus: words recur, longer chunks rarely do.
    rng = random.Random(7)
    vocab = ["the", "cat", "dog", "saw", "a", "big", "red", "ball", "ran", "to", "old", "man"]
    sents = [" ".join(rng.choice(vocab) for _ in range(rng.randint(3, 6))) for _ in range(48)]
    labels = [str(i % 2) for i in range(len(sents))]
    src = re.sub(r'<input use="train">[^<]*</input>', '<input use="train">' + "|".join(sents) + "</input>", src)
    src = re.sub(r'<output use="train">[^<]*</output>', '<output use="train">' + "|".join(labels) + "</output>", src)
    src = re.sub(r'<input use="test">[^<]*</input>', '<input use="test">' + "|".join(sents[:8]) + "</input>", src)
    src = re.sub(r'<output use="test">[^<]*</output>', '<output use="test">' + "|".join(labels[:8]) + "</output>", src)
    config = tmp_path / "MM_ladder_learn.xml"
    config.write_text(src.replace(
        "<digitWholes>false</digitWholes>",
        "<digitWholes>false</digitWholes>\n    <boundaryTypes>none</boundaryTypes>\n"
        "    <boundaryLearningRate>40.0</boundaryLearningRate>\n"
        "    <boundaryUpdateEvery>16</boundaryUpdateEvery>", 1))
    init_config(path=str(config), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    cfg = Models.BaseModel.load_config(str(config))
    TheData.load("inline", dat=dict(cfg["architecture"]["data"]))
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(config), data=TheData)
    ws = m.wholeSpaces[0]
    A = ws._ATOMIC_COLUMNS
    space_cols = [A + r for r, cl in ws._predicate_rows.items() if WHITESPACE in cl]
    assert space_cols
    b0, e0, a0, _ = ws._predicate_masks()
    assert not bool(b0.any() or e0.any() or a0.any())
    opt = m.getOptimizer(lr=1e-3)
    for _ in range(6):
        m.runEpoch(optimizer=opt, batchSize=4, split="train")
    b_on, e_on, a_on, _ = ws._predicate_masks()
    assert any(bool(b_on[c]) or bool(e_on[c]) or bool(a_on[c]) for c in space_cols), \
        (ws.begins_weight[space_cols].tolist(), ws.ends_weight[space_cols].tolist())
    letter_bytes = list(range(ord("a"), ord("z") + 1))
    assert not any(bool(b_on[c]) or bool(e_on[c]) or bool(a_on[c]) for c in letter_bytes)
    units, atoms, ids, mask, offsets = _stage(m, ["the cat saw a dog"])
    # Words are units now; whether the space is its own unit or attached to
    # the word depends on which of the two space boundaries turned on first
    # (both are legitimate under the memory-load criterion).
    stripped = [u.strip() for u in units[0]]
    assert "cat" in stripped and "dog" in stripped


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


# -- contracts 6-7: the counts, admissions and acquired predicates round-trip ----

def test_utility_state_and_predicates_round_trip_a_checkpoint(tmp_path):
    m = _build_ladder()
    cs = m._concept_owner()
    ws = m.wholeSpaces[0]
    with torch.no_grad():
        cs.ensure_chunk_prior().fill_(50.0)
    opt = m.getOptimizer(lr=1e-3)
    m.runEpoch(optimizer=opt, batchSize=8, split="train", max_batches=6)
    ws.__dict__.setdefault("_row_bytes", {})[7] = [ord("a"), ord("e")]   # an acquired predicate
    counts = cs.utility_counts()["n"]
    admitted = dict(cs.__dict__.get("_chunk_admitted", {}))
    rows = dict(cs.__dict__.get("_chunk_rows", {}))
    assert counts > 0 and admitted
    path = tmp_path / "ladder.ckpt"
    m.save_weights(str(path))
    m2 = _build_ladder()
    m2.load_weights(str(path), require_match=False)
    cs2 = m2._concept_owner(); ws2 = m2.wholeSpaces[0]
    assert cs2.utility_counts()["n"] == counts
    assert {tuple(k): v for k, v in cs2.__dict__.get("_chunk_admitted", {}).items()} == \
        {tuple(k): v for k, v in admitted.items()}
    assert {tuple(k): v for k, v in cs2.__dict__.get("_chunk_rows", {}).items()} == \
        {tuple(k): v for k, v in rows.items()}
    assert list(ws2.__dict__.get("_row_bytes", {}).get(7, [])) == [ord("a"), ord("e")]
    # The learned boundary weights are parameters and round-trip too.
    assert torch.equal(ws2.begins_weight.detach(), ws.begins_weight.detach())
