"""The PS word store (type="words") for the reverse() recommenders.

doc/plans/2026-07-12-word-store-typed-reverse.md (v2, Alec's review): the
word store IS the PS RadixLayer's promoted collection — this pass LABELS it
(``RadixLayer.word_ids`` / ``word_text``, no new state) and, gated
``<PartSpace><wordStore>``, routes the Method-2 free-derivation un-fold's
recommender candidates through it (basis = PS ``subspace.what``, restricted
to the word rows via ``left_rows``/``right_rows``; percept id == row).
"""

import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

_DATA = os.path.join(_PROJECT, "data")
_SMOKE = os.path.join(_DATA, "MM_meronomy_smoke.xml")
_DEFAULTS = os.path.join(_DATA, "model.xml")


# -- the label: RadixLayer's typed word view ---------------------------------

def _standalone_store():
    from Layers import RadixLayer
    ps = RadixLayer(8, promotion_threshold=2, promotion_min_length=2,
                    word_bounded=True)
    return ps


def test_word_ids_are_promoted_multibyte_entries():
    ps = _standalone_store()
    assert ps.WORDS_TYPE == "words"
    assert ps.word_ids().numel() == 0            # empty store: no words
    # Two sightings promote a word; spell_out seeds byte percepts.
    ps.observe_chunk(b"hello")
    ps.observe_chunk(b"hello")
    ps.spell_out(b"hello")                       # one promoted percept now
    ps.spell_out(b"cat")                         # unpromoted: seeds c/a/t bytes
    ids = ps.word_ids().tolist()
    texts = {ps.word_text(i) for i in ids}
    assert texts == {"hello"}                    # no seeded byte percepts
    hello_id = ps.get_id(b"hello")
    assert hello_id in ids


def test_single_byte_word_needs_standalone_attestation():
    ps = _standalone_store()
    ps.spell_out(b"a")                           # seeds the 'a' byte percept
    assert ps.word_ids().numel() == 0            # seeded byte alone: not a word
    ids = ps.word_ids(standalone_bytes={ord("a")}).tolist()
    assert {ps.word_text(i) for i in ids} == {"a"}


def test_word_ids_index_the_shared_basis_rows():
    ps = _standalone_store()
    ps.observe_chunk(b"dog")
    ps.observe_chunk(b"dog")
    ps.spell_out(b"dog")
    wid = int(ps.word_ids()[0])
    # percept id == row index of the store's codebook (the shared basis).
    assert 0 <= wid < int(ps.codebook.shape[0])
    assert ps.bytes_for(wid) == b"dog"


# -- the recommender's typed restriction (the machinery the unfold drives) ---

def test_radial_recommender_recovers_signed_pair():
    """Alec 2026-07-13: 'min should be a radial min to deal with signed
    activations' — the radial filters recover a SIGNED pair from its
    radmax fold, where the lattice filters degenerate to sentinels."""
    from Layers import Ops
    # Same-sign dims fold to the larger magnitude; the OPPOSITE-sign dim
    # (index 3) ANNIHILATES to zero — the wildcard the filters must honor.
    w1 = torch.tensor([-0.9, 0.2, 0.0, -0.1])
    w2 = torch.tensor([-0.3, 0.8, 0.1, 0.6])
    W = torch.stack([w1, w2])
    y = Ops._radmax(w1, w2).unsqueeze(0)         # the radial fold parent
    assert float(y[0, 3]) == 0.0                 # annihilated dim
    x1, x2 = Ops._binary_op_recommend(y, W, 'union', radial=True)
    got = {tuple(x1.reshape(-1).tolist()), tuple(x2.reshape(-1).tolist())}
    assert got == {tuple(w1.tolist()), tuple(w2.tolist())}


def test_recommender_row_restriction_recovers_stored_pair():
    from Layers import Ops
    w1 = torch.tensor([0.9, 0.9, 0.9, 0.0])
    w2 = torch.tensor([0.0, 0.9, 0.9, 0.9])
    decoy = torch.tensor([0.9, 0.9, 0.9, 0.9])   # would win unrestricted
    W = torch.stack([decoy, w1, w2])
    y = torch.maximum(w1, w2).unsqueeze(0)
    rows = torch.tensor([1, 2])
    x1, x2 = Ops._binary_op_recommend(y, W, 'union',
                                      left_rows=rows, right_rows=rows)
    got = {tuple(x1.reshape(-1).tolist()), tuple(x2.reshape(-1).tolist())}
    assert got == {tuple(w1.tolist()), tuple(w2.tolist())}


# -- model fixture ------------------------------------------------------------

_CACHE = {}


def _build(tmp_path_factory, word_store=True):
    key = "on" if word_store else "off"
    if key in _CACHE:
        return _CACHE[key]
    import Models
    import Language
    from util import init_config, TheXMLConfig
    with open(_SMOKE) as f:
        xml = f.read()
    if word_store:
        # Anchor on the REAL element (line-start indent) — the header
        # comment also mentions <synthesis>meronomy</synthesis>.
        anchor = "\n    <synthesis>meronomy</synthesis>"
        assert anchor in xml
        xml = xml.replace(
            anchor,
            anchor + "\n    <chunkPromotionThreshold>2"
                     "</chunkPromotionThreshold>\n"
                     "    <wordStore>true</wordStore>", 1)
    d = tmp_path_factory.mktemp("word_store")
    p = os.path.join(str(d), f"MM_word_store_{key}.xml")
    with open(p, "w") as f:
        f.write(xml)
    init_config(path=p, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    Models.TheData.load(TheXMLConfig.get("data.dataset", default="phrases"))
    _CACHE[key] = m
    return m


def _train_forward(m):
    """One training-mode forward over one batch; returns the raw items."""
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    m.train()
    m.forward(x)
    return items


# -- gating + the live collection ---------------------------------------------

def test_knob_default_off(tmp_path_factory):
    m = _build(tmp_path_factory, word_store=False)
    assert getattr(m.perceptualSpace, "word_store_reverse", None) is False


def test_reading_promotes_the_batch_words(tmp_path_factory):
    m = _build(tmp_path_factory, word_store=True)
    assert m.perceptualSpace.word_store_reverse is True
    items = _train_forward(m)
    items = _train_forward(m)                    # 2 sightings >= threshold
    store = m.perceptualSpace.percept_store
    texts = {store.word_text(i) for i in store.word_ids().tolist()}
    expected = {w for s in items for w in str(s).split() if len(w) >= 2}
    assert expected and expected.issubset(texts)


# -- the unfold consumes the words --------------------------------------------

def _basis_op_index(reducer):
    """First reducer op whose host layer takes the recommender basis."""
    for i, ad in enumerate(reducer.ops):
        gl = getattr(ad, "gl", None)
        rev = getattr(gl, "reverse", None)
        if rev is not None and "basis" in rev.__code__.co_varnames:
            return i, gl
    return None, None


def _spy_unfold(m):
    """Drive one synthetic recorded fold step through the unfold with a
    reverse() spy; return the captured (basis, left_rows) call."""
    reducer = m._stm_reducer()
    assert reducer is not None, "smoke grammar lost its arity-2 reduce ops"
    idx, gl = _basis_op_index(reducer)
    assert idx is not None, "no basis-threaded reverse among reducer ops"
    D = int(m.conceptualSpace.stm.concept_dim)
    R = len(reducer.ops)
    marg = torch.zeros(1, 1, R); marg[0, 0, idx] = 1.0
    can = torch.tensor([True])
    object.__setattr__(m, "_stm_reduce_op_trace", [(marg, can)])
    captured = []

    def spy(parent, basis=None, left_rows=None, right_rows=None, **kwargs):
        captured.append((basis, left_rows))
        return parent, parent               # any 2-tuple satisfies the walk

    orig = gl.reverse
    gl.reverse = spy
    try:
        S = torch.rand(1, D) * 0.1
        out = m._reverse_reduce_unfold(S)
        assert out is not None and out.shape[:2] == (1, 2)
    finally:
        gl.reverse = orig
        object.__setattr__(m, "_stm_reduce_op_trace", None)
    assert len(captured) == 1
    return captured[0]


def _recognized_rows(m):
    for cs in (list(getattr(m, "conceptualSpaces", []) or [])
               or [m.conceptualSpace]):
        fn = getattr(cs, "recognized_word_rows", None)
        r = fn() if callable(fn) else None
        if r is not None and int(r.numel()) > 0:
            return cs, r
    return None, None


def test_one_to_one_recognition_registers_under_words(tmp_path_factory):
    """Refined 1:1 (Alec): the part-aggregation (ONE promoted percept — the
    trie's created word) must be EQUAL IN SPAN to the WS word-property
    span. Multi-part runs and span mismatches do not register; first
    sighting suffices."""
    m = _build(tmp_path_factory, word_store=True)
    cs = m.conceptualSpace
    store = m.perceptualSpace.percept_store
    # 'zebra' promoted -> ONE pid; 'yak' unpromoted -> byte run; 'quix'
    # promoted but its tile span will NOT match any WS property span;
    # 'zebr' promoted -> a LETTERS-PART that is not the entire word
    # 'zebry' (its own extent, 4, cannot fill the 5-wide property span).
    store.observe_chunk(b"zebra")
    store.observe_chunk(b"zebra")
    zebra_pid = store.spell_out(b"zebra")
    assert len(zebra_pid) == 1
    yak_pids = store.spell_out(b"yak")
    assert len(yak_pids) > 1
    store.observe_chunk(b"quix")
    store.observe_chunk(b"quix")
    quix_pid = store.spell_out(b"quix")
    assert len(quix_pid) == 1
    store.observe_chunk(b"zebr")
    store.observe_chunk(b"zebr")
    zebr_pid = store.spell_out(b"zebr")
    assert len(zebr_pid) == 1 and len(store.bytes_for(zebr_pid[0])) == 4
    N = 3 + len(yak_pids)
    pid_2d = torch.tensor([zebra_pid + yak_pids + quix_pid + zebr_pid],
                          dtype=torch.long)
    groups = torch.tensor([[0] + [1] * len(yak_pids) + [2] + [3]],
                          dtype=torch.long)
    ws = getattr(cs, "terminalSymbolSpace_ref", None) or \
        getattr(cs, "wholeSpace_ref", None)
    assert ws is not None and getattr(ws, "_mereology_raise", False)
    D = int(ws.subspace.what.getW().shape[-1])
    vec = torch.rand(1, N, D) * 0.1
    # PS tile record: zebra (0,5); yak byte slots share the word tile
    # (6,9); quix (10,14); the 'zebr' part rides the word tile (15,20)
    # ('zebry') — the tile record stamps the WORD span on the slot even
    # though the part fills only 4 of its 5 bytes. WS property spans
    # stage (0,5), (6,9), (15,20) — quix's span has no twin.
    tile_spans = [[(0, 5)] + [(6, 9)] * len(yak_pids)
                  + [(10, 14)] + [(15, 20)]]
    saved_spans = getattr(ws, "_staged_analysis_spans", None)
    object.__setattr__(
        ws, "_staged_analysis_spans",
        torch.tensor([[[0, 5], [6, 9], [15, 20]]], dtype=torch.long))
    try:
        cs._maybe_autobind_meta(
            pid_2d, vec, word_groups=groups, tokens=None,
            word_texts=[["zebra", "yak", "quix", "zebry"]],
            tile_spans=tile_spans, percept_store=store)
    finally:
        object.__setattr__(ws, "_staged_analysis_spans", saved_spans)
    reg = getattr(cs, "_recognized_words", {}) or {}
    assert reg.get("zebra") == zebra_pid[0]
    assert "yak" not in reg          # multi-part: the trie has no word yet
    assert "quix" not in reg         # span mismatch: no WS property twin
    assert "zebry" not in reg        # letters-part does not FILL the word
    # The in-model structure: A_zebra rides Parts(WORDS); A_yak does not.
    wom = cs._word_obj_meta
    A_zebra = wom["zebra"][0]
    A_yak = wom["yak"][0]
    words_cid = cs._words_concept_id
    parts = cs.concept_parts(words_cid)
    assert ("sym", A_zebra) in parts
    assert ("sym", A_yak) not in parts
    rows = cs.recognized_word_rows()
    assert rows is not None and zebra_pid[0] in rows.tolist()


def test_unfold_falls_back_when_no_words_yet(tmp_path_factory):
    m = _build(tmp_path_factory, word_store=True)
    _train_forward(m)
    ps = m.perceptualSpace
    saved = ps.percept_store
    _cs_list = (list(getattr(m, "conceptualSpaces", []) or [])
                or [m.conceptualSpace])
    saved_regs = [getattr(cs, "_recognized_words", None) for cs in _cs_list]
    from Layers import RadixLayer
    try:
        # Clear BOTH sources — the in-model recognition registry AND the
        # store's promoted collection: no word rows -> the existing basis
        # pick (first dim-matched codebook: wholeSpace, conceptualSpace).
        for cs in _cs_list:
            object.__setattr__(cs, "_recognized_words", None)
        ps.subspace.percept_store = RadixLayer(
            int(saved.dim), word_bounded=True)
        basis, left_rows = _spy_unfold(m)
        assert left_rows is None
        assert basis is m.wholeSpace.subspace.what
    finally:
        ps.subspace.percept_store = saved
        for cs, r in zip(_cs_list, saved_regs):
            object.__setattr__(cs, "_recognized_words", r)


def test_unfold_ignores_words_when_knob_off(tmp_path_factory):
    m = _build(tmp_path_factory, word_store=False)
    _train_forward(m)
    basis, left_rows = _spy_unfold(m)
    assert left_rows is None                     # no typed restriction
    assert basis is m.wholeSpace.subspace.what   # the existing pick, unchanged


# -- open-fronts Task A: percept store rides checkpoints ----------------------

def test_percept_extras_ride_the_checkpoint_envelope(tmp_path_factory):
    m = _build(tmp_path_factory, word_store=True)
    _train_forward(m)
    _train_forward(m)
    ps = m.perceptualSpace
    store = ps.percept_store
    assert len(store) > 0
    words_before = {store.word_text(i) for i in store.word_ids().tolist()}
    assert words_before
    blob = m._collect_vocab_extras()
    assert blob is not None and "ps_percept_extras" in blob
    # Restore into a FRESH store on the same shared basis (the reload path:
    # _restore_vocab_extras runs pre-state-dict on a fresh construction).
    from Layers import RadixLayer
    saved = store
    try:
        ps.subspace.percept_store = RadixLayer(
            int(saved.dim), word_bounded=True, basis=ps.subspace.what)
        assert len(ps.percept_store) == 0
        m._restore_vocab_extras(blob)
        got = ps.percept_store
        assert len(got) == len(saved)
        assert ({got.word_text(i) for i in got.word_ids().tolist()}
                == words_before)
        for w in words_before:
            assert got.get_id(w.encode()) == saved.get_id(w.encode())
    finally:
        ps.subspace.percept_store = saved


def test_storeless_configs_emit_no_percept_key(tmp_path_factory):
    m = _build(tmp_path_factory, word_store=False)
    # The OFF smoke still has a radix/meronomy store; emptiness is what
    # gates the key. A fresh (unused) model's store is empty only before
    # any forward — emulate by swapping an empty store.
    from Layers import RadixLayer
    ps = m.perceptualSpace
    saved = ps.percept_store
    try:
        ps.subspace.percept_store = RadixLayer(
            int(saved.dim), word_bounded=True, basis=ps.subspace.what)
        blob = m._collect_vocab_extras()
        assert blob is None or "ps_percept_extras" not in blob
    finally:
        ps.subspace.percept_store = saved


# -- open-fronts Task B: word-bearing-fold filtering ---------------------------

def test_slot_kind_stacks_mirror_the_push_discipline():
    from Layers import ShortTermMemory
    stm = ShortTermMemory(batch=2, capacity=4, concept_dim=3)
    stm.ensure_batch(2)
    assert getattr(stm, "_slot_kinds", None) is None    # recording off
    stm.note_push_all("other")                          # no-op while off
    stm.kinds_enable(2, depths=[1, 0], kind="other")
    ks = stm._slot_kinds
    assert ks == [["other"], []]                        # carried content tagged
    stm.note_push_all("other")
    stm.note_push_masked([True, False], "word")
    assert ks[0] == ["word", "other", "other"]
    assert ks[1] == ["other"]
    stm.clear()
    assert ks == [[], []]


def test_unfold_stamps_sequential_where_offsets(tmp_path_factory):
    """The fold order IS the position (Alec): emitted word slots get
    sequential byte offsets stamped into the CS where band; sentinel /
    zero slots stay unwritten."""
    m = _build(tmp_path_factory, word_store=True)
    _train_forward(m)
    _train_forward(m)
    ps = m.perceptualSpace
    store = ps.percept_store
    _sb = getattr(m.wholeSpaces[0], "_standalone_run_bytes", None)
    rows = store.word_ids(standalone_bytes=_sb)
    assert rows.numel() >= 2
    basis = ps.subspace.what
    W = basis.getW()
    sub = m.conceptualSpace.subspace
    D = int(m.conceptualSpace.stm.concept_dim)
    d_what = (D - int(getattr(sub, "nWhere", 0) or 0)
              - int(getattr(sub, "nWhen", 0) or 0))
    r0, r1 = int(rows[0]), int(rows[1])
    out = torch.zeros(1, 3, D)
    out[0, 0, :d_what] = W[r0, :d_what]
    out[0, 1, :d_what] = W[r1, :d_what]          # slot 2 stays a zeros tail
    m._stamp_unfold_where(out, d_what, basis, rows)
    w_enc = sub.whereEncoding
    idx = [int(i) for i in w_enc.resolve(D)]
    len0 = len(store.bytes_for(r0))
    expect = w_enc.encode(torch.tensor([[0.0, float(len0 + 1)]]))
    assert torch.allclose(out[0, 0, idx], expect[0, 0], atol=1e-5)
    assert torch.allclose(out[0, 1, idx], expect[0, 1], atol=1e-5)
    assert float(out[0, 2, idx[0]:].abs().sum()) == 0.0   # tail unwritten


def test_words_summary_row_running_mean(tmp_path_factory):
    """Alec's §3a call: the WORDS codebook face is the order-capped
    SUMMARY ROW — the well-known 'words' atom (WS row 0) carries the
    running mean of member word-whole rows, fold record stamped full."""
    m = _build(tmp_path_factory, word_store=True)
    _train_forward(m)
    ws = m.wholeSpace
    cs = m.conceptualSpace
    ww = getattr(ws, "_word_whole_ss", {}) or {}
    p2r = ws._ws_pos_to_row
    keys = [k for k in ww if ww[k] in p2r][:2]
    assert len(keys) == 2, "need two bound word-wholes on the smoke"
    W = ws.subspace.what.getW()
    row0 = ws.well_known_atoms["words"]
    saved_reg = getattr(cs, "_recognized_words", None)
    saved_wid = getattr(cs, "_words_concept_id", None)
    saved_row0 = W[row0].detach().clone()
    try:
        object.__setattr__(cs, "_recognized_words", {})
        object.__setattr__(cs, "_words_concept_id", None)
        v1 = W[p2r[ww[keys[0]]]].detach().clone()
        v2 = W[p2r[ww[keys[1]]]].detach().clone()
        A1, A2 = cs.new_concept(), cs.new_concept()
        cs._register_recognized_word(A1, "syn_w1", 1, ws=ws,
                                     whole_pos=ww[keys[0]])
        cs.apply_pending_words_summary(ws)   # the stem drain (deferred write)
        assert torch.allclose(W[row0].detach(), v1, atol=1e-5)
        cs._register_recognized_word(A2, "syn_w2", 2, ws=ws,
                                     whole_pos=ww[keys[1]])
        cs.apply_pending_words_summary(ws)
        assert torch.allclose(W[row0].detach(), (v1 + v2) / 2, atol=1e-5)
        cb = ws.subspace.what
        rams = getattr(cb, "ramsification", None)
        if rams is not None:
            assert cb.abstraction_order(row0) == int(rams.shape[1])
    finally:
        with torch.no_grad():
            W.data[row0] = saved_row0
        object.__setattr__(cs, "_recognized_words", saved_reg)
        object.__setattr__(cs, "_words_concept_id", saved_wid)


# -- open-fronts Task C: the WS word-whole rows resolve for the SS driver ----

def test_ws_word_whole_registry_resolves_to_rows(tmp_path_factory):
    m = _build(tmp_path_factory, word_store=True)
    _train_forward(m)
    ws = m.wholeSpace
    reg = getattr(ws, "_word_whole_ss", None)
    p2r = getattr(ws, "_ws_pos_to_row", None)
    assert reg, "the autobind must have bound word-wholes on this config"
    rows = sorted({int(p2r[p]) for p in reg.values() if p in p2r})
    assert rows, "word-whole positions must resolve through _ws_pos_to_row"
    W = ws.subspace.what.getW()
    assert all(0 <= r < int(W.shape[0]) for r in rows)


def test_forward_records_kind_tagged_trace(tmp_path_factory):
    m = _build(tmp_path_factory, word_store=True)
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    m.eval()
    with torch.no_grad():
        m.forward(x)
    stm = m.conceptualSpace.stm
    ks = getattr(stm, "_slot_kinds", None)
    assert ks is not None and len(ks) == len(items), \
        "kind recording must be live on wordStore configs"
    trace = getattr(m, "_stm_reduce_op_trace", None)
    assert trace, "the sweep must have traced folds"
    assert all(len(step) == 4 for step in trace), \
        "trace steps must carry operand kinds while recording"


# -- cross-batch graph severing (LAST in file: builds a DIFFERENT config) ----

# -- the signed-space snap (Alec's design call, 2026-07-14) -------------------

def test_word_pair_snap_recovers_folded_pair():
    """Joint pair snap (Decision B): argmax over word-row PAIRS of
    <radmax(w_i, w_j), parent> recovers the exact folded pair. No
    feasibility filter — the radial order filter admitted no real row
    against trained composites, so EVERY free-derivation operand was the
    all-ones ⊤ sentinel (measured 2026-07-14; the 'structural 0.8711'
    was cos(⊤, nearest word row))."""
    from Layers import Ops

    torch.manual_seed(0)
    W = torch.randn(6, 16) * 0.5
    rows = torch.arange(6)
    parent = Ops._radmax(W[2], W[4]).unsqueeze(0)
    x1, x2 = Ops.word_pair_snap(parent, W, rows)
    got = {tuple(x1.reshape(-1).tolist()), tuple(x2.reshape(-1).tolist())}
    want = {tuple(W[2].tolist()), tuple(W[4].tolist())}
    assert got == want


def test_word_pair_snap_orders_by_side_dot():
    """The pair's x1 is the HIGHER one-sided-dot member (the documented
    deterministic tie-break — review finding: the swap branch had zero
    coverage while emission order feeds the where-stamp downstream)."""
    from Layers import Ops

    D = 12
    w_hi = torch.zeros(D)
    w_hi[:6] = 1.0                       # one-sided dot 6.0
    w_lo = torch.zeros(D)
    w_lo[6:9] = 0.5                      # one-sided dot 0.75
    W = torch.stack([w_lo, w_hi])        # store order must not matter
    parent = Ops._radmax(w_hi, w_lo).unsqueeze(0)
    x1, x2 = Ops.word_pair_snap(parent, W, torch.arange(2))
    assert torch.equal(x1.reshape(-1), w_hi)
    assert torch.equal(x2.reshape(-1), w_lo)


def test_intersection_snap_priming_recovers_present_words():
    """Conjunction's selection differs from disjunction's (Alec 2026-07-14:
    'we may want a different kind of codebook selection for conjunction as
    opposed to disjunction'). The meet is LOSSY — radmin(a, b) collapses to
    ``a`` whenever ``a`` is dominated, so the fold fit alone cannot recover
    the second operand (every candidate agrees on the parent's support).
    Priming (which words are present) selects the DISTINCT present pair —
    the information disjunction's join never loses and so never needs."""
    from Layers import Ops

    a = torch.tensor([1., 1, 0, 0, 0, 0])        # the shared core (== meet)
    b = torch.tensor([1., 1, 1, 0, 0, 0])        # dominates a (extra dim 2)
    c = torch.tensor([1., 1, 0, 1, 0, 0])        # dominates a (extra dim 3)
    W = torch.stack([a, b, c])
    rows = torch.arange(3)
    parent = Ops._radmin(a, b).unsqueeze(0)
    assert torch.equal(parent.reshape(-1), a)    # the meet destroyed b's dim 2
    x1, x2 = Ops.word_pair_snap(parent, W, rows, op_name='intersection',
                                priming=torch.tensor([1., 1., 0.]))
    assert {tuple(x1.reshape(-1).tolist()), tuple(x2.reshape(-1).tolist())} \
        == {tuple(a.tolist()), tuple(b.tolist())}
    # Flip priming to c -> the meet-aware selection follows the present set.
    y1, y2 = Ops.word_pair_snap(parent, W, rows, op_name='intersection',
                                priming=torch.tensor([1., 0., 1.]))
    assert {tuple(y1.reshape(-1).tolist()), tuple(y2.reshape(-1).tolist())} \
        == {tuple(a.tolist()), tuple(c.tolist())}


def test_grammar_layer_reverse_snap_is_op_respecting():
    """The snap lives INSIDE the grammar layers (Alec 2026-07-14): recovery
    is owned by each GrammarLayer's reverse() and dispatched by WHICH layer —
    UnionLayer does the join snap, IntersectionLayer the meet-aware snap — not
    a union-for-all imposed at the un-fold driver."""
    from Language import UnionLayer, IntersectionLayer
    from Layers import Ops

    class _Basis:
        def __init__(self, W):
            self._W = W

        def getW(self):
            return self._W

    a = torch.tensor([1., 1, 0, 0, 0, 0])
    b = torch.tensor([1., 1, 1, 0, 0, 0])
    W = torch.stack([a, b, torch.tensor([1., 1, 0, 1, 0, 0])])
    rows = torch.arange(3)
    # Union: radmax(a, b) recovered from the fit alone.
    up = Ops._radmax(a, b).unsqueeze(0)
    ux1, ux2 = UnionLayer().reverse(up, basis=_Basis(W),
                                    left_rows=rows, right_rows=rows, snap=True)
    assert {tuple(ux1.reshape(-1).tolist()), tuple(ux2.reshape(-1).tolist())} \
        == {tuple(a.tolist()), tuple(b.tolist())}
    # Intersection: radmin(a, b) == a (lossy meet) — priming selects (a, b).
    ip = Ops._radmin(a, b).unsqueeze(0)
    ix1, ix2 = IntersectionLayer().reverse(
        ip, basis=_Basis(W), left_rows=rows, right_rows=rows,
        left_priming=torch.tensor([1., 1., 0.]), snap=True)
    assert {tuple(ix1.reshape(-1).tolist()), tuple(ix2.reshape(-1).tolist())} \
        == {tuple(a.tolist()), tuple(b.tolist())}


def test_union_snap_needs_no_priming():
    """Disjunction's join is well-determined: the folded pair is recovered
    from the fit alone (priming is an accepted no-op refinement, not a
    requirement — the asymmetry with conjunction above)."""
    from Layers import Ops

    torch.manual_seed(0)
    W = torch.randn(6, 16) * 0.5
    rows = torch.arange(6)
    parent = Ops._radmax(W[1], W[4]).unsqueeze(0)
    bare = Ops.word_pair_snap(parent, W, rows)                 # no priming
    primed = Ops.word_pair_snap(parent, W, rows,
                                priming=torch.zeros(6))         # neutral
    want = {tuple(W[1].tolist()), tuple(W[4].tolist())}
    for got in (bare, primed):
        assert {tuple(got[0].reshape(-1).tolist()),
                tuple(got[1].reshape(-1).tolist())} == want


def test_word_side_snap_support_match_beats_l2():
    """Decision A's rationale (Alec): recovering N from ADJ(N) — the
    modifier attenuates dims of the noun, so the true noun is L2-FAR
    from the parent (its unmodified support is missing there) yet
    dot-CLOSE (absent dims contribute nothing). Raw dot must pick the
    noun over an L2-closer low-magnitude decoy."""
    from Layers import Ops

    D = 16
    noun = torch.zeros(D)
    noun[:8] = 1.0                       # broad support (cat over 𝟙 domain)
    parent = torch.zeros(D)
    parent[:4] = 1.0                     # ADJ kept dims 0-3 only
    # L2-closer decoy (||p-decoy|| = 1.0 < ||p-noun|| = 2.0) whose PATTERN
    # over the visible support differs — a uniform rescale of the parent
    # is genuinely indistinguishable from it, so the discriminating decoy
    # must mismatch in shape, not just scale.
    decoy = torch.zeros(D)
    decoy[:4] = torch.tensor([0.9, 0.9, 0.3, 0.3])
    # Big-norm decoy: raw dot picks it (10.0 > 4.0) — pins the support-
    # restricted DENOMINATOR (10/√50 ≈ 1.41 < noun's 2.0), the measured
    # large-row-dominance fix a raw-dot mutant would silently drop.
    big = torch.zeros(D)
    big[:2] = 5.0
    W = torch.stack([decoy, noun, big])
    w_star, _ = Ops.word_side_snap(parent.unsqueeze(0), W, torch.arange(3))
    assert torch.equal(w_star.reshape(-1), noun)


def test_word_side_snap_minimal_residual():
    """word∧other steps: the residual carries the parent EXACTLY on the
    dims the snapped word does not radially dominate and zero elsewhere
    (minimal residual — biases the next backward step toward the
    UNEXPLAINED content), so radmax(word, residual) reconstitutes the
    parent on every non-annihilated dim."""
    from Layers import Ops

    torch.manual_seed(1)
    W = torch.randn(5, 12) * 0.3
    # The word row is a sparse STRONG mask (Alec's model: a word is an
    # adjective pre-applied to the 𝟙 domain) so the fold retains its
    # support against the composite co-operand. Dim 4 PARTIALLY overlaps
    # the composite (0 < |w*| < |parent|) — the case that separates the
    # minimal radial residual from plain subtraction (review finding:
    # disjoint supports made a subtractive mutant byte-identical).
    W[3] = 0.0
    W[3, :4] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    W[3, 4] = 0.1
    rows = torch.arange(5)
    comp = torch.zeros(12)
    comp[4:] = torch.randn(8) * 0.5
    comp[4] = 0.6
    parent = Ops._radmax(W[3], comp).unsqueeze(0)
    w_star, resid = Ops.word_side_snap(parent, W, rows)
    assert torch.equal(w_star.reshape(-1), W[3])
    # THE minimal-residual contract in one line (review finding: the old
    # reconstitution checks passed a parent-echo residual too).
    p = parent.reshape(-1)
    want = torch.where(p.abs() > w_star.reshape(-1).abs(), p,
                       torch.zeros_like(p))
    assert torch.equal(resid.reshape(-1), want)


def test_unfold_word_word_step_snaps_to_store_rows(tmp_path_factory):
    """A 4-tuple word∧word trace step dispatches the joint pair snap: the
    emitted operand and the final carry are EXACT store rows — the
    ⊤-sentinel degeneracy is gone from the word-bearing un-fold."""
    m = _build(tmp_path_factory, word_store=True)
    _train_forward(m)
    _train_forward(m)                                # promote the words
    from Layers import Ops

    store = m.perceptualSpace.percept_store
    # Pin the un-fold's candidate source: clear the (shared cached
    # model's) recognition registry for the duration so the store's
    # promoted collection is the live set in BOTH isolation and
    # full-file orderings; restored below.
    cs_list = (list(getattr(m, "conceptualSpaces", []) or [])
               or [m.conceptualSpace])
    saved_reg = [getattr(c, "_recognized_words", None) for c in cs_list]
    for c in cs_list:
        object.__setattr__(c, "_recognized_words", None)
    wid = store.word_ids()
    assert int(wid.numel()) >= 2
    W = m.perceptualSpace.subspace.what.getW().detach()
    r1, r2 = int(wid[0]), int(wid[1])
    reducer = m._stm_reducer()
    R = len(reducer.ops)
    d_what = int(W.shape[1])
    stm = m.conceptualSpace.stm
    D = int(stm.concept_dim)
    S = torch.zeros(1, D)
    S[0, :d_what] = Ops._radmax(W[r1], W[r2])
    marg = torch.zeros(1, 1, R)
    marg[0, 0, 0] = 1.0
    can = torch.tensor([True])
    lw = torch.tensor([True])
    rw = torch.tensor([True])
    object.__setattr__(m, "_stm_reduce_op_trace", [(marg, can, lw, rw)])
    try:
        out = m._reverse_reduce_unfold(S)
    finally:
        object.__setattr__(m, "_stm_reduce_op_trace", None)
        for c, reg in zip(cs_list, saved_reg):
            object.__setattr__(c, "_recognized_words", reg)
    assert out is not None and int(out.shape[1]) >= 2
    got = {tuple(out[0, i, :d_what].tolist()) for i in range(2)}
    want = {tuple(W[r1].tolist()), tuple(W[r2].tolist())}
    assert got == want


def test_unfold_trace_free_recovers_two_word_root(tmp_path_factory):
    """Trace-free scope (Alec 2026-07-14, step 2): the derivation recovers a
    2-word root exactly from the grammar reverse ops with NO forward record
    and NO recognition registry (the store word_ids are the candidate set),
    emitting only store rows. DEEPER trees (>2 words) need separable composite
    children the collapsed root cannot yet yield — bounded by the STM cap and
    gated on root separability (step 3)."""
    from Layers import Ops

    m = _build(tmp_path_factory, word_store=True)
    _train_forward(m)
    _train_forward(m)
    store = m.perceptualSpace.percept_store
    cs_list = (list(getattr(m, "conceptualSpaces", []) or [])
               or [m.conceptualSpace])
    saved_reg = [getattr(c, "_recognized_words", None) for c in cs_list]
    for c in cs_list:
        object.__setattr__(c, "_recognized_words", None)
    object.__setattr__(m, "_stm_reduce_op_trace", None)   # NO forward record
    try:
        wid = store.word_ids()
        assert int(wid.numel()) >= 2
        W = m.perceptualSpace.subspace.what.getW().detach()
        r1, r2 = int(wid[0]), int(wid[1])
        d_what = int(W.shape[1])
        D = int(m.conceptualSpace.stm.concept_dim)
        S = torch.zeros(1, D)
        S[0, :d_what] = Ops._radmax(W[r1], W[r2])
        out = m._reverse_reduce_unfold(S)
    finally:
        for c, reg in zip(cs_list, saved_reg):
            object.__setattr__(c, "_recognized_words", reg)
    assert out is not None
    emitted = {tuple(out[0, i, :d_what].tolist())
               for i in range(int(out.shape[1]))
               if float(out[0, i, :d_what].abs().sum()) > 0}
    store_rows = {tuple(W[int(r)].tolist()) for r in wid.tolist()}
    assert emitted.issubset(store_rows)
    assert {tuple(W[r1].tolist()), tuple(W[r2].tolist())}.issubset(emitted)


def test_grammar_reverse_ops_from_generate_section(tmp_path_factory):
    """Trace-free reverse (Alec 2026-07-14): the reverse operation set is
    determined from the grammar's <generate> section, symmetric to how the
    forward reducer is built from <compose>. Enumerates the arity-2
    snap-capable reverse ops (union / intersection at minimum)."""
    m = _build(tmp_path_factory, word_store=True)
    ops = m._grammar_reverse_ops()
    names = {n for n, _ in ops}
    # The SS lattice pair (radmax / radmin under radialStmReduce) is what the
    # STM reducer folds with on this config; the reverse set mirrors it.
    assert "disjunction" in names and "conjunction" in names, names
    for _n, host in ops:
        assert int(getattr(host, "arity", 1)) == 2
        assert "left_rows" in host.reverse.__code__.co_varnames


def test_reverse_chooser_picks_fold_op_by_roundtrip(tmp_path_factory):
    """The reverse derivation finds its OWN op — no forward record: it scores
    each grammar reverse op by ROUND-TRIP fit (op.compose(op.reverse(parent))
    vs parent) and picks the best. A radmax(w1, w2) root is explained by
    union, not intersection, and recovers the exact word pair."""
    from Layers import Ops

    m = _build(tmp_path_factory, word_store=True)
    _train_forward(m)
    _train_forward(m)
    cs_list = (list(getattr(m, "conceptualSpaces", []) or [])
               or [m.conceptualSpace])
    for c in cs_list:
        object.__setattr__(c, "_recognized_words", None)
    store = m.perceptualSpace.percept_store
    wid = store.word_ids()
    assert int(wid.numel()) >= 2
    W = m.perceptualSpace.subspace.what.getW().detach()
    r1, r2 = int(wid[0]), int(wid[1])
    basis = m.perceptualSpace.subspace.what
    parent = Ops._radmax(W[r1], W[r2])
    ops = m._grammar_reverse_ops()
    (name, _host), (x1, x2), _fit = m._reverse_choose_op(
        parent, ops, basis, wid.to(W.device))
    assert name == "disjunction"          # radmax fold, discovered by fit
    got = {tuple(x1.reshape(-1)[:W.shape[1]].tolist()),
           tuple(x2.reshape(-1)[:W.shape[1]].tolist())}
    assert got == {tuple(W[r1].tolist()), tuple(W[r2].tolist())}


# -- step 3: Method-1 -> Method-2 distillation (root separability) -----------

def test_leaf_distill_default_off(tmp_path_factory):
    """<training><leafDistillWeight> defaults 0.0 -> no term, no head —
    byte-identical (the house default-off contract)."""
    from Layers import TheError

    m = _build(tmp_path_factory, word_store=True)
    assert float(getattr(m, "leaf_distill_weight", 0.0)) == 0.0
    terms = []
    orig_add = TheError.add

    def spy(name, value, weight=1.0, **kw):
        terms.append(name)
        return orig_add(name, value, weight=weight, **kw)

    TheError.add = spy
    try:
        opt = m.getOptimizer(lr=0.01)
        m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
    finally:
        TheError.add = orig_add
    assert "leaf_distill" not in terms
    assert getattr(m, "_leaf_distill_head_module", None) is None


def test_leaf_distill_trains_root_toward_leaves(tmp_path_factory):
    """Step 3 (Alec: 'the training of method-1 will give information to
    method-2'): with the knob on, a train batch adds the leaf_distill term —
    the leaf-decoder head regenerates the Method-1 EXACT leaves
    (_stm_pre_reduce_slab) from the collapsed root (_stm_single_S), so the
    root cannot satisfy it while collapsing distinct sentences to one
    attractor. The head exists, carries gradients after the step, and is
    handed to the optimizer."""
    from Layers import TheError

    m = _build(tmp_path_factory, word_store=True)
    m.leaf_distill_weight = 0.1
    terms = {}
    orig_add = TheError.add

    def spy(name, value, weight=1.0, **kw):
        if name == "leaf_distill" and torch.is_tensor(value):
            terms[name] = float(value.detach())
        return orig_add(name, value, weight=weight, **kw)

    TheError.add = spy
    try:
        opt = m.getOptimizer(lr=0.01)
        m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
    finally:
        TheError.add = orig_add
        m.leaf_distill_weight = 0.0
    assert terms.get("leaf_distill", 0.0) > 0.0
    head = getattr(m, "_leaf_distill_head_module", None)
    assert head is not None
    opt_params = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) in opt_params for p in head.parameters()), \
        "the lazily-built head must be handed to the live optimizer"


def test_percept_concept_reverse_index_and_row(tmp_path_factory):
    """Step (b) substrate (snap design doc §ontology): wholes and parts
    co-occurring at one `.where`/`.when` form a concept's SUPPORT; the
    concept, once formed, is location-independent. The PARALLEL path forms
    the word/object pair (``create_word_object_meta``); the SERIAL path
    RESOLVES and lights it up — it does not mint. The forward folds the
    OBJECT concept ``B``; Method-2 un-folds into object concepts and
    TRANSLATES them back to word concepts (the exact reverse). Pins:
    percept -> (A, B) reverse tie -> B's order-0 ``similarity_codebook``
    row, and the B -> A translation."""
    m = _build(tmp_path_factory, word_store=True)
    cs = m.conceptualSpace
    A, B, C = cs.create_word_object_meta([3, 5], word_whole=None, key="hello")
    # Each word-part percept resolves to BOTH members of the pair ...
    assert cs.concept_of_percept(3) == A
    assert cs.concept_of_percept(5) == A
    assert cs.object_concept_of_percept(3) == B
    assert cs.concept_of_percept(99) is None          # untied percept
    assert cs.object_concept_of_percept(99) is None
    # ... the fold target is the OBJECT concept's order-0 row ...
    row = cs.concept_codebook_row_of_percept(3)
    assert row is not None and row == cs._csw_concept_row(0, B)
    W = cs.similarity_codebook.getW()
    assert 0 <= int(row) < int(W.shape[0])
    # ... and the decode-side translation recovers the word concept.
    assert cs.word_concept_of_object(B) == A
    assert cs.word_concept_of_object(A) is None       # not an object concept


def test_concept_row_content_lights_up_the_resolved_row(tmp_path_factory):
    """The serial read's resolve+gather (step (b)): a percept tied to a
    word/object pair lights up the OBJECT concept's ``similarity_codebook``
    row (the forward folds object concepts) — unit-norm (the signed
    hypersphere) and EXACTLY the resolved row; an untied percept gives
    mask False and a zero row (the caller keeps the computed idea there).
    Cross-word DISTINCTNESS is structural (distinct concepts -> distinct
    rows) and capacity-gated: the smoke config's order-0 block holds ONE
    row (caps0=1, nVectors=2), so only real configs (caps0 >= the
    word/object inventory) can seat several — the re-ladder phase
    exercises that."""
    import torch.nn.functional as _F

    m = _build(tmp_path_factory, word_store=True)
    cs = m.conceptualSpace
    A1, _, _ = cs.create_word_object_meta([3], word_whole=None, key="hello")
    content, mask = cs.concept_row_content(torch.tensor([3, 99]))
    assert mask.tolist() == [True, False]
    assert torch.allclose(content[0].norm(), torch.tensor(1.0), atol=1e-5)
    assert float(content[1].abs().sum()) == 0.0            # untied -> zero row
    W = cs.similarity_codebook.getW()
    r1 = cs.concept_codebook_row_of_percept(3)
    assert torch.allclose(content[0], _F.normalize(W[r1], dim=-1, eps=1e-8),
                          atol=1e-6)


def test_concept_index_read_sizes_inventory_by_explicit_nvectors(
        tmp_path_factory):
    """The serial arm sizes per-stage CS inventories by the tower's TILE
    count (vocabulary-blind: 4 words collided onto caps0=2 rows and the
    concept read HURT — 2026-07-15 pre-flight). Under
    ``<conceptIndexRead>`` an EXPLICIT ``<ConceptualSpace><nVectors>`` is
    honored as a floor so the inventory seats the word/object concepts
    (~2x distinct words at order 0); without the gate every existing
    config keeps byte-identical shapes."""
    import Models
    import Language
    from util import init_config

    cfg = os.path.join(_PROJECT, "data", "matrix",
                       "MM_20M_grammar_wordstore.xml")
    with open(cfg) as f:
        xml = f.read()
    assert "<radialStmReduce>" in xml and "<nVectors>8</nVectors>" in xml
    xml = xml.replace("<radialStmReduce>",
                      "<conceptIndexRead>true</conceptIndexRead>\n    "
                      "<radialStmReduce>", 1)
    xml = xml.replace("<nVectors>8</nVectors>", "<nVectors>16</nVectors>", 1)
    d = tmp_path_factory.mktemp("cir_sizing")
    p = os.path.join(str(d), "wordstore_cir.xml")
    with open(p, "w") as f:
        f.write(xml)
    init_config(path=p, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BaseModel.from_config(p)
    try:
        assert getattr(m, "concept_index_read", False) is True
        for c in (list(getattr(m, "conceptualSpaces", [])) or []):
            assert int(c.nVectors) >= 16, \
                f"stage CS starved: nVectors={c.nVectors}"
            caps = c._order_caps()
            assert caps and int(caps[0]) >= 8            # seats 4x(A+B)
    finally:
        # The module _CACHE models were built under the smoke config;
        # restore its XMLConfig so later cached-model tests stay valid.
        init_config(path=_SMOKE, defaults_path=_DEFAULTS)
        Language.TheGrammar._configured = False


def test_two_epoch_training_severs_cross_batch_graph():
    """Cross-epoch training pin (the ladder5 relaunch crash, 2026-07-13):
    epoch 1 on the wordstore config broke the pending backward
    ("[1016] at version 1") — the serial arcs' persistent carriers
    (STM ``_live_buffer``, router ``_last_output``/``_last_root_state``,
    ``routing_state.rule_probs``, ``_stm_last_reduce_routing``,
    ``_stm_single_S``, ``_intent_boosts``) are plain attributes, so
    ``_detach_persistent_state`` never severed them and epoch N+1's
    forward chained epoch N's (already-stepped) graph into its loss.
    Two epochs must run; the named carriers must EXIST and be graph-free
    after. Runs LAST in this file (recon_bench re-inits the process
    config/data to the matrix wordstore variant — the smoke-config
    ``_CACHE`` tests above must not run after that repoint)."""
    import random

    import numpy as np
    import recon_bench

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = recon_bench._resolve_config(
        "data/matrix/MM_20M_grammar_wordstore.xml")
    m, dev, lr, batch_size = recon_bench._build_model(cfg)
    opt = m.getOptimizer(lr=lr)
    for _ in range(2):                     # epoch 1 crashed pre-fix
        m.runEpoch(optimizer=opt, batchSize=batch_size, split="train")

    def graph_free(t):
        return not (torch.is_tensor(t) and t.grad_fn is not None)

    # Key carriers must EXIST (a missing carrier would make the
    # graph-free checks vacuous) ...
    stm_buf = m.conceptualSpace.stm._buffer
    rr = getattr(m, "_stm_last_reduce_routing", None)
    router = m.symbolSpace.subspace.languageLayer
    last_out = getattr(router, "_last_output", None)
    assert torch.is_tensor(stm_buf) and stm_buf.numel() > 0
    assert isinstance(rr, dict) and rr, "reduce routing must be stashed"
    assert torch.is_tensor(last_out), "router _last_output must be stashed"
    # ... and carry NO graph across the tick boundary.
    assert graph_free(stm_buf)
    assert graph_free(getattr(m, "_stm_single_S", None))
    assert all(graph_free(v) for v in rr.values()), \
        [k for k, v in rr.items() if not graph_free(v)]
    assert graph_free(getattr(m.perceptualSpace, "_intent_boosts", None))
    assert graph_free(last_out)
    assert graph_free(getattr(router, "_last_root_state", None))
    rs = getattr(m.symbolSpace.subspace, "routing_state", None)
    assert rs is None or graph_free(getattr(rs, "rule_probs", None))
