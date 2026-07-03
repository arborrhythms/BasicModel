"""End-to-end: the shared untyped square concept store (v3).

Population (at mint) writes untyped edges onto ONE shared AttentionLayer;
the forward (gated on symbolicOrder>0 + parallel) runs the iterated wave
encoder + dictionary decoder. Byte-identical when off.
"""
import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

import Spaces
import Language
from Layers import WORD
from test_basicmodel import _populate_test_config

_DATA = os.path.join(os.path.dirname(_BIN), "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")
_D = 8


def _cs_active(nS=64, order=3):
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    cs = Spaces.ConceptualSpace([nP, _D], [nS, _D], [nS, _D])
    object.__setattr__(cs, "_symbolic_order", order)
    object.__setattr__(cs, "_serial", False)
    return cs


def _build(name):
    import Models
    from util import init_config
    p = os.path.join(_DATA, name)
    init_config(path=p, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m


# -- population at mint, keyed by ramsified order -----------------------------

def test_word_symbol_reserves_order0_snap_row():
    cs = _cs_active()
    A, B, C = cs.create_word_object_meta([1, 2], WORD, key="cat")
    # A (word) is order 0: it RESERVES its order-0 codebook row (the snap
    # reads it); its PS parts / WS whole live in the reference store, NOT as
    # weight columns (P2 symbolic-only rework).
    assert cs._concept_source_order(A) == 0
    a_row = cs._csw_concept_row(0, A)
    assert a_row is not None
    assert cs.concept_weights(a_row) == []          # no edges at order 0
    assert set(cs.concept_parts(A)) == {1, 2}       # the reference store
    assert cs.concept_wholes(A) == [int(WORD)]


def test_meta_is_ordered_pair_over_subsymbols():
    cs = _cs_active()
    A, B, C = cs.create_word_object_meta([1, 2], WORD, key="cat")
    # C (meta) is the sec-4c ordered pair [whole=A, part=B]: order 1, ONE
    # untyped edge per sym constituent (v3: direction lives in the records).
    assert cs._concept_source_order(C) == 1
    c_row = cs._csw_concept_row(1, C)
    got = dict(cs.concept_weights(c_row))
    assert cs._csw_concept_row(0, A) in got
    assert cs._csw_concept_row(0, B) in got
    assert len(got) == 2                            # no bias: pair only


def test_chain_link_edge_to_rest_link_is_populated():
    """v3 pin for the FIXED defect: v2's ``so < order`` stratification
    silently dropped the head link's edge to the REST link whenever the
    ramsified cap clamped both to the SAME order -- the vine's recursion
    edge. The untyped square store keeps every sym-constituent edge."""
    cs = _cs_active(order=1)                # cap clamps every link to order 1
    A1, _, _ = cs.create_word_object_meta([1], 2, key="w1")
    A2, _, _ = cs.create_word_object_meta([3], 4, key="w2")
    A3, _, _ = cs.create_word_object_meta([5], 6, key="w3")
    head = cs.create_joint_concept([A1, A2, A3], key=("w1", "w2", "w3"))
    alloc = Spaces._concept_alloc_of(cs)
    ly = alloc.layer(0)
    head_row = ly.row_of(("pool", head))
    rest = [x for (r, x) in alloc.records(head)
            if r == "part" and isinstance(x, tuple) and x[0] == "sym"][0]
    rest_row = ly.row_of(("pool", int(rest[1])))
    assert head_row is not None and rest_row is not None
    # the chain-link edge lives, via the public read-out
    assert rest_row in dict(cs.concept_weights(head_row))


def test_population_inactive_is_noop():
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D, wordDim=_D,
        outputDim=_D, nInput=nP, nPercepts=nP, nConcepts=64, nSymbols=64,
        nWords=64, nOutput=64, nWhere=0, nWhen=0)
    cs = Spaces.ConceptualSpace([nP, _D], [64, _D], [64, _D])   # NOT active
    cs.create_word_object_meta([1, 2], WORD, key="cat")
    assert getattr(cs, "_sparse_fam", None) is None    # nothing populated


def test_order_block_overflow_is_safe():
    cs = _cs_active(nS=8, order=3)                     # order 0 cap = 4
    rows = [cs._csw_concept_row(0, c) for c in range(6)]
    assert rows[:4] == [0, 1, 2, 3]                    # fills the block
    assert rows[4] is None and rows[5] is None         # overflow -> None (safe)


# -- the forward glue ---------------------------------------------------------

def test_symbolic_phase_snap_runs_even_unpopulated():
    """P3: the order-0 snap is ALWAYS defined against the codebook -- a_0
    flows (and yields activations) even before any edges exist; the higher
    orders read as zero. The returned content NEVER substitutes the carrier
    (decision 10) -- it feeds the losses/SS leg only."""
    cs = _cs_active()
    cs.eval()                                          # no EMA write here
    D_dict = int(cs.similarity_codebook.getW().shape[-1])
    settled = torch.randn(2, 64, D_dict)
    content, acts = cs.cs_symbolic_phase(settled)
    assert acts is not None and acts.shape == (64, 2)
    start0, end0 = cs.order_slice(0)
    assert torch.any(acts[start0:end0] != 0)           # a_0: live snap
    assert torch.all(acts[end0:] == 0)                 # higher orders: empty
    assert content.shape == (2, 64, D_dict)            # the symbolic slab


def test_symbolic_phase_inactive_is_noop():
    cs = _cs_active(order=0)                            # symbolicOrder=0
    settled = torch.randn(2, 64, _D)
    out, acts = cs.cs_symbolic_phase(settled)
    assert out is settled and acts is None


# -- driver config ------------------------------------------------------------

def test_sparse_concept_config_builds_and_stamps():
    m = _build("MM_sparse_concept.xml")
    # symbolicOrder=3: the wave iteration budget K (task 8.3, K=1 leaves deep links dark)
    assert m.serial is False and m.symbolicOrder == 3 and m.symbol_tower is True
    css = [cs for cs in m.conceptualSpaces if cs._sparse_active()]
    assert css
    cs = css[-1]
    # P2/P3: the _n_ps_codes/_n_ws_codes source-layout stamps retired with
    # the percept families, _sparse_replace with the two-phase cutover; the
    # sparse gates ride _symbolic_order/_serial alone.
    assert not hasattr(cs, "_n_ps_codes") and not hasattr(cs, "_n_ws_codes")
    assert not hasattr(cs, "_sparse_replace")


def test_sparse_concept_forward_smoke():
    import Models
    from util import TheXMLConfig
    m = _build("MM_sparse_concept.xml")
    Models.TheData.load(TheXMLConfig.get("data.dataset", default="xor"))
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    m.train()
    out = m.forward(x)                                 # runs the parallel loop
    assert out is not None


# -- follow-up 2: sparse weights are trainable (in the optimizer) -------------

def test_getparameters_includes_csw_after_population():
    cs = _cs_active()
    base = set(id(p) for p in cs.getParameters())
    cs.create_word_object_meta([1, 2], WORD, key="cat")     # populates weights
    after = cs.getParameters()
    csw = [ly.values for fam in cs._sparse_fam.values() for ly in fam
           if ly is not None and ly.values is not None]
    assert csw                                              # weights exist
    after_ids = set(id(p) for p in after)
    assert all(id(p) in after_ids for p in csw)             # all included
    assert len(after) >= len(base)


def test_getparameters_byte_identical_when_inactive():
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D, wordDim=_D,
        outputDim=_D, nInput=nP, nPercepts=nP, nConcepts=64, nSymbols=64,
        nWords=64, nOutput=64, nWhere=0, nWhen=0)
    cs = Spaces.ConceptualSpace([nP, _D], [64, _D], [64, _D])   # NOT active
    cs.create_word_object_meta([1, 2], WORD, key="cat")
    assert [id(p) for p in cs.getParameters()] == [id(p) for p in cs.params]


def test_model_optimizer_picks_up_csw_weights():
    m = _build("MM_sparse_concept.xml")
    cs = [c for c in m.conceptualSpaces if c._sparse_active()][-1]
    # populate per-order weights on the model's own CS, then ask for the optimizer
    cs.create_word_object_meta([1, 2], WORD, key="cat")
    cs.create_word_object_meta([3, 4], WORD, key="dog")
    csw_ptrs = {ly.values.data_ptr() for fam in cs._sparse_fam.values()
                for ly in fam if ly is not None and ly.values is not None}
    assert csw_ptrs
    opt = m.getOptimizer(lr=0.01)
    opt_ptrs = {p.data_ptr()
                for g in opt.param_groups for p in g["params"]}
    assert csw_ptrs <= opt_ptrs                            # all weights optimized


def test_conceptual_sbow_situates_live_sparse_codes():
    """Phase-3 completion (plan C1): under a sparse-active config the parked
    t=0 slab is GRAD-BEARING and conceptual_sbow_loss situates the composed
    codes themselves -- the substitutability gradient reaches the sparse
    family values (the legacy path parked a detached slab and could only
    rotate the codebook rows)."""
    import Models
    from util import TheXMLConfig
    m = _build("MM_sparse_concept.xml")
    # Stage 0: the CS whose settled slab is parked at the cutover (and where
    # production's autobind populates -- _commit_autobind_from_stash is
    # stage-0 only).
    cs = [c for c in m.conceptualSpaces if c._sparse_active()][0]
    # Mint a BIAS-BOUNDED joint alongside the meta: the snap's rectified
    # readout can hard-zero a constituent row at a random init (a_0 = 0 ->
    # d a_1 / d value = tanh' * a_0 = 0, a grad-dead draw -- the documented
    # init-blindness), but the joint's EVERYTHING bias edge reads the
    # CONSTANT 1, so its value's gradient is alive at ANY init: the
    # mechanism check is deterministic.
    A1, _B1, _C1 = cs.create_word_object_meta([1], 2, key="w1")
    A2, _B2, _C2 = cs.create_word_object_meta([3], 4, key="w2")
    cs.create_joint_concept([A1, A2], key=("w1", "w2"))
    Models.TheData.load(TheXMLConfig.get("data.dataset", default="xor"))
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    m.train()
    m.forward(x)                                          # parks the slab
    slab = getattr(m, "_cs_parallel_slab", None)
    assert slab is not None
    assert slab.requires_grad                             # LIVE, not detached
    loss = m.conceptual_sbow_loss()
    assert loss is not None and loss.requires_grad
    loss.backward()
    got = any(ly.values.grad is not None and ly.values.grad.abs().sum() > 0
              for fam in cs._sparse_fam.values() for ly in fam
              if ly is not None and ly.values is not None)
    assert got, "SBOW gradient must reach the sparse family values"


def test_csw_weights_update_under_optimizer_step():
    cs = _cs_active(nS=16, order=1)                    # two-block [8 | 8]
    cs.add_concept_edge(8, 0, weight=0.5)
    cs.add_concept_edge(9, 2, weight=1.0)
    what = torch.randn(16, _D)
    a_0 = torch.rand(8, 2)
    vals = cs._sparse_families(1)[1].values
    before = vals.detach().clone()
    opt = torch.optim.SGD([vals], lr=1.0)
    content, _ = cs.cs_forward_content(a_0, what)
    opt.zero_grad()
    content.sum().backward()
    opt.step()
    assert not torch.allclose(vals.detach(), before)              # trained


def test_demux_feedback_is_views_of_the_mixed_carrier():
    """P3 decision 7 (EXECUTION NOTES 4): the pump's C->P / C->S handoffs
    carry the per-tower WINDOWS of the MIXED carrier (combine.views), NEVER
    combine.reverse -- the exact inverse returns each tower its OWN input
    (zero information transfer; the root-caused sO=1 frozen-loss defect)."""
    import Models
    from util import TheXMLConfig
    m = _build("MM_sparse_concept.xml")
    Models.TheData.load(TheXMLConfig.get("data.dataset", default="xor"))
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    m.eval()
    with torch.no_grad():
        m.forward(x)
    cs = m.body_stages[-1]["cs"]
    assert cs._sparse_active()
    carrier = getattr(cs.subspace, "_bind_carrier", None)
    assert carrier is not None
    views = cs.combine.views(carrier)
    part_fb = cs._subspaceForPS.materialize()
    assert torch.allclose(part_fb, views[0], atol=1e-6)     # part window
    ws_fb = getattr(cs, "_subspaceForWSDemux", None)
    assert ws_fb is not None and cs._subspaceForWS is ws_fb
    assert torch.allclose(ws_fb.materialize(), views[1], atol=1e-6)
    # ...and NOT the exact-inverse legs (the killed mutation): the windows
    # of the mix differ from the recovered pre-mix inputs.
    legs = cs.combine.reverse(carrier)
    assert not torch.allclose(views[0], legs[0], atol=1e-4)


def test_two_phase_forward_cutover_stamps_terminal_activations():
    """P3 e2e: under the sparse driver config the pump runs 2-stream and the
    POST-PUMP cutover stamps the terminal CS with the settled activations
    (the SS leg + losses read them); the snap's EMA identity trace moves the
    order-0 codebook rows while training."""
    import Models
    from util import TheXMLConfig
    m = _build("MM_sparse_concept.xml")
    Models.TheData.load(TheXMLConfig.get("data.dataset", default="xor"))
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    cs0 = m.body_stages[0]["cs"]
    assert cs0._sparse_active()
    W = cs0.similarity_codebook.getW()
    start0, end0 = cs0.order_slice(0)
    before = W.detach().clone()
    m.train()
    m.forward(x)
    last_cs = getattr(m, "_combine_last_cs_sub", None)
    assert last_cs is not None
    acts = getattr(last_cs, "_concept_activations", None)
    assert acts is not None and int(acts.shape[0]) == int(cs0.nVectors)
    # EMA identity trace: order-0 rows moved; higher-order rows untouched.
    after = W.detach().clone()
    assert not torch.equal(after[start0:end0], before[start0:end0])
    assert torch.equal(after[end0:], before[end0:])
    # The rectified snap can read EXACTLY zero at a random init (every
    # slot-mean projection clamped -- the documented init-blindness); the
    # EMA trace is precisely the mechanism that makes it discriminative:
    # after a few traced forwards the winning rows align with the field and
    # the snap symbols come alive.
    for _ in range(3):
        m.forward(x)
    last_cs = getattr(m, "_combine_last_cs_sub", None)
    acts = getattr(last_cs, "_concept_activations", None)
    assert torch.any(acts[start0:end0] != 0)             # live snap symbols
