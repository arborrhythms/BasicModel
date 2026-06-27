"""End-to-end: the ramsified per-order sparse concept transform.

Population (at mint, keyed by ramsified order) builds per-order torch.sparse
weight tables; the forward (gated on symbolicOrder>0 + parallel) replaces the
concept content with the encoder + dictionary decoder. Byte-identical when off.
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


def _cs_active(nS=64, order=3, n_ps=8, n_ws=8):
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
    object.__setattr__(cs, "_n_ps_codes", n_ps)
    object.__setattr__(cs, "_n_ws_codes", n_ws)
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

def test_word_symbol_populates_order0_ps_ws():
    cs = _cs_active(n_ps=8, n_ws=256)               # WORD (=100) is a WS feature
    A, B, C = cs.create_word_object_meta([1, 2], WORD, key="cat")
    # A (word) is order 0: parts are PS codes (block offset 0), whole is the WS
    # code WORD (block offset n_ps=8). _csw rows are local within order 0.
    assert cs._concept_source_order(A) == 0
    a_row = cs._csw_concept_row(0, A)
    srcs = [s for (s, _w) in cs.concept_weights(0, a_row)]
    assert 1 in srcs and 2 in srcs                  # PS parts at offsets 1, 2
    assert (8 + int(WORD)) in srcs                  # WS whole at offset n_ps+WORD


def test_meta_is_higher_order_over_subsymbols():
    cs = _cs_active(n_ps=8, n_ws=8)
    A, B, C = cs.create_word_object_meta([1, 2], WORD, key="cat")
    # C (meta) refs the sub-symbols A (order 0) and B; it is order 1.
    assert cs._concept_source_order(C) == 1
    c_row = cs._csw_concept_row(1, C)
    # its weights live in order 1's table, sourced from the SS_0 block.
    assert len(cs.concept_weights(1, c_row)) >= 1
    offsets, _ = cs.cs_source_layout(1, 8, 8)
    ss0 = offsets[0]
    assert all(s >= ss0 for (s, _w) in cs.concept_weights(1, c_row))


def test_population_inactive_is_noop():
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D, wordDim=_D,
        outputDim=_D, nInput=nP, nPercepts=nP, nConcepts=64, nSymbols=64,
        nWords=64, nOutput=64, nWhere=0, nWhen=0)
    cs = Spaces.ConceptualSpace([nP, _D], [64, _D], [64, _D])   # NOT active
    cs.create_word_object_meta([1, 2], WORD, key="cat")
    assert getattr(cs, "_csw_store", None) is None    # nothing populated


def test_order_block_overflow_is_safe():
    cs = _cs_active(nS=8, order=3, n_ps=8, n_ws=8)     # order 0 cap = 4
    rows = [cs._csw_concept_row(0, c) for c in range(6)]
    assert rows[:4] == [0, 1, 2, 3]                    # fills the block
    assert rows[4] is None and rows[5] is None         # overflow -> None (safe)


# -- the forward glue ---------------------------------------------------------

def test_sparse_concept_forward_noop_when_unpopulated():
    cs = _cs_active()
    folded = torch.randn(2, 64, _D)
    # similarity_codebook rows != folded slots here, and no weights -> fallback
    out = cs._sparse_concept_forward(folded, None, None)
    assert out is folded


def test_align_rows_pads_and_trims():
    cs = _cs_active()
    t = torch.randn(3, 2)
    assert cs._align_rows(t, 5).shape == (5, 2)        # pad
    assert cs._align_rows(t, 2).shape == (2, 2)        # trim
    assert cs._align_rows(t, 0) is t                   # unset -> no-op


# -- driver config ------------------------------------------------------------

def test_sparse_concept_config_builds_and_stamps():
    m = _build("MM_sparse_concept.xml")
    assert m.serial is False and m.symbolicOrder == 1 and m.symbol_tower is True
    css = [cs for cs in m.conceptualSpaces if cs._sparse_active()]
    assert css
    cs = css[-1]
    assert int(getattr(cs, "_n_ps_codes", 0)) > 0
    assert int(getattr(cs, "_n_ws_codes", 0)) > 0


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
    cs = _cs_active(n_ps=8, n_ws=256)
    base = set(id(p) for p in cs.getParameters())
    cs.create_word_object_meta([1, 2], WORD, key="cat")     # populates weights
    after = cs.getParameters()
    csw = [v for v in cs._csw_vals.values() if v is not None]
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
    csw_ptrs = {p.data_ptr() for p in cs._csw_vals.values() if p is not None}
    assert csw_ptrs
    opt = m.getOptimizer(lr=0.01)
    opt_ptrs = {p.data_ptr()
                for g in opt.param_groups for p in g["params"]}
    assert csw_ptrs <= opt_ptrs                            # all weights optimized


def test_csw_weights_update_under_optimizer_step():
    cs = _cs_active(nS=16, order=1, n_ps=8, n_ws=8)
    cs.add_concept_weight(0, 0, 0, weight=0.5)
    cs.add_concept_weight(0, 1, 2, weight=1.0)
    what = torch.randn(16, _D)
    ps = torch.rand(8, 2)
    ws = torch.rand(8, 2)
    vals = cs._csw_vals[0]
    before = vals.detach().clone()
    opt = torch.optim.SGD([vals], lr=1.0)
    content, _ = cs.cs_forward_content(ps, ws, what)
    opt.zero_grad()
    content.sum().backward()
    opt.step()
    assert not torch.allclose(cs._csw_vals[0].detach(), before)   # trained
