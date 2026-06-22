"""Global attention (B): free, content/relation-driven attention over a TYPED
addressable space (doc/specs/reading-attention.md "(B) Global attention").

Two pieces land together and are exercised here:

  * the **stochastic element** -- the two-pass superposition temperature on the
    attention selection (``ReadingAttention.superposition_scale``): t=0 is
    sharp/exploit + byte-identical, a higher temperature flattens for
    exploration. Reading attention honours it too (covered in
    test_reading_attention.py); here we check it on global attention;

  * the **typed ``.where``** ranging over {input window, STM, LTM, symbol/whole
    codebook} -- one distribution competing across spaces, a typed ``.where`` +
    a soft-read of the addressed content. Gated ``<globalAttention>``, dark by
    default (the read is parked, not fed back -> byte-identical).
"""
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import pytest
import torch

_DATA = os.path.join(os.path.dirname(_BIN), "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _entropy(a):
    a = a.detach()
    return float(-(a * torch.log(a.clamp_min(1e-12))).sum(-1).mean())


def _spaces(B=2, D=16, big_codebook=False):
    from Spaces import GlobalAttention as GA
    V = 65536 if big_codebook else 11
    return [
        {"id": GA.SPACE_INPUT, "keys": torch.randn(B, 5, D)},      # per-batch
        {"id": GA.SPACE_STM, "keys": torch.randn(B, 3, D)},        # per-batch
        {"id": GA.SPACE_LTM, "keys": torch.randn(7, D)},           # shared
        {"id": GA.SPACE_CODEBOOK, "keys": torch.randn(V, D),       # shared
         "boosts": torch.ones(V)},
    ]


# ---------------------------------------------------------------------------
# (a) the GlobalAttention module in isolation
# ---------------------------------------------------------------------------

def test_ranges_over_all_addressable_spaces():
    from Spaces import GlobalAttention as GA
    ga = GA()
    sp = _spaces(B=2, D=16)
    out = ga(concept_q=torch.randn(2, 16), symbol_q=torch.randn(2, 16),
             spaces=sp, temperature=0.0)
    assert out is not None
    assert out["alpha"].shape[1] == 5 + 3 + 7 + 11, "Mtot = sum of space sizes"
    assert sorted(set(out["space_of"].tolist())) == [
        GA.SPACE_INPUT, GA.SPACE_STM, GA.SPACE_LTM, GA.SPACE_CODEBOOK]


def test_typed_where_and_soft_read_shapes():
    from Spaces import GlobalAttention as GA
    ga = GA()
    sp = _spaces(B=2, D=16)
    out = ga(concept_q=torch.randn(2, 16), symbol_q=torch.randn(2, 16),
             spaces=sp, temperature=0.0)
    assert tuple(out["where"].shape) == (2, 2)
    assert float(out["where"].min()) >= 0.0 and float(out["where"].max()) <= 1.0
    assert tuple(out["content"].shape) == (2, 16)             # Dc = min width
    assert out["space_id"].shape == (2,)
    assert bool(((out["space_id"] >= 0) & (out["space_id"] < GA._N_SPACES)).all())


def test_large_shared_codebook_is_memory_safe():
    # A 65536-row shared codebook must NOT be broadcast to [B, V, D] (it is
    # matmul'd). If it were materialized this would blow memory; reaching the
    # asserts means the shared path held.
    from Spaces import GlobalAttention as GA
    ga = GA()
    sp = _spaces(B=2, D=16, big_codebook=True)
    out = ga(concept_q=torch.randn(2, 16), symbol_q=torch.randn(2, 16),
             spaces=sp, temperature=0.0)
    assert out["alpha"].shape[1] == 5 + 3 + 7 + 65536
    assert torch.isfinite(out["content"]).all()


def test_gradient_stops_at_keys():
    # A downstream loss on the soft-read content trains the scorer ONLY; the
    # codebook/LTM/input keys and the query receive no gradient (the EMA/
    # persistent stores stay frozen; orders.md §6 "Learning").
    from Spaces import GlobalAttention as GA
    ga = GA()
    B, D = 2, 16
    inp = torch.randn(B, 5, D, requires_grad=True)
    cb = torch.randn(11, D, requires_grad=True)
    cq = torch.randn(B, D, requires_grad=True)
    sp = [{"id": GA.SPACE_INPUT, "keys": inp},
          {"id": GA.SPACE_CODEBOOK, "keys": cb, "boosts": torch.ones(11)}]
    out = ga(concept_q=cq, symbol_q=None, spaces=sp, temperature=0.0)
    out["content"].pow(2).sum().backward()
    assert inp.grad is None and cb.grad is None and cq.grad is None
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in ga.parameters())


def test_temperature_flattens_peaked_distribution():
    # The stochastic element: with a peaked prior (space_bias), a higher
    # temperature raises entropy and bleeds mass off the preferred space.
    from Spaces import GlobalAttention as GA
    ga = GA()
    ga.space_bias.data = torch.tensor([3.0, -3.0, -3.0, -3.0])  # prefer INPUT
    sp = _spaces(B=2, D=16)
    cq, sq = torch.randn(2, 16), torch.randn(2, 16)
    e0 = _entropy(ga(concept_q=cq, symbol_q=sq, spaces=sp, temperature=0.0)["alpha"])
    e9 = _entropy(ga(concept_q=cq, symbol_q=sq, spaces=sp, temperature=0.95)["alpha"])
    assert e9 > e0 + 0.1, "higher temperature must flatten the distribution"


def test_temperature_zero_is_sharpest():
    from Spaces import GlobalAttention as GA
    ga = GA()
    ga.space_bias.data = torch.tensor([3.0, -3.0, -3.0, -3.0])
    sp = _spaces(B=2, D=16)
    cq, sq = torch.randn(2, 16), torch.randn(2, 16)
    a0 = ga(concept_q=cq, symbol_q=sq, spaces=sp, temperature=0.0)["alpha"]
    a5 = ga(concept_q=cq, symbol_q=sq, spaces=sp, temperature=0.5)["alpha"]
    assert float(a0.detach().max()) >= float(a5.detach().max()), (
        "t=0 is the sharpest (exploit)")


def test_empty_spaces_returns_none():
    from Spaces import GlobalAttention as GA
    ga = GA()
    assert ga(concept_q=torch.randn(2, 8), symbol_q=None, spaces=[],
              temperature=0.0) is None
    # a space with zero candidates is skipped
    assert ga(concept_q=torch.randn(2, 8), symbol_q=None,
              spaces=[{"id": 0, "keys": torch.randn(2, 0, 8)}],
              temperature=0.0) is None


# ---------------------------------------------------------------------------
# (b) the model wiring (MM_global.xml)
# ---------------------------------------------------------------------------

def _build(name):
    import Models, Language
    from util import init_config
    p = os.path.join(_DATA, name)
    init_config(path=p, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m


def _batch(m):
    import Models
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    return m.inputSpace.prepInput(items)


def test_global_on_builds_module():
    from Spaces import GlobalAttention
    m = _build("MM_global.xml")
    assert m.global_attention_enabled
    assert isinstance(m.global_attention, GlobalAttention)


def test_global_off_has_no_module():
    m = _build("MM_reading.xml")            # readingAttention only
    assert not getattr(m, "global_attention_enabled", False)
    assert getattr(m, "global_attention", None) is None


def test_forward_parks_typed_obs_over_spaces():
    from Spaces import GlobalAttention as GA
    m = _build("MM_global.xml")
    x = _batch(m)
    m.eval()
    with torch.no_grad():
        out1 = m.forward(x)[2]
        out2 = m.forward(x)[2]
    assert torch.isfinite(out1).all()
    assert torch.equal(out1, out2), "dark global attention stays deterministic"
    obs = getattr(m, "_global_attention_obs", None)
    assert obs is not None
    # input window + STM + codebook are present (LTM only under ltmConsolidation)
    seen = set(obs["space_of"].tolist())
    assert GA.SPACE_INPUT in seen and GA.SPACE_CODEBOOK in seen
    assert tuple(obs["where"].shape)[1] == 2
    assert torch.isfinite(obs["content"]).all()


def test_addressable_spaces_gathers_input_stm_codebook():
    from Spaces import GlobalAttention as GA
    m = _build("MM_global.xml")
    x = _batch(m)
    m.train()
    with torch.no_grad():
        m.forward(x)
    in_sub = m._lex_embed_stem(x)
    ps = m.perceptualSpace.forward(in_sub)
    prev = m.conceptualSpaces[0]._subspaceForWS
    spaces, _ = m._addressable_spaces(prev, ps)
    ids = {s["id"] for s in spaces}
    assert GA.SPACE_INPUT in ids
    assert GA.SPACE_CODEBOOK in ids


def test_ltm_space_appears_when_store_present():
    # The LTM address space is gathered from symbolSpace.ltm_store; stage a
    # synthetic TernaryTruthStore so the path is exercised without
    # <ltmConsolidation>.
    from Spaces import GlobalAttention as GA
    from Layers import TernaryTruthStore
    m = _build("MM_global.xml")
    if getattr(m, "symbolSpace", None) is None:
        pytest.skip("no symbolSpace on this config")
    x = _batch(m)
    m.train()
    with torch.no_grad():
        m.forward(x)
    in_sub = m._lex_embed_stem(x)
    ps = m.perceptualSpace.forward(in_sub)
    D = int(ps.materialize().shape[-1])
    store = TernaryTruthStore(D, capacity=8)
    store.slots[:3] = torch.randn(3, 3, D)
    store.count = torch.tensor(3)
    object.__setattr__(m.symbolSpace, "ltm_store", store)
    prev = m.conceptualSpaces[0]._subspaceForWS
    spaces, _ = m._addressable_spaces(prev, ps)
    ltm = [s for s in spaces if s["id"] == GA.SPACE_LTM]
    assert ltm, "LTM space must be gathered when ltm_store has rows"
    assert ltm[0]["keys"].shape[0] == 3 and ltm[0]["keys"].dim() == 2


def test_global_params_reach_optimizer():
    m = _build("MM_global.xml")
    opt = m.getOptimizer(lr=0.01)
    gp = {p.data_ptr() for p in m.global_attention.parameters()}
    op = set()
    groups = list(getattr(opt, "param_groups", []) or [])
    for o in getattr(opt, "optimizers", []) or []:
        groups.extend(o.param_groups)
    for g in groups:
        for p in g["params"]:
            op.add(p.data_ptr())
    assert gp and gp.issubset(op)


def test_superposition_temperature_threads_to_global():
    # The model-level superposition temperature must REACH global attention's
    # selection. (At init the full-model distribution is dominated by the ~65k
    # codebook candidates and sits near the entropy ceiling, so the robust
    # invariant is: the explore temperature changes the distribution and does
    # not sharpen it -- entropy non-decreasing.)
    m = _build("MM_global.xml")
    # Non-zero preference so the temperature has something to scale (an
    # untrained scorer emits ~0 logits -> uniform regardless of temperature).
    m.global_attention.space_bias.data = torch.tensor([3.0, -3.0, -3.0, -3.0])
    x = _batch(m)
    m.eval()
    with torch.no_grad():
        m.forward(x)
    in_sub = m._lex_embed_stem(x)
    ps = m.perceptualSpace.forward(in_sub)
    prev = m.conceptualSpaces[0]._subspaceForWS
    object.__setattr__(m, "_superposition_temperature", None)
    m._global_attention_step(prev, ps)
    a0 = m._global_attention_obs["alpha"].clone()
    object.__setattr__(m, "_superposition_temperature", 0.95)
    m._global_attention_step(prev, ps)
    a9 = m._global_attention_obs["alpha"].clone()
    object.__setattr__(m, "_superposition_temperature", None)
    # L1 mass difference (robust to the ~65k-codebook near-uniform regime where
    # element-wise allclose is fragile under different RNG): the two
    # temperatures must produce measurably different distributions.
    assert float((a0 - a9).abs().sum()) > 1e-4, (
        "the explore temperature must reach + change global attention")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
