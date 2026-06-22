"""The global-attention CONSUMER (doc/specs/reading-attention.md "(B)"):
feed the parked soft-read back into the head (the "answer") so the output loss
trains the retrieval. Gated ``<globalAttentionConsume>`` (requires
``<globalAttention>``); dark by default (the soft-read stays parked).

The LTM address space is the parsed TruthSet (``ltm_store``), so "reading over
the TruthSet stored in LTM" is the SPACE_LTM read fed back here. The full QA
training (a question/answer dataset as the supervised target) is the data-wiring
follow-on; this slice lands the mechanism + the gradient path.
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


# ---------------------------------------------------------------------------
# (a) GlobalAttention.consume in isolation
# ---------------------------------------------------------------------------

def test_consume_zero_init_is_noop():
    from Spaces import GlobalAttention
    ga = GlobalAttention()                       # consume_gate zero-init
    symbols = torch.randn(2, 5, 8)
    content = torch.randn(2, 8)
    out = ga.consume(symbols, content)
    assert torch.equal(out, symbols), "zero-init gate must be a no-op residual"


def test_consume_gate_injects_on_leading_width():
    from Spaces import GlobalAttention
    ga = GlobalAttention()
    with torch.no_grad():
        ga.consume_gate.fill_(0.5)
    symbols = torch.zeros(2, 3, 8)
    content = torch.ones(2, 8)
    out = ga.consume(symbols, content)
    assert torch.allclose(out, torch.full_like(out, 0.5)), (
        "gate>0 must add gate*content on the common leading width")


def test_consume_handles_2d_and_3d_symbols():
    from Spaces import GlobalAttention
    ga = GlobalAttention()
    with torch.no_grad():
        ga.consume_gate.fill_(1.0)
    c = torch.ones(2, 6)
    out3 = ga.consume(torch.zeros(2, 4, 6), c)   # [B, N, D]
    out2 = ga.consume(torch.zeros(2, 6), c)      # [B, D]
    assert out3.shape == (2, 4, 6) and out2.shape == (2, 6)
    assert float(out3.detach().mean()) == 1.0 and float(out2.detach().mean()) == 1.0


def test_consume_none_content_is_noop():
    from Spaces import GlobalAttention
    ga = GlobalAttention()
    s = torch.randn(2, 8)
    assert torch.equal(ga.consume(s, None), s)


def test_consume_width_mismatch_slices_to_common():
    from Spaces import GlobalAttention
    ga = GlobalAttention()
    with torch.no_grad():
        ga.consume_gate.fill_(1.0)
    symbols = torch.zeros(2, 10)                 # wider than content
    content = torch.ones(2, 4)                   # narrower
    out = ga.consume(symbols, content)
    assert float(out[:, :4].mean()) == 1.0 and float(out[:, 4:].abs().max()) == 0.0


# ---------------------------------------------------------------------------
# (b) the model wiring
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


def test_consume_flag_off_by_default_on_global_config():
    # MM_global has <globalAttention> but NOT the consumer -> parked (dark).
    m = _build("MM_global.xml")
    assert m.global_attention is not None
    assert not getattr(m, "global_attention_consume", False)


def test_qa_config_builds_with_consumer_and_ltm():
    m = _build("MM_qa.xml")
    assert m.global_attention_consume and m.global_attention is not None
    assert m.ltm_consolidation
    # the LTM store (the "book") exists for the SPACE_LTM read to range over
    assert getattr(m.symbolSpace, "ltm_store", None) is not None
    x = _batch(m)
    m.eval()
    with torch.no_grad():
        out = m.forward(x)[2]
    assert torch.isfinite(out).all()


def test_zero_gate_matches_consume_off():
    # With the consumer on but the gate at its zero init, the head output equals
    # the consume-off forward (the residual is a no-op) -> the flag is dark until
    # the gate trains.
    m = _build("MM_qa.xml")
    x = _batch(m)
    m.eval()
    with torch.no_grad():
        out_off_flag = (lambda: (setattr(m, "global_attention_consume", False),
                                 m.forward(x)[2].clone())[1])()
        setattr(m, "global_attention_consume", True)
        out_on_gate0 = m.forward(x)[2].clone()
    assert torch.equal(out_off_flag, out_on_gate0), (
        "consume on with zero gate must equal consume off")


def test_gate_feeds_the_read_into_the_answer():
    m = _build("MM_qa.xml")
    x = _batch(m)
    m.eval()
    with torch.no_grad():
        setattr(m, "global_attention_consume", False)
        base = m.forward(x)[2].clone()
        setattr(m, "global_attention_consume", True)
        m.global_attention.consume_gate.fill_(0.5)
        fed = m.forward(x)[2].clone()
    assert not torch.equal(base, fed), "a non-zero gate must feed the read back"


def test_nonzero_gate_does_not_corrupt_the_reverse():
    # The consume swaps body_sub's event for the head ONLY: body_sub is also
    # ``_combine_last_cs_sub`` (what the reverse / reconstruction reads), so it
    # MUST be restored. With a NON-ZERO gate the head output changes but the
    # carrier the reverse reads stays byte-identical to consume-off. (Regression
    # for the dead-restore bug: the event lives on body_sub.event.W, not a
    # ``_event`` attr, so a getattr snapshot would never restore it.)
    m = _build("MM_global.xml")
    x = _batch(m)
    m.eval()
    with torch.no_grad():
        setattr(m, "global_attention_consume", False)
        head_off = m.forward(x)[2].clone()
        carrier_off = m._combine_last_cs_sub.materialize().clone()
        setattr(m, "global_attention_consume", True)
        m.global_attention.consume_gate.fill_(0.5)
        head_on = m.forward(x)[2].clone()
        carrier_on = m._combine_last_cs_sub.materialize().clone()
    assert not torch.equal(head_off, head_on), "the gated read must reach the head"
    assert torch.equal(carrier_off, carrier_on), (
        "the reverse carrier must be restored (reconstruction unaffected)")


def test_answer_loss_trains_retrieval():
    # The output loss backprops through the read into the scorer + the consume
    # gate -- retrieval that helps the answer is rewarded.
    m = _build("MM_qa.xml")
    x = _batch(m)
    m.global_attention.consume_gate.data.fill_(0.3)   # active so grad flows
    m.train()
    m.zero_grad(set_to_none=True)
    out = m.forward(x)
    out[2].pow(2).sum().backward()
    ga = m.global_attention
    assert ga.consume_gate.grad is not None and float(ga.consume_gate.grad.abs().sum()) > 0
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in ga.scorer.parameters()), "the answer loss must train the scorer"


def test_ltm_truthset_read_is_fed_to_the_answer():
    # Stage a synthetic LTM store (the parsed TruthSet) and confirm the LTM read
    # reaches the consumer: the soft-read content (which ranges over SPACE_LTM)
    # injected into a head changes it.
    from Spaces import GlobalAttention as GA
    from Layers import TernaryTruthStore
    m = _build("MM_qa.xml")
    if getattr(m, "symbolSpace", None) is None:
        pytest.skip("no symbolSpace")
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
    assert any(s["id"] == GA.SPACE_LTM for s in spaces), "the TruthSet must be addressable"
    m._global_attention_step(prev, ps)
    obs = m._global_attention_obs
    assert obs is not None and obs.get("content") is not None
    with torch.no_grad():
        m.global_attention.consume_gate.fill_(0.5)
    symbols = torch.zeros(int(obs["content"].shape[0]), 4, D)
    fed = m.global_attention.consume(symbols, obs["content"])
    assert float(fed.abs().sum()) > 0, "the LTM-inclusive read must reach the head"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
