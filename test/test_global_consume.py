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
# (a) PrimedSymbolReader.consume in isolation
# ---------------------------------------------------------------------------

def test_consume_zero_init_is_noop():
    from Attention import PrimedSymbolReader
    ga = PrimedSymbolReader()                       # consume_gate zero-init
    symbols = torch.randn(2, 5, 8)
    content = torch.randn(2, 8)
    out = ga.consume(symbols, content)
    assert torch.equal(out, symbols), "zero-init gate must be a no-op residual"


def test_consume_gate_injects_on_leading_width():
    from Attention import PrimedSymbolReader
    ga = PrimedSymbolReader()
    with torch.no_grad():
        ga.consume_gate.fill_(0.5)
    symbols = torch.zeros(2, 3, 8)
    content = torch.ones(2, 8)
    out = ga.consume(symbols, content)
    assert torch.allclose(out, torch.full_like(out, 0.5)), (
        "gate>0 must add gate*content on the common leading width")


def test_consume_handles_2d_and_3d_symbols():
    from Attention import PrimedSymbolReader
    ga = PrimedSymbolReader()
    with torch.no_grad():
        ga.consume_gate.fill_(1.0)
    c = torch.ones(2, 6)
    out3 = ga.consume(torch.zeros(2, 4, 6), c)   # [B, N, D]
    out2 = ga.consume(torch.zeros(2, 6), c)      # [B, D]
    assert out3.shape == (2, 4, 6) and out2.shape == (2, 6)
    assert float(out3.detach().mean()) == 1.0 and float(out2.detach().mean()) == 1.0


def test_consume_none_content_is_noop():
    from Attention import PrimedSymbolReader
    ga = PrimedSymbolReader()
    s = torch.randn(2, 8)
    assert torch.equal(ga.consume(s, None), s)


def test_consume_width_mismatch_slices_to_common():
    from Attention import PrimedSymbolReader
    ga = PrimedSymbolReader()
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
    from configuration_fixtures import small_retained
    from recon_bench import _build_model
    with small_retained(name) as path:
        model, *_ = _build_model(path)
    return model


def _batch(m):
    import Models
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    return m.inputSpace.prepInput(items)


def _generation_forward(m, x):
    """The learned reader's consumer is generation, outside the numeric head."""
    output = m.forward(x)
    carrier = getattr(m, '_combine_last_cs_sub', m.conceptualSpace.subspace)
    idea = carrier.materialize().detach()
    generated = m._generation_read(idea, record=getattr(m, '_last_sentence_understanding', None))
    return (*output[:2], generated, *output[3:])


@pytest.mark.slow
def test_consume_gate_is_zero_by_default_on_global_config():
    m = _build("MM_global.xml")
    assert m.answer_attention is not None
    assert not hasattr(m, "answer_attention_consume")
    assert not m.answer_attention.consume_gate.detach().any()


@pytest.mark.slow
def test_qa_config_builds_with_consumer_and_ltm():
    m = _build("MM_qa.xml")
    assert m.answer_attention is not None
    assert not hasattr(m, 'answer_attention_consume')
    assert m.ltm_consolidation
    # the LTM store (the "book") exists for the SPACE_LTM read to range over
    assert getattr(m.symbolSpace, "ltm_store", None) is not None
    x = _batch(m)
    m.eval()
    with torch.no_grad():
        out = m.forward(x)[2]
    assert torch.isfinite(out).all()


@pytest.mark.slow
def test_zero_gate_matches_consume_off(monkeypatch):
    # Compare the actual zero gate with a head whose consume function is the
    # identity. No retired flag can stand in for bypassing the read.
    m = _build("MM_qa.xml")
    x = _batch(m)
    from configuration_fixtures import freeze_admission
    freeze_admission(m, x, monkeypatch)
    m.eval()
    consume = m.answer_attention.consume
    with torch.no_grad():
        monkeypatch.setattr(m.answer_attention, 'consume', lambda symbols, content: symbols)
        out_off_flag = _generation_forward(m, x)[2].clone()
        monkeypatch.setattr(m.answer_attention, 'consume', consume)
        out_on_gate0 = _generation_forward(m, x)[2].clone()
    assert torch.equal(out_off_flag, out_on_gate0), (
        "consume on with zero gate must equal consume off")


@pytest.mark.slow
def test_gate_feeds_the_read_into_the_answer(monkeypatch):
    m = _build("MM_qa.xml")
    x = _batch(m)
    from configuration_fixtures import freeze_admission
    freeze_admission(m, x, monkeypatch)
    m.eval()
    with torch.no_grad():
        m.answer_attention.consume_gate.zero_()
        base = _generation_forward(m, x)[2].clone()
        m.answer_attention.consume_gate.fill_(0.5)
        fed = _generation_forward(m, x)[2].clone()
    assert not torch.equal(base, fed), "a non-zero gate must feed the read back"


@pytest.mark.slow
def test_nonzero_gate_does_not_corrupt_the_reverse(monkeypatch):
    # Generation consumes its own operand. The carrier used by reconstruction
    # must remain byte-identical even when the retrieval gate is active.
    m = _build("MM_global.xml")
    x = _batch(m)
    from configuration_fixtures import freeze_admission
    freeze_admission(m, x, monkeypatch)
    m.eval()
    with torch.no_grad():
        m.answer_attention.consume_gate.zero_()
        head_off = _generation_forward(m, x)[2].clone()
        carrier_off = m._combine_last_cs_sub.materialize().clone()
        m.answer_attention.consume_gate.fill_(0.5)
        head_on = _generation_forward(m, x)[2].clone()
        carrier_on = m._combine_last_cs_sub.materialize().clone()
    assert not torch.equal(head_off, head_on), "the gated read must reach the head"
    assert torch.equal(carrier_off, carrier_on), (
        "the reverse carrier must be restored (reconstruction unaffected)")


@pytest.mark.slow
def test_answer_loss_trains_retrieval():
    # The output loss backprops through the read into the scorer + the consume
    # gate -- retrieval that helps the answer is rewarded.
    m = _build("MM_qa.xml")
    x = _batch(m)
    m.answer_attention.consume_gate.data.fill_(0.3)   # active so grad flows
    m.train()
    m.zero_grad(set_to_none=True)
    out = _generation_forward(m, x)
    out[2].pow(2).sum().backward()
    ga = m.answer_attention
    assert ga.consume_gate.grad is not None and float(ga.consume_gate.grad.abs().sum()) > 0
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in ga.scorer.parameters()), "the answer loss must train the scorer"


@pytest.mark.slow
def test_ltm_truthset_read_is_fed_to_the_answer():
    # Stage a synthetic LTM store (the parsed TruthSet) and confirm the LTM read
    # reaches the consumer: the soft-read content (which ranges over SPACE_LTM)
    # injected into a head changes it.
    from Attention import PrimedSymbolReader as GA
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
    prev = getattr(m, '_combine_last_cs_sub', None) or m.conceptualSpace.subspace
    spaces, _ = m._addressable_spaces(prev, ps)
    assert any(s["id"] == GA.SPACE_LTM for s in spaces), "the TruthSet must be addressable"
    m._answer_attention_step(prev, ps)
    obs = m._answer_attention_obs
    assert obs is not None and obs.get("content") is not None
    with torch.no_grad():
        m.answer_attention.consume_gate.fill_(0.5)
    symbols = torch.zeros(int(obs["content"].shape[0]), 4, D)
    fed = m.answer_attention.consume(symbols, obs["content"])
    assert float(fed.abs().sum()) > 0, "the LTM-inclusive read must reach the head"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
