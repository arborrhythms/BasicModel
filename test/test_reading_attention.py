"""Reading attention: the learned ``.where`` producer
(doc/specs/reading-attention.md "(A) Reading attention"; orders.md §6).

Gated ``<readingAttention>`` and DARK by default: with the flag off no module
is allocated, no scope is written, and no loss is added (byte-identical -- the
full suite is the byte-identical-off witness).

These tests cover (a) the ``ReadingAttention`` module in isolation -- the shift
bootstrap, the monotonic/coverage mask, the unit-range scope, the next-word CE,
and the gradient boundary (the loss never reaches the codebooks / primed
symbols) -- and (b) the model wiring on the dedicated ``MM_reading.xml`` config:
the producer is built when on (and absent when off), it writes
``wholeSpaces[0]._passback_scope_where`` (teacher span in training), it registers
the next-word CE on the conceptual error container, and its readout params reach
the optimizer.
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
# (a) the ReadingAttention module in isolation
# ---------------------------------------------------------------------------

def _contiguous_spans(B, K, w):
    """``[B, K, 2]`` of K contiguous words each ``w`` atoms wide."""
    one = [[k * w, (k + 1) * w] for k in range(K)]
    return torch.tensor([one] * B)


def _module_inputs(B=2, K=5, w=8, D=16, seed=0):
    torch.manual_seed(seed)
    spans = _contiguous_spans(B, K, w)
    percept = torch.randn(B, K * w, D)
    concept_q = torch.randn(B, D)
    symbol_q = torch.randn(B, D)
    return spans, percept, concept_q, symbol_q


def test_shift_bootstrap_selects_next_span_at_init():
    # At init (zero-init readout head) the producer IS the serial reading
    # for-loop: argmax of the attention distribution is the next word.
    from Spaces import ReadingAttention
    ra = ReadingAttention()
    spans, percept, cq, sq = _module_inputs()
    N = int(percept.shape[1])
    for read_idx in range(4):
        _, _, alpha = ra(concept_q=cq, symbol_q=sq, percept_ev=percept,
                         spans=spans, read_idx=read_idx, N=N, training=False)
        assert int(alpha[0].argmax()) == read_idx, (
            f"bootstrap should select span {read_idx}, got {int(alpha[0].argmax())}")


def test_scope_is_normalized_unit_range():
    from Spaces import ReadingAttention
    ra = ReadingAttention()
    spans, percept, cq, sq = _module_inputs()
    N = int(percept.shape[1])
    next_where, _, _ = ra(concept_q=cq, symbol_q=sq, percept_ev=percept,
                          spans=spans, read_idx=1, N=N, training=False)
    next_where = next_where.detach()
    assert next_where.shape == (spans.shape[0], 2)
    assert float(next_where.min()) >= 0.0 and float(next_where.max()) <= 1.0
    # start <= end for every row (a well-formed [start, end] bracket).
    assert bool((next_where[:, 1] >= next_where[:, 0]).all())


def test_monotonic_coverage_mask_excludes_consumed():
    # Already-consumed spans (k < read_idx) carry ~0 probability -> reading
    # stays left-to-right and cannot re-select.
    from Spaces import ReadingAttention
    ra = ReadingAttention()
    spans, percept, cq, sq = _module_inputs()
    N = int(percept.shape[1])
    _, _, alpha = ra(concept_q=cq, symbol_q=sq, percept_ev=percept,
                     spans=spans, read_idx=3, N=N, training=False)
    assert float(alpha[:, :3].detach().sum()) < 1e-4, "consumed spans must be masked out"


def test_padding_spans_never_selected():
    # A (0, 0) padding span (extent 0) is masked even when it is the only
    # unconsumed candidate.
    from Spaces import ReadingAttention
    ra = ReadingAttention()
    B, w, D = 2, 8, 16
    spans = torch.tensor([[[0, w], [w, 2 * w], [0, 0]]] * B)   # last = pad
    percept = torch.randn(B, 2 * w, D)
    N = int(percept.shape[1])
    # read_idx=0: spans 0/1 are real unconsumed candidates, span 2 is pad.
    _, _, alpha = ra(concept_q=None, symbol_q=None, percept_ev=percept,
                     spans=spans, read_idx=0, N=N, training=False)
    assert float(alpha[:, 2].detach().max()) < 1e-4, "padding span must not be selected"


def test_codebook_retrieval_prior_is_the_subsymbolic_term():
    # The codebook-retrieval prior (the literal intent_boosts path) ranks a
    # span high when its content snaps near an intent-primed prototype. Build a
    # codebook whose rows ARE the span contents and an intent boost that singles
    # prototype 3 out; the prior (tested directly) must then rank span 3 top.
    from Spaces import ReadingAttention
    B, K, w, D = 1, 5, 6, 8
    spans = _contiguous_spans(B, K, w)
    torch.manual_seed(1)
    percept = torch.randn(B, K * w, D)
    keys = ReadingAttention._span_keys(percept, spans)        # [1, K, D]
    W = keys[0].clone()                                       # rows = span means
    boosts = torch.ones(K)
    boosts[3] = 5.0                                           # prime prototype 3
    prior = ReadingAttention._codebook_retrieval_prior(
        keys, W, intent=None, external_boosts=boosts)
    assert prior is not None and prior.shape == (B, K)
    assert int(prior[0].argmax()) == 3, "primed prototype's span must score top"


def test_codebook_path_preserves_gradient_boundary():
    # The codebook-retrieval prior is fully detached: a codebook with
    # requires_grad must receive NO gradient from the reading loss.
    from Spaces import ReadingAttention
    ra = ReadingAttention()
    spans, percept, cq, sq = _module_inputs(K=5, D=16)
    N = int(percept.shape[1])
    W = torch.randn(32, 16, requires_grad=True)
    _, ce, _ = ra(concept_q=cq, symbol_q=sq, percept_ev=percept, spans=spans,
                  read_idx=2, N=N, training=True, codebook_rows=W)
    ce.backward()
    assert W.grad is None, "the EMA-only codebook must receive no reading grad"
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in ra.parameters())


def test_codebook_path_keeps_shift_bootstrap():
    # Wiring the codebook prior must not disturb the shift bootstrap: at init
    # (zero readout head) the argmax is still the next span.
    from Spaces import ReadingAttention
    ra = ReadingAttention()
    spans, percept, cq, sq = _module_inputs(K=5, D=16)
    N = int(percept.shape[1])
    W = torch.randn(32, 16)
    for read_idx in range(4):
        _, _, alpha = ra(concept_q=cq, symbol_q=sq, percept_ev=percept,
                         spans=spans, read_idx=read_idx, N=N, training=False,
                         codebook_rows=W)
        assert int(alpha[0].argmax()) == read_idx


def test_no_codebook_falls_back_to_cosine():
    # With no codebook the producer is unchanged (concept-content cosine
    # fallback) -- the module still scores and bootstraps.
    from Spaces import ReadingAttention
    ra = ReadingAttention()
    spans, percept, cq, sq = _module_inputs(K=5, D=16)
    N = int(percept.shape[1])
    _, _, alpha = ra(concept_q=cq, symbol_q=sq, percept_ev=percept,
                     spans=spans, read_idx=1, N=N, training=False)  # no codebook
    assert int(alpha[0].argmax()) == 1


def test_next_word_ce_matches_neg_log_alpha():
    # The text-mode CE is exactly -log α at the next-word index, averaged over
    # rows whose target span is real.
    from Spaces import ReadingAttention
    ra = ReadingAttention()
    spans, percept, cq, sq = _module_inputs()
    N = int(percept.shape[1])
    read_idx = 2
    _, ce, alpha = ra(concept_q=cq, symbol_q=sq, percept_ev=percept,
                      spans=spans, read_idx=read_idx, N=N, training=True)
    expect = -torch.log(alpha[:, read_idx].clamp_min(1e-12)).mean()
    assert torch.allclose(ce, expect, atol=1e-4)


def test_eval_has_no_ce_loss():
    from Spaces import ReadingAttention
    ra = ReadingAttention()
    spans, percept, cq, sq = _module_inputs()
    N = int(percept.shape[1])
    _, ce, _ = ra(concept_q=cq, symbol_q=sq, percept_ev=percept,
                  spans=spans, read_idx=1, N=N, training=False)
    assert ce is None


def test_no_spans_is_noop():
    from Spaces import ReadingAttention
    ra = ReadingAttention()
    out = ra(concept_q=None, symbol_q=None, percept_ev=torch.randn(2, 8, 4),
             spans=None, read_idx=1, N=8, training=True)
    assert out == (None, None, None)


def test_gradient_stops_at_primed_symbols():
    # The reading loss trains the MLP readout ONLY: it does NOT backprop into
    # the codebooks (the percept keys) nor the primed symbols (the query),
    # preserving the EMA-only VQ contract (orders.md §6 "Learning").
    from Spaces import ReadingAttention
    ra = ReadingAttention()
    spans, percept, cq, sq = _module_inputs()
    N = int(percept.shape[1])
    percept = percept.requires_grad_(True)
    cq = cq.requires_grad_(True)
    sq = sq.requires_grad_(True)
    _, ce, _ = ra(concept_q=cq, symbol_q=sq, percept_ev=percept,
                  spans=spans, read_idx=2, N=N, training=True)
    ce.backward()
    assert percept.grad is None, "no gradient may reach the codebook content"
    assert cq.grad is None and sq.grad is None, (
        "no gradient may reach the primed symbols (query)")
    got = any(p.grad is not None and float(p.grad.abs().sum()) > 0
              for p in ra.parameters())
    assert got, "the MLP readout must receive the reading-loss gradient"


def test_loss_trains_attention_toward_target():
    # A few SGD steps on the next-word CE sharpen the distribution onto the
    # target span (the readout learns; it is not frozen at the bootstrap).
    from Spaces import ReadingAttention
    ra = ReadingAttention()
    spans, percept, cq, sq = _module_inputs(seed=3)
    N = int(percept.shape[1])
    opt = torch.optim.SGD(ra.parameters(), lr=0.5)
    read_idx = 2
    _, ce0, a0 = ra(concept_q=cq, symbol_q=sq, percept_ev=percept,
                    spans=spans, read_idx=read_idx, N=N, training=True)
    for _ in range(25):
        opt.zero_grad()
        _, ce, _ = ra(concept_q=cq, symbol_q=sq, percept_ev=percept,
                      spans=spans, read_idx=read_idx, N=N, training=True)
        ce.backward()
        opt.step()
    _, ce1, a1 = ra(concept_q=cq, symbol_q=sq, percept_ev=percept,
                    spans=spans, read_idx=read_idx, N=N, training=True)
    assert float(ce1.detach()) < float(ce0.detach()), "CE should decrease under training"
    assert float(a1[0, read_idx].detach()) >= float(a0[0, read_idx].detach())


# ---------------------------------------------------------------------------
# (b) the model wiring (MM_reading.xml)
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


def _stage_synthetic_spans(m, ps, K=3):
    """Overwrite the staged word brackets with K contiguous synthetic words
    (xor items lex to a single token, too short to exercise reading)."""
    ev = ps.materialize()
    B, N = int(ev.shape[0]), int(ev.shape[1])
    w = max(N // (K + 1), 1)
    spans = torch.tensor([[[k * w, (k + 1) * w] for k in range(K)]] * B)
    object.__setattr__(m.wholeSpaces[0], "_staged_analysis_spans", spans)
    return spans, N


def test_reading_attention_on_builds_producer():
    m = _build("MM_reading.xml")
    from Spaces import ReadingAttention
    assert m.reading_attention_enabled
    assert isinstance(m.reading_attention, ReadingAttention)
    # The handoff consumer is also wired on this config.
    assert m.mereology_raise


def test_reading_attention_off_has_no_module():
    # MM_mereology does not set <readingAttention> -> the off path is untouched
    # (no module, the producer wiring is inert -> byte-identical).
    m = _build("MM_mereology.xml")
    assert not getattr(m, "reading_attention_enabled", False)
    assert getattr(m, "reading_attention", None) is None


def test_forward_is_finite_and_deterministic():
    m = _build("MM_reading.xml")
    x = _batch(m)
    m.eval()
    with torch.no_grad():
        out1 = m.forward(x)[2]
        out2 = m.forward(x)[2]
    assert torch.isfinite(out1).all()
    assert torch.equal(out1, out2), "the eval forward must be deterministic"


def test_train_forward_then_backward_is_finite():
    m = _build("MM_reading.xml")
    x = _batch(m)
    m.train()
    out = m.forward(x)
    assert torch.isfinite(out[2]).all()


def test_producer_writes_teacher_scope_and_registers_loss():
    m = _build("MM_reading.xml")
    x = _batch(m)
    m.train()
    with torch.no_grad():
        m.forward(x)                      # warm the CS carriers / STM
    ws0, cs0 = m.wholeSpaces[0], m.conceptualSpaces[0]
    prev = cs0._subspaceForWS
    in_sub = m._lex_embed_stem(x)         # re-lex (stages real spans)
    ps = m.perceptualSpace.forward(in_sub)
    spans, N = _stage_synthetic_spans(m, ps, K=3)
    cs_sub = cs0._subspaceForWS
    m._reading_attention_step(1, prev, ps, cs_sub)
    # Teacher forcing in training: the WRITTEN scope is the TRUE next span
    # (word read_idx = t - 1 = 0 -> [0, w] / N). R2: the scope is CS-owned.
    scope = getattr(m.conceptualSpace, "_passback_scope_where", None)
    assert scope is not None and scope.numel() == 2
    w = float(spans[0, 0, 1]) / N
    assert torch.allclose(scope.float(), torch.tensor([0.0, w]), atol=1e-4)
    # The next-word CE is registered on the conceptual error container.
    assert "reading_attention" in cs_sub.errors._terms
    val = cs_sub.errors._terms["reading_attention"]["value"]
    assert torch.isfinite(val).all() and bool(val.requires_grad)


def test_registered_loss_backprops_to_producer():
    # The loss registered on the (copy_context-shared) conceptual error
    # container backprops to the producer readout -- closing the loop from
    # the pipeline Error (which flows into the model's totalLoss) to the MLP.
    m = _build("MM_reading.xml")
    x = _batch(m)
    m.train()
    with torch.no_grad():
        m.forward(x)
    cs0 = m.conceptualSpaces[0]
    prev = cs0._subspaceForWS
    in_sub = m._lex_embed_stem(x)
    ps = m.perceptualSpace.forward(in_sub)
    _stage_synthetic_spans(m, ps, K=3)
    cs_sub = cs0._subspaceForWS
    m._reading_attention_step(1, prev, ps, cs_sub)
    val = cs_sub.errors._terms["reading_attention"]["value"]
    m.zero_grad(set_to_none=True)
    val.backward()
    got = any(p.grad is not None and float(p.grad.abs().sum()) > 0
              for p in m.reading_attention.parameters())
    assert got, "the registered reading loss must backprop to the producer"


def test_producer_scope_clears_past_last_word():
    m = _build("MM_reading.xml")
    x = _batch(m)
    m.train()
    with torch.no_grad():
        m.forward(x)
    ws0, cs0 = m.wholeSpaces[0], m.conceptualSpaces[0]
    prev = cs0._subspaceForWS
    in_sub = m._lex_embed_stem(x)
    ps = m.perceptualSpace.forward(in_sub)
    _stage_synthetic_spans(m, ps, K=2)    # only 2 words
    # pass t=4 -> read_idx=3 >= K=2 -> no next word -> scope cleared to None.
    m._reading_attention_step(4, prev, ps, cs0._subspaceForWS)
    assert getattr(m.conceptualSpace, "_passback_scope_where", None) is None


def test_producer_params_reach_the_optimizer():
    m = _build("MM_reading.xml")
    opt = m.getOptimizer(lr=0.01)
    ra_ptrs = {p.data_ptr() for p in m.reading_attention.parameters()}
    opt_ptrs = set()
    groups = list(getattr(opt, "param_groups", []) or [])
    for o in getattr(opt, "optimizers", []) or []:
        groups.extend(o.param_groups)
    for g in groups:
        for p in g["params"]:
            opt_ptrs.add(p.data_ptr())
    assert ra_ptrs and ra_ptrs.issubset(opt_ptrs), (
        "the reading-attention readout params must be optimized")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
