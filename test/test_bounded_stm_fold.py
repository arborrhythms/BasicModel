"""Bounded-STM fold gates: capacity invariant + per-word ingestion."""
import os, sys
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import torch, warnings
import Models, Language
from util import init_config

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_PROJECT, "data")

def _model():
    init_config(path=os.path.join(_DATA, "MM_grammar.xml"),
                defaults_path=os.path.join(_DATA, "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(os.path.join(_DATA, "MM_grammar.xml"))
    Models.TheData.load("xor")
    return m

def test_stm_never_exceeds_cap_after_forward():
    m = _model(); m.train()
    cap = int(m.conceptualSpace.stm.capacity)
    loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m.forward(x)
    depth = m.conceptualSpace.stm._depth
    assert int(depth.max().item()) <= cap, f"STM depth {int(depth.max())} > cap {cap}"


def test_sentence_end_reduces_toward_root():
    m = _model(); m.train()
    loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m.forward(x)
    S, post_depth = m._stm_reduce_to_single_S()
    assert int(post_depth.max().item()) <= max(1, 3), "absolute rows must collapse near root"
    assert torch.isfinite(S).all(), "root state must be finite"


def test_binary_reducer_is_tier_free():
    import inspect, Language
    src = inspect.getsource(Language.BinaryStructuredReductionLayer)
    assert "op_tier_idx" not in src and "position_tier" not in src, "tier machinery must be gone"


def test_compose_single_reduction_tier():
    m = _model()
    lang = None
    for mod in m.modules():
        if type(mod).__name__ == "LanguageLayer" and len(getattr(mod, "_binary_layers", {})) > 0:
            lang = mod; break
    assert lang is not None, "no configured LanguageLayer found"
    assert len(lang._binary_layers) == 1, (
        f"expected a single reduction tier, got binary tiers {list(lang._binary_layers.keys())}")
    # all reduce ops now live in the one tier
    only = next(iter(lang._binary_layers.values()))
    assert only.r_reduce >= 8, f"merged tier should hold all reduce ops, got r_reduce={only.r_reduce}"


def test_lift_lower_stay_invertible_cs_ops():
    """Task 7 (per user directive, 2026-06-05): lift/lower remain ordinary
    C-tier (CS-internal) invertible sigma/pi ops returning non-quantized
    results -- they are NOT re-expressed as SS codebook round-trips (which
    would be lossy and break invertibility; codebook queries to SS are
    always quantized, but lift/lower must not be). The C/S tier delta was
    already removed in Task 5; the only remaining lift/lower delta is the
    conceptual-ORDER signature, not a tier move.
    """
    for cls in (Language.LiftLayer, Language.LowerLayer):
        assert cls.tier == 'C', f"{cls.__name__} must stay an ordinary C-tier op"
        assert cls.invertible is True, (
            f"{cls.__name__} must stay invertible (non-quantized result)")


def test_cap_equivalence_short_sentence():
    """Task 9: STM capacity parameter — equivalence gate.

    Spec claim: for N_total_pushes <= capacity, the force-to-fit
    back-pressure check (_max_depth_host >= capacity) is NEVER True before
    any push, so the forced reduce is a no-op.  When the condition never
    fires, two runs of the same sentence with cap=N_total and cap=N_total+K
    produce bit-identical STM end-states (same S, same depth).

    Implementation note — the xor/MM_grammar model uses depth-before-push
    semantics: just before the N_total-th push, _max_depth_host is
    N_total-1.  The check N_total-1 >= N_total is False, so NO reduce fires.
    This is the force-to-fit no-op condition ("capacity >= N_total").

    Architecture of STM pushes for this model:
      * ConceptualSpace.forward calls _stm_shift_and_push once per word
        position (parallel CS stage push, no back-pressure guard).
      * _per_word_body_step calls push_step_masked once per word position
        (serial loop push, back-pressure-guarded).
    With N_static=8 word positions, total pushes = 16 per forward pass.
    The DEFAULT_CAPACITY=8 is below N_total=16, so back-pressure fires
    for the default cap; we must use cap >= N_total to enter the no-op
    regime.  This test establishes N_total empirically with a large cap,
    then verifies the two-runs equivalence under cap=N_total and
    cap=N_total+4 (both in the no-pressure regime).

    Approach: one model, vary capacity between two reset runs (avoids
    cross-model random-init differences).  eval()+no_grad() makes each
    forward deterministic — no weight updates, no dropout variation.
    """
    m = _model()
    m.eval()

    stm = m.conceptualSpace.stm
    ws = stm._word_subspace

    loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)

    # --- Phase 1: measure N_total using a large cap (no pressure possible)
    LARGE_CAP = 32
    ws.idea_ensure_capacity(LARGE_CAP)
    # Reset to clean state before measurement pass
    m.conceptualSpace.Reset(hard=True)
    ws._idea_capacity = LARGE_CAP
    ws._idea_buffer = torch.zeros(
        int(ws._idea_buffer.shape[0]), LARGE_CAP,
        int(ws._idea_buffer.shape[2]))
    ws._idea_max_depth_host = 0
    ws._idea_depth.zero_()

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m.forward(x)
    N_total = int(stm._max_depth_host)
    assert N_total > 0, "forward produced zero STM pushes; data/model mismatch"
    assert N_total <= LARGE_CAP, (
        f"N_total={N_total} exceeded measurement cap={LARGE_CAP}; "
        f"increase LARGE_CAP")

    # --- Phase 2: run with cap=N_total, verify no back-pressure fires
    def _reset_to_cap(cap):
        """Hard-reset the STM and set idea-stack capacity to `cap`."""
        m.conceptualSpace.Reset(hard=True)
        ws._idea_capacity = cap
        ws._idea_buffer = torch.zeros(
            int(ws._idea_buffer.shape[0]), cap,
            int(ws._idea_buffer.shape[2]))
        ws._idea_max_depth_host = 0
        ws._idea_depth.zero_()

    # cap=N_total: depth-before-push semantics guarantee the last push sees
    # _max_depth_host = N_total - 1 < N_total = cap → no reduce fires.
    _reset_to_cap(N_total)
    bp_count_tight = [0]
    _orig_reduce = m._stm_bounded_reduce_step
    def _counting_reduce(*args, **kwargs):
        # Force-to-fit (ingestion back-pressure) calls
        # _stm_bounded_reduce_step() with no protect_depth; the sentence-end
        # sweep passes protect_depth=. Discriminate on the argument, not a
        # hard-coded line number, so this robustly counts ONLY ingestion fires
        # (a line-number check would silently miscount after any Models.py edit).
        if not args and kwargs.get('protect_depth') is None:
            bp_count_tight[0] += 1
        return _orig_reduce(*args, **kwargs)
    m._stm_bounded_reduce_step = _counting_reduce
    try:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m.forward(x)
    finally:
        m._stm_bounded_reduce_step = _orig_reduce

    assert bp_count_tight[0] == 0, (
        f"force-to-fit back-pressure fired {bp_count_tight[0]} time(s) "
        f"with cap={N_total} == N_total; expected 0 (no-op regime). "
        f"depth-before-push semantics: max_depth_host reaches N_total-1 "
        f"before the last push, so N_total-1 >= N_total is False.")
    assert int(stm._max_depth_host) == N_total, (
        f"expected max_depth_host=={N_total} after no-pressure run, "
        f"got {stm._max_depth_host}")

    # --- Phase 3: snapshot end-state for cap=N_total
    S_tight = (m._stm_single_S.detach().clone()
               if getattr(m, "_stm_single_S", None) is not None
               else stm._buffer[:, :1, :].detach().clone())
    depth_tight = stm._depth.clone()

    # --- Phase 4: run with cap=N_total+4, same input
    _reset_to_cap(N_total + 4)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m.forward(x)
    S_loose = (m._stm_single_S.detach().clone()
               if getattr(m, "_stm_single_S", None) is not None
               else stm._buffer[:, :1, :].detach().clone())
    depth_loose = stm._depth.clone()

    # --- Phase 5: assert equivalence — cap >= N_total in both runs means
    # no back-pressure fired in either; the reduce-to-S collapse operates
    # on the same full N_total-item STM, so the outputs are bit-identical.
    assert torch.equal(depth_tight, depth_loose), (
        f"STM depth differs: cap={N_total} → {depth_tight.tolist()}, "
        f"cap={N_total+4} → {depth_loose.tolist()}")
    assert torch.equal(S_tight, S_loose), (
        f"STM sentence-S differs between cap={N_total} and cap={N_total+4}; "
        f"max |Δ| = {(S_tight - S_loose).abs().max().item():.3e}. "
        f"Both runs are in the no-pressure regime (cap >= N_total={N_total}), "
        f"so identical STM content before reduce-to-S should yield "
        f"identical S.")
