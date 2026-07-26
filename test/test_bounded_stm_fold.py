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


def test_binary_reducer_is_space_role_free():
    import inspect, Language
    src = inspect.getsource(Language.BinaryStructuredReductionLayer)
    assert "op_space_role_idx" not in src and "position_space_role" not in src, "space_role machinery must be gone"


def test_compose_single_reduction_space_role():
    m = _model()
    lang = None
    for mod in m.modules():
        if type(mod).__name__ == "LanguageLayer" and len(getattr(mod, "_binary_layers", {})) > 0:
            lang = mod; break
    assert lang is not None, "no configured LanguageLayer found"
    assert len(lang._binary_layers) == 1, (
        f"expected a single reduction space_role, got binary space_roles {list(lang._binary_layers.keys())}")
    # all reduce ops now live in the one space_role
    only = next(iter(lang._binary_layers.values()))
    assert only.r_reduce >= 8, f"merged space_role should hold all reduce ops, got r_reduce={only.r_reduce}"


def test_lift_lower_stay_invertible_cs_ops():
    """Task 7 (per user directive, 2026-06-05): lift/lower remain ordinary
    CS-space_role (CS-internal) invertible sigma/pi ops returning non-quantized
    results -- they are NOT re-expressed as SS codebook round-trips (which
    would be lossy and break invertibility; codebook queries to SS are
    always quantized, but lift/lower must not be). The CS/SS space_role delta was
    already removed in Task 5; the only remaining lift/lower delta is the
    conceptual-ORDER signature, not a space_role move.
    """
    for cls in (Language.LiftLayer, Language.LowerLayer):
        assert cls.space_role == 'CS', f"{cls.__name__} must stay an ordinary CS-space_role op"
        assert cls.invertible is True, (
            f"{cls.__name__} must stay invertible (non-quantized result)")


def test_cap_equivalence_short_sentence():
    """A non-binding hard capacity leaves the sentence root unchanged.

    Capacity now also controls the *soft* occupancy-pressure threshold, so
    arbitrary capacities are intentionally not equivalent at a nonzero
    ``stmReduceTau``. Set that threshold to zero here to isolate the hard
    demand controller, then compare two capacities larger than the observed
    live-stack peak. Neither run may apply a demand reduction and both must
    produce the same final root/depth.
    """
    m = _model()
    m.eval()
    m.stm_reduce_tau = 0.0

    stm = m.conceptualSpace.stm
    ss = stm._word_subspace

    loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)

    # Measure the live-stack peak with a deliberately loose capacity.
    LARGE_CAP = 32
    ss.idea_ensure_capacity(LARGE_CAP)
    m.conceptualSpace.Reset(hard=True)
    ss._idea_capacity = LARGE_CAP
    ss._idea_buffer = torch.zeros(
        int(ss._idea_buffer.shape[0]), LARGE_CAP,
        int(ss._idea_buffer.shape[2]))
    ss._idea_max_depth_host = 0
    ss._idea_depth.zero_()

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m.forward(x)
    peak_depth = int(stm._max_depth_host)
    assert peak_depth > 0, "forward produced zero STM pushes; data/model mismatch"
    assert peak_depth < LARGE_CAP, (
        f"peak_depth={peak_depth} reached measurement cap={LARGE_CAP}; "
        f"increase LARGE_CAP")

    def _reset_to_cap(cap):
        m.conceptualSpace.Reset(hard=True)
        ss._idea_capacity = cap
        ss._idea_buffer = torch.zeros(
            int(ss._idea_buffer.shape[0]), cap,
            int(ss._idea_buffer.shape[2]))
        ss._idea_max_depth_host = 0
        ss._idea_depth.zero_()

    def _run_without_demand(cap):
        _reset_to_cap(cap)
        demand_applied = []
        original_reduce = m._stm_bounded_reduce_step

        def _recording_reduce(*args, **kwargs):
            reduced = original_reduce(*args, **kwargs)
            if kwargs.get("demand", False):
                demand_applied.append(reduced.detach().clone())
            return reduced

        m._stm_bounded_reduce_step = _recording_reduce
        try:
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                m.forward(x)
        finally:
            m._stm_bounded_reduce_step = original_reduce
        assert not any(bool(mask.any()) for mask in demand_applied), (
            f"hard capacity demand applied with non-binding cap={cap}")
        root = (m._stm_single_S.detach().clone()
                if getattr(m, "_stm_single_S", None) is not None
                else stm._buffer[:, :1, :].detach().clone())
        return root, stm._depth.clone()

    # Leave explicit headroom above the observed peak so neither capacity can
    # enter the demand path. Sixteen is also the historical upper bound for
    # this eight-column fixture, keeping the assertion stable if grammar
    # choices reduce its current peak.
    tight_cap = max(16, peak_depth + 1)
    S_tight, depth_tight = _run_without_demand(tight_cap)
    S_loose, depth_loose = _run_without_demand(tight_cap + 4)

    assert torch.equal(depth_tight, depth_loose), (
        f"STM depth differs: cap={tight_cap} → {depth_tight.tolist()}, "
        f"cap={tight_cap+4} → {depth_loose.tolist()}")
    assert torch.equal(S_tight, S_loose), (
        f"STM sentence-S differs between cap={tight_cap} and "
        f"cap={tight_cap+4}; "
        f"max |Δ| = {(S_tight - S_loose).abs().max().item():.3e}. "
        "Both runs are outside hard capacity demand and use the same "
        "capacity-independent zero soft threshold.")
