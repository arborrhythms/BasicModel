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
    src = inspect.getsource(Language.OperationSelectionLayer)
    assert "op_space_role_idx" not in src and "position_space_role" not in src, "space_role machinery must be gone"


def test_compose_single_reduction_space_role():
    m = _model()
    layer = m.symbolSpace.languageLayer.operation_layer
    assert layer is m._stm_reducer()
    assert layer.r_reduce >= 8
    assert layer.r_apply > 0


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


def test_closing_budget_is_fixed_even_when_unary_choices_do_not_shrink(monkeypatch):
    m = _model()
    stm = m.conceptualSpace.stm
    stm.begin_forward(1, device=torch.device('cpu'))
    stm._depth.fill_(3)
    stm._buffer.fill_(1)
    calls = []
    def unary_round(**kwargs):
        calls.append(kwargs['row_gate'].clone())
        return torch.ones(1, dtype=torch.bool)
    monkeypatch.setattr(m, '_stm_operation_step', unary_round)
    m.syntacticOrder = 0
    root, depth = m._stm_reduce_to_single_S()
    assert len(calls) == 2 * stm.capacity
    assert depth.tolist() == [-3]
    assert not m._compose_complete.any()
    assert torch.isfinite(root).all()
