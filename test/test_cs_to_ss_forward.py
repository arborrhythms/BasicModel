"""Phase 4: CS -> SS is .forward()-mediated.

The symbol bind leg is produced by ``SymbolSpace.forward_concept_to_symbol``
(the concept arrives through ``forward``; the symbol is the row-aligned view of
the concept), NOT by the retired ``ConceptualSpace._build_symbol_leg`` reach
into the WholeSpace meta codebook + the stashed ``_model_symbolSpace`` pointer.
A symbol-tower config in PARALLEL mode runs the 3-stream bind through it.
"""

import pytest
import inspect
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

_DATA = os.path.join(os.path.dirname(_BIN), "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")


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






@pytest.mark.slow
def test_xor_symbols_parallel_forward_smoke():
    """XOR_symbols.xml is concept_order_limit=0 (word_brackets=False -> PARALLEL).
    Symbols have no codebook (rule 9): the bind is of the two percept streams,
    and the forward must run without error."""
    import Models
    from util import TheXMLConfig
    m = _build("XOR_symbols.xml")
    assert not hasattr(m, 'symbol_tower') and m.word_brackets is False
    assert int(m.conceptualSpace.combine.n_streams) == 2
    Models.TheData.load(TheXMLConfig.get("data.dataset", default="xor"))
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    m.eval()
    with torch.no_grad():
        m.forward(x)


@pytest.mark.slow
def test_forward_concept_to_symbol_without_a_field_is_none():
    """A carrier with no concept field gives no leg: the leg is the symbols'
    two lanes, and a concept event is not published as symbols."""
    m = _build("XOR_symbols.xml")
    ss = m.symbolSpace
    D = 8
    B, N = 2, 3
    sub = Spaces.SubSpace(inputShape=(N, D), outputShape=(N, D),
                          nInputDim=D, nOutputDim=D)
    sub.set_event(torch.randn(B, N, D))
    assert ss.forward_concept_to_symbol(sub) is None


@pytest.mark.slow
def test_forward_concept_to_symbol_empty_is_none():
    m = _build("XOR_symbols.xml")
    ss = m.symbolSpace
    empty = Spaces.SubSpace(inputShape=(1, 8), outputShape=(1, 8),
                            nInputDim=8, nOutputDim=8)
    assert ss.forward_concept_to_symbol(empty) is None
    assert ss.forward_concept_to_symbol(None) is None


@pytest.mark.slow
def test_symbol_leg_of_a_field_whose_store_holds_no_symbol_is_none():
    """A field over a store with no symbol reads none (the leg's lanes,
    gradient and contents are certificate 6 in test_item17_symbols)."""
    m = _build("XOR_symbols.xml")
    ss = m.symbolSpace
    D, B, N = 8, 2, 3
    sub = Spaces.SubSpace(inputShape=(N, D), outputShape=(N, D),
                          nInputDim=D, nOutputDim=D)
    sub.set_event(torch.randn(B, N, D))
    object.__setattr__(sub, "_concept_activations", torch.rand(N, B, 1, 2, requires_grad=True))
    object.__setattr__(sub, "_concept_inventory_rows", torch.arange(N))
    assert ss.forward_concept_to_symbol(sub) is None
