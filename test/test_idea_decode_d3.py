"""Idea-decode Stage D3 CONSUMER (doc/old/2026-06-20-idea-decoder.md): under
<ideaDecode>, the grammar <generate> reverse runs on the SYNTACTIC WholeSpace
(symbolSpace.wholeSpace, the one with the SyntacticLayer + a populated
subspace -- NOT wholeSpaces[0]) and DRIVES the reverse seed, so the surface
words come from the grammar. Shape-guarded: drives on an exact match (the
symbol_dim==concept_dim invariant); compact-symbol configs fall back unchanged
(they need a learned symbol->concept expander). Default off -> byte-identical
(the reverse-roundtrip suite is the off-path witness).
"""
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import torch
_DATA = os.path.join(os.path.dirname(_BIN), "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")


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


def _forward(m):
    import Models
    from util import TheXMLConfig
    Models.TheData.load(TheXMLConfig.get("data.dataset", default="xor"))
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    m.eval()
    with torch.no_grad():
        m.forward(x)
    return m


def test_idea_decode_defaults_off():
    m = _build("MM_mereology.xml")
    assert getattr(m, "idea_decode", None) is False


def test_grammar_decode_runs_on_syntactic_ws():
    """WS-reference fix: the grammar <generate> reverse uses the SYNTACTIC WS
    (symbolSpace.wholeSpace), which has a live SyntacticLayer + populated
    subspace -- so it returns a real [B,1,symbol_dim] decode (NOT None as it did
    when seeded from the empty wholeSpaces[0])."""
    m = _forward(_build("MM_mereology_serial.xml"))
    ws = m._idea_decode_ws()
    assert ws is m.symbolSpace.wholeSpace
    assert type(getattr(ws, "syntacticLayer", None)).__name__ == "SyntacticLayer"
    with torch.no_grad():
        gen = m._run_idea_decode_generate()
    assert gen is not None
    gev = gen.materialize()
    assert gev is not None and gev.dim() == 3


def test_consumer_drives_seed_on_shape_match():
    """The consumer SETS the reverse seed to the grammar decode when shapes
    match exactly (the symbol_dim==concept_dim invariant)."""
    m = _forward(_build("MM_mereology_serial.xml"))
    with torch.no_grad():
        gev = m._run_idea_decode_generate().materialize()
        seed = m.conceptualSpace.subspace
        seed.set_event(torch.zeros_like(gev))      # shape-matched seed
        m.idea_decode = True
        out = m._idea_decode_drive(seed)
        assert torch.allclose(out.materialize(), gev, atol=1e-5)   # driven
        assert m._idea_decode_parked is not None                   # parked too


def test_consumer_falls_back_on_width_mismatch():
    """Compact-symbol configs (symbol_dim << concept_dim) need a learned
    expander; until then the consumer leaves the seed UNCHANGED (no corruption
    of the carried reconstruction)."""
    m = _forward(_build("MM_mereology_serial.xml"))
    with torch.no_grad():
        gev = m._run_idea_decode_generate().materialize()
        seed = m.conceptualSpace.subspace
        big = torch.zeros(gev.shape[0], 1, 1020)   # concept width != symbol_dim
        seed.set_event(big)
        before = seed.materialize().clone()
        m.idea_decode = True
        out = m._idea_decode_drive(seed)
        assert torch.equal(before, out.materialize())             # unchanged


def test_reverse_off_path_unchanged():
    """idea_decode off -> reverse takes the existing path (drive skipped)."""
    m = _forward(_build("MM_mereology.xml"))
    with torch.no_grad():
        snap = m.conceptualSpace.stm.snapshot()
        if snap is not None:
            m.conceptualSpace.subspace.set_event(snap)
            rev = m.reverse(m.conceptualSpace.subspace)   # idea_decode off
            assert rev is not None
