"""Phase 0 contract tests -- analysis/synthesis dual-input plan
(doc/plans/2026-06-08-analysis-synthesis-dual-input.md, rev. 2026-06-09).

Orientation under test (the CORRECTED one):

  percepts_in -- the ATOM view  -> PS (bottom-up synthesis, Sigma/union)
  concepts_in -- the UNITY view -> SS (top-down analysis, Pi/intersection)

  InputSpace.forward(input) -> (percepts_in, concepts_in)
    percepts_in: the atom-stream SubSpace; content channel [B, N, 1]
    concepts_in: ONE width-N event [B, 1, N] (same values, unity view)

Knob-acceptance contracts (<synthesis> on PS, <analysis>/<lexer> on SS) land
with Phase 4, where the schema changes; they are deliberately not stubbed
here.
"""
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path: sys.path.insert(0, _BIN)

import pytest
import torch


def _build(name):
    import Models, Language
    from util import init_config
    p = os.path.join(os.path.dirname(_BIN), "data", name)
    init_config(path=p, defaults_path=os.path.join(
        os.path.dirname(_BIN), "data", "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m


def _staged_batch(m):
    import Models
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    return m.inputSpace.prepInput(items)


def test_inputspace_forward_returns_dual_view():
    # Phase 1 contract: InputSpace.forward emits BOTH views of one source.
    m = _build("MM_20M.xml")
    x = _staged_batch(m)
    with torch.no_grad():
        percepts_in, concepts_in = m.inputSpace.forward(x)
    # Atom view: the lexed SubSpace; its content channel is [B, N, 1]
    # (N atoms of width-1 byte content; the where/when band is coordinate
    # metadata, not content).
    assert hasattr(percepts_in, "materialize"), (
        "percepts_in must be the atom-stream SubSpace")
    what = percepts_in.materialize(mode="what")
    assert what is not None and what.dim() == 3 and what.shape[-1] == 1, (
        "atom view content channel must be [B, N, 1], got "
        f"{None if what is None else tuple(what.shape)}")
    # Unity view: ONE width-N event.
    assert torch.is_tensor(concepts_in) and concepts_in.dim() == 3, (
        f"concepts_in must be a [B, 1, N] tensor, got {type(concepts_in)!r}")
    B, one, N = concepts_in.shape
    assert one == 1, (
        f"unity view must be a SINGLE event, got {tuple(concepts_in.shape)}")
    assert B == what.shape[0] and N == what.shape[1], (
        "the two views must present the SAME B and N "
        f"(atoms {tuple(what.shape)}, unity {tuple(concepts_in.shape)})")


def test_dual_views_share_values():
    # The two views are views of ONE presentation: same values, different
    # shape. (Analysis is non-altering; the unity view is the byte content
    # verbatim, not a transform of it.)
    m = _build("MM_20M.xml")
    x = _staged_batch(m)
    with torch.no_grad():
        percepts_in, concepts_in = m.inputSpace.forward(x)
        what = percepts_in.materialize(mode="what")
    assert torch.equal(
        concepts_in.squeeze(1).to(torch.float32),
        what[..., 0].to(torch.float32)), (
        "unity-view values must equal the atom-view content channel")


def test_ss_accepts_unity_kwarg_loudly():
    # Phase 0 contract: SymbolicSpace.forward ACCEPTS the optional
    # IS_concepts kwarg; until Phase 2 wires the consumption, passing a real
    # unity must FAIL LOUDLY (a silently dropped concept input is the
    # failure mode this plan exists to prevent).
    m = _build("MM_20M.xml")
    ss = m.symbolicSpace
    seed = m._empty_seed_ss
    out = ss.forward(seed, IS_concepts=None)  # None == legacy path, no raise
    assert out is not None
    with pytest.raises(NotImplementedError):
        ss.forward(seed, IS_concepts=torch.zeros(2, 1, 16))


def test_full_forward_green_with_dual_view():
    # The orchestration shim threads the tuple; the model forward is intact
    # and the unity view is parked for Phase 2 (staged, unused).
    m = _build("MM_20M.xml")
    x = _staged_batch(m)
    with torch.no_grad():
        out = m.forward(x)[2]
    assert out is not None and torch.isfinite(out).all()
    staged = getattr(m, "_staged_concepts_in", None)
    assert staged is not None and staged.dim() == 3 and staged.shape[1] == 1, (
        "the forward must park the unity view on _staged_concepts_in "
        "(Phase 1: staged, unused; Phase 2 consumes it at SS stage 0)")


@pytest.mark.xfail(
    reason="Phase 7 (reconstruction recombination) not landed", strict=False)
def test_inputspace_reverse_two_branch():
    # Phase 7 contract: reverse recombines BOTH branches.
    m = _build("MM_20M.xml")
    x = _staged_batch(m)
    with torch.no_grad():
        percepts_in, concepts_in = m.inputSpace.forward(x)
        recon = m.inputSpace.reverse(percepts_in, concepts_in)
    assert recon is not None
