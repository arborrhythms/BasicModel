"""WholeSpace property codebook -- ``.what`` rows as whole-ranging properties.

Phase 3 of the part/whole refactor: on WholeSpace a ``.what`` codebook row
is a PROPERTY (e.g. a sinusoid value) rather than a character atom. When
``materialize`` hands the per-position selection (``_index``) to a codebook
via ``mode="property"``, it produces a per-position region membership over
the whole input -- the region that HAS the property (``> 0``) and the
region that does NOT (``<= 0``). The read is opted into per call by the
``mode`` argument (there is no flag); every other read stays on the
unchanged ``lookup`` path (doc/plans/2026-07-10-wholes-are-types-
segmentation.md, T1).
"""

import os
import sys
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch
import torch.nn as nn

import Spaces
from Spaces import Codebook, SubSpace, WholeSpace


def _property_codebook(V=6, D=8, seed=0):
    # A "property" codebook is just a codebook read via mode="property";
    # there is no flag -- the mode argument IS the per-call opt-in
    # (doc/plans/2026-07-10-wholes-are-types-segmentation.md, T1).
    torch.manual_seed(seed)
    cb = Codebook()
    cb.W = nn.Parameter(torch.randn(V, D))
    return cb


def test_materialize_property_shape_and_range():
    cb = _property_codebook()
    idx = torch.tensor([[0, 1, 2, 3, 4]])          # [B=1, N=5]
    region = cb.materialize_property(idx, n_positions=5)
    assert region.shape == (1, 5, 5)               # [..., n_positions]
    assert torch.all(region >= -1.0) and torch.all(region <= 1.0)


def test_property_segments_into_have_and_havenot():
    # A property must be able to carve the input: at least one position with
    # the property (> 0) and at least one without (<= 0) across the field.
    cb = _property_codebook(seed=3)
    idx = torch.arange(6).view(1, 6)
    region = cb.materialize_property(idx, n_positions=16)
    assert (region > 0).any(), "no has-property region produced"
    assert (region <= 0).any(), "no complement region produced"


def test_property_region_is_low_frequency():
    # A property ranges coarsely over the whole input: the region signal must
    # be low-frequency (few sign changes), not character-frequency oscillation.
    # The basis tops out at ~half cycles over the field, so zero crossings are
    # bounded by ~2*half regardless of how long the input is.
    D = 8
    half = D // 2
    cb = _property_codebook(V=4, D=D, seed=5)
    n = 128
    region = cb.materialize_property(torch.zeros(1, 1, dtype=torch.long), n_positions=n)
    sig = region[0, 0]                                  # [n]
    crossings = int((sig[1:] * sig[:-1] < 0).sum())
    assert crossings <= 2 * half + 2, f"too many sign changes ({crossings}) for low-freq property"


def test_materialize_property_is_differentiable():
    cb = _property_codebook(seed=1)
    idx = torch.zeros(1, 4, dtype=torch.long)
    region = cb.materialize_property(idx, n_positions=4)
    region.sum().backward()
    assert cb.W.grad is not None and torch.isfinite(cb.W.grad).all()


def test_subspace_routes_property_mode_to_codebook():
    D = 8
    sub = SubSpace([4, D], [4, D], nInputDim=D, nOutputDim=D)
    sub.what = _property_codebook(V=4, D=D, seed=2)
    sub.set_index(torch.tensor([[0, 1, 2, 3]]).unsqueeze(-1))   # [B,N,M=1]
    region = sub.materialize(mode="property")
    assert region is not None
    assert region.shape[-1] == 4                  # n_positions from selection


def test_subspace_property_mode_routes_any_codebook():
    # There is no property flag: mode="property" is the per-call opt-in, so
    # ANY .what codebook is read as a property when the mode is requested
    # (doc/plans/2026-07-10-wholes-are-types-segmentation.md, T1).
    D = 8
    sub = SubSpace([4, D], [4, D], nInputDim=D, nOutputDim=D)
    cb = Codebook()
    cb.W = nn.Parameter(torch.randn(4, D))        # a plain codebook
    sub.what = cb
    sub.set_index(torch.tensor([[0, 1, 2, 3]]).unsqueeze(-1))
    region = sub.materialize(mode="property")
    assert region is not None
    assert region.shape[-1] == 4                  # n_positions from selection


def test_subspace_property_mode_none_without_index():
    # The only remaining None conditions: no .what, or no selection.
    D = 8
    sub = SubSpace([4, D], [4, D], nInputDim=D, nOutputDim=D)
    sub.what = _property_codebook(V=4, D=D, seed=2)
    assert sub.materialize(mode="property") is None   # no _index set


def test_property_codebook_public_reads_share_membership_view(monkeypatch):
    """No public row-read surface may expose a property's raw Parameter.

    Optimizer state is allowed to drift outside the unit cube between the
    update and its post-step projection.  ``getW``, full prototypes, selected
    lookups, and sinusoidal property materialization must nevertheless observe
    the same UNORM-STE membership values.
    """
    monkeypatch.setattr(Spaces, "meronomy_enabled", lambda: True)
    raw = torch.tensor([
        [-2.0, 0.25, 2.0, 0.75],
        [1.50, -0.50, 0.40, 0.60],
        [0.10, 0.20, 0.30, 0.40],
    ])
    cb = Codebook()
    cb.is_percept_store = True
    cb.W = nn.Parameter(raw.clone())
    expected = raw.clamp(0.0, 1.0)

    assert torch.allclose(cb.getW(), expected)
    assert torch.allclose(cb.prototype(), expected)
    idx = torch.tensor([[1, 0, 1]])
    assert torch.allclose(cb.lookup_rows(idx), expected[idx])
    assert torch.allclose(cb.lookup(idx), expected[idx])

    # The whole-ranging sinusoidal reader is downstream of ``lookup``.  A
    # control codebook holding the already-clamped rows therefore has to be
    # exactly equivalent to the marked store with out-of-range raw storage.
    control = Codebook()
    control.W = nn.Parameter(expected.clone())
    got = cb.materialize_property(idx, n_positions=7)
    want = control.materialize_property(idx, n_positions=7)
    assert torch.allclose(got, want)


def test_canonical_ws_stage_composes_transformed_property_rows(monkeypatch):
    """Stage-0 property mixing reads memberships, never raw coefficients."""
    monkeypatch.setattr(Spaces, "meronomy_enabled", lambda: True)
    P, D, N = 8, 6, 2
    cb = Codebook()
    cb.is_percept_store = True
    cb.W = nn.Parameter(torch.full((P, D), -2.0))
    stage = SimpleNamespace(
        property_basis=True,
        subspace=SimpleNamespace(what=cb, muxedSize=D),
        inputShape=(N, D),
        nWhere=0,
        nWhen=0,
        nDim=D,
    )
    input_bytes = torch.tensor([[[65, 49]]])       # capital letter + digit
    carrier, width = WholeSpace._stage0_carrier(stage, input_bytes, None)

    # Both bytes activate canonical property predicates, so this would be a
    # negative carrier if stage-0 reached into raw W.  The transformed rows
    # are all zero and therefore so is their mixture.
    assert stage._stage0_property_membership.sum() > 0
    assert width == D
    assert torch.equal(carrier, torch.zeros_like(carrier))
