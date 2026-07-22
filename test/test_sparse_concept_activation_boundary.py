"""Sparse native-fold -> ConceptualSpace codebook activation boundary.

The PS and WS recursions remain in their own 128-wide WHAT coordinates.  A
staged concept-row identity plus a scalar activation is the only thing that
crosses into the 1024-wide conceptual WHAT coordinates; the codebook row is
the dimensional increase.  These tests deliberately expose no native feature
vector to the decoder, so a coordinate projection cannot satisfy the contract.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Spaces import ConceptualSpace, Space  # noqa: E402


class _LookupOnlyCodebook(nn.Module):
    """Tiny codebook spy that makes a full-inventory read fail loudly."""

    def __init__(self, rows):
        super().__init__()
        self._rows = nn.Parameter(rows.clone())
        self.lookup_calls = 0
        self.lookup_sizes = []

    @property
    def W(self):  # pragma: no cover - reached only by an incorrect decoder
        raise AssertionError("sparse decode must not read the full codebook")

    def getW(self):  # pragma: no cover - reached only by an incorrect decoder
        raise AssertionError("sparse decode must not read the full codebook")

    def prototype(self):  # pragma: no cover - incorrect full-inventory read
        raise AssertionError("sparse decode must not read the full codebook")

    def active_prototypes(self):  # pragma: no cover - incorrect scan surface
        raise AssertionError("sparse decode must not scan active prototypes")

    def lookup_rows(self, indices):
        self.lookup_calls += 1
        self.lookup_sizes.append(int(indices.numel()))
        return self._rows[indices.long()]

    # ``Codebook.lookup`` is also a valid O(K) indexed read surface.  Keep the
    # alias so the test constrains complexity rather than one method spelling.
    lookup = lookup_rows


def _bare_cs(rows):
    """Construct only the state used by ``decode_sparse_concept_rows``."""
    cs = ConceptualSpace.__new__(ConceptualSpace)
    nn.Module.__init__(cs)
    cs.nWhat = 1024
    cs.nWhere = 4
    cs.nWhen = 4
    cs.concept_dim = 1032
    cs.inputShape = [8, 1032]
    cs.outputShape = [8, 1032]
    cs.similarity_codebook = _LookupOnlyCodebook(rows)
    return cs


def _decode(cs, rows, activations, band):
    """Name the proposed sparse boundary once for easy API adjustment."""
    return cs.decode_sparse_concept_rows(rows, activations, band)


def test_sparse_decode_gathers_exact_known_rows_and_preserves_band():
    codebook = torch.zeros(4, 1032)
    codebook[1, :1024] = torch.linspace(-1.0, 1.0, 1024)
    codebook[2, 700] = 3.25
    # Poison the codebook's trailing columns: WHERE/WHEN comes from the source
    # event and must never be replaced by learned concept-row coordinates.
    codebook[:, 1024:] = 99.0
    cs = _bare_cs(codebook)

    rows = torch.tensor([[2, 1]], dtype=torch.long)
    activation = torch.ones(1, 1, 2)
    band = torch.arange(16, dtype=torch.float32).reshape(1, 1, 2, 8)
    decoded = _decode(cs, rows, activation, band)

    assert decoded.shape == (1, 1, 2, 1032)
    assert torch.equal(decoded[0, 0, 0, :1024], codebook[2, :1024])
    assert torch.equal(decoded[0, 0, 1, :1024], codebook[1, :1024])
    assert torch.equal(decoded[..., 1024:], band)


def test_unknown_row_is_zero_what_and_never_aliases_row_zero():
    codebook = torch.zeros(2, 1032)
    codebook[0, :1024] = 7.0
    cs = _bare_cs(codebook)

    rows = torch.tensor([[-1, 0]], dtype=torch.long)
    activation = torch.ones(1, 1, 2)
    band = torch.randn(1, 1, 2, 8)
    decoded = _decode(cs, rows, activation, band)

    assert torch.equal(decoded[0, 0, 0, :1024], torch.zeros(1024))
    assert not torch.equal(
        decoded[0, 0, 0, :1024], codebook[0, :1024])
    assert torch.equal(decoded[0, 0, 1, :1024], codebook[0, :1024])
    # Unknown identity suppresses conceptual WHAT, not its fixed location/time.
    assert torch.equal(decoded[..., 1024:], band)


def test_six_fold_scalars_all_receive_gradient_without_coordinate_projection():
    codebook = torch.zeros(3, 1032)
    # Coordinate 900 is well outside either native 128-wide perceptual space.
    # Reaching it exactly proves that the dimensional increase is a concept-row
    # decode, not padding/repeating/projecting the native feature coordinates.
    codebook[2, 900] = 4.0
    cs = _bare_cs(codebook)

    rows = torch.tensor([[2]], dtype=torch.long)
    activation = torch.tensor(
        [[[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]],
        requires_grad=True,
    )
    band = torch.zeros(1, 6, 1, 8)
    decoded = _decode(cs, rows, activation, band)

    assert decoded.shape == (1, 6, 1, 1032)
    assert torch.allclose(
        decoded[0, :, 0, 900], activation[0, :, 0] * 4.0)
    decoded[..., 900].sum().backward()
    assert activation.grad is not None
    assert torch.equal(activation.grad, torch.full_like(activation, 4.0))


def test_sparse_decode_cost_tracks_selected_rows_not_inventory_width():
    # A deliberately nontrivial inventory size makes accidental full reads
    # visible through the lookup-only spy, while the requested work remains
    # exactly B*N regardless of the six source activations sharing each row.
    codebook = torch.randn(257, 1032)
    cs = _bare_cs(codebook)
    rows = torch.tensor([[3, 200], [17, -1]], dtype=torch.long)
    activation = torch.ones(2, 6, 2)
    band = torch.zeros(2, 6, 2, 8)

    decoded = _decode(cs, rows, activation, band)
    assert decoded.shape == (2, 6, 2, 1032)
    assert cs.similarity_codebook.lookup_calls == 1
    assert cs.similarity_codebook.lookup_sizes == [rows.numel()]


def test_source_native_fold_activation_is_normalized_before_concept_decode():
    """Native width cannot become an accidental activation-strength prior."""
    small = torch.ones(1, 2, 8)
    wide = torch.ones(1, 2, 128)

    assert not hasattr(ConceptualSpace, "native_fold_presence")
    a_small = Space.native_fold_activation(small, 8)
    a_wide = Space.native_fold_activation(wide, 128)

    expected = torch.ones(1, 2)
    assert torch.allclose(a_small, expected)
    assert torch.allclose(a_wide, expected)


def test_membership_activations_decode_to_signed_bounded_concepts():
    torch.manual_seed(19)
    codebook = torch.nn.functional.normalize(
        torch.randn(7, 1032), dim=-1)
    cs = _bare_cs(codebook)
    rows = torch.tensor([[1, 5, 3]], dtype=torch.long)
    activation = torch.rand(1, 6, 3)
    band = torch.zeros(1, 6, 3, 8)

    decoded = _decode(cs, rows, activation, band)

    what = decoded[..., :1024]
    assert bool((what >= -1.0).all())
    assert bool((what <= 1.0).all())
