"""Direct grammar/decode searches must ignore inactive codebook reserve."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


class _PrefixCodebook:
    """Small duck-typed stand-in for Codebook's active read contract."""

    def __init__(self, rows, active):
        self.W = torch.nn.Parameter(torch.as_tensor(rows, dtype=torch.float32))
        self._active = int(active)

    def getW(self):
        return self.W

    def active_row_count(self):
        return self._active

    def active_prototypes(self):
        return self.W[:self._active]


class _PerceptStore:
    def __init__(self, rows, active):
        self.codebook = torch.as_tensor(rows, dtype=torch.float32)
        self._size = int(active)






def test_radix_ws_decode_searches_bound_active_intersection_with_remap():
    from Layers import RadixLayer

    radix = RadixLayer(dim=2, initial_cap=4)
    radix.insert(b"active", init_vector=torch.tensor([0.0, 1.0]))
    ws_cb = _PrefixCodebook(
        [
            [0.8, 0.1],  # active + bound: the only decodable candidate
            [1.0, 0.0],  # active + bound but no META path: must not win
            [0.2, 0.2],
            [1.0, 0.0],  # bound but inactive: must not win
        ],
        active=2,
    )

    class _WholeSpace:
        subspace = SimpleNamespace(what=ws_cb)
        _ws_row_to_pos = {0: 10, 1: 20, 3: 30}
        _pos_kind = {11: "ps", 31: "ps"}
        _ps_pos_to_row = {11: 0, 31: 0}

        @staticmethod
        def taxonomy_children(pos):
            # Position 20 is a plain bound WS atom: no children and no META
            # parent. Although its vector is an exact query match, selecting it
            # would emit b"" and truncate the word.
            return {10: [11], 20: [], 30: [31]}.get(int(pos), [])

        @staticmethod
        def taxonomy_parent(pos):
            return None

        @staticmethod
        def is_meta(pos):
            return False

    assert radix.reverse(
        torch.tensor([1.0, 0.0]), symbolic_space=_WholeSpace()) == b"active"


def test_chunk_reverse_and_peel_preserve_active_prefix_row_ids():
    from Language import SynthesizeLayer

    cb = _PrefixCodebook(
        [[0.8, 0.1], [0.0, 1.0], [0.2, 0.2], [1.0, 0.0]], active=2)
    whole = torch.tensor([1.0, 0.0])
    chunk = SynthesizeLayer(nInput=2, nOutput=2)

    part, _residual = chunk.analyze(whole, basis=cb)
    assert torch.equal(part, cb.active_prototypes()[0])

    # An explicit set containing only a reserved id has no real candidate.
    part, residual = chunk.analyze(
        whole, basis=cb, left_rows=torch.tensor([3]))
    assert torch.equal(part, whole)
    assert torch.equal(residual, torch.zeros_like(whole))

    with torch.no_grad():
        parts, _residual = chunk.peel(whole, cb, max_parts=1)
    assert parts and parts[0][0] == 0


def test_impenetrable_trust_slices_physical_ema_to_active_prefix():
    from Layers import ImpenetrableLayer

    active_rows = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    basis = SimpleNamespace(
        vq=SimpleNamespace(cluster_size=torch.tensor([1.0, 3.0, 99.0, 99.0])))

    trust = ImpenetrableLayer()._trust(active_rows, basis)

    assert torch.allclose(trust, torch.tensor([0.25, 0.75]))
