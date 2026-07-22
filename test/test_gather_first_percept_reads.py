"""Large radix stores must transform only selected/occupied percept rows."""

from __future__ import annotations

import os
import sys
import warnings
from types import SimpleNamespace

import torch


os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


def _percept_basis(rows):
    from Spaces import Codebook

    rows = torch.as_tensor(rows, dtype=torch.float32)
    basis = Codebook()
    basis.create(
        nInput=1,
        nVectors=int(rows.shape[0]),
        nDim=int(rows.shape[1]),
        customVQ=False,
        monotonic=True,
    )
    basis.setW(torch.nn.Parameter(rows.clone()))
    basis.is_percept_store = True
    return basis


def _forbid_full_read(monkeypatch, basis):
    def fail(*_args, **_kwargs):
        raise AssertionError("full percept Codebook.getW() read")

    monkeypatch.setattr(basis, "getW", fail)


def test_word_local_radix_gathers_before_unorm_with_identical_gradient(
        monkeypatch):
    import Spaces
    from Spaces import PartSpace

    monkeypatch.setattr(Spaces, "meronomy_enabled", lambda: True)
    rows = torch.tensor([
        [-2.0, 0.25, 3.0],
        [0.10, 0.20, 0.30],
        [4.00, -5.0, 0.75],
        [0.90, 0.80, 0.70],
        [0.40, 0.50, 0.60],
        [0.15, 0.35, 0.55],
    ])
    basis = _percept_basis(rows)
    ps = object.__new__(PartSpace)
    torch.nn.Module.__init__(ps)
    ps.subspace = SimpleNamespace(
        what=basis,
        nWhat=3,
        nWhere=0,
        nWhen=0,
        whereEncoding=SimpleNamespace(nDim=0),
    )
    ps._meronomy_words = False
    ps._radix_where_indices = []
    ids = torch.tensor([[1, 4, 1], [5, 4, 3]], dtype=torch.long)

    _forbid_full_read(monkeypatch, basis)
    event = ps._radix_part_events(ids)

    assert torch.equal(event.detach(), rows[ids].clamp(0.0, 1.0))
    event.sum().backward()
    expected_counts = torch.bincount(ids.reshape(-1), minlength=len(rows))
    expected_grad = expected_counts.to(rows.dtype).unsqueeze(-1).expand_as(rows)
    # The UNORM clamp is straight-through, so gather-first preserves the exact
    # selected-row gradient, including coordinates outside [0, 1].
    assert torch.equal(basis.W.grad, expected_grad)


def test_slot_radix_embedding_gathers_without_full_codebook_read(monkeypatch):
    """Cover the non-word-major ``_embed_radix`` gather site as well."""
    import Language
    import Models
    from util import init_config

    cfg = os.path.join(_PROJECT, "data", "MM_xor.xml")
    defaults = os.path.join(_PROJECT, "data", "model.xml")
    init_config(path=cfg, defaults_path=defaults)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = Models.BasicModel.from_config(cfg)
    ps = model.perceptualSpace

    class _Upstream:
        _host_tokens = [["hello", "world"]]

        @staticmethod
        def materialize(mode=None):
            return torch.zeros(1, 8, 1, dtype=torch.long)

    _forbid_full_read(monkeypatch, ps.subspace.what)
    out = ps._embed_radix(_Upstream())
    assert out.materialize().shape[:2] == (1, ps.outputShape[0])


def test_radix_active_searches_do_not_read_physical_reserve(monkeypatch):
    import Spaces
    from Layers import RadixLayer

    monkeypatch.setattr(Spaces, "meronomy_enabled", lambda: True)
    store = RadixLayer(dim=2, initial_cap=16)
    store.insert(b"left", init_vector=torch.tensor([0.9, 0.1]))
    store.insert(b"right", init_vector=torch.tensor([0.1, 0.9]))
    store._basis.is_percept_store = True
    _forbid_full_read(monkeypatch, store._basis)

    active = store.active_prototypes()
    assert active.shape == (2, 2)
    assert store.associate_span(torch.tensor([0.9, 0.1])) == 0
    assert store.reverse(torch.tensor([0.9, 0.1])) == b"left"


def test_symbolize_uses_radix_occupied_prefix_without_full_read(monkeypatch):
    import Spaces
    from Language import SymbolizeLayer
    from Layers import RadixLayer

    monkeypatch.setattr(Spaces, "meronomy_enabled", lambda: True)
    store = RadixLayer(dim=2, initial_cap=16)
    store.insert(b"word", init_vector=torch.tensor([0.9, 0.1]))
    store._basis.is_percept_store = True
    _forbid_full_read(monkeypatch, store._basis)

    ws_rows = torch.nn.Parameter(torch.tensor([[0.1, 0.9], [0.8, 0.2]]))

    class _WholeSpace:
        def __init__(self):
            self.subspace = SimpleNamespace(
                what=SimpleNamespace(active_prototypes=lambda: ws_rows))
            self._ws_pos_to_row = {}
            self.ps_row = None

        def ensure_ps_position(self, row):
            self.ps_row = int(row)
            return 10 + int(row)

        @staticmethod
        def ensure_ws_position(row, kind="ws"):
            return 20 + int(row)

        def insert_meta(self, ps_pos, ws_pos, fused_vec=None):
            self._ws_pos_to_row[30] = 0
            return 30

        @staticmethod
        def record_lbg_pull(ws_pos, vec):
            return None

        @staticmethod
        def maybe_split_lbg(ws_pos):
            return None

    ws = _WholeSpace()
    layer = SymbolizeLayer(
        nInput=2,
        nOutput=2,
        wholeSpace=ws,
        perceptualSpace=SimpleNamespace(percept_store=store),
    )

    layer.forward(torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9]))
    assert ws.ps_row == 0


def test_post_step_projection_is_embedding_only(monkeypatch):
    from Models import BasicModel
    from Spaces import Codebook, Embedding

    codebook = Codebook()

    def fail():
        raise AssertionError("radix Codebook.normalize() must not run")

    monkeypatch.setattr(codebook, "normalize", fail)
    radix_owner = SimpleNamespace(
        perceptualSpace=SimpleNamespace(
            subspace=SimpleNamespace(what=codebook)))
    BasicModel._normalize_perceptual_embedding(radix_owner)

    embedding = Embedding()
    calls = []
    monkeypatch.setattr(embedding, "normalize", lambda: calls.append(True))
    lexicon_owner = SimpleNamespace(
        perceptualSpace=SimpleNamespace(
            subspace=SimpleNamespace(what=embedding)))
    BasicModel._normalize_perceptual_embedding(lexicon_owner)
    assert calls == [True]
