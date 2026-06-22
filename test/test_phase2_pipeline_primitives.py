"""Unit tests for Phase 2 Pipeline primitives: ReverseAdapter, CachePoint."""
import sys
from pathlib import Path

# basicmodel/bin must come before bin/ so that basicmodel's parse.py is found
# first (both directories contain a parse.py with different APIs).  Insert in
# reverse order so basicmodel/bin ends up at index 0.
_project = Path(__file__).resolve().parent.parent           # basicmodel/
_wo_root = _project.parent                                   # WikiOracle/
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch

from Models import ReverseAdapter, CachePoint


# --- Test helpers ---

class _IdentitySpace:
    """Minimal stand-in with forward/reverse returning the same subspace."""

    def __init__(self):
        self.forward_calls = 0
        self.reverse_calls = 0

    def forward(self, subspace):
        self.forward_calls += 1
        return subspace

    def reverse(self, subspace):
        self.reverse_calls += 1
        return subspace


class _FakeSymbols:
    """Stand-in subspace with materialize() + set_event() + is_empty()."""
    def __init__(self, tensor):
        self._t = tensor
    def materialize(self):
        return self._t
    def set_event(self, t, compute_activation=False):
        self._t = t
    def set_muxed(self, t):
        self._t = t
    def is_empty(self):
        return self._t.numel() == 0


# --- ReverseAdapter ---

def test_reverse_adapter_calls_reverse_not_forward():
    inner = _IdentitySpace()
    adapter = ReverseAdapter(inner)
    sentinel = object()
    out = adapter.forward(sentinel)
    assert out is sentinel
    assert inner.reverse_calls == 1
    assert inner.forward_calls == 0


def test_reverse_adapter_no_own_parameters():
    inner = torch.nn.Linear(2, 2)
    adapter = ReverseAdapter(inner)
    # The only parameters should be inner's.
    own = [p for n, p in adapter.named_parameters() if not n.startswith("wrapped.")]
    assert own == [], "ReverseAdapter must not introduce new learnable parameters"


# --- CachePoint ---

def test_cache_point_stashes_and_returns_identity():
    cp = CachePoint()
    x = torch.tensor([1.0, 2.0, 3.0])
    out = cp(x)
    assert out is x
    assert cp.last is x


def test_cache_point_updates_on_each_call():
    cp = CachePoint()
    a, b = torch.tensor([1.0]), torch.tensor([2.0])
    cp(a)
    cp(b)
    assert cp.last is b


from Models import GrammarMergeGlue  # noqa: E402


# --- GrammarMergeGlue ---

def test_grammar_merge_glue_averages_pairs():
    x = torch.randn(2, 8, 4)
    ws = _FakeSymbols(x.clone())
    g = GrammarMergeGlue(stage_idx=0, initial_n=8, is_last=False)
    g.forward(ws)
    y = ws.materialize()
    assert y.shape == (2, 4, 4)
    expected = (x[:, 0::2, :] + x[:, 1::2, :]) / 2
    assert torch.allclose(y, expected, atol=1e-6)


def test_grammar_merge_glue_is_last_identity():
    x = torch.randn(2, 8, 4)
    ws = _FakeSymbols(x.clone())
    g = GrammarMergeGlue(stage_idx=0, initial_n=8, is_last=True)
    g.forward(ws)
    assert torch.equal(ws.materialize(), x)


def test_grammar_merge_glue_roundtrip_via_diff_cache():
    x = torch.randn(2, 8, 4)
    ws = _FakeSymbols(x.clone())
    g = GrammarMergeGlue(stage_idx=0, initial_n=8, is_last=False)
    g.forward(ws)
    g.reverse(ws)
    assert torch.allclose(ws.materialize(), x, atol=1e-6)


def test_grammar_merge_glue_empty_passthrough():
    ws = _FakeSymbols(torch.zeros(0, 0, 0))
    g = GrammarMergeGlue(stage_idx=0, initial_n=8, is_last=False)
    out = g.forward(ws)
    assert out is ws


def test_space_forward_arities():
    """Each Space's forward() exposes its post-2026-05 reconciliation arity.

    Cross-space combination moved out of ``_sourced_input``/``*_ref``
    into explicit ``forward`` arguments supplied by the recurrent cell:
      * PartSpace.forward(x_subspace)        -- Stage 1.A single-arg
      * ConceptualSpace.forward(PS_subspace, SS_subspace=None)
      * WholeSpace.forward(CS_subspaceForWS, IS_concepts=None)
        -- the optional stage-0 UNITY input (analysis/synthesis
        dual-input plan, rev. 2026-06-09)
      * InputSpace / ModalSpace / OutputSpace    -- single arg
        (InputSpace now RETURNS the dual view tuple; its argument
        arity is unchanged)
    The optional second args (CS, SS) default to ``None`` so standalone
    single-arg callers still work.

    Stage 1.A substrate refactor (doc/plans/
    2026-05-26-two-loop-pi-sigma-substrate.md): PartSpace.forward
    is single-arg now. The body composes ``pi(x) + sigma(x)`` on the
    same materialized input; the legacy two-input
    ``tanh(pi_input(IS) + pi_concept(C_prev))`` contract (with
    CS_subspaceForPS as the second arg) is retired.

    Post-2026-05-21 SentenceState dissolution: the per-sentence ``work``
    carrier was retired. Grammar / serial-processing state lives directly
    on ``SymbolicSubSpace`` (cursor, recur_pass) and is reached via
    ``subspace.symbolSpace``. Forwards take ONLY data SubSpaces.
    """
    import inspect
    from Spaces import (InputSpace, PartSpace, ModalSpace,
                        ConceptualSpace, WholeSpace, OutputSpace)
    expected = {
        InputSpace: (1, 1),
        PartSpace: (1, 1),   # x_subspace -- Stage 1.A single-arg
        ModalSpace: (1, 1),
        ConceptualSpace: (1, 2),   # PS_subspace req, SS_subspace opt
        WholeSpace: (1, 2),     # CS_subspaceForWS req, IS_concepts opt
        OutputSpace: (1, 1),
    }
    for cls, (min_req, max_params) in expected.items():
        sig = inspect.signature(cls.forward)
        params = [p for name, p in sig.parameters.items() if name != "self"]
        required = [p for p in params
                    if p.default is inspect.Parameter.empty
                    and p.kind in (p.POSITIONAL_ONLY,
                                   p.POSITIONAL_OR_KEYWORD)]
        assert len(params) == max_params, (
            f"{cls.__name__}.forward has {len(params)} content params "
            f"(expected {max_params}): {list(sig.parameters)}")
        assert len(required) == min_req, (
            f"{cls.__name__}.forward has {len(required)} required "
            f"(expected {min_req}): {list(sig.parameters)}")
        assert "work" not in sig.parameters, (
            f"{cls.__name__}.forward must not carry the retired Phase-1 "
            f"'work' carrier: {list(sig.parameters)}")


def test_all_spaces_have_single_arg_reverse():
    """Each Space.reverse() takes exactly one content arg (the subspace).

    Post-2026-05-21 SentenceState dissolution: the ``work`` carrier is
    gone from reverse too. Reverse generates its own reconstructed
    estimates of intermediate SubSpaces (reconstruction contract); it
    never reads forward bookkeeping.
    """
    import inspect
    from Spaces import InputSpace, PartSpace, ModalSpace, ConceptualSpace, WholeSpace, OutputSpace
    for cls in (InputSpace, PartSpace, ModalSpace, ConceptualSpace, WholeSpace, OutputSpace):
        sig = inspect.signature(cls.reverse)
        params = [p for name, p in sig.parameters.items() if name != "self"]
        assert len(params) == 1, (
            f"{cls.__name__}.reverse has {len(params)} content params "
            f"(expected 1): {list(sig.parameters)}")
        assert "work" not in sig.parameters, (
            f"{cls.__name__}.reverse must not carry the retired Phase-1 "
            f"'work' carrier: {list(sig.parameters)}")


# --- build_pipelines() smoke tests ---

def _make_mm_xor_model():
    """Helper: load MM_xor config with xor data and return a BasicModel."""
    import os
    from data import TheData
    from Models import BaseModel
    from util import ProjectPaths
    config_path = os.path.join(
        ProjectPaths.PROJECT_DIR, "basicmodel", "data", "MM_xor.xml")
    TheData.load("xor")
    model, _ = BaseModel.from_config(config_path, data=TheData)
    return model


def test_build_pipelines_creates_body_stages():
    """build_pipelines() constructs body_stages as an nn.ModuleList of ModuleDicts.

    Replaces the prior pipeline_fwd nn.Sequential check. After the
    2026-05-11 module consolidation, the entire pipeline is method-
    based: ``_forward_stem`` / ``_forward_body`` / ``_forward_head``
    plus the per-stage ``body_stages`` ModuleList. FlattenKWrapper
    is gone.
    """
    import torch.nn as nn_
    model = _make_mm_xor_model()
    assert isinstance(model.body_stages, nn_.ModuleList)
    assert len(model.body_stages) == model.subsymbolicOrder
    for stage in model.body_stages:
        assert isinstance(stage, nn_.ModuleDict)
        assert "cs" in stage and "ws" in stage
    # Pipeline boundaries are methods, not attributes. ``_forward_stem``
    # stays retired (the IR forward inlines InputSpace+PartSpace
    # directly into ``_forward_per_stage``); ``reverse()`` reconstructs input
    # from the terminal ConceptualSpace state (§5 of the recurrent-cell plan).
    # The head-seeded ``_run_pipeline_rev`` primitive was removed 2026-06-07.
    assert callable(getattr(model, '_forward_body', None))
    assert callable(getattr(model, '_forward_head', None))
    assert not hasattr(model, '_forward_stem')
    assert callable(getattr(model, 'reverse', None))
    assert not hasattr(model, '_run_pipeline_rev')
