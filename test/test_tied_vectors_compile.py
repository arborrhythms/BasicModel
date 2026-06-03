"""``torch.compile(fullgraph=True)`` must be able to trace a read of
``WordVectors._vectors``.

Regression (the "graph break" fix): the tied-storage ``_vectors`` property
read ``object.__getattribute__(self, "_tied_param_getter")``. Dynamo cannot
trace ``object.__getattribute__`` (gb0156: "Dynamo does not know how to
trace method ``__getattribute__`` of class ``type``"), so *any* compiled
forward that reads ``wv._vectors`` raised ``torch._dynamo.exc.Unsupported``
under ``fullgraph=True``.

The production trigger is IR-mode ``Models.create_ir_mask``:

    null_vec = codebook.getW()[codebook.null_percept_idx]

``codebook.getW()`` (``Spaces.Codebook.getW``) reads ``self.wv._vectors``,
so the IR brick body could not be compiled with ``fullgraph=True`` (it is
compiled unconditionally on CPU/CUDA). ``data/idempotent.xml`` reproduced
the crash end-to-end.

These tests pin the property to stay trace-able for both the tied and the
untied storage path.
"""
import torch
import torch.nn as nn

from embed import WordVectors


class _StubCodebook(nn.Module):
    """Minimal tie target: exposes ``getW()`` returning a Parameter,
    mirroring ``Spaces.Codebook`` from the property's point of view."""

    def __init__(self, n, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, d))

    def getW(self):
        return self.W


def _compiled_read(wv):
    """Compile (fullgraph) a tensor op that reads ``wv._vectors``.

    Returns the scalar result; raises ``torch._dynamo.exc.Unsupported``
    if the property cannot be traced.
    """
    torch._dynamo.reset()

    # ``backend="eager"`` keeps the test on the Dynamo TRACE path (where the
    # graph break lived) without invoking the Inductor C++ toolchain -- the
    # latter is unrelated to this regression and is itself broken in repo
    # checkouts whose path contains a space (iCloud "Mobile Documents").
    # ``fullgraph=True`` still turns any graph break into a hard error.
    @torch.compile(fullgraph=True, backend="eager")
    def f(x):
        return (wv._vectors * x).sum()

    return f(torch.ones(4, 8))


def test_untied_vectors_compiles_fullgraph():
    """Untied storage (local Parameter) read under fullgraph compile."""
    wv = WordVectors(torch.randn(4, 8), ["a", "b", "c", "d"])
    out = _compiled_read(wv)
    assert out.dim() == 0


def test_tied_vectors_compiles_fullgraph():
    """Tied storage (routes through ``codebook.getW()``) under fullgraph."""
    wv = WordVectors(torch.randn(4, 8), ["a", "b", "c", "d"])
    wv.tie_to_codebook(_StubCodebook(4, 8))
    out = _compiled_read(wv)
    assert out.dim() == 0
