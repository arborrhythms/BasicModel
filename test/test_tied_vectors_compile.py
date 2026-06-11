"""``torch.compile(fullgraph=True)`` must be able to trace a read of
``WordVectors._vectors``.

Regression (the "graph break" fix): the ``_vectors`` property used to
read ``object.__getattribute__(self, "_tied_param_getter")``. Dynamo
cannot trace ``object.__getattribute__`` (gb0156: "Dynamo does not know
how to trace method ``__getattribute__`` of class ``type``"), so *any*
compiled forward that reads ``wv._vectors`` raised
``torch._dynamo.exc.Unsupported`` under ``fullgraph=True``.

The production trigger is IR-mode ``Models.create_ir_mask``:

    null_vec = codebook.getW()[codebook.null_percept_idx]

``codebook.getW()`` (``Spaces.Codebook.getW``) reads ``self.wv._vectors``,
so the IR brick body could not be compiled with ``fullgraph=True`` (it is
compiled unconditionally on CPU/CUDA). ``data/idempotent.xml`` reproduced
the crash end-to-end.

Step 3 (2026-06-10 symbolic-iteration plan) retired the TIED storage path
(``tie_to_codebook``); storage is PS-local permanently. The property
indirection remains precisely for this traceability contract, so the
untied pin stays.
"""
import torch

from embed import WordVectors


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
    """PS-local storage (the permanent layout) read under fullgraph."""
    wv = WordVectors(torch.randn(4, 8), ["a", "b", "c", "d"])
    out = _compiled_read(wv)
    assert out.dim() == 0


def test_tie_api_is_retired():
    """``tie_to_codebook`` is gone; the getter stays permanently None."""
    wv = WordVectors(torch.randn(4, 8), ["a", "b", "c", "d"])
    assert not hasattr(wv, "tie_to_codebook"), (
        "WordVectors.tie_to_codebook was retired (Step 3 of the "
        "2026-06-10 symbolic-iteration plan)")
    assert wv._tied_param_getter is None
