"""Lower the BasicModel tensor core to an ExecuTorch .pte file (Tasks D2/D3).

Usage
-----
    python bin/export_mlx.py data/MM_20M.xml out.pte

The script has TWO clearly separated phases:

  Phase 1 — torch.export (runs here, executorch NOT required)
    Build the model, stage one batch, call ``torch.export.export`` on the
    ``ExportCore`` adapter (defined in this script) whose ``forward`` IS
    ``forward_core(model, .)``.  The export adapter lives HERE, not on
    ``BasicModel`` -- the model keeps a single forward path; the only model
    surface this needs is the ``_forward_per_stage(in_sub_override=...)`` hook.

  Phase 2 — ExecuTorch lowering (requires executorch + MLX/Apple delegate)
    Lower the ExportedProgram to .pte via ``to_edge_transform_and_lower``
    and write the binary.  ALL executorch imports are inside a try/except
    so an absent executorch produces a clear message + exit(2) (the test
    recognises exit(2) as "skip", NOT a hard failure).

PARTITIONER IMPORT NOTE
-----------------------
This script was authored with ``executorch`` ABSENT from the venv, so the
import paths below are best-effort and MUST be finalised against the
installed executorch + delegate version before the lowering can actually
run.  The strategy is:

  1. Try the dedicated MLX delegate first (hypothetical path
     ``executorch.backends.apple.mlx``).  An "MLXPartitioner" may exist
     in a future executorch release or an Apple-side package.
  2. Fall back to the shipping Apple MPS delegate
     (``executorch.backends.apple.mps.partition.mps_partitioner.MPSPartitioner``).
  3. If neither is available, exit(2) with a clear message.

The torch.export portion (Phase 1) is FULLY EXERCISABLE without executorch.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

# ---------------------------------------------------------------------------
# Resolve bin/ onto sys.path so we can import Models, Language, util
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Fix KMP duplicate-lib issue common on macOS (harmless on Linux).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Export BasicModel tensor core to ExecuTorch .pte")
    p.add_argument("model_xml",
                   help="Path to the model XML config (e.g. data/MM_20M.xml)")
    p.add_argument("output_pte",
                   help="Destination path for the .pte binary")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Phase 1 helpers (no executorch needed)
# ---------------------------------------------------------------------------

def _build_model(model_xml: str):
    """Load and eval the model from *model_xml*. Returns the BasicModel."""
    import Models
    import Language
    from util import init_config

    _REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
    defaults = os.path.join(_REPO_ROOT, "data", "model.xml")
    # init_config expects a path WITHOUT the .xml suffix (it appends it).
    # Pass the raw path — init_config handles both forms.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        Language.TheGrammar._configured = False
        init_config(path=model_xml, defaults_path=defaults)
        m, _ = Models.BasicModel.from_config(model_xml)
    m.eval()
    return m


def stage_for_core(model, x):
    """HOST pre-step: lex + chunk + embed ``x`` -> staged ``[B, N, D]`` tensor.

    Reuses the model's NORMAL staging (start spaces + ``inputSpace.forward``)
    and parks the stem shell on ``model._staged_in_sub`` for ``forward_core``
    to rebind the export argument into. The export concern lives HERE in the
    (opt-in, speculative) script rather than on ``BasicModel`` -- the model
    keeps ONE forward path; the only model surface this needs is the
    ``_forward_per_stage(in_sub_override=...)`` injection hook.
    """
    import torch
    if isinstance(x, torch.Tensor):
        try:
            x = x.to(next(model.parameters()).device)
        except StopIteration:
            pass
    model._start_spaces_for_forward()
    model._spaces_started_for_forward = True
    # Eager stem: lex (IS) -> embed (PS) -> finalize bookkeeping, so the
    # host-side tokenization runs OUTSIDE the exported tensor core
    # (2026-06-07 eager-embed-stage).
    in_sub = model._lex_embed_stem(x)
    model._staged_in_sub = in_sub
    # Pure tensor core: park an empty inter-sentence seed so the in-trace read
    # is a pure attr read (no predictor call inside the core).
    model._staged_intersentence_seed = None
    model._intersentence_seed_staged = True
    return in_sub.materialize() if in_sub is not None else None


def forward_core(model, staged):
    """TENSOR-ONLY core: rebind ``staged`` into the parked stem shell and run
    the SAME ``_forward_per_stage`` the normal forward uses (via the
    ``in_sub_override`` hook). Returns the head prediction tensor (index 2).

    The random IR mask (``create_ir_mask``'s BERT-style ``torch.bernoulli``
    hide-a-token) is DISABLED for the core: it is a training / ``infer()``
    infill corruption, and the exported core is the DEPLOYMENT inference
    graph -- a baked-in bernoulli would make every .pte call nondeterministic
    and parity untestable. Zeroing ``mask_rate`` here also keeps the
    ``bernoulli`` node out of the ``torch.export`` trace (``create_ir_mask``
    early-returns on ``rate <= 0``). Save/restore so the live model's
    training-path masking is untouched."""
    in_sub = model._staged_in_sub
    in_sub.set_event(staged)
    _saved_mask_rate = model.mask_rate
    model.mask_rate = 0.0
    try:
        out = model._forward_per_stage(None, in_sub_override=in_sub)
    finally:
        model.mask_rate = _saved_mask_rate
    return out[2] if out is not None else None


def _get_staged_input(m):
    """Load the xor dataset, pull one batch, return the staged [B,N,D] tensor."""
    import Models
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    staged = stage_for_core(m, x)
    return staged


def _run_torch_export(m, staged):
    """Phase 1: trace the tensor core with torch.export. Returns ExportedProgram."""
    import torch

    class ExportCore(torch.nn.Module):
        """``nn.Module`` whose ``forward`` IS ``forward_core(model, .)`` --
        torch 2.11's ``torch.export.export`` wants an ``nn.Module``, not a
        bound method. Holds the model by reference (it owns every Parameter
        the core reads; no re-registration)."""

        def __init__(self, model):
            super().__init__()
            object.__setattr__(self, "_model", model)

        def forward(self, staged):
            return forward_core(self._model, staged)

    print("[export_mlx] Phase 1 — torch.export.export ...", flush=True)
    ep = torch.export.export(ExportCore(m), (staged,))
    print(f"[export_mlx] torch.export OK: {ep}", flush=True)
    return ep


# ---------------------------------------------------------------------------
# Phase 2 helpers (executorch required — ALL imports guarded)
# ---------------------------------------------------------------------------

def _get_partitioner():
    """Return an ExecuTorch Apple-Silicon partitioner, preferring the MLX delegate.

    Finalised against executorch 1.3.1 (installed here; it BUNDLES the MLX
    backend + ``mlx.metallib`` runtime):
      * MLX is the purpose-built Apple-Silicon-GPU delegate
        (``executorch.backends.mlx.partitioner.MLXPartitioner``), built for this
        executorch / torch 2.12 stack -- the right default. Requires Apple
        Silicon + the Metal compiler (ships with Xcode).
      * CoreML (``CoreMLPartitioner``) is a fallback, but its coremltools 9.0
        crashes converting torch-2.12 graphs (optimize_linear MIL pass).
      * MPS is DEPRECATED (removed in executorch 1.4); last resort only.

    ``EXPORT_MLX_BACKEND=mlx|coreml|mps`` forces one backend; default tries
    MLX -> CoreML -> MPS. Returns (partitioner_instance, name_str).
    """
    forced = os.environ.get("EXPORT_MLX_BACKEND", "").strip().lower()

    def _mlx():
        from executorch.backends.mlx.partitioner import MLXPartitioner  # type: ignore[import]
        return MLXPartitioner(), "MLXPartitioner"

    def _coreml():
        from executorch.backends.apple.coreml.partition import CoreMLPartitioner  # type: ignore[import]
        return CoreMLPartitioner(), "CoreMLPartitioner"

    def _mps():
        from executorch.backends.apple.mps.partition.mps_partitioner import (  # type: ignore[import]
            MPSPartitioner,
        )
        from executorch.exir.backend.compile_spec_schema import CompileSpec  # type: ignore[import]
        # MPS (deprecated) requires compile_specs and ships no helper; pass a
        # minimal fp16=off spec to match the fp32 export.
        specs = [CompileSpec("use_fp16", bytes([0]))]
        return MPSPartitioner(compile_specs=specs), "MPSPartitioner (deprecated MPS fallback)"

    order = {"mlx": [_mlx], "coreml": [_coreml], "mps": [_mps]}.get(
        forced, [_mlx, _coreml, _mps])
    last_exc = None
    for _fn in order:
        try:
            return _fn()
        except (ImportError, ModuleNotFoundError) as exc:
            last_exc = exc

    raise ImportError(
        "No ExecuTorch Apple partitioner importable (tried "
        "executorch.backends.mlx / apple.coreml / apple.mps)."
        + (f" Last error: {last_exc}" if last_exc else "")
    )


def _lower_and_write(ep, output_pte: str) -> None:
    """Phase 2: lower ExportedProgram to .pte and write to *output_pte*.

    ALL executorch imports are inside this function so their absence never
    crashes Phase 1.
    """
    try:
        # ExecuTorch edge-transform + lower API (executorch >= 0.3).
        from executorch.exir import to_edge_transform_and_lower  # type: ignore[import]
    except (ImportError, ModuleNotFoundError):
        # Older executorch used a two-step API; try that as a fallback.
        try:
            from executorch.exir import to_edge  # type: ignore[import]
            _LEGACY_API = True
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError("executorch.exir not importable") from exc
    else:
        _LEGACY_API = False

    partitioner, p_name = _get_partitioner()
    print(f"[export_mlx] Phase 2 — lowering with {p_name} ...", flush=True)

    if not _LEGACY_API:
        edge_prog = to_edge_transform_and_lower(ep, partitioner=[partitioner])
    else:
        # Legacy two-step: to_edge -> .to_backend -> .to_executorch
        from executorch.exir.backend.backend_api import to_backend  # type: ignore[import]
        edge_prog = to_edge(ep).to_backend([partitioner])

    et_prog = edge_prog.to_executorch()
    pte_buf: bytes = et_prog.buffer

    os.makedirs(os.path.dirname(os.path.abspath(output_pte)) or ".", exist_ok=True)
    with open(output_pte, "wb") as fh:
        fh.write(pte_buf)

    size_kb = len(pte_buf) / 1024
    print(f"[export_mlx] wrote {output_pte} ({size_kb:.1f} KB)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)

    # -----------------------------------------------------------------------
    # Phase 1 — always runs (torch.export only, no executorch needed)
    # -----------------------------------------------------------------------
    m = _build_model(args.model_xml)
    staged = _get_staged_input(m)
    ep = _run_torch_export(m, staged)

    # -----------------------------------------------------------------------
    # Phase 2 — executorch lowering (guarded)
    # -----------------------------------------------------------------------
    try:
        import executorch  # noqa: F401  — presence probe
    except (ImportError, ModuleNotFoundError):
        print(
            "[export_mlx] executorch not installed; cannot lower to .pte — "
            "install executorch + the MLX/Apple delegate and re-run.",
            file=sys.stderr,
        )
        sys.exit(2)  # exit(2) = "skip" signal (not a hard failure)

    try:
        _lower_and_write(ep, args.output_pte)
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            f"[export_mlx] executorch import error during lowering: {exc}\n"
            "Ensure executorch and the MLX/Apple delegate are installed and "
            "that the import paths in export_mlx.py match the installed version.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
