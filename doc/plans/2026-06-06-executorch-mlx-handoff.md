# ExecuTorch / MLX Lowering -- Completion Handoff (Phase D, Tasks D2/D3)

Date: 2026-06-06
Status: handoff. D1 done + verified; D2/D3 scaffolded but UNVERIFIED (executorch
absent in the dev venv at authoring time, so the lowering + runtime parity have
never actually executed).

Note: this doc uses ASCII arrows (`->`) and operators (`<`, `>=`) deliberately --
no Unicode glyphs, so `make doc` / xelatex does not choke (see the LaTeX-not-Unicode
project convention).

## 1. What is DONE and verified (D1)

The BasicModel tensor core is exportable with `torch.export`. `bin/Models.py` exposes:

- `stage_for_core(x) -> staged [B, N, D] tensor` -- the HOST pre-step (lex + chunk +
  embed). Equivalent to what `_begin_step` parks on `self._staged_in_sub`, but
  RETURNED as a plain tensor.
- `forward_core(staged) -> pred tensor` -- the TENSOR-ONLY core (IS-embed already
  done -> PS -> CS -> SS -> OS) that reads the `staged` ARGUMENT (not `self`) and
  returns the head prediction.
- `export_core_module() -> nn.Module` -- a thin `_ForwardCoreModule` wrapper whose
  `forward` IS `forward_core`. Required because torch 2.11's
  `torch.export.export` rejects a bound method (`Expected mod to be an instance of
  nn.Module, got method`).

Verified: `torch.export.export(m.export_core_module(), (staged,))` returns a genuine,
non-vacuous `ExportedProgram` (~2046 graph nodes, 59 distinct aten ops over the real
PS->CS->SS->OS pipeline); `ep.module()(staged)` matches eager `forward_core(staged)`
at max abs diff 0.0. Pinned by `test/test_mlx_export.py::test_forward_core_exports`.

## 2. What is NOT done (D2/D3 -- this handoff)

`executorch` is NOT installed in the dev venv, so:

- `bin/export_mlx.py` runs Phase 1 (torch.export) fine, then exits(2) at the
  executorch probe with a clear message. The Phase-2 lowering has never run.
- `test/test_mlx_export.py::test_mlx_lower_writes_pte` (D2) and
  `::test_pte_runtime_parity` (D3) SKIP cleanly (executorch absent).

So the MLX `.pte` lowering and runtime parity are UNVERIFIED and the import/API
paths below are best-effort -- they MUST be finalized against the installed
executorch + delegate version.

## 3. Environment prerequisite

- torch 2.11.0 (already present; `torch.export` works).
- Install ExecuTorch + the Apple/MLX delegate on an Apple-silicon Mac.
- The dev checkout path contains a SPACE (iCloud). cpu inductor compile dies on that
  space; `torch.export` does NOT use inductor so it is unaffected, but the executorch
  `.pte` build toolchain may also choke on the path -- if it errors on the path, run
  from a space-free checkout.

## 4. The open question: WHICH delegate?

The plan says "MLX delegate / MLXPartitioner". Reality (confirm against the install):

- A dedicated ExecuTorch -> MLX backend/partitioner (`executorch.backends.apple.mlx...`)
  is HYPOTHETICAL -- it may not exist. MLX (Apple's array framework) is separate from
  ExecuTorch; an ExecuTorch->MLX delegate may not ship.
- The shipping Apple backend is MPS:
  `executorch.backends.apple.mps.partition.mps_partitioner.MPSPartitioner`.
- `bin/export_mlx.py::_get_partitioner()` (~line 118) tries the MLX path first and
  falls back to MPS. FINALIZE this: run `pip show executorch`, inspect
  `executorch/backends/apple/`, and either confirm an MLX delegate exists or accept
  MPS as the realistic Apple-accelerated target (and adjust the "MLX" naming intent).

## 5. D2 -- finalize the `.pte` lowering

File: `bin/export_mlx.py` (Phase 2: `_get_partitioner` + `_lower_and_write`).

1. `pip install executorch` (+ the chosen delegate). Confirm `import executorch` works.
2. Run `python bin/export_mlx.py data/MM_20M.xml /tmp/mm20m.pte`. Phase 1 already
   works; Phase 2 now attempts lowering and will surface the first real API mismatch.
3. Fix the partitioner import in `_get_partitioner` (~line 118) to the real module
   path.
4. Confirm the lowering API in `_lower_and_write` (~line 150):
   `to_edge_transform_and_lower(ep, partitioner=[p]).to_executorch().buffer`. The
   script has a legacy `to_edge(ep).to_backend([p])` fallback; keep whichever matches
   the installed executorch.
5. Gate: `test/test_mlx_export.py::test_mlx_lower_writes_pte` should now RUN (not skip)
   and assert the `.pte` exists with size > 0.

## 6. D3 -- finalize runtime parity

File: `test/test_mlx_export.py::test_pte_runtime_parity` (currently a clean skip).

1. Load the `.pte` via the ExecuTorch runtime. Best-guess (finalize against the
   installed bindings): `from executorch.runtime import Runtime` (then `Runtime.get()`
   / `load_program` / `load_method`), OR
   `executorch.extension.pybindings.portable_lib._load_for_executorch(path)`.
2. Run it on the SAME staged input (`staged = m.stage_for_core(x)`), compare to eager
   `m.forward_core(staged)`, assert `max abs diff < 1e-2`.
3. Wrap with the host chunk pre-step so the end-to-end answer matches
   (`stage_for_core` -> `.pte` run -> compare). The autobind/taxonomy host POST-step
   is excluded from the core (it is non-traceable host Python), so compare at the
   `forward_core` boundary, not the full `forward`.
4. Gate: `test_pte_runtime_parity` passes (parity `< 1e-2`).

## 7. Known obstacles / risks

1. EXPORT MUST BE FROM A FRESH MODEL. `torch.export` is clean only BEFORE an eager
   forward runs. A post-forward re-export trips a data-dependent guard
   (`GuardOnDataDependentSymNode`) in a host-side routing finiteness check
   (`bin/Spaces.py`, `_intra_routing_for_predict`'s `if not torch.isfinite(routing).all()`).
   For D2 (build -> stage -> export -> lower, all from fresh) this is fine. If you ever
   need to re-export after a forward, functionalize that branch with
   `torch._check` / `guard_or_false` -- do NOT remove the finiteness check (fail-loud
   policy).
2. FIXED BATCH SIZE. The export is traced for the example batch (B=4, via
   `num_streams=4`). For a variable-batch `.pte`, pass `dynamic_shapes=` to
   `torch.export.export` marking the batch dim dynamic, and confirm the delegate
   supports dynamic shapes (MPS may not -- you may need a fixed B and re-export per
   batch size).
3. PARALLEL CORE ONLY. `forward_core` targets the MM_20M PARALLEL tensor core. The
   serial grammar path (chart / grammar ops, data-dependent control flow) is NOT
   export-friendly; serial models stay host-driven. Do not point `export_mlx.py` at a
   serial config.
4. dtype / AMP. The export traces in the model's eval dtype (fp32 on cpu). For an
   MLX/MPS fp16 target, confirm dtype handling at lowering (the `_begin_step` AMP
   staging is host-side and not in the core).

## 8. Acceptance criteria

- `test/test_mlx_export.py` -- all 5 tests PASS with NO skips (D1 export + D2 lowering
  + D3 parity) on a Mac with executorch + the delegate installed.
- `python bin/export_mlx.py data/MM_20M.xml out.pte` writes a non-empty `.pte`.
- Runtime parity `< 1e-2` vs eager `forward_core`.

## 9. References

- `bin/Models.py`: `stage_for_core`, `forward_core`, `export_core_module`,
  `_ForwardCoreModule`, `_forward_per_stage(..., in_sub_override=...)`,
  `_stem_subspace_from_tensor`.
- `bin/export_mlx.py`: the lowering script (Phase 1 torch.export + Phase 2 executorch).
- `test/test_mlx_export.py`: the 5 gates (3 pass now; D2/D3 skip until executorch).
- Plan: `doc/plans/2026-06-06-dimensional-governance-completion.md` Phase D.
