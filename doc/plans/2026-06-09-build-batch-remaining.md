# 2026-06-09 build batch -- remaining work (granular)

Status: **active execution plan.** Landed this session: **#11** (regression
head) verified; **#13** (PS-codebook retire / SS-quantize default) APPLIED
but left **4 regressions** (the subagent was killed before it verified the
full suite). Suite is at **13 failed / 2131 passed** (9 deferred IS-shape +
4 from #13). Each task below returns the suite to the **9-failed** baseline
(regression-verified; the **D** semantic validation is deferred).

Design cross-refs: `2026-06-09-asymmetric-vq-symbolic-ss.md` ($\S$4 routing,
$\S$5 combiner, $\S$6 semantic; $\S$7 tasks 7--14). Decisions:
`2026-06-08-mm20m-architecture-backlog.md` $\S$1 (#11), $\S$2 (#13).

Verification (every task): sibling venv --
`BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager PYTHONPATH=bin "<WikiOracle/.venv
python>" -m pytest test/ -q` (the tests; `make test`'s report writer also
needs `PIL` in `basicmodel/.venv`). Each task: suite returns to **9 failed**
(no NEW failures), `test_basicmodel` green, XOR still solves + reconstructs.

## Landed
- **#11** -- config-honoured regression head + `<readout>` knob
  (`MM_20M` $\to$ `sigmoid`; default `identity` is byte-identical to the old
  bare-linear head). Verified; `test_basicmodel` 206.
- **#13** -- PS `<codebook>` removed (PS $=$ subsymbolic `none`); SS
  `quantize` default. **APPLIED, NOT clean** -- 4 regressions (Task 0).

## Task 0 -- stabilize #13 (the 4 regressions)
PS-codebook-removal side-effects. Resolve so PS $=$ `none` does not break the
paths that relied on a PS codebook surface:
- `test_basicmodel.py::TestReconstructionSymbols::test_forward_output_shape_unchanged`
- `test_explicit_dimensions.py::TestIdempotentCliRuns::test_cli_does_not_crash`
- `test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct`
- `test_ir_fullgraph_compile.py::test_idempotent_forward_compiles_fullgraph_eager`

Likely a config/reader fix (XOR_exact still needs a percept surface for
reconstruction; the forward output shape must be preserved). **User
hypothesis to test first:** some of these may be *resolved by Task 1 / Task 3*
(parallel quantize / the new combiner) rather than a standalone patch -- run
those and re-check before hand-fixing.
Acceptance: suite back to **9 failed**; `test_basicmodel` green.

## Task 1 -- C-8: parallel SS quantize -- **LANDED (2026-06-09, uncommitted)**
`Codebook.quantize()` genuinely fires in the parallel path: the stage-0
unity consumption (`_stage0_unity_forward`) builds its evidence at the
CARRIER width (`W = nDim`, the codebook row width), snaps it through the
SS codebook, and pools the SNAPPED carrier down to the narrow combine
code. Pinned by `test_parallel_ss_quantize_fires` (call-counting -- the
0-call probe, codified). Pre-snap $z$ + selected indices are threaded as
forward-locals (`_stage0_z_pre_snap` / `_stage0_indices`) for the recon
gather. Discovered en route: the parallel no-op had TWO layers -- the
single-stage empty-seed early-return AND the non-empty leg's early
`nOutputDim` trim, which would have destroyed the 1024-wide carrier the
codebook needs (8 vs 1024). **Deferred to the Task-3 leg** (no shipping
config exercises the non-empty parallel SS leg): retarget the two trims
from `nOutputDim` to `nDim` and rewrite the SS snap dispatch
(reversible/nonreversible branches) to the asymmetric contract. The
wide(W)$\leftrightarrow$narrow(muxed content) symbol-code definition is
an open design point owned by Task 3.

## Task 2 -- C-9 + C-11/12: asymmetric STE routing, drop commitment + EMA
**PARTIALLY LANDED (2026-06-09, uncommitted):**
- ~~Remove the standard commitment loss and the EMA codebook update for
  the SS VQ~~ -- DONE at the VQ level: `vq.ema_update = False` (new
  master gate on `VectorQuantize`; default True preserves PS/CS) and
  `vq.commitment_weight = 0.0`, hardwired for every SS stage in
  `BaseModel` construction. Pinned by `test_ss_vq_asymmetric_flags`
  (codebook bit-stable across a training-mode forward). The SPACE-level
  `commitmentBeta` branch (the `reversible` dispatch) remains live until
  the dispatch rewrite (Task-3 leg).
- Forward / output STE, $z_q = z_e + \mathrm{sg}(e[\mathrm{idx}] - z_e)$
  -- DONE on the stage-0 path (gradient $\to$ the evidence/upstream;
  code detached).
- Reverse / recon plain gather $z_q = e[\mathrm{idx}]$ (gradient $\to$
  codebook) -- **LANDED (2026-06-10, uncommitted)**: the SS VQ codebook
  is promoted to an `nn.Parameter` (enlisted in the optimised params)
  and the stage-0 snap emits `ss_codebook_recon` -- a plain
  differentiable gather of the SELECTED rows against the DETACHED
  evidence, lifted onto the pipeline-chained error container by
  `_forward_body`. Pinned by `test_ss_codebook_recon_gradient`
  (gradient support == exactly the selected rows; encoder leg blocked)
  and `test_ss_recon_term_reaches_pipeline_errors`. The codebook now
  trains by the exact input-faithfulness gradient -- the EMA
  replacement. Keep the $\pm 1$ operating range (evidence is
  tanh-squashed).

## Task 3 -- C-10: 2-stream ILL combiner -- **LANDED (2026-06-10, uncommitted)**
`ConceptualCombine` is the 2-stream SLOT bind (geometry corrected by
Alec, 2026-06-10): the streams stack along the VECTOR axis ($N$ slots
each, $2N$ total) and ONE cascade runs over the flattened
$2N \cdot D = 16\,n\mathrm{Dim} = 2^{14}$ slab (cross-slot reach, zero
pad; $D$ = the full muxed event width -- option B, the band
participates). Purely linear; exact reverse $\mathrm{ILL}^{-1}
(\mathrm{CS}) \to (\mathrm{PS}, \mathrm{SS})$ with NOTHING threaded
alongside -- the whole augment machinery (`aug_dim`, `reverse_dropped`,
`_combine_augments`) is deleted. The PS / SS VIEWS are the bind's
slot-halves (`combine.views`: vectors $0..N{-}1$ / $N..2N{-}1$ of the
STORED mix -- the row-windows a Phase-5/6 `.active` index view
expresses literally; views-of-the-bind keeps the ILL in the head's
gradient path, where views-of-the-inverse would cancel it to zero).
The threaded `_combine_carriers` are the FULL binds (the recurrent /
reconstruction state -- no rank shed, contraction gone at the source).
`_reverse_body` inverts each stage's bind exactly and surfaces the PS
stream (the percept leg) for the input decode. **Glue (decided by Alec,
2026-06-10, superseding the interim sum):** the views are literally
STACKED (2N vectors) and CS glues them through the learned
**$[2N, N]$ corpus-callosum matrix** over the slot axis
(`ConceptualCombine.callosum`, init = averaging the two hemispheres;
128 params at 16x8). When `OutputSpace` eventually accepts both views,
it consumes the stack directly. **Containment (processing contract,
2026-06-10):** the whole bind calculation lives on
`ConceptualSpace.bind_streams` / `unbind` (the body loop orchestrates
only), and the carrier rides ON the SubSpaces (`_bind_carrier`); the
reverse prefers the SubSpace-carried bind. The A6 interSentence seed
primes the SYMBOLIC side (added into $SS_t$ at stage 0) since the CS
stream is retired. Round-trip + XOR regression green (suite 2152
passed / 0 failed).
**Accumulated scope, updated:**
- ~~the asymmetric RECON gather~~ -- LANDED (see Task 2 above);
- ~~the SS snap-dispatch rewrite~~ -- **LANDED for the PARALLEL leg
  (2026-06-10, uncommitted)**: the reversible/nonreversible branches
  drop the nearest-target pull (`symbol_residual`), the space-level
  commitment, and the cb_commit emission -- the legacy single-objective
  crutches -- and the reversible branch NAMES via a no-grad snap
  (`_naming_indices`) when widths agree and EMA cannot fire. The
  SERIAL/grammar leg keeps the legacy coupling behind an explicit
  ``syntacticLayer is not None`` gate -- without EMA/commitment/residual
  its codebook is fully orphaned (the stage-0 recon gather is
  parallel-only) and `test_mm_grammar_learns_xor_signal` dies; the
  serial asymmetric recon leg is D-territory. The two
  `nOutputDim`$\to$`nDim` trim retargets in the non-empty parallel SS
  leg remain open (no shipping config exercises that leg);
- the wide$\leftrightarrow$narrow symbol-code definition (today:
  per-slot adaptive pooling of the snapped carrier at SS stage 0; the
  sum-of-halves head view at the bind) -- design point owned by
  Phase 5/6;
- **#13 second half** (SS `<codebook>` knob removal / quantize
  hardwire): **CANCELLED as knob-removal (Alec, 2026-06-10) — the tag
  STAYS as the subsymbolic-vs-symbolic iteration switch, applied to
  the CS→SS leg; see the dual-input plan's "`<codebook>` knob STAYS"
  section (which also schedules the `insert_paired_word` retirement
  and the MM_20M 4-D second-order acceptance).** The mechanism work
  below remains valid and load-bearing for the quantize mode.
  Original status: **MECHANISM FOUND + OUTPUT COLLAPSE FIXED
  (2026-06-10, uncommitted).** The trained
  collapse had TWO independent components, now separated by a
  variant-bisection ladder (full table in the dual-input plan's Phase-5
  section, rev. 2026-06-10):
  (a) **OUTPUT collapse -- FIXED.** `Codebook.quantize`'s per-call
  `replace_W(self.vq.codebook)` (an EMA-mode prototype refresh)
  re-pointed the space-owned basis `W` at the VQ's gradient-orphaned
  random matrix under the asymmetric hardwire, severing the trained
  prototypes from every `W` consumer; XOR_exact's output loss sat at
  the constant-predictor floor for all 600 epochs. Fix: the refresh is
  GATED on `vq.ema_update` (Spaces.py). With everything else ON
  (recon gather, adopt-on-first-sight, roles, naming), XOR_exact's
  OUTPUT now learns under `<codebook>quantize</codebook>`
  (output 0.0010; predictions $-0.01/0.96/0.83/-0.02$). Suite 2156/0.
  The earlier "three hypotheses eliminated" chain stands; the
  additionally acquitted: recon-term competition, optimizer param
  append, adoption, roles, naming, RNG/init shift, ``_active``
  override (stays None), grad flow through ``set_event``.
  (b) **Recon-TEXT breakage -- FIXED (2026-06-10 late); GATE PASSED.**
  Root cause: the tied-storage layout (PS lexicon rows live in the SS
  codebook ``W`` as orth+semantic PAIRS via ``insert_paired_word``)
  remaps ``key_to_index[word] -> orth_idx`` but leaves ``index_to_key``
  in insertion order; ``decode_reverse_meta`` rendered rows
  POSITIONALLY, so every word decoded as the string at list position =
  its row index. Fixed at the decode: row->word via the INVERSE of
  ``key_to_index`` + nearest-row search restricted to mapped rows.
  **Acceptance: XOR_exact with SS quantize = 4/4 OK, byte-exact text +
  XOR predictions; suite 2156/0.** The #13b hardwire (schema removal,
  config strips, reader default) is now UNBLOCKED -- land it with the
  known full recipe; Task 4 unblocks behind it. Caveat to carry: the
  shipping ``none`` fixture's recon shows unseeded run-to-run
  flakiness (see the dual-input plan Phase-5 caveat); seed before
  attributing. Full mechanism narrative: dual-input plan, Phase-5
  section (rev. 2026-06-10).

## Task 4 -- BPE/MPHF: shared byte codebook -- **CORE LANDED (2026-06-10, uncommitted)**
Share the byte / `PerceptStore` codebook across chunking front ends so a
`bpe`/`mphf` chunk decodes through the SAME byte codebook on the reverse path
that `radix` uses (segmentation becomes a strategy over one shared
embedding/codebook, not a private table). See
`2026-06-08-mm20m-architecture-backlog.md` $\S$4.

**Landed (suite 2162/0):** bpe/mphf construct the `PerceptStore`
(RadixLayer with its own Codebook basis; `W` Parameter-promoted, the
radix recipe -- the lexicon Embedding keeps `subspace.what` as the
word-synthesis surface). The ChunkLayer vocabulary MIRRORS into the
store: full sync at wire time (after the BPE auto-load), incremental on
every `train_step` promotion and artifact `load`; insertion in chunk-id
order + RadixLayer's sequential allocation give **`percept_id ==
chunk_id`**, ASSERTED at every mirror site (fail loud -- the store must
be dedicated). Byte identity resolves through the new
`ChunkLayer.bytes_for` -> `store.bytes_for` (the same reverse surface
radix uses); `id_to_bytes` is DEMOTED to the segmentation-side mirror
(trie building, GPU tables, artifact format unchanged) and the unwired
fallback. Capacity: the vocab caps at `n_vectors` (>= 256 enforced for
bpe), so the store never grows past its Parameter -- alignment safe by
construction. Pinned by `TestSharedByteStore` (6 tests: construction +
alignment both modes, Parameter permanence, promotion mirroring,
smoke-prompt round-trip through the store, unwired fallback).

**Follow-up (the hard cut, optional):** delete `id_to_bytes` and point
ALL segmentation-side readers (trie `_ensure_trie`, the embed walkers,
`bpe_to_lexicon_keys`, the GPU static tables, the artifact writer) at
the store. Mechanical but wide (the artifact schema keeps
`id_to_bytes` as its serialization either way); do it when touching
those readers anyway.

## Task 5 -- C-13: semantic arrangement -- **MECHANISM LANDED (2026-06-10, uncommitted)**
`SymbolicSpace.semantic_arrangement_loss(indices)`: pode $=$ the
activated rows' centroid (attraction), antipode $=$ the rest of the
codebook's centroid (repulsion); both poles DETACHED so the gradient
lands only on the active rows. Gated by the new
`<semanticArrangement>` weight on SymbolicSpace (default 0 $=$ OFF);
computed post-snap at stage 0 and lifted onto the pipeline error
container as `ss_semantic_arrangement` alongside the recon term.
Pinned by `test_semantic_arrangement_mechanism` (off-by-default;
gradient support $=$ exactly the activated rows). The semantic PAYOFF
is deliberately unasserted -- that is **D**'s corpus gate ($\S$8: XOR
cannot validate the semantic side). The single-snap vs sparse
multi-symbol code question ($\S$6) stays open.

## Deferred
- **D** -- corpus validation on the serial / grammar path. XOR cannot
  validate the symbolic side (`asymmetric-vq` $\S$8: parallel $=$ no VQ; parity
  $\neq$ semantics); needs a real corpus and present supervision.
- **`[4,-1,5]` IS-shape bug** -- RESOLVED 2026-06-09 (commit e85bc1f): a
  widening PS now sizes its fold + forwardBegin reshape at the embedded
  percept width (`_pi_width`); suite at 0 failed. The structural follow-on
  lives in `2026-06-08-analysis-synthesis-dual-input.md` (rev. 2026-06-09,
  orientation corrected): the IS feed becomes a dual view, IS $\to$ PS as
  `[B,N,1]` (atoms, bottom-up synthesis), IS $\to$ SS as `[B,1,N]` (unity,
  top-down analysis).

## Suggested sequencing
0 (or fold into 1/3) $\to$ 1 $\to$ 2 $\to$ 3 $\to$ 4 $\to$ 5. Commit per task
for granularity; keep the suite at 9-failed after each.
