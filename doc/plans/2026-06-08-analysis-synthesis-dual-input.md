# Analysis / Synthesis Dual-Input Plan

Date: 2026-06-08
Revised: **2026-06-09 — orientation corrected.** The original draft assigned
analysis to the perceptual side and synthesis to the symbolic side; that was
backwards. This revision supersedes the draft's side-assignments throughout.
Status: Active execution plan

## Orientation (the corrected through-line)

```
SymbolicSpace (SS)                  PerceptualSpace (PS)
------------------                  --------------------
TOP-DOWN                            BOTTOM-UP
ANALYSIS                            SYNTHESIS
Pi  -- product, intersection        Sigma -- sum, union
starts from UNITY                   starts from ATOMS
  concepts_in: [B, 1, N]              percepts_in: [B, N, 1]
Universals: meaning/term            Particulars: specifically
generalities (spyi-mtshan,          characterized entities
don-spyi, sgra-spyi)                (rang-mtshan)
front ends: analyse (meronymic)     front ends: the chunkers
  + the lexers                        (radix, bpe, byte)
codebook of generalities; the       content codebook with a snap
divisions snap to it                distance (the percept store);
                                    the eidetic ground
```

**Analysis (SS, top-down).** Analysis starts with the input as one undivided
unity — `[B, 1, N]`, a single event of width N — and divides it: lexer cuts,
meronymic part–whole divisions, aspects. Each analytic step is an
**intersection**: adding a constraint narrows, the way universals compose
(RED ∩ BALL). The divisions reference the shared `.where`/`.when` coordinate
frame, exist as **boundaries in the data** (like an edge), and **do not alter
the data**. They **snap to the SS codebook**. Analytic characterization is
**large-scale and approximate** — mean value over a region, extent/support
size, low-pass features: coarse descriptions of large perceptual pieces. A
coarse one-over-many description is precisely a meaning-flavored generality;
that is why analysis is the symbolic side.

**Synthesis (PS, bottom-up).** Synthesis starts with the input as N atoms —
`[B, N, 1]`, N events of width 1; bytes as particulars — and **unions** them
into larger wholes: chunks, recurring surface forms, words. Synthesis also
does not alter the data: it **represents chunk content in a codebook with a
snap distance** (the radix/percept store), so a recurring particular is
recognized as the *same* percept row, and byte-exact reconstruction continues
to flow through the percept ids. An exact this-one surface form is a
specifically characterized particular; that is why synthesis is the
perceptual side.

**Duality on reverse.** Each branch's reverse runs its dual operation:
`PS.reverse` un-unions (aggregate → atoms); `SS.reverse` un-divides (parts →
unity). Model-level reconstruction therefore combines a bottom-up content
stream with a top-down scaffold stream (§6).

**Why this is the correct orientation.** It restores coherence with two
standing commitments the draft contradicted:

1. the Buddhist mapping (§9) already grounded `rang-mtshan` (particulars) in
   PS and the generalities (`spyi-mtshan`/`don-spyi`/`sgra-spyi`) in SS;
2. the descriptor codebook (§7) already held the LF/coarse descriptor rows —
   which are analysis outputs — in SS.

It also leaves the working XOR recipe intact (`asymmetric-vq` §1: radix +
1-vector SBOW on PS), since the chunkers stay put.

## Goal

Reformulate the two loops as an explicit analysis/synthesis pair:

- **dual view**: `InputSpace.forward()` emits the same source input under two
  views — an atom view for the perceptual branch and a unity view for the
  symbolic branch.
- **recurrent integration**: `PerceptualSpace` and `SymbolicSpace` both
  receive input-derived evidence, then the existing C-tier recurrence
  combines percepts, symbols, and concepts.
- **reconstruction**: reverse recombines the perceptual and symbolic branches
  back into the input surface with an `InvertibleLinearLayer`.
- **operators**: Pi (analysis/intersection) and Sigma (synthesis/union) sit
  on the correct sides — **they swap** relative to today's code.
- **configuration**: the perceptual `<chunking>` knob becomes `<synthesis>`
  (union modes); a new SS-side `<analysis>` knob hosts `analyse` and the
  lexer modes.

The immediate shape contract:

```text
InputSpace.forward(input) -> (percepts_in, concepts_in)

percepts_in: [B, N, 1]   # atoms  -> PS (bottom-up synthesis)
concepts_in: [B, 1, N]   # unity  -> SS (top-down analysis)
```

## Current Context

The current forward path is:

```text
_lex_embed_stem:
  InputSpace.forward(x) -> in_sub          (single SubSpace)
  PerceptualSpace.embed_stem(in_sub)       (chunk+embed, host-eager)
  InputSpace.finalize_stem(in_sub, ps)

_forward_body:
  PS_sub_stage0 = PerceptualSpace.forward(in_sub)
  for each stage:
    SS_sub = SymbolicSpace.forward(prevCS_forSS)
    CS_sub = ConceptualSpace.forward(contribution, SS_sub)
    ConceptualCombine(PS_t, SS_t, CS_t)
```

Standing contracts this plan changes or relies on:

- `InputSpace` is a pure raw lexer returning ONE SubSpace. It becomes the
  dual-view source; **lexer ownership moves to SS** (lexing = analytic
  cutting).
- `PerceptualSpace` owns chunking/embedding after the 2026-06-07
  peer-removal. **It keeps them** — the chunkers are synthesis. The percept
  store remains the authoritative surface-content codebook (and the shared
  decode surface for backlog §4's BPE/MPHF byte-codebook sharing).
- **PS currently owns `self.pi` (PiLayer); SS currently owns `self.sigma`
  (SigmaLayer). These swap** (§3): Pi is analysis (SS); Sigma is synthesis
  (PS).
- `SymbolicSpace.forward()` currently takes only `CS_subspaceForSS`; it gains
  the stage-0 unity input (§2).
- `ConceptualCombine` currently mixes three streams; the 2-stream
  `CS = ILL([PS ‖ SS])` change is tracked separately
  (`2026-06-09-asymmetric-vq-symbolic-ss.md` §5, task C-10) and composes with
  this plan: feed better PS/SS streams into whichever combiner is live.
- Reverse reconstruction currently seeds from concepts, walks the body
  reverse, then `PerceptualSpace.reverse()` and `InputSpace.reverse()`.

## Design

### 1. Split InputSpace Output (dual view)

```python
percepts_in, concepts_in = inputSpace.forward(inputData)
```

- `percepts_in` is the **atom view** `[B, N, 1]`: the raw elements as N
  separate width-1 events. The PS front ends (chunkers) consume atoms
  directly — note the radix already learns its own chunking from a raw
  stream (`asymmetric-vq` §1), so no lexer pre-cut is required on this side.
- `concepts_in` is the **unity view** `[B, 1, N]`: the whole presentation as
  one width-N event, the thing analysis divides.
- Both views retain metadata to reconstruct source ordering, dtype, and
  padding/null mask. They are VIEWS of one buffer, not copies, wherever
  possible.
- Tuple contract stays the public shape; a compatibility shim may live only
  inside model orchestration during migration, never as a silent permanent
  fallback.

### 2. Add Input to SymbolicSpace (stage-0 unity)

```python
SymbolicSpace.forward(CS_subspaceForSS, IS_concepts=None)
```

- Stage 0 reads `IS_concepts` (the unity view).
- Later stages read the recurrent `CS_subspaceForSS`.
- The contribution is composed INSIDE SymbolicSpace (additive/compositional);
  `ConceptualCombine` continues to see exactly one `SS_t` stream.
- Repeated symbolic input injection (stages > 0) is a later knob; the first
  implementation reads input once, mirroring PS.

`_forward_body` becomes:

```text
percepts_in, concepts_in = staged dual view from InputSpace
PS_sub_stage0 = perceptualSpace.forward(percepts_in)

for t:
  if t == 0:
    SS_sub = ss.forward(prevCS_forSS, concepts_in)
  else:
    SS_sub = ss.forward(prevCS_forSS)
  CS_sub = cs.forward(contribution, SS_sub)
  ConceptualCombine(PS_t, SS_t, CS_t)
```

### 3. Pi / Sigma Swap

Today: `PerceptualSpace.pi = PiLayer(...)` (Spaces.py ~8241) and
`SymbolicSpace.sigma = SigmaLayer(...)` (Spaces.py ~12685). Corrected:

- **PS owns Sigma** (`SigmaLayer`): synthesis, sum/union — the bottom-up
  fold over atoms/percepts.
- **SS owns Pi** (`PiLayer`): analysis, product/intersection — the top-down
  fold over the unity/carrier.

Implementation notes:

- The span/sizing machinery transfers as-is: the global `<sigmaPi>` mode
  (last | butterfly | full) drives both folds unchanged; the embedded-width
  sizing rule (`_pi_width`, 2026-06-09 — a widening space sizes its fold at
  the embedded `nOutputDim`, not the raw `nInputDim`) follows the fold to its
  new owner (rename to `_fold_width`).
- `ConceptualSpace.sigma_in` (the per-stage fold Stage 10 migrated out of
  PS): review its name/role during this phase — it was named for the OLD
  orientation. **Open decision; do not rename blindly.**
- **Checkpoint policy — DECIDED (2026-06-09, Phase 3 commit): fresh-start.**
  `ps.pi.*` / `ss.sigma.*` parameter keys swap to `ps.sigma.*` / `ss.pi.*`
  with NO state_dict migration: no production checkpoints exist (tests build
  fresh; local `output/` artifacts are regenerable). A checkpoint saved
  before the swap will fail loudly on load, which is the desired behavior.

### 4. Top-Down Analysis on SS (analyse + the lexers)

Move the analytic front ends to SymbolicSpace:

- **Lexer ownership IS → SS**: lexing is analytic cutting (a boundary
  decision over the unity). The `<lexer>` knob moves from the InputSpace
  config section to SymbolicSpace; InputSpace emits the dual views only.
- **`analyse` (the meronymic analyzer) moves PS → SS**: part–whole division
  is top-down analysis.
- Analytic outputs: boundaries/divisions in the shared `.where`/`.when`
  frame + large-scale measurements over the resulting parts (mean, extent,
  low-pass). Non-altering: the unity buffer is never rewritten.
- The divisions snap to the SS codebook (§7): a division is identified as a
  kind (a generality), and the coarse measurements ride with it.
- Host-side tokenization stays OUT of the compiled graph: the SS-side lexing
  gets the same eager-stem treatment `embed_stem` pioneered (an eager
  pre-step staging the cut metadata before the compiled body). No per-token
  host sync in the training path; report-only metadata stays lazy.

### 5. Bottom-Up Synthesis on PS (the chunkers)

- **radix, bpe, byte stay on PS** — they union atoms into chunks/surface
  forms. The percept store remains the authoritative surface-content
  codebook, with its snap distance recognizing recurring particulars.
- **lexicon and mphf are strictly codebooks** (they neither analyze nor
  synthesize): they stay parked on PS short-term, flagged to move later —
  likely toward SS as content-codebook front ends, since naming is
  representational. Decide when they actually block something.
- Byte-exact reconstruction continues to flow through the percept ids
  (`asymmetric-vq` §2: PS carries lossless reconstruction); the SS-side
  reverse decodes surface forms THROUGH this shared PS store (backlog §4).
- The PS fold becomes `sigma` (§3); the chunk+embed pipeline itself is
  unchanged by this plan beyond the knob rename (§8).

### 6. Reconstruction: Recombining Both Branches

Reverse reconstruction synthesizes the original input from both branches:

```python
InputSpace.reverse(percepts_recon, concepts_recon)
```

Recombination layer:

```text
flatten(percepts_recon): [B, N]    (bottom-up content stream)
flatten(concepts_recon): [B, N]    (top-down scaffold stream)
concat:                  [B, 2N]
InvertibleLinearLayer:   [B, 2N] -> [B, N]  (approximate)
                      or [B, 2N] -> [B, 2N] (exact, augment-threaded)
```

Open engineering choice (pick ONE explicitly and test it; do not hide
exactness assumptions in comments):

- rectangular `ILL(2N, N)` — approximate, matches the output width;
- square `ILL(2N, 2N)` + threaded/zeroed N-wide augment — exact, mirroring
  `ConceptualCombine`.

### 7. SS Descriptor Codebook (generalities)

One representational codebook in SymbolicSpace, holding GENERALITIES:

- rows carry kind/role metadata: meaning-general (`don-spyi`), term-general
  (`sgra-spyi`), and LF/coarse characterizations (frequency, mean, low-pass,
  region statistic) — the natural outputs of top-down analysis;
- `.active` selects/weights rows; `.where`/`.when` place the selected
  generality over its extension; materialization is
  `.what[active] @ (.where, .when)`.

**Redistribution from the draft**: the draft held HF *and* LF rows in one SS
codebook. Corrected: **HF surface contents (pixel/byte/glyph-grade rows) are
PS percept-store rows** (particulars); SS keeps the generality rows.
Cross-resolution placement (an HF content and an LF characterization
materialized over the same `.where`/`.when`) is a PS × SS cooperation at the
shared coordinate frame, not an intra-SS affair.

This supports both directions:

- **analyze synthesized concepts**: SS emits generality candidates with
  placements; analysis tests them against the PS supports.
- **synthesize analyzed concepts**: PS yields supports/chunks; SS chooses
  generality rows that describe them.

### 8. Configuration: `<synthesis>` and `<analysis>`

- PS: `<chunking>` is renamed `<synthesis>` (it is the union knob). Values:
  `radix | bpe | byte` (+ `lexicon | mphf` parked, §5). `none` aliases
  `byte`.
- SS: new `<analysis>` knob hosting `analyse` and the lexer modes; the
  `<lexer>` element moves into the SymbolicSpace section.
- Legacy `<chunking>` / IS-side `<lexer>`: rejected loudly OR accepted via a
  documented one-release compatibility shim — pick one policy and state it
  in model.xsd comments. No silent permanent aliasing.
- Schema, every shipping config, and the sweep harness
  (`test/_sweep_chunking.py` string-substitutions) update together.

### 9. Philosophy Documentation

Rename:

```text
basicmodel/doc/BuddhistParallels.md -> basicmodel/doc/Philosophy.md
```

Update links in `README.md`, `doc/Spaces.md`, `doc/Logic.md`, and any
comments/tests pointing at the old filename.

Sections, before the Buddhist material:

```text
# Philosophy

## Analytic / Synthetic (Kant)

## Epistemic Levels (Ramsey)

## Buddhist Epistemology
```

Content:

- **Kant**: analysis decomposes a given unity into its conditions
  (top-down); synthesis combines given elements into an object (bottom-up).
  Map: SS analyzes the presented unity; PS synthesizes the manifold of
  atoms; reconstruction is their joint employment.
- **Ramsey**: the project-specific epistemic-level mapping; verify the exact
  Ramsey terminology before presenting it as scholarship.
- **Buddhist epistemology**, with the corrected orientation:
  - `rang-mtshan` — specifically characterized particulars: PS, bottom-up,
    grounded in the eidetic percept store over exact atoms.
  - `spyi-mtshan` — generally characterized entities: SS, top-down.
  - `don-spyi` / `sgra-spyi` — meaning-/term-generality: descriptor ROLES in
    the one SS generality codebook, materialized via
    `.active`/`.where`/`.when`.

Bridge text:

```text
Input reconstruction combines specifically characterized perceptual
particulars with generally characterized symbolic divisions. The perceptual
branch synthesizes bottom-up — union of atoms into recurring surface forms,
content-represented in the percept store within a snap distance — so exact
reconstruction stays grounded. The symbolic branch analyzes top-down —
intersection/division of the presented unity into parts characterized at
large scale — so meaning- and term-generalities never masquerade as
particulars.
```

## Implementation Phases

### Phase 0: Contract Tests

Add tests BEFORE changing behavior:

- `InputSpace.forward()` returns two outputs with exact shapes
  `percepts_in [B, N, 1]` and `concepts_in [B, 1, N]` on a tiny byte/raw
  fixture; both views address the SAME underlying values.
- `SymbolicSpace.forward()` accepts the optional `IS_concepts` unity input.
- Existing single-stream assumptions fail loudly at the model boundary
  rather than silently dropping the unity input.
- `InputSpace.reverse(percepts, concepts)` produces `[B, N]` /
  source-compatible reconstruction.
- Non-altering analysis: an SS analysis pass leaves the unity buffer
  byte-identical (boundaries/measurements are overlay outputs).
- Knob acceptance: `<synthesis>` accepted on PS, `<analysis>` (+ `<lexer>`)
  on SS in a tiny config; legacy spellings follow the §8 policy.

### Phase 1: InputSpace Dual-View Split

- `InputSpace.forward()` emits `(percepts_in, concepts_in)`.
- `_lex_embed_stem` passes the atom view into `PerceptualSpace.embed_stem`;
  `finalize_stem` keeps PS bookkeeping until the split is stable.
- The unity view is staged for SS but UNUSED this phase (no behavior change
  downstream).

Acceptance: tiny model forward runs; shape tests pass; suite stays green
(2138-passed baseline, commit e85bc1f); no codebook behavior change.

### Phase 2: Symbolic Input Branch

- `SymbolicSpace.forward(CS_subspaceForSS, IS_concepts=None)`; stage 0
  converts the unity view into symbolic evidence, composed additively inside
  SS.
- A test proves stage-0 SS output changes when `IS_concepts` changes; a test
  proves stages > 0 run without direct input.

### Phase 3: Pi / Sigma Swap

- `PerceptualSpace.pi (PiLayer)` → `PerceptualSpace.sigma (SigmaLayer)`;
  `SymbolicSpace.sigma (SigmaLayer)` → `SymbolicSpace.pi (PiLayer)`.
- Sizing/spans transfer verbatim (`<sigmaPi>` global; `_fold_width`).
- Decide + record the checkpoint key policy (§3); review
  `ConceptualSpace.sigma_in` naming (open decision).

Acceptance: suite green; XOR still solves + reconstructs (the recipe is
PS-side and untouched, but the SS fold type changed — regression-gate it).

### Phase 4: Knob Split + Front-End Migration

- `<chunking>` → `<synthesis>` on PS (radix/bpe/byte; lexicon/mphf parked).
- `<analysis>` + `<lexer>` land on SS; `analyse` dispatch and the lexers
  move; SS-side eager-stem staging for host tokenization.
- Schema + configs + sweep harness updated together (§8).

Acceptance: smoke prompts reconstruct byte-perfect through the PS store on
every synthesis mode; analysis modes produce non-altering
boundaries+measurements; no compiled-path host sync regressions.

### Phase 5: Materialization Split

- `SyntheticSubSpace` (PS): content spell-out through the percept store.
- `AnalyticSubSpace` (SS): division/generality realization via
  `(.active, .where, .when)` against the generality codebook.
- Shared base behavior only where genuinely identical; callers stop assuming
  `.what[active]` is the whole event.

Acceptance: tests distinguish bottom-up content materialization from
top-down generality materialization.

### Phase 6: SS Generality Codebook

- Formalize descriptor roles (meaning-general / term-general / LF-coarse) as
  row metadata in ONE SS codebook (§7); HF surface rows live in the PS
  store.
- An LF generality and an HF content row can be materialized over the same
  `.where`/`.when` (PS × SS cooperation).

Acceptance: role metadata round-trips; analyzed supports can be described by
generality rows without becoming percept-store content; synthesized
generalities can be checked against PS supports.

### Phase 7: Reconstruction Recombination

- `PerceptualSpace.reverse()` produces the percept branch;
  `SymbolicSpace`/`ConceptualSpace.reverse()` exposes the concept branch;
  `InputSpace.reverse(percepts_recon, concepts_recon)` recombines via the
  ILL (§6) — exact OR approximate, chosen explicitly.

Acceptance: direct IS forward→reverse roundtrip; model-level reverse finite
on `MM_xor` or smaller; exactness claim matches the implemented mode.

### Phase 8: Philosophy Rename and Link Cleanup

Doc-only (§9). Acceptance: no broken links; README points to
`Philosophy.md`; the orientation table of this plan is reflected in the
`rang-mtshan`/`spyi-mtshan`/`don-spyi`/`sgra-spyi` mapping.

## Risks

- **Pi/Sigma swap vs checkpoints**: parameter keys flip owners; without a
  state_dict migration, existing checkpoints silently mismatch. Decide
  policy in Phase 3, loudly.
- **Lexer-ownership blast radius**: `<lexer>` is read from the InputSpace
  section by many configs/tests; `MM_xor` pins `lexer=byte` for radix — but
  the chunkers consume the raw atom view after Phase 1, so the pin becomes
  inert; verify rather than assume.
- **Host-sync on SS**: lexing is host-side tokenization; without the
  eager-stem pattern it lands inside the compiled body and breaks fullgraph.
- **Shape churn**: tests asserting single-arg `InputSpace`/`SymbolicSpace`
  contracts fail as expected migration work, not regressions.
- **Exactness ambiguity**: `2N → N` recombination cannot be a bijection over
  arbitrary branch pairs; exact mode needs the square layer + augment.
- **Generality drift**: the SS codebook must not absorb HF surface content
  (that is the PS store's job), or analysis collapses back into chunking.
- **XOR recipe**: `lexer=raw` + radix + 1-vector SBOW must survive every
  phase (it is the standing regression guard; `asymmetric-vq` §8).
- **Philosophy rename blast radius**: the old filename is referenced in
  docs, comments, and tests.

## Open Decisions (tracked, not blocking)

- `ConceptualSpace.sigma_in` naming/role after the swap (§3).
- ~~Checkpoint key migration vs fresh-start (§3)~~ — DECIDED: fresh-start
  (Phase 3, 2026-06-09).
- Exact vs approximate recombination ILL (§6).
- Final home of `lexicon`/`mphf` (§5).
- Legacy-knob policy: loud rejection vs one-release shim (§8).

## Done Criteria

- `InputSpace.forward()` emits the dual view: atoms `[B, N, 1]` to PS, unity
  `[B, 1, N]` to SS; both branches receive input-derived evidence.
- **Pi lives on SS (analysis, intersection); Sigma lives on PS (synthesis,
  union).**
- `analyse` + the lexers run on SS, breakdown-only and non-altering, with
  large-scale approximate characterization snapping to the SS codebook.
- The chunkers run on PS under `<synthesis>`, content-representing with a
  snap distance, byte-exact reconstruction intact through the percept store.
- A single SS codebook stores generality rows (meaning-/term-/LF-coarse
  roles) materialized via `.active`/`.where`/`.when`; HF surface contents
  stay in the PS store.
- Analytic and synthetic materialization are separate enough that SS cannot
  absorb surface content and PS cannot mint generalities.
- Reverse reconstruction uses both branches through an explicit, tested
  recombination layer.
- `Philosophy.md` replaces `BuddhistParallels.md` and documents Kant,
  Ramsey, and the corrected Buddhist mapping.
