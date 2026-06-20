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

## Processing contract (Alec, 2026-06-10)

A standing rule for every phase below and all future work:

- **Contain calculations in Spaces** — the Models body loop orchestrates
  only.
- Within Spaces, the surface is mainly **`__init__` / `forward` /
  `reverse` / `reset`**.
- **Data passes to those methods inside SubSpaces**, and Spaces keep
  **little or no state** — clearing the way for **pipeline parallelism
  over Spaces**.

Status (2026-06-10): the 2-stream bind calculation moved into
`ConceptualSpace.bind_streams` / `unbind` (demux / fit / slot-stack /
cascade / corpus-callosum glue), with the bind carrier riding ON the
SubSpaces (`_bind_carrier`) rather than model state; the Models helpers
delegate to the Space statics. **Known contract debt, tracked:**

1. `concepts_in` (the unity view) is a raw tensor; the contract wants a
   SubSpace carrier (Phase 5/6, with the `.active`-views work).
2. SS stage-0 forward-locals (`_staged_analysis_spans`,
   `_stage0_recon_loss`, `_stage0_z_pre_snap`, `_stage0_indices`) park on
   the Space instance; they should ride the SubSpaces once the unity is
   SubSpace-carried.
3. Model-level staging (`_staged_in_sub`, `_staged_concepts_in`,
   `_combine_carriers` as a test handle) is orchestrator staging — kept,
   but the reverse path now prefers the SubSpace-carried bind.
4. The Space lifecycle methods are `Start`/`Reset`/`End` today; the
   contract names `reset` — align naming when next touching the
   lifecycle.

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

### 6. Reconstruction: Recombining Both Branches — PAINTING (rev. 2026-06-10)

Reverse reconstruction synthesizes the original input from both branches:

```python
InputSpace.reverse(percepts_recon, concepts_recon)
```

**Design (Alec, 2026-06-10 — supersedes the draft's concat+ILL
recombiner):** the recombination is PAINTING, not a learned mix:

- the **Universal view paints the background** — the coarse top-down
  scaffold lays down the canvas first;
- the **Atomic view is averaged in** (or painted over) — the exact
  bottom-up content lands on that background; **start with averaging**,
  graduate to paint-over (atomic wins where it has support) if averaging
  proves too soft.

```text
background = broadcast(concepts_recon)        # universal paints first
recon      = average(background, percepts_recon)   # atoms averaged in
             (later: paint-over -- atomic replaces where supported)
```

No recombination parameters. **LANDED (2026-06-10, uncommitted)** with
one refinement learned from the XOR_exact gate: the painted COMBINED
surface rides the SubSpace as ``_painted_event`` while the event itself
stays the ATOMIC reverse -- the exact word/byte decode reads the
continuous reversed tensor (XOR_exact's codebook-free chain), and
painting it (even only the padding) decodes to junk words. The
conceptual branch is the stage-0 SS stream of the bind's exact inverse,
stamped by ``_reverse_body`` and carried across the PS handoff
(``_concepts_recon`` -- SubSpace-carried; ``reverse`` stays single-arg
per the processing contract). **Open hook:** promoting the painted
surface to a consumer (the reconstruction loss and/or the report's
Reconstructed column) -- decide when the decode can read the atomic
branch separately. The retired alternative (flatten/concat/ILL,
rectangular-vs-square) is recorded here for history only.

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

### The `<codebook>` knob STAYS: subsymbolic vs symbolic iteration (Alec, 2026-06-10 — SUPERSEDES the #13b knob-removal)

The SS `<codebook>` tag is NOT removed. It is the iteration-mode switch:

| `<codebook>` | PS iterations (up to conceptualOrder) | SS iterations |
|---|---|---|
| `none` | subsymbolic | **subsymbolic** (continuous legs; the whole loop is subsymbolic, then reconstitute) |
| `quantize` | subsymbolic | **symbolic** (the SS leg QUANTIZES its input) |

The knob's codebook applies to the **input from CS** (the t>0 recurrent
leg, `CS_subspaceForSS`) — NOT the input from IS (stage-0 unity), where
the ANALYSIS codebook lives. Two separate codebook families:

1. **Meronymic/analysis codebooks on INPUT** — the stage-0 analytic
   generality store (adopt-on-first-sight, recon gather, descriptor
   roles — the Phase-5/6 machinery) governed by `<analysis>`, and the
   PS-side percept/lexicon stores (`<synthesis>`, Task-4 shared byte
   store). Always present; independent of the `<codebook>` knob.
2. **The symbol codebook on the CS leg** — engaged by
   `<codebook>quantize</codebook>`; it captures, in place, the
   correspondence the retired cross-space path was groping at: the
   **code-as-written vs the code-for-the-concept** — the first is a
   SYMBOL for the second and must be interpreted. Codebook row =
   concept code (full `nDim`-wide vector); row id / narrow code = the
   written symbol (the 4-wide ID channel of the SS output). It emits
   **ONE symbol at a time**, and per APOHA the emitted symbol's
   copart is ZEROS everywhere — "the concept of the cup appears to
   the mind through the negation of non-cup"; the exclusion is the
   appearance, not padding. In symbolic iterations **the codebook
   REPLACES the Pi operation** (selection-by-exclusion stands in for
   computed intersection) — which is why thought is top-down, serial,
   and can only be ACTED ON by Sigma: the emitted symbol is discrete
   and given, composable but not re-analyzable (Alec, 2026-06-10;
   details in `2026-06-10-symbolic-iteration-codebook.md` Step 1).

**This codebook REPLACES `insert_paired_word`** (the PS→SS reach-across
that wrote an orth row + random semantic partner per lexicon word and
remapped `key_to_index` onto SS rows). Retirement blast radius,
inventoried 2026-06-10: the `Embedding.insert` peer hook
(Spaces.py ~3713), the `stage_oov` bulk migration (~3446–3495,
`wv.tie_to_codebook`), the autobind `mark_word_atom` fallback,
`embed.WordVectors.tie_to_codebook`, and five pinning test files
(`test_tied_orth_storage`, `test_lexicon_ownership`,
`test_tied_vectors_compile`, `test_tying_storage_shared`,
`test_unified_lexicon_codebook`). The lexicon keeps PS-LOCAL storage
permanently; the decode already resolves row→word through the inverse
of `key_to_index` (identity for untied lexicons), so the decode layer
needs no further change.

**Implementation order (next iteration):**
1. CS-leg symbolization (parallel leg, t>0, quantize): snap the CS
   input against the SS codebook; the event becomes the SYMBOLIC code
   (value substitution is correct HERE — these are symbolic
   iterations) with honest STE via adopt-on-first-sight re-homed to
   this leg (virgin rows adopt CS-input vectors; the bug-A lessons
   hold: no `replace_W` clobber, recon gather trains rows toward
   concept codes, EMA off).
2. Stage-0 machinery re-homes under `<analysis>` (the input-side
   analytic store becomes its own basis object, decoupled from the
   `<codebook>` gate).
3. Retire `insert_paired_word` + the tie (inventory above).
4. **The 4-D second-order acceptance (MM_20M, SS quantize):** the SS
   symbolic iteration emits `[1024, 8]` narrow codes (8 = 4 what +
   2 where + 2 when; the 4-wide what IS the symbol ID); the REVERSE
   keys the codebook by ID and recreates the full 1024-wide concept
   representation — "reconstruct by keying the codebook with
   indices"; quantization is what allows returning just the
   activation values from CS, enabling SECOND-ORDER symbols. Lands as
   the acceptance test of step 1 (xfail until then).

XOR_exact stays `<codebook>none</codebook>` BY DESIGN (the
all-subsymbolic mode), not as a gap. The 2026-06-10 quantize-mode
fixes (the `replace_W` gate, the decode inverse-map) remain valid and
load-bearing for the quantize mode regardless.

### Phase 5: Materialization Split — **MECHANISM FOUND + FIRST HALF LANDED (2026-06-10, uncommitted)**

- `SyntheticSubSpace` (PS): content spell-out through the percept store.
- `AnalyticSubSpace` (SS): division/generality realization via
  `(.active, .where, .when)` against the generality codebook.
- Shared base behavior only where genuinely identical; callers stop assuming
  `.what[active]` is the whole event.

Acceptance: tests distinguish bottom-up content materialization from
top-down generality materialization.

**Status (rev. 2026-06-10).** The split materialized EMPIRICALLY before it
materialized structurally — as the #13 mechanism. The blocker (XOR_exact
4/4 → 0/4 the moment SS gets `<codebook>quantize</codebook>`) was bisected
by a ladder of trained-CLI variants, each toggling one suspect (probe
scripts under /tmp, fixture flip+restore per run; all post-dispatch-rewrite,
all with the stage-0 forward already NON-ALTERING, `z_q = carrier`):

| variant | what differs | XOR output |
|---|---|---|
| B    | Codebook basis present, every `self.codebook` gate off | **learns** |
| C2a  | full path minus the stage-0 recon term | flat (0.175 floor) |
| C3   | C2a minus adoption | flat |
| C4   | gates off (B) + Models VQ hardwire ON | **learns** |
| C5   | C3 with `Codebook.quantize` forced to the fallback branch (no `vq()` / no `replace_W`) | **learns** |
| C6   | C3 with the full `vq()` call, only `replace_W` skipped | **learns** |
| fix  | everything ON, `replace_W` gated on `vq.ema_update` | **learns** (0.0010) |

CONVICTION: `Codebook.quantize`'s per-call `replace_W(self.vq.codebook)` —
an EMA-mode service (refresh the prototype after an in-forward EMA write)
that, under the asymmetric hardwire, re-pointed the space-owned basis `W`
at the VQ's gradient-orphaned random matrix on every call and severed the
space's trained prototypes from every `W` consumer. Acquitted on the way:
reverse mechanics, loss terms, plain materialization, `set_event` grad
flow, `_active` override (stays None — event slot is a plain Tensor),
recon-term loss competition, optimizer param append, adoption, roles,
naming, RNG/init shift (baseline robust across unseeded runs).

The TWO-STORE reading this forces (the split, realized inside the SS
basis): **`basis.W` is the SYNTHETIC/materialization store** (task-
gradient-trained, space-owned; what `.what` consumers read) and
**`vq.codebook` is the ANALYTIC/generality store** (recon-gather-trained;
quantize NAMES against it: indices/roles/adoption). Landed:

- `Codebook.quantize`: the `replace_W` refresh gated on `vq.ema_update`
  (EMA codebooks keep it; asymmetric SS leaves `W` alone).
- Stage-0 forward NON-ALTERING (`z_q = carrier`): analysis "does not
  alter the data; it does snap to a codebook" — the snap is annotation,
  not value replacement. (The earlier STE substitution put drifting
  codebook VALUES into the bind's SS half; with rows reassigned under the
  recon gather, the late XOR transition could never consolidate. STE
  alone was NOT the whole story — see the table — but it is contrary to
  the stated analysis semantics and is retired on this path.)
- `adopt_stage0_evidence` (host-eager, in `_lex_embed_stem` beside span
  staging; the data-dependent `unique` cannot live in the compiled body):
  VIRGIN rows (descriptor role UNASSIGNED) adopt the evidence vector that
  selects them — data-dependent init in the lexicon's
  insert-on-first-sight idiom; makes the STE/naming honest from step one.
  No `replace_W` (two stores). `test_ss_vq_asymmetric_flags` updated:
  first training forward may adopt ONCE; thereafter bit-stable.
- `_stage0_carrier` factored out (shared by the compiled stage-0 forward
  and the eager adoption pass).

**SECOND HALF — OPEN (the analytic NULL descriptor + the wiring leak).**
With gates off (variant B) the recon TEXT is already junk (`'A \x01'`),
so the breakage is basis-object-level, not gated logic. Fill-probe
finding: under `none` the SS `.what` Tensor receives NO per-batch fill at
all — the only write is the per-batch CLEAR (`_clear_runtime_basis` →
`setW(None)` from `Start`); the `none` fixture's exactness is a property
of the basis-less passthrough. Input recon under a Codebook basis
converges ~4× worse, degrading from when output training kicks in
(epochs ~300+).

**Design decision (Alec, 2026-06-10):** ASCII text carries little
universal content — but NULL vs non-NULL IS the right text analysis.
Today text synthesis spells only up to the first NULL; a FULL-FRAME
reconstruction depends on NULLs being represented by SS. The SS
contribution to text recon is therefore a FUNCTIONAL descriptor: a
codebook row that materializes as "insert NULL over this row's `.where`
extent" (a pointer to behavior, not content). Implementation sketch:
a `ROLE_NULL_FILL` descriptor role; pad spans' zero-evidence names to
that row at stage 0; the painting reverse materializes it by writing
NULL across the span — universal background = NULL, atoms at support,
full frame byte-exact. This also resolves Phase 7's consumer-promotion
objection: with NULL painted (instead of coarse-mean upsample), the
painted surface can BE the event for text — padding decodes to NULL,
which the display contract already filters.

**Acceptance (Alec): the byte-exact XOR_exact gate STAYS for #13b.**

**Historical check (2026-06-10, Alec's lead "it worked until the commit
before this"):** detached-worktree runs of the SS-quantize flip at
e85bc1f (pre-plan) and 4ce3f2d (committed plan HEAD): at BOTH, the
OUTPUT learns (legacy EMA loop alive: `replace_W` copied EMA-tracked
values — coherent) and the recon TEXT is the SAME junk (`'A \x01'`).
So: bug A (output collapse) was a true regression of the UNCOMMITTED
asymmetric leg (the Parameter promotion flipped `replace_W` from
value-copy into object-rebind) — found and FIXED today, parity
restored. Bug B (text) is NOT a regression: byte-exact recon through
an SS Codebook basis never existed — even a healthy EMA codebook
reverses to 8 coarse centroids, not lexicon rows. Nothing to bisect in
history; the NULL-descriptor design below is NEW capability and the
whole remaining #13b gate.

Transplant probe — **RUN (2026-06-10): WIRING LEAK CONFIRMED.** `none`
weights loaded into a gates-off Codebook-basis model (`strict=False`;
15 missing keys, all `what.W`/`what.vq.*`), one seeded eval epoch each:
`allOut` max-diff 6.9e-3, **`lastIn` max-diff 1.793** — the REVERSE
output differs at byte scale with identical weights.

**BUG B CONVICTED + FIXED (2026-06-10, late). GATE PASSED.** A getW
spy on the SS basis named the consumers: the PS lexicon's vector
storage IS the SS basis (the 2026-05-27 tied-storage refactor —
`WordVectors.tie_to_codebook`); `Embedding.insert` hands every fresh
word to `SymbolicSpace.insert_paired_word`, which writes an ORTH row
(the PS vector) + a random SEMANTIC partner row into the SS codebook's
`W` and remaps the PS-side `key_to_index[word] -> orth_idx` — while
`index_to_key` deliberately keeps plain insertion order ("we don't
re-sparsify it here"). The decode (`decode_reverse_meta`) rendered
rows POSITIONALLY via `index_to_key[idx]`, so every word decoded as
whatever string sat at LIST position = its ROW index ('\x01'→'\x03',
'\x04'→'\t' — the 2k+1 pairing arithmetic; junk from insert time, no
training needed; self-lookup probe `/tmp/xor_tie.py`). Fix (in
`decode_reverse_meta`): row→word via the INVERSE of `key_to_index`
(identical behavior for untied lexicons where the maps coincide), and
the nearest-row search RESTRICTED to mapped rows so semantic/virgin
rows cannot shadow a word row. Self-lookup exact in both modes.

**Acceptance run: XOR_exact with SS `<codebook>quantize</codebook>`,
everything on (replace_W gate + non-altering stage-0 + adoption +
recon gather + decode fix): 4/4 OK — byte-exact text AND XOR
predictions** (0.16/1.01/1.05/0.02). Suite 2156/0. The #13b knob-
removal hardwire is UNBLOCKED (and Task 4 with it).

Codebook ADMISSIBILITY (Alec's question, answered): SS insertion is
triggered by PS lexicon inserts — every word `Embedding.insert` adds
(ASCII bootstrap atoms, lexed words on first sight) flows to
`insert_paired_word` → orth+semantic row pair in SS `W`; capacity
doubles via `grow_to` on exhaustion (XOR_exact's nVectors=8 grew to
512). Stage-0 analysis naming writes NOTHING into `W` (it names
against the analytic store `vq.codebook`; adopt-on-first-sight is the
only writer there).

CAVEAT (open observation): the SHIPPING `none` fixture's recon shows
run-to-run flakiness post-session (predictions always 4/4; recon 4/4,
4/4 early-day, then 3/4, 3/4, 2/4 — different rows each time;
unseeded CLI). The `none` training path is untouched by inspection
(gates off; decode change is render-only and identity for untied
maps), but attribution needs a SEEDED fixture run — consider seeding
the CLI (hygiene task) before reading too much into single runs.

STILL OPEN (forward-looking, not gating): the NULL-descriptor
materialization above (full-frame recon; pairs with Phase-7 painted-
surface promotion), and Alec's second-order-symbol acceptance: under
quantize, CS should be able to return ACTIVATION VALUES ONLY, with
reconstruction re-keying the codebook by INDEX (the 4-wide symbolic-ID
channel of MM_20M's 8x1024 muxed event) — add an MM_20M test that the
4-D ID round-trips to the full conceptual representation.

### Phase 6: SS Generality Codebook — **CORE LANDED (2026-06-10, uncommitted)**

- Descriptor roles formalized as row metadata in ONE SS codebook (§7):
  `Codebook.descriptor_roles` buffer + `ROLE_LF_COARSE` /
  `ROLE_MEANING_GENERAL` (don-spyi) / `ROLE_TERM_GENERAL` (sgra-spyi)
  constants, `set_descriptor_role` / `get_descriptor_role` /
  `ensure_descriptor_roles` API (mirroring the category_ids /
  part_parents idiom). The stage-0 evidence snap tags its selected rows
  LF-COARSE (analysis outputs ARE the coarse characterizations).
  Materialization is unchanged by the role — roles say which FACE of
  generality a row carries. Pinned by
  `test_descriptor_roles_lf_coarse_tagging`.
- STILL OPEN (rides with Phase 5): the HF/LF cross-resolution placement
  acceptance (an LF generality and an HF percept-store row materialized
  over the same `.where`/`.when` — PS × SS cooperation), role
  persistence in the ckpt bundle, and checking synthesized generalities
  against PS supports.

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
