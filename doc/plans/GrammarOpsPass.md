# Grammar-Ops Corrective Pass — Plan Seed

> **Status: SEED (proposals awaiting author sign-off).** Successor to
> [MeronomyPlan.md](MeronomyPlan.md) (complete: stages 0–9 landed,
> `<meronomy>on` cutover live at basicmodel `ca437cd`, suite 2314
> passed / 0 failures). Scoped by the author's corrections of
> 2026-06-11 (recorded in MeronomyPlan §6): syntactic validity is the
> grammar files' job, not the masks'; the masks are lexical; the
> syntaxes do not unify through De Morgan; semantics enters as
> attention over symbols, not semantic rules. Items marked
> **[PROPOSAL]** are defaults to veto; items marked **[DECIDED]**
> restate author direction.

## 0. Strategy

The grammar operators today are syntactic, total, symmetric, and
exclusive: any rule applies over any operands ("colorless green ideas
sleep furiously" composes as readily as sense), one verb is one weight
matrix or none, comprehension and production knowledge are unrelated,
and serial parsing parks perception. This pass makes the operators
**lexically selective** (masks), **semantically guided** (intent
priming over both towers), **kernel-audited** (per duty), and
**attention-capable** (preattentive synthesis with conflict
preemption) — without adding a single hard guard: every mechanism is a
multiplicative factor on the rev-c soft rule superposition, with
discreteness only at commit points.

The unifying object (§6 below): one weight algebra,

    weight(rule, word, operands, role, context) =
        rule_probability(rule, tier)          # which rules fire here
      × lexical_gate(word)                    # which slice of the shared operator
      × role_gate(analysis | synthesis)       # per-direction competence
      × intent_priming(operands, intent)      # what context licenses

with commits at the single-writer tick and at mint, and preemption by
the truth measure's conflict region.

## 1. Role-collapsed grammar migration (syntax stays in the grammars)

**[DECIDED]** Syntactic validity — which categories a rule takes — is
defined by the `.grammar` files, NOT by operator masks. The standing
migration: all grammars to the role-collapsed format (roles defined by
**operator argument position**; format spec in
`doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md`
§4).

- Already migrated: `default.grammar`, `xor.grammar`,
  `shamatha.grammar`, `role_collapsed.grammar` (the reference).
- Straggler: `complete.grammar` (no role-collapsed header) — audit and
  migrate; sweep for any other non-conforming grammar files.
- Tests: format conformance check per grammar file; the existing
  grammar-driven suites stay green through the migration.

## 2. Lexical masks on lift/lower (one matrix, many verbs)

**[DECIDED]** The lift/lower masks are **lexical modulation of shared
operator matrices**: differentiate walking from running without paying
a weight matrix per verb. One shared verb operator; a per-word mask
selects its slice.

- Mechanism seed (already in the code): the per-call low-rank ``gate``
  on the LDU diagonal (`_d_effective`, `Layers.py`; flows through
  `LiftLayer` :2286 / `LowerLayer` :2501, and through the
  `MeronymicFoldAdapter` unchanged). The mask IS a gate vector.
- **[PROPOSAL]** Mask source: the word's own code — a small learned
  projection from the lexical embedding to the gate's rank space,
  owned by LiftLayer/LowerLayer (one projection per operator, shared
  across the vocabulary; the per-verb difference rides entirely in the
  embedding → gate map). Gate parametrization stays the existing
  tanh-bounded convention.
- **[PROPOSAL]** Training: end-to-end with the composition losses (no
  separate mask objective); the registration loss (§5 of the spec)
  applies unchanged since gating preserves the weight law (the
  contractive clamp bounds the gated diagonal).
- Tests: two verbs sharing one matrix produce distinct compositions
  with masks on and indistinguishable ones with masks ablated; the
  weight law (§10.10) holds under arbitrary gates; gate flows through
  the adapter on the membership path.

## 3. Kernels: no audit — the boundary is already duty-clean

**[DECIDED] (author, 2026-06-11): no per-duty kernel audit.** The
grammar ops are decidedly NOT order-preserving, and do not need to
be: they are not producing new parts or wholes within a space — they
COMBINE existing ones in interesting ways. Order preservation is the
law of meronomy CONSTRUCTION, and construction lives exactly where the
cutover put it: the two meronymic slots (PS.sigma / SS.pi, the
membership kernels behind the adapter). Everything above the slots —
Lift/Lower, union/intersection/negation, the chart ops — keeps its
signed/odds kernels permanently; cancellation and sharpening are
features of combination, not defects. The only constraint that
survives from the old audit framing: the lexical gate-slicing surface
(§2) must keep working on the signed kernels — it already does
(`_d_effective` gating predates the meronomy).

## 4. Rule sharing across directions — grammar tier only (OPEN)

**What the De Morgan construction does and does not claim (author
question, answered).** It is NOT one operation acting as both AND and
OR. It is an algebraic identity of the membership chart: with
complement ``c(m) = 1 − m``, the product-family kernel conjugated by
complement IS the probabilistic-sum family —
``π(m) = exp(W·log m + b)`` gives ``a·b`` (AND), and
``σ(m) := c(π(c(m))) = 1 − exp(W·log(1−m) + b)`` gives ``a + b − ab``
(OR) — two functions, one parameter set plus one fixed involution
(`SigmaLayer2` holds a `PiLayer2` and owns zero parameters;
bit-exact in `test_de_morgan_exact_with_shared_kernel`). The legacy
Sigma/Pi needed two implementations because they lived in different
CHARTS (additive-tanh vs multiplicative log-odds in the signed
domain), where no exact complement links the pair; in ``[0,1]`` the
link is an identity. Note the signed domain contains the same pattern:
``RadMax(a,b) = −RadMin(−a,−b)``. **The two slots do NOT share
parameters with each other** — PS's σ and SS's π each own their
kernel; only the FORM is shared.

**[OPEN — author wants English motivation; scope corrected]** The
cross-direction sharing proposal applies to **grammar-tier rules
only**. Perceptual analysis methods (on-center/off-surround
segmentation, boundary-finding, figure/ground — in this codebase the
chunking/lexing tier: `ChunkLayer`, BPE, the percept store) are
PRE-grammatical, of a different character from the combinatorial ops,
and their generative duals (rendering) are different machines — no
sharing claim touches that tier.

English motivation offered for the grammar tier:
1. **Cross-modal structural priming**: comprehending a passive raises
   the rate of producing passives minutes later — one stored rule
   exercised from both directions.
2. **Receptive→productive transfer without re-learning**: hear "to
   porch the newspaper" once; both parse and produce novel denominal
   verbs immediately — one combinatorial fact, two directions.
3. **Self-monitoring**: speakers parse their own speech with the
   grammar that produced it; two inventories would systematically
   miss production-specific structure.
4. **The asymmetries are access, not knowledge**: "the horse raced
   past the barn fell" is parse-hard but produce-rare — the same
   reduced-relative rule under different per-role weights; likewise
   comprehension's larger repertoire (you parse Faulkner without
   writing Faulkner).

If adopted: one grammar-tier rule inventory (each rule invertible),
the two directions as its two readings, **per-role weights** (the
`role_gate` factor) carrying the comprehension/production asymmetry.
Decision deferred to author review of the evidence above.

## 5. Semantic guidance = attention over symbols (single intent, both towers)

**[DECIDED]** The grammars stay syntactic. Meaning-sensitivity comes
from **priming over symbols**, not semantic rules.

- Existing machinery: attention modes
  (`off|primer|second-order|low-rank`, `Spaces.py:125`) and the
  primed-reverse-generation boost weights (`left_priming` /
  `right_priming` on `Ops._binary_op_recommend`, `Layers.py:11489`;
  plan `2026-05-20-primed-reverse-generation.md`) — currently SS-side.
- **[DECIDED]** Mirror the priming into **both codebook towers**: with
  towers-as-codebooks (spec §5), priming is boost weights over rows;
  the PS/extent tower deserves it as much as the SS/intent tower —
  primed *recognition* alongside primed *retrieval*.
- **[DECIDED]** Combine both grammars with a **single intent**: one
  current-intent code priming both towers simultaneously, weighting
  the analytical and synthetic superpositions toward the same context.
- **[PROPOSAL]** Mechanics: the intent code is the current idea (top
  of the PS-side workspace / the pursued goal event); per-tower boosts
  are its graded similarity against each codebook's rows (one matmul
  per tower, the pole-quotient law on the reference-bound rows);
  boosts enter the recommender and the rule superposition through the
  existing priming plumbing — no new mechanism, one new producer.
- Tests: priming a tower biases recognition/retrieval rankings
  monotonically in boost weight; the same intent code moves both
  syntaxes' dispatch in the same semantic direction; intent off ⇒
  byte-identical to today.

## 6. Preattention, the serial lock, and the two truth sets

**Definition (the phrase "mutex granularity," plainly):** how much of
the system the serial parser's lock covers — the whole substrate
(perception parks while parsing) versus the workspace only (one move
— split | shift | reduce — per tick) with parallel perception
continuing beneath. The refinement adopts the fine grain: the
single-writer rule stands within a workspace; preattentive parallel
synthesis runs beneath it.

**[DECIDED] Two sets of truths (author, 2026-06-11; doctrinal names
per the Gelug two-truths mapping, Philosophy.md).** The truth store
conflates two kinds of content, and only one may feed preattention.
**Absolute truths are ideas; relative truths are relations between
ideas** (causal implication is the worked example of the relative
set):

- **Absolute truths (ideas)** — timeless region constraints over
  conceptual space, INCLUDING material implication ``(¬a ∨ b)``.
  Evaluable in one pass by the luminosity/coverage criterion; rooted
  at the grammar's absolute-truth start state (``ABS_T`` — already
  declared, e.g. `complete.grammar`'s
  ``<start name="absolute_truth">``). **This set, and only this set,
  is the preattention filter**: the conflict region —
  ``min(T_k, F_k)`` mass from `TruthLayer.luminosity`, measured,
  never stored — computed over the absolute set is the preemption
  signal (contested evidence between the percept stream and the
  parse's commitments captures the serial thread).
- **Relative truths (relations between ideas)** — the worked example
  is causal implication: ``if a then b`` as a PROCEDURE, not a
  region — a state of affairs (NP1 at t₁), a change (VP), a
  consequent (NP2 at t₂). Typed temporal triples (the ``.when`` band
  already carries the shape). NOT expressible as material truth:
  evaluation = evaluate NP1, run one-or-more VP reasoning steps (the
  serial loop — `reason`/`derive`-style stepping), check NP2 —
  verification by SIMULATION or relational evaluation, not by
  coverage. Relative truths never enter the luminosity measure:
  recording ``if a then b`` as ``¬a ∨ b`` both loses the temporal
  content and corrupts the absolute set with a region assertion the
  causal rule never licensed.
- **[PROPOSAL]** Mechanics: two stores (cleanest under
  measured-not-stored — causal entries are not region-shaped, so one
  store with a type flag would force every coverage computation to
  mask) — `TruthLayer` stays the absolute store; a sibling
  causal store holds the typed triples and is consumed only by the
  reasoning loop.

**[PROPOSAL]** Preemption policy: threshold + hysteresis on the
absolute set's conflict mass (per-dimension max, not the mean — one
sharply contested witness should interrupt); action ladder:
checkpoint the workspace → reground (re-run recognition on the
contested span) → frame-split (fork contexts so both readings
survive) — abort only as the degenerate case.

**[PROPOSAL]** Unification: the serial-*with*-attention masks of
`2026-05-29-stm-serial-parallel-modes.md` — meronymic (CS→PS,
expectation painting) and taxonymic (CS→SS, retrieval restriction) —
are the same algebra as §5's priming: multiplicative weights on soft
superpositions, produced by context; implement as intent-priming
producers rather than separate mask machinery.

Fallback if one-workspace preattention proves too coarse: the mode
flag moves from model-global to per-workspace (each workspace serial
or parallel with its own single-writer rule).

## 6b. The Gaussian window (IN RESERVE — superseded as primary by §6c)

> **2026-06-11 (author):** with the sentence protocol of §6c, the
> Gaussian masking is DROPPED as the primary mode-combiner. Its
> duties devolve: global context → the gist pump; local context → the
> workspace stacks themselves; expectation → the per-word
> predict-then-perceive loop; refresh → re-pump on preemption (or per
> clause boundary if needed). The identity-fade construction below
> remains recorded because it is exact and zero-cost to keep on
> paper — the proven fallback if discrete pumping ever needs
> smoothing (e.g., gist staleness on very long sentences). With the
> window dropped, §10.11 needs NO read-only-fold carve-out: parallel
> computation happens only in parallel PUMPS (ordinary mode); serial
> pumps stay pure; the sole clarification is that mode is a PER-PUMP
> property within a serial sentence.

**Why a mutex at all (author, 2026-06-11):** per apoha (meaning by
exclusion — the symbolic cut over a non-conceptual presentation
field) and dual-system theory, the subsymbolic and symbolic processes
INTERFERE at any single pump of the loop; yet one network must both
learn words (subsymbolic) and compose them (symbolic), so one model
must operate in both modes.

**The resolution: the mutex serializes COMMITS, not computation**
(rev-c law: discreteness only at commit points, never in dispatch).
Interference is a write-write phenomenon; per pump there is ONE
symbolic commit (shift | reduce | mint). The parallel substrate
computes continuously beneath as a READ-ONLY modulator — it primes
what the next commit sees and never writes the workspace. Word
learning writes slow weights (codebooks, EMA); composition writes
fast state (the workspace); the per-pump exclusivity binds the
fast-state commit only.

**The Gaussian word-mask is EXACT in the membership chart.** Each
fold's identity element makes masked-out operands exactly invisible
(the §10.2 identity/absorber theorems, already pinned by tests):

    σ side (identity 𝟘):  m′ = 1 − (1 − m)^w   — w = 0 ⇒ m′ = 𝟘,
        and the operand block multiplies log 𝟙 = 0: EXACTLY zero
        contribution for any admissible weights.
    π side (identity 𝟙):  m′ = m^w              — w = 0 ⇒ m′ = 𝟙,
        exactly invisible likewise.

Both are the same operation — a per-position scale on log-mass — and
both are PURE INPUT TRANSFORMS on the membership operands: no layer
changes; implementable against PiLayer2/SigmaLayer2 as they stand.

**Consequence: serial and parallel are the two ends of one dial.** A
Gaussian over word positions centered on the serial cursor, width
``s``: ``s → 0`` is pure serial (the fold sees only the focal word);
``s → ∞`` is pure parallel (the whole slab); between, serial
processing WITH graded contextual/meronymic enrichment from the
windowed parallel fold beneath. The context field enters the serial
loop at the two priming points: it modulates codebook selection (the
match commits serially; its evidence is context-primed) and enriches
the workspace ambient the shift reads.

**Amendment required (one line):** spec §10.11's "no whole-slab folds
fire" becomes "the per-tick 2N fusion stays bypassed (its WRITE duty
migrated into the shift); the folds may fire PREATTENTIVELY —
Gaussian-windowed, identity-faded, read-only — modulating but never
writing the workspace."

**[PROPOSAL]** Mechanics: window center = the serial cursor of the
per-word loop; width = per-model config with a learned-parameter
option (the serial↔parallel dial; plausibly annealed or
attention-controlled later); the windowed field's row-priming reuses
the §5 plumbing.

## 6c. The sentence protocol: parallel pump first (author design)

**[DECIDED shape (author, 2026-06-11; refined same day)]:** in serial
mode, every sentence gets an independent **parallel prelude of
`conceptualOrder` steps** to seed the codebook towers, then the
serial task. Meaning on one side, syntax on the other (the logo: the
star-shaped parallelism of semantics; the binary-branching tree of
syntax) — and instead of gating one another, the modes INTERLEAVE at
pump granularity. Refinements (author):

- **The analysis is the parallel bit**: the π carving of the domain
  into parts (with the σ fusion) is the parallel act — the Gelug
  first moment, direct and non-conceptual: nameless crossing, no
  table lookup, no story (see Philosophy.md §"Direct Perception and
  the Conceptual Overlay"). **The serial bit happens in STM** — the
  conceptual overlay: naming, recognition, memory, preference,
  elaboration.
- **The prelude runs up the tier ladder**: `conceptualOrder` parallel
  steps (the per-order derivation path), not one — the scene
  description is built order-by-order, seeding BOTH towers. The σ/π
  layers are the real subsymbolic shapers of meaningfulness in
  conceptual space (codebook words are produced/conditioned by
  them): the prelude primes semantics at that moment and shapes it
  to a minor degree (small EMA — the word-learning guarantee).
- **Collapse with back-action**: where serial collapse is mandated,
  serial processing selects a STORY (the soft superposition of
  readings collapses, one commit per pump) and the wavefunction of
  reality is simultaneously perturbed — the commits write (workspace,
  bindings, EMA); meaning-making back-acts on the state.
- **The product is one of the two truths**: a completed sentence
  yields an **absolute truth** (an idea — a region-shaped extent,
  ABS_T-rooted, luminosity-evaluable) or a **relative truth** (a
  relation between ideas — causal implication being the worked
  example), per §6's two truth sets under their doctrinal names.

The mutex (one commit per pump) is honored throughout — interference
is avoided by temporal separation.

Consequences:
- **Word-learning guarantee.** Subsymbolic word learning (codebook
  EMA, adoption, percept↔form association) lives on the parallel
  path; the guaranteed pump keeps the lexicon learning even in
  serial mode. One network, both modes, every sentence.
- **Unification with §6b**: the protocol is a WIDTH SCHEDULE on the
  Gaussian dial — pump zero at width ∞ (the slab), serial pumps at
  finite width (the moving field): ``[∞, s, s, …]`` per sentence.
- **Gist routing**: pump zero's output feeds the §5 single intent,
  priming both towers for every subsequent serial pump
  (`bind_streams`'s stage-0 ``seed_payload`` hook is an existing
  landing point).

**[PROPOSAL] decisions:**
1. Pump zero's commit: intent-only (gist = priming context; nothing
   in the workspace) vs workspace floor (push the gist as the initial
   idea — the star becomes slot zero of the tree's workspace; costs a
   slot).
2. Pump zero learns (full normal parallel pump, EMA on — the
   word-learning guarantee) vs frozen read-only gist. "Independent"
   suggests learning ON.
3. Optional consolidation sandwich: one parallel pump AFTER the
   parse, turning the §10.11 coincidence law (serial reduce-chain
   extent ≈ parallel σ extent) from a test into a self-consistency
   TRAINING SIGNAL (gist-vs-parse agreement). Deferred unless wanted —
   but see §6d, which strengthens its case (the consolidation pump is
   the channel by which serial lexical evidence reaches the rows).

## 6d. Codebook update law: percepts ← parallel, references ← serial
## (author, 2026-06-11 — IMPLEMENTED)

**[DECIDED, corrected from the earlier parallel-only draft]:**
**percepts are shaped by the parallel pass; the symbols (references)
are shaped by the serial pass.** The serial pass is the only one that
does referential lookup, so it is the only one that invokes the
referential taxonomy *qua references* — and therefore the only one
licensed to shape it. **STE is fine in both cases**: the real
distinction is whether parallel mode is allowed to shape references
(**it is not**) and whether serial mode is allowed to shape
non-references (**it is not**). This keeps things rational, although
it may represent an overly optimistic arrangement with respect to
human thinking (recorded caveat).

Grounding in the three-pieces decomposition (doc/Architecture.md
§"Three pieces"): the mereological towers + the symbol table and
attention run in PARALLEL; **thought is a subsequent isolation of
attention over that space, enabled by references — and so it shapes
the references.**

**Implementation (landed with this revision):**
- `Spaces.reference_update_mask(serial_mode, reference_ids, n_rows)`
  — the law: rows allowed to move = references when serial,
  non-references when parallel.
- `VectorQuantize.update_mask_fn` chokepoint — the EMA write, the
  EMA accumulators (frozen rows neither decay nor accumulate, so a
  later mask change cannot make them jump from stale statistics), and
  dead-code expiry/revival all respect the mask; byte-identical to
  the legacy path when no mask is installed. Learnable-codebook mode
  gets the same partition via a row-gradient hook.
- `Space.install_reference_update_law(table_getter, side)` — wires a
  space's codebook VQ to the law with a LAZY table getter;
  `side='object'` (PS/extent tower: bound object ids are references)
  or `side='word'` (SS/intent tower: bound word ids). Installed for
  both towers at model construction (Models.py) and self-installed by
  SymbolicSpace at table creation. Dark by construction: meronomy
  off, no table, or no codebook ⇒ mask None ⇒ legacy behavior.
- References gains `bound_objects()` / `bound_words()`.

## 7. Open decisions for author sign-off

1. §2 mask-source projection: per-operator learned projection from
   the lexical embedding (default) vs direct embedding-slice.
2. §4 grammar-tier rule sharing (one invertible inventory + per-role
   weights) vs independent directional inventories — pending author
   review of the English evidence in §4. Segmentation-tier methods
   excluded either way.
3. §5 intent = top-of-workspace idea (default) vs dedicated goal
   state; priming enters rule dispatch as well as codebook lookup?
4. §6 causal-store mechanics: sibling store (default) vs typed
   entries in one store; the causal triple schema (state@t₁, change,
   state@t₂) and its consumption by the reasoning loop.
5. §6 preemption thresholds and the action ladder; per-dimension max
   as the trigger statistic.
6. §6c sentence protocol: pump zero's commit (intent-only vs
   workspace floor); pump zero learns (default: yes — the
   word-learning guarantee); gist-refresh policy (on preemption only,
   vs also at clause boundaries); the optional consolidation sandwich
   (coincidence law as training signal). (§6b's Gaussian window is in
   reserve, not a live decision.)

## 8. Out of scope (recorded)

- Semantic rules in the grammar files (explicitly not the path — §5).
- Per-duty kernel audit (struck 2026-06-11: grammar ops are
  combiners, decidedly not order-preserving; membership kernels live
  at the meronymic slots only — §3).
- Rule sharing at the segmentation/perceptual-analysis tier
  (on-center/off-surround and kin are pre-grammatical, different in
  character from the combinatorial ops; §4 scope correction).
- Causal conditionals as material truths (``if a then b`` is never
  stored as ``¬a ∨ b``; §6).
- Stored derivational identity (resolved 2026-06-11: identity is
  content + participation; spec §8).
- Any new stored epistemic state (measured-not-stored stands; the
  conflict region is computed, κ stays struck).
- Validity-as-selection masks on operators (superseded by §1 + §2 —
  syntax in grammars, masks are lexical).
