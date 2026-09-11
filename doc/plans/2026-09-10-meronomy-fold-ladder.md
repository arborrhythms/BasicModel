# Meronomy as the fold ladder: synthesis and analysis without the radix trie

Status: plan, 2026-09-10 (Alec; reviewed against BasicModel `fc93560`).
Drives the perception change that the
[mathematical thinking plan](2026-09-09-mathematical-thinking.md) Phase 9
ran into. Design background:
[Mereology.md, "Mereological Algorithm"](../Mereology.md#mereological-algorithm)
(the two mereological towers; its adjoint framing is superseded by the meeting rule below) and [Architecture.md](../Architecture.md)
§C (attention as the origin of the reading scope).

## Context

The successor experiments of 2026-09-09/10 showed where the stack binds
concepts to words: the radix trie promotes `12` into an opaque chunk by
recurrence, the aligned word loop keys a concept on the word's surface
string, and the analysis cut uses four fixed byte types. Decisions (Alec,
2026-09-10): concepts must not be bound to words; a run of digits is a
compound concept, never one stored concept per number; chunking should stop
where concepts say the basic level is; the subsymbolic loop takes over the
radix trie's role (a likely speed-up, to be measured, not assumed);
the WholeSpace tower begins with everything and synthesizes downward by
division, learning types by learning boundaries ("spaces bound words"),
and the two towers meet at Rosch's basic level (they need not be duals);
all synthesis / analysis modes except `meronomy`
move to `Legacy.py`, and `meronomy` gets the new implementation. The old
`meronomy` was an alias of `radix` in PartSpace and of the word cut in
WholeSpace, so nothing that worked is lost.

The plan is written as a set of contracts first (algebra, carriers,
boundary predicate, statistics, mutation boundary, identity, phrase
meaning) and phases second, because the reused code does not by itself
provide the properties the phases need; each contract names what the code
provides today and what the phase must add.

## What exists (reuse)

| piece | where | role in the new design |
|---|---|---|
| parameter-free set union in membership space, `1 - prod(1 - m_i)` (probabilistic sum) | `Layers.MeronymicFoldAdapter.compute_aggregate_over_set` (the serial word assembler) | today's rung 0; replaced by the max join (contract 1) |
| learned sigma folds, M-way `atanh`-sum fold, balanced split | `Layers.SigmaLayer.synthesize_over_set`, `generate` | the learned rungs and their numerical inverse |
| isotonic projection | `bin/Mereology.py` `project_monotone` (edge loop, POCS), `where_containment_edges` (quadratic in spans); `join_from_bottom` / `meet_from_top` stay as utilities, unused by the design | the cross-tower order guarantee, bounded per contract 5 |
| compiled fold ladders per word tick | `Models._aligned_part_fold_ladder`, `_aligned_whole_fold_ladder`; `PartSpace.fold_event_ladder`, `WholeSpace.fold_event_ladder` | unary feature folds on an assembled event; retire into the forwards (they do not join spans, contract 2) |
| per-concept fold depth | `Spaces.record_concept_fold_support` | a derivation record; the basic level is estimated separately (contract 4) |
| vectorised type-run cut over property signatures | `Spaces._type_run_spans` (CPU factories, `.item()`), `_build_property_signature_lut` (untagged rows absent, rows >= 63 ignored, canonical fallback when tags are empty), `stage_analysis_spans`, `stage_word_property_weights` | the cutter's run logic; its predicate becomes learned (contract 3) and its host-side parts move behind the mutation boundary (contract 5) |
| digit whole (2026-09-10) | `<digitWholes>`, `Spaces._digit_signature_bits`, the singleton mask of `_type_run_spans` | the explicit form of the singleton predicate the learner must express (contract 3) |
| standalone router (Viterbi / soft-DP tiling to byte terminals) | `perceptual_analyzer.MeronymicRouter`, `MeronymicAnalyzer` | reference for the routing pressure; not on the live path |
| radix store + byte fallback | `Layers.RadixLayer` (longest match, promotion gates), `BytesFallbackEncoder` | Legacy; the test oracle for Phase 4 |
| legacy selection | `Legacy.LEGACY_PART_SYNTHESIS_MODES` (already contains `radix`), `normalize_part_synthesis_mode`, `embed_part_stem`; canonical `meronomy` still sets `synthesis_mode = "radix"` directly in `Spaces.py` | remove the canonical dependency on radix; add the analysis cuts |
| word-major staging | `Spaces._embed_radix_word_major`, `synthesize_word_parts`, `_radix_part_events`; `Models._stage_serial_concept_rows` (surface-keyed, `key = str(value)`) | replaced by ladder staging and identity per whole (contract 6) |
| concept relations | `ConceptualSpace.concept_parts`, `concept_wholes` (deduplicated relation sets, not counts), `_automatic_concept_admitted` (capacity gate only), `cs_forward_content` / `_order_caps` (per-order top-K) | the admission site and the readout window; counts are new state (contract 4) |
| grammar chunk | `Language.ChunkLayer` (`chunk(C, C) = left + right`; PEEL reverse), `complete.grammar` rules; `forwardGrammarWeight` (a bounded local structural contrast on committed folds, chooser-only) | the phrase-level site (contract 7) |
| analysis-mode resolver | `Spaces.py` (`grammatical` / `meronomy` resolve to `raw` at the InputSpace seam) | collapse to one mode |

Mode inventory (`data/*.xml`): synthesis `radix` 17 fixtures, `bpe` 5,
`lexicon` 4 (incl. the `model.xml` default), `meronomy` 6 (BasicModel,
MM_20M_*, nanochat); analysis `word` 8, `raw` 8, `byte` 6, `meronomy` 7.
26 tests touch `RadixLayer` / `percept_store` / `synthesis_mode`. The
inventory is refreshed at the start of Phase 0.

## Decision

- `<synthesis>meronomy</synthesis>` is the subsymbolic fold ladder: bytes
  are atoms; rung 0 joins the atoms of a whole by max; the learned rungs raise the
  order of that code; a code is admitted to the PartSpace codebook by the
  `chunk` operation; no trie in perception.
- `<analysis>meronomy</analysis>` is the descending ladder: the unity at
  the top, the byte-complete tiling at the floor, and one tiling per
  admitted boundary predicate between them, all computed at pass 0; a
  divided whole's code is the meet of its positions' property activations
  (min over positions, contract 1); a property becomes a boundary type when the
  wholes it yields are attested. The two towers are not duals: each
  synthesizes codes monotonically, one upward by joining, one downward by
  dividing, and they meet at the basic level (contract 4).
- All other modes move to `Legacy.py`; the old behaviour stays reachable as
  `radix` / `word` through Legacy as the oracle for the Phase 4 comparison
  and is removed after it.

Why the ladder rather than feedback to the trie ("bigger / smaller
chunk"): the trie is a host island (Python longest-match, hit-count
dictionaries, a growth callback before every compiled body) while the fold
ladder is already compiled and fixed-shape. Whether that is a speed-up is a
Phase 4 measurement (graph breaks, recompilations, peak memory, bytes per
second), not a premise.

## Contracts

### 1. Algebra and the information kept for reconstruction

Two laws exist in the code for joining parts: the serial word assembler's
membership union `1 - prod(1 - m_i)` (`MeronymicFoldAdapter`), the
probabilistic sum, and `Mereology.join_from_bottom`'s `tanh(sum atanh)`,
the Einstein sum. Both are strict t-conorms, each with an exact `1 - x`
dual, and both make the joined whole grow with the number of parts.

Decision: neither. The PartSpace join is the max over parts per
coordinate, and with the WholeSpace meet as the min over positions (below)
the two towers use the Gödel pair, the only idempotent t-norm and
t-conorm, the lattice proper. Reasons: the code is the category, and a
category does not count its parts (multiplicity and order live on the
witness and in the occurrence's `.where` bindings, contracts 1 and 6, so
`11` and `1` are distinct occurrences with the same rung-0 category code);
a whole's max dominates every sub-span's max by construction, so the
containment order holds inside a whole without projection and the isotonic
projection is needed only across the towers; and max is the operation the
grammar already calls `union`. The stated cost: max passes gradient only
to the strongest part per coordinate, where a sum would reach every part;
with a wide event and few parts per whole this is mild, and the learned
rungs above the base have dense gradients regardless. The learned sigma
rungs keep their `atanh` algebra: they are feature folds that raise order,
not lattice operations, and need not share the base's law. The strict sums
are recorded as the alternative considered; `join_from_bottom` and the
assembler's union stay in the code as utilities and Legacy respectively.

The WholeSpace value is a meet in the literal sense, the intersection over
the whole's positions: a divided whole's code is the activation of the
properties that hold at every position of its extent. For the type-run cut
this is exact by construction (a run is the maximal span with a constant
property signature, so "common to all positions" is the signature). With
the learned soft predicates of contract 3 the intersection is a soft AND
over positions, and it must be the idempotent one, the min (Gödel t-norm):
a property holds in a whole to the degree it holds at its weakest position,
independent of length; a strict t-norm such as the Einstein product would
decay with the number of positions and make long wholes propertyless. Join
by max over parts and meet by min over positions are the Gödel pair, both
idempotent, so a whole's category code depends neither on how many times a
part recurs nor on how long the whole is; neither is derived from the
other by complement. Descent from
everything is then literal: the unity's meet is the properties common to
the whole sentence (usually almost none), and every division yields wholes
with more properties in common, so the meet grows as the extent shrinks
while the join grows as the extent grows. Above the base, the learned pi
rungs raise order from the whole's meet as the sigma rungs do from the
part's join. `meet_from_top` stays in `Mereology.py` as a utility but is
not part of the design; the order between the towers (whole dominates
part) is carried by the isotonic projection over `.where` containment,
which is indifferent to how either tower produced its codes. The live assembler's rung 0 changes law in
Phase 1 (it changes anyway when its inputs become byte atoms); Phase 0
byte-identity concerns the Legacy move, not rung 0.

Where the logic lives: the join is the only OR in perception; there is no
AND in either tower, which is what keeps both monotone and orderable by
containment. Negation enters first at ConceptualSpace: a concept is
defined by having some perceptual building blocks and lacking others, an
AND over present parts with a NOT over absent ones, expressed in the
concept's part relations and the `not` / `non` grammar operations.

Neither the balanced split (`SigmaLayer.generate`, which recomposes to the
parent, not to the original children) nor De Morgan duality gives
byte-sequence invertibility. Reconstruction therefore does not rely on
numerical inversion. The carrier keeps an **ordered constituent witness**
per whole: the constituent codebook ids in surface order, their relative
byte positions within the whole, multiplicity (repeated ids), lengths, and
the valid mask; fixed capacity per whole (the whole-length cap of contract
2), padded. Reconstruction of the INPUT replays the witness exactly.
Generated content has no input witness and must not read one. Its
descent is: the learned rungs invert numerically (the balanced split,
which recomposes to the parent within tolerance); rung 0, being a max, has
no numerical inverse, so its constituents are recovered by domination
against the atom rows (an atom is present in a generated whole when the
whole's code dominates that atom's row per coordinate, within the
admission radius: the PEEL step of `chunk.reverse` restricted to atoms),
and their ORDER comes from the symbolic side, the parse that generated the
whole or the witness stored with the admitted concept the whole was
retrieved as. A generated whole that is neither an admitted concept nor
the product of a parse has a constituent set but no order, and is reported
as such rather than serialised in an arbitrary order.

Tests, kept separate: numerical inversion of the learned rungs (the split
recomposes to the parent within tolerance); constituent recovery at rung 0
(domination against the atom rows recovers exactly the atoms of a whole,
including under permutation and repetition); stored-surface replay (byte-exact,
including permutations of the same atoms, repeated bytes, identical
adjacent digits such as `11`); and answer-side generation with the input
witness deliberately unavailable.

### 2. The two towers run automatically: join up, divide down

Synthesis and analysis are both automatic, every pass, and neither
waits for a loss:

- **Synthesis (join by similarity).** Within each whole of the current
  tiling, the part is the max over the atoms under it (contract 1); at
  rungs above, the sigma folds raise its order. Nothing decides whether
  to join; the tiling decides where.
- **Analysis (divide by difference).** Each whole divides wherever the
  codes under it differ. Two sources of difference, both read from what
  is already there:
  1. **Property differences** between adjacent positions: the symmetric
     difference of the two signatures, typed by the property that
     changes and in which direction. A property absent at `i-1` and
     present at `i` is the **left boundary** of a run of it (it begins);
     present at `i-1` and absent at `i`, the **right boundary** (it
     ends). Consecutive occurrences of a property produce no boundary.
     These give the class rungs (letter run, digit run, space run,
     punctuation) and replace the "left of X" predicates.
  2. **Part-code differences** inside a class run: adjacent positions
     whose atom codes are farther apart than the admission radius (the
     radius identity uses, contract 6) divide the run below the class
     level without any sub-class being defined.
- **New whole rows (unsupervised).** When the positions bound to a whole
  row cluster in their codes, the row splits along its principal pull
  by the LBG rule already on WholeSpace ([Mereology.md, "Automatic
  analysis: dividing wholes"](../Mereology.md#lbg-division)); the new
  row's analyzer predicate is the set of atoms whose codes fall on its
  side, read through the callosum. Its begins / ends boundaries are then
  candidates like any other. This is how "vowels" comes to exist. The
  LBG gate under the property basis is lifted for this.
- **Carrier axes.** `[B, W, M, D]` at rung 0 (whole capacity, atoms per
  whole, event width), `[B, W, D]` above; a rung consumes the atoms of
  one whole (rung 0) or the whole's previous-rung code; spatial joining
  across wholes happens only where the tiling ladder has a coarser whole
  (a coarser whole's base is the max over its atoms). The tiling ladder
  is computed at once at pass 0 (the unity at the top, the byte floor at
  the bottom, one tiling per admitted boundary type between); PartSpace
  climbs it with the pass index. Overlength wholes divide by the finest
  available boundary; the compiled loop's fixed residual capacity keeps
  its fail-loud contract for a whole it cannot lay out.
- **Coverage.** Top-k selection (`fieldRetrieve` / `fieldAttend`)
  chooses which wholes ConceptualSpace attends this pass; unattended
  wholes stay on the carrier and are re-offered under the thinking
  loop's forced-closure pressure knob (open question Q2, resolved: one
  knob).

### 3. Boundary types, the sameness level, and what the learner decides

- **Representation.** Per WholeSpace property row `p`, two learned
  bounded weights: `begins_p` (the left boundary of a run of `p` bounds a
  whole) and `ends_p` (its right boundary does). Per property class, one
  **sameness level**: whether a run of `p` coheres at the class rung
  (letters: a run is one whole) or at the atom rung (digits: every atom
  its own whole). The sameness level replaces the singleton weight; it
  generalises `<digitWholes>` without a digit-specific rule, and it is a
  Rosch question (which rung is basic for this class) the learner
  answers. The signature slab is a bool `[B, N, P]`; the cut at `i` is
  `OR_p (begins_p AND p begins at i) OR OR_p (ends_p AND p ends at i)`,
  plus every atom boundary inside a class whose level is the atom.
- **Whitespace.** A space run is a whole like any other; it enters the
  loop as a unit whose grammatical operation is the null operation
  (`stop`), which the reduce pass applies at once, so it costs no STM
  and no discard mask exists. "Space begins" and "space ends" compete
  as boundary types like the others.
- **Pre-seeding (minimal).** The whole lexicon is seeded the way the
  part lexicon is: with atoms. Its atoms are the atomic wholes, one per
  character value (the run of that character), the smallest set of
  segmentations from which every tiling composes; the eight class rows
  (letter, digit, whitespace, punctuation, capital, control, high byte,
  pad) are an optional prior, not the seed. Boundary types are derived
  from rows, never stored. Classes are groupings the learner finds
  (coarser tilings that recur at lower density), and every finer whole
  is a division. The seeded rows are splittable like any other.
- **What the learner decides.** Not whether finer wholes exist (that is
  automatic, contract 2) but which are salient: the score update over
  `begins_p` / `ends_p` and the sameness level per class, by the
  recurrence and density of the wholes each candidate tiling yields
  against the current tiling (the memory-load criterion), at the
  presentation boundary, outside autograd; the reconstruction and
  answer costs reach the weights through the straight-through threshold
  as well. `<boundaryTypes>none</boundaryTypes>` starts with no boundary
  type on (the cold-start test); the class concepts remain seeded.

### 4. Category utility, the meeting rule, and the basic level

- **One estimator.** For any category `c`, admitted or candidate, over
  the presentations seen: `n_c` (presentations containing `c`), `n_cf`
  (containing `c` with feature `f`), `n_f`; a feature is a constituent
  row (the rung below) or a containing whole row (the rung above);
  Laplace smoothing `utilitySmoothing`, minimum evidence
  `utilityMinCount`; a concept and a concept-feature pair count at most
  once per presentation; counts commit at the training path's boundary
  and are frozen under evaluation. `CU(c) = P(c) * (Σ_f P(f|c)² −
  Σ_f P(f)²)`, normalised across rungs by the number of candidates at
  the rung. A candidate phrase's `n_c` is its recurrence and its
  features are its member concepts, the same formula; its gain is
  `CU(phrase)` minus the best of its members'.
- **The meeting rule.** The basic level of a presentation is the rung at
  which a part joined from below and a whole divided from above
  coincide in extent and are both attested (the callosum's
  part-is-whole identity); utility is the estimator and the tie-breaker
  there, ties toward the coarser rung. `P(c)` is the working-memory
  pressure (few, frequent wholes), the predictability term the
  long-term-memory pressure (a bounded set of distinct wholes that
  predict their parts).
- **Bootstrap.** The seeded class concepts give synthesis its first
  domains; boundary types and sameness levels update at the boundary
  from the epoch's counts; synthesis admission commits per presentation.
  The circularity (synthesis needs domains, boundaries need attested
  wholes) is broken by the seeded classes rather than by a byte floor.
- **Salience of a division.** A row split by LBG exists regardless; its
  begins / ends boundaries earn salience only if the wholes they yield
  recur (the same learner, contract 3).

### 5. The mutation and compilation boundary

Moving code into `forward()` does not make it graph-safe: the admission
allocator, fold-support records and property tags mutate Python
containers; `_type_run_spans` uses CPU factories, data-dependent sizes and
`.item()`; `project_monotone` loops over edges with scalar extraction. The
contract:

- Inside the captured forwards: fixed-capacity tensor state only (counts,
  ids, witness, the bool signature slab), bounded loops (the rung count,
  `M`, `W`), no `.item()`. The cut is re-expressed as a tensor run
  labelling (cumulative sums over the boundary mask) with a fixed `W`.
- Proposals, not mutations: a forward writes admission proposals (rung,
  code, witness, utility) into a fixed-size proposal buffer on the
  SubSpace; the owner commits them in `Reset` at the presentation boundary,
  outside the captured graph, exactly once (a committed proposal is marked,
  so a backward replay or a second pass cannot admit twice, and the tensors
  the backward needs are never mutated in place).
- Visibility: new ids, counts, tags and projected codes become visible at
  the next presentation, never mid-graph.
- Monotone projection scope: only the admitted row's containment
  neighbourhood (its constituents and its containing wholes on the current
  witness), so the edge set is linear in `M` rather than quadratic in the
  number of spans; the global projection stays an offline consistency pass.
- Measurements before any speed claim: graph breaks, recompilations, peak
  memory and bytes per second on identical checkpoints, corpora, hardware
  and budgets, against the Legacy `radix` path.

### 6. Persistent identity versus an occurrence's `.where`

A primitive concept must be reusable at another position, in another
sentence and after save / load; an occurrence must keep ordered spatial
bindings. Two keys:

- Persistent identity: `(rung, nearest codebook row)` with the row's
  admission radius as the metric; no absolute address in the key (or one
  concept per observation follows), and no unordered aggregate alone (or
  `12` and `21` collapse).
- Occurrence: `(identity, ordered .where bindings of its constituents)` on
  the carrier's witness; two occurrences of the same identity at different
  positions are the same concept twice.
- Prototype motion: when learned weights or the monotone projection move a
  row, the nearest-row metric is evaluated against the row's current code;
  an occurrence that no longer falls within the radius of its row is a
  miss, never a silent re-keying.
- Checkpoints: the aligned-protocol round-trip defect (math plan Phase 9)
  is a prerequisite of this contract, not an open question: ids,
  references, selected rungs, utility counts, boundary predicates and the
  next allocation round-trip through save / load; capacity exhaustion is
  tested without row recycling.
- Numerals: `12`, `21` and `11` use reusable digit concepts and distinct
  ordered occurrences; no stored concept per numeral string.

### 7. Chunk admission and idiomatic meaning

`ChunkLayer` adds its children and PEEL recovers a basis element with a
residual; that does not supply a non-compositional meaning or identify the
parse uniquely. The contract: an admitted phrase is a concept, that is a
concept id with a row in ConceptualSpace's concept dictionary (the
`nVectors` signed-unit atoms `cs_forward_content` reads; the table a
word's concept `A` gets a row in through `create_word_object_meta`),
initialised from the additive composition and then trained by the
reconstruction and answer costs like any row, so its meaning can diverge
from its parts; it is not a PartSpace percept row or a WholeSpace property
row. The witness keeps the parse (the constituent phrase ids and order),
so reconstruction is unaffected. Credit: the chooser's existing
`forwardGrammarWeight` is a bounded local structural contrast and is left as
it is; the new loss is the admission objective, the category-utility gain
of the phrase over its parts, which updates the boundary and admission
parameters, and its effect on the intended choices is shown by a test that
perturbs the utility and observes the chooser's pick change. Tests use
idiomatic and literal uses of the same phrase with frequency-matched
compositional controls; "stays two" means no atomic lexicalisation, not
prevention of the ordinary grammatical reduction.

## Chunking as a conceptual operation

Background: the basic level is the level of a taxonomy at which categories
carry the most information, have the highest cue validity, and are most
differentiated from one another (Rosch, Mervis, Gray, Johnson & Boyes-Braem
1976, "Basic objects in natural categories"; Rosch 1978, "Principles of
categorization"); it is the entry point of perception (Jolicoeur, Gluck &
Kosslyn 1984) and the most codable level in language. It is not fixed: with
expertise the subordinate level becomes as differentiated and as fast as
the basic level (Tanaka & Taylor 1991), so the level is learned per domain.
The computable form is category utility (Corter & Gluck 1992), the gain in
predicting a category's features from knowing the category (contract 4).

`chunk` is ONE conceptual operation, owned by ConceptualSpace, applied at
three sites with one criterion (contract 4) and one credit (contract 7):

1. **Subsymbolic loop.** At pass `t` ConceptualSpace reads the attended
   wholes at every rung and applies `chunk`: admit (or retrieve) the
   concept at the rung of maximal utility among admissible candidates. The
   loop continues from the chunked concept; "a bigger / smaller context" is
   one rung up or down. The recorded fold depth is the derivation; the
   utility counts are the estimate of the basic level, and their drift with
   evidence is the expertise effect.
2. **Word loop.** The units pushed to STM are the chunked concepts, bounded
   by `<fieldAttend>` under the coverage schedule of contract 2.
3. **Grammar (idioms).** The structural op `chunk(C, C)` is the same
   operation at the phrase level under contract 7.

## Loop placement

Every rung of both ladders and the `chunk` operation live inside the
subsymbolic and symbolic loops, that is inside the Spaces' `forward()` /
`reverse()` (the processing contract: calculations in Spaces' `__init__` /
`forward` / `reverse` / `Reset` only; data in SubSpaces). No new public
methods on Spaces; helpers are private and called only from those entry
points; the model orchestrator only sequences passes; durable mutation
happens in `Reset` per contract 5.

| loop | pass | forward (Spaces entry point) | reverse (mirrored) |
|---|---|---|---|
| subsymbolic | 0 | `InputSpace.forward`: byte atoms with `.where`, unity view (both exist) | `InputSpace.reverse`: bytes from the witness |
| subsymbolic | 0 | `WholeSpace.forward`: the whole tiling ladder from the learned predicates (unity at the top, byte-complete at the floor) and the min value per whole per rung | `WholeSpace.reverse` |
| subsymbolic | 0 | `PartSpace.forward`: rung 0 = max over each whole's atoms, witness written | `PartSpace.reverse`: witness replay (input) or domination against the atom rows with order from the parse (generated) |
| subsymbolic | t >= 1 | `WholeSpace.forward`: pi_t over each whole's meet at the rung-t tiling; `PartSpace.forward`: sigma_t over each whole's rung t-1 code, and the max over contained finer wholes for wholes first present at the rung-t tiling | the same `reverse` at the mirrored pass |
| subsymbolic | every t | `ConceptualSpace.forward`: `chunk` over the attended wholes (retrieve-k / attend-k, proposals by utility) | `ConceptualSpace.reverse` |
| symbolic | reduce | `LanguageSpace.forward(snapshot)` -> `SymbolSpace.forward` (the grammar chooser): `chunk` as a candidate op | `LanguageSpace.reverse(snapshot)` -> `SymbolSpace.reverse`: `chunk.reverse` (PEEL) |
| boundary | presentation end | `Reset`: commit proposals, update counts, at epoch end update boundary weights | |

The symbolic loop is called as `forward()` / `reverse()` like every other
Space. Today `SymbolSpace.forward` / `reverse` already wrap `compose` /
`generate`, but `LanguageSpace` exposes `compose(snapshot)` /
`generate(snapshot)` and the model calls `languageSpace.compose(...)` at
three sites. Phase 0 makes `LanguageSpace.forward` / `reverse` the public
entries (returning the reduction plan exactly as `compose` / `generate`
do), the model calls them, and `compose` / `generate` become private
helpers of the forwards.

Consequences for existing code: `fold_event_ladder`,
`_aligned_part_fold_ladder` / `_aligned_whole_fold_ladder` and the
`stage_analysis_spans` / `stage_word_property_weights` staging retire into
the forwards (rung outputs are the forward's event ladder, carried on the
SubSpace as the fold carriers already are). The eager `embed_stem` exists
only to keep host tokenization out of the compiled body; with no trie there
is no host tokenization, so the stem reduces to `InputSpace.forward` (byte
lexing) and `PartSpace.forward` runs inside the body. The compilation
boundary changes in content (contract 5), not in kind.

## Phases

First execution slice (Alec): Phases 0 through 2b, with a separate
acceptance checkpoint after each phase so that a single final successor
score cannot hide which replacement helped or broke; Phases 3 and 4 after
review.

### Phase 0 — Legacy move and the loop entry rename

Refresh the inventory. `radix` is already in Legacy's synthesis set and
dispatch; the remaining change is removing the canonical `meronomy`
dependency on radix (`synthesis_mode = "radix"` in the PartSpace
constructor) and migrating analysis ownership: add
`LEGACY_WHOLE_ANALYSIS_MODES` and a `stage_analysis_spans` legacy
dispatch for `byte` / `raw` / `sentence` / `word` / `grammatical`.
`model.xsd` keeps the enums with a "legacy" note; the `model.xml` default
`lexicon` resolves through Legacy. Rename the symbolic loop entries
(`LanguageSpace.forward` / `reverse`).

Acceptance: byte-identical behaviour, established by comparing parameter
names, checkpoint keys, masks, losses and outputs on the fixtures before
and after (a pinned-output test per touched fixture), not by unchanged
test counts alone.

Status (2026-09-10): landed. `LanguageSpace.forward` / `reverse` are the
entries (`compose` / `generate` delegate to them); the three model call
sites use `forward`; `Legacy.LEGACY_WHOLE_ANALYSIS_MODES` and
`stage_analysis_spans_legacy` dispatch the non-meronomy cuts, and the
canonical branch reads one mode; the PartSpace side needed no code (radix
was already a Legacy front end; `meronomy` stays radix-backed until
Phase 1). Pinned outputs (answers, symbolic state, staged spans) on
`MM_xor`, the verb fixture and the canonical successor config are
identical before and after. Tests: `test/test_meronomy_ladder.py`
(Phase 0 section).

### Phase 1 — synthesis as the fold ladder

Step 0 (prerequisite): fix the aligned-protocol checkpoint round trip of
concept identities (contract 6) with its own test.

1. Atoms are bytes: the PartSpace codebook's byte alphabet rows, `.where`
   = byte start (`InputSpace.forward`, exists).
2. Domains from the analysis tiling of the current pass: a part never
   crosses a whole boundary (with the digit predicate, `12` never fuses
   below the grammar).
3. Rung 0 = max over a whole's atoms with the witness written (contract
   1); rungs t >= 1 = the sigma folds over the whole's code (contract 2).
   Carrier `[B, W, M, D]` -> `[B, W, D]` per rung.
4. Admission by `chunk` under contract 4 (admissible after
   `<admissionCount>` sightings within `<admissionRadius>`, rung by
   utility), as proposals committed in `Reset` (contract 5); the admitted
   row records its rung, its witness surface and its local containment
   projection.
5. Selection: retrieve-k / attend-k over the ladder (`<fieldRetrieve>` 16,
   `<fieldAttend>` 8) with the coverage schedule (contract 2).
6. Reverse: witness replay for input reconstruction; for generated
   content the split through the learned rungs, domination against the
   atom rows at rung 0, and order from the parse (contract 1).
7. Word loop: identity per whole (contract 6) replaces the surface-keyed
   word concept; a whitespace word with two digit wholes pushes two
   concepts.

Files: `bin/Spaces.py` (`PartSpace.forward` / `reverse`, `InputSpace.reverse`
witness replay, `ConceptualSpace.forward` proposals, `Reset` commits),
`bin/Layers.py` (the witness and proposal carriers as SubSpace data; the
max join and min meet as private helpers of the forwards),
`bin/Mereology.py` (the local containment projection), `bin/Models.py`
(orchestration only;
the aligned fold-ladder wrappers and `embed_stem`'s PartSpace part retire),
`data/model.xsd` (`wholeCapacity`, `wholeLength`, `fieldRetrieve`,
`fieldAttend`, `admissionRadius`, `admissionCount`, `utilitySmoothing`,
`utilityMinCount`), `doc/Params.md`, `doc/Mereology.md`, `doc/Spaces.md`,
`doc/Componentization.md`.

Status (2026-09-10, step A landed): `PartSpace._embed_ladder_word_major`
is the canonical stem on the aligned serial path: the units are the
staged wholes (the model hands the tiling to PartSpace transiently, as it
does the growth callback), the atoms are byte rows with exact spans (the
witness is the existing ids / mask / offsets / part-spans record), no
longest match and no promotion run; an overlength unit is cut at the
residual capacity and marked. Rung 0 joins by max
(`MeronymicFoldAdapter.set_law = "max"` on the canonical adapters). The
non-word-major meronomy path is still radix-backed (pending). Fixture
`data/MM_ladder.xml` (3.4M parameters, BasicModel topology, successor
corpus, digit wholes); tests in `test/test_meronomy_ladder.py` (step A
section): units, byte atoms, byte-exact witness replay under permutation
and repetition, the capacity cut, the max law.

Status (2026-09-10, step D-0 landed): rung-0 admission by recurrence in
the ladder stem: a unit seen `chunkPromotionThreshold` times (the
admission count) gets a row seeded with its rung-0 code (the max over its
atom rows), queued and committed at the boundary flush, never while online
learning is frozen; digits are separate units so `12` is never admitted.
The word store therefore fills from units, not trie chunks
(`test_word_store.py` green). Successor corpus on the canonical config
with the ladder stem, 20 epochs: 19 % (the digit-whole floor was 21 %),
same per-fact pattern (digit count separated, digit identity not read),
so the stem is a non-regression and the identity work is step D proper.

Acceptance checkpoints, in order: (a) ordered carriers and byte coverage
(the contract 1 tests, including permutations, repeated bytes, `11`;
empty and overlength input; batch-row isolation; finite gradients); (b)
stable admission and checkpointing (admission at the recurring rung, no
double admission across a backward replay, frozen inventory under
evaluation, save / load round trip of contract 6); (c) the digit-identity
tests: matched-length facts distinguished by digit identity, reordered and
repeated digits, translation of an occurrence to another position; the
successor corpus on the canonical topology is a non-regression floor only
(21 %, attributed to digit count in the pilot report), with the per-fact
table showing two digit parts per two-digit word.

### Phase 2 — analysis as the descending ladder

1. Candidates: every WholeSpace property row, with the learned predicate of
   contract 3 (`b_p`, `s_p`) and the bool signature slab.
2. Rungs: the tiling ladder is computed at pass 0 from the predicates
   ordered by boundary weight (contract 2): the unity at the top, the
   byte-complete tiling at the floor, and one intermediate tiling per
   admitted predicate (on the first epoch: the canonical priors, or none
   under `<boundaryTypes>none</boundaryTypes>`); the output is the ladder of
   nested tilings with the min value per whole per rung.
3. Boundary-type admission: at epoch end (`Reset`), a property's boundary
   weight is updated by the utility of the wholes its cut yielded during
   the epoch (contract 4), so space wins for text, digit stands alone
   inside numerals, and letter flips stay below the basic level, none of
   them privileged in advance.
4. The meeting rule (contract 4): analysis gives synthesis its domains, a
   whole that synthesis cannot admit sends analysis down a rung (the
   within-whole division, generalised), and where a synthesized part and a
   divided whole coincide in extent and are both attested, that is the
   basic level and the callosum records the identity.

Files: `bin/Spaces.py` WholeSpace (`forward`: the tiling ladder from the
learned predicate; `Reset`: predicate update), `bin/Layers.py` (the
predicate parameters on the property SubSpace), `doc/Mereology.md`
"Analyzer" section, `doc/Architecture.md`.

Status (2026-09-10, step 1 landed): `WholeSpace.stage_analysis_spans`
(canonical branch) stages the two-rung tiling ladder at pass 0, the
space-bounded coarse tiling over the unit tiling, and the unit-to-whole
parent map (`_staged_tiling_ladder`, `_staged_unit_parent`); the unity
and the byte floor are implicit. The unit tiling follows the contract 3
priors: space and punctuation are boundaries, letter / digit flips are not
(`w0`, `abc123` are one unit each; the property-signature cut still
divides them as the WholeSpace view inside the unit), and the digit
singleton makes every digit a unit when `<digitWholes>` is on. The
learned predicates (contract 3) and the boundary-type admission (step 3)
are next; until then the two rungs are the canonical priors.

Status (2026-09-10, step 2 landed): the boundary / singleton predicates
are parameters on every canonical WholeSpace (`boundary_weight`,
`singleton_weight`, logits over the property rows, hard 0.5 threshold in
the cut; `_build_boundary_predicates`), initialised from the canonical
priors or, under `<boundaryTypes>none</boundaryTypes>`, all off; the unit
tiling is cut from them (`_predicate_unit_spans`) when the property basis
is on, byte-identical to the priors cut. Under `none` a sentence is one
unit with whitespace kept as content (the cold start of step 3). The
pinned state-dict keys of the meronomy configs moved by two per WS stage
(`test_dual_towers.py`). Step 3, the score update that lets the cold
start learn space as the basic boundary, is next; the chunk prior's score
update by utility gain (`utilityPriorRate`) and the utility-gain gate on
phrase admission landed with Phase 2b.

Status (2026-09-10, step 3 landed): the boundary learner
(`WholeSpace._observe_candidate_tilings`, `_update_boundary_predicates`,
`WholeSpace.Reset`): at every presentation the canonical cut also stages,
per property row, the tiling "cut where this row flips" (whitespace / pad
discarded), the discard-only tiling and the current tiling, and accrues
their wholes' surfaces; at the boundary a row's logit moves by
`boundaryLearningRate` times its tiling's score (recurrence of its
wholes minus `boundaryDensityWeight` times wholes per presentation)
against the current tiling's, a row whose flips add no cut beyond the
discard boundaries earning nothing and the discard rows earning the
discard tiling's score. The cold start is byte-complete. Test: under
`<boundaryTypes>none</boundaryTypes>` the successor corpus turns the
whitespace rows on within three short epochs and leaves the letter rows
off, and `plus` emerges as a unit. Limits: whitespace is still a
boundary-only (discarded) class rather than a whole type (demoting it is
a later step), the learner is a score update at the boundary rather than
autograd through the cut, and the singleton weights are not yet learned
(the digit singleton stays a prior).

Acceptance: the untagged cold start learns space as the basic boundary on
a small text corpus with the canonical fallback verified off; the digit
singleton is learned on the successor corpus; the tilings nest; the cut is
byte-identical to today's when the predicates equal the four classes.

### Phase 2b — chunk in the grammar (idioms)

The `chunk` structural op becomes a chooser candidate on the STM reduce
pass with admission under contract 7; a chunked phrase is one concept in
STM and in LTM slots with its own trained row; `chunk.reverse` (PEEL
against the store) plus the witness is its analysis.

Files: `bin/Language.py` (`ChunkLayer` role, the admission proposal, the
admission objective), `data/complete.grammar` (already lists the rules),
`bin/Spaces.py` (utility for phrases), `doc/Language.md`, `doc/STM.md`,
`doc/Training.md`.

Acceptance: idiomatic and literal uses of the same phrase with
frequency-matched compositional controls; the idiom is admitted with a row
that diverges from the additive composition, the control is not
lexicalised; perturbing the utility changes the chooser's pick.

Status (2026-09-10, mechanism landed): `data/ladder.grammar` is
`complete.grammar` plus the `chunk` compose / generate rules, so `chunk`
is a candidate of the CS reducer (`MM_ladder.xml` uses it; other configs
are unchanged). The STM mirrors each slot's coarser whole and unit
position (host-eager, like the slot kinds; the compiled tensor peer is
deferred); the reduce step passes a structural prior to the reducer
(`BinaryStructuredReductionLayer.forward(op_prior=...)`): `chunk` is
forbidden across wholes and carries the learned `chunk_prior` logit
(zero at init) inside one. A `chunk` chosen on a same-whole pair proposes
the pair's concept ids; `ConceptualSpace` commits utility counts
(contract 4: once per presentation, at the training path's boundary) and
admits a recurring proposal (`admissionCount`) as a concept over its
member concepts. Landed after: the utility-gain gate on admission (a recurring
proposal is admitted only when the phrase's utility exceeds its members'
best) and the score update of `chunk_prior` by that gain
(`utilityPriorRate`); the clause rung of the tiling ladder (runs bounded
by punctuation; the whole two adjacent words share), without which no
two words ever shared a whole and `chunk` never fired on text; the
inline idiom / literal fixture `data/MM_ladder_idiom.xml` (`kick the
bucket` vs `kick the ball`, frequency-matched), on which recurring
pairs are admitted with positive gains. Not yet done: the admitted
phrase's own row wired into the answer path so that the idiom's meaning
can diverge from the additive composition while the literal control
stays compositional (today an admitted phrase is a relation-only concept
over its members), and the compiled-path mirror. Tests: ladder grammar has
`chunk`; licensing only inside one whole; counts accrue once per
presentation; a recurring same-whole pair is admitted as a phrase.

### Phase 3 — configs, tests, docs

Fixtures naming `radix` / `word` move to `meronomy` / `meronomy` once the
Phase 1-2 gates pass; the 26 radix-touching tests are re-pointed or moved
to a Legacy test module; `doc/Mereology.md` "Convergence and Supported
Modes", the `doc/Params.md` `synthesis` / `analysis` rows and the What
specification's reconstruction / answer separation describe the ladder;
`README.md` links this plan. Historical radix results and newly
demonstrated behaviour are labelled separately in the benchmarks.

### Phase 4 — measurement against the radix path

On the successor corpus and on a text corpus (the B24 band), against the
Legacy `radix` oracle on identical checkpoints, corpora, hardware and
budgets: identity (an admitted part recurs to the same row),
reconstruction exactness, answer quality, bytes per second, compile and
startup cost, graph breaks and recompilations, peak memory, and the
digit-whole successor pair. Gate for removing the Legacy front ends: no
loss on identity, reconstruction or answers, and a throughput gain on
text; the oracle stays for one release after the gate.

## Documentation per phase

Runtime documentation is updated with each phase, not only in Phase 3:
[Architecture](../Architecture.md), [Componentization](../Componentization.md)
and [Spaces](../Spaces.md) for ownership, carrier shapes and loop
placement (Phases 1-2); [Language](../Language.md), [STM](../STM.md) and
[Training](../Training.md) for admission, retained constituents, credit
and update timing (Phases 1, 2b); [Params](../Params.md),
[Mereology](../Mereology.md) and the
[What specification](../specs/2026-07-27-teaching-modes-and-next-iteration.md)
for configuration, algebra and the reconstruction / answer separation.

## Resolutions (2026-09-11)

The gaps listed at the end of the first slice, resolved as decisions:

1. **What reaches the answer path.** A unit's STM slot holds its concept
   row's atom (its identity at the meeting rung) with its position band;
   digit identity reaches the answer through the grammar's composition
   of slots, not through a separate attended-rung feed. The answer seed
   stays the root idea. The experiment that tests this is the successor
   pair on the ladder path with the per-unit concepts now carrying atom
   identity (Phase 1 step 7 landed); its result decides whether the
   answer path's capacity, not the representation, is the limit.
2. **The admitted phrase's row.** An admitted phrase gets a codebook row
   (the concept row allocation the word concept uses), initialised from
   the additive composition of its members; the reduce step's `chunk`
   result snaps to that row when the pair matches an admitted phrase
   (the C-to-S round trip), so the row is what STM holds and what the
   reconstruction and answer costs train; the witness keeps the parse.
   That is what lets an idiom's meaning diverge from its parts while a
   literal control stays compositional, and it is the idiom test's
   remaining prerequisite.
3. **Utility across rungs.** One estimator (contract 4) for admitted
   and candidate categories; the improvised phrase formula is retired.
4. **Persistence.** The utility counts, phrase hits and admissions, and
   the boundary evidence ride the structural extras of the checkpoint
   and round-trip with it (contract 6); the predicates are parameters
   and already do.
5. **The compiled path.** Eager-only licensing of `chunk` is accepted
   for the slice; the provenance slab `(whole, unit, clause)` per STM
   slot becomes a fixed-shape tensor carried through the functional
   push in the next slice.
6. **Open questions.** Q1: two knobs, because they play different roles:
   the admission radius decides identity (nearest row), the LBG variance
   threshold decides division; neither is derived from the other. Q2:
   the coverage schedule shares the thinking loop's forced-closure
   pressure knob. Q3: the cold start is re-tested under the difference
   construction with the class concepts seeded and no byte floor.

Decisions (Alec, 2026-09-11) on the four questions the resolutions raised:

- **The sameness level is learned by the utility score**, the same
  learner as the boundary types (contract 3), not set by grammatical
  demand.
- **Whitespace units are presented** to the loop, and their null
  operation folds them at once; the loop step is perceptually
  important even where later conceptual analysis discards it, and the
  compiler may optimise the null step away later.
- **The seeded whole rows are splittable, and the seed is minimal.**
  Just as the part lexicon is seeded with every character as an atom,
  the whole lexicon is seeded with the smallest set of segmentations
  from which every tiling can be composed: the atomic wholes, one per
  character value (a run of that character), rather than the eight
  hand-tagged classes. Letter, digit, space and punctuation are then
  groupings the learner finds (coarser tilings that recur at lower
  density), with the eight class rows available only as an optional
  prior. Everything is a division or a grouping of the atomic wholes.
- **One geometry knob.** The admission radius (identity: nearest row)
  and the LBG variance threshold (division) keep their names, but the
  radius is defined from the variance: `admissionRadius` defaults to
  the square root of `lbgThreshold` (the standard deviation along the
  split axis), overridable.

Status (2026-09-11, second slice landed): the difference-typed
boundaries (`begins_weight`, `ends_weight`) and the per-column sameness
level (`atom_level`) over the atomic wholes (256 columns, one per byte
value) plus the property rows, cut from a bool signature slab
(`_predicate_unit_spans`), with the class rows as the `canonical` prior
and the atomic tiling as the `none` cold start; whitespace runs presented
as units under `<whitespaceUnits>` with the grammar's `null` unary
operation (`NullLayer`, in `ladder.grammar`); the boundary learner with
the memory-load score (recurrence minus `boundaryDensityWeight` times
density minus `boundaryTypeWeight` times distinct-per-occurrence),
greedy per update window (`boundaryUpdateEvery`), ties toward class rows
(test: a varied generated corpus turns a whitespace boundary on and
leaves letter bytes' boundaries off); LBG on the property inventory
(`record_property_pull` from each unit's rung-0 code,
`maybe_split_property_row` at the boundary, the new row's predicate
acquired from the pulls' bytes, `_grow_boundary_weights`); the phrase row
(allocated at admission from the additive composition, snapped to by a
matching chunk in the reduce step); persistence of the utility counts,
phrase hits / admissions / rows / gains and the acquired predicates
through the structural extras; `admissionRadius` defined from
`lbgThreshold`. Not done: the compiled provenance slab (eager-only
licensing stands), the vowel test (a corpus whose codes split the letter
row; the mechanism is exercised only by construction), and training the
phrase row through the answer path end to end.

Implementation of these resolutions was the second slice (status above); what remains of it: the difference-
typed boundaries and the sameness level (replacing the flip weight and
the singleton), whitespace as a null-operation unit, the seeded class
concepts, the LBG gate lifted on the property inventory with predicate
acquisition, the phrase row, the persistence of the counts, and the
compiled provenance slab.

## Open defects found on the way

- The Phase 2b provenance mirrors (slot wholes and units) are host-eager;
  the compiled tensor peer (`functional_push_step_masked`) carries no such
  slab yet, so `chunk` licensing is eager-only for now (contract 5).
- `BaseModel.dispatch_per_row_reset` on the aligned ladder fixture fails
  in `WholeSpace._whole_ancestors` (`taxonomy_parent_map` is absent under
  the property basis) before any fold-ladder code runs; the training
  path's sentence boundary (`runEpoch`) does not take that route. Tests
  drive boundaries through the training path until this is fixed.

## Open questions

- Q1 The admission radius and `lbgThreshold`: one knob or two.
- Q2 Whether the coverage schedule of contract 2 should share the thinking
  loop's forced-closure pressure knob or have its own.
- Q3 Whether category utility, normalised across rungs as in contract 4,
  is stable under the byte-complete cold start, or needs the canonical
  priors of contract 3 as a permanent floor.

## Verification

- Phase 0: pinned outputs, parameter names, checkpoint keys, masks and
  losses identical on the touched fixtures; the suite green.
- Phase 1: the acceptance checkpoints (a), (b), (c) above, each with its
  named tests in `test/test_meronomy_ladder.py`; the reconstruction
  round-trip and radix spell-out tests green in their Legacy scope.
- Phase 2 / 2b: the acceptance tests above; the type-run cut
  byte-identical under the four-class predicates.
- Phase 4: the measurement table in `doc/benchmarks/`.
