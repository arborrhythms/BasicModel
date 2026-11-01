# Mereology

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record. Record: [two truths §1, "The row, field by field"](specs/2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided); [bind plan §6–§6.3](plans/2026-10-19-bind-thought-operation.md); [occurrence spec §13–§14](specs/2026-09-30-occurrence-tense-aspect.md); [expectation spec rev 5](specs/2026-10-08-expectation-at-every-level.md).
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Parthood, subsumption and kind statements ("cats are animals") are **folds** in the meronomy and the taxonomy over symbols, not rows; equality is membership in a singleton; no `REL_PARTOF` row exists.
> - Rows: an LTM1 row is **one** concept slot; an LTM3 row is **three symbol references** (argument, kind, argument) of kind definition, attribution or conditional. The three-slot relation row with a REL code in the middle, null outer vectors, `refs`, `rel_type`, `role_mask`, `order`, `polarity`, `grammatical_mode`, `row_ids`, the address columns and the semantic sidecar are superseded: a row holds its slots, `.when` (model time, a column composition never reads), `truth_plus`/`truth_minus` (the lanes, renamed from `c_plus`/`c_minus`), the access count (renamed from `witness_count`) and `surprise`, and nothing else. **Amended (decided 2026-10-23, [step 2 §71](plans/2026-10-21-step2-one-ltm.md); recorded 2026-10-30):** an LTM1 row is two tagged references with its kind read from the referents, not three. As built at `3718d9d0dc`, `LTM0` still carries `c_plus`/`c_minus`, `source_trust`, `order`, `row_ids`, `origin`, `record_kind`, `polarity` and `witness_count` columns; only `refs`, `grammatical_mode` and the index columns are retired (audit 2026-10-30; decided 2026-11-01, Alec: the unused columns are removed and the row brought to this layout, [thinking spec §32.2](plans/2026-10-27-item2-5-thinking.md) item 17).


> **As built at `3718d9d0dc` (audit 2026-10-30; read second).** The body below is the
> record; where it differs, the code is this. Concepts are allocator ids with
> one row each in `similarity_codebook`; parthood, kind and definition
> statements are LTM1 `Part(member, whole)` rows owned by `ConceptAllocator.parts`,
> mirrored as sigma edges on the one `ConceptualAttentionLayer` (written by
> `cover_members`, taught by `DefinitionFold.loss` from the LTM1 lanes) and a
> centroid row; the META taxonomy (`insert_meta`, `taxonomy_parent_map`,
> `meta_pair_to_idx`, `meta_trust`, `part_chain`) and `record_cross_tower_meronomy`
> are gone. Concept composition is `fold_presence` (scatter min and max with
> $q^{w}$ and $1 - (1 - q)^{w}$); `PiLayer2`, `SigmaLayer2` and
> `MeronymicFoldAdapter` are library classes with no production consumer;
> `Ops.union` is an elementwise max and `Ops.intersection` an elementwise min
> where a zero coordinate is unspecified. `ImpenetrableLayer` is unwired and no
> `impenetrable*` knob is read. Percept parthood is `.where` span containment
> (`RunStructureLayer`, `FormContainment`); `Ops.partOf` has no caller. The
> property split is `record_property_pulls` / `maybe_split_property_row` with
> fixed thresholds; only `lbgThreshold` is read, for the admission radius.
> Interrogative mode is a structural `what` wrapper; `QueryLayer` and `queryPart`
> survive only in checkpoint migration.

Single-page reference for the mereological grammar: parthood as the
fundamental operation, the five mereological relations, and the
`ImpenetrableLayer` regularizer.

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

Mereology is the bridge from subsymbolic perception to explicit concept order.
Where an LLM usually leaves part/whole structure implicit in token statistics,
BasicModel stores and trains it as codebook geometry. This is the architecture's
closest point of contact with Formal Concept Analysis: part-percepts and
whole-percepts define a fuzzy extent/intent relation, and concepts are the
ordered links over that relation. DisCoCat enters one level later, when the
language layer composes these ordered meanings according to typed grammar.

> **Part/whole spaces (2026-06-12; base class retired 2026-07-10).** The
> perceptual side names the duality directly: `PartSpace` (bottom-up
> synthesis over atoms) and `WholeSpace` (top-down analysis over unity),
> both subclassing `Space` directly — there is no `PerceptualSpace` base.
> A thin `PerceptualSpace(Space)` base briefly existed (no
> params/submodules; only `NULL_PERCEPT_KEY` and isinstance sites) but was
> removed as part of the dual-towers rev-2 pyramid rework, once PS/WS
> became symmetric duals sharing one `forward(in_sub, CS_out)` signature.
> At the corpus callosum, objects are analysed and synthesized by sending
> them back through the towers — wholes get split, parts get chunked. For
> the object/reference (sign/symbol) vocabulary, see [Spaces.md](Spaces.md)
> and [Philosophy.md](Philosophy.md).

> **Meronomy reconciliation (2026-06-11; MeronomySpec wins — plan §2).**
> Four reframings of this page's claims:
>
> 1. **`part()` is the graded retrieval surrogate, not *the* parthood
>    relation.** Exact dominance (`Ops.partOf` — elementwise $\le$
>    reduced with `all`) is the semantic ground truth; the clipped
>    cosine below is its retrieval-time score. `part()` stays total
>    and form-level over everything — including symbol codes and
>    binding-table-bound rows — and that totality is its cordon duty
>    (spec §1, §7): geometric relations answer "looks like";
>    *asserted* parthood lives in the truth store and the towers'
>    registration, answers "is", and is a different call. Reading one
>    relation as the other is the category error.
> 2. **Fusion = elementwise max stays as the order-theoretic join.**
>    The learned $\sigma$-fold satisfies $\sigma \succeq \max$ (MeronomySpec §10.1) —
>    join versus parametric whole-maker; the whole-maker can only
>    over-cover the join, never under-cover it.
> 3. **The region partition (incl. exact `equal == 0` underlap) is a
>    scoring vocabulary only.** No abstraction classifier consumes it
>    (withdrawn, spec §9): no adjacency exists over witness
>    dimensions, so disconnection is meaningless there.
> 4. **Contiguity is sequential, never dimensional** (spec §2): it is
>    owned by `Mereology.Contiguous` on the derivational axis; $\sigma$
>    extents over witness dims are single lumps always.

> **2026-05-29 delta — binary GrammarLayer reverses take Basis, not W.**
> `Ops._binary_op_recommend(result, W, op_name, …)` (the
> mereology-guided recommender that walks `W` rows to find an
> operand pair such that $\mathrm{op}(x_1, x_2) \approx \mathit{result}$) is still keyed on
> the raw `W` tensor at the low-level kernel signature. But the
> public `UnionLayer.reverse(parent, basis=None)` /
> `IntersectionLayer.reverse(parent, basis=None)` now accept a
> Codebook / Basis object (typically `WholeSpace.subspace.what`)
> and extract `W = basis.getW()` internally before dispatching to
> `Ops.disjunctionReverse` / `Ops.conjunctionReverse`. The chart
> reverse / signal-router dispatch (`bin/Language.py::unreduce()`)
> passes `basis=space_role_basis` at the call site; no back-ref is stored
> on the layer.

*A superseded passage beginning "**Codebook IS the meronymic structure.** The standalone" moved to [DevelopmentHistory](DevelopmentHistory.md) (2026-10-30).*

---

## Parthood as Clipped Cosine Projection

Parthood is the single fundamental operation. Every other mereological
relation is defined in terms of it.

> **Terminology.**
> Below, $A, B$ are **percept** vectors (dimensionally-embedded codes in the
> `PartSpace`/`WholeSpace` codebooks); `part`/`whole`/`equal` are geometric
> relations over those percepts. "concept" is reserved for a `ConceptualSpace`
> relation tying one part-percept to one whole-percept (by reference), and
> "symbol" for a 0-D `SymbolSpace` reference to a concept.

For two percept vectors $A, B \in \mathbb{R}^D$:

$$
\mathrm{part}(A, B) = \frac{\max(0,\ A \cdot B)}{|A|\,|B|}
$$

The clipped cosine projection lies in $[0, 1]$ and satisfies Boole's
contrapositive:

$$
\mathrm{part}(A, B) = \mathrm{part}(-B, -A)
$$

because $(-B) \cdot (-A) = A \cdot B$ and norms are sign-invariant.

**Empty-operand contract (asymmetric).** `part(A, B)` returns $1$ when $|A|$
is near zero regardless of $|B|$ --- the empty set is part of everything
--- but returns $0$ when $|A|$ is non-empty and $|B|$ is near zero ---
nothing non-empty is part of the empty set. Empty-vs-empty
($|A| \approx |B| \approx 0$) resolves to $1$ (the $|A|$ branch wins the
tie). See `Ops._part_kernel` (`bin/Layers.py`).

---

## The Full Suite

Every member of the suite is expressible through `part`. Each `Basis`/`Space`
method takes a `scalar` flag (default `scalar=False`, the *vector* form used
inside the grammar layers, e.g. $\mathrm{part}(x,y) = x \cdot (y / \|y\|)$
elementwise); `scalar=True` collapses it to the clipped-cosine scalar in
$[0, 1]$ tabulated below, which is what every other formula on this page and
`ImpenetrableLayer` actually consume (`Basis.part(..., scalar=True)`).

| Method | Formula (`scalar=True`) | Interpretation |
|--------|---------|----------------|
| `part(A, B)` | $\max(0, A \cdot B) / (|A|\,|B|)$ | Fundamental. A is part of B. |
| `whole(A, B)` | $\mathrm{part}(B, A)$ | A contains B. |
| `equal(A, B)` | $\mathrm{part}(A, B) \cdot \mathrm{part}(B, A)$ | Mutual parthood, $[0, 1]$. |
| `overlap(A, B)` | $0 < \mathrm{equal}(A, B) < 1$ | Strictly-partial mutual parthood (indicator). |
| `underlap(A, B)` | $\mathrm{equal}(A, B) = 0$ | No mutual parthood (indicator). |
| `boundary(A, B)` | $|\mathrm{part}(A, B) - \mathrm{part}(B, A)|$ | Asymmetry of containment (zero under clipped cosine; for bases with asymmetric `part`). |
| `copart(A, B)` | $1 - \mathrm{part}(A, B)$, clamped to $[0, 1]$ | The part of B *not* accounted for by A. |

*A superseded passage beginning "**Exact dominance vs. retrieval score.** `part`/`whole`/`equ" moved to [DevelopmentHistory](DevelopmentHistory.md) (2026-10-30).*

---

## Region Partition

`equal(A, B)` partitions into three disjoint regions:

| Region | Condition |
|--------|-----------|
| Underlap | $\mathrm{equal} = 0$ |
| Overlap | $0 < \mathrm{equal} < 1$ |
| Identity | $\mathrm{equal} = 1$ |

Under clipped cosine, `part` is symmetric, so the three regions collapse to
a single scalar classifier. For a future `Basis` with asymmetric `part`, the
general five-relation table applies:

**Five-relation table** (general case, $\tau \approx 1$, $\epsilon \approx 0$):

| Relation | Condition |
|----------|-----------|
| `disjoint(a,b)` | $P[a,b] < \epsilon$ and $P[b,a] < \epsilon$ |
| `part(a,b)` | $P[a,b] > \tau$ and $P[b,a] < \epsilon$ |
| `part(b,a)` | $P[a,b] < \epsilon$ and $P[b,a] > \tau$ |
| `equal(a,b)` | $P[a,b] > \tau$ and $P[b,a] > \tau$ |
| `overlap(a,b)` | Both partial |

**Asymmetric subsumption.** Since `part` is symmetric under clipped cosine,
classical asymmetric subsumption is not encoded in raw magnitude. It is
recovered **relationally** via figure / ground: compare $\mathrm{part}(A, B)$
against $\mathrm{part}(A, \neg B)$. This is what makes Boole's contrapositive
hold exactly.

---

## Mereological Fusion

Fusion of a truth set is the **least upper bound** (LUB / join) over stored
truth vectors:

$$
\mathrm{fusion} = \max_i\, \mathrm{truths}[i]
$$

(elementwise max). In bivector space, the fusion vector names the top-right
corner of the axis-aligned bounding hyperrectangle dominating every stored
truth. Fusion is the geometric dual of luminosity: LUB (join) vs GLB (meet).

DoT is already baked into each stored truth (`stored[i] = activation_i *
degree_i` in `TruthLayer.record`), so fusion is trust-weighted automatically.

See [Logic.md](Logic.md) section **Fusion** for full discussion and the
leading-bivector / paired-index layout caveat.

---

*Moved to [DevelopmentHistory](DevelopmentHistory.md#impenetrablelayer-five-relations-regularizer) (2026-10-30): "ImpenetrableLayer: Five-Relations Regularizer".*

## Order-raising (building the meronymic lattice)

Gated behind `<mereologyRaise>` (default off $\to$ byte-identical).

*A superseded passage beginning "**Terminology note.** The cross-tower link that ties one par" moved to [DevelopmentHistory](DevelopmentHistory.md) (2026-10-30).*

The two towers stay **in-kind** — `PartSpace` *composes parts $\to$ parts*,
`WholeSpace` *analyses wholes $\to$ wholes* — and the **concept (META node) is the
cross-tower link**, associated with both an overlapping part-percept and
whole-percept (`insert_meta(ps_pos, ws_pos, fused_vec=None, *, ema=0.1,
trust=None)`; `ss_pos` is the pre-rename spelling). The native views meet
through the concept's definition over their memberships. At init PartSpace
has primitive parts and WholeSpace has supplied property rows defined over
primitive input. The attended field binds their location and level;
object-kind testimony belongs to item 7.

The live 11c decision uses evidence and geometry. A *both* reading prompts
refinement in the order-0 field. Its retained supporting occurrences are
attributed through the symbol definitions and tested by `RunStructureLayer`.
Contiguous support continues pi refinement; discontiguous support can be
symbolized one order higher by sigma after local refinement has stalled.
A large number of constituents is never a reason to raise. Ordinary
symbolization establishes an order-1 individual; context promotion beyond
that order uses `ConceptualSpace.maybe_raise_order` and the same convergence
gate. No WS fold-depth node is minted as a substitute for conceptual order.
See [Architecture](Architecture.md) and the
[11c residue receipt](benchmarks/2026-09-24-item10/README.md) for the rule,
patience parameter and controlled Felix/cat gate.

`delete_meta`, `unlink_child`, and `ps_children_of_whole` still maintain
native cross-tower bindings. These operations do not by themselves establish
an object kind; testimony remains item 7's responsibility. Retained brackets
belong to the attentive field, never to a permanent conceptual location.

### Automatic analysis: dividing wholes by LBG splitting {#lbg-division}

*A superseded passage beginning "Parts are synthesized automatically (the join over atoms, ev" moved to [DevelopmentHistory](DevelopmentHistory.md) (2026-10-30).*

**On the property inventory (2026-09-11).** The legacy split above is
gated off under the canonical property basis (its constructor calls the
eight-class inventory stable). The fold-ladder plan lifted that gate
with a dedicated pair on WholeSpace, `record_property_pull` and
`maybe_split_property_row`: after the ladder stem, each unit's rung-0
code (the max over its atoms) pulls on every property row its bytes
hold, with the unit's bytes recorded beside the pull (the stem records a
whole presentation at once through `record_property_pulls`, one
device-to-host copy and a byte-to-row lookup table, so the cost does not
scale with the packed row's units); at the sentence
boundary a row whose pulls' max per-coordinate variance exceeds
`lbgThreshold` after `lbgMinCount` pulls splits along its mean pull,
the fresh row taking as its **analyzer predicate** the bytes whose pulls
fell on its side (`_row_bytes`), so the predicate is read from the parts
rather than declared. The new row becomes a column of the boundary
learner at once (`begins` / `ends` / atom level start off). That is how a
whole such as "vowels" comes to exist without being defined: the letter
row's members separate in code, the row splits, the new row's predicate
is the vowel set, and whether "vowel begins" ever bounds anything is the
utility learner's decision.

**The boundaries themselves (contracts 2-3, landed 2026-09-11).** The
unit tiling is cut from a bool signature slab over the atomic wholes
(one column per byte value, the minimal seed of the whole lexicon, as
the byte rows are of the part lexicon) and the property rows: a
column's beginning or ending bounds a whole when its `begins_weight` /
`ends_weight` is on, the typed symmetric difference of adjacent
positions, so consecutive occurrences produce no boundary; a column at
the atom level (`atom_level`) makes every position it holds a whole by
itself (the digit rule without a digit-specific answer). Whitespace runs
are units like any other under `<whitespaceUnits>`, their grammatical
operation the `null` identity. The learner (`_update_boundary_predicates`)
scores each candidate move by the memory-load criterion, reuse minus
the working-memory cost (wholes per presentation) minus the long-term
cost (distinct wholes per occurrence), greedily against the current
tiling over a window of presentations, ties toward the more general
column; a varied text corpus turns a whitespace boundary on from the
atomic cold start and leaves single letters' boundaries off.

*Moved to [DevelopmentHistory](DevelopmentHistory.md#the-corpus-callosum-links-the-towers-part-isa-whole) (2026-10-30): "The corpus callosum links the towers (part `isa` whole)".*
**Design contract -- types from properties, word-sized `.where` (Alec, 2026-07-03).**
The enabled analysis methods are a set of PROPERTIES identifying the TYPES we
attend to, dividing the input (left of space, right of space, punctuation,
numbers) into word-sized objects. Each object's `.where` then either (a)
ASSOCIATES with a maximal part from PS -- a promoted percept covering exactly
that span -- or (b) is SENT BACK to PS to produce a parts-based definition
covering exactly that `.where` (spell-out bounded by the span).

### Explicit concepts $\longleftrightarrow$ implicit subsymbolic representations (dual-coded, 2026-06-21)

*A superseded passage beginning "**Terminology note.** This section formerly called the expli" moved to [DevelopmentHistory](DevelopmentHistory.md) (2026-10-30).*

The architecture carries **two coordinating structures** for the same content — an
**explicit relational** tower (the concepts) and an **implicit subsymbolic** one —
and order-raising is what keeps them in correspondence. We have *both*, not one or
the other.

- **Explicit (concepts).** A concept is **built directly** (not folded): an
  index-value relation entry tying a `PartSpace` part-percept code to a `WholeSpace`
  whole-percept code, with self-reference (`('sym', id)` members) so concepts nest
  into the taxonomy. Concepts associate part-percepts and whole-percepts **in virtue
  of a common whole**, and **identity on insertion is keyed to the exact ordered
  pair** — `WholeSpace.meta_pair_to_idx[(ps_pos, ws_pos)]` for `insert_meta`,
  `ConceptAllocator.relate_idx[(part, whole)]` (`bin/Layers.py`) for `relate` —
  **not** same-part-OR-same-whole: a repeat insert of the identical pair
  reinforces the existing concept, but a new pairing that merely shares a part
  or a whole with an existing concept mints a distinct one. `.where`
  **varies trial-to-trial** and is *not* the identity or trigger key — it keeps
  only its spatial/meronomy role (extent, containment). The Concept codebook
  (`taxonomy` / `taxonomy_parent_map` / `meta_pair_to_idx` / `meta_trust` /
  `part_chain`) is this structure. (This refines the earlier
  `.where`/co-occurrence framing above: the *trigger* to abstract is **many
  constituents under a common concept**, not a shared `.where`.) These
  relations are the INDEX side; the concept's subsymbolic content (the
  `ConceptDim` atom + its untyped sparse edge decomposition, populated at mint
  by `_populate_concept_weights`) is the representational side — see the next
  bullet.
- **Implicit (subsymbolic).** Each concept has one distributed code in the
  ConceptualSpace dictionary. The parallel knowing field carries independent
  positive and negative evidence per occurrence. `cs_forward_content`
  composes disjunctive parts by max independently on each pole.
  `conceptualPi` enables intersections within the order-0 field; their
  required contributions reduce by min over nonzero evidence per pole.
  Higher orders combine symbols through sigma and never hold pi edges. Exponents are
  nonnegative, and a negated part selects the opposite symbol pole. The
  taper admits both poles together. Final symbol readout unions occurrences
  and scales the two directions of the one code, preserving both separately
  from neither. The many-to-one abstraction lives on those part edges.
  [Architecture](Architecture.md#decided-in-direction-a-concept-is-sigma-over-pi-alec-2026-09-21)
  specifies the folds, discovery and reverse; the serial idea-vector path
  remains distinct from this paired conceptual field.

**Coordination.** Definitions persist by concept id, while the attended
order-0 field binds rows per turn. Native memberships read those definitions;
their distributed codes support similarity, retrieval and tied
reconstruction. Above order 0, a definition refers to symbols in the
preceding order. Symbolization raises the order. A large constituent count
does not trigger it: automatic context promotion beyond order 1 requires
discontiguous both support after stalled refinement. The explicit
definition and its code remain two views of the same concept, with no
second tower of permanent row identities.

---

## Why This Design

- **Parthood as projection.** One formula, Boole-contrapositive exact,
  continuous in $[0, 1]$. The old composite formula
  $\operatorname{conjunction}(1 - \operatorname{dist}(x, x \cap y),
  1 - \operatorname{dist}(y, x \cup y))$ mixed set-valued operands
  with a distance.

- *A superseded passage beginning "**Overlap penalty (not antisymmetry).** Legacy `Impenetrable" moved to [DevelopmentHistory](DevelopmentHistory.md) (2026-10-30).*

- **Trust via VQ EMA.** Already tracked for VQ commit loss.

- **Parthood is preserved by Pi / Sigma.** Under `monotonic=true`, Pi/Sigma
  are restricted to $W \geq 0$. User-supplied truth-set bivectors live on
  the non-negative paired `[pos, neg]` cone, where componentwise $\leq$ is
  the parthood partial order. Each lift / lower preserves parthood
  pole-by-pole for that truth surface. The bivector layout keeps
  contradiction `[1, 1]` distinct from ignorance `[0, 0]` under positive
  matmul. See
  [Spaces.md "Monotonicity of the lift / lower chain"](Spaces.md#monotonicity-of-the-lift--lower-chain).

---

## Mereological Algorithm: Meronymic Synthesis and Analysis {#mereological-algorithm}

> **Main algorithm.** One monotone synthesizer maps parts to wholes and one
> monotone analyzer maps wholes to parts. Together they build a codebook whose
> part/whole codes carry a partial order by construction: a concept lattice.
> The analyzer is the complement-mirror of the synthesizer, so the two are one
> adjoint design rather than unrelated procedures.

The live configuration surface is:

```xml
<PartSpace><synthesis>meronomy</synthesis></PartSpace>
<WholeSpace><analysis>meronomy</analysis></WholeSpace>
```

PartSpace owns bottom-up synthesis and WholeSpace owns top-down analysis. The
legacy `<chunking>` spelling and PartSpace `analyse` mode are rejected by code.
Other synthesis and analysis modes remain available.

### Relation to FCA, LLMs, and DisCoCat

The algorithm is the code-facing form of the Formal Concept Analysis analogy:
byte-span containment supplies the extent order, whole/property rows supply the
intent side, and learned part/whole links form a fuzzy concept lattice. Unlike
an LLM, the order is explicit rather than only an effect of token co-occurrence
hidden in weights. DisCoCat enters above this ordered substrate, where typed
grammar composes conceptual meanings; it does not define the perceptual order.

### The Adjoint Operators

Parts and wholes form a bounded lattice. The operators start from opposite
poles---`NOTHING` ($\bot$, all zero) and `EVERYTHING` ($\top$, all
one)---already named in [`bin/Layers.py`](../bin/Layers.py). (`ATOM` /
`UNIVERSE` were the pre-2026-07-02 names; both survive as back-compat
aliases.)

| Property | Synthesizer | Analyzer |
|---|---|---|
| direction | parts $\to$ whole | whole $\to$ parts |
| pole / start | $\bot$: all zero | $\top$: all one |
| operation | join: $0 \lor p_1 \lor p_2 \lor \cdots$ | meet: $1 \land c_1 \land c_2 \land \cdots$ |
| produces | dominating wholes (suprema) | subordinate parts (infima) |
| monotonicity | increasing: `whole > parts` | decreasing: `part < whole` |
| configuration | `<synthesis>meronomy</synthesis>` | `<analysis>meronomy</analysis>` |

They are exact De Morgan duals under percept complement:

```text
analyzer(x) = complement(synthesizer(complement(x)))
complement(x) = 1 - x
```

Implement the join-from-$\bot$ synthesizer; obtain the analyzer by reflection.
These are sigma/pi *lattice roles* (union from zero and intersection from one),
not the `SigmaLayer`/`PiLayer` butterfly folds. The perceptual butterfly folds are deleted. Rank is mereological level
in the same native cube; conceptual order counts symbolizations. See
[Percept Geometry](Spaces.md#percept-geometry-positive-unit-hypercube).

### The Order Lives in the Codes

The part/whole codes themselves are coordinatewise monotone on the `[0,1]`
presence cube: every whole dominates its parts. That ordered codebook is the
mereological embedding. In FCA terms, the monotone adjoints provide the Galois
closure; the ordering of the codes is what distinguishes a lattice from an
unordered lookup table.

SBOW is deliberately excluded here. It situates free conceptual relations by
distributional substitutability, whereas the mereological layer is grounded by
span order and nearest-row identity. Moving percept rows with SBOW would fight
the closure and could redirect token decode. See
[SBOW Situates Concepts, Not Percepts](Spaces.md#percept-sbow).

### The Part Specification: `.where` Containment

For text, parthood is byte-span containment:

```text
P is-part-of W  iff  span(P) is contained by span(W)
```

`record_cross_tower_meronomy` and `RunStructureLayer.contained_mask` read the
order from `.where` brackets. It is exact and complete for contiguous parts,
including a cross-boundary part that the local build did not combine:

```text
"abcd" = [0,4]   "ab" = [0,2]   "cd" = [2,4]   "bc" = [1,3]
[1,3] is contained by [0,4]  =>  "bc" is-part-of "abcd"
```

Thus a synthesizer may build `abcd` locally as `ab join cd` and still recover
the global constraint involving `bc`. Non-contiguous selections such as `ac`
lie outside this poset by design.

### Synthesizer: Build, Then Project

```text
synthesize(parts):
    seed  = bottom join part_1 join ... join part_k
    poset = {(whole, part): span(part) is contained by span(whole)}
    code  = project_monotone(seed, poset)
```

The join seed dominates the parts used in the local build, but it may initially
violate a cross-boundary constraint. Projection repairs the code against the
complete `.where` order. Max is idempotent, so repeating a part does not accumulate evidence.
Higher rank changes the extent or level while retaining the native cube.
Cross-order conceptual abstraction is sigma over symbols and symbolization.

### Redistribution as Isotonic Projection

For every containment edge and coordinate, require:

$$
\operatorname{code}(W)_i \geq \operatorname{code}(P)_i.
$$

The constraints are isotonic and decouple per coordinate. Redistribution is
therefore per-coordinate isotonic regression on the `.where` containment DAG:
project the codes to the nearest point that satisfies every part/whole edge.
The projection is convex and minimum-norm, so it moves only what the order
requires. A newly promoted part requires projection only in its `.where`
neighborhood.

The differentiable pressure is the Vendrov-style hinge
$\sum_i \max(0, P_i-W_i)$. When training content jointly, use projected
gradient: the hinge supplies pressure and the projection supplies the hard
guarantee. A test construction in which `join(ab, cd)` initially violated the
cross-boundary `bc` relation projected to zero violations while remaining in
`[0,1]`.

### Analyzer: the `1-x` Mirror

The analyzer is meet-from-$\top$, reflected through `1-x`, and produces
subordinate parts. It cuts a whole by boundary properties. The first live
approximation fixes word boundaries with `OR(space, punctuation)` via
`char_class_region([WHITESPACE, PUNCT])`. The fuller design grows a composable,
factorable basis of relational properties such as left/right-of-X and splits a
newly promoted whole until it reaches the word level.

`InputSpace` supplies both the atomic part view and the unity whole view.
`PartSpace` tokenization is selected by `<synthesis>`; WholeSpace division is
selected by `<analysis>`. `analysis=raw` preserves the whole byte buffer,
`analysis=word` uses the word boundary detector, and `analysis=meronomy` routes
the top-down role through the meronymic path.

### Convergence and Supported Modes

With `subsymbolicOrder > 0`, synthesis chunks small parts upward and analysis
splits large wholes downward. Each pass feeds the next until the two meet at an
operational basic level. The current boundary rule pins that level to words as
a first approximation; learning the more general Rosch-style basic level is an
open extension.

Since 2026-09-10 (doc/plans/2026-09-10-meronomy-fold-ladder.md) `meronomy`
is the only canonical mode on both sides: PartSpace synthesis is the fold
ladder (units = the staged wholes, atoms = bytes with an ordered witness,
rung 0 = the max over atoms, admission of recurring units at the boundary
flush) and WholeSpace analysis is the descending tiling ladder (the
four-class unit tiling with the digit singleton until the boundary
predicates are learned). The analysis side's boundaries are learned predicates over the
WholeSpace property rows (`boundary_weight`, `singleton_weight`),
initialised from the four classes or, under `<boundaryTypes>none</boundaryTypes>`,
from a byte-complete cold start, and moved at each sentence boundary by
the recurrence and density of the wholes each row's cut would yield (the
memory-load criterion: whitespace turns on first). The other synthesis
modes (`radix`, `lexicon`, `bpe`, `mphf`, `byte`) and analysis cuts
(`byte`, `word`, `raw`, `sentence`, `grammatical`) are legacy front ends
dispatched by `bin/Legacy.py`. The two towers are not duals: the join is the only OR in
perception, the meet over positions is the only AND, and negation enters
first at the conceptual level.
Rank is mereological level in both towers, and all ranks use the same
`[0,1]^D` membership cube. Conceptual order counts symbolizations.


### Live Routing and Implementation Boundary

The live reading boundary is `Models._lex_embed_stem`: InputSpace supplies the
byte slab and WholeSpace stages the property-defined unit spans; PartSpace's
meronomy ladder retains every unit's ordered byte witness and admits recurring
units into the percept store. The compiled word loop consumes the staged part
IDs, masks and offsets. Grammar operations consume object references above this
boundary; a white-space unit remains in perception and byte reconstruction.

The standalone `perceptual_analyzer.py` prototype, including its cosine-based
merge router and endpoint-sum span codec, was removed in the October 1 suite
trim. It is not the live routing implementation. Its exact byte-replay behavior
is covered by `test/test_meronomy_utf8.py` and `test/test_meronomy_ladder.py`;
within-whole division is covered on WholeSpace by
`test/test_within_whole_division.py`. The live start-only WhereEncoding and its
`test/test_where_bracket.py` tests remain. The use and coverage audit is in
[the trim receipt](benchmarks/2026-10-01-stage1-and-suite-trim/suite-trim/precursor-coverage-map.json).

The live span bookkeeping and ordered byte witness establish the path and order
specification. **(2026-06-25, LANDED standalone.)** The hard `join-from-bottom`
plus isotonic-projection guarantee is implemented in
[`bin/Mereology.py`](../bin/Mereology.py) (`join_from_bottom`, `meet_from_top`,
`project_monotone`, `mereological_synthesize`, `mereological_analyze`) and
exercised over projected trees in [`bin/Meronomy.py`](../bin/Meronomy.py)
(`MeronomyTree` and friends), with `test/test_mereology.py`,
`test/test_meronomy.py`, and `test/test_meronomy_laws.py` covering both. It is
not yet applied to the live percept codebook; the relational analyzer basis
and learned convergence criterion are the remaining next-boundary work. The
native max synthesis and conceptual SBOW remain unaffected.

Implementation references:

- poles, membership folds, and character-class regions:
  [`bin/Layers.py`](../bin/Layers.py)
- `record_cross_tower_meronomy` and `stage_analysis_spans`:
  [`bin/Spaces.py`](../bin/Spaces.py)
- `.where` containment edges, the join/meet/isotonic-projection primitives,
  and the mereological synthesizer/analyzer (`where_containment_edges`,
  `join_from_bottom`, `meet_from_top`, `project_monotone`,
  `mereological_synthesize`, `mereological_analyze`):
  [`bin/Mereology.py`](../bin/Mereology.py)
- the projected-tree exerciser (`MeronomyTree`, word/phrase synthesis over
  it): [`bin/Meronomy.py`](../bin/Meronomy.py)
- presence and complement geometry:
  [Spaces.md](Spaces.md#percept-geometry-positive-unit-hypercube)
- conceptual situating: conceptual-similarity-space plan

**Same module, different topic.** `bin/Mereology.py` — home to the
join/meet/projection primitives above — also carries the unrelated
`Mereology` mixin: the contemplative-awareness measure family (`Contiguous`,
`Continuous`, `Peaceful`, `Area`, `Luminosity`), mixed into `BaseModel` first
in MRO (`class BaseModel(Mereology, nn.Module)`, `bin/Models.py`). See
[doc/research/three-surfaces.md](research/three-surfaces.md) for that
geometry — the two share a file and a name, not an implementation.

### `isPart` as a Grammar Layer

`IsPartLayer` lifts parthood into the role-collapsed grammar as a binary
operator (arity 2). Under a query model it dispatches to `queryPart`
(`_dispatch_method_name_for_rule`), so `isPart` and `isEqual` are the
relative operators recognized by `_relative_start_categories`
(`_RELATIVE_OP_NAMES`). Its identity vector lives in the `WholeSpace`
operator codebook and participates in the same soft superposition as the
other operators.
