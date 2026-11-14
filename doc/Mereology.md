# Mereology

Single-page reference for parthood: the one fundamental operation and the
relations defined from it, the two places parthood is held (asserted, as
relations in memory, and geometric, in the codes), and the synthesis and
analysis that build the percept lattice.

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

Mereology is the bridge from subsymbolic perception to explicit concept order.
Where an LLM usually leaves part/whole structure implicit in token statistics,
BasicModel stores and trains it as codebook geometry. This is the architecture's
closest point of contact with Formal Concept Analysis: byte-span containment
supplies the extent order, whole/property rows supply the intent side, and
concepts are the ordered links over that fuzzy extent/intent relation.
DisCoCat enters one level later, when the language layer composes these
ordered meanings according to typed grammar; it does not define the
perceptual order.

The perceptual side names the duality directly: `PartSpace` (bottom-up
synthesis over atoms) and `WholeSpace` (top-down analysis over unity) both
subclass `Space` and share one `forward(in_sub, cs_out)` signature. Objects
are analysed and synthesized by sending them back through the towers: wholes
get split, parts get chunked. For the object/reference (sign/symbol)
vocabulary, see [Spaces.md](Spaces.md) and [Philosophy.md](Philosophy.md).

## Asserted and geometric parthood

Parthood is held in two ways, and reading one as the other is a category
error. Geometric relations answer "looks like"; asserted parthood answers
"is", and is a different call.

1. **Asserted parthood is a relation in memory.** A parthood, subsumption or
   kind statement ("cats are animals") is one relation, $Part(x, y)$ (Alec,
   2026-10-20/21): an `LTM1` row of two references with the truth pair, owned
   by `ConceptAllocator.parts`. It has no typed row, and no `REL_PARTOF` row
   exists; the row is read by the types of its arguments (`LTM1.readings`):
   two situations give a conditional, a situation and a speech individual an
   attribution, an object and a word a definition, two kinds a subsumption,
   and equality is membership in a singleton. `LTM1` and the taxonomy over
   symbols are therefore one structure, its edges kept as rows (explicit) and
   as folds (implicit) that the rows train: each row is mirrored as a sigma
   edge on the one `ConceptualAttentionLayer` (written by `cover_members`,
   taught by `DefinitionFold.loss` from the row's lanes) and a centroid row.
   The meronomy at order 0 is the perceptual case. The row layout is in
   [STM](STM.md#10-ltm-as-the-chain-of-stm-end-states).
2. **Exact dominance is the semantic ground truth among codes.** `Ops.partOf`
   is elementwise $\le$ reduced with `all`; `intuit`'s containment tests are
   its callers. `part()`, the clipped cosine below, is its graded retrieval
   score. It stays total and form-level over everything, symbol codes
   included.
3. **Fusion is the elementwise max, the order-theoretic join.** `Ops.union`
   is an elementwise max and `Ops.intersection` an elementwise min in which a
   zero coordinate is unspecified. There is no learned perceptual whole-maker
   beside the join: a part's code is the max of its constituents' codes.
4. **The region partition is a scoring vocabulary only.** No abstraction
   classifier consumes it, the exact `equal == 0` underlap included: no
   adjacency exists over witness dimensions, so disconnection is meaningless
   there.
5. **Contiguity is sequential, never dimensional.** It is owned by
   `Mereology.Contiguous` on the derivational axis; $\sigma$ extents over
   witness dimensions are single lumps always.

**Identity is decided by containment, not distance** (Alec, 2026-10-20).
`bind` compares a request $q$ with a candidate $c$ by the two meet ratios
$|q \wedge c| / |c|$ and $|q \wedge c| / |q|$ (`IdentityRows.meet_ratios`),
the meet being the coordinatewise minimum, and admits a candidate only when
one ratio is 1. An underspecified request (a pronoun, *the cat*) is contained
in every candidate of its kind, so location, recency and expectation decide
among them. A form's cue is likewise a learned region tested by containment.

**Selection is the mereological algebra.** Tense, aspect and the preposition
select by an idempotent meet with no inverse, so on the mat twice is on the
mat; a verb proper belongs to the other algebra, a change, a group element
with an inverse ([Logic](Logic.md#evidence-lanes)).

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
$[0, 1]$ tabulated below, which is what every other formula on this page
consumes (`Basis.part(..., scalar=True)`).

| Method | Formula (`scalar=True`) | Interpretation |
|--------|---------|----------------|
| `part(A, B)` | $\max(0, A \cdot B) / (|A|\,|B|)$ | Fundamental. A is part of B. |
| `whole(A, B)` | $\mathrm{part}(B, A)$ | A contains B. |
| `equal(A, B)` | $\mathrm{part}(A, B) \cdot \mathrm{part}(B, A)$ | Mutual parthood, $[0, 1]$. |
| `overlap(A, B)` | $0 < \mathrm{equal}(A, B) < 1$ | Strictly-partial mutual parthood (indicator). |
| `underlap(A, B)` | $\mathrm{equal}(A, B) = 0$ | No mutual parthood (indicator). |
| `boundary(A, B)` | $|\mathrm{part}(A, B) - \mathrm{part}(B, A)|$ | Asymmetry of containment (zero under clipped cosine; for bases with asymmetric `part`). |
| `copart(A, B)` | $1 - \mathrm{part}(A, B)$, clamped to $[0, 1]$ | The part of B *not* accounted for by A. |

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

**Reversing a join or a meet.** A max or a min is many-to-one.
`UnionLayer.reverse(parent, basis=None)` and
`IntersectionLayer.reverse(parent, basis=None)` take a codebook, typically
`WholeSpace.subspace.what`: with one, `Ops.unionReverse` /
`Ops.intersectionReverse` walk its rows through `Ops._binary_op_recommend`
for an operand pair whose join or meet approximates the parent; without one
the reverse raises. The layer stores no reference to the codebook.

`ImpenetrableLayer`, the five-relations regularizer over codebook rows, is a
library class that no production path constructs, and no `impenetrable*`
setting is read; its design is in
[DevelopmentHistory](DevelopmentHistory.md#impenetrablelayer-five-relations-regularizer).

---

## Order-raising (building the meronymic lattice)

Gated behind `<mereologyRaise>` (default off).

The two towers stay **in-kind**: `PartSpace` *composes parts $\to$ parts*
and `WholeSpace` *analyses wholes $\to$ wholes*, and the **concept is the
cross-tower link**: an order-0 concept's literals are basic percepts, each of
which names an occurrence's ordered parts and its whole boundaries (see
"Basic-percept premises" below). At init PartSpace
has primitive parts and WholeSpace has supplied property rows defined over
primitive input. The attended field binds their location and level.

The decision to raise uses evidence and geometry. A *both* reading prompts
refinement in the order-0 field. Its retained supporting occurrences are
attributed through the symbol definitions and tested by `RunStructureLayer`.
Contiguous support continues pi refinement; discontiguous support can be
symbolized one order higher by sigma after local refinement has stalled.
A large number of constituents is never a reason to raise. Ordinary
symbolization establishes an order-1 individual; context promotion beyond
that order uses `ConceptualSpace.maybe_raise_order` and the same convergence
gate. No WS fold-depth node is minted as a substitute for conceptual order.
See [Architecture](Architecture.md) for the rule and its
patience parameter.

`ps_children_of_whole` reads the native percept parts reachable through a
concept's word bindings. Such bindings do not by themselves establish an
object kind. Retained brackets belong to the attentive field, never to a
permanent conceptual location.

### Automatic analysis: dividing wholes by LBG splitting {#lbg-division}

Wholes are divided by splitting property rows on WholeSpace
(`record_property_pull`, `maybe_split_property_row`). After the ladder stem,
each unit's rung-0 code (the max over its atoms) pulls on every property row
its bytes hold, with the unit's bytes recorded beside the pull (the stem
records a whole presentation at once through `record_property_pulls`, one
device-to-host copy and a byte-to-row lookup table, so the cost does not
scale with the packed row's units). At the sentence boundary a row whose
pulls' largest per-coordinate variance exceeds a fixed threshold (0.5) after
at least eight pulls splits along its mean pull, the fresh row taking as its
**analyzer predicate** the bytes whose pulls fell on its side, so the
predicate is read from the parts rather than declared. `lbgThreshold` is
read only for the admission radius. The new row becomes a column of the
boundary learner at once (`begins` / `ends` / atom level start off). That is
how a whole such as "vowels" comes to exist without being defined: the letter
row's members separate in code, the row splits, the new row's predicate is
the vowel set, and whether "vowel begins" ever bounds anything is the utility
learner's decision.

**The boundaries themselves.** The
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

**Design contract -- types from properties, word-sized `.where` (Alec, 2026-07-03).**
The enabled analysis methods are a set of PROPERTIES identifying the TYPES we
attend to, dividing the input (left of space, right of space, punctuation,
numbers) into word-sized objects. Each object's `.where` then either (a)
ASSOCIATES with a maximal part from PS -- a promoted percept covering exactly
that span -- or (b) is SENT BACK to PS to produce a parts-based definition
covering exactly that `.where` (spell-out bounded by the span).

### Explicit concepts $\longleftrightarrow$ implicit subsymbolic representations

The architecture carries **two coordinating structures** for the same content, an
**explicit relational** one (the concepts) and an **implicit subsymbolic** one,
and order-raising is what keeps them in correspondence. We have *both*, not one or
the other.

- **Explicit (concepts).** A concept is **built directly** (not folded): an
  index-value relation entry tying a `PartSpace` part-percept code to a `WholeSpace`
  whole-percept code, with self-reference (`('sym', id)` members) so concepts nest
  into the taxonomy. Concepts associate part-percepts and whole-percepts **in virtue
  of a common whole**, and **identity on insertion is keyed to the exact ordered
  pair** (`ConceptAllocator.relate_idx[(part, whole)]`, `bin/Layers.py`),
  **not** same-part-OR-same-whole: a repeat insert of the identical pair
  reinforces the existing concept, but a new pairing that merely shares a part
  or a whole with an existing concept mints a distinct one. `.where`
  **varies trial-to-trial** and is *not* the identity or trigger key: it keeps
  only its spatial/meronomy role (extent, containment). The allocator's
  relation rows (`ConceptAllocator.parts`, the `LTM1` stratum) are this
  structure. These relations are the INDEX side; the concept's subsymbolic
  content (its code and sparse edge decomposition, written at mint by
  `_populate_concept_weights`) is the representational side (the next
  bullet).
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
  continuous in $[0, 1]$, with no set-valued operand mixed into a distance.

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
legacy front ends are listed under "Convergence and Supported Modes" below.

### The Adjoint Operators

Parts and wholes form a bounded lattice. The operators start from opposite
poles---`NOTHING` ($\bot$, all zero) and `EVERYTHING` ($\top$, all
one)---named in [`bin/Layers.py`](../bin/Layers.py).

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

The synthesizer is the join from $\bot$; the analyzer is obtained by
reflection. These are sigma/pi *lattice roles* (union from zero and
intersection from one), not the `SigmaLayer`/`PiLayer` butterfly folds. Rank
is mereological level in the same native cube; conceptual order counts
symbolizations. See
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

`RunStructureLayer` reads the order from `.where` brackets (its
`contained_mask`). It is exact and complete for contiguous parts,
including a cross-boundary part that the local build did not combine:

```text
"abcd" = [0,4]   "ab" = [0,2]   "cd" = [2,4]   "bc" = [1,3]
[1,3] is contained by [0,4]  =>  "bc" is-part-of "abcd"
```

Thus a synthesizer may build `abcd` locally as `ab join cd` and still recover
the global constraint involving `bc`. Non-contiguous selections such as `ac`
lie outside this poset by design.

A contiguous event takes its actual mereology. A sentence or clause whose
words form one run of the input is a whole in WholeSpace's sense: a maximal
run whose parts are the words and whose `.where` is the bracket. Its form is
the leaf-weighted centroid of its parts' forms (`compose_blocks`), drawn
toward the mean of the WholeSpace properties that witness the run
(`FormCentroids`). An event interrupted by another clause has no single run;
its centroid is then a type over disjoint parts, where the mereology means
less.

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
splits large wholes downward, each pass feeding the next. The two towers meet
at the basic level: the extent at which a part joined from below and a whole
divided from above coincide and are both attested. That extent is a basic
percept (`BasicPercepts`), whose row names the occurrence's ordered parts and
its whole boundaries; recurrence promotes it to a durable row. The level is
not fixed in advance: it moves with evidence per domain, as Rosch's basic
level moves toward the subordinate with expertise (see "Basic-percept
premises" below).

`meronomy` is the only canonical mode on both sides:
PartSpace synthesis is the fold ladder (units = the staged wholes, atoms =
bytes with an ordered witness, rung 0 = the max over atoms, admission of
recurring units at the boundary flush) and WholeSpace analysis is the
descending tiling ladder (the four-class unit tiling with the digit singleton
until the boundary predicates are learned). The analysis side's boundaries are
learned predicates over the WholeSpace property rows (`begins_weight`,
`ends_weight`, `atom_level`), initialised from the four classes or, under
`<boundaryTypes>none</boundaryTypes>`, from a byte-complete cold start, and
moved at each sentence boundary by the recurrence and density of the wholes
each row's cut would yield (the memory-load criterion: whitespace turns on
first). The other synthesis modes (`radix`, `lexicon`, `bpe`, `mphf`, `byte`)
and analysis cuts (`byte`, `word`, `raw`, `sentence`, `grammatical`) are
legacy front ends dispatched by `bin/Legacy.py`. As built the two towers are
not mirror images: the join is the only OR in perception, the meet over
positions is the only AND, and negation enters first at the conceptual level.
Rank is mereological level in both towers, and all ranks use the same
`[0,1]^D` membership cube. Conceptual order counts symbolizations.

### Live Routing and Implementation Boundary

The live reading boundary is `Models._lex_embed_stem`: InputSpace supplies the
byte slab and WholeSpace stages the property-defined unit spans; PartSpace's
meronomy ladder retains every unit's ordered byte witness and admits recurring
units into the percept store. The compiled word loop consumes the staged part
IDs, masks and offsets. Grammar operations consume object references above this
boundary; a white-space unit remains in perception and byte reconstruction.
Exact byte replay is covered by `test/test_meronomy_utf8.py` and
`test/test_meronomy_ladder.py`, within-whole division by
`test/test_within_whole_division.py`, and the start-only `WhereEncoding` by
`test/test_where_bracket.py`.

The live span bookkeeping and ordered byte witness establish the path and
order specification. The hard join-from-bottom plus isotonic-projection
guarantee is implemented standalone in
[`bin/Mereology.py`](../bin/Mereology.py) (`where_containment_edges`,
`join_from_bottom`, `meet_from_top`, `project_monotone`, `mereological_synthesize`, `mereological_analyze`) and
exercised over projected trees in [`bin/Meronomy.py`](../bin/Meronomy.py)
(`MeronomyTree`), with `test/test_mereology.py`, `test/test_meronomy.py` and
`test/test_meronomy_laws.py` covering both. It is not applied to the live
percept codebook, where the native max synthesis orders codes inside a whole
([Spaces](Spaces.md#fold-ladder)).

Implementation references:

- poles, membership folds, and character-class regions:
  [`bin/Layers.py`](../bin/Layers.py)
- `stage_analysis_spans`: [`bin/Spaces.py`](../bin/Spaces.py)
- presence and complement geometry:
  [Spaces.md](Spaces.md#percept-geometry-positive-unit-hypercube)
- conceptual situating:
  [SBOW Situates Concepts, Not Percepts](Spaces.md#percept-sbow)

**Same module, different topic.** `bin/Mereology.py` — home to the
join/meet/projection primitives above — also carries the unrelated
`Mereology` mixin: the contemplative-awareness measure family (`Contiguous`,
`Continuous`, `Peaceful`, `Area`, `Luminosity`), mixed into `BaseModel` first
in MRO (`class BaseModel(Mereology, nn.Module)`, `bin/Models.py`). See
[doc/research/three-surfaces.md](research/three-surfaces.md) for that
geometry — the two share a file and a name, not an implementation.

### `isPart` as a Grammar Layer

`IsPartLayer` lifts parthood into the grammar as a binary operator (arity 2).
It remains only for explicit historical grammars and checkpoint migration;
production grammars use the structural `part` / `whole` family, and an
interrogative is a structural `what` wrapper around the completed idea. A
rule is relative when its left-hand side is a declared relative start
category or its declared relation kind is relative (`_relative_rule_id_set`).

## Basic-percept premises

A basic percept is a percept with both parts and a whole, the extent at which
synthesis and analysis meet (above); the name follows Rosch's basic level, the
level of categorization at which members share a shape and a configuration of
parts, which is where part knowledge concentrates (Tversky and Hemenway
1984). Basic percepts take over the tying of parts to wholes from the first
symbolic layer, so no symbolic order needs a mereological read.

Mereological bounds belong to perception. `BasicPercepts.subspace_bounds`
takes the identity join of parts and the property meet of wholes.
`FormContainment.audit` checks those basic rows and their boundaries; symbolic
order-0 feature weights do not supply the containment premises.
`MereologicalCodes` reads the resulting basic codes and learned conceptual
context, with no direct two-tower symbolic form derivation. A word's form is
the weighted centroid of the percepts of its written definitions whenever any
definition weight is positive, including when the word owns an admitted basic
percept; only when every weight is zero is the form that basic percept's own
code, and a word with neither keeps the zero form (`MereologicalCodes.derive`;
Alec, 2026-11-10). The conceptual field reads live basic evidence, including
its complementary pole, so property memberships retain their gradient, and
the compose-time weighted presence rule applies. Reverse divides a concept's
demand over its written basic-percept alternatives by weight
([Spaces](Spaces.md#common-order-zero-reverse)).
