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

Mereology is the relation of part and whole among percepts. "Mereology is
part-whole relation at the perceptual level which, when it occurs over
symbols, becomes a taxonomy (because the sigma fold makes the concepts
abstract)" (Alec, 2026-11-20). Among percepts a whole holds its parts; among
symbols a kind holds its members. Percepts and concepts live in topological
spaces; symbols do not.

Parthood is held in two ways, and reading one as the other is a category
error. Geometric relations answer "looks like"; asserted parthood answers
"is", and is a different call.

1. **Asserted parthood is a relation in memory.** A parthood, subsumption or
   kind statement ("cats are animals") is one relation, $Part(x, y)$ (Alec,
   2026-10-20/21): an `LTM1` row of two references with the truth pair and
   an activity, owned by `ConceptAllocator.parts`. It has no typed row, and
   no `REL_PARTOF` row exists; the row is read by the types of its arguments (`LTM1.readings`):
   two situations give a conditional, a situation and a speech individual an
   attribution, an object and a word a definition, two kinds a subsumption,
   and equality is membership in a singleton. `LTM1` and the taxonomy over
   symbols are therefore one structure, its edges kept as rows (explicit) and
   as folds (implicit) that the rows train: each row is mirrored as a sigma
   edge on the one `ConceptualAttentionLayer` (written by `cover_members`,
   taught by `DefinitionFold.loss` from the row's lanes).
   The meronomy at order 0 is the perceptual case. The row layout is in
   [STM](STM.md#10-ltm-as-the-chain-of-stm-end-states).
2. **Exact dominance is the semantic ground truth among codes.** `Ops.partOf`
   is elementwise $\le$ reduced with `all`; `intuit`'s containment tests are
   its callers. `part()`, defined below, is its graded form: it is 1
   exactly where dominance holds. It stays total and form-level over
   everything, symbol codes included.
3. **Fusion is the elementwise max, the order-theoretic join.** `Ops.union`
   is an elementwise max and `Ops.intersection` an elementwise min in which a
   zero coordinate is unspecified. There is no learned perceptual whole-maker
   beside the join: the parts half of a percept's code is the greatest of
   its parts' codes, and its wholes half the least of its wholes' codes
   ("The Percept's Code" below).
4. **The region partition is a scoring vocabulary only.** No abstraction
   classifier consumes it, the exact `equal == 0` underlap included: no
   adjacency exists over witness dimensions, so disconnection is meaningless
   there.
5. **Contiguity is sequential, never dimensional.** It is owned by
   `Mereology.Contiguous` on the derivational axis; $\sigma$ extents over
   witness dimensions are single lumps always.

**Identity is decided by parthood, not distance** (Alec, 2026-10-20 and
2026-11-18). A request $q$ and a candidate $c$ are compared by parthood in
both directions, $\mathrm{part}(c, q)$ and $\mathrm{part}(q, c)$ as defined
below (`ConceptualMatch.match`). One is a part of the other when one of the
two is 1, and the two are the same when both are. `bind`, the merge of two
identities in MTM and the link of a memory admit a candidate when one is a
part of the other. An underspecified request (a pronoun, *the cat*) is a part
of every candidate of its kind, so location, recency and expectation decide
among them. A form's cue is likewise a learned region tested by containment.

**Selection is the mereological algebra.** Tense, aspect and the preposition
select by an idempotent meet with no inverse, so on the mat twice is on the
mat; a verb proper belongs to the other algebra, a change, a group element
with an inverse ([Logic](Logic.md#evidence-lanes)).

---

## Parthood as Shared Size

Parthood is the single fundamental operation. Every other mereological
relation is defined in terms of it. It is the parthood of *The Whole Part*
(pages 184 and 185; one of the books attached to the author's ORCID
record): the size of what two things share, over the size of the first.

> **Terminology.**
> Below, $A, B$ are **percept** vectors (dimensionally-embedded codes in the
> `PartSpace`/`WholeSpace` codebooks); `part`/`whole`/`equal` are geometric
> relations over those percepts. "concept" is reserved for a signed value
> that a Sigma gives over percepts or symbols
> ([Architecture](Architecture.md#sigma-and-pi-layers)), and "symbol" for a
> row of a Pi above order zero, which has no dimensions of its own.

For two vectors $A, B$ whose coordinates lie in $[0, 1]$:

$$
\mathrm{part}(A, B) = \frac{|A \cap B|}{|A|}
= \frac{\sum_i \min(A_i, B_i)}{\sum_i A_i}
$$

What two vectors share is the smaller value on each coordinate, and a size
is the sum of the coordinates. The value lies in $[0, 1]$. It is 1 exactly
when $A$ is at or below $B$ on every coordinate, which is the exact
dominance of `Ops.partOf`, and 0 when the two share nothing.

**Parthood is not symmetric.** $\mathrm{part}(A, B)$ is how much of $A$
lies in $B$, and $\mathrm{part}(B, A)$ how much of $B$ lies in $A$. The two
values together tell apart the five relations of two things:

| Relation | $\mathrm{part}(A, B)$ | $\mathrm{part}(B, A)$ |
|---|---|---|
| apart | 0 | 0 |
| $B$ inside $A$ | below 1 | 1 |
| $A$ inside $B$ | 1 | below 1 |
| the same | 1 | 1 |
| overlapping | between 0 and 1 | between 0 and 1 |

A likeness of two vectors, such as their cosine, is the same in both
directions and cannot tell a part from its whole: for a region and a part
that is three tenths of it, parthood gives 0.3 and 1, and the cosine gives
0.548 both ways. A cosine remains the measure where every concept is
compared at once; it is a likeness and not parthood (Alec, 2026-11-18).

**Signed values.** A negative and a positive value share nothing (Alec,
2026-11-18). On each coordinate two values of the same sign share the
smaller of their sizes, two values of opposite signs share zero, and a size
is the sum of the absolute values:

$$
\mathrm{part}(A, B) =
\frac{\sum_i [A_i B_i > 0]\,\min(|A_i|, |B_i|)}{\sum_i |A_i|}
$$

This is the unsigned formula applied to the stored form of a signed value,
its two unsigned lanes: the positive part on the for lane and the negative
part on the against lane. Where a concept is stored as two lanes it needs
no rule of its own. It is 1 exactly when every coordinate of $A$ lies
between zero and the same coordinate of $B$. A thing and its negation are
apart, $\mathrm{part}(A, -A) = 0$, and exchanging the lanes of both
changes nothing, $\mathrm{part}(-A, -B) = \mathrm{part}(A, B)$.

**Empty-operand contract (asymmetric).** `part(A, B)` returns $1$ when $|A|$
is near zero regardless of $|B|$ --- the empty set is part of everything
--- but returns $0$ when $|A|$ is non-empty and $|B|$ is near zero ---
nothing non-empty is part of the empty set. Empty-vs-empty resolves to $1$.
The readers that decide identity take an empty first operand as no
evidence: an empty description is offered nothing, and
`ConceptualMatch.match` returns 0 for it.

**In the source.** One function computes it: `ConceptualMatch.match`
returns parthood in both directions. The readers that decide identity call
it on the stored lanes. `Ops._part_kernel` with `scalar=True`, the scalar
read of the grammar's `part`, calls it on the two lanes of each operand,
its positive part and its negative part, and keeps the empty-operand
contract above (`bin/Layers.py`).

---

## The Full Suite

Every member of the suite is expressible through `part`. Each `Basis`/`Space`
method takes a `scalar` flag (default `scalar=False`, the *vector* form used
inside the grammar layers, e.g. $\mathrm{part}(x,y) = x \cdot (y / \|y\|)$
elementwise); `scalar=True` collapses it to the scalar in $[0, 1]$
tabulated below, which is what every other formula on this page consumes
(`Basis.part(..., scalar=True)`).

| Method | Formula (`scalar=True`) | Interpretation |
|--------|---------|----------------|
| `part(A, B)` | $|A \cap B| / |A|$ | Fundamental. How much of A lies in B. |
| `whole(A, B)` | $\mathrm{part}(B, A)$ | How much of B lies in A. |
| `equal(A, B)` | $\mathrm{part}(A, B) \cdot \mathrm{part}(B, A)$ | Mutual parthood, $[0, 1]$; 1 is identity. |
| `overlap(A, B)` | $0 < \mathrm{equal}(A, B) < 1$ | Strictly-partial mutual parthood (indicator). |
| `underlap(A, B)` | $\mathrm{equal}(A, B) = 0$ | No mutual parthood (indicator). |
| `boundary(A, B)` | $|\mathrm{part}(A, B) - \mathrm{part}(B, A)|$ | Asymmetry of containment. |
| `copart(A, B)` | $1 - \mathrm{part}(B, A)$, clamped to $[0, 1]$ | The share of B that A does not account for. |

---

## Region Partition

`equal(A, B)` partitions into three disjoint regions:

| Region | Condition |
|--------|-----------|
| Underlap | $\mathrm{equal} = 0$ |
| Overlap | $0 < \mathrm{equal} < 1$ |
| Identity | $\mathrm{equal} = 1$ |

The product does not say which of the two is the part. The two directions
of `part`, with $P[a,b] = \mathrm{part}(a, b)$, give the five relations
($\tau \approx 1$, $\epsilon \approx 0$):

| Relation | Condition |
|----------|-----------|
| `disjoint(a,b)` | $P[a,b] < \epsilon$ and $P[b,a] < \epsilon$ |
| `part(a,b)` | $P[a,b] > \tau$ and $P[b,a] \le \tau$ |
| `part(b,a)` | $P[a,b] \le \tau$ and $P[b,a] > \tau$ |
| `equal(a,b)` | $P[a,b] > \tau$ and $P[b,a] > \tau$ |
| `overlap(a,b)` | Both partial |

**Subsumption is in the measure.** $\mathrm{part}(A, B) = 1$ with
$\mathrm{part}(B, A) < 1$ is $A$ inside $B$; no comparison with a negation
is needed. Exchanging the lanes of both operands leaves parthood unchanged.
The contrapositive holds for full parthood under the complement $1 - x$ of
unsigned values: $A \le B$ exactly when $1 - B \le 1 - A$. It does not
hold for the exchange of lanes, which gives $\mathrm{part}(B, A)$.

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
cross-tower link**: the first Pi alone reads the codes, and an order-0
concept reads the basic percepts' rows, each a row of the first Pi over its
parts and its wholes, through the Not (see
"Basic-percept premises" below). At init PartSpace
has primitive parts and WholeSpace has supplied property rows defined over
primitive input. The attended field binds their location and level.

An order is raised in two ways (Alec, 2026-11-19).

*Implicitly:* any concept formed above symbols is one order above the
concepts those symbols read. An order rises at a Sigma and nowhere else.

*Explicitly, by statements that share a side* (Alec, 2026-10-21: "'Is' does
not cross order by itself, but when used with multiple identical RHS or LHS,
it creates an equivalence class"). "Felix is a cat" and "Ciao is a cat" are
two part relations kept in LTM with one whole. They call for a concept that
ranges over both, so a second concept of cat is formed one order higher. MTM
is where it is made: the kind collects the members' symbols, and where a
member has no symbol MTM makes one in nominating the kind, after looking for
one that already holds it ([Orders](Orders.md), "Orders are learned").

A reading that is both, for at one of a concept's percepts and against at
another, still prompts refinement in the order-0 field
(`ConceptualSpace.refine_over_collected`): its retained supporting
occurrences are tested by `RunStructureLayer`, contiguous support continues
the refinement, and discontiguous support can be symbolized one order
higher after local refinement has stalled. A large number of constituents
is never a reason to raise. No WS fold-depth node is minted as a substitute
for conceptual order.

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

- **Explicit (relations).** What is asserted of concepts is held as
  relations in memory: the `LTM1` rows of two references
  (`ConceptAllocator.parts`), read by the types of their arguments
  ("Asserted and geometric parthood" above). `.where` varies from one
  occurrence to the next and is never an identity or a key: it keeps only
  its role as a place.
- **Implicit (the layers).** The same content is held in the matrices of
  one structure, a Pi, a Not and a Sigma, at every order
  ([Architecture](Architecture.md#sigma-and-pi-layers)). A row of a Pi is a
  basic percept at order zero, or a symbol above it, a thing: it is made
  implicitly by the Pi or nominated by MTM when its constituents always
  occur together, and each row brings one concept of the Sigma above it to
  start with. A basic percept's constituents are coordinates of perceptual
  space and a symbol's are coordinates of conceptual space; each row is a
  cone over the coordinates its members share (Alec, 2026-11-20). The Not
  offers each row's value and not-value, and a concept is one signed value,
  $y_c = \tanh(\sum_i w_{ci} s_i)$, a kind, whose size is its certainty.
  Its weights are at zero or above; the Not is the only place a sign is
  changed, and a Not entry is a concept's entry that reads a row's
  not-value. The many-to-one abstraction lives on the Sigma's entries; the
  Pi's entries at order zero are the mereology, a percept over its parts and
  wholes.

**Coordination.** An asserted relation and the layers are two views of one
content: a definition between concepts ("balloons are rubber") is an edge of
the Sigma taught from its `LTM1` part row (`DefinitionFold`), and kept under
the penalty that thins every entry no cost supports. Above order 0 a kind
collects from every epistemic level below its own, so orders are not
strictly ramified, while a symbol's cone lies in the conceptual space of one
order. The order rises only at a Sigma, made implicitly or nominated by MTM,
never by the number of constituents.

---

## Why This Design

- **Parthood as shared size.** One formula, not symmetric, continuous in
  $[0, 1]$ and exact dominance at 1, with no set-valued operand mixed into
  a distance.

- **Parthood is preserved by Pi and Sigma while no Not entry is read.** A
  map whose weights are all zero or above keeps the order of its inputs:
  where one vector is at or below another in every coordinate, so are
  their outputs. Checked on 4,000 such pairs, this holds for a Pi with any
  entries of zero or above, for a Sigma $\tanh(\sum_i w_{ci} s_i)$ with
  every $w_{ci}$ at zero or above, and for the union $1 - \prod(1 - x)$, and fails as soon as the Sigma has
  one Not entry. A Pi's output lies at or below every input it reads and
  the union's at or above, so the Pi makes the order below and the union
  the order above. Monotone weights cannot make evidence against; that
  takes the against lane or a Not entry. The two lanes keep "both" `[1, 1]`
  apart from "neither" `[0, 0]`. See
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

### The Percept's Code: Parts and Wholes on Two Lanes {#percept-code}

Alec, 2026-11-19: "parts build bottom-up with zeros as a min, and wholes
build top-down with ones as a max. This coincides with parts building using
max/sums, and wholes building using min/products."

**The layout** (Alec, 2026-11-19: "64 and 64 until proven that we need
more"). A percept's code has a parts half and a wholes half of 64
coordinates each, and each half has two lanes, evidence for and evidence
against, every value within 0 to 1. A "negative number" in perception is
evidence against: Alec's -1 is 1 on the against lane, and zero on both lanes
is no evidence. No coordinate is set aside for a length or a felt value.

| | Parts half | Wholes half |
|---|---|---|
| starts at | all zeros, the bottom | all ones, the top |
| built | upward, from its parts | downward, from its wholes |
| a percept's code | the greatest of its parts' codes | the least of its wholes' codes |
| over the percepts of one reading | the union, $1 - \prod (1 - v)$ | the product, $\prod w$ |
| for lane | the lower bound: the parts it has | the upper bound: the wholes it lies in |
| against lane | the upper bound, as evidence against: the parts that the forms holding it add | the lower bound, as evidence against: what its upper bound cuts away |

The greatest and the least are taken for a code, which is a bound; the
union and the product where percepts are present by degree and are joined
in one reading. On values of 0 and 1 the two pairs agree. So a percept's
code is at or above each of its parts' codes and at or below each of its
wholes' codes (Alec, 2026-11-14).

**A form's code is the set of its parts** (Alec, 2026-11-19). A part is a
run of adjacent characters. A form's parts are its characters and the longer
parts that exist and lie inside it. The space is a character like a letter,
"so that the saliency map can select against it from within parts", and a
form may hold one (`ice cream`). No end marks are kept, so a form has one
code wherever it stands: `is` alone and `is` inside `this`. A longer part is
added only where two forms would otherwise have the same parts ("we should
also do pairs only when needed"): the part added is the shortest run that
one of them holds and the other does not (`ry` for `archer` and `archery`).
A part once added is carried by every form that holds that run, those
admitted before it and after it. So parthood between forms is exact: a form
found inside another lies wholly inside its code. The exact code is the list
of the form's parts (`WordIdentity.atoms_of`); the row of fixed width is its
fold, each part's fixed bits drawn over the 64 coordinates
(`WordIdentity.atom_code`). Of the first 19,821 distinct words of a
dictionary, 4 share a code at 64 and 64 with the wholes half below, and none
once the next part is added for them.

**What tells against a form** is its strict wholes in the part tower and
nothing else (Alec, 2026-11-19: "only strict wholes ( in the part tower, not
the property tower ) are evidence against"). Its against lane in the parts
half holds the coordinates that its holders add, a holder being a stored
form that holds it as a run: the `y` of `archery` for `archer`. The holders
that count are the nearest, those that add the fewest characters
(`WordIdentity.nearest_holders`), since a short form is held by hundreds of
words; a form then has 0.72 against entries on average and at most 32.
Where a coordinate would be both, for stands.

**A whole is a row of the table of wholes** (Alec, 2026-11-19: "we should
be storing it as a codebook. Starting from the set of all values, we
discriminate at some boundary"). A whole is defined by what belongs to it: a
set of characters for a class of characters, a number for a feeling. Its
code is the OR of the codes of the catalog parts that belong to it, the
catalog being drawn a second time in the wholes half's own coordinates
(`WordIdentity.catalog_code`, `PrimitiveProperties.codes`). The whole catalog
OR'ed is the class of all characters, all ones, which every percept lies in
(Alec: "A percept cannot have no wholes"). So a class inside another lies
at or below it with no rule for it, and a class of characters is the OR of
the characters taught to it ("that would give a principled way to build
character classes"). A whole may also be given by other wholes ("just like
parts build from parts, properties may express in terms of other
properties"): its code is the AND of theirs and lies inside each
(`PrimitiveProperties.define_by_rows`: "capital letter" by "letter" and
"capital"). Each known form is its own tightest whole, the OR of its parts'
catalog codes (`WordIdentity.tightest`). Measured at 64 coordinates, seven
classes of characters coded by OR (all characters 64 ones, letter 57,
capital and lower case 44 each, digit and vowel 24 each, whitespace 11)
are each at or below every class they lie in, and no other pair is so by
accident. A class of whole words fills the 64 coordinates quickly (one word
31 ones, five words 57), so a class of more than a few words bounds almost
nothing.

**The wholes half is an upper bound.** Under the class of all characters
alone nothing is cut away and the half says nothing. A reader that asks
whether one form lies inside another reads the for lane alone, where the
part order holds. The code is read by the first Pi alone; no concept reads
it.

**A scalar is the same mechanism over the bins of its value** (Alec,
2026-11-19: "the exact same mechanism, except part growth and whole
contraction become the same operation on scalars"). The catalog is the
value's bins in order, 64 bins of the felt value. The parts half is the bins
at or below the value; the wholes half is all ones with the bins above it
cut away; for an exact value the two are one code, and the against lane is
the bins above. On a line the union of two parts codes is the code of the
greater value and the intersection of two wholes codes that of the lesser,
so growing the parts and contracting the wholes move one cut. On two
dimensions they differ: of the 2,016 pairs of boxes on an 8 by 8 grid the
intersection is a box in all, and the union in only the 1,232 where one box
lies inside the other.

**The analysis from above is conceptual attention** (Alec, 2026-11-19: "we
need something driving that analysis top-down just like synthesis drives
bottom-up part construction"). No Sigma writes into the wholes half of a
code. A request over concepts is carried down by the transposes of the
matrices of the way up, through the Sigma's weights, the Not and the first
Pi's own entries, onto every member of a row, its parts on the part tower
and its properties on the property tower, and onto its where
([Architecture](Architecture.md#one-attention)). Two forms that no class of
characters tells apart, such as `archer` and `archery`, are told apart by
the next part above.

**The way back of a form starts from the form alone** (Alec, 2026-11-19: "we
are not always doing reconstruction: sometimes we do generation"). No place
of a part is taken from a reading. "The whole stores its parts in order, and
longest first": the store keeps each form's characters in order
(`RadixLayer.inverse_table`), and a code that is no stored form has no
spelling.

**A string read as a word it is not is taught when it is observed** (Alec,
2026-11-19). Nothing is added to a form's code for it. What is admitted is
the part that tells the string from the word, and not the string ("We don't
want to admit Felixx, we want to admit the part that discriminates it from
Felix: "xx""), by an idempotent round over cases gathered from the training
harness's logs (`WordIdentity.admit_part`, `part_between`).

### The Part Specification: the Set of Parts

For text, parthood between forms is read from their codes: one form is a
part of another exactly when its set of parts lies inside the other's
("The Percept's Code" above), and so when its code lies wholly inside the
other's on the for lane:

```text
P is-part-of W  iff  parts(P) is contained by parts(W)
"bc" in "abcd":  {b, c} is contained by {a, b, c, d}
```

`.where` is a place and nothing more. There is one where, the place on the
constant sinusoid; there is no second where for how a thing's parts stand to
one another, and no test of spans decides a part
([Spaces](Spaces.md#percept-modalities)). The order inside a form is in its
code only as far as it tells known forms apart, so a string that is no known
form may read as one until the part that tells them apart is taught.

The standalone synthesizer below takes its containment edges as given
(`where_containment_edges`), so a test may build `abcd` locally as
`ab join cd` and still recover the constraint involving `bc`.

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
subordinate parts. It cuts a whole by boundary properties. A percept's wholes
half is this analyzer's result: all ones under the class of all characters,
cut down by each narrower whole it lies in. The first live
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
percept (`BasicPercepts`), a row of the first Pi over its parts and wholes,
which becomes a row only when MTM nominates it on recurrence, never on a
first reading. The level is
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
legacy front ends dispatched by `bin/Legacy.py`. The two towers mirror each
other in the code: the parts half grows from all zeros by the OR of its
parts, and the wholes half shrinks from all ones by the AND of its wholes.
Evidence against is perceived, on each half's against lane, and no percept
is negated: a sign is changed only by the Not, through a Not entry.
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
takes the greatest of a percept's parts' codes for its parts half and the
least of its wholes' codes, its own tightest whole among them, for its
wholes half; `BasicPercepts.against` gives its against lane.
`FormContainment.audit` checks those rows and their boundaries.

**A basic percept is a row of the first Pi** over its parts and wholes
(`PerceptPi`, `bin/OrderZero.py`). The row selects for and against (Alec,
2026-11-19: "Pi should select both strong for and strong against"). Its for
entries are its parts and wholes, joined by the product, so each must be
present. Its against entries are the parts that its nearest holders add,
joined by the union, so any one present tells against it: the row of
`archer` reads for 1 and against 0 on `archer`, and for 1 and against 1 on
`archery` (`PerceptPi.opposed`). In a row the run of parts is the percept's
presence, read wherever it lies in the extent and the same on both lanes,
since a run of parts has no against of its own. The wholes are read where
the run lies, on their own lanes: the for lane is the product of the wholes'
for lanes, and the against lane the union of their against lanes, since one
whole witnessed not to hold denies the row. So the character `7` under the
property "capital" reads for 0 and against 1. A single row is for or
against; "both" stands across the rows of one reading, one row carrying a
property for and another against (Alec, 2026-11-19).

A row exists only when MTM nominates it on recurrence, and each nominated
row brings one concept of order zero to start with. A basic percept is an
occurrence, an event (Alec, 2026-11-20: "all of their constituent percepts
occur at the same time, so there is no basis for abstraction into a
noun"): the word over its occurrences is its concept of order zero, the word
as a thing is a symbol over that concept, and a kind of words, such as noun,
collects those symbols. Its constituents are coordinates of perceptual
space, and its row is the cone over the coordinates its occurrences share. The tower of parts and wholes below
the basic percepts is perception, made on first sight. Parts and wholes
unite at the basic percept, and above it only symbols are collected. So a
concept above order zero carries no perceptual form: neither a centroid of
the percepts it collects nor a copy of its word's code. Alec, 2026-11-12:
"symbols should not take the centroid of the percepts that they collect;
they are epistemic points." A reader that needs such a concept's form takes
its word's, through the definition that maps the word to it. The way back
inverts the first Pi exactly and gives each percept its parts
([Spaces](Spaces.md#common-order-zero-reverse)).
