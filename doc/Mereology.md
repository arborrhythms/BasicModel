# Mereology

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
> on the layer. See
> [doc/old/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md](old/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md).

> **Codebook IS the meronymic structure.** The standalone
> `MereologicalTree` sidecar that formerly stored explicit parent /
> equality links was retired in favour of pure-geometric parthood
> on the `WholeSpace` bivector codebook. The grammar layers
> `PartLayer`, `IsEqualLayer`, `EqualLayer`, and `QueryLayer` operate directly on
> codebook bivector activations via clipped cosine projection --- no
> separate adjacency table, no `<architecture><mereologicalTreeSize>`
> XML knob (silently ignored if present). Asserted meronymic
> relations are learned by training pulling the codebook geometry
> into the right configuration.

---

## Parthood as Clipped Cosine Projection

Parthood is the single fundamental operation. Every other mereological
relation is defined in terms of it.

> **Terminology (per `doc/old/2026-06-21-terminology-percepts-concepts-symbols.md`).**
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

**Exact dominance vs. retrieval score.** `part`/`whole`/`equal`/`overlap`/
`underlap`/`boundary`/`copart` above are the graded clipped-cosine *score*
(see the Meronomy reconciliation note above, item 1). The boolean dominance
trio `Ops.partOf(S1, S2)` / `Ops.wholeOf(S1, S2)` / `Ops.overlapOf(S1, S2)`
(`bin/Layers.py`) is the separate exact ground truth: elementwise $\le$ /
$\ge$ reduced over the last dim with `all` (`overlapOf` is the zero-directed
elementwise min, `Ops._radmin`) --- not a `part`-suite formula, and not what
this page's scores approximate at retrieval time.

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

## ImpenetrableLayer: Five-Relations Regularizer

`ImpenetrableLayer` regularizes the WholeSpace whole-percept codebook. It
classifies each ordered pair of codebook rows $(i, j)$ into one of the five
mereological relations (using `Basis.part`), and penalizes partial overlap
when paired with a trust mismatch. The learned half of the [Codebook
Uniqueness Contract](Spaces.md#codebook-uniqueness-contract).

### Penalty

$$
\mathcal{L} = \mathrm{overlap\_weight} \cdot
   \frac{\sum_{i \neq j} \mathrm{overlap}(i, j)
                        \cdot |\mathrm{trust}(i) - \mathrm{trust}(j)|}
        {K(K-1)}
   + \mathrm{variance\_floor\ term}
$$

Overlap strength is damped to zero as the pair approaches `equal`:

$$
\mathrm{overlap\_strength}(i, j) =
   \min(P[i,j],\ P[j,i]) \cdot
   \left( 1 - \max(P[i,j],\ P[j,i])^k \right)
$$

with $k = \mathrm{equal\_suppression}$ (default 4.0). This is a continuous
formula, not the discrete five-way classification below it: $\min(P[i,j],
P[j,i])$ already reads near-zero for `disjoint` and strict `part` (one or
both scores near 0), so only `equal` needs an explicit damp --- the
$(1 - \max(\dots)^k)$ factor pushes the penalty toward zero as a pair's
mutual parthood approaches 1.

A separate `variance_floor` term guards against row collapse.

### Trust

Per-row usage frequency, derived from VQ EMA counts:

$$
\mathrm{trust}(i) = \frac{\mathrm{cluster\_size}[i]}{\sum_j \mathrm{cluster\_size}[j]}
$$

When VQ is absent, trust falls back to $\|cb[i]\| / \max_j \|cb[j]\|$.

> **Relative relations and tetralemma trust.** A relative sentence in
> the `part` / `isEqual` predicate family is preserved at depth 3 as
> `[predicate, idea1, idea2]` and may be learned into the codebook as a
> ternary META edge (predicate as parent, the two ideas as children) via
> `WholeSpace.insert_relation`. An accepted relation carries a
> **tetralemma trust 4-tuple** $(t, f, b, n)$ (TRUE / FALSE / BOTH /
> NEITHER, summing to $1$) from the TruthSet posture, gated by the
> content-aware learn-score against `<truthCriterion>`. See
> [STM.md Section 9](STM.md#9-relative-vs-absolute-end-states).

### Diagnostics

After `forward()`: `last_overlap_loss` and `last_variance` populate whenever
their weight is nonzero. `last_relation_counts` (dict summing to $K(K-1)$)
populates only under `MODEL_DEBUG` --- the five `.sum().item()` calls it
needs are host syncs that would otherwise break CUDA-graph capture.

### Configuration

XML knobs (under WholeSpace):

| Knob | Default | Meaning |
|------|---------|---------|
| `impenetrableOverlap` | 0.0 | Weight of overlap $\times$ trust-diff penalty |
| `impenetrableVariance` | 0.0 | Minimum variance regularizer weight |
| `impenetrableAntisymmetry` | 0.0 | Legacy antisymmetry weight, still accepted by schema |
| `impenetrableTransitivity` | 0.0 | Legacy transitivity weight, still accepted by schema |

---

## Order-raising (building the meronymic lattice)

Gated behind `<mereologyRaise>` (default off $\to$ byte-identical). Full design +
code map: [doc/old/mereological-order-raising.md](old/mereological-order-raising.md).

> **Terminology note.** The cross-tower link that ties one part-percept to one
> whole-percept by reference is a **concept** (a `ConceptualSpace` relation);
> `insert_meta` allocates it into the **Concept codebook** --- the
> `WholeSpace.taxonomy` / `taxonomy_parent_map` / `meta_pair_to_idx` /
> `meta_trust` / `part_chain` tables. The earlier `_sym_*` placeholder name
> no longer exists in code (it survives only in older comments); prose below
> uses "concept" for the relation.

The two towers stay **in-kind** — `PartSpace` $\sigma$ *composes parts $\to$ parts*,
`WholeSpace` $\pi$ *analyses wholes $\to$ wholes* — and the **concept (META node) is the
cross-tower link**, associated with both an overlapping part-percept and
whole-percept (`insert_meta(ps_pos, ws_pos, fused_vec=None, *, ema=0.1,
trust=None)`; `ss_pos` is the pre-rename spelling). An object's identity is
**isomorphic**: $\sigma$-up meets $\pi$-down at the object. At init there are only **atoms**
(part-percepts in PartSpace) and the **universe** (the top whole-percept in
WholeSpace); objects emerge from attention.

Three balancing forces keep a single coherent lattice and converge it:

1. **Link the tightest relation** — the largest part-percept $\leftrightarrow$ the smallest
   whole-percept (via `.where` extent); skip a link a bigger-part/smaller-whole
   already subsumes; drop useless concepts.
2. **Too many parts $\to$ synthesize a higher-order part** (`maybe_raise_order`):
   when a whole accumulates more than `K_many` parts, mint a higher-order PART
   that subsumes them, with **abstraction order** one above its constituents
   (tracked via the ramsification table — order 0 = atom, 1 = basic category,
   …) and explicit `part_chain` provenance (a higher-order part is abstract, so
   its `.where`/`.when` are discontiguous and the meronymy is tracked, not read
   off `.where`). Idempotent per whole.
3. **Only one part $\to$ Lewis' Singleton $\to$ analyse the whole** ($\pi$ divides) or drop
   the spurious link.

Link surgery: `delete_meta` / `unlink_child` (invert `insert_meta`'s
registrations); `ps_children_of_whole` counts a whole's parts. The higher-order
code is synthesized by `SigmaLayer.synthesize_over_set` (the M-way generalization
of the binary atanh-sum fold) — or, in the "subsymbolic first" phase, the
mean-combine of constituents, with `part_chain` as the source of truth. (First
pass: the raise fires correctly when a whole has many parts; the live pid-keyed
autobind binds 1 percept$\to$1 concept, so similarity-based many-to-one binding + the
prune-and-rebind of moot edges are noted follow-ups.)

### The corpus callosum links the towers (part `isa` whole)

The full mechanism (see the spec): the **corpus callosum** in ConceptualSpace (the
learned `[2N,N]` `self.callosum` glue) is what joins the towers into one meronomy. A
part-percept `A` (PartSpace) and a whole-percept `B` (WholeSpace) have `.what`
codes from *different* codebooks (incomparable), but their **`.where` is
comparable** — so the callosum asks whether `A.where` is *part of* `B.where`
(a direct span-tuple containment test, `record_cross_tower_meronomy`; see the
2026-07-09 delta below for why this is no longer a `WhereEncoding` decode);
when it is (and no greater part / lesser whole intervenes, per codebook
activation) it forms the **concept `A isa B`** (token `isa` type; the type *names*
the token). Co-occurrence at the same `.where`/`.when` drives the link, weakening
when they dissociate — a **Hebbian** coupling between codebooks.

Since the 2026-06-16 redesign `.where` carried an **endpoint-sum bracket**
`[start, end]` (angle = span center, magnitude = extent); containment was
`A.start >= B.start && A.end <= B.end`, and **contiguity** (adjacency / gap)
between sibling parts was the same endpoint comparison. An instant snaps to
zero extent, so atoms compare as points. (See `doc/Spaces.md` for the
encoding. 2026-07-04: `.when` left the bracket — it is the 4-dim start
ladder; temporal extents ride the record store / exact clock, not the band.
**2026-07-09:** the muxed `.where` band itself moved to a 2-rung quadrature
*ladder* over the byte START only (`WhereEncoding`, `bin/Spaces.py`) — range
rung + resolution rung, no END in the band. `WhereEncoding.decode_span` is
retired; the endpoint-sum bracket codec survives only on the analyzer side,
as `EndpointSumWhere` (`bin/perceptual_analyzer.py`), a distinct codec. The
runtime containment test above no longer goes through a `WhereEncoding`
decode at all: `WholeSpace.record_cross_tower_meronomy` (`bin/Spaces.py`) and
`RunStructureLayer.contained_mask` (`bin/Layers.py`) read `.where` span
tuples directly.)

**Design decision (2026-06-29, REVISIT) -- the `.where` support of percepts vs
symbols.** Every PERCEPT (a part from PartSpace or a whole from WholeSpace) MUST
carry a real `.where` (`nWhere > 0`): a definite `[start, end]` extent. A percept
with no location is not admissible. A SYMBOL (a concept-level relation entry, or an
SS reference) MAY instead either (a) carry `nWhere = 0` (no location of its own), or
(b) carry a RECOMBINED `.where` equal to the bounding span of its constituents --
`start` = the leftmost constituent's `start`, `end` = the rightmost constituent's
`end` (the support of a symbol is the sum of the supports of its members). If the
recombination is hard to implement, the fallback is to give every symbol a `.where`
of "everywhere" (universal support). THE ANT-COLONY QUESTION: is the location of an
ant colony IN each ant (distributed -- every ant keeps its own `.where`), or is
there a SINGLE location that subsumes all the ants' locations (the bounding region)?
The recombined bounding-span (b) takes the single-subsuming-extent view; "everywhere"
is its degenerate limit. This choice determines whether a concept's `.where` is
distributed over its parts or collapsed to one subsuming extent, and should be
revisited.

Word $\leftrightarrow$ object cannot be linked this way (too unlike; no convex set is specific
enough), so a **second-order meta-object** is synthesized in PartSpace, **outside
`.where`/`.when`**, fusing the word-code and object-code into the symbol used in
serial communication (the MetaSymbol). *(Language update 2026-07-02, revised
same day by the two-phase rework: in the ramsified CS this is a FIRST-order
META-concept -- the sec-4c ORDERED PAIR $[whole=word\text{-}symbol,
part=object\text{-}symbol]$, roles as positional slots of an ordered pair
rather than containment claims; the typed read-out recovers (word, object) by
INTERSECTION with the word-symbol class (`meta_word_object`), and the
reference-table pairing remains the serial-mode access path. See
doc/Architecture.md sec A.)* Because symbols are outliers, the
part$\leftrightarrow$whole **concepts** live in a **two-code LUT** and the symbolic taxonomy is
**relations over symbol indices** — which are the TruthLayer's **absolute truths**
(propositions) and **relative truths** (`RelativeTruthStore`), to be integrated.
Taxonomy relations are also learned **explicitly from trusted language** ("cats are
furry" $\to$ `[cats] <= [furry]`).

**Granularity / the part-whole ratio** is the correctness signal (and the MM_20M
mean-collapse fix): *many parts $\to$ one whole* = under-analysed; *one part $\to$ one whole*
= over-analysed. An incorrect ratio is the criterion to request **further synthesis
($\sigma$, e.g. radix) or analysis ($\pi$, property tiling) within the problematic `.where`** —
until words emerge as parts. MM_20M collapses because its byte chunker never climbs
and its analyser never descends, so no word-granularity parts exist for XOR.

**Design decision (2026-06-29, REVISIT; brake updated 2026-07-03) -- locking in the
"basic level" of analysis.** The mereological analysis should converge on and
STABILISE at Eleanor Rosch's BASIC LEVEL -- the granularity where the part-whole
ratio is correct (neither many-parts-one-whole nor one-part-one-whole). The radix
climb builds parts UPWARD (byte -> prefix -> word -> ...); left unbounded it
over-climbs (a whole short sentence promotes to one percept). Two principled brakes
hold it at the basic level: (a) a STRUCTURAL bound -- word tiling: a chunk cannot
cross a word boundary, where the cut is WHITESPACE AND PUNCTUATION (separator runs
-- whitespace or punctuation -- are themselves maximal same-class tiles that get
observed/promoted just like word tiles; see ``PartSpace._embed_radix`` under
``_meronomy_words`` and ``RadixLayer``'s ``word_bounded`` promotion gate); and (b)
the part-whole-ratio convergence signal above (request more synthesis/analysis
until the ratio is right). The ideal is to LOCK IN whatever basic level the corpus
implies. As a pragmatic HEDGE -- and because **word-level is essentially the basic
level for a symbolic LLM** -- the analyser cuts on the word tiling (PS
observes/promotes per tile so parts top out at word size), which is a hard
structural bound rather than a size threshold: the earlier ``<basicLevelMinSize>``
/ ``<basicLevelMaxSize>`` letter-count bounds were too loose to separate a short
sentence from its words on their own, could not express "don't cross a boundary"
at all, and are now REMOVED (dead config, never wired to anything -- the word
tiling supersedes them); the full ratio-driven basic-level convergence remains the
to-revisit principled version.

**Design contract -- types from properties, word-sized `.where` (Alec, 2026-07-03).**
The enabled analysis methods are a set of PROPERTIES identifying the TYPES we
attend to, dividing the input (left of space, right of space, punctuation,
numbers) into word-sized objects. Each object's `.where` then either (a)
ASSOCIATES with a maximal part from PS -- a promoted percept covering exactly
that span -- or (b) is SENT BACK to PS to produce a parts-based definition
covering exactly that `.where` (spell-out bounded by the span).

### Explicit concepts $\longleftrightarrow$ implicit subsymbolic representations (dual-coded, 2026-06-21)

> **Terminology note.** This section formerly called the explicit part$\leftrightarrow$whole
> relation entry a "symbol." Per the convention, that relation — one part-percept
> tied to one whole-percept by reference, held in the Concept codebook's
> `taxonomy` / `taxonomy_parent_map` / `meta_pair_to_idx` / `meta_trust` /
> `part_chain` tables — is a **concept**. The 0-D `SymbolSpace` **symbol** is a
> separate, intensional reference *to* such a concept and is not what those
> entries are. Prose below says "concept"; the code no longer carries a
> `_sym_*` placeholder (that name survives only in older comments).

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
- **Implicit (subsymbolic).** Each concept has a corresponding **learned vector** —
  a strictly-positive `ConceptDim` **atom** (a feature signature) stored in the CS
  concept dictionary (`similarity_codebook`, softplus-rectified). For a higher-order
  concept the *production* is no longer "$\sigma$ then quantize", nor the earlier
  iterated sparse wave; it is a **feedforward $\sigma$-pyramid**
  (`cs_forward_content`, bin/Spaces.py; dual-towers rev 2, 2026-07-12): the single
  untyped square `ConceptualAttentionLayer` computes each rung $k$'s rows in one
  hop, $a^k = \tanh(W [a^{<k} \mid 1])$, gathered under a per-batch top-K taper
  (`order_slice(k)` / `_order_caps`), with $K$ = `symbolicOrder` bounding the
  rung count — **no fixed point and no re-injection** (each rung reads only the
  rows already settled below it; nothing is fed back into its own input). The
  concept code is the final rung's activation
  scaling its positive atom ($a \cdot \mathrm{softplus}(atom)$; radial: magnitude = certainty,
  sign = present vs anti-present). The many$\to$one abstraction is carried by the sparse
  WEIGHTS (which sources contribute, with what sign), not by a $\sigma$-fold + VQ snap; the
  gradient reaches the weights, the source activations, and the dictionary
  (forward-connected). `PerceptDim` and `ConceptDim` are decoupled — a concept is in a
  different vector space than its percepts, never a sum of percept vectors.

**Coordination.** Order-raising is the coupling between the two. When the explicit
tower raises a higher-order part-percept/whole-percept (built directly, inserted on
the **over-collected side** of the higher-order table — too-many-parts $\to$ higher-order
part, too-many-wholes $\to$ higher-order whole), it **creates the corresponding implicit
subsymbolic higher-order representation** via the sparse wave above (new untyped
edges from the relation's row to its constituents on the shared square store). The explicit concept
**indexes / names** the implicit vector; the implicit vector is the concept's
**subsymbolic content**. The two co-evolve and must agree: a directly-built
concept-index paired with a quantized vector that is a valid point in the higher-order
mereological codebook. Direct index + quantized representation are the two
complementary halves of one concept.

This is the dual-coded (neuro-symbolic) core: discrete concept **relations over
indices** in lock-step with continuous **quantized representations** — neither alone,
but the two coordinating.

---

## Why This Design

- **Parthood as projection.** One formula, Boole-contrapositive exact,
  continuous in $[0, 1]$. The old composite formula
  $\operatorname{conjunction}(1 - \operatorname{dist}(x, x \cap y),
  1 - \operatorname{dist}(y, x \cup y))$ mixed set-valued operands
  with a distance.

- **Overlap penalty (not antisymmetry).** Legacy `ImpenetrableLayer`
  penalized mutual parthood $P[i,j] \cdot P[j,i]$ directly, pushing
  *every* overlapping pair apart including synonyms. The five-relations
  design leaves `equal` pairs alone and only penalizes the ambiguous middle
  region, gated by trust mismatch.

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
not the `SigmaLayer`/`PiLayer` butterfly folds. The butterfly folds remain the
higher-order abstraction machinery. See
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
complete `.where` order. On sparse, bounded-depth grounding, the soft union
stays away from the all-one corner. Cross-order abstraction is handled by the
separate butterfly folds rather than by repeatedly stacking this join.

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

The adjoint pair can subsume alternate text front ends, but the implementation
continues to support the existing synthesis modes (`radix`, `lexicon`, `bpe`,
`mphf`, and `byte`) and analysis cuts (`byte`, `word`, `raw`, `sentence`,
`grammatical`, and `meronomy`).

### Live Routing and Implementation Boundary

`MeronymicRouter` in [`bin/perceptual_analyzer.py`](../bin/perceptual_analyzer.py)
scores candidate merges against the perceptual codebook and routes them with
the shared `binary_tiling_viterbi` / `binary_tiling_soft_dp` primitives (see
[Language.md](Language.md), *Shared Weighted-Deduction Framework*). In a cold
state the codebook contains the whole-input vector and byte vectors, so the
router decomposes the surface to byte terminals. As merge promotion learns
words bottom-up, the same router reproduces word-level runs.
`analyze_routed` preserves the raw bytes of UTF-8 segments and reconstructs a
byte-exact surface for replay.

The live router and span bookkeeping establish the path and the complete order
specification. **(2026-06-25, LANDED standalone.)** The hard `join-from-bottom`
plus isotonic-projection guarantee is implemented in
[`bin/Mereology.py`](../bin/Mereology.py) (`join_from_bottom`, `meet_from_top`,
`project_monotone`, `mereological_synthesize`, `mereological_analyze`) and
exercised over projected trees in [`bin/Meronomy.py`](../bin/Meronomy.py)
(`MeronomyTree` and friends), with `test/test_mereology.py`,
`test/test_meronomy.py`, and `test/test_meronomy_laws.py` covering both. It is
not yet applied to the live percept codebook; the relational analyzer basis
and learned convergence criterion are the remaining next-boundary work. The
existing butterfly folds and conceptual SBOW remain unaffected.

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
- conceptual situating:
  [conceptual-similarity-space plan](plans/2026-06-23-conceptual-similarity-space.md)

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
