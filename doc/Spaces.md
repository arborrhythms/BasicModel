# Spaces

> **Architecture relation.** The Space/SubSpace split is where BasicModel most
> directly departs from conventional LLMs: hidden state is routed through named
> percept, concept, symbol, STM, and codebook stores instead of one residual
> stream. PartSpace/WholeSpace plus the Concept codebook provide the fuzzy
> Formal Concept Analysis reading (extent, intent, and concept order), while
> SymbolSpace and the grammar host provide the DisCoCat-like composition path
> from typed syntax to vector meaning.

> **Terminology (one noun per space).** A **percept** is a PartSpace or
> WholeSpace thing, dimensionally embedded and extensional; its two subtypes
> are **part-percepts** (atoms, synthesized bottom-up) and **whole-percepts**
> (properties or regions, analysed top-down). A **concept** is a signed
> value that a Sigma gives over basic percepts at order zero, or over
> symbols above it. A **symbol** is a row of a Pi above order zero: it has no
> dimensions, no vector, place or form of its own, and is held apart from
> the concepts; a symbol is not a WholeSpace row. There are **objects** and **references**, and a reference
> is either a **sign**, a quantized version of its referent (the same space,
> snapped to a codebook row), or a **symbol**. Code identifiers such as
> `subspace.what` keep their names.

> **Perception.** `PartSpace` and `WholeSpace` both subclass `Space`.
> PartSpace combines ordered parts; WholeSpace divides the field by
> primitive properties. Neither owns a sigma/pi fold layer. Native codes stay
> in the same `[0,1]^D` cube, on two lanes, for and against; rank is
> mereological level, not a change of dimensional space. The first Pi, over
> a basic percept's parts and wholes, and the Sigma that reads the basic
> percepts begin the one structure, Pi, Not, Sigma, that repeats at every
> order ([Architecture](Architecture.md#sigma-and-pi-layers)). No percept or
> symbol is negated: a percept carries perceived evidence against on its
> against lane, and a sign lives only in a weight of a Sigma. Objects are
> analysed and synthesized by sending them back through the towers: wholes
> get split and parts get chunked.

> **`subspace.what`: atoms and properties.** On `PartSpace` the `.what` rows
> are **atoms** (letters and words, gathered by `Codebook.lookup`). On
> `WholeSpace` the `.what` rows are **properties**, and properties are what
> wholes *are*: a `.what` row is a binary proposition over the field, and
> only a proposition can bifurcate the input into two clearly identified
> regions, has-type and complement, the way `materialize_property` does.
> Types apply to **runs of characters, not single characters**. BasicModel
> begins with a small basis of roughly eight ASCII-class properties (letters,
> capitals, digits, punctuation, whitespace and the remaining byte classes),
> one codebook row per property. This is an initial learned basis rather than
> a closed ontology: dynamic properties may activate unused rows within the
> declared WS capacity, which may be declared at PartSpace scale. Such
> admission never allocates a concept or symbol in WholeSpace. Property
> activation may overlap: an ASCII capital activates both `letter` and
> `capital`, and that conjunction does not mint a ninth property row. A whole
> is a **maximal constant-type run**; words are runs of letters, punctuation
> or digits. `materialize` hands the per-position selection
> (`subspace._index`) to the codebook, which produces atoms via `lookup` or a
> per-position **region membership** via `Codebook.materialize_property`
> (`mode="property"`): the region that has the property (`> 0`) and the
> region that does not (`<= 0`). Properties are **low-frequency by
> construction**: there is too much detail to learn a codebook over the whole
> input unless the atoms are low-frequency, so the property basis frequency
> is tied to the input length: the k-th harmonic completes only about k/2
> cycles over the whole field. The learnable property row is the
> low-frequency atom; the basis enforces the coarseness.

> Grammar dispatch lives on `SymbolSubSpace.languageLayer`. The grammar's
> layers (`change`, `lift`, `lower` and the rest) own their grammatical
> operations; perceptual towers do not supply their parameters.

## Overview

BasicModel is a pipeline of five **spaces** plus a grammar host
(`SymbolSpace`), each performing a distinct representational
transformation. Data flows forward from raw input to task output; the
reverse pass reconstructs the original input from the symbolic
representation. The subsymbolic role is filled by `PartSpace`, and the
grammar runs from `SymbolSubSpace.languageLayer`, the signal router.

**`SymbolSpace` literally subclasses `Space`** (`class
SymbolSpace(Space)`, [`bin/Language.py`](../bin/Language.py)), but it is the
downstream reference/grammar coordinator rather than a third learned percept
inventory. It constructs via
`nn.Module.__init__` directly, deliberately SKIPPING `Space.__init__`'s
object/what/where/when VQ-basis build, and holds a `SymbolSubSpace`
coordinator (the typed-STM stack + grammar dispatch carrier) as
`self.subspace`. `SymbolSpace` is a transparent container: reads that
miss its own attrs fall through `__getattr__` to the held coordinator,
and writes forward through `__setattr__`, so `symbolSpace.X` call sites
are unchanged whether `X` lives on `SymbolSpace` or on the coordinator
it forwards to.

```
Forward (space ownership): InputSpace -> {PartSpace, WholeSpace, SymbolSpace,
                              LanguageSpace} -> ConceptualSpace -> OutputSpace
Reverse:                    OutputSpace -> SymbolSpace -> ConceptualSpace
                              -> {PartSpace, WholeSpace} -> InputSpace
```

The forward line describes ownership rather than a same-tick linear chain:
PS/WS form the subsymbolic peer result; SS returns a symbol reference; and
LanguageSpace returns a grammar/routing plan. ConceptualSpace is the sole
reducer and owner of its `CSsub`, `CSsym`, and `CSLang`/STM commits.

### SymbolSpace and LanguageSpace: reference, grammar, and scheduling

`SymbolSpace` is the discrete, inspectable boundary of the model: a symbol is
a typed, zero-dimensional reference to a ConceptualSpace concept, carrying its
reference activation rather than a third perceptual codebook row.  This makes
the concept-to-symbol transition the natural place to expose grammatical role,
derivation provenance, and a stable reference for an external truth or
reasoning system.  It does **not** make SymbolSpace the owner of a separate
continuous concept store; concept allocation, conceptual STM, and their
mutation remain ConceptualSpace responsibilities.

The grammar operator inventory is executed by `LanguageLayer`, registered once
under `SymbolSubSpace.languageLayer`.  `LanguageSpace` is the runtime
scheduling facade for that same layer: in the serial peer pipeline it consumes
the completed symbolic result, chooses grammar operations, returns immutable
choice tensors, and owns the grammar latch timing. It does not duplicate
`LanguageLayer` parameters or grammar state, and it cannot directly mutate
ConceptualSpace STM. ConceptualSpace applies each returned Binary or Unary
choice to its own buffer/depth/order/reference tensors through its `CSLang`
transaction lane. `CSLang` is only the ownership name of the existing CS STM;
it does not register another module or copy its state. This preserves an
inspectable separation between:

- **concept references** (`SymbolSpace`),
- **grammar execution and its persistent state** (`SymbolSubSpace` /
  `LanguageLayer`), and
- **pipeline timing** (`LanguageSpace`).

`SymbolSubSpace.reconstruction_stack` holds the temporary reading record:
selected operations, actual operands, native references and the owned spelling
targets. Each word has three fixed operation-round slots, followed by a bounded
closing budget. Every round chooses from both arities through the same softmax;
there is no separate unary pass. The compiled loop masks inactive words and
publishes the actual end state explicitly.

The sentence driver uses this record for tied reconstruction and for a
path's one departure. Reconstruction keeps the kept path's gradient live;
targets and durable memory are detached. After the paths of a sentence are
compared, only the kept end state is committed and the operation record is
cleared
([Architecture](Architecture.md#whole-paths-and-concept-episodes-2026-11-11)).
The LTM entry and the completed `Understanding` contain no reading program,
leaves or spelling snapshot; output generates from the occupied state and
the model's current grammar.

When `forwardGrammarWeight` is nonzero, the same fixed trace layout also
carries a bounded local loss for each committed fold. Unary and binary layers
contrast their type-valid alternatives using detached child/candidate
evidence, so only the grammar chooser is updated. This gives the forward
derivation direct pressure without extending the autograd path across words.

An application-level truth engine may bind those stable references to typed
propositions, provenance, certainty, or an operator tree.  That policy belongs
above the BasicModel runtime; the model's invariant is that a symbolic result
can be traced back to its concept reference and grammar operation without
silently collapsing it into a perceptual row.

The recurrent character lives in (a) STM accumulation across words in serial
mode, and (b) the T-pass parallel refinement loop driven by
`<subsymbolicOrder>` via `WS.forward(cs_out=...)` / `CS.forward(...)`
iterations. PS itself runs once, at stage 0; the per-stage recurrence
advances through the prior stage's CS output, not repeated PS calls (see
**Perceptual and conceptual ownership** below).

![WikiOracle Space Hierarchy](diagrams/vector_spaces.svg)

---

## Base Class: Space

All spaces inherit from `Space`, which manages:

- **Shape management.** `inputShape` / `outputShape` as `[nObjects, nDim]`.
- **Codebook / VQ quantization.** When `nVectors > nActive`, a codebook holds
  candidate vectors; top-k selection gives the bottleneck.
- **Reshape flag.** When in/out object counts differ, the `[B, nIdeas, nDim]`
  tensor is flattened before the next space and restored on the way back.
- **Attention.** The `<attention>` element (`off` | `primer` |
  `second-order` | `low-rank`), read per Space with default `off`.
- **`set_sigma` propagation.** Ergodic-mode noise level cascades from the
  top-level model down through every child layer.

### Reset cascade: hard vs. soft {#reset-cascade-hard-vs-soft}

Every space exposes `Reset(batch=None, hard=True)`; both arguments are part
of the required signature.

| Call form | Scope | Use |
|-----------|-------|-----|
| `space.Reset()` | All rows | Whole-state wipe (epoch boundary) |
| `space.Reset(batch=b, hard=True)` | Row `b` only | Document boundary |
| `symbolSpace.soft_reset(batch=b)` | Row `b` sentence-scoped state | Grammar `<start>` reduction |

A hard reset clears, for the affected rows, the parse stack, `_last_svo`, the
sentence-completed flag, the codebook commit accumulator, the discourse
history, `serial_cache`, `_ar_embedded` and `_end_of_stream`.

A soft reset clears the per-sentence working buffers (the parse, category and
reconstruction stacks, `_last_svo[b]` and the sentence-completed flag). It
does **not** touch the discourse history (`SymbolExpectation`'s chronological
context and predictor) or codebook EMA, which are document-scoped.
ConceptualSpace clears its STM softly at a sentence or document reset and
hard at the epoch reset ([STM.md](STM.md#1-stm-data-model), "Lifecycle").

Reset is dispatched from `runEpoch`, never from inside `runBatch` (the pure
compute brick --- see [Architecture.md](Architecture.md)).

---

## Perceptual and conceptual ownership {#sigma-pi-ownership}

Perception is analysis and synthesis of what is present. PartSpace combines
parts bottom-up, a code growing from all zeros by the OR of its parts;
WholeSpace divides the field top-down, a code shrinking from all ones by the
AND of its wholes ([Mereology](Mereology.md#percept-code)). At every order
the same structure follows: a Pi, whose rows are basic percepts at order
zero and symbols above it, each row with two values, for and against; then
the Not, which joins each row's two values into one signed value; then a
sparse Sigma, whose weights are at zero or above, giving the concepts.

| Space | Owns | Forward |
|---|---|---|
| PartSpace | native percept inventory, radix ids, the parts and their codes | find the sentence's forms and parts; a form's parts half = the greatest of its parts' codes |
| ConceptualSpace | the basic percepts' Pi (`BasicPercepts`, `PerceptPi`), the Not and the Sigma of order zero (`order_zero_not`, `order_zero`), the Pi, Not and Sigma of each order above, and STM | read the first Pi on the rows the lookup names, join each row's two values in the Not, read the concepts of order zero from those signed values, then each order's symbols and concepts |
| WholeSpace | the table of wholes (`PrimitiveProperties`) and analysis spans | read each whole where a form's run of parts lies; a percept's wholes half = the least of its wholes' codes |

The two perceptual `forward(in_sub, cs_out=None)` signatures remain.
Subsymbolic passes can change attention or level but apply no learned fold
inside a tower. There is no perceptual layer stack kept beside this route.

**Composition (per-mode):**

- **SERIAL / GRAMMATICAL** (`<serial>true</serial>`): one
  iteration per word.

  ```
  CSSub(w):    IS(w) -> { PS(w), WS(w) } -> CS reduction
  CSSym(w-1):  symbolicOrder x (CS -> SS -> CS)
  CSLang(w-2): word-reference deposit -> object-reference resolution
               -> Language's operation choices -> CS apply/STM commit
  ```

  These are three peer-pipelined dependency stages over adjacent word indices.
  The braces inside CSSub are peers, not a left-to-right call order: Input is
  never routed from PS into WS. The eager lexical boundary stages a separate
  property activation for every WS word, so WS reads only the current word
  rather than repeatedly analysing the full padded sentence. Within CSSym,
  the `CS -> SS -> CS` recurrence is intentionally sequential and runs exactly
  `<symbolicOrder>` times to promote conceptual order. CSLang deposits the
  word concept into CS STM; interpreting that reference replaces it with the
  paired object concept before Language builds the object concepts into a
  syntax tree, one operation per round
  ([Language](Language.md#one-operation-per-round)). `CSsub`/`CSsym`
  snapshots handed to a non-owner are read-only, and only CS computes or
  commits conceptual or STM transitions. The CSSub-to-CSSym and
  CSSym-to-CSLang latches merely retain completed tensor products; they are
  not duplicated Space state. Where sentences end,
  `_run_sentence_word_bricks` drives these stages one sentence at a time,
  perceiving the sentence once and then reading it along whole paths
  ([Architecture](Architecture.md#whole-paths-and-concept-episodes-2026-11-11));
  otherwise the three stages run through one tensor `while_loop`, in which W
  is a storage capacity.

- **PARALLEL** (`<serial>false</serial>`): T iterations of PS over
  CS.

  ```
  contribution = PS.forward(IS)      # PS runs ONCE, at stage 0 only -- not
                                     # repeated per stage (single-pass subsymbolic decision)
  for t in 1..T = <subsymbolicOrder>:
      # Universe glue contract: sparse-parallel
      # (symbolicOrder > 0) offers the universe EVERY stage; otherwise it
      # bootstraps stage 0 only and t > 0 stays carrier-driven.
      WS_t = WS.forward(unity if (sparse_parallel or t == 0) else None,
                        cs_out=prevCS_forSS)
      CS_t = CS.forward(contribution, WS_t)   # STM[t] = combine (parallel write; no shift)
      contribution = CS_t             # stage k+1's contribution IS stage k's CS output
  router.dispatch_at_C(STM)           # grammar ops after STM population
  ```

### Change, Lift and Lower

`change`, `lift` and `lower` are first-class `GrammarLayer` operators at
the CS (`space_role='CS'`), not rule-id annotators over a shared substrate:

- `change`: `arity=2`, `clause="S"`, invertible. It is predication:
  `"the boy runs"`. It applies the verb phrase's gain to its subject in
  the basis the phrase carries, through `MeaningAction`, the map
  $Q\,\mathrm{diag}(e^{w})\,Q^{\top}$ in the chart on each evidence lane.
  Outside a meaning layout it owns an internal `self._sigma: SigmaLayer`
  for the pairwise additive (sigma-style) math, and the reverse goes
  through the same layer. `verb(VP, NP)` gives the verb phrase that basis,
  read from the NP's meaning ([Language](Language.md)).
- `lift`: `arity=1`. It raises its operand one order.
- `lower`: `arity=1`. It is admitted only on an operand of order one or
  more, and gives a member of it, one order down. A determiner is a
  modifier and has no operator of its own.

The chart of `change` is bounded in the backward: its internal
`SigmaLayer` carries `_bounded_backward`, so `atanh(x)` keeps its exact
forward value and clamp while the gradient slope is capped at the tangent at
`|x| = 0.9` (`Layers.bounded_atanh`, a custom autograd Function saving only
the input). Operands at $\pm 1$ are ordinary (max-law unit codes, `chunk`
sums, saturated folds), and the exact derivative $1/(1-x^2)$ (about
$5 \cdot 10^6$ at the clamp) compounded through the nested folds of a long
packed row into NaN parameters. Space-level and readout layers keep the
exact chart: the cap under-converged the crisp XOR fits
(`test_explicit_dimensions.py`), and the compounding needs the nesting.

The signal router (`SymbolSubSpace.languageLayer`) dispatches them as
reduce ops at the CS, weighted by `Grammar.rule_probability` (the
per-position copy/reduce score head). See [Language.md](Language.md#grammar)
for the GrammarLayer specifics.

### Butterfly mode on `GrammarLayer`

`GrammarLayer(butterfly=True, N=N)` allocates an FFT-style
**element-pair** cascade over a flattened `[B, M]` view (`M` = `N`
padded to the next power of two; `n_levels = log2(M)`), NOT a packed
per-node matrix Parameter. Storage is three per-node LDU-triplet
`nn.Parameter`s shared by every `GrammarLayer` subclass ---
`butterfly_L: [n_levels, M//2]`, `butterfly_d: [n_levels, M//2, 2]`,
`butterfly_U: [n_levels, M//2]` (sub-diagonal, the two diagonal
scalars, super-diagonal of each node's `2 x 2` LDU block) plus a
`butterfly_perms` buffer (per-level bit-reversal permutations placing
XOR-neighbour elements adjacent). Identity init (`L=0, d0=d1=1, U=0`)
makes the cascade identity at construction. `_butterfly_pair_forward`
/ `_butterfly_pair_reverse` (`GrammarLayer`,
[`bin/Layers.py`](../bin/Layers.py)) implement the closed-form `2x2`
LDU pair op ONCE at the base class -- sign-flip on the off-diagonals +
reciprocals on the diagonals, no `torch.linalg.solve`. Every
`GrammarLayer` subclass (`SigmaLayer`, `PiLayer`, `IntersectionLayer` /
`UnionLayer`, `ConjunctionLayer` / `DisjunctionLayer`, ...) reuses this SAME pair op
via `self._butterfly_forward` / `self._butterfly_reverse` in butterfly
mode; there is no per-subclass `_butterfly_pair_op` override or
distinct pairwise math (atanh/einsum/tanh, min/max, etc.) --- the
subclass identity only matters for the non-butterfly forward/reverse
math.

Parameter savings: `O(N · log N)` scalars per cascade vs `O(N² · D²)`
for a single big matrix. The `<sigmaPi>` mode selects the grammatical
layers' implementation. Perceptual passes do not allocate these layers.
The primitive XOR gate uses located conceptual definitions and sigma over
their symbols; see [DevelopmentHistory](DevelopmentHistory.md#located-xor-and-native-perception-2026-09-24).

---

## Normalization and Ranges

| Space | Data Contract | Geometry |
|-------|--------------|----------|
| InputSpace | Measured tensor features scale to presence; signed text embeddings retain sign | `[0,1]` for presence data; `[-1,1]` for text embeddings |
| PartSpace | Modal/demuxed what/where/when encoding. Marked radix/meronomy percept stores expose one-sided presence; the orthographic Lexicon remains signed | `[0,1]^d` for percept-store rows; signed projective unit ball for Lexicon rows |
| ConceptualSpace | One opaque code per concept: no where/when band (`canonical_shape` is `(0, 0)`); a concept has generalised over the where and when modalities and cannot be divided (Architecture, "Concepts are opaque; percepts and symbols are located"). The code is `[form \| meaning]`, every coordinate in `[0,1]`; evidence is two lanes, for and against | positive cube; two evidence lanes in `[0,1]` |
| WholeSpace | Whole-percept property memberships | `[0,1]` presence |
| SymbolSpace | symbols: rows of a Pi above order zero, each two unsigned lanes, with no dimensions of their own | `[0,1]` on each lane |
| OutputSpace | Rescaled from activation range to original data range | Data range |

Grammar and chart machinery lives on downstream `SymbolSpace`; it is not
owned by the WholeSpace property inventory.

**Data scaling.** `Data` computes global `input_min`/`input_max` and
`output_min`/`output_max` at load time. InputSpace uses `Data.normalize(x,
"input")` to scale measured presence data to `[0,1]`; signed text embeddings
remain in `[-1,1]`. OutputSpace uses `Data.denormalize(x, "output")` to restore
the original output range.

**Demuxed mode.** When `InputSpace.demuxed=true`, what/where/when components
are stored independently in the SubSpace rather than concatenated. ModalSpace
routes each component through independent PartSpaces; downstream spaces
see an identical muxed tensor via `materialize()`.

---

## Percept Geometry: Positive Unit Hypercube {#percept-geometry-positive-unit-hypercube}

> **Design.** A percept is a vector of independent **presence** features. Each
> coordinate is a membership in `[0,1]`: `0` is absent or nothing, `1` is fully
> present or everything, and intermediate values are partial presence. A
> percept's code holds two such vectors, its for lane and its against lane;
> what perception calls a negative number is evidence on the against lane,
> never a stored negative. For a percept the opposite of 1 is 0, as in
> Boole (Alec, 2026-11-17).

The percept hypercube is the grounded extent side of the architecture. An LLM
usually learns such grounding indirectly through token statistics; BasicModel
keeps percept presence as an explicit carrier. Formal Concept Analysis enters
when perceptual extents are paired with whole/property intents to form concept
order. DisCoCat enters later, when grammar composes conceptual meanings rather
than raw percept memberships.

Percepts and concepts share the positive cube for content but read it
differently: a percept's coordinates are presences, while a concept's code
keeps its form in the percept cube and its meaning in two nonnegative lanes,
and its polarity and certainty live in evidence lanes rather than in a sign.
Signed coordinates are rejected permanently in identity and concept content:
a native concept carrier must lie in $[0, 1]$ (`IdentityRows.carrier` raises
otherwise), learned maps on the identity path preserve the cube by
construction (see [Bounded Encodings](#percept-bounded-encodings)), and an
artifact holding signed legacy identity points is retired rather than clipped
or converted. Keeping the roles distinct avoids treating a percept's unused
negative half as if it were sensory content.

### Presence, Not Signal {#percept-presence}

A percept feature answers *how present is this?*

| Value | Meaning |
|---|---|
| `0` | nothing or absent; the empty corner |
| `1` | everything or fully present |
| `x in (0,1)` | partial presence |

There is no negative presence. Evidence against a percept is a presence too,
on the against lane: the parts that a form holding it adds, the wholes its
upper bound cuts away ([Mereology](Mereology.md#percept-code)). Zero on both
lanes is no evidence. On the percept path:

- `ConceptualSpace.factor_percept` (a staticmethod, [`bin/Spaces.py`](../bin/Spaces.py))
  keeps the percept INPUT non-negative (`percept.clamp(min=0)`; percepts are
  one-sided) but the EVIDENCE it returns against a CS concept row is
  SIGNED in `[-1, +1]`: selection is by absolute value, so an anti-aligned
  row, a "known false" exclusion, is reachable rather than clamped away; a
  zero percept yields zero evidence (the dot-product tautology).
- The Pi of every order (`PiLayer3`, and the first Pi `PerceptPi`) operates
  on memberships in `[0,1]` through `log`/`exp`, flooring the bottom
  element with `EPS_LOG` before `log(0)` (`membership_log`); its derivative
  passes wherever a membership lies above the floor and at most one, one
  included.
- A marked radix/meronomy Codebook (`is_percept_store`) is read through a UNORM
  straight-through clamp, so forward lookup and reverse decode see the same
  `[0,1]` rows while the float master remains trainable.

The signed orthographic Lexicon is a separate path. Its sign is form content,
so it stays in the projective unit-ball geometry described under
[Lexicon](#lexicon-projective-unit-ball).

### Level Sets: an Intensity Is a Kind {#percept-level-sets}

Every input intensity is categorized. Alec, 2026-11-13: "all input
intensities are tuned into level-sets of percepts", and in the codebook
"percept activations have a kind there but no magnitude". A letter is the
plain case, present or absent. A number is treated the same way.

- **A level is bounded by a part and a whole.** Alec, 2026-11-12: "Each
  percept is categorized against perceptual parts and wholes, whose codes
  are the values. So the order zero concept is a generalization over
  basic percepts." A felt value's percept knits a bound it contains, its
  part, with a bound that contains it, its whole.
- **A value is its place in one sorted table of values** (Alec,
  2026-11-19: "we are enclosing the scalar by a lessthan sort"). The
  code's coordinates are bins of the felt value, 64 of them: the parts
  half holds the bins at or below the value, the wholes half is all ones
  with the bins above it cut away, and the against lane holds the bins
  above it (`BasicPercepts.value_code`). It is the mechanism of a form's
  code with the bins as the catalog
  ([Mereology](Mereology.md#percept-code)).
- **Order survives without magnitude.** A lower value's parts half lies
  inside a higher one's, so `Ops.partOf` on two of them is less-than. No
  percept carries an amount, and any two levels can still be compared. The
  bins are fixed, 64 for each place, so a value's code does not change when
  another value arrives; each place has a table of its own, on coordinates
  of its own, and a collection of values is read by its bounds, not through
  the Sigma.
- **A concept generalizes levels.** A concept over several levels takes
  their common part and their covering whole: an interval on one
  dimension, a hyperrectangle on several.
- **Why a kind.** A magnitude in the code would be operated on directly,
  and "arithmetic do not operate on the codes: that would break codebook
  lookup."
- **What is left out.** "Im reserving the feeling of that continuity for
  direct perception, but that is not something that we can model."

**The number modality.** A row of numbers of any shape is read this way:
a level percept at each place. The XOR proofs are two places on one axis,
and a 28 by 28 image (`data/MM_image.xml`) is 784 places on two axes, "the
only different being the percept dims" (Alec, 2026-11-13). One reader,
`NumericPercepts`, serves any number of places.

- **The feeling.** A number $n$ from zero to infinity is felt as
  $f(n) = g/(1+g)$ with $g = \ln(1+n)$: zero at 0, one only at infinity,
  and logarithmic between, so one is felt as .409 and ten as .706. The map
  is fixed and has no state.
- **A level at a place.** A reading felt above zero is a level percept at
  its place: one basic percept for each pair of place and level met. The
  same level at two places is two percepts, so XOR's two inputs stay
  distinct. Zero is no presence. No coordinate of the percept cube is
  set aside for one place, so the cube's width does not grow with the
  number of places.
- **A collection per place.** Each place has a table of values of its
  own, on coordinates of its own. A collection of the level percepts met
  at a place is read by its bounds, 0 or 1, and not through the Sigma
  (`BasicPercepts.read_collection`); its for lane is the
  reading's membership in that collection, not the felt value.
- **Back from the field.** `NumericPercepts.reverse` returns each place's
  number from the level its concept holds.
- **The scalar beside the words** is the case with no place, and is
  perceived, collected and read by the same functions.

That intensity is a code and never an amplitude was decided for images on
2026-10-25 ([FutureWork](FutureWork.md#intensity-as-codes-not-amplitude-alec-2026-10-25-decided)).

### Modalities: One Perceptual Space Each {#percept-modalities}

Alec, 2026-11-13: "the .where for imagery has to be 2D." He weighed one
shared space for all the senses and chose otherwise: "perhaps the
integration path is leave each modality in its own perceptual space with
varying .where, and integrate only at the level of zero-order concepts by
adding a .modality field according to the percepts that they unite. this
still allows perceptual space to have basic-level perceptual objects on a
modal basis."

- **Each modality has its own perceptual space.** A modality declares how
  many place axes it has. A place is one address in the modality's range
  of `.where`, whatever its number of axes: a place on two axes is its
  row-major index, which the declared shape decodes to row and column.
  For example, row 3, column 5 of a 28 by 28 image is address
  3 * 28 + 5 = 89 within the image's range. The band stays the four
  coordinates of the one two-rung ladder, which recovers the exact
  address (`WhereEncoding.decode_index`), and no vector is widened.

  | Modality | Place axes | Place shape |
  |---|---|---|
  | the scalar beside the words | 0 | one address |
  | text | 1 | positions in a sentence |
  | XOR | 1 | two positions |
  | an image | 2 | 28 by 28; parts and wholes are boxes |

- **A percept is a level at a place.** A pixel is a level percept at a
  place on two axes, as a letter is a byte percept at a place on one.
  Parts and wholes, and so basic percepts, are knit within one modality,
  so basic-level perceptual objects are modal.
- **Modalities meet at order-zero concepts.** An order-zero concept
  collects basic percepts from one modality or from several, and one that
  collects from several is where the modalities are integrated. No shared
  space of places lies below it. Text's `.where` is an order in a sequence
  and an image's is a place on a surface, so one space for both would give
  them distances that mean nothing.
- **Modality and order are read from the place.** Alec: "the .where that
  is baked into all concepts in virtue of muxing the .where of a percept
  will include a .where in a range that can uniquely identify the modality
  and order of a percept." The model's places are one address space,
  divided into fixed, disjoint ranges (`WhereRegistry`), and every muxed
  vector carries its address in its band. The ranges are one input range
  for each modality (text, a numeric input, the scalar beside the words,
  each sized by its configuration; the scalar's has one address), the
  part rows, the whole rows, the symbols and the live conceptual slots. A
  symbol has two addresses, one for each lane; rows lie in their order's
  block, so the range gives the order ([Orders §12](Orders.md)). Each
  modality's input has a range of its own, so the range gives the modality
  too. Neither is stored beside the vector. The number of place axes would
  not tell text from XOR, which both have one; the range does.
- **The where is one fixed parameter** (Alec, 2026-11-19: "a pytorch param
  with the sinusoid written to it"). The `.where` stamped onto percepts
  flows from one tensor that holds the constant sinusoid, a row for each
  address of the registry, written from `WhereEncoding.encode`. "The
  parameter itself must stay fixed if the sinusoid coding is to remain
  meaningful across all .where": nothing of it is learned, nothing is
  written into it as a state, and it is in no optimizer's groups. It takes
  a gradient, so the salience map of a look lands on it, and what it lights
  there says where to look
  ([Architecture](Architecture.md#one-attention)). Each percept has its own
  band, the wires of the parameter's row at its place. A reading's
  `.where` is its own, the band of the event at its start, and is not
  combined over its percepts (Alec, 2026-11-19: "It's own, flowing from
  the fixed .where parameter"). Only the first Pi reads the where, as a
  look's match of place; no concept has a weight on a where wire.

### Complement, Copart, and Negation {#percept-complement}

A percept's opposite is its complement:

```text
antipode_percept(x) = 1 - x
```

Mereologically, the complement of a part is its **copart**, the rest of the
whole. With the whole normalized to `1`, the carrier `[part, copart] = [x,
1-x]` has dependent axes; the copart is derived rather than stored.

This is not the catuskoti/tetralemma carrier. That truth representation has two
independent axes so it can distinguish BOTH from NEITHER:

| Carrier | Axes | Redundant? | Layer |
|---|---|---|---|
| part/copart `[x, 1-x]` | dependent | yes; collapse to one `[0,1]` coordinate | percept |
| catuskoti `[TRUE, FALSE]` | independent | no | concept/truth |

Complement on `[0,1]` and negation in centered signed coordinates are the same
reflection. Let `y = x - 1/2`; then `1-x = 1/2-y`. The operation is shared, but
the stored carriers and their semantics remain distinct.

**Consequence for expectation and attention.** The cube has a complement but
no additive inverse: `x + (1 - e)` leaves `[0,1]`, and nothing added to a
presence cancels it. Expectation therefore never subtracts from content. At
the concept level it is a gain on the arriving content with the residual in
two separate lanes; at the percept level the signed residual is computed
beside the observed signal, which the forward path carries unchanged (see
[Expectation across the architecture](ExpectationRetention.md)). What the
unsigned carrier does support is withdrawal, the non-affirming `non`, and
that is how attention works here: the reading scope and the priority surface
exclude the non-object without predicting it. No percept or symbol is
negated. A percept's against lane is perceived evidence, not a negation; the
one place a sign is changed is the Not, a layer that offers each row's value
and its not-value; a Not entry, a concept's entry that reads a row's
not-value, at the basic level records that something expected is absent. It
is made by the gradient, which an omission teaches, or by a declared Not
member (Alec, 2026-11-17 and 2026-11-19). The operator `not` swaps the two lanes of what
it is put on, which is its own inverse, and `non` removes both, keeping the
operand as a marked witness, which cannot be undone (Alec, 2026-11-12:
"non- removes both positive and negative evidence, while not swaps the two
types of evidence."; `NonLayer`; [Logic](Logic.md#evidence-lanes)). Neither
negates a coordinate, so sensation is never subtracted from. Only what is
conceptualized can be
attenuated: content that reaches no concept, an unparsed span or a percept
with no concept, is met in full every time.

### Sigma/Pi Membership Lattice {#percept-membership-lattice}

The meronymic fold/split operators form a bounded lattice:

| Operator | Role | Identity | Absorber |
|---|---|---|---|
| union | synthesis: parts join | `0`: `x union empty = x` | `1`: `x union everything = everything` |
| pi | analysis: wholes meet, and a row requires its members | `1`: `x intersection everything = x` | `0`: `x intersection empty = empty` |

On values present by degree the union is $1 - \prod_i (1 - x_i)$ and the Pi
the product $\prod_i x_i$; within a code, which is a bound, they are the
greatest and the least, and on 0 and 1 the two pairs agree. These are
set-membership roles. A membership zero rules out a whole where a form's run
lies; `(0,0)` on the two lanes is no evidence.

### The Fold Ladder {#fold-ladder}

WholeSpace's constant-signature runs supply positions and word units supply
extents. The ordered canonical PartSpace ids, masks and brackets retain the
surface witness. Recurrence admits composite parts at the boundary, and a
part formed in a turn does not aggregate again in that turn. Higher rank
means a larger part or a finer whole, in the same membership cube. It is not
a sequence of learned perceptual transforms.

A word's code is the **set of its parts**: its parts half is the
coordinatewise greatest of its parts' codes, and its wholes half the least
of its wholes' codes, its own tightest whole among them
([Mereology](Mereology.md#percept-code)). Repeated constituents are
idempotent. Two forms with the same characters, such as anagrams (`cat` and
`act`) or `1` and `11`, would have the same parts, so the shortest run that
one holds and the other does not is added as a part, and the two codes
differ. Over the first 19,821 distinct words of a dictionary, 4 share a code
at 64 and 64 before that part is added, and none after.

A code is a bound, so it is built by the greatest and the least, which do
not grow with how often a constituent recurs or decay with a whole's
length; a whole's greatest dominates every sub-span's by construction, so
containment order holds inside a whole without projection. The code is
read by the first Pi alone, for the what and for the where; nothing joins
the codes of a reading's percepts.

A percept's identity carries no absolute address: a basic percept is a row
of the first Pi over its parts and wholes (`BasicPercepts.reserve`), and its
form is told apart from every other by its set of parts. An occurrence is
that identity at its own place; two occurrences of one identity at
different places are the same percept twice.

### Percept-to-Concept Seam {#percept-concept-seam}

**The rows' signed values.** For one reading, the first Pi gives each basic
percept present two values, for and against, from its parts, its properties
and its against entries; it is the only reader of the perceptual code (Alec,
2026-11-19: "the perceptual code is used exclusively to distribute codes in
the hypercube, and the only thing touching that is the pi"). The Not
(`NotLayer3`, `order_zero_not`) joins each row's two values into one signed
value, for less against, $s_i = f_i - g_i$ in $[-1, 1]$; an entry of the Not
gives one concept that row's pair exchanged, $g_i - f_i$. The reading's own
`.where` is its own, and its `.when` is the stamp of the model's clock; the
where enters at the first Pi, as a look's match of place.

**The concepts of order zero.** A sparse Sigma (`SparseSigmaLayer3`,
`order_zero`) reads those signed values: for a concept $c$,

$$y_c = \tanh\Big(\sum_i w_{ci}\, s_i\Big),$$

the sum over the concept's entries, $w_{ci}$ at zero or above and $s_i$ the
signed value of row $i$, the value or the not-value as the entry chooses. It has no bias, and
its rows are compressed, as at every order above. It is monotone: with no
Not entry, raising a row's value never lowers a concept. The size of an
entry is the Sigma's weight and its sign is the Not's; the gradient makes a
Not entry where it carries an entry through zero, and a declared Not member
makes one explicitly. Order zero is n:n on basic percepts: each nominated
row brings one concept, which starts reading only its own row, at one, and a
member that joins later writes an entry at one on its row. That is how a row
starts and not a limit: a row may be read by any number of concepts, and a
concept may read any number of rows. A concept reads no
percept's code, no place and no time, so it answers only to the rows it is
given: the concept of `alpha` reads nothing on "beta delta", and where a
form's holder stands in its place (`archer` in `archery`) the row reads
against and its concept reads nothing. The Sigma of order zero is not
penalized.

Conceptual fields carry no per-position tensors or per-concept coordinates.
A concept is opaque and keeps the `.where` and `.when` it was given; no
concept has a weight on a where wire. See
[Architecture](Architecture.md#sigma-and-pi-layers).

`WhereRegistry` assigns each model fixed, nonoverlapping ranges for each
input modality's places, PartSpace rows, all WholeSpace rows, symbol lanes
and the live conceptual slots. Each symbol lane has one address. Text's
input percept and symbol occurrences carry the word's input byte start;
repeated occurrences use their different input locations,
without extra symbol slots. Mixing serial readers declare word slots, so their
fixed input-address extent is the existing word-slot capacity times the existing
per-word residual-part capacity (at least the raw InputSpace extent). This
converts unit counts into byte addresses using the fixed residual layout.
The address guard still rejects overflow. Aligned readers retain their raw-byte
extent. The mode is resolved before this fixed registry is constructed. Thought-produced symbols use their symbol-row
address. The registry encodes locations through one model-owned `.where` ladder, held as
the one fixed where parameter (see "Modalities" above). Its long period
covers the registry and its short period resolves adjacent locations. The
band carries the address; an integer location is decoded from it. Codebook
identity remains the row index. Every perceptual
codebook reserves physical `nVectors` at construction, and admission changes
occupancy in that allocation; checkpoint capacities must match.

Concept ids persist. Above order zero the structure repeats: a Pi whose rows
are symbols over the concepts below, each symbol two unsigned rows, one on
each lane, then the Not, which joins each symbol's pair into one signed
value, for less against, then a Sigma that reads those values with weights
at zero or above ([Architecture](Architecture.md#sigma-and-pi-layers)).

### Geometry Split {#percept-geometry-split}

The implementation keeps percept and concept/symbol carriers distinct:

| Property | Percepts |
|---|---|
| space | positive unit hypercube `[0,1]^D` |
| coordinate | per-axis presence |
| opposite | complement `1-x` |
| sides | one-sided |
| metric | membership distance; presence MSE for the marked store |
| magnitude | presence per axis |

| Property | Concepts |
|---|---|
| space | code `[form \| meaning]` in the unit cube: the form in the percept cube, the meaning in two nonnegative lanes |
| coordinate | form presence; meaning as a placement direction scaled by trust |
| opposite | `not` exchanges the two evidence lanes; no coordinate is negated |
| sides | a concept's value is one signed number whose size is its certainty, stored as two unsigned lanes, its positive part for and its negative part against; "both" is two concepts, one for presence and one for absence (Alec, 2026-11-19) |
| metric | parthood both ways. One function, `ConceptualMatch.match`, compares a content with conceptual space for the reference and field readers and returns how much of each candidate lies in the content and how much of the content lies in each candidate, on the meaning lanes where the code has them. The other parthood reads (the hierarchy, the basic percepts, case selection) call the same measure directly (`IdentityRows.meet_ratios`). The measure is provisional ([Mereology](Mereology.md#parthood-as-shared-size)) |
| magnitude | the for lane's radius is the concept's trust |

A symbol has no code: it is two unsigned rows of a Pi, one on each lane, and
is held in tensors apart from the concepts. The orthographic Lexicon (below)
keeps a sign, as form content.

A concept's meaning direction is placed, not optimized: at each
kept close a placed row takes one meet-ratio subgradient step toward its
context, with negatives drawn deterministically from the
row's content key as a brake against collapse, and is renormalized in the
positive cube (`MereologicalCodes.place_rows`). A raised class is placed by
its context like any concept. The context is the field at every order: what
is present and what the folds give from it, so a present token brings its
types (`FieldProjection.close`). It is taken before any weighting by the
likeness of places, so no place is moved by a value computed from places, and
a step moves a row only relative to rows of its own order.

A symbol is also not a synonym for a concept or a whole. A concept is a
kind, a signed value a Sigma gives; a symbol is a thing, a row of a Pi over
the conceptual space of one order, made implicitly by the Pi or nominated by
MTM when its constituents always occur together. There is no symbol of a
concept: a symbol is a region of sameness that may hold one concept or
several ("my dog" and "Voyager" are two concepts and one symbol), and none is
made for a concept as a rule (Alec, 2026-11-20).

### SBOW Situates Concepts, Not Percepts {#percept-sbow}

The `conceptual_sbow_loss_codes` path in [`bin/embed.py`](../bin/embed.py)
situates concept codes by substitutability on the non-serial path. The
canonical serial path places meanings instead (above): sentence context
attracts each observed concept and deterministic negatives repel it, without
making the dictionary optimizer-owned.

Percepts are deliberately not SBOW-situated:

1. Percept identity is grounded by nearest-row decode in its perceptual metric;
   moving a row changes the token or signal it denotes.
2. Byte spans, `.where`/`.when`, and the meronymic tower anchor its structural
   role.

Distributional movement would therefore break grounding even if the
composition algebra remained invertible. Concept codes can move by
substitutability because their identity is mediated; percept rows cannot.

### Bounded Encodings {#percept-bounded-encodings}

The bounded ranges admit efficient fixed-point storage:

| Format | Range | Intended carrier |
|---|---|---|
| UNORM | `[0,1]` | percept presence |
| SNORM | `[-1,1]` | signed concept/symbol values |
| Q-format fractional | bounded signed or unsigned range | CPU/DSP equivalent |

Training remains in `float32`/`bf16`. Forward-time range enforcement keeps a
float master and uses a straight-through clamp or row projection; the
percept-to-concept seam does not use tanh/atanh. Over a bounded unit range,
SNORM16 has a roughly `3e-5` uniform grid and is finer than bf16 across most of
the interval; SNORM8 is materially coarser.

Learned maps over conceptual codes (the meaning-lane kernels, marker
translations) keep values in the unit cube by construction. A map acts in the
cube's centered chart, $x \mapsto \operatorname{atanh}(2x - 1)$ with finite
rails (`cube_chart`), and its result is mapped back by the inverse chart
$y \mapsto (\tanh y + 1)/2$ (`cube_unchart`), whose whole real domain lands in
$(0, 1)$; a translation adds in the chart and is exactly the identity at zero
(`cube_translate`), and an additive combination is scaled to stay within the
chart's range. Inputs are asserted to lie in the cube (`assert_cube`), and no
result is clipped, so a declared inverse holds.

### The Cube as a Lattice; Two Form Subspaces

Under the componentwise order the percept cube $[0,1]^D$ is a lattice with
*nothing* at the bottom and *everything* at the top; componentwise `min` and
`max` are its meet and join (`Ops.intersection` and `union`), and with the
complement $p \mapsto 1 - p$ as its order-reversing involution it is a Kleene
algebra: De Morgan exact, with no sign pooling. Parthood is a partial order on
it: a part and a type need not be comparable, ties are allowed, and mutual
containment means equal vectors. The unknown, $p = \tfrac12$ (bipolar zero), is
fixed by the complement and by meet or join with itself; false absorbs the meet
and true the join, and nothing pools: $\min(1 - x, y) \ne \min(x, 1 - y)$ in
general, so a composite still says which constituent was negated. On a
concept's two evidence lanes the same order gives Belnap's four values:
the field's `intersection` and `union` are the weighted least and greatest
on each lane (`SparseLayer.fold_presence`), and
`not` exchanges the lanes. The kernel over codes
(`Ops._conjunction_kernel`) is not one of these connectives: it binds code
directions by a normalized product, a binding in the vector-symbolic sense
whose sign would be a phase, so it cannot carry negation; the lattice pair is
offered to grammars as `intersection` and `union`.

One percept vector of `percept_width` coordinates serves every tower (Alec,
2026-10-26). At text-model construction it splits into a parts half
`[0, identity_width)` of `identityPairDim` coordinates and a wholes half
`[identity_width, percept_width)` of `propertyDim` coordinates, 64 and 64
(Alec, 2026-11-19), each part or whole drawn on `identityOnes` and
`propertyOnes` coordinates, 3 by default (`resolve_identity_layout`). Each
half holds two lanes, for and against (`BasicPercepts.codes` and
`against_codes`). The `.where`/`.when` band follows `percept_width`. Numeric
configurations have no wholes half. The split is fixed at construction, and
a checkpoint of another layout is rejected.

| row | parts half | wholes half |
|---|---|---|
| part (a character or an added run) | its fixed bits (`WordIdentity.atom_code`) | its bits drawn again in the wholes half's coordinates (`WordIdentity.catalog_code`) |
| whole (a row of the table of wholes) | not read: a whole bounds only the wholes half | the OR of the catalog codes of the parts that belong to it, or the AND of the wholes it is given by (`PrimitiveProperties.codes`); the class of all characters is all ones |
| basic percept | for: the greatest of its parts' codes; against: the parts its nearest holders add | for: the least of its wholes' codes, its own tightest whole among them; against: what that bound cuts away (`BasicPercepts.against`) |

A part's content lies in the parts half and a whole's in the wholes half;
the other half holds the bound that lets one coordinatewise order read both.
The parts half sorts by containment of parts and the wholes half by what
belongs to a whole, so a narrower whole's code is at or below a wider one's
with no rule for it. Location selects which parts and wholes an occurrence
has; no `.where`, `.when` or span coordinate enters a code.
`BasicPercepts.subspace_bounds` computes the two halves of a basic row, and
`FormContainment.audit` certifies nesting within each half and among basic
percepts, reporting ties without failing them. Composed
wholes keep the leaf-weighted centroid of `compose_blocks` and its exact
inverse `split_centroid`, since a join has no inverse; a description's form
coordinates are likewise folded by centroid so they stay in the cube.

### Live Paths and Migration Status {#percept-live-path}

The tracked baseline enables the global meronomy chart, but the actual carrier
still depends on the input and synthesis mode:

| Path | Live representation |
|---|---|
| measured tensor (`input_presence=True`) | normalized to `[0,1]` |
| text embedding (`input_presence=False`) | retained in `[-1,1]` |
| radix/meronomy percept store | UNORM `[0,1]`; presence decode |
| orthographic Lexicon | signed projective unit ball |
| concept code | `[form \| meaning]` in `[0,1]`; evidence in two lanes |

The canonical radix/property path is membership-native throughout: the codebook
read and the many-to-one word union (max) consume and return `[0,1]` directly,
and there is no learned perceptual fold. The signed truth evaluation chart is not
used at this seam. The orthographic Lexicon remains a compatibility store and
is not the canonical `BasicModel.xml` surface path.

### Mereological Guarantees {#percept-guarantees}

The encoding is hybrid.

**Guaranteed by construction:**

- Byte-position structure - radix longest-match, slot order, the exact
  id-to-bytes table, `.where`/`.when` brackets, and run containment - does not
  depend on vector position.
- When a fold is configured as invertible/butterfly,
  `compose(generate(y)) == y` follows from its LDU/butterfly form. A bare linear
  fold does not provide that guarantee.

**Anchored by the perceptual metric:**

- Vector-to-token identity uses complement-aware presence MSE on the marked
  percept store and wrapped/projective lookup on the signed Lexicon path.
- A form's code is the set of its parts, and that set determines which form
  it is: two forms that would have the same parts are told apart by an
  added part. Radix supplies the sentence's forms through a trie, its
  longest match; it is not place-value arithmetic. The way back reads each
  form's stored characters in order (`RadixLayer.inverse_table`).

Moving percept rows would leave byte structure and configured invertibility
intact while redirecting nearest-row decode. The model could compose and
decompose consistently yet name the wrong token, which is precisely why
percepts remain anchored.

See also [Mereology](Mereology.md) and [Logic](Logic.md).

---

## Codebook Similarity Metric

`Codebook` wraps `VectorQuantize`. Similarity metric per space:

| Store | Geometry and metric | Retrieves |
|---|---|---|
| PartSpace radix/meronomy | `[0,1]^d`; presence MSE | percept-presence row |
| WholeSpace property basis | `[0,1]^d`; property-membership metric | whole-percept property row |
| SymbolSpace | a symbol is two unsigned Pi rows, one for each lane, with no code of its own; serial references keep their event contract | the symbol's row and lane |
| Conceptual similarity codebook | unit rows; dot product | SBOW concept row |

The ConceptualSpace entry is the concept dictionary's *situating* metric, not
the forward concept-production path. Its input magnitude carries belief
certainty, and lookup ranks $\arg\max_i(x \cdot c_i)$.

### Presence and Wrapped-MSE (Perceptual / Symbolic)

These codebooks store *what something looks like*, so the right notion is
coordinate-wise distance. The marked percept store uses
`_presence_mse_score`: ordinary unwrapped MSE on `[0,1]^d`, monotonically
remapped so identical rows score `1` and complementary corners score `-1`.
The signed Lexicon and unmarked stores use `_wrapped_mse_score`, which first
wraps the coordinate difference into the signed unit cell. In both cases,
lookup chooses the row with the largest similarity (equivalently, the smallest
MSE in the active geometry).

For an ordinary, unwrapped Euclidean codebook, retrieval expands
$\|x - c_i\|^2$:

$$
\|x - c_i\|^2 = \|x\|^2 + \|c_i\|^2 - 2\,(x \cdot c_i)
$$

$\|x\|^2$ is constant across $i$ and drops from argmin:

$$
\arg\min_i \|x - c_i\|^2 = \arg\max_i (x \cdot c_i - \tfrac{1}{2}\,\|c_i\|^2)
$$

`VectorQuantize` can therefore keep $\|c_i\|^2$ in `_b_norms_sq`:

```python
indices = (flat @ codebook.T - 0.5 * b_norms_sq).argmax(dim=-1)
```

One matmul + one broadcast subtract + one argmax. Skips the `sqrt`, the per-row
$\|x\|^2$ add, and the cdist autograd plumbing.

### Dot product (Conceptual)

> **Note.** This dot-product metric is the `similarity_codebook`'s retrieval
> metric (used by the substitutability / SBOW *situating* signal), NOT the
> forward concept-production path. Concepts are produced by the Pi and the
> Sigma of each order, and a symbol has no code. Dictionary identity and
> evidence are separate. See **ConceptualSpace → The cycle of the orders**.

ConceptualSpace concepts are *named directions* in belief space. $x \cdot
c_i$ gives the *signed strength of belief that $x$ affirms concept $i$*:

- $+1$ fully affirms; $0$ orthogonal; $-1$ fully denies

Two consequences:

1. **Codebook must be unit L2-norm** for this read; the VQ/EMA paths that use
   it maintain their own unit invariant.
2. **Input must NOT be normalized.** The magnitude *is* the certainty signal.
   Cosine similarity would divide it out.

For *ranking*, $x \cdot c_i$ and $\cos(x, c_i) = (x \cdot c_i) / \|x\|$ are
monotone-equivalent (positive constant cancels). Omitting input normalization
preserves certainty and costs less:

```python
# codebook is unit L2-norm (maintained by EMA)
indices = (flat @ codebook.T).argmax(dim=-1)
```

### Configuring the metric

`use_dot_product` is a class attribute on `Codebook` (default `False`). Set it
on a Space subclass to opt in --- `ConceptualSpace` does this. Inputs are not
normalized, so the effective meaning of the underlying
`VectorQuantize.use_cosine_sim` flag is "codebook unit-norm; rank by dot
product".

---

## Mereological rank and conceptual order

Native PartSpace and WholeSpace codes remain in the same cube as their
rank changes. Rank describes the size of a part or the fineness of a whole
cut. There is no perceptual fold sequence to invert.

Conceptual order counts symbolizations: observations are order 0,
particulars are order 1, kinds are order 2. A concept's order is read from
the block of the concept inventory its row lies in (`order_slice`), never
stored beside the vector ([Orders §12](Orders.md)). One word form can address
concepts at several orders; `interpret` reads the expected one, else the one
of highest order, as specified in
[Lexicon](Lexicon.md#word-forms-and-concept-orders). A proper name selects
a particular, not an order-0 event.

## Codebook Uniqueness Contract

Every codebook entry is identified by its **row index** and must carry
**distinct `.what` (`WhatEncoding`) content**:

- **`.where` --- a location, never a codebook row key.** A model-owned
  `WhereRegistry` assigns each input modality's places, then fixed PartSpace
  and WholeSpace slices, then symbols and the live conceptual slots. The
  symbol range holds two addresses for each concept row, one for each lane,
  using exact integer coordinates so addresses remain distinct on CPU, CUDA
  and MPS.
  Constructing another model cannot move these ranges, and the code of each
  address is held in one fixed parameter (see "Modalities"). Content lookup and
  native relation references do not use where coordinates as identities.
- **`.what` --- distinct prototype content.** Identical `.what` collapses to the
  same parthood identity (`equal(A, A) = 1`) --- a redundant pair the network
  can't distinguish.

How distinctness is kept:

| Source | Mechanism |
|---|---|
| PartSpace percept store | native ids from the canonical radix tiling; a form's code is the set of its parts, and two forms that would share one are told apart by the shortest run one holds and the other does not (`WordIdentity`); a basic percept is a row of the first Pi over its parts and wholes (`BasicPercepts.reserve`) |
| PartSpace Lexicon | cosine-margin pode/antipode SBOW training, for trained Lexicons |
| InputSpace vocabulary | shares PartSpace's Lexicon (text); manual (raw) |

No regularizer enforces `.what` distinctness on the WholeSpace or
ConceptualSpace codebooks: `ImpenetrableLayer`, the overlap penalty designed
for it, is a library class that no production path constructs
([Mereology](Mereology.md#mereological-fusion)). Distinct content there is
learned, and together with the distinct row indices it keeps the parthood
lattice well-formed.

---

## Lexicon (Projective Unit Ball)

The **Lexicon** ([`bin/Layers.py`](../bin/Layers.py)) backs PartSpace
word embeddings and WholeSpace whole-percept prototypes. Each row is a vector
$w_i$ in the **projective unit ball** --- the closed ball $B^D = \{x : \|x\|_2
\le 1\}$ with the **negation identification** $w \sim -w$ realizing real
projective space $\mathbb{RP}^D$.

**Terminology pin** --- three notions sometimes conflated:

- **Pode** of $(a, b)$: midpoint $(a + b)/2$; SBOW positive-pair attractor.
- **Wrapped pode**: midpoint via the $\pm$-quotient, $(a - b)/2$; the
  midpoint through *negation* of $b$.
- **Antipode** of a single point $p$: furthest point. On the flat torus
  unique ($\mathrm{wrap}(p + 1)$); on $\mathbb{RP}^D$ **not unique** --- the
  maximum-distance set is the orthogonal hyperplane.

So $-w$ is the **negation** of $w$, *not* the antipode.

### Distance and lookup

For $a, b \in B^D$ the projective squared distance is

$$
d_{\mathbb{RP}}^2(a, b) = \min(\|a-b\|_2^2,\; \|a+b\|_2^2)
= \|a\|_2^2 + \|b\|_2^2 - 2\,|\langle a, b\rangle|.
$$

With $\operatorname{pode}(a, b) = (a + b)/2$ and $\operatorname{wpode}(a, b)
= (a - b)/2$, $d_{\mathbb{RP}}(a, b) = 2 \cdot \min(\|a -
\operatorname{pode}\|,\ \|a - \operatorname{wpode}\|)$. The lookup picks
whichever rep of $b$ ($b$ or $-b$) is closer to $a$.

Sorting by smallest $d_{\mathbb{RP}}^2$ = sorting by largest
$\operatorname{score}(x, w_i) = |\langle x, w_i\rangle| - \tfrac{1}{2}\|w_i\|_2^2$.

Implementation: cache `W_norm2 = W.square().sum(-1)` once per optimizer
step; top-k is `(x @ W.T).abs() - 0.5 * W_norm2` followed by `torch.topk` ---
dense matmul + abs + broadcast subtract. No $V \cdot D$ outer-product.

The `Lexicon` API:

```python
lexicon = Lexicon(V, D)
lexicon.project_unit_ball_()         # after optimizer.step()
W_index, W_norm2 = lexicon.lookup_index()

# Projective (RP^D) --- antipode-aware, default.
idx, dist_sq, scores = Lexicon.topk_rp(x, W_index, W_norm2, k=32)

# Plain L2 --- for sites where w and -w are distinct.
idx, dist_sq, scores = Lexicon.topk_l2(x, W_index, W_norm2, k=32)

# Pairwise primitives:
Lexicon.rp_distance_sq(a, b)
Lexicon.rp_similarity(a, b)
Lexicon.rp_pode(a, b)
Lexicon.rp_wrapped_pode(a, b)
Lexicon.rp_closer_rep(a, b)          # sign(<a, b>) * b
```

For $V \gtrsim 10^5$, use `topk_rp_chunked` to bound peak score-tensor size.

### SBOW training: pode (attractor) and antipode (repulsion target)

- **Pode (attractor).** Positive-pair updates pull $a$ and $b$ toward
  $\operatorname{pode}(a, b)$; the gradient picks the closer of $b$ and $-b$
  for shorter-arc attraction.
- **Antipode (balancing repulsion target).** Negative-pair updates push the
  row toward the furthest point. On $\mathbb{RP}^D$ this is a $(D-1)$-sphere,
  so SBOW samples a representative orthogonal direction.

Negative-sampling gradient has two regimes by $\mathrm{sign}\langle a,
b\rangle$: positive case is standard contrastive repulsion along $(a - b)$;
negative case pushes $a$ away from $-b$ along $(a + b)$.

After every optimizer step the trainer calls `lexicon.normalize()`, clipping
$\|w_i\| \le 1$. `W_norm2` should be refreshed when weights change.

Torus primitives (`Lexicon.wrap`, `Lexicon.delta`, etc.) and the
`torus=True` constructor flag remain as **legacy** static methods (the
earlier Lexicon used the flat torus $T^D = [-1, 1)^D$ with wrapped MSE). New
code must use the `rp_*` primitives.

---

## InputSpace

**Role.** Receives the raw source buffer and lifts it into the model's
internal working dimensionality.

**Text mode forward.** Delegates tokenization to `Lex`, producing a span table
of `(start, end, type)`. Each span $\to$ a vector with two components:

- `nWhat` dims --- token content, encoded via `Basis` / `Codebook` (the word
  embedding lookup).
- `nWhere` dims (4) — two quadrature pairs encode one registry location.
  One model-owned encoding serves input positions, PartSpace, every
  WholeSpace, and symbols, written once into one fixed parameter, a row
  for each address (see "Modalities: One Perceptual Space Each"). The long period is the next power of two covering
  the complete registry plus its seam guard. The fine period is at least
  256 positions and grows with large registries to keep float32 coarse-phase
  error inside the branch bound. Integer modulo occurs before float conversion;
  `decode_index` combines the two phases as integers, preserving adjacent
  addresses above 2²⁴. A word occurrence uses its input byte start in the
  registry's input slice, not its part row's address. Extent endpoints stay
  in perception's span metadata. A part's end follows from its start and
  byte length; a whole's end cannot be recovered from the start-only band.
- `nWhen` dims (4): the same two-rung construction over field time, with
  periods derived from LTM capacity. `.when` is model time, one tick per
  sentence and per new thought idea on one clock
  ([STM.md](STM.md#memory-module-and-thought-stamps)). Every element
  observed in one field has the same advancing subjective `when_time`.
  Per-word reconstruction excludes these shared dimensions, while grammar
  still transports them. Any field-time objective should score the timestamp
  once per field. The model has one integer clock, and its code is made by
  one encoder when the clock advances and held on the model beside it;
  every reader, the spaces and the tense operators among them, reads it
  there. Tense is the onset-vs-`now`
  relation; `shift_time` rotates both pairs coherently, each at its own
  $\omega$. Both `.where` and `.when` are start-only two-rung ladders
  (`WhereEncoding`, `WhenStartDurationEncoding`).

Result: `[nActive, nWhat + nWhere + nWhen]` tensor.

**Text mode reverse.** Inverts the span encoding: each vector $\to$ nearest
codebook entry, then spans $\to$ characters via the stored offset table.

**Numeric mode.** On a numeric file whose grammar declares field operators
only (`data/MM_xor.xml`, `data/MM_image.xml`), `NumericPercepts` reads each
number as a level percept at its place ([Level Sets](#percept-level-sets)),
and no layer is learned at the input. Otherwise tensor data passes through unchanged;
`LiftingLayer` projects native input dim (e.g. 784 for MNIST) to `nDim`.
Non-embedding inputs are scaled to `[-1, 1]` via the global data min/max.

**Key parameters.**

| Parameter | Description |
|-----------|-------------|
| `nActive` | Sequence length |
| `nDim` | Output dim per vector |
| `lexer` | Set on `<WholeSpace>`: lexing is analytic cutting, and InputSpace executes the intake |
| `codebook` | Whether input values are discrete |
| `demuxed` | Store what/where/when independently |

The band is architectural, not a per-Space setting:
`architecture.canonical_shape(section)`
([`bin/architecture.py`](../bin/architecture.py)) gives (nWhere, nWhen) =
(4, 4) on InputSpace, PartSpace, ModalSpace, WholeSpace and SymbolSpace, and
(0, 0) on ConceptualSpace and OutputSpace. Percepts and symbols are located
and timed; a concept is one opaque code that has generalised over where and
when, and the answer has neither.

**Invertibility.** Always non-invertible; reverse is a separate reconstruction
using the span table.

### Document streaming and `valid_mask`

Documents longer than `nOutput` bytes are not truncated. `TheData` maintains a
per-row cursor `(doc_idx[b], offset[b])` and `next_tick()` returns
`(input, output, hard_eos)` where `input` is a `[B, nOutput]` slab containing
the next $\le$ `nOutput` bytes from each row's current document. `hard_eos[b]` is
a host-side bool set when row `b`'s cursor exhausts the current document. A
short fill at document end NULL-pads the slab tail; `valid_mask: [B, K] bool`
flips False for padded positions, and state-mutation propagation skips them.

**Cursor universal --- trial mode for non-AR data.** `next_tick()` is the single
dispatch for both AR text byte (rolling cursor) and non-AR data (numeric).
In trial mode (`slab_bytes` not set), each tick yields one batch of trials
with `hard_eos = [True] * B`. The runEpoch outer loop drives `ds.next_tick()`
directly for both modes; the DataLoader exists only so existing tests can
grab `loader.dataset`.

**Packed word-loop training.** A self-supervised word model may instead set
`<packSentences>true</packSentences>`. `runEpoch` then drives
`next_packed_tick()` over the same B contiguous streams. Each row appends
chronologically consecutive, complete sentences until another sentence would
overflow W, the raw byte slab, or the root FIFO. Boundary tensors let CSLang
soft-reset that row between sentences without padding all rows to a shared
sentence length. The last sentence drains through loss and backward before
the eager boundary resets it. `hard_eos` retains its stronger meaning:
exhaustion of that row's contiguous corpus stream.

`_end_of_stream` is a host-side `list[bool]` diagnostic only; the canonical
hard-reset signal is the cursor's `hard_eos`.

### Within-sentence AR retirement (2026-05-14)

There is no next-token objective within a sentence, and no next-sentence
objective either: nothing estimates a sentence, and what carries from one
sentence to the next is the recent situation (see
[STM.md](STM.md#11-the-recent-situation)). Within-sentence training is
IR-only. `InputSpace.forward` emits `[B, N, D]` (left-aligned, right-padded
to N) and `_forward_per_stage` runs a single masked-LM pass:

1. **Stem**: `InputSpace.forward` + `PartSpace.forward` $\to$
   `[B, N, D]`.
2. **Mask**: `create_ir_mask` replaces a `mask_rate` fraction of WHAT
   positions with `NULL_PERCEPT`; the pre-mask event is stored on
   `_ir_pre_mask_input` as the loss target.
3. **Body**: T stages on B rows (no per-cursor walk, no causal
   mask).
4. **Head**: `outputSpace` $\to$ `[B, N, predDim]`. The head is a side
   channel --- IR loss is computed at the subsymbolic (PS), not at the head.

`runBatch` reads `_ir_mask_positions` and `_ir_pre_mask_input` and scores the
PartSpace subspace against the pre-mask event at the masked positions by MSE.
Reconstruction is always from concepts; there is no selection among
concepts, symbols or both.

---

## PartSpace

**Role.** Find a sentence's forms and their parts. Native percept ids come
from the canonical radix tiling, one lookup at a sentence's start, and the
first Pi is read only on the rows that lookup names. A form's code is the
set of its parts ([Mereology](Mereology.md#percept-code)); a basic percept's
row is made when MTM nominates it on recurrence. Membership reads the parts
found, never re-expanding and re-matching bytes.

PartSpace owns the percept store and the configured synthesis front end.
`synthesize_word_parts` takes the greatest of active constituent codes in
`.what` and retains the first active constituent's position band. There is
no `sigma`, per-pass fold stack or learned transformation of the native
cube. Each form's stored characters supply the exact way back.
`forward(in_sub, cs_out=None)` keeps the shared perceptual interface. A look
dims what lies away from where it asks by degree, with no floor
([Architecture](Architecture.md#one-attention)).

### The Two Stems, and Where a Word's Bracket Comes From {#two-stems}

The stem is the first step of reading. `PartSpace` cuts a row's bytes
into percepts and lays them in a table of slots. There are two stems,
and the configuration decides which one runs (`PartSpace._embed_radix`).

- **The word-major stem** (`_embed_radix_word_major`; the files that
  compose or examine words). A slot is a word. Each word carries its
  parts and the byte at which each part starts
  (`InputSpace._ar_word_part_offsets`, `[B, W, P]`); the first of them
  is the word's own start.
- **The whole-slab stem** (the rest of `_embed_radix`; a file with field
  operators only). A slot is one emitted percept. A word held as one
  percept takes one slot; any other word is spelled out, a slot for
  each part, and the bytes between two words take slots too. The stem
  computes each word's interval (`Meronomy.percept_spans`) to cut the
  word texts (`word_texts`), to say which word a slot's part belongs
  to (`word_groups`), and to keep as the word's interval
  (`word_intervals`), since a slot need not be a whole word.

A **word bracket** is the interval of the row's bytes that a word
occupies. The first look of a sentence, the open run, reads the percepts
inside every word bracket at once (`FamiliarTier.open_sentence`), with a
grammar or without one. Attention's later looks are activities over the
sentence's percept bank and are not made of word brackets
([Architecture](Architecture.md#one-attention)). `ModelAttention.stage_input` makes the brackets
(`model._attention_spans`, `[B, W, 2]`) from the intervals the stem
kept. Every stem keeps each word's interval, as `Meronomy.percept_spans`
cut it, in `word_intervals` beside `word_texts` on
`PartSpace._forward_input`, and `stage_input` reads them the same way
on both stems, trimming a word's leading and trailing whitespace. The
row is not searched again for a word's text. A row with no word texts
has no brackets.

"hello world" on the whole-slab stem, at a width of eight slots:

| Slot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| holds | h | e | l | l | o | the space | w | o |
| the word of that part | 0 | 0 | 0 | 0 | 0 | none | 1 | 1 |
| the bracket kept at that slot | "hello", bytes 0 to 5 | "world", bytes 6 to 11 | | | | | | |

The second word is cut at the eighth slot, and its text and bracket are
whole, since both come from the row and not from the slots. On this
stem the brackets are kept by the word's number on the same row of
slots: the bracket of "world" is at slot 1, which holds a letter of the
first word. Its interval is the word's, and the first look reads by the
interval. The key taken from the slot is the letter's; only a walk
reads a key, and no walk runs on this stem in the tests.

---

## ConceptualSpace

**Role.** Forms concepts from perceptual and symbolic sources, by one
structure at every order: a Pi, then a Not, then a Sigma. At order zero the
first Pi's rows are the basic percepts, each with two values, the Not offers
each row's value and not-value, and a sparse Sigma over those values gives
the concepts ("Percept-to-Concept Seam" above). Above it a Pi makes symbols
over the conceptual space of the concepts below, the Not offers each
symbol's value and not-value, and a Sigma makes the concepts of the next
order. A Pi makes things and a Sigma makes kinds; an order rises at a Sigma
and nowhere else. Symbols and
concepts are held in separate tensors. `CS.forward` itself performs STM
bookkeeping. The design is in
[Architecture](Architecture.md#sigma-and-pi-layers).

**Aligned serial geometry.** The BasicModel path is distinct from the
sparse-parallel phase described below. PS and WS each expose eight live
locations. With `subsymbolicOrder=4`, their three cumulative sigma/pi folds
are fitted without moving feature coordinates across locations and stacked as
`[B, 6, 8, D]`. CS emits the equal local mean while retaining that six-source
carrier and the ordered fold paths. The concept written to STM carries actual
subsymbolic order 3 plus support from orders 1, 2, and 3 on both towers. Its
row lies in its order's block of the fixed `nVectors` inventory
(`order_slice`), so the row's address gives its order. The same location
rule applies when aligned binding is chosen in parallel. `ConceptualCombine` is the separate conditional mixing-matrix
path and is not part of serial BasicModel.

**Geometry.** A concept's code is `[form | meaning]` (`MereologicalCodes`):
the form is the weighted centroid of the percepts of its written definitions,
or its own basic percept's code when every weight is zero (see
[Mereology](Mereology.md)), and the meaning holds two nonnegative lanes, the
for lane being the concept's placement direction scaled by its trust radius.
That form belongs to a concept of order zero. A concept above order zero
carries no perceptual form; a reader that needs one takes the form of the
concept's word, through the definition that maps the word to it.
Every coordinate lies in $[0, 1]$, and no sign is stored. A concept's
activation is one signed number whose size is its certainty, stored as two
unsigned lanes.

![A concept: sigma over pi, one signed code, two evidence lanes](diagrams/concept_geometry.svg)

**Owned matrices and carrier.** The conceptual inventory owns the first Pi
(`BasicPercepts.pi`, a `PerceptPi` stored as compressed rows with an index
of columns beside them), the Not and the sparse Sigma of order zero
(`order_zero_not`, `order_zero`), and above order zero a Pi, the Not and a
Sigma for each order, the Pi and the Sigma stored as compressed rows
that grow into capacity allocated beforehand. A row of any Pi exists only
when the Pi makes it or MTM nominates it on recurrence, and each row brings
one concept to start with, so every order's Sigma is square in its rows: as
many concepts as rows. The
clean-stack STM retains the incoming event and its short-lived tree. Input
reconstruction starts from the live order-zero field, independently of
that tree. On the way back the Pi inverts exactly, and the Sigma returns
the rows' signed values by least squares where there are enough concepts
and not otherwise; a signed value is split into its two unsigned parts before
a percept or a symbol holds it (`signed_lanes`: -.5 is .5 on the against
lane). No input perception spans enter the native decoder.

**The cycle of the orders.** At an order with $n$ concepts:

```text
y_c    = tanh(sum_i w_ci s_i)           # n signed values (SparseSigmaLayer3)
lanes  = signed_lanes(y)                 # stored as 2n unsigned lanes
rows   = Pi(lanes)                       # the symbols, each a row on its own
                                         # lane (PiLayer3)
signed = Not(rows)                       # each symbol's value (for less
                                         # against) and not-value; an entry
                                         # chooses one
y_next = tanh(sum_i w_ci signed_i)      # the next Sigma, w at zero or above
```

A Pi row is $z = \exp(W \log \mathrm{clamp}(m, \epsilon, 1) + b)$, with its
diagonal at one or more, the rest at zero or more and its bias at zero or
less, so a row requires the members it has weight on and never exceeds its
own reading. A Sigma's weights are at zero or above and it has no bias, so
it is monotone; the size of an entry is the Sigma's and its sign is the
Not's. A Not entry is made implicitly, where the gradient carries an entry
through zero, and explicitly, by a declared Not member. The types of the
three steps (Alec, 2026-11-20): the Pi has unsigned weights on signed
concepts, "so it must allocate two lanes in order to operate on a given
concept"; the Not takes a thing's two unsigned lanes and gives one signed
value, possibly negated; the Sigma has unsigned weights on those signed
values and "does not need this doubling". A symbol's cone lies in the
conceptual space of one order; a kind collects from every epistemic level
below its own, and between orders only activation passes. The entries of every Pi and of every Sigma above order
zero are learned under a penalty that thins the values of the entries that
exist; the Sigma of order zero is sparse and is not penalized.

`_order_caps()` gives the room of each order. Order zero has as many
concepts as there are basic percepts, and each order above has half the
units of the one below (Alec, 2026-11-19: "Geometric taper is not a bad
rule"), connected to the two rows beneath it, its own order's symbols and
the concepts of the order below; this is storage allocated
beforehand, not a number of entries expected. A concept's order is read
from the block its row lies in (`order_slice`). Participation is detached
EWMA use.

**Activation carrier.** `subspace.activation` remains the ordinary pipeline
presence gate. The conceptual knowing field is the separate paired
`_concept_activations` tensor, whose occurrence axis is retained in the
checkpoint and whose two poles travel through thought effects and the SS
leg. Serial idea vectors and the sentence close have their own contracts
([STM.md](STM.md)); neither is a substitute for this paired field.

**Shared conceptual hierarchy** (Alec, 2026-11-08). `ConceptualSpace` owns one
native hierarchy for word meanings, classes and classes of classes. A class
is placed by its context like any concept; at kept closes a word draws its
classes together among classes and a class its words together among words,
and no position crosses an order. Formal grammar
arguments are ordinary concepts prebuilt one order above words and use this
same learning path. Groups over arguments lie one further order up.
`PartsOfSpeech` owns the declaration mapping and grammar reads, not another
conceptual space or category codebook. The fixed declaration reservation is
included at initial model construction; codebooks and trust columns do not
resize during use. Form and meaning retain their separate coordinates and
learning paths. See
[Architecture](Architecture.md#hierarchical-conceptual-context-alec-2026-11-08).

### Eager word interpretation (fullgraph forward)

The compiled per-batch forward (`BaseModel.enable_compiled_step`,
`torch.compile(fullgraph=True)`) must carry **zero graph breaks**. Concept
admission, definition writes and ConceptualSpace taxonomy updates are host-side:
`.item()` loops, Python dict/set mutation, and reserve activation cannot be
traced safely. They therefore do not run inside the compiled forward. This
allocation is entirely separate from any dynamic WholeSpace property growth.

The reading calls the full `interpret` unary at its eager word boundary,
under either binding. The canonical aligned serial path already has that
boundary; the other readings use `_stage_reading_word_concepts` before
entering their graph. Parts fuse first, and the object's row replaces the
word's row. A new word reserves that inventory row and one definition row
together. Repeated canonical parts are a lookup, even after learned
property memberships move.

`_commit_autobind_from_stash` still runs at the sentence boundary for its
remaining work, including category-codebook updates and recognised words.
Forcing interpretation does not disable those duties. The word-to-object
link is an identity-based definition row in `LTM1`
([STM.md](STM.md#10-ltm-as-the-chain-of-stm-end-states), "Definitions").

`BASIC_FULLGRAPH=0` relaxes the strict gate to enumerate graph breaks;
`MODEL_COMPILE=eager` traces without the Inductor C++ backend.

### ShortTermMemory

ConceptualSpace owns `self.stm`, a `ShortTermMemory` instance: a per-batch
working set of unquantized CS "ideas". `CS.forward` is STM bookkeeping, with
no atomic fold layer; the concepts of the orders (see **The cycle of the
orders**) never enter the STM as its chunks.

| Property | Default | Configurable via |
|---|---|---|
| Capacity | 8 chunks (within Miller's $7 \pm 2$ band) | `<ConceptualSpace><stmCapacity>N</stmCapacity></ConceptualSpace>` |
| Storage | the fourteen `STMChunks.FIELDS` tensors, `[batch, capacity, ...]`, fixed at construction | working state, not saved |
| Cleared | softly at a sentence or document reset (the unfinished sentence's operands); hard at the epoch reset | `ConceptualSpace.Reset` (soft), the epoch reset (hard) |

The signal router (`SymbolSubSpace.languageLayer`) consumes
`stm.snapshot()` as its slab input for grammar op dispatch. The chunks, the
API, serial and parallel sequencing, attentional filtering, the
relative-vs-absolute end-states and the LTM chain of end-states are in the
dedicated [STM.md](STM.md) chapter.

---

## WholeSpace

**Role.** WholeSpace is the top-down analysis tower over whole-percepts. Its
inventory consists only of properties that divide the input field into regions.
It is perceptual and therefore strictly upstream of concepts and symbols.

**Owned state.**

- Every model owns exactly one `model.wholeSpace`, whose `subspace.what` is
  the canonical whole-percept/property codebook at width `nDim`. Every
  ConceptualSpace stage, the byte cut, word fills, lessons and attention read
  this same inventory. `bindingDepth` counts ConceptualSpace stages and
  ramsification slots; it does not multiply WholeSpace. BasicModel initially
  contains roughly eight ASCII-class properties. There is no `wholeSpaces`
  alias or second `analysis_store` codebook.
- The `<analysis>` policy reads primitive memberships and partitions the field.
  There is no `PiLayer` in WholeSpace.
- Dynamic property learning may add rows to this inventory. Its allocator and
  checkpoint policy are independent of ConceptualSpace capacity.

WholeSpace does **not** own word meanings, concepts, definitions, taxonomy,
grammar symbols, or symbol prototypes. ConceptualSpace owns the concept
inventory and taxonomy; the common LTM store owns definition rows; SymbolSpace owns
downstream references and grammar dispatch. Consequently a WholeSpace state dict must not contain a
tensor sized like the conceptual inventory merely because aligned binding is
enabled.

**Whole-percept presence.** Each whole records graded presence in `[0, 1]`
on two lanes, and may hold by degree, such as a distance on a ratio scale
(Alec, 2026-11-19). A whole is a row of the table of wholes defined by what
belongs to it: the catalog parts it holds, or other wholes it is given by
(`PrimitiveProperties.define_by_rows`). Its code is the OR of its parts'
catalog codes or the AND of those wholes' codes, and the class of all
characters is all ones (`PrimitiveProperties.top`); a class of characters
is the OR of the characters taught to it. A basic percept's row reads each
of its wholes where its run of parts lies, and not over the whole reading,
so a space elsewhere in a sentence is no evidence against a word. A whole
witnessed not to hold, a narrower whole the percept does not lie in, tells
against it; a whole not observed asserts nothing
([Mereology](Mereology.md#percept-code)).

**Alignment.** At concept formation, native PS and WS readings bind within
the same attentive field. This is an alignment of the transient `nOutput`
axis, not equality of the two codebook inventories. PS, WS, and CS may therefore
have different `nVectors` values.

**Codebook geometry.** `subspace.what.getW().shape == (nVectors, nDim)`.
Each row describes one whole-percept property. As new properties become useful,
`nVectors` may eventually approach PartSpace scale without changing what the
rows mean.

**One inventory.** One inventory makes every reader agree: the word fill and
the serial open read the same memberships, the byte cut and the lessons teach
the same rows, and a property row has one `.where` address. A checkpoint that
carries a second `analysis_store` copy, concept or symbol prototypes on the
WholeSpace paths, or a WS taxonomy is migrated at load without rewriting the
source artifact: the incompatible tensors are dropped with a warning, the one
copy with changed memberships or observations is retained (competing changed
inventories are rejected), and later symbol and conceptual addresses move
down by the removed copies' capacity, with the persisted location bands
translated and identity and occurrence keys keeping their meaning. The
bracket-context projection starts at zero when loading the former attention
scorer. The retained inventory keeps its optimizer state.

See [Philosophy.md](Philosophy.md), [Logic.md](Logic.md),
[Mereology.md](Mereology.md), and [Language.md](Language.md).

---

## Monotonicity of the lift / lower chain

A map whose weights are all zero or above is an **order-preserving map on
the positive cube**: where one vector is at or below another in every
coordinate, so are their outputs.

$$
a \leq b \text{ componentwise} \Longrightarrow f(a) \leq f(b) \text{ componentwise}
$$

Checked on 4,000 pairs of vectors, one at or below the other, it holds for:

1. **A Pi** with entries of zero or above. Its output lies at or below every
   input it reads (a product of numbers between 0 and 1), so the Pi makes
   the order below: what needs all its members.
2. **The union** $1 - \prod (1 - x)$. Its output lies at or above every input
   it reads, so it makes the order above: what any member is enough for.
3. **A Sigma** $\tanh(\sum_i w_{ci} s_i)$ with every $w_{ci}$ at zero or
   above. It rises with every input, but is not always above it.

It fails as soon as a Sigma has one Not entry. So the chain keeps parthood
lane by lane while no Not entry is read, and a mind that reads no Not entry
speaks only of what is. Monotone weights cannot make
evidence against: that takes the against lane of a percept's code, or a Not
entry. The two lanes keep "both" `[1, 1]` apart from "neither" `[0, 0]`, where
one signed axis would let the two cancel under summation.

See [Mereology](Mereology.md#percept-code), [Logic.md](Logic.md) and
[Philosophy.md](Philosophy.md).

---

## OutputSpace

**Role.** Maps symbolic (or syntactic) vectors to task targets. Four
branches, selected at construction ([`bin/Spaces.py`](../bin/Spaces.py)):

- **Field readout** (`self.field_readout`), on every grammar file except
  one that declares the `pi` and `sigma` folds and no examination or
  vector operator: the output reads the field at the path's end through
  one linear layer, a weight for each concept row of the inventory and
  each evidence lane, and a bias, all zero at construction (`read_field`).
  A retired row's weights are zeroed (`retire_field_row`). The XOR proofs
  read 0 and 1 this way, as $x_1 + x_2 - 2\,(x_1 \wedge x_2)$ with signed
  weights. With this branch `forward` is not used.
- **Nonlinear / activation mode** (`<nonlinear>` on `<OutputSpace>`,
  `self.nonlinear_output`): an `InvertibleLinearLayer` acts on the scalar
  SYMBOL ACTIVATION vector (`[B, nSymbols] -> [B, nOutput]`), wrapped with
  `atanh` $\to$ linear $\to$ `tanh` to reproduce the nonlinearity a
  `PiLayer` would apply internally — only PS / CS may own a
  `SigmaLayer`/`PiLayer`, so this path builds the same nonlinear surface
  directly instead of delegating to one. `forward` reads
  `vspace.materialize(mode="activation")`; `reverse` mirrors it:
  `tanh(linearLayer.reverse(atanh(x)))`.
- **Unquantized regression head** (`self._regression_head`, true when
  `nVectors <= 1` or `<codebook>none</codebook>`): a head that can only
  name a single codebook prototype would have a VQ snap collapse to the
  row-mean, so this branch NEVER snaps regardless of a configured
  `<codebook>quantize</codebook>`. `forwardLinear` is bias-free
  (`x @ W`); a learned scalar intercept `self._readout_bias` (zero-init)
  is added by `_apply_readout` so a `{0,1}` target (e.g. XOR) can still
  shift off the feature mean. `reverse` undoes it via `_invert_readout`
  before the inverse linear. The head is always linear plus bias, never
  squashed.
- **Codebook (quantized, symbolic) head** (`elif self.codebook`, when
  `nVectors > 1` with a codebook): the post-linear event snaps through
  `self.subspace.get_vectors().forward(output)` — genuinely symbolic,
  multi-vector outputs only.

**Forward.** Vector-mode (the two non-activation branches): `y = W_out * x +
b_out` through the configured linear chain (`InvertibleLinearLayer` when
`<invertible>`, else a `LinearLayer` pair), then the regression-head or
codebook branch above. Always `reshape=True` --- the `[B, nSymbols,
symbolDim]` tensor is flattened before projection (`OutputSpace` is the
sole flattener).

**Reverse.** Pseudoinverse of `W_out` (vector-mode; regression-head
un-readout runs first), or the activation-mode inverse above. Text mode
snaps each output vector to the nearest codebook entry (nearest-neighbour
lookup).

**Text mode generation.** Autoregressive: each step's predicted token vector
is snapped to its nearest codebook entry and fed back as input until
`maxResponseLength` or EOS.

**Key parameters.** `nActive`, `nDim`, `nVectors`.

**Layer.** `LinearLayer` / `InvertibleLinearLayer` (vector-mode) or the
wrapped `InvertibleLinearLayer` (activation-mode), with `(bias, temp)` for
ergodic mode.

**Range.** Forward rescales `[-1, 1]` to the original data range via
`Data.denormalize()`. Reverse applies `Data.normalize(x, "output")`.

**Invertibility.** Pseudoinverse (vector-mode); the activation-mode branch
is exactly invertible via its `InvertibleLinearLayer`; not exactly
invertible in general.

## One basic-percept basis

`BasicPercepts` is perception's fixed-capacity table. Each row is a basic
percept, a row of the first Pi (`PerceptPi`) over its parts and wholes, with
for entries joined by the product and against entries, the parts its
nearest holders add, joined by the union. A row exists only when MTM
nominates it on recurrence (`concept_mint_count` readings, a reading being
one sentence of one stream, so a form that stands twice in a sentence is
read once); nothing becomes a row on its first reading, and a new word is
perceived at once but `interpret` cannot find it until its row is
nominated. A basic percept is an occurrence: the word over its occurrences
is its concept of order zero, and the word as a thing is a symbol over that
concept (Alec, 2026-11-20). Each nominated row brings one concept of order
zero to start with, whose column
of the Sigma starts from the row's own read pair
(`BasicPercepts.column`). The first Pi holds only the mereology's entries;
no concept writes into it. Compose whole reads and typed word slots use
this same basis.

Basic-percept rows are not withdrawn. Reservation appends an identity,
promotion retains it, and concept forgetting does not remove it; a full
table raises.

## Common order-zero reverse

`ConceptualField` carries concept IDs, paired evidence, where and when; its
perception member is `None` on the decoder path. `SentenceField` retains the
live input field across the sentence's paths, publishes the serial knowing carrier,
and obtains a separate descended view after understanding. The parallel path
keeps the symbolic carrier for higher-order readers while taking the same
descended view for reconstruction.

The way back inverts each layer as far as it goes. The Sigma of order zero
returns the rows' signed values by least squares where there are enough
concepts, and not otherwise; a signed value on the way back is replaced by
its two unsigned parts before a percept holds it, so -.5 is .5 on the
against lane and nothing on the for lane. The first Pi inverts exactly. At
order zero the way down also carries the vertical spread into the first
layer: a part that is present raises the rows that read it, and a row gives
a share to each of its parts, so where a look's percepts were withheld the
way back reads the expected ones in their place
([Architecture](Architecture.md#sigma-and-pi-layers)).

`FieldReverse.NativeActivity` holds selected native rows, strengths and continuous
percepts at the covered positions. `NativeDecode` uses preallocated spelling,
length and position buffers, and vectorized traversal. A form is put back
from the form alone, its stored characters in order. Its place is recreated
with it (Alec, 2026-11-19: "recreated percepts have recreated locations that
place them in the out buffer"): a concept gives each percept it recreates a
band of where wires, through the same weights by which it reads the where;
the band is matched against the code of every place by a softmax over the
places of the out buffer, and what is put back at a place is the sum over
the recreated percepts of what each is, times its weight there. So the
reconstruction's error reaches both the weights that read the what and those
that read the where. The reverse reads no perception spans. `PartSpace.reverse` and `InputSpace.reverse` carry both the
byte report and differentiable percept face. A missing aligned field or a failing
reverse stage raises to the caller.

Output's `generate` is separate from this reverse. Its occurrence writer selects
terminal rows, preserves both activation poles and supplies output coordinates.
The same decoder then realizes the answer; a numeric output reads the field
through `OutputSpace.read_field` where the file has a field readout, and
otherwise uses its forward head. A numeric input's reverse returns each
place's number from its level (`NumericPercepts.reverse`). Thought updates
knowing at existing field brackets, without extra thought-occurrence columns.
