# Philosophy

This document maps the model's machinery onto three philosophical accounts
of cognition: Kant's analysis/synthesis pair, Ramsey's treatment of
theoretical structure, and Buddhist epistemology (pramana theory). The
through-line is the corrected analysis/synthesis orientation:

```text
WholeSpace (SS)                  PartSpace (PS)
top-down ANALYSIS                   bottom-up SYNTHESIS
Pi -- product, intersection         Sigma -- sum, union
starts from UNITY [B, 1, N]         starts from ATOMS [B, N, 1]
Universals / generalities           Particulars / specifically
                                    characterized entities
```

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

The philosophical mapping is compatible with, but not reducible to, the
engineering comparison with LLMs. LLMs motivate the problem of opaque fluent
prediction. Formal Concept Analysis gives a modern mathematical reading of the
analytic/synthetic and extension/intension split: concepts are supported by an
extent and constrained by an intent. DisCoCat gives the corresponding account
for language: grammatical form composes distributional meanings into a sentence
meaning before truth or valid cognition can be assessed.

## Analytic / Synthetic (Kant)

For Kant, **synthesis** is "the act of putting different representations
together, and grasping their manifoldness in one cognition" (A77/B103);
**analysis** decomposes a given whole into its distinguishable conditions.
Synthesis is bottom-up — a manifold of intuition is combined into an object;
analysis is top-down — a given unity is divided into the concepts it falls
under.

The model realizes the pair literally:

- The **perceptual branch synthesizes**: the input arrives as a manifold of
  atoms (`[B, N, 1]`), and the bottom-up Sigma (union) fold — with the
  chunking front ends (radix/BPE/byte) — combines atoms into recurring
  surface forms held in the percept store.
- The **symbolic branch analyzes**: the same input arrives as one undivided
  unity (`[B, 1, N]`), and the top-down Pi (intersection) fold divides it —
  lexer cuts, part-whole divisions, coarse large-scale characterizations
  (means over large regions).
- **Reconstruction is their joint employment.** Kant's dictum that thoughts
  without content are empty and intuitions without concepts are blind maps
  directly: symbolic generalities without perceptual particulars cannot
  spell out a surface; perceptual particulars without the symbolic scaffold
  carry structure but no meaning. `InputSpace.reverse(subspace)` recombines
  both branches: the conceptual reconstruction rides the incoming SubSpace
  (`_concepts_recon`), and the private helper `_paint_reconstruction`,
  called inside `reverse`, paints it together with the atomic percept
  branch.

## Epistemic Levels (Ramsey)

The codebase's term **"Ramsified"** (e.g. `RamsifiedModel.xml`; spaces
replicated "Ramsified across conceptual orders") descends from the **Ramsey
sentence**: replace a theory's theoretical terms with existentially bound
variables, so the theory's content lives in the *roles its structure
realizes* rather than in privileged names. In the model, the per-order
conceptual/symbolic spaces are exactly such role-realizations — what a space-role
*is* is exhausted by its position in the recurrent loop, not by an intrinsic
label; the same structural theory is instantiated at each conceptual order.

> Scholarship hedge (recorded deliberately): this section states the
> project's working usage. Verify the mapping against Ramsey's "Theories"
> (1929) before presenting it as Ramsey exegesis; the "epistemic levels"
> phrasing in particular is the project's own.

## Buddhist Epistemology

Input reconstruction combines **specifically characterized perceptual
particulars** with **generally characterized symbolic divisions**, under the
corrected orientation:

- **`rang-mtshan`** — specifically characterized particulars: the PS side,
  bottom-up, grounded in the eidetic percept store over exact atoms. An
  exact this-one surface form, recognized as a recurring particular within
  the store's snap distance.
- **`spyi-mtshan`** — generally characterized entities: the SS side,
  top-down. Divisions of the presented unity, characterized at large scale.
- **`don-spyi`** (meaning-generality) and **`sgra-spyi`** (term/sound-
  generality) — descriptor **roles** inside the one SS generality codebook,
  not separate codebooks; `.active` selects rows, `.where`/`.when` place
  them over perceptual supports.

The perceptual branch synthesizes bottom-up — union of atoms into recurring
surface forms, content-represented in the percept store within a snap
distance — so exact reconstruction stays grounded. The symbolic branch
analyzes top-down — intersection/division of the presented unity into parts
characterized at large scale — so meaning- and term-generalities never
masquerade as particulars.

The remainder of this document is the original pramana-theory mapping of
the truth ontology.

## Direct Perception and the Conceptual Overlay (Gelug)

In Gelug terms, the **first moment of encounter** with an object is a
direct, non-conceptual cognition: the object appears before it has
been taken up by naming, recognition, memory, preference, or
conceptual elaboration. **Subsequent moments** are often mediated by
conceptual construction, habit, and afflictive projection, so the mind
no longer meets the object simply as it appears, but through an
overlay of interpretation.

The architecture realizes this distinction structurally:

- **The first moment is the parallel prelude.** Percepts cross the
  corpus callosum **nameless** — the interface law (spec §3) factors
  a percept into content (row selection) and evidence (magnitude),
  with no naming side-channel — and the σ/π folds carve the scene:
  the analysis, in its carving of the domain into parts, IS the
  parallel act. The prelude pumps **up to the model's
  `subsymbolicOrder`** (the per-order derivation ladder), seeding both
  codebook towers with a scene description that is pre-conceptual in
  exactly the Gelug sense: no table lookup, no naming, no story. The
  σ and π layers are the real subsymbolic shapers of meaningfulness
  within conceptual space — the words in the codebooks are produced
  and conditioned by them — so the parallel scene description primes
  semantics at that moment and shapes it to a minor degree (the small
  EMA step; the word-learning guarantee).
- **The subsequent moments are the serial STM process.** Naming (the
  binding-table deref), recognition (the codebook snap), memory (the
  workspace stacks), preference and afflictive projection (intent
  priming, heat), conceptual elaboration (shift/reduce composition):
  each serial pump meets the object through the gist-primed,
  intent-weighted overlay, no longer as it first appeared. The serial
  bit happens **in STM**.
- **Collapse and back-action.** Where serial collapse is mandated,
  serial processing selects a **story** — the soft superposition of
  readings collapses, one commit per pump — and the wavefunction of
  reality is simultaneously perturbed: selection is not passive
  observation; the commits write (workspace, bindings, codebook EMA),
  so the act of meaning-making back-acts on the state that produced
  the appearance.
- **The two truths (satya-dvaya).** A completed sentence's meaning is
  either an **absolute truth** — an *idea*: a region-shaped extent,
  evaluable by the luminosity/coverage criterion, rooted at the
  grammar's absolute-truth start state (`exist_O1`, the EXISTS
  operator's output: `<start name="absolute_truth">exist_O1</start>`
  in `data/complete.grammar`; the legacy `ABS_T` token is retired) —
  or a **relative truth** — a *relation between ideas* (causal
  implication, $NP_1$ at $t_1 \to VP \to NP_2$ at $t_2$, is the worked
  example), verified relationally or by simulation through the serial
  reasoning loop, never by coverage. Only absolute truths feed the preattentive
  filter; relative truths are the conceptual overlay's own products
  and are evaluated within it.

**Expectation belongs to the overlay and cannot reach the first moment**
(2026-09-20). The expected next idea is held as a sign-reversed *negative
image* and added to the sealed idea, so what is conceived is what was not
predicted. Perceptual space carries no negation, so the first moment is met
in full whatever is expected; attention, which excludes without predicting,
is the faculty that can reach it. See
[Expectation as a Negative Image; Attention as Exclusion](#expectation-as-a-negative-image-attention-as-exclusion-2026-09-20).

At the corpus callosum, objects are analysed and synthesized — by
sending them back to PerceptualSpace (the shared base of the two
perceptual views): wholes get split and parts get chunked. In symbolic
"mode", the objects that get sent back are **symbols**. The semiotic
vocabulary: there are **objects** and **references**, and a reference
is either a **sign** or a **symbol**. A *sign* is a quantized version
of the referent — it stays in the referent's space, snapped to a
codebook row (the direct-perception side of the ledger). A *symbol* is
an unrelated version of the referent, of much lower dimensionality —
an arbitrary stand-in, related to its referent only through the
binding table (the conceptual-overlay side; cf. *sgra-spyi*, the term
generality, versus the object it evokes).

Protocol summary: any sentence, when processed, has
``subsymbolicOrder`` parallel steps to seed the codebook towers, and
then meaning-making happens over the complete sentence, producing an
absolute truth (an idea) or a relative truth (a relation between
ideas).

## Purpose

WikiOracle's truth ontology closely parallels Buddhist **pramana theory**,
particularly as developed by Dignaga, Dharmakirti, and --- for the tetralemma
--- Nagarjuna. Pramana theory asks:

> How does a *valid cognizer* obtain reliable knowledge?

This document maps WikiOracle's ontology to the **sources of valid
cognition** in Buddhist logic and shows how the **tetralemma** is represented
using **4-valued (quaternary) truth logic** and **non-affirming negation**.

## Valid Cognition in Dharmakirti

Two primary pramanas:

| Source | Sanskrit | Meaning |
| --- | --- | --- |
| Direct perception | *pratyaksa* | immediate, non-conceptual awareness of unique particulars (*svalaksana*) |
| Inference | *anumana* | conceptual reasoning operating on universals (*samanyalaksana*) |

Dharmakirti identifies four subtypes of pratyaksa: sensory (*indriya*),
mental (*manasa*), self-awareness (*svasamvedana*), and yogic (*yogijnana*).
Testimony (*sabda*) reduces to inference from speaker reliability (*apta*).

WikiOracle mapping:
- **pratyaksa** $\to$ Feeling (direct, pre-conceptual, *svasamvedana*)
- **anumana** $\to$ Fact and Operator (conceptual, propositional)

Testimony is inference from trust --- exactly how WikiOracle treats Indirect
Truth.

## Mapping to WikiOracle Truth Objects

WikiOracle expresses the same epistemic structure through six truth types,
organized into Direct Truth and Indirect Truth.

### Direct Truth

| Type | Epistemic Role | Buddhist | Sanskrit |
| --- | --- | --- | --- |
| **Feeling** | immediate hedonic tone; $\pm$1 = *vedana* | direct perception / self-awareness | *pratyaksa* / *svasamvedana* |
| **Fact** | conceptual proposition with truth value in [-1, +1] | inference / conceptual cognition | *anumana* / *kalpana* |
| **Operator** | logical transformation (and/or/not/non) | logical pervasion / formal reasoning | *vyapti* / *prayoga* |

Feelings are *pratyaksa* because they are pre-conceptual, non-linguistic,
and immediate --- the raw experiential signal before elaboration. Dharmakirti
is explicit: pratyaksa apprehends *svalaksana* and is non-conceptual
(*nirvikalpaka*). The moment something is formulated as a proposition with a
truth value, it is conceptual and falls under *anumana*.

### Indirect Truth

| Type | Epistemic Role | Buddhist | Sanskrit |
| --- | --- | --- | --- |
| **Reference** | citation grounding a claim | scripture / textual source | *agama* |
| **Provider** | another cognizer supplying claims | trustworthy person / valid cognizer | *apta* / *pramana-purusa* |
| **Authority** | reference to another body of conversations | trustworthy testimony | *apta-vacana* |

All three reduce to inference from source reliability.

### The Epistemic Pipeline

```
feeling (direct perception) -> fact (conceptual judgment) -> operator -> new fact
```

Authorities influence which providers are trusted; logical validity is
determined by operators and evidence.

## Frame-Relative Truth

WikiOracle evaluates facts relative to **epistemic frames** defined by
authorities and priors:

```
fact = (proposition, frame, truth_value)
```

Different frames may legitimately assign different truth values:

| Frame | Earth age |
| --- | --- |
| Biblical literalist | ~6000 years |
| Geological science | ~4.5 billion years |

Both recorded simultaneously without contradiction; truth is **frame-indexed**.

## Quaternary Truth and the Catuskoti

WikiOracle's truth analysis encodes Nagarjuna's tetralemma (*catuskoti*),
per *Mulamadhyamakakarika*. Writing affirmation and negation as independent
poles $[a_P, a_N]$:

| State | Sanskrit | $[a_P, a_N]$ | WikiOracle reading |
| --- | --- | --- | --- |
| True | *asti* | `[1, 0]` | affirmed |
| False | *nasti* | `[0, 1]` | negated |
| Both (inconsistency) | *ubhaya* | `[1, 1]` | affirmed and negated (across frames/sources) |
| Neither (unknown) | *anubhaya* | `[0, 0]` | neither affirmed nor negated |

Operations are 4-valued, respecting De Morgan under pole-swap negation
$\neg[a_P, a_N] = [a_N, a_P]$:

- Conjunction: $[\min(a_P, b_P), \max(a_N, b_N)]$ (truth-min, falsity-max)
- Disjunction: $[\max(a_P, b_P), \min(a_N, b_N)]$

> Implementation note (2026-07): the 2-dim bivector activation *substrate*
> described above was retired in 2026-05 --- every inter-component
> activation now carries one signed Degree-of-Truth scalar in $[-1, 1]$
> (read it as $a_P - a_N$; `ActiveEncoding`, `bin/Spaces.py`). The
> four-valued *analysis* survives where it matters, in `TruthLayer`
> (`tetralemma_balance_penalty`, `consistency`, `suggest_clarifications`
> are all live, `bin/Layers.py`). On the scalar carrier the grammar
> operators reduce accordingly: conjunction/disjunction are plain
> elementwise `torch.min` / `torch.max` over the whole activation vector
> (`ConjunctionLayer` / `DisjunctionLayer`, `bin/Language.py`, via
> `Ops.intersection` / `Ops.union` with `monotonic=True`), and negation
> remains the flip (sign flip on the scalar; the pole-swap form survives
> as the demuxed-bivector kernel, `Ops._negation_kernel(monotonic=True)`).

The *Both* corner is no longer conflated with *indeterminate* nor exiled to
feeling. Inconsistency is a first-class analysis state which the loss
can detect and suppress (see `TruthLayer.tetralemma_balance_penalty` and
`TruthLayer.consistency` / `suggest_clarifications`).

Feelings (*vedana*) continue to occupy the *Neither* position semantically
--- pre-conceptual, outside propositional truth --- but are distinguishable
from *unknown* concepts by carrying no codebook commitment.

When multiple frames are considered simultaneously, *Both* emerges naturally
as frame disagreement.

## Negation and Logical Operators

Buddhist logic distinguishes two types of negation:

| Type | Sanskrit | Meaning |
| --- | --- | --- |
| Affirming negation | *paryudasa* | negation implying an alternative predicate |
| Non-affirming negation | *prasajya-pratisedha* | pure removal of a predicate |

WikiOracle's operators map to Dharmakirti's theory of inference:

| Operator | Buddhist | Sanskrit |
| --- | --- | --- |
| `not` (`NotLayer`) | affirming negation / implies the opposite | *paryudasa* |
| `non` (`NonLayer`) | non-affirming negation / pure removal | *prasajya-pratisedha* |
| `conjunction` (`ConjunctionLayer`) | positive concomitance / co-presence | *anvaya* |
| `disjunction` (`DisjunctionLayer`) | negative concomitance / co-absence | *vyatireka* |

> Implementation note (2026-07): these are **grammar operators** --- rule
> names in `data/complete.grammar` dispatched to the layer classes in
> `bin/Language.py` --- not markup. The corpus format is plain text; no
> `<not>` / `<non>` / `<and>` / `<or>` XML tags are parsed anywhere.

All operators are instances of **logical pervasion** (*vyapti*) --- the
necessary connection between reason and conclusion grounding valid inference.

`non(a)` interprets as: *the conceptual commitment to a is removed*. This
produces **epistemic openness** rather than contradiction. Valid cognitions
deepen stable conceptual attractors; false cognitions weaken them, producing
a truth-weighted energy landscape similar to a Hopfield memory.

The two negations are also two faculties (2026-09-20). `not` needs a signed
carrier and is how **expectation** enters: the predicted idea, sign-reversed,
is added to what was composed. `non` needs no sign and is how **attention**
works: the non-object is removed and nothing is put in its place, so
attention needs no prediction. See
[Expectation as a Negative Image; Attention as Exclusion](#expectation-as-a-negative-image-attention-as-exclusion-2026-09-20).

## Truth Lattice

| State | Interpretation | Examples |
| --- | --- | --- |
| True (*asti*) | affirmed in frame | `<truth text="..." trust="+0.9"/>` (an XML `<truthSet>` row) |
| False (*nasti*) | rejected in frame | `<truth text="..." trust="-0.9"/>` |
| Both (*ubhaya*) | disagreement across frames | frame-indexed contradiction (what `TruthLayer.consistency` flags) |
| Neither (*anubhaya*) | outside truth lattice | feeling --- excluded from training by design intent (see the implementation note below) |

## Feelings, Vedana, and the "Neither" Position

Feelings occupy the *neither* position. They are not truth-evaluable
propositions --- they are **direct perception** (*pratyaksa*), specifically
**self-awareness** (*svasamvedana*): the reflexive presence of experiential
content.

The $\pm$1 values correspond to **vedana** (hedonic tone):

- **+1**: *sukha-vedana* --- pleasant
- **-1**: *duhkha-vedana* --- unpleasant

Vedana arises from contact (*sparsa*) --- the meeting of sense organ, sense
object, and consciousness. Pre-conceptual and non-linguistic.

In WikiOracle (design intent):
- Feelings are to be **excluded from model training**.
- Feelings are to be **excluded from TruthSets** --- they carry no epistemic
  weight.
- Canonical examples: poetry, greetings, hedged claims, subjective expressions.

This preserves the tetralemma without logical explosion.

> Implementation note (2026-07): the exclusion above is stated design
> intent, not implemented machinery. No `<feeling>` tag exists, and there
> is no feeling-specific training or TruthSet filter; the `<truthSet>` /
> trust machinery (provisioning in `bin/Models.py`, `TernaryTruthStore` in
> `bin/Layers.py`) is real but generic over truth texts. The hedonic
> *reading* of the stored trust sign (next subsection) is the current
> stand-in for a feeling channel.

### Trust-sign as hedonic tone --- luminosity as joy

The same $\pm$1 scalar the Truth Lattice reads epistemically (*asti* / *nasti*)
can be read **hedonically** as vedana: the sign of a stored truth's trust value
in long-term memory is its felt tone --- $+1$ *sukha* (welcome), $-1$ *duhkha*
(unwelcome). One number, two readings; which is in force is a matter of
attention, not storage.

Under the hedonic reading, **luminosity** --- the catuskoti coverage of a region
of the lattice (how fully its four corners are determined) --- reads as **joy**
rather than truth: illumination weighted by trust-sign measures how *welcome* the
determined content is, not merely how *settled* it is. The epistemic and
contemplative luminosities coincide in magnitude and differ only in the
sign-weighting.

## Expectation as a Negative Image; Attention as Exclusion (2026-09-20)

Four statements of Alec's (2026-09-20) fix how expectation and attention
enter the mind. The mechanism is specified in the
[accessible-mind spec §2.6](specs/2026-09-20-accessible-mind-subsystems.md#26-expectation);
this section records what it means and what it answers to.

> Prediction is the negation (the affirming negation, or sign-reversal) over
> the conceptual activation, which is left in conceptual space and added to
> the incoming conceptual activation. For a perfect prediction we then
> conceive of nothing.

> The object appears to the mind as the negation of the non-object. So we
> apply expectation to everything but the object of observation, which allows
> it to (for example) be recognized more quickly.

> I'm not sure negation can be applied to zero-order concepts, which are just
> perceptual assemblages, since perceptual space does not carry negation.

> Opposite of non-pot is a non-affirming negation. So I guess attention does
> not need full prediction in order to focus (which makes sense).

### The rule

With `o` the idea composed from a sentence, `ê` the idea that was expected,
`κ` how strongly each role was expected to be filled, `g` a gain and `m` the
attention on each role, what is **conceived** is

```text
c = o − g · (1 − m) ⊙ κ ⊙ ê
```

A sentence that was perfectly predicted conceives nothing. One that departs
from its prediction conceives the departure. A role that was expected and did
not come conceives `−ê`: an absence. And `o = c + g(1 − m)κê` exactly while
the estimate is kept, so nothing of the input is lost: the sentence is met
from the expected point instead of from zero. It is a change of origin, not a
distortion.

This is the opposite sign to the natural first idea, that expecting something
should *raise* it in the mind. Raising the expected makes comprehension lean
toward its own prediction: the input is no longer accepted as it is, and a
predictor trained on what was understood is then trained on its own echo.
Subtracting leaves the understanding alone. "If we do not predict, we process
exactly what is there" (Alec); under this rule that is equally true when we
do predict, and the difference expectation makes is confined to what is
*new*.

### Perception carries no negation

The third statement is settled by the percept geometry
([Spaces](Spaces.md#percept-geometry-positive-unit-hypercube)): a percept is a
presence in `[0, 1]`, "one-sided; its opposite is the complement `1-x`, not
the signed negation `-x`". The cube has a complement but no additive inverse,
and cancelling by addition needs one. Zeroth-order concepts are assemblages
of percepts; sign enters conceptual activation only at the first rung above
them. So the negative image cannot be formed at order 0 at all. It is an
*idea* — and the negation of a zeroth-order point is not another such point
but a region, everything the exclusion leaves, which makes it higher-order —
and it is added where signed serial form exists: to the sealed idea.

Read against [the Gelug account above](#direct-perception-and-the-conceptual-overlay-gelug),
this is exact. The **first moment** — the parallel prelude, percepts crossing
nameless — is out of expectation's reach by construction: sensation is never
subtracted from. Expectation belongs wholly to the **conceptual overlay**.
The purity of accepting the input is the purity of the first moment, and it
is guaranteed by the geometry, not by restraint.

It also forces the two negations of
[the operators table](#negation-and-logical-operators) apart into two
faculties, by the carrier each needs:

| | unsigned carrier | signed carrier |
| --- | --- | --- |
| Where | presence `[0,1]`; the priority surface; the reading scope | conceptual activation `[-1,1]` above order 0; ideas |
| Negation | `non` (*prasajya-pratisedha*): withdrawal, "not taken up" | `not` (*paryudasa*): sign reversal, "the opposite is affirmed" |
| Faculty | **attention**, as selection | **expectation**, as the negative image |
| Failing to register | the unattended is not conceived: no trace, no absence (inattentional blindness) | the expected is cancelled: nothing new is conceived, and the estimate stands for it (habituation) |
| Omission | nothing: one does not miss what one was not looking for | the uncancelled image: the absence is conceived |

One can habituate only to what one can conceptualise: content that reaches no
idea has no negative image and is met in full every time.

### Eliminative and collective engagement

Gelug epistemology divides awarenesses by how they engage their objects. A
direct perceiver is a **collective engager** (*sgrub 'jug*): it engages its
object by the power of the thing, and everything that is of one substance
with the object appears to it together. A conceptual consciousness is an
**eliminative engager** (*sel 'jug*): it gets at its object by eliminating
what is not that object, and so never takes in all of the object's features.

The architecture has both, in the same places. Order-0 presence is engaged
collectively: all of it, nothing subtracted. Conception is eliminative twice
over: a code is already a projection, the nearest row with every other row
eliminated; and what is conceived of a sentence is what the negative image
leaves.

### Apoha: attention excludes, and needs no prediction

*Apoha* (*gzhan sel*, exclusion of the other) is Dignaga's and Dharmakirti's
account of how a concept can apply to many particulars without a real
universal: "cow" is whatever is not non-cow. The second and fourth statements
above apply it to attention, and the kind of negation decides what attention
needs. The exclusion by which an object appears — opposite from non-pot — is
**non-affirming**: it removes the non-object and puts nothing in its place.
It therefore needs no estimate of what the non-object *is*, only which thing
is the object.

- **Attention focuses without prediction.** One can attend in a wholly novel
  scene. In the architecture attention is the unsigned machinery that was
  already there — the reading scope over the percepts, and the intent channel
  of the priority surface at the competitive readout — and neither reads the
  predictor. What they exclude is *not taken up*: not conceived, no trace, no
  absence.
- **Where there is also a prediction, it is applied to everything but the
  object.** The context is cancelled to the extent that it was expected, and
  the object of observation is conceived in full, expected or not. This is
  the one place the two faculties meet, and with no prediction it costs
  attention nothing.
- **Facilitation is reduced competition.** Serial thinking takes one thing at
  a time. With the non-object excluded, and cancelled where it was expected,
  the object is the only content with magnitude and wins without a contest.
  Expectation silences; attention is what makes the expected thing *easier*
  to see. They are different faculties, and the experimental literature that
  treats them as one reports a paradox (below). How soft the exclusion is
  decides whether an unexpected, unattended thing captures the mind or goes
  unseen.
- **Exclusion and selection are one act at a competitive readout.** A readout
  that depends only on relative score is unchanged by a shift common to all
  candidates, so lowering every non-object by `δ` selects exactly what
  raising the object by `δ` selects. A concept can therefore work by
  exclusion alone, with no positive universal standing behind it — the
  nominalist point of *apoha*, as arithmetic.
- **A question is an expectation with a hole in it.** Its bound roles are the
  context and its open role is the object of observation. Laid over a
  remembered frame, the bound roles cancel and what remains conceived is the
  answer. *Answering is subtracting the question.*

### Absence is inferred, never perceived

Dharmakirti counts **non-observation** (*anupalabdhi*) among the three kinds
of valid reason, and restricts it: only the non-observation of what is
*suitable to appear* (*drsyanupalabdhi*) establishes an absence; the
non-observation of what would not have appeared anyway establishes nothing
beyond the lack of a warrant to affirm.

The negative image gives the same account. Nothing in the input says that the
dog did not bark. The sentence is silent, and the silence is informative only
against the expectation of a bark: the uncancelled image `−ê` *is* the
absence, conceived. So:

- an absence is never composed from the input, only **concluded** in thought,
  as an inference;
- it is licensed only where the thing **would have been observed**: the role
  was strongly expected (`κ`) and within the scope of what was read;
- a conceptual activation of zero is **unknown, not absent** (presence `½`).
  "Confirmed" and "never in play" both conceive nothing, and only the
  retained estimate tells them apart.

Negative facts therefore exist only against expectations, which is also the
psycholinguistic finding: a denial is natural only where the affirmative was
plausible (Wason 1965).

### Identity is an expectation

> Identity has to be carried by expectation or prediction because (at least
> from a philosophical point of view) identity does not exist. (Alec,
> 2026-09-21)

"The lion runs. The lion is tired." Nothing in the second sentence says that
its lion is the first one, and on the view this document takes
([Implicit Existence and Svabhava](#implicit-existence-and-svabhava)) there is
no further fact that would: what persists is a continuum of moments, and "the
same lion" is imputed upon it. Recognition, *this is that*, was for the
Buddhist epistemologists a conceptual cognition built on memory, not a
perception; Hume called the identity we ascribe to things and persons a
fiction of the imagination, produced by resemblance and causation among
successive perceptions.

So the architecture stores no identity. A later sentence's word may be tied by
reference to an earlier row
([two truths §3.5](specs/2026-09-16-two-truths-ideas-and-relations.md#35-object-permanence-a-word-may-translate-to-an-earlier-occurrence-decided-2026-09-21)),
and that reference records only that the mind *took* the two as one. What
carries the individual across the gap is the predictor: each anchor in its
situation is a standing prediction that something continues. Empty the
situation and the same words name only their types.

This is also how identity is found in people. Object permanence in infants is
*measured* as surprise when a hidden object fails to persist (Baillargeon
1987): it is an expectation before it is anything else. An object file keeps
an object's identity through changes of its features by continuity, not by
matching them (Kahneman, Treisman & Gibbs 1992). And the expectation is
strong enough to override the evidence: most people fail to notice that the
stranger they are talking to has been replaced by another person during a
brief interruption (Simons & Levin 1998).

Since the input cannot settle identity, it cannot correct it either. Its
consequences can. What is *said* of an individual is composed from the input
alone, so a prediction that leaned on a wrong identity fails on content, and
that surprise is what revises the imputation. Identity is the one place where
expectation is allowed into composition, and it is kept honest from
downstream.

### Beginner's mind

At `g = 0` nothing is subtracted. Every sentence is conceived in full, as if
for the first time, and no absence can be conceived, because there is nothing
for the world to fall short of. That is a formal reading of **beginner's
mind** (*shoshin*). The world-model goes on learning — estimating continues;
only its application to what is conceived is suspended — and every document
begins this way, since there is no prior context to expect from.

Two cautions keep the reading honest. Beginner's mind in this sense is **not
non-conceptual**: a code is still a projection, so conception remains an
eliminative engager; what is suspended is the *prior*, not the concept. And
it is **not free**: with nothing cancelled nothing stands out, so load rises,
and the absences, which are real information, are given up. It is a setting
to be able to reach, not a default to prefer.

**Learning has its own beginner's mind** (Alec, 2026-09-21). The negative
image keeps *comprehension* honest at any gain, because it is added after
composition. *Learning* is a separate matter. Where a word's meaning settles
depends on its context, and that context can be what occurred or, in part,
what was expected. In people it is both: anomalous playing cards are seen,
and reported, as the normal cards that were expected (Bruner & Postman
1949), and evidence is assimilated to the belief it was meant to test (Lord,
Ross & Lepper 1979). With a share `w` of the context taken from the
estimate, the step toward what occurred shrinks to `(1 − w)` and the rest
goes toward what was already believed; at `w = 1` nothing new can be
learned. "We get perfect learning only when we drop our preconceived ideas
about the situation." `w` is a model variable, not a prohibition: zero is
the unbiased learner, above zero the human one. And the model has two knobs
where a person has one — it can conceive through its expectations and still
learn without them.

The contemplative literature's two families of attention regulation (Lutz et
al. 2008) fall on the rule's two parameters: **focused attention** holds one
object of observation and excludes the rest (`m` on one object); **open
monitoring** holds no object and lets what arises arise (`m = 0`, with the
gain lowered).

### Desire, lack and feeling-tone

A desire is a standing prediction. Held as a negative image it makes the
*absence* of the desired thing perceptible: what is wanted and not there is
conceived as `−ê`, a **lack**. Valence becomes reference-dependent:
disappointment is an uncancelled image, relief a cancelled threat, and the
same outcome is gain or loss according to what was expected.

This separates lack from
[feeling-tone](#feelings-vedana-and-the-neither-position). *Vedana* arises
from contact and is pre-conceptual: it lives where there is no negation, so
there is pleasant, unpleasant and neutral sensation but **no lack in
sensation**. Lack needs the signed carrier. It is a conceived absence, a
product of the overlay — "preference and afflictive projection" in the Gelug
list above — and at `g = 0` it cannot be formed. On this reading craving
conditions suffering through a specific mechanism: the standing prediction is
what makes the world fall short.

The same sign creates a degenerate optimum, the "dark room": a mind rewarded
for conceiving little would seek what it can predict, or stop looking. The
rule in the spec is that residual credit may train how an estimate is
*formed* and never what is *observed or conceived* — not the gain, not the
object of observation, not reading attention.

### Correspondences, and where they strain

The full table, with sources, is in the
[spec's §7](specs/2026-09-20-accessible-mind-subsystems.md#7-correspondence-with-the-psychological-literature).
In brief:

- **The negative image is a known circuit.** Cerebellum-like structures in
  weakly electric fish learn, by anti-Hebbian plasticity, a negative image of
  the sensory consequences of the animal's own discharge; it is added to the
  input and cancels it, leaving what the world added (Bell 1981; Bell, Han &
  Sawtell 2008). It is the reafference principle (von Holst & Mittelstaedt
  1950), and it gives the estimate its two signs: positive in production,
  where the predicted idea is what is said, and negative in comprehension,
  where it is what need not be conceived. Generation is the dual of
  comprehension in sign as well as in direction.
- **Habituation, and its release.** Sokolov's (1963) neuronal model: a
  repeated stimulus builds a model, the response fades, and any mismatch —
  including an omission — brings it back.
- **Omission responses.** The omission of an expected sound evokes activity
  with the expected sound's signature (SanMiguel et al. 2013); dopamine
  neurons dip at the moment of an omitted reward (Schultz, Dayan & Montague
  1997).
- **Attention reverses silencing.** Prediction silences unattended signals
  and attention reverses the effect (Kok, Rahnev et al. 2012); expectation and
  attention are distinct and routinely confounded (Summerfield & Egner 2009).
  The rule reproduces that 2 × 2, the better identification of an object in
  its expected scene (Palmer 1975), and inattentional blindness (Simons &
  Chabris 1999).
- **Pointer plus tag.** A scripted event is remembered as a pointer to the
  script plus tags for what was atypical; typical actions are falsely
  recognised and atypical ones kept (Graesser, Gordon & Sawyer 1979). The
  store's linked estimate and observation rows are that layout, and a
  perfectly predicted sentence is the first thing forgotten.

Where it strains: expectation also *sharpens* the representation of what was
expected (Kok, Jehee & de Lange 2012), and this design has no sharpening;
cortical gain control is mostly divisive, not subtractive (Carandini & Heeger
2012); and the classic report that Zen practitioners do not habituate
(Kasamatsu & Hirai 1966) did not replicate (Becker & Shapiro 1981), though
habituation of startle is reduced with intensive practice (Antonova, Chadwick
& Kumari 2015).

> Scholarship hedge (recorded deliberately): this section states the
> project's working usage. The division of awarenesses into eliminative and
> collective engagers follows the Gelug presentation (Lati Rinbochay & Napper
> 1980; Klein 1986; Dreyfus 1997). Gelug textbooks divide exclusions three
> ways — objective, mental, and non-affirming-negative exclusions — and the
> secondary sources consulted (Berzin, below) class the *mental* exclusions,
> the categories through which conception works, as affirming negations. The
> project takes the exclusion by which attention isolates its object,
> opposite from non-pot, as non-affirming (Alec); which textbook category
> that answers to should be checked against Klein and Dreyfus before it is
> presented as exegesis. The readings of *anupalabdhi* (Kellner 2003), of
> beginner's mind, of craving and of recognition as conceptual are the
> project's own mappings onto the mechanism.

**References.** Antonova, Chadwick & Kumari (2015), *More meditation, less
habituation? The effect of mindfulness practice on the acoustic startle
reflex*, PLoS ONE 10(5). Becker & Shapiro (1981), *Physiological responses to
clicks during Zen, Yoga, and TM meditation*, Psychophysiology 18. Bell (1981),
*An efference copy which is modified by reafferent input*, Science 214. Bell,
Han & Sawtell (2008), *Cerebellum-like structures and their implications for
cerebellar function*, Annual Review of Neuroscience 31. Baillargeon (1987),
*Object permanence in 3½- and 4½-month-old infants*, Developmental Psychology
23. Bruner & Postman
(1949), *On the perception of incongruity: a paradigm*, Journal of
Personality 18. Berzin, *Negation
phenomena: implicative and non-implicative* and *Special features of the
Gelug tradition*, Study Buddhism (studybuddhism.com). Carandini & Heeger
(2012), *Normalization as a canonical neural computation*, Nature Reviews
Neuroscience 13. Dreyfus (1997), *Recognizing Reality: Dharmakirti's
Philosophy and Its Tibetan Interpretations*, SUNY. Hume (1739), *A Treatise
of Human Nature*, I.iv.6, "Of personal identity". Graesser, Gordon & Sawyer
(1979), *Recognition memory for typical and atypical actions in scripted
activities: tests of a script pointer + tag hypothesis*, Journal of Verbal
Learning and Verbal Behavior 18. Kahneman, Treisman & Gibbs (1992), *The
reviewing of object files: object-specific integration of information*,
Cognitive Psychology 24. Kasamatsu & Hirai (1966), *An
electroencephalographic study on the Zen meditation (Zazen)*, Folia
Psychiatrica et Neurologica Japonica 20. Kellner (2003), *Integrating
negative knowledge into pramana theory: the development of the
drsyanupalabdhi in Dharmakirti's earlier works*, Journal of Indian Philosophy
31. Klein (1986), *Knowledge and Liberation*, Snow Lion. Kok, Jehee & de
Lange (2012), *Less is more: expectation sharpens representations in the
primary visual cortex*, Neuron 75. Kok, Rahnev, Jehee, Lau & de Lange (2012),
*Attention reverses the effect of prediction in silencing sensory signals*,
Cerebral Cortex 22. Lati Rinbochay & Napper (1980), *Mind in Tibetan
Buddhism*, Snow Lion. Lutz, Slagter, Dunne & Davidson (2008), *Attention
regulation and monitoring in meditation*, Trends in Cognitive Sciences 12(4).
Palmer (1975), *The effects of contextual scenes on the identification of
objects*, Memory & Cognition 3. Lord, Ross & Lepper (1979), *Biased
assimilation and attitude polarization*, Journal of Personality and Social
Psychology 37(11). SanMiguel, Widmann, Bendixen, Trujillo-Barreto
& Schröger (2013), *Hearing silences: human auditory processing relies on
preactivation of sound-specific brain activity patterns*, Journal of
Neuroscience 33. Schultz, Dayan & Montague (1997), *A neural substrate of
prediction and reward*, Science 275. Siderits, Tillemans & Chakrabarti (eds.)
(2011), *Apoha: Buddhist Nominalism and Human Cognition*, Columbia. Simons &
Chabris (1999), *Gorillas in our midst: sustained inattentional blindness for
dynamic events*, Perception 28. Simons & Levin (1998), *Failure to detect
changes to people during a real-world interaction*, Psychonomic Bulletin &
Review 5. Sokolov (1963), *Perception and the
Conditioned Reflex*, Pergamon. Summerfield & Egner (2009), *Expectation (and
attention) in visual cognition*, Trends in Cognitive Sciences 13(9). von Holst
& Mittelstaedt (1950), *Das Reafferenzprinzip*, Naturwissenschaften 37. Wason
(1965), *The contexts of plausible denial*, Journal of Verbal Learning and
Verbal Behavior 4.

## The Four Foundations of Mindfulness

Awareness must range over more than sense input. The four foundations of
mindfulness (*satipatthana*) name what attention can take as its object; each
maps onto one addressable store of
[global attention](Architecture.md#addressable-attention--the-typed-where):

| Foundation | Sanskrit | Attention store |
| --- | --- | --- |
| Body | *kaya* | the input window + the percept codebooks (PART / WHOLE) |
| Feeling-tone | *vedana* | the sign of the LTM trust value ($\pm$1) |
| Mind | *citta* | short-term memory (STM) |
| Mental objects | *dhamma* | the symbol codebook (SYMBOL) |

One mechanism serves all four: pointing the typed `.where` at a codebook is
recall, at the input window is perception. Without addressable symbolic content
there is no fourth foundation --- the typed address space is what makes
*dhammanupassana* (mindfulness of mental objects) possible for the model.

## Implicit Existence and Svabhava

WikiOracle's [grammar](Language.md) allows sentences without an explicit
verb phrase: $S \to NP$. The bare noun phrase "Fire!" carries an implicit
existential predicate. This has a philosophical cost.

Nagarjuna's *Mulamadhyamakakarika* argues that the fundamental error of
conceptual thought is attributing **svabhava** (inherent existence) to
phenomena. "Fire" without an explicit existential predicate appears to
stand on its own --- as if fire possessed permanent, independent self. This
is the reification that **sunyata** (emptiness) challenges.

| Concept | Sanskrit | Meaning |
| --- | --- | --- |
| Inherent existence | *svabhava* | mistaken belief in independent permanent things |
| No-self | *anatta* / *anatman* | nothing has a fixed independent self |
| Dependent origination | *pratityasamutpada* | things arise in dependence on causes |
| Emptiness | *sunyata* | phenomena are empty of inherent existence |

Making existence explicit --- "fire exists" --- restores existence as a
*relation* rather than an *attribute*. The live grammar enforces this at
the root: its absolute-truth start state is `exist_O1`, the output of the
EXISTS operator (`<start name="absolute_truth">exist_O1</start>`,
`data/complete.grammar`), so even the bare "Fire!" completes only by
passing through an explicit existential predicate --- existence is
supplied as a relation, never presumed as an attribute.

> Implementation note (2026-07): an earlier grammar expressed the same
> point as the rule $VP \to \varepsilon$ iff $MP \to \varepsilon$ (absence
> of existential predicate implies absence of modal frame). That rule is
> retired --- the role-collapsed grammar has no VP/MP nonterminals ---
> and the `exist_O1` root above is its successor.

## Shamatha Speech and Single-Pointedness

Dakpo Tashi Namgyel's requirement for single-pointedness models as a
restriction on what counts as one object in speech. A complete DNF specifies
all logical commitments about an object, but logical completeness alone is
not contemplative one-pointedness --- the parts must also remain a single
spatiotemporal field.

WikiOracle's Shamatha Speech target adds a contiguity condition to the DNF
object grammar:

- Every conjunction or disjunction may range over all active percepts.
- Merged parts must have connected `where()` support.
- Merged parts must have continuous or adjacent `when()` support.
- Disconnected supports are scattered aggregates, not one object.

Differs from serial speech. Serial follows a cursor through time. Shamatha
Speech sees the whole current percept field but only permits logical
composition that preserves a single object of attention.

## Psychological grounding (2026-09-16)

The architecture is meant to be human in its mechanism, not only in its
output: a mind we can read, and a model of our own. Each claim above has a
counterpart in the psychology of memory, language and belief. This section
records the counterparts that support the design, and then the places where
the design departs from what is known about people, so that each departure
is a stated choice or a stated gap rather than an accident.
[Architecture.md](Architecture.md#cognitive-grounding-dense-perceptual-vs-sparse-symbolic-2026-07-02)
grounds the perceptual/symbolic split (complementary learning systems,
dual process, systematicity, grounded cognition, basic level); this section
covers memory, the two truths, expectation, testimony and feeling.

### Support

- **The two truths are two memory systems.** An *idea* (absolute truth, one
  fused point with its derivation) is what text comprehension builds as the
  **situation model**, the integrated representation of one state of
  affairs that survives when the text's wording is gone (Zwaan & Radvansky
  1998); it is episodic in Tulving's sense, a particular. A *relation*
  (relative truth, `row R row`) is **semantic memory**: general, structured,
  read by inference rather than by matching (Tulving 1972). Semantic memory
  is organised as a hierarchy over which properties are inherited
  (Collins & Quillian 1969), which is what the part rows between concept
  rows and the taxonomy index derived from them provide. The
  [two-truths spec](specs/2026-09-16-two-truths-ideas-and-relations.md)
  makes the split explicit in LTM.

- **Gist survives, form fades; meaning is stored fused.** Recognition
  memory for a sentence's wording is lost within a few seconds of hearing
  it while memory for its meaning persists (Sachs 1967); fuzzy-trace theory
  separates a durable gist trace from a fragile verbatim trace (Reyna &
  Brainerd 1995). Fusing an absolute sentence to one point and keeping its
  factoring only as a replayable derivation is the same asymmetry:
  understanding rests on the gist, the form is reconstructed.

- **Propositions embed by reference.** Text memory is a network of
  propositions in which an embedded proposition is an argument of the
  higher one, `SAY[he, P]`, and recall follows the propositional hierarchy
  (Kintsch & van Dijk 1978). This is the relative case of the spec: a
  clause that cannot fuse is referenced by row. Long-term working memory
  explains how experts hold far more than a span's worth of material by
  keeping **retrieval structures** in working memory that point into LTM
  (Ericsson & Kintsch 1995); a pushed row reference on the STM is a
  retrieval structure.

- **Attribution and content are separate traces.** People keep what was
  said while losing who said it (source monitoring: Johnson, Hashtroudi &
  Lindsay 1993), and a message from a discredited source gains force over
  time as the source memory decays faster than the content (the sleeper
  effect: Hovland & Weiss 1951; Kumkale & Albarracín 2004). Storing "he
  said P" as an operator relation over P's own row, rather than fusing the
  two, is the representation these effects require. Testimony reducing to
  inference from source reliability (Dharmakirti's *apta*, above) has its
  modern form in **epistemic vigilance** (Sperber et al. 2010): hearers
  calibrate belief to the source, not to the confidence the speaker
  expresses, which is the spec's rule that a sentence never sets its own
  trust.

- **The parallel prelude and the serial pump are preattentive and
  attentive processing.** Feature-integration theory separates a parallel
  preattentive stage that registers features across the whole field from a
  serial attentive stage that binds them into objects one at a time
  (Treisman & Gelade 1980; Neisser 1967). The Gelug "first moment" that
  crosses the corpus callosum nameless, followed by the serial STM process
  that names, recognises and composes, is the same two-stage structure, and
  it says what the parallel stage may and may not do: carve, not name.

- **Working memory is small and serial composition is chunked.** The focus
  of attention holds about four items (Cowan 2001; Miller 1956 for the
  older seven), and skilled memory works by **chunking** into learned units
  (Gobet et al. 2001), against a slave-store and central-executive
  organisation (Baddeley & Hitch 1974). The STM's few live slots, the
  shift/reduce fold that keeps them few, and admitted phrases that snap to
  their own concept row are the model's chunking.

- **Priming is spreading activation.** Semantic priming (Meyer &
  Schvaneveldt 1971) and its spreading-activation account (Collins &
  Loftus 1975) describe activation flowing along associative edges and
  decaying with distance; the energy-dissipating diffusion over concept
  store edges that primes the next read is that mechanism.

- **Expectation is the objective, and comprehension predicts.** Predictive
  processing holds that perception and cognition proceed by predicting
  input and learning from the discrepancy (Rao & Ballard 1999; Friston
  2010; Clark 2013). In language, comprehenders predict upcoming content at
  every level and prediction is graded (Kuperberg & Jaeger 2016), and
  prediction during comprehension runs the production system in reverse
  (Pickering & Garrod 2013). The concept-to-concept expectation of the next
  sentence, scored by discrepancy, and generation as the dual of
  comprehension, are both of a piece with this. Since 2026-09-20 the
  estimate enters comprehension with *negative* sign, after composition,
  so that what is conceived is the discrepancy itself and the input is
  never bent toward the prediction
  ([above](#expectation-as-a-negative-image-attention-as-exclusion-2026-09-20)); in production the same estimate is what is said.

- **Reasoning by simulation and by rule are both real.** Relative truths
  are evaluated "relationally or by simulation": simulation is the
  **mental models** account (Johnson-Laird 1983), in which people reason by
  constructing and inspecting models of situations; relational evaluation
  with modus ponens as a primitive is the **mental logic** account (Braine
  & O'Brien 1998). Keeping both readers is the empirically safe position;
  which one people use varies with content.

- **Feeling precedes and is outside propositional truth.** Affective
  reactions arise before and independently of the cognitive appraisal that
  would justify them ("preferences need no inferences": Zajonc 1980), and
  bodily hedonic markers guide judgement without being judgements (Damasio
  1994). *Vedana* in the *neither* position, and the hedonic reading of the
  trust sign, are consistent with both.

- **Symbols, signs and the word/object generalisation.** The doc's *sign*
  (a quantised version of the referent) and *symbol* (an arbitrary stand-in
  related only through the binding table) are Peirce's icon and symbol.
  Symbolic reference proper is not the word-to-object link but a
  **generalisation over many such links** (Deacon 1997), which is what the
  META node as a generalisation over both the word-concept and the
  object-concept encodes; grounding the symbol side in the perceptual
  towers is the answer to the symbol grounding problem (Harnad 1990).

- **Mindfulness is metacognitive monitoring.** Pointing attention at one's
  own stores (STM, LTM trust sign, the symbol codebook) and reading them
  is the monitoring half of the monitoring/control architecture of
  metacognition (Flavell 1979; Nelson & Narens 1990). The four foundations
  as addressable stores are that architecture with the stores named.

### Discrepancies

Where the model and the human evidence differ. Alec's direction
(2026-09-16): **parameterise optimal versus human operation** so both can
be explored in one model, rather than fixing either. An `<operation>`
profile in `model.xml` (`optimal` | `human`) selects the human-like
behaviour for every item marked **profile** below; items marked **future**
are collected in [FutureWork.md](FutureWork.md); items marked **closed**
are not discrepancies on inspection.

1. **Default belief (profile).** People believe what they comprehend and
   must spend effort to unbelieve it (the Spinozan account: Gilbert 1991).
   Under `optimal` the model is Cartesian: comprehension registers a row
   with trust `0`, and only provenance asserts it, because the human
   default is what liars exploit and a model that can be read must not
   acquire beliefs it was never given. Under `human` comprehension asserts
   at the source's provenance trust, and unbelieving is a later,
   effortful revision.

2. **Verbatim retention (profile, future).** People lose wording within
   seconds (Sachs 1967). Under `optimal` every idea's derivation is kept
   losslessly for reconstruction. Under `human` derivations decay while
   fused points persist, the gist/verbatim asymmetry made dynamic. This
   is one face of the forgetting model in FutureWork.md.

3. **Forgetting and consolidation (profile, future, urgent).** LTM is
   append-only to capacity, with recency as the only decay; people
   consolidate selectively and forget (McClelland, McNaughton & O'Reilly
   1995; Ebbinghaus 1885). The store's capacity is a wall, so a forgetting
   model is needed soon regardless of profile. Direction: forget what is
   not fully understood (low luminosity) and what does not integrate with
   other ideas (few references to or from it). Specified in FutureWork.md.

4. **Document boundaries (future).** The hard reset at a document boundary
   clears transient context; people carry context across texts and
   conversations. A training convenience, not a cognitive claim; carrying
   context across documents is in FutureWork.md.

5. **Level of prediction (closed).** Human comprehension predicts at the
   word level too, with reading times tracking surprisal (Hale 2001; Levy
   2008). The model expects the next sentence's concepts, and the grammar
   unfolds an expected idea into words through the reverse chain, so
   word-level expectation is derived from concept-level expectation rather
   than computed separately. Not a discrepancy; surprisal is a readout of
   the unfolding.

6. **One word, one object (decided, future).** People have polysemy and
   synonymy as the norm, resolved by context. Decision: META concepts
   generalise over **more than two** concepts, several words and several
   objects, and the discrimination among them happens at interpretation
   time, from context, not at binding time. The two-truths spec §3.4 records
   this; the binding table's one-row-per-word law is replaced by the
   n-ary META, and the interpretation-time selection is in FutureWork.md.

7. **Strict inheritance (closed).** Hierarchical semantic memory predicts
   inheritance, but people show typicality effects and exceptions (Rips,
   Shoben & Smith 1973): a penguin is a bird that does not fly. Part rows
   carry their own trust, so inheritance is graded and an exception is a
   lower-trust or negative row, not a contradiction.

8. **Explicit existence (withdrawn).** An earlier draft set psychological
   essentialism (Gelman 2003) against the grammar's forced `exist`. The
   two are not about the same thing: essentialism concerns hidden category
   essences, the `exist` start concerns whether an NP is found and trusted
   in LTM. No discrepancy is claimed.

9. **Inversion is not exact (future).** The reconstruction trace will be
   partially dropped over time (item 2), so inversion becomes a learned
   approximation from a lossy trace rather than an exact replay. That is
   what gives inversion a role in learning: with a complete trace,
   reconstruction teaches nothing. The trace-dropping schedule is in
   FutureWork.md; Architecture.md's note that brains approximate rather
   than invert then applies to the model too.

10. **Silencing without sharpening (open).** In people expectation both
    reduces the response to what was predicted and sharpens its
    representation (Kok, Jehee & de Lange 2012), and cortical gain control
    is largely divisive rather than subtractive (Carandini & Heeger 2012).
    The model has the subtractive silencing and leaves facilitation to
    attention; it has no sharpening. References and the rest of the
    comparison are [above](#expectation-as-a-negative-image-attention-as-exclusion-2026-09-20).

**References.** Baddeley & Hitch (1974), *Working memory*, in Bower (ed.),
The Psychology of Learning and Motivation 8. Braine & O'Brien (1998),
*Mental Logic*, Erlbaum. Clark (2013), *Whatever next? Predictive brains,
situated agents, and the future of cognitive science*, Behavioral and Brain
Sciences 36(3). Collins & Loftus (1975), *A spreading-activation theory of
semantic processing*, Psychological Review 82(6). Collins & Quillian (1969),
*Retrieval time from semantic memory*, Journal of Verbal Learning and Verbal
Behavior 8. Cowan (2001), *The magical number 4 in short-term memory*,
Behavioral and Brain Sciences 24(1). Damasio (1994), *Descartes' Error*,
Putnam. Deacon (1997), *The Symbolic Species*, Norton. Ebbinghaus (1885),
*Über das Gedächtnis*. Ericsson & Kintsch (1995), *Long-term working
memory*, Psychological Review 102(2). Flavell (1979), *Metacognition and
cognitive monitoring*, American Psychologist 34(10). Friston (2010), *The
free-energy principle: a unified brain theory?*, Nature Reviews Neuroscience
11. Gelman (2003), *The Essential Child*, Oxford. Gilbert (1991), *How
mental systems believe*, American Psychologist 46(2). Gobet, Lane, Croker,
Cheng, Jones, Oliver & Pine (2001), *Chunking mechanisms in human
learning*, Trends in Cognitive Sciences 5(6). Hale (2001), *A probabilistic
Earley parser as a psycholinguistic model*, NAACL. Harnad (1990), *The
symbol grounding problem*, Physica D 42. Hovland & Weiss (1951), *The
influence of source credibility on communication effectiveness*, Public
Opinion Quarterly 15. Johnson, Hashtroudi & Lindsay (1993), *Source
monitoring*, Psychological Bulletin 114(1). Johnson-Laird (1983), *Mental
Models*, Harvard. Kintsch & van Dijk (1978), *Toward a model of text
comprehension and production*, Psychological Review 85(5). Kumkale &
Albarracín (2004), *The sleeper effect in persuasion: a meta-analytic
review*, Psychological Bulletin 130(1). Kuperberg & Jaeger (2016), *What do
we mean by prediction in language comprehension?*, Language, Cognition and
Neuroscience 31(1). Levy (2008), *Expectation-based syntactic
comprehension*, Cognition 106. McClelland, McNaughton & O'Reilly (1995), as
cited in Architecture.md. Meyer & Schvaneveldt (1971), *Facilitation in
recognizing pairs of words*, Journal of Experimental Psychology 90. Miller
(1956), as cited in Architecture.md. Neisser (1967), *Cognitive
Psychology*, Appleton-Century-Crofts. Nelson & Narens (1990), *Metamemory:
a theoretical framework and new findings*, The Psychology of Learning and
Motivation 26. Pickering & Garrod (2013), *An integrated theory of language
production and comprehension*, Behavioral and Brain Sciences 36(4). Rao &
Ballard (1999), *Predictive coding in the visual cortex*, Nature
Neuroscience 2. Reyna & Brainerd (1995), *Fuzzy-trace theory: an interim
synthesis*, Learning and Individual Differences 7. Rips, Shoben & Smith
(1973), *Semantic distance and the verification of semantic relations*,
Journal of Verbal Learning and Verbal Behavior 12. Sachs (1967),
*Recognition memory for syntactic and semantic aspects of connected
discourse*, Perception & Psychophysics 2. Simons (1987), *Parts: A Study in
Ontology*, Oxford. Sperber, Clément, Heintz, Mascaro, Mercier, Origgi &
Wilson (2010), *Epistemic vigilance*, Mind & Language 25(4). Treisman &
Gelade (1980), *A feature-integration theory of attention*, Cognitive
Psychology 12. Tulving (1972), *Episodic and semantic memory*, in Tulving &
Donaldson (eds.), Organization of Memory. Zajonc (1980), *Feeling and
thinking: preferences need no inferences*, American Psychologist 35(2).
Zwaan & Radvansky (1998), *Situation models in language comprehension and
memory*, Psychological Bulletin 123(2).

## Summary

| WikiOracle | Category | Buddhist Epistemology | Sanskrit |
| --- | --- | --- | --- |
| Feeling | Direct Truth | direct perception / self-awareness / hedonic tone | *pratyaksa* / *svasamvedana* / *vedana* |
| Fact | Direct Truth | inference / conceptual cognition | *anumana* / *kalpana* |
| Operator | Direct Truth | logical pervasion / formal reasoning | *vyapti* / *prayoga* |
| Reference | Indirect Truth | scripture / textual source | *agama* |
| Provider | Indirect Truth | trustworthy person / valid cognizer | *apta* / *pramana-purusa* |
| Authority | Indirect Truth | trustworthy testimony | *apta-vacana* |

The system models **conventional truth dynamics** consistent with the
logical structure of Dharmakirti and the tetralemma of Nagarjuna. Plural
frames coexist, inference operates within frames, feelings provide the
perceptual ground from which concepts arise, and non-affirming negation
preserves epistemic openness.
