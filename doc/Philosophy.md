# Philosophy

This document maps the model's machinery onto three philosophical accounts
of cognition: Kant's analysis/synthesis pair, Ramsey's treatment of
theoretical structure, and Buddhist epistemology (pramana theory). The
through-line is the corrected analysis/synthesis orientation
(doc/old/2026-06-08-analysis-synthesis-dual-input.md, rev. 2026-06-09):

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

The architecture realizes this distinction structurally
(doc/old/MeronomySpec.md; doc/old/GrammarOpsPass.md §6c --- the section
numbering lives only in that archived doc):

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
