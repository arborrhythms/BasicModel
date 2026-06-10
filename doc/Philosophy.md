# Philosophy

This document maps the model's machinery onto three philosophical accounts
of cognition: Kant's analysis/synthesis pair, Ramsey's treatment of
theoretical structure, and Buddhist epistemology (pramana theory). The
through-line is the corrected analysis/synthesis orientation
(doc/plans/2026-06-08-analysis-synthesis-dual-input.md, rev. 2026-06-09):

```text
SymbolicSpace (SS)                  PerceptualSpace (PS)
top-down ANALYSIS                   bottom-up SYNTHESIS
Pi -- product, intersection         Sigma -- sum, union
starts from UNITY [B, 1, N]         starts from ATOMS [B, N, 1]
Universals / generalities           Particulars / specifically
                                    characterized entities
```

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
  carry structure but no meaning. `InputSpace.reverse(percepts, concepts)`
  recombines both branches.

## Epistemic Levels (Ramsey)

The codebase's term **"Ramsified"** (e.g. `RamsifiedModel.xml`; spaces
replicated "Ramsified across conceptual orders") descends from the **Ramsey
sentence**: replace a theory's theoretical terms with existentially bound
variables, so the theory's content lives in the *roles its structure
realizes* rather than in privileged names. In the model, the per-order
conceptual/symbolic spaces are exactly such role-realizations — what a tier
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

WikiOracle's concept activation is a 2-dim bivector `[aP, aN]` encoding
Nagarjuna's tetralemma (*catuskoti*), per *Mulamadhyamakakarika*:

| State | Sanskrit | `[aP, aN]` | WikiOracle reading |
| --- | --- | --- | --- |
| True | *asti* | `[1, 0]` | affirmed |
| False | *nasti* | `[0, 1]` | negated |
| Both (inconsistency) | *ubhaya* | `[1, 1]` | affirmed and negated (across frames/sources) |
| Neither (unknown) | *anubhaya* | `[0, 0]` | neither affirmed nor negated |

Operations are 4-valued, respecting De Morgan under pole-swap negation
$\neg[aP, aN] = [aN, aP]$:

- Conjunction: `[min(aP, bP), max(aN, bN)]` (truth-min, falsity-max)
- Disjunction: `[max(aP, bP), min(aN, bN)]`

The *Both* corner is no longer conflated with *indeterminate* nor exiled to
`<feeling>`. Inconsistency is a first-class activation state which the loss
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
| `<not>` | affirming negation / implies the opposite | *paryudasa* |
| `<non>` | non-affirming negation / pure removal | *prasajya-pratisedha* |
| `<and>` | positive concomitance / co-presence | *anvaya* |
| `<or>` | negative concomitance / co-absence | *vyatireka* |

All operators are instances of **logical pervasion** (*vyapti*) --- the
necessary connection between reason and conclusion grounding valid inference.

`non(a)` interprets as: *the conceptual commitment to a is removed*. This
produces **epistemic openness** rather than contradiction. Valid cognitions
deepen stable conceptual attractors; false cognitions weaken them, producing
a truth-weighted energy landscape similar to a Hopfield memory.

## Truth Lattice

| State | Interpretation | Examples |
| --- | --- | --- |
| True (*asti*) | affirmed in frame | `<fact trust="+1">` |
| False (*nasti*) | rejected in frame | `<fact trust="-1">` |
| Both (*ubhaya*) | disagreement across frames | `<fact trust="0">` / frame-indexed contradiction |
| Neither (*anubhaya*) | outside truth lattice | `<feeling>` --- excluded from training |

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

In WikiOracle:
- Feelings are **excluded from model training**.
- Feelings are **excluded from TruthSets** --- they carry no epistemic weight.
- Canonical examples: poetry, greetings, hedged claims, subjective expressions.

This preserves the tetralemma without logical explosion.

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
*relation* rather than an *attribute*. The grammar rule $VP \to \varepsilon$
iff $MP \to \varepsilon$ reinforces this: absence of existential predicate
implies absence of modal frame.

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
