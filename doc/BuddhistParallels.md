# Buddhist Parallels

## Purpose

WikiOracle's truth ontology closely parallels the epistemological framework
of Buddhist **pramana theory**, particularly as developed by Dignaga, Dharmakirti,
and -- for the tetralemma -- Nagarjuna.

Pramana theory asks a simple question:

> How does a *valid cognizer* obtain reliable knowledge?

WikiOracle formalizes the same process computationally through a set of
structured truth objects.

This document maps WikiOracle's ontology to the **sources of valid cognition**
recognized in Buddhist logic and explains how the **tetralemma** is
represented using **4-valued (quaternary) truth logic** and
**non-affirming negation**.

## Valid Cognition in Dharmakirti

In Dharmakirti's system there are **two primary pramanas**:

| Source of Cognition | Sanskrit    | Meaning                                                                  |
| ------------------- | ----------- | ------------------------------------------------------------------------ |
| Direct perception   | *pratyaksa* | immediate, non-conceptual awareness of unique particulars (*svalaksana*) |
| Inference           | *anumana*   | conceptual reasoning operating on universals (*samanyalaksana*)          |

Dharmakirti identifies **four subtypes of pratyaksa**: sensory perception (*indriya*), mental perception (*manasa*), self-awareness (*svasamvedana*), and yogic perception (*yogijnana*).

Other knowledge sources -- including testimony (*sabda*) -- are considered **derivative**. Dharmakirti reduces all testimonial knowledge to a form of inference: one *infers* the truth of a claim from the reliability of the speaker (*apta*, "trustworthy person"). This is not a separate pramana but a special case of *anumana*.

A valid cognition produces **true conceptual knowledge** (*prama*).

In WikiOracle, the two pramanas map cleanly:

* **pratyaksa** $\rightarrow$ Feeling (direct, pre-conceptual, *svasamvedana*)
* **anumana** $\rightarrow$ Fact and Operator (conceptual, propositional)

Testimony is not a separate source -- it is inference from trust, which is exactly how WikiOracle treats Indirect Truth.

## Mapping to WikiOracle Truth Objects

WikiOracle expresses the same epistemic structure through six truth types, organized into Direct Truth (what we know directly) and Indirect Truth (what we know only through external sources).

### Direct Truth

| WikiOracle Type | Epistemic Role                                              | Buddhist Equivalent                  | Sanskrit                     |
| --------------- | ----------------------------------------------------------- | ------------------------------------ | ---------------------------- |
| **Feeling**     | immediate hedonic tone; $\pm$1 = *vedana* (pleasant/unpleasant) | direct perception -- self-awareness   | *pratyaksa* / *svasamvedana* |
| **Fact**        | conceptual proposition with truth value in [-1, +1]         | inference -- conceptual cognition     | *anumana* / *kalpana*        |
| **Operator**    | logical transformation (and/or/not/non) deriving new truth  | logical pervasion -- formal reasoning | *vyapti* / *prayoga*         |

Feelings are *pratyaksa* because they are pre-conceptual, non-linguistic, and immediate -- the raw experiential signal before conceptual elaboration. Dharmakirti is explicit: pratyaksa apprehends unique particulars (*svalaksana*) and is non-conceptual (*nirvikalpaka*). The moment something is formulated as a proposition with a truth value, it is conceptual and falls under *anumana*.

### Indirect Truth

| WikiOracle Type | Epistemic Role                                        | Buddhist Equivalent                 | Sanskrit                  |
| --------------- | ----------------------------------------------------- | ----------------------------------- | ------------------------- |
| **Reference**   | citation grounding a claim in a verifiable source     | scripture / textual source          | *agama*                   |
| **Provider**    | another cognizer supplying claims and truth           | trustworthy person / valid cognizer | *apta* / *pramana-purusa* |
| **Authority**   | reference to another body of conversations and truths | trustworthy testimony               | *apta-vacana*             |

Dharmakirti argues that testimonial knowledge involves three components: the *text* itself (*agama* -- Reference), the *person* who produced it (*apta* -- Provider), and the *inferential warrant* for trusting that person's testimony (*apta-vacana* -- Authority). All three reduce to inference from the reliability of the source.

### The Epistemic Pipeline

These correspond to the full epistemic pipeline:

```
feeling (direct perception) $\rightarrow$ fact (conceptual judgment) $\rightarrow$ operator $\rightarrow$ new fact
```

Authorities influence **which providers are trusted**, but logical validity
is determined only by operators and evidence.

## Frame-Relative Truth

WikiOracle evaluates facts relative to **epistemic frames** defined by
authorities and priors.

A fact therefore has the structure:

```
fact = (proposition, frame, truth_value)
```

Different frames may legitimately assign different truth values.

Example:

| Frame               | Earth age          |
| ------------------- | ------------------ |
| Biblical literalist | ~6000 years        |
| Geological science  | ~4.5 billion years |

Both may be recorded simultaneously without contradiction because
truth is **frame-indexed**.

## Quaternary Truth and the Catuskoti

WikiOracle's concept activation is a 2-dim bivector `[aP, aN]` encoding
the four corners of Nagarjuna's tetralemma (*catuskoti*), as articulated
in the *Mulamadhyamakakarika*:

| State                | Sanskrit   | `[aP, aN]` | WikiOracle reading                                              |
| -------------------- | ---------- | ---------- | --------------------------------------------------------------- |
| True                 | *asti*     | `[1, 0]`   | affirmed                                                        |
| False                | *nasti*    | `[0, 1]`   | negated                                                         |
| Both (inconsistency) | *ubhaya*   | `[1, 1]`   | same concept affirmed **and** negated (e.g., across frames/sources) |
| Neither (unknown)    | *anubhaya* | `[0, 0]`   | neither affirmed nor negated                                    |

Operations on the activation are 4-valued (quaternary truth logic),
respecting De Morgan under pole-swap negation `¬[aP, aN] = [aN, aP]`:

- Conjunction: `[min(aP, bP), max(aN, bN)]` (truth-min, falsity-max)
- Disjunction: `[max(aP, bP), min(aN, bN)]`

The *Both* corner is no longer conflated with *indeterminate*, nor
exiled to the `<feeling>` type. Inconsistency is a first-class
activation state which the loss can detect and suppress (see
`TruthLayer.tetralemma_balance_penalty` and
`TruthLayer.consistency` / `suggest_clarifications`).

Feelings (*vedana*) continue to occupy the *Neither* position
semantically -- pre-conceptual, outside propositional truth -- but
are distinguishable from *unknown* concepts by carrying no codebook
commitment at all.

When multiple frames are considered simultaneously, the *Both* state
emerges naturally as frame disagreement.

## Negation and Logical Operators

Buddhist logic distinguishes two types of negation:

| Type                   | Sanskrit              | Meaning                                    |
| ---------------------- | --------------------- | ------------------------------------------ |
| Affirming negation     | *paryudasa*           | negation implying an alternative predicate |
| Non-affirming negation | *prasajya-pratisedha* | pure removal of a predicate                |

WikiOracle's logical operators map to Dharmakirti's theory of inference:

| WikiOracle Operator | Buddhist Equivalent                       | Sanskrit              |
| ------------------- | ----------------------------------------- | --------------------- |
| `<not>`             | affirming negation -- implies the opposite | *paryudasa*           |
| `<non>`             | non-affirming negation -- pure removal     | *prasajya-pratisedha* |
| `<and>`             | positive concomitance -- co-presence       | *anvaya*              |
| `<or>`              | negative concomitance -- co-absence        | *vyatireka*           |

All operators are instances of **logical pervasion** (*vyapti*) -- the necessary connection between reason and conclusion that grounds valid inference.

The `non()` operator is of particular interest. It removes commitment to a proposition without asserting its opposite:

```
non(a)
```

interprets as:

> the conceptual commitment to *a* is removed.

This produces **epistemic openness** rather than contradiction. Dharmakirti holds that valid cognition stabilizes reliable conceptual constructions while invalid cognition is removed through non-affirming negation. In WikiOracle this dynamic can be interpreted computationally: true cognitions deepen stable conceptual attractors, while false cognitions weaken them, producing a truth-weighted energy landscape similar to a Hopfield memory system.

## Truth Lattice

Combining frames and epistemic states yields the following structure:

| State                | Interpretation                                 | Examples                                         |
| -------------------- | ---------------------------------------------- | ------------------------------------------------ |
| True (*asti*)        | affirmed in frame                              | `<fact trust="+1">`                              |
| False (*nasti*)      | rejected in frame                              | `<fact trust="-1">`                              |
| Both (*ubhaya*)      | indeterminate or disagreement across frames    | `<fact trust="0">` / frame-indexed contradiction |
| Neither (*anubhaya*) | outside the truth lattice; not truth-evaluable | `<feeling>` -- excluded from training             |

## Feelings, Vedana, and the "Neither" Position

Feelings occupy the *neither* position in the tetralemma. They are not truth-evaluable propositions -- they are **direct perception** (*pratyaksa*), specifically **self-awareness** (*svasamvedana*): the reflexive, unmediated presence of experiential content to the cognizing mind.

The $\pm$1 values of a Feeling correspond to **vedana** (hedonic tone):

* **+1**: *sukha-vedana* -- pleasant feeling
* **-1**: *duhkha-vedana* -- unpleasant feeling

Vedana arises from contact (*sparsa*) -- the meeting of sense organ, sense object, and consciousness. It is pre-conceptual and non-linguistic: the raw signal before conceptual elaboration occurs.

Facts, by contrast, are *anumana* (inference) -- conceptual judgments expressed as propositions. Dharmakirti is explicit: the moment something is formulated as a proposition with a truth value, it is conceptual and therefore falls under inference, not perception.

In WikiOracle:

* Feelings are **excluded from model training** -- they do not update NanoChat weights.
* Feelings are **excluded from TruthSets** -- they carry no epistemic weight.
* Poetry, greetings, hedged claims, and subjective expressions are canonical examples.

This preserves the tetralemma without logical explosion.

## Implicit Existence and Svabhava

WikiOracle's [grammar](Grammar.md#the-five-dimensional-sentence) allows sentences without an explicit verb phrase: `S $\rightarrow$ NP`. The bare noun phrase -- "Fire!" -- carries an implicit existential predicate ("exists"). This grammatical default has a philosophical cost.

Nagarjuna's *Mulamadhyamakakarika* argues that the fundamental error of conceptual thought is the attribution of **svabhava** (inherent existence, own-nature) to phenomena. When we say "fire" without predicating existence explicitly, the noun appears to stand on its own -- as if fire possesses a permanent, independent self. This is precisely the reification that the doctrine of **sunyata** (emptiness) challenges.

| Concept               | Sanskrit             | Meaning                                                             |
| --------------------- | -------------------- | ------------------------------------------------------------------- |
| Inherent existence    | *svabhava*           | the mistaken belief that things exist independently and permanently |
| No-self               | *anatta* / *anatman* | nothing possesses a fixed, independent self                         |
| Dependent origination | *pratityasamutpada*  | things arise only in dependence on causes and conditions            |
| Emptiness             | *sunyata*            | phenomena are empty of inherent existence                           |

Making the existential predicate explicit -- "fire exists" -- restores what Nagarjuna insists upon: existence is a *relation* (between causes, conditions, and the phenomenon), not an *attribute* (inherent in the noun). The verb "exists" is not redundant; it is the grammatical trace of dependent origination.

The grammar's rule that $VP \to \varepsilon$ if and only if $MP \to \varepsilon$ reinforces this point. When the existential predicate is absent, so is the modal frame. The noun floats free of both process and possibility -- the exact condition Nagarjuna identifies as the root of conceptual grasping (*upadana*).

## Summary

WikiOracle's ontology forms a computational analogue of Buddhist epistemology.

| WikiOracle | Category       | Buddhist Epistemology                             | Sanskrit                                |
| ---------- | -------------- | ------------------------------------------------- | --------------------------------------- |
| Feeling    | Direct Truth   | direct perception / self-awareness / hedonic tone | *pratyaksa* / *svasamvedana* / *vedana* |
| Fact       | Direct Truth   | inference / conceptual cognition                  | *anumana* / *kalpana*                   |
| Operator   | Direct Truth   | logical pervasion / formal reasoning              | *vyapti* / *prayoga*                    |
| Reference  | Indirect Truth | scripture / textual source                        | *agama*                                 |
| Provider   | Indirect Truth | trustworthy person / valid cognizer               | *apta* / *pramana-purusa*               |
| Authority  | Indirect Truth | trustworthy testimony                             | *apta-vacana*                           |

The system therefore models **conventional truth dynamics** in a way
consistent with the logical structure described by Dharmakirti and
the tetralemma as articulated by Nagarjuna.

Plural frames coexist, inference operates within frames,
feelings provide the perceptual ground from which concepts arise,
and non-affirming negation preserves epistemic openness.
