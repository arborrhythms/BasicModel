# Proposal: Semantic Differentiation of Verbs by NP Masking and Eigenvalue Modification

## 1. Summary

This proposal defines a compact representation of verbs as operations over noun-phrase embeddings. The central claim is that a verb does not transform an entire noun phrase uniformly. Rather, a verb selects a relevant substructure of the NP, applies a local semantic transformation to that substructure, and then recombines the transformed portion with the unmodified complement of the NP.

In this framework, a verb is represented by:

```text
verb v = (R_v, P_v, Δ_v, τ_v)
```

where:

```text
R_v = required or expected mereological part-schema
P_v = projection/mask over the relevant NP features or parts
Δ_v = semantic edit, possibly represented as an eigenvalue modification
τ_v = temporal/event contour of the verb
```

The resulting composition is:

```text
NP₂ = Comask_v(NP₁) + VP_v(Mask_v(NP₁))
```

or more formally:

```text
x₂ = (I - P_v)x₁ + T_v(P_vx₁)
```

For many verbs, an edit formulation is preferable:

```text
x₂ = x₁ + P_vΔ_v(x₁)
```

This allows a verb to modify only those features of the NP that it semantically touches, while preserving the remaining identity and background features of the NP.

## 2. Motivation

Traditional syntax allows a broad abstraction:

```text
NP + VP → Sentence
```

However, this abstraction does not explain why some combinations are ordinary, some metaphorical, and some nonsensical:

```text
The person walks.
The car walks.
The number seven walks.
```

All three are syntactically valid. The difference lies in whether the NP contains the parts, features, or affordances on which the VP can operate.

Rather than treating this as a hard selectional restriction, this proposal treats the verb as an always-applicable operator. Nonsense does not arise because the verb refuses to apply. Nonsense arises because the transformed subspace cannot be coherently recombined with the untransformed complement of the NP.

## 3. NP Representation

An NP embedding represents a conceptual whole. It may be decomposed into:

```text
NP = whole + parts + relations + affordances + state features
```

For example, the NP `person` may include:

```text
body
limbs
posture
location
agency
perception
memory
legal identity
biographical continuity
social role
```

The NP `car` may include:

```text
body
wheels
engine
road relation
artifacthood
ownership
passenger capacity
motion state
```

The NP `number seven` may include:

```text
abstract object
ordinal/cardinal value
mathematical relations
symbolic representation
```

Each NP therefore has a mereological structure: a decomposition into relevant parts and their relations.

## 4. Verb Representation

A verb is represented as a compact operator over a projected subspace of the NP.

```text
v = (R_v, P_v, Δ_v, τ_v)
```

### 4.1 Required Part-Schema: `R_v`

`R_v` specifies the parts or affordances the verb normally operates on.

For example:

```text
R_walk =
  body
  locomotor supports
  ground contact
  posture
  cyclic motion
  trajectory
  self-propulsion
```

```text
R_rust =
  material surface
  oxidizable substrate
  environmental exposure
  chemical-state feature
```

```text
R_believe =
  cognizer
  mental-state register
  proposition
  epistemic commitment
```

This schema is not a hard syntactic precondition. It is a description of what the verb knows how to modify.

### 4.2 Projection or Mask: `P_v`

`P_v` selects the portion of the NP relevant to the verb.

```text
P_vx = verb-relevant projection of NP x
```

For `walk`, the projection includes motion, posture, locomotor morphology, ground relation, and trajectory. It excludes most identity features.

For example:

```text
P_walk(person) → legs, posture, gait, motion-state, ground-contact
P_walk(car) → wheels, motion-state, ground-contact, trajectory
P_walk(seven) → nearly empty
```

The complement is:

```text
C_v = I - P_v
```

So:

```text
C_vx = NP features not modified by the verb
```

### 4.3 Semantic Edit: `Δ_v`

`Δ_v` is the semantic force of the verb. It modifies the projected portion of the NP.

For many verbs, the edit form is preferable:

```text
x₂ = x₁ + P_vΔ_v(x₁)
```

This says that the verb changes the NP state rather than replacing the NP.

For `walk`, the edit may increase or modify features such as:

```text
in-motion
grounded locomotion
cyclic gait
agentive bodily movement
trajectory
energy expenditure
```

For `rust`, the edit may modify:

```text
surface oxidation
material degradation
color change
chemical exposure
structural weakening
```

For `believe`, the edit may modify:

```text
epistemic commitment
mental-state alignment
propositional attachment
confidence weighting
```

### 4.4 Temporal/Event Contour: `τ_v`

Verbs also encode temporal structure.

Examples:

```text
walk    → durative activity
arrive  → punctual achievement
rust    → gradual process
know    → stative condition
break   → change-of-state event
give    → transfer event
```

Thus two verbs may modify similar feature regions but differ in temporal contour.

For example:

```text
walk  = sustained locomotion
step  = discrete locomotor event
run   = higher-intensity locomotion
limp  = asymmetric or impaired locomotion
```

## 5. Composition

The core composition operation is:

```text
x₂ = (I - P_v)x₁ + T_v(P_vx₁)
```

This means:

1. preserve the parts of the NP not touched by the verb;
2. transform the verb-relevant portion;
3. recombine the transformed part with the preserved complement.

For many verbs, this is better written as a residual edit:

```text
x₂ = x₁ + P_vΔ_v(x₁)
```

The residual form preserves identity by default. A person who walks remains the same person, but with an updated motion-state.

## 6. Eigenvalue Interpretation

If the NP embedding is decomposed into modes or eigenfeatures, then the verb can be represented compactly as an edit to those eigenvalues.

Let:

```text
x = Qλ
```

where:

```text
Q = basis of conceptual/eigenfeature directions
λ = eigenvalue or feature-magnitude vector
```

A verb may act as:

```text
λ₂ = λ₁ + p_v ⊙ δ_v
```

where:

```text
p_v = mask over verb-relevant eigenfeatures
δ_v = verb-specific eigenvalue edit
⊙ = elementwise multiplication
```

Then:

```text
x₂ = Qλ₂
```

or:

```text
x₂ = Q(λ₁ + p_v ⊙ δ_v)
```

This provides a compact spectral representation of verb meaning.

For example:

```text
walk:
  p_v selects locomotion/posture/trajectory modes
  δ_v increases cyclic self-motion and grounded trajectory modes
```

```text
rust:
  p_v selects surface/material/chemical-state modes
  δ_v increases oxidation and degradation modes
```

```text
believe:
  p_v selects epistemic/mental-state/proposition-binding modes
  δ_v increases commitment or acceptance modes
```

## 7. Literal, Metaphorical, and Nonsensical Outputs

This model does not require a precondition of semantic validity. The verb always applies.

The important question is whether the resulting recombination is coherent:

```text
x₂ = x₁ + P_vΔ_v(x₁)
```

The resulting `x₂` may be:

```text
literal
metaphorical
fictional
nonsensical
```

depending on whether the transformed subspace coheres with the NP complement.

### 7.1 Literal Case

```text
The person walks.
```

The NP contains the relevant part-structure:

```text
body
legs
posture
ground contact
self-motion
```

The verb has strong purchase, and the recombination is coherent.

```text
person + walking-state → coherent person-undergoing-walking
```

### 7.2 Metaphorical or Coerced Case

```text
The car walked down the road.
```

The car lacks literal legged gait, but it has partial overlap with the walk-schema:

```text
body
ground contact
trajectory
motion
```

The result may be interpreted as:

```text
slow, uneven, stepwise, or awkward vehicle motion
```

The phrase is not blocked. It is stabilized through analogy.

### 7.3 Near-Nonsense Case

```text
The number seven walked.
```

The NP has almost no overlap with the walk-schema. The projected subspace is nearly empty:

```text
P_walk(seven) ≈ 0
```

The verb still applies, but the output does not form a coherent literal object-state unless the context supplies fiction, animation, or metaphor.

## 8. Diagnostic Measures

The model can use several diagnostic quantities.

### 8.1 Verb Purchase

```text
purchase_v(x) = ||P_vx||
```

This measures whether the verb has relevant NP material to transform.

A normalized version is:

```text
purchase_v(x) = ||P_vx|| / ||x||
```

Low purchase indicates that the NP lacks the parts or features the verb normally modifies.

### 8.2 Mereological Coverage

```text
coverage_v(x) = match(A(x), R_v)
```

where:

```text
A(x) = mereological analysis of NP x
R_v = required part-schema of verb v
```

This measures whether the NP contains enough of the verb’s expected part-structure.

### 8.3 Output Coherence

```text
coherence_v(x₂) = K(x₂)
```

where `K` measures whether the resulting NP state lies near a learned manifold of coherent entity-states or event-states.

This distinguishes:

```text
high purchase + high coherence    → literal use
partial purchase + recoverable coherence → metaphor/coercion
low purchase + low coherence      → nonsense
```

## 9. Verb Classes as Derived Clusters

Traditional semantic verb classes can be recovered as clusters of similar masks and edits.

For example:

```text
locomotion verbs:
  walk, run, crawl, limp, jump
```

These share similar projections over motion, posture, body, ground-contact, and trajectory features, but differ in their eigenvalue edits and temporal contours.

```text
material-change verbs:
  rust, melt, freeze, evaporate, burn
```

These share projections over material state, surface, phase, temperature, and chemical composition.

```text
cognition verbs:
  believe, know, imagine, remember, doubt
```

These share projections over mental-state, proposition-binding, memory, attention, and epistemic commitment.

Thus verb classes need not be primitive syntactic categories. They can be derived from similarities among:

```text
R_v
P_v
δ_v
τ_v
```

## 10. Relation to Syntax

The syntactic rule remains broad:

```text
NP + VP → Lift
```

But syntax is enriched by semantic operator profiles.

The grammar does not need to block expressions like:

```text
The theorem coughed.
The car limped.
The city slept.
The corporation panicked.
```

Instead, it composes them and then evaluates the resulting semantic state.

This preserves the generativity of language while explaining why some combinations are literal, some metaphorical, and some nonsensical.

## 11. Proposed Implementation

### 11.1 NP Encoding

Represent each NP as:

```text
x = entity embedding
A(x) = mereological decomposition
```

The decomposition may be represented as:

```text
parts
relations
affordances
state features
identity-preserving features
```

### 11.2 Verb Encoding

Represent each verb sense as:

```text
v = (R_v, p_v, δ_v, τ_v)
```

where:

```text
R_v = part-schema
p_v = feature/eigenvalue mask
δ_v = edit vector
τ_v = event contour
```

For polysemous verbs, store multiple verb senses:

```text
drive₁ = operate vehicle
drive₂ = move under power
drive₃ = compel
drive₄ = propel
```

Each sense has its own mask and edit.

### 11.3 Composition

For an intransitive verb:

```text
x₂ = x₁ + p_v ⊙ δ_v(x₁)
```

For a transitive verb:

```text
x₂_subject = x_subject + p_subject,v ⊙ δ_subject,v(x_subject, x_object)
x₂_object  = x_object  + p_object,v  ⊙ δ_object,v(x_subject, x_object)
```

The event representation is:

```text
E_v = Lift(x_subject, v, x_object)
```

For ditransitives:

```text
give(agent, theme, recipient)
```

the verb updates multiple participant states:

```text
agent: loss/control-transfer edit
theme: possession/location-transfer edit
recipient: acquisition/control edit
```

### 11.4 Coherence Evaluation

After composition:

```text
x₂ = x₁ + P_vΔ_v(x₁)
```

evaluate:

```text
purchase_v(x₁)
coverage_v(x₁)
coherence_v(x₂)
```

These diagnostics classify the output as literal, metaphorical, coerced, fictional, or nonsensical.

## 12. Example Table

| Verb | Required Part-Schema `R_v` | Mask `P_v` | Eigenvalue/Edit `δ_v` | Temporal Type |
|---|---|---|---|---|
| walk | body, supports, ground contact, trajectory | gait, posture, motion | increase cyclic self-locomotion | durative activity |
| run | body, supports, ground contact, trajectory | gait, speed, exertion | increase high-speed locomotion | durative activity |
| sleep | organism, rest cycle, consciousness | arousal, posture, activity | decrease wakeful activity | state |
| rust | oxidizable surface, exposure | material surface, chemistry | increase oxidation/degradation | gradual process |
| melt | solid material, heat/phase | phase-state, temperature | shift solid toward liquid | change of state |
| believe | cognizer, proposition | epistemic state, commitment | increase acceptance/commitment | mental state |
| give | agent, theme, recipient | possession/control relations | transfer possession/control | event |
| break | structured object, integrity | structure, integrity | decrease cohesion/integrity | achievement |

## 13. Advantages

This approach has several advantages:

1. It preserves broad syntactic compositionality.
2. It explains selectional effects without hard selectional restrictions.
3. It distinguishes literal, metaphorical, and nonsensical outputs.
4. It gives a compact representation of verb meaning.
5. It allows verb classes to emerge from shared masks and edits.
6. It integrates naturally with a mereological theory of noun phrases.
7. It supports reversible or residual semantic updates.
8. It allows polysemy by representing each verb sense as a separate mask/edit pair.

## 14. Core Claim

The semantic content of a verb can be represented as a compact operation over an NP embedding:

```text
Verb meaning = mereological part-schema + feature/eigenvalue mask + semantic edit + event contour
```

A verb does not simply attach to an NP. It selects a relevant substructure of the NP, modifies that substructure, and recombines the result with the preserved complement of the NP.

Literal meaning occurs when the NP contains the parts necessary for the verb and the recombined output is coherent.

Metaphor occurs when the NP partially contains or analogically simulates the relevant part-structure.

Nonsense occurs when the verb applies but the NP lacks the relevant parts, or when the transformed subspace cannot coherently recombine with the NP complement.

## 15. Minimal Formula

The minimal formula is:

```text
x₂ = x₁ + P_vΔ_v(x₁)
```

with diagnostics:

```text
purchase_v(x₁) = ||P_vx₁||
coverage_v(x₁) = match(A(x₁), R_v)
coherence_v(x₂) = K(x₂)
```

This gives a compact, computable account of how verbs semantically differentiate according to their modification of NP masks and eigenvalue states.
