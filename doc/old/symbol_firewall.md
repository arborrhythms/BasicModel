# Firewall Composed on Symbols

## Summary

This document proposes a **symbol firewall** for a symbolic/subsymbolic language model architecture. The purpose of the firewall is to ensure that all computation occurs on **known, typed, introspectable units** rather than on an unrestricted opaque residual stream.

The units are not necessarily words. They may be concepts, percepts, relations, parts, roles, affordances, prototypes, memories, discourse objects, or latent feature-bundles. The requirement is that every computational unit must have:

1. a stable code or address;
2. a semantic type;
3. a meaning-bearing role in the model;
4. an introspectable interface;
5. a bounded relation to other units.

The goal is not to eliminate subsymbolic computation. The goal is to prevent subsymbolic computation from becoming an unanalyzable transformer-like substrate.

---

## Core Claim

A safer symbolic/subsymbolic LLM should not allow computation over anonymous dense state. Every operation must be composed over known units.

In short:

> All computation must be symbol-composed, even when the symbols are not words.

A “symbol” in this sense is any model-internal unit that has a code, type, reference, role, and inspectable meaning. Words are only one class of symbols. Other symbols may represent visual forms, affordances, grammatical roles, semantic features, mereological parts, latent prototypes, discourse entities, or abstract operators.

---

## The Problem

Transformers scale because they allow high-bandwidth dense computation over token embeddings and residual streams. But this creates a safety and interpretability problem: the model may compute through distributed patterns that do not correspond to any stable, inspectable semantic object.

A hybrid symbolic/subsymbolic architecture improves on this only if the subsymbolic side is prevented from becoming another opaque residual stream.

The danger is:

```text
symbolic parser + symbolic labels + large unrestricted latent stream = transformer with decorations
```

The desired architecture is:

```text
typed symbols + attached latent states + bounded operations + emitted semantic deltas = introspectable computation
```

---

## What Counts as a Symbol?

A symbol is not merely a word token. A symbol is a computational unit with semantic standing.

A unit qualifies as a symbol if it has:

| Property | Description |
|---|---|
| Code | A stable internal address, index, key, vector-code, or graph node ID |
| Type | A semantic or computational category |
| Meaning | A defined role in interpretation or generation |
| Interface | Declared read/write permissions and valid operations |
| Provenance | Some account of how the unit was introduced or inferred |
| Relations | Links to other units, such as part-of, instance-of, causes, modifies, refers-to |
| Introspection | A way to inspect its current state, confidence, and semantic effect |

Examples include:

- word senses;
- entities;
- noun phrases;
- verb phrases;
- predicates;
- relations;
- perceptual prototypes;
- latent affordance codes;
- mereological parts;
- discourse referents;
- speaker intentions;
- memory traces;
- scene objects;
- sensory features;
- grammatical roles;
- semantic masks.

Thus the architecture can retain nonverbal and subsymbolic cognition without allowing unstructured opacity.

---

## The Firewall Principle

The symbol firewall enforces the following rule:

> No computation may operate on, mutate, or preserve model state unless that state is attached to a known symbolic unit or is immediately compressed into one.

This means that dense computation is permitted only when it is:

1. **attached** to a symbol, concept, percept, relation, or role;
2. **typed** by a declared semantic category;
3. **masked** by a relevant feature or part structure;
4. **bounded** in scope and persistence;
5. **auditable** through an emitted semantic delta.

Subsymbolic computation is therefore not forbidden. It is domesticated.

---

## Architectural State

A model state may be represented as:

```text
State = (G, C, V, M, P, U)
```

Where:

| Component | Meaning |
|---|---|
| `G` | Symbolic graph of concepts, entities, relations, parts, and discourse objects |
| `C` | Codebook of known units, including verbal and nonverbal concepts |
| `V` | Latent vectors attached to symbolic units |
| `M` | Masks controlling which features, parts, or dimensions may be read or written |
| `P` | Policies, type constraints, safety constraints, and factuality constraints |
| `U` | Uncertainty state: ambiguity, confidence, conflict, unresolved reference |

The central design constraint is that `V` cannot become a free-floating global latent memory. Each latent vector must be attached to an entry in `G` or `C`.

---

## Computation Contract

Every computational module must satisfy a contract of this form:

```text
Operation:
  input_units:      known symbolic units
  input_types:      declared semantic/computational types
  read_mask:        permitted features, parts, or dimensions
  write_mask:       permitted features, parts, or dimensions
  latent_kernel:    optional subsymbolic computation
  output_delta:     symbolic or symbol-attached state change
  explanation:      inspectable account of the operation's semantic effect
  confidence:       uncertainty or quality score
  provenance:       record of sources and dependencies
```

A dense module may perform work that is not directly expressible in words, but it must still report what kind of thing it did.

For example:

```text
Module: MetaphorFitKernel
Input:  [source_concept, target_concept]
Reads:  prototype geometry, affective tone, affordance overlap
Writes: metaphorical-fit score, candidate mapping edges
Emits:  “The source and target share motion/containment structure, but differ in agency.”
```

The exact latent geometry need not be fully verbalized. But the module’s role, input, output, and semantic effect must be inspectable.

---

## Firewall Invariants

The symbol firewall enforces these invariants.

### 1. No anonymous state

There may be no persistent latent state without an associated symbolic address.

Bad:

```text
z_next = F(z_current, token)
```

Good:

```text
concept[apple].latent.surface_features = F(concept[apple].latent.surface_features, context)
```

---

### 2. No unrestricted global residual stream

The architecture must not contain a large hidden vector that all modules can read and write freely.

All cross-module communication must pass through symbols, typed latent fields, masks, or emitted deltas.

---

### 3. All latent vectors are owned by symbols

A dense representation may exist only as the latent interior of a concept, percept, relation, scene, memory, or operator.

The latent vector is therefore not self-standing. It is an attribute of an inspectable unit.

---

### 4. All operations declare read/write masks

A module must declare which parts of a unit it reads and which parts it may modify.

For example, the verb “polish” may read/write the surface features of an object, while “own” modifies social/legal relation features rather than physical texture.

---

### 5. Semantic mutation must emit a delta

If computation changes the meaning, type, salience, relation, or state of a unit, it must emit a symbolic delta.

Example:

```text
Delta:
  unit: apple_17
  changed_feature: surface_luster
  cause: polish(John, apple_17)
  confidence: high
```

---

### 6. Unresolved latent work must be compressed or discarded

Temporary latent workspaces may exist during computation, but they cannot persist indefinitely unless they are compressed into a symbolic unit, attached latent field, uncertainty object, or candidate set.

---

### 7. New concepts require promotion

If a latent pattern recurs and proves useful, it should be promoted into the codebook as a named or addressable concept.

Promotion may create:

- a new concept;
- a new feature;
- a new relation;
- a new prototype;
- a new affordance;
- a new discourse role;
- a new operator.

This keeps the architecture from hiding long-term knowledge in latent space.

---

## Verb Application Example

A verb should not apply to an NP as an undifferentiated vector. It should apply to the NP through a semantic mask.

A verb can be represented as:

```text
Verb = (type_constraints, read_mask, write_mask, latent_transform, emitted_delta)
```

For example:

```text
polish(John, apple)
```

The verb “polish” requires that the object have something like a surface. It reads and modifies surface-related features.

```text
Input unit: apple
Relevant parts: surface, material, visible texture
Read mask: dullness, roughness, coating, color, reflectance
Write mask: luster, smoothness, visible cleanliness
Output delta: apple.surface_luster increased
```

By contrast:

```text
own(John, apple)
```

does not primarily modify the apple’s surface, color, mass, or edibility. It modifies a social/legal relation.

```text
Input units: John, apple
Relevant relation: possession/control/claim
Read mask: agent, object, social context
Write mask: ownership relation
Output delta: owns(John, apple)
```

Thus the same NP can participate in different computations because different operators activate different masks.

---

## Handling Nonverbal Concepts

The firewall does not require that every concept be verbal.

Some units may be meaningful but hard to name:

- visual gestalt;
- rhythm;
- affective tone;
- bodily affordance;
- metaphorical resonance;
- scene coherence;
- phonological fluency;
- perceptual similarity;
- prototype distance.

These units can remain subsymbolic internally, but they must have codes and interfaces.

For example:

```text
Concept code: perceptual_gestalt_4812
Type: visual-gestalt
Meaning: recurring shape/texture configuration
Attached latent: yes
Readable by: scene parser, analogy kernel, memory matcher
Writable by: perceptual update module
Inspectable as: nearest prototypes, examples, contrastive features, confidence
```

This allows the architecture to represent meaningful nonverbal cognition without pretending that all meaning is reducible to language.

---

## Audit Record

Every computation should be able to produce an audit record.

Example:

```yaml
operation_id: op_92831
operator: polish
input_units:
  - John: entity/person
  - apple_17: entity/physical-object
constraints_checked:
  - object_has_surface: passed
  - object_is_modifiable: passed
read_mask:
  apple_17:
    - surface_texture
    - surface_luster
    - material_finish
write_mask:
  apple_17:
    - surface_luster
    - visible_cleanliness
latent_kernel:
  name: surface_change_kernel
  version: 0.3
output_delta:
  - apple_17.surface_luster: increased
  - apple_17.visible_cleanliness: increased
confidence: 0.91
explanation: The verb applies to the object's surface features and increases luster/cleanliness.
```

The audit record does not expose every floating-point operation. It exposes the semantic accountability of the computation.

---

## Difference from a Transformer

A transformer allows computation to occur through a dense residual stream distributed across token positions. Some information may be recoverable through probing, but the architecture does not require each computation to be attached to an inspectable semantic unit.

The symbol firewall changes the default.

| Transformer | Symbol-firewalled architecture |
|---|---|
| Dense residual stream | Symbol-attached latent state |
| Token-centered | Concept-centered |
| Attention over positions | Routing over typed units and relations |
| Interpretability by probing | Interpretability by construction |
| Distributed hidden computation | Mask-bounded local computation |
| Latent state can persist opaquely | Latent state must attach, compress, or disappear |

The result is not a purely symbolic system. It is a constrained hybrid system in which dense computation remains powerful but cannot evade conceptual accountability.

---

## Safety Significance

The architecture is safer because its computations can be inspected at the level of known units.

A safety system can ask:

- What concepts were active?
- Which units were read?
- Which units were modified?
- Which masks were applied?
- Which constraints were checked?
- Which latent kernels were used?
- What semantic deltas were emitted?
- What uncertainty remains?
- Which claims depend on which sources or inferences?

This makes safety analysis structural rather than merely behavioral.

Instead of only testing outputs, one can inspect the internal route by which the output was produced.

---

## Minimal Acceptance Criteria

An implementation satisfies the firewall if the following are true:

1. Every persistent state object has a symbolic code.
2. Every code has a type and meaning-bearing role.
3. Every dense vector is attached to a symbol, concept, percept, relation, or operator.
4. Every operation declares its input units and output deltas.
5. Every operation has read/write masks.
6. Every semantic mutation is auditable.
7. No unrestricted global residual stream is available for hidden computation.
8. Temporary latent workspaces are either compressed, attached, or discarded.
9. Recurrent useful latent patterns are promoted into the codebook.
10. Safety and factuality constraints operate over the symbolic graph, not only over final text.

---

## One-Sentence Formulation

A symbol firewall ensures that all computation is composed over known meaning-bearing units: not all units are words, but all persistent units have codes, types, interfaces, and inspectable semantic effects.
