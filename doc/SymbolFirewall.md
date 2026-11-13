# The Symbol Firewall

*A governing architectural principle. This document states the principle
and maps each invariant onto the structures that realize it, or names the
gap.*

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

The firewall is the negative definition of BasicModel's relation to LLMs: it
rejects a large unrestricted residual stream as the place where meaning lives.
The positive replacement is split across Formal Concept Analysis-like concept
order, where persistent state has extent/intent structure, and DisCoCat-like
grammar composition, where semantic mutations are typed operations over known
units. The firewall requires those units to remain inspectable.

## Two spaces, two metrics, one index: the bifurcation is complete (Alec, 2026-10-20)

Alec: "the centroid definition of parts and wholes from order zero symbols
can continue into higher order symbols, but the conceptual space is not
shared with perceptual space; it is a similarity space where the metric is
not conditioned by distances between objects in the physical world. And
that bifurcation needs to be complete; we don't want leakage."

- **Perceptual space** (PartSpace and WholeSpace, the [0, 1] percept cube
  under the meronomy) holds **forms**. A word form's code is the fold of
  its letters and letter-wholes; its metric is overlap of form. It
  identifies *which word* (step 1 of the identity staircase: the same form
  is the same word symbol) and nothing about meaning. "cat" is near "cot"
  here and that is all it is for.
- **Conceptual space** holds **meanings**. It is a similarity space placed
  by context (the registered priming surface, SBOW), with its own cube and
  its own meet; its metric is never conditioned by form or by physical
  distance. "cat" is near "dog" here. The centroid definition continues
  upward in *this* space: a kind is the fold over its members, an
  individual the fold over its occurrences, a higher-order symbol's
  concept the fold over the concepts it collects — the conceptual
  mereology (a cat's parts; the taxonomy's kinds), a second tree, not the
  perceptual one.
- **The symbol index is the only link.** A symbol has a form in the one
  space and a meaning in the other, tied at the shared index (word DEF
  object). Nothing else crosses: no metric, no gradient, no fold reads
  across the boundary. Any computation that compared a form code with a
  meaning code, or let letter overlap move a meaning, or let meaning move
  a form code, is leakage and a defect.
- **Psychological standing.** Two stores linked by an entry is the
  settled picture of lexical access: Levelt's lemma and lexeme, the form
  lexicons against the semantic system, a form-specific visual word form
  area (Dehaene & Cohen); double dissociations (semantic dementia spares
  regular reading; deep dyslexia reads "cat" as "dog"; surface dyslexia
  the reverse); the tip-of-the-tongue state (Brown & McNeill 1966) as a
  link that fails between two intact stores; separable form and semantic
  priming. The qualification: humans show small form–meaning
  *interaction* — mixed errors above chance (Dell 1986), sound symbolism,
  a small reliable form–meaning systematicity (Monaghan et al. 2014).
  That is shared **activation**, not a shared **metric**: form activates
  its symbol, the symbol primes its meaning, the meaning's neighbours
  prime their symbols and forms — the priming surface over the symbol
  index does exactly this and would produce mixed errors with no geometry
  crossing. The architecture is Levelt on representation and Dell on
  dynamics, which is where the field settled.
- **Consequences.** `bind`'s content term, the priming spread
  (`ActivationPredictor`'s predicted next state), placement by the
  registered surface, and the two identity models (intersection, union)
  are all **in conceptual space**. The perceptual fold enters identity
  only as form identification (the word symbol).

## Hierarchical context preserves the firewall (Alec, 2026-11-08)

A higher-order concept is valid conceptual context for its members. `ConceptHierarchy` derives class directions from member meanings and sends class context back down membership edges at kept closes, through the same architecture at every order. This includes word concepts and formal-argument concepts. Grammar only declares the arguments and supplies kept memberships; it owns no separate class-learning geometry.

The downward update changes meaning directions and preserves form coordinates and confidence radii. A paired chart without a meaning band cannot supply a conceptual view: its form coordinates are never used as a surrogate direction. A whole conceptual chart can supply its native conceptual value, but the implemented downward placement step requires a meaning band. No operator is inserted into perceptual or lexical form context to achieve this influence. [Architecture](Architecture.md#hierarchical-conceptual-context-alec-2026-11-08) gives the current rule and [the certificates](../test/test_argument_concepts.py) exercise ordinary and formal classes.

## Principle

> All computation must be composed over known, typed, introspectable units —
> even when the units are not words. No computation may operate on, mutate, or
> preserve model state unless that state is attached to a known symbolic unit or
> is immediately compressed into one.

A *symbol* here is any model-internal unit with a stable **code** (address),
a semantic **type**, a meaning-bearing **role**, an **interface** (declared valid
operations), **relations** to other units, and a way to **introspect** its state.
Words are one class of symbol; percepts, parts, roles, prototypes, discourse
referents, operators, and grammatical categories are others.

The goal is not to forbid subsymbolic computation. It is to prevent the
subsymbolic side from degenerating into an opaque, transformer-like residual
stream:

```
symbolic parser + symbolic labels + large unrestricted latent stream
    = a transformer with decorations            (the thing we reject)

typed symbols + symbol-attached latent + bounded ops + emitted deltas
    = introspectable computation                (the thing we build)
```

This is the same commitment expressed structurally elsewhere in the codebase:
the five-Space pipeline ([Architecture.md](Architecture.md)), the Space/SubSpace
contract ([Spaces.md](Spaces.md)), the mereology of percepts and wholes
([Mereology.md](Mereology.md)), and the truth/luminosity audit
([Logic.md](Logic.md), [Reasoning.md](Reasoning.md)). The firewall names the
doctrine those pieces jointly satisfy.

## Why this architecture already largely satisfies it

The architecture is **symbol-composed by construction**. There is no global dense
hidden vector that modules read and write freely; state lives in codebook rows
and short-term-memory slots that carry addresses, types, and meronomic relations.
The firewall is therefore mostly a *naming and auditing* discipline over that
structure, plus a small set of identified gaps.

## Invariant $\to$ code scorecard

| # | Invariant | Status | Anchor |
|---|---|---|---|
| 1 | No anonymous persistent state | **satisfied** | `Codebook` rows are addressed by integer position; `part_parents` meronomy (Spaces.py, `Codebook.part_parents`, `Codebook.set_part_parent`) |
| 2 | No unrestricted global residual stream | **satisfied** | Five-Space pipeline; cross-module signal goes through STM + the grammar router, not a shared hidden vector ([Architecture.md](Architecture.md), [Spaces.md](Spaces.md)) |
| 3 | All latent vectors owned by symbols | **satisfied** | Every concept is an allocator id with a `similarity_codebook` row; a word's latent is bound by the DEF row `interpret` writes (Interpret.py) and parthood by LTM1 Part rows |
| 4 | Operations declare read/write masks | **partial** | Every grammar operation declares its subsystem reads and writes (`AccessibleMind.OperatorEffects`) and its footprint over form, meaning and poles (`OperatorFootprint`), checked at grammar load; a face may only remove a write, and a subsystem is written once per round (`EffectRound`). Mode-partitioned `reference_update_mask`, `update_mask_fn` (installed by `install_reference_update_law`), `intent_priming_weights` (Spaces.py), inert under the default `<architecture><meronomy>` setting: the law returns no mask unless it is on. The per-feature mask is the verb's support below. |
| 5 | Semantic mutation emits a delta | **partial** | Truth-domain only: luminosity + `extrapolate` deltas ([Logic.md](Logic.md), [Reasoning.md](Reasoning.md)). Not yet generalized to all mutations. |
| 6 | Temporary latent is compressed or discarded | **satisfied (by design)** | Bounded STM depth + reset cascade ([STM.md](STM.md), [Spaces.md](Spaces.md)); no indefinite latent workspace |
| 7 | Recurrent useful patterns are promoted | **satisfied** | MTM recurrence promotion (`FamiliarTier.promote_candidates`, `conceptMintCount`) |

Rows #4 and #5 are partial: the instance below advances them, and the
remaining gaps are listed under *Future work*.

## A delivered instance: verb application as a masked, sparse semantic edit (invariants #4, #5)

This is not new doctrine: it is the firewall principle applied at one seam.
(An earlier instance, the MetaSymbol participation categories, is retired;
see [DevelopmentHistory](DevelopmentHistory.md#a-metasymbol-participation-categories--a-typed-symbol-owns-a-latent).)

A verb does not transform a noun phrase as an undifferentiated vector. On a
configuration with a meaning block, `lift` applies the verb phrase to its
subject through `MeaningKernels.MeaningAction`: the positive-definite map
$Q\,\mathrm{diag}(e^{w})\,Q^{\top}$, applied with the same $w$ and $Q$ to
each evidence lane in the cube chart. The edit is masked: it is restricted
to each lane's defined coordinates, so an absent coordinate stays absent and
the lanes never mix; for a transitive verb phrase the basis $Q$ is built from
Householder reflections read from the object's meaning and confined to the
object's support, the identity elsewhere, and for an intransitive one it is
the basis its verb group shares (`VerbGroups`). The only per-verb quantity is
the **sparse** log-spectrum $w$, a zero-initialized linear readout of the
verb phrase, soft-thresholded (threshold .1, straight-through) and bounded to
$\pm 4$: an untrained verb is the identity, and a coordinate enters its
support only when its push recurs. `adverb` is the same action in one shared
basis. The inverse solves the same restricted system, so the edit is exactly
invertible given the verb. The full account, with the numerical verb spectrum
that configurations without a meaning block keep, is in
[Language](Language.md#grammarlayer-forward--reverse-inventory).

This is the firewall's verb example made real: the same NP participates in
different computations because each verb phrase supplies its own sparse
spectrum and each object its own basis and support, and the modification is
local, bounded, invertible and reportable.

## Future work (out of scope for this pass)

The firewall holds mostly by construction. Closing it fully needs:

- **Generalized semantic deltas** (#5): emit a structured delta for *every*
  symbol-state mutation, not only truth-domain inferences.
- **Per-operation audit records**: an `operation_id` stamping input units, masks
  checked, latent kernel, output delta, confidence, provenance.
- **Type/role metadata catalog** (#2 hardening): a queryable registry of every
  symbol's declared type and role.
- **Safety/factuality constraints over the symbolic graph**, generalized beyond
  the current truth machinery.

## Acceptance criteria

An implementation satisfies the firewall when: every persistent state object has
a symbolic code with a typed, meaning-bearing role; every dense vector is
attached to a symbol; every operation declares input units, read/write masks, and
an output delta; every semantic mutation is auditable; no unrestricted global
residual stream exists; temporary latent is compressed/attached/discarded;
recurrent useful patterns are promoted; and safety constraints operate over the
symbolic graph rather than only over final text. Today, criteria tied to
invariants #1, #2, #3, #6, #7 hold by construction; #4 and #5 are advanced by the
instance above and completed by the future work.

## One-sentence formulation

> A symbol firewall ensures that all computation is composed over known
> meaning-bearing units: not all units are words, but all persistent units have
> codes, types, interfaces, and inspectable semantic effects.
