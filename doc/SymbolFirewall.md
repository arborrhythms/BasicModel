# The Symbol Firewall

*A governing architectural principle. This document is the code-anchored,
normative version of an earlier source proposal — it states the principle
and maps each invariant onto the structures that already realize it (or
names the gap).*

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

The firewall is the negative definition of BasicModel's relation to LLMs: it
rejects a large unrestricted residual stream as the place where meaning lives.
The positive replacement is split across Formal Concept Analysis-like concept
order, where persistent state has extent/intent structure, and DisCoCat-like
grammar composition, where semantic mutations are typed operations over known
units. The firewall requires those units to remain inspectable.

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
The firewall is therefore mostly a *naming and auditing* discipline over what is
already built, plus a small set of identified gaps.

## Invariant $\to$ code scorecard

| # | Invariant | Status | Anchor |
|---|---|---|---|
| 1 | No anonymous persistent state | **satisfied** | `Codebook` rows are addressed by integer position; `part_parents` meronomy (Spaces.py, `Codebook.part_parents`, `Codebook.set_part_parent`) |
| 2 | No unrestricted global residual stream | **satisfied** | Five-Space pipeline; cross-module signal goes through STM + the grammar router, not a shared hidden vector ([Architecture.md](Architecture.md), [Spaces.md](Spaces.md)) |
| 3 | All latent vectors owned by symbols | **satisfied** | META taxonomy binds every latent to a position: `WholeSpace.insert_whole`, `WholeSpace.insert_meta`, `WholeSpace.taxonomy_parent` (Spaces.py; note `ConceptualSpace` also defines an unrelated, same-named `taxonomy_parent` — disambiguate by class) |
| 4 | Operations declare read/write masks | **partial** | Mode-partitioned `reference_update_mask`, `update_mask_fn` (installed by `install_reference_update_law`), `intent_priming_weights` (Spaces.py). **No per-operation *feature* mask** until the verb edit below. |
| 5 | Semantic mutation emits a delta | **partial** | Truth-domain only: luminosity + `extrapolate` deltas ([Logic.md](Logic.md), [Reasoning.md](Reasoning.md)). Not yet generalized to all mutations. |
| 6 | Temporary latent is compressed or discarded | **satisfied (by design)** | Bounded STM depth + reset cascade ([STM.md](STM.md), [Spaces.md](Spaces.md)); no indefinite latent workspace |
| 7 | Recurrent useful patterns are promoted | **satisfied** | LBG codebook split + meta auto-bind `ConceptualSpace._maybe_autobind_meta` (Spaces.py) |

"Partial" rows (#4, #5) are where this body of work makes concrete progress; the
remaining gaps are listed under *Future work*.

## Two delivered instances

These are not new doctrine — they are the firewall principle applied at two
specific seams, each gated dark by default (flag-off is byte-identical).

### A. MetaSymbol participation categories — *a typed symbol owns a latent
category* (invariants #3, #7)

A word's syntactic category is learned as its **frequency of participation across
operator roles** (`<method>_I<n>` inputs / `<method>_O1` outputs), stored as a
small `VectorQuantize` codebook keyed by the **MetaSymbol** (the equivalence
class uniting a word with its object). The latent category is therefore *owned*
by an addressable symbol — not floating in a residual stream — and recurrent
role profiles are *promoted* into stable centroids that decay when unused
(`codebook_retire=False`). The structured grammar layers condition routing on
this typed category context through `<categoryCodebook>` for all transform
choosers. Design: [Language.md](Language.md) "Participation Categories".

### B. Verb application as a masked, sparse semantic edit (invariants #4, #5)

A verb does not transform a noun phrase as an undifferentiated vector. It selects
the NP substructure it touches and edits only that, preserving the complement
(`x₂ = x₁ + p_class ⊙ δ_v`, in the atanh content domain). Crucially, **the mask
`p_class` is not a free parameter**: a verb applies to a noun *class* ("jump" $\to$
"things with legs"), and class membership is read from the NP's **own
eigen-signature** — its activated eigenfeatures in the codebook geometry, the
learned class identity it already carries. Where the NP lacks a feature (out of
class) `p_class` $\approx 0$ and that feature is preserved. The only per-verb parameter
is a **sparse** eigenvalue edit `δ_v` over the shared lift operator's content
eigenbasis (the LDU diagonal is the eigenvalue-like component, in `Layers.py`
— not `Language.py`: `InvertibleLinearLayer`'s `d` factor, carried through
`SigmaLayer`'s triangular L/D/U factorization), made sparse by an in-forward
soft-threshold plus the L1 hook `gate_l1_loss`. The edit is a **zero-init residual branch**
(untrained ⇒ no-op ⇒ the sigma fold), and stashes an introspectable
`adverb_purchase` diagnostic — a first, bounded form of the emitted semantic
delta. NOTE (2026-06-20, corrected): only the `verbEigEdit` **residual branch** in
`LiftLayer.forward` was removed (the sparse edit that used to ride on top of the
symmetric sigma fold); that mechanism is now the **adverb** eigenmodifier. A
separate, still-live eigen-spectrum edit remains on the verb itself:
`LiftLayer.apply_verb` / `unapply_verb` (inherited by `VerbLayer`) apply the
verb's own sparse log-eigenvalue spectrum via `LiftLayer._verb_spectrum_w`,
gated by `architecture.verbSpectrum` (force-built on by `VerbLayer`, its own
class, subclassing `LiftLayer`) — so the verb still performs a live eigenvalue
edit, just a different one from the removed residual. The live grammar path for
the mask/mean mechanism described above is `adverb`, implemented by
`AdverbLayer` (also its own class, subclassing `LiftLayer`), which force-builds
the zero-init projection and calls `LiftLayer.apply_adverb`. `<adverbEigEdit>`
remains only as the legacy direct-`LiftLayer` helper flag, following an
earlier source proposal for semantic verb/NP mask eigenvalue edits.

This is the firewall's verb example made real: the same NP participates in
different computations because different operators activate different
class-defined masks, and the modification is local, bounded, and reportable.
(Wiring the *discrete* taxonomy class node — `taxonomy_parent` / `part_parents`
— in place of the NP's eigen-signature proxy is a noted future refinement; it
needs per-slot symbol identity threaded into the lift op, the same plumbing the
category codebook's Phase 2 introduced.)

## Future work (out of scope for this pass)

The firewall is ~70% realized by construction. Closing it fully is deferred:

- **Generalized semantic deltas** (#5): emit a structured delta for *every*
  symbol-state mutation, not only truth-domain inferences.
- **Per-operation audit records**: an `operation_id` stamping input units, masks
  checked, latent kernel, output delta, confidence, provenance (the proposal's
  audit-record shape).
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
two instances above and completed by the future work.

## One-sentence formulation

> A symbol firewall ensures that all computation is composed over known
> meaning-bearing units: not all units are words, but all persistent units have
> codes, types, interfaces, and inspectable semantic effects.
