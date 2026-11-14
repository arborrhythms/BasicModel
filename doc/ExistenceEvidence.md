# Existence evidence and grammatical descriptions

An idea is conceived and does not exist (decided 2026-10-20). A noun phrase's
default verb phrase is read as *occurs*: presence to the mind in a situation
and nothing more, which holds of an idea, a percept and a relation alike.
Existence proper is a further relation asserted by evidence, and that evidence
is what the readers below weigh.

The grammatical `exist` is `isTrue`'s conceptual face: it reports
the presence of a concept's content as two independent lane magnitudes
(`ThoughtConceptualCapability.extent`) and reads no LTM row. Whether accepted
LTM evidence supports a description is `isTrue`'s question: it reads an ended
proposition's pair, trust and witnesses by reference, or the best matching
row's (`ThoughtLTMCapability.truth_evidence`); estimates, covers, episode
anchors and questions without evidence never answer. Neither face brings a
frame into STM; `query` does that. See
[AccessibleMind](AccessibleMind.md) and [QueryContracts](QueryContracts.md).

The standalone `TruthGroundedReasoner` reader below keeps a maximum-support
contract. It is not the normal thought executor; the storage and admission
rules are shared.

## Standalone reader contract

The reader's `existence_evidence` asks whether the referent of a complete
conceptual description is supported by accepted LTM facts. Its input retains
all occupied NP1/VP/NP2 roles, explicit presence, bindings, semantic scope and
constituent references.
Matching NP1 alone cannot establish a relation. A different VP, NP2, binding,
scope or explicit constituent reference changes the requested description.
The lookup rejects unequal widths; it does not flatten or truncate roles.
[Full lookup](../bin/reasoning.py).

Every occupied role must clear `tau_id` under the existing signed-magnitude
identity score. The minimum role score multiplies the fact's signed trust.
For an exact match, the original degree is preserved. Matching facts contribute
to separate positive and negative support values by maximum, never addition:
repeating a source cannot increase its supported degree. Proposition polarity
determines whether a stored degree supports or refutes the request.
[Matching and support](../bin/reasoning.py).

The result includes both degrees and each matching fact's stable occurrence,
origin, source text, bindings, scope and references. No match means unknown.
Low support remains visible even below the posture threshold; strong positive
and negative support yields `BOTH`. Missing required metadata is reported as
incomplete evidence. The normal controller retains the checked typed result
and its sources through child returns, history and final answer preparation,
including missing-context diagnostics.
[Posture](../bin/reasoning.py), [controller](SelectedMeaning.md).

The reasoner's `exist()` leaves conceptual content and its pair, and its
`is_true()` an ended proposition's pair and witnesses; neither returns a
scalar. Checked evaluation and the normal controller use the rich evidence
result. Conceptual presence is never a substitute for a fact.
[Compatibility view](../bin/reasoning.py).

## Meaning ownership and storage

`ConceptualMeaning` is an owned value, not another memory or planner. Roles are
`[3, D]`, with a separate Boolean presence mask, polarity and an optional
sentence kind (`idea` or `relation`); bindings, scope and typed references are
immutable metadata. It carries no grammatical mode: whether a sentence asserts
is a fact of its kept program. When the `ask` rule lies on the kept head path
(an `ask` inside an operand does not count), the clause's evidence pair and
source evidence are set to $(0,0)$, so an asked clause asserts nothing. A live
clone keeps its current-step gradient. The explicit adapter converts STM order
to infix NP1/VP/NP2 order; depth three uses `[1, 2, 0]`, and depth two uses
`[1, 0]`. The predictor and observation writers share that adapter. A two-role
input retains its VP in the store and the legacy chain reader.
[Value](../bin/Meaning.py),
[adapter](../bin/Meaning.py),
[prediction adapter](../bin/Layers.py),
[observation writer](../bin/Models.py).

`LTM0`, the occurrence store, owns each row's idea vector and its two evidence
lanes, beside its occurrence identity and required-metadata flag. A row's kind
is read, not stored (`LTM0.kind_of`): `estimate` from its expectation provenance, `episode`
and `cover` from its anchors, `question` while it has open references, a direct
document's declared kind, `inference` for a thought conclusion with evidence,
`fact` or `observation` by its source's authority when it has evidence, and
`unverified` otherwise. The sentence close writes its row with a kind read from
structure: `question` when the clause has open references, `unverified` when it
is closed with evidence $(0,0)$ (an asked yes/no sentence), and `observation`
otherwise. Every reader treats `unverified` as making no truth claim. A root
whose evidence is $(0,0)$ commits no event to any identity, since an unasserted
sentence changes no state. Only facts qualify for the standalone reader.
Recording an external input records an observation; explicit TruthSet admission
can accept it as a fact, and a TruthSet may assert only a closed clause:
asserting one with open references raises. Setting writer origin alone does not
accept a fact. A question or estimate cannot certify its own referent.
[Store](../bin/Memory.py),
[admission](../bin/ClauseRow.py),
[write boundary](../bin/Models.py).

Durable occurrence references contain a store namespace and a monotonically
allocated ID. Row positions may change during compaction; those references do
not. Reset clears records without reusing their occurrence IDs. The standalone
reader compares retained constituent-reference identities. The bounded
nested-meaning reader and retention graph are described in
[NestedRetention](NestedRetention.md); selected composition and all three
observation writers preserve nested occurrences
([SelectedMeaning](SelectedMeaning.md)).
[Occurrence identity](../bin/Memory.py),
[compaction](../bin/Memory.py),
[reset](../bin/Memory.py).

## Checkpoints

Role vectors and scalar/enum columns remain tensor state. A versioned
`truth_semantics` entry in the model's existing structural sidecar preserves
bindings, scope, constituent references and source text. Restore checks the
store namespace and occurrence IDs before attaching that metadata. A tensor
fingerprint binds each occurrence to its sidecar content; removing scope,
replacing it or swapping metadata between records cannot restore a different
description under the original occurrence ID. Source-text updates cannot
replace missing required metadata.
There is no dictionary-valued PyTorch `_extra_state` entry.
[Save](../bin/Models.py),
[restore](../bin/Models.py),
[metadata validation](../bin/Memory.py).

A bare tensor-state restore can lack the sidecar. Its required-metadata flag
makes scoped/source-bearing evidence unavailable until the matching metadata
is restored. Missing metadata cannot silently mean empty scope. A partially
saved set of the new semantic columns is rejected even by a non-strict load.
[Guarded read](../bin/Memory.py),
[checkpoint migration](../bin/Memory.py).

A checkpoint in an older row layout is not loaded: loading one fails with the
retired column's name or the unsupported layout version. No legacy row is
reclassified, given a presence mask or certified from part of its roles.
[Layout check](../bin/Memory.py).

## Predictions and fact admission

Expectation's estimates are typed `estimate` rows. Estimates, observations,
questions and unverified rows do not become accepted facts through scalar
conversion or high confidence. Fact admission preserves the complete meaning,
scope and provenance under the store's explicit contract. Ordinary thought
history records checked support and typed results; it does not promote
generated content into truth. See [Queries](../bin/Queries.py),
[thought history](ThoughtHistory.md) and [SelectedMeaning](SelectedMeaning.md).

## Gradients

The meaning-value clone preserves live representation gradients, while durable
store writes detach them. Fact lookup, metadata matching, thresholding and
support aggregation are hard operations with scalar evidence; they provide no
ordinary derivative through their choices, and they add no learned parameter
or loss. The architecture-wide gradient boundaries are documented in
[GradientFlow](GradientFlow.md).
[Owned value](../bin/Meaning.py),
[detached write](../bin/Memory.py),
[hard lookup](../bin/reasoning.py).
