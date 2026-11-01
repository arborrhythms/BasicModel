# Existence evidence and grammatical descriptions

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record. Record: [two truths §1, "The row, field by field"](specs/2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided); [bind plan §6–§6.3](plans/2026-10-19-bind-thought-operation.md); [occurrence spec §13–§14](specs/2026-09-30-occurrence-tense-aspect.md); [expectation spec rev 5](specs/2026-10-08-expectation-at-every-level.md).
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Rows: an LTM1 row is **one** concept slot; an LTM3 row is **three symbol references** (argument, kind, argument) of kind definition, attribution or conditional. The three-slot relation row with a REL code in the middle, null outer vectors, `refs`, `rel_type`, `role_mask`, `order`, `polarity`, `grammatical_mode`, `row_ids`, the address columns and the semantic sidecar are superseded: a row holds its slots, `.when` (model time, a column composition never reads), `truth_plus`/`truth_minus` (the lanes, renamed from `c_plus`/`c_minus`), the access count (renamed from `witness_count`) and `surprise`, and nothing else. **Amended (decided 2026-10-23, [step 2 §71](plans/2026-10-21-step2-one-ltm.md); recorded 2026-10-30):** an LTM1 row is two tagged references with its kind read from the referents, not three. As built at `3718d9d0dc`, `LTM0` still carries `c_plus`/`c_minus`, `source_trust`, `order`, `row_ids`, `origin`, `record_kind`, `polarity` and `witness_count` columns; only `refs`, `grammatical_mode` and the index columns are retired (audit 2026-10-30; decided 2026-11-01, Alec: the unused columns are removed and the row brought to this layout, [thinking spec §32.2](plans/2026-10-27-item2-5-thinking.md) item 17).
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.


The normal grammatical `exist` runs through the single thought controller and
the indexed cue reader. It sums masked role similarity times signed fact trust,
separates positive and negative support, and clips each at one. More matching
facts can increase familiarity. Polarity determines the support direction;
scope and bindings filter candidates. Only facts certify truth. `exist` brings
no frame into STM; a cued `what` does that. See
[AccessibleMind](AccessibleMind.md) and [QueryContracts](QueryContracts.md).

The older standalone `TruthGroundedReasoner` reader below retains its original
maximum-support compatibility contract pending review. It is not the normal
thought executor. The storage and admission rules remain shared.

## Historical standalone reader contract

`Exist` asks whether the referent of a complete conceptual description is
supported by accepted LTM facts. Its input retains all occupied NP1/VP/NP2
roles, explicit presence, bindings, semantic scope and constituent references.
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

`exist()` and `is_true()` retain a scalar compatibility view, positive minus
negative support. That scalar loses conflict information. Checked evaluation
and the normal controller use the rich evidence result. Neither path consults model
activation as a substitute for a fact.
[Compatibility view](../bin/reasoning.py).

## Meaning ownership and storage

`ConceptualMeaning` is an owned value, not another memory or planner. Roles are
`[3, D]`, with a separate Boolean presence mask, grammatical mode and polarity;
bindings, scope and typed references are immutable metadata. A live clone keeps
its current-step gradient. The explicit adapter converts STM order to infix
NP1/VP/NP2 order; depth three uses `[1, 2, 0]`, and depth two uses `[1, 0]`.
The predictor and observation writers share that adapter. A two-role input
retains its VP in the store and the legacy chain reader.
[Value](../bin/Meaning.py),
[adapter](../bin/Meaning.py),
[prediction adapter](../bin/Layers.py),
[observation writer](../bin/Models.py).

`LTM0` remains the owner of role vectors and fact evidence. Fixed
buffers add role presence, grammatical mode, polarity, record kind, occurrence
identity and the required-metadata flag. Supported kinds are `fact`, `question`,
`estimate`, `observation` and `unverified`. Only assertive facts qualify for
Exist. Recording an external input records an observation; explicit TruthSet
admission can accept it as a fact. Setting writer origin alone does not accept
a fact. A question or estimate cannot certify its own referent.
[Store](../bin/Layers.py),
[admission](../bin/Layers.py),
[write boundary](../bin/Layers.py).

Durable occurrence references contain a store namespace and a monotonically
allocated ID. Row positions may change during compaction; those references do
not. Reset clears records without reusing their occurrence IDs. Exist compares retained constituent-reference identities. The bounded
nested-meaning reader and retention graph are described in
[NestedRetention](NestedRetention.md); selected composition and all three
observation writers now preserve nested occurrences ([SelectedMeaning](SelectedMeaning.md)).
[Occurrence identity](../bin/Layers.py),
[compaction](../bin/Layers.py),
[reset](../bin/Layers.py).

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
[metadata validation](../bin/Layers.py).

A bare tensor-state restore can lack the sidecar. Its required-metadata flag
makes scoped/source-bearing evidence unavailable until the matching metadata
is restored. Missing metadata cannot silently mean empty scope. A partially
saved set of the new semantic columns is rejected even by a non-strict load.
[Guarded read](../bin/Layers.py),
[checkpoint migration](../bin/Layers.py).

Older checkpoints lack all these columns. Explicit provisioned/user TruthSet
origins may retain fact status; unclassified conversation rows become
unverified. Old nonzero slots provide a best-effort presence mask. A legacy
relation with no recoverable VP is unverified, rather than certified from NP1.
Old missing scope/source text is not reconstructed from model activation.
[Legacy migration](../bin/Layers.py).

## Predictions and fact admission

Checked `arma` results are typed estimates. Estimates, observations, questions
and unverified rows do not become accepted facts through scalar conversion or
high confidence. Fact admission preserves the complete meaning, scope and
provenance under the existing store's explicit contract. The addressee/testimony
registry and legacy world-lemma writer are removed. Ordinary thought history
records checked support and typed results; it does not promote generated
content into truth. See [Queries](../bin/Queries.py),
[thought history](ThoughtHistory.md) and [SelectedMeaning](SelectedMeaning.md).

## Gradients

The meaning-value clone preserves live representation gradients, while durable
store writes detach them. Fact lookup, metadata matching, thresholding and
support aggregation are hard operations with scalar evidence; they provide no
ordinary derivative through their choices. Later learned query selection needs
its declared policy credit. This migration adds no learned parameters or new
loss, and does not itself train useful questioning. The architecture-wide
gradient boundaries remain documented in [GradientFlow](GradientFlow.md).
[Owned value](../bin/Meaning.py),
[detached write](../bin/Layers.py),
[hard lookup](../bin/reasoning.py).

*Moved to [DevelopmentHistory](DevelopmentHistory.md#historical-validation-september-16) (2026-10-30): "Historical validation (September 16)".*
