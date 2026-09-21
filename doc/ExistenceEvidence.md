# Existence evidence and grammatical descriptions

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

`TernaryTruthStore` remains the owner of role vectors and fact evidence. Fixed
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

## Historical validation (September 16)

Current controller/effect receipts are in [Testing](Testing.md).

The repository's 51 new probes pass in 3.11 s; ten affected existing files
pass 281 tests with one skip and three warnings in 89.56 s. The full suite
passes with exit status zero: **4279 passed, 51 skipped, 7 xfailed, 185 warnings, 4 subtests passed in 4175.57s (1:09:35)**. All 561 runtime/test/configuration
files remained frozen.
[Full-suite green](benchmarks/2026-09-16-existence-evidence-data/full-suite-green.log),
[source manifest](benchmarks/2026-09-16-existence-evidence-data/full-suite-source-manifest.json).
[Focused tests](benchmarks/2026-09-16-existence-evidence-data/focused-green.log),
[affected tests](benchmarks/2026-09-16-existence-evidence-data/affected-green.log).

Original lookup, metadata and ingestion probes failed before implementation.
Follow-up candidate probes exposed lost final-result provenance, incomplete
checkpoints, replaced scope, public-input flattening and incomplete legacy
fact admission; each failed before its corresponding fix.
[Lookup red](benchmarks/2026-09-16-existence-evidence-data/lookup-red.log),
[metadata red](benchmarks/2026-09-16-existence-evidence-data/metadata-red.log),
[ingestion red](benchmarks/2026-09-16-existence-evidence-data/ingestion-red.log),
[provenance red](benchmarks/2026-09-16-existence-evidence-data/provenance-red.log),
[sidecar red](benchmarks/2026-09-16-existence-evidence-data/sidecar-red.log),
[public-input red](benchmarks/2026-09-16-existence-evidence-data/public-input-red.log),
[legacy admission red](benchmarks/2026-09-16-existence-evidence-data/legacy-admission-red.log).

The isolated candidate checks are preparation evidence. Repository checks
above describe the September 16 runtime. Grammatical routing, ordinary thought
history and nested retention now use the one controller. Residual policy credit
remains item 2. Taxonomy evidence is documented separately
in [Taxonomy queries](TaxonomyQueries.md).
