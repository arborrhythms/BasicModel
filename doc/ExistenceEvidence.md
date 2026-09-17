# Existence evidence and grammatical descriptions

Implementation reference, September 16. Focused probes pass; affected-file
and full-suite validation are recorded in integrated specification §15. The normal grammatical query route and ordinary thought controller remain
separate work. [Query contracts](QueryContracts.md) supplies the explicit
checked shared-VP adapter. [Conceptual-taxonomy PartOf](TaxonomyQueries.md)
now has its own evidence reader.

## What the lookup establishes

`Exist` asks whether the referent of a complete conceptual description is
supported by accepted LTM facts. Its input retains all occupied NP1/VP/NP2
roles, explicit presence, bindings, semantic scope and constituent references.
Matching NP1 alone cannot establish a relation. A different VP, NP2, binding,
scope or explicit constituent reference changes the requested description.
The lookup rejects unequal widths; it does not flatten or truncate roles.
[Full lookup](../bin/reasoning.py#L156).

Every occupied role must clear `tau_id` under the existing signed-magnitude
identity score. The minimum role score multiplies the fact's signed trust.
For an exact match, the original degree is preserved. Matching facts contribute
to separate positive and negative support values by maximum, never addition:
repeating a source cannot increase its supported degree. Proposition polarity
determines whether a stored degree supports or refutes the request.
[Matching and support](../bin/reasoning.py#L156).

The result includes both degrees and each matching fact's stable occurrence,
origin, source text, bindings, scope and references. No match means unknown.
Low support remains visible even below the posture threshold; strong positive
and negative support yields `BOTH`. Missing required metadata is reported as
incomplete evidence. The legacy kernel retains these sources through its final
result, including missing-context diagnostics.
[Posture](../bin/reasoning.py#L586),
[kernel lookup](../bin/thinking.py#L268),
[kernel aggregation](../bin/thinking.py#L398).

`exist()` and `is_true()` retain a scalar compatibility view, positive minus
negative support. That scalar loses conflict information. Checked evaluation
and the kernel use the rich evidence result. Neither path consults model
activation as a substitute for a fact.
[Compatibility view](../bin/reasoning.py#L213).

## Meaning ownership and storage

`ConceptualMeaning` is an owned value, not another memory or planner. Roles are
`[3, D]`, with a separate Boolean presence mask, grammatical mode and polarity;
bindings, scope and typed references are immutable metadata. A live clone keeps
its current-step gradient. The explicit adapter converts STM order to infix
NP1/VP/NP2 order; depth three uses `[1, 2, 0]`, and depth two uses `[1, 0]`.
The predictor and observation writers share that adapter. A two-role input
retains its VP in the store and the legacy chain reader.
[Value](../bin/Meaning.py#L67),
[adapter](../bin/Meaning.py#L15),
[prediction adapter](../bin/Layers.py#L10058),
[observation writer](../bin/Models.py#L133).

`TernaryTruthStore` remains the owner of role vectors and fact evidence. Fixed
buffers add role presence, grammatical mode, polarity, record kind, occurrence
identity and the required-metadata flag. Supported kinds are `fact`, `question`,
`estimate`, `observation` and `unverified`. Only assertive facts qualify for
Exist. Recording an external input records an observation; explicit TruthSet
admission can accept it as a fact. Setting writer origin alone does not accept
a fact. A question or estimate cannot certify its own referent.
[Store](../bin/Layers.py#L8674),
[admission](../bin/Layers.py#L8912),
[write boundary](../bin/Layers.py#L8925).

Durable occurrence references contain a store namespace and a monotonically
allocated ID. Row positions may change during compaction; those references do
not. Reset clears records without reusing their occurrence IDs. This migration
compares retained constituent-reference identities; it does not yet implement
arbitrary nested-meaning traversal or claim its separate acceptance gate.
[Occurrence identity](../bin/Layers.py#L8781),
[compaction](../bin/Layers.py#L9125),
[reset](../bin/Layers.py#L9163).

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
[Save](../bin/Models.py#L4409),
[restore](../bin/Models.py#L4693),
[metadata validation](../bin/Layers.py#L8823).

A bare tensor-state restore can lack the sidecar. Its required-metadata flag
makes scoped/source-bearing evidence unavailable until the matching metadata
is restored. Missing metadata cannot silently mean empty scope. A partially
saved set of the new semantic columns is rejected even by a non-strict load.
[Guarded read](../bin/Layers.py#L8789),
[checkpoint migration](../bin/Layers.py#L8863).

Older checkpoints lack all these columns. Explicit provisioned/user TruthSet
origins may retain fact status; unclassified conversation rows become
unverified. Old nonzero slots provide a best-effort presence mask. A legacy
relation with no recoverable VP is unverified, rather than certified from NP1.
Old missing scope/source text is not reconstructed from model activation.
[Legacy migration](../bin/Layers.py#L8863).

## Testimony and predictions

Registered addressees declare whether a result is testimony, estimate, question
or observation. ARMA is an estimate. Estimate values cannot become truth
intervals or accepted facts, even when scalar and highly trusted. Tensor
content, unparsed text and non-finite values do not become a positive truth
assertion by conversion failure. Accepted numeric testimony about an Exist
description retains the complete description, scope and named source.
The legacy Part testimony/materialization adapter has no grammatical VP;
those rows remain `unverified` for Exist until the structured-query migration.
[Evidence kind](../bin/thinking.py#L85),
[registration](../bin/thinking.py#L325),
[admission](../bin/thinking.py#L355).
[Legacy materialization](../bin/reasoning.py#L507).

## Gradients

The meaning-value clone preserves live representation gradients, while durable
store writes detach them. Fact lookup, metadata matching, thresholding and
support aggregation are hard operations with scalar evidence; they provide no
ordinary derivative through their choices. Later learned query selection needs
its declared policy credit. This migration adds no learned parameters or new
loss, and does not itself train useful questioning. The architecture-wide
gradient budget remains documented in [GradientFlow](GradientFlow.md).
[Owned value](../bin/Meaning.py#L67),
[detached write](../bin/Layers.py#L8925),
[hard lookup](../bin/reasoning.py#L156).

## Validation

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
above use the installed runtime and test files. Grammatical routing, ordinary thought history, nested retention and
residual policy credit remain open. Taxonomy evidence is documented separately
in [Taxonomy queries](TaxonomyQueries.md).
