# Expectation across the architecture

Item 4 follows [the October 19 rev 5 specification](specs/2026-10-08-expectation-at-every-level.md).
`SymbolExpectation` owns one activation predictor shared by the word, composed
symbol, closing and document reads. It predicts independent for/against
activations over native identity rows. It has no byte, ARMA, root-only or
slot-structured prediction head. The existing three-role memory envelope is
retains a representative identity; the full symbol bank is the forecast.

The content gain is `m * (1-g*κ + g*κ*s)`, where `s` is the saturated sum of
absolute lane residuals divided by expected lane mass plus epsilon. The
entire saturation is detached, including its denominator. Each symbol keeps
one gain over both evidence lanes and retains its raw identity. Expected
activation separately primes selection. It is never added to comprehension.
Perceptual mean and variance supply per-coordinate salience and a scalar
region gain; codebook lookup reads the original coordinates.

A bank identity requires raw-key distance at most `1e-5`. An unknown composed
value passes whole and contributes no target to its nearest bank key. The STM
keeps the raw identity and original projection coefficient. Compose scores a
separate gained activation view, including operators that read a coefficient
from a norm; its selected candidate retains the raw identity. Order-zero
poles receive P at perception and S once at their symbol consumer. Repeated
reads cannot compound S. The sentence prime and S read are cleared before
closing thoughts and before the next peer-word input.

The local residual has excess and omission signs for each evidence lane.
It trains only the predictor: both target and all source context are detached.
The row and document gains compose multiplicatively when thought reads the
closing. A cold document has no document attenuation. Open grammatical
objects keep their expected prime and are spared attenuation. Only explicitly
`non`-withheld dimensions lose the prime. The raw observation, prior and content remain available to
awareness even when the content gain is zero. An omission can support an
explicit checked `not` inference; it is never an external observation.

## One pair, one observation

`TernaryTruthStore` is the single durable owner. A warm external boundary writes
an `estimate` row before its `observation` (or interrogative `question`) row and
links their stable occurrences in both directions. Only the observation enters
the chronological predictor context. Provisioned truths and internal thoughts
are not additional external observations. Estimate rows are never facts or their
own corroboration, and ordinary LTM evidence readers exclude them.

The estimate retains the complete bank: native identity, order, raw key and
two activation logits for every symbol. Checked bindings and scope come from
preceding source occurrences. They never copy the incoming target. Provenance
records those occurrences, the external stream/document and the intended
observation. The observation retains its own raw carrier and metadata. When
the durable row fuses several observed symbols, its
`_observed_symbol_activations` metadata retains the original two-lane target,
aligned by the prior's native IDs. Recomputing that target by quantizing the
fused description would change the measured residual. Checkpoint replay uses
the retained target, including negative evidence, rather than doing so.

`expectation_pair(gain=..., object_mask=...)` reconstructs the prior and derives
content, gain, expected activation, excess and omission. There is no third
durable delta or conceived row. Restoration reads the exact stored observation,
so it also works at zero content gain. A new pair requires capacity for both rows and fails before publishing either
when forgetting is required. Reads resolve stable occurrences through the store index,
including after compaction; they do not scan unrelated rows.

For current pairs, surprise is the detached mean squared activation residual
over the row's attended support, bounded by `s/(1+s)`, independent of bank size,
content norm and gain. Local prediction
training uses bounded activation MSE. Pre-Item-4 retained role estimates migrate once at load into an explicit
symbol bank; their historical surprise is preserved. Unknown surprise remains
`-1`. Item 5's forgetting formula, pending-source protection, stable links,
compaction and conservative concept reclamation are unchanged.

The existing semantic sidecar fingerprints links with scope and source text,
checks both directions on restore, and participates in origin-compaction
reachability. A tensor-only restore leaves rows with missing required sidecar
metadata unavailable. No second memory owner is introduced.

## Local prediction credit

Prediction residuals train only the shared activation head. There is no
anticipatory-query reward or separate expectation policy. Reading the object
mask of a nested question follows its owned occurrence on the controller's
ordinary work meter. The held prior remains immutable when its local loss is
replayed against a newly arrived observation.

Source encodings, observed targets, thought effects and durable history are
detached. Only the shared predictor receives local E gradients. Both the
paired-trial keep rule and chooser credit use `R + A`; E is independent.
Mechanism checks and learning measurements have separate receipts in the
[Item 4 record](plans/2026-10-19-item4-start.md). Carried Item 0 findings and
5.5 learning gates are not cleared by these checks.

## Migration and regression dispositions

`architecture.prediction` is rejected as a retired setting. The old comprehension projection (`InterSentenceLayer.cast`/`prime`),
serial `_c_prior`, parallel seed slab, per-source residual fire flag and its
arm/fire APIs are gone. Old `cast.*` checkpoint tensors are discarded at the
owning discourse loader; there is no replacement input projection. Source-row
batch sizing now has an explicit host integer; sentence completion and SVO
reset retain their own existing owners.

The previous attention context receives zero-initialized columns for the bank
prime and normalized percept salience. Percept moments update only during
training on external inputs and are checkpointed; an
older checkpoint starts them cold. Old slot/ARMA head tensors migrate explicitly
to a fresh shared activation predictor, while retained observations and their
semantic sidecar survive. Transient pending graphs do
not survive checkpoint restore.

The relevant regression contracts are causal predictions, detached ownership,
identity preservation at zero gain, full-bank pair restoration, exactly-once
external observations, document/row isolation, local E training, R+A selection,
and the actual probability correction for the exploration mixture. Historical
ARMA, vector-subtraction and slot-prediction assertions are superseded by these
contracts; their retirement is not evidence that a learning gate passed.

All four gain strengths default to zero in both canonical configurations. A
nonzero operating point must first pass the native zero-gain parity gate.
