# Expectation across the architecture

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record. Record: [two truths §1, "The row, field by field"](specs/2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided); [bind plan §6–§6.3](plans/2026-10-19-bind-thought-operation.md); [occurrence spec §13–§14](specs/2026-09-30-occurrence-tense-aspect.md); [expectation spec rev 5](specs/2026-10-08-expectation-at-every-level.md).
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Rows: an LTM1 row is **one** concept slot; an LTM3 row is **three symbol references** (argument, kind, argument) of kind definition, attribution or conditional. The three-slot relation row with a REL code in the middle, null outer vectors, `refs`, `rel_type`, `role_mask`, `order`, `polarity`, `grammatical_mode`, `row_ids`, the address columns and the semantic sidecar are superseded: a row holds its slots, `.when` (model time, a column composition never reads), `truth_plus`/`truth_minus` (the lanes, renamed from `c_plus`/`c_minus`), the access count (renamed from `witness_count`) and `surprise`, and nothing else.
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.
> - `.where` is inside the concept's reserved band (no column); **symbols have `.where` and `.when`**, concepts keep the extent and never the clock; `.when` is model time, sinusoidally encoded; the document and the stream are **context wholes** in WholeSpace, so "sentence *k* of document *d*", source address hashes, timestamps and situation wholes are superseded.
> - Expectation is a **gain field** `c = m⊙a⊙o` with the residual in two lanes, never a subtraction or a sign-reversed image; per level with a local error (no inversion across levels — the invertible path is item 3's); one activation predictor over the symbol bank (the slot-structured `SentenceExpectation`, `BracketExpectation`, the byte level and the ARMA heads are retired); keep and credit both read `R + A`; defaults gains 0, prime on, `φ = .5`.


Item 4 follows [the October 19 rev 5 specification](specs/2026-10-08-expectation-at-every-level.md).
`SymbolExpectation` owns one activation predictor shared by the word, composed
symbol, closing and document reads. It predicts independent for/against
activations over native identity rows. It has no byte, ARMA, root-only or
slot-structured prediction head. The existing three-role memory envelope
retains a representative identity; the full symbol bank is the forecast.

The content gain is `m * (1-g*κ + g*κ*s)`, where `s` is the saturated sum of
absolute lane residuals divided by expected lane mass plus epsilon. The
entire saturation is detached, including its denominator. Each symbol keeps
one gain over both evidence lanes and retains its raw identity. Expected
activation separately primes selection. It is never added to comprehension.
The shipped defaults are four zero gains, `expectationPrime=true` and
`explorationUniformFraction=0.5` (Alec's B3 decision). The sole
[§9.1 parity configuration](../test/fixtures/item4-parity.json) keeps
prediction enabled, turns off the compose and candidate-attention selection
prime and restores uniform eligible alternatives (`φ=1`). It is a gate
fixture; zero attenuation alone does not promise the previous selection law.
Reference variants start with neutral candidate-value columns and reference
projection in the shared chooser. The enabled prime still biases their
selection, but its logit span is bounded by `log(1.1)` in binding menus.
With neutral initial scores this keeps each conditional probability within
`[1/1.1, 1.1]` of uniform, for any retained set. This calibrates the fixed
prior; learned scores remain unrestricted and both neutral paths receive
ordinary gradients. Other menus keep their existing prime scale.
Perceptual mean and variance supply per-coordinate salience and a scalar
region gain; codebook lookup reads the original coordinates.
Closing confidence κ is per symbol, the maximum of its two predicted
activations. It is never taken from the representative envelope's role slot.
Word and S reads use κ=1: their predicted lane magnitudes supply the local
certainty, without a second confidence gate. Closing reads retain the
explicit presence-confidence gate required by repair G.

A native handle identifies an activated symbol in its bank address domain.
Without a handle, a bank identity requires raw-key distance at most `1e-5`.
An unknown composed value passes whole and contributes no target to its nearest bank key. The STM
keeps the raw identity and original projection coefficient. Compose scores a
separate gained activation view, including operators that read a coefficient
from a norm; its selected candidate retains the raw identity. Order-zero
poles receive P at perception and S once at their symbol consumer. The S read
uses the actual for/against lanes, including purely negative evidence. Repeated
reads cannot compound S. The sentence prime and S read are cleared before
closing thoughts and before the next peer-word input.
P's gain affects the incoming content, but the S residual and its gain both
compare the original lane pair with the S prior. P cannot change that local
observation or manufacture an omission for S.

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

The estimate retains native identity, order and two activation logits for
every symbol. Full keys are retained for the observation's support and the
representative envelope; absent keys are explicit, never reconstructed from
later mutable weights. Before an observation is linked, a future prior may
retain every key. The 1,024-symbol × 1,032-coordinate measurement is in
[the repair receipt](benchmarks/2026-10-19-item4-repair/README.md).
Checked bindings and scope come from
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
over the soft union of observed and expected support: each symbol's weight is
`max(o, ê)` across its two lanes, and the weighted error is bounded by
`s/(1+s)`. Confident omissions count; empty bank padding does not dilute the
score. Both writers apply the same withheld marks and formula, independent
of content norm and gain. Local prediction
training uses bounded activation MSE. Pre-Item-4 retained role estimates migrate once at load into an explicit
symbol bank with unknown handles and orders (`-1`); their historical surprise is preserved. Unknown surprise remains
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

An order-zero arrival has two local training reads of the shared head: the
word read predicts the next admitted word from its preceding word context;
the S read predicts the symbolic field before the ensuing composition and
scores the resulting field. These distinct causal contexts and targets are
intentional. They do not apply the content gain twice. Each local loss keeps
its existing declared weight; neither enters the R+A trial judge.

Source encodings, observed targets, thought effects and durable history are
detached. Only the shared predictor receives local E gradients. Both the
paired-trial keep rule and chooser credit use `R + A`; E is independent.
This includes thought departures: keeping, credit and the audit use the
same sum, even when R and A disagree. Work remains metered and reported,
without an additional work-price term in the judge.
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
prime and normalized percept salience, and loses the retired structured
forecast slot. The expectation-free proposal view has zero prime and cold
percept salience (`z=1`), with only the open request and gist as needs.
Percept moments update only during
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
The selection prime and exploration fraction `φ=.5` are separate from these
gains and remain enabled by Alec's B3 decision. The original close-out's
enabled-default/parity comparison is recorded in its receipt. The later
[outstanding-gates round](benchmarks/2026-10-19-item4-step0/README.md) compares
the declared all-gains-.5 operating point with all gains zero, both prime on
and φ=.5; it also repairs initialization and calibrates the binding prime.
Its candidate review and single landing sweep are separate from the earlier
close-out and remain required. All 16 standing XOR cases pass in both arms;
the one-successor decomposition certificate still fails in both. The nonzero
configuration is therefore a measured candidate, not an accepted new default.
