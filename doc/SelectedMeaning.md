# Selected linguistic meaning and one controller

Item 1's **architecture and production wiring are complete**. The September 20
clarification keeps concrete language-quality and structural-preference checks
as design/evidence goals at this stage. Neither learned natural-wording quality
nor learned questioning utility is established. The separate codec in `288b56b`
did not satisfy this architecture; its passing regression receipt never proved
otherwise. See [Testing](Testing.md) for receipts and
[KernelRetirement](KernelRetirement.md) for the earlier test dispositions.

## Ownership and execution

`AnswerProgram` owns signed full-width leaves, native identities, WORD rows,
lexical provenance and the forward's actual compose actions.
`LanguageSpace.program_meaning` only recovers that selected derivation. It has
no trainable fallback, surface interpreter or separate decoder. Technical
anchors are grammar provenance; natural words acquire no predefined operator.
BasicModel configures `reverseOutput` to use the declared generate walk and
ordinary perceptual inverse. Other model configurations retain their existing
synthesis compatibility. Input teachers never choose output actions.

Word → operator associations train the existing compose MLP through the ordinary
reconstruction/answer objectives. Its binary first hidden layer now receives
both operand and role means **and signed left/right differences**. Symmetric
candidate values therefore cannot erase word order. The extra input projection
belongs to that same MLP and scalar choice head; it has no separate classifier,
decoder, optimizer or loss. Its zero initialization preserves prior predictions
and initialization order. Older checkpoints add only this zero projection;
name-based optimizer restoration preserves existing moments and leaves the new
parameter fresh. Current checkpoints restore its learned weights and moments.

No additional LM is introduced. Learned numerical work remains inside declared
grammar operators such as `lift`, `verb` and `lower`. An optional future LM must
be an explicit selectable operator with ordinary configuration and training.
The shipped anchor tables contain technical spellings only: the predefined
natural word `equals` is removed. This does not rewrite previously owned traces.

BasicModel enables `outputInLoop` and `outputPolicyWeight=1.0`. The generate
policy receives supplied-answer credit through the existing `runBatch` total;
unlabeled FineWeb supplies no such credit. Reconstruction still trains the
compose MLP and shared numerical operators. Independent generation numerical
ownership remains item 3. These settings do not create semantic labels or
make an unsupported parse into a canonical relation: `program_meaning` still
requires a supported, selected structural form.

`ConceptualMeaning.constituents` owns complete nested role triples and local
references. `TernaryTruthStore.bind_constituents` validates references, depth,
cycles and capacity before writing children in postorder. Children become
questions or unverified occurrences, never facts by containment. Eager,
pending and packed observation share this owner. Durable records detach;
current selected values preserve their ordinary gradient.

`BasicModel.run_selected_thought` is the only thought controller. Its menu
comes from `<thought>`. All public reasoning entry points and normal answer
resolution share the completed-row boundary, `WhatInteractionMemory` and one
`QueryWorkBudget`. A plain request can select `what(Q)` using its already-owned
begin/descent occurrence. No speculative LTM row is needed to offer it.
Children run in the same controller; their typed returns are causal sources
of parent conclusions. Cutoff permits only the bounded return/finish drain.

The frame kernel, addressees, testimony, `NeuralToolUser`, `TruthInterval`,
`WhatStepChooser`, the separate prediction scorer, geometric `legacy_*`
readers, nearest-word realisers and soft bridge-policy experiment are removed.
Nonzero `answerLossWeight` and `predictNextLossWeight` fail configuration.
Their checkpoint parameters are discarded, not reinterpreted as new policies.

## Policy input, credit and restoration

The controller sees full masked root, active and candidate role triples;
mode and polarity for each; ordered binding/scope metadata; level, pressure
and actual evidence; and bounded attended STM/LTM content. The sole chooser
is an MLP. Metadata has no extra language model: two 64-byte categorical
fields per meaning use binary byte categories, occupancy and overflow bits.
Typed native references are alpha-renamed jointly against the canonical roles,
so binding equality survives renaming and allocator magnitudes never enter.
The bounds are a representation limit, not a claim of arbitrary scope support.

Each context read spends the shared meter on at most four STM/history records
and four recent LTM records. Small budgets reserve work for execution. STM and
ordinary history use the active batch row. LTM admits shared accepted facts
and the row's held occurrence references; other rows' private observations and
estimates cannot supply content. The read reports a bounded/incomplete view.
Attention uses the live active meaning and detached memory values. It has no
answer target or additional learned head. `ThoughtFeatures.context_width`
owns the input dimension (`15D + 3509`, plus two action flags).

Zero final weights preserve the deterministic execute/conclude baseline.
Candidate values and checked results detach; root and active values stay live.
Each eligible supplied-answer row earns `-answer_error - 0.01 * actual_work`
with one EMA baseline, added to the real `runBatch` total. Old thinking-weight
aliases select this same objective once. FineWeb has no answer labels, so
residual credit remains item 2. Old incomplete-context policy weights and
optimizer moments reset explicitly; current policies restore strictly.

Typed answers retain truth roles, complete predictions, every set member,
checked code payloads and nested child results. Missing content remains
unavailable. Resolution owns the answer before generation; generation does
not re-execute readers. History checkpoints preserve nested typed results.

## Evidence and retained design goals

The review probes distinguish positive/negative requests, mode, bindings and
scope; test reference-renaming invariance and row-local memory; and fit the
actual chooser to take one or two nested `what(part)` descents. The resulting
MLP choices run without a scripted selector, receive actual episode credit and
retain causal return sources. This is a mechanism test with authored routing
labels. It does not show useful learned decomposition or transitive reasoning
caused by nested questions. Native multi-edge reader proofs are separate tests.

The deleted codec's 36 synthetic examples, noun-only holdout and restricted
function vocabulary are **not accepted language-learning evidence**. Its
natural-wording and optimizer tests are retired with that architecture.
The replacement checks establish a narrower mechanism: a real text batch runs
through normal `runBatch`, retains its actual forward programs, updates the
compose MLP including its order inputs through reconstruction, and invokes the
declared generate walk. The same unlabeled batch leaves the generate policy
unchanged. An equal-candidate test isolates learned sensitivity to operand and
role order. Checkpoint, optimizer and full-graph checks cover that added input.
These are correctness/wiring checks, not a held-out linguistic study.

The retained empirical protocol uses forward-parsed text, holds out **complete
relation wording as well as nouns**, includes converse and alternate-sense
controls, and evaluates generation from the normal vocabulary followed by
recomposition of meaning. Multiword canonical meaning and natural output quality
remain unproven. No fourth interpreter or realiser may supply success.

The preference goal is to use understandable structural operators whenever
they carry the meaning. Any opaque operator competes through the ordinary
grammar MLP, with structural choices preferred at equal fit. Measure the share
of sentences routed through opaque operations, and require it to fall as
structural coverage grows on the same corpus. This landing does not implement
a preference guarantee or report that routing study; those remain explicit
goals rather than synthetic passing tests.

The matched-compute, multi-seed direct-answer/no-subgoal comparison remains
item 4 and gates every claim of learned utility, even when these mechanisms
and the full regression suite pass.
