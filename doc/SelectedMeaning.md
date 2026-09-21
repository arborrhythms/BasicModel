# Selected linguistic meaning and one controller

**Item 1's working language gate is met on the bounded supervised curriculum**
described below. This establishes grammar-owned wording in that setting;
general English and useful learned questioning are not established.
Structural-preference and routing-share measurements remain design goals.
The separate codec in `288b56b`
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

Word → operator associations use the existing compose MLP. Its binary first
hidden layer receives full-width operand/category means and signed differences.
The copy/wait scores also receive neighboring operand differences through the
same hidden layer. A marker's isolated copy preference can therefore change
when a suitable neighbor arrives. Both input projections start at zero without
advancing the RNG; old chooser states extend by zeros, while current states
must restore their learned weights. Name-based optimizer restoration preserves
existing moments. Adding the `surface` rule changes the complete grammar's
catalogue and category widths: a checkpoint from the earlier complete grammar
is not an interchangeable training resume. Rule indices are appended, not
renumbered; no claim of a general compose-catalogue migration is made.

The declared binary `surface` operator preserves its complete right-hand
semantic subtree, including mode, polarity and scope. Its numerical carrier is
`W(marker) + content`, so outer grammar choices can still read the marker. A
small MLP **inside that operator** learns the marker for free generation; the
other child is the residual. Recomposition is exact by construction. Tied input
reconstruction instead uses only the recorded occurrence's operand witness.
The existing `preposition` projection can discard a semantically neutral marker.
Neither operator reads spelling, selects a thought operation, or accesses LTM.
Meaning recovery projects only wrappers present in the actual selected tree;
it never guesses a head from arbitrary numerical `lift`/`verb` outputs.

No additional LM is introduced. An optional future LM must be an explicit
selectable grammar operator with ordinary configuration and training. Technical
anchor spellings remain provenance only; natural relations have no predefined
word → operator table. The shared-table architecture uses the first-stage concept allocator for both
the thought registry and retained WORD/OBJECT identities. Configurations with
separate stage dictionaries keep their terminal registry. All thought readers
use that registry's owner. A taxonomy neighbor without an allocated payload is
reported as an unavailable reference with incomplete evidence; reading it never
allocates a row or fabricates an answer value.

The optional `grammar` dataset supplies text and separate structural/output
annotations. `GrammarLessons` has no parameters or inference entry point. It
scores teacher states only after the student's own input program and output
have been fixed. Explicitly annotated operand variables receive full-code
augmentation in teacher states; these do not replace captured student leaves.
Compose cross entropy trains the existing unary/binary chooser and matches the
live copy/reduce occupancy threshold. Separately annotated output trees train
the existing generate chooser and selected numerical operators. Desired output
words must already belong to the ordinary input vocabulary. The shipped
[MM_grammar_wording.xml](../data/MM_grammar_wording.xml) selects this curriculum;
`grammarLessonWeight` adds its loss to the actual `runBatch` total. Unlabelled
corpora and evaluation splits supply no grammar lessons. Shared weights remain
trained through their own uses under the [separate-state contract](GradientFlow.md).

Declarative relation answers retain all three canonical roles rather than a
lossy numerical root. The normal generate walk chooses its own actions over
those roles. Only **after** it emits word concepts does the lexical inverse
compare them with all known WORD/OBJECT rows that own spellings, using tiled
full-width cosine comparisons. It has no relation templates, function-word
whitelist, current-input vocabulary restriction or fallback meaning decoder.
`AnswerConstruction.texts` and `WhatAnswer.text` expose that generated wording.
The ordinary perceptual/numeric output remains available. Independent numerical
generation ownership remains item 3.

`ConceptualMeaning.constituents` owns complete nested role triples and local
references. `TernaryTruthStore.bind_constituents` validates references, depth,
cycles and capacity before writing children in postorder. Children become
questions or unverified occurrences, never facts by containment. Eager,
pending and packed observation share this owner. Durable records detach;
composed requests may remain live in the episode; checked thought effects detach.

`BasicModel.run_selected_thought` is the only thought controller. Its menu
comes from `<thought>`. All public reasoning entry points and normal answer
resolution share the completed-row boundary, `WhatInteractionMemory` and one
`QueryWorkBudget`. A plain request can select `what(Q)` using its already-owned
begin/descent occurrence. No speculative LTM row is needed to offer it.
Children run in the same controller; their typed returns are causal sources
of parent conclusions. Cutoff permits only the bounded return/finish drain.

The frame kernel, addressees, testimony, `NeuralToolUser`, `TruthInterval`,
`WhatStepChooser`, the separate prediction scorer, geometric `legacy_*`
readers, whole-meaning nearest-word realisers and soft bridge-policy experiment
are removed. The lexical inverse of already generated word concepts is a
different boundary, described above.
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

Each context read charges the shared meter for at most four STM/history/knowing
records, four recent discourse ideas and four frames already brought into
serial context by `what`. Small budgets reduce these limits to reserve work
for execution. A recent LTM write alone supplies no chooser content. Indexed
retrieval admits shared facts and this row's stream; other streams' observations
and estimate/question rows cannot become retrieved frames. See
[AccessibleMind](AccessibleMind.md) for cue ranking and index ownership.
Attention reads the active meaning and detached memory values; its final chooser features detach. It has no
answer target or additional learned head. `ThoughtFeatures.context_width`
owns the input dimension (`15D + 3509`, plus two action flags).

Zero final weights preserve the deterministic execute/conclude baseline.
Candidate values, checked results and root/active chooser observations detach. Output also cuts state at the concluded idea; shared generation operators remain trainable.
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
The replacement includes normal `runBatch` checks and a reproducible CPU
language run, seed 931. All 1,599 training sentences pass through the real
forward path before their detached leaves supply teacher states. The focused
run trains the existing generate chooser/operator for 1,000 steps, then the
compose MLP for 8,000; it holds the input encoding fixed and disables input
reconstruction during this isolated language study. Ordinary reconstruction,
loss assembly, optimizer updates and the shipped XML/loader are checked
separately. This is supervised structural learning, with explicitly annotated
operand-code augmentation, not a claim of unsupervised or full end-to-end
production convergence.

The 32 development cases were used while refining the architecture. An
additional 24 cases then introduced three unseen complete relation wordings
(`also are equal to`, `are also equal to`, `also are part of`) and eight unseen
nouns in both operand orders. These are new combinations of learned marker
words, not a claim to infer an unknown relation lexeme without evidence.
The fresh run gets **56/56** comprehension/control cases, **53/53** generated
relation sentences and **53/53** reparsed meaning matches. The three alternate
uses of `have` do not acquire a false part relation. Generation chooses from
all 178 observed word spellings; its function vocabulary is not restricted.
No held-out annotation enters parsing or generation. Metadata recomposition
checks role identity/order, mode, polarity, bindings and scope. The separate
wrapper check also preserves the complete role values.

[Testing](Testing.md#working-grammar-wording-gate-september-20) records commands,
development failures, the normal training smoke, the explicit slow language
run and the full default receipt. The slow case is retained for reproduction;
the preference measurements below need not become artificial passing tests.

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
