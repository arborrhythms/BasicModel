# Selected linguistic meaning and one controller

Item 1 implementation, September 20, 2026. Receipt details live in
[Testing](Testing.md#selected-meaning-and-one-controller-september-20).

## Ownership and execution

`AnswerProgram` owns the selected sentence's signed full-width leaves,
native concept identities, WORD rows, lexical provenance and compose actions.
`LanguageSpace.program_meaning` is pure: it recovers a selected semantic fold
or applies learned lexical alignment, without a reader execution or memory
write. An explicit semantic fold takes precedence. Capture retains a hard
meaning decision, including unknown, so later training cannot reinterpret an
earlier sentence. Technical anchors remain grammar provenance; natural words
have no predefined operator meaning.

`ConceptualMeaning.constituents` retains complete nested role triples and
local references. `TernaryTruthStore.bind_constituents` validates references,
depth, cycles and total capacity before writing children in postorder. Children
become questions or zero-trust unverified occurrences, never facts merely
because the containing sentence was observed or admitted. Eager observation,
pending-boundary drain and packed-sentence drain use this same owner. Durable
records detach; live selected roles preserve their current-step gradient.

`BasicModel.run_selected_thought` is the only controller. The grammar's
`<thought>` catalogue defines its executable menu. `reason_about`,
`think_about`, `answer_query` and normal answer resolution all enter the
completed-row boundary guard; sentence forward/reverse phases cannot execute
thoughts. Text is understood once, and serving reuses the captured value.
`ThinkingKernel`, its frames/addressees/Testimony/next-op policy and the entire
`NeuralToolUser` class are deleted. The old What parity selector is not
constructed, called or restored. The separate `reason_predict_next` blend and scorer are also deleted. The
reviewed numerical bridge-loss experiment remains an independent function
with no recurrent query entry.

Each episode has one `WhatInteractionMemory` owner and one `QueryWorkBudget`.
Operation choices, references, payloads, evidence scans, traversal and children
all pay the same meter. `what(Q)` pushes a child context in this controller;
the child can choose repeatedly, and its typed return is recorded as the causal
source of the parent's conclusion. Exhaustion preserves acquired evidence and
permits only the active-depth return drain plus one root finish. Exceptional
execution also drains open contexts and discards incomplete policy records.

## Answers, credit and restoration

Typed adapters preserve truth roles, detached predictions, every ordered set
member, checked code atoms/references and nested typed child results. An
unavailable payload cannot become a zero-valued answer. `resolveAnswer` freezes
the resulting derivation; `reverseOutput` uses it without another checked read.
Generated set members share the row's finite word budget and report truncation.

There is one thought hard-choice credit contract, documented in
[GradientFlow](GradientFlow.md). The chooser sees full root/active/candidate
schemas (`9D + 17` inputs, including masks and action/evidence fields), with
root and active values live and hard candidate values detached. Native IDs and
addresses are dispatch metadata. Each eligible supplied-answer row earns
`-answer_error - 0.01 * actual_shared_work`, with one EMA baseline. This term
is added to the actual `runBatch` total. Legacy thinking weights migrate by
maximum into `selectedThoughtPolicyWeight`, enabling that objective once.

Ordinary live values survive until the sole optimizer boundary; checked
results detach immediately. History checkpoints recursively preserve typed
`ThoughtResult` children and restore detached data, row-local references and
spent budget. Retired controller and separate prediction-scorer weights are discarded. Lazy saved language,
controller and synthesis topology is restored before strict key validation;
unused synthesis layers are not manufactured during loading.

## Learned natural wording

`model.configure_meaning_learning(word_rows, word_values, hidden=48)` installs
an optional language parameter block from the existing canonical binary
operator catalogue and owned WORD vocabulary. Only full-width word/role
payloads enter its networks. Native identities label operand supervision and
copy selections, never numerical features. The block learns operation and
mode classification, canonical operand pointers, and a conditional GRU
word/copy decoder. Unknown or low-confidence lexical decisions decline to
supply a meaning. Compose and generation have a 128-word bound.

`runBatch(..., meaning_supervision=((meaning, realization_program), ...))`
accepts one annotation per completed row. A `None` meaning supplies an unknown
label; a `None` realization omits decoder supervision. Targets are consulted
only after output, and `linguistic_meaning` is included in the trained total.
Language parameters join the real optimizer once, including when configured
after optimizer construction, and restore from the ordinary checkpoint.

The deterministic learning test trains 36 annotated examples over six noun
pairs, including converse/paraphrase, equality and negative controls. Held-out
“a bicycle has a wheel” starts unknown and learns the same canonical `part`
identity and role orientation as “a wheel is part of a bicycle.” “Contains”
and another held-out noun pair preserve that identity across all three writers.
“A bicycle owns a wheel” and possessive “a person has a bicycle” remain unknown.
The checked taxonomy thought consumes the learned identity, generation emits
“a bicycle has a wheel,” and recomposition recovers the same roles. No natural
spelling is added to `<Anchors>`. A separate real-model test verifies training
total, optimizer membership, parameter change and strict checkpoint restore.

These are the item 1 gates, not a claim of general language or multistep
reasoning utility. The trained lexical block currently covers complete binary
concept operands; structural composition handles explicit unary/nested forms.
Item 2 retains expectation/residual credit, item 3 retains the generation
catalogue migration and broader output learning, and item 4 retains causal
held-out utility, reconstruction controls and throughput evidence.

FineWeb supplies no answer labels: this chooser receives no reward there
until item 2 adds residual credit. The full multi-seed plan §4 utility study
remains item 4. See the [61-test migration table](KernelRetirement.md).
