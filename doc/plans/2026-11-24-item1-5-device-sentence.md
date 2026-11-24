# Item 1.5 proposal for a device-resident sentence

**Approved representation.** The chooser writes its chosen syntax directly into each candidate’s STM. Downstream model code takes one head and follows the existing idea and relation references. This supersedes the proposed device implementation of `ReadingIndex`. The retained STM head passes through MTM. MTM stages recent percepts, concepts and event operands. Eligible content transfers to LTM with one owner at a time, consolidating into LTM0 wherever possible.

## Sentence boundary

Staging may admit input bytes, encode static grammar rule names and allocate capacity before execution. From wide-open attention through the single whole-field reconstruction, narrowed looks, candidate scoring, selected backward, optimizer step and retained close, decisions and values stay on the device. The read benchmark should cover the entire execution entry, including its called backward and close. Post-sentence reporting and checkpoint serialization may copy results out. Any pre-existing postprocessing that participates in a decision or optimizer update must be inside the measured boundary.

An earlier strict capture probe on the ordinary four-page reading update stopped in `InputSpace._lex_batch` when it turns a tensor into a NumPy array and calls `tobytes()`. Where a host source exists, input admission can carry its actual encoded, clipped bytes forward with the staged slab. Inputs that originate on the accelerator still need a device-native byte path. Preserve ASCII substitution, byte limits, embedded NUL handling, packed-row ownership and provenance. Count admission time in end-to-end measurements. Treat any content, grammar or fixation decision made after admission as sentence work.

The sentence crosses the input boundary as a byte tensor with its valid extent. Single bytes are the initial parts. The model chooses fixations, assembles parts into learned spellings and parses from its observations. Input admission supplies no word lookup or word boundaries. At sentence close a retained spelling can be read back through its ordered parts. The current strict capture probe reaches the learned-span conversion in the lexical stem.

## The STM head is the parse

**At any model time, STM and LTM each hold at most one root idea or relation per stream.** Each batch row takes its time from its own ModelState. The value retained for that stream at that time is the root. For example, if the value at time t is the relation `pair(A, B)`, the entry at t is that relation. Following it reaches A and B. Neither operand acquires a separate entry at t. If the chooser reduces the whole expression to idea J, J alone occupies that position. The same rule applies during an unfinished parse and after retention.

Alec specified that each sequence link addresses its own idea and the next link, that reductions build STM1 over their operands or produce an STM0 idea, and that everything entering LTM has one head. The working sequence can hold up to eight things under the 1.7 specification. Respect the configured working capacity and existing behavior for longer inputs. [Item 1.7, §19.146, point 1](2026-11-14-sigma-pi-layers.md#19146-alec-the-chain-in-the-two-memories-the-single-head-the-looks-of-a-model-with-no-grammar-momentum-stays-2026-11-23)

**What exists.** `SentenceStrata.Tree.write_edges` already writes located word occurrences, prefixes each application with the chosen operator symbol, orders the operands with `store.parts.sequence`, and returns the head through `trace.head`. The inverse’s `SentenceField.said_words` already follows this head for located words. The gap is how the structure gets there. `Models` builds a `ReadingIndex` from recorded actions. `write_edges` then walks that Python index to manufacture the relations. Other consumers still use its action numbers, parents, forest and operand arrays.

**How the chooser would write it.** In the examples below, a pair is one STM1 row with two typed references. Each name for a word denotes a particular located STM0 occurrence.

1. Admit a leaf from the observed bytes at the fixation the model chose. A raw sequence of A, B, C is `pair(A, pair(B, C))`. The final occurrence itself terminates the existing sequence encoding. Identical word symbols at different places have distinct occurrence handles. A model with no grammar can retain this chain directly.
2. When the chooser takes a binary operation on heads L and R, the operand head is `pair(L, R)`. The application head is `pair(operator_symbol, operand_head)`. A unary application is `pair(operator_symbol, L)`. Bind links the operand it actually consumes and retains the bound reference used by the computation. Use the returned handles immediately. During construction, the current STM root reaches the ordered operands still available to the chooser. A reduction updates that root. The numeric result is the live idea computed by that operation, with its existing gradient lifetime.
3. A completed symbolic expression has one application head. A completed idea may itself be the head. Where learning or exact inversion still needs the derivation, keep its ordinary STM references reachable for that lifetime. A partial reduction exposes one chain head over the remaining constituents, in order. This preserves partial and silent readings without inventing an operator to join the roots. On promotion, translate the winning reachable references into LTM in the existing close order and give the consumer the translated head.
4. Follow the head to find operands, constituent words and their argument positions. Use ordinary temporary traversal state. A shared occurrence reached twice is visited in each operand position. Replace `ReadingIndex`, `recorded_steps`, and the stored Python forest as model interfaces. Any old-action adapter needed for historical data belongs at load; a comparison of the former index with the emitted relations belongs in tests.

**Worked example.** In the XOR grammar, let A and B be distinct word occurrences. One schedule admits A, applies `interpret` to A, admits B and applies `conjunction`. Another admits A and B, applies `interpret` to A in the older occupied seat, then applies `conjunction`. With the two schedules supplied to the current `ReadingIndex`, interface validator and STM edge writer, both yielded exactly `pair(conjunction, pair(pair(interpret, A), B))`. There is one parse here. The time at which B was admitted is not syntax. Neither schedule was sampled from a trained chooser. Interpreting B instead gives `pair(conjunction, pair(A, pair(interpret, B)))`, which is visibly different.

**What must be recoverable.** The head needs the exact chosen operator, its ordered operands and distinct leaf occurrences. Declared operators must have an unambiguous execution interface. Check overloaded names against the existing catalog’s canonical rule, kernel and formal argument concepts; where the name alone is ambiguous, reference the specific declared operation through STM. If a chosen operation such as a transposition is itself taught or inverted, preserve its symbol and operand links, including which declared output is used for a multi-output operation. The current edge writer skips transpositions. Mere changes to evaluation order need no parse entry. No node needs persistent `kind`, `local`, `word`, action number or parent columns.

**What the loss uses.** For example, `SentenceCredit.forward_discard` currently indexes the actual operand and result tensors by action number. Bind that loss to the chosen application’s live operands and result when it is taken, or follow its STM references while those values are live. Preserve any sampled probability or departure comparison that the policy objective actually consumes at the choice. Test the scalar losses and gradients on mixed winners. These quantities are transient training values. The committed parse consists of the reachable ideas and relations.

Each trial owns its new STM rows. The opening can be shared. Each forecast sees only the observations already made by its own path. The single whole-field reconstruction and the existing clock cadence are preserved.

Word expectation advances separately as each trial observes a word. For example, an exploration that opens on B scores B before observing A, even if the greedy walk reached B after A. Scoring an output-only departure can reuse the greedy reading because all arrivals are identical. A further optimization would share the forecasts up to the first differing observation, then batch the suffixes. Reuse requires identical causal inputs, including arrival order, active rows and evidence lanes. The same target word at the same source position is insufficient.

A morphological basic percept is defined by its parts and wholes. The existing percept code assembles their for and against evidence and reads it through Pi. Spelling across the sentence boundary uses ordered observed bytes. The centroid and its associated evidence must still come from the percept definition, including learned parts and the evidence supplied by their holders.

The input can recover parts directly from observed atoms and their locations.
A learned part can occur at any byte offset, including inside an unfamiliar
whole. The spatial aperture selects parts within it and wholes enclosing it.
It does not impose a common starting byte.

For reconstruction, exclude the current occurrence when choosing the next
part. Use the largest compatible overlap that preserves the bytes already
placed and makes the resulting assembly strictly longer. From `hel`, the
part `ello` shares `el` and `llo` shares `l`. Either yields `hello`.
Where locations are given, distinct occurrences retain their places. The
single reconstruction follows the opening attention pass.

### Proposed experiment in frequency and non-overlapping parts

Alec proposed using frequent parts for a quick reading. One ordered,
non-overlapping cover of the observed bytes would suffice for concatenation.
Frequency would guide the usual cover. During the presentation a later look can address the
observed bytes. If reanalysis is to work after the presentation, each kept
part must also expand to its ordered elements. A rare substring crossing
an earlier cut can then be proposed before it has a frequency count.

Use `sho | pping` as the first cover of `shopping`, with `shop` absent from
the inventory. On a later look the model should be able to attend to the
first four bytes across that cut and propose their ordered content. A kept
observation can teach `shop` under the existing learning rules. A later
reading may then use `shop | ping`. Both covers preserve both occurrences
of `p`. A discarded exploration must leave the shared inventory untouched.

In a fresh inventory, explicitly admitting the first four observed bytes
added `shop` to `shopping` while preserving `sho` and `pping`. This checks
admission. A separate MTM test supplies the aperture and verifies that the
production nomination code proposes the four observed bytes even when the
cover cuts after `sho`. Prefixing a space shifts the aperture by one and gives
the same candidate. The current chooser scores existing reading brackets,
and the native nomination code consumes complete brackets. A test of discovery must
let the model produce the subspan without supplying `shop` or its endpoints.
Compare training sentences per second and time to the same restored reading
accuracy before adopting this proposal.

## Copy the retained head into LTM

Each stream has at most one root per tick. Four pages read together can each close with a different root at the same tick. Index ModelState by batch row to select the closing stream and its time. The current implementation broadcasts a scalar clock across rows. An idea root resolves to an LTM0 row. A relation root resolves to an LTM1 ordered pair whose endpoints retain their reference types. STM0 ideas copy to LTM0 and STM1 links copy to LTM1 under the same logical structure.

LTM0 holds fixed-width idea vectors in a dense matrix. For a batch of compatible cue vectors Q and eligible idea rows V, `Q @ V.T` produces one content score per cue and idea. Use the lookup’s existing eligibility, normalization and scoring rules. LTM1 holds the links for traversing a relation in order. If t holds the relation `pair(A, B)`, A and B can participate in idea lookup through their LTM0 rows while the sole root at t addresses the pair in LTM1.

Reduction belongs to the chooser in STM. The winning head may denote the original operator-prefixed expression, an expression with reduced idea operands, or a single reduced idea. All three use the same commit. For example, if STM has already reduced `interpret(A)` to idea I, it may hand commit `pair(conjunction, pair(I, B))`. If the entire expression has become idea J, it may hand commit J. The copy takes the head it is given. It does no grammar selection, replay or further reduction.

Commit the winning STM root as the sole LTM value for that time. Preserve the structure it owns, copying underlying storage where necessary and reusing existing durable references. Copying a child preserves its place in the structure. Only the root is entered for that time. Unfolding the translated LTM head must encounter the same symbols, idea values and pair structure as unfolding the supplied STM head. Preserve each occurrence’s existing location and time. Destination-local row names may differ. Translate references to copied rows consistently, including references held by ideas; existing durable references and shared symbols keep their identities. LTM owns its tensors after STM expires. The copy itself cannot reconstruct any words or operators that reduction has already discarded. An inverse that needs them must use the retained representation and pass its existing behavioral gate.

`ReadingMemory.commit` now copies the reachable ideas with their original stamps, translates occurrence references, and interns reachable pairs in dependency order. Two streams can retain the same durable relation even when all relation seats are occupied. The copy checks the capacity for all streams before publishing the first. Tests cover a single free idea seat, shared children, external references, STM expiry, and checkpoint restore.

The grammar-free and partial-reading paths use that copy. The ordinary grammatical close still writes from its admitted clause. Its recent STM relation head also consumes an occurrence name for a point, and existing checkpoints omit the stream and tick of a relation close. The historical root lookup and the changeover of the ordinary grammatical close are open.

## Percept and concept staging

MTM is the implementation name for the staging area for basic percepts and concepts. STM and LTM share the idea and ordered-relation representation used to retain a sentence. The two mechanisms meet when a word occurrence refers to a concept. Copying that occurrence preserves the reference and leaves concept admission with its existing owner.

Each exploration can observe its own provisional growth. Only the winning stream’s observations are eligible to persist, in the existing close order. Exercise mixed winners against isolated runs from the same opening and compare STM, LTM, clock, RNG, optimizer and gradients. The current `MediumTermMemory` class also exposes event and checkpoint interfaces used by grounding. Preserve their existing lifetimes and compatibility while changing sentence storage. Alec subsequently placed recent noun and event operands in MTM, with transfer to LTM when eligible. Preserve the current teaching and expiry rules. The broader staging-area review and its compile check belong to item 1.4.

## Acceptance and rollout

1. Start with an observed sentence on the current path and record its reachable STM and LTM heads. Then test direct STM emission at each selection. Compare operator identities, exact operand order, distinct occurrences, results, losses, gradients and the inverse. Include a single word, repeated words, a raw chain, unequal lengths, nested applications, bind, transpose, partial reductions and mixed winners. Vary the admitted words independently of padding. Once the consumers pass using the STM head, delete the runtime reader and action replay. Prove independence by exercising the ordinary training and held-out entry points with only the STM form available.
2. At a fixed time t, check STM during construction and LTM after commit and restore. A nested relation must yield exactly one entry at t and both ordered operands through that entry. A fully reduced idea must yield just that idea at t. An empty time may have zero entries. Inspect the temporal records as well as the returned head to catch duplicate same-time writes. Compare the reachable winning STM and restored LTM structures. Verify that concept references retain their identity, only the winning observations affect staged growth and discarded paths leave memory, clock and gradients intact.
3. Exercise the complete ordinary training entry with strict fullgraph capture and instrumented execution. Count explicit reads, native scalar/equality operations, device-to-host copies and data-dependent output sizes. Require zero in-sentence synchronization on the actual accelerator. A captured subroutine alone cannot establish this. On Metal, audit the host-binary64 trust update and the clock-band rotation in `RowTime.how_far_back`. Any device replacement must preserve their numerical contracts.
4. Repeat fresh reading, both verbal XOR boundaries, numeric XOR and raised-class gates. Repeat the matched sentences/s and time-to-accuracy measurements. Do not change acceptance thresholds or force fixations to obtain capture.

The integration points for Claude are the chooser’s STM writes, `SentenceStrata`, `Extension`, `Comprehension`, `WholePaths`, `CandidateGrowth`, `MediumTermMemory` and the reference/grounding writers. Coordinate changes to the staging and grounding interfaces independently of the STM-to-LTM copy.
