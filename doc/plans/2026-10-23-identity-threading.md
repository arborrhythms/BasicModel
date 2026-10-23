# Identity threading: MTM values, bind in the grammar, events as occurrences of their subject (Claude, 2026-10-23; for Codex)

**Status:** the identity line of item 3.5 step 2, sequenced by
[step-2 spec §36](2026-10-21-step2-one-ltm.md#36-decided-alec-2026-10-22-identity-first-the-3-5-sequence)
after storage-only 2b (published `39272e7a6`) and before derivation
removal. Codex's brief is [§68](2026-10-21-step2-one-ltm.md#68-codex-identity-3435-against-the-landed-storage-source--2026-10-22);
§3 answers it point by point. Under
[§39](2026-10-21-step2-one-ltm.md#39-standing-rule-amended-alec-2026-10-22-codex-continues-documenting)
Codex reviews this spec, fixes what the code contradicts, and continues,
documenting each decision; only reversing a decision in §1 comes back to
Alec.

## 1. Decided (Alec, 2026-10-22 and 2026-10-23)

1. **MTM stores working content** (step-2 §35). Each identity holds one
   current value; names map to identities.
2. **Thread through identity** (step-2 §34): a verb acts on its subject's
   identity, not on the mention that names it ("if the verb acts on a
   pronoun, that is not the right/complete subject").
3. **bind is a compose-grammar operator** that the chooser calls during
   the parse, "with two locations instead of one": the noun phrase and
   the antecedent's location. The chooser picks operator and location
   together: the menu lists bind once per accessible location, plus the
   dummy. So the phrase is bound before the verb composes with it, and
   "things would be bound by the close".
4. **Value update:** a verb's result replaces the identity's current
   state; a later description mention joins its content into it; the
   introducing noun phrase is the initial value; history stays in LTM0.
5. **Events** (amended 2026-10-23 by [step-2 §71](2026-10-21-step2-one-ltm.md#71-decided-alec-2026-10-23-ideas-in-ltm0-binary-chains-in-ltm1-no-sentence-row)):
   there is no sentence row. A sentence whose words are all existing
   concepts compresses to one LTM0 idea, the VP applied to its subject's
   identity, written as an occurrence in that identity's history; a
   sentence that cannot compress is an LTM1 chain of two-reference rows
   whose top row is the sentence. Role evidence comes from MTM while it
   lasts, and a recalled chain brings its identities back into MTM
   through `query()` (step-2 §72). A mention's reading is kept in MTM as
   a name/value pair, not as a row.
6. **No word is another word's object; nothing is special about "plus
   12".** Alec: "'plus 12' should be an NP and a VP. Why are you treating
   it as special?" (The earlier "keep word-as-object" entry misread a
   cut-off answer and is withdrawn.) Alec, 2026-10-23: "It is an object,
   but we should only know that: VP = VP NP." So "12" is the verb's
   object, the noun phrase in the VP = VP NP rule, and that position is
   all the grammar knows about it: the §10 grammar-only contract. It is
   never the word "plus"'s definition object. The close must never
   define a word as naming another word's id, and the storage round's
   N1 special case goes with that fix (§2, "Definitions").
7. **Order:** identity first, derivation removal after (§36). The
   derivation readers stay until the replacements pass the same gates.

## 2. The mechanism, in code names

**MTM entry.** `MediumTermMemory.Token` gains `value` (a detached
tensor, the identity's current state) and the pool gains `names`, a map
from a mention's occurrence address to its token key, which also holds
the mention's reading, the word or symbol it read to (§72: readings are
name/value pairs in MTM). The token key stays
the introducing occurrence's address. `open` sets `value` to the
introducing phrase's content; a new `update(token, value, *, when)`
replaces it after a verb; `bind` joins a description mention's content
into it (coordinatewise max, as `token_value` joins today). `token_value`
returns the stored value; its union over occurrences becomes a
diagnostic, `history_value`, used by nothing live. `forget`, `merge`,
capacity eviction and the LTM0 reclamation hook (`mtm.forget`) are
unchanged; expiry and promotion stay 2c's.

**bind in the compose grammar.** A new `BindLayer(GrammarLayer)` with
`method_name 'bind'`, unary on the stack (`bind_argout1 =
bind.forward(bind_argin1)`), its antecedent supplied as the action's
reference through the existing reference-expanded menu
(`reference_data['binary_refs']` / `operation_refs`), exactly the
mechanism the determiner variants used before line 3 collapses them: one
action per accessible location, plus the dummy (reference 0). Its forward
returns the identity's current value when a location is chosen, and the
phrase's own content for the dummy (an own object). Accessible locations
are the current frames' tokens (`ReferenceClose.accessible` over
`SituationFrame`s) plus the trial overlay's own new tokens, so a second
mention in one sentence can bind to the first. The journal records the
choice (`ClauseJournal` `formation`, today's `binding_choices` path) so
the close commits it. `definite` and `lower` keep their folds and lose
their reference variants (line 3's decision); a determiner's result is
an NP the chooser may then bind.

**The trial overlay.** Each trial (exploit and explore, `WalkTrials`,
`SentenceFork`) starts from the same MTM snapshot and owns an
`MTMOverlay`: new tokens, value updates and name mappings made during
that trial. Reads go overlay first, then the pool. `ReferenceClose.commit`
applies the kept trial's overlay once, after `write_clause`; a discarded
trial's overlay is dropped. `preview`'s `('new', n)` records become
overlay tokens, so R10's same-sentence alias is the overlay read.

**The verb acts on the identity.** `LiftLayer` and `VerbLayer` are
unchanged: their subject operand is whatever is on the stack, which after
bind is the identity's current value. The close publishes the sentence's
VP result to the subject's token (`update`) in the overlay, before the
next clause reads it; within a sentence the publication happens at the
clause boundary that `ClauseJournal` already marks for companions.

**Events.** `ClauseRows.write_clause` writes the sentence row as today
with `role_refs` for its three slots; after bind, slot 0 and slot 2
references are the bound tokens' keys (addresses) and slot 1 is the verb
symbol. `ReferenceClose.commit` appends the sentence's occurrence to the
subject token's history (`mtm.event(token, occurrence, when)`, a history
entry distinct from a joined mention) and calls `update` with the VP
result. Event order within one close follows the kept operation sequence,
not `RowTime`, which stamps all members of a close alike.

**Role evidence from MTM.** Each token's history records, per event, the
verb symbol and the role the identity filled (subject or object), set by
`mtm.event(token, occurrence, verb, role, when)` at the kept close.
`ReferenceClose._participants`, `expected_roles` and `prediction` read
those histories instead of Part pairs: `expected_roles(predicate)`
counts events of that verb with an object; `prediction(identity,
predicate, role)` is the share of that verb's events in which the
identity filled the role. Evidence lasts while the token does (expiry
and promotion stay 2c's); a recalled chain brings its identities and
their events back into MTM through `query()`. G3 keeps its assertions
within that lifetime.

**Equality substitution through identity.** "x is three" is an equality
event whose slot 0 token (x) gets `update(value of three)` by decision 4,
so a later "x plus two" composes with x's current value and equals
"three plus two" by evaluation. `ThoughtBinding.integrate` keeps its
witnessed-equality edges and, until the removal round, its component
congruence; the identity route is added beside them so both decomposition
arms pass on this source with the readers present.

**The scored snapshot.** `ReferenceReconstruction.cost` compares the
mention's content with the record's value as captured at the bind, in
the overlay record; a later `update` does not change that comparison.
First mentions free, same-close aliases covered, arrivals and thought
episodes excluded, as landed.

**Tensors and gradient.** Overlay values are tensors staged into the
trial's state at the host boundary and passed through the compiled
operation path; no Python pool read inside the compiled body. Within a
sentence, the VP's output is a function of the identity value, so output
prediction's gradient reaches the VP and the bind through the derivative
(Alec: "binds should get credit assignment through the derivative"); the
discrete bind choice keeps its R + A score-function credit. At keep, the
committed value is detached: the graph ends at the kept close.

**Definitions.** How the dual-role word arose: in the joint-objective
corpus ("1 plus 2"), an untrained chooser composes "plus" and "2" under a
headed operation; `PartClosing.prepare` records a word origin for every
planned term by walking to its head word, and `preflight` calls
`_definition(symbol, word)` for each, so the head word "plus" is defined
as naming the phrase's symbol, which for an objectless "2" standing for
itself is the word "2". Three changes, in this line:

1. `Interpret.define` rejects an object that is a word (`'word'` in
   `symbol_types`, or in `definitions.word_ids`): a definition names an
   object, never a word.
2. `PartClosing._definition` skips a symbol that is a word, for the same
   reason, before the mint path is removed with close-time minting in the
   removal round (§11.1), which retires `_word_origins` and `_definition`
   entirely.
3. The N1 special case in `ConceptualSpace.word_surface_for_row` (a word
   with an object spells nothing) is removed; the reader returns to
   `Interpret.lexical_form`. Its regression
   (`test_word_bank_form_survives_use_as_another_words_object`) becomes
   the guard's test: defining a word as another word's object raises, and
   both words keep their own forms.

**Objects that are not concepts, and clauses (Alec, 2026-10-23).** A
sentence is written as a three-slot row whose roles reference other rows
(`ClauseRows.write_clause`: a role holds the child's occurrence address,
or a symbol with a point); nested clauses chain through those references,
as the nested-quote fixture shows (root, "speaker" scope, referencing the
inner row, referencing the quoted leaf). These references are semantic
bindings, kept through derivation removal (step-2 §33, R5); what is
removed is the Part-edge tree of compose operations. So "he said she said
X" is three rows chained by reference, and it stays that way. When a
verb's object is not a concept, as in "x plus 4723587", the object slot
references the object's mention row, with its word reading, and nothing
is compacted into a fused idea; a numeral's value is resolved in thought
by chained successor sentences, not stored. An identity's current value
is updated by a VP whose object is resolved; an unresolved object leaves
the event as an open reference, which thought may fill, as a pending name
is today.

**Lifetime.** `values` and `names` serialize with the pool in
`ConceptualSpace`'s checkpoint extras (the resolver extra is version-only
today); document and stream isolation follow the frames; reclamation of
an LTM0 row still calls `mtm.forget`.

## 3. Answers to §68

1. *Name.* A mention, keyed by its occurrence address, mapped to a token;
   never a global spelling key. Pending-name metadata stays separate.
2. *Sequence.* Snapshot, overlay, bind at the chooser's call, VP reads the
   value, publish at the clause boundary, commit the kept overlay once.
   Provisioning uses a temporary overlay that never replaces the caller's.
3. *State, kind, history.* Replace by VP, join by mention, kind as
   metadata, LTM0 immutable; the singleton guard and the settled-join rule
   are unchanged; a bound native individual gets an occurrence-keyed entry
   whose value is working state, the bank row untouched.
4. *Events and roles.* The sentence row's slots; the verb and roles are
   read from them; order inside a close from the kept sequence.
5. *Substitution.* Witnessed equalities kept; congruence kept until
   removal; evaluation through identity added.
6. *Scoring.* The value at the bind, in the overlay record.
7. *Boundaries.* Staged tensors, derivative credit within the sentence,
   detach at keep, checkpointed pool.

## 4. Gates (§68's certificates, plus the existing ones)

1. "alec went to town, and then he was attacked": the second VP receives
   the first VP's result through the subject identity; fixed operations.
2. Change of state: "went to town, then went home" keeps the latest
   place, not their union.
3. Two aliases read one updated value; a description mention joins.
4. Both-trial isolation with opposite keep outcomes; stream and
   provisioning isolation.
5. Event and mention counts; joined-site protection; immutable LTM0 rows
   after later value changes.
6. Unchanged: G1–G3, the decomposition arms on their original snapshots
   and seeds, the pending-premise certificates, the reference files, the
   eight-file set and the regression files, on one frozen hash. No
   learning claim.
