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

## 5. Codex: code review and implementation decisions — 2026-10-23

Reviewed against main `8799d4b11`, which records Claude's four incoming
documents unchanged; production is still the reviewed storage source
`6a36b7cee`. This section precedes implementation. §39 governs the work.
The following corrections implement §1 and do not reverse a decision in
it. Every resulting test port will be listed with its exact diff in the
identity receipt.

1. **Layout is explicitly deferred.** §2's two descriptions of sentence
   storage mix today's layout with §§71–72. Today `ClauseRows.write_clause`
   stores an idea's fused point in slot 0 and its derivation reference in
   role 0; that role is not invariably the subject. A relative close also
   returns its Part handle, separately from its LTM0 occurrence. Do not
   replace these layouts or their readers here. Recover the subject,
   object and verb from the kept grammatical operation journal, and carry
   their temporary correspondence to the actual written occurrence into
   MTM. Keep all existing asserted relations and component congruence.
   The layout-and-removal spec owns the later migration and the recalled
   binary-chain representation.

2. **The clause journal is not a live boundary callback.**
   `ClauseJournal.finish_clause` runs after the word bricks and explicitly
   never executes an operator. Updating an overlay there cannot make an
   earlier VP's result available to the next VP. The live seams are
   `LanguageSpace.choose_operation`, `ClauseScope.apply` and the two
   operation loops in `Models` (ordinary and closing). Carry tensor
   identity state through those loops and the existing sentence/fork
   state, and update it when the chosen operation completes a clause.
   `ClauseJournal` later consumes recorded choices/values, rather than
   replaying numerical operations. No Python MTM read enters the compiled
   body. An unresolved object records the event without applying a
   completed-state replacement.

3. **The menu is one global chooser, not a second resolver.**
   `ReferenceContext.prepare_operands` currently expands references on
   determiner and implicit participant operands, with order-one work
   deferred to `ReferenceClose.preview`. Add the explicit unary compose
   `bind` face and offer its accessible token locations plus reference
   zero (dummy). Collapse the lower/definite variants without changing
   their folds or stamped orders. Bind keeps its operand's grammatical
   stamp; neither lexical definitions nor a word-order test gates a
   grammar choice. The kept parse's positions and determination still
   decide noun sites. A recorded compose bind is authoritative at close:
   preview reads its choice and scored value and does not select it
   again. Unbound noun sites become own objects; elided roles and arrival
   thought operations retain their declared contracts. The thought
   `bind.thought` face remains distinct from `bind.forward`.

4. **Provisional identity state belongs to the trial.** The current
   `('new', n)` objects appear only after composition; they cannot serve
   the new operator. Stage existing frame-accessible tokens at the host
   boundary, and give newly determined/bound phrases trial-local handles
   that are disjoint from durable addresses. Include their values,
   aliases and operation correspondence in the same state that
   `SentenceFork` snapshots and restores. A later operation reads the
   latest value through that handle. At kept close, map provisional
   handles to the writer's actual occurrence addresses once. This adds
   no conceptual seat, stored parse or LTM identity column.

5. **Current value is not the historical union.**
   `ReferenceClose.token_value` currently detaches and unions retained
   occurrences, and `individuals` can union those observations again.
   Both live reads must use the current MTM value for a pooled identity;
   historical union remains diagnostic only. Description mentions join;
   completed VP results replace. Keep the mention's value-at-bind in its
   plan for `reconstruction.reference`, so later updates cannot change
   that cost. Overlay values keep the within-sentence graph; only the
   durable committed pool detaches. A bound native individual now gets
   the occurrence-keyed working entry specified in §3, without changing
   its bank row.

6. **Names, events and counting have different ownership.** Preserve
   occurrence-keyed mention membership and its count, so the settled-join
   and singleton guards still concern joined mentions. `names` holds a
   mention's token and reading. Store event history separately on the
   token, with occurrence, verb and role; a sentence's subject ownership
   must not turn an event into another joined mention. Record object
   participation without overwriting that object's value with the
   subject's VP result. Deduplicate repeated admission of the same event
   and role, and order new events by the kept operation sequence. Pool
   eviction/reclamation removes its owned names and events together.

7. **Role evidence changes owner.** `_participants`, `prediction` and
   `expected_roles` presently read Part pairs and `parts.access_count`.
   Replace that source with MTM event histories, preserving the existing
   recurrence threshold and per-role proportion. G1–G3 remain semantic
   certificates. The test that deletes Part rows to erase role evidence
   describes the retired owner; port it to pool expiry/forgetting and
   checkpoint restoration, and disclose every changed check. No Part
   fallback may conceal missing MTM events.

8. **Checkpoint and provisioning have concrete seams.**
   `ConceptualSpace.vocab_extras/load_vocab_extras` own the conceptual
   sidecar; the resolver's extra is version-only. Serialize the pool in
   that existing owner, validate names, values and event references on
   restore, and treat a legacy checkpoint without MTM as an empty pool.
   `_ltm_ingest_truth_texts` already saves/restores caller recall history;
   its temporary identity context needs the same exception-safe boundary.
   Frames still bound accessibility by document/stream; expiry and
   promotion policy remain 2c's work.

9. **The definitions fault is confirmed.** `Interpret.define` does not
   reject a word object, and `PartClosing._definition` can admit one when
   a headed phrase retains the operand word's symbol. Add the §2 guard
   and skip, before any write; remove N1's surface special case. Port
   its regression to rejection and unchanged native forms, as §2
   explicitly requires. This is a definition-integrity check, not a
   lexical part-of-speech decision. Other malformed definitions fail
   loudly; no fixture or production row is silently relabeled.

Implementation follows these seams, then one frozen source with the
specified scoped gates and both original decomposition arms. Every
failed run remains retained. Claude reviews before publication; no
learning claim and no step-2 full sweep in this line. Item 3.0 has not
been reached.

## 6. Codex: read-ahead alignment with LTM and STM — 2026-10-23

Read step-2 §§70–72 again after Alec's request, including the new decision
that the eight chunks are shared by the open sentence, recalled chains
and recent memory. Also checked `STM.md`, the two-truths specification,
and todo's sequence. Their older amendment blocks describe superseded
layouts; §§71–72 govern the upcoming layout-and-removal round.

The identity implementation keeps current values, mention readings and
role histories in MTM. The tensor overlay is the trial's transaction on
that pool, not another STM or a larger parse stack. A stack constituent
carries its correspondence to a transaction identity; the existing
operation journal carries the choices and events. Today's writer exports
a temporary mapping to the actual occurrence beside a returned Part
handle. This mapping is not a stored column and can change when an event
becomes a compressed LTM0 idea or an LTM1 chain head. No READING row,
relation-type column, stored sentence derivation, additional chunk budget
or independent recall store is introduced by the identity implementation.
The identity owner therefore survives the later replacement of the row
and chunk representations.

## 7. Codex: implementation checks and ports — 2026-10-23

Development evidence is retained under
`doc/benchmarks/2026-10-23-identity-threading/development/`, including each
initial RNG, observed reading and failing state. These checks are not
promoted to evidence for the final frozen source.

- `foundation-01`: six MTM cases passed before the old native-individual
  empty-pool assertion failed. §3 now requires an occurrence-keyed working
  entry. The renamed test keeps both native reference assertions and
  replaces only the empty-pool check with the introduction key, native
  symbol, count two and unchanged value checks.
- The role-evidence owner test now checkpoints/restores MTM and forgets
  its entries instead of clearing a Part cache and deleting Part rows.
  Its expected role and participant-content assertions remain verbatim.
- The N1 test follows §2's explicit replacement: defining a word as
  another word's object must raise before writing. Its two absent-form
  checks become the native word's form; the prohibited definition lookup
  becomes empty, and the other word's native form is checked too. The
  object and perceptual identity checks remain verbatim.
- `foundation-02`: 27 passed and four pre-existing skips before a menu
  compatibility error: a reference-rule fixture need not have
  `method_name` when no identity overlay is supplied. The explicit-bind
  dispatch now checks the overlay first and reads that optional field
  safely. No fixture or assertion change is needed for this error.

The compose bind joins the description into the selected current value
before a subsequent VP. This follows §1.4's join rule; the pre-join value
is separately retained in the scored record for reference reconstruction.
An operation journal, rather than the final value alone, owns event order.

`foundation-03` passed 72 checks with four existing skips, including G1–G3
and the new eager/compiled overlay derivative checks. `compose-01` was
interrupted before a certificate outcome: the development runner had not
enabled the arrival capture plugins. Its log and source are retained and
it supplies no evidence. The runner now enables full arrival captures and
both-trial keep probes from the outset; the next draw uses unchanged
production source.

`compose-02` captured both trials before a commit-time correspondence
failure. The keep probe records seven R ties retaining greedy and one
explore win (R 0.03606738 versus 0.03246064); every A was zero. No
certificate assertion was reached. The cause is the existing model-time
stamp: `_commit_sentence` copies the prepared root before writing it, so
its Python identity differs from the plan's root. The writer now exports
the actual root occurrence explicitly alongside its per-node temporary
mapping. The identity-event committer uses that mapping for the prepared
root; no LTM row or test assertion changes.

`compose-03` passed both pending-premise orders with unchanged assertions
and full captures. The native-symbol isolation audit then found that a
global `token_of(native)` lookup could cross streams when the same native
individual appears in both. Frame occurrence membership now selects the
working token; a bare native id resolves only when unambiguous. A new
two-stream regression checks distinct current values and empty-frame
inaccessibility. Provisioning's existing recall-isolation fixture now
supplies the conceptual owner's MTM and additionally checks temporary
pool ownership and restoration; all its previous assertions remain.

`reference-01` passed 54 checks with four existing skips before the second
old native-individual empty-pool assertion, in `test_reference_arrivals`.
Its reference, resolved meaning and pending-name assertions stay verbatim;
the renamed test replaces only the empty-pool check with the occurrence
key, native identity and stored value. The role-history audit also excludes
a standalone mention from event admission: its cue is not a predicate
event. Grammar-completed events still use the kept role correspondence.

`reference-02` reached the native-arrival test after thirteen passes. Its
old single-entry check exposed two independently mutable values for a
native candidate and the pending name selecting it. The pending singleton
now joins the native candidate's accessible working entry, preserving the
settled-site guard. The two fixture arms resolve the old name through
`token_of`; they check the chosen owner's kind, both retained members and
counts three (retained mention) or two (bare native candidate). Their
selected native symbol, derived content and closed-site checks remain.
`reference-03` caught a misplaced local `reading` assignment in this
repair; it now belongs to commit, with no test change. `reference-04`
passed sixteen cases before a thought-bind witness included the subject's
event as well as its mention. Events locate the current identity but do
not add mention witnesses: the witness reader now restricts pooled
mention evidence to the name membership map. The locatable-address
assertion stays unchanged.

`reference-05` confirms that witness fix, then reaches the analogous
native-arrival review fixture. Its prohibition on synthetic companion rows
and all prepare/commit immutability checks stay. The renamed test follows
the pending mention's `token_of` owner and checks the native introduction
and pending mention as that owner's two members; its audit reference is
that owner. The native symbol, count two and successful refinement checks
remain. The earlier single anonymous entry was superseded by §3's bound
native working entry, not by a storage-layout change. New captures include
the pool's serialized names and events as well as its values.

`reference-06` passes thirteen cases before the minimal joined-site guard
fixture reaches event publication without a clause. A refinement-only
plan has no grammatical event to publish; event commit now returns for
that case. No fixture or assertion changes. The compose menu also stages
the existing observed exclusion pairs from the host: it applies the same
eight-observation, never-together rule to the mention and each current
value. This retains the earlier incompatibility decision when binding
moves into the chooser, without pool reads inside compiled composition.

`reference-07` passes 39 cases before N1's replacement checks expose a
second spelling assumption: `Interpret.lexical_form` still treated every
defined identity as an object. It now reads an admitted word's own
description, and continues to read an object's unique defining word. This
completes §2's native-form behavior; no additional test port. The menu
audit also keeps the bind dummy at zero after expectation priming, rather
than letting that bias price it. New identity correspondence and checkpoint
fields receive shape, ownership and value validation before publication.

`integration-01` passes eight cases, including the real eager chooser,
before the new fullgraph fixture constructs a Python namespace inside its
compiled function. Its bank is now created at the host boundary, as in
production; every assertion is unchanged (§17's same-round setup rule).
The raw forward publisher also clears an absent identity transaction so
it cannot expose a stale preceding sentence's journal.

`integration-02` reaches the AOT-compiled gradient check. AOT materializes
an unused parameter gradient as zeros, whereas eager leaves it absent.
The new assertion now accepts absent or exactly zero, still requiring
no pathwise chooser update; its following nonzero score-function-gradient
assertion is unchanged. This is recorded as a changed assertion under
§39. The new bind declaration is appended to each compose catalogue so
existing operation indices retain their order for snapshot comparison.

`integration-03` passes ten cases before the new journal fixture names a
connective outside its CS-only test catalogue. It now uses that catalogue's
`sum` to contain the two completed clauses; every assertion is unchanged.
The unresolved-object audit adds a resolution bit to trial locations: a
bind to an anonymous, unresolved token must not turn an unresolved VP into
a completed state update merely because it selected a location. This bit
is staged with the value and propagated through the journal; it is neither
an LTM column nor a noun-class decision.

`integration-04` confirms the real chooser and exclusion checks, then
finds that a same-close alias of a new token is offered an arrival re-run.
Only an anonymous own object is a new pending site; a mention already
joined by compose bind is not. The arrival planner now distinguishes
them. The new journal fixture introduces its subject through determination,
then binds the alias, so it isolates the specified two-VP identity path
from anonymous first-name arrival behavior. Its assertions stay verbatim.

`integration-05` reaches commit and finds a third event for two completed
VPs. The direct-Clause compatibility adapter was also observing the
recorded program's containing expression. Captured identity journals now
publish events only from their recorded completed operations; direct
Clause fixtures retain their existing grammatical adapter. The event
count assertion stays two.

`integration-06` passes all 28 identity and storage contract cases. The
source audit covers the changed modules, five grammar catalogues, existing
test ports and the new identity certificates. The scoped freeze selects
the regression and native eight-file sets, all reference files, relevant
changed-module files and both original decomposition arms. Development
passes are not promoted to the freeze; every failed source capture stays
in this receipt. Publication still requires Claude's review.

## 8. Codex: first frozen checks and their ports — 2026-10-23

Frozen `9623fa959` passes the regression eight-file set (85 cases) and both
pending-premise orders unchanged. The native eight-file set has two
failures; its complete results and captures remain under
`final/source-01`. Nothing from this hash substitutes for the next
frozen source's gates.

1. `test_native_row_is_the_only_content_and_witnesses_are_live_queries`
   edits a native bank row after opening its token and expects the old
   bank/history union. §1.1 and §1.4 instead retain the token's current
   value. Rename the test to that contract and change its one candidate
   content assertion from `a[0]+a[1]` to `a[0]`. Its changed-bank check,
   live witness counts and no-chart assertions remain verbatim.
2. The c/e closing fixture records every leaf at order zero and has no
   noun site. Its former implicit formation variant supplied an open x;
   collapsing those variants removes that accident. Apply the already
   authorized §13 mechanism-fixture port: set x's reading through
   `reference_orders` to one before presentation, exactly as the forced
   pending-premise certificate does. No rule, forced choice, keep law,
   assertion or learning certificate changes.

Before either port, `c-e-probe-01` reproduced the failure from its retained
initial RNG; every captured initial conceptual-owner parameter is equal.
Both trials and costs are retained. The declarative row ties at R
0.2335897386 and keeps greedy; the question keeps greedy at R 0.1626923084
versus 0.1726922989. A is zero throughout. Both declarative readings have
zero reference sites, so this is a missing fixture order, not a wrong
binding or a keep-law dispute. The original private presentation generator
was not captured for this c/e run; the probe claims equality of the
captured numerical initialization, not identity of source addresses.

The original decomposition snapshots gain a measurement-only catalogue
adapter: restore all surviving parameter rows by grammar rule/category
identity, preserving the prior retired-predictor mapping. Only the added
bind action/role rows keep their original-seed initialization. The replay
ledger records every mapped row, input hash and original seed; no parameter
snapshot, test assertion or production checkpoint loader is rewritten.

## 9. Codex: nested closing query repair — 2026-10-23

`ports-01` and its retained-initialization replay `c-e-probe-02` reach the
c/e fixture's mint assertion after the x reading port. The query answers
the unresolved inner x from its own enclosing equality's subject. This is
a mechanism defect, not a failed mint assertion to port. `ports-02`, with
ancestor exclusion in place, instead chooses an unrelated retained row:
masking the only open role leaves an entirely unconstrained query. Its
captured thought readings, both trial costs and kept choice are retained.

The closing now carries enclosing occurrences in the existing
`_query_exclude` list, including the actual occurrence beside a returned
Part handle. The write boundary carries that occurrence through its root
copy. `best_match` returns no witness when no constrained role remains;
an arbitrary held or primed row cannot answer a bare free name. Bind's
location menu remains the operation for selecting an antecedent. A new
executor regression supplies a held row and verifies that an entirely
open name obtains no witness or evidence from it.

The nested query also inherits the enclosing close's source evidence and
live mode. Its leaf's identification poles are not independent authority
to mint an answer to a question. The c/e fixture supplies the order-one
reading for `what` as well as `x`, under the same §13 mechanism-fixture
authorization. All c/e assertions, query/conclude preferences, live
departures and ForcedGrammar remain unchanged. No learning certificate
gets fixture orders. These repairs preserve the derivation layout and
readers; they correct the closing/query boundary exposed by moving noun
binding into composition. No decision in §1 is reversed.

`ports-03` passes the mint and open-question semantics and reaches the
retained-row kind assertion: the now-explicit pending child is stored as
`unverified`. `ClauseRows.write_clause` now assigns `question` to an open
child as it already does for an open root; this adds no assertion authority
or evidence. Admission preflight and the write use the same kind. The
original row remains open at its occurrence, with no rewrite or change to
the certificate's kind assertion.

`ports-04` passes c/e and 105 further cases, then the newly included
`test_item6_2_thinking.py` reaches four old boundary-fixture cases that
expect a bare equality append to bind a prior name. That expectation was
retired by the accepted §10.4 core before this line, and no live resolver
is exercised in that fixture. Supply its anonymous token and selected
refinement explicitly in MTM, as the kept close would do. The four cases'
existing assertions, including the native antecedent, open raw row,
question kind, witness address and successful working read, stay verbatim.
This is a fixture rewrite under §39, not a return of reader-side binding.

`ports-05` passes all 60 selected thinking/storage cases. Source
`2331c11b0` then passes the 85 regression cases; its native group is
interrupted for artifact packaging, not a test result. One full capture
is 167 MiB, beyond the publication host's per-file limit. Captures above
2 MiB are compressed losslessly, with original and compressed SHA-256
hashes in `packed-captures.json`; subsequent captures use the same format.
The unpublished freeze is retained in `source-02/withdrawn-freeze.bundle`
and replaced before publication. Production and tests are unchanged by
this packaging correction. `source-03` runs all scoped groups on its
replacement hash; nothing is promoted from either earlier source.

## 10. Codex: retained annotations versus open slots — 2026-10-23

Frozen `10a36531c` passes the 85 regression cases. Its native group has
122 passes, seven existing prerequisite skips and one failure:
`test_one_empty_region_query_permits_conclude_and_only_source_mints[True]`.
The c/e certificate and both pending-premise orders pass unchanged.

The new question-kind rule read the retained `_open_references` annotation
even after a thought filled that role with a minted local constituent.
Use `open_slots` for the actual unfilled roles, beside the existing null
relation-operand check. The old annotation remains immutable provenance;
it does not turn a concluded inference into a question. No assertion or
fixture changes. The failed source and capture stay in `source-03`; the
replacement source reruns all required groups without promoting results.
