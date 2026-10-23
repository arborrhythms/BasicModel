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

**MTM and concepts (Alec's question, 2026-10-23).** MTM writes nothing
into a concept. An identity's current value is working state; its
permanent record is the LTM0 occurrences the close writes, immutable, and
the LTM1 membership edges (identity in kind). Promotion (2c, three
distinct sentences) mints the identity as a concept whose content is the
fold over those occurrences, rebuildable; a kind's content never changes
from a member's state, only its extension does, by an edge. One current
path contradicts this and yields to it: `IdentityRows.expose_form` and
`absorb_form` write a named identity's codebook row as the union of its
content and each kept bind's content during training. The identity line
retires that write; the value lives in MTM until promotion.

Indices: concepts are indexed by id in the shared symbol/concept index;
`.where` is a band inside a symbol's percept vector, not an index; LTM0
is keyed by occurrence address and stamped `.when`; STM frames hold
occurrence addresses with their symbols and activations
(`SituationAddress`). MTM keys are occurrence addresses, a token's kind
is a concept id, and a native individual is a symbol id, so MTM resolves
by address and id, never by `.where` or `.when`. MTM is populated only
from STM at the close: an identity enters it through a mention in a
frame, and bind's candidates are the frames, so a concept is bound only
when a mention of it is in STM. A kind is referenced by id without being
pulled into STM. Recall (`query()`) brings LTM entries into STM, and
their identities re-enter MTM.

**A known individual changes: "Felix dies" (Alec's question,
2026-10-23).** Felix is a promoted individual: a concept in the index at
order 1, the class of his occurrences, with a codebook row whose content
is the fold over them, a rebuildable cache (Orders §9). When "Felix" is
read, the word is interpreted as that object (`Interpret.forward`) and
the mention enters STM; bind gives the mention Felix's identity, an
occurrence-keyed MTM entry whose kind is the native symbol and whose
value is working state (§3, point 3); the bank row is not touched. "Dies"
acts on that value and replaces it, and the close writes the idea as an
LTM0 occurrence in Felix's history, with its `.when`. The concept is not
modified by the sentence. It changes only through its occurrences: at the
next consolidation, promotion or rebuild, the fold over Felix's
occurrences includes a death, so the invariant that characterizes him
shrinks (alive is no longer shared by every occurrence) while the set
grows by one. His current state outlives MTM only as his latest
occurrence: when he is next recalled, `query()` brings his occurrences
back into STM and MTM, and his value is re-derived from the most recent
by `.when`, so "Felix is dead" is read from the history, not from the
concept. Kind membership is untouched: Felix stays a cat, since kinds do
not change, members do (Orders §7). An individual is a temporal object
with a trajectory (Orders §4); after death the trajectory is complete and
the concept persists as that completed class.

**MTM is the identity relation over percept parts (Alec, 2026-10-23).**
"Since MTM is basically stitching together identities from other places,
is it possible that it is just the percepts that correspond to the
concepts of STM?" Yes. A token's content is occurrences, each a located
percept part (address, `.where`, `.when`); bind stitches a new mention's
location to earlier ones, which is spatiotemporal continuity, the object
file. The meanings stay in the concept parts of STM's chunks and LTM0's
rows; MTM holds no meaning of its own. The current value is a read over
the stitched chain, the latest event's idea joined with the description
mentions after it, so the stored `value` of §35 is a cache of that read,
kept for working speed. This is why MTM never writes into a concept. The
chain outlives the eight chunks because its occurrences are LTM0 entries;
expiry bounds the stitching, and promotion makes a stitched chain a
concept, the class of its occurrences.
Alec: "Or does it need its own concepts?" No. The cached value is the
identity's *state*, the latest event's idea joined with later
descriptions, a working vector that changes with every verb; the
individual's *concept* is its character, the fold over all its
occurrences, minted at promotion as an order-1 class with a row in the
index. MTM holds state, kind, events and stitched percepts, and no
entry in the index or on the priming surface; before promotion there
is continuity only, and an expired unpromoted chain is gone
(re-identification is future work). The continuum before the
designation, in Dharmakīrti's terms.

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

## 11. Codex: scoped relation and catalogue audit — 2026-10-23

While `731cb5791` completes its changed-module pass, the retained failures
identify three corrections for the next source. No test assertion or
existing chooser forcing will change:

1. **Child question kind.** The root's writer-supplied kind must remain
   authoritative. Before installation of canonical references, a completed
   relative clause can have temporary nulls in its meaning; those are not
   free variables. Assign a child the question kind only when an explicit
   open-reference annotation still denotes an actual open slot. Completed
   roots and unindexed numerical children keep their existing kinds.
2. **Dummy eligibility.** The first dummy-zero assignment currently
   overwrites a scorer's `-inf`, bypassing the existing forced relation
   fixtures and repeatedly choosing bind. Preserve nonfinite exclusions
   there, as the later post-priming assignment already does. Every eligible
   dummy remains exactly zero; the fixed operations, seats and fixture
   sizes stay unchanged. The unchanged no-duplicate-seat and unaligned
   relation assertions remain gates.
3. **Time-reader inventory.** Add `MediumTermMemory.state` and
   `MediumTermMemory.restore` to the explicit fixture inventory as model-time
   checkpoint readers. The inventory equality and all other assertions stay
   verbatim; no timestamp is introduced on the forward path.
4. **Introductions under later wrappers.** The interleaving capture has
   two actual determinations followed by a bind to one of those tokens.
   Recovering only the outer reference site loses the earlier introduction,
   although the live menu correctly offered it. Retain those introductions
   as temporary children of the kept clause and plan them before their uses.
   They get today's occurrence rows; this neither stores an overlay nor
   starts the layout round. No candidate is masked or silently replaced.
5. **A dummy's unchanged argument.** In the retained eight-seat mixing
   failure, dummy binds wrap a carried OBJECT whose working lexical value
   differs from its bank row. Such a no-op must keep the carried reference,
   rather than minting a second seat after comparing with the bank. Fold
   only an unchanged bind (same point and order, no selected binding); real
   transformations keep the ordinary argument admission path.
6. **Public reference fixture.** `test_public_reading_fuses_the_reference_
   before_capture_and_write` still adds a reference variant to `lift`.
   Port the fixture to a declared compose bind and a staged held token.
   The live operand is the joined description/current value; test that
   value and the recorded identity choice, rather than the retired lift
   reference variant. Keep its capture, closing and inverse checks live;
   enumerate the exact assertion edits in the receipt.

## 12. Codex: read-ahead into layout and STM — 2026-10-23

Read `2026-10-23-layout-and-removal.md` and the final decisions at step-2
§72. LTM0 will hold one-vector ideas; unresolved compositions will become
binary tagged LTM1 chains, with no sentence row, stored derivation, or word
bag. Placement moves to the close; word retrieval uses content or bare-word
chain references. STM's shared eight chunks and least-recently-read
non-operand eviction follow that round. These are successor contracts,
not changes to this source. The identity overlay is temporary trial state;
its names, values and events remain the MTM boundary those replacements
will consume. This round retains the journal, three-slot occurrence rows,
readers and close-time minting. The newly diagnosed lost-introduction fix
uses that existing writer and does not introduce READING rows or chain tags.

**Scoped evidence retained.** `source-04` completed all 1,160 selected
changed-module cases, with 13 failures. The additional compiled parity
failure is the same unchanged-bind wrapper losing its child's Part symbol;
its full journal is retained. Development `scoped-01` passes 65 cases
covering the relation roots, eight-seat cases, time catalogue, introduction
recovery and the interleaving schedule. `scoped-02` is a collection error
(mistyped selector), retained. `scoped-03` exposed a fixture setup error in
the port: the held occurrence was left at order 0 and correctly absent
from the object menu. Set that held object to order 1, matching its declared
role; the assertions do not change for this correction.

**Read-ahead amendment seen during this audit.** Claude has now recorded
Alec's move of layout-and-removal and STM chunks to item 3.0 in step-2
§72 and todo. The identity implementation is unchanged by that move; 2c
follows it on today's storage. `scoped-04` further identified the port's
ladder lift as a generic fold, not its intended S: declare that grammatical
position in the unit fixture and rebuild its overlay catalogue. The same
assertions pass in `scoped-05`, including the inverse and gradient checks.

## 13. Codex: copied owner at the closing boundary — 2026-10-23

`scoped-compiled-01` reaches the copied model and fails the unchanged Part
edge check. This is distinct from the original wrapper failure, which is
repaired: `copy.deepcopy(model)` copies both tables but leaves the store's
weak `_concept_owner` and index closures pointing to the original space.
`configure_model_index` then returns on its copied boolean alone. A direct
owner probe confirms that the copy has its own parts and model, but its
store's conceptual owner is the original. This is an ownership repair,
not a fixture port: compare the attached owner with the current registry
space before the fast return; on mismatch, reattach the index/read closures
and clause owner. Retain one row-remap observer per store. A new unit
certificate copies an already bound model/store, reattaches, and checks
that writes and content reads belong only to the copy. The existing
compiled parity assertions, snapshots and seeds stay unchanged. The
read-only `scoped-compiled-02` close probe records the two store identities
and planned edge to authenticate this diagnosis before changing source.

## 14. Codex: one site order for scoring and admission — 2026-10-23

Frozen `4763623e6` passes the 85 regression cases. The native set completes
130 cases with 122 passes, seven existing prerequisite skips and one
failure in the live initial binding-distribution test; both pending-premise
orders and c/e pass unchanged. No gate is promoted to the next source.

The retained failing journal has 32 actions, two nested determinations and
a final bind to the second token. It is not an excessive-depth parse. The
preview now orders introductions before uses, but `_commit_sentence`
reattaches the scored items by zipping them with `sites()`'s original root-
first order. The first own-object decision is therefore attached to the
root, and its later use to an ancestor inside that root: `prepare` correctly
exposes this as a recursive alias. Put the chronological ordering in
`sites()` itself so preview, admission, answer views and every other caller
share one order. No reranking, choice or assertion changes. Extend the
wrapped-introduction certificate to finish the admitted journal again and
reattach decisions through the same zip as production before preparing and
committing it. The full failed journal, plans, initial state and exception
remain under `final/source-05`; the source stays frozen.

The focused `site-order-01` check passes 22 cases, including the unchanged
live initial binding-distribution test and both admission paths. The
original compiled parity case also passes in `scoped-compiled-03` after
§13's ownership repair. These are diagnosis checks; source-06 runs every
required scoped gate again on its own frozen hash.

## 15. Codex: a determination inside a later non-head branch — 2026-10-23

Frozen `6da7938e6` passes all 85 regression cases. Its native set completes
122 passes, seven existing prerequisite skips and one live initial-menu
failure: a compose bind names an unwritten introduction. The captured
journal is authoritative: it introduces tokens in earlier determinations,
then references them after another determination has used their phrases
in its non-head operand. `participant_terms` walks only that later head,
so it never inspects the earlier determining operation. The transaction
owns the token, but closing loses its introduction. This differs from
§14's scoring/admission order mismatch, which remains repaired.

The fix is to inspect every operation subtree, assigning noun status only
at that operation's declared positions. An unselected non-head operand
is visited with `noun=False`; its bare word stays excluded, but any
nested determining operation still has its own description position.
This retains the already selected identity, without selecting a different
antecedent, making a marker into a noun, or changing a chooser menu. The
retained `participant-probe.json` verifies the skipped subtree, and
`participant-proposed-result.json` reproduces the same failure in a fixed
journal and then gets the same two own/use records with the proposed
walk installed only in that diagnostic process. The frozen source remains
unchanged while its other scoped groups finish.

Extend the same-round wrapped-introduction regression to place the first
determination in a later determination's non-head branch, with both direct
and re-admitted paths. All existing assertions remain. The marker and
predicate exclusion certificates run unchanged beside it. The proposed
walk and probe are retained under `final/source-06`; every new source gets
its own scoped gates.

## 16. Codex: completed scoped pass and boundary fixtures — 2026-10-23

All requested groups finish on `6da7938e6`: 1,423 passes, twelve skips and
nine failures, retained in `final/source-06/gate-summary.json`. The skips
include five module prerequisites/device cases and the seven native
prerequisites; they are not claimed as passes. No result is promoted.

- The live initial-menu failure is §15. The retained initial RNG passes
  after the traversal repair in `retained-menu-01`; its captured-state
  certificate is a diagnosis check, not another frozen gate.
- The router-context fixture's unary stub returns `value + 1`, which can
  leave the identity cube before the binary face reads it. Make the stub
  `1 - value`, which is bounded through any number of selected unary
  applications. Both structural-context assertions stay verbatim. No
  production bound or numerical operator changes.
- The two pending-relation storage fixtures still expect a bare equality
  append to bind a name. That reader was retired at §10.4. As with the
  four direct item-6 boundary fixtures in §7, explicitly supply the kept
  MTM introduction and its selected `bind` after both rows are present.
  The asserted symbol, null removal, source address, document, position
  and immutable raw row all stay verbatim, in both orders.
- The production completed-question failure comes from the nested branch
  of `closing_question`: after inheriting its parent's authority, an
  unsupported region can cease to need an episode, but the branch returns
  it unconditionally. Recheck `needs_episode` after that inheritance and
  continue to other children when it is false. A new certificate checks
  both the closed region and a later sibling that still has a free role;
  the latter must remain a question. `boundary-01` passes all 54 focused
  identity, participant, router and math-repair cases after these fixes.

Both original-snapshot decomposition arms fail their unchanged proof
assertion in both successor cases. The keep probes retain both trials,
their R/A components and the unchanged tie decision. Their fixture still
requests mint/reference/open alternatives on ordinary operations, but
§1.3 gives those reference choices to explicit compose `bind`. Its words
also all have order zero: its old open-name rank no longer creates an
order-one site. Port the fixture's existing choices, not its conclusions:
record the declared noun readings through `reference_orders` under the
same unit-fixture rule as reconstruction §13; express the former mint
choice through determination and a retained reference/open choice through
`bind`. Use the same preferred source references and question-only open
name. The question's `what` stays open; no closure is forced, no answer is
supplied to a premise, and no assertion, snapshot, seed or budget changes.
All exact fixture edits ride in the receipt's test-port diff.

`decomposition-01` retains the first port attempt: merely stamping every
noun and always binding makes first mentions anonymous, so the legacy
fixture's preferred-source callback encounters a genuinely open reference.
The callback now tolerates an open source; the port preserves the original
mint fallback via `lower_unary` instead of replacing it with the dummy.
The former `store.refs` ranking read is also removed: its occurrence-only
column was always unfilled, so the same +6 occurrence rank is retained.
Every attempt remains captured for review.

`decomposition-02` retains the next fixture failure. The old fixture's
unary choices only ran at a one-slot root; its offset therefore named the
first operation window column. The newly explicit NP operations also run
at depth two, where the top stack operand is the second window column.
Use `min(depth, 2) - 1` for their reference slice and unary action offset.
This changes which operand the fixture requests, not the requested bind,
mint or dummy, and adds no production mask.

## 17. Codex: direct pending roles after explicit compose bind — 2026-10-23

`decomposition-03` and `decomposition-04` retain both trial readings and
keep probes. The fixture first needed its original bare-operand guard on
the explicit NP operations too: a completed S or VP can carry a head's
word, but that does not request another bind of the entire phrase.

The remaining production failure is a lost role correspondence. The old
operation reference path wrote `open_reference` on the operand and its
parent role. Explicit compose bind instead owns a child site, and prepare
marks only that child's pending name. The root still points to its child
but has no pending-role annotation, so `closing_question` asks about the
anonymous child alone, dropping the equality predicate and complement.
Carry a one-slot child's pending name and site record to the direct role
that references it during prepare. The existing MTM working reader then
clears a resolved role or keeps the actual pending role open. This is not
an equality-derived bind or a nested-position redesign: only the direct
`('clause', n)` correspondence is used. Add a direct-root regression and
leave all existing decomposition assertions and thought forcing unchanged.

**Further retained diagnostics.** `direct-role-01` passes 55 identity,
arrival and math-repair cases, including the new direct subject/complement
question cases. `decomposition-05` still fails: forcing determination on
all constants consumes a fourth operation before lift in the unchanged
three-round word loop. `decomposition-06` restricts the reading stamps and
explicit NP operations to the fixture's named references (x/y/what, plus
z for two successors), leaving constants at their original readings. The
proof assertion still fails; both trial captures and keep probes remain.
A focused read-only Claude diagnosis is requested before further changes
to resolution semantics. It is not the final frozen-source review.

The diagnosis also exposed a capture limitation: the contract observer
saved tensor states and MTM plus complete journals, but not LTM0's separate
semantic sidecar. Future captures add `semantic_extras`, Part state,
ConceptualSpace extras and the model's structural sidecar. Existing captures
remain unchanged and are not claimed to contain that missing sidecar.

## 18. Codex: Claude's focused decomposition diagnosis — 2026-10-23

The actual [read-only report](../benchmarks/2026-10-23-identity-threading/development/claude-decomposition-diagnosis/report.md)
is retained with its request, exact source diff and tool stream. It is a
diagnosis of the failed working source, not a landing verdict.

1. **Fixture A.** A surface wrapper drops the predicate scope flag, while
   preserving the overlay's producing-action column. The fixture's scope
   test therefore repeatedly bound a completed phrase and exhausted the
   unchanged closing budget. A bare pushed word has source column zero;
   any opened NP, VP or S has its producing action. Use that column only
   in the forced fixture to preserve its original bare-operand scope.
   No production menu, repetition rule, forced conclusion or budget changes.
2. **Production B.** `git show b38889eb7` confirms both the order-one
   leaf's `forward_identity` and prepare's unconditional named-open path
   predate this line. Explicit compose bind now supplies a resolved token
   before the close. Keeping that child pending stops preparation of the
   entire enclosing relation, so its retained Part readers never receive
   the relation this round promises to preserve. For a recorded compose
   bind to a resolved kind/native reference or a completed same-close
   alias, attach that existing kind/alias and do not mark its name open.
   The numerical point remains the selected current value; the actual
   chosen token is retained in the plan and MTM commit. Anonymous and
   unresolved references keep the pending path. This implements §1.3 and
   preserves §1.7's current storage, without replacing the readers.
3. **Direct role repair D.** Restrict the §17 correspondence to participant
   roles 0 and 2. `with_slots` already deduplicates its open annotations.
   Add a resolved-compose-bind counterpart to the pending-child tests.
4. **Remaining audit C.** An unminted completed object and an anonymous
   pending name can both lack a kind. The final source must distinguish
   those cases without inferring resolution from a seat or from content
   magnitude. The focused report proposes explicit pending metadata on
   MTM and a versioned checkpoint migration; this is audited below before
   the next freeze, alongside the staged and committed readers.

Claude flags possible decisions about order-zero binds and repeated binds.
This round preserves the existing no-order-gate and no-menu-mask rules:
only stamped order-one participant sites create an identity object under
the settled grammar contract, and the operation budget bounds repeated
choices. No exclusion or new grammar rule is introduced. The future order
and layout designs remain deferred. Using a resolved token's kind in the
current Part writer is the compatibility bridge explicitly required by
§1.7, not an early migration to a different stored layout.

`decomposition-07` retains the next failure and its keep probes: both
questions now have only `what` open, and the premise Part relations are
present. The fixture still attaches trailing punctuation with `verb`,
turning the bound noun into a predicate head. Its old operation-level
reference variant subsequently discarded that wrapper; compose bind now
correctly precedes it. Port that marker attachment to the existing surface
operation's declared marker position: transpose the top pair, then surface.
This keeps the bound noun as the head instead of asking a verb operation
to carry punctuation. It changes only this forced grammar fixture; the
same operators, original budgets, snapshots, seeds, assertions and live
keep apply. The concrete action port is listed in the receipt.


## 19. Codex: resolved values without a minted kind — 2026-10-23

Claude's §18C distinction is explicit `Token.pending` metadata. It describes
resolution, not nounhood or an individual seat. Named own objects are
pending; completed own objects and mints are not, even without a minted
kind. Staging and the working reference reader use this bit. Refinement
with a kind clears it; merging a pending token into a resolved token also
clears it, without overwriting the antecedent's kind. VP value updates do
not by themselves resolve a pending name. No identity or new column is
written to LTM0.

Checkpoint version 2 stores the bit and validates it before replacing the
pool. Version 1 migrates using its historical `kind is None` convention;
that old format cannot retrospectively distinguish an unminted value.
New certificates cover both states at the host stage and working reader,
round-trip state, atomic malformed-state rejection, migration, and merge.
The resolved-compose-bind direct subject/complement certificate checks the
current Part relation, no open role, actual selected token and immutable
rows/bank after commit. No existing assertion changes.

`decomposition-08` passes both zero-arm successor cases with all four
companion questions correct at the original snapshots and seeds. It is a
diagnosis pass only; both arms run on the next frozen source. `resolved-01`
passes 101 cases with four prerequisite skips and exposes an overbroad
§17 propagation in the existing R10 alias certificate. A fused value's
structural children are not relative-clause roles. Restrict propagation to
present participant slots of a relative clause. `resolved-02` then passes
all 102 cases with the same four skips; every R10 assertion stays verbatim.
The failed capture remains retained. Source-07 will run the complete scoped
selection, including both decomposition arms, with no earlier result promoted.


## 20. Codex: retained c/e query failure on source-07 — 2026-10-23

Frozen `59b7c8b87` passes all 85 regression tests. Its native set completes
122 passes, seven prerequisite skips and one c/e failure at the unchanged
line 348: the question must remain open. Both pending-premise orders pass
with counts 0 bound/8 open and 0 bound/24 open. No source is changed while
the other frozen groups finish.

The full failure, semantic sidecar, MTM, both readings and keep costs are in
`final/source-07`; `ce-probe.json` and `ce-memory-probe.json` are compact
extracts, not replacement captures. Both trials tie R and A; keep is greedy.
The question now retains its actual equality and only its direct requested
referent is open. Query selects the other sentence's equality, then treats
its fused operand occurrence as a complete answer, even though that
operand's named child is pending. The child is open in MTM; its fused parent
row has neither a derivation reference nor a semantic child reference.
This is a mechanism failure, not a revised expected answer or a learning
claim. A focused read-only Claude diagnosis examines the retained-storage
boundary before the next repair. No assertion, forcing, objective or keep
rule changes; the current hash remains frozen for all remaining gates.


### Claude's diagnosis and the repair, same date

The [actual read-only report](../benchmarks/2026-10-23-identity-threading/final/source-07/claude-ce-diagnosis/report.md)
confirms the loss at `ClauseRows.write_clause`: an absolute child with an
open descendant is not PartClosed, but its fused row has no remaining
resolution information. This predates identity. The previous c/e pass
queried the isolated word instead of the actual equality; the corrected
direct-role question exposes it. Neither query's Part reader nor the keep
selected a wrong interpretation.

Record `_unresolved_operands` on the retained parent's existing semantic
sidecar for a referenced child that is incomplete strictly below that
child (`incomplete(child) and not pending(child)`). Each entry pairs the
parent role with the occurrence reference already in that role. This is
a write-once resolution annotation, not a new row, tensor, column, parse
or identity. Direct named children keep their existing open-slot path.
The old fused placeholder is never recomputed when its inner name later
resolves, so it must not become a completed answer merely because time
passes; re-evaluating retained expressions stays with the later design.

`ThoughtBinding.integrate` refuses a marked operand as the final answer,
while preserving the equality as a rewrite and permitting its unmarked
reverse endpoint. The ordinary one-frame `ThoughtReferences.fill` uses
the same answer boundary, preventing a non-equality bypass. Exclude the
annotation from `semantic_metadata`, so it does not change chooser inputs.
Tests cover write-time marking, resolved/direct-pending counterparts,
sidecar round-trip, both equality directions, ordinary fill and identical
metadata features. The c/e assertion and all other existing assertions
stay verbatim. Claude identifies no reversal of §1; §39 permits this
repair. Apply after the source-07 scoped run ends, then freeze and run every
required gate again on the repaired source.


## 21. Codex: repeated equality is not progress — 2026-10-23

`unresolved-01` passes all 58 focused unit/reference cases but retains the
c/e failure and keep probe. The new annotation is present on the equality;
the first answer direction is correctly refused. A previous read of that
same equality, however, lets `equivalent` reach the opposite endpoint and
answer the still-free question with its own fixed side. This is the same
backwards no-progress path the existing integration guard already rejects
for a bound region. Both c/e trials still tie and keep greedy.

`unresolved-02` reproduces exactly that defect by giving the new equality
unit its prior result: eleven pass and the repeated/free case fails. Extend
the existing guard to a free region when its source only matches through
an equality path and the proposed answer is that region's fixed side.
An explicitly reflexive direct witness remains permitted; a new test
checks it. The valid reverse endpoint and all previous substitution proofs
remain live. No assertion, query rank, chooser feature, objective or keep
rule changes. This is another proof-validity repair under §39, not an
alternative accepted answer. Every failed run stays in the receipt.


`unresolved-03` passes 61 focused cases. The c/e question now correctly
stays open with nothing minted, but the unchanged operation assertion
finds a second query before conclude: retrieval still returns the marked
frame as an answer witness before integration refuses it. Apply the same
resolution check at `ThoughtLTMCapability.best_match`'s answer boundary.
For a free equality side, the marked endpoint cannot supply the answer;
its unmarked reverse endpoint remains usable when the query identifies
the marked occurrence. Other queries retain their direct role alignment.
This changes which retrieved evidence can answer a free role, not the
operation/bind menus. Add both direction checks; retain the recorded scan
count. No test assertion is ported. The failed run and keep probe remain.


`unresolved-04` passes all 72 focused cases, including the c/e certificate
with every original assertion. Source-07 is complete with 1,442 passes,
twelve prerequisite/device skips and its one retained c/e failure.
Source-08 runs the whole scoped selection again. The ten additional direct
consumers of the changed thought readers join the module selection; the
114 module files are partitioned into two independent process groups,
each file run once on the same frozen source. This is still a scoped pass,
not step 2's deferred full sweep. No previous source supplies evidence.

## 5. Grounding: the psychological stores and Dharmakīrti (Claude, 2026-10-23, at Alec's request)

**Three stores, three literatures.**

- *STM as chunks.* Miller's seven plus or minus two (1956) and Cowan's
  four (2001) are counts of chunks, not items, and a chunk is a bound
  group; §72's eight shared chunks of ideas and relations is that
  reading. Baddeley's episodic buffer (2000) is the shared set itself: a
  limited store that binds information from the slave systems and from
  long-term memory into integrated episodes, which is what a chunk
  holding a recalled chain's head beside the open sentence's constituents
  does. Ericsson and Kintsch's long-term working memory (1995) is recall
  by cue: retrieval structures in LTM are reached from STM, and what they
  return occupies working memory, which is `query()` loading a chain's
  head into a chunk.
- *MTM as object files and file cards.* Kahneman, Treisman and Gibbs
  (1992) distinguish the object file, a temporary episodic record indexed
  by spatiotemporal continuity and updated as the object changes, from
  the type, the long-term recognition category. MTM is the object file:
  keyed by its introducing occurrence, holding the current state, updated
  by each event, with the kind as a separate reference. Pylyshyn's visual
  indexes (1989, 2001) are the pointer-only version MTM had until §35;
  the object-file literature is why it now holds content. In discourse
  semantics the same object is Heim's file card (1982): one card per
  discourse referent, written to as the discourse proceeds, which is a
  name/value store with VP updates; Kamp's discourse referents (1981)
  with their accessibility conditions are bind's candidate frames;
  Recanati's mental files (2012) are the same thing for names. Situation
  models (Zwaan and Radvansky 1998) track protagonists' states across
  sentences and update them at events, which is identity threading;
  event segmentation theory (Zacks et al. 2007) puts the update at the
  event boundary, which is the close.
- *Episodic and semantic memory as the two parts of every entry.*
  Tulving (1972): episodic memory holds dated occurrences, semantic
  memory holds concepts and their relations. Alec (2026-10-23): episodic
  memory is the percept part of STM and LTM, semantic memory the concept
  part; every entry is `[form | meaning]`, and `.when` belongs to the
  percept part. The concept index with LTM1's edges is the semantic
  store proper. Complementary learning systems
  (McClelland, McNaughton and O'Reilly 1995) make consolidation a slow
  extraction of structure from replayed episodes; promotion by recurrence,
  three distinct sentences, is that extraction, and the minted concept's
  content as the fold over its occurrences is the extracted structure.
  Alec's episodic-memory note, several LTM0 entries per time, is Tulving's
  episode proper.

**Dharmakīrti.** The fit is close enough to use as a check on the design.

- *Two objects of cognition.* The unique particular (svalakṣaṇa) is
  momentary, located, causally efficacious and grasped directly by
  perception without conceptual construction; the universal
  (sāmānyalakṣaṇa) is a conceptual construction by exclusion (apoha) and
  is what inference and language handle. An LTM0 idea, "an absolute truth
  understood directly ... a single vector ... not necessary in any
  codebook" (§71), is the particular; a concept in the index, with its
  for and against lanes, is the universal, and the project's apoha reading
  (Philosophy, "Apoha: attention excludes") already treats exclusion as
  what forms it. Dharmakīrti's two truths, the ultimately real particular
  and the conventionally real universal, are the project's two truths,
  ideas and relations, in his vocabulary.
- *Identity as a continuum.* On momentariness (kṣaṇikatva) nothing
  persists; a continuing object is a continuum (santāna) of causally
  linked moments, and its identity is imputed (prajñapti). An MTM token is
  a santāna: its occurrences are the moments, its current value the
  latest, its identity a designation that binding makes, which is
  "identity is its occurrences" and Philosophy's uncollapsed model.
- *Recognition and determination.* For Dharmakīrti, "this is that same
  one" is a conceptual determination (adhyavasāya, niścaya) that follows
  perception; it is a judgment, not a perception, and can be wrong. Bind
  is that judgment: scored, with the dummy as the option to withhold it,
  made at the noun phrase's determination, and revisable only as
  re-identification, which is future work. "An object at its
  determination" is his term as well as ours.
- *Memory.* Memory (smṛti) is conceptual for Dharmakīrti, a construction
  from traces, not a valid means of knowing the particular again. "If we
  need the original sentence, reconstruct from the vector; it will not be
  exact" (§30) is the same claim.

References: Baddeley 2000, *Trends Cogn. Sci.* 4; Baddeley and Hitch
1974; Cowan 2001, *Behav. Brain Sci.* 24; Ericsson and Kintsch 1995,
*Psychol. Rev.* 102; Heim 1982, *The Semantics of Definite and Indefinite
Noun Phrases*; Kahneman, Treisman and Gibbs 1992, *Cognitive Psychology*
24; Kamp 1981, "A theory of truth and semantic representation";
McClelland, McNaughton and O'Reilly 1995, *Psychol. Rev.* 102; Miller
1956, *Psychol. Rev.* 63; Pylyshyn 2001, *Cognition* 80; Recanati 2012,
*Mental Files*; Tulving 1972, in *Organization of Memory*; Zacks et al.
2007, *Psychol. Bull.* 133; Zwaan and Radvansky 1998, *Psychol. Bull.*
123. Dharmakīrti: *Nyāyabindu* I (perception, the two objects) and
*Pramāṇavārttika*, Pratyakṣa chapter; Dunne 2004, *Foundations of
Dharmakīrti's Philosophy*; Dreyfus 1997, *Recognizing Reality*; Siderits,
Tillemans and Chakrabarti (eds.) 2011, *Apoha*; Arnold 2012, *Brains,
Buddhas, and Believing*; Tillemans, "Dharmakīrti", *Stanford Encyclopedia
of Philosophy*.


## 22. Codex: incoming MTM/concept clarification — 2026-10-23

Claude's concurrent §2 clarification explicitly retires a remaining live
write: `ReferenceClose.commit(training=True)` and `resolved` still call
`IdentityRows.expose_form`, allocating a cue-region concept or unioning
bound content into its bank row. This conflicts with the identity value's
MTM ownership. The incoming text and grounding section were preserved
verbatim in `248f8f8d7`; Philosophy's companion edit is committed unchanged
separately. The duplicate numbered §5 grounding heading is Claude's text;
its named anchor remains distinct from the earlier code review.

Source-08's scoped pass is deliberately interrupted before this repair:
517 passes and one skip were recorded, no failed assertion; the remaining
cases are incomplete, not passes. The interruption ledger, logs and all
partial captures remain; no source-08 result is promoted. The prior full
source-07 failure and both actual focused Claude diagnoses remain too.

Remove both production exposure calls and retire `expose_form` and
`absorb_form`. Keep mention loss, answer supervision and score-function
credit unchanged. Existing native cue regions remain readable for legacy
state; promotion stays 2c's. The unit helper `form_exposure` now explicitly
mints its supplied prior cue region, instead of calling a retired learning
write. This fixture rewrite changes no assertion. Four new cases cover
kept bind training and delayed mention credit, each with/without an existing
cue region: no new allocation, no region mutation, no row or bank change;
the delayed loss still requires gradients. This implements §1's MTM value
ownership and the incoming §2 clarification without a decision reversal.


`bank-isolation-01` passes 123 cases with four existing prerequisite skips.
The read-ahead's known-individual example reinforces the same separation:
a VP changes the current MTM value, not its native concept. Promotion,
consolidation and rebuilding remain their scheduled work. Source-09 will
run all scoped groups on its own frozen hash, including every direct
consumer of IdentityRows; the source-08 partial pass supplies no evidence.


## 23. Codex: full scoped source-09 boundary failures — 2026-10-23

Frozen `81ba14861` completes every requested group: 1,582 passes, twelve
existing prerequisite skips, three failures. Regression/native groups,
G1–G3, c/e, both pending-premise orders, and both original-snapshot
and original-seed decomposition arms pass. The wider changed-module
selection exposes three defects; this source is not submitted for LAND or
published. No earlier evidence is promoted. All captures and replays are
in the identity receipt's `final/source-09`.

Before changing source, replay the three retained initial RNG states with
read-only boundary and keep probes (`diagnosis`). The presentation corpus's
private generator was not captured; that limitation is explicit, and the
replays are diagnostic only. Both mechanism errors reproduce. The first
fails during exploit preview before the other trial or its keep exists;
previous completed closes retain both R/A readings and decisions. The
admission failure has both readings and the actual keep: row 0 compares
R 1.214051008 versus 1.591376066, row 1 ties at 1.480019212; A is zero,
and both keep greedy. The decoder case has no sentence keep.

1. A grammatical `definite_unary` over a relation opens a live overlay
   handle, but `ClauseJournal` requires `region(description)` and a fused
   `has_point` before recovering its site. Its later compose bind reaches
   an unwritten introduction. The captured missing handle is 24, opened
   before the later relation/bind wrappers. It needs a consistent
   occurrence-backed determined object that preserves the underlying
   relation, not a menu restriction or an invented resolved value. The
   focused Claude diagnosis is requested before choosing that repair.
2. `resolve_lexical_references` keeps an order-0 source word in preview
   when no corresponding object exists, but replaces it with -1 during
   admission. The identical reading then acquires extra child clauses and
   trips the frozen-structure guard. Proposed narrow fix: retain the
   word's identity fallback at order zero in both phases; leave absent or
   ambiguous higher orders unchanged. Every existing assertion stays.
3. The old two-operation decoder fixture supplies Difference an operand
   `[1, 2]`, outside the conceptual cube. The inverse probe shows Sum
   available and Difference rejected, so there is no second choice. The
   proposed fixture port scales its event and bank uniformly by 0.5,
   preserving shapes, both operations and all existing assertions. The
   production decoder domain stays unchanged.

These proposals are not code changes yet. The actual read-only Claude
response will be retained beside the three probes; it is a focused
diagnosis, not final review. §39 remains the authority to repair and
continue without reversing §1.


## 24. Codex: source-09 diagnosis accepted and repaired — 2026-10-23

The actual read-only Claude diagnosis is retained at
`final/source-09/diagnosis/claude/report.md`. It finds no §1 reversal and
accepts all three repairs under §39; this is not a LAND verdict.

A relative description's grammatical determination retains the exact value
opened by its numerical operation journal. Its own-object clause owns that
value and its introducing handle; the original relation is a child with
its references intact. No guessed fused meaning, word reading, seat or
menu rule is added. The site has no lexical reading: `subject_word_id=-1`
and an explicit introduction marker. Required introducing actions are
retained independently of later bind wrappers, and chronological sites
still put the introduction first. A dummy on an already determined object
keeps that handle and remains free in reference reconstruction. An event
whose tag-3 introduction has vanished raises instead of silently losing
that event. Separate determinations still create separate handles.

Order-zero reference capture now uses the source word in both phases when
there is no association; an ambiguous order-zero reading is unknown in
both phases. Higher orders keep their prior behavior. This preserves the
identity fallback and the scored/admitted clause correspondence without
changing R, A, the keep or forced choices.

The exact existing-test fixture port is limited to
`test_free_policy_has_no_pathwise_credit_between_supported_operations`:
its event `[1,1]` becomes `[.5,.5]` and bank `[[1,0],[0,1],[1,2]]` becomes
`[[.5,0],[0,.5],[.5,1]]`. Shapes, operations and every assertion remain
verbatim. `diagnosis/unchanged-decoder.json` verifies that the decoder,
eligibility method and old fixture were unchanged from the pre-identity
source. No test is retired. New certificates and retained-RNG diagnostics
will check the repairs before the next freeze; no source-09 result is
promoted.

## 25. Codex: occurrence arguments at the thought writer — 2026-10-23

The retained-RNG development replay `boundary-repair-01` gets past the
source-09 introduction error, then stops at `ThoughtStream.write`: a
concluded relation contains an existing LTM0 occurrence as its left
argument, but the adapter accepts only symbols and local constituents.
`thought-boundary-01` captures the conclusion and both trial readings
before the repair. The actual keep ties on R 0.503043473 and A 0 and
keeps greedy; neither the keep nor the conclusion is changed. The corpus
uses an uncaptured private generator, so these are diagnostic replays,
not source-hash gate evidence.

The adapter now translates that validated occurrence to its existing row
address for the ordinary `ClauseRows`/`PartClosing` writer. The existing
derivation and mint path still applies, including the derivation reference
on the completed occurrence; no new table, tag, row layout, object-to-word definition,
answer target or bank update is introduced. Missing occurrence arguments
still fail loudly. This completes a reader/writer boundary required by
identity threading and reverses no §1 decision. New tests cover both
argument roles with unchanged input rows and the asserted relation pair.
No existing assertion is changed or test retired by this repair.

## 26. Codex: boundary certificates and the next freeze — 2026-10-23

`boundary-certificates-01` retains five failures in new certificates. Two
expose the dummy wrapper still losing its relative site: recovery must
accept its recorded bind request and keep the determined phrase's
nonlexical carrier. One exposes `word_concepts` including the word itself:
order-zero ambiguity checks exclude words, exactly as the resolver does.
The other two are new fixture correspondence errors: the writer returns
`('sym', handle)` and today's completed occurrence stores that derivation
reference, not its input roles. Correct the new fixture's handle unpacking
and replace its new raw-input-role assertion with exact
`(('sym', result), None, None)`. The asserted pair, evidence and unchanged
prior row checks stay. The original failed run is retained. This is no
layout change and no existing-test assertion port.

`boundary-certificates-02` passes all 71 cases. `boundary-guard-01` passes
the extra loud-loss check. The relation-backed fixture covers two equality
updates, a later bind, an intervening dummy, and direct versus readmitted
plans; both mention and event counts and immutable rows/bank are checked.
Separate determinations remain separate. `thought-boundary-02` completes
the unforced math presentation from the retained initial RNG after the
writer repair. These are development checks only. The next frozen source
runs every requested scoped group anew; all failures above stay retained.

The newly incoming layout text further specifies the STM/LTM0 time band:
a concept gets the current stamp on entering an STM chunk, a recalled
entry retains its past stamp, and an estimate can be stamped ahead. The
separate LTM0 `when` column retires in that later round. This identity line
keeps today's columns and close stamps unchanged; it does not partially
implement the item-3 layout.

## 27. Codex: corrected read-ahead time representation — 2026-10-23

Claude's incoming layout amendment corrects the preceding read-ahead
summary: ConceptualSpace has no reserved time band. The later STM row
appends `RowTime` beside the concept vector and retains the LTM0 `when`
column; the symbol's muxed source-position band is a different quantity.
The earlier in-band wording in §26 describes the now-superseded draft.
These are incoming item-3 documentation edits, preserved unchanged. They
change no identity source, gate, row layout or current closing stamp on
frozen `d55544d98`.

## 28. Codex: growing stream groups at pending-name reads — 2026-10-23

Source-10 `d55544d98`'s second module group retains a production failure in
`test_ordinary_driver_two_three_eight_stream_groups_after_single_write`:
`pending_names` reads stream column 1 from an old table with one column.
The existing test exercises real groups of 1, 2, 3 and 8. The row is scoped
to stream 0, so it cannot belong to a stream whose column has not yet been
created. The read must reject that membership before indexing the column.
The identical unsafe access also exists in `_arrival_plan`.

The pre-change diagnostic `stream-boundary-01` restores the captured
initial RNG and reproduces the error with stream shape `[8192, 1]`, old
row 0's membership `[true]`, and requested stream 1. Full scoring/readings
and the preceding greedy tie are retained. The failing close stops during
exploit preview, before its second trial or keep exists. As before, the
private presentation generator is not restored; this is diagnostic, not
promoted first-draw gate evidence.

Under §39, both arrival readers gain the same upper-bound membership
check. A row with no stream membership retains its existing unscoped
behavior; an explicitly scoped row cannot leak to a new stream. New unit
certificates cover both readers and scoped/unscoped rows without changing
the table width. The original driver assertions stay verbatim, as do all
other existing assertions. No menu, objective, layout or §1 decision changes.
The complete source-10 results remain retained; the repaired source must
own a new frozen scoped pass.

The complete source-10 pass ends with 1,608 passed, twelve existing
prerequisite skips and this one failure. All requested cases complete;
both pending-premise orders, both original-snapshot decomposition arms,
c/e, the regression/native groups and the first module group pass. None
is promoted to the next source.

`stream-guard-01` passes six cases and exposes a setup omission in the new
unscoped-row unit: its hand-built plan lacks preview's zero `cost` field.
Add that field exactly as production supplies it, retaining every
assertion (§17's same-round setup exception). No existing fixture changes.

`stream-guard-02` passes all seven arrival-review cases.
`stream-boundary-02` passes the unchanged real 1/2/3/8-stream driver from
the retained initial RNG. These are diagnostics; source-11 freezes the
repair and reruns all requested scoped gates on its own hash.

## 29. Codex: source-11 scoped gates complete — 2026-10-23

Frozen `bf44bbc47804899e470d4150f261496631b28a64` completes all 1,623
selected cases across 132 files: 1,611 passed, twelve existing skips and
zero failures, with no compile-cache retry. The receipt's source audit,
assertion ledger, exact port patch and per-group results all name that
source. Both decomposition arms preserve their original snapshots/seeds
and all surviving parameters. The unchanged pending-premise certificate
records forward 0 bound / 16 open, reverse 0 / 32, with no wrong binding.
G1–G3, c/e, the unforced presentation, optimizer-gradient log and the real
1/2/3/8-stream driver pass. Earlier failures and their pre-change probes
remain retained. No earlier source result substitutes for these gates.

The skips are nine absent phase artifacts and three opt-in slow MPS runs;
none is a pass or learning evidence. This is the scoped identity pass,
not step 2's full sweep. The production source, tests and measurement
fixtures have not changed after freeze. Claude now reviews the frozen
source and receipt before publication. Item 3.0 has not been reached.

## 30. Codex: landing review and typed operands — 2026-10-23

Claude's actual read-only review of frozen `bf44bbc47` returns **DO NOT
LAND** despite the complete green scoped pass. The identity branch of
`ReferenceContext.prepare_operands.propose` returned before the existing
`resolve_order` lookup. That accidentally changed non-order-one operand
values, although capture still resolved their references. Decision §1.3
removes reference variants, not typed-concept lookup. Restore that lookup
for orders other than one, preserve implicit literal operands, and retain
exactly one option. Order one remains the raw operand until explicit bind.
This restores the decided contract and reverses no §1 choice.

The existing kind-point certificate is parametrized with and without the
identity state. Its point and reference assertions stay; add an exact
single-option assertion. Seven new cases cover implicit orders zero/two,
order one, literals and an explicitly requested generic type. The public
binding/capture fixture gains the origin assertion Claude requested:
`entry.identity_origins[0, :2] == (2, occurrence[-1])`; its existing
assertions stay. `typed-operand-01` passes all sixteen cases. No test is
retired. Source-12 reruns every scoped group; nothing is promoted from 11.

Claude's nonblocking decomposition note is also closed: the measurement
adapter obtains old action names from the original grammar, checks the
binary catalogue unchanged, and maps surviving unary rows by name in
`apply_anchor`, `tool_embedding` and `what_projection`. The name ledger
independently authenticates the row map before bitwise parameter checks.
Original snapshots, seeds, budgets and all certificate assertions stay.

Two explicitly nonblocking event-history notes remain follow-ups: a
re-close at an event occurrence can drop its history through `forget`;
an unfactored leaf verb may lack the point `_participants` requires.
Record both in FutureWork for the event/recall contract; no speculative
re-close policy or new verb representation is imposed in this repair.

The incoming layout draft has changed again since §27: Alec now locates
time in the percept part of the paired vector and retires the separate
columns in that future round. Its percept/concept and model-time wording
is preserved unchanged in the layout spec and Philosophy. It remains
item 3's design, not an identity-line storage change. Today's columns,
derivation readers and close-time minting stay.

## 30. Claude's review of source-11 and the repair in progress — 2026-10-23

Reviewed: frozen `bf44bbc47` (1,611 passed, twelve existing skips, no
failures, 132 files), the receipt, §§6–29, the delegated review's DO NOT
LAND, and the uncommitted repair in the working tree.

**The delegated finding is real, and the repair is right.** At the
frozen source, `ReferenceContext.prepare_operands` returned an operand
unchanged whenever the identity overlay was supplied, so a kind or
generic operand composed from the word's own leaf value instead of its
existing typed concept; every production call takes that branch. The
working-tree repair resolves the typed concept for every order but 1,
keeps the literal exemption, and leaves order 1 to bind, which is §1.3 as
decided. The kind-point test is now parametrized over the overlay, and a
seven-case test pins the operand rule. The `identity_origins` check the
review asked for is added. Freeze that as source-12, run the same scoped
groups once, review, publish.

**The line otherwise matches §1.** MTM holds current values, names with
readings, and verb-and-role event histories, with a native individual
keyed by its occurrence and the bank never written; `expose_form` and
`absorb_form` are gone. Bind is a compose rule (`bind_argout1 =
bind.forward(bind_argin1)`), one menu entry per accessible location plus
the zero dummy, credited through the derivative and the score function;
determiners keep their folds and lose their variants in the menu
construction, not in the grammar text. VP results replace, descriptions
join, only the kept close publishes. Definitions reject a word as another
word's object and the N1 case is removed. No fixture port weakens a
contract; the decomposition arms keep their snapshots and seeds. Every
failed source is retained.

**Carried, not blocking.** `MediumTermMemory.token_of` falls back to an
occurrence that is some subject's event, and `forget` then drops that
event from every history; it cannot fire inside one close, since opens
and binds precede `_commit_events`, but a re-close at an event occurrence
would lose history silently. Guard it in source-12 if cheap, else in 2c.
The decomposition expansion maps three parameter arrays by position,
relying on bind being the last compose rule; a name-based assertion
closes that in a later round.

## 31. Codex: source-12 scoped gates complete — 2026-10-23

Frozen `9ccc4911c5a6792cbe97de8e80de373b03cb41ce` passes all requested
scoped groups: 1,619 passed, twelve existing skips, zero failures; all
1,631 selected cases complete across 132 distinct files. There is no
compile-cache retry. The source audit covers 880 files; source, tests and
measurement fixtures are unchanged after freeze. Both decomposition arms
retain their original snapshots/seeds, independently checked action names
and bitwise surviving parameters. The unchanged pending certificate has
forward 0 bound / 16 open, reverse 0 / 32, and no wrong binding. G1–G3,
c/e, ordinary presentation, gradient logging and multi-stream driving
pass. Every earlier failed source and its probes remain retained.

The twelve skips are nine absent phase artifacts and three opt-in slow
MPS/Inductor cases; none supplies a learning or GPU-execution claim. This
is the scoped identity pass, not step 2's full sweep. Claude's actual
landing review of this source and its receipt precedes publication.

## 32. Codex: capture the typed operand reference — 2026-10-23

Claude's actual source-12 review returns DO NOT LAND: the restored typed
value reaches composition, but identity programs skip the journal branch
that carries an implicit operand's typed reference. Explicit kind/generic
requests already resolve at capture; implicit S subjects and part/whole/
equal operands do not. For an S subject, the main reference is already
updated from `operand_refs`, but its factored reference and composition
leaf remain raw. A relation's reference also remains raw.

`typed-journal-01` retains eight deterministic failures, covering orders
zero/two and lift/part/whole/equal. The fixture passes the actual
`prepare_operands` result into an identity program and `finish_clause`.
Its raw reference is 1 and the selected typed reference is 7. These unit
cases contain no exploit/explore trials or keep, so no keep probe exists.

Under step-2 §39, share the journal's existing head/reference lookup, then
let an identity program adopt a different native reference only for a
known requested order other than one, excluding kind/generic modes. Exit
that operand's legacy branch immediately: no open, mint or formation
choices are revived. The numerical value stays the exact compose result.
The eight new certificates check main and factored references, the
composition leaf reference and the typed point. Every existing assertion
stays; no test is retired. This restores correspondence and reverses no
§1 decision. Source-13 owns a new complete scoped pass.

Two review notes are recorded precisely. The identity path's implicit
literal exemption keeps both raw point and reference; the old ordinary
path exempted only reference binding after typed lookup. This is an
explicit identity-path choice accepted in the source-12 review, not an
exact restoration of the ordinary path as §30 could imply. The
decomposition adapter also directly checks that the current unary names
minus bind equal the original unary sequence. No snapshot, seed, budget
or numerical row map changes. The event-history notes remain FutureWork.

Alec's latest instruction changes the review hand-off: finish and supply
the frozen source and receipt to Alec, who mediates with Claude. No direct
Claude review is requested for source-13, and publication waits for that
mediated review. This does not remove the review requirement.

`typed-journal-02` passes all 78 live-reference, sentence-journal and
identity-threading cases after the repair, including the eight new cases with
assertions unchanged from their pre-repair failures. This is development evidence only;
the frozen source owns every required scoped group anew.

## 33. Codex: source-13 complete; Alec mediates review — 2026-10-23

Frozen `65ab7fbe9d5c67271b0bea7e4853c8ba2b690363` passes every requested
scoped group: 1,627 passed, twelve existing skips, zero failures; all
1,639 selected cases complete across 132 distinct files, with no
compile-cache retry. The audit verifies 886 production/configuration,
test and measurement files unchanged after freeze. Both decomposition
arms retain their original snapshots/seeds, direct name checks and
bitwise surviving parameters. The unchanged pending certificate records
forward 0 bound / 16 open and reverse 0 / 32, with no wrong binding.
G1–G3, c/e, the ordinary presentation, gradient logging and multi-stream
driver pass. The eight new typed-journal cases pass on this hash.

Every failed source and diagnostic remains retained. Nine skips need
absent phase artifacts and three are opt-in slow MPS/Inductor cases;
none counts as a pass or learning/GPU evidence. No full sweep ran.

The [review hand-off](../benchmarks/2026-10-23-identity-threading/final/source-13/review-handoff.md)
and receipt are complete for Alec to mediate with Claude. No direct
review is requested for this source and no LAND verdict is claimed.
Publication of BasicModel main and the WikiOracle bump waits for that
review. No additional gates are scheduled unless review changes source.
2c follows identity's eventual landing; item 3.0 is not reached.

## 34. Claude's review of source-13 — 2026-10-23: LAND

Reviewed: frozen `65ab7fbe9`, the source-12 delegated review (DO NOT
LAND on the stored reference of implicit typed operands), §§31–33, the
review hand-off, the two production hunks since `bf44bbc47` and the eight
new cases.

**The two repairs close the typed-operand gap from both sides.** Source-12
restored the typed concept's value for operands at orders other than 1
(`ReferenceContext.prepare_operands`, the identity branch). Source-13
makes the journal store the matching reference: in `ClauseJournal.
finish_clause`, an identity program adopts the operation's reference for a
leaf only when the role is an implicit binding at a known order other than
1, not kind or generic, and the reference is a different native id; it then
leaves the legacy branch, so no formation, open or mint choice returns and
order 1 stays bind's. The eight cases pass the actual `prepare_operands`
result into an identity program and `finish_clause` and assert the main
reference, the factored reference, the composition leaf's reference and the
exact typed point, for lift, part, whole and equal at orders 0 and 2; they
failed before the repair and pass unchanged after it. Gates on the frozen
hash: 1,627 passed, twelve existing skips, no failures, all 1,639 selected
cases across 132 files; both decomposition arms on their original
snapshots and seeds with a direct unary-name check; the pending-premise
certificates unchanged. Everything the source-11 review verified is
unchanged apart from these hunks.

**One note, not blocking.** The identity branch also skips the legacy
alias resolution for a trial's newly requested singleton (`pending_order`,
through `reference_orders`), which belongs to close-time minting; that
path ends with the minting in item 3's removal round, so nothing is lost
now, but the receipt should say the branch skips it. The literal-exemption
divergence, the FutureWork event-history notes and the position-mapped
decomposition rows stand as recorded.

**Verdict: LAND `65ab7fbe9`.** Publish main and bump WikiOracle; then 2c.
