# Iteration 1 of 3.5's remaining work: the swap, the verb and the modal (Claude, 2026-10-23; for Codex)

**Status.** Item 3.5's remaining work runs as two iterations (Alec,
2026-10-23): lines 3, 4 and 5, then lines 6 and 7. Asked how much of
"operations on meaning, centroids on form" to bring forward for the verb,
Alec chose **"The whole swap now"**, so it moves from item 3 into this
iteration. The order within it:

1. **The swap** (§2).
2. **Line 4, the verb** (§3), on the swapped source.
3. **Line 3's lessons**, rerun at their original budgets and seeds on the
   swapped source. Their stop on the old source stands as the record
   (`benchmarks/2026-10-23-item3-5-stage4e-np`, source 05: stage3's source
   order 8/8 with both CE bars failed; the NP lower lesson's order-1
   occurrence 0/1, the chooser preferring `product`). The swap changes
   what composes, so its artifacts are regenerated after it.
4. **Line 5, the modal** (§4): an alternative timeline of the one
   identity; the modal withholds the lowering.

Under step-2 §39 Codex reviews this spec against the code, fixes what the
code contradicts and continues, documenting each decision; only reversing
a decision in §1 comes back to Alec. §6 is the iteration's gate rule.

## 1. Decided

1. **Operations on meaning, centroids on form, now** (Alec, 2026-10-23;
   layout spec §1 item 12). Every grammar and logical operation acts on
   the meaning block, the logical ones by their exact rules and the
   grammatical ones by learned kernels; a reduce's percept part is the
   centroid of its constituents' forms; zeroth-order concepts start as
   random codes and are moved by the priming surface; generation reverses
   the meaning composition and reaches forms through the symbol index.
2. **The verb is a group element acting on its subject's identity**
   (occurrence spec §14.1): an invertible map from the identity's state
   before to its state after; the undefined verb is the identity element
   `e`, read as unknown; a verb is a diagonal of log-eigenvalues `w`, zero
   by default, its support admitted by recurrence; the teacher is the
   log-ratio of the identity's observed after-frame over its before-frame;
   states are relations, not group elements.
3. **Bases** (Alec, 2026-10-23): a transitive verb acts in its object's
   basis, `Q(object) · diag(e^w) · Q(object)ᵀ`; **an intransitive verb
   keeps the learned verb groups** ("Keep verb groups for intransitives"):
   a shared basis per group, found by joint diagonalization, minted on
   recurrence, capacity `verbGroups` 24 (occurrence §14.1). A numeral
   object is a count, not a basis: "plus 2" is the successor applied twice,
   by thought.
4. **A copular property joins the identity's state like a description**
   (Alec, 2026-10-23): "Felix is tired" joins tired into Felix's state, and
   is an observed after-frame for the verb's teacher.
5. **Identity threading** (identity spec): a VP result replaces the
   identity's state, a description joins it, values live in MTM, event
   histories record verb and role, the graph ends at the kept close.
6. **Modals are of the same order as verbs and nouns; an MP is a
   difference of VPs as a VP is of NPs; its form is the adverb; a
   counterfactual is an alternative timeline of the one identity, an
   abstraction kept above order 0 by withholding the lowering** (Alec,
   2026-10-22 and 2026-10-23; §4).
7. No language in code: no marker is named as possibility or necessity.

## 2. The swap, in code names

**`GrammarLayer.compose_blocks` (`Layers.py`).** Today it gives each
operation's learned kernel the operands' *form* blocks and passes one
operand's *meaning* through, or applies a declared bilattice rule
(`MeaningCodes.compose`). The blocks swap roles:

- **Form block: the centroid.** The composed form is the centroid of the
  operands' forms, each weighted by the number of leaves it covers (a
  count carried beside each stack item and in the journal); no parameters.
  A contiguous event's centroid is its whole's form; a discontiguous one
  is a type over disjoint parts (Orders §9). The `.where` and `.when`
  bands keep their bracket (the `bands` helper in
  `ClauseJournal.finish_clause`).
- **Meaning block: the operations.** Operations with declared exact rules
  keep them: the logical ones (`not`, `non`, `intersection`, `union`,
  `conjunction`, `disjunction`) and `sum`. Every grammatical operation gets a learned kernel on the
  meaning block that acts alike on the positive and the negative halves
  and never mixes them (the two lanes), initialized to today's passthrough
  of its meaning head, so the meaning side starts exactly as now:
  - `lift` is §3's verb action, a map in the same chart whose weights the
    VP supplies, in place of lift's learned additive sigma on the meaning
    block (Orders §10's "lift keeps its sigma" then holds only where there
    is no meaning block; Claude's reading);
  - `verb` keeps the verb's meaning, the source of its `w`;
  - `adverb` is §3's modifier diagonal;
  - `lower`, `definite` and `universal` a pi fold on meanings;
  - the other grammatical operations (compound, preposition, the surface
    operations, product, the relation faces) one learned kernel each,
    identity-initialized; Codex lists them from the catalogue;
  - `bind` returns the identity's value, as now.
- **The form kernels retire.** Their parameters leave the model through an
  explicit, versioned migration that drops them with a drift record.
  Artifacts trained with them are regenerated, not converted; line 3's are
  regenerated anyway.

**Reverse and generation.** `generate_from_grammar_context`,
`Language._bounded_binary_reconstruction` and the reverse step: the
meaning kernels' inverses split a parent's meaning; the form side splits a
centroid exactly given one operand and the leaf counts; the candidate pair
search scores the meaning residual of the meaning kernel's fold, with the
centroid residual on forms as the second constraint, instead of the
form-kernel residual alone. `SentenceUnderstanding.readback_scores` keeps
identifying a leaf's word by its form block, now against centroid splits.

**The reconstruction objective.** `Models._reconstruct_sentences`,
`_reconstruct_trial` and the `reconstruction` owner group
(`Models.py` ≈2848): reconstruction trains the meaning kernels and their
inverses through generation, and its parameter ownership moves from the
form kernels to them.

**Configurations (corrected on checking, 2026-10-23).** Only a
configuration whose conceptual space is wider than its form has a meaning
block (`MereologicalCodes`: the complement is the conceptual width minus
the percept-event width): `BasicModel`, `BasicModel_answers_tied_benchmark`,
`MM_grammar`, `XOR_grammar`, and fixtures that widen the conceptual space,
among them line 3's lesson models (`identity_experiment.make_model`:
232 wide, meaning 128). `XOR_exact` (16/16), `MM_xor` (14/14),
`MM_math_chain` (104/104; its `meaningWidth` 32 has no room) and the
default `model.xml` (9/9) have none: their vector is all meaning and their
kernels already act on it, so the swap and §3's verb action leave them
unchanged this iteration. The XOR standing gate and the decomposition arms
run on those configurations and do not move; they must pass unchanged.

**Unaffected.** The readers that were already meaning-only (layout spec,
the side-effect audit): bind's meet, identity candidates, the reference
term, queries, placement's context codes, answer credit. **Not in this
iteration:** the stored derivation and its readers, close-time minting,
three-slot rows, LTM0 one slot and LTM1 chains, STM chunks, sentence-level
stamps, `.when` in the percept part, recall by time — item 3.

## 3. The verb (line 4)

**The action.** `lift`'s meaning rule:

    S.meaning = act(subject.meaning; w = W_v(VP.meaning), Q)

- `W_v` is a readout on the meaning width, built like
  `LiftLayer._verb_spectrum_w`: linear, zero-initialized, soft-thresholded
  at τ with the straight-through gradient, so a fresh verb is `e`.
- `act` applies the diagonal in the cube chart (`cube_chart`, as
  `_verb_rail_transform` does), the same `w` and `Q` on both halves. A
  coordinate absent in the subject stays exactly absent. With `w = 0`
  everywhere the result is the subject's meaning exactly, by construction
  (`torch.where` on the support, not a chart round trip).
- **`Q` for a transitive VP** (the VP's object handle, stack column 2,
  which `MTMOverlay.apply` already carries into the event journal, is set):
  `k` Householder reflections whose vectors are learned readouts of the
  object's meaning, confined to the object's support and the identity
  elsewhere; `k` is `<verbBasisReflections>`, default 4. The object's
  value is read from the identity overlay by that handle.
- **`Q` for an intransitive VP:** its verb's group basis `Q_g`, or `I`
  while the verb has no group. `VerbGroups` holds up to `<verbGroups>`
  (24) orthogonal bases, each identity at initialization and exactly
  invertible (Cayley or Householder parametrization; Codex chooses). A
  verb's group is host-side bookkeeping kept with the checkpoint: its
  teacher residual under its current basis is recorded at each teacher
  event; when the residual exceeds `<verbGroupFit>` on three distinct
  occurrences, the verb joins the existing group whose basis gives the
  smallest residual, or mints a group if none falls below the threshold
  and capacity remains; at capacity it joins the best and the fit is
  reported. `Q_g` learns through its member verbs' teacher terms, which is
  joint diagonalization by gradient.
- Without identity threading (no overlay), `Q = I`.

**The teacher.** Within a sentence the overlay's values are pathwise, so
the reference term and output prediction reach `W_v` and the bases through
the derivative. Across sentences the graph ends at the kept close, so the
log-ratio teacher of §14.1 is an explicit term:

- `MediumTermMemory.Event` gains `before` (the subject's value captured at
  its bind, the snapshot the identity line already scores) and `basis`
  (the object's meaning at the event, or the verb's group), both detached.
- When a later kept close gives identity I observed content (a description
  mention joining, a copular property joining, or an equality's
  complement) and I's latest event has verb `v`, the term `verb_teacher`
  is the mean squared chart difference between `act(before; W_v(v), Q)`
  and I's newly observed value, over the coordinates defined in both.
  `before`, the observed value and the verb's row are detached, so only the
  readouts and bases learn; concept rows stay placed by the priming
  surface. Its weight is a model.xml knob in the Error registry, like
  `reconstruction.reference`.
- At its optimum `w` is the log-ratio in `Q`'s coordinates; the soft
  threshold admits a coordinate only when the push recurs across
  occurrences.

**The copular join.** In `MTMOverlay.apply` and at the commit, a `part`
relation whose part is a bound identity joins the whole's meaning into
that identity's state, as a description mention does; equality still
replaces. Claude's reading, for Codex to confirm against the catalogue: a
`whole` relation mirrors it, joining the part's meaning into the whole's
identity ("the box contains apples").

**The modifier.** `adverb(VP, ADV)`: `VP'.meaning = act(VP.meaning;
w = W_m(ADV.meaning), Q_m)`, with `Q_m` one shared learned basis,
identity-initialized, so a modifier starts as `e`.

## 4. The modal (line 5): an alternative timeline of the one identity

**The discussion, in Alec's words (2026-10-23).** "The question of modals
is exemplified by 'what is the meaning of possible vs necessary'. The
multiple different timelines approach as a fifth dimensional ordinate
over events seems great; I had previously envisioned two NP creating a VP
as a difference, and two NPVP creating a MP as a difference. But perhaps
that is about learning the MP, and the form of the MP can be as an
adverb?" Then, on Claude's first mechanism: "we seem to be baking
counterfactuals into MTM. Do they exist in other places? I would assume
they are created naturally in concepts higher than order 0, since they
are abstract." Then: "So it would take recombination of abstraction to
create something not present in reality." Then: "MTM still holds
identities, which are connected NPs or things. But it also holds
timelines, which are connected VPs happening to a single identity."

**Withdrawn:** Claude's "possible identity" mechanisms (a bind-menu entry;
then a learned actuality `p(m)` with a `possible` record kind). They put
the branch in the wrong unit. The unit of a counterfactual is a timeline,
not an identity. Claude's claim that MTM holds only stitched percept parts
is corrected: MTM stitches perceived occurrences, NP mentions into an
identity and VP events into a timeline, and holds working state for each,
while holding no concept row.

**Decided (Alec, 2026-10-23).**

1. **One identity, several timelines.** A counterfactual is an alternative
   timeline of the one actual identity: Joe stays Joe, with his connected
   NPs, names, kind and promotion; what branches is the chain of VPs on
   him. `Token` gains `timelines`, a tuple of (events, value); today's
   `events` and `value` become timeline 0, the actual, whose value is the
   identity's current state; an alternative timeline's value is the
   identity's state along that branch, which "he would have been tired"
   reads.
2. **Recombination, kept above order 0** ("Good correction"). A modal
   clause is the ordinary composition, the same `lift`, `verb` and `lower`
   kernels on the identity's state, with the lowering to order 0 withheld,
   so its result is an idea at order 1 or above: an abstraction of a
   possible event, created in concepts higher than order 0 and not present
   in reality. It is written as an idea at that order and never as an
   order-0 occurrence of the subject. The order is the mark; there is no
   record kind. "Did Joe run?" reads order-0 occurrences; "could Joe run?"
   reads the abstractions and the alternative timelines. Only NP + VP
   lowers; Orders §10 is amended, reversing 2026-10-22's "VP + MP lowers".
3. **Which timeline a clause extends.** A lowered event (order 0) extends
   the actual timeline and updates the actual state. An unlowered event
   extends an alternative timeline of its subject and updates that
   timeline's value, never the actual. Whether the modal applies is the
   chooser's choice of the `adverb` operation, learned by exposure like
   any attachment; nothing names a marker.
4. **Continuation (Alec: "Ok, but make a note in the doc that we are doing
   this instead of creating a new timeline for every 'could have'").** A
   modal clause continues its subject's most recent open alternative
   timeline, and opens a new one only when none is open. So "Joe could
   have run. He would have been tired." is one branch, and so is "Joe
   could have run. Joe could have swum." unless the chooser separates
   them. **This is a default adopted for simplicity, not a claim about
   discourse: the alternative, a new timeline for every "could have", was
   considered and not taken.** If exposure shows two counterfactuals of
   one identity must be kept apart, the chooser gains the choice to open a
   new timeline (an action, credited like the others), and this note is
   the record of why it did not have it first.
5. **Possible and necessary.** Possible: the event holds on some timeline.
   Necessary: on every timeline of the identity, the actual included; in
   the limit over a kind, the generic ("cats breathe"), two truths'
   relative truth at the kind level. The dual holds in the lanes.
6. **The second difference, as a teacher.** Two NP-VPs of one identity at
   one time on different timelines: "Joe walked" (actual) and "he could
   have run" (alternative). The MP's diagonal (`W_m`, `Q_m` of §3) maps
   the actual event to the alternative; across sentences `modal_teacher`
   compares them at equal `when`, as `verb_teacher` compares before and
   after; within a sentence the derivative reaches it as it reaches the
   other kernels.
7. **Lifetime.** Alec: "Psychologically, we get attached to identities,
   which suggests some identities (like self) rarely if ever go away. But
   perhaps there is a path from MTM to concepts of LTM?" There is, and 2c
   built it: three distinct sentences promote a token to an order-1
   concept in the index (`MemoryConsolidation.consolidate`), whose row
   persists under forgetting by value, and a recalled or re-mentioned
   native individual re-enters MTM keyed by its occurrence with the
   concept as its `native` (`ReferenceClose`, the identity line). So an
   MTM entry expires while its identity persists as a concept, and
   attachment is re-witnessing: an identity mentioned in nearly every
   sentence, the self the limiting case, is promoted at once, has its
   value refreshed at every close, and is never selected by forgetting.
   Alternative timelines expire with their token and at the document's
   end (2c's teardown); their events persist as ideas at order 1 or above
   in LTM0, composed over the identity's symbol, so a recalled identity's
   counterfactuals are reachable by reference. Promotion folds only
   order-0 occurrences into the concept's content; the abstractions are
   attached by reference, not folded in.
8. **MTM is a source for conceptual space, and stays one** (Alec,
   2026-10-23: "composed over the identity's symbol means that the mind or
   conceptual space has access to MTM symbols. In Yogācāra terms, that is
   what creates the bījas in the ālaya: without that witnessing, we would
   not become attached to identity"). Two channels, both present today and
   both required of every later layout: a bound mention composes over the
   identity's current value (`BindLayer` reads it from the menu staged by
   `MTMOverlay.stage` from `ReferenceClose.token_value`), and the idea the
   close writes keeps a reference to the identity (the mention occurrence
   that `names` maps to the token, or the promoted `symbol`), while the
   kept close writes back into MTM (`mtm.open`, `bind`, `update`, `event`
   in `ReferenceClose.commit`). An alternative timeline's ideas are
   composed over the identity in the same two ways. The layout round
   keeps both when it compresses a sentence to one idea.
   *Integration (Alec, 2026-10-23): the overlay is a transition
   implementation; the target is to prepend MTM symbols to the symbol array
   conceptual space reads, before the order rungs, so identities are
   available to compositions at every order and bind composes two field
   positions (identity spec §2, "Integration by prepending"). Built in
   item 3's STM round unless pulled forward; this iteration keeps
   timelines in `Token`, mirrored by the overlay, so they migrate.*

## 5. Certificates

The swap:

1. The form block is the leaf-weighted centroid; a contiguous run's
   centroid is its whole's form.
2. Every grammatical operation's meaning starts as today's passthrough,
   exactly; no learned kernel mixes the lanes.
3. The reverse: a centroid splits exactly given one operand and the
   counts; a meaning kernel inverts where it is invertible.
4. The swap's certificates run on a paired configuration; the XOR
   standing gate, the decomposition arms (both on configurations without a
   meaning block), G1–G3 and the pending-premise certificates pass
   unchanged.

The verb:

5. Zero support is an exact passthrough; `unapply` with the same verb and
   basis recovers the subject's meaning; lanes never mix; absent
   coordinates stay absent.
6. One verb and subject with two objects of disjoint support give
   different results, each confined to its object's support.
7. An intransitive verb whose residual recurs mints a group; a second
   verb joins it; capacity is reported.
8. "The cat ran", then "the cat is tired": the copular property joins and
   fires `verb_teacher` once; `w` moves toward the log-ratio on defined
   coordinates only.

The modal:

9. With no `adverb` application, timelines, identity values and every
   existing assertion are unchanged; `Token.timelines` has one timeline
   whose events and value equal today's.
10. A modal clause writes its idea at order 1 or above and no order-0
    occurrence of its subject; the actual state and timeline 0 are
    unchanged; the subject's alternative timeline gains the event and its
    value is the VP applied to the state it branched from.
11. A second modal clause about the same subject continues the open
    alternative timeline (§4 item 4), and its VP acts on that timeline's
    value: "X could have run; X would have been tired" leaves X's actual
    state untouched and the alternative's state changed twice.
12. "X walked" (actual) and "X could have run" (alternative) at equal
    `when` fire `modal_teacher` once, moving the modifier's diagonal toward
    the map from the actual event to the alternative on defined coordinates
    only; a modifier with no alternative event teaches nothing.
13. A modal marker reaches both bracketings through `adverb` in a
    fixed-operation fixture; "did" reads order 0 and "could" reads the
    abstractions in a fixed-operation query.
14. Lifetime: an alternative timeline expires with its token and at the
    document's end; its ideas remain in LTM0 at their order; a promoted
    identity's concept folds order-0 occurrences only.
15. Line 5's learning gate (todo): generated corpora with a modal marker
    as a free leaf and as a verb affix; "did" and "could" questions
    answered from order 0 and from the abstractions; budgets and seeds
    fixed in advance; reported, not rescued.
16. Source: an idea written from a clause with a bound subject has the
    identity's value in its content (the bind's operand equals
    `token_value` at the bind) and a reference to the identity (its
    mention's occurrence maps to the token, or its promoted symbol);
    an alternative timeline's idea has both as well.

## 6. The iteration's gate rule (Alec, 2026-10-23: two iterations)

- Development runs keep a log only; a receipt starts at the freeze.
- The swap and the verb are frozen once together, before line 3's lessons
  rerun on that source (artifacts need a frozen source); the modal is
  frozen once after; then one review of the whole iteration through Alec,
  then publication.
- Scoped gates sized to the change: the changed modules' tests, §5's
  certificates and the standing ones (G1–G3, the decomposition arms, the
  pending-premise certificates); the twelve-draw series only if binding
  changes.
- One full sweep at the end of 3.5, after lines 6 and 7.

## 7. Answers and open points

- *Meaning side* (Alec, 2026-10-23): "The whole swap now."
- *Intransitive basis* (Alec, 2026-10-23): "Keep verb groups for
  intransitives."
- *Copular state* (Alec, 2026-10-23): it joins like a description.
- *The modal* (Alec, 2026-10-23): an alternative timeline of the one
  identity; the modal withholds the lowering ("Good correction");
  continuation default noted ("Ok, but make a note"); lifetime through
  promotion (§4).

## 8. Code review before implementation (Codex, 2026-10-23)

Claude's supplied five-document edit is committed unchanged at
`40c541fb53`. A subsequent concurrent edit withdrew §4's commit-time
possible identities and marked line 5 held. Alec has been asked which
version governs that line; the independent swap, verb and lesson work
continues. The earlier lesson stop remains evidence on its own source.

1. **The corrected configuration scope agrees with the code.**
   `GrammarLayer.__init__` records `meaning_layout` only for embedding
   configurations with conceptual width greater than PartSpace width.
   Keep that boundary. The numeric configurations and their random draws
   do not acquire new modules. The versioned grammar catalogue, rather
   than a global tensor-width heuristic at restore, identifies the swapped
   operators and reports retired form parameters and initialized meaning
   parameters, including optimizer disposition.
2. **A compose adapter alone is insufficient.** `compose_blocks` is used
   by the live chooser, but bounded pair reconstruction and its eligibility
   test call `compose` directly and score only the form slice. The binary
   inverse dispatch also reads the old `_sigma`/`_pi` implementation.
   All three paths must use the same swapped fold and its inverse.
   `readback_scores` continues to identify terminal words by form.
3. **Leaf counts are missing.** The six-column identity stack carries
   identity, predicate, object, resolved, word and source; none is a leaf
   count. Carry counts beside that stack and through the operation journal;
   unary operations preserve them, binary reductions add them, and
   transpose moves them with the value. A bare operator call defaults to
   two single constituents. A reverse with supplied counts solves the
   centroid equation; free candidate search must account for its proposed
   children's counts instead of borrowing a retained derivation.
4. **The actual meaning-head contract is narrower than the prose.**
   Today's meaning passthrough is `meaning_head_role`, default left,
   independent of grammatical `head_role`; lower explicitly selects the
   right meaning. Preserve this exact initial meaning, including the
   relation faces, unless §3 explicitly replaces it. The grammatical
   catalogue includes synthesize, product, lift, verb, compound, adverb,
   lower (its definite/universal faces), surface, preposition, their unary
   faces, min/max, isEqual/isPart, equal, part/whole/assertPart, implies
   and the identity/wrapper faces. Structural bracket actions and
   transpose are not reductions. Interpret and bind retain their declared
   reading/value behavior. The logical rules remain exact.
5. **Support and the inverse need an explicit construction.** Merely
   rotating a sparse vector then masking absent coordinates loses the
   inverse: the inverse of a restricted matrix is not the restriction of
   its inverse. Use the positive-definite action
   `Q diag(exp(w)) Qᵀ` restricted to each lane's defined coordinates;
   `unapply` solves that same restricted system. Q and w are shared by
   the two lanes, and a transitive action is identity outside the object's
   support. Zero w uses an exact passthrough with the action's derivative;
   it must not cut the zero-initialized readout's training path. This
   reconciles §1's invertibility with §5's exact absence requirement
   without mixing lanes or inventing evidence at an absent coordinate.
6. **The object handle is already available; the group is not.**
   `MTMOverlay` stages values before the numerical loop and carries the
   VP object in stack column 2. Stage group assignments with the word
   readings at the same host boundary; never read the pool or a Python
   assignment dictionary inside the compiled body. Group bases use
   Householder pairs, initialized to the identity. Membership and fit
   witnesses are host checkpoint state; bases and readouts are parameters.
7. **Copular updates and across-sentence teaching are new code.** The
   overlay currently updates lift/equality, not part/whole, and
   `MediumTermMemory.Event` contains only occurrence, verb, role and time.
   Add detached before/basis snapshots at kept commit, and consume each
   later observation once. Register the teacher through the existing
   Error registry and reconstruction parameter owner; no private optimizer
   or backpropagation into retained concept rows. `part(A,B)` joins B into
   A's identity; `whole(A,B)` is its converse and joins B into A's identity
   when A is the whole participant. Equality keeps replacement semantics.
8. **Gate and evidence boundaries.** Development runs retain logs only.
   The swap/verb freeze starts its receipt and precedes lesson generation;
   lessons retain their original schedules and stop at the first failed
   behavior. The old-source results are not promoted. The modal freeze,
   if authorized after the concurrent §4 revision, has its own scoped
   evidence. No full sweep is run in this iteration.

## 9. Swap and verb implementation decisions (Codex, 2026-10-23)

The development runs below are logs, not frozen-source receipts. The
concurrent modal edits now specify alternative timelines, replacing both
of the earlier mechanisms. Alec has been asked to reconcile that change
with the explicit p(m) hand-off; no modal implementation is included in
the swap/verb work.

- **Catalogue and call paths.** `MeaningKernels` owns the parameter-free
  centroid, a shared-lane chart-affine head map, and the diagonal action.
  The grammatical catalogue's direct compose calls, structural adapters,
  unary faces, free pair search and reverse dispatcher use these maps.
  Compound's paired face no longer requires the retired form-side case
  fold to be available. Its unpaired face still does. Duck-typed numerical
  operators without a `meaning_layout` remain on their existing path;
  development 12 exposed this missing optional-attribute guard.
- **Inversion.** A paired binary action needs a context witness or a
  candidate bank. No balanced pseudo-split replaces its missing argument.
  Free search fits the centroid mixture from each candidate pair, scores
  both the meaning and form residuals, and recovers the selected head's
  meaning through the inverse. Selected symbols supply exact forms;
  this avoids a floating-point centroid round trip changing terminal
  identity. Known-context inversion solves the centroid with supplied
  counts. The generation owner and reconstruction parameter group are
  unchanged; their shared modules now contain the meaning parameters.
- **Versioned retirement.** Grammar catalogue `meaning_kernel_schema=1`
  reports every retired paired form parameter and every initialized
  meaning parameter. Positional grammar alignment runs first, including
  aliases; optimizer moments for retired or newly initialized parameters
  are discarded. A schema-1 checkpoint is never silently reinitialized.
  The configurations without a meaning block acquire no new module.
- **Verb observations and optimization.** Event checkpoint version 4 adds
  detached before/basis snapshots and a once-only teacher mark, accepting
  legacy four-field events. Re-observing an event preserves that mark.
  The observation is admitted at the kept close. The existing paired
  sentence optimizer has already stepped at that boundary, and further
  sentence steps may follow in the batch. Consequently the close retains
  only detached teacher inputs; after those paired steps the ordinary
  batch Error registry builds the teacher graphs at one parameter version.
  There is no extra optimizer, update or graph crossing a close. The
  teacher is an explicitly weighted reconstruction term (default 0.1),
  and neither its before/after observations nor its verb row learns from
  it. An evaluation close can retain history and fit witnesses but creates
  no teacher gradient. Group assignments and distinct recurrence witnesses
  are checkpointed host state, staged as tensors before composition.
- **Copular content.** `part` and `whole` join the other participant into
  their left participant's working identity; equality keeps replacement.
  The event journal and kept commit carry that same state. The new
  two-sentence certificate uses frame staging, fixed operations, the real
  journal, preview, writer and commit: a subsequent copular observation
  teaches the previous verb once. Separate tests cover the absence mask,
  object support, group capacity, inverse and deferred graph boundary.

**Existing-test ports under step-2 §39.** No test is retired. In
`test_operators_round2c.test_zero_meaning_departures_and_one_reader_step`,
for both paired configurations, replace the compose arm's unequal-root
assertion by exact equality: their zero meanings remain zero and both
exact operations now carry the same centroid. Replace its unconditional
comparison weights `[.5,.5]` by those weights on an actually selected
compose departure, and kept-only weights otherwise. The retained probe
shows XOR's four rows have zero eligible binary forks, `compose_round=-1`,
tied R+A and kept-only exposure; MM still has compose departures. All
activation, evidence, keep, update-count and optimizer-step checks stay.
In `test_mind_generativity`, the training fixture supplies the known right
operand to `reverse_blocks` instead of calling the retired form inverse
without a witness. Its free stored reads still receive only the bank,
and every assertion stays.

**Development failures retained.** Runs 01/06 corrected new fixture API
uses (read-only config property; Error's objective keyword). Run 08's
new inverse-gradient fixture had both a zero chart context and a live
forward/inverse cancellation; use a nonzero context and an independently
supplied parent, preserving its assertions. Run 10 collected no tests
because a selector did not exist. Run 12 exposed the reverse dispatch
compatibility defect and the two ports above; 13/14 retained the
no-departure comparison exposure before its port. All development logs
will ride with the frozen receipt; none substitutes for its gates.

**Resumed after Alec's pause (2026-10-23).** The updated hand-off releases
§4's alternative timelines after this freeze and line 3's rerun. Claude's
ten-document update is committed unchanged at `82191630d6`; the parent's
Ethics amendment stays for the publication bump. Development 18 was
interrupted at Alec's pause (ten passed); its log remains. Resumption 19
passes both previously failing exact-lane assertions unchanged. Their fix
is to recover the independent stance of each signless candidate before
scoring the exact logical fold, not only after choosing the pair. Invalid
out-of-cube inverse solutions are not valid candidate decompositions.
A direct numerical `LiftLayer(3,3)` in that run exposed a width mismatch:
an explicitly smaller numerical operator must not inherit its enclosing
configuration's wider meaning block. The layout is now admitted only for
the configured conceptual width (or a width-free grammatical face).
Run 20 passes that repair and the stored-generation file, with no assertion
port. The source audit retains the unchanged four out-of-scope XML files.

## 10. First frozen checks: inverse boundaries and retained fixtures (Codex, 2026-10-23)

The swap/verb production commit is `b7487f8821`; the unchanged additional
Philosophy edit is `374d7e0aa6`, the first gate hash. Certificates passed
16/16; the native set completed 130 cases (122 passed, seven prerequisite
skips, one failure); the reference regressions passed 110/110; the operator
set completed 192 cases (191 passed, one prerequisite skip). Full pending
captures and keep probes are retained, and both orders passed. The
reconstruction run exposed eight more failures in the slow historical
migration file. It was interrupted while the next file was running so the
failed source could be repaired; its completed results and interruption
remain in the receipt. Unrun groups supply no evidence.

1. **Paired direct inverses.** The synthesis forward had moved to a
   centroid and meaning kernel while its direct reverse still returned
   the old chart-sum split. Refuse that split without a context or candidate
   search, as the structural reverse already does. The same audit closes
   two bypasses: paired `EqualLayer.forward` uses its meaning kernel, and
   its direct reverse and compound's old case-bank reverse cannot return a
   retired split. Known-context `reverse_blocks` and the bounded search
   remain the actual inverses. Numerical configurations are unchanged.
2. **The synthesis fixture's old form equation is superseded by §1.1.**
   Its four-coordinate layout is unchanged. Initialize its declared
   two-lane kernel, supply the known right operand, and recover the other
   form by `2 * parent - left`, instead of chart subtraction. Keep its
   cube, recomposition and recovered-operand assertions. Its name now
   states the centroid/meaning contract. No input or capacity is resized.
3. **A diagnostic byte score must accept bytes.** Native input
   reconstruction now returns integer surface identities. The optional
   `reverseReconstruct(target=...)` diagnostic reached MSE with a Long
   tensor. Promote the two surfaces to their shared floating type before
   computing the existing diagnostic. Byte surfaces carry no conceptual
   where/when bands; do not slice those bands from a byte string. Do not change any training term,
   weight, target flow or keep comparison.
4. **The old supplied-traversal probes called the free-R entry point.**
   A compose journal no longer directs free reconstruction. Port the two
   explicitly supplied three-leaf inverse fixtures to
   `DerivationReconstruction.reconstruct`, the existing supplied-text
   reader that still owns those co-operands. Their shapes, distinct
   occurrence activations, pre-fold structure, recovered values, root-damage
   and inverse-error assertions stay. Their local error is the supplied
   inverse's coordinate discrepancy, not a claim about the free-R term.
   The signed fixture now shifts its values by .5 into the identity cube,
   preserving the three distinct activations and all assertions. No production
   reader is changed and no journal is reintroduced into R.
5. **Other historical fixture/API ports.** The verb witness fixture's
   random values enter the identity cube by adding .5 and its nonzero spectral
   weight is .002 instead of .2, avoiding saturation after that shift; its seed, widths
   and three inverse assertions stay. The zero-valued ownership probe
   installs its observer after the paired sentence run and clears only
   the reconstruction cache before capture of the four-item execution adapter
   (the full adapter would republish the old completed result); the unchanged `len(calls)==1`
   now measures creation of that owned result, not all paired-trial
   scorings. The enablement fixture explicitly starts disabled on the model, then enables the same path; the retired
   config flag no longer disables reconstruction for a retained grammar reading. The enablement and root-damage
   fixtures supply three native bank words at the lexical positions as their
   terminal field, and STOP to the native free decoder; no intermediate
   transformed word or whitespace slot is presented as an exact terminal.
   Root damage stays in the cube and updates the sentence root as well as
   the final field. The compiled derivative fixture also enters the cube. These behavioral fixtures use the
   established eager-reading seam; the explicit compiled inverse probe
   still calls `torch.compile` itself. No fixture is resized.
6. **Two assertion ports in the slow migration file.** The selected sum
   inverse is `2 * parent - witness`, so its parent derivative is 2:
   `expected=[1.,2.]` becomes `[2.,2.]`; the inactive row's two passthrough
   derivatives remain 2. The native surface is now a byte identity with
   variable length: its damaged-surface check uses `not torch.equal`
   instead of `not torch.allclose`, so different byte lengths are compared
   as different surfaces. No other assertion is weakened or retired.

These are §39 repairs and fixture ports within the decided swap and the
existing free/supplied-reader separation. They do not reverse a §1
decision. Every old/new assertion and fixture change rides with the final
source audit; the failed first freeze is not promoted as passing evidence.

**Development follow-up.** The fallback free reader's bank contains exact
WORD/OBJECT copies. `decoder_eligibility` counted two identical terminal
codes as two explanations and rejected even the supplied primitive. Its
existing rule is about *different* tied one-code explanations: identical
copies now count as one, while different tied codes remain ambiguous. A
new regression checks both. This changes neither the candidate bank nor
any binding menu, and applies to the shared decoder; the unchanged XOR
and decomposition assertions must certify it on the next frozen source.
Development 21/23/25 and the two diagnostic probes retain each failure.

The development process that predated the duplicate-terminal repair was
interrupted after three completed compile tests (653 seconds); its partial
run and warnings remain. A second development experiment staged the two
historical packed-reconstruction stress fixtures eagerly before compiling
their reconstruction, but was interrupted before either completed. Its
patch is retained as an unlanded experiment; that test file is restored
unchanged. Neither partial run certifies the stress tests. The changed
meaning action has its own eager/fullgraph gradient certificate, and the
native fullgraph word-chunk file is still fully opted in.

**Scope correction before the repaired freeze.** The roundtrip file mixes
loss-band/dispatch checks with historical three-epoch free-reconstruction
learning bars on the unchanged MM configurations. Those learning bars are
item 0, not this mechanism landing (§6). The initial development invocation
incorrectly opted the whole file into RUN_SLOW; its partial run (nine cases
completed before interruption) is retained. On the repaired source that
file runs under its normal default slow boundary as `legacy-loss-smoke`;
the separate reconstruction migration and objective files remain opted
in; the unchanged compile file keeps its default slow boundary and runs
its three general compiler contracts. The native fullgraph word-chunk
file and compiled meaning-action certificate remain opted in, as do XOR, decomposition,
G1–G3 and pending-premise gates. No assertion or skip marker is changed
for this scope correction, and no interrupted learning run is called a
pass. The original-budget line-3 and modal-corpus learning runs remain
exactly the requested reported gates, with no rescue.

## 11. Lesson restore repair (Codex, 2026-10-23)

The repaired swap/verb source `d3450e16e8` passed every scoped mechanism
group, both decomposition arms and XOR. Its original warm-up completed
76 presentations and 38 paired updates. Stage 3 then failed **before its
first presentation**: `VerbGroups.get_extra_state` returned a dictionary,
where the native weight preflight and the lesson's exact-retention adapter
require tensor envelopes. The failed restore, initial state and warm-up
are retained on that source; this is not a failed behavior gate.

Use the existing `_pack_state`/`_unpack_state` envelope for the group's
host metadata and declare it variable metadata at native restore, just
like occupation counts. The initial dictionary envelope remains readable
through the module's checkpoint preparation. No learned tensor, group
assignment, witness, lesson budget, seed or assertion changes. A native
save/load regression checks a minted group and a pending fit witness,
including the first freeze's envelope. Regenerate the warm-up and resume
the original lesson sequence on the repaired frozen source. This is a
§39 persistence repair, not a rescue of a learning result.

## 12. Working-view port and stationary reconstruction (Codex, 2026-10-23)

Source `f83d0081bf` passed nine scoped groups; identity/driver completed
182 passes and one failure. Its captured failure in
`test_one_native_sentence_owns_only_its_end_state_after_the_boundary`
reproduced: a thought filled a stored question's working reading while
the test compared that reading with the raw occurrence slots. Use
`store.row(row)['meaning'].roles` for both storage layouts, instead of
using it only for Part rows and `store.slots[row]` otherwise. The equality
assertion and every other assertion stay verbatim; the written row is not
changed. The captured replay then passes. This is a §39 expected-view
port, not a change to the binding or storage contract.

The original warm-up passed. Stage 3 restored, but its first two training
batches each took roughly 150 seconds and the process reached a 33.4 GiB
footprint on a 36 GiB host. The run was interrupted during its third
training batch; initial, last-batch and failure states and all keep probes
are retained. No behavior gate had completed, and the partial run is not
learning evidence.

A first-batch profile found 1,089 reconstruction trips. Once every live
top had no legal inverse or terminal, the walk repeated exactly the same
pending state to its budget. Stop at that fixed state, retaining its
pending depth, empty trace suffix and already allocated full-budget random
draws. A regression compares the complete old-budget result, trace, RNG
state and input pullback, with one readable row and one pending row, in
eager and fullgraph execution. Nothing admits a new move or changes a
budget, candidate, residual, objective or keep. The first-batch diagnostic
falls to 2.3 seconds with the same four cross-entropies and two paired
updates; the original-budget lesson must still run on the new freeze.

Two exact action simplifications also stay: the modifier uses its one
shared basis without expanding it per candidate, and an eager action
with no object and no assigned group uses the identity basis directly.
The general compiled path and eager shortcut have matching outputs and
pullbacks. A separate selected-face graph experiment gave no useful speed
improvement and is withdrawn; its patch and logs remain in development.
There are no other existing-test assertion changes or retired tests in
this repair. Development checks: 54 passed and one existing prerequisite
skip; the captured end-state replay passed separately.

## 13. Paired determination at the close (Codex, 2026-10-23)

Frozen source `deb8bfbbc9` passed all ten scoped groups. The original
warm-up completed. Stage 3 stopped while closing its eighteenth training
batch, after seventeen completed batches, before any final behavior gate:
`PartClosing` required a `lower` result to equal its description. The
retained keep shows row 1 choosing the explore reading `lower(word,
ask(word))`, with R 3.056869 against 3.076869 and equal A. Both readings,
full initial/last-batch/failure states and the failure remain in source 04.
A fresh model restored from the last-batch checkpoint did not reproduce
the whole-batch failure; that diagnostic is retained and is not certified
as an exact replay. A fixed-operation test on the production paired model
reproduces the writer failure both for a standalone determination and
when a relation consumes it.

The old inline alias is valid for the unchanged configurations, whose
lower is an exact projection. In §2's paired configurations the result
has its own form centroid and learned meaning. Close that result as the
ordinary binary fold it now is: retain its operand pair, keep the fused
occurrence value, and seat it only when a containing relation needs it as
an argument. The second planning visit must also retain that result,
rather than aliasing it back to the description. No stored layout changes.
The old rejection of an altered exact-projection lower remains unchanged
and passes. No existing assertion is changed or test retired.

The longer learned walks also exposed a resource cost: the eager decoder
retained the inverse graph of every menu option on every trip, although
its choice is hard and has no straight-through gradient. Compute the same
menu and legality without a gradient, then recompute the selected inverse
with the live parent and candidate inputs. The selected value path keeps
its gradient; no MTM read is detached. Preserve operator diagnostic
activations across that recomputation. Compiled execution keeps the full
menu path. A regression compares outputs and pullbacks to the parent,
candidate bank and meaning-kernel parameters against the full menu; the
existing eager/fullgraph walk comparison still checks traces, pending
state and RNG. No choice, inverse eligibility, budget, loss, keep law or
learning assertion changes. Development logs precede the next freeze;
all original lessons and scoped gates must run again on that hash.

Preserve the first menu's zero-valued graph contribution as well: rejected
operators retain their zero gradients, rather than acquiring absent
gradients that would change a momentum update. The new regression checks
that distinction and the complete optimizer step against the full eager
menu for a terminal and a compound. Its initial compiled reference exposed
an existing eager/compiled distinction in unused-gradient materialization;
use the full eager menu as the reference for this eager-only change. The
assertions stay; the earlier failed setup checks are retained. Final
focused development check: 46 passed, one existing prerequisite skip.
The optional timing diagnostic completed the model call but failed while
serializing its report; it supplies no timing or learning claim.

## 14. The pending-walk fixture's duplicate codes (Codex, 2026-10-23)

Frozen source `d37ee1824` passes certificates, native, reference regressions,
operators, XOR and both original decomposition arms. Reconstruction
completes 162 cases with one failure in the newly included output-walk
file. Identity/driver, legacy loss smoke and the lessons have not started
on this hash; no earlier source supplies them.

`test_walk_reports_truncation_when_work_is_pending` supplies two copies
of exactly the same terminal code and expects no emit. That setup predates
§10's correction: duplicate WORD/OBJECT copies are one explanation, not
an ambiguous pair. Its captured replay confirms that STOP is the sole
legal first move, with one emit and the lower slot still pending. The
writer and selected-inverse repair are not involved.

Port the fixture to two distinct, equally poor candidate codes: the top's
first two coordinates are (.8, .8), and the two bank codes' first two
coordinates are (.1, .2) and (.2, .1), with their remaining coordinates
zero. Neither is a terminal explanation or a supported decomposition.
Keep its width, two bank rows, validity mask, budget, policy preference and
`assert bool(truncated.all()) and n_emitted.tolist()==[0]` verbatim. The
comment now describes unresolved work, rather than claiming an identity
unary must consume three trips. The captured replay passes after this
fixture-only port; no production code or other assertion changes. Failed
probe setup attempts and both semantic replays are retained in development
53–56. The next hash runs the scoped groups and original lessons afresh.

## 15. Modal code review before implementation (Codex, 2026-10-23)

The released §4 replaces both withdrawn possible-identity proposals. The
unmodified Claude edits were committed before this work. Nothing here
changes a §1 decision. This review follows the swap/verb mechanism freeze
and the line-3 rerun; its source-specific result is in the receipt.

1. `Token.value` and `Token.events` currently own only the actual state.
   Make `timelines` their owner and keep those names as timeline-0 views.
   Version the checkpoint, validate the complete pool before replacement,
   and migrate the old one-timeline state. Pool capacity, age, names,
   native identity and document teardown stay on the token, not a branch.
2. `MTMOverlay` has one value array, a seven-column stack and a six-column
   action journal. Stage the latest alternative beside the actual value
   and carry its numerical composition while the parse is open. Neither
   branch is an extra bind candidate or identity. A preceding adverb on
   the VP selects the alternative subject value for lift; an adverb on an
   already-composed S uses that S's alternative evaluation and cancels its
   provisional actual event. Transparent wrappers preserve that scope.
   The committed timelines stay in Token; no new persistent overlay owner
   or all-order identity integration is introduced.
3. The current journal retains each completed S and `node_order` preserves
   its subject stamp. Actual NP+VP must instead lower to order 0, with
   adverb withholding it. The marker is the selected operation and the
   resulting order, never a word or a new record kind. An inner S covered
   by that modifier must not escape as an actual companion row.
4. `ReferenceClose._commit_events` currently updates the token from the
   final overlay value. Preserve that final update, including later mention
   joins, and also retain each selected event's before/after snapshots so
   the teacher has the correct correspondence between multiple events. Commit into actual or latest alternative
   by the recorded grammatical order, keeping the subject's mention
   reference in the written idea. A bound value still enters composition
   through the existing bind menu; no detach is added to that read.
5. `VerbLearning` already admits kept observations once and defers their
   numerical graphs until after the paired updates. Use the same boundary
   for `modal_teacher`, matching actual and alternative subject events at
   equal `when`. Only a retained modifier with both event values supplies
   this teacher; an absent branch supplies no target. Its term trains the
   existing adverb action and does not enter the sentence's keep.
6. The occurrence reader currently has no order restriction. Fixed-order
   queries need to distinguish order 0 from abstractions, and timeline
   evidence needs the two-lane some/every fold, including actual timeline
   0. Carry the grammatical request's order through the existing query
   metadata; no lexical name recognizes a modal or a question marker.
7. Consolidation currently folds one-slot mention rows, including order-1
   mentions. Keep the three-distinct-sentence threshold, but fold only
   the identity's order-0 occurrences. Alternative ideas remain reachable
   through their subject reference and are excluded from concept content.
   The existing promotion fixture will supply actual occurrences for its
   three mentions; its membership, count and content assertions remain.
8. Generated-corpus protocol is fixed before the modal freeze, including
   its budgets, corpus partition and initialization state.
   The two forms are a free modifier leaf and a verb affix; exposure alone
   chooses the parse, bind and adverb. Failure is reported and carried to
   item 0, without seed, budget, forcing or menu rescue.

The paired `lift` path needs the subject's stamp minus one for NP+VP;
`result_order` currently follows lift's declared VP head, while the writer
separately follows the subject without lowering. These cannot both be the
new order mark. Record the chosen event order in the paired timeline
journal and let the writer use it. Unpaired numerical configurations keep
their existing contracts and gates. Relative description joins remain
observations, not alternative events merely because their rows have an
order-1 stamp. Only the selected modifier withholds the predication.

The generated protocol and any existing test ports are listed before the
modal freeze. Checkpoint ports change only the actual-timeline access path;
old versions still load as a single actual timeline. There is no separate
saved alias for `value`/`events`, and no test enshrines the overlay's current
order-1 bind limitation. The modal question reads the existing event or
abstraction; it must not assert its own alternative event.

The line-3 sequence on `65451694e209edafacb5b74c87b091be6ca8342b`
ends at description: source order 8/8, both original CE bars failed;
lower and definite each pass bracket/occurrence 1/1; description brackets
4/4 and answers 1/4 after 24 presentations/12 paired updates. No retry or
later lesson supplies a pass. The remaining chain is carried to item 0.
Claude's later Philosophy edit is restored byte-for-byte and committed at
`bc9ae646f`; its exact patch was held outside the frozen lesson tree.

Additional implementation boundaries from this review:

- Opening an alternative shares the actual event prefix and the recorded
  branch-before value. Shared events across timelines are valid; duplicate
  occurrence/role pairs within a timeline are not. A later actual event
  does not advance a branch.
- A post-S adverb cancels only its covered provisional event. It restores
  the prior actual value only while that event is still the latest actual
  writer; a later independent actual event or description join survives.
  No second parser or re-choice is introduced.
- Substitute alternative meaning operands before the existing chooser
  scores the binary candidates, preserving their form coordinates. Bind's
  candidates, dummy, validity and score are unchanged. The selected
  numerical shadow supplies post-S attachment, using the same kernel,
  counts and basis, with no pool access in the compiled body.
- A question must not assert an event merely by asking. Retained event
  correspondence (subject identity and predicate), not cosine similarity
  alone, distinguishes initialized verbs with equal state results. The
  bounded thought reader keeps its work meter and unknown evidence.
- The actual view remains the default. The existing existence and truth
  read faces can reduce the timelines with some/every, respectively; no
  lexical marker classifier or new grammar operator chooses that fold.
  The actual timeline participates in both folds; missing evidence is
  unknown, not negative.

## 16. Modal implementation and corpus protocol (Codex, 2026-10-24)

These decisions implement §4 without reversing §1. Development logs stay
separate from the receipt that starts at the modal freeze.

- `Token.timelines` owns `(events, value)`. `value` and `events` are views
  of actual timeline 0. Checkpoint version 5 writes only this owner;
  versions 1–4 migrate to one actual timeline. The complete pool is
  validated before replacement, including event snapshots and widths.
  Capacity, sixteen-close expiry, names, promotion and document teardown
  still belong to the identity. A branch is never a bind candidate.
- `TimelineOverlay` mirrors the latest alternative and carries temporary
  scope, branch evaluations, prior actual values and the latest writer.
  The chooser scores branch meaning operands before selecting the same
  existing operation. Form coordinates, bind menus, dummy, validity and
  credit are unchanged. A post-S modifier restores the covered actual
  value and its unresolved flag only if no later actual writer has
  superseded it. No reference choice is rerun. Both bracketings are checked;
  post-attachment rollback and its pathwise derivatives are checked in
  eager and compiled execution.
- The temporary journal records the event order and covered-event
  suppression. The writer keeps the outer idea and the subject reference;
  it does not write the covered S as an actual occurrence. Event admission
  uses the semantic S's position, not the later attachment's position, so
  a later independent actual event cannot enter an earlier branch's
  prefix. Transparent unary wrappers retain the original event address.
  Object participants retain their own branch value, not the subject's.
- `modal_teacher` matches subject events at equal model time and consumes
  each alternative once. Like `verb_teacher`, it defers the graph until
  after paired updates. The weight is `training.modalTeacherWeight`,
  default 0.1, under reconstruction. Only explicit teacher observations
  end their graphs; no detach is added to an MTM read during composition.
- A question does not assert its event. `closing_question` carries the
  selected subject, predicate and order to `TimelineRead`. The existing
  bounded thought reader matches those references, not the cosine of
  initialized equal-valued events. Actual requests read order 0; abstract
  requests read order 1 or above. The existence face folds some timelines
  and the truth face folds every timeline, with actual included and the
  dual lane rule. Missing evidence is unknown. A truncated read cannot
  certify every. After expiry, retained Part arguments still locate an
  idea through its subject reference, under the same record meter; they
  cannot reconstruct an expired branch organization or certify every.
  If content postings are absent, this exact-reference read scans only
  within that meter. No new stored identity, derivation or record kind is
  introduced. The non-timeline query path is unchanged.
- Mention recurrence still admits promotion at three distinct sentences.
  Its value fold includes only order-0 mention/event occurrences, never
  alternative ideas. The latter keep their reference after document end.

**Existing test ports (no retired tests).**
`test_invalid_identity_checkpoint_does_not_replace_the_pool` moves the
invalid value, invalid event and expected value accesses to timeline 0;
the event corruption retains invalid role 1 in the version-5 tuple.
`test_pending_checkpoint_migration_and_validation_are_atomic` explicitly
builds the legacy value/events layout before testing version 1. The
expiry/promotion corruption test changes its expected-value access to
timeline 0. The promotion test supplies an actual order-0 close around
each of its four mention presentations; its assertions remain verbatim.
The receipt includes the exact patch, including both expected-value
assertion ports. All other existing assertions remain unchanged.

Development failures are retained: the new checkpoint comparison first
used a tensor helper on string metadata; the version-5 existing fixtures
still addressed removed flat fields; a transparent wrapper lost its event
address; and a retained idea lacked a content cue after expiry. Their
repairs preserve the semantic checks. The same-round close helper now
returns the prepared clause, as production does, for query construction.
The independent-event test additionally checks that an earlier branch
does not inherit a later actual event.

**Certificate 15, fixed before execution.** The complete protocol is
[`corpus-protocol.json`](../benchmarks/2026-10-23-item3-5-verb-modal/modal/corpus-protocol.json).
A fresh seed, 394616477, is drawn once and fixed for both forms. Each form
has four subjects, two verbs and eight generated documents: six training,
two held out by `(subject index + verb index) mod 4 == 3`. Each document
has an actual event, then an alternative, and two separately presented
questions. Two epochs give 24 training examples/72 sentence presentations
and 36 paired updates per form; evaluation gives four examples/twelve
sentences, without updates. The two premises are presented without any
supplied target, then the question is presented as a separate closing in
the same document. This protocol correction precedes its first execution: a
supervised multi-sentence batch would otherwise credit every premise. Batch size 4, learning rate .003, the native
232-coordinate paired lesson configuration and a fresh model per form.
The same initial draw is used for the free modifier leaf and verb affix.
There is no prerequisite restore, stage menu, grammar lesson, reference
order, segmentation hint or premise answer. Only training questions have
answer supervision. Both exact held-out answers and the correctly
referenced actual/alternative event histories must pass at 100%; failures
move to item 0 without retries, extra budget, forcing or another seed.

No binding mechanism changes: this modal source runs the pending-premise
certificates once in the scoped groups, not a twelve-draw series. There is
no full sweep in this iteration. The all-order MTM-prepending integration
and retirement of the transition overlay remain item 3; no test pins the
current order-1 bind limitation.

## 17. Frozen-run fixture and harness repairs (Codex, 2026-10-24)

Frozen modal source `45513baca6` passes all 43 certificates, 191 operator
cases with one existing skip, 55 query cases, 180 identity/driver cases
with three existing skips, and the unchanged XOR standing gate. The
eight-file group completes with one fixture-lifetime failure; the
regression and checkpoint groups do not start after it. Decomposition
collection and the corpus setup expose two missing diagnostic imports;
neither starts a model or provides behavioral evidence. Every run remains.

1. `test_meet_ties_kind_requests_and_excludes_missing_defining_content`
   binds the fixture's store to `_`, then overwrites `_` with `score`'s
   output. The resolver intentionally holds the production store weakly.
   The captured-RNG diagnostic forces collection between score calls:
   the first sees a store and the second sees `None`. Keep the store in
   a named local for this test and the adjacent
   `test_content_meet_cannot_read_form_while_form_can_identify_its_region`,
   which has the same two-score lifetime defect. Every assertion stays
   verbatim. This is a fixture ownership repair, not a changed resolver
   fallback or binding score.
2. The corpus runner's import search omitted the directory containing
   `identity_keep_probe`. Add that existing diagnostic directory. Its
   failed run retained the initial RNG but created no model and presented
   no data; the protocol seed, corpus, budgets and gates do not change.
3. The new gate directory omitted `identity_decomposition.py` and its
   `original-complete.grammar`. Copy both byte-for-byte from the prior
   measured protocol, so both arms use the same original snapshots,
   seeds and parameter-name maps. The collection failure is retained;
   it supplies no decomposition pass. Preflight every plugin's location
   before the corrected run.

No production file changes after `45513baca6`. The correction freeze
contains these fixtures, the harness path and the retained receipt. The
full declared scoped groups run on that hash, with no earlier result
promoted; there is still no full sweep and no corpus rescue. These are
§39 repairs, not a reversal of §1.

## 18. Frozen evidence and mediated hand-off (Codex, 2026-10-24)

The correction source is `01effdcaf9dd9adfbc460a791e06feb06b38b940`.
All ten scoped groups complete on that hash: **733 passed, 11 existing
skips**, no failures, all 744 selected cases. Production remains
byte-identical to `45513baca6`; the correction is exactly §17. Both
original decomposition arms and XOR pass. G1–G3 and both pending-premise
orders pass, with full captures and keep probes: forward 0 bound / 16
open, reverse 0 bound / 32 open. No earlier source supplies a pass and
no binding change calls for a twelve-draw series.

Certificate 15 completes both forms at the declared seed and budgets:
36 paired updates per form, then four held-out examples. Free leaf and
affix each score **0/4 exact answers and 0/4 required timeline histories**.
Both remain failed learning evidence and move to item 0 without rescue,
as §16 declares. The line-3 stop on `65451694e` also stands: description
answers 1/4; adjective through copular identity were not run. No artifact
is promoted and no copular prerequisite is lifted.

The [receipt](../benchmarks/2026-10-23-item3-5-verb-modal/README.md) gives
certificate 9–16 correspondence, per-group worker totals, exact assertion
and fixture ports, every retained failure and source audit. The modal
changes two expected-value paths to actual timeline 0, ports canonical
checkpoint/actual-occurrence fixtures, and retains the two weak-store
fixtures' owners; no existing test is retired. The lossless archive records
SHA-256 and byte counts for every file. These are evidence/docs updates
only, with no further source change or gate rerun.

The complete swap/verb/modal iteration now waits for Claude's review
through Alec. Publication of main and the parent Ethics edit/WikiOracle
bump waits for that verdict. No full sweep; lines 6–7 and the end-of-3.5
sweep remain. Item 3.0 has not been reached.

## 19. Claude's review of iteration 1 — 2026-10-24: LAND, with carried notes

Reviewed: §§8–18, the receipt (`benchmarks/2026-10-23-item3-5-verb-modal`),
the hand-off, the production diff from `ddfc3bdcd6` (22 files, 1,601
lines: `MeaningKernels`, `VerbLearning`, `ModalLearning`, `TimelineOverlay`,
`TimelineRead` new), the test diff (15 files), the port ledgers, and the
frozen sources `65451694e` (swap and verb: 811 passed, 26 existing skips,
837 cases) and `01effdcaf9` (modal: 733 passed, 11 existing skips, 744
cases). Both pass XOR, both decomposition arms on their original snapshots
and seeds, G1–G3 and the pending-premise certificates; no test is retired
(the two removed names are the two declared renames) and no skip is added.

**Each §1 decision is implemented as decided, on the paired configurations.**
`compose_blocks` makes the form a leaf-weighted centroid and gives the
meaning block to the operation: exact rules where declared, `MeaningAction`
for `lift` and `adverb`, a lane-preserving `LaneKernel` for the other
grammatical faces, all exact at initialization through `exact_initial`
while keeping the derivative. The verb acts as `Q diag(e^w) Qᵀ` on each
lane in the cube chart, never mixing lanes, with absent coordinates kept
exactly absent and the inverse as the restricted SPD solve; a transitive
VP takes `Q` from the object's meaning through Householder reflections
confined to the object's support, an intransitive one from its verb's
group, groups minted by a third distinct failed fit up to `verbGroups`.
Copular `part` and `whole` join; equality replaces. `verb_teacher` and
`modal_teacher` compare detached snapshots on defined coordinates, fire
once per event, enter the Error registry under reconstruction after the
paired steps, and reach no concept row. `Token.timelines` owns the actual
and alternative branches; a modal clause keeps its order above 0
(`TimelineOverlay.choice`: lift lowers to `left_order − 1`, the modifier
withholds it), writes only its idea, updates only the branch, and a post-S
modifier restores the covered actual write only while it is the latest
writer. Promotion folds order-0 occurrences only; `some` and `every` fold
timelines with both lanes and `every` fails on an incomplete read; a
question asserts nothing. Certificate 16 holds both MTM channels. The
configurations without a meaning block are untouched, as corrected in §2.
No marker is named anywhere in `bin`.

**Learning results, correctly handled.** Line 3's rerun on the swapped
source stops at description (answers 1/4) after stage 3 (order 8/8, CE
bars failed) and lower and definite (1/1 each); certificate 15's two
corpus forms score 0/4 on answers and histories. Both move to item 0 with
no rescue, as the protocol fixed in advance. These are the measurements
3.5 owes, not its mechanism gates.

**Carried notes, none blocking.**

1. *A new derivation reader.* `TimelineRead.read`'s fallback after a
   token expires reads subject and predicate through `PartRead.
   argument_reader`. It joins the removal round's list (layout spec §2):
   after removal it reads the chain's references instead.
2. *Two paths by layout.* `Language.py` now carries 33 `meaning_layout`
   guards: the unpaired configurations keep the form kernels by decision,
   and that path is the legacy one once every configuration carries a
   meaning block. Retire it then, not before.
3. *Indistinguishable at initialization.* With parameter-free centroids
   and identity-initialized kernels, two grammatical faces with the same
   head operand compute the same output until their kernels learn; the
   round2c port records it. Operator choice therefore learns from the
   chooser's priors and close-time credit, not from forward differences,
   at the start of every lesson.
4. *Equal `when`.* `modal_teacher` pairs an alternative event with an
   actual one at equal model time. Model time is the close's tick, not
   narrated time, and 2c found several closes can share one tick while
   consecutive sentences usually do not. The intended pairing is by branch
   point: the actual event the alternative was opened from (its recorded
   `before`). *Checked in the retained keep records (`corpus/*/keep.jsonl.gz`,
   21 batches per form): no pool entry ever held an alternative timeline,
   and no event carries `taught` or `modal_taught`, so neither teacher
   fired in either form.* The corpus failure is therefore upstream of the
   teachers: the chooser never applied `adverb` to the modifier marker,
   so no branch opened, and with note 3's identical forward values at
   initialization nothing but answer credit could have favored it within
   36 paired updates. Item 0's retry should start from operator choice,
   and the teacher's pairing should be by branch point before then.
   `verb_teacher` had no target by design: the corpus observes no identity
   after its event.
5. *Performance.* Stage 3 reached 33 GiB and 150 s per batch before the
   stationary-walk stop (§12) brought the first batch to 2.3 s; the eager
   decoder's selected-inverse change (§13) is the second saving. Item 1's
   measurement should include both paths.
6. *Six freezes for one line.* Each repair re-froze because the lessons
   need a frozen source. For iteration 2, run a short lesson smoke before
   the freeze so the freeze count returns to §6's one per line.

**Verdict: LAND `01effdcaf9`** (production identical to `45513baca6`).
Publish BasicModel main, include the parent `doc/Ethics.md` edit with the
WikiOracle bump, and push the parent main. Lines 6 and 7 follow as
iteration 2, after Claude's context-wholes spec.


## 20. Iteration 2: review of the existing specifications against the code (Codex, 2026-10-24)

Reviewed occurrence §§4–5 and §§12–13.1, with §13.2's gate and the later
decisions read for precedence; parts-of-speech §3, with its operand-flow
table and chooser contract; and the 5.5 plan's nested-admission decision.
The source is published BasicModel `f306fc28aa190264a6c841267155686f9f71ea3d`
(reviewed modal freeze `01effdcaf9`, production unchanged), referenced by
published WikiOracle `319a79373be58d5309989193fec49b7ef1b64ffe`. The requested
corpus paragraph and read-only keep-record audit landed before those
pushes. This section is a code review, not an implementation or a new
freeze. No test, lesson or sweep was run for it; no assertion changed.
Claude's context-wholes specification comes before that implementation,
and review continues through Alec.

### 20.1 What is already present, and what remains

| Contract | Present on this source | Work still to establish |
| --- | --- | --- |
| Occurrence §12: history order | `OccurrenceRows.stream_rows` and `recency_weights`, and `get_stm_chain`, use the store's order within a stream. Re-witnessing does not move a row. | Preserve that order when context wholes replace source/stream columns; do not infer it from equal model ticks. |
| Occurrence §§12–13.1: surface and inverse | `SurfaceLayer`, its shared unary face and explicit self-inverse `TransposeLayer` exist; the retired analyzers and split/insert faces are absent. | Exercise the native paired form/meaning path, including a nested marker and its inverse; remove the remaining language-naming text. |
| Occurrence §§4–5: VP phases and selection | MTM has actual/alternative event histories, with `before`, `after`, verb and role. `CaseSelection` supports bounded native concept cases on the unpaired path. | A VP phase view, its repeated idempotent selection, reference-event content and phase questions are not implemented by those pieces. The paired compound does not use the case fold. |
| Occurrence §12: nested admission | One named suffix lesson, default off, witnessed stem, irregular kept whole; scoped parts persist afterwards. | Native paired composition/inverse coverage and admission cost, under the existing scoped decision. |
| Parts-of-speech §3 | Kept-program direct-operand counts, carried words, direct chooser probabilities, report-only cosines and checkpoint state already exist. | Audit their current native integration and close line 7 with scoped evidence; no new category codebook is needed. |
| Documents and streams as context wholes | Today's source-address and stream columns still serve several owners and readers. | Claude's new spec must define the replacement and migration. This review does not build the item-3 layout or STM integration. |

The old inventory is not an accurate removal list. `surface_tense.py`,
`surface_morphology.py`, `TenseLayer`, `AspectLayer`, `MorphologyLayer`,
`_lift_when` and `_lower_when` are already absent. So are the legacy
timestamp column and `Clause.eternal`; LTM0's load migration drops the old
timestamp. `test_item55_mechanisms.test_absences` already checks the retired
mechanisms. The remaining `.when` is model time, not a temporal operand.
The obsolete analyzer, situation-whole and four-surface-operation passages
must be read with the later amendments, not rebuilt.

### 20.2 VP phases: the actual gap is in the meaning and query paths

Occurrence §4 assigns the selection to the catalogue's compound; §13.1
calls its semantic rule `position`. There is no `PositionLayer` or live
grammar entry by that name. `CompoundLayer` (`Language.py:3355`) declares
the right operand as its case head. Its unpaired path requires
`selected_cases` and folds them; `CaseSelection.stage_case_bank` stages
existing native concept cases, while `select_cases` uses the modifier's
observed memberships. This corrects the older statement that modifier
conditioning is absent everywhere. It does not supply the phases of a
newly composed VP or a bound identity's event.

On a paired layout, `CompoundLayer.compose` instead calls
`GrammarLayer.compose_blocks` (`Layers.py:2492`): centroid on form, a
generic meaning kernel on meaning. The compiled candidate path
(`Language.py:7287`) supplies `SelectedCases` only when `meaning_layout`
is absent, and paired reverse validation follows the paired kernel too.
Consequently the existing compound tests do not establish the phase
contract on BasicModel. The old `fold_cases` also subtracts negative from
positive membership in a signed code; copying it into the paired meaning
block would collapse the two evidence lanes.

`PrepositionLayer` (`Language.py:3638`) does not fill this gap. Its
unpaired computation returns the right operand; its paired computation is
another meaning kernel. Its comment about editing `.where` describes no
such live operation. The transparent unpaired preposition tests remain
valid for that configuration and must not be weakened to accommodate a
different paired rule.

Iteration 1 supplies an event history and verb action, superseding §4's
old dependency on an unbuilt item 6.5 teacher. It does not yet expose
before/change/after as selectable cases. `TimelineRead.request/read`
matches a grammatical subject, verb and order against timeline events;
it has no phase selection or R operand. An event's stored snapshots alone
cannot answer whether only part of its change was asserted. In particular:

- Before and after must remain distinguishable on reverse. Progressive
  selection must not entail the completed event; repeated selection must
  be idempotent, not another application of a verb gain.
- R is now or a referred-to event's content, never the referenced row's
  `.when`. Close ticks and §19's teacher-pairing note do not implement
  narrated time. Preserve that distinction in both composition and query.
- No selection retains the whole case set and answers phase questions as
  unknown. It must not silently select the present or a completed event.
- `TimelineOverlay.choice` treats the chosen `adverb` operation as modal
  scope and withholds lowering. Routing all tense/aspect selections through
  that operation would incorrectly turn ordinary phase selections into
  alternative events. The phase selector must retain VP order as §4 says;
  it must not add a lexical exception to modal scope.

**Recommendation for the implementation specification:** retain one
case-selection owner, the existing compound, for the semantic `position`
rule. Generalize its staged case view to NP cases and VP phases, using the
bound value and event content, with a lane-preserving fold on paired
layouts. Stage the required view outside the compiled body. Keep marker
packaging separate from selection and call the selector again for another
selection; do not add temporal/spatial variants, a second scorer or a
phase-gain rule. State how the absent-marker face gets its learned
selection object and how R is carried in that view. This is a proposed
bridge to today's code, not a new operator, a catalogue rename or a stored
phase table. No code or checkpoint schema is changed here.

### 20.3 Surface forms and nested admission

`SurfaceLayer` (`Language.py:3537`) has two accepted layout paths. The
unpaired one owns the marker map/prior and chart reconstruction; the
paired one has no such form parameters and uses the centroid plus meaning
kernel. `MarkerUnaryLayer` shares the binary owner's kernel and its reverse
emits no extra leaf. Binary surface takes a marker on the left; a marker
on the other side is reachable by the existing charged transpose of the
two newest constituents. There is no reason to add another surface face,
split or insert. Existing surface/mechanism tests cover reverse witnesses,
unary output and transpose cost, but direct unpaired layer fixtures alone
do not certify the paired native route and counted centroid inverse.

`WordIdentity.nested_lesson/observe_nested` (`WordIdentity.py:240–296`)
admits only the active suffix after a witnessed stem, including when the
stem is witnessed after its marked word. It does so after one such pair;
there is no distinct-stem threshold. That is **not a newly found decision
violation**: the 5.5 plan's later Alec decision, under “Before the 4e pass,”
explicitly supersedes the proposed stricter statistic with lesson scoping.
Occurrence §12 retains both stages of that history. Keep the named-pattern,
default-off decision and the existing immediate-admission assertions in
`test_nested_wholes`, rather than introducing an unapproved threshold.
The artifact-backed companion is still prerequisite-dependent, not a new
learned pass. No prefix/infix/general substring admission is implemented.

The performance residue is concrete: the candidate search considers only
the current word and that word plus the active marker, but an admitted
change still revisits nested parts, reindexes word forms and rebuilds the
word hash/trie. A bounded lesson smoke should measure that path and prove
that the staged sub-word parts reach composition and reverse while the
word remains the attention unit. A unit test that merely carves a word
does not show that the chooser uses the marker. No marker label should
choose a grammar operation.

The language-naming inventory also needs correction. `Models.py:6900` has
one such docstring and `:6923` a comment, not two live docstrings. Additional
names remain in configuration/test comments and the test identifier
`test_given_truth_texts_are_plain_english` and its diagnostic text.
Those need neutral descriptions under §13.0, with assertion semantics
unchanged. The inactive tense/morphology/aspect planning comments in
`complete.grammar` and `ladder.grammar` are stale too. This review has not
edited code, fixtures, test names or configurations.

### 20.4 Direct-operand word classes: audit the existing owner

`PartsOfSpeech.formal_arguments`, `flow_mask` and `observe_kept`
(`PartsOfSpeech.py:16–120`) already implement the central §3 contract.
Each carried lexical occurrence receives its input-argument count in a
kept program; producing operations pass their words according to §2b,
with result-argument counts and both transpose outputs recorded. Clauses
clear the carrier. Rejected readings and bracket annotations do not
supply those counts. `collapse` reports occupant-distribution cosines;
it merges nothing and trains no prototypes.

The probability table is staged once for the paired trials
(`Models.py:17634`). The kept-program update is at `Models.py:20494`,
followed by the table refresh. `Language.py:7017` adds the appropriate
direct-argument probabilities to operation scores; `update_flow` preserves
one contribution per carried word rather than averaging a phrase into a
new class. Checkpoint state keeps the formal-argument schema and counts.
The output-argument counts exist but do not implement output typing.
`PartClosing` also records admitted-symbol occupations; that is not a
license to count bracket supervision as word evidence.

`test_parts_of_speech.py` already covers kept-only/exactly-once counts,
unseen distributions, direct scores, carrier flow, compiled execution,
checkpoint restoration and report-only collapse. Its phase-artifact cases
remain prerequisite-dependent. Line 7 should close the current integration
with a short native paired lesson smoke and scoped tests, then document
any actual gap; it should not build a second categorizer. The old §3 noun
prior tied to an individual column is superseded by grammar-only parts of
speech and is not to be revived. Order-2 kinds, output typing, order in
parts of speech and the learning claims stay on their recorded later
items. Neither labels nor a forced stage menu supply learning evidence.

### 20.5 Inputs to Claude's context-wholes specification

The current owners still use `document_keys`, `sentence_index`,
`sentence_content_keys`, `address_keys` and `stream`. Consumers are broader
than the row exporter: `Occurrence` stages and resolves source addresses;
`ClauseRow` upserts them; `ReferenceContext` excludes replaced occurrences;
`Models`, `data` and `Teacher` drive document/stream boundaries;
`Queries`, `MemoryIndex` and `ThoughtStream` read provenance or scope;
`Forgetting` protects/releases addressed rows; `MediumTermMemory.close`
and `end_document` use document identity for expiry and promotion.

The new spec needs to preserve occurrence handles and token/mention/event
references while replacing provenance columns; say how the same retained
row participates in more than one stream; preserve first store order
under re-witnessing; and define document-end teardown of all timelines,
sixteen-close expiry, recurrence across three distinct documents and
promotion across three distinct sentences. Context membership must not
silently replace those distinctness tests with model ticks. The chain's
replacement reader must still distinguish streams without ordering them
against one another. No stream-as-seat identity or resurrection of word
adjacency follows from this work.

This is an interface inventory, not an alternative context-wholes design.
The item-3 tagged layout, derivation removal, close-time mint retirement
and MTM-prepending/STM integration remain out of scope. In particular,
do not deepen the transition overlay to solve phase or context storage,
and do not retire `TimelineRead`'s current derivation fallback here.
The §19 branch-point teacher note remains carried to the item-0 retry;
changing `.when` semantics is not a substitute for that repair.

### 20.6 Gate plan after the specification is ready

The six placement variants in occurrence §13.2 are not yet implemented in
`test/constituent_corpus.py`; it still produces the earlier constituent
families. The phase gate needs those variants with one grammar and one
configuration, including no-marker → unknown, partial change without
completion, repeated selection, stacked selections and both surface
directions. Report behavioral learning separately from controlled phase
and inverse certificates. A failed learned operator choice is not repaired
by a marker table, forced choice, wider menu, extra budget or another seed.

Before each line's single freeze, run the §19 short lesson smoke on the
native paired configuration: input staging, live composition, kept close,
reverse and checkpoint restore must all execute, including any new nested
or phase path. Development runs retain logs; the receipt starts at the
freeze. Then size the gates to the actual changed modules, preserving
surface/compound/POS tests and occurrence §8's applicable certificates,
G1–G3, the unchanged XOR standing gate, original decomposition arms and
pending-premise certificates.
A twelve-draw series is needed only if binding changes. No full sweep
until the end of 3.5. Record every fix and assertion port under step-2 §39,
and carry every failed run. There is no gate result or assertion-port
request in this review.

Iteration 1 is published; iteration 2 is reviewed here while Claude writes
the context-wholes spec. Lines 6–7, their scoped landings and the single
end-of-3.5 sweep remain. Item 3.0 has not been reached.

## 21. Context wholes released; iteration 2 proceeds (Codex, 2026-10-24)

Alec supplied Claude's [context-wholes specification](2026-10-24-context-wholes.md)
while §20 was being completed and explicitly released its implementation
with the rest of lines 6–7 under step-2 §39. Claude's spec and the exact
identity/todo additions are committed unchanged at `0f90e9b2d3`. The
context spec's dated §6 reviews the code and records the required
adaptations: overlapping memberships without merging identities, retained
membership after pool expiry, checkpoint migration, transient addresses,
eager staging before the compiled prelude, store-order recency, bounded
context lifetime and bounded context queries. §20 remains the review of
the existing surface, phase, admission and occupation specifications.

Proceed on main. Line 6 includes context wholes and its twelve fresh
pending-premise draws because the menu gains context locations. A short
native lesson smoke precedes each line's single freeze; gates and receipts
are sized to the change. Line 7 follows with its own frozen source. Review
of the complete iteration is through Alec, then publication of main and
the WikiOracle bump. No learning claim, no ordinary-identity prepend or
derivation/layout removal in this line. Item 3.0 has not been reached.
