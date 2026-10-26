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

## 22. Replacement context mechanism and development status (Codex, 2026-10-24)

§21's MTM context mechanism and twelve-draw instruction are superseded by
Alec's WholeSpace decision. Claude's replacement context spec, identity
spec and todo edits were committed unchanged at `23eed97d20`. The dated
review and implementation decisions are context spec §§7–9. The withdrawn
build and all its logs remain in the iteration's development directory;
none is promoted as evidence for the replacement.

The active implementation uses native WholeSpace run forms and approximate
content membership. It removes the stream column and document-provenance
sidecar, with explicit test ports. It adds no context address, MTM whole,
bind location or twelve-draw series. Active runs carry context across
batches; retained rows can supply only the trace their composition kept.
The context spec records the read path after symbolic promotion, the
initial membership threshold, migration and capacity behavior. No learned
abstraction or exact document recovery is claimed.

Surface inverses, nested admission and direct-operand occupations have
development coverage; the existing categorizer remains the owner for
line 7. The VP-phase gap from §20.2 remains: the paired compound uses a
meaning kernel, and event snapshots are not yet selectable VP phases with
R carried as content. The six generator variants and phase questions must
be connected to that implementation, not accepted on the basis of the
existing generic case selector. This is work remaining in line 6, not a
new prerequisite skip or a move of its mechanism to item 0.

Neither line has frozen or published. A receipt starts only at its freeze;
development logs and the exact assertion/fixture inventory are retained
separately. Keep one freeze per line after its lesson smoke, then the sized
gates, one review of the iteration through Alec, and publication. No full
sweep until 3.5 ends. Item 3.0 has not been reached.

## 23. Cover replacement and preflight stops (Codex, 2026-10-24)

§22's WholeSpace run build is withdrawn. Claude's context spec, todo,
Orders, FutureWork and Architecture edits were committed unchanged at
`47a2216e0b`. Context spec §§11–12 record the §2b review, adaptations,
ports and two stops requiring decisions. The active tree restores exact
batch streams and their original readers, opens an ordinary order-1
concept and LTM cover row per document, and reads segment membership.
No run row, MTM context identity, contextual word blend or membership
threshold remains; _consolidation_sources stays removed. WholeSpace
contributes only to order-0 composed form centroids; tower intervals
and derive are unchanged. Promotion uses evidence-weighted centroids.

Preflight passed XOR's eight cases. Both original decomposition arms
failed at two successors because cheaper live readings dropped required
premises. Keep probes and a diagnostic-only RNG intervention isolate
the added cover draw's effect on the departure. An existing 24-row
forgetting fixture needs a twenty-fifth row for the required cover before
its boundary. No seed, keep rule, forced scope, capacity or existing
behavioral assertion has changed to make these pass. Context spec §12
gives recommendations and an exact capacity-port proposal for Alec.

All evidence remains development evidence, including every failure and
both withdrawn builds. Line 6 has not frozen; VP phases, six generated
variants and final gates remain. Line 7 follows with its own smoke/freeze;
one iteration review precedes publication. No sweep or twelve-draw series
has run. Item 3.0 has not been reached.

## 24. Final cover correction, inverse smoke and the remaining compound decision (Codex, 2026-10-24)

Context spec §§13–16 supersede §23's two stops. The forgetting fixture
preallocates 32 rows in both constructors, asserts 32 and uses 8/32 and
6/32. No runtime capacity changes. Its eight sentences, interval, one
boundary pass and behavioral assertions remain verbatim. The cover's
checkpointed dedicated generator leaves the global RNG untouched.
`preflight-02` passes XOR (8) and both original decomposition arms (2 each),
without the diagnostic intervention. The three §13 notes are recorded
individually in context §15.

Run 66 passes the 32 live surface, occupation and case-mechanism checks
(3 prerequisite skips), but its native lesson smoke exposes an invalid
free unary inverse. A dictionary word's centered form can be closer while
uncentering the different parent would leave the membership chart.
`FormCentroids.unary` now admits that centered equation only if its
recovered form is finite and within the chart. The plain equation remains
available; no coordinate is clipped and no chooser action is forced.
Run 67 passes all 19 cover/inverse checks and the native lesson smoke.
The failing run is retained. This fixes the new inverse's domain, with no
existing assertion changed. The stream-placement fixture port and its
failed intermediate setups are recorded in context §16.

**Decision requested from Alec before the phase change.** The phase rule
in occurrence §§4–5 and §13.1 is idempotent selection of a head's cases.
The accepted swap in this spec §§1–2 gives paired `compound` a direct,
identity-initialized affine meaning kernel. After learning, that kernel
need not be a projection: applying it twice can change the meaning twice.
Neither the operator's current paired forward nor its paired inverse
reads `SelectedCases`. This is the gap identified in §20.2, not a cover
or preallocation failure.

Recommendation: keep one learned kernel, but use it to form the modifier's
case test; intersect that test with the retained cases and fold the result
without mixing positive and negative evidence. Carry the retained case
selection through subsequent applications so repeating the same test is
idempotent. The head supplies its cases (including a VP's before/change/
after); the object supplies content, including a referred-to event for R.
No `.when` column is a composition operand. Its absent-marker face shares
the same kernel. This replaces direct affine meaning composition and its
affine inverse for paired `compound`; it requires a bounded case search
on generation, as the unpaired case selector already does.

That replacement is submitted through Alec because it changes the
accepted operation contract, rather than being treated as a routine
inverse repair under §39. No phase implementation, grammar-menu change,
phase assertion port or frozen source is included in this section. The
six generated variants depend on that mechanism. Independent cover,
surface and direct-operand audits continue; line 6's single freeze waits
for the remaining mechanism, followed by line 7 and the iteration review.

## 25. Correcting the scope of §24 and implementing the case owner (Codex, 2026-10-24)

On rereading §§1–2, the affine parameterization is an implementation
choice, not Alec's decision. The decision is one learned kernel per
grammatical operation, shared by its faces and preserving the evidence
lanes. A learned case test followed by a case fold satisfies it. Therefore
§24 should not have been an approval stop; its question is optional
steering, and work continues under §39. No elapsed silence is treated as
approval. The authority is the existing §39 hand-off.

`CaseView` retains the head's cases, bilateral memberships and an accumulated
content test. Selection intersects tests; refolding averages each evidence
lane independently. Unselected cases give unknown to a case question, and
a partial content selection cannot certify the whole request. The paired
compound's existing kernel forms the test, using the unfiltered head as its
stable context. It is not reapplied as a gain to the previously selected
value. Form remains the leaf-weighted centroid, including the separately
supplied WholeSpace center when applicable. Neither the case view nor its
kernel reads memory or a clock inside the tensor body.

Two exact ports in `test_meaning_kernels.py` accompany that semantic change:
`test_other_catalogue_faces_use_the_same_centroid_contract` expects the
compound's right head, rather than the left modifier, when no cases are
available; all other classes and assertions remain unchanged.
`test_direct_paired_reverses_reject_a_retired_split` now requires the
compound's `reverse_blocks` to reject an affine split and request candidate
search, as its direct reverse already does. The other classes retain the
original exact inverse assertions. Run 71's failure of the old affine
assumption is retained. Run 73 passes all 36 case-view, compound and
meaning-kernel checks. These are development checks; case staging, parse
flow, generation and the phase questions still have to be connected and
verified before the line's smoke and freeze.

## 26. Phase content, parse flow and the six placements (Codex, 2026-10-24)

The phase bridge is separate from the identity overlay. `CaseFlow` carries
head membership, the accumulated content test, the current reference value
and the selected-content journal beside the conceptual stack. The eager
owner stages reachable native cases and accessible actual-event snapshots
(before, the verb operator, after), within the existing candidate bound.
Added phase slots are local bank positions, never native symbols or bind
candidates. Current reference content comes from each accessible identity's
current value, counted once per head/identity; multiple available identities
supply a centroid, not a new identity or an asserted correspondence. Explicit
selection instead receives its actual object operand, which can carry a
referred event. Neither path reads `.when` to compose a phase.

The binary and absent-marker faces of `compound` share one kernel. The
absent-marker face uses that staged current reference as its implicit object;
with native cases but no event history it uses the head's own content. It
preserves the same case set and accumulated intersection, so the same
selection is idempotent. A second call can restrict the first selection,
without creating a temporal/spatial operator, a marker lookup or another
scorer. Both shipped catalogues declare the absent-marker face and its
one-child reverse. No phase call withholds lowering as `adverb` does.
`test_shipped_grammar_supplies_subtyping_without_a_predefined_surface` ports
its exact face count from 1 to 2; its binary head/order/anchor assertions
stay, and it also checks the unary head/order and absence of a surface
anchor. Run 88 retains the old count failure.

A kept subject event records the selected **content**, not its test mask.
The mask alone could appear to cover coordinates that were never among the
head's cases. Phase questions carry that content in an ordinary meaning
constituent; `_query_case` contains only its typed local reference, keeping
numerical payloads out of semantic metadata. Each evidence lane must cover
the whole requested content. No selection, an empty selection or a proper
fragment does not certify completion or its negation. Event clocks do not
enter this comparison. Expired event history cannot certify the phase and
reports `phase_history_expired`; ordinary retained event queries keep their
existing derivation fallback. MTM checkpoint version 7 copies and validates
this optional event value; versions 1–6 remain readable, with no phase value.
No read adds a detach to MTM's staged content.

The retained inverse reconstructs the counted form centroid against its
co-operand and marks discarded head meaning undefined. It does not run an
affine inverse of selection or copy the missing head from the answer target.
Free generation uses the primed candidate search and the same case fold;
the absent-marker inverse emits one candidate and no extra leaf. Existing
forward-discard credit therefore applies to genuinely lost content. The
surface and exact-rule inverses keep their own contracts.

Development repairs are retained: run 77 exposed a conditionally unbound
module import in a compiled loop (78 passes both modes); 80 exposed a tensor
incorrectly put in query metadata; 81–82 are launcher collection errors;
83 exposed exact decimal comparison against float32 evidence in the new
fixture (the two scalar checks now use `pytest.approx`, with the same .8/.2
values); 87–88 exposed a staging variable shadowing the reference accumulator.
Run 84 passes 57 phase, modal and thought checks. Run 86 completes the short
unforced native lesson (one training and one evaluation batch), but passes
0/2 answers, selections and exact reconstructions. That is a development
observation, not learning evidence promoted to the freeze.

`constituent_corpus.case_corpus` supplies the six §13.2 placements; their
labels never enter the model. `phase_experiment.py` uses one configuration,
the complete menu, no reference orders, segmentation hints or grammar
forcing, and answer targets only for separate training questions. The
frozen protocol is three epochs of 18 training cases in batches of two,
then six held-out cases, for each placement. One first RNG draw is retained
and shared across the six placements; no result selects a seed or budget.
It reports exact answers, selection counts and zero native reconstruction R,
retaining both trials, keep decisions, initial and failure states. Failed
learning goes to item 0 without rescue. The smoke uses only one batch from
each split and makes no learning claim. The single line-6 freeze still
follows its final smoke and preflight; line 7 follows, then review through
Alec. Nothing in this section is published evidence.

## 27. Integration ports and cover capacity (Codex, 2026-10-24)

Development run 90 stopped in the decomposition measurement loader before
behavior, because its archived grammar was missing. Run 93 then exposed the
new generate action's unmapped parameter row. `preflight-05` authenticates
compose and generate action names against the original grammar, restores
every surviving original parameter exactly, and retains initialization only
for the new `compound_unary` rows (alongside the already accepted `bind`
rows). Run 94 passes both original decomposition arms, two cases each, at
their original snapshots and seeds. No behavioral certificate changed.
Run 92 passes 213 integrated regressions, with seven existing skips.

Run 96 passes 277 operator/reader checks and skips four, with six failures.
One production adaptation is missing: `_program_entries` must allow its
minimal capture owner to lack `languageSpace`, just as it allows absent
optional form metadata. The optional lookup is now nested; no assertion
changes. Two tests assume there is no prior frame at a document's first
sentence. Under §39, their exact ports are:

- `test_real_packed_ends_train_before_the_next_sentence`: context length
  `[sid + 1] * 2` becomes `[sid + 2] * 2`, counting the cover; the last-frame,
  occurrence, stored-value, gradient and train/commit assertions stay.
- `test_zero_meaning_departures_and_one_reader_step`: compare components
  0 and 2 (R and A) between readings, rather than all three. A polarity
  departure now changes expectation against the cover, which is legitimate
  context. Captured credit still has zero R+A advantage and keeps greedy;
  every keep, root, evidence, reader and optimizer assertion stays. The
  objective and predictor are unchanged.

Run 97 is retained as a diagnostic launcher failure: it omitted the CPU
setting and selected MPS, where the existing float64 centroid diagnostic is
unsupported. Run 98 sets the same CPU environment as the scoped gates and
retains initial/failure states, cover allocations and both trials' credit.
It confirms the expectation-only difference and the capacity issue below.

**Decision routed to Alec before freeze.** The three XOR_grammar cases
(two round-2c arms and the round-3a native-root check) fill six concept seats
with four admitted word concepts and the first two covers; the third cover
fails before any reading or keep. Four one-sentence batch documents require
four covers, hence eight total seats. No safely redundant or unreferenced
seat exists. The proposed patch changes only construction-time
`ConceptualSpace.nVectors` in `data/XOR_grammar.xml`, 6 to 8; output width,
data, budgets, seeds and assertions stay. This is a preallocation, never a
runtime resize. It is held unapplied because the standing no-fixture-capacity
rule needs an explicit exception beyond the separate §12.2 LTM correction.
The patch is `development/proposed-xor-preallocation.patch`. No cover is
suppressed and no seat is reserved. Independent development checks continue;
line 6 is not frozen and line 7 has not started.

## 28. Line-7 read-ahead: direct operand occupations (Codex, 2026-10-24)

Parts-of-speech §3's mechanism is already live. `formal_arguments` builds
fixed prototypes from the declared compose arguments and results.
`PartsOfSpeech.observe_kept` replays only the selected program: each carried
word occurrence increments its direct argument, and the head-flow table
carries the resulting occurrences onward. A symmetric phrase preserves both
occurrences, a headed phrase preserves all the head operand's words, and a
clause stops their flow. `Models._commit_sentence` is the admission seam;
the two trial reads share the eagerly staged occupation probabilities and
cannot update counts. `Language.OperationSelectionLayer` adds these direct
argument probabilities to operation scores. `GrammarCheckpoint` migrates
occupation columns by their declared names and does not turn old category
weights into observations. No definition or word-order lookup decides a
part of speech here.

The new absent-marker compound face follows this existing one-operand head
rule. It needs no new prototype class, lexical table or learned category
codebook. Line 7 will add a focused repeated-occurrence certificate through
binary composition, the unary face and clause completion; compare the host
journal with the tensor flow, and retain the existing kept-only native,
checkpoint and gradient checks. The narrow scope is mechanism verification,
not a claim that unseen words' classes have been learned. Its lesson smoke,
freeze and scoped evidence follow line 6 in the requested order. This is
read-ahead only, not a second source freeze or a landing receipt.

### §27 capacity proposal corrected after the complete native group (2026-10-24)

The eight-file development group completes with 121 passed, seven existing
skips and two more capacity failures: `test_forced_not_and_or_descend_and_snapshot_certificate`
and `test_both_training_trials_share_bootstrap_snapshot`. Both fail at the
third cover of the first presentation, before their reading/keep. The latter
presents the same four documents three times, with a new cover on each
reading as §2b requires. Its twelve covers plus four word seats need sixteen
occupied positions. The unchanged initial order partition leaves two unused
order-0 positions, so **18 is the minimum construction-time inventory**:
partition (6, 3), ordinary tail 9, hence 12 order-1 positions. No reservation
or allocator change is proposed. Eight would cover only the first read;
the earlier proposal is retained as `superseded-xor-eight-seat-proposal.patch`.
`proposed-xor-preallocation.patch` now changes 6 to 18 and remains unapplied.
The exact inventory is in `xor-capacity-diagnosis.json`. All output widths,
input rows, repeated readings, targets, seeds and assertions stay. G1–G3,
both pending-premise orders and the other live native checks pass on the
current development source. No frozen evidence is claimed.

### Another bounded fixture exposed by cover retention (2026-10-24)

The identity/driver group also reaches
`test_compose_pair_driver.py::test_evaluation_repeats_the_same_program_without_an_explore_trial`.
Its `MM_xor_loopback` configuration has eight concept seats, four words and
four covers after its first evaluation. Its second reading needs four more
covers and fails before composition. This is a separate configuration from
both the XOR standing gate and the eight-seat `MM_grammar` gates.
The unapplied `proposed-loopback-preallocation.patch` changes only that
configuration's ConceptualSpace construction capacity, 8 to 16. The existing
order partition (8, 4) then has an ordinary tail of four, permitting the
eight retained covers; output width stays eight. No assertion, input, repeat
count or runtime shape changes. This needs the same narrow preallocation
exception as the XOR proposal, and is not applied.

A diagnostic tested restoring an initial checkpoint before the second call
instead of changing capacity. Run 102 failed in the diagnostic wrapper's
signature (fixed with `wraps`); run 103 still exhausts capacity. Its pre-read
checkpoint predates dynamic word admission, and loading weights into the
populated instance does not rewind that inventory. This is not evidence for
a valid fixture port. Both logs and the checkpoint are retained; no reset
or checkpoint change is made to production, and the existing evaluation
test is unchanged. The capacity failure is reported as such, not hidden by
suppressing covers or changing the determinism assertion. The remaining
checks continue on the unchanged development source.

## 29. Predictor isolation diagnosis (Codex, 2026-10-24)

The identity/driver development group completed with two failures: the
loopback capacity failure above and the predictor-isolation check in
`test_real_packed_runbatch_trains_prediction_and_representation_with_teacher`.
The latter initially looked like a production gradient leak. Replay of its
captured initial process and generator states in run 108 distinguishes the
terms: `expectation.activation` and `expectation.symbols` reach only the six
shared predictor parameters. `expectation.components` has graph connections
to ten representation parameters (all zero-norm in that captured trial).
Covers introduce the native individuals that activate that pre-existing B
energy path. There is no completed keep at the failure: the assertion fires
in the first trial's backward probe.

The component contract is explicit in
`test_independent_components.py::test_sparse_coding_is_confined_to_verbs_and_cannot_train_noun_signatures`:
its observed roles must receive nonzero gradients while no noun dictionary
is trained. Detaching that target would break the accepted contract. No
production loss, gradient route, MTM read or objective will change.
Under §39, the exact fixture port is to form the diagnostic loss from
`registry.terms()` named `expectation.activation` and `expectation.symbols`,
including their weights, in place of `registry.total(objective='expectation')`.
The test already disables word prediction to isolate this head. All of its
assertions, including encoder disconnection, observed predictor gradients,
zero-weight Adam behavior and detached retained context, remain verbatim.
The independent sparse-component certificate remains unchanged and runs
beside it. The source edit and verification follow the current reader run.

Runs 104, 106 and 107 are retained diagnostic-launcher failures (missing
capture-helper imports); 108 uses the established full capture environment
and reproduces the actual assertion with per-term parameter gradients.
These are development logs, not a freeze or a receipt.

## 30. Authorized XOR preallocation (Codex, 2026-10-24)

Alec authorized the corrected proposal: `data/XOR_grammar.xml` changes
only `ConceptualSpace.nVectors`, from 6 to 18. The twelve covers and four
word concepts required by the unchanged three-presentation bootstrap
certificate fit the existing partition. Output width, inputs, budgets,
readings and assertions stay unchanged. This is construction-time
preallocation; no runtime shape or allocation policy changes. The earlier
eight-seat proposal remains retained and superseded. The independent
`MM_xor_loopback` proposal is still awaiting its own authorization.

The edit follows completion of the running, source-guarded reader group,
so its captures continue to identify one source. Development checks of
the affected cases then precede the line-6 lesson smoke and single freeze.

## 31. Preallocation rationale for review (Codex, 2026-10-24)

Alec asked that the preallocations be justified to Claude, noting that the
code has too many unjustified variables. These are three existing fixture
capacity fields, not new parameters. Their inputs and repeats remain fixed;
none changes a compiled shape at runtime or reserves seats for a reading.

| fixture | fixed capacity | derivation and headroom |
| --- | ---: | --- |
| `test_real_model_forgets_at_document_boundary_then_reads_writes_and_reloads` | LTM 32, both constructors | Eight unchanged sentences can write 24 rows; the new cover requires a twenty-fifth. Alec chose 32 for construction-time headroom. High/low marks remain exactly eight/six, expressed as 8/32 and 6/32; one boundary pass and every behavioral assertion remain. |
| `data/XOR_grammar.xml` | concepts 18 | The bootstrap certificate presents four documents three times: twelve covers plus four admitted words. With unchanged output width six, the existing partition is (6, 3), followed by nine ordinary tail seats. That supplies twelve order-1 positions and leaves two order-0 positions unused. Eighteen is the minimum inventory under that partition. |
| `data/MM_xor_loopback.xml` | concepts 16 | The evaluation determinism certificate presents four documents twice: eight covers plus four words. With unchanged output width eight, the existing partition is (8, 4), followed by four ordinary tail seats. That supplies eight order-1 positions and leaves four order-0 positions unused. Sixteen is the minimum inventory under that partition. |

The XOR change was explicitly authorized at eighteen; Alec's follow-up
instruction to justify the preallocations is applied to the separate
sixteen-seat loopback proposal as well. That change follows the running
source-guarded checks. Both remain bounded fixture changes: no new allocator
rule, learning budget, document count, output width or behavioral assertion.
The eight-seat `MM_grammar` gates are unaffected. The superseded eight-seat
XOR and twenty-five-row LTM proposals and every failed run are retained.

The reader development group passes all 55 cases. The storage group passes
111 and fails only `test_every_read_word_has_a_concept_and_symbol[XOR_grammar.xml]`
at the same six-seat cover allocation, before composition. It joins the
authorized preallocation verification. These results are development logs;
the receipt begins at the single line-6 freeze.

## 32. Exact candidate lane recovery (Codex, 2026-10-24)

With the authorized inventory, run 110 passes six cases and reaches two
strict zero-R failures in the unchanged final-b certificates. Runs 112 and
114 replay their captured initial states, retain the full models and both
trials' terms and keeps. The charge is `reconstruction.content`, about
2.9e-9; word identity, bytes and recomposition are exact. The two readings
tie and keep greedy. No reference choice, answer target or keep rule is
changed. The predictor port separately passes on its captured state, as do
all eighteen independent-component checks; the sixteen-seat loopback
certificate passes with every assertion unchanged.

The lane inverse takes signless candidate codes, then solves their evidence
magnitudes. On the failing second read the candidates already reproduce the
nonempty negative lane exactly. Their common coordinates differ by one
float32 step (.1601281613111496 versus .1601281464099884). The min inverse
nevertheless rescales the first candidate to the least admissible magnitude,
shrinking its exclusive coordinates from .10675211995840073 to
.10675210505723953. Those exclusive coordinates cannot be inferred from a
min root. This is an unnecessary change to an already exact candidate, not
a missing word or a failed allocation.

Under §39, `MeaningCodes.recover_lanes` keeps the candidate pair's nominal
magnitudes when their min/max already equals a nonempty parent lane exactly;
otherwise its existing least-min/greatest-max solve remains. An empty lane
still uses that solve and cannot acquire unobserved evidence. The two lanes
remain independent, and the mean solve is unchanged. This uses the free
candidate bank, with no input operand, target, tolerance, clipping, weight or
new parameter. The non-injective inverse's exact-candidate preference changes;
the reconstruction objective and every existing assertion stay verbatim.
Four new cases retain the captured magnitudes across both operators and both
poles, checking exact content and exact recomposition. The unchanged final-b
pair is rerun before the smoke and freeze. All failures remain retained.

## 33. Retained form assertion follows the published swap (Codex, 2026-10-24)

The expanded lane-recovery file exposes an older form assertion in
`test_corner_composition_lane_rules_and_form_unchanged`. It compares the
form block of an operator configured with a meaning block to a second call
on a form-only input. Those calls have different contracts since iteration
1's published swap: the former is the parts' centroid, while the latter
retains its legacy exact form rule. The failure precedes lane recovery.
All 32 conjunction/disjunction corner cases fail this one assertion; the
sixteen sum cases already agree because their old rule is the mean.

Under §39, the assertion's expected value changes from
`op.compose(a[:, :2], b[:, :2])` to `(a[:, :2] + b[:, :2]) / 2`, with
`rtol=0, atol=0` unchanged. This is one assertion over 48 parameterizations;
the evidence-lane assertion and every input remain verbatim. It enforces the
already published centroid rule independently of the implementation. The
failed runs are retained and no test is retired.

## 34. Line-6 freeze audit (Codex, 2026-10-24)

The final development preflight passes the 147-case lane/form file, both
strict final-b certificates replayed from their captured failures, the eight
XOR standing cases, and both original decomposition arms (two cases each,
original snapshots and seeds). The last native lesson smoke completes one
training batch and one held-out batch, with six paired updates. It reports
0/2 exact answers, 2/2 selection counts and 0/2 exact reconstructions. This
is a runtime smoke with failed learning targets, not a learning claim or
evidence promoted to the frozen gates. Its complete states and every earlier
failure remain under `development/`.

The source audit records every changed source and fixture hash, syntax
checks, removed identifiers and exact assertion/fixture ports. No test is
retired. No run wholes, MTM wholes, contextual word blend or membership knob
remain. Covers use the ordinary reference menu, so no twelve-draw series is
introduced. The three preallocations and their arithmetic are in §31. The
new lane inverse preference and the retained form assertion port are in
§§32–33; no reconstruction weight, answer target or keep rule changes.

Line 6 freezes once on main. Its receipt begins there and runs the scoped
groups and all six generated placements by that hash. Line 7 follows with
its own smoke, freeze and scoped evidence. Both wait for the one iteration
review through Alec before publication. Item 3.0 has not been reached.

## 35. Frozen line-6 failures and captured diagnosis (Codex, 2026-10-24)

Source `e9c53252aa` passes the 82 mechanism certificates and 110 reference
regressions. The native group completes 122 passes, seven prerequisite skips
and one c/e failure at the unchanged line 348. All remaining groups continue
on that same source before any implementation changes.

The full capture shows the question's explored parse contains
`part(what, is)` inside its equality. Its R is .14269229769706726 versus
.16269230842590332 for greedy; A is zero for both, so explore is correctly
kept. The query then takes the other sentence's equality as an answer at
match .7132023572921753. That row marks its fixed right operand unresolved,
but `best_match.can_answer` checks only whether the *free* role is marked
outside the equality-specific path. Content from the unresolved right side
therefore appears to establish a match against a different fixed reference.
No cover is the answer witness. The same annotation already refuses an
unresolved *answer* and admits an exact address of an unresolved operand;
this is the missing fixed-operand check, not a new identity or query target.

A captured diagnostic replay filters precisely that unsupported match and
passes the original c/e test with every assertion, reading and keep unchanged.
The diagnostic is not a frozen gate pass. The intended repair applies the
same resolution boundary in retrieval and ordinary fill: a marked fixed
operand cannot establish a content match to another reference. Direct reads
that explicitly name that operand retain their valid opposite endpoint.
No query rank, bind menu, forced choice, objective or keep rule changes.

The fixed-budget corpus also exposes a runtime error after training creates
multiple lexical readings. The grouped operation executor flattens position
and reading-variant axes; the new phase view still indexes the original
positions. Unary phase selection then receives six candidate positions with
a two-position view. Both initial and failing states, updates and keeps are
retained for each affected variant; these are execution failures, not failed
learning targets. The repair will preserve the position axis for contextual
case operations, using the existing per-variant execution pattern. A direct
binary/unary reproduction checks values and pullbacks against separate
variant execution in eager and fullgraph modes. No new parameter, capacity,
selection or budget follows from the repair. Source changes wait until the
current frozen invocations finish; every original run remains retained.

The identity/driver group reveals a third runtime failure, in the existing
joint-objectives packed-run certificate. Its configuration has no meaning
block, yet the new retained phase inverse indexes `meaning_layout[0]`.
Both unary and binary case branches must require the paired layout before
using that inverse. The unpaired unary face already composes as an identity;
its inverse declaration and reverse must say the same. The unpaired binary
face retains its existing native-case search. This corrects an inverse
contract without introducing a new operation, objective term or parameter.
A new regression covers the unary values, inverse domain and gradients in
eager/fullgraph modes and the binary case search. Its first compiled setup
constructed a namespace inside the trace; moving that fixture construction
outside the tensor function keeps every assertion. That setup failure and
the original production capture remain retained. No existing assertion is
changed by any of these three repairs.

The query-boundary diagnostic also passes all 55 existing thought/query
consumer cases with their assertions unchanged. The unpaired inverse repair
passes the original packed-run failure from its captured initial state.
The matching free unary decoder must make the same paired-layout check:
without a meaning block its absent selection is the identity there too.
The new certificate checks both retained and free inverse availability,
exact values and gradients. These diagnostic results justify the repair;
they do not replace gates on the repaired frozen source.

The remaining query group exposes an additional retained-inverse domain
failure in `test_numeric_and_renamed_sentence_paths_keep_oracles_out[texts1]`.
An inverse-derived parent reaches a lane kernel outside the cube. Its
initial state and failed run are retained; a read-only replay now records
each inverse input, output and defined-coordinate mask to locate where the
invalid value first appeared. No clipping, tolerance or assertion change is
proposed before that diagnosis. Both original decomposition arms and the
XOR standing gate pass on the unchanged first frozen source.

## 36. Inverse domain and provisioning-count follow-ups (Codex, 2026-10-24)

The retained trace locates the first invalid inverse value at `SumLayer`:
a bound identity's current value no longer admits the retained sum split.
The inverse correctly marks every returned coordinate undefined; its
largest placeholder is 1.9996763467788696. Two identity inverses carry that
undefined value, then `VerbLayer` passes it to the cube chart despite the
all-false defined-coordinate mask. This is evaluation of an already
unavailable inverse, not a roundoff or cube-domain decision.

Under §39, `inverse_step` supplies a numerical zero at coordinates already
marked undefined before invoking the next map. The false mask stays false;
zero is not asserted as recovered content. Defined coordinates are never
clipped, detached or repaired, and an out-of-cube defined input still
rejects loudly. The new eager/fullgraph cases check that form remains
exact, missing meaning stays undefined and a defined invalid input still
raises. The original failing run and the read-only inverse trace remain;
its captured replay verifies the production path before the repair freeze.

The context/storage group completes 111 cases with two failures. The
occurrence-coordinate certificate reaches the same unpaired-inverse defect
already diagnosed in §35; every assertion stays. The other failure is a
count port in `test_real_provisioning_is_not_an_external_prediction_stream`:
three provisioned one-sentence documents now write three cover rows beside
the original three sentence rows. Port `assert len(store) == 3` to
`assert len(store) == 6` and add a separate three-cover check. The prohibition
on external prediction binding, the empty context and absent next
expectation assertions stay verbatim. No test is retired. This changes no
fixture capacity, input, document count or behavioral threshold.

The supplemental forward-discard file exposes a historical fixture mismatch,
not another inverse failure. Its October-10 retained frames predate the
published swap and contain the old transformed forms. The action indices
still name `lower` and `non` correctly; all lower meaning inverses are in
range, while 50–61 form coordinates per lower frame violate the current
centroid inverse. The local certificate does not replay a forest or carry
leaf counts. Port only its copied output form: the mean of the two retained
operand forms for a binary, or the single operand form for a unary. Keep
all actions, all three meaning values, saved costs and assertions verbatim;
the immutable archive is untouched. This makes the fixture express the
current form rule without changing the meaning-loss certificate. The failed
probe, per-step domain audit and original grammar snapshot are retained.

The first working-tree application of the diagnostic patch lost one level
of indentation in the grouped unary branch. Development unit and replay
launches failed at import before executing the model. Those logs remain;
the indentation is corrected and syntax checks now precede the reruns.
This was an application error, not a frozen source or a fixture failure.

The first form-only port still fails because the archived operand forms
are signed coordinates, not current cube coordinates; the per-step audit
retains that correction to the initial diagnosis. Complete the fixture
migration by applying the existing `cube_unchart` to the copied form block
before taking each local output's centroid. No meaning coordinate, action,
cost, assertion or archived byte changes. This is a fixture carrier port,
not a production clamp or an inverse-domain relaxation. The first port's
failure is retained beside the original failure.

For the repair protocol, the arithmetic-isolation file runs one case per
worker invocation under the same 900-second limit. Its first frozen worker
completed five cases in 873 seconds, close to that aggregate limit. This
changes test-process grouping only; every selector remains, with the same
fixture, training steps, suite limit and assertions. Other query cases keep
the existing grouping. No model or lesson budget is enlarged.

The applied repair passes the c/e, unpaired packed-run and occurrence-time
captures. The signed-form fixture port passes all three forward-discard
checks. The domain replay's test body also passes, but its runner reports
`source_changed` because the independent forward-discard fixture was edited
while it ran. That invocation is retained as invalidated, not a clean pass;
the same captured domain case is repeated after source edits finish. The
lesson smoke and remaining pre-freeze checks run with the source stable.

## 37. Line-6 repair freeze audit (Codex, 2026-10-24)

The repaired source passes the eighteen new regression/provisioning cases,
the three forward-discard cases after the signed-form fixture migration,
and all four original production-failure captures (c/e, packed unpaired
read, occurrence coordinates and the stable domain replay). The short
lesson smoke completes one training and one held-out batch, six paired
updates, with 0/2 answers, 0/2 selection counts and 0/2 exact reverses.
Those learning targets fail; the smoke supplies runtime coverage only.

The cumulative audit hashes 63 source/fixture files and parses every changed
Python file. Relative to the first freeze, exactly one existing assertion
changes: provisioning row count 3 to 6, with a separate three-cover check.
The historical frame fixture changes only its copied form representation;
every assertion stays. Four new regression files cover the production
repairs. The capacity values are unchanged; their arithmetic is also written
beside the two concept capacities and the 32-row fixture. Every failed or
invalidated invocation remains retained. The repaired source receives a new
frozen hash; the first freeze remains failed. All scoped groups and the six
corpus placements run on the repair hash, with no promoted pass and the
corpus's same captured first draw. Line 7 follows before review through Alec.

## 38. Repair-gate scope and corpus runner interruption (Codex, 2026-10-24)

The final module-to-test audit adds three direct-reader files once on
`04c1132797`: `test_understanding_readout.py`,
`test_live_query_occurrences.py` and `test_reference_index.py`. All twelve
cases pass. No completed group repeats; `supplemental-scope.json` records
why these files were added to the original selector list. No source or
assertion changes accompany this scope correction.

The preallocation receipt also traces the legacy `grow_inventory` call:
`Spaces._concept_alloc_of` constructs these native stores with all
`nVectors` rows, so admission asks for at most the existing size and that
helper returns without growth. The three field values remain construction-
time capacities; the receipt gives each required inventory and unused
partition or chosen headroom.

The six corpus placements were initially queued three at a time. With
several presentations taking minutes, Codex attempted to replace only the
queue controller and run all six independent placements concurrently on
the 14-core, 36-GiB host. Terminating the controller also caused its tool
execution session to terminate the three active descendants. This was an
orchestration error: verb-form, separate-leaf and noun-form stopped after
7, 7 and 22 completed batches. Their logs, keeps, initial states, last
batch-start states and interruption records remain under source-02/corpus.
They are interrupted evidence, not passes or production failures.

The three previously queued placements continue from their original first
starts. The interrupted three replay from the exact same captured initial
RNG and the full original budget in source-02/corpus-replay; no trained
checkpoint, new draw, source change or learning rescue is used. The replay
protocol retains the initial-state checksum. The new controller cannot know
the exit code of a process it did not parent, so its records leave that code
null and report the missing completion files explicitly. The original
controller's exit 143 and the actual interruption are retained. The
read-only process and tensor samples are timing diagnostics only; no
execution setting or model parameter was changed.


The end-of-3.5 default sweep follows line 7's freeze and scoped gates, as
§6 requires, before the combined review through Alec. It uses the preceding
step-2 sweep's resource limits. The two subprocess-harness files run once
in a clean measurement environment, separately from the otherwise default
collection with capture plugins: their intentional replacement of
PYTHONPATH made inherited PYTEST_PLUGINS fail five child probes in the
preceding sweep. This is a disjoint execution grouping, not an omitted
file, new skip, test port or repeated sweep. The prepared protocol records
it before collection; no full sweep has run yet.


## 39. Completed line-6 evidence and line-7 admission (Codex, 2026-10-24)

On `04c1132797`, all fourteen scoped groups finish: 1,250 passed, twelve
existing prerequisite skips and no failures. All six corpus placements
complete their original budgets from the captured first draw; all answer
and exact-reconstruction scores are 0/6, and selection counts are 6/6 only
for the unmarked form, 0/6 otherwise. Their full captures and keep audit
are in the line-6 receipt. All six learning gates move to item 0. Some
kept programs carry selected-output flags and two forms have limited
verb-teacher activity, but no retained event carries a case value; no modal
teacher or alternative timeline appears. No learning claim follows.

Line 7 installs the repeated-occurrence certificate prepared under §28,
unchanged, after all line-6 source guards close. It adds no production code,
capacity or parameter. Its eager/fullgraph cases independently specify the
carrier and exact occupation-count deltas through symmetric composition,
a headed binary, the absent-marker unary and clause completion. Every
existing assertion stays. The short kept-only paired native lesson precedes
its freeze. The first smoke command omitted the capture helper's directory
from PYTHONPATH and failed at collection before any test ran; that log is
retained. The corrected runner supplies the same complete observer paths
already used by line 6, with no test or production change.

The corrected smoke passes all three cases: two new eager/fullgraph cases
and one existing paired native lesson. Line 7 now freezes the test-only
change. No existing assertion changes and no test is retired. Its six
scoped files run once by hash, followed by the one default end-of-3.5 sweep
under §38's disjoint harness protocol. Both lines wait for review through
Alec; no publication or item-3.0 start is implied by this freeze.

## 40. End sweep and scoped repairs (Codex, 2026-10-24)

**Capacity rationale superseded by Alec's correction in §43.** The run
results below remain historical evidence; counting each training presentation
as a new book is not the storage contract.

The single default sweep completes on `0bd65c5e2cd1594a8d2e517cbdbbc274751f379c`:
6,504 selected and completed, 6,144 passed, 324 existing skips, 35 failures
(33 call, two setup) and one non-strict XPASS. The source guard passes and
the two execution groups cover the collection exactly once. The four call
reports emitted by the existing flag-aggregation test count as one node;
all raw reports remain. This is a failed sweep, not a green landing. Only
scoped repair runs follow; no second full sweep is proposed.

The read-only keep probes restore the original initial RNG exactly for
all five replayed tests (tensor audit retained). Three unchanged eight-seat
MM fixtures have four words and four covers occupying the whole inventory
before the first kept close requests another concept. Both trials and the
keep are retained: they tie and keep greedy. The 18-seat XOR allocation is
a minimum for the three-presentation bootstrap, not a claim that a
400-epoch training fixture fits. Two long-training setup failures exhaust
covers before their quality assertions run. No capacity, presentation count,
forcing, keep decision or objective is changed to hide these failures.
The receipt carries these additional capacity decisions to Alec with the
bounded allocation arithmetic; a larger number is not selected by trial
and error.

The reread probe corrects an initial diagnosis: its sixteen rows are four
covers, four estimates, four observations and four thoughts, not twelve
covers. Its test must select the four sentence observations before checking
stable addresses, timestamps and the two witnesses. The first observation
now has a preceding cover, hence expectation pairs also occur at the first
sentence. General occurrence reads continue to include covers.

Three independent production repairs need no assertion changes. A plain
callable in an inverse fixture has no `__dict__`; the diagnostic activation
preserver tolerates that interface. Structural adapters use the optional
`meaning_layout` attribute for plain operation implementations. A scalar
verb-group subscription inside nested `cond`/`while_loop` triggers Dynamo's
`PendingUnbackedSymbolNotFound`; tensor `index_select` preserves values and
gradients without extracting a scalar. The retained minimal reproduction
fails on the sweep hash, and a regression exercises that same nested path.
The two live-reference fixtures receive the current tree accessor and
`timelines` argument; all their assertions stay.

Other ports account explicitly for covers in provisioning/allocator counts,
truth-only views, expectation history and the question refusal message, and
use the published form centroid in the old form-kernel fixture. Every exact
assertion and fixture change will be listed in the repair receipt. The two
prepared-answer fixtures assume the retired additive paired `lift` inverse;
they require current operator fixtures while retaining their boundary,
length and credit checks. No test is retired. The sweep remains the source
of every original failure; repair evidence is separate.

## 41. Decoder diagnostics and remaining capacity decisions (Codex, 2026-10-24)

**The capacity recommendation below is withdrawn by §43.** It described
the implementation's repeated covers, not a legitimate book-cover lifetime.

The first scoped repair run passes 203 cases with five existing skips and
five failures. The toy decoder fixtures always returned a status tuple;
the selected inverse now requests values alone. Port their declared
`return_status` behavior, keeping all actions, gradients and assertions.
The reread's second close retains eight estimates and eight thoughts, plus
eight covers and the same four observation rows: the total is 28, not the
initial port's 20. The failed 20-row port remains retained; the observation
address/time/two-witness checks stay verbatim.

The scalar basis repair exposes a second compiler defect. Its retained FX
graph has 39 placeholders and 38 inputs: the extra placeholder is an unused
`ForwardTokenAOTInput`. The matching library's `unlift_tokens` returns
without removing an unused effect token when no top-level effect nodes
remain. A global token-policy diagnostic does not repair it; no such global
setting is applied. A native-root domain-check diagnostic is being tested
without disabling any nested checks or changing the compiler backend.

The prepared-answer diagnostics identify a production omission as well as
the obsolete additive-lift fixture. `reverseOutput` passes
`require_symbols=False` for an aligned carrier even when it has a meaning
block. The swap requires a candidate bank for that meaning inverse, and the
held bank is already present. A diagnostic supplying it restores the
four-versus-one length check with a current union-centroid fixture. The
production branch must use the bank for paired configurations regardless of
alignment; pure-form aligned configurations keep their existing behavior.
The shared-inverse fixture uses a current determiner kernel with distinct
marker/content forms and a known marker, preserving the length, state-credit
and parameter-ownership checks. Its obsolete full-vector equal-children
setup is ported to the two explicitly supplied operands, not to a weaker
boundary or a fabricated inverse. Exact ports and results accompany the
repair receipt.

Capacity decisions remain explicit. The three unchanged eight-seat MM
word fixtures have no free row after four words and four covers. The
400-epoch XOR fixture needs at least 1,600 cover seats for training alone,
before evaluation and other concepts; the authorized 18-seat bound belongs
to the three-presentation bootstrap. I recommend construction-time
preallocation derived separately for each bounded consumer, with the full
presentation budget and existing partition recorded. I do not recommend a
new knob, a guessed margin, a shorter budget, a forced reading, or silent
cover reuse. Increasing the expressly unchanged eight-seat configuration,
or changing the fresh-cover rule, requires Alec's decision before applying
it. The receipt states this limitation next to the original justifications.


## 42. Sweep repairs and the scope of the capacity arithmetic (Codex, 2026-10-24)

**The cover arithmetic and proposed freeze below are superseded by §43.**
The independent repair results remain development evidence.

The second scoped development run completes 58 cases: 38 pass, nineteen
existing skips and one failure. Both prepared-answer ports pass, including
the unchanged four-versus-one and two-versus-two length checks. The failure
is in my bank condition: a pure-form bank can have `meaning_start` equal to
its full width, not only None. Testing for a nonempty complement preserves
the pure-form answer path; the original mixed-target mask and conditioner
update assertions then pass unchanged in the third repair run.

The prepared inverse's gradient port is explicit: after the swap a fixed
selected form has no trainable form kernel. Its percept-only loss must not
have a nonzero gradient to a meaning-only inverse. Rename that test to
`test_percept_answer_keeps_meaning_inverse_and_state_out_of_credit` and
replace its nonzero inverse-gradient assertion with zero-or-None. Keep
its two emitted words, absent state gradient, reconstruction ownership and
output-backward exclusion checks. Its independent marker and content replace
the obsolete equal-children setup. A separate new test differentiates a
meaning loss through the inverse of a given, detached parent and requires a
nonzero meaning-kernel gradient. This changes a test expectation to the
decided swap; it adds no detach or objective change to production.

The root-domain-check compiler diagnostic also fails. The successful
interface diagnostic identifies the complete AOT omission: one unused,
explicitly described `ForwardTokenAOTInput` remains in the FX graph after
its caller has already unlifted it, together with an empty constant output.
Inductor receives 38 arguments for 39 placeholders. Removing that dead
interface also requires removing its output-stride entry and shifting the
user-output indices. `CompilerEffects` performs this narrow normalization
only for reconstruction's Inductor callback. It neither removes live effects
nor changes a backend, domain check, numerical value, parameter, gradient
boundary or global compiler setting. Negative tests keep ordinary, live and
already-complete interfaces untouched. The native compiled ingestion case
passes; the eight-seat MM case now reaches its independent capacity failure.
All failed intermediate compiler diagnostics remain retained.

The capacity audit finds a second half of the existing-tail repair:
`PartClosing.preflight` still permits only the higher-order partition while
its writer also uses the unpartitioned tail. Align that preflight with the
writer, without adding rows. A new relation-argument test checks the tail
seat and unchanged capacity and bank shape. The old refusal fixture removes
all order-1 seats, including the tail, by assigning the rest of its same
physical inventory to another order; every assertion stays. The third scoped
run completes 31 cases: 28 pass, two existing skips and the unchanged
eight-seat exhaustion. No further capacity is applied.

For Claude's review, the 18- and 16-seat figures are minima for the mandatory
cover-plus-word inventory of the named bounded certificates under the
existing partitions. They are not a promise that every possible derivation
or every longer consumer fits. The explicit 32-row fixture has seven spare
rows chosen by Alec. None is a new knob or a runtime resize. The long XOR
budget implies 400 × 4 = 1,600 fresh cover allocations for training alone;
that is workload accounting, not a chosen new simultaneous-capacity bound.
The remaining ordinary concepts, evaluation presentations and reclamation
behavior must be accounted for before selecting another construction value.
The expressly unchanged eight-seat fixtures need Alec's decision. The
recommendation is a documented per-consumer construction bound, without a
shorter document, budget, forced parse, concealed reuse or guessed margin.

The repaired source will freeze once after its short lesson smoke and exact
assertion/fixture inventory. Scoped repair gates follow on that hash. The
one complete sweep remains failed and is not repeated. Known capacity
failures stay visible for review through Alec; no publication or item-3.0
start is authorized by these repairs.

## 43. A book cover is not an epoch allocation (Codex, 2026-10-24; for Claude through Alec)

Alec corrects the preceding recommendation: “a book cover happens once at
the beginning of LTM.” He also flags “storing the same LTM memories for
multiple epochs” as “probable magic” and requests discussion with Claude
after the run. The claim that 400 epochs over four documents *requires*
1,600 covers is withdrawn. It counts allocations made by today's cursor;
it does not justify a capacity, and an epoch does not establish a new book.
The 18- and 16-seat recommendations used the same repeated-cover assumption
and need reassessment. Their historical arithmetic and failed runs remain.
No additional capacity is selected. The separately authorized 32-row fixture
accounts for one eight-sentence document and its initial cover, not epochs.

The code discrepancy is concrete. `DocumentCover.begin_sentence` calls
`open_cover` when the document changes, the cursor restarts at an earlier or
equal position, or the previous cover is missing. `end_document` removes
the cursor entry. `open_cover` mints a fresh symbol and uses that symbol in
the cover's write address; it has no retained-book reuse path. Thus another
training presentation currently creates another cover even when the book
and source sentences are the same. This behavior is a review finding, not
an accepted interpretation of Alec's correction.

The retained reread probes distinguish memory writes rather than assuming
that every row duplicates. `Models._commit_sentence` gives the observation
writer the existing document/position/content address. `OccurrenceRows`
finds that address again. In the four-document fixture, one presentation
has 16 rows: four covers, four estimates, four observations and four thoughts.
The second presentation has 28: eight covers, eight estimates, the same four
observations and eight thoughts. The observation addresses and `when` remain;
their witness counts become two. The original failure, failed intermediate
count port, passing development port and keep probes are all retained in
the sweep receipt. The 28-row/eight-cover assertions describe the current
implementation only; they must not become a certificate for repeated books.

An existing lower-level test already guards row reuse:
`test_four_occurrences_after_four_hundred_presentations_and_full_upsert`
keeps four rows after 400 presentations, while requiring 400 witnesses
per row and the latest value. It stays unchanged. The review therefore
needs to distinguish row duplication from repeated evidence and state
updates; replacing the cover allocation alone does not settle both.
`LTM0.append_meaning` replaces the row's content and joins each evidence
pole by maximum while incrementing `witness_count`; it does not add the
same evidence numerically on every presentation. `MediumTermMemory.event`
also replaces an existing `(occurrence, role)` event while its token is
retained. The review concerns the actual lifecycle of these updates, not
a claim that every memory mechanism blindly appends.

**Discussion requested, with recommendation.** Keep the once-per-book cover
decision fixed. Review the training replay boundary before sizing more
storage: which LTM records should be reused, which genuinely new prediction
or thought events may be stored, and whether replay should add witnesses,
evidence, MTM events or recurrence/promotion credit. The implementation also
needs to preserve the association between a reused sentence and its original
cover in the `.when`-ordered segment model. My recommendation is to separate
training presentations from new source events and make those write rules
explicit, then reassess the affected preallocations and count ports against
that contract. This is not a proposal to reset memory each epoch, change
learning budgets, add a provenance sidecar or silently suppress writes.

The native answer-credit diagnostic already in progress ended at its
existing 1,800-second worker limit: exit 124, zero of one cases completed,
5.02 GiB peak memory. It is an incomplete timeout, not an assertion result
or a pass. Its log, initial state and process sample remain in
`sweep/development/answer-credit-native*`; the source diff's file hashes
were unchanged throughout. After the run, only the two configuration
comments were corrected; their values remain 18 and 16. The prepared repair
freeze is held for this review, with the working repairs and every failed
run retained. No second sweep, publication or item-3.0 start follows from
these development results. Alec continues to mediate the discussion with
Claude.

## 44. Claude's review of iteration 2, and the replay contract — 2026-10-24

Reviewed: context spec §§15–16, iteration §§24–43, the line-6 receipt
(first freeze `e9c53252aa` failed and retained; repair freeze
`04c1132797`: 1,250 passed, twelve existing skips, no failures across
fourteen groups, the six placements complete), the line-7 receipt
(`0bd65c5e2c`: 22 passed, two existing skips), the end sweep on that hash
(6,504 completed, 6,144 passed, 324 existing skips, 35 failures, one
xpass), the 35 failing nodes, the preallocation rationale and its
withdrawal, the uncommitted scoped repairs, and the cover and address code.

**What stands.** Line 6 is built as §2b decided: covers as order-1
concepts with cover rows in exact batch streams, segments by `.when`,
readers excluding covers as propositions, the dedicated cover generator,
the span-whole centroid for order-0 forms with its inverse, the surface
inverses, and VP phases as idempotent case selection through `compound`
and its absent-marker face (`compound_unary`, the unary face §12 allows,
not a new rule). Codex's §25 self-correction was right: the affine kernel
was an implementation choice, so replacing it with a learned case test
was §39 work. The six placements and both smokes fail their learning
targets and go to item 0 with no rescue; the placements' audit shows no
case value ever reached a kept event, which, like iteration 1's corpus,
puts the failure upstream of the teachers. Line 7 adds a certificate and
nothing else. The declared centroid discrepancy (meaning-side fold) is
recorded for item 3 as agreed. Seven test ports in line 6 are explained
one by one in the receipt; no test is retired.

**What blocks landing.** The sweep failed, and the repair freeze is rightly
held, because most of the remaining failures turn on one unsettled rule:
what a repeated presentation of the same source writes. Codex's
implementation opens a fresh cover whenever the cursor restarts, and the
reread probe shows a second presentation of four one-sentence documents
adding four covers, four estimates and four thoughts (16 rows to 28) while
the four observations are re-witnessed in place. Alec: "a book cover
happens once at the beginning of LTM"; storing the same memories for
several epochs is "probable magic." The capacity arithmetic built on
repeated covers (18 and 16 seats, 1,600 covers for 400 epochs) was
withdrawn by Codex on that correction, correctly.

**The replay contract, recommended.** An epoch is the optimizer's repeat
of a source, not a new experience. Every LTM write is addressed by its
source, and a repeated presentation refreshes the row it already wrote,
never appends:

1. *Observations* already do this: `_address_for_write` keys a sentence by
   document, position and content, so a re-read finds its row, joins the
   evidence poles by maximum and increments `witness_count`, which is
   definedness growing with exposure, the legitimate effect of repetition.
2. *Covers* are addressed by the document, not by a minted symbol:
   `open_cover` writes `document_key=('cover', domain, symbol)`, so a
   re-presentation cannot find it and mints again. Address the cover by
   the document key; `begin_sentence` reopens only when no cover row
   exists at that address. One cover per book, found again at each
   presentation, its `.when` unmoved, so the segment it heads is intact.
3. *Estimates* are addressed with the prediction's content
   (`_estimate_write_address`), which changes as parameters learn, so each
   epoch appends a new estimate. Address them by document and position
   alone, so the latest prediction replaces the earlier one.
4. *Thoughts* likewise: an episode for the same question at the same
   position replaces its row.
5. *MTM* already replaces an event for the same occurrence and role while
   the token lives; recurrence and promotion count distinct covers and
   distinct rows, so epochs add no credit; the sixteen-close clock is
   unaffected.

With this, a store holds one record per source event however many epochs
run, and no segment is disturbed, because no row moves and none is added.

**Capacity, as a formula.** A fixture's concept inventory is admitted
words, plus one cover per distinct document, plus the concepts its closes
mint, plus the fixed partition's slack, computed and recorded per fixture;
"preallocate more" is headroom above that floor, Alec's to choose, never a
derivation from epochs. The 18 and 16 are recomputed on that basis, and
the three eight-seat MM fixtures, which fill with four words and four
covers, are sized by it too, since their contract (four documents) has
not changed; the 400-epoch XOR fixture needs no cover beyond its four.

**The remaining sweep failures** are the cover-count and form-rule ports
Codex has in hand (round4a0's reread becomes 16 rows under the contract,
not 28), the prepared-answer fixtures assuming the retired additive lift
inverse, the toy decoder signatures, the compiler dead-token repair and
the aligned-carrier bank condition. None reverses a decision.

**Verdict:** not landable until the repair freeze; the replay contract
above is the one decision needed before it, and the port list changes with
it. Then one scoped repair freeze on the sweep's hash, the scoped repair
gates, and the review of the whole iteration through Alec; the failed
sweep stays as the record.

## 45. Decided: a presentation begins with empty memories (Alec, 2026-10-24)

Alec, asked what a repeated presentation of the same source should write:
"If we know we will be running multiple epochs, probably just delete the
LTM between runs (updating memories in place is probably best left to
percepts/concepts)." Asked how fixture capacities are then justified:
"See previous answer." This settles §43 and withdraws §44's refresh-in-place
recommendation: an epoch is not an experience to be merged with the last
one; it is the same experience again, and what persists between the two is
only what learning changed, the percepts and concepts (parameters). The
memories of the experience, every store, begin empty.

**The boundary.** One model-level operation, called at the start of each
training epoch in `BasicModel.runEpoch` (split `train`), beside the
existing per-epoch `expectation.reset()` and `what_memory.reset()`, which
are the same idea for two smaller memories. It empties:

1. LTM0 of every row a presentation wrote (`ORIGIN_CONVERSATION`); rows
   given rather than experienced, `ORIGIN_PROVISIONED` and `ORIGIN_USER`,
   stay, since `provision_ltm` runs once. `LTM0.reset` where nothing is
   provisioned, `clear_origin` otherwise; Codex chooses and writes why.
2. LTM1 of every relation whose arguments are cleared LTM0 occurrences;
   relations between concepts (`relate`) are concept structure and stay.
3. MTM entirely (`restore(None)`): its identities were stitched from the
   cleared occurrences. The cover cursor `_document_covers`, the thought
   document-end flags, the forgetting counters (`forget_*`), the
   consolidation recurrence record (distinct covers per concept) and any
   other host-side cache holding an occurrence reference, which Codex
   audits and lists in the receipt. Recurrence and promotion therefore
   credit distinct documents within one presentation only; epochs add none.
4. STM is per-sentence state and needs nothing.

Direct training loops that re-present a source without `runEpoch`
(`test_mm_xor._fit_xor`, the bootstrap certificates that present four
documents three times) call the same boundary between presentations. A
reread *within* one presentation is a real experience and keeps today's
behavior: the address finds the row, poles join, `witness_count` grows;
`test_four_occurrences_after_four_hundred_presentations_and_full_upsert`
remains the store-level contract for that case and says nothing about
training.

**The cover concept persists; the cover row does not.** The cover is a
concept and concepts never die; its seat could not be reused anyway
(`ConceptAllocator.new_concept` is monotonic and `drop` keeps the row). So
the model keeps a map from document to cover symbol, checkpointed beside
`cover_rng_state`; `open_cover` mints only for a document not in the map,
and after the boundary writes the document's cover row afresh at the same
address `('cover', domain, symbol)`, with the new presentation's `.when`,
so the segment it heads is whole. Concept reclamation leaves mapped cover
concepts alone. One cover per book, for the life of the model.

**Capacity, sized for one presentation.** Concept seats: admitted words,
plus one cover per distinct document across the splits a fixture presents,
plus the concepts its closes mint, plus the fixed partition's slack; LTM
rows: one presentation's rows plus the provisioned seed. Each value is
written beside its derivation; nothing is derived from an epoch count. The
18 (`XOR_grammar`) and 16 (`MM_xor_loopback`) are recomputed on that basis,
the three eight-seat `MM` fixtures are checked against it (four words and
four covers fill eight exactly, so any minted concept needs a larger
construction value with its justification), and the 400-epoch XOR fixture
needs its four covers and no more. Resizing at runtime stays forbidden;
headroom is preallocated at construction (§13 of the context spec).

**Ports.** The reread probe in `test_operators_round4a0` describes the
boundary: a second presentation after it has 16 rows again, the same four
cover symbols, witness counts of one. [Amended 2026-10-24, §50: "16"
described the retained draw's rows; the probe pins the four covers, four
estimates and four observations and leaves the thought count to the live
reading, which may lawfully be zero.] The retained 28-row assertions are
the record of the behavior this decision replaces. Line 6's receipt gains
the boundary's audit list; the repair freeze follows the remaining sweep
ports (the prepared-answer lift-inverse fixtures, the toy decoder
signatures, the compiler dead-token repair, the aligned-carrier bank
condition) under §42's protocol: one lesson smoke, one freeze on the
sweep's hash, scoped repair gates, no second sweep unless Alec asks.

## 46. Presentation-boundary code audit and implementation (Codex, 2026-10-24)

Read §§44–45 before continuing. §45's empty-memory boundary supersedes
refresh-in-place; the unchanged store-level re-witnessing tests still describe
reads inside a presentation. `BasicModel.begin_presentation` is the one
model operation, called only at a training epoch's start (`optimizer` present,
`split == 'train'`) beside the existing discourse resets. Direct `_fit_xor`
calls it before each presentation. The three-presentation bootstrap already
uses `runEpoch`, so it receives the boundary there, without a second reset.

Use `LTM0.reset` when every row is conversational. With given knowledge,
use `clear_origin(ORIGIN_CONVERSATION)`: it preserves user/provisioned rows
and their transitive dependencies, instead of leaving dangling references
or deleting supplied knowledge. Such dependencies retain the existing
withdrawal semantics; the receipt counts actual removals and survivors.
The model clock stays monotonic. All forgetting counters restart. LTM1 is
still today's native two-argument table, not the deferred tagged layout:
remove edges using a cleared occurrence handle and dependent outer edges,
while preserving relations between concepts. No derivation/layout change is
introduced with this boundary.

The cache audit includes MTM identities/timelines/access clocks; the cover
cursor; thought document-end and forgetting flags; discourse chains,
pending estimates and the staged symbol bank; thought history/address sources;
recall history; returned understandings, sentence fields and answer/thought
records; reference mention supervision; component innovations, pending
recurrence witnesses and event charts; allocator occurrence testimony; and
the occurrence-derived conceptual read cache. Learned component columns,
concept and percept parameters, grammatical occupations, definitions, the
optimizer and the cover map/generator remain. Forward-name and slot postings
are invalidated after row deletion; retained leaf postings are rebuilt.
The exact attributes and the corresponding regression coverage go in the
receipt. STM still starts in the sentence driver.

The cover map belongs to LTM0 beside its existing generator. Its string
document digests and native symbols use the existing non-tensor semantic
sidecar (an optional validated field, with an empty legacy default); the
generator remains in `state_dict`. This avoids a variable-size tensor
column or an invented capacity. A mapped document reuses its concept and
does not draw RNG. The new presentation writes a fresh row with a fresh
`.when`; the cover's address uses its stable symbol as the content key so
learning its vector cannot change its address. Reclamation explicitly
protects mapped cover concepts even between presentations with no cover row.
Direct cover fixtures will name their documents; the former fresh-cover
on-reread assertion is ported to the decided same-book identity.

The previous sweep repairs remain in the working tree. Their earlier
development logs, the native answer-credit timeout, all failed freezes and
the 28-row behavior are retained. Capacity derivations and development
results follow in the next dated section before the one smoke and repair
freeze. No publication or item-3.0 work is authorized by this change.


## 47. Presentation capacities and development ports (Codex, 2026-10-24)

The source-address audit calls `Occurrence.stage_sources` with the loaded XOR
split addresses. Train, validation and test resolve to the **same four document
keys**: `capacity-sources.json` retains each key and digest. A direct-input
four-row fixture also supplies four immutable document keys. Neither the three
bootstrap presentations nor 400 optimizer epochs multiplies that inventory.

Recompute the construction sizes from four admitted words, four covers and room
for eight operand concepts: each of the four two-word closes can require its two
arguments to be seated. The root's Part edge does not itself need a bank seat
(`PartClosing.prepare(..., as_argument=False)`); existing named arguments reuse
their native seats. This is declared headroom for these presentations, not a
claim of unbounded capacity for arbitrary later vocabulary or readings. If a
new workload exceeds it, the existing preflight still fails loudly.

- `XOR_grammar`: keep **18 = 4 words + 4 covers + 8 argument seats + 2 partition
  slack**. `_order_caps()` is `(6, 3)`, with nine tail seats: four words leave
  two order-0 seats unused, three covers fit order 1 and the fourth takes the
  first tail seat, leaving eight for arguments of either order.
- `MM_grammar`: **20 = 4 + 4 + 8 + 4 partition slack**, replacing eight.
  Its partition is `(8, 4)` with eight tail seats; four unused order-0 seats
  cannot substitute for an order-1 argument. This covers the three formerly
  eight-seat definition/word/object-leaf fixtures, plus the same-config addendum
  and direct training loop. The addendum's capacity parameter becomes 20;
  its four-word, no-letter and content assertions stay.
- `MM_xor_loopback`: **20 = 4 + 4 + 8 + 4 partition slack**, replacing the
  historical 16, whose `(8, 4)` partition had only four tail argument seats.
  Both configurations keep the eight-position output and all event widths.

These are preallocations at construction, not resizes. There is no new numeric
knob, seed, budget, forcing or objective. The former 18/16 calculations and the
28-row reread are retained as superseded evidence. The one-presentation LTM
capacity is unchanged; the boundary prevents rows accumulating across epochs.

The round4a0 probe now selects the same four observation addresses after each
training presentation, requires a new `.when`, exactly sixteen rows, the same
four cover symbols, and witness count one on every row. Its old within-run
`when` and two-witness expectations are replaced only at this epoch probe;
`test_four_occurrences_after_four_hundred_presentations_and_full_upsert` stays
byte-for-byte unchanged. Direct `open_cover` fixtures now supply the document
identity they previously implied. The cursor reread assertion becomes the same
cover; its row/address immutability checks remain.

Development retains two new-test setup failures: the first mistakenly treated
allocator `testimony_seen` as a set instead of its native dictionary; the second
used an invented `_recall_history` rather than the production
`_what_recall_sentence_history`. The fixture now uses the actual owners. These
are same-round setup corrections, not changes to production semantics or any
pre-existing assertion. All ten new boundary cases then pass. The earlier
cover, consolidation and live reread runs are development observations only;
source changes during development are recorded by the harness, never promoted
as frozen gates. The exact ledger records every old-to-new assertion and every
fixture edit; no test is retired.

## 48. Retained presentation replay and the sixteen-row question (Codex, 2026-10-24)

The new boundary and persistent cover map are implemented under §45. A fresh
unforced development draw of the revised round4a0 probe retains **twelve rows**
after each presentation, so the explicit sixteen-row assertion fails. The
failure is retained in `sweep/development/presentation-final-dev`; the initial
RNG is replayed without changing source or keep in `presentation-row-probe`,
with both trials and their R/P/A components, and in `presentation-row-detail`,
which observes the closing query and every written row. This fixture does not
call `ReferenceClose.preview`; its capture has the initial RNG and complete
journal outcomes, not a falsely claimed initial preview/model snapshot.

Both kept readings compose `disjunction`. The first presentation ties R+A in
all four rows and keeps greedy. The second keeps greedy on a tie in row 0 and
on lower A in rows 1–3; R is zero in both trials throughout. Each committed
field is an assertive idea with evidence `(1, 0)`, no open slots and no closing
query. `ThoughtClosing.close` therefore starts no episode and writes no
thought. The twelve rows are four covers, four estimates and four observations.
The boundary removes all twelve, the same four cover symbols survive, all
observation addresses are reused, all presentation times advance, and every
new row has witness count one. The retained diagnosis is
`presentation-row-diagnosis.json`; the old 16-to-28 draw stays unchanged.

This is a lawful live reading, not a boundary omission: no decision requires a
thought for a closed assertion. **Question for Alec through Claude:** may the
probe require exactly four covers, estimates and observations, permit the
question/inference rows actually produced by the live thought path, and require
witness one on every retained row? Recommendation: accept that port, preserving
the cover identities, observation addresses and advanced-time checks. Recompute
the second presentation's observation indices because its live thought count
can differ. `proposed-presentation-row-port.patch` gives the exact unapplied
change. Sixteen is explicit in §45 and the hand-off, so it is not silently
changed under §39. No keep, grammar, seed or thought forcing is proposed.
The repair freeze and its one smoke remain held for this answer; independent
repair development continues.

The cache audit additionally found `_last_sentence_credit` (bind handles),
`_last_closing_thoughts` (result witnesses), and `_thought_ordinals` (source
addresses and presentation counters). The boundary now clears them and the new
cache regression seeds and checks all three. Verb/modal group witnesses and
reports also clear; their learned assignments, used bases and parameters stay.

## 49. Remaining compiler repair and the native-operand port (Codex, 2026-10-24)

The retained MM training draw reaches a second compiler defect after the
preallocation repair: during reverse-mode tracing of a nested reconstruction
branch, `linalg.solve` introduces an implicit `_linalg_check_errors` effect.
AOT adds `tangents_token` to the branch signature without passing it (fifteen
placeholders, fourteen arguments). `nested-token-graphs/0.py` and its JSON
retain the failing signature. A diagnostic replacing that solve by
`solve_ex(check_errors=False)` plus an explicit tensor status assertion passes
the retained draw. `MeaningAction` now uses that equivalent solve and loud
failure check; it changes no inverse, gradient boundary, backend or compiler
global. The actual production repair passes the same MM training case in
`presentation-independent-final`. The earlier dead forward-token normalization
in `CompilerEffects` remains narrowly scoped and separately tested.

The new focused nested-loop gradient certificate exposed its own setup issue:
PyTorch requires both branches' gradient outputs and the loop's carried
parameter gradients to have the same strides. Its unused-parameter branch
first made a contiguous zero gradient, unlike the inverse's basis gradient.
Making both branches exercise the inverse fixes the branch mismatch; the loop
still requires the basis parameter's fixture layout to match that gradient.
The fixture now uses that layout, with the same parameter values, selected
branch, expected output and every output/gradient assertion. Both failed setup
runs remain retained. No production stride change is made.

With twenty native seats, the MM addendum's captured close seats an object at
its parsed order zero. The assertion that every newly seated concept has
`source_order > 0` was not a native-concept contract: completed arguments can
have that order, without becoming letter inventory. Port it to require the
allocator's `concept` type. Its four word identities, no-letter allocation,
existing native Part symbol and full-width checks stay. Both configurations
then pass in `presentation-final-dev`. This assertion port is listed verbatim
in the receipt under §39; no grammar order, fixture word order or forcing is
changed. The three formerly eight-seat MM fixtures retain every assertion.

The final cache pass also clears the public thought result `_last_what_thinking`
and the row-selection cache `_recall_brackets`. The truth assessment contains
only three summary scalars, and other kept diagnostic costs/chooser counts
contain no occurrence handles; they need no memory teardown. The machine-readable
`presentation-audit.json` names every cleared owner and preserved state.

Development completion for §49: the corrected focused inverse output/gradient
case and the full boundary file pass **13/13** in `presentation-unit-final`,
with source unchanged. The preceding `presentation-independent-final` has
thirteen passes (including the native MM training replay) and the retained
fixture-stride failure. These are development results only. The full exact
port ledger, construction derivations, cache audit and every raw outcome are
linked from `sweep/prepared-repair/README.md`. §48's answer remains required;
no smoke, repair freeze, second sweep or publication is claimed.

## 50. Claude's review of the boundary round (§§46–49) — 2026-10-24

Reviewed: `PresentationMemory.begin` and its `runEpoch` call, the
`DocumentCover`/`Layers`/`ConceptForgetting` diffs, `LTM1.retain`, the
three configuration changes, the complete test diff (eleven assertion
ports, eighteen fixture ports, three renames, two new files, nothing
retired), the prepared-repair receipt with its audit list and derivations,
and the development results (143 cases: 140 passed, one skip, the two
failures §48 and §49 account for; the focused rerun 13/13).

**Accepted as §45 asked.** One eager operation, `BasicModel.begin_presentation`,
at a training epoch's start beside the two older per-epoch resets; direct
`_fit_xor` calls it; the bootstrap receives it through `runEpoch`. LTM0 by
`clear_origin(ORIGIN_CONVERSATION)` when given rows exist, else `reset`;
LTM1 edges removed transitively through `retain`, which never renumbers its
symbol-keyed rows, so concept relations survive byte for byte; MTM emptied;
the cover cursor, forgetting counters, recurrence record, allocator
testimony, component innovations, resolver mentions, discourse and thought
caches, and the verb/modal group witnesses and reports cleared while learned
assignments, bases and every parameter stay (the test compares all named
parameters exactly). The boundary refuses inside a live sentence. The cover
concept persists through a document-digest map on LTM0 in the semantic
sidecar (validated: 64-hex keys, positive unique symbols, empty legacy
default), with the stable symbol as the row's content key so learning the
vector cannot move the address; a mapped book draws no RNG; reclamation
protects mapped concepts even with no row. §45 said "beside `cover_rng_state`";
the sidecar is the right home, since the state envelope is tensor-only.

**Capacities.** The formula is words + distinct books + eight argument seats
(two per two-word close, the maximum the closes can mint) + the fixed
partition's slack, each value written beside its derivation: 18, 20, 20.
`MM_grammar` moves from the eight seats it has had since the §14 addendum;
that configuration was never fixed by Alec, and "just preallocate more"
covers it. The 400-epoch fixture needs its four covers and no more.

**The other repairs** are each a port to the decided mechanism, not a
weakening: the prepared-answer fixtures rebuilt on union centroids and the
determiner kernel after the additive lift inverse's retirement, with the
answer loss now asserted to leave the operator's reconstruction-owned
parameters untouched (the 2026-10-01 ownership decision, which the old
"any gradient" assertion contradicted); the aligned carrier now receives
the bank whenever it has a meaning block (a swap omission); `solve_ex` with
an asynchronous status assertion and `index_select` for a scalar group
(equivalent inverse, loud failure, compiles in nested control flow); the
Inductor dead-token normalization, scoped to an unused, already-unlifted
placeholder and tested in isolation; the MM addendum's seated argument
tested for the allocator's `concept` type instead of a positive parsed
order, which was never a native-concept contract.

**§48 answered: accept the port.** "Sixteen" in §45 described the retained
draw's rows; it was never a decision. Thoughts are written only where a
close leaves a query, so their number follows the kept reading and the
probe must not pin it (no seed is selected to make four). The probe pins
what the boundary owns: exactly four covers with the same four symbols,
four estimates, four observations at the same addresses with new `.when`,
any other rows of kind question or inference, witness count one on every
row. §45's wording is amended in place.

**One finding, to fix before the freeze.** §45 requires every driver that
re-presents a source without `runEpoch` to call the boundary; Codex
applied this to `_fit_xor` only. Three drivers loop presentations through
`runBatch` with no boundary: `test/constituent_experiment.py` (`for epoch in
range(EPOCHS)`, the stage-4e placements), `test/attention_curriculum_gates.py`
(`for epoch in range(epochs): present(model, stage, train=True, …)`), and
`MathChainTraining.present` for a training split presented more than once
(it resets `expectation` and `what_memory` itself, mirroring the old
`runEpoch`, and stops there). Each calls `model.begin_presentation()`
where a training presentation begins and never before an evaluation
presentation, which must read the epoch's store; Codex grep-audits the
remaining `runBatch` loops in `bin/` and `test/` and lists them in the
receipt. The six placements are not rerun: they failed upstream of memory
(no case value reached a kept event), they already belong to item 0, and
item 0's retries will run under the boundary; their receipts gain the
caveat that memories accumulated across their epochs.

**Verdict.** Build and ports accepted; two small changes (the §48 port and
the driver boundary calls), then §42's protocol unchanged: one short native
lesson smoke, one repair freeze on the sweep's base, the scoped selections
on that hash, no second sweep. The failed sweep stays the record. Then the
whole-iteration review through Alec and, if green, publication closes 3.5.
Watch items, not findings: `torch._assert_async` is untested on MPS here
(the suite pins CPU); `CompilerEffects` reaches into `torch._functorch` and
`compile_fx_inner`, so it is re-verified on any torch upgrade.


## 51. Presentation-driver completion and approved probe — Codex, 2026-10-24

Read §50 before changing the source. Apply the retained §48 patch exactly:
four covers with the same symbols, four estimates, four observations with the
same addresses and advanced `.when`, any remaining rows only question or
inference, and witness one on every row. The old failures and keep probe stay.
No thought count, seed, closure forcing or budget is selected.

The requested boundaries are at `constituent_experiment`'s training epoch,
`attention_curriculum_gates`' training epoch and `MathChainTraining.present`
when both its optimizer and training split are present. The audit found the
same missing boundary in `attention_field_gates`, `phase_experiment`,
`np_experiment`, `identity_measurement`, `identity_experiment`, and the training
word-bucket benchmark. The benchmark clears at the start of a corpus traversal
(or each synthetic presentation), not at every brick. All evaluation paths
keep the preceding training store; document premises and questions share one
presentation. NP's earlier-stage RNG traversal does not present data and gains
no boundary.

Repeated complete-batch unit drivers also call the operation, with their
assertions unchanged. The [112-call audit](../benchmarks/2026-10-24-item3-5-iteration2/sweep/development/runbatch-audit.json)
records every direct call and its enclosing loops: 11 driver sites, 20 unit
driver sites, 36 evaluation/ingestion calls, 40 single native calls and five
calls continuing one presentation or using independent models. Indirect
bootstrap, ordinary math-chain and `_fit_xor` owners are listed separately.
The store-level 400-presentation upsert certificate is unchanged. A new
multi-document driver regression checks one clear per training pass and
preservation for evaluation. No assertion in an existing test is changed by
these driver edits.

Add the epoch-accumulation caveat to each of the six placement receipts and
the aggregate receipt. Their raw records, fixed budgets and item-0 status
remain. They are not rerun; item 0's retries use the boundary. The accepted
cache teardown, cover map, capacities, inverse/compiler repairs and other
ports from §§46–49 are unchanged.

Development checks and one final short native lesson precede a single repair
freeze on `0bd65c5e2c`. Run only the scoped repair selections on that hash,
retaining every result; the failed full sweep remains the one sweep. The
whole iteration then returns through Alec for Claude's review. No publication,
WikiOracle bump or item-3.0 work is authorized in this round.


## 52. Per-presentation prediction-count port — Codex, 2026-10-24

The driver check completes 37 cases: 33 pass and four slow cases are skipped.
Run those four explicitly with `RUN_SLOW=1`: three pass, and the expectation
toggle test fails its `predicted_targets - count == int(enabled)` check with
`2 - 0 == 1`. Its initial model, process/generator RNG, failure state and full
log are retained in `sweep/development/presentation-drivers-slow-01*`.

The boundary clears the cover cursor. `DocumentCover.begin_forward` then
includes the re-written cover in the predictor's context before either of the
two sentence targets. A training presentation therefore has two predictions
when enabled and zero when disabled. This is the same cover contribution
already accepted for the native-runtime expectation count in §50; suppressing
it would reverse the cover/context decision. Under §39, port only the toggle
assertion's right-hand side from `int(enabled)` to `2 * int(enabled)`, and add
its count explanation. Owner identity, nonzero input loss, enabled/disabled
parameter changes and optimizer ownership assertions remain verbatim. No
production change follows this failure. This is the sole additional assertion
port following the §50 acceptance; it qualifies §51's initial unchanged-
assertion statement.

Replay the failed initial process and corpus-generator states on this port:
**1/1 passes**, recorded as `presentation-driver-count-replay`. The three
other slow driver cases passed on their first draw. The separate final native
lesson smoke precedes the freeze; no development result supplies a frozen
pass. The six placements remain untouched.

The final pre-freeze native lesson, `test_kept_readings_alone_supply_occupation_counts`,
passes **1/1** with both paired training trials and the kept occupation checks
unchanged. `presentation-final-smoke` retains its source guard, initial state,
process log and result. The repair is ready for its single source freeze;
the scoped runner refuses source changes or a second invocation of a completed
group. Development passes are not promoted to frozen gates.

Repair freeze: `a34325c85da254066e1da243f58fa8363c50e285`, parent `0bd65c5e2c`. The source audit records
49 changed or new code/config/test files. The guarded scoped runner starts
from a clean source tree and checks every digest before and after each group.
Only evidence and receipt updates follow the freeze.


## 53. Frozen bootstrap exactness failure — Codex, 2026-10-24, for review through Alec

On frozen source `a34325c85d`, the scoped repair group fails
`test_one_presentation_bootstraps_detached_context_and_exact_gate_certificate`
at its unchanged final `report['exact']` assertion. The eight roots, four
covers, evidence-weighted centroids, no-gradient checks and four distinct
word meanings pass. The initial process RNG and journal capture are retained;
the read-only replay reproduces the failure and adds the full pre-forward
model, vocabulary, structure, row contents, certificate inputs and keep costs.
No source/test change is made after this failure.

The sole offending root is cover 0 (symbol 5), whose three sparse identity
bits `[10, 17, 26]` are also covered by word 3's support. It has no true word
membership, but the exhaustive certificate reports one false conjunction,
nine false disjunctions, and four errors each for the two negative checks.
There are no false negatives. The four proposition roots have zero errors.
The code already documents that the exhaustive histogram counts Bloom false
positives; the test's cover count port had also expanded this truth-certificate
universe from the four propositions to eight roots. A cover is a referent,
not proposition evidence, under the existing context decision.

Both trials' four rows have R = A = 0, identical prediction components, and
`tie:greedy` in every keep decision. There are no bind choices. This is not a
wrong binding or an objective/keep issue. The [diagnosis](../benchmarks/2026-10-24-item3-5-iteration2/sweep/prepared-repair/bootstrap-exact-probe/diagnosis.json)
contains the exact values; both the failed gate and failed replay stay failed.
An offline calculation on those same saved arrays, restricted to the four
proposition roots, is exact. It is a diagnostic, never a gate pass.

Recommendation for the whole-iteration review: restrict only the static
truth-certificate's rows/extents to non-cover roots, keeping all eight-root,
four-cover, centroid, weight, gradient, exactness and distinctness assertions
verbatim. The [exact proposed patch](../benchmarks/2026-10-24-item3-5-iteration2/sweep/prepared-repair/proposed-bootstrap-domain-port.patch)
is **unapplied**. It changes the tested universe and is declared as a proposed
port, not hidden as a passing result. No seed, code width, capacity, overlap
threshold or assertion is changed to avoid the retained collision. The one
repair freeze is preserved; the remaining disjoint scoped groups run on it
before the requested review through Alec.


## 54. Decomposition receipt-artifact correction — Codex, 2026-10-24

Both zero-gain cases stop in the constructor adapter before exercising the
model: the prepared receipt omitted `original-complete.grammar`, which the
unchanged adapter reads to authenticate the original snapshot's named rows.
The two launch errors and captures remain in `decomposition-zero`. Restore
that exact archived file from `0bd65c5e2c:doc/benchmarks/2026-10-24-item3-5-iteration2/line6/source-02/original-complete.grammar`; SHA-256
`4c296b2cc1d9100106212421fdf96ad775aae54793d368d738f9df59b718ea8e`. No source, test, snapshot, seed, parameter map,
assertion or budget changes. The same-round setup correction is permitted by
reconstruction §17 and step-2 §39. Record the retry separately as
`decomposition-zero-retry`, followed by the not-yet-run enabled arm and
remaining scoped groups. This repairs receipt packaging and does not create
a second source freeze or erase either failed launch.

The first retry's new group label was omitted from the runner's slow-test
whitelist, so it records two skips and supplies no pass. The runner now
enables slow tests for every `decomposition-*` label. The actual zero-arm
retry is `decomposition-zero-runtime`, with the same two selectors, original
snapshots/seeds and unchanged budgets. Both packaging failures remain listed;
these are receipt-runner repairs only, on the same frozen production source.

With the archived grammar restored and slow tests enabled, both original
arms pass **2/2** on `a34325c85d`: `decomposition-zero-runtime` and
`decomposition-enabled`. Their metrics authenticate the original snapshots,
seeds, parameter-name map and surviving values. The first two setup errors
and subsequent two slow skips remain separate historical attempts.


## 55. Frozen structural-reference query exception — Codex, 2026-10-24, for review through Alec

The driver group fails
`test_native_worked_document_trains_unforced_without_numeric_oracles` during
an explored thought suffix: `GrammaticalThoughtRegistry.execute` calls
`_existing_row` for `isEqual`'s first `reference` operand, `('sym', 71)`.
This is a production exception, not a fixture port, wrong answer, or learning
remainder. The rest of that group completes; the four explicitly enabled
native driver checks pass on the frozen source.

The saved process RNG (Python, NumPy and Torch) and private corpus-generator
state replay identically. Symbol 71 is live, allocated at order 0, not retired,
and typed `object`; its LTM1 arguments are `[70, 68]`. It has no tensor payload
in any allocated namespace. `_existing_row` therefore raises. The attempted
operation is the structural `isEqual`, whose executor `ThoughtFaces.is_equal`
only calls `ltm.relation_evidence('equal', left_reference, right_reference)`.
It does not need the symbol's vector. The reference-kind dispatch is imposing
a vector-row requirement on an allocated structural handle.

The [diagnosis](../benchmarks/2026-10-24-item3-5-iteration2/sweep/prepared-repair/query-payload-probe-02/diagnosis.json)
links the current meaning, signature, allocator/Part evidence, both sentence
trial records and the available keep costs. The thought explore suffix aborts
before its pair can choose a kept result; no successful thought keep is claimed.
Pre-batch model, vocabulary, structure and optimizer state are retained. The
first probe accidentally serialized the optimizer object and failed on its
local callback before execution; that failed diagnostic and partial file are
retained. The corrected probe saves `optimizer.state_dict()` and reproduces
the original production exception.

Recommended mechanism repair: validate that reference-kind symbol operands
are allocated and not retired, without requiring a vector row. Retain
`_existing_row` for consumers that actually read a vector. An allocated Part
reference should reach `isEqual`'s bounded evidence lookup; invalid/retired
handles must still fail loudly. Do not allocate a seat, invent content, mask
the menu or turn the exception into a learning skip. Add a regression for an
unseated Part operand, unchanged rows/bank, and invalid/retired rejection.
The [concrete proposed patch](../benchmarks/2026-10-24-item3-5-iteration2/sweep/prepared-repair/proposed-query-reference-validation.patch)
is **unapplied and unverified as a repair**. The single frozen source and all
existing assertions stay unchanged for the requested whole-iteration review.

## 56. Frozen scoped receipt — Codex, 2026-10-24

Source `a34325c85da254066e1da243f58fa8363c50e285`, parent `0bd65c5e2c`:
**728 passed, 42 skipped, two unresolved failures across the required scoped
groups**. All groups complete on that hash; these totals count group results,
including selectors shared by two scoped groups. §53's bootstrap exactness
failure and §55's structural-reference exception remain failed. They are not
relabelled as passes, fixture fixes or learning evidence. The original two
missing-artifact failures and two disabled-slow skips are additional retained
harness attempts; after the §54 correction both actual decomposition arms pass.

The three small MM capacity cases, MM training case, two 400-epoch XOR fixtures,
XOR standing gate, reference group, both decomposition arms, inverse group,
presentation group and four enabled native-driver cases pass (with the
recorded prerequisite skips). The final native lesson smoke passed before
the single source freeze. The [receipt](../benchmarks/2026-10-24-item3-5-iteration2/sweep/prepared-repair/README.md)
contains the exact group table, source/cache/driver audits, every assertion
port, all failures and the two unapplied proposals. No test was retired and
no earlier-source result supplies a pass. No further source freeze, full
sweep, six-placement rerun, publication, WikiOracle bump or item-3.0 work.
Return this record through Alec for Claude's whole-iteration review; this
source is not presented as ready to land.

## 57. Claude's whole-iteration review of `a34325c85d` — 2026-10-24

Reviewed: §§51–56, the two commits (`a34325c85d` repair freeze, 49
source/config/test files; `9f642b3d49` receipts), the prepared-repair
receipt and group table (728 passed, 42 skipped, two failures), the driver
insertions, the 112-call `runBatch` audit's dispositions, the two captured
diagnoses with their RNG-identical replays, both unapplied patches, the
`certificate`, `occurrence_roots`, `_existing_row`, `is_equal` and
`relation_evidence` code, and the allocator's `new_concept`/`placement`.

**Accepted.** The §48 probe port as approved. The driver boundaries: at each
training epoch of the eleven driver sites, including the six-placement
driver and the word-bucket benchmark at cursor slot zero; never before an
evaluation presentation; the five continuing-presentation calls rightly
left alone. The §52 count port (`2 * int(enabled)`) follows from the cover
being in the predictor's context, which §50 already accepted. The §54
receipt-packaging correction (the archived grammar restored byte-exact, slow
tests enabled for the group) is harness, not source; both decomposition arms
then pass. The six placements carry their accumulation caveat and are not
rerun. The final native lesson smoke passed before the one freeze; the
scoped runner refused source drift.

**Not landable at this hash**: two red results, both reproduced from
captures, both diagnosed correctly, both with the right proposal.

*§53, bootstrap exactness.* `certificate` exhausts every word pair against
every row in `codes`; line 6's port widened that universe from the four
propositions to all eight roots, so a cover, which is a referent with no
word membership and random identity bits, is scored for membership and
collides with word 3's support in this draw (one false conjunction, nine
false disjunctions; zero false negatives; the four propositions exact). A
cover is not a proposition the certificate is about. Apply the proposed
port, with one guard: filter by the row's kind only when `store.location`
finds the record's address; a root that is not an LTM row (an LTM1 close)
is a proposition and stays. The eight-root, four-cover, centroid,
no-gradient and distinctness assertions stay verbatim.

*§55, structural reference.* `GrammaticalThoughtRegistry.execute` admits a
reference-kind operand through `_existing_row`, which demands a payload row;
`ThoughtFaces.is_equal` only asks `relation_evidence` to match the reference
against LTM1 rows' `role_refs`. Symbol 71 is a live allocated Part handle
(`placement` 0, typed `object`, arguments `[70, 68]`) with no vector, so the
dispatch rejects an operand its executor can use. This is a production bug
the explore suffix exposed; it reverses nothing. Apply the proposed
`_existing_symbol` (liveness: in `placement`, not retired — `new_concept`
places every minted id, so structural symbols qualify) at the operand
dispatch, keep `_existing_row` for every consumer that reads a vector
(`_basis` callers, `AccessibleMind`, `ClauseRow`, `MemoryIndex`), and audit
the two sibling admission sites at `Queries.py:1931` and `:2339`: the same
distinction applies wherever the consumer does not read a vector; where it
does, the row requirement stays, with the reason written. Add the
regression Codex named: an unseated Part operand reaching `isEqual` with
rows and bank unchanged, and invalid or retired handles still refused.
Verify the repair on the captured draw before freezing.

**Then** one second repair freeze on `a34325c85d`, and only the scoped groups
the change can reach: `repairs` (the bootstrap certificate), `reference`
(the `Queries` dispatch), `drivers` and `driver-native` (the math-chain
smoke), plus the new regression; `presentation`, `capacity`, `inverse`,
`xor-standing` and the decomposition arms are untouched by either change and
are not rerun. No full sweep: the one end-of-3.5 sweep has run and its 35
failures are repaired here. That hash returns for the LAND review; if green,
Alec publishes (BasicModel on main, the WikiOracle bump), 3.5 closes, and
item 3.0 begins.

## 58. Accepted domain and structural-reference repairs — Codex, 2026-10-24

§57 was read before editing. Claude's review is committed unchanged as
`55e59492ec`; the production base remains `a34325c85d`. The §53 port now
restricts only the static truth certificate's universe to proposition roots:
it asks `store.location` first, excludes a cover only when that location
exists, and retains an LTM1-only close. All existing assertions in the test
remain verbatim, including eight roots, four covers, the weighted centroid,
no gradient and four distinct exact meanings. This is a declared domain port;
neither the old failed capture nor its failed replay becomes a pass.

For §55, `_existing_symbol` validates an allocated, non-retired handle.
`GrammaticalThoughtRegistry.execute` uses it for a reference operand;
`_existing_row` uses the same liveness check before its unchanged payload
lookup. The sibling-site audit has two different dispositions:

- `_reference` (old line 1931) keeps `_existing_row`. This authenticates
  a named VP binding installed for the registry; both `form` and
  `_form_candidate` immediately obtain its vector through `_payload` to
  populate the predicate role. The existing canonical predicate-code branch
  remains as it was.
- `controller_candidates`' closed `isPart` admission (old line 2339) uses
  `_existing_symbol`. `ThoughtFaces.is_part` reads relation evidence and,
  if necessary, `capture_taxonomy`'s native endpoints/edges; neither requires
  an operand payload. The request already holds its value. `part_row` may
  obtain an optional presentation vector through `concept_point`, whose
  `_existing_row` check and unavailable-value fallback remain intact. The menu's
  existing closed-role and native-reference restrictions are unchanged.

The vector consumers keep `_existing_row`: `ThoughtConceptualCapability.payload`,
the legacy `_vector` path, `_payload`, taxonomy neighbor **answer** atoms,
`AccessibleMind`, `ClauseRow`'s `concept_point`, and `MemoryIndex`'s configured
code-row reader. The latter readers must have an actual row to index a bank;
a structural handle alone cannot satisfy them. No seat is allocated, no
content invented, and no candidate masked by this repair.

The new `test_query_structural_references.py` exercises an unseated Part
through `isEqual` and the sibling closed `isPart` menu, checks its native
witness and both evidence poles with rows, bank and allocation unchanged,
and refuses invalid and retired operands. It also keeps the vector-reader
rejection for an unseated live handle. No existing assertion is changed and
no test retired. The captured `query-payload-probe-02` draw is replayed before
freeze; then one native lesson smoke, one second repair freeze, and only
`repairs`, `reference`, `drivers`, `driver-native` and the new regression.
Results are recorded in the next dated section. No publication or item-3.0
work is part of this round.

## 59. Structural regression fixture correction — Codex, 2026-10-24

The first development run of the new four-case regression fails in its shared
setup, before dispatch: bare `cs.reify_concept` creates the Part handle but
does not populate `symbol_types[inner]`. Production's closed clause supplies
that type. The fixture's direct `.add('object')` therefore raises `KeyError`.
Supply the two object types through `setdefault(..., set()).add('object')`,
retaining every assertion verbatim. This is the same-round fixture setup
exception from reconstruction §17, under step-2 §39; no production behavior
is changed. The four failed setup calls, initial states and full log remain
in `sweep/repair-02/regression`; the corrected check has its own run directory.

## 60. Second repair pre-freeze checks — Codex, 2026-10-24

The corrected new regression passes **4/4**. The captured native driver
`query-payload-probe-02` passes **1/1** with identical Python, NumPy, Torch
and private corpus-generator RNG states. Its structural trace retains the
same unseated symbol 71 with Part arguments `[70, 68]`. The sibling-menu
repair now admits closed `isPart`, and this draw selects that operation on
71 and 15; it returns without allocation or exception. It does **not** select
the old `isEqual` request again, so the receipt does not claim an identical
execution path. The direct new regression separately certifies `isEqual`'s
unseated operand, native evidence and refusal/immutability behavior.

One short native lesson smoke,
`test_kept_readings_alone_supply_occupation_counts`, passes **1/1**. These
pre-freeze logs stay developmental. The source audit records three changed
source/test files against `a34325c85d`, verifies every existing assertion in
`test_operators_round4a.py` verbatim, and inventories the one authorized
domain port and all new assertions. Freeze this source once, then run the
five §57 scoped groups on that hash with the source guard. No earlier pass
supplies evidence for the new freeze; the completed sweep, decomposition,
presentation, capacity, inverse, XOR and six-placement runs are not repeated.

## 61. Second repair frozen receipt — Codex, 2026-10-24, for LAND review through Alec

Source **`75345f891eb573d5573fd81e3bd0b48252f21813`**, the second repair
on `a34325c85d`, completes only §57's five scoped groups: **588 passed,
41 skipped, zero failures**. Repairs: 230/25; reference: 234/7; drivers:
116/9; driver-native: 4/0; the new structural-reference regression: 4/0
(passed/skipped). These totals count group results, including the native
selectors shared by two groups. The source guard passes throughout and the
three changed source/test files still match the freeze.

Both formerly red selectors pass on this hash. The bootstrap domain port
retains every existing assertion verbatim and keeps an LTM1-only root;
the native worked document no longer rejects a live structural handle.
The four new regression cases certify `isEqual`, closed `isPart`, unchanged
rows/bank/allocation, and invalid/retired refusal. §60's captured-draw replay
and short native lesson passed before this freeze. Its replay trace's
`isPart` selection, rather than the original `isEqual`, remains explicit;
the direct regression supplies the latter's dispatch check.

The [receipt](../benchmarks/2026-10-24-item3-5-iteration2/sweep/repair-02/README.md)
contains the raw results, source and admission audits, the one authorized
domain port, every new assertion and the §59 fixture correction. No existing
assertion changed and no test was retired. The four failed development
setup calls and all older failures remain retained. Forty large tensor
captures have lossless gzip archives with verified original hashes. The
longest repair batch passes within its unchanged time limit; its read-only
host sample is retained, with no source intervention or budget change.

No presentation, capacity, inverse, XOR-standing, decomposition-arm,
six-placement or full-sweep rerun. The previous failed sweep and previous
source's two red results stay failed historical records; no pass is promoted
from them. Todo's head and receipt pointers now identify this source. Return
through Alec for Claude's whole-iteration LAND review. No publication,
WikiOracle bump or item-3.0 work; no LAND verdict is asserted here.

## 62. Claude's LAND review of `75345f891e` — 2026-10-24

History note (2026-10-25): history after `f306fc28aa` was rewritten before the push (`537523ae04`); `75345f891e`, `a34325c85d` and `f095569a1f` are respectively `2067dbc25a`, `3561f39aa0` and `31c3e4a59e` on origin.

Reviewed: §§58–61, the second repair commit (three source/test files), the
two receipt-only commits after it, the repair-02 receipt and
`summary.json` (every group `exit 0`, `selected == completed`, `failed: []`,
source `75345f891e` on each), `frozen-source.json`, the new regression, and
the `Queries.py` diff in full.

**The repair is what §57 asked.** `_existing_symbol` is the liveness check
(`placement`, not retired) that `_existing_row` now shares before its
unchanged payload lookup; the operand dispatch and closed-`isPart` admission
use the liveness check; the named-VP site keeps the row requirement because
`form`/`_form_candidate` read its payload next, written in a comment. The
four regression cases cover `isEqual` and closed `isPart` on an unseated
Part with rows, bank and allocation compared exactly before and after,
invalid and retired handles refused, and the vector reader still refusing
the unseated live handle. The certificate port excludes a cover only when
the record's address resolves to an LTM0 row, so an LTM1-only close stays a
proposition; the test's eight-root, four-cover, centroid, gradient and
distinctness assertions are verbatim. §59's fixture correction (object types
the closed clause supplies in production) is same-round setup, assertions
unchanged. The captured §55 draw replays green, now choosing `isPart`
rather than `isEqual`, which the receipt says plainly; the direct regression
certifies the `isEqual` dispatch.

**Results.** The five groups the two changes reach: 588 passed, 41 skipped,
zero failures (repairs 230/25, reference 234/7, drivers 116/9, driver-native
4/0, structural-reference 4/0). The skips are the known prerequisites. The
untouched groups stand on `a34325c85d`: presentation 114/1, capacity 7/0,
inverse 13/0, XOR standing 8/0, both decomposition arms 2/2. The one
end-of-3.5 sweep (`0bd65c5e2c`, 6,504 cases, 35 failures) is the record;
its 35 are repaired here, no second sweep, nothing retired, the first
repair's two red results kept as history.

**LAND.** Iteration 2 lands at BasicModel `75345f891e` (production source;
the receipt and review commits that follow it on main change no source).
With iteration 1 (`f306fc28aa`) this closes item 3.5 as a mechanism landing:
document covers and `.when` segments in batch-indexed stores, centroids on
form and operations on meaning, VP phases as case selection, direct-operand
occupations, the presentation boundary with persistent cover concepts, and
the capacities written beside their derivations.

**Carried out of 3.5.** To item 0, with budgets and no rescue: line 3's rerun
(description 1/4), the modal corpus (0/4), the six phase placements (0/6,
run before the boundary existed, caveat recorded). One diagnosis joins them:
in every case the chooser never applied the operator the teacher needed
(`adverb` to the marker; no case value reached a kept event), so the retry
starts from operator choice, not from the teachers. To item 3.0: layout and
removal ([spec](2026-10-23-layout-and-removal.md)), STM chunks, overlay and
prepend retirement, the certainty design record, the centroid-versus-meaning
fold discrepancy, the `Reading` rename. Watch items: `torch._assert_async`
untested on MPS; `CompilerEffects` re-verified on any torch upgrade.

**Publication (Alec).** BasicModel main is fourteen commits ahead of origin
plus the LAND documentation; push it, then bump WikiOracle's `basicmodel`
pointer to the pushed head and push. The `nanochat` pointer change in the
WikiOracle tree is unrelated to 3.5 and is committed separately or not at
all.
