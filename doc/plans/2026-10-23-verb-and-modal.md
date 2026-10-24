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
