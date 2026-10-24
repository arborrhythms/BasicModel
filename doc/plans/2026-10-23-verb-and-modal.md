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
4. **Line 5, the modal** (§4), differencing over possible identities.

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
   difference of NP-VPs across an identity's possible identities, as a VP
   is a difference of NPs across time; its form is the adverb** (Alec,
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

## 4. The modal (line 5): differencing over possible identities

Alec (2026-10-23): "The question of modals is exemplified by 'what is the
meaning of possible vs necessary'. The multiple different timelines
approach as a fifth dimensional ordinate over events seems great; I had
previously envisioned two NP creating a VP as a difference, and two NPVP
creating a MP as a difference. But perhaps that is about learning the MP,
and the form of the MP can be as an adverb?" Asked where the timeline
ordinate is stored: "Since identities are used for verb differencing, it
would make sense to collect 'possible identities' for modal differencing."
Asked what the modal learns from: the same.

**Decided, as read by Claude.**

- **A VP is a difference across time, an MP a difference across possible
  identities.** The verb's teacher compares one identity's states before
  and after (§3); the modal's compares an identity's actual state with the
  state of one of its possible identities at the same time ("Joe walked;
  he could have run"). The timelines of the fifth ordinate are an
  identity's possible identities.
- **Possible and necessary.** The indicative is about the actual identity;
  *possible* is true of some possible identity; *necessary* of the actual
  and every possible one, the dual. So a necessary modal learns a small
  difference between actual and possible states on what it asserts, and a
  possible modal a larger one: the two are told apart by what they learn,
  not by name.
- **Form.** The `adverb` rule: `adverb(VP, M)` the *de re* reading,
  `adverb(S, M)` the *de dicto*; occurrence §14.3's `position` with minted
  worlds is withdrawn.

**In code names (Claude's reading, corrected on checking, 2026-10-23).**
The first version put a "new possible identity" entry beside every
accessible identity in the bind menu. Checking found three gaps: it
changed binding at initialization (more options, so G1–G3 and the
twelve-draw series would move); the actual-or-possible choice had no
credit, since the evidence arrives in later sentences, after the kept
close where credit ends; and nothing in LTM recorded possibility once MTM
let the token go. The corrected mechanism:

- **The modifier carries actuality.** `AdverbLayer`'s meaning rule gains
  `p(m)`, a learned readout of the modifier's meaning, soft-thresholded
  and clamped to [0, 1], zero-initialized, so every modifier starts actual
  and nothing changes until it is learned; binding is unchanged.
- **Possible identities.** `MediumTermMemory.Token` gains `actual`, the
  token it is a possible identity of. At the kept close, an event carrying
  a modifier with `p(m) > ½` is written to a possible identity of its
  subject, opened with the subject's current value and `actual` naming it;
  the VP replaces the possible identity's state, never the actual's
  ("Joe could have run" does not make Joe tired). A possible identity is an
  ordinary entry for capacity, expiry and checkpointing.
- **Durable mark.** The event's LTM0 rows get a new record kind,
  `possible`: queries for what happened skip it, as they skip questions
  and estimates, and a question whose own modifier reads as possible
  reads it. "Did Joe run?" and "could Joe run?" are then answered from
  different rows after MTM has expired.
- **`modal_teacher`, the difference of two NP-VPs.** When a later kept
  close observes the actual identity A (an event, a description, a copular
  property or an equality) after an event e carrying modifier m, the
  observation is compared with A's state with e applied and without it,
  all detached; the closer one is the target for `p(m)` (possible when
  "without" fits better), a binary cross-entropy term. "Joe could have run.
  Joe walked." teaches the modifier toward possible; "Joe ran quickly. Joe
  is tired." teaches it toward actual. Nothing names a marker.
- **The modifier's diagonal** (`W_m`, `Q_m`) trains through the derivative
  within the sentence, like the other kernels.
- **Possible versus necessary.** The definitions above stand. This
  iteration builds one possible identity per modal event, which separates
  the actual from the possible; telling necessity from possibility needs
  the actual and several possibles to be compared, and stays this line's
  recorded open point.

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

9. With every `p(m)` at zero, binding, identity values and every existing
   assertion are unchanged.
10. A modifier with `p(m) > ½` opens a possible identity whose `actual`
    names the subject's token and whose value starts at its state; the VP
    replaces the possible identity's state; the actual state is unchanged;
    the rows are written with kind `possible` and skipped by queries for
    what happened.
11. A modifier whose event the actual identity's later observation
    contradicts fires `modal_teacher` once and moves its `p` toward
    possible; one whose event the observation bears out moves toward
    actual. The fixture's markers are corpus tokens, never named in code.
12. A modal marker reaches both bracketings through `adverb` in a
    fixed-operation fixture.
13. Line 5's learning gate (todo): generated corpora with a modal marker
    as a free leaf and as a verb affix; "did" and "could" questions
    answered from the actual and the `possible` rows; budgets and seeds
    fixed in advance; reported, not rescued.

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
- *The modal* (Alec, 2026-10-23): collect possible identities for modal
  differencing; the MP is learned as a difference of two NP-VPs; its form
  is the adverb (§4).
