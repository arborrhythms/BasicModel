# Item 1.45: verbs in the field of concepts, and episodic memory as percepts

*Claude, 2026-11-23. A design for Alec, begun at his word. It has no text
for Codex: nothing here is built before items 1.6 and 1.5, and the open
questions of §9 come first. Sections 1 and 4 are Alec's; the rest is
Claude's reading and proposal, and says so.*

## 1. Alec's words

Opening it (2026-11-23):

> I would like to revisit verbs, specifically with respect to how nouns and
> verbs might manifest in extension. We currently have them as matrix
> operators that do modification, but I am wondering about an earlier model
> that we had, where they were basically learned on temporal differences
> between nouns. So after we learn a set of nouns as an extension, and we
> have an intension for those which is the types over them, we learn verbs
> in both cases. The intension is currently the matrix multiply, but there
> is no extension: the verbs are not learned from direct observation, and I
> think that is the failing that we need to address, because it would take
> care of the extension of a verb.
>
> So if verbs were the categorization of differences on nouns, i.e. a set of
> learned operations over all transformations of objects, how best to
> shoehorn that into the current architecture? and if we solve that, then is
> our current intension the correct one, or should we just run a single noun
> through the learned algorithm over verbs to apply that transform, and that
> is its intension (applied to a point)?

Of Claude's first answer: "I think the direction is good also, although it
means collapsing the four forms of operators back to two, and building
higher-order verb classes." And:

> One thing that it points to is that just as LTM slots may reference a noun
> or verb in their epistemic memories, we can use indices those slots (or a
> content match) to determine which memories are collected into nouns; so
> perhaps we can store basic percepts on LTM to train the pi-not-sigma to
> recognize nouns even after the fact. Then training verbs would be
> analogous, because the differences are present there.

Then, deciding and setting the task:

> Episodic memory should be percepts: we can restimulate the concepts from
> them anyway.
>
> You can begin to write a spec 1.4, lets actually get things working as
> spec'd in 1.6 and quickly in 1.5 before we have more churn.
>
> Before you remove the four forms, we need to figure out how verbs map onto
> the existing field of concepts. So a point of knowing can be in an
> arbitrary location in conceptual space, but my thesis is that knowing in
> parallel is implemented by lighting up a set of existing concepts. For
> nouns, the parts of those nouns are lit to the same degree, and the whole
> to the fractional degree of participation. For verbs, we are now able to
> light up parts of verbs in the sense of events, and running has an
> extension on the set of observations of running.
>
> If we have that, then the parallel knowing becomes possible in the first
> "wisdom" pass, but we have the operators necessary to produce a
> representative point within that field that may not even be a part of that
> field.
>
> If you agree with this AND you think that we can implement it without
> causing lots of churn to the other operators, lets mark it as 1.65.
> Otherwise, please write it as 1.45, to be done after 1.6 and 1.5

## 2. Where it is placed, and why

**Item 1.45, after items 1.6 and 1.5.** ("spec 1.4" above is read as this
spec.)

Claude agrees with the thesis, and by the test Alec set the item passes:
the mapping of §5 changes no operator. It is placed at 1.45 all the same,
for a reason the test does not name:

- *A change can be observed only where an object has a state the model
  perceives.* With text alone, what stands before and after a verb is two
  sentences about one identity. Their percepts are the letters of two
  sentences, not two states of a thing. So before item 1.6 there is nothing
  to observe a change *of*: the extension of a verb would again be made of
  what was said, which is what `VerbLearning` reads today (§3). The felt
  world of item 1.6 (an object with an intensity at a place) is the first
  thing in the model that can be seen at one time and seen changed at the
  next.
- *It changes what LTM keeps and what the open field holds.* Both were
  rebuilt in item 1.7 (the looks, the chain in LTM, `ReadingMemory`, whose
  removal Codex is asked to explore, sigma-pi §23.8). Building on them
  before they have been used by 1.6 is the churn Alec means to avoid.
- *Replaying memory multiplies the passes*, which is item 1.5's matter.

Nothing of item 1.6 or 1.5 waits on this item. Item 1.6's scenes are still
(a word of place, not a word of change); a second scene for one object is
this item's addition to that world.

## 3. What exists (read in the source at `47b4192fe5`)

*The verb's intension.* A verb acts on a noun phrase coordinate by
coordinate in the chart, as a gain and a shift:
`Language.apply_verb` gives `unchart(e^w * chart(NP) + s)`, with `w` and `s`
read from the verb word's code by a learned readout (`_verb_spec`,
`_verb_shift`), in a basis per verb group (`MeaningKernels.VerbGroups`).
`unapply_verb` is its exact inverse given the verb. The modifier uses the
same transform (`_modifier_log_gain`, `_modifier_shift`).

*What teaches it.* `VerbLearning.observe` takes an identity's state before
the verb (`event.before`, an MTM event) and its next occurrence after, and
trains the readout on the error between the action's prediction and the
state after, in the chart (`MeaningAction.teacher`). Three limits:

- It runs only where a verb word is present (`event.verb_meaning`).
- The teacher skips every coordinate where the state before or the state
  after is zero (`defined = before.ne(0) & observed.ne(0)`). "The water
  froze. It became ice" with ice not there before teaches the gain nothing
  on ice.
- The difference is used once as a target and is kept nowhere. There is no
  concept of the change, no set of changes, no type over them.

The third is the missing extension Alec names.

*What LTM keeps of a moment.* An LTM0 row holds `episodicConcepts` (4)
references to concepts, each with one signed value
(`episodic_concepts`, `episodic_values`; `EpisodicMemory.select`: "the
strongest field readings, over all orders, in one fixed list"). It holds no
percept and no place. This is Alec's change of 2026-11-11 from percepts to
concepts.

*What the basic Pi reads.* Each occurrence in the bank is a row of
`.what`, `.where` and `.when` wires in two lanes
(`BasicPercepts`: `table [B, occurrences, width + where + when, 2]`). Each
row of the first Pi keeps its own entries on the `.where` and on the
`.when` wires (`where_entries`, `when_entries`, four wires each), and the
row's reading of an occurrence is the product of its match on both
(`field_factor(w, where) * field_factor(t, when)`). So the first Pi already
reads `.when` as it reads `.where`.

*How a row comes to be.* Nothing becomes a Pi row on its first reading; MTM
nominates rows on recurrence (`FamiliarTier.promote_candidates`, rules 20
to 22 of sigma-pi §19.10).

*How the field is lit.* `ConceptSpread.spread`: "up once by the fan, then
down once by the share, each joined as the product on complements".

*The clock.* One tick is one sentence; the present's code is the one
`.when` of everything read in that sentence.

## 4. Decided by Alec

1. **Episodic memory holds percepts.** "Episodic memory should be percepts:
   we can restimulate the concepts from them anyway." This replaces the
   change of 2026-11-11 (an entry of order-zero concepts).
2. **The four forms of the operators stay** until it is worked out how
   verbs map onto the existing field of concepts. Nothing in this item
   removes or merges an operator.
3. **The thesis.** Knowing in parallel is lighting a set of existing
   concepts. For a noun, its parts are lit to the same degree and its whole
   to the fractional degree of participation. For a verb, its parts are
   events: "running has an extension on the set of observations of
   running". The first pass, with attention wide open, then knows verbs in
   parallel as it knows nouns. The operators produce a representative point
   in that field, which need not be a part of the field.
4. **Order of work.** Items 1.6 and 1.5 first, working as specified.

## 5. The mapping proposed (Claude)

### 5.1 A noun in the field, as built

A noun's concept is a row whose members are its parts. Lit from below, the
whole receives from each part by the fan (its fractional participation);
lit from above, the parts receive by the share. This is §4's point 3 for
nouns, and it is what `ConceptSpread.spread` computes. Nothing is proposed
here.

### 5.2 An event is a whole over `.when`, by the same layers

A word is a whole whose parts lie at several places: the first Pi joins
them across `.where`. An event is a whole whose parts lie at two times: the
same Pi can join them across `.when`. Nothing is subtracted.

- *Gained:* the Pi of what is there after and the Not of it before.
- *Lost:* the Pi of what was there before and the Not of it after.
- *Kept:* the Pi of it before and after.
- *A verb class:* the Sigma over events that share what is gained and lost,
  whatever the thing that changed.

This is the Pi-Not-Sigma of every order (rule 29), read across time where a
noun's is read across place. It answers the case the gain cannot: ice that
was not there before is "ice after, and not ice before", which needs no
state before to scale.

Claude's first answer proposed keeping the change as a difference taken in
the chart. That is withdrawn in favour of this: the difference would have
been a new operation and a new kind of code, and this is neither.

### 5.3 Where the state before comes from

From episodic memory, as percepts (§4, 1; §6). At the opening of a
presentation, the percepts last stored for what is present are recalled
into the open field beside the present ones, each with its own `.when`. The
field then holds one thing at two times, and the one pass with attention
wide open reads both.

### 5.4 How a verb is lit

- *By observation:* an event's row is lit by its members, the states before
  and after, as a word's row is lit by its letters. The verb class is lit
  from below by the fan, to the degree the event is one of its kind.
- *By the word:* a verb word names its class as a noun names its object
  (item 1.6's tie of word to object). Lit from above, the class lights its
  events by the share: "running" lights the observations of running.

Both are `ConceptSpread.spread` as built. The first pass therefore knows a
verb in parallel in the same sense as a noun, which is the thesis.

### 5.5 The representative point

Unchanged. `verb` and `change` go on producing the phrase's point from the
verb word's code and the noun phrase's point, and that point may lie
outside the field. Whether the readout (`_verb_spec`, `_verb_shift`) can
then be read off the verb class's own row, and whether the four forms
reduce to two, is asked only after §5.2 to §5.4 are measured. It is not in
this item's build.

### 5.6 Higher-order verb classes

A verb class is one order above the events it collects, as a kind is above
its members; classes of classes (motion, change of state) are one order
above that. They are rows nominated on recurrence like any other. They
would take the place of the separately fitted bases of
`MeaningKernels.VerbGroups`, but only when §5.5's question is taken up.

## 6. Episodic memory as percepts (Claude's proposal for Alec's decision)

- *The entry.* A fixed number of the open field's basic percepts, each as
  its row in the percept store, its `.where` and one signed value. The
  `.when` is the row's own stamp. A percept is a row number, not a code, so
  the entry stays small. The number is a key of `data/model.xml` beside
  `episodicConcepts`, which it replaces.
- *Restimulation.* Recalling an entry presents its percepts as an open
  field and runs the one pass up. The concepts are what that pass lights
  now, not what it lit then: a noun learned since is recognized in an old
  memory. That is "even after the fact".
- *What changes for recall.* Today a cue is matched against an entry's
  concept references (`EpisodicMemory.matches`). With percepts in the
  entry, either the cue is carried down to percepts and matched there, or
  the candidates are restimulated and matched as concepts. The first is one
  matrix product over LTM0, which is what LTM0 was shaped for; the second
  is a pass per candidate. Claude proposes the first, and restimulation
  only of what is recalled.
- *What it costs.* Every reader of `episodic_concepts` and
  `episodic_values` changes (`EpisodicMemory.select`, `stage`, `write`,
  `matches`, `query`, `for_meaning`, `tree_entry`, `entry_set`,
  `held_rows`, `stm_key`, and the looks of a model with no grammar). No
  operator does.

## 7. Collecting memories into a noun, and into a verb, after the fact

- *By index.* The entries that reference one identity are that identity's
  occurrences. Offered again to MTM's count of recurrence, they nominate
  rows over a longer span than MTM holds. No new cost.
- *By content.* A match finds memories that were not tied at the time. As
  built the match decides by a bar, and the corpus count of sigma-pi §23.7
  shows the trouble with a bar: at 0.7 it joined 110 unequal collections,
  and at 1.0 it identified nothing beyond exact matches. So content is used
  only to nominate, by graded evidence summed over recurrences, and never
  to identify on one match.
- *Verbs.* Two entries of one identity at successive times, recalled
  together, are an event's two states. The LTM1 chain gives "next"
  (sigma-pi §19.146). What must be durable is the identity's state at each
  occurrence, as percepts; an entry of a sentence's strongest readings is
  not that, since a reading that drops out of a short list would be read as
  a loss.

## 8. What would show it (to be measured from input to output)

Each on the felt world of item 1.6, with a second scene for one object:

1. *A change is observed with no verb word.* The same object in two scenes
   with one intensity changed, presented without a sentence, several times:
   a row for the event is nominated.
2. *A verb names its events.* The verb word presented with those scenes:
   asked of a new object that changes the same way, the model answers with
   the verb.
3. *A verb lights its events.* The verb word presented alone: the field
   holds the event rows, read through an answer and not by inspection.
4. *What was not there.* An object that gains what it did not have: the
   event is learned, where the gain learns nothing.
5. *After the fact.* Scenes stored before the noun is taught; the noun is
   then taught; the stored scenes, recalled, are answered with the noun.
6. *No operator moved.* The tests of `verb`, `change`, `modifier`,
   `adverb` and the event cases of `compound` pass unchanged.

## 9. Open, for Alec

1. *Is an event a row of the first Pi over two `.when`, as §5.2 has it?*
   The alternative is its own layer for time. Claude proposes the first:
   one structure, and the wires are already there.
2. *Earlier and later, not the tick.* A row's `.when` entries hold a code
   of a time. An event must be the same event whenever it happens, so its
   two members need "before" and "after" in relation to each other, as a
   word's letters hold places in relation to the word. How the first Pi
   makes a word independent of where it stands is the pattern to follow;
   Claude has not traced it for `.when` and proposes nothing yet.
3. *How many percepts an entry holds*, and whether an identity's state is
   stored with each of its occurrences or only the sentence's strongest
   (§7, last point).
4. *"The four forms".* Claude reads them as an operator's use on a point
   (compose, generate) and its use on the field (the two field forms of
   2026-11-12). If Alec means another four, §4 point 2 and §5.5 are to be
   reworded.
5. *Whether `VerbLearning` stays as it is until then.* Claude proposes yes:
   it is the only teacher the verb's point has.
