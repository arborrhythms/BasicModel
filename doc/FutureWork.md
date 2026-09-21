# Future work

Cross-cutting items that are decided in direction but not yet specified or
built, with the decision that opened them and the document that owns the
detail once it exists. Section-local future-work lists elsewhere
([Language.md](Language.md#future-work-nouns-from-partspace-adjectives-from-wholespace),
[SymbolFirewall.md](SymbolFirewall.md#future-work-out-of-scope-for-this-pass))
stay where they are; this file indexes the items that span documents.
Items marked **urgent** block ordinary operation before long.

## 1. Operation profile: optimal versus human

**Decision (Alec, 2026-09-16).** Parameterise optimal versus more-human
operation so that both can be explored in one model. One `<operation>`
element in `model.xml` (`optimal` | `human`, default `optimal`) selects the
human-like behaviour for every item that
[Philosophy.md](Philosophy.md#discrepancies) marks **profile**:

| Item | `optimal` | `human` |
|---|---|---|
| Default belief | Cartesian: comprehension registers at trust `0`; provenance asserts | Spinozan: comprehension asserts at provenance trust; unbelieving is a later revision |
| Verbatim retention | derivations kept losslessly | derivations decay; fused points persist |
| Forgetting | capacity wall, recency only | forgetting model (§2) |

The profile is a switch over documented behaviours, never a hidden
heuristic; each behaviour has its own test under both settings. Params.md
gains the element when the first profile-dependent behaviour lands.

## 2. Forgetting and consolidation (urgent)

Specified in [the forgetting spec](specs/2026-09-16-forgetting.md)
(2026-09-16). Summary: `<ltmCapacity>` is the limit; a pass runs when the
store is nearly full, never mid-sentence and not more often than a minimum
interval, and deletes the lowest-valued unprotected rows down to a target
occupancy. Value combines trust (high trust preferred), utility (a row that
is easily deduced from other rows is not worth keeping: low expectation
surprise for ideas, one-step derivability for relations) and contribution
to luminosity (dissonant rows are forgotten; full coverage gain is optional
because it is expensive). Relations over deleted rows cascade; references
are remapped on compaction. The human profile adds an age term. The earlier
direction here, that isolated rows should be forgotten, is withdrawn: an
isolated row can score high on all three terms.

## 3. Lossy reconstruction trace; inversion as learning

**Decision (Alec, 2026-09-16).** Parts of the reconstruction trace will be
dropped over time, so inversion is a learned approximation from a lossy
trace, not an exact replay. With a complete trace reconstruction teaches
nothing; the loss is what gives the reconstruction objective its role in
learning.

To specify: which parts of a derivation are dropped first (leaf identities
before structure, or the reverse), on what schedule (age, use, profile),
what the reverse chain must recover from the fused point alone when the
trace is gone, and how the reconstruction loss is weighted between a full
trace and none. This is the verbatim/gist asymmetry (Sachs 1967; Reyna &
Brainerd 1995) made dynamic, and it shares its schedule with §2.

**Scheduled (Alec, 2026-09-20).** Dropping detail is the precursor stage of
forgetting and is gradual: see
[forgetting spec §4a](specs/2026-09-16-forgetting.md#4a-detail-before-rows-alec-2026-09-20).
What the reverse chain must recover from the fused point alone is
*generativity*
([accessible-mind spec §2.0](specs/2026-09-20-accessible-mind-subsystems.md)),
measured by that spec's test 12. What remains here is the weighting of the
reconstruction loss between a full trace and none.

## 4. Context across documents

The hard reset at a document boundary clears transient context. People
carry context across texts and conversations. To specify: what survives a
document boundary under the `human` profile (the expectation observation
view, priming heat, the What interaction slots), how a new document's
first sentences are scored against a carried context without penalising a
genuine topic change, and whether the boundary becomes a learned soft
signal rather than a cursor fact. The current per-row document key and
clause-scoped reset in the integrated plan §8.3 and §11 are the base.

## 5. N-ary META and interpretation-time discrimination

**Decision (Alec, 2026-09-16).** META concepts generalise over more than
two concepts: several words and several objects (synonymy and polysemy),
and the discrimination among them happens at interpretation time, from
context, not at binding time. The two-truths spec §3.4 records the
decision; the one-row-per-word law of the binding table is replaced by the
n-ary META. To specify: the META row's member set and its allocation,
what `deref(word)` returns when a word belongs to a META with several
objects (the member set, ranked by context), where in the seal the choice
is made (the chooser that already selects readings), and how a wrong
choice is credited by expectation and reconstruction.

## 6. English arithmetic Q/A curriculum

**Proposal (Alec, 2026-09-20).** Generate English questions and answers from
bounded mathematics to supply output-training data and reproducible held-out
evaluation, for example “What is two plus two?” → “Four.” This belongs with
generation and supervised output (NEXT item 3 in [todo](../todo.md)); later
reasoning-utility comparisons remain item 4. This records a curriculum direction,
not an implemented English dataset or a new learning result.

Start with a small vocabulary of number NPs and a learned successor VP:
`successor(two) = three`. Teach the finite counting facts, then compose
addition from succession and multiplication from addition. Common sums and
products may be memorized through ordinary learned associations and memory;
evaluate that separately from deriving unfamiliar combinations. These are
grammatical meanings selected by the existing MLPs. Arithmetic on concept
coordinates or symbol addresses does not implement numerical addition.

Beyond the initial vocabulary, build number NPs compositionally using decimal
place value and carry, with addition and multiplication as compound VPs.
Teach English number names alongside this structure, including irregular
forms such as “eleven” and “twelve”; English spelling is not itself the
place-value algorithm. Do not allocate an independently memorized atomic NP
for every larger result. Variable binding and algebraic identities need their
own later lessons; decimal notation alone does not establish those abilities.

The existing [math generator](../bin/exact.py) can produce exact labels on the
data/evaluation side. Its current [loader](../bin/data.py) supplies numeric
one-hot answer labels, so English answer text and the normal supplied-answer
training path still need to be wired. Present the question through ordinary
compose, let thought establish the answer, and realize it through generate
and the normal vocabulary. During end-to-end Q/A, the desired answer and
oracle calculation are scoring targets, never an answer seed or a runtime
calculator. A lesson that supplies an already encoded, on-manifold idea to
train realization is useful too, but is identified separately from a test of
answering the question. Follow the concluded-idea gradient boundary and shared
operator contract in [the integrated plan §8.4](plans/2026-09-15-next-sentence-as-the-production-objective.md).

Generate separate training and evaluation sets with fixed seeds and partitions
over mathematical problems before applying wording templates. Keep equivalent
commuted problems together when evaluating unseen arithmetic, and keep
memorized-fact and paraphrase-only results identifiable. Reserve new operand
combinations, complete question wordings, larger composed numbers, carry
patterns and derivation depths for distinct evaluations. All base successor
facts may be taught; repeating them in evaluation measures acquisition, not
generalization. Score answer meaning as well as English realization through
the full vocabulary. Retain renamed-vocabulary controls for arithmetic meaning,
and the matched-compute comparisons before claiming useful learned reasoning.

Existing [successor-VP probes](../test/test_verb_successor.py) establish local
operator behavior under their own setup; they do not establish this end-to-end
English Q/A curriculum. The older [mathematical-thinking specification](specs/2026-09-09-mathematical-thinking.md)
retains the original motivation, but its retired controller and interpreter
are not part of this proposal.

## 7. Episodic memory: a few active indices beside each row

**Decision (Alec, 2026-09-20).** LTM holds serial form only: a row is one
idea per role, and several things at once can be stored only by chaining
NP / VP into a compound sentence, which approximates episodic memory and
keeps only what was *said*
([accessible-mind spec §2.7.1](specs/2026-09-20-accessible-mind-subsystems.md)).
The extension that makes episodic memory proper is small: **room, next to
each three-slot row, for a few indices of the most highly active percepts or
concepts** at the moment the row was written. Those are the things that were
present but never composed into the sentence.

What is stored is indices with their activations, not vectors — a sparse
field over the existing tables, so it is re-read through the current
dictionary and costs a few numbers per row beside three idea vectors. With
the row's where / when / source binding it is an episodic trace: a stored
state, bound to its occasion.

To specify: how many (a handful); which tables (perceptual rows from the two
towers, conceptual rows of order 0 and above, or both); the selection rule
(top activation at the seal, excluding the codes the sentence itself
contributed, since the derivation already holds those); whether retrieval
**reinstates** them into parallel knowing, bounded and decayed, so that
remembering differs from knowing; their use as retrieval cues and as the
evidence for discriminating a word's senses at interpretation time (§5);
their remapping when a table is compacted; and their place in forgetting —
they are detail, so they go first
([forgetting spec §4a](specs/2026-09-16-forgetting.md#4a-detail-before-rows-alec-2026-09-20)).

The psychological grounding is in the accessible-mind spec §7: the episodic
trace as an index to the pattern of activity rather than its content (Teyler
& DiScenna 1986), reinstated at retrieval (Danker & Anderson 2010); the fast
system as the binder of what co-occurred (O'Reilly & Rudy 2001); and
multiple-trace memory, where stored instances keep the senses a single
reading loses (Hintzman 1986; Jamieson et al. 2018).

## 8. Recall as estimate plus residual

**Direction (Alec, 2026-09-20).** Expectation is a negative image: what is
conceived of a sentence is the composed idea less what was predicted
([accessible-mind spec §2.6](specs/2026-09-20-accessible-mind-subsystems.md#26-expectation)).
The store already keeps the two terms as a linked `estimate` / `observation`
pair with the residual derived ([ExpectationRetention](ExpectationRetention.md)),
and the observation row keeps the full idea. That is the `optimal` layout and
item 2 builds on it unchanged.

The `human` profile goes one step further, as a stage of
[forgetting §4a](specs/2026-09-16-forgetting.md#4a-detail-before-rows-alec-2026-09-20):
the expected part of a row is detail, and detail goes first. A row coarsened
to its **tag** keeps only the residual and the pointer to its context, and is
recalled by adding the *current* world-model's estimate back. Recall then
drifts toward the schema as the world-model changes, which is the human
pattern: typical actions falsely recognised, atypical ones kept (Graesser,
Gordon & Sawyer 1979; Bartlett 1932; Brewer & Treyens 1981).

To specify: when a row is coarsened to its tag (after its estimate row is
gone, or before); what the pointer must retain so the estimate can be formed
again (the context rows, which may themselves have been forgotten); how a
tag row is ranked and unfolded, since a residual is not an idea on the
manifold and generativity is measured on ideas; and how a tag row's
provenance says that part of what it recalls is reconstruction.

## 9. Exclusion at the readout

**Direction (Alec, 2026-09-20).** "The object appears to the mind as the
negation of the non-object", and that negation is non-affirming: it removes
the non-object and puts nothing in its place
([accessible-mind spec §2.6.7](specs/2026-09-20-accessible-mind-subsystems.md#267-attention-the-exclusion-of-the-non-object)).
So attention needs no prediction in order to focus, and the architecture
already has it in that form: the reading scope, and the intent channel of the
priority surface at the concept pyramid's per-order top-K (floor of zero,
never a veto), behind `<relevance>`.

What is not built is the interaction at the level of rows. Item 2 applies
the estimate "to everything but the object of observation" at the level of
roles, where it cannot touch composition. At the level of rows it would make
an expected object quicker to recognise *within* a sentence: expected
non-object rows of order ≥ 1 lose priority, so the object competes against
less.

To specify: how the estimate, an idea, becomes a priority over rows
(`quantize`, restricted to rows of order ≥ 1 by the ramsification table);
what sets the object of observation at this level (conceptual activation as
the origin of reading attention,
[Architecture](Architecture.md#symbolic-weights-reconstruction-parse-time-attention-2026-06-30));
and the price, which is the reason this waits — selection at the readout can
change what is composed, so the purity invariant of spec §2.6.3 no longer
holds by construction. The residual must still be learned from what was
there, not from what was selected (statistical learning needs attention:
Turk-Browne, Jungé & Scholl 2005), which may mean composing twice. Decide
after item 2's measurements (spec test 32) show whether role-level sparing is
enough.
