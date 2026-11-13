# Training

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record.
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Expectation is a **gain field** $c = m \odot a \odot o$ (attention's gain $m$, expectation's gain $a$, the arriving signal $o$) with the residual in two lanes, never a subtraction or a sign-reversed image; per level with a local error (no inversion across levels — the invertible path is item 3's); one activation predictor over the symbol bank (the slot-structured `SentenceExpectation`, `BracketExpectation`, the byte level and the ARMA heads are retired); keep and credit both read `R + A`; defaults gains 0, prime on, `φ = .5`.
> - Amended 2026-10-29/30 (Alec): at the percept level the residual is signed and the forward path carries `expected + residual`, so "never a subtraction" is superseded there (built in item 2.7's final round); at the concept level the expectation is `ActivationPredictor`'s predicted next state, which at item 2.5 round B is also the spread that primes attention, the chooser and retrieval; `intuit` is a separate top-down layer with its own weights, trained only by output. Round B and the one-backward path are built (accepted as implemented, Alec, 2026-11-01). [Current gradient boundaries and normalization review](GradientFlow.md#current-gradient-boundaries-and-loss-normalization) supersede the historical ownership descriptions below where the selected chooser setting differs.
> - **Accepted as implemented (Alec, 2026-11-01).** The one-backward path and round B's work are accepted; the three lesson-13 arms are measurements, not gates; the defaults stay `separateChooser=true` and `reconstructionJitter=0` until Alec changes them; a standing test or certificate that fails on an unresolved deviation is marked a strict xfail naming it, and the stage 2 freeze locks down what works. "Remains unaccepted" in the bullet above is superseded.
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.
> - `.where` is inside the concept's reserved band (no column); **symbols have `.where` and `.when`**, concepts keep the extent and never the clock; `.when` is model time, sinusoidally encoded; a stream is the batch row that wrote a row, and a document is the cover row that starts it in that stream (reused at every later presentation; a row belongs to the document of the latest cover before it), so "sentence *k* of document *d*", source address hashes, timestamps and situation wholes are superseded (2026-10-24).


The current cross-architecture reference is
[Gradient flow across the architecture](GradientFlow.md): what each objective
trains, where gradients stop, and how optimizer ownership is audited.

**The developmental corpus (2026-10-27).** The pinned recipe is
`data/corpus-sources.json`. Assemble it in a fresh directory:

```sh
.venv/bin/python bin/prepare_corpus.py --recipe data/corpus-sources.json --output data/corpus/assembly-01
```

The recipe pins its inputs: Wordbank's public response slices and the
AO-CHILDES transcript tables by immutable git object hashes, the FineWeb shard
by SHA-256. `bin/prepare_corpus.py` assembles three groups in order. Wordbank
contributes each word at the earliest month at which at least half of the
responses report production (`production_threshold` 0.5; a missing response
counts as negative); questionnaire annotations are removed, slash variants are
kept, and no lexical category reaches the learner. AO-CHILDES contributes
adult utterances to children under six years (`max_age_days` 2190) in annual
bands (`band_days` 365), ordered by age, transcript and utterance, with their
original wording and with end punctuation taken from each utterance's recorded
type. Written FineWeb text follows. Stage and age metadata schedule input;
they supply no grammatical classes, object definitions or answers. Splits are
document-disjoint, 8/1/1 by document index, and each keeps its chronology.
Assembly is bounded in memory by SQLite. The source tables, the SQLite
assembly and the generated shards are ignored artifacts; the manifest records
every source fingerprint, the filtering counts, the stage sizes and the hashes
of the recipe and the assembler.

`data/BasicModel_corpus.xml` is the measured million-training-sentence sizing
candidate. It is **not yet a fitting production configuration**: its full
constructor exceeded the 24 GiB guard on this 36 GiB host. The loader and
fixed word-identity inventory can be audited independently with
`bin/corpus_audit.py` and `bin/corpus_capacity.py`.
Training and evaluation finish each corpus stage across all streams before
advancing. Packs stop at document boundaries. A mid-presentation checkpoint
saves consumed stream positions, including with prefetch, plus the priming,
identity, live STM and expectation continuation state. Replaying a tick count under a
later learned tiling is not a safe resume method. Native update parity is a
separate measurement; cursor equality alone does not establish it.

**Item 0.5 small-stage rehearsal (2026-11-13).** The frozen sequence is
`data/training-corpus.json`; `data/eval/stage-models.json` declares the unchanged
small configuration, bounded document/epoch budget and held-out readings. Run
`.venv/bin/python bin/stage_models.py --output <new-local-directory>` (also
available through `bin/corpus_lessons.py --stage-protocol`). The producer saves
M0 and each completed stage on the same live model. A standalone
`--only-lesson` run is a diagnostic and cannot register a stage. All lesson
thresholds remain undeclared; completed exposure alone cannot register one.

`bin/eval_operators.py::operator_table` restores each saved checkpoint in an
isolated process, verifies tensor bytes, and reads the same full validation
manifest. It reports selected-rule use, each existing layer's own parameter
change and held-out game production. Unavailable readings remain null;
parameter-only readings after failed validation are explicitly incomplete.
The 31 game variants are frozen under `data/games/`, with coverage in
`inventory.json`. Their intended rules are diagnostic labels, never required
learner choices. Game training remains stopped at the two whole-sentence
signal nulls recorded in the [item 0.5 return](plans/2026-11-09-item0-5-test-development.md#19-item-05-return-codex-2026-11-12).

The H9/H10 task recipe, files and protocol live in `data/eval/studies/`.
`bin/study_corpus.py` reproduces their recorded hashes. Small diagnostics use
`bin/eval_studies.py smoke --arm <name> --output <new-directory>` or
`--gate <name> --side treatment|control`. Missing runtime controls are reported
before training; there is no substitute learner. Order/determiner controls
share ordinary held-out text, and vocabulary controls transform the text and
answer together. Recorded query-work charges include discarded trials but are
not total compute. These small diagnostics do not satisfy the million-sentence
qualification, three-seed comparisons, or undeclared acceptance bars.

**Concept dictionary ownership (October 2).** Codes are parameters
trained only by reconstruction, without a norm constraint. The canonical
configurations set `conceptualContextLearningRate=0`; contextual rotation and
the concept VQ EMA refresh no longer write the dictionary. Admission does
not replace a reserved code with a normalized feature/part composition.
Expectation and supplied answers train their own readers with detached
sources. The leaf is still code times signed activation. Distributional pressure on the
codes is deferred until reconstruction and XOR hold; a unit-sphere constraint
on the codes is not planned to return.

> For the current ownership of the training lifecycle and the proposed split
> between launch resolution, corpus cursors, host orchestration, compiled tensor
> steps, objectives, and checkpoints, see
> [Runtime Architecture and Componentization](Componentization.md).

> **2026-05-29 deltas:**
>
> - **Embedding unit-cell wrap.** The training loop calls `normalize()`
>   on the `Embedding` basis (`perceptualSpace.subspace.what`) right
>   after `optimizer.step()` in `bin/Models.py::runBatch`. That
>   `normalize()` is a periodic unit-cell **wrap** into $[-1, 1)$ via
>   `embed._wrap_unit_ball` (torus geometry), *not* an L2 ball
>   projection (`Layers.Lexicon.normalize` does ball-project, but is
>   not on this path). Keeps embedding vectors from drifting out of
>   the unit cell under `JOINT` / `BACKPROP` modes.
> - **Reconstruction is unconditional from concepts.** The `<reconstruct>`
>   element (and `reconstructEnum`) was RETIRED (2026-06-09); there is no
>   knob to set or override. Reconstruction is seeded from the understanding:
>   the open attention pass's field and the sentence's end slots
>   ([Within-sentence reconstruction objective](#within-sentence-reconstruction-objective)).
> - **Supervised output loss restored.** `bin/Models.py::runBatch`
>   computes MSE on labeled datasets. Unlabeled corpora such as FineWeb
>   train only through reconstruction/prediction objectives.

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

Training keeps the LLM-like objectives of prediction and reconstruction, but
does not make next-token likelihood the whole story. Early objectives shape the
embedding and codebook substrate; later objectives train the Formal Concept
Analysis-like concept order, the DisCoCat-like grammar composition path, and the
truth/reasoning machinery that tests composed meanings. This is why the staged
curriculum treats prediction as substrate-building and question answering as a
separate deliberate objective.

## Overview

*A superseded passage beginning "For the current production expectation rollout, use the" moved to [DevelopmentHistory](DevelopmentHistory.md) (2026-10-30).*

Two phases: **embedding pretraining** and **network training**. Embedding
pretraining builds word vectors from a large corpus. Network training uses
those vectors as input representation and learns to predict and reconstruct
sentences.

The two phases can overlap: `<trainEmbedding>` controls whether and how
embeddings continue to evolve during network training.

---

## The staged objective curriculum

> *Amended 2026-11-11.* The labels System 1 and System 2 in this section name
> two learning rules (occurrence and task error) and predate the definition of
> 2026-11-09, by which System 1 is everything the loop needs to close without
> thinking and System 2 is the three grammars; both are trained by gradient
> ([Architecture](Architecture.md#dual-process-system-1-knowing-and-system-2-2026-11-09),
> [GradientFlow](GradientFlow.md)). The text below is kept as the record.

Network training climbs five objectives, from substrate-building reconstruction
to deliberate question answering. The order is not arbitrary — it is the
architecture's **two learning rules** in their natural developmental order:

- **Distribution / occurrence** moves the concept dictionary: concepts form by
  being *seen*, with
  no gradient and no goal. This is *prediction as substrate* — unconscious,
  statistical, **System 1**. You cannot ask "why did the empire fall?" until
  *empire* and *fall* are codebook rows, placed by exposure through the
  sentence-local rotation rule above. EMA is used by other configured VQ stores.
- **Gradient on a task error** moves the attention readouts and the relation
  store: the model learns *where to look* and *what relates to what* by being
  *tested* — directed, credit-assigned, **System 2**.

So "bootstrap with prediction, then answer questions" is just the EMA
substrate-builder running first and the gradient deliberate-layer on top
(*prime subsymbolically from what you see, then pump symbolically*). Reconstruction (stages 1–2) is the **gate**: a question can only
be *answered* once an idea can be turned back into words, so the grammar-operation
inverses those stages need are the load-bearing prerequisite for the whole
curriculum.

| stage | objective | trains | dominant rule | status |
|---|---|---|---|---|
| 1 | reconstruct, **keep syntax**, fill the missing **words** | the lexical/leaf inverse + the codebook (word$\leftrightarrow$slot) | EMA (System 1) | recon loss + codebook decode exist; masking + leaf fill to build. **Needs no new grammar inverses** (the kept parse tree drives the existing reverse). |
| 2 | reconstruct with **no syntax / no words** (from the idea alone) | the full generative inverse (deliverable C) + abstraction | mixed | reverse path exists but is parse-tree-*dependent* $\to$ must be **decoupled** to drive from the primed symbolic space (attention) instead of `generate_rules`. |
| 3 | predict the **next sentence** | the inter-sentence (discourse AR) predictor | EMA $\to$ gradient | exists (`<prediction>interSentence`). |
| 4 | answer a question by **reasoning** (no search) | relations, modus ponens (`consequents()`), the catuṣkoṭi/trust | gradient (System 2) | the truth stores + `consequents` exist; the QA framing + consistency loss to build. |
| 5 | answer a question by **LTM search** | global attention (the typed `.where`) + the soft-read fed back + the explorer | gradient (System 2) | global attention (B) built/dark; the consumer (feed the read back, train by the answer) + book paging are deferred. |

Plain next-token/next-sentence prediction *under-trains this architecture's*
retrieval and relation machinery (the local window usually suffices, so the
separate global attention gets no reward); that is why prediction is the
**substrate** here, not the goal — stages 4–5 train the deliberate layer, and the
stochastic exploration finally earns its keep in stage 5 (search has only a
distal reward, so it must explore to break symmetry).


## The generated supervised curriculum, in order (Alec, 2026-10-09)

The objectives above are what is trained; this is the order in which the
model is taught, by **generated corpora** whose targets name parts of the
input — a word, a word's turn, a sentence's part, an answer — and never an
attention action, an operator or an order rule. What a stage teaches is a
learned prior a later corpus can revise; nothing here is built into the
architecture (word order in particular is not universal across languages,
so reading order is taught, not wired). Each stage is a gate: the enabled
model against its disabled control, one fresh unseeded run per mode, no
retries, the corpus generated from a small declared vocabulary. A stage is
read as a result only when its control passes it.

*Amended 2026-11-07 (Alec).* What the curriculum orders is the operations
learned. The order in force is the ten lessons of
[Concept hierarchy learning and curriculum order](#concept-hierarchy-learning-and-curriculum-order-alec-2026-11-08):
clauses and questions come before determiners and modifiers, so stage 4b's NP
lessons follow 4c and 4d. The stages below keep their corpora and gates.

| # | stage | generated corpus | what the lesson names | gate | trains |
|---|---|---|---|---|---|
| 1 | **Identity** | one percept whole (a letter run) presented, re-presented, and re-presented after an intervening item | *same* — the second occurrence reconstructs from the first's concept and row | the same percept gets the same concept row (the identity audit: no duplicate rows); the item is recognised again after the intervening one | identity by construction (fixed codes for a form's boundary-marked letter pairs and its length); the row as the item's persistence. Object permanence *across sentences* is stage 5, not here |
| 2 | **Identifying words** | one-word fields, known and novel words | the word — its reconstruction, and *which* word (the symbol) | readback 100 % for known words; a novel word is minted once and recognised on its second presentation | the word whole → concept read; the support mask of one candidate |
| 3 | **Reading words in order** | fields of two, then more, words; the lesson gives the words in turn | the part whose turn it is — "the first word", then "the second": a cross-entropy on the candidate scorer toward that candidate, reads teacher-forced during the lesson; the order is the lesson's, from the data | every read supports one word in the lesson's order on permuted fields after training; lesson CE falls to near zero within 64 presentations; control = source order; list reconstruction is reported alongside, not used as this gate | the attention scorer's choice of the next read; one read per placement; the eight-space as STM |
| 4a | **Simple sentences: identities** | `x is three.` `a cat is an animal.` — two words joined by `is`, generated over variables/values and kinds | the sentence; the supplied answer to `what is x ?` | reconstruction 100 %; the question answered 100 % on held-out pairs | the one relation as a verb (`is`); the `what` question bound from supplied answers only |
| 4b | **Simple sentences: NP** | `the red ball`, `a big cat` — determiner, adjective, noun | the phrase; its asked parts ("the middle word", "the noun"); the supplied answer to `what is red ?` | reconstruction of the compound 100 %; asked parts; the property question | composition within a role and its inverses (adjective ∧ noun, sub-typing); the determiner as the bind-or-mint cue; word → operator learned, never anchored |
| 4c | **Simple sentences: NP + VP** | `the cat sleeps.` `a dog runs.` — subject and intransitive verb | the sentence; the supplied answer to `what sleeps ?` | reconstruction and the question 100 % | the verb as a pattern of change, categorically apart from nouns (a noun names an individual; a verb is one fixed change that maps the individual before to the individual after); the clause's idea, NP and VP fused to one point |
| 4d | **Simple sentences: relations** | `the cat is on the mat.` `x is bigger than y.` `the dog chases the cat.` — two NPs and a relation | the sentence; supplied answers on either role — `what is on the mat ?`, `what chases the cat ?`, `what does the dog chase ?` | reconstruction; both role questions 100 %; `the dog chases the cat` and `the cat chases the dog` answered differently | the two noun roles distinguished by order; relation operators |
| 4e | **Constituents** (added 2026-10-16) | the 4b–4d sentences with their brackets: `[the black cat] [runs] .`, `[the dog] [chases [the cat]] .` | the constituents — a cross-entropy on the compose chooser toward the bracketed reading, as stage 3 does for order; the NP operators (mint/bind) applied at each bracketed NP | free reading reproduces the brackets on held-out sentences of the same kinds; every singular NP writes an occurrence | constituency, taught by lesson, never wired; the NP event that mints an anonymous individual |
| 5 | **Tying sentences together: object permanence by reference** | `the cat sleeps. it is black.` `a dog runs. the dog is brown.` — a pronoun or a definite re-mention referring to an earlier occurrence, with and without an intervening sentence | the referent (which earlier occurrence); the supplied answer to a question about the first sentence asked after the second (`what is black ?` → `the cat`) | the reference resolves to the right occurrence 100 %; the repeated noun reuses its row, `a dog` mints a new one; the question across sentences answered | identity is its occurrences tied by references in slots; the row persists when not being read — object permanence proper; "identity is an expectation" |
| 6 | **Graded composition** | the standing fixtures' vocabulary arranged so each item needs one more round than the last — clauses and embedding after the simple kinds — read in order and shuffled | reconstruction; the surprise at every level | the ordered set beats the shuffled on derivation optimality (the share of items that end on the optimal derivation) | expectation at every level; one departure per sentence |
| 7 | **Context as content** | several documents in parallel sharing vocabulary; a document split by another; a fact stated in one needed in another; a word whose referent differs between documents | the answer, which only the right context gives | batch differentiation; return after interruption (stage 5's permanence at the document level); shared truth; priming as a prior; expectation from the right history | the situation code in content; retrieval as a candidate entertained into a slot |
| 8 | **Thinking** | one-step problems first (`x is three. what is x plus one ?`), then chains, worked steps scored as intermediate answers | the answer and the worked steps | the deferred math-chain measurement under its four corrections (one-step problems first; worked steps scored as intermediate answers; an episode that can bind a found candidate; the budget counted in steps, with each query's record charge bounded separately); decomposition demonstrated at the closing | the thought loop and its credit |
| 9 | **The age-appropriate corpus** | Wordbank → `interpret` → AO-CHILDES by age band ([FutureWork, gradual training](FutureWork.md)) | text; supplied answers where the corpus has them | the stall diagnostic flat per level | everything above, at scale; item 3's corpus |
| 10 | **The full train** | the target corpus (item 0) | — | the standing gates unchanged or better | — |

**Word classes in order (Alec, 2026-10-21; built in item 3).** Parts of
speech are learned "in virtue of learning certain words first (proper
nouns, then nouns (mass and non-count), then count nouns (for languages
that have them)". Proper nouns come from demonstratives, which point at
events: "this is Felix, that is Felix" makes Felix the class of two
events, an individual at order 1. Count nouns come from definitions over
individuals: "Felix is a cat, Ciao is a cat" makes cat a kind at order 2.
This refines stages 4a and 4b; the mechanism, a raise by equivalence
class, is [Orders §10](Orders.md#10-orders-are-learned-definitions-raise-three-compositions-lower-alec-2026-10-21-decided-built-in-item-3).
Until it is learned, unit tests force parts of speech (Alec: "they may
take a while to learn, and we need to test without that").

**Item 6 part 2 amendment (Alec, 2026-10-10).** The
identity-from-data lesson design
governs the stage 1 and stage 5 identity lessons: vary object, property and
verb factors independently; grow support from one to three; counterbalance
pronoun candidates' recency, role and order; include imperfect determiner cues,
held-out cue/content conflicts, later distinguishing properties for same-kind
individuals and document-dependent referents. In stage 5, `a` and `the` are
evidence for identity choices, not unconditional mint/bind instructions.
Measure the declared candidate-count/recency/role strata and both controls
(a follow-up that neither candidate predicts leaves binding at chance; training
on a recency-biased set and testing on the reversed set exposes a positional
shortcut) before changing the mechanisms. Pronoun lessons place each
candidate in each position equally often ("the cat saw the dog. it barked."
beside "... it purred."), so that only content the follow-up predicts decides.
Rule retirement requires a passing
lesson without that rule; a rule whose lesson fails without it stays, with the
failure reported, and a retired rule is removed, not kept beside its learned
replacement.

The [first text corpus](../data/identity_from_data/README.md)
implements this lesson inventory before any identity mechanism changes. The
grader keeps missing and colliding candidates in the denominator, separates
resolved coverage from conditional accuracy, and counts semantic rows rather
than word rows. Its known-policy controls check the grader; native errors and
unreached evaluations do not constitute a learned binding or chance result.
None of the existing rules is retired by this
preparation.

Why this order. Stages 1–3 teach what a word is and how to take one at a time;
4a–4d teach sentences by the operations they need, one more at each step —
a relation between two atoms, composition within a role, a verb against a
noun, two roles told apart by order; only then, in 5, are sentences tied to
one another, because a reference needs sentences to refer to and a row to
persist. Stage 7's "return after interruption" is stage 5's permanence raised
to the document. Stages 1–3 absorb item 6.1's attention stages; 4b's asked parts are
its fourth. Stage 3 is the lesson that failed when reading order was left to
sparse credit alone: the scorer learned content, not position.
Free decoding of compounds — unfolding a composed child against a bank that
holds only words — is **not** a stage gate: reconstruction in every gate is the
inversion along the actual derivation, excluding undefined coordinates from
the inside loss and reporting their unrecovered words/coordinates separately.
Stage 3 gates on reads and lesson CE instead of list reconstruction
(Alec, 2026-10-10). Free compound decoding is item 6's
work, whose inverse will clean up against the primed words *and* STM's composed
wholes.

**One departure, read at the end.** The order also follows from how choices
are credited. A path departs once from the greedy derivation, completes the
rest greedily, and its cost $R + A$ is read at the end of the sentence; a chain
of random choices would be rolling for all sixes. The comparison measures
$Q(s, a)$, the cost of choice $a$ in state $s$ followed by the greedy
completion, which is the right credit, $Q(s, a) = V^*(s')$, exactly when that
completion is already optimal. The last choice is always credited exactly,
since its completion is empty, and each earlier choice is credited exactly once
every choice after it is optimal: backward induction, Viterbi's principle of
optimality. One draw per derivation therefore suffices, on four conditions.
The completion must be near-optimal, which a graded corpus supplies one step at
a time: each item needs one more operation than those before it, added at the
bottom of the derivation (phrases before clauses before sentences; for
thinking, a question answered by a premise, then one successor step, then
two). Credit is first-order: a better derivation that differs at two choices
at once shows no gain to a single departure, so each stage adds one new
decision. Coverage is about one over the derivation's depth, since the
departure lands uniformly over the eligible choices. And the signal is the
outcome, $R + A$, never predictability, with the uniform floor $\varphi$
keeping every alternative reachable; the temperature and the floor set only
how far the departure's alternative draw looks, while the exactness comes from
comparing two derivations of one input that differ in one choice. Without the
graded order, completions off the greedy path are never learned and early
departures are credited against untrained suffixes. The stall diagnostic reads
this per level: an explore-win rate at zero while the level's surprise stays
high says the data step is too steep there. A shuffled order is the control.


### The generated math suite: what it exercises and what it cannot (2026-10-20)

Alec: "mathematical equations written as English sentences offer one of
the best automatically-generated testbeds for helping a model learn stage
by stage." Every sentence has a checkable meaning, the difficulty is a
dial, and the corpus is free — which is what the Viterbi order needs. It
runs under `RUN_SLOW` ([Testing](Testing.md#fast-regression-and-slow-experiments))
and doubles as the compiler test at scale. Its coverage:

| exercised, often better than anywhere else | consumed by |
|---|---|
| identity by name: variables as proper nouns, re-mention, equality as singleton membership, definitions, substitution of equals | 3.5 `bind` step 1 |
| composition and roles: `NP VP`, `VP → VP NP` twice, nesting, "x minus one" ≠ "one minus x" | the grammar operators ([Language](Language.md)); 3.5 |
| the verb as a group element with exact ground truth: plus/minus inverses, plus commutes, times/divided-by a second group; translation along the successor chain of opaque numerals is diagonal in the chain's Fourier basis, so the additive and multiplicative bases are an exact target for joint diagonalization | 3.5 verb groups |
| relations and chaining: conditionals, premises in either order, one- and two-step decomposition, sub-questions | the thought loop ([Reasoning](Reasoning.md)); item 0's deferred measurement |
| questions, answers, binding: `what` opened and bound, the answer line as supervision, yes/no on the truth lanes | `bind`, `R + A` |
| modality and counterfactuals: a world differing in one restriction with a computable consequent | the second difference ([Philosophy](Philosophy.md#two-differences-the-verb-across-time-the-modal-across-possibility-alec-2026-10-19)) |
| ellipsis and coordination: "x is three and y four"; "x and y are three"; "x or y" in the both/neither corners | the elided site; the lanes |
| generics against particulars; quantifier scope | ideas and relations; LTM2 |
| attribution with conflicting sources and known truth | LTM2 attribution; trust in the lanes |
| surface variety under one meaning: "equals", "is", "sums to", the marker as word or affix | one operation over any surface carrier |
| perceptual identity of word forms: letters and letter-wholes are percepts (PartSpace, WholeSpace), so nested admission, marker minting and the meet over letter-wholes are exercised; the referent's quantity has no percept, and under textual grounding it needs none — object-concepts are minted and learned through the definitions and kind statements the corpus supplies | the identity staircase steps 1–2 |

| not reached | why |
|---|---|
| space, motion, prepositions, and most verb classes | arithmetic yields two or three of the two dozen verb groups |
| aspect: activities, progressive, perfect, durations | the corpus has states and instantaneous changes only |
| lexical ambiguity, garden paths, pragmatics, metaphor | math English is unambiguous by construction |
| vocabulary growth and morphology beyond ordinals and plurals | a closed vocabulary |
| desire and the social channel | nothing wants anything |

Two cautions: the model can learn the sublanguage's template distribution
rather than English, so the graded math corpus is interleaved with the
age-appropriate natural corpus of item 3 rather than run alone; and since
numerals are opaque words it tests the mechanisms, never arithmetic
competence, which is what it is for.

**Mathematics enters only as language** (Alec, 2026-09-09). The runtime has
no arithmetic primitive, lookup or substitution operation: numerals are
opaque number words with arbitrary concept addresses, `plus` and `minus` are
ordinary transitive verbs, and addition is iterated succession. The counting
grade states the successor, `n plus one`, for every numeral; a larger sum is
reached by chaining it in thought, each intermediate result an LTM row.
`bin/exact.py` parses, generates and scores problems for the corpus and for
evaluation only. Its oracle answers and traces are scoring targets and never
reach the learner as inputs, answer seeds or a calculator;
`test_arithmetic_isolation.py` checks that a supervised update cannot reach
the exact arithmetic or a fallback code.

### Examples by stage, and the construction each introduces

Every example uses only the constructions of its own stage and the stages
before it. The ledger at the end says where each construction first appears,
so that containment can be checked at a glance. Vocabulary is small and
declared: variables `x y z`, numerals as opaque words `three seven nine`,
colours `red blue green gold`, nouns `cat dog ball mat`, adjectives `big
black`, verbs `sleeps runs eats chases`, and the function words each stage
admits.

**1. Identity** — single percept wholes; no words as concepts yet, no
sequence, no sentence.

    cat          cat          dog          cat
    red          red          blue         red

**2. Identifying words** — one word per field; a novel word (`zib`) minted
once, recognised the second time.

    red          blue         cat          zib          zib

**3. Reading words in order** — lists of content words, no function words,
no verb, no punctuation; the lesson names "the first word", "the second".

    red blue
    blue red
    cat dog red
    gold green blue red

**4a. Identities** — two atoms and `is`; the period; the `what` question.
No determiners, no adjectives, no action verbs.

    x is three .          what is x ?      → three
    y is seven .          what is y ?      → seven
    z is nine .           what is z ?      → nine

**4b. NP** — determiners and attributive adjectives; the phrase as a unit;
asked parts and the property question. No verb other than the question's
`is`.

    the ball              the red ball          a big cat
    the black dog         a red ball

    the red ball          the middle word ?     → red
    the red ball          what is red ?         → the ball

**4c. NP + VP** — an intransitive verb after an NP of 4b; the subject role.
One noun role only.

    the cat sleeps .      what sleeps ?         → the cat
    a dog runs .          what runs ?           → a dog
    the black cat eats .  what eats ?           → the black cat

**4d. Relations** — a second noun role: transitive verbs, prepositions,
comparatives, and kind identity between two NPs. Order tells the roles
apart.

    the dog chases the cat .     what chases the cat ?        → the dog
                                 what does the dog chase ?    → the cat
    the cat chases the dog .     what chases the dog ?        → the cat
    the cat is on the mat .      what is on the mat ?         → the cat
    x is bigger than y .         what is bigger than y ?      → x
    a cat is an animal .         what is an animal ?          → a cat

**4e. Constituents** — the 4b–4d sentences again, with their brackets
supplied as the lesson (the generator knows them). The lesson names the
constituents the way stage 3 names the order: a cross-entropy on the
compose chooser toward the bracketed reading. At each bracketed singular
NP the reading applies the NP operator (mint, or bind for a re-mention)
and captures the NP's own meaning before the sentence fold. Its occurrence
row is written at the kept close. The lesson's CE teaches the waits
as well as the folds; it never replaces the student's kept program.
Adjective descriptions use `intersection`; this corpus contains no
noun–noun `compound` and requires no case lesson.
The gate, declared before the fixed-budget run (Alec, 2026-10-17), is free
reading on held-out sentences of the trained kinds: **≥90% exact brackets**,
**one occurrence for ≥95% of singular NPs with zero duplicate writes**, and
**one order-1 occurrence at every lowered NP**. Mint and bind construct an
order-1 member from either an order-0 type or a raised description, and kind
constructs order 2, so the description's stamp is reported, not required to
be 2. The bare-description control (no occurrence for an unlowered
description) is reported, not gated (2026-10-18). Missing NPs still fail.
Stop at a failed gate; native identity learning waits for this gate, and a
green mechanism sweep does not pass it.

    [the black cat] [runs] .
    [the dog] [chases [the cat]] .
    [the cat] [is on [the mat]] .

**5. Tying sentences together** — two or three sentences; a pronoun or a
definite re-mention refers to an earlier occurrence; a question about the
first sentence is asked after the second. Every sentence is a 4c/4d
sentence; the only new thing is the reference.

    the cat sleeps . it runs .                         what runs ?   → the cat
    a dog runs . the dog sleeps .                      what sleeps ? → the dog   (one row)
    a dog runs . a dog sleeps .                                                   (two rows)
    the cat sleeps . a dog runs . the cat eats .       what eats ?   → the cat   (same row as the first)
    the cat is on the mat . it sleeps .                what sleeps ? → the cat

Also at stage 5 (2026-10-16): the change lesson for verbs — the same
individual mentioned before and after a VP, so that the VP's operation
can be learned as the difference between its occurrences:

    the cat runs . the cat is tired .          (runs: → tired)
    the dog sleeps . the dog is rested .       (sleeps: → rested)

**6. Graded composition** — one more round per item: coordination,
relative clauses, embedding; read in order and shuffled.

    the cat and the dog run .
    the cat that sleeps is black .
    the dog chases the cat that runs .

**7. Context as content** — documents in parallel, interrupted, sharing
facts.

    A: x is three .   B: x is seven .      A: what is x ? → three   B: what is x ? → seven
    A₁: the cat sleeps .   B: a dog runs .   A₂: what sleeps ? → the cat

**8. Thinking** — `plus` and chained identities, one step first; the
successor is a verb, numerals stay opaque words.

    x is three . what is x plus one ?             → four      (three plus one is four . given)
    y is x . x is three . what is y ?             → three

| construction | first appears |
|---|---|
| a single percept whole; repetition | 1 |
| a novel word; a word as a concept | 2 |
| several words; order; a named position | 3 |
| `is` between two atoms; the period; `what … ?`; variables and numerals | 4a |
| determiners `the`, `a`; attributive adjectives; the phrase as a unit; asked parts | 4b |
| an intransitive verb; the subject role | 4c |
| a second noun role; transitive verbs; prepositions; comparatives; kind identity between NPs; `does` | 4d |
| a pronoun; definite re-mention; a document of several sentences; a question across sentences | 5 |
| coordination; relative clauses; embedding | 6 |
| parallel documents; interruption; facts shared across documents | 7 |
| `plus`; identities chained across sentences; worked steps | 8 |

---

## Phase 1: Embedding Pretraining (`make train` / `embed.py train`)

Produces a static embedding artifact (e.g. `BasicModel.kv`). Output path from
`<embeddingPath>`. Under `make train` (`bin/train.py`), Phase 1 runs **only**
when the artifact is absent or `--force-embeddings` is passed (byte-lexer
configs skip it entirely); Phase 2 always runs.

Phase 1 is a legacy path for word-level lexicons. Under the byte lexer it does
not run, `model.xml` leaves `embeddingPath` empty, and `MereologicalCodes.derive`
writes every row's meaning from its placement, so pretrained vectors would not
survive. Conceptual meaning is learned online, by placement at kept closes.

### Pipeline

```
FineWeb-EDU parquet shards
    -> Pass 1: stream documents, lex + parse, count words, build vocabulary
    -> Pass 2: stream documents again, train SBOW per sentence, discard examples
    -> Save WordVectors to sentence.pt
```

### SBOW (Sentence Bag of Words)

The current streaming SBOW trainer uses a `Lexicon` embedding table only: no
vocabulary projection head and no full softmax. Given a sentence of $N$ words,
it builds a Gaussian-weighted leave-one-out in-group center (`pode`) for each
word, samples a same-power random out-group, and trains two attractive terms:

- the word vector is pulled toward its in-sentence pode;
- random out-group vectors are pulled toward the pode's torus antipode.

Implementation: `StreamingSBOWTrainer._train_sentence()` (private) in
`bin/embed.py`. `sim(...)` below is `Lexicon.similarity`, the wrapped-MSE
torus similarity in $[-1, 1]$ --- not a dot product.

```python
vecs = embeddings(idx)              # [N, D]
pode = inner_kernel @ vecs           # [N, D], diagonal excluded
antipode = Lexicon.antipode(pode)
loss = -logsigmoid(sim(pode, vecs)).mean()
loss -= logsigmoid(sim(antipode, out_vecs)).mean()
```

### CBOW (Continuous Bag of Words)

CBOW predicts each target word from its context; every in-vocab word in the
sentence takes a turn as the target ($N$ examples per sentence), with a
leave-one-out padded context mean:

$$\bar{c}_i = \frac{1}{|C_i|} \sum_{j \in C_i} v_j$$

Loss applies the same negative-sampling objective per-word to the padded
context mean rather than the leave-one-out centroid:

$$\mathcal{L}_{\text{CBOW}} = -\log \sigma\big(s(\bar{c}_i, v_{w_i})\big) - \frac{1}{K}\sum_{k=1}^{K} \log \sigma\big(-s(\bar{c}_i, v_{w_k^-})\big)$$

where $s(a, b)$ is the wrapped-MSE torus similarity (`_wrapped_mse_score`,
$1 - 2\,\mathrm{mean}(\delta_{\text{wrap}}^2)$ in $[-1, 1]$ --- not a dot
product), $K = 64$ is the negative-sample count, and $w_k^-$ are uniformly
sampled words. No full-softmax vocabulary head is involved. Implementation:
`PretrainModel.train_step` $\to$ `_neg_sampling_loss` in `bin/embed.py`.

### Two-Pass Architecture

- **Pass 1 (vocabulary):** Stream all documents, count every word. Words
  meeting `min_count` are promoted; the model (a `Lexicon` embedding table
  only --- no vocab-projection head) is allocated once at final vocab size.
- **Pass 2 (training):** Stream documents again. Per sentence, run one SBOW
  step and discard examples. Multiple epochs re-stream the data.

### Configuration (train.py / environment overrides)

| Variable | Default | Description |
|----------|---------|-------------|
| `BASIC_DATASET` | XML default | Dataset/data source selector |
| `BASIC_MAX_DOCS` | XML default | Max documents to process |
| `BASIC_NUM_SHARDS` | XML default | FineWeb-EDU shard count |
| `BASIC_NUM_EPOCHS` | XML default | Phase 2 epoch override |
| `BASIC_BATCH_SIZE` | XML default | Phase 2 batch-size override (`--batch-size`) |
| `BASIC_MAX_TOKENS` | XML default | Token budget |
| `BASIC_MAX_BATCHES` | XML default | Phase 2 batch cap |
| `BASIC_RANDOM_SHARDS` | `0` | `1` = randomize FineWeb shard order (`--random-shards`) |
| `BASIC_CHECKPOINT_EVERY_BATCHES` | XML `checkpointEveryBatches` (0) | Mid-epoch periodic checkpoint cadence in batches; 0 disables |
| `BASIC_RUN_TEST` | unset | Enable test passes; optional value caps test batches |

### Memory Considerations

For SBOW, dominant cost is the `Lexicon` table plus optimizer state. With 200K
vocab and 100 dims:

- Embedding: 200K $\times$ 100 $\times$ 4 bytes = ~80MB
- Optimizer state depends on the chosen optimizer and device

Streaming SBOW trains per sentence and normalizes the Lexicon after each step.

---

## Phase 2: Network Training (`Models.py`)

The network learns to predict and reconstruct sentences using pretrained
embeddings.

### Within-sentence reconstruction objective

Every sentence read with a grammar reconstructs its input twice, through one
decoder, `BasicModel.reverse` (`SentenceField.reconstruct`; BasicModel enables
`teacherReconstruction` and `reconstructInLoop` in
[BasicModel.xml](../data/BasicModel.xml)). The **input reconstruction**
reverses the parallel field of the open attention pass, with no grammar,
against the detached native events of the presented input
(`reconstruction_reverse.what/where/when`). The **order-zero reconstruction**
runs the generate walk on the sentence's untransformed end slots, places the
emitted terminals as new occurrences, and reverses that order-zero field
through the same decoder against the same events
(`reconstruction_order_zero.what/where/when`). Both are event errors per band,
weighted by `reconstructionScale`: the first trains perception, definitions
and basic percepts; the second trains compose, the grammar inverse and the
definitions it selects. The decoded bytes' identity and missing positions
against the input are audits, not trained (`reconstruction.free_bytes`,
`reconstruction.coverage`). A sentence with no admitted surface candidate
contributes no reconstruction term and increments
`reconstruction_unavailable_sentences`. The targets never supply candidate
spellings or reconstructed ideas. [Architecture](Architecture.md), "The three
stages: input, thinking, output", gives the loop.

The owned `InputReconstruction` retains the recovered ideas, the realized
input event, the costs per sentence and truncation. Ownership is established
during understanding, before later staging can overwrite the carriers.
`reverseReconstruct` returns that completed event; an explicitly supplied
target only changes its optional diagnostic event score
([Models.py](../bin/Models.py)).

Observed targets are detached. A choice is credited by its whole path: a
departure's reward is its complete path's change in $R + A$
([What questions and the two primary costs](#what-questions-and-the-two-primary-costs)).
Compose lessons update only their chooser. The parameter ownership and
concluded-answer cut are specified in [GradientFlow](GradientFlow.md).

Separate reconstruction compilation caches its backward and disables donation
of saved buffers. This permits repeated gradient reads for per-objective
diagnostics even when the first backward used the cache only once. The compiler
normalizes this PyTorch build's disabled-donation metadata without permanently
changing its global setting ([Models.py](../bin/Models.py)).

Every retained meronomy reading with a grammar, in either binding, uses tied
reconstruction; `reconstructInLoop=false` cannot disable it. The old reading
modes, detached student runtime and D3 path are retired. Whole-slab/non-grammar
configurations
retain masked-LM reconstruction: `create_ir_mask` replaces selected WHAT
positions with `NULL_PERCEPT`, retains `_ir_pre_mask_input`, and scores those
positions. Supplied-answer learning remains a separate objective; reconstruction
does not manufacture answer labels.

**Adaptive omission.** In training, part of each sentence is hidden before
identity admission, and reconstruction must restore it (`bin/Omission.py`).
Two rates are kept, one for whole words and one for percept slots within the
kept words, and one word per sentence is always kept whole. Omission is out of
band: a hidden slot keeps its place, bracket and time and has no content,
rather than a corrupted value, since a good model reproduces the corruption it
sees. With $v$ the long-term reconstruction accuracy on visible content, $a$
the accuracy on hidden content and $r$ = `omissionTarget`, each rate is
$m = \mathrm{clip}\left(\frac{v - r}{v - a}, 0, m_{\max}\right)$, with
$m_{\max}$ = `omissionLimit` (and $m_{\max}$ once $a \ge r$): nothing is hidden
until visible accuracy passes the target, and the rule limits itself because
$a$ falls as more is hidden. The accuracies are running averages over
`omissionHorizon` sentences, updated once per present sentence from the kept
reconstruction (`Omission.observe`). The order-zero reconstruction is scored
against the input before omission at every place; the input reconstruction,
which has no grammar, is scored only where content was observed. Only
reconstruction restores hidden content; output stays silent on text that asks
nothing. `omissionTarget = 1` disables omission and its random draws.

| Knob | Effect |
|------|--------|
| `maskRate` | Bernoulli mask probability at the subsymbolic (PS) (BERT default 0.15) |
| `reconstructionScale` | Independent priority of each relative reconstruction term. It no longer subtracts from the answer's weight. |
| `reconstructInLoop` | Historical fixture opt-in; retained meronomy grammar readings always reconstruct their understanding. Training/evaluation use the owned byte objective once. |
| `reconstructionBasisLimit` | Positive count, default 16: the sentence bank adds up to this many most-primed other symbols to its own rows. It also bounds free-inverse candidates per side (at most its square in pairs). Independent of word, STM and field capacity. Missing candidates/inverses report incompleteness. |
| `grammarLessonWeight` | Priority for explicit compose/generate lessons; each lesson updates only its own chooser. `forwardGrammarWeight` is retired and rejected. |
| `<reconstruct>` | RETIRED (2026-06-09; `reconstructEnum` removed). There is no target space-role knob: reconstruction is seeded from the understanding and weighted by `reconstructionScale`. |

### Construction trace and derivation pressure

The serial reading records its concluded root and slots, rule choices,
operand positions and per-word references. No witness offsets or operation
frames enter the shared understanding. Sentence-local journals use the loop's static word-capacity bound,
so changing the sentence length does not create a new capture shape. Each
path freezes these fields and the post-seen-write priming snapshot into one
`SentenceUnderstanding`. The tied inverse and the detached answer consume the
same object; the public inverse returns its completed reconstruction rather
than trying to replay a discarded journal.

The primed bank contains the sentence's own symbols plus the most activated
other symbols, per batch row. Surface-bearing symbols are reconstruction
candidates; all valid symbols supply detached answer context. The priming of
the next read is the activation predictor's forecast of the next state
(`ConceptSpread.open_prediction`), read detached; nothing diffuses along
concept store edges. The bank and all record fields are described in
[GradientFlow](GradientFlow.md).

The compose chooser is credited by the whole-path comparison: a score-function
term, the departure's probability times the detached difference between the
complete paths' costs (`output.compose_score_function`). Explicit grammar
lessons, where configured,
train only the corresponding compose or generate chooser. The former
`forwardGrammarWeight` structural proxy and detached reverse chooser are
retired. A current checkpoint containing student keys is migrated by dropping
those keys and their optimizer entries, preserving shared compose weights.
Reader Adam state is preserved by name; reconstruction starts momentum SGD
when loading an older Adam checkpoint. No replacement student is created. The historical
student contract is retained in the migration section below.

The following references describe composition and reconstruction objectives;
they do not establish fidelity or stability of this implementation:

- Original DisCoCat lifts grammatical reductions to semantic morphisms that
  compose constituent meanings into a whole; it does not require those
  generally information-losing maps to be invertible
  ([Coecke, Sadrzadeh, and Clark, 2010](https://arxiv.org/abs/1003.4394)).
- Functorial language models obtain a probability distribution over word
  sequences from a grammar-to-meaning monoidal functor, suggesting grammatical
  likelihood as a generative objective
  ([Toumi and Koziell-Pipe, 2021](https://arxiv.org/abs/2103.14411)). A recent
  DisCoCat tensor encoder also found that contrastive training over explicit
  derivations improves sensitivity to word order and predicate roles
  ([Lo et al., 2025](https://aclanthology.org/2025.starsem-1.25/)).
- FCA defines concepts as fixed points of the Galois connection, and formal
  concepts are sufficient optimal factors for reconstructing Boolean
  object--attribute incidence matrices
  ([Denniston, Melton, and Rodabaugh, 2013](https://arxiv.org/abs/1309.5134),
  [Belohlavek and Vychodil, 2010](https://doi.org/10.1016/j.jcss.2009.05.002)).
  For graded activations, fuzzy Galois connections extend the same construction
  to complete residuated lattices
  ([Belohlavek, 1999](https://belohlavek.inf.upol.cz/publications/Bel_Fgc.pdf)).

The former detached student's exact-leaf objective is retired. The retained
reconstruction signal scores the tied inverse of the understanding.

### `<trainEmbedding>`: Embedding Update Mode

Controls whether and how embeddings evolve during network training. When
enabled (CBOW, SBOW, or BOTH), implements EM-like alternation:

- **E-step (network):** Forward + masked prediction + backward + optimizer.
- **M-step (CBOW/SBOW):** Run one embedding update on the same sentence.

| Value | Embedding | Model layers | Description |
|-------|-----------|-------------|-------------|
| `NONE` | Frozen | Trained | Only model layers train |
| `CBOW` | CBOW (padded context, own optimizer) | Trained | Model layers train separately |
| `SBOW` | SBOW (centroid, own optimizer) | Trained | Faster variant |
| `BACKPROP` | Backprop only | Trained | Codebook trained purely via model loss |
| `BOTH` | SBOW post-batch | Trained | Two optimizers |
| `JOINT` | Single backward | Trained | Single optimizer: combined model + SBOW loss |

### Gradient Flow Through Perceptual Embeddings

This section concerns perceptual embeddings, not the buffer-owned concept
dictionary above. `<trainEmbedding>` determines whether embedding parameters
appear in the
**main optimizer** --- that is the whole freezing mechanism. There is no
mode-conditional `detach()` of the codebook weight anywhere:
`optimize_embedding = train_embedding not in ("NONE", "CBOW", "SBOW")`, and
`getOptimizer` filters embedding params out of the main param list when it is
False. Under NONE/CBOW/SBOW gradients may still flow through the lookup, but
no main-optimizer step ever moves the rows:

| `trainEmbedding` | In main optimizer | Embedding rows moved by |
|------------------|-------------------|-------------------------|
| `NONE` | No | Nothing (frozen) |
| `CBOW` | No | CBOW pretrainer (own optimizer) |
| `SBOW` | No | SBOW pretrainer (own optimizer) |
| `BACKPROP` | Yes | Model loss only |
| `BOTH` | Yes | Model loss + post-batch SBOW step |
| `JOINT` | Yes | Single combined loss |

- **`NONE`** is recommended for constrained hardware (Apple MPS <16GB).
- **`CBOW`/`SBOW`** maintains clean EM separation.
- **`BACKPROP`** is the simplest trainable mode. Embedding shaped entirely
  by the task; best for small vocabularies.
- **`BOTH` risks gradient interference** --- reconstruction pulls embeddings
  apart (distinguishability); SBOW pulls co-occurring words together. These
  forces can conflict because they use separate optimizers.

### `JOINT`: Single Combined Loss

At batch end, JOINT adds embedding error to the registered model objective
in the same optimizer update:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{model}} + \lambda \cdot \mathcal{L}_{\text{SBOW}}$$

where $\lambda$ is `<embeddingScale>` (default 0.1).

The default, `separateChooser=false`, takes one backward over the sentence's
objectives in two traversals (`GradientProjection.backward`): the R side
(reconstruction, scaled by its jitter draw, and every cost not on the answer
side) and the A side (output, the generate lessons and the choosers' credit).
In each optimizer unit (a concept row, a definition edge, otherwise a whole
tensor) and in Adam's second-moment metric, the A gradient loses only its
component that opposes R, and the two are summed: answer learning never undoes
reconstruction. With `separateChooser=true`, reconstruction, expectation and
output instead have separately named objectives in Error and disjoint
parameter owners: reconstruction owns perception, trainable codes, compose
operators and their tied inverses, and the compose chooser; the supplied
answer owns its readers and stops at the common understanding record; and
restricted backward enforces these boundaries even when a forward operator is
shared. In both modes expectation owns its predictors with source and target
detached, attention keeps its stopped consumption boundary, intuit's weights
remain output-only, and compose and generate lessons update only their own
choosers. `<separateChooser>` defaults to `false` (the code and
`data/model.xml`); `<reconstructionJitter>` defaults to zero and scales
reconstruction by one `U[1-j,1]` draw per sentence. The percept predictor's
arrival adaptation is loop state, with its accumulated loss in the shared
sentence optimizer in both modes. Each epoch reports the R and A gradient
norms reaching the understanding, the projection's cosine and the fraction of
the answer gradient it removed, the per-operator dissonance and the per-group
chooser variance; diagnostic work is excluded from the training-step time. See
[GradientFlow](GradientFlow.md#current-gradient-boundaries-and-loss-normalization).
`branchDiagnosticsEvery` reports actual objective writers per optimizer parameter.
See [GradientFlow](GradientFlow.md) for the complete term and record inventory.

Meronomy is the only retained text mode. Every retained serial grammar reading
reconstructs through its understanding. Grammar-free configurations retain
perceptual reconstruction. A sentence with no admitted surface candidate has
no reconstruction term and is counted. Keep and credit read $R + A$ over whole
paths: each stream keeps its strictly cheapest complete path, ties keeping the
greedy one, and expectation is excluded from the comparison and trains only
the predictors. The answer cost has one exposure per sentence, given to the
kept path and shared with the paths whose departure is in compose and their
greedy path.

*A superseded passage beginning "Inter-sentence MSE/contrastive losses are consumed independe" moved to [DevelopmentHistory](DevelopmentHistory.md) (2026-10-30).*

*A superseded passage beginning "SBOW loss uses the same negative-sampling objective, with $s" moved to [DevelopmentHistory](DevelopmentHistory.md) (2026-10-30).*

### Why MSE over Embeddings (not Cross-Entropy)

- "cat" prediction when target is "kitten" incurs less loss than "democracy"
  --- embedding space captures semantic similarity.
- Output dim is the embedding dim (~100) vs. vocab size (~200K).

### Training Loop

The live aligned P/W-to-concept handoff uses plain Sigma:
`a_c = tanh(b_c + sum_i W_ci * evidence_i)`. Inputs are up to eight identified
part/whole-code references at a location, not fold RMS scalars; there is no
source-count mean or gate. BasicModel opts into weak incoming-connection L1
with `ConceptualSpace/conceptReadoutL1=0.01` (absent/zero disables it). Each
concept's weights/bias are gathered from `IndexedSigmaConceptsFromPercepts`
at sentence staging, then consumed by
the eager or compiled word loop. Sparse gradients use the compact row-local
optimizer and appear under reconstruction in the ownership audit.
Within the single optimizer step, a diagonal-metric L1 prox soft-thresholds
only admitted input connections. Its threshold is
`lr * (lambda / distinct_observed_concepts) / (sqrt(v_hat) + eps)`;
biases and unused slots are excluded. `concept_readout_l1` is a separate
detached reporting cost, not another backpropagated penalty. Output-gradient
state boundary stays unchanged; L1 itself is an intentional sparsity/accuracy
tradeoff, not a promise of monotonic reconstruction improvement. Policies are
batch-local, never applied in evaluation, and cleared at the next `zero_grad`
after a skipped update.
If the readout has no learning gradient (`grad=None`), it is left untouched:
sparsity alone must not collapse a definition without accuracy feedback.
Coupled reconstruction updates and missing-gradient no-update behavior are
tested separately.
There is no second optimization step. Full processed native fold state remains
separate from this scalar readout and is released at batch/sentence teardown.
Checkpoint coefficient tensors and the conceptual vocabulary's reference
manifest must be restored together.

Data flows through `SentenceStreamDataset` wrapped in `DataLoader`. The
ordered training list is split into `B = batchSize` contiguous slabs of
length `L = len(split) // B`; at step `t`, row `b` is item `b * L + t`.
Temporal context is coherent across steps. No per-epoch global shuffle.

When `<packSentences>true</packSentences>`, the real `runEpoch` optimizer
loop advances each contiguous row until the next complete sentence would
exceed `serialWordCapacity`, the raw byte slab, or the fixed sentence-root
FIFO. Thus one B-wide optimizer brick can contain a different number of
sentences in every row while preserving each row's corpus chronology. A
sentence never crosses a brick. Under the canonical `<analysis>meronomy
</analysis>` the brick budget and the word-to-sentence layout are counted
in the analysis ladder's units (whitespace and digit units included; the
joining space belongs to the sentence it follows), the same cut the stem
stages, so the packed layout and the staged unit mask agree by
construction (`WholeSpace.unit_spans_of_bytes`,
`InputSpace.sentence_unit_count`). The packer runs on the prefetch
thread, so the unit tiler reads a host copy of the boundary predicates,
primed when the predicates are built or updated and refreshed by every
main-thread tiling (never the accelerator parameters; a worker thread
without the copy fails loud); a boundary update between packing and
staging is caught by the stem's alignment check. Explicit word-to-sentence IDs drive the
row-local soft resets inside CSLang; the final sentence in each row remains
live through loss, backward, discourse, and the contextual concept update,
then resets at the eager brick boundary. The log reports complete sentences
per second at epoch or wall-clock-cap exit.

For each B-wide batch:

1. `_start_spaces_for_forward()` calls `Start()` on every Space
   (`runBatch` pre-invokes it before the compiled step; `forward()`
   self-invokes when not externally started).
2. `InputSpace.forward()` lexes/embeds once into `[B, N, D]`
   (left-aligned, right-padded to N). No K axis, no cursor unfold.
3. In whole-slab, grammar-free configurations `create_ir_mask` replaces a
   `mask_rate` fraction of WHAT positions with `NULL_PERCEPT`; the pre-mask
   event is stashed on `_ir_pre_mask_input` and the position mask on
   `_ir_mask_positions`. A reading with a grammar uses adaptive omission
   instead.
4. `_forward_body` runs T stages on B rows.
5. `_forward_head` produces `[B, N, predDim]` (a side channel --- IR
   loss is computed at the subsymbolic (PS), not at the head).
6. `runBatch` reads the trained sum from `Error.total()`; the registered
   terms are listed under
   [Relative errors and their owners](#relative-errors-and-their-owners-item-69-october-2).
   Raw fidelity metrics remain available but do not duplicate them.
7. Each sentence runs one greedy path and `explorePaths` alternatives, each
   reusing the greedy prefix and departing once. All are costed under the same
   parameters; each stream keeps its strictly cheapest $R + A$ path (ties keep
   the greedy one), and input and thought effects commit only after the
   comparison. The sentence then takes one optimizer step, in either ownership
   mode. The batch-end step runs only when a remaining registered term
   differentiates. Diagnostics observe these updates without adding one.
   Percept expectation's predictor steps its own local Adam once per word
   inside the perception loop.
8. Embedding training (`CBOW`/`SBOW`/`BOTH`) runs once per batch.

---

## What questions and the two primary costs

### Relative errors and their owners (item 6.9, October 2)

**The walk residual (2026-10-30).** The general registry rule below has one
exception. Generation's walk residual is three absolute chart errors, form,
for and against (`reconstruction.walk_residual.*`, `output.walk_residual.*`):
each sums all chosen steps and the unresolved stack energy, and enters the
registry with baseline one and the active-stream mask. Exactness is absolute
MSE `1e-8`; an admissible nonfinite residual raises before menu scoring. Term
balancing is avoided where possible (Alec): the projection protects
reconstruction from the answer side without a scale balance, and both
ownership settings compare paths by $R + A$. See the
[explicit formulas and boundaries](GradientFlow.md#the-normalization-contract-and-the-generation-exception).

`Layers.Error` stores every constituent's numerator, detached uninformed
baseline, active count, configured priority and owner. Contributions with
the same name combine before division. The trained objective is the ratio
of their means, not a mean of independently normalized rows. Squared errors
use the origin's mean squared target norm; categorical and binary errors
use uniform predictions (`log K`, `log 2`, or `log 256` for a byte). Neither
a batch-fitted mean/base rate, an epsilon floor nor a running loss scale is
used. An exactly zero squared target makes an unnormalized penalty; an
empty target mask contributes zero. A nearly solved error stays small.

Every error defaults to weight 1. Existing explicit priorities in the merged
configuration, including `model.xml`, remain in place. The old
`(1-r)*answer+r*reconstruction` blend is retired: `r` weights reconstruction
independently. The [complete term inventory](GradientFlow.md#trained-total-and-diagnostic-objectives)
specifies each comparison, baseline, weight, placement and gradient reach.
In registry names, the trained families are: the answer cost (`answer`, side
A); `reconstruction_reverse.what/where/when`, the input reconstruction through
the System 1 decoder, scored at observed places;
`reconstruction_order_zero.what/where/when`, the order-zero reconstruction
through compose, the generate walk and the same decoder, scored against the
input before omission; `reconstruction.free_percept`, the decoded percepts
against the percept forecast's residual; the walk residuals and departure
credits of reconstruction and output; the expectation predictor's own terms
(`expectation.*`, source and target detached); and the explicit grammar
lessons (`grammarLessonWeight`). `reconstruction.free_bytes`, the decoded
bytes' identity against the input, and `reconstruction.coverage`, the missing
positions, are audits with `trained=False`.

`definition_sparsity` (soft-L0), `gate_l1` (operator gates), and
`truth.falsity/balance` have no targets and retain their configured penalty
strengths without normalization. `concept_readout_l1` records the distinct
observed concepts' incoming L1 with `l1Lambda`; its sole training owner is
the proximal operation at each applicable sentence or batch optimizer step.
A batch without a batch-end backward stages no additional update.
`selected_thought_policy` is a targetless score-function objective with its
configured policy weight and a detached return baseline. These terms, and
zero-target squared penalties, are reported separately from relative errors. `Error.total()` sums trained terms;
`total(kind='relative')`, `total(kind='penalty')` and `breakdown()` expose
the distinction. Raw fidelity metrics and missing-candidate counts have
`trained=False`.

A path is scored from one immutable understanding record, shared by
reconstruction and the answer reader. Its primed symbols are the sentence's
own rows plus the most activated others, with a separate priming surface for
each batch row. Every path of a sentence is scored before the sentence's one
optimizer step, and the kept path is the strictly cheapest by $R + A$.
Reconstruction uses momentum SGD (0.9), and the readers and expectation
predictors use Adam. The configured learning rate is unchanged; there is no
adaptive normalization of reconstruction steps. The audit records code and
chooser-anchor displacements against their gradients and each sentence's
modal derivation over training epochs.
[GradientFlow](GradientFlow.md#the-training-step-october-1) lists the record
encoding and ownership audit, including inactive optimizer parameters.

Every training batch is a batch of `WhatQuestion`s (`bin/What.py`), not a bare
input/output tensor pair. `runBatch` derives them from the cursor rows
(`_questions_for_batch`: supervised when the dataset has outputs, present
otherwise, future on `predict` trials, inference at runtime) or from the
`<whatCurriculum>` schedule, and scores two independently weighted,
independently normalized primary costs recorded before `backward()`:

| Cost (`primary_costs()`) | Compares | Trains |
|---|---|---|
| `input_reconstruction` (+ `input_reconstruction_reverse`, `reverseReconstruct()`'s own cost) | the reconstructed input with the presented (or clean) input | the bottom-up understanding and its input-associated inverse |
| `answer_construction` | the realized response from `reverseOutput()` against an available, separately supplied `Data.what(What.supervised(...))` answer; automatic temporal targets are evaluation metrics only ([Models.py](../bin/Models.py)) | the generate chooser, the question conditioner and OutputSpace's percept adapter; on shared parameters its gradient is projected against reconstruction's (with `separateChooser=true`, shared operators transmit it without an answer update) |

The desired answer enters loss preparation only after the model response is
fixed.

**Answers, silence and referents.** Output is never a direct consequence of
understanding the question; it is the answer that thinking produced. The
answer is resolved only from a concluded, accepted thought; with no conclusion
the output role is empty and the model is silent. `Data.what` distinguishes
three cases. A split that supplies no outputs, or an inference presentation,
has no desired answer and is masked out, so text that supervises no output
trains through reconstruction only. A row with a supplied answer is scored
against it. A row of a supervised split with no supplied answer, or a sentence
for which no question is supplied, has `WhatAnswer.silence`: an available
target whose text is the empty surface (the end marker alone, through the same
embedding as text targets) and whose tensor value is the zero vector.
Producing words where nothing was asked therefore costs A, and an empty
construction against silence costs zero, which teaches both answering and not
talking. A loss-side `_answer_silence_mask`, one flag per stream and sized at
construction, records which rows carry silence; it never enters model context.
When the supplied answer names a referent (`ReferentTarget`) rather than a
sentence, the answer cost adds a binding term to the surface comparison
(`BindingAnswers.cost`): the exact identity costs 0; an unbound answer, a
thought not accepted or a target with no concept costs 1; any other binding
costs one minus the best coverage of its content by the target's concepts. The
term is chosen by the kind of the supplied answer, never by a configuration
flag.

---


*Moved to [DevelopmentHistory](DevelopmentHistory.md#historical-reconstruction-objectives-and-migration-2026-09-12) (2026-10-30): "Historical reconstruction objectives and migration (2026-09-12)".*
## Epoch report: throughput and word units

`runEpoch` closes a packed epoch with one line, e.g.
`Packed training throughput: 412 complete sentences in 61.2s = 6.73
sentences/s (16 optimizer bricks, B=8, W=64; word units 99.6% of 5,120
words)`.  The per-batch `batch = k (Δ=…s)` lines give the steady-state
rate once compilation has warmed up.  `word units` is the assurance that
the model operates over words now that the analysis tiling is learned
rather than fixed (`<boundaryTypes>`, doc/Mereology.md): the fraction of
whitespace-delimited words of the epoch's presentations that were staged
as exactly one unit (`BaseModel.word_unit_fraction()`; counters reset per
epoch).  Under the canonical boundary types plain text reads 100%; a
numeral under `<digitWholes>` is deliberately several digit units, so it
counts as a word that is not one unit; the atomic cold start
(`<boundaryTypes>none</boundaryTypes>`) reads 0% until the boundary
learner has acquired the whitespace boundary (test/test_meronomy_ladder.py).

## SBOW vs CBOW

| Property | CBOW | SBOW | Masked Prediction (Phase 2) |
|----------|------|------|----------------------------|
| Targets per sentence | N (every in-vocab word) | N (every word) | N (one per masked pos) |
| Context | Padded LOO mean of other in-vocab words | LOO centroid of N-1 | All unmasked |
| Positive updates | N per step | N per step | N per step |
| Repulsive force | K=64 random negatives pushed from the context mean | random out-group to torus antipode | Implicit via MSE |
| Signal density | High | High | High |
| Loss | two `-logsigmoid` negative-sampling terms (wrapped-MSE scores) | two `-logsigmoid(sim(...))` terms | MSE over embeddings |
| Updates embeddings | Yes (own optimizer) | Yes (own optimizer) | Only for `BACKPROP`/`BOTH`/`JOINT` |
| Used in (`<trainEmbedding>`) | `CBOW` | `SBOW`, `BOTH`, `JOINT` | `BACKPROP`, `BOTH`, `JOINT` |

Both embedding trainers now make every in-vocab word a target under negative
sampling; neither uses a full-softmax projection head. The remaining
difference is geometric: CBOW queries a uniform padded context mean and
pushes $K = 64$ random negatives away from it, while the streaming SBOW
builds a Gaussian-weighted leave-one-out pode and pulls random out-group
vectors toward its torus antipode.

---

## Integrated checkpoint architecture

| Artifact | Example | Contents | Updated by |
|----------|---------|----------|-----------|
| XML config | `data/BasicModel.xml` | Architecture, objectives, corpus and checkpoint path | Hand-edited |
| Checkpoint | `output/BasicModel.ckpt` | Model and embedding state, vocabulary/BPE extras, optimizer, counters, RNG and corpus manifest | Phase 2 |

The integrated checkpoint is resumable. Mid-epoch resume requires the same
corpus manifest and batch size; mismatches fail loudly instead of replaying a
different cursor. `--force-embeddings` remains a deliberate Phase-1 rebuild,
not a separate model artifact contract.

Every owner's checkpoint state carries a version. Loading an older layout, or
state in retired coordinates, raises and names the owner; nothing is
converted lossily or backfilled with defaults. Where old content maps exactly
onto the new layout, an explicit migration converts it and reports every key
or column it drops. A lesson artifact that cannot be loaded is regenerated by
running its producing lesson again; a test treats an artifact as a
prerequisite and never trains to produce one. Modules built on first
use, such as the question conditioners, are instantiated from the shapes
saved in a checkpoint before the strict key audit
(`_materialize_answer_path_from_checkpoint`), so a fresh model loads them with
`require_match=True`; a module created after the optimizer exists joins it
before the first backward that uses it, never twice.

---

## Embedding Space Geometry

`Lexicon` defaults to projective unit-ball lookup in the model, while the
streaming SBOW trainer still constructs its training table with `ball=False`
for the legacy torus pode/antipode objective. See [Lexicon.md](Lexicon.md) for
the current geometry and the migration note.

---

## Profiling

`--profile` wraps Phase 2 in `cProfile`:

```bash
python bin/train.py --model data/BasicModel.xml --profile --max-docs 500
# make train_micro is the capped micro run (no --profile flag of its own;
# invoke bin/train.py directly to add it). make bench_remote profiles the
# recon bench on ArborStudio via torch.profiler, not cProfile.
```

Output: a `.prof` file in `output/profiles/` plus a top-30 summary to stdout.
View with `snakeviz`:

```bash
pip install snakeviz
snakeviz output/profiles/train_20260319_111441.prof
```

For live profiling, `py-spy` can attach by PID (requires root on macOS):

```bash
pip install py-spy
sudo py-spy record -o profile.svg --pid <PID> --duration 30
```

## Category utility and chunk admission (2026-09-10)

At the training path's sentence boundary `ConceptualSpace` commits the
presentation's utility observations (one per resolved unit: the concept with
its constituent rows and property rows as features) and the chunk proposals
made by the reduce step; nothing is committed while online learning is frozen,
as in evaluation. The estimator of the basic level is category utility
(Corter & Gluck 1992),
$\mathrm{CU}(c) = P(c)\left(\sum_f P(f \mid c)^2 - \sum_f P(f)^2\right)$
over the concept's features, Laplace-smoothed by `utilitySmoothing` and
withheld below `utilityMinCount` presentations (`category_utility`); a concept
and a concept-feature pair count at most once per presentation. A proposal
that has recurred `admissionCount` times is admitted as a concept over its
member concepts only if the phrase's utility exceeds the best utility of its
members. That gain also moves the learned `chunk_prior` logit by
`utilityPriorRate`, so phrases that predict their parts better than their
parts do become easier to chunk.

An admitted phrase also gets a row of its own in the concept dictionary,
initialised from the additive composition of its members; when the
reduce pass chunks a pair that matches an admitted phrase, the folded
parent's STM reference is set to that row, so later reads and the
answer path use the phrase's row and the losses train it. The utility
counts, the phrase hits, admissions, rows and gains, and the acquired
predicates of split property rows ride the structural extras of the
checkpoint.

## Existence evidence and observation admission (September 16)

The three forward observation writers retain canonical role presence,
including the depth-two VP, and mark records as observations. Explicit
TruthSet admission can accept a fact; an origin tag or high activation alone
cannot. Internal questions, predictions and unverified legacy relations remain
ineligible for `Exist` fact support. The lookup preserves both signed degrees
and provenance, and adds no gradient objective or trainable parameters.
[Observation adapter](../bin/Models.py),
[admission](../bin/Layers.py),
[lookup](../bin/reasoning.py).

A given truth (`<truthSet>`, `store_truths`) is a plain sentence with a trust
value; nothing out of band says what kind of sentence it is. The grammar
decides whether it is an idea or a relation, as for any input, and provenance
supplies only trust, origin and the source text. Provisioning therefore waits
for training: an untrained chooser cannot be expected to read a given truth in
the right form, so a test that needs relation rows from provisioned text uses
a trained checkpoint or forces the derivation and says so.

See [Existence evidence](ExistenceEvidence.md) for complete-description matching,
fact admission, checkpoint migration and the hard lookup boundary, and
[GradientFlow](GradientFlow.md) for the architecture-wide gradient boundaries.

## Conceptual-taxonomy queries (September 16)

Successful checked taxonomy queries preserve native proof sources and cannot
materialize world facts. Their bounded reader traversal is distinct from
policy-selected nested `what` execution. The frame-kernel curriculum,
`legacy_*` readers and optional bridge/prediction policy objectives are removed.
Only the normal thought chooser receives episode credit. Mechanism tests do
not establish learned utility. See [Taxonomy queries](TaxonomyQueries.md) and
[SelectedMeaning](SelectedMeaning.md).

## Checked queries and native grammatical referents (September 16)

Checked query contracts and shared VP binding add no objective or optimizer
parameter. Captured answer programs retain the native concept IDs selected by
forward's OBJECT/WORD row decision; an unknown referent stays unknown. These
addresses remain separate from numerical semantic features and survive later
staging. Pure grammatical formation retains live operand gradients. Hard
lookup is nondifferentiable and durable descriptions stay detached. The normal
controller retains live episode values and receives supplied-answer policy
credit. Residual credit now has its own baseline and trajectory replay on that
same chooser; learned utility still requires measurement. See
[ExpectationRetention](ExpectationRetention.md).
[Capture](../bin/Models.py),
[owned programs](../bin/Understanding.py),
[formation](../bin/Queries.py),
[details](QueryContracts.md).

## Ordinary history and live query credit (September 17)

Ordinary history supports same-level continuation, strictly nested descent and
return, explicit finish, and one shared work budget. Episode-mode meanings and
selected thought-occurrence reads stay live through the single optimizer step;
after a finished episode, the existing owner detaches them. Restored history
is detached and cannot refresh prior budget or pressure. Storage itself adds no
loss or optimizer parameter; the normal selected controller adds its separate,
default-off policy term on top of these retained records. Residual reward and
learned utility still require their own evidence. See
[ordinary thought history](ThoughtHistory.md) and [gradient flow](GradientFlow.md).

## Sentence and reasoning permission

When tied reconstruction is enabled, it must be owned by the `Understanding`
before `resolveAnswer()` opens completed rows for checked reasoning. Supplied
executors in `understand()` and `what()` run under the same sentence mask, and
output realization cannot start a query. Permission restores on error and does
not change optimizer ownership or gradient balancing. See
[Query phases](QueryPhases.md).

## Shared selected-query work

Selected query accounting is a transient host meter, not a trainable feature or
loss. It charges the same allowance before selected preparation, executor
calls, native reads, and nested callbacks, while preserving existing live
operand gradients and durable-record detachment. It neither creates residual
policy credit nor learned utility. The normal controller creates it from the
episode allowance, propagates it and commits its actual final cost once; its
separate residual estimator uses the final metered work alongside prediction
error; the supplied-answer estimator keeps its own baseline. See
[shared query work](QueryWork.md) and [gradient flow](GradientFlow.md).

## Restoring dependent occurrences

Stateless tensor restore withdraws request authority immediately. Physical
pruning waits for the semantic sidecar and ordinary thought owners, then keeps
their reachable constituents and discards only orphans. Tensor-only restore
therefore retains zero-trust content while ownership metadata is unavailable.
Checkpoint content is detached; a normal live episode retains its existing
gradient route. See [nested retention](NestedRetention.md).


## Annotated grammar wording curriculum

`MM_grammar_wording.xml` selects `dataset=grammar` and the shipped JSON text
curriculum. `Data.loadGrammarLessons` retains separate train/validation/test
annotations, while normal text enters the forward and output paths without
those annotations. Afterwards, `runBatch` adds `grammarLessonWeight` times the
compose/generate lesson loss to the real trained total. It requires matching
unpacked source rows and captured words; evaluation, missing annotations and
prediction-only trials receive no such term. It fabricates no ordinary answer
labels. Both choosers are the existing grammar MLPs, and numerical generation
learning belongs to declared operators. This loss is supervised structural
credit, not a second thought-policy objective. Shared parameters still obey
the separate-state/shared-operator contract in [GradientFlow](GradientFlow.md). See [SelectedMeaning](SelectedMeaning.md)
for the bounded wording result and its limits, and [GradientFlow](GradientFlow.md)
for the detach/ownership boundaries.

## Item 2.7 lesson-13: ideas before relations (Alec, 2026-10-28)

The version-2 `data/corpus-lessons.json` has two curriculum phases. Four distinct
idea documents name both measured objects with demonstratives and supplied-answer
questions. Four relation documents state their kind and a kind-level sentence,
then ask for the kind and yes/no membership. Each phase runs four epochs, each
document once per epoch. The epoch boundary clears experience and MTM once before
the first document; document covers are rewritten without clearing familiarity.
The corpus checkpoint still preserves an incomplete epoch on resume.

`corpus_lessons.py` uses the math corpus's `ChainDocument`/`ReferentTarget` path;
answers are loss-side data. No order, grammatical operation or part of speech is
supplied. Every epoch reports objects and their orders, classes and members,
candidate counts/symbols/support/both status, and scored answers. The ideas exit
requires two order-1 objects; relations requires an order-2 kind containing both.
The relations phase is the registered producer of `copular_identity`. It writes a
qualified artifact only on acquisition success; a failed acquisition remains the
reported result.

**The final curriculum (2026-10-29).** `corpus_lessons.py --curriculum
lesson-13` preserves the original two phases and 160 training sentences.
`--curriculum final` selects the separately frozen proper, mass, count-kind and
kind-verb documents in `data/corpus-lessons.json`, four documents and four epochs
per phase, with supplied answers through `MathChainTraining.present`. Read-only
measurements inspect kept meanings, references, witnessed orders, candidates,
answer scores and class both-corner fractions. Per-phase held-out recombinations
run without an optimizer or vocabulary admission; their answers never enter the
training vocabulary union. A failed acquisition remains the result. Only an
unforced passing relations phase can write `copular_identity`.

At a corpus stage boundary, `Corpus.admit_stage` admits that stage's training
vocabulary before native input construction. Capacity trials declare their grid,
28 GiB aggregate process-tree RSS guard, host swap/pressure samples and one-hour
limit before launch. On macOS the same limit also guards charged physical
footprint, including compressed memory and lifetime peaks between samples.
`corpus_capacity.py --train-brick` loads the selected metadata, admits the
training vocabulary through the first native stage with all 28 document windows
nonempty, and enters that native cursor without training the skipped prefix.
It touches reservations, allocates the configured dense optimizer state,
and runs a populated B28/W256 forward/backward/step. `--search-grid` brackets and
refines inventories, relations, witnesses, bridge and recurrence in that order;
a timeout or non-memory error stops that search as inconclusive or failed. A
constructor-only result never establishes a fit. The selected full-prefix run
uses stage admission, ordinary checkpoint continuation and the same guard.
The B28/W256 capacity search and the selected corpus prefix belong to the
`RUN_SLOW` run registered in `test/test_corpus_large_slow.py` (Alec,
2026-10-29); partial capacity measurements do not select a production size.

**Percept expectation and residual reconstruction (2026-10-29).** For the
perceptual layer, the residual rule supersedes the earlier gain-only rule
(Alec, 2026-10-29/30). `PerceptExpectation` learns at the eager word boundary
with detached context and observation, using its own Adam state outside the
main optimizer. The forecast made before arrival is frozen for every path of
the sentence. The decoded percepts are compared with the detached
`observed - expected_at_arrival` target (`reconstruction.free_percept`); that
loss cannot update this local predictor. Forward recombination retains the
full attended signal and its Jacobian. Every checkpoint preserves the private
optimizer, including a completed epoch with no corpus cursor; a trained
checkpoint missing that state fails on load.

## Concept hierarchy learning and curriculum order (Alec, 2026-11-08)

The architecture learns class direction from member meaning and conditions members from higher-order context through `ConceptHierarchy`. Kept grammar applications supply members to prebuilt formal-argument concepts, using this general mechanism. Each argument starts at zero radius and learns confidence by the ordinary views-and-agreement rule. Part-of-speech learning has no special centroid, count distribution or feedback rule. Directional conditioning changes no perceptual form and no trust radius.

What the curriculum orders is the operations the model learns (Alec, 2026-11-07); the parts of speech follow from the operations, as the occupant classes of their formal arguments. An operation's arguments presuppose what earlier operations produce: an object before a predicate over objects, a clause before a question about it, a predicate before its modifiers. Once word concepts are stable, object concepts are learned through `interpret`; every word then occupies `interpret`'s one input, so all words share one class, and the clause and verb operations' arguments divide them into noun and verb classes. Questions come early, because testing comprehension teaches syntactic category: the answer to "who ran?" is a subject noun. Intransitive comes before transitive, and simple before complex as far as possible. Nouns and verbs are therefore learned first by the curriculum, not by a grammatical-class prerequisite in the learning algorithm. The architecture enforces only the coarse order, words before operations and NP + VP before other operations, through the configured authority gates, which remain data-driven: empty lift arguments report zero readiness. The finer order is the corpus presentation's, which models child development and the Piagetian stages. A kept application can record members while its resulting relation has low authority, and incidental learning from an example met before its lesson is never suppressed. The mechanism is described in [Architecture](Architecture.md#hierarchical-conceptual-context-alec-2026-11-08); demonstrating transfer and stable category distinctions belongs to the lesson evidence.

The ten lessons of `data/lessons/parts-of-speech.json`:

| lesson | operations learned | parts of speech that follow | the lesson ends when |
|---|---|---|---|
| 0 words | none | none: forms are kept and gain trust by views | a set fraction of admitted forms have a trust radius at least `definitionTrust` |
| 1 interpret | `interpret`, with `sign` beside it | one class: every word occupies the same argument | a set fraction of defined words have a denotation |
| 2 clause | `lift` (intransitive: "Bill ran.") | nouns (subject occupants) and verbs (VP occupants) | the two occupant classes are definite |
| 3 ask | `ask` with the question words ("who ran?") | interrogatives; the answers test lesson 2's classes | the answers to the held-out questions |
| 4 verb | `verb` (transitive: "Bill chased Felix.") | object nouns; whether subject and object collapse into one class is read, never declared | the object class is definite, and the answers |
| 5 modifiers | `intersection`, `product` | adjectives | the modifier class is definite |
| 6 determiners | `lower`, `definite` | determiners | the determiner class is definite |
| 7 adverb | `adverb` | adverbs | the adverb class is definite |
| 8 prepositions | `preposition`, `surface`, `compound` | prepositions and the remaining function words | the class is definite |
| 9 relations | `equal`, `part`, `whole`, `implies`, `generic`, `query`, `infer`, `isTrue` | none: a clause has no word occupant | the relation groups |

The verb lessons use names and bare plurals as subjects and objects, so that no untaught operation is needed; an article met before its lesson is a form that occupies no argument yet. A class is definite when its members fit it strictly better than the other populated arguments and it has enough views: definiteness is separability times view evidence, and operation trust is the minimum definiteness of an operation's inputs ([Architecture](Architecture.md), "Argument fit, separability and boundary conditioning").

**Readiness is a measurement.** Each lesson ends on a measurement, never on an age or a sentence count, and its threshold is data beside the corpus: the `criterion` of each stage in `data/lessons/parts-of-speech.json`, declared by Alec and not yet set. `ConceptualSpace.trust_measurements()` and `gate_measurements()` supply the values to the lesson runner (`bin/corpus_lessons.py`). Each lesson's budget is declared before it runs, and no lesson extends its budget to pass. A lesson produces an artifact that the next lesson restores as its fixture.

**Gates are behaviors (Alec, 2026-10-18).** "In writing a UG [a universal grammar], we do not demand which operators must be chosen, only which behaviors must be exhibited." No lesson gate names an operator. A lesson's gates are the behaviors its sentences exhibit: the held-out constituents, the answers to its questions, one individual per singular noun phrase with no duplicate or colliding occurrence, and the measurements its manifest declares. The kept operators, order stamps, unexpected occurrences and meaning norms are reported beside the gates, never required; whether a bare description licenses a later reference is measured by a later sentence, not demanded of a reading.

## The staged training corpus as data (item 0.5, 2026-11-12)

The training corpus of item 0 is named by one manifest,
`data/training-corpus.json`: part A (words only; simple sentences and
synthetics; math) as tracked, frozen files, and part B (the developmental
assembly, which is built locally, and the web shard, which the preparation
command downloads and verifies)
recorded by hash. Every held-out
set is listed there: the lesson held-out records, the games' validation,
test and transfer splits, the math validation, test and beyond splits, the
next-word gate and the held-out cost set. Part A's files: the ten lesson
stages in `data/lessons/parts-of-speech.json`, the games under
`data/games/` and the graded math rows under `data/math/`, each with its
manifest, generator command and seed. The design is the
[item 0.5 plan](plans/2026-11-09-item0-5-test-development.md) §6 to §9 and
§17.

Learning is judged by four measurements, each reported per epoch: the cost
on the held-out cost set (`data/eval/heldout-cost.json`, the union of part A's
validation splits); acquisition, by each lesson's criterion; answers, bound and
exact text, on the held-out questions; and each chooser's departure win rate,
the fraction of its departures whose complete path is kept over the greedy
path. Chooser gradient variance is reported, not a criterion.
