# Item 0.5: the test development pass (Alec, 2026-11-09; Claude's spec for Codex)

> **Status, 2026-11-09:** written the day Alec placed the item. §1 is his decision; §2 to §12 are Claude's filling-out; §13 lists the assumptions made; §14 holds the five decisions as they were put to him and §15 his answers of the same day, which restaged the training corpus (§8, §9.1, §9.2). No round text for Codex is in this document yet: one is written when Alec asks.

## 1. The decision (Alec, 2026-11-09)

> The transformation-games note and all other testing and harness work can move to 0.5 which becomes a test development pass that can happen on Codex, so that you can do the compiler work in 1.0 at the same time. So lets re-write the todo and the 0.5 testing spec to include existing test and harness work, as well as the generated corpus (transformation-games, the synthetic math problems corpus). 0.5 can also be responsible for creating a corpus to train the model, starting with the synthetic examples just mentioned, and then continuing to the training corpus used by NanoChat.

Read as five things:

1. A new countdown item, 0.5, between item 1 and item 0 in [the todo](../../todo.md): a test development pass, built by Codex.
2. It runs side by side with item 1, the compiler work, which Claude builds. For this pair the standing division (Codex builds, Claude specifies and reviews; todo, 2026-09-21) is set aside.
3. It takes the transformation-games note (the test development feedback of 2026-11-09) and the test and harness work that sat in items 2, 1 and 0.
4. It owns the generated corpora: the transformation games and the synthetic math problems corpus.
5. It owns the training corpus: the synthetic examples first, then the corpus NanoChat trains on.

## 2. What the item is for

Item 0 is one long run, and several things it reads do not exist yet or have never run. Item 8's task generator is not written. 6.5's six gates have no harness. The math-chain measurement waits on four corrections. The registry `test/fixtures/phase-artifacts.json` names producers for ten lessons, the lesson data `data/corpus-lessons.json` holds two stages, and all twenty phases are unregistered. Item 0.5 builds these before the run and runs each once on a small model, so that the long run is spent on learning and not on finding out that a harness does not run.

On the day item 0 starts, four things are true:

- **Harnesses.** Every measurement item 0 reads has a harness that has run once on a small model, and either its threshold is declared (the declarations are Alec's) or it is marked "recorded, not tuned".
- **Corpus.** The training corpus is data: frozen files under one manifest with hashes, every held-out set listed in one place and certified disjoint from the training documents.
- **Lessons.** Each lesson has a corpus, a criterion and one recorded run on a small model: an artifact registered, or a null recorded.
- **Suite.** The bounded suite batches safely, and the weekly tier has been green once on the hash that holds both this item and item 1.

The item changes no learner mechanism and makes no learning claim at scale. A small-model result shows that the harness runs and gives a first reading; it is never an acceptance.

## 3. The boundary with item 1

Two builders work at once, so every file has one owner.

| Owner | Files |
|---|---|
| Item 0.5 (Codex) | the test harness (`test/bounded_tests.py`, `test/test_report.py`, `test/slow_tests.py`, `test/slow_weekly.py`, `test/phase_artifacts.py`, `test/fixtures/`) and the tests this item adds; corpus and lesson data under `data/` (`data/games/`, `data/math/`, `data/eval/`, `data/corpus-lessons.json`, `data/corpus-sources.json`, the new `data/training-corpus.json`) and the small configurations this item adds; the generator and measurement modules `bin/exact.py`, `bin/math_chain_corpus.py`, `bin/MathChainTraining.py`, `bin/eval_*.py`, `bin/prepare_corpus.py`, `bin/corpus_lessons.py`, `bin/corpus_audit.py`, `bin/corpus_capacity.py` and the new `bin/game_corpus.py`; [Testing](../Testing.md) and [Training](../Training.md) |
| Item 1 (Claude) | the learner's runtime and compile path: `bin/Models.py`, `bin/Language.py`, `bin/Layers.py`, `bin/Spaces.py`, `bin/Memory.py` and the other runtime modules; the tests of the compile path that its rounds add or port, named in each round; the comparison of compiled and eager runs, the four runtime failure groups, the per-sentence timing, the three `torch._assert_async` sites and the eager hot spots of todo item 1 |
| Shared, named in the round text that touches them | `bin/data.py`, `bin/Corpus.py`, `bin/train.py`, `bin/CorpusCheckpoint.py`, `Makefile`, `data/model.xsd`, `todo.md` |

Four rules keep the two apart:

- An existing test file has one editor per round, and a round text lists the test files it edits.
- Item 0.5 edits a runtime file only to add a read-only observer call site (the logger of §5, H5, is the one known case), and the round text names the function.
- An observer never writes model state.
- A mechanism defect found by one of this item's tests keeps its strict assertion and is recorded with its evidence. That piece of work stops there and goes to Alec; the rest of the round continues.

How two writers share one repository (one checkout or two, and whose sweep runs when) is decision D1 of §14.

## 4. Rules that bind this item

1. **Math is never direct.** Numerals are opaque words, successor is the one verb, and the learner has no numeric code or readout. `bin/exact.py` is an oracle for generating and checking rows only ([mathematical thinking](../specs/2026-09-09-mathematical-thinking.md)).
2. **No language in code.** No language is named or special-cased in code, configuration, fixture names or test names. The words of a corpus are data: recipes and rows hold them, and a generator only substitutes slots.
3. **No predefined surface.** An operator has no declared surface word. A game's pairs teach its marker; nothing names it to the learner.
4. **Forcing.** A unit test may force a part of speech; a learning certificate forces nothing. A mechanism probe never satisfies a learning gate.
5. **Corpora are data** (Claude's proposal with this item). A corpus is generated once and frozen, with its hashes in a tracked manifest. A test never regenerates a corpus and never calls a network or a language model. Regenerating one is a source change, which the frozen source hash shows, because `test/bounded_tests.source_snapshot` hashes every `.json` under `data/`.
6. **Seeds.** A seed that names a generated dataset is that dataset's identity and is recorded in its manifest. No seed is chosen so that an assertion passes. A learning run is fresh and unseeded, one per arm, with no retry.
7. **Shapes.** A fixture's capacities and shapes are never resized; the input changes instead.
8. **Proportion.** Fast checks until an implementation is finished, one default sweep per landing, rounds combined.
9. **Nulls.** A null result is recorded as a null with its budget. Nothing is regraded, extended or drawn again.
10. **Receipts.** Receipts stay local under `doc/benchmarks/`, which git ignores. What must last is a tracked manifest, a registry entry or a line in a document.

## 5. The existing test and harness work, by origin

The moved todo text is under item 0.5 in [the todo](../../todo.md), unchanged. This table is the index; "from" is the item the work sat in.

| | Work | From | What item 0.5 does |
|---|---|---|---|
| H1 | Bounded-test batching | 2 | Restore safe, faster batching inside the 8 GiB cap without raising a cap or omitting a case; reconsider [fixture reuse](2026-09-17-bounded-fixture-reuse.md). |
| H2 | D5, the two decomposition snapshots | 2 | Regenerate both from their recorded seeds under the 256-wide layout, keep both old hashes beside the new ones, and run both `test_math_chain_decomposition` arms at budgets 512 and 768 with the original assertions. First check: after item 2.0 retires the decomposition teacher, confirm what each arm still measures. If an arm has nothing left to measure, that piece stops and the strike is Alec's. |
| H3 | The deferred slow marks | 2 | The 26 core, 4 tower-projection and 3 native-packing cases, both decomposition arms and `test_review17_measurement_wiring_on_one_ordinary_batch`, run once under `RUN_SLOW`. |
| H4 | The weekly sweep | 2 | `make test_all` green once on one hash, and the default sweep green at the ten-worker 28 GiB ceiling. Taken as the joint exit of items 0.5 and 1 (§12): one run on the hash that holds both, so each of item 1's four runtime groups is repaired in it or excepted by name. |
| H5 | The per-interval logger | 2 (from 4) | The readings listed in the moved text, written every $n$ batches by an option beside `checkpointEveryBatches` (`bin/Models.py`, `data/model.xsd`, [Params](../Params.md)). Read-only, reusing captured readings, with its collection cost measured and reported. Item 0's stop conditions read it. New test: `test/test_interval_log.py`. |
| H6 | The resume certificate | 2 | Extend `test/test_corpus_checkpoint.py` to the real `runBatch` on a small configuration, with the byte-identical checks listed in the moved text; one `bin/train.py` command with `checkpointEveryBatches`, run again to resume, in a receipt. Extended here to the interleaved stream of §8. |
| H7 | Two stale entries | 2 | Remove the deleted test's entry from `DEFAULT_CERTIFICATES` in `test/slow_tests.py`; confirm the 7.5 shared-operator report assertion is gone and strike the [FutureWork](../FutureWork.md) sentence that defers its verification. |
| H8 | The math-chain measurement | 0 | §7: corrections 1 and 2, the declaration, one dry run. |
| H9 | Item 8's causal study | 0 | The generator of the 256 held-out tasks with its frozen hashes, the scaffolding of the four arms (learned thought, direct answer, equal actual compute, reconstruction only), and the nested routing catalogs that include one opaque candidate. The runs and the gate are item 0's. |
| H10 | 6.5's six learning gates | 0 | The harness that does not exist: held-out anaphora against the rule-based binding at matched exposure, verb reuse, prediction against the detached-target control, shuffled order, renamed vocabulary, and mint and bind against a shuffled-determiner control. Seeds 0, 1 and 2, never tuned. |
| H11 | The held-out cost set | 0 (2.0's round) | The set behind item 2.0's held-out cost metric, drawn from the validation splits of §8's manifest and frozen by hash. |
| H12 | Occurrence §8's measurements | 0 (from 5.5) | Confirm that the six-placement harness still runs after item 2.0. Recorded, not tuned. |
| H13 | The registry and the lessons | 0, the lessons item | §9. |
| H14 | The next-word manifest | 0 (from 4) | §8's disjointness certificate covers `data/eval/nanochat_grammar_gate.json`; one scoring dry run with `bin/eval_nanochat_grammar.py` on a small model, reported without a claim. |
| H15 | Transformation games | the note of 2026-11-09 | §6. |
| H16 | The generated math corpus | 1 (added 2026-10-20) | §7. The comparison of compiled and eager runs on that corpus stays in item 1. |
| H17 | The training corpus | Alec, 2026-11-09 | §8. |

What stays where it is:

- **Item 2:** deleting the legacy paths, the powers-of-two pass, a meaning block for every configuration, and the document corrections. These change mechanism, configuration or prose, not tests.
- **Item 1:** the work listed in §3.
- **Item 0:** every run at scale; the registrations at the trained checkpoint; the capacity search and selected-prefix run of `test/test_corpus_large_slow.py`; the measured constants; the learning records carried at their original budgets, whose assertions this item does not touch; and the capture at sentence 132 with its repair at the producer.

## 6. Transformation games

Alec's note of 2026-11-09 (verbatim under item 0.5 in [the todo](../../todo.md)): teach one operation through a corpus devoted to it, so that the learner recognizes the exercise; test acquisition inside the game and transfer outside it; and use the evidence to revisit arbitrary structural restrictions such as the limit on several unary operations at one location.

### 6.1 What a game is

A game is one operation $f$ of the grammar and a corpus of pairs $(s, t)$ with $t = f(s)$, or triples $(s_1, s_2, t)$ with $t = f(s_1, s_2)$ for a binary operation. A game is named by its `rule_name` in `bin/Language.py` and a variant, never by a word of any language. Each game has its own context whole (`ContextWhole`, `bin/Teacher.py`), so the exercise is recognizable as one. A configuration holds one whole today (`architecture.data.contextWhole`), so Codex confirms by a fast check whether a lesson stage can carry its own whole; if it cannot, a game is its own small configuration.

### 6.2 Rows and files

One row per pair, in `data/games/<rule>/<variant>.jsonl`:

```json
{"game": "not", "variant": "clause", "tier": "template", "split": "train", "key": "not-clause-000017", "sources": ["felix runs."], "target": "felix does not run."}
```

`data/games/manifest.json` lists every file with its sha256, row counts per split, tier and provenance. The split is by content: the sha256 of the first source's normalized text, modulo 10, gives train (0 to 7), validation (8) and test (9), so a source sentence keeps its split in every game it appears in. A fourth split, `transfer`, holds pairs whose fillers come from the recipe's transfer list: words the learner knows from ordinary lessons and never meets inside the game.

### 6.3 Two tiers

- **Template tier.** `bin/game_corpus.py` (`expand(recipe, *, seed)`, `verify(rows, recipe)`) expands a recipe `data/games/recipes/<rule>.json`. A recipe holds frames with slots and filler entries, and each filler entry lists every form its frames need, so the code substitutes slots and knows nothing about any language (§4 rule 2). The pairs are correct by construction.
- **Written tier.** Varied pairs written once by a language model (Alec's note) and frozen. The manifest records the model identifier, the date, the sha256 of the prompt, the counts `verify` accepted and rejected, and the keys of a sampled audit list that a person or the reviewer read. Who writes and audits this tier is decision D3.

Both tiers are frozen rows. A test reads the manifest and the rows and nothing else (§4 rule 5).

### 6.4 Presentation and training signal

A game document is the source sentence or sentences, then the target. The item adds no training signal, so the target is taught by one that exists. There are two candidates:

- (a) next-sentence production: `WhatQuestion.future(where, 1)` under `<whatCurriculum>` (`bin/What.py`). Its learning gate is still open in item 0's records (the future-label arm did not improve in 30 updates).
- (b) a lesson record whose `answer` is the target, through `bin/corpus_lessons.py`. Answers there are single words today; a whole-sentence answer is untested.

The first fast check of the games round establishes which of the two trains whole-sentence production on a single pair repeated. Codex uses that one and records the choice. If neither does, the games work stops at that finding, which is a mechanism question for Alec; the corpora are still built and frozen. No cue word is declared for a game (§4 rule 3): the context whole and the pairs carry it.

### 6.5 Measurements

With $\hat t_i$ the sentence the learner produces for source $s_i$, over $n$ held-out pairs of game $g$:

- **Acquisition**, on the `test` split: $A_\text{prod}(g) = \frac{1}{n}\sum_{i=1}^{n} [\hat t_i = t_i]$.
- **Vocabulary transfer**, the same count on the `transfer` split: $T_\text{vocab}(g)$, and the carry-over $C(g) = T_\text{vocab}(g) / A_\text{prod}(g)$.
- **Context transfer**, outside the game's context whole: ordinary lesson records whose answer depends on the operation (for example "Felix does not run. Ciao runs. Who runs?", answered "Ciao"). $T_\text{context}(g) = a_\text{with} - a_\text{without}$, the accuracy on those records of a model that had the game against one that did not.
- **Composition**, for two learned games $g$ and $h$: $T_\text{compose}(g, h)$, the acquisition count on pairs that need both.

Two probes are reported beside them and never satisfy a gate (§4 rule 4): $A_\text{read}$, the fraction of targets whose kept program applies the game's rule, read by `rule_name`; and $A_\text{mean}$, the agreement of the target's selected meaning with the rule's layer `forward` applied to the source's meaning (for `not`, `NotLayer.forward`).

Each game runs three arms: enabled; ablated (the rule removed from the configuration's grammar file, a change of data only; if a rule cannot be removed by data the arm is dropped and the report says so); and shuffled pairing (each source paired with another source's target). One fresh unseeded run per arm, no retry, the budget declared in the manifest before the run (the gate form of [Training](../Training.md)). The bars are Alec's to declare; until he does, the readings are recorded, not tuned.

### 6.6 Inventory

Candidates, in the lesson order of [parts of speech §5b](../specs/2026-10-18-parts-of-speech.md), each confirmed against `GRAMMAR_LAYER_CLASSES` and `_OPERATOR_SURFACE_SCHEMAS` before its recipe is written: `not`, `non`; `conjunction`, `disjunction`; `intersection`, `product`; `lower_unary`, `definite_unary`; `adverb`; `transpose`; `part`, `whole`; `implies`; `generic`; and the question and equality rules under the names the code has. The first round builds three: `not`, `conjunction`, and the unary-chain game of §6.7 (D3).

### 6.7 Unary chains

The note asks that the restriction on several unary operations at one location be revisited as lessons supply evidence. Today `UnaryHistory.penalty` charges the $k$-th unary application at one operand $(k-1)^2$ logit units, and generation stops a run at `unary_run < R1` (`bin/Models.py`). The chain game has targets that need $k = 1, 2, 3$ unary operations at one operand (for example a negated, determined noun phrase), and reports $A_\text{prod}(g; k)$ for each $k$ under the present penalty and bound. That is the evidence. An arm with the penalty off needs a configuration option in a learner file, which is not this item's to add; it waits on Alec's word after the first readings.

### 6.8 Games as lessons

A game enters training as a stage of the lesson manifest: a `source` reference to its file in `data/games/` and a `criterion` (§9), producing a phase artifact in the registry when its criterion is met.

## 7. The synthetic math problems corpus

The generator exists (`bin/exact.py`, `bin/math_chain_corpus.py`, `bin/MathChainTraining.py`); the rows are generated at test time and nothing is frozen. Item 0.5 makes the corpus data and prepares its measurement.

- **Frozen rows.** `data/math/grade-<nn>.jsonl`, one serialized `ChainDocument` per row (`key`, `sentences`, `question`, `answer`, `pair`, `steps`, `statement_references`), with `data/math/manifest.json` holding the generator command, the seed as the corpus's identity, the sha256 of each file and the counts. New test: `test/test_math_corpus_frozen.py` (the files match the manifest; the oracle `ExactVerifier` accepts every row).
- **Grades.** Counting first, then one-step problems, then chains by depth (correction 1 of the 2026-10-08 protocol; the Viterbi order of [thinking §13.9](../specs/2026-10-07-thinking.md)), with distractors as their own grade.
- **Splits.** Both `split_by_surface` and `split_by_structure`, plus the protocol's held-out pairs and beyond-range pairs.
- **The four corrections** (`protocol-corrections-14-13.json`, deferred to item 0). Corrections 1 and 2 are this item's: the grade order above, and worked steps presented as intermediate answers. If correction 2 needs a new term in an objective, it is mechanism and goes to Alec. Correction 3 is what the measurement reads (the episode binds a found candidate or concludes with the best match). Correction 4, the budget counted in steps, changes the work meter; it is mechanism, outside this item, and it precedes the measurement.
- **The declaration.** The measurement is declared again after item 2.0 retires the decomposition teacher and books the choice reward to output: the three conditions, the bar of 9 of 10 trainings and the chain bar of the protocol, with Alec's word on any number that changes. One dry run on the smallest grade shows that it runs.
- **Item 1's use.** The frozen rows are the input of item 1's comparison of compiled and eager runs, identical by construction on both sides. Each row's throughput is recorded per run.
- Math is never direct (§4 rule 1): the rows are sentences, and nothing numeric reaches the learner.

## 8. The training corpus

One manifest, `data/training-corpus.json`, names every stage of the training corpus in order, with each stage's files, hashes, counts and split rule. Item 0 trains from this manifest and from nothing else. The staging is Alec's of 2026-11-09 (§15): three small stages that can be analyzed operator by operator, a saved model with tests after each, and only then the large corpus.

**Part A: the analyzable stages.**

| | Stage | What it holds | Source | State today |
|---|---|---|---|---|
| A1 | Words only | lessons 0 and 1 of [parts of speech §5b](../specs/2026-10-18-parts-of-speech.md) (`words`, `interpret`): forms kept and trusted, each word given its object-concept, no operation on two words | `data/corpus-lessons.json` | no stage data (§9) |
| A2 | Simple sentences and synthetics | lessons 2 to 9 of §5b (`clause`, `ask`, `verb`, `modifiers`, `determiners`, `adverb`, `prepositions`, `relations`) and the transformation games | `data/corpus-lessons.json`, `data/games/` | two lesson stages exist; the games are new (§6) |
| A3 | Math | the graded math corpus | `data/math/` | generator exists, rows not frozen (§7) |

- **One model, three stages, in order.** The model is saved before A1 and after each stage: $M_0$, $M_{A1}$, $M_{A2}$, $M_{A3}$.
- **Read operator by operator.** After each stage the operator table of §9.1 is written from the saved models. Whether everything has been learned to some degree is read from that table.
- **Saved, with tests.** The three trained models are registered and tested (§9.2).
- **The gate.** Part B does not start until Alec has read the table after A3 and says so. A rule with no rise is named in the question put to him.

**Part B: the large corpus, after the gate.**

| | Stage | Source | State today |
|---|---|---|---|
| B1 | Developmental text | the assembly `data/corpus/assembly-01/` (local; 57,210 documents, 3,421,205 sentences), recipe `data/corpus-sources.json` | built |
| B2 | NanoChat's base corpus | `karpathy/fineweb-edu-100b-shuffle` through `bin/data.py` | one shard of 1,822 on disk; it is already the last stage of the assembly |
| B3 | NanoChat's chat-stage mixture | SmolTalk, MMLU `auxiliary_train`, GSM8K `main`, the spelling tasks, the identity conversations | not prepared, and not prepared in this item unless Alec says |

- **Data only.** Item 0.5 prepares B1 and B2 as data: manifest entries, hashes and the held-out list. No training on part B happens in this item.
- **Interleaving.** During part B the documents of part A may be interleaved at a fraction Alec declares at the gate (zero is allowed), placed by document index and not by a random draw, so a resumed run reads the identical stream (H6 certifies this). The interleaving is a change to `CurriculumStreamDataset` in `bin/Corpus.py`, a shared file, named in its round.
- **Size.** The shard count of B2 is set from item 1's measured sentences per second and Alec's time budget; the floor is item 0's 1,000,000 sentences, which the shard on disk already covers.
- **Held-out sets, in one list.** The manifest lists every held-out set: each stage's validation and test splits, the next-word manifest `data/eval/nanochat_grammar_gate.json`, the games' `test` and `transfer` splits, the math held-out and beyond-range pairs, item 8's 256 tasks and the held-out cost set (H11). New test in the default tier, `test/test_training_corpus_manifest.py`: no held-out item appears in a training file, by key and by content hash, read from the manifests without loading a corpus.
- **Storage.** Large data stays out of git (`data/fineweb/`, `data/corpus/`). Manifests with hashes are tracked, so they enter the frozen source hash. Fetching a shard is a preparation step, run once outside any test.

## 9. Lessons and registered artifacts

The state today: `data/corpus-lessons.json` holds two stages, `ideas` and `relations`, and a final round. The registry `test/fixtures/phase-artifacts.json` declares producers for ten lesson phases (`words`, `interpret`, `clause`, `ask`, `verb`, `modifiers`, `determiners`, `adverb`, `prepositions`, `relations`) by reference to the table of parts of speech §5b, and no stage data exists for them. All twenty phases are unregistered, so every trained-tier test skips.

Item 0.5 builds the test-development half of what the documents call the lessons item:

- stage data for the ten rows, small and from a declared vocabulary;
- a `criterion` field per stage, read by `bin/corpus_lessons.py` after each epoch, so a lesson ends on its criterion and not on an epoch count;
- restoration of a registered artifact into the model of a trained-tier test, which today trains a fresh model after the prerequisite is checked ([Testing](../Testing.md));
- two tiny trained artifacts registered first, `xor_grammar` and `mm_xor_grammar`, whose producers and criteria are already in the registry;
- the two class measurements read before a lesson is said to have taught a part of speech (a word's class follows its participations; the collapse table of parts of speech §2b beside the argument radii).

One recorded run per lesson on a small model: an artifact registered if the criterion is met, a null recorded if not. The lessons item's mechanism questions (finer gate progression, whether the chooser reads trust, the grammar-XOR runs that do not converge) stay in [FutureWork](../FutureWork.md) and are Alec's.

### 9.1 The operator table (Alec, 2026-11-09)

Alec: "All of that is analyzable in terms of what effect learning is happening on particular operators, and we can in general make sure that everything gets learned 'to some degree'." The table is the instrument. It has one row per rule name of `TheGrammar.rules_upward` and one group of columns per saved model of part A ($M_0$, $M_{A1}$, $M_{A2}$, $M_{A3}$; §8). Each cell holds three readings, taken with the model in evaluation and nothing written to it:

- **Use**, $n(r)$: how many times rule $r$ is chosen in the readings of one fixed validation set, the union of the three stages' validation splits, the same documents for every model.
- **Change**, $\Delta(r) = \lVert \theta_r - \theta_r^{0} \rVert / \lVert \theta_r^{0} \rVert$, over the parameters of the rule's own layer in `GRAMMAR_LAYER_CLASSES`, against $M_0$. A rule with no parameters of its own shows none.
- **Probe**, $A_\text{prod}(g_r)$: production accuracy on the `test` split of the rule's game (§6.5), evaluated without training. A rule whose game is not built shows no probe.

New module `bin/eval_operators.py`, this item's (§3): `operator_table(models, manifest)` returns the rows, and the stage run writes the table beside each saved model. The call sites that read the chosen rules and the layer parameters are read-only observers, named by function in the round text (§3).

How the table is read:

- *Learned to some degree:* after A3 the rule's probe is above its probe at $M_0$; where there is no probe, its use and its change are both above zero. This is Claude's reading of "to some degree" (§13, assumption 9). No number is tuned, and Alec may declare a floor.
- *Which stage taught it:* the stage across which the probe rises.
- *What a later stage cost it:* a fall of the probe from one saved model to the next is listed by rule name and never averaged away.
- The table is a reading on one fresh, unseeded run. It satisfies no learning gate of item 0 (§4). A rule with no rise is recorded as a null and named in the gate question of §8.

The games of §6.6 beyond the first three are built at the template tier in the third round, so every rule with a surface has a probe before the gate.

### 9.2 Saved models and their tests (Alec, 2026-11-09)

Alec: "We can also save off models from this stage and add testing for them."

- **Saved.** The stage run saves the model before A1 and after each stage. The three trained ones are registered in `test/fixtures/phase-artifacts.json` as phases `stage_words`, `stage_sentences` and `stage_math`, each with its producer (the stage run's command, the source hash, the hash of `data/training-corpus.json`) and its criterion: every lesson criterion of the stage, and for A3 the criterion declared with the math measurement (§7). The model files stay local; the registry entries and their hashes are tracked.
- **Tests, in the trained tier.** New file `test/test_stage_models.py`, skipped while a phase is unregistered. For each registered model: it restores into a model built from its recorded configuration (the restoration of §9); its file matches the registered hash; its own stage's criteria hold on the validation split; its columns of the operator table reproduce from the file. Retention, the earlier stages' criteria read on the later models, is recorded and not asserted until Alec declares a bar.
- **Which model continues (Alec, 2026-11-09, §15.1).** The experiments of part A, with their table, their saved models and their tests, run on a small model: the configuration the lesson runs use today (`data/MM_corpus_lessons.xml`). The full-size model is kept for the large training. Item 0 opens that training with the same three stages at the long run's configuration, because a model cannot change size between part A and part B, and continues into part B from its own model saved after A3. It writes the table once more from those models, and the gate of §8 is read there, with the small run's table beside it.
- **Math may need more.** Alec: the full-size model "may be necessary for the mathematical corpus". The first fast check of the third round reads whether A3 fits the small configuration. If it does not, the math stage runs on a second, larger model that is first taken through A1 and A2, whose analysis stays the small model's; the larger model uses an existing configuration, never a resized one, and Alec is told before the full-size one is used.

## 10. What item 0 reads

| Measurement of item 0 | Harness | Thresholds | In item 0.5 |
|---|---|---|---|
| The five `fineweb` certificates | exists | exposure of 1,000,000 sentences | nothing new; confirm that a missing exposure skips and never passes |
| The causal gate (from 9 and 8) | H9 | arms and thresholds declared by Alec | generator, hashes, arms scaffolding, one small run per arm |
| 6.5's six gates | H10 | declared by Alec, seeds 0, 1, 2 | the harness, one small run |
| Occurrence §8 | H12 | recorded, not tuned | confirm it runs |
| Next-word scoring (from 4) | H14 | recorded, not tuned | disjointness, one dry run |
| Math chain | H8, §7 | the protocol's bars, declared again | corpus, corrections 1 and 2, one dry run |
| Games | §6 | declared by Alec | corpora, three games on a small model |
| Lesson criteria | §9 | the `criterion` fields | stage data, one run per lesson |
| The operator table, and the gate before the large corpus | §9.1 | read by Alec; no number tuned | the module, and one table from the small three-stage run |
| Saved stage models | §9.2 | each stage's criteria | three registered small models and `test/test_stage_models.py` |
| Held-out cost (2.0) | H11 | recorded | the frozen set |
| Stop conditions | H5 | item 0's | the logger |
| Resume | H6 | byte-identical | the certificate |
| Capacity search (from 2.7) | exists | item 0's | nothing; it runs in item 0 |

## 11. Rounds

Three combined rounds, each one Codex text written when Alec asks, each with fast checks until its implementation is finished and one default sweep at its landing:

1. **Harness.** H1 to H7, H11, H12 and H14.
2. **Corpora.** §6 (three games), §7, the lesson stage data of §9 and the manifest of §8 (part A's files; part B as data), with their certificates.
3. **Lessons and stage models.** The lesson criteria and the artifact restoration of §9, the remaining games at the template tier, the three-stage run on the small configuration, the operator table (§9.1), the three registered models and their tests (§9.2), H9 and H10.

The weekly sweep of H4 runs once, at the joint exit with item 1.

## 12. Exit

- Every row of §10 has its harness and one recorded small-model run, with its thresholds declared or marked "recorded, not tuned".
- `data/training-corpus.json` is complete, part A's three stages as files and part B as data (§8), and its disjointness certificate passes in the default tier.
- The three stages have run once on the small configuration; the operator table (§9.1) has been put to Alec; each of the three saved models is registered or its null recorded, and `test/test_stage_models.py` passes in the trained tier for each registered one.
- Each lesson of §9 and each built game has a registered artifact or a recorded null.
- `make test_all` is green once, and the default sweep is green at the ten-worker 28 GiB ceiling, on one hash that holds items 0.5 and 1.
- [Testing](../Testing.md) and [Training](../Training.md) describe the corpora, the games and the registry as built.

## 13. Assumptions (overturn any)

1. Item 0.5 is the test-development half of the lessons item; the lessons item's mechanism questions stay in FutureWork.
2. Two borderline runtime items stay in item 1: the nested-admission timing and the three `torch._assert_async` sites.
3. The todo rewrite moves text and deletes none. The dated log at the top of the countdown stays where it is; it can move to [DevelopmentHistory](../DevelopmentHistory.md) later, on your word.
4. A mechanism defect found by item 0.5 goes to you with its strict assertion kept (§3).
5. Math-chain correction 4 is mechanism, built outside this item before the measurement is declared.
6. The unary penalty-off arm waits on your word (§6.7).
7. Game and math rows are tracked in git, since they are small; the natural corpora stay local.
8. *Added with §15.* "Words only" is lessons 0 and 1 of parts of speech §5b; "simple sentences and synthetics" is lessons 2 to 9 with the transformation games; "math" is the graded math corpus (§8).
9. "To some degree" is read from the operator table as a rise over the untrained model, with no tuned number (§9.1).
10. The stage models of this item are small rehearsals; item 0 runs the three stages again at the long run's configuration and continues into the large corpus from its own third saved model (§9.2). *Decided by Alec the same day (§15.1).*
11. "NanoChat and others" is the developmental assembly and NanoChat's base corpus; the chat-stage mixture is not prepared in this item (§8).
12. The games beyond the first three are built at the template tier in the third round, so every rule with a surface has a probe before the gate (§9.1).

## 14. Decisions for Alec

> **Answered 2026-11-09 (§15).** The five decisions are kept below as they were put. Two things differ from this text: D1's recommendation was revised in conversation to a detached worktree with no branch, and D4 was decided differently from its recommendation.

- **D1. Two writers, one repository.** Codex freezes source by a hash over `bin/`, `test/` and `data/`, so any edit of mine inside a Codex round breaks its freeze, and two default sweeps cannot run at once at the ten-worker 28 GiB ceiling. The rule of 2026-11-03 is main only, with no sandbox branches. *Recommendation:* for this pair only, a second git worktree and branch for item 1, merged by you at each landing; sweeps take turns. *Alternative:* one checkout, the file ownership of §3, freezes that alternate.
- **D2. Who reviews.** *Recommendation:* Codex reviews my compiler rounds by the method I use on its rounds (hash, patch, certificates, one sweep); I review item 0.5's rounds briefly.
- **D3. The written tier and the first games.** *Recommendation:* Codex writes and verifies the written-tier pairs inside its round, frozen with the provenance of §6.3, and I read the sampled audit list at review; the first three games are `not`, `conjunction` and the unary-chain game.
- **D4. How far the corpus goes.** Which NanoChat stages item 0.5 prepares, and at what size. *Recommendation:* stages 1 to 5, with stage 5's shard count set after item 1 measures throughput; stage 6 only after the base run, and without GSM8K until you say how digit-heavy word problems sit with the rule that math is never direct. The developmental assembly stays as stage 4, before the NanoChat base corpus.
- **D5. Order, and who builds item 1.5.** Item 1.5 precedes item 1 by your word of 2026-10-30, and both it and item 2's remaining housekeeping change the runtime files that item 1 works in. *Recommendation:* Codex finishes item 2.0, then the shrunk item 2; then the pair starts, with item 1.5 on my side ahead of item 1, while Codex is on item 0.5. *Alternative:* Codex builds item 1.5 before the pair starts. With either, the declarations that the todo collects "at the start of item 0" (item 8's arms and thresholds, 6.5's gates, the held-out set) and the game bars are needed before item 0.5's third round, not at the start of item 0.

## 15. Alec's answers (2026-11-09)

After a discussion of D1 and D5, Alec:

> D4 is my only point of contention.
>
> We should have the simple corpus (training on words only). then training one simple sentences and synthetics. then training on math. All of that is analyzable in terms of what effect learning is happening on particular operators, and we can in general make sure that everything gets learned "to some degree". We can also save off models from this stage and add testing for them.
>
> It is only at that point that we unleash training on the million-plus sentence corpus of Nonochat and others.

Recorded as follows. "My only point of contention" is read as acceptance of D1, D2, D3 and D5 as last recommended. Where the discussion left a question without its own answer, the reading is marked, and it is his to overturn.

- **D1, decided: a second worktree, detached, no branch.** This is the recommendation as revised in the discussion. Item 1 is built in a second worktree of this repository at a detached commit, so main stays the only branch (the rule of 2026-11-03). Claude's work stays uncommitted there, and Alec commits it onto main at each landing. Codex's freeze in the main checkout is untouched by it. The machine has 36 GiB and the default sweep's ceiling is 28 GiB, so sweeps and item 1's timing runs take turns: "at the same time" means editing and fast checks in parallel. *Reading:* "is a detached worktree acceptable?" had no answer of its own.
- **D2, decided as recommended.** Codex reviews Claude's compiler rounds by the method Claude uses on its rounds (hash, patch, certificates, one sweep); Claude reviews item 0.5's rounds briefly.
- **D3, decided as recommended.** Codex writes and verifies the written-tier pairs inside its round, frozen with the provenance of §6.3; Claude reads the sampled audit list at review. The first three games are `not`, `conjunction` and the unary-chain game.
- **D4, decided by Alec, and not as recommended.** The corpus is staged: words only; then simple sentences and synthetics; then math. Each stage is analyzed for its effect on particular operators, so that everything is learned to some degree, and models are saved from these stages and tested. Only then does training start on the million-plus sentence corpora. This is built into §8 (parts A and B and the gate between them), §9.1 (the operator table) and §9.2 (the saved models and their tests). The recommendation it replaces taught the lessons, games and math first and then interleaved them into the large corpus, with no gate, no reading per operator and no saved models. Claude's readings inside his decision are assumptions 8 to 12 of §13; the one that most needs his word is 10, which model continues into the large corpus.
- **D5, decided as recommended in the discussion.**

| | Codex | Claude |
|---|---|---|
| now | item 2.0, rounds 5 and 6 | the reviews of those rounds |
| then | item 0.5, rounds 1 and 2, which do not depend on the runtime files | item 1's baseline and plan, read-only, when the machine is quiet; then item 1.5 |
| then | item 2's remaining housekeeping, in a gap between rounds | item 1 |
| then | item 0.5, round 3, on a hash that holds item 1.5 (Claude's expectation, untested: the lesson runs and the D5 snapshots should not be made twice) | item 1, continued |
| exit | one weekly sweep on the hash that holds both items | |

*Readings:* item 1.5 is on Claude's side, and item 2's remainder goes in a gap; neither had an answer of its own. The declarations that the todo collects "at the start of item 0" (item 8's arms and thresholds, 6.5's gates, the held-out set) and the game bars are needed before item 0.5's third round.

### 15.1 Which model continues (Alec, later the same day)

Assumption 10 was put to him as three arrangements: (1) rehearse the three stages on a small model in item 0.5 and run them again at full size as the opening of item 0; (2) run them once at full size and continue from those saved models; (3) keep the saved models for study and tests only and start the long run fresh. Alec:

> lets do 1.
> the full-size model (on this machine) should only be used for the large dataset training; it is overkill for the small synthetic experiments (although it may be necessary for the mathematical corpus).

Recorded in §9.2: the experiments, the operator table, the saved models and their tests are the small model's; the full-size model is for the large training, which opens with the three stages; the math stage may take a larger model. How the larger model is reached (a second model taken through A1 and A2 first) is Claude's filling-out.
