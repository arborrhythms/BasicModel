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

1. **Math is never direct.** Numerals are opaque words, successor is the one verb, and the learner has no numeric code or readout. `bin/exact.py` is an oracle for generating and checking rows only.
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
| H1 | Bounded-test batching | 2 | Restore safe, faster batching inside the 8 GiB cap without raising a cap or omitting a case; reconsider fixture reuse: test helpers build a fresh model at every `from_config` call, so one compact preallocated model per compatible fixture family in a bounded worker, with a verified complete reset, could serve the tests that do not need fresh construction; construction, reset, compilation, forward and backward are timed separately before any speedup is claimed. |
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

Candidates, in the order of the ten lessons (`data/lessons/recipes/00-words.json` to `09-relations.json`), each confirmed against `GRAMMAR_LAYER_CLASSES` and `_OPERATOR_SURFACE_SCHEMAS` before its recipe is written: `not`, `non`; `conjunction`, `disjunction`; `intersection`, `product`; `lower_unary`, `definite_unary`; `adverb`; `transpose`; `part`, `whole`; `implies`; `generic`; and the question and equality rules under the names the code has. The first round builds three: `not`, `conjunction`, and the unary-chain game of §6.7 (D3).

### 6.7 Unary chains

The note asks that the restriction on several unary operations at one location be revisited as lessons supply evidence. Today `UnaryHistory.penalty` charges the $k$-th unary application at one operand $(k-1)^2$ logit units, and generation stops a run at `unary_run < R1` (`bin/Models.py`). The chain game has targets that need $k = 1, 2, 3$ unary operations at one operand (for example a negated, determined noun phrase), and reports $A_\text{prod}(g; k)$ for each $k$ under the present penalty and bound. That is the evidence. An arm with the penalty off needs a configuration option in a learner file, which is not this item's to add; it waits on Alec's word after the first readings.

### 6.8 Games as lessons

A game enters training as a stage of the lesson manifest: a `source` reference to its file in `data/games/` and a `criterion` (§9), producing a phase artifact in the registry when its criterion is met.

## 7. The synthetic math problems corpus

The generator exists (`bin/exact.py`, `bin/math_chain_corpus.py`, `bin/MathChainTraining.py`); the rows are generated at test time and nothing is frozen. Item 0.5 makes the corpus data and prepares its measurement.

- **Frozen rows.** `data/math/grade-<nn>.jsonl`, one serialized `ChainDocument` per row (`key`, `sentences`, `question`, `answer`, `pair`, `steps`, `statement_references`), with `data/math/manifest.json` holding the generator command, the seed as the corpus's identity, the sha256 of each file and the counts. New test: `test/test_math_corpus_frozen.py` (the files match the manifest; the oracle `ExactVerifier` accepts every row).
- **Grades.** Counting first, then one-step problems, then chains by depth (correction 1 of the 2026-10-08 protocol: with one departure per derivation, credit reaches deep derivations only when the problems arrive graded, the last step alone and then the last two, and not when depths are mixed), with distractors as their own grade.
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
| A1 | Words only | lessons 0 and 1 (`words`, `interpret`): forms kept and trusted, each word given its object-concept, no operation on two words | `data/corpus-lessons.json` | no stage data (§9) |
| A2 | Simple sentences and synthetics | lessons 2 to 9 (`clause`, `ask`, `verb`, `modifiers`, `determiners`, `adverb`, `prepositions`, `relations`) and the transformation games | `data/corpus-lessons.json`, `data/games/` | two lesson stages exist; the games are new (§6) |
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

## 16. Reconciled with items 2.0 and 1.8 (Claude, 2026-11-12)

Alec, 2026-11-11: "Let's not worry about who does 1.5 , 1, 0.5; we just
want to work in parallel, with you doing most of the spec work." Asked
whether Claude could start item 1.5, 1 or 0.5 while Codex completes item
2.0 and builds item 1.8, and told that 1.5 and 1 optimize and compile the
code 1.8 rewrites while rounds 1 and 2 here do not depend on the runtime:
"Wait for the 2.0 push." Codex pushed the whole-path landing
(`b35a4bd316`, recorded as `d88acd3fab`). Claude builds rounds 1 and 2 in
a second worktree, `basicmodel-item0-5`, detached at `d88acd3fab` with no
branch (D1), and leaves the work uncommitted for Alec.

What changed since this plan was written, and what follows here:

1. *The runtime moves under 1.8.* Item 1.8
   ([specification](2026-11-11-item1-8-last-architecture-round.md))
   changes the memory columns and recall, the sentence loop's operators,
   the output read and the numeric input. The harness items that run the
   model are therefore built after 1.8 lands, so that each is run once on
   the code item 0 will use: H2, H3, H5 (its call site is in
   `bin/Models.py`, which 1.8 edits), H6, H12 and H14's scoring dry run.
   Round 1 now: H1, H7, H11, H14's disjointness half and H18 below; H4
   stays the joint exit.
2. *H2.* The decomposition teacher was retired in item 2.5 (decision (a),
   2026-11-04). H2's own first check stands: what each arm still measures
   is read before anything is regenerated, and an arm with nothing left
   to measure goes to Alec.
3. *H18, new: the source hash counts a git-ignored file.* The receipts'
   method (`test/bounded_tests.source_snapshot`) includes the local
   `data/corpus/assembly-01/manifest.json`, which git ignores, so a clean
   checkout of a landed commit cannot reproduce its receipt's hash: the
   whole-path round returned `26dd5a0bda60` over 1,051 files and its
   landing hashes to `ad3efc4a71b7` over 1,050, differing only in that
   file. Two fixes: leave git-ignored files out of the snapshot, or track
   that manifest. Either changes the method, so it takes effect at a
   clean point (after 1.8's landing), never inside a round in flight.
   Claude's default, for Alec: leave ignored files out.
4. *Games.* 1.8 renames the field rules and makes `not` a field operator;
   `conjunction` stays a vector operator. The first three games keep their
   rule names (`not`, `conjunction`, and the unary chain), which exist
   before and after 1.8. Their recipes, rows and manifest are built now;
   §6.4's check of which training signal teaches a whole-sentence target
   runs the model and waits for 1.8. The written tier was to be written by
   Codex (D3); with Claude building, Claude writes it and Codex, or Alec,
   reads the audit list.
5. *Math.* Unchanged. "Math is never direct" stands beside 1.8's scalar,
   whose grounding of number words is a lesson of its own (1.8 §3.10); the
   rows here stay sentences.
6. *Lessons.* The stage data of the ten lessons is built now. Their
   criteria are read by `bin/corpus_lessons.py`, a runtime path, and their
   runs belong to round 3, after 1.5.
7. *The two tiny artifacts of §9* (`xor_grammar`, `mm_xor_grammar`) wait:
   1.8 moves every XOR learning certificate to numbers (1.8 §3.9).
8. *Review.* D2 had Codex review Claude's compiler rounds; the same method
   (hash, patch, certificates) is proposed for these rounds, through Alec.

## 17. Rounds 1 and 2 as built (Claude, 2026-11-12; on the branch `testing`, §18)

*Round 2, the corpora.* Each is data under a tracked manifest with its
generator command and seed (the seed is the corpus's identity, §4 rule 6),
and a fast certificate that reads only the frozen files (§4 rule 5).

- *Math* (§7): `bin/math_chain_corpus.py freeze --seed 20261112`, nine
  grades in `data/math/grade-00.jsonl` to `grade-08.jsonl` (828 rows),
  in correction 1's order: counting, identity, alias, plus zero, plus one,
  plus two to ten, the same with two distractors, depth two, depth three
  (test only, deeper than any trained chain). Held-out pairs only in
  validation and test, beyond pairs only in `beyond`, the other grades
  split by `exact.split_by_surface`'s rule. Correction 2 is data: each
  worked step is a question whose answer is supplied
  (`ChainDocument.questions`, read by `ChainDocument.target`), so the
  existing answer supervision scores it and no objective term is added.
  Certificate: `test/test_math_corpus.py`, which checks every answer and
  intermediate answer with its own evaluator.
- *Games* (§6): `bin/game_corpus.py` (`expand`, `verify`, `content_split`),
  with no word of any language in it; recipes for `not` (clause, copula),
  `conjunction` (subjects, predicates) and `unary_chain` (k = 1, 2, 3: the
  plan's own example, a negated, determined noun phrase, intended through
  `definite_unary`, `lower_unary` and `not`). Template tier only; the
  written tier is not built. In-game fillers are words no lesson uses;
  transfer fillers are lesson words. Certificate: `test/test_game_corpus.py`.
- *Lessons* (§9): the ten phases' stage data in
  `data/lessons/parts-of-speech.json` (`bin/lesson_stages.py`, reusing the
  games' expander), in the lesson driver's format, so `stage_documents`
  reads each stage; held-out records apart, at most a quarter of a
  variant's training records. Each stage's criterion names its
  measurement from parts of speech §5b with the threshold left null for
  Alec. `data/corpus-lessons.json` is unchanged, because item 1.8's
  measurement reuses its arm; §8's A1 and A2 read the new file.
  Certificate: `test/test_lesson_stages.py`.
- *The manifest* (§8): `data/training-corpus.json` (`bin/training_corpus.py
  freeze`), stages A1 to B3 in order with hashes, the held-out list, and
  the one scan of the developmental assembly for sentences that occur only
  in held-out items: 953 such sentences, 5 occurrences in 3,421,205
  sentences, all from lesson 0's held-out single words; recorded, not
  required to be zero, since part B is read after the gate.
  Certificate: `test/test_training_corpus.py`.

*Round 1, the harness.* Built: H7 (the deleted test's entry removed from
`DEFAULT_CERTIFICATES`; no test asserts the 7.5 report any more, recorded
in FutureWork), H11 (`data/eval/heldout-cost.json`, the union of part A's
validation splits, 191 items, hashed in the manifest; the metric itself is
item 0's to declare) and H14's disjointness half (every next-word gate item
is a test document under the shard's split rule). Not built, with the
reason: H1 is measured only by full sweeps, which cannot share the machine
with Codex's; H2, H3, H5, H6, H12 and H14's dry run run the model and
follow item 1.8 (§16 item 1); H4 is the joint exit.

*Findings, for Alec* (H20 is of the same kind as H18: a clean checkout
is not the measured one):

- *H19, withdrawn (Claude, later on 2026-11-12).* It was recorded as a
  test downloading the web shard into the fresh worktree. That was a
  misreading: `data/fineweb/shard_00000.parquet` (94,511,680 bytes) has
  been tracked in git since `b917793f37`, the worktree's copy carries the
  checkout's time to the second, and `test/test_math_chain.py` does not
  touch it. The `local` flag of stage B2 in `data/training-corpus.json` is
  therefore half right (that shard is tracked; later shards would be
  fetched); it stays until the manifest is next built.
- *H20.* `test/test_doc_links.py` fails 58 cases in a clean checkout,
  because the documents link to local, git-ignored material: receipts
  under `doc/benchmarks/`, old results under `output/tests/` and one
  checkpoint folder. Proposed: the test resolves only tracked targets and
  reports links into ignored paths as local, not as broken.

## 18. A second machine (Alec, 2026-11-12)

### 18.1 Decided

Alec, 2026-11-12: "I will actually use codex on the other machine to do
the 0.5 work. You can remain supervisor. (For both 1.8 and 0.5)". Then,
by question tool:

- *Where the work goes.* "Give me text to paste. It will have to pull 2.0
  from GitHub and work on its own branch", and for Claude's rounds 1 and
  2: "Push, but on branch 'testing'". Rounds 1 and 2 are one commit on top
  of `9f7923d67f` on the branch `testing`; Codex continues there and
  returns there; Alec lands it.
- *The model as it is in 2.0.* Asked whether the items that run the model
  wait for 1.8: "You can run the model as it exists in 2.0, can't you?",
  then "Yes, run them on 2.0 now". This replaces §16 item 1: H2, H3, H5,
  H6, H12, H14's dry run and §6.4's check run on 2.0's code, and after 1.8
  lands only the numbers item 0 reads are measured again. H8's dry run
  (§7) is of the same kind and joins them (Claude's reading).
- *H20.* "doc links to [git-ignored] receipts should have the relevant
  portions of those receipts inlined immediately. That can happen here,
  now, and be pushed on main." Amended the same day: "Actually, all work
  on testing, please. Then push.", "I'm suggesting that receipts are
  irrelevant" and "I do not want to check irrelevant content into
  GitHub". So on `testing` every receipt reference is removed with
  nothing inlined, and the plans and specifications are deleted, except
  this plan and the item 1.8 specification, which go when their rounds
  land; the standing documents take in what they relied on. A return
  links no ignored path.
- *H18* waits for 1.8's landing, since it changes how every receipt's
  hash is computed (Claude's reading; Alec's answer addressed H20).
- *The written tier.* With Codex building, D3 stands as decided: Codex
  writes and verifies the pairs, Claude reads the audit list. §16 item 4's
  reading is withdrawn.

- *A small repository.* "Please do rewrite main's history to drop the
  old receipts, we can do that after 1.8 so that we have a downloadable
  repo after 1.8 of only a few MB (we probably have test files in there
  also, that can be replaced with downloadable links as a part of the 0.5
  work). So perhaps we can integrate 1.8 and 0.5, and then we will have
  the small repo we wanted." The order: item 1.8 lands on main, this item
  lands from `testing`, then main's history is rewritten to drop
  `doc/benchmarks/` and every large file no longer in the tree. Measured
  on 2026-11-12, GitHub holds 2.35 GB, of which 2.06 GB are receipts
  committed on main before they were ignored; the tracked tree is 203 MB,
  173 MB of it in twenty files over 1 MB. This round moves the large data,
  fixture and paper files out of the tree (§18.2 item 5). Then: "30 mb repo is
  fine. Don't remove important code history!" The rewrite drops only
  receipts and large data and paper files that come from other sources;
  every commit and its code stays. And: "Let's just aim for under 100
  mb." And: "But files tracked in other repos should not be checked in
  here. Of those are pre-build binaries or weight files, we probably want
  to keep them." Read as: a file whose source is elsewhere leaves the
  tree; a binary or weight file made here stays (Claude's reading).

### 18.2 Codex text (Claude, 2026-11-12; for Alec to paste on the second machine)

~~~
Item 0.5, the test development pass, continues on this machine (Alec,
2026-11-12). Claude wrote this text and reviews your return; Alec decides
and lands. Codex on another machine is building item 1.8 on main at the
same time: never push main, and never force-push.

Get the code. Work on the branch testing: main at 9f7923d67f (item 2.0
complete) plus Claude's commits with rounds 1 and 2 of this item and the
documentation pass (the plans and specs folded into doc/*.md). In a
BasicModel checkout that exists, run git fetch origin, then
git checkout -b testing origin/testing; otherwise
git clone -b testing https://github.com/arborrhythms/BasicModel basicmodel.
Have git-lfs installed first, so that data/mnist_train.csv is the real
file (as a pointer, its case skips). Run make install (Python 3.12; .venv from the unpinned requirements.txt).
When origin/main moves, merge it into testing.

Read first, in doc/plans/2026-11-09-item0-5-test-development.md: §4 (the
rules), §5 (the H items), §16 and §17 (what was reconciled and built), and
§18.1 (Alec's decisions of today, which override §16 where they differ).
The other plans and specs are deleted: doc/*.md is the design, and
doc/Testing.md holds the rules for tests and measurements.

Check the copy before building any local data. This passes 22 cases:
env -u RUN_TRAINED -u RUN_SLOW .venv/bin/python -m pytest -q -p no:cacheprovider test/test_math_corpus.py test/test_game_corpus.py test/test_lesson_stages.py test/test_training_corpus.py
Each manifest's command, run with its output in a scratch directory,
reproduces the frozen files byte for byte. The games and lessons builders
need their recipes copied inside the output directory first, and the
lessons builder does not create that directory; rebuild the training
manifest with --root at a copy of the data it reads and --local-root .,
then compare data/training-corpus.json and data/eval/heldout-cost.json.
Then build the developmental assembly once:
.venv/bin/python bin/prepare_corpus.py --recipe data/corpus-sources.json --output data/corpus
(1.5 GiB; it checks every source against its recorded digest). The web
shard data/fineweb/shard_00000.parquet is tracked. Never run
bin/training_corpus.py freeze --root . without the assembly in place: it
writes B1's manifest_sha256 and part_b_scan as null.

The round, one return:
1. Review rounds 1 and 2 as built (§17) against the plan and §4; repair
   what you find and record each repair. Known warts: the two builder
   limits named above; stage B2's local flag (§17, H19 withdrawn); and
   test_training_corpus.py's local part B case, which fails once the
   assembly is built here, because the assembly's manifest records
   assembly_seconds, a wall-clock time, so the manifest_sha256 recorded
   for B1 cannot match another build (B1 needs an identity that leaves the
   timing out). A repair that touches a frozen corpus freezes it again
   with the same seed, and its rows must come out identical.
2. H1 (§5): the bounded-test batching, measured with complete sweeps on
   this machine, which runs no other sweeps.
3. The games' written tier (§6.3, D3): you write and verify the varied
   pairs once and freeze them with the model identifier, the date, the
   prompt's sha256, the counts verify accepted and rejected, and the keys
   of a sampled audit list for Claude to read.
4. On 2.0's model (§18.1): H2, starting with its first check (what each
   decomposition arm still measures, before anything is regenerated; an
   arm with nothing left to measure goes to Alec); H3, once under
   RUN_SLOW; H5 with test/test_interval_log.py, its hook in bin/Models.py
   kept to one call because item 1.8 edits that file; H6; H8, the
   declaration of §7 written for Alec's word with every changed number
   marked, and one dry run on grade 0; H12; H14's scoring dry run; and
   §6.4's check of which training signal teaches a whole-sentence target
   (if neither does, the games' training stops at that finding, which
   goes to Alec). Record each number as measured on 2.0.
5. Large files out of the tree (§18.1): move out every tracked file over
   1 MB whose source is elsewhere, and every paper (Alec: "files tracked
   in other repos should not be checked in here"); binaries and weight
   files made here stay.
   - data/fineweb/shard_00000.parquet (90.1 MB; its source is BASE_URL in
     bin/data.py), data/mnist_test.csv (17.5 MB) and data/mnist_train.csv
     (a git-lfs file): fetched from their public sources.
   - doc/research/*.pdf (25 papers, 55.4 MB): each becomes a link to its
     public source where the research notes cite it, and none is fetched;
     a paper with no public source stays, and §19 lists it for Alec.
   - test/fixtures/2026-10-10-item6/identity-rst/validation/retained-scorer.pt
     (12.3 MB) is a weight file made here: it stays.
   A fetched file is named in a tracked manifest with its URL, byte size
   and sha256; one preparation command downloads and verifies all of them;
   a test that needs one skips, naming that command, when the file is
   absent (tests never use the network, §4 rule 5). Record in §19 the
   tracked tree's size before and after (git ls-tree -r -l HEAD).
6. Not in this round: H18 (after 1.8 lands), H20 (done on testing, §18.1),
   H4 (the joint exit with item 1), and round 3 (§11).

Rules: §4 of the plan. No seed is chosen so that an assertion passes; a
fixture's capacities and shapes are never resized; legacy paths are
deleted, not kept; comments state technique. Find causes yourself and
repair them where they arise; a mechanism added only to pass a gate is
not a repair. A setup defect in a test of this round is fixed without
stopping, its assertions unchanged. Continue without stopping and record
each decision you take for Alec; stop only for a design question, written
for Alec with your recommendation. A failed
test/test_mm_xor.py::TestMMXorConvergence::test_convergence or
test/test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class
stops the round.

Testing: fast checks while you build; one default sweep when you freeze,
env -u RUN_TRAINED -u RUN_SLOW make test, nothing deselected.

Return as §19 of the plan: what changed and why, each item's certificate
and numbers, the sweep's result, findings, and decisions for Alec.
Receipts stay local under doc/benchmarks/ on this machine, where Claude
cannot read them, so §19 carries inline every number, command and hash
the review needs, and links no ignored path. Freeze: commit on testing;
record the commit and the source hash (test/bounded_tests.source_snapshot,
SHA-256 of its compact sorted JSON, and whether
data/corpus/assembly-01/manifest.json was present); push testing.
~~~

### 18.3 The review round here: integration, and the lesson gates (Alec, 2026-11-13)

Alec, 2026-11-13: "0.5 is being completedin one round on the other machine,
so we will have to integrate against the pushed 1.5 source and make this
change before subitting. can you make sure that is in the purview of your
review round?" Claude reads "1.5" as the pushed checkpoint of item 1.8
(`main`, `ab5176e0e2`); the round below also integrates with the pushed
source of this item (`origin/testing`, `b6370617d3` when this was written),
so it holds under either reading.

No further text goes to the second machine. Claude's review of its return
(§19, on `origin/testing`) covers, before anything is submitted:

1. *The review of §19* as returned.
2. *Integration.* The uncommitted document work of this worktree is based on
   `96307818d2` and is brought onto the pushed `testing`; and `testing` is
   integrated with the pushed 1.8 source, so that what is submitted runs on
   it.
3. *The lesson gates*, made here.

The lesson gates are the outside gate of the 1.8 specification's §17 point
18. Alec: "Operators are "a priori" concepts that we are using for the
control of the architecture itself. Lets not do that; the gating of those
operators might be able to be done *in the tests*, which would act like
piagetian stages signalling that the test suite can move on to the next
stage. Are there tests that we could administer that would determine if
the correct operators are being used in the correct locations? If so, the
test gating should be moved outside."

- *What each lesson expects is declared.* Each variant of each recipe
  (`data/lessons/recipes`) states the operator applications its sentences
  should contain: the operator by its rule name in the grammar file the
  lessons run under, and the slots that are its operands, in order. Claude
  proposes them from the templates and the grammar; which operator is
  right at a place is Alec's to decide, from the table.
- *It is carried into the records.* `lesson_stages.expand` writes, for each
  generated record, every expected application as the operator's name and
  the places of its operands in the rendered sentence. The corpus is
  rebuilt with its recorded seed; its sentences do not change.
- *Two measures*, on a stage's held-out records presented without training
  (`corpus_lessons.heldout`):
  - `operator_placement`: the share of expected applications found in the
    tree of the path the model kept. One is found when a node of that tree
    has that operator and its operands' words are exactly the declared
    places. Structure around it does not count against it, since a
    sentence can have more than one right tree.
  - `heldout_answers`: the share of held-out questions answered with the
    supplied answer (`text_correct`, `bound_correct`, as scored now).
- *The criterion.* Each recipe's `criterion` names these two measures in
  place of `occupant_definiteness`, which no code computes. Their
  thresholds stay null until Alec sets them from the numbers;
  `test_lesson_stages.py` keeps its assertion that none is set.
- *The gate.* The lesson driver takes a stage's two measures after each
  pass and records them. While a threshold is null it moves on as it does
  now and says so; once thresholds are set, a stage is passed when both
  are met and the next begins then. Nothing in the model changes for this.
- *Later.* A test of the field a sentence is left with waits for item
  1.8's extend. The prerequisites of `test/phase_artifacts.py` are not
  changed before thresholds exist.
- *The relations stage* (`09-relations.json`). Alec, 2026-11-13: "If a
  sentence like "cats have fur" does not write a relation, then we have
  not correctly parsed the sentence as a relative truth, or a relation
  between two ideas (have expresses parthood, is expresses identity
  (unless predication), both are relations that are written to STM1 by a
  correct grammatical parse, are stored in LTM1, and teach the conceptual
  taxonomy)." So this stage's expected operator applications are `part`
  at the verb of a sentence of parthood and `equal` at the verb of a
  sentence of identity, each between the sentence's two ideas, and its
  held-out measure counts a sentence as right only where the relation is
  in `LTM1` after it. No word is tied to an operator in code: the recipe
  says which slot each operator is expected at.

### 18.4 A proposal for the lesson gates (Claude, 2026-11-13; for Alec to correct)

Alec: "please just make a proposal and well go from there." Nothing here
is set in a recipe yet.

**The operators expected, by stage.** A slot is a named place of a
recipe's variant. Interpretation of a leaf is forced while reading and is
not counted.

| Stage | Sentences | Operator expected, and between which slots |
|---|---|---|
| 0 words | "{w}." | none: one word |
| 1 interpret | "This is {n}." "What is this?" | `sign` at "this"; `equal` between "this" and {n}; `ask` at "What" |
| 2 clause | "{n} {v}." "{p} {v}." | `verb` between the subject and {v}; `lower_unary` at the bare plural {p} |
| 3 ask | "{n} {v}." "Who {v}?" | `ask` at "Who"; `verb` between "Who" and {v} |
| 4 verb | "{a} {t} {b}." | `verb` between {t} and {b}, and between {a} and that phrase |
| 5 modifiers | "{j} {p} {v}." | `intersection` or `product` between {j} and {p} |
| 6 determiners | "The {c} {v}." "A {c} {v}." | `definite` between "The" and {c}; `lower` between "A" and {c} |
| 7 adverb | "{n} {v} {a}." | `adverb` between {v} and {a} |
| 8 prepositions | "{n} {v} {p} the {l}." | `definite` between "the" and {l}; `preposition` between {p} and that phrase; `compound` between {v} and the prepositional phrase |
| 9 relations | "{n} is a {c}." "A {c} is an animal." | `lower` between "a" and {c}; `part` at "is", between the two ideas |

Four rows are the least sure and are the ones to correct first: the bare
plural of stage 2; the operator that joins a subject to an intransitive
verb; `compound` as what attaches a prepositional phrase; and `part` at
"is a" in stage 9, which is a predication and so, by Alec, not `equal`.

**The thresholds.** One number for both new measures at every stage:
0.875, that is one record in eight may miss.

| Stage | Held-out records | May miss |
|---|---|---|
| 0 words | 3 | 0 |
| 1 interpret | 2 | 0 |
| 2 clause | 7 | 0 |
| 3 ask | 4 | 0 |
| 4 verb | 24 | 3 |
| 5 modifiers | 12 | 1 |
| 6 determiners | 8 | 1 |
| 7 adverb | 12 | 1 |
| 8 prepositions | 15 | 1 |
| 9 relations | 12 | 1 |

- *Which measures a stage has.* `operator_placement` at stages 1 to 9.
  `heldout_answers` at the stages whose variants ask a question: 1, 3, 4
  and 9. Stage 0 keeps `definition_trust_fraction` and stage 1 keeps
  `denotation_fraction`, each at 0.875 too. Stage 9 keeps
  `relation_groups` as it is.
- *Twice running.* A stage is passed when its measures hold on two passes
  in a row. The held-out sets are small, and one pass can be met by
  chance.
- *From the first numbers.* No stage has been measured against these.
  The first run reports each measure beside its threshold, and the
  thresholds are corrected from that before any stage is made to wait.
