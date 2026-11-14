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

## 19. Item 0.5 return (Codex, 2026-11-12)

This section is the review record; measurements below are on item 2.0,
`9f7923d67f8921ef5c14afdf2fe18c536802018b`. Work began at testing
`96307818d257ce72a24bd5a1cc81897200cac944`. During the work Alec committed
`0e34c23775` (machine-specific editor settings), `0f29e41182` (MNIST), and
`840b5f91d3` (FineWeb) in the same checkout. They are retained in the final
history. Item 1.8 runs on the other machine; no item 1.8 implementation is
attempted here. This machine has 24 GiB RAM and eight CPU cores. The environment
is Python 3.12.13, torch 2.15.0, numpy 2.5.4, pytest 9.2.0, pandas 3.1.1 and
pyarrow 26.0.1, from the unpinned requirements installed with `make install`.

Alec's later directions override the pasted instructions above:

- Stored papers stay stored; no research PDF is replaced with a link or removed.
- Probe evaluation may run the model. H5 uses a separate evaluation process so
  its admissions and RNG draws do not change the learner.
- The historical H3 count of “26 core” cases is treated as obsolete after Alec's
  answer “probably obsolete.” Its selector list and receipts were not present
  in this checkout or recoverable from the tracked plans. No tests are deleted,
  and execution of that unidentified list is not claimed.
- Push `testing`, verify synchronization, then update and switch this checkout
  to `main` for subsequent work. Never push or force-push `main` in this round.

### 19.1 Corpus repairs and frozen written pairs

The pre-build command from §18.2 passed **22 cases in 38.71 s**. The original
math, template games and lessons were reproduced byte for byte at seed
**20261112** before any local assembly was built. The builders now create their
output directory and copy their recipes from an external recipe directory;
the recipe files remain part of the frozen corpus. No original row changed.
The final games/lessons/training reproduction compared **39 files**, all
byte-identical, including the new written tier. Original tracked JSONL rows
were also compared directly to the starting Git tree: **16 files identical**.

The actual assembly destination is `data/corpus/assembly-01`, rather than the
parent `data/corpus` shown in the pasted command. Run:

```sh
.venv/bin/python bin/prepare_corpus.py --recipe data/corpus-sources.json --output data/corpus/assembly-01
.venv/bin/python bin/game_corpus.py freeze --seed 20261112 --recipes data/games/recipes --out data/games --written data/games/written
.venv/bin/python bin/lesson_stages.py freeze --seed 20261112 --recipes data/lessons/recipes --out data/lessons
.venv/bin/python bin/training_corpus.py freeze --root . --local-root .
```

The assembly verified its source digests and built **57,210 documents /
3,421,205 sentences in 127.579667667 s**. The old raw B1 manifest digest
`cafa9a495c1d16ac73eb0ddff1e2e825fdb1b9c76d7cc5c34a048e64f785bd0b`
differed from this build's raw
`431d188f9a5379ef9e1ef1cd09b278adb6ed169a4008b36fb405197e02e61303`
only because of `assembly_seconds`. B1 now hashes compact sorted JSON excluding
that one field: **`d84a6bb65e59d77480203d00b508ec6cb94b9fbe84a3304634f2f7429c959e40`**.
The freezer checks that the assembly manifest, SQLite database and first web
shard exist before writing; missing part B can no longer overwrite the frozen
identity with nulls. The SQLite connection closes explicitly. B2's `local=true`
is correct now that the shard is external local data.

After builder repairs, the four corpus files passed **24 cases in 8.65 s**.
After written pairs and external-data preparation, the four corpus files plus
`test/test_prepare_data.py` passed **28 cases in 13.95 s**. The held-out-only
scan has **983 distinct sentences, five assembly matches**, compared with
953/five before written pairs. The held-out cost validation items did not change.

The written recipes and rows are in `data/games/written/` and the eleven
`*.written.jsonl` files. They contain **12 not, 12 conjunction and 12 unary-chain
pairs; 36 accepted, zero rejected**. They were authored and verified once,
without replacing rejected draws or changing the source-hash split. Provenance:
model **gpt-6**, date **2026-11-12**, prompt SHA-256
**`636ab4b01d97e01d7a13338169b14a58bcbb4244e75a82d8bf5da2efa748fbce`**.
The full prompt and verification frames are tracked data. Claude's audit is
pending; the fixed sampled keys to read are:

| Game | Audit keys |
|---|---|
| not | `not-written-000001`, `not-written-000003`, `not-written-000009`, `not-written-000011` |
| conjunction | `conjunction-written-000003`, `conjunction-written-000006`, `conjunction-written-000008`, `conjunction-written-000011` |
| unary_chain | `unary_chain-written-000002`, `unary_chain-written-000003`, `unary_chain-written-000005`, `unary_chain-written-000006` |

Final corpus file SHA-256 values:

| File | SHA-256 |
|---|---|
| `data/math/manifest.json` | `d64787a9597ab081c4e16ce246e2b51c2a3a0df05d46aa5801822b742233686b` |
| `data/games/manifest.json` | `500b2e6972308b99d80977fe39b2d11ad7209551d9860da9586a082c04eb8c1d` |
| `data/lessons/manifest.json` | `5f777efe11356816724a43e35fd57686486ca1810c26d55e135c4aad94a59818` |
| `data/training-corpus.json` | `696b06a9d542bea7e8e16df96173397387915dbac8a89fdd81fcae7a2c342b1a` |
| `data/eval/heldout-cost.json` | `ef9e02ec98487337e211cd56bd801b1056ba4f56afea082bea30092adbdb948f` |

### 19.2 External data and preserved local material

`bin/prepare_data.py` is the single explicit download/verification command:
`.venv/bin/python bin/prepare_data.py`. `data/external-files.json` records every
URL, size, digest and archive conversion. Its SHA-256 is
`f34ee62cb96ade7f1164152127c84b138ca5729ad33fa3cfb896c68c7cd97cfe`.
All three files were verified in place, and the command was independently run
into an empty destination: all three downloads reproduced the exact bytes.
Pytest replaces the FineWeb download entry point with a local-only lookup, and
MNIST consumers require local files; absence skips with the preparation command.
The intentional test of an unfetched historical LFS pointer remains.

| Local file | Bytes | SHA-256 |
|---|---:|---|
| `data/fineweb/shard_00000.parquet` | 94,511,680 | `6a3450ea745ee6b6830c09819c0f2c72682eba443022de595bcb50ceb3092c22` |
| `data/mnist_train.csv` | 109,640,201 | `06e64b0bf65de89c522348319ce5f63f43fb1cb40e1ce3c2ff43c1d3b8a0e14c` |
| `data/mnist_test.csv` | 18,303,650 | `c5adfcc9ce29069f8a988a6af1dcdc009bba975d2c98a52233661f79d880292b` |

The Kaggle archive is 15,948,628 bytes, SHA-256
`b6783c2e65a110a0891e930b4eecd330670eacd7d4730ed01191934fe8876573`.
Its raw train/test members have digests
`fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d` /
`51c292478d94ec3a01461bdfa82eb0885d262eb09e615679b2d69dedb6ad09e7`.
The declared header and CRLF conversion reproduce the original CSVs exactly;
no example is transformed or dropped. Existing corrupt files fail verification.
All research papers, `BasicModel.pdf`, and the locally made retained scorer
weights stay tracked. The seven preexisting planning notes stay in the ignored
`scratch_preexisting_notes/` directory for Alec's review; none was deleted.
The starting tracked-tree size was **206,872,716 bytes**, counting the old MNIST
training LFS pointer as 134 bytes, as `git ls-tree -r -l HEAD` does.

### 19.3 H2: native snapshots and strict mechanism findings

Both arms still measure forced ask/query/descend/return/bind through the real
training driver at one and two successors. Retirement of the decomposition
teacher did not retire these thought paths. Both original seeds and budgets
remain: **4199722147 / 512**, **2428440506 / 768**. The old recorded first hash
was `dbd7315f8a49f581f5f2bb0f837e0075ef9dbada63cc94cc9db6c0d1ee367d34`;
only the second prefix `f1d141ab...` survives in tracked prose. Its full digest
was not recovered and is not invented.

The old parameter-mapping adapter was removed from the fixture recorder.
Construction now captures the current native parameter names and shapes from
the original RNG streams; it does not load or remap old weights. The current
`MM_math_chain.xml` has event width **104**, meaning width **32**, and STM
capacity **8**. Restoring the historical 256-wide layout would resize this
fixture, so the present layout is recorded explicitly and unchanged. New
parameter-content hashes, identical before and after the observer-only port:

| Successors | Native parameter hash |
|---|---|
| 1 | `3ac2357a6cdc6791a87a5903e8c50f960f77eda0c3321605a394555e4c64c3ac` |
| 2 | `728d131deb1ffd76be08a65b24a28a5d0663e5160b1a6c030c4e638c4297dd99` |

Command: `PYTHONPATH=bin:test:test/fixtures/2026-10-25-stm-repair RUN_SLOW=1
DECOMPOSITION_SNAPSHOT_DIR=<new-local-directory> BASICMODEL_DEVICE=cpu
.venv/bin/python -m pytest -q -p no:cacheprovider -p recorded_initialization
test/test_math_chain_decomposition.py`.
The initial **2 failures / 7.60 s** were the observer's obsolete 13-field STM
assumption; it now follows the native **14 fields**, including chunk truth,
with capacity eight unchanged. After that setup port: **2 failures / 12.68 s**.
The proof assertions are unchanged and the piece stops for Alec:

- One successor changes state outside the strict footprint: `when_time`,
  `_cs_field_concept_ids`, STM `_thought_contents`, subspace concept activations,
  codes, IDs and inventory rows, `_thought_stm`, and `_model_tick`.
- Two successors fails in `ConceptHierarchy.observe_member(parent=64,
  member=146)` from the kept sentence's parts-of-speech update: the class is
  not one order above its member.

Recommendation: inspect these producers against item 1.8's contract before
changing the footprint or class assertion. No second initialization was drawn
and no shape or assertion was weakened.

### 19.4 H14: next-word scoring dry run

The scorer originally staged all 16 alternatives into a one-stream model.
It now reads bounded waves within the existing stream capacity, captures their
word readings, pads only the input sequence dimension, and scores the full
choice bank together. The absent XML seed no longer silently means seed zero.
The initial setup failure is retained; no model capacity changed.

Command: `.venv/bin/python bin/eval_nanochat_grammar.py score --model
data/MM_corpus_lessons.xml --fresh --device cpu --limit 4 --output <local-json>`.
The scoring run completed in **17.072407667 s**: four items, 16 choices,
chance 0.0625, intact and shuffled top-1 both **0.25**, both MRR
**0.4340277777777778**, true NLL **2.7725887298583984**, target variance **0**,
effective rank **0**, intact-minus-shuffled **0**. These are tied scores from
an untrained model, not evidence of learning. Initial affected scorer/logger
checks passed **9 cases / 38.85 s** after allowing the normal compiler cache;
the earlier sandbox-only cache permission error is not counted as a model result.

### 19.5 H3: named slow cases and setup ports

The eight explicitly named cases completed once under `RUN_SLOW=1
RUN_SLOW_FRACTION=0.5`, CPU, one worker, 8 GiB, 1,800 s per worker and
10,800 s overall, with one case/file per worker. Command:

```sh
RUN_SLOW=1 RUN_SLOW_FRACTION=0.5 BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python test/test_report.py test/test_priming_energy.py::test_cs_to_ps_projection_live test/test_priming_energy.py::test_cs_to_ws_projection_live test/test_priming_energy.py::test_projection_bounded_across_epochs test/test_priming_energy.py::test_one_canonical_surface_per_tower test/test_packed_reconstruction_parity.py::test_native_packing_preserves_each_sentence_and_owned_program test/test_review14_contracts.py::test_review17_measurement_wiring_on_one_ordinary_batch --workers 1 --memory-gib 8 --batch-size 1 --max-files 1 --timeout 1800 --suite-timeout 10800
```

Initial result: **eight completed, eight failed**, with no cap exceeded. H2's
two arms are recorded separately above. The four projection cases all stop at
`Models._reverse_body`: `ps_rec, _ws_rec = rec` receives more than two values.
This is a producer finding; no runtime repair or weaker assertion is made here.

The three packing cases initially requested `What.present` answers from the
unrelated `MM_ladder` dataset while supplying their own raw inputs. The runtime
correctly rejected the misaligned targets. Both vocabulary warm-up and layout
measurement now explicitly use `What.inference`, which supplies no answer.
All sentences, model shapes and parity assertions are unchanged. Diagnostic
replays of the three original recorded random states then **passed 3/3**, in
**24.018155792 / 21.920409834 / 21.633024250 s** including process startup;
peak process-tree memory was **2,098,285,040 / 2,097,383,944 / 2,099,333,640 bytes**.
These are setup-port replays, not replacement fresh draws.

The review17 audit counted four walk menus after the runtime had grown to seven.
Its observer now counts attention, compose, candidate, bind, reconstruction,
thought and output, excluding rounds already used. The expected counts are
computed independently from the eligibility masks; the strict `walk_count`
assertion is unchanged. Replaying the child probe's original random state then
reached a different assertion: **zero compose score-function steps, expected
four** (`test/observer_probe.py:38`). It failed in **11.828133167 s**, peak
**3,551,072,024 bytes**. The retired-teacher checks before that assertion passed;
subsequent checks were not reached. This strict finding stops the observer piece
for Alec rather than silently replacing its expected count. All diagnostic
replays restored the captured Python/NumPy/torch states and the original
`PYTHONHASHSEED=2137108569`; no passing initialization was selected.

### 19.6 H6: real continuation certificate, pending producer decision

`test/test_corpus_checkpoint.py` now contains a slow-tier certificate that runs
real `runBatch` updates using the unchanged `MM_corpus_orders.xml` geometry,
with sentence packing enabled. Twelve source sentences (four simple clauses
repeated three times) leave a genuine mid-document checkpoint after the first
packed batch. The same next batch is run uninterrupted and after native restore.
The test compares byte fingerprints of both losses, kept-program actions, word
IDs and case values/selections, and the complete final `state_dict`. Immediately
after restore it explicitly compares the surprise, access-count, forgetting,
relevance and placement ledgers. The independent comparison reader is retired
in 2.0, so the test asserts its absence rather than inventing replacement weights.

Two setup defects were repaired without weakening assertions: program views
must be copied before commit invalidates them; and four sentences all fitted
one batch, so the input was extended to twelve, with capacities unchanged.
The real comparison then **failed in 39.66 s**. The immediate retention-ledger
restore checks passed. The uninterrupted/resumed reconstruction-loss hashes were
`dbb4d6b1a053970cfc7c207e50a1f63bcedda8b7791090c2036588611574fb73` /
`9e9ebb4d20b087c33f1a3d4f06d544703ac8d8e5dab2560f2b7a596a3de25465`.
Both answer losses were zero, hash
`df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119`.
The first, uninterrupted-next and restored-next batches each consumed four
sentences, in **13.222 / 11.252 / 10.020 s** respectively. The failure is
retained, not retried for a pass. The kept-program/final-state assertions stay
strict but were not reached after the loss assertion failed.

Reproduction entry point: `RUN_SLOW=1 BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test
.venv/bin/python -m pytest -q -p no:cacheprovider
test/test_corpus_checkpoint.py::test_real_next_batch_matches_checkpoint_losses_and_kept_programs`.
The retained checkpoint SHA-256 is
`9e88533fcf892ce000cdec4c11447aff965cd3d424228967c25d75b02c6b9c8c`;
its exact configuration is
`081e1b6383183f9f93e24a42e8f081479caba1d1f238f300fd209dd2f1bb272f`,
manifest `fed2020917227978b59a48c5a9c7766fb46ba4932184c1c39bd151c9f98885b2`,
and rows `cb6fc2ae07a10c7e4f5d08a94b9e9ae0f7e840cd072f4f695fe30264c9c17172`.
The direct initial test did not retain its before-construction RNG state;
the saved checkpoint contains the continuation RNG used for diagnosis.

Under §3's mechanism-defect rule, H6 stops at this finding. The interleaved
stream extension and the `train.py` resume acceptance command remain pending;
they are not claimed as certified. Recommendation: repair native continuation
at its producer before adding a second stream scheduler to that certificate.

### 19.7 H8: proposed declaration for Alec

This is a declaration for review, not authorization to start the item-0
campaign. The three conditions are answer plus expectation, expectation only,
and zero attention budget. Frozen grades and existing intermediate-answer
supervision implement corrections 1 and 2; no objective was added. Credit remains
reconstruction plus answer. The grade-0 dry runs execute one unseeded attempt per
condition and do not reset or select a passing start.

| Number | Previous declaration | Proposed campaign | Change |
|---|---:|---:|---|
| Epochs per condition | 17 | 17 | unchanged, pending representative timing |
| Fresh paired starts | 10 | 10 | unchanged; capture each unseeded first draw |
| Required successful trainings | 9 of 10 | 9 of 10 | unchanged |
| Attention budget | 32; zero control 0 | 32; zero control 0 | unchanged |
| LTM capacity | 131,072 | 131,072 | unchanged |
| Batch size | 8 | 8 | unchanged |
| Learning rate | 0.001 | 0.001 | unchanged |
| Epoch wall-time basis | eight hours | not inferred from grade 0 | old basis withdrawn; new basis pending |
| Worker timeout | 24 hours | not yet redeclared | no timeout increase authorized here |
| Chain accuracy bar | referenced in the old ignored protocol | retain the original bar once recovered | numeric value unavailable; Alec must supply it |
| This development check | not the campaign | one epoch, 40 documents, 60 declaratives, nine batches per condition; 1,800 s / 8 GiB guard | new dry-run budget only |

The referenced old protocol's numeric chain bar is absent from the tracked
plans and standing documents. It is not reconstructed from a result or replaced
with a new threshold. A grade with no answer targets cannot time question,
worked-step or deep-chain work or establish an answer gate; those limitations
prevent using this dry run as a full-campaign runtime projection. Correction 4,
step-based work accounting, remains outside this item as §7 requires.

The actual grade-0 runs used CPU, one torch thread, compile backend `none`,
eager peer reading, unchanged `MM_math_chain.xml` shapes and learning rate
0.001. Command for each condition:

```sh
.venv/bin/python test/item05_measurements.py math --condition <condition> --output <new-local-directory>
```

Each completed its sole epoch, 40 documents / 60 declaratives / nine batches.
The work includes real forward/backward/optimizer steps. Every answer loss was
zero because this grade has no question targets; reconstruction and expectation
work executed. These development timings are not warmed throughput or learning
claims, and compile/recompile phase timings are not claimed.

| Condition | Internal seconds | Guard wall seconds | Peak process-tree bytes | First / last reconstruction loss |
|---|---:|---:|---:|---|
| `answer_expectation` | 298.992322333 | 301.698018875 | 4,862,694,248 | 0.00575305474922 / 0.018265908584 |
| `expectation_only` | 426.288238334 | 429.392807958 | 4,196,274,616 | 0.00582458917052 / 0.0187019258738 |
| `zero_budget` | 358.115439708 | 364.514282084 | 4,001,976,664 | 0.00604714872316 / 0.0178549475968 |

The harness was unchanged across these runs, but disabled logger/test work was
still being finished during the first two development epochs. Their source
snapshots are recorded separately; this is not one frozen paired campaign:

- `answer_expectation`: `77294cecce58abbe8312a99f33b32dcbcf70ff4040d73c2c89776e9a87a77d7b`.
- `expectation_only`: `63dbb9cbada914079aacb74822b82720c27074ddfa94d0fec003ae2343d8d48a`.
- `zero_budget`: `d69526500b2bb7553f6f2e9bd77db3ac1bd777c68aec3aa1bf7f95320e2633d7`.

### 19.8 Whole-sentence game signal: null at the declared budget

Command: `.venv/bin/python test/item05_measurements.py game --condition
<future|supplied> --output <new-local-directory>`. Each signal received one
fresh unseeded attempt, **16 updates, batch one, learning rate 0.001**, CPU,
one thread, compile backend `none`, unchanged `MM_corpus_lessons.xml` shapes,
with the same 1,800 s / 8 GiB guard. The frozen training pair was
`not-clause-000002`: “Bella runs.” → “Bella does not run.” The future arm uses
`WhatQuestion.future(0, 1)` with source and target in one document; the supplied
arm stages the full target through the existing lesson answer path. No parse
or answer was forced.

| Signal | Internal / guard seconds | Peak bytes | Complete target among observed outputs |
|---|---|---:|---:|
| future | 47.782687500 / 50.642138209 | 1,163,626,344 | 0/16 |
| supplied | 51.898928541 / 54.506942416 | 1,098,581,912 | 0/16 |

The future arm produced an empty string each time, with answer loss zero and
reconstruction loss 0.010708170943 → 0.0107099786401. The supplied arm produced
“runs” on update 13 and an empty string on the other 15 updates. Its answer
loss was 0.000548491138034 initially and 0.000548126641661 on the final update,
with a 478.653015137 spike on update 13; reconstruction was
0.00926270894706 → 0.00926536042243. Both source hashes were
`d69526500b2bb7553f6f2e9bd77db3ac1bd777c68aec3aa1bf7f95320e2633d7`.

These are outputs observed during the 16 updates, not an independent
post-update or held-out evaluation. Neither signal demonstrated complete
sentence production within this budget. Under §6.4, games training stops here
for Alec; neither signal is selected for a larger run. The corpora remain
frozen. Recommendation: inspect target credit and production at the existing
producer before authorizing a larger budget or a new objective.

### 19.9 H12: six placement attempts

`test/identity_experiment.py` no longer selects seed 20261010 or inherits seed
42 from `BasicModel.xml` for these measurements. The harness captures the first
unseeded state and shares that recorded state across all six placements.
`test/phase_experiment.py` now copies program values and credit before commit
invalidates their transient views. Capacities, shapes, assertions and the
existing paired-update wrapper are unchanged.

Command for each variant, using the same initially absent RNG path:

```sh
.venv/bin/python test/phase_experiment.py <new-local-directory> --variant <variant> --smoke --initial-rng <shared-first-rng.pt>
```

The smoke protocol is one epoch over the first two training documents, then
the first two held-out documents; batch two, learning rate 0.001, CPU, one
thread, eager peer reading, compile backend `none`, 1,800 s / 8 GiB guard.
The model's four-stream capacity is unchanged. Five variants completed with
three paired optimizer updates each and zero disabled batch-end updates.

| Variant | Guard seconds | Peak bytes | Exact answers / selections / reverse (of 2) |
|---|---:|---:|---|
| `verb_form` | 16.847331333 | 1,264,830,360 | runtime failure; no evaluation |
| `separate_leaf` | 27.355718250 | 1,414,858,792 | 0 / 0 / 0 |
| `noun_form` | 28.426374208 | 1,550,223,520 | 0 / 0 / 0 |
| `one_leaf_two_selections` | 26.026067708 | 1,396,443,152 | 0 / 0 / 0 |
| `two_leaves` | 27.051150250 | 1,374,980,112 | 0 / 0 / 0 |
| `unmarked` | 26.067261375 | 1,181,370,144 | 0 / 2 / 0 |

The `verb_form` first batch failed in `ThoughtChunks.stage → read → _index`
with “thought chunk is absent at its .when address”; exception cleanup then
failed in `Thoughts.replay_thoughts` with “thought occurrences must be strictly
chronological.” Its initial state, batch state and failure state were retained;
no retry was made. The unmarked arm's two correct selection counts are its
zero-selection control, not successful answer or reconstruction learning.
All six source audits were unchanged during their run, based at Git
`840b5f91d3fe53d1837ddca01a1ab0c94e83a97d`, source snapshot
`d69526500b2bb7553f6f2e9bd77db3ac1bd777c68aec3aa1bf7f95320e2633d7`.
Thus all six were attempted once; the harness is not certified as six successful
executions. The thought-memory producer failure goes to Alec.

### 19.10 H5 and the checkpoint dependency

`bin/IntervalLog.py` adds `intervalLogEveryBatches` (zero disables) and
`intervalLogFile`, with schema and parameter-table entries. The only change to
`Models.py` is one call in `BasicModel.runEpoch`, after batch/cursor counters
advance and before periodic checkpointing. One observer call in
`Forgetting.model_boundary` records every completed forgetting pass. Neither
observer writes learner state.

Interval rows capture reconstruction loss, expectation discrepancies, LTM
occupancy/capacity/forgetting count and report, provisioned-fact luminosity,
and order-zero inventory. Luminosity folds positive and negative fact poles
independently using the native formula; user/conversation origins are excluded.
The fixed probe worker owns a checkpoint copy, CPU evaluation, and its own RNG
and runtime globals. It reads four XOR plus 68 FineWeb probes, four fixed
generic-subject probes, word/sentence both rates, and one native reconstruction
sample. Probe failure or timeout is reported as failure, never as a measurement.

A full probe execution before the restore identity check was added completed
in **68.748045958 s** including process startup (**67.072013791 s** internally),
peak **857,998,944 bytes**. It reported XOR CP 0 (within/between 0/0, 2/4 pairs),
FineWeb CP −0.0565628260374 (within 0.292700588703, between 0.236137762666,
967/1,311 pairs), generic subjects 3/4, word both 0/90 and sentence both 0/76.
The reconstruction sample “Felix is a cat.” was available but truncated, byte
cost 0.0761762037873. **These are descriptive copied-model results only:** the
following restoration mismatch disqualifies them as learner stop-condition
readings.

The four descriptive generic-subject observations from that same disqualified
copy were:

| Probe | Expected | Observed | Correct |
|---|---|---|---|
| cats breathe | relative | relative | True |
| a cat is an animal | relative | absolute | False |
| this cat breathes | absolute | absolute | True |
| the cat is black | absolute | absolute | True |

The H6 checkpoint diagnostic compared native loaded tensors before any further
training. Six state-dictionary names alias one shared 4,229 × 392 float32
concept matrix: `conceptualSpaces.{0,1,2}.{similarity_codebook.W,layers.2.W}`.
`load_state_dict` and vocabulary restoration preserved its saved digest
`7f1e07c0f817b63e683c0cb08752ce3096bbe2234f48169d799c603daefee1a9`.
`_restore_structural_extras` changed it to
`86c9a1e42b2a66418371d4144d6b492ae64351ff2e4bdbe72a2062a75027e752`;
aligned-codebook resynchronization kept that changed digest. **21 values**
changed, in rows **12, 15, 4183, 4216, 4217**, columns
**360, 361, 364, 365, 367, 372, 373**, maximum absolute difference
**2.9802322387695312e-8**. Torch RNG matched the checkpoint before and after
optimizer construction. This identifies a byte-identity defect in structural
restoration; it does not prove that these differences alone explain H6's later
loss mismatch. No checkpoint adapter or tensor overwrite was added.

The logger now checks every restored state tensor byte-for-byte before running
probes and rejects such a copy. A real diagnostic replayed the checkpoint's
next batch once to reach a live interval boundary, then compared the parent
before and after collection: state tensors, vocabulary/structural extras,
corpus runtime, optimizer state, all three RNG streams and progress counters
were **unchanged**. The interval at batch two took **4.141079292 s** and
correctly reported the copy failure. Captured scalar values were reconstruction
**0.0325614511967**, concept expectation **0.117779292166**, LTM **42/16384**,
forgetting passes **0**, luminosity **0**, inventory **18/4229**. The guarded
diagnostic including its one continuation batch took **21.137955333 s**, peak
**1,979,371,048 bytes**, source
`3c476a928b5685dcc290415a4e822570b358c00eb9be03fca298e09ff931c076`.

Two preceding diagnostic setup attempts are retained: one used `_optimizer`
before `runBatch` assigned it; the other requested captured batch costs directly
after loading, before any batch had populated them. The final diagnostic uses
the returned optimizer and a real continuation batch; neither setup failure is
counted as a model result. Recommendation: fix native checkpoint identity under
H6 before enabling these probe rows as stop conditions. H5 scalar logging and
parent invariance are measured; valid copied-model probe acceptance remains
blocked by that producer finding.

### 19.11 Recorded initialization identities

All hashes in this table are SHA-256 of retained first-state files; they
identify local diagnostic evidence, not seeds selected for a pass. The local
receipt directory is deliberately not linked.

| Attempt | Initial random-state SHA-256 |
|---|---|
| `math-answer_expectation` | `ba1882d7c32ae298b4178852c905abdc0d6a880fa03d731ed74890df6928f491` |
| `math-expectation_only` | `2b172d20c5ddf119c53467129d66b80940279a851803fd875b8eda187f695b2c` |
| `math-zero_budget` | `85ce483bfda4768e2a4716432b123098db203c3c3a59fb60c4c1dce8c00c0e0d` |
| `game-future` | `92fd8438cf1abded11d35162228b64010fdab85f3ba25cc52491a8929a06684a` |
| `game-supplied` | `217bafcc2d9803624b5c80305869d0199a02c0f61eac8f33e7c88fa6cf040b36` |
| six placements, shared first draw | `e98e973167ab439dcc8794d20db29ccdd4dd2df826f8501f411f3fd726360c68` |
| `test/test_priming_energy.py::test_cs_to_ps_projection_live` | `ca4a38cef95a4780f631754d07401d2d60b8756a3b3cf038f16d2455c139fda4` |
| `test/test_priming_energy.py::test_cs_to_ws_projection_live` | `043d396d8976906483a1d772e60e16c808fdcdcac865e48b1d3810edbc01ece9` |
| `test/test_priming_energy.py::test_projection_bounded_across_epochs` | `1d8ff4bcde0e5a148e6f5d3e498f3e1fd83058dfe810a2ee01acb9f66536253f` |
| `test/test_priming_energy.py::test_one_canonical_surface_per_tower` | `8adba8971035767e8a6ee6da533261a6b726d8d55b61ab7594805a3767cf75c3` |
| `test/test_packed_reconstruction_parity.py::test_native_packing_preserves_each_sentence_and_owned_program[0]` | `022c0830355dbb8855e81f4b2e6e9a15b5774c902f90764348ea48028e4c51f5` |
| `test/test_packed_reconstruction_parity.py::test_native_packing_preserves_each_sentence_and_owned_program[1]` | `125752577d1f04cfdb4235d1fb6fde33f191423cefdb1c701e11174f41afee7b` |
| `test/test_packed_reconstruction_parity.py::test_native_packing_preserves_each_sentence_and_owned_program[2]` | `99974fc7024bfa0ce0240e901e48a2c643cd5bc8cbac94086082414a3ad3eed1` |
| `test/test_review14_contracts.py::test_review17_measurement_wiring_on_one_ordinary_batch` | `8b6a00302251086a8718ae4263fdc78fa0fe1e2077bd0d6efa04b793eccc9cec` |
| review17 child before model construction | `f5d9a7f877986b984a29358a0f3176d13fff86edf9e5ffc02595fcd0b292ddb1` |

### 19.12 H1 and freeze checks

The inherited runner already batches up to **256 cases / 16 files** and
recycles workers at resource boundaries. Its constants had drifted to a
12 GiB worker cap and a 9 GiB scheduling share despite H1's 8 GiB ceiling.
Both are restored to **8 GiB**. The 8 GiB machine reserve and four-core CPU
reserve are unchanged, giving this machine **two workers / 16 GiB aggregate**.
The standing testing guide's worker formulas, recycling threshold and memory
examples now describe these limits; its separate weekly objective ceiling is
unchanged.
No fixture cache was introduced: reusable full models would need a verified
reset of vocabularies, allocators, runtime journals, optimizer state and global
construction registries. Such a reset is not established by the current
fixtures. No construction/reset/compile/forward/backward speedup is claimed.

The final affected-file command selected the complete files below with
`env -u RUN_TRAINED -u RUN_SLOW make test TEST_ARGS='<files>'` (the receipt used
the equivalent direct `test/test_report.py` entry point):
`test_interval_log.py`, `test_nanochat_grammar_eval.py`, `test_forgetting.py`,
`test_bounded_test_runner.py`, `test_training_device.py`, `test_mnist.py`,
`test_basicmodel.py`, `test_packed_reconstruction_parity.py`,
`test_corpus_checkpoint.py`, `test_game_corpus.py`, `test_math_corpus.py`,
`test_lesson_stages.py`, `test_training_corpus.py`, `test_prepare_data.py` and
`test_doc_links.py`, all under `test/`. Result: **366/366 completed, 356 passed,
10 slow cases skipped, zero failures, 196.007174000 s**. There were five
workers including recycled continuations; the maximum reported worker peak
was below 5.65 GiB. New logger checks include unchanged parent state, disabled
logging, failed children, every forgetting pass, the independent luminosity
formula and rejecting a changed checkpoint copy; the scorer check keeps all
candidates in one comparison bank across bounded input waves.

The complete default sweep was launched as
`env -u RUN_TRAINED -u RUN_SLOW make test`, with only `--run-dir` supplied to
select a fresh local receipt destination. It collected **7,154 cases**, with
nothing deselected, on source SHA-256
`3c476a928b5685dcc290415a4e822570b358c00eb9be03fca298e09ff931c076`.
The assembly manifest was present. Worker limits remained **1,800 s / 8 GiB**;
overall limit **10,800 s**, aggregate **16 GiB**, two workers. A separate
read-only monitor watches the durable case reports and interrupts this sweep
if either of the two §18.2 stop cases fails; it cannot turn a failure into a pass.

Before seeing the default sweep's final result, the H1 comparison is declared
as a second full coverage run with **64 cases / four files per worker**, on the
same frozen source, using the same two-worker, 8 GiB per-worker / 16 GiB total,
1,800 s / 10,800 s limits and no simultaneous workload. This compares smaller
batches with the retained 256/16 default without reproducing the invalid
12 GiB cap. It is a batching measurement, not a replacement acceptance run;
first-sweep failures remain failures. The same hard-stop monitor applies. It
runs only if the first sweep completes without a required round stop. Shared
compiler caches and fresh unseeded random draws are limitations of any
wall-time comparison; no isolated mechanism speedup will be inferred.

### 19.13 Decisions carried to Alec

- H2: retain the strict footprint and class-order assertions; repair the
  thought-state and kept-sentence producers before recertifying the two arms.
- H3: repair the reverse-body return contract; decide the review17 audit's
  compose-credit contract explicitly instead of changing four expected steps
  to the observed zero. The historical unidentified 26-case list is obsolete
  per Alec; the three packing setup ports preserve all parity assertions.
- H6/H5: repair structural restore's byte identity, then establish uninterrupted
  continuation before extending interleaving or CLI resume and before treating
  copied-model probes as learner stop conditions. Scalar logging remains usable.
- H8: provide or explicitly retire the unrecovered numeric chain-accuracy bar
  and approve a new representative timing basis before the full campaign.
  No budget increase or campaign launch is inferred from grade zero.
- H12: investigate the first verb-form thought-chunk failure; carry the five
  completed nulls at their original budgets.
- Games: neither existing signal demonstrated the complete target in the
  declared check. Inspect production/credit before choosing either for training.
- Written tier: Claude still needs to read the twelve frozen audit keys in
  §19.1. Verification of 36/36 authored pairs is not that human review.

H4, H18, H20 implementation and round 3 remain outside this round as directed;
H20's previously completed work is preserved. No learning gate is declared
passed by a mechanism probe, a skipped test or an incomplete measurement.

### 19.14 Default-sweep failure inventory

The first sweep's failures are retained independently of later setup-port
checks or the H1 timing comparison. The completed sweep recorded
the following **23 failures**:

| Case or family | Finding |
|---|---|
| `test_compose_pair_driver.py::test_complete_paths_train_in_one_step_and_commit_once` | Observer calls `.detach()` on `_compose_forced_slots=None`. A non-reading departure can reuse the completed composition and legitimately have no new forced slots. A setup port must explicitly verify that branch, keep the existing assertions, and never manufacture a composed departure. |
| `test_math_chain_section14.py::test_forced_c_e_closings_exercise_empty_search_mint_and_question_storage` | `Thoughts.replay_thoughts` rejects nonchronological occurrences both during `commit_thought` and during `controller_error` cutoff. |
| `test_output_path_supervised.py::test_mixed_supplied_numeric_and_automatic_text_trains_only_supplied_row` | The thought-history chronology assertion fails during the actual output path. |
| `test_output_synthesis.py::test_what_report_separates_families_and_thinking` | The thought-history chronology assertion fails during the public report path. |
| `test_operators_round4a0.py::test_when_reader_inventory_covers_current_attribute_and_dynamic_readers` | Inventory names `PartSpace._radix_part_events`; the actual time-band access moved into `_radix_code_events`, while the old method remains a wrapper. The inventory assertion itself is unchanged. |
| `test_reasoning_cde_model.py::TestReasoningCDEModel::test_answer_query_degrades_gracefully` | `MemoryLink.chunk_candidates → MemberRead.read → ClauseRow.point_of_row` reaches a shared-concept index that is not attached. |
| `test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row` | Native feature columns are `{0}`, expected `{4, 402}`. The strict feature assertions are retained for a producer/contract decision. |
| Sixteen `test_whole_presence.py` parametrizations | Missing original `whole-pullback-1.pt`; execution stops before the numerical assertions. |

The missing fixture is the sentence-four whole from the item-2.5 native mean
receipt, originally at
`doc/benchmarks/2026-10-31-item2-5-completion/addendum-10/native-mean/whole-pullback-1.pt`.
Historical commit `1e02cebf5d` describes total membership approximately
**1.23037e-7**, but does not contain the tensor file or enough values to recover
it. No matching blob appears in Git history, and a search including ignored
files under the local project trees found no copy. The sixteen cases are the
forward/pullback bounds (two dtypes × two masses), tiny-mass JVP/weight gradcheck
(two × two), and compiled pullback (two backends × two dtypes × two masses).
The original shapes and assertions cannot be preserved by inventing a different
whole. Alec has been asked for the original machine/path; these cases are not
converted to skips or assigned replacement tensors.

The non-missing-input failures' recorded pre-setup RNG identities are:

| Node ID (under `test/`) | Random-state SHA-256 | PYTHONHASHSEED |
|---|---|---|
| `test_compose_pair_driver.py::test_complete_paths_train_in_one_step_and_commit_once` | `d2a2922756ee7d2bb323a732efcfb51084c1f794c9c82b74067ebc8953bf6fb6` | 1381815608 |
| `test_math_chain_section14.py::test_forced_c_e_closings_exercise_empty_search_mint_and_question_storage` | `c98c3d84678deea3ae5e988fb4a6a59b6eea466a5be876f078409baf5267dd6a` | 1381815608 |
| `test_output_path_supervised.py::test_mixed_supplied_numeric_and_automatic_text_trains_only_supplied_row` | `48cc51d2db05d3b31f06e142ca2ecc492b18c428ac5b9c64b7e5aeb2a1fcde24` | 1381815608 |
| `test_output_synthesis.py::test_what_report_separates_families_and_thinking` | `78e816fb6b11f1ebe6635cc71dd5996745a03acd109bbe2a317d56460939b262` | 1381815608 |
| `test_operators_round4a0.py::test_when_reader_inventory_covers_current_attribute_and_dynamic_readers` | `404e9c21a083e3808d541b65c95a40fd1d95fab160c91931a5e0fef8bc533052` | 1381815608 |
| `test_reasoning_cde_model.py::TestReasoningCDEModel::test_answer_query_degrades_gracefully` | `96963dd4e585c792fedd7d7130294fd296d8aabeb9227f838cddb9896269e174` | 1381815608 |
| `test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row` | `731310802a6ba1b25f4869866f907919068551d0116d989e5b69b44c9f6f1293` | 1381815608 |

The existing non-strict expected-failure case
`test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input`
was an unexpected pass. Its marker and assertion were not changed. The two
required stop cases both passed: shared-concept raising in **0.204194417 s**,
and `TestMMXorConvergence::test_convergence` in **9.448233541 s**.

The default sweep completed **7,154/7,154 cases** in **4,080.850160500 s**
(runner time; outer `make` command **4,082.253496875 s**). Unique case outcomes:
**6,804 passed, 326 skipped, 23 failed, one unexpected pass**. The 173 workers
include recycled continuations and the seven large-memory workers. Peak
reported worker process-tree memory was **7,884,118,520 bytes**, below the
8 GiB cap; neither memory nor time cap was exceeded. Ordinary coverage was
**7,147/7,147**, large-memory coverage **7/7**. No required whole-round stop
was triggered. `make` exited **2** because the runner reported test failure.
`TestOrthogonalFlags::test_flags_match_expected` emits four passing call reports
for one case; the totals above count its node once. Result JSON SHA-256:
`4199826398ea8937b00a30f7b1fbc156945281c196f56d1a8b115476bc896a56`.

The final grammar-XOR reconstruction case completed its configured **400
epochs** and recovered **4/4** word multisets with all inverses available; the
field-readback audit was also **4/4**. Saved final answers were
**0.3009, 1.0010, 0.8918, 0.2727**, against **0, 1, 1, 0** (printed precision).
This is the one unseeded reconstruction run in the default sweep; the separate
trained-tier classification certificate was skipped and is not claimed.

### 19.15 H1 comparison timeout and required round stop

The predeclared comparison ran as
`env -u RUN_TRAINED -u RUN_SLOW make test TEST_ARGS='--batch-size 64 --max-files 4'`,
with an additional fresh `--run-dir` for its local receipts. Both runs selected
the same **7,154 cases** and validated the same **1,090 source files**, SHA-256
`3c476a928b5685dcc290415a4e822570b358c00eb9be03fca298e09ff931c076`.
The assembly manifest was present. No other sweep or model measurement ran
concurrently. The worker, aggregate and deadline limits were unchanged.

The comparison **did not complete**: **4,253/7,154 cases completed**, comprising
**4,054 passed, 195 skipped and four failed**. The ordinary phase stopped on
timeout; the seven large-memory cases were not started. Runner time was
**3,997.675058291 s**, outer `make` time **3,998.823302416 s**. The runner exited
**124**, `make` exited **2**, and there were **174 workers** including recycled
continuations. Maximum reported worker memory was **6,797,859,680 bytes** and
aggregate memory **12,946,470,184 bytes**, both below their caps. The first
sweep's aggregate peak was **12,107,985,616 bytes**. Comparison result JSON
SHA-256: `d55f962a74a3a139c397c33e8914954d643c26f5c500a0e329c3842d3ab4aa75`.

Worker 172 reached its **1,800 s** deadline after **1,801.046661375 s**, while
running
`test_percept_expectation_final.py::test_arrival_brick_compiles_and_equals_eager_loop_state_loss_and_gradient`.
Its peak was **482,018,600 bytes**. Worker 173 was consequently aborted after
**1,747.489444542 s**, peak **1,443,530,744 bytes**, while running
`test_perceptual_loopback.py::TestRecurrentCellAndOutputViews::test_full_forward_runs_recurrent_cell`.
Neither active case produced a completed pytest outcome. Their first-sweep
passes took **2.577253667 s** and **15.118661541 s** respectively; those passes
do not erase this timeout, and the timeout's cause is not established by the
logs. Initial worker RNG hashes were
`dbcde269f0fefea13074fe76db859bac9bab47a05adc3282de6b05cd4cdeaa3d` and
`b676b78ed672acb60015072503a44929d4a344607db651dee6fe41261ab0b8f1`.

The four completed failures were:

| Case (under `test/`) | Comparison finding | Pre-setup RNG SHA-256 |
|---|---|---|
| `test_math_chain_section14.py::test_forced_c_e_closings_exercise_empty_search_mint_and_question_storage` | Unlike its first-sweep chronology failure, this draw reached line 419: the minted committed meaning still has open slot `('referent', 0)`. | `9b5fd012305f78c266515795d3f183627d4e3f208f55097ee205b50b5369d871` |
| `test_output_synthesis.py::test_what_report_separates_families_and_thinking` | Thought-history chronology fails again. | `05190bd7faa6eb0ae3d0f2e5a42aadd9da8967d7d86d4cce141a4d67e94ebba6` |
| `test_operators_round4a0.py::test_when_reader_inventory_covers_current_attribute_and_dynamic_readers` | The same obsolete `_radix_part_events` inventory entry. | `62a984215c531b88db36dec30bad3bd1880336ad5ca2524535214dd5b38206a7` |
| `test_percept_transitivity.py::test_native_kept_readings_have_transitive_percept_codes` | Additional case fails thought-history chronology during closing and `controller_error` cutoff. | `6b062544a85c39b96c6522cf1059013c7e5c43b98d7847f1c579c7d2e530e2cb` |

The comparison's recorded `PYTHONHASHSEED` was **4241412517**. Both named
hard-stop cases passed again: shared-concept raising **0.214859458 s**, MM XOR
convergence **5.605781541 s**. The incomplete sweep itself requires the round
to stop under [Testing](../Testing.md). **No model run, diagnostic replay,
fixture port or further test sweep followed the timeout.** No cap was raised,
and the comparison was not retried. The 256-case/16-file default is retained;
the incomplete comparison supports no throughput conclusion or speedup claim.

Two setup ports identified from the completed first sweep therefore remain
unapplied: read the compose path's own `compose_forced_slots` and explicitly
verify the non-reading branch before representing its empty mask; and point
the time-reader inventory at `PartSpace._radix_code_events`. Their existing
assertions and fixture shapes remain unchanged. These await the next authorized
round, together with investigation of the timeout and the strict producer
findings. The original whole-presence capture's location remains unanswered;
its sixteen failures remain visible, with no replacement tensors or skips.

This is a **review handback with open findings**, not a green acceptance or a
completed H1 batching certificate. The complete default sweep in §19.14 is the
acceptance record; neither the comparison's passing cases nor the earlier fast
checks regrade its failures.

### 19.16 Freeze and repository handback

Implementation freeze commit on `testing`:
**`711203ffc8323c06145f214ba5bdd76be7d682c3`**. Its source identity is
**`3c476a928b5685dcc290415a4e822570b358c00eb9be03fca298e09ff931c076`**,
SHA-256 of `test/bounded_tests.source_snapshot` serialized with
`json.dumps(snapshot, sort_keys=True, separators=(',', ':'))`.
It contains **1,090 source entries**; `data/corpus/assembly-01/manifest.json`
was **present**, with the raw and normalized identities in §19.1. H18 is
deliberately deferred, so that ignored manifest still contributes to this hash.
The subsequent documentation commit records this freeze without changing its
source identity. No model execution or tests followed the required stop;
the staged patch passed `git diff --cached --check`.

Summing blob lengths from `git ls-tree -r -l HEAD`, the tracked tree changed
from **206,872,716 bytes** before this work to **94,184,121 bytes** including
this final handback. The implementation freeze alone was **94,182,343 bytes**.
These are tracked-tree sizes, not Git history or materialized LFS sizes.
The stored papers and retained scorer weights are unchanged. External datasets,
local receipts and the seven planning notes remain on this machine.

The handback preserves Alec's cleanup commits `0e34c23775`, `0f29e41182` and
`840b5f91d3`. The final pre-freeze fetch still found `origin/main` at
`9f7923d67f8921ef5c14afdf2fe18c536802018b`; no newer main change required a merge.
Publication is to **`testing` only**, for Alec's review and landing. Per his
later direction, this local checkout returns to the fetched **`main`** after
the testing push is verified. This handback does not merge testing into main,
push main, rewrite history, or remove the preserved local material.
## 20. Continuation and round 3 (2026-11-13)

Alec authorized all remaining 0.5 rounds without intermediate review, with
integration and review after 1.8, and then specified **testing only**. This
continuation does not alter §19's results, budgets, source identity or nulls.
It does not authorize learner-mechanism changes under §3. Stored papers and
the seven ignored planning notes remain local and unchanged.

### 20.1 Starting checkpoint and environment

`testing` starts this continuation at merge `91e71c40cf`, incorporating
`origin/main` checkpoint `ab5176e0e2` into prior testing tip `b6370617d3`.
The incoming checkpoint describes accepted 1.8 work but still schedules
§§3.16–3.18; it is not the final joint-exit hash. The actual local `main`
ref remains `9f7923d67f`. No main push or force push is authorized.
An invalid duplicate loose ref, `refs/heads/main 2`, prevented fetch; its
contents were preserved in the ignored continuation receipts before removing
it from Git's ref namespace. The real main ref was not changed.

The first compose setup replay failed during PyTorch compiler import, before
its test assertions: `duplicate extern kernel: conv1x1_via_mm`.
It exited 1 in 10.358644583 s, peak 799,835,720 bytes. An installed-package
audit checked 2,475 canonical Python files against torch's RECORD: zero
missing and zero mismatches. It found 14,632 unregistered duplicate files
with numbered names and canonical counterparts, totaling 508,817,512 bytes.
Those files were moved to ignored local quarantine, preserving their bytes;
no registered package file or version changed. RECORD SHA-256:
`b98ea4d12125be094c139d7db8e5dafb6fe19948b5164fb7abac531a6a1aaf07`.
Compiler import then passed. This finding does not attribute §19's earlier
H1 timeout to the environment.

The compose observer now reads the completed path's forced-slot mask instead
of a transient model member restored by `WholePaths.finish`. A later path
without a reading walk has no compose fork; the observer checks that fact
and records an all-false mask. Test assertions and fixture shapes are
unchanged. Replay with the original RNG SHA-256
`d2a2922756ee7d2bb323a732efcfb51084c1f794c9c82b74067ebc8953bf6fb6`
and `PYTHONHASHSEED=1381815608` passed: 1 case, pytest 4.59 s,
guard elapsed 7.448905125 s, peak 1,095,370,576 bytes. The reader inventory
also follows the renamed `PartSpace._radix_code_events` implementation.

H18 is taken at this new measurement boundary: source snapshots exclude
Git-ignored local data but continue to hash tracked files, even when an
ignore pattern matches, and untracked files not ignored. Non-repository
test roots retain their previous behavior. §19's old hashes are historical
and are not recomputed under this rule.

### 20.2 Declared rehearsal protocol, before learning

The existing `data/MM_corpus_lessons.xml` configuration is used unchanged.
The small rehearsal budget is one epoch and the first two documents in
lexicographic document-key order per lesson, plus the first two training
documents in key order per math grade. Each lesson's first two held-out
documents supply its bounded diagnostic reading. Full validation identities
remain available to the saved-model operator table; a bounded diagnostic is
labelled as such and does not substitute for full validation. No learning
seed is set or selected, and no arm is retried or extended after a null.
Each process retains the 8 GiB, 1,800-second guard. Stage continuation uses
the same live small model; saved checkpoints are separately restored and
must match bytes before a saved-model reading is accepted.

All ten lesson bars and the math chain bar remain undeclared. The harness
records readings and explicit null registrations until declared criteria
are met. A missing measurement, restoration failure or runtime exception
cannot become a zero, an acquired lesson, or a passing trained artifact.
The earlier two whole-sentence signal nulls remain binding: remaining game
corpora and production probes are built, but game learning is stopped at
that finding. A2's rehearsal therefore reports this incomplete component.
No full-size model is substituted or resized.

H9's four arms and H10's six paired comparisons get frozen task data and
executable measurement scaffolding. Their small smoke budget is one epoch,
two training documents and two held-out documents per arm; differences in
actual work are recorded, never called matched merely because counts were
requested. Existing full-study seeds 0, 1 and 2 and exposure qualifications
remain unchanged. Missing runtime control surfaces are reported as
unavailable; the harness does not implement a second learner to mimic them.

Owned implementation files added or edited in this continuation will be
listed with the final measurements below. Existing test edits begin with
`test_compose_pair_driver.py`, `test_bounded_test_runner.py`, and the
`when-readers-round4a0.json` inventory. Fast whole-file checks precede the
single combined default freeze sweep. The joint weekly exit remains pending
the final compiler/runtime integration.

### 20.3 Rehearsal launch record

Before launching learning, the affected source checks completed 90/90 cases
(89 passed, one inventory metadata failure) in 287.672293958 s, aggregate
peak 4,588,574,672 bytes. The new `ScalarPercepts.join` inventory row lacked
`previous_absolute_time` and `now`; those descriptive fields were supplied
without changing its strict reader assertion. Subsequent new harness/data
checks passed 36 cases and skipped the three unregistered stage models
(default tier), in 9.68 s. The first stage-runner unit check caught the
math corpus's evaluation-only grade; the runner now preserves that split
and never trains those held-out rows.

The H6 run on the merged checkpoint completed 3 cases: the scheduling/
retention check passed; both real next-batch variants failed exact loss-byte
equality. Guard elapsed 70.239827792 s, peak 1,814,170,816 bytes; pytest
67.83 s. The mixed variant alternates complete four-sentence documents
from two sources, keeping the original twelve sentences and model shapes.
Its first and next batches each consumed four sentences, so its resume
crosses the source-document boundary. Assertions remain strict.

The bounded math preflight contains 9 grades, 8 with training rows, 16
training documents and 104 sentences. Maximum sentence size is 25 bytes,
6 whitespace words and a conservative 12 native units, within the existing
32-unit serial word capacity. No full-size model or resized fixture is used.

If a stage producer stops on a runtime defect, later stages are unattempted
continuations, never newly initialized substitutes. Each *unattempted*
lesson may receive one separate standalone diagnostic at the already
declared two-document, one-epoch budget; failed or completed lessons are
not retried. Such standalone diagnostics cannot register stage models.
All H9/H10 small-arm runs retain the same two-document budget, with missing
control implementations recorded as unavailable before any training.
The actual full-study seeds and exposure gates remain unchanged.

The missing local MNIST files and FineWeb shard zero were restored with
`.venv/bin/python bin/prepare_data.py`, which verified their tracked sizes
and SHA-256 values. The training manifest was then regenerated with the
assembly present. Its expanded held-out-only set has 2,456 sentences;
80 occurrences were found in the 3,421,205-sentence developmental assembly.
This is the existing reported overlap scan, not a claim of part-B disjointness.

The first rehearsal execution found a harness setup defect: string math and
study document keys reached `Teacher.where`, which requires native integer
addresses. The adapter now assigns document ordinals and preserves each
original key as `external_id`; a teacher-boundary regression check preserves
both grouping and sentence order. Twelve focused cases passed in 5.02 s.
The failed setup attempts are retained. Replays use their captured original
Python/NumPy/torch RNG states and unchanged budgets, without selecting seeds
or extending a null. The continuous stage replay necessarily repeats its
A1/A2 prelude to reach A3; those repeated measurements are not independent
learning trials. A separate inspection also corrected held-out evaluation:
word-order/determiner training controls read the same ordinary held-out text,
while a bijective vocabulary control transforms held-out answers as well.
No original study run reached evaluation before these fixes.

The replay's first ten lessons reproduced every recorded per-batch input/output
cost and every operator-parameter delta exactly. Thus the A1/A2 prelude did not
introduce another selected initialization. The original setup failures occurred
before a math/study training batch could complete.

The additional H6 CLI diagnostic uses four frozen word-lesson documents,
alternating source labels, in the unchanged `MM_corpus_orders.xml` capacities.
Only dataset/manifest, packing, auto-load/save and `checkpointEveryBatches=1`
are changed in a local XML copy. The identical command is executed twice:
`.venv/bin/python bin/train.py --model <local-resume.xml> --num-epochs 1
--batches 1 --batch-size 1`, with CPU, `MODEL_COMPILE=none` and a local
`BASIC_WEIGHTS_PATH`. Exit status and cursor/exposure advancement are recorded
separately from the already failing exact next-update parity certificate.

Before the new default sweep, the H1 comparison is declared again on this
continuation's final source: the retained 256-case/16-file default against
64 cases/four files, complete selection in both, two workers, 8 GiB per
worker, 16 GiB aggregate, 1,800 seconds per worker and 10,800 seconds overall.
No model work runs concurrently. The comparison follows only if the first
sweep completes without a mandatory stop. The repaired environment and new
source are the boundary of this measurement; neither explains away §19's
prior timeout. No fixture reuse, cap increase or mechanism timing claim is
introduced. Both sweeps preserve every failure independently.

Saved-model observer repairs after the rehearsal freeze: game probes now use
a shared runtime document and runtime question addresses; the operator observer
reads only selected, active programs at `_commit_sentence`, while their native
references are still pinned, and publishes counts only after the original
commit succeeds. An initial saved-M0 evaluation exposed the journal lifetime
error; its failed table is retained and the corrected reader is replayed on
the same saved checkpoints. Each table reader has its own unchanged
8 GiB/1,800-second guard; the serial supervisor only aggregates results and
does not terminate one reader at the same instant as its own timeout. Partial
validation is retained as incomplete and cannot populate full-set use counts.
Evaluation freezes the existing argument-learning flag as well as vocabulary
growth and reads checkpoint class geometry before presenting held-out input.
The same saved word model is re-evaluated without a learning replay. Future
stage runs also persist each completed batch's costs/exposure before a timeout.

### 20.4 Completed bounded rehearsal readings

All commands used CPU, eager model execution, one thread, an 8 GiB process-tree guard and a 1,800-second process limit. The original setup attempts remain failures; their captured RNG states were used only for the documented repair replay.

| Phase | Training documents | Sentences | Batches | Outcome | Criterion |
|---|---:|---:|---:|---|---|
| words | 2 | 2 | 2 | null | undeclared; value 0.0 |
| interpret | 2 | 4 | 4 | null | unavailable; value None |
| clause | 2 | 2 | 2 | null | unavailable; value None |
| ask | 2 | 4 | 4 | null | unavailable; value None |
| verb | 2 | 4 | 4 | null | unavailable; value None |
| modifiers | 2 | 2 | 2 | null | unavailable; value None |
| determiners | 2 | 2 | 2 | null | unavailable; value None |
| adverb | 2 | 2 | 2 | null | unavailable; value None |
| prepositions | 2 | 2 | 2 | null | unavailable; value None |
| relations | 2 | 6 | 6 | null | unavailable; value None |
| math-00 | 2 | 3 | 3 | null | unavailable; value None |
| math-01 | 2 | 6 | 6 | null | unavailable; value None |
| math-02 | 2 | 8 | 8 | null | unavailable; value None |
| math-03 | 2 | 8 | 8 | null | unavailable; value None |
| math-04 | 2 | 12 | 12 | null | unavailable; value None |
| math-05 | 2 | 22 | 22 | null | unavailable; value None |
| math-06 | 2 | 26 | 26 | null | unavailable; value None |
| math-07 | 2 | partial, not persisted | partial, not persisted | interrupted by process limit | not measured |

The word lesson measured stability fraction 0.0. Every later saved lesson reader rejected changed shared conceptual dictionary tensors before claiming a criterion. No threshold was supplied and no artifact was registered. The last training grade was interrupted; the evaluation-only grade was not reached and receives no training. MA3 was not produced. The original child result remains running because the process was terminated; its outer receipt records timeout after 1,801.020528375 s, peak 1,970,343,440 bytes. Future runner versions persist each completed batch so an interruption retains partial exposure. No partial counts are invented for this run.

| Process | Status / exit | Seconds | Peak bytes |
|---|---|---:|---:|
| replay-stages | running / 124 | 1801.020528375 | 1970343440 |
| replay-causal-learned_thought | evaluation_failed / 1 | 27.943362250 | 1649741016 |
| replay-gate-anaphora-treatment | evaluation_failed / 1 | 26.069032375 | 1642056896 |
| replay-gate-verb_reuse-treatment | evaluation_failed / 1 | 20.604904042 | 1720405208 |
| replay-gate-prediction-treatment | evaluation_failed / 1 | 42.089977083 | 1706364192 |
| replay-gate-shuffled_order-treatment | evaluation_failed / 1 | 28.684582042 | 1926532504 |
| replay-gate-shuffled_order-control | evaluation_failed / 1 | 28.134227875 | 1807469832 |
| replay-gate-renamed_vocabulary-treatment | evaluation_failed / 1 | 23.235783209 | 1871990120 |
| replay-gate-renamed_vocabulary-control | evaluation_failed / 1 | 33.280430084 | 1929268656 |
| replay-gate-mint_bind-treatment | evaluation_failed / 1 | 38.798760042 | 1759841664 |
| replay-gate-mint_bind-control | evaluation_failed / 1 | 39.963697708 | 1740852632 |
| causal-direct_answer | unavailable / 0 | 4.364600292 | 402359448 |
| causal-equal_actual_compute | unavailable / 0 | 4.369102459 | 402097280 |
| causal-reconstruction_only | unavailable / 0 | 4.368825083 | 402310320 |
| gate-anaphora-control | unavailable / 0 | 4.640222833 | 401802392 |
| gate-verb_reuse-control | unavailable / 0 | 4.379710875 | 401933488 |
| gate-prediction-control | unavailable / 0 | 4.383959250 | 402113664 |

Study-arm exposure and query work (successful runtime charges, including discarded trials; not total compute):

| Arm | Updates | Sentences | Query charges | Result |
|---|---:|---:|---:|---|
| replay-causal-learned_thought | 6 | 6 | 415 | evaluation_failed |
| replay-gate-anaphora-treatment | 6 | 6 | 541 | evaluation_failed |
| replay-gate-verb_reuse-treatment | 4 | 4 | 272 | evaluation_failed |
| replay-gate-prediction-treatment | 6 | 6 | 551 | evaluation_failed |
| replay-gate-shuffled_order-treatment | 6 | 6 | 323 | evaluation_failed |
| replay-gate-shuffled_order-control | 6 | 6 | 478 | evaluation_failed |
| replay-gate-renamed_vocabulary-treatment | 6 | 6 | 284 | evaluation_failed |
| replay-gate-renamed_vocabulary-control | 6 | 6 | 448 | evaluation_failed |
| replay-gate-mint_bind-treatment | 4 | 4 | 336 | evaluation_failed |
| replay-gate-mint_bind-control | 4 | 4 | 331 | evaluation_failed |
| causal-direct_answer | — | — | — | unavailable |
| causal-equal_actual_compute | — | — | — | unavailable |
| causal-reconstruction_only | — | — | — | unavailable |
| gate-anaphora-control | — | — | — | unavailable |
| gate-verb_reuse-control | — | — | — | unavailable |
| gate-prediction-control | — | — | — | unavailable |

Rehearsal source SHA-256: `a98f30eb93e7d7838b09aac261641acac4bfa345d0ce64a8957a342ca8ffccdb`. Original initialization identities replayed after the setup fix:

- `replay-stages`: `ce3b6733a982e8e805f1346424ba2222afeb0d8defe30059fe44cabe8d02d6db`.
- `replay-causal-learned_thought`: `d152b0a7fcc66ca625a4dff2cb7aebf65d64f66f6bbb1aa5a455a174c903a0e9`.
- `replay-gate-anaphora-treatment`: `ed25e8bc3f20db4a3e53efb849d0b80eab5411b920dff7b01625afe579addb7b`.
- `replay-gate-verb_reuse-treatment`: `129d4e054e5cd74c47878f749bff905e9a1867f6286d79ac9d4f3633eb6cb724`.
- `replay-gate-prediction-treatment`: `3d3e8910ecc417aa7be5aaba51ffa52e0116eb88b30c82dc27366b1e3cb2c6ba`.
- `replay-gate-shuffled_order-treatment`: `4b81951cd64be07aeb09251b1934467fa4f26c8aec231ec543acfb36bfc17747`.
- `replay-gate-shuffled_order-control`: `388d32b4366edd12f0323fe5a5f0ac31f833d0030d2c6c338d4e37cbefed18a2`.
- `replay-gate-renamed_vocabulary-treatment`: `d62f4dbd66daa883c8019975876a600185966c73cd00bc741cb2ffc8cf4e71e2`.
- `replay-gate-renamed_vocabulary-control`: `5359311a7ed788654fce7a10e6895056c035a0e23533993dbe3ff4804d09f20e`.
- `replay-gate-mint_bind-treatment`: `b5a273ca2a234454e37117b5e2505e2997a2d314d9bc8ac548715533a7d93f20`.
- `replay-gate-mint_bind-control`: `b3d4f6bb6dd1e1699385bdfc56ab08a0bf5cf208629f0dd5a87eeb67b46e3a1d`.

Checkpoint identities (local weights, not registrations):

- `M0`: `5c4e3b692e0b68676456a441c80cd244003a04cf8e8244c3c10721c43f8c5477`.
- `MA1`: `c77d4f5d5d16e4ec48576f874009ad3ae60dda5251a835a9649c36279bbe1b9f`.
- `MA2`: `deac8c7bb643067ffc31a7d81e87232769e518192b286fbb596c8d3f074abb3b`.
- `MA3`: not produced.

Frozen data reproduced byte for byte: 90 compared files, zero differences. The original three game variants were unchanged. New template variants extend the inventory to 31; intended-rule labels remain diagnostics. The H9/H10 training set and learning-heldout set each contain 320 tasks; causal-heldout contains exactly 256. Content and manifest hashes:

- `data/games/manifest.json`: `694059aab1b6bc6c315198244239b9f6ecdcaa0e439ab0c25c2f0e0524c7a01a`.
- `data/games/inventory.json`: `a82372b716f47b52104845324a42201abf652d292bdaf7bb06f78481cbc399ad`.
- `data/eval/stage-models.json`: `94db779660672ac98a8cb63a2c41a12245766580d5c4b213da09127228d70a61`.
- `data/eval/studies/recipe.json`: `29ab35509da83ad1ab5467c427427448811007cc13dceae40789d35926b31c51`.
- `data/eval/studies/manifest.json`: `75d340009202dd968ad55d1bf21eceafb793976c7a3d2c93602e6b942ce6fcad`.
- `data/eval/studies/protocol.json`: `229586ae2c6665d6feb0f8f8d78ba09875a4f07b122810542016f257713804dd`.
- `data/training-corpus.json`: `687922eb2c5d38b93999464619f3f5379e2ca428b65ef61b6fb5785040dffb31`.
- `data/eval/heldout-cost.json`: `9ae11509855b0cb1913426db5d594e4e1e0e3638c914037b781bb18a7269ef83`.

The corrected saved-word reading has **2 admitted forms, 0 stable, 0 named,
0 denoting**, stability fraction **0.0**. All **92** argument radii and every
entry of the 92-by-92 cosine table are zero before held-out presentation.
The initial reader reported four admitted forms and three nonzero radii after
probe presentation; those are contaminated post-probe readings and are not
checkpoint geometry. The corrected measurement restores the same checkpoint
and trains nothing. Its process completed in 7.709020209 s, peak
1,040,336,600 bytes. Threshold remains undeclared, registration null.

The stage producer's live, pre-evaluation parameter observations are retained
below. They do not replace byte-exact saved-model readings. `none` means the
rule has no owned parameters; relative change is undefined for a zero-norm
baseline. MA3 is absent because the rehearsal timed out.

| Rule | Owned parameters | Relative change M_A1 | Relative change M_A2 |
|---|---:|---:|---:|
| divide | 0 | none | none |
| descend | 0 | none | none |
| gloss | 0 | none | none |
| intersection | 0 | none | none |
| union | 0 | none | none |
| interpret | 0 | none | none |
| sign | 288 | none | none |
| part | 288 | none | none |
| whole | 288 | none | none |
| equal | 288 | none | none |
| implies | 288 | none | none |
| generic | 288 | none | none |
| ask | 288 | none | none |
| not | 0 | none | none |
| non | 0 | none | none |
| conjunction | 0 | none | none |
| disjunction | 0 | none | none |
| sum | 0 | none | none |
| product | 288 | none | none |
| lift | 4096 | 0 | 1.09442186664e-05 |
| verb | 288 | none | none |
| compound | 288 | none | none |
| adverb | 2624 | 0 | 0 |
| lower | 288 | none | none |
| definite | 288 | none | none |
| universal | 288 | none | none |
| preposition | 288 | none | none |
| surface | 288 | none | none |
| transpose | 0 | none | none |
| lower_unary | 288 | none | none |
| definite_unary | 288 | none | none |
| preposition_unary | 288 | none | none |
| surface_unary | 288 | none | none |
| compound_unary | 288 | none | none |
| bind | 0 | none | none |

### 20.5 Final evaluation and verification boundary

The post-observer fast command selected `test_game_probes.py`,
`test_stage_models.py`, `test_nanochat_grammar_eval.py`, `test_lesson_stages.py`
and `test_study_harnesses.py`: 25 passed, 3 default trained skips, one failure,
15.43 s. The failure occurred during Inductor artifact loading in the configured
`~/.cache/torchinductor` cache; a direct temporary-file check confirmed
`PermissionError: Operation not permitted` under the restricted execution
environment. It is retained as an incomplete fast check, not a model failure
regraded by selecting another initialization. The remaining scoped checks
and sweeps use the bounded runner with access to its existing compiler cache.
After the pinned-journal observer correction, the complete stage/probe files
passed 6 cases with 3 default trained skips in 4.14 s.

The initial saved-model table attempt finished in 14.984667416 s, peak
950,536,496 bytes, with 35 declared rule rows. M0 exposed the observer's
expired-reference setup defect; MA1 and MA2 were rejected by strict restoration;
MA3 was explicitly absent. The corrected M0 reader repeats inference only
on the same saved file. The live-parameter table above is preserved separately.

The MA1 interval probe attempt exited 1 in 5.145660042 s, peak 788,350,464
bytes, before its 72+4 probe readings: the same six shared dictionary aliases
failed byte identity. It supplies no fabricated zero-valued probe metrics.
The initial CLI invocation exited 1 in 11.854784166 s, peak 992,037,216 bytes.
Its one training batch completed and checkpointed before `_reconstructionReport`
failed to stack variable-length input tensors (lengths 5 and 6). Its checkpoint
therefore permits the intended second invocation; the reporting failure is
retained and no runtime code is changed.

The identical second CLI invocation loaded the checkpoint and exited 1 in
10.058625042 s, peak 1,001,523,600 bytes. `--batches 1` is a cumulative cap:
the resumed training loop performed zero further batches and the final report
again failed on lengths 5 and 6. Both checkpoint hashes were
`4f9fe08ed9db84cbe60b1bb88528e3b1f8feb7c0f532125f06834558cba4cfce`;
training-step count, total batches, epoch batches and cursor steps remained
1, 1, 1 and `[1]`, respectively. Thus this command demonstrated loading and
cap preservation, not resumed advancement or next-update equality. It does
not replace the failing H6 uninterrupted-versus-restored test.

The final operator table used source
`60004b6d4eee5067e2f465858884db9f3137479cdbd201438306b064d5bbc016`.
The corrected M0 observer completed 489 of the frozen 999 validation
sentences before its unchanged 1,800-second guard stopped it at
1,801.017922834 s, peak 1,075,627,808 bytes. MA1 and MA2 exited 1 in
5.161082750 s / 788,956,720 bytes and 5.420678833 s / 792,479,280 bytes,
respectively, at the six-alias byte-identity check. MA3 was absent. All 35
declared rule rows remain in the table, with full-set use counts and probes
unavailable. M0 partial counts are explicitly partial; the live parent
parameter observations in §20.4 remain separate evidence.

The H10 referent collector now reads the native occurrence-close audit,
copying selected word identities before the pinned journal expires. It
requires a fresh first referent, two distinct mention occurrences and the
same resolved referent; an operator name never satisfies this measurement.
Ambiguous instrumentation yields unavailable, and a missing admitted form
yields false. The observer restores the original commit binding and ignores
stale audit objects. Prediction-head evaluation runs without gradients.
The saved M0 baseline command was
`.venv/bin/python bin/eval_studies.py evaluate --config data/MM_corpus_lessons.xml
--checkpoint <M0.ckpt> --manifest data/eval/studies/manifest.json --control ordinary
--family mint_bind --limit 2 --output <reading.json>`, under CPU, eager evaluation,
one thread, no automatic load/save, 8 GiB and 1,800 s. It exited 0 in
10.312872583 s, peak 846,972,440 bytes, on source
`72a4aa561a4159988a1bfea27f20c9d289b4cc12e3edc768fcdd22975f42a3bb`.
The two tasks were `mint_bind-09af68b6a36b6ac8` and
`mint_bind-0b018fbe060b21ae`: referent accuracy 0/2 (forms not admitted),
answer accuracy unavailable, input reconstruction error 0.10000000149011612.
Prediction activation MSE was 0.2900126576423645 for ordered and shuffled
contexts, 0.29018527269363403 for context-free input, identically across
seeds 0, 1 and 2; two targets, bank size 141, both activation variances zero.
The query meter charged 236 units over 14 successful attempts: 208 bracket,
26 serial-word-loop and 2 space-read. This is a collector baseline, not a
trained comparison or a measure of total compute.

The registry retains 54 null phase artifacts. It records outcomes for the
ten lessons, three stage models, both obsolete combined XOR producers and
31 game variants. Each game's training remains unattempted after §19.8's
two signal nulls. The current numeric XOR class tests are not producers for
the old combined grammar criteria; the separate grammar inverse test still
exists and is not retired by this registry correction.

The final focused command was `env -u RUN_TRAINED -u RUN_SLOW make test`
with the complete files `test_study_harnesses`, `test_stage_models`,
`test_game_probes`, `test_game_corpus`, `test_lesson_stages`,
`test_training_corpus`, `test_math_corpus`, `test_phase_artifacts`,
`test_bounded_test_runner`, `test_interval_log`, `test_nanochat_grammar_eval`,
`test_operators_round4a0` and `test_doc_links` (all under `test/`, suffix `.py`).
All 178 selected cases completed: 163 passed, six marked skips and nine
documentation-link failures, 240.111877792 s, peak aggregate 6,540,761,928
bytes. Compilation remained enabled, and the earlier compiler-cache case
passed with cache access restored.

The nine link failures came from untracked, Git-ignored numbered copies of
retired documents, not the current tracked documentation. All ten `* 2.md`
copies under `doc/plans` and `doc/specs` were moved byte for byte to
`scratch_icloud_notes_2026_11_13/`, retaining their relative paths: 404,001
bytes, audit SHA-256
`ce0ecf604e37811e6faf577b62fffe4aacab03e1974556cd7b5bad7de3fb64ff`.
No link assertion or tracked document was removed. The seven notes in
`scratch_preexisting_notes/` are separate and unchanged. The complete
`test/test_doc_links.py` then passed all 51 cases under the bounded runner.
There were no numbered duplicate Python/configuration/manifest files under
`bin`, `test` or `data` at freeze.

### 20.6 Frozen source and landing verification

The final source snapshot has 1,154 entries; SHA-256 of its compact,
sorted JSON is
`1ca4a02418452145d854d1b4b8d957764941399e74d964d8e221729dd64f0e82`.
`data/corpus/assembly-01/manifest.json` was present. The last pre-freeze
fetch left `origin/main` at `ab5176e0e2` and `origin/testing` at
`b6370617d3`; the working branch was `testing`, HEAD `91e71c40cf`.
Compilation checks passed for all nine added or edited `bin/` modules.

Owned implementation files in this continuation:

- Added `bin/eval_games.py`, `bin/eval_lessons.py`, `bin/eval_operators.py`,
  `bin/eval_studies.py`, `bin/stage_models.py` and `bin/study_corpus.py`;
  edited `bin/corpus_lessons.py`, `bin/eval_nanochat_grammar.py` and
  `bin/training_corpus.py`.
- Added the stage protocol and six study files under `data/eval/`, the game
  inventory and 28 template recipes with frozen rows; updated the game,
  training and held-out-cost manifests. The original three game variants
  and stored papers were preserved.
- Edited `test/bounded_tests.py`, `test/phase_artifacts.py`,
  `test/fixtures/phase-artifacts.json` and
  `test/fixtures/when-readers-round4a0.json`.
- Existing test edits: `test/test_bounded_test_runner.py`,
  `test/test_compose_pair_driver.py`, `test/test_corpus_checkpoint.py`,
  `test/test_game_corpus.py`, `test/test_lesson_stages.py`,
  `test/test_nanochat_grammar_eval.py` and `test/test_phase_artifacts.py`.
  New tests: `test/test_game_probes.py`, `test/test_stage_models.py` and
  `test/test_study_harnesses.py`.
- Updated this plan, `doc/Testing.md`, `doc/Training.md` and `todo.md`.
  No learner-runtime file was edited in this continuation.

The complete default command is `env -u RUN_TRAINED -u RUN_SLOW make test`,
with only `TEST_ARGS=--run-dir <new-local-receipt-directory>` added. It
selects 7,235 cases with no deselection: two workers, 256 cases / 16 files,
8 GiB per worker, 16 GiB aggregate, 1,800 seconds per worker and 10,800
seconds for the suite. No other model work runs concurrently. The two
mandatory stop selectors in the original request are monitored during the
run; a failure stops further work rather than triggering a retry.

### 20.7 Default sweep result

The default sweep completed all **7,235 / 7,235** selected cases on the
frozen source: **6,885 passed, 328 skipped, 21 failed, one XPASS**.
Bounded-runner exit 1 (`test_failure`), make exit 2; runner elapsed
**4,655.268484375 s**, wrapper elapsed **4,655.550953583 s**.
There were **151 workers** including resource-boundary recycling and the
large-memory phase. Peak worker footprint was **7,466,556,184 bytes**;
peak aggregate footprint **11,485,425,928 bytes**. No worker timed out,
exceeded its memory cap, or aborted. The frozen source was checked again
after completion and was unchanged. SHA-256 of the compact ordered selected
node-ID list is
`fba02a255721ec9963780d2d2dd181845c477b80a6f1f6dfc4ceb9c17219b308`.
Counts use one outcome per selected node ID; three extra successful reports
for `test_use_flags.py::TestOrthogonalFlags::test_flags_match_expected`
do not inflate the total.

Both mandatory stop gates passed:
`test/test_mm_xor.py::TestMMXorConvergence::test_convergence` in
0.810372375 s and
`test/test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class`
in 0.221694542 s. The last large-memory worker, containing
`test/test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct`,
also passed: worker elapsed 872.769260083 s, peak 4,536,210,496 bytes.
Thus the predeclared H1 comparison is eligible; this does not regrade any
default failure.

Five failures are runtime/assertion findings, with their assertions intact:

| Selector | Observed failure |
|---|---|
| `test/test_identification_evidence.py::test_native_sentence_carries_paired_identification_to_its_row` | Evidence `(0.5487958192825317, 0.0)` instead of `(0.8, 0.6)`. |
| `test/test_per_row_hard_reset.py::test_runEpoch_dispatches_per_row_reset_per_batch` | `MemoryConsolidation._preflight`: consolidation exceeds the concept row pool. No capacity was changed. |
| `test/test_reading_round.py::test_native_read_sites_carry_one_token_across_three_documents` | The read token has no retained occurrence: `entry.occurrences == ()`. |
| `test/test_reasoning_cde_model.py::TestReasoningCDEModel::test_answer_query_degrades_gracefully` | The tied reconstruction bank was not staged after concept admission. |
| `test/test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row` | Native feature columns `{0}` instead of `{4, 402}`. |

Their pre-setup RNG identities, in the table's order, are:

1. `578194a6360300a58eb288cf2b060f313a22c2ccaa2dec76b5027918a50a9bf3`.
2. `77d36cf733bb5850b81a8f45c05b9da3ccee57f36f7ca711f7f1d8ea5fd3cc7f`.
3. `2d500816ebda100e6b5ba317fc989317af006c23cfec6ecd629c171d18e2bf71`.
4. `e722e3fcf3affdd1be01c249a0f36242202d488592e9d04880f88f665d271cc2`.
5. `12da30ba4a9acb906cca40f168cce9d45ce095be1e83a493999a72959050d568`.

All five used `PYTHONHASHSEED=2498025303`; torch 2.15.0, executorch
1.6.0, pytest 9.2.0. These are captured streams, not selected learning seeds.

The other **16 failures** are the still-missing original
`doc/benchmarks/2026-10-31-item2-5-completion/addendum-10/native-mean/whole-pullback-1.pt`
capture. The selectors under `test/test_whole_presence.py` are:

- `test_whole_presence_forward_and_weight_pullback_are_bounded` and
  `test_whole_presence_tiny_mass_jvp_and_weight_gradcheck`, each with
  `captured-dtype0`, `captured-dtype1`, `smallest_normal-dtype0` and
  `smallest_normal-dtype1` (eight cases).
- `test_whole_presence_compiled_pullback_matches_decided_bounds_at_tiny_mass`,
  the Cartesian product of `aot_eager` / `inductor`, `captured` /
  `smallest_normal`, and `dtype0` / `dtype1` (eight cases).

The XPASS is
`test/test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input`.
Its existing `xfail(strict=False)` explicitly permits this result; the
historical bar remains unchanged. It is not counted as an additional failure.

### 20.8 H1 completed comparison

The second command was `env -u RUN_TRAINED -u RUN_SLOW make test` with
`TEST_ARGS=--batch-size 64 --max-files 4 --run-dir <new-local-receipt-directory>`.
It retained two workers, 8 GiB per worker, 16 GiB aggregate, 1,800 seconds
per worker and 10,800 seconds overall. Both complete sweeps used the same
ordered 7,235-case selection and source hashes recorded above.

| Reading | Default 256 / 16 | Comparison 64 / 4 |
|---|---:|---:|
| Completed | 7235 | 7235 |
| Worker processes | 151 | 271 |
| Elapsed seconds | 4655.268484375003 | 4793.822899542007 |
| Peak worker bytes | 7466556184 | 7616601240 |
| Peak aggregate bytes | 11485425928 | 11027918464 |
| passed | 6885 | 6886 |
| skipped | 328 | 328 |
| failed | 21 | 20 |
| xpassed | 1 | 1 |

The comparison elapsed difference was +2.976%.
Both runner results are `test_failure` (exit 1; make exit 2), with no
incomplete selection, worker timeout, memory-cap termination or aborted
worker. Both mandatory stop gates passed again: MM XOR convergence in
0.829557958 s and the shared-concept denotation gate in 0.207299209 s.
The retained default remains **256 cases / 16 files**. This is one ordered
comparison using the existing compiler cache and fresh unseeded learning;
the different failure outcomes also matter. It is not an isolated estimate
of construction, reset, compilation, forward or backward costs, and no
fixture-reuse or component-speedup claim is made.

The comparison repeats the identical paired-evidence values, concept-row
pool exhaustion, native feature-column mismatch and 16 missing-capture
failures from §20.7. It also fails
`test/test_serial_mode_integration.py::test_serial_mode_does_not_slow_short_streams`:
`MemoryConsolidation.recurrence_groups` raises because angular recurrence
exceeds the constructed `recurrenceCapacity`. This occurs during the
baseline forward loop, before the speed assertion; it supplies no completed
speed comparison. That case passed in the default sweep.

Conversely, `test_native_read_sites_carry_one_token_across_three_documents`
and `TestReasoningCDEModel::test_answer_query_degrades_gracefully` passed in
the comparison (6.682046458 s and 1.295084708 s). Their default failures
remain failures. Run order, fresh random streams and process grouping were
not separately controlled, so the differing outcomes are not attributed
to one cause. The same permitted non-strict XPASS occurred in both runs.

The comparison runtime-failure streams used `PYTHONHASHSEED=1866004773`
(same library versions as §20.7). SHA-256 identities:

- `test/test_identification_evidence.py::test_native_sentence_carries_paired_identification_to_its_row`: `02fa1fbf8fd42ac6801f3257d8c374cdf3e084828ede3d229caf865a5f723a56`.
- `test/test_per_row_hard_reset.py::test_runEpoch_dispatches_per_row_reset_per_batch`: `6e2630baed59dfa57b584a3b94def96ca05bfd3c4a4ba2d81f3d86698dce771e`.
- `test/test_serial_mode_integration.py::test_serial_mode_does_not_slow_short_streams`: `0ed36dc96ffff28054b05729b26b1d5ec81edbe3f142bb571638dbe27a3db9db`.
- `test/test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row`: `0e1dc9b28f61f116b27a64bf380e69402fbbdc62817c875e4039be842599d09f`.

### 20.9 Remaining integration work

All authorized test-development rounds now have implementation and recorded
bounded outcomes. Item 0.5 is **not a green exit**. No learner mechanism was
changed to hide a failure, no fixture capacity or assertion was relaxed,
and no unsuccessful learning arm was extended or selected away. Remaining
requirements for the later integration pass are:

- Repair byte-exact saved-model restoration and next-update parity; finish
  a CLI continuation that advances after restoration, with the reporting
  failure and cumulative-cap behavior above preserved as evidence.
- Supply the original whole-pullback capture and resolve the named runtime
  failures and cross-run differences without substituting a capture.
- Define the unavailable study controls, opaque routing implementation and
  acceptance bars; qualify a whole-sentence game learning signal. Until
  then the game and stage registrations remain null and Part B stays gated.
- Resolve the recorded A3 rehearsal and full operator-reading timeouts at
  a separately declared measurement boundary; this continuation did not
  extend either budget or rerun their learning.
- Run the joint default/weekly exit on the final item-1 integration hash.
  The accepted checkpoint used here is not that final joint hash.

Stored papers remain stored, and the seven pre-existing planning notes
remain available in `scratch_preexisting_notes/`. Publication is to
**testing only**; local main and the parent repository pointer are outside
this landing.

### 20.10 Publication identity

Implementation and measured return commit on `testing`: `170ea508ec5e40a264a34d37d07b730c0f9142dc`.
The following metadata-only commit records this identity; it changes no
validated executable, test, fixture, corpus or configuration. The source
snapshot was rechecked after committing and still has 1,154 entries, SHA-256
`1ca4a02418452145d854d1b4b8d957764941399e74d964d8e221729dd64f0e82`;
`data/corpus/assembly-01/manifest.json` remains present.

Tracked blob size (`git ls-tree -r -l`): before this continuation,
1,271 files / 94,586,951 bytes at `91e71c40cf`; implementation
1,344 files / 96,155,856 bytes at `170ea508ec`.
Stored research papers have no diff in this continuation. The seven original
planning notes remain ignored local files, preserved for inspection.

The H1 wrapper completed in 4,794.725143750 s. Its final grammar-reconstruction
worker passed in 874.446467084 s, peak 4,542,813,272 bytes. Both complete
sweep source snapshots match the implementation source above. Publication
uses a normal push to `origin testing`, with no main update or force push.

## 21. Review of §19 and §20 (Claude, 2026-11-13)

Alec, 2026-11-13: "please sync testing branch there so that you can do
what you need, integrate with your docs branch, push back to github on
testing branch, and then give me code to paste to codex in 0.5 thread".
Reviewed on `testing` at `d3c76cf98fc2`.

**Verified.**

- `testing` holds the accepted item 1.8 checkpoint (`91e71c40cf` merges
  `ab5176e0e2bc`). It was 10 commits ahead of `main` and none behind, so
  `main` can move to it without a merge. Seven of its files are also open
  in the first machine's uncommitted 1.8 work: two one-line `IntervalLog`
  hooks (`Forgetting.py`, `Models.py`), the document identities of
  `corpus_lessons.py`, and four files whose mode alone changed.
- No learner mechanism changed. The source changes are evaluation
  harnesses and data tools, the two `IntervalLog` hooks, which do nothing
  unless `intervalLogEveryBatches` is set, and the evaluator's capture of
  held-out word scores. The changed tests are setup ports with their
  assertions kept, as §19.5 and §19.14 describe.
- *The written tier.* Claude read the twelve keys of §19.1. All twelve
  targets are right for their game: the negations (`will not`, `cannot`,
  `is not`), the two conjunctions joined by `, and`, and the determiner
  chains (`The horses`, `All the birds`). Two pairs repeat a sibling's
  frame with other names (`not-written-000009` and `000011`;
  `conjunction-written-000006` and `000008`), so the tier adds less
  variety than it could; within a transfer split that is by design.
  Accepted.

**Repaired here.** The sixteen `test_whole_presence.py` failures had one
cause: the test read its capture from `doc/benchmarks/`, which git
ignores, so the file never reached the second machine. The original
exists on the first machine in 24 copies with one SHA-256,
`20ba5b9649d35a762a7c882e86400a6cc6a5d8235ccfd6f8ae9adb99ce435452`. It is
now tracked at
`test/fixtures/2026-10-31-item2-5-whole-presence/whole-pullback-1.pt`, and
the test checks that hash before loading (`2746737f49`). Nothing was
regenerated. The file passes all 37 cases on the first machine.

**The other five failures of §20.7** are learner-side and stay with item
1.8. Two are among the old failures the 1.8 return already lists
(`test_reasoning_cde_model.py::TestReasoningCDEModel::test_answer_query_degrades_gracefully`,
`test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row`),
and `test_reading_round.py::test_native_read_sites_carry_one_token_across_three_documents`
is repaired in the first machine's 1.8 work. They were not rerun here.

**For the integration with 1.8** (no action in this round):

- `IntervalLog` reads `truth_layer` and the LTM forgetting count; item
  1.8's reduction changes both, so it is ported when 1.8 lands.
- The 31 games are named for today's operators. Alec's operator list of
  2026-11-13 (1.8 specification §22.8) withdraws several of them
  (`equal`, `generic`, `definite`, `universal`, `verb`, the one-operand
  faces). The games stay; their names and diagnostic labels are mapped
  then.
- The lesson-gate proposal, §18.4, names operators by today's list too.

**For Alec.** The decisions of §19.13 stand as Codex wrote them (H2, H3,
H5 and H6, H8, H12, the games' signal). The written tier is accepted. My
docs branch is merged onto `testing` as `9630b8b371`.

## 22. Codex text: test data for images (Claude, 2026-11-13; for Alec to paste on the second machine)

~~~
Item 0.5, a testing round on the second machine. Testing only: change no
learner mechanism, no configuration of the model, no capacity and no
fixture's sizes. Never push or force-push main; push testing only, and only
at the end. Claude's review of your return is §21 of
doc/plans/2026-11-09-item0-5-test-development.md.

0. Sync. git fetch origin; git switch testing; git merge --ff-only
   origin/testing. HEAD must be 2746737f49 or a descendant of it. That
   brings Claude's docs merge (9630b8b371) and the whole-presence capture
   (2746737f49). Run test/test_whole_presence.py: all 37 cases must pass
   here, where the 16 captured cases failed before (§20.7). Report the
   result.

1. Image test data. The first machine is building a path for the number
   modality that reads XOR and 28 by 28 images by one reader (designed in
   §22.9 of the item 1.8 specification, which is on the first machine and
   not yet on testing; you do not build that path). It needs fixed
   two-level images. Build only the data.

   a. A builder, bin/image_corpus.py (default name), with a freeze command
      like the other builders:
      .venv/bin/python bin/image_corpus.py freeze --seed 20261112 --out data/images
      It writes generated 28 by 28 images whose every pixel is 0 or 1, in
      ten classes labelled 0 to 9 by integer, with no word anywhere.
      Default shapes, one per class: a horizontal bar, a vertical bar, a
      falling diagonal, a rising diagonal, a plus, an X, a square outline,
      a filled square, a ring and a filled disc. Position, size and stroke
      vary within bounds drawn from the declared seed. Default counts per
      class: 100 train, 20 validation, 20 test. No image appears twice
      across or within splits. Store each split as JSONL, one image per row
      with its label, its split and its 784 pixels packed as 196 hex digits
      in row-major order (default). Write a manifest with the seed, the
      command, the counts and each file's SHA-256, as data/games and
      data/lessons do. The builder reproduces every file byte for byte.

   b. A two-level MNIST subset, derived and never tracked. A function in
      the same module reads the prepared MNIST files (bin/prepare_data.py;
      require them through test/local_data.require_mnist as the MNIST
      tests do), maps a pixel to 1 when its value is at least 128 of 255
      and to 0 otherwise, and takes a fixed prefix: the first 1000 rows of
      train and the first 200 of test (default). Track only a manifest of
      that rule with the SHA-256 of the resulting arrays and labels.

   c. One loader for both: image_corpus.load(name, split) returns a
      float tensor [N, 28, 28] of 0s and 1s and a long tensor [N] of
      labels, for name 'shapes' or 'mnist'. The first machine's
      certificates will call it.

   d. test/test_image_corpus.py: the freeze reproduces byte for byte; every
      pixel is 0 or 1; shapes and label counts are as declared; the splits
      are disjoint; the loader agrees with the files; the MNIST subset
      reproduces its manifest hashes, and skips with the preparation
      command when the files are absent. It runs in the default sweep and
      constructs no model.

   e. Add the corpus to data/training-corpus.json's part A only if its
      schema has a place for an evaluation set; otherwise list it nowhere
      and report. A short paragraph in doc/Training.md under the staged
      training corpus says what the images are and how to rebuild them.

2. Run the new test file, test/test_whole_presence.py, and the existing
   corpus tests (test_math_corpus.py, test_game_corpus.py,
   test_lesson_stages.py, test_training_corpus.py, test_prepare_data.py)
   before pushing. Run no full sweep in this round.

Return as §23 of the plan: each default kept or changed, the counts, the
file hashes, the test results with times, and the pushed commit. Then push
testing.
~~~

## 23. Image-data testing return (Codex, 2026-11-13)

### 23.1 Sync and scope

Fetched origin, switched to `testing`, and fast-forwarded from
`d3c76cf98fc28df0f637f4ff27163407d4cf0adc` to
`0693c8e98fab23d4fc625c9a3dc48ad1b1108710`. Verified that
`2746737f49` is an ancestor. Read §21. The original tracked whole-presence
capture has SHA-256
`20ba5b9649d35a762a7c882e86400a6cc6a5d8235ccfd6f8ae9adb99ce435452`.
The immediate bounded check passed **37/37**, including all sixteen cases
that previously lacked this capture: **25.775733 s** runner elapsed,
**20.729941 s** worker elapsed, **6.445119 s** summed test-call time,
peak worker/aggregate memory **629,017,312 bytes**. Command:
`env -u RUN_TRAINED -u RUN_SLOW make test 'TEST_ARGS=test/test_whole_presence.py --run-dir <new-local-receipt-directory>'`.

This round adds only the image-data builder/loader, frozen data, its tests
and documentation. It changes no learner mechanism, model configuration,
capacity, existing fixture size or assertion. The first machine's number
reader is outside this change. Stored papers and the seven local planning
notes are preserved. No full sweep ran in this round, as §22 directs.

### 23.2 Defaults and reproducibility

All requested defaults are kept:

| Default | Result |
|---|---|
| Module and shape command | `bin/image_corpus.py`; `.venv/bin/python bin/image_corpus.py freeze --seed 20261112 --out data/images` |
| Image dimensions and levels | 28 × 28, every pixel 0 or 1 |
| Labels | Integer 0–9; no words rendered or supplied as image targets |
| Classes, in label order | Horizontal bar, vertical bar, falling diagonal, rising diagonal, plus, X, square outline, filled square, ring, filled disc |
| Counts per class | 100 train, 20 validation, 20 test |
| Total shapes | 1,000 train + 200 validation + 200 test = **1,400**, all globally distinct |
| Serialization | One image per JSONL row, fields `label`, `split`, `pixels`; 196 lowercase hex digits for 784 pixels |
| Pixel order | Row-major, first pixel in the most significant bit of the first byte; zero padding retained |
| MNIST threshold | Original integer pixel ≥128 maps to 1, otherwise 0 |
| MNIST selection | First 1,000 train and first 200 test rows after the header, original order, no shuffle or balancing |
| Shared loader | `image_corpus.load(name, split)` for `shapes` or `mnist`; CPU float32 `[N,28,28]` and int64/long `[N]` |

The previously unspecified shape bounds are now explicit in the manifest:
integer radius 4–11 (bounding-square side 9–23, odd), centers from radius
through 27−radius on each axis, and stroke parameter 1–3, all inclusive.
Axial/diagonal bands have width 2×stroke−1; square and ring borders use
stroke as their inward thickness. Filled square/disc have no stroke draw.
Y increases downwards, so the falling diagonal runs upper-left to
lower-right. Python `random.Random` (MT19937) draws in class order, then
train/validation/test order; duplicate pixel arrays are rejected globally,
regardless of label. The seed was declared before generation and unchanged.
The manifest records raster rules and row order. Commands use canonical
repository-relative output names, with no timestamp or temporary path in
the generated bytes.

The shape freeze needs no prepared MNIST. The separate command
`.venv/bin/python bin/image_corpus.py freeze-mnist --out data/images`
authenticates the prepared CSVs against `data/external-files.json` and writes
only `mnist-manifest.json`. Both derived pixel and label arrays remain in
memory. Pixel hashes use uint8 `[N,28,28]` C-order bytes; label hashes use
signed little-endian int64 `[N]` C-order bytes. The loader checks the frozen
shape file hash or the derived MNIST hashes before returning tensors.
The MNIST tests call `test/local_data.require_mnist`; absent data skips with
`.venv/bin/python bin/prepare_data.py`. A simulated absent-data test verifies
that skip and the application loader's preparation error without fetching.

Part A of `data/training-corpus.json` has training stages A1–A3 and no
evaluation-set slot. Its separate top-level `heldout` list is not a Part A
slot. Per §22(e), the image corpus is **not registered anywhere in that
manifest**; the manifest is unchanged. A paragraph under the staged corpus
in `doc/Training.md` explains the independent image data and both commands.

### 23.3 Counts and SHA-256 identities

All five tracked data files are under `data/images/`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `train.jsonl` | 236000 | `f72633edbb201dff56d8635c7d65d864f17ca394b5e84e63979c00a0e3eafa50` |
| `validation.jsonl` | 48200 | `b92d17161622daaeb18ef86f7b691a89664234df7724ff950973e07ed08f68c8` |
| `test.jsonl` | 47000 | `75f16d687917ba97acb05af9931a475880bf28692ccc2c5e32325833cbfa33d7` |
| `manifest.json` | 1878 | `a2dc9928f203454fecdad9549254af8f4f72c08ba8c07ef9ba3e0a66b50c6c60` |
| `mnist-manifest.json` | 1550 | `711bd6fb0a296c6117c2ff13e267d164c91606e442be0b383c48d4c54b216191` |

Every shape class has exactly the split counts above. MNIST prefix label
counts, in label order 0–9, are:

- Train: `[97, 116, 99, 93, 105, 92, 94, 117, 87, 100]` (1,000).
- Test: `[17, 28, 16, 16, 28, 20, 20, 24, 10, 21]` (200).

| Derived array | SHA-256 |
|---|---|
| MNIST train binary pixels | `6e5429e6869a3863a9bcb655d2c2dbd333689da1405df83c303faf5076d61f03` |
| MNIST train labels | `7c8733e936a08984ed9643de8fd50871b49bc9fd173030252f50154c302fc247` |
| MNIST test binary pixels | `4e5d91afd4806551c95e4a8846fb9a9b064ef9854dff4e23d47482584d76412e` |
| MNIST test labels | `da7f2703def944efe765ca288252560f04f9d92058c8e1a665bd14bf57bbee72` |

Prepared source CSV SHA-256: train
`06e64b0bf65de89c522348319ce5f63f43fb1cb40e1ce3c2ff43c1d3b8a0e14c`;
test `c5adfcc9ce29069f8a988a6af1dcdc009bba975d2c98a52233661f79d880292b`.
Neither source CSV nor a derived MNIST array is added to Git.

### 23.4 Requested checks and publication

Ran the seven requested files together, after the freeze:

```sh
env -u RUN_TRAINED -u RUN_SLOW make test 'TEST_ARGS=test/test_image_corpus.py test/test_whole_presence.py test/test_math_corpus.py test/test_game_corpus.py test/test_lesson_stages.py test/test_training_corpus.py test/test_prepare_data.py --run-dir <new-local-receipt-directory>'
```

**90/90 passed**, zero failures/skips/xfails/xpasses. Runner elapsed
**32.444118 s**; worker elapsed **27.446453 s**. The table gives summed pytest
call durations, excluding collection, setup/teardown and runner overhead:

| File | Passed | Test-call seconds |
|---|---:|---:|
| `test_image_corpus.py` | 23 | 0.786726 |
| `test_whole_presence.py` | 37 | 2.998289 |
| `test_math_corpus.py` | 5 | 0.045232 |
| `test_game_corpus.py` | 7 | 0.186083 |
| `test_lesson_stages.py` | 7 | 0.019812 |
| `test_training_corpus.py` | 8 | 0.137190 |
| `test_prepare_data.py` | 3 | 0.003907 |

The new 23 default-sweep cases construct no model. They verify exact freeze
reproduction, all ten class geometries and variation, counts, binary pixels,
global disjointness, independent hex decoding and loader agreement, MNIST
manifest reproduction, source-prefix agreement, exact 127/128 threshold
boundary, prefix truncation, missing-data behavior and invalid split/name
rejection. Both real MNIST splits were present and checked. The reproduction
case is the explicit exception requested by §22(d) to §4's general rule
against regeneration inside tests; it writes only to pytest's temporary
directory. No network or language model is used.

Runner limits stayed at two workers, 8 GiB per worker, 16 GiB aggregate,
1,800 s per worker, 10,800 s suite, batch size 256 and max 16 files. The
seven-file selection fit one worker, peak **628,853,400 bytes** (also peak
aggregate). No retries, timeouts or capacity changes occurred. PyTorch
2.15.0, pytest 9.2.0; the compiled whole-presence paths remained enabled.

Receipt identities, recorded inline because local receipts are ignored:

- Initial whole-presence `result.json`: `2101ec167f7a28c18b0260361f633c89b0b6354d9467c3d0eefacaf9f0da29c6`.
- Seven-file `result.json`: `471beb17bbb9b4191021d604173cb48d57fed21c231d92ece1f5fd44609f88d4`.
- Seven-file `source-manifest.json`: `73f0c2761b957da3c64868651e60fdcb0e07f7bce99563c15c5ee37f9b21d068`.

The validated source snapshot has **1,158 entries**, SHA-256 of compact
sorted JSON
`746920adb75721862be00b1b3d28db5a92f8a27cade140feb7dcc8598b9744c4`.
It matches `test/bounded_tests.source_snapshot` after testing;
`data/corpus/assembly-01/manifest.json` is present. JSONL identity is carried
by the frozen shape manifest's file hashes. Publication is one normal push
of `testing` after recording the implementation commit below. Local `main`
remains `9f7923d67f8921ef5c14afdf2fe18c536802018b`.

Implementation commit pushed on `testing`:
**`5b59d5b8db19f4d26534701c67ad82aace231a39`**. The following metadata-only
commit records that identity; both are included in the single end-of-round
push. It changes no validated source or data. The final remote branch tip
is the metadata commit, whose hash is reported in the chat handback.
