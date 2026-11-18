# Computation by wiring, not by moving data: part 1 of item 1.5, and item 1.6 until 2026-11-18 (Claude; investigated, measured and re-scoped on 2026-11-17)

Alec asked for this as a new item between 1.7 and 1.5: "can you
investigate the possibility of pushing the architecture in this
zero-data movement direction? It's 1.6 so that subsequent optimization
becomes easier, with fewer cpu or python data reads/writes."

**Renumbered** (Alec, 2026-11-18: "Please move all 1.6 work into
1.5"). This work is the first part of item 1.5 since then, and the
number 1.6 names another item: the number modality, the math corpus
and the felt world for the math corpus. Where this document says
"item 1.6" or "this item" it means this work. Where it hands
something to "item 1.5" it means that item's second part: the lookups,
the folds and the percept pass. The file keeps its name, because the
specifications cite it by its path.

This document says what a sentence moves today, what each movement
would become, the steps, and what Alec decides. Its first form was an
investigation by reading. The same day it was measured (section 3), and
two sweeps of the source answered its two open points. The measurement
changes the order of what matters: section 6 is revised, and Alec
decided the revision the same day (section 9). Nothing is built.

## 1. Alec's words

- 2026-11-17, the request: "The form of MTM can probably be the same as
  percepts or symbols, since it serves as a testing ground for percepts
  or symbols. So an easily promoted and easily forgotten store, as
  opposed to the percepts that they promote into. Given that STM copies
  over into LTM, and MTM is percepts, and STM trees are potentially
  activation traces over the order-recursion, we have remarkably few
  structures. In conjunction with lack of read/write, I see this as the
  basis for optimization that works in virtue of wiring instead of
  copying data around. That would work well biologically also."
- 2026-11-17, earlier (1.8 §68.30): "consider that STM is in fact a tree
  that might be unfolded in the same process as conceptual recursion,
  except that it uses an activation trace instead of a buffer that gets
  read/written. If we combine that with extension instead of intension,
  we pretty much have a buffer-free mind (it works via activation of
  structures rather than reading/writing contents)."
- 2026-11-17 (1.8 §68.29): ""the small stack" is exactly the ideas and
  references that will be copied into LTM0 and LTM1 at the end of the
  round. Feel free to work in place in LTM if possible".
- 2026-11-17 (1.8 §68.36): "Sentence nouns tie to identity in MTM, and
  that identity is either transitory when deleted from MTM or becomes
  promoted as a symbol at the next order."
- 2026-11-17, later: "It seems like 1.8 is almost complete, and 1.7 is
  fully specified. Can we update and finish open questions in 1.6 as a
  prelude to GPU optimization?"

## 2. What counts as moving data

Four kinds, each of which can be counted.

1. *A host read:* Python takes a value out of a tensor, or walks a
   Python table, to decide what to do next. On a GPU each one waits for
   the device.
2. *A copy between stores:* a value written into a second store.
3. *A clone for a trial:* state copied so that a path that may be
   dropped cannot disturb the one that is kept.
4. *A second form:* one thing kept in two layouts, each kept current.

*The earlier baseline* (1.8 §53, the lesson's sentences, before the
tree-first work). First sentence: 80,483 host calls, of which 53,915
read a scalar; 52,497 of them in the close. Sixteen sentences: 1,634,871
host calls. Its largest readers were `MereologicalCodes.derive` (29,538),
`LTM1.references` (23,620) and `LTM1.contains` (12,238).

## 3. What a sentence costs today: a measurement (2026-11-17)

*What was run.* One worked document of the math chain
(`test/test_math_chain.py::test_native_worked_document_trains_unforced_without_numeric_oracles`):
12 sentences of about 8 words, 21 paths composed and scored, one
optimizer step for each sentence. Codex's source as 1.8 §59.44 returns
it (§59.40 with one change, in `Comprehension.Evaluation.leaf`), on the
CPU, nothing compiled, under `cProfile` (44.3 s). The counts are exact.
The shares of time are indicative: the profiler weighs a short Python
call more than a tensor operation, and when the word steps are compiled
the read's own operations fuse. The profile is `census16/worked_1.prof`
in the session's scratchpad.

| For one sentence | Count |
| --- | ---: |
| Python calls | 10,001,837 |
| calls into the tensor library | 1,096,486 |
| `.tolist()` on a tensor | 22,030 |
| `.cpu()` | 1,445 |
| `.nonzero()` | 992 |
| `.item()` | 80 |

| Phase of a sentence | Share of its time |
| --- | ---: |
| scoring the paths (`score`) | 45% |
| the read (`compose`) | 26% |
| the sentence's optimizer step (`_sentence_train_step`) | 12% |
| the kept close (`_commit_sentence`) | 9% |
| the rest | 7% |

| Where the calls come from | For one sentence | Kind (section 2) |
| --- | --- | --- |
| The hook on every tensor that autograd saves (`SentenceCompose.saved_sentence_values`) | 63,344 calls of the hook; 53,499 of the 80,383 detaches; its own queries are about two fifths of all calls into the tensor library | 3 |
| `FieldReverse.reverse`, 9 calls | 77,327 of the 88,210 `new_zeros` and 38,590 of the 54,801 `stack`: two zero buffers for each of 4,096 primitive properties, at every call | 4 |
| `MereologicalCodes.derive`, 867 calls | 32,785 `zero_` and 78,808 row lookups | item 1.5's |
| The walk of the pinned tree (`SentenceStrata.Tree.operator`) | 8,924 of the 9,032 reads of `LTM1.references`, two `.tolist()` each: four fifths of all `.tolist()` | 4, step A |
| `ConceptualMeaning.__post_init__`, 1,380 constructions | 341,816 calls of `_freeze_metadata` (about 250 for each construction); 2,760 clones | 4 |
| `SentenceBoundary.detached` | 80,364 calls, recursive, from 227 entries; 8,449 of the 15,502 dataclass `replace` | 3 |
| `Thoughts.replay_thoughts` | 5,658 `replace` | 1.8 §69 |

*What it changes.*

- Step A is confirmed: the walk of the pinned tree is the largest
  source of host reads, on a lesson document as in the probe of 4.2.
- The other two steps of the first proposal are small in time. MTM's
  work for a sentence is at most 5% of it (2.08 s of 42.75 s summed
  over its functions, which overlap; `MemoryLink.menu` and the values
  computed on read are most of it), and the kept rows' write
  (`_append_observed_meaning`) is 0.3%.
- The largest sources of calls were not in the proposal at all: the
  hook on saved tensors, the reverse's zero buffers, the derived forms,
  meanings built again and again, and detaching by walking containers.
- The first proposal asked what share of a sentence the work for each
  path is (section 9, point 3). Scoring the paths is 45% of a sentence.

## 4. What a sentence moves today

Read on 2026-11-17 in Codex's source, first of 1.8 §59.31 and then of
§59.40, by four sweeps of `bin/`; the lines marked *checked* were read
again by Claude.

### 4.1 The stores, and what holds them

| Store | Held in tensors | Held in Python objects |
| --- | --- | --- |
| STM (`STMChunks`) | all 14 fields, [streams, 8 chunks, ...] | nothing |
| Ideas (`LTM0`, `STM0`, both `IdeaRows`) | `slots`, `trust`, `referent`, `activity`, the episodic columns | `LTM0._row_times`, `_rows_by_time`, the coordinate counts; `STM0.working` (a dict of meaning metadata) |
| References (`LTM1`, `STM1`) | `refs`, `ref_kinds`, the two truth lanes | the indexes `_pairs`, `_parts`, `_wholes`, `_references` |
| MTM (`MediumTermMemory`, in `bin/Memory.py`) | `transient_ids`, `transient_used` | everything else: `entries` (at most `mtmCapacity`, 64), `names`, `pending`, `closes`, `accesses` (*checked*) |
| Basic percepts (`BasicPercepts`) | `codes`, `promoted`, `count`, `places` | `index`, `words`, `occurrences`, `selectors` |
| The concept allocator | `singleton_values`; `parts` (an `LTM1`) | about fifteen dicts and four sets (`symbols`, `relate_idx`, `word_descriptions`, `individuals`, ...) |
| The Sigma and Pi edges (`SparseLayer`) | `values` | the lists `_rows`, `_cols` and the dict `_index` |
| A captured reading (`Reading`) | most of its 53 fields | the dataclass itself |
| A meaning (`ConceptualMeaning`) | `roles`, `role_mask` | `role_refs`, `bindings`, `scope` as nested tuples, frozen again at every construction (*checked*) |

MTM's own description is already wiring: "References form a path
across kept closes; values are computed on read. The pool owns
membership and recorded action operands, never a second copy of an
idea's content" (*checked*). Its cost is its form: frozen dataclasses
replaced at every change, and scans of every entry.

### 4.2 The tree is kept in five forms

1. The live stack of the read (`STMChunks`).
2. The journal's record: one row of 74 integers for each action, with
   each operand's provenance, and a slab of the operands' values
   (`ChunkJournal`).
3. `Reading`: taken for each scored path and each stream by
   `Models._program_entries`, with about fifteen gathers, several
   clones, and a host loop over all MTM entries.
4. The pinned tree: `SentenceStrata.capture` writes an occurrence row
   for each word and operator into an STM0 bank and an ordered pair
   for each application into an STM1 bank, once for each path and
   stream, by a host loop over `program.actions.tolist()` (*checked*).
   It then sets the reading's `actions` and `leaves` to None, and
   `Reading.__getattribute__` answers them by `Tree.actions()` and
   `Tree.leaves()`, which rebuild them from the rows and pairs each time
   they are read (*checked*).
5. The host walks: thirteen files of `bin/` each loop over the actions
   again on the host, most with a Python stack of their own
   (`Assessment`, `ClauseJournal`, `Comprehension`,
   `DerivationReconstruction`, `Extension`, `FormCentroids`,
   `GrammarEvidence`, `Interpret`, `PartsOfSpeech`, `ReferenceContext`,
   `SentenceCredit`, `SentenceStrata`, `eval_operators`) (*checked*).

In a probe on tests, every read of `references` was a walk of the pinned
tree (268 of 268, and 1,720 of 1,720). On the lesson document of section
3 it is 107,083 of 108,389.

### 4.3 Copies between stores, for each sentence

| From, to | Where | How |
| --- | --- | --- |
| MTM to STM | `PrependedSymbols.reconcile` and `stage` | every STM field cloned, then host loops over streams, seats and MTM entries (*checked*) |
| a scored path's clause to its private stack | `PartClosing.stage`, from `Comprehension.Evaluation.finish` | into the trial's `IdeaStack` (an `STM0` with its own `STM1`), under names from the trial's clock |
| the kept clause to LTM | `ClauseRows.write_clause` (`bin/ClauseRow.py`), `PartClosing._write`, `IdeaRows.append_meaning` | the clause closed again from the start, by a host recursion, one row at a time |
| STM to its recent store | `STMChunks.keep_close`, `SentenceStrata.keep_close`, `_clause_end_state` | every STM field cloned; one row appended, its relations copied from LTM1 |
| MTM to percepts, concepts, symbols | `promote_candidates`, `promote_identity`, `PerceptCollections.collect` | a host loop over all of MTM's entries each sentence (*checked*); a copy only where one is promoted |

*A correction.* The first form of this document said that a kept clause
is written twice by the same recursion, and that its staged rows could
be committed as a block. That is wrong on the facts. The staged rows are
made before the reference close has decided anything, and the kept
clause is changed by `ReferenceClose.prepare` between the two: its
bindings, its references, its site children and its companions, so that
the number, the kind and the referents of the rows can all differ. The
kept write also does what staging does not: it gives each row its time,
its trust, its episode and its first activity, holds a pending meaning
in MTM, removes an equal pair, and may consolidate. And it is 0.3% of a
sentence (section 3).

### 4.4 Clones for a trial

For each scored path: `_program_entries` (4.2, form 3), the pinned tree
(form 4), a copy of the concepts held in STM (`HeldConcepts`), a private
stack of ideas and references (`IdeaStack`), and for a path that
departs, a deep clone of STM, the language state and the feedback
(`SentenceFork.snapshot`, *checked*). At the kept close the chosen
`Reading` and comprehension's result are cloned again (`detached`).

Three more, found by the measurement.

- *Every saved tensor.* A sentence batch runs inside
  `saved_sentence_values` (`bin/Models.py`), which hooks every tensor
  that autograd saves. It exists because the sentences of a packed row
  share one perception forward across their optimizer steps, and a step
  writes the parameters in place.
- *Every meaning.* A `ConceptualMeaning` clones its roles and its mask
  and walks all of its metadata whenever it is constructed, and a
  `replace` constructs one.
- *Every boundary.* `SentenceBoundary.detached` walks dicts, tuples and
  dataclasses to their tensors and rebuilds each container on the way.

`bin/Models.py` alone has 133 lines that clone, 191 that detach, 67
that call `tolist`, 35 `item` and 28 `cpu`.

## 5. The few structures, and three rules

Alec's sentence names four. With the field, they are five.

| Structure | What it is | What exists already |
| --- | --- | --- |
| Rows | percepts at order zero and symbols above; MTM is the rows on trial | `BasicPercepts` holds candidates beside promoted rows; rule 20 of item 1.7 has MTM nominate every row |
| Concepts | the Sigma over rows; a signed value each | item 1.7 |
| Ideas and references | one layout, short-lived in STM and kept in LTM | `STM0` is an `IdeaRows` as `LTM0` is; `STM1` is an `LTM1` |
| The trace | what a read did: its actions, their operands, their orders | the journal's record already carries each operand's provenance |
| The field | the concepts' activations | 1.8 §68.18: "the field is just the concept activations" |

- *One home.* A thing is stored once. Whatever else needs it holds its
  index.
- *A mark, not a copy.* A change of standing (on trial to promoted,
  short-lived to kept) changes a count, a flag or a boundary.
- *Not stored if it can be run again.* What the layers give from the
  trace is not kept as contents.

## 6. The steps, as revised

Each step is measured by the counts of section 3, and each must leave
results exactly equal, as 1.8 §65's certificate 2 requires of part C.

**Step A. One record of a read.** Decided; built with part C of item
1.8, whose §65.5 has its rules and certificates.

- The trace is the journal's record and its slab of values. `Reading`
  reads it; the pinned occurrence rows and pairs of the tree are no
  longer written; `Tree` and `CompositionView` answer from the trace.
  One index of the trace is made once for a path, and the thirteen
  readers take it.
- *Measured:* it removes four fifths of a sentence's `.tolist()` reads.

**Step B. MTM in the form of rows.** Decided in direction (section 9,
point 2); after item 1.7, whose rules 19 to 21 it completes. Its worth
is the structure and where it lives, on the device with the rows; in
time it is at most 5% of a sentence today.

*The rule: an entry is a row, and whatever else it carries is a pair.*

| What a `Token` carries (read in `bin/Memory.py`) | Its home |
| --- | --- |
| `support`, a candidate's weighted member rows (at most `field_width`) | the row's own entries in the Pi |
| `count`, `when`, `kind`, `pending`, `where`, `native` | columns of the rows |
| `symbol`, `transient`, `order`, `mass` | nothing: once kept the row is the symbol; its index is its id on trial; its order is the Pi's; its mass is the sum of its entries |
| `readings`, kept within the horizon, and `accesses` | for each stream the clock of the last reading and of the last access, and the greatest of the readings on each lane: columns. The horizon is a comparison of clocks, not a list that is pruned |
| `occurrences`, with `names`, `mentions` and `members` | pairs from the row to each occurrence, in the reference store's layout (`STM1` on trial, `LTM1` when kept), each with its stream and clock. Alec, 2026-09-30: an identity is nothing but its occurrences |
| `timelines` and `event_links`: an event's occurrence, role and time | pairs from the row to the occurrence's row, with the role and the timeline; the verb and the time are read from that row |
| a candidate's `contexts`, the chains, `participants`, `sign_views` | pairs, each with its kind |
| `MTM.pending`, a held meaning with its step | a row of the trial's ideas, with the step as a column |

- *Promotion and lapse are marks.* `promote_candidates` and
  `promote_identity` mark a row as kept, and its entries and its pairs
  are already where a kept row's are. What they copy today goes: the
  support written again as constituents (`remember_candidate`), the
  edges written into the allocator's parts, `BasicPercepts.reserve` and
  `promote`. A lapse frees the row and its pairs.
- *No scan.* The owner of an occurrence is found by a read of the pairs
  by column (item 1.7's rule 19 keeps an index of columns), where
  `token_of` scans every entry and every event today. The loops over all
  entries in `PrependedSymbols.reconcile` and `stage`,
  `ReferenceClose`'s `accessible`, `MemoryLink.menu` (one inside
  another), `Extension.held_context` and `mtm.close` become reads of
  columns under a mask.
- *Nothing is staged into STM.* `PrependedSymbols.reconcile` and `stage`
  go. The concepts held in STM for a trial (`HeldConcepts`) are the
  trial's provisional members of those rows.
- *Removed as unread* (found by the sweep, to be confirmed at the
  build): `refinement['both']`, which nothing reads;
  `refinement['member_concepts']`, which nothing writes;
  `MediumTermMemory.mint_count`, which nothing reads.
- *The values an event snapshots* (`before`, `case_value`,
  `verb_meaning`, `basis`, `modifiers`, and the participant's state
  after the change). They are the recorded operands of the action,
  taken from the journal's record at the kept close
  (`ReferenceClose.link_events`, *checked*), and copied into the
  `Event`. An event outlives its sentence: it is read until its entry
  lapses (`value_of` takes the latest) and until the teaching of verbs
  has used it. *Claude's proposal:* at the kept close they are written
  once as rows of the short-lived ideas (`STM0`), and the event's pair
  holds their indices; they lapse with the row. That is the one copy
  that exists today, given a home in the ideas' layout. Whether the
  kept trace of step A could hold them instead, with no copy, depends on
  how long a trace is kept, and is read at the step's text.
- *Certificates.* No `Token`, `Event`, `Timeline` or `Name` object and
  no dict of entries. MTM's state is columns and pairs on the device. A
  promotion and a lapse change a mark and copy no value. No loop over
  entries in a sentence. The lesson's replays are exactly equal.

**Step C is withdrawn** (decided, section 9). It was to commit the kept
path's staged rows as a block. The staged rows are not
the kept rows (4.3), and the kept write is 0.3% of a sentence.

**Step D. A meaning is made once.** New, from the measurement; decided
(section 9).

- *Today:* 1,380 meanings are constructed for a sentence. Each
  construction reads the device three times to validate
  (`isfinite().all()`, `mask.any()`, `mask.tolist()`), clones `roles` and
  `role_mask`, and walks `role_refs`, `bindings` and `scope` to freeze
  them, about 250 calls each. Most constructions are a `replace` that
  changes one field.
- *The rules.* (1) Metadata is frozen where it is made, by its writer,
  and a meaning that is given frozen metadata does not walk it again.
  (2) A meaning is validated where it enters, from a store or a fixture,
  and not at a `replace` of a meaning that was valid. (3) `roles` is
  cloned only where the caller keeps writing to what it handed over;
  the build lists those callers.
- *A trial of rule 1 alone* (a scratch process, the checkout untouched;
  `census16/freeze_once.py`): a tuple that `_freeze_metadata` has
  already returned is given back as it is. The document of section 3,
  with no profiler and no seed: six runs as it is took 36.4 to 39.2 s
  (mean 37.6); five runs with the rule took 27.2 to 36.9 s (mean 31.7).
  The draw differs from run to run, and Codex was running tests on the
  same machine for part of the time, so this is a size, not a number:
  between a twentieth and a quarter of the document's time. (Six of the
  sixteen runs stopped early on the defect of 1.8 §59.39, which the
  text of 1.8 §68.54 removes.)
- *Certificates.* For a sentence, the constructions, the calls of
  `_freeze_metadata` and the clones, before and after. Every meaning of
  the lesson's replay equal: its roles, its mask and its metadata.

**Step E. What is cut from the graph is cut where it is written.** New,
from the measurement; decided (section 9).

- *Today:* `SentenceBoundary.detached` is entered 227 times for a
  sentence and visits 80,364 values, rebuilding every dict, tuple and
  dataclass it passes. Rebuilding a meaning constructs it (step D).
- *The rule.* A durable store detaches what it writes, which
  `ConceptualMeaning` already says of them. A boundary names the
  tensors it cuts. No container is walked or rebuilt to detach.
- *Certificates.* The calls for a sentence, before and after. The
  gradients of one batch equal (atol 1e-6), and no tensor of a finished
  sentence still holds a graph.

**Step F. The hook on every saved tensor: a question for item 1.** New,
from the measurement; measured there first (decided, section 9).

- *Today:* 63,344 calls of the hook for a sentence in an eager run, and
  two fifths of all calls into the tensor library.
- *Not known:* how many remain when the word steps are compiled, since
  a compiled step saves far fewer tensors. That is the first thing to
  measure.
- *The ways, if it still weighs.* (a) The shared perception forward
  reads a frozen copy of the parameters it uses, one copy for a packed
  row, and nothing is hooked. (b) One optimizer step for a packed row.
  The second changes what is learned from, and is Alec's.
- It is not built in this item.

**Handed to item 1.5 with their numbers.**

- `FieldReverse.reverse` makes two zero buffers for each of 4,096
  primitive properties at every call and stacks them: 77,327
  `new_zeros` and 38,590 `stack` for a sentence. One buffer, written by
  one scatter, is item 1.5's "folds as scatters".
- `MereologicalCodes.derive` is called 867 times for a sentence. Item
  1.5 already has the derived forms kept current as a tensor.

**Not proposed.**

- *Comprehension for the kept path only.* Capture, comprehension and
  the preview of the reference close run once for each scored path,
  because a path's cost includes its meaning's costs. Alec left the
  choice among paths as it is (section 9, point 3). Its measured share
  is now known: scoring is 45% of a sentence.
- *STM's functional fields.* They are small, and trials and the
  compiler rely on them.

## 7. What GPU optimization receives

- *Where the compiled part ends today.* Five word steps are compiled as
  whole graphs (`stage_cs_sub`, `stage_cs_sym`, `choose_operation`,
  `apply_language_choice` and the percept's arrival;
  `SentenceCompose.compile_word_brick`). Everything else of a sentence
  runs on the host: its open, the scoring of its paths, its optimizer
  step, its close, its thought and its boundary. By section 3 that is
  three quarters of a sentence's time. Eleven functions of `bin/` are
  declared host islands (`@torch.compiler.disable`).
- *Two numbers to carry through items 1.6, 1.5 and 1,* taken on the
  document of section 3: the host reads of a tensor for a sentence
  (24,547 today: `tolist`, `cpu`, `nonzero`, `item`), and the calls into
  the tensor library for a sentence (1,096,486 today). On a GPU the
  first are waits and the second are launches. Neither falls by moving
  the same work to the device.
- *The source as a whole* (a static count of `bin/`, 198 files): 408
  `.tolist()`, 244 `.item()`, 169 `.cpu()`, 185 tests of a tensor's
  `any` or `all` by `bool`, 97 `.nonzero()`.
- *What each item removes.*

| Item | What it removes |
| --- | --- |
| 1.8 | the walk of the pinned tree (step A); the estimate of a sentence and its cost for each path (§68.53); the replays of thought (§69); the tables as tensors and one graph for a sentence (§64 to §66) |
| 1.7 | the dictionaries of edges: the Pi and the Sigma above order zero as compressed rows on the device (rule 19); the closure of the definitions as the layers' own forward |
| 1.6 | MTM's objects and its loops (step B); meanings built again (step D); detaching by walking (step E) |
| 1.5 | the derived forms; the reverse's buffers; lookups as one product; the percept pass once for a sentence |
| 1 | what is compiled; the hook on saved tensors (step F); sentences in batches |

- *A sentence's percepts, once* (Alec, 2026-11-17: "I think that
  attention in GPU reading over a sentence may work most quickly if we
  compute perceptual features for the whole sentence at the start (IOW,
  do all codebook lookups by building an array of percept indices for
  the .what and an array of .where with the same indices)"). It is item
  1.5's percept pass, decided on 2026-10-30: "the percept pass runs once
  per sentence over all words with gradients, since the percept depends
  on neither grammar state nor the expectation". Half of it exists: the
  two arrays are there (`native_indices` and `native_part_spans` of the
  perceptual space's forward input), and the open pass reads every word
  bracket from them in one call (`FamiliarTier.open_sentence`, the first
  look of 1.8 §60.1). What is not so yet, measured on the document of
  section 3: rows of a codebook are looked up 1,924 times for a sentence
  (`Codebook.lookup_rows`), 63 of them reading a whole table (`getW`,
  7% of the profiled time); and each later look of 1.8 §60.1, rule 3,
  reads its percepts again. With the rows gathered once into an array
  beside the `.where` array, a look takes its percepts by a mask on
  `.where`, and nothing is looked up after the sentence's start. The
  set of rows cannot change inside a sentence, since admission and
  nomination are at its boundaries.
- *What stays on the host by decision:* tokenization and the admission
  of a new type (item 1.5), and the grammar's serial choices.
- *This item's exit:* the two numbers taken again on the same document,
  and §53's inventory taken again on the lesson.

## 8. Risks

- *MTM is read in many places.* About sixty readers and writers of a
  `Token`'s fields in twelve files (`FamiliarTier`, `MemoryLink`,
  `ReferenceClose`, `PrependedSymbols`, `PerceptCollections`,
  `ScalarPercepts`, `Extension` and others). Step B is the largest step
  here, for the smallest gain in time.
- *Exactness.* Each step changes where a value lives or how often it is
  made, not what it is. The lesson's exact replays are the test.
- *Steps D and E touch how gradients are cut.* Their certificates
  compare gradients, not only values.
- *One document, eager, on the CPU.* The measurement is to be taken
  again on the source that items 1.8 and 1.7 leave, and with the word
  steps compiled, before any of these steps is given to Codex.

## 9. Decided

**Decided** (Alec, 2026-11-17: "All good"), each as recommended then.

1. *The one form of the tree is the trace,* and the pinned rows and
   pairs are no longer written (step A). This reverses the choice of
   item 2.0's round 5, where the pinned tree in STM0 and STM1 is the
   authority and the journal is not.
2. *MTM and the rows it promotes into are one table,* in which a row on
   trial is marked as kept, so a promotion moves nothing (step B).
3. *The choice among complete paths is left as it is* in this item, and
   what share of a sentence the work for each path is, is measured
   first. It is measured: 45%.
4. *Step A is built with part C of item 1.8* (1.8 §65.5), and steps B
   and C are item 1.6, after 1.7.

**Decided** (Alec, 2026-11-17, of the four points Claude put to him:
"the four points are good to go with your recommendations").

5. *Step C is withdrawn.* Its premise was wrong and its share is 0.3%.
6. *Steps D and E are added* to this item.
7. *Step B stays,* after item 1.7, for its structure and not for time;
   its proposal for an event's snapshots is read again before its text.
8. *Step F is measured in item 1 first,* with the word steps compiled.

*Claude's default, not put to him:* within the item, D and E first,
since they are small and measured, and B last.
