# Item 1.5: lookups as one matmul, whole/part folds as scatters, the percept pass per sentence (Alec, 2026-10-30; Claude's reading and design, in code names; decisions taken the same day, §5)

Alec's decision of 2026-10-30: these optimizations are not future work.
They are an item of the countdown, placed before item 1 so that the
compiler work compiles what this item builds under `fullgraph`, and before
item 0. The cost reading of the same day (§2) was first written into
[FutureWork](../FutureWork.md#throughput-levers-for-the-serial-loop-item-1-candidates)
and moved here the same day; that section keeps a pointer. The September
throughput levers recorded there remain item 1 candidates; this item
absorbs lever 4 (the host islands and device CSR) and lever 2 (the
known-word lookup) where §3 names them.

## 1. The record (Alec, 2026-10-30)

- Content-based lookups over absolute truth concepts, in LTM or in
  conceptual space, as a single matmul.
- The same matmul might someday apply to percepts, although that may not be
  faster if it is radix-true-like part composition.
- Percept composition is slow; parallelize it for a GPU.
- Whole and parts may need work; concept lookup seems fast, and there are
  not many orders. Grammar operations will be hard to speed up, except for
  their queries over existing cognitive structures.
- Not future work: before item 0, and before the compile work, to make
  sure they compile with `fullgraph`.
- Later the same day, on the decisions of §5: the lookups are content-based
  matching of a vector against concepts or LTM0 entries, as a matmul; MTM is
  name/value pairs, and since words are percept codes rather than
  orthographic there, is that an exact match? (answered in §5: no);
  "this is great, we have the start of a parallel perceptual engine"; one
  round if possible, since Codex does this much in an iteration when it is
  already specced and spends too much time in testing for fine-grained
  todos.

## 2. Where the time goes as built (Claude, 2026-10-30)

Read at `3718d9d0dc` plus Codex's uncommitted round B and one-backward
trees. No profile was taken: the one-backward timing arms were running on
the machine, so the ranking is structural. The regime measured today is
lesson-13 at batch 1, eager, one CPU thread, at 8 to 10 s per sentence,
against the July compiled baseline of 6.8 sentences/s at batch 24. Every
cost below scales with occupancy (occupied rows, lexical concepts), not
with capacity: lesson-13 shows the fixed per-word overhead and hides what
`BasicModel.xml`'s 262,144 LTM0 rows and 65,536 concepts would expose.

1. **The derived codebook read at every word.** `forecast_percept`
   (`ModelAttention.py`) gathers every lexical concept row before each
   word's percept, and `MereologicalCodes.derive` walks them in Python with
   one host sync per row (`_form_valid[row]`). The forms are invalidated
   whenever the feature parameter changes, so every optimizer step forces
   a re-derive, and the re-derive walks the whole feature-edge dictionary
   (`_definitions`) with a host read per edge. That is $O(L)$ Python per
   word, shared by both trials. The symbol field then runs three distance
   passes over the whole symbol bank per word, per trial: batched, but
   proportional to the inventory.
2. **The second trial reruns the whole word loop** with masking, although
   everything before the departure site is identical and the fork
   machinery (`SentenceFork`) exists. Under `<separateChooser>` false the
   shared prefix already sits in one graph.
3. **LTM indices with the wrong complexity.** LTM1's dict index is rebuilt
   over all relation rows on every `add` and the next read
   (`LTM1._index`); `EntryRead.cued_rows` builds its candidate list with a
   list membership test, $O(N^2)$ in occupied LTM0 rows before the work
   meter applies; `relation_evidence` scans LTM1 linearly although
   `rows_referencing` exists.
4. **Per-sentence Python sweeps.** `recurrence_groups` over all occupied
   LTM0 rows per stream per kept close (two syncs per row);
   `_witnessed_rows` over all concept rows per MTM entry in
   `observe_field` and `promote_candidates`; `DefinitionFold.examples`;
   `cs_read_memberships` and `_compose_order0` over host COO rows with a
   sync per edge; full derives (`open_prediction`, `stage_symbol_bank`)
   and `getW()` calls used only for a device.
5. **The compose rounds run eagerly in training**, about five host syncs
   per round, times $(3W + \text{closings}) \times 2$ trials; only the CS
   stages and the arrival brick are compiled.
6. **Dormant but shaped the same:** heat retrieval (off; when on, two
   `.item()` per reverse step and a full `getW()`).

Not the problem: the choosers' candidate scoring (compose scores a bounded
window in one batched call; thought stacks all actions into one logits
call, twice), attention admission (sentence-local, batched),
`ActivationPredictor` (chunked), `BindingAnswers.cost`.

**Percepts.** Within a sentence the percept of every word is independent of
grammar state, STM and the expectation: the arrival chart's forward value
is exactly the observed percept (`_PerceptIdentity` in `Expectation.py`)
and the word bricks take empty feedback (`_run_sentence_word_bricks` in
`Models.py`). The GPU math inside the bricks is already parallel over
streams and parts (gather and `amax` over $P$ parts, the WS matmul, the
bank decode); what is serial is the Python around it: $W = 256$ columns per
sentence with a host sync each, two small compiled launches per word, and
the forecast's host work per word. The stem already computes
`synthesize_word_parts` over $[B, W, P]$ without gradients
(`_embed_ladder_word_major` in `Spaces.py`). Host-side and
order-dependent: tokenization (the radix trie, per-byte id loops) and
admitting new word types (identity mint and rekey, `interpret`); these run
once per batch. Known words are recomposed from their atoms at every
occurrence although the identity row already holds the `amax` of its
atoms; parts are padded to $P = 64$ against about 11 real, so most of each
gather is masking.

**Whole/part composition is the shared slow shape on both sides**: a fold
over definition or part edges that runs as Python per edge on the concept
side and as a padded gather on the percept side. With few orders the fold
is a few passes.

## 3. The work (design, in code names)

Each step is a re-implementation with the same outputs where it says
"equal"; the matmul lookup is the one place where the output is a ranked
set rather than a value, and §4 says how it is certified.

1. **Sparse stores to device CSR; forms as a tensor.** The feature-edge
   store (`layer.features`, today an `_index` dict and a values parameter)
   and the membership matrices (`matrix._rows`, `_cols`) get preallocated
   device CSR views with fixed edge capacity (fail loud on exhaustion;
   never resize). `MereologicalCodes.derive` becomes: for the rows whose
   validity bit is clear, one segmented scatter per order
   (`scatter_reduce` with `amax`/`amin` over the subspace bounds, the same
   $q^{w}$ and $1 - (1 - q)^{w}$ folds `fold_presence` writes), all rows at
   once, then set the bits. Invalidation is incremental: the optimizer step
   clears the bits of rows whose edges' values changed, not all rows.
   `_definitions`, `_subspace_bounds`, `cs_read_memberships` and
   `_compose_order0` lose their Python edge loops. Equal.
2. **Content lookups as one matmul.** `EntryRead.cued_rows` builds no
   list: the cue's roles against the LTM0 roles tensor in one product,
   masked by kind, scope and the occupied-role mask, top-k under the work
   meter, explicit cues (references, retrieved, `rows_referencing` on
   SYMBOL refs) prepended. `relation_evidence` reads through
   `rows_referencing`. `LTM1._index` is maintained on `add` (dict insert),
   not rebuilt. `_witnessed_rows` becomes a maintained boolean tensor over
   the inventory; `witnessed_match` is one cosine over the masked rows.
   `recurrence_groups` folds the new row into its seat instead of
   regrouping the store; `DefinitionFold.examples` reads the CSR view.
   MTM's `match_candidate` and the familiar tier's `witnessed_match` are
   the same product over the masked codebook rows (§5, D2).
   Same candidates and scores as today at the same meter (§4).
3. **The percept pass once per sentence.** `stage_cs_sub` and
   `stage_cs_sym` run over $[B, W, P]$ with gradients, as the stem's
   no-grad version does, with ragged parts (a CSR of part ids per word)
   in place of the 64-wide padding; the per-word loop keeps only what
   depends on the previous word: `observe_percept`'s arrival state, which
   is a causal prefix mean and a shift and so is also computed for all
   words at once, and `forecast_percept` scored for all words in one
   predictor pass with its inputs and targets detached as §16.1 requires.
   The known word's percept reads the identity row's `amax` (lever 2),
   byte-equal to recomposition. Equal.
4. **Fork the second trial at the departure site.** The explore trial
   resumes from `SentenceFork`'s recorded state at the departure word
   instead of rerunning the prefix with masking. Equal in actions,
   probabilities, costs and updates, since the prefix is identical by
   construction.
5. **Fullgraph cleanliness.** Every brick touched above traces under
   `fullgraph=True` with no host read inside it (`MODEL_COMPILE=eager`
   proves the trace without Inductor); the eager compose rounds and their
   compilation in training remain item 1's.

Out of scope: the grammar operations themselves (serial by design, one
round at a time over STM); tokenization and new-type admission, which stay
on the host once per batch and are cached per word type; heat retrieval
(off).

## 4. Gates and measurement

- **Equality certificates.** Steps 1, 3 and 4: byte-equal forward values,
  costs and parameter updates against the current path on the standing
  fixtures (XOR, packing, the native worked document) and on one lesson-13
  sentence batch. Step 2: the same ranked candidate set and scores as the
  current `cued_rows`, `relation_evidence` and `witnessed_match` at the same
  work meter, on the thought fixtures and one lesson-13 question epoch.
- **Compile certificates.** Each rewritten brick traces under
  `fullgraph=True`; the one-backward certificates (16(a), 16(b), 16.2's
  arrival state and compiled brick) stay green.
- **One timing measurement**, before and after, proportionate: lesson-13 at
  the one-backward measurement configuration (eager, batch 1) and the
  math chain compiled at the July baseline's batch and brick size,
  sentences/s and peak memory, no seed. The July figure (6.8 sentences/s at
  batch 24) is retaken on the same commit as the after.
- **Allocation reads.** Fixed CSR capacities and the ragged part store are
  preallocated; no resize anywhere; concepts never die (validity bits only).
- The usual: no language named in code, no seed pinning to pass, fail loud,
  no `RUN_SLOW` before November, receipts under `doc/benchmarks/` with
  `frozen-source.json`, ports with before/after text, commands with
  environments.

## 5. Decisions (Alec, 2026-10-30)

- **D1, order:** as written ("whatever works").
- **D2, the content matrix:** Alec's intent is content-based matching of a
  vector against concepts or LTM0 entries as one matmul. For concepts that
  is the query code against the codebook of derived forms,
  $[V, \text{code width}]$. For LTM0 it is the cue's roles against the
  stacked roles of all rows, $[N_0, R, D]$, normalized per role, masked by
  the cue's occupied roles and averaged, which is what `cued_rows` computes
  row by row today (the "roles tensor" of the question meant this, not
  MTM). MTM is not an exact match: `MediumTermMemory.match_candidate`
  scores every entry by cosine, and `FamiliarTier.witnessed_match` compares
  the support centroid against each witnessed row of the same order by
  cosine at `concept_match_cos`, with the witnessed support set required
  equal and only the literal key exact. Those two loops are the same
  product: the centroid against the masked codebook rows, top-1 over the
  threshold, the support-set equality as a mask from the constituents
  store. The against lane stays a later extension.
- **D3, memory:** the ragged gather, accepted ("the start of a parallel
  perceptual engine").
- **D4, rounds:** one Codex text for steps 1 to 4, fully specified here:
  Codex does this much in one iteration when it is already specced, and
  spends too much time in testing for fine-grained todos. One freeze, one
  timing measurement.

## 6. Pointers

- Cost reading moved from FutureWork's throughput-levers section
  (2026-10-30); the September levers stay there as item 1 candidates.
- Todo: item 1.5 in the countdown, between items 2 and 1; item 0 waits on
  it.
- Code sites named above: `ModelAttention.py` (`forecast_percept`,
  `observe_percept`), `Expectation.py` (`recombine_percept`,
  `ActivationPredictor`), `MereologicalCodes.py` (`derive`, `_definitions`,
  `_subspace_bounds`), `Spaces.py` (`cs_read_memberships`,
  `_compose_order0`, `synthesize_word_parts`, `_embed_ladder_word_major`,
  `_witnessed_rows`), `EntryRead.py` (`cued_rows`), `LTM1.py` (`_index`,
  `add`), `Queries.py` (`relation_evidence`), `MemoryConsolidation.py`
  (`recurrence_groups`), `FamiliarTier.py` (`observe_field`,
  `promote_candidates`, `witnessed_match`), `Models.py`
  (`_run_sentence_word_bricks`, `stage_cs_sub`, `stage_cs_sym`,
  `SentenceFork`).
