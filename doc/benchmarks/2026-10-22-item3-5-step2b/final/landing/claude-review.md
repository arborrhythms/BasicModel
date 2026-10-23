## Verdict: **HOLD**

I could not test any of this: per your instructions I only read the files, and ran nothing. I reviewed the frozen source `3c5cc2a1d5243dd80c2cdcfc3cc2ebcdaa98c56c` against its gate summary (683 cases: 675 passed, 8 existing skips, 0 failures), the landing receipt and ledgers, §§36–57, bind §10.4, and the production files themselves.

Most of the round is sound. Two problems block landing, and a third should be fixed in the same re-freeze. None of them needs a new sweep.

### Blockers

**B1. Removing the stored mode also changes live behavior, which §22 kept.** The receipt says no `LTM0.MODES` consumers remain (`landing/README.md:50-54`). That is true of the list of mode names, but not of the mode value itself:
- The sidecar now saves only `role_refs`, `bindings` and `scope` (`Layers.py:9533-9534`), and `raw_meaning_of` no longer passes a mode (`Layers.py:9134-9139`). So every stored row reads back as `mode="assertive"` (the default at `Meaning.py:153`).
- **Closing questions:** `ThoughtClosing.py:19-24,38-43` replaces the live `parsed` meaning with the row just written, then sets `surface_open = source.mode == 'interrogative'`. That is now always false. A yes/no question with zero evidence and no open slot, stored as an ordinary row, gets no `question()` call, so `needs_episode` (`ThoughtReferences.py:52-55`) is false and no episode opens. §22 names this exact live use: "the live parse keeps its mode in this round… deciding whether a yes/no question… opens an episode."
- **Answer seeds:** `Models.py:20407-20410` builds the sentence's end state from the stored meaning. `Models.py:8714-8722` then no longer skips interrogative fields, so questions become answer seeds on the training/answer path.
- **Held questions:** `Models.py:5738-5749` resolves a stored question through `Queries.py:1131` (`meaning_of`). It now always reads as not interrogative, so the loop always stops.
- **Fix:** take surface mode from the live clause or parse, not from the re-read row. In `closing_question`, use `parsed`/`clause.meaning.mode`. In `_commit_sentence`, apply `clause.meaning.mode` to the retained meaning. Port or explicitly decide the `Models.py:5745` reader, and add a regression for a zero-evidence yes/no question through `write_clause` + `closing_question`. Record these consumers in the §41.3 list.

**B2. The completed-sentence count ports are not complete.** A completed relative `write_clause` now leaves one LTM0 row; your own `test_step2b_storage.py` and `test_sentence_end_state` assert `len(store) == 1` for `part_clause()`. These unported checks will fail on the frozen hash:
- `test/test_relative_sentence_codebook_insertion.py:15`: `len(store) == 0` after `part_clause` writes.
- `test/test_lies_can_be_learned.py:10`: the same.
- `test/test_selected_relation_meaning.py:397`: `len(store) == 0` after `commit_reading` of a completed Part.
- Also worth checking: `test_taxonomy_policy.py:51`, `test_replacement_kernel_contracts.py:39,124`, `test_sentence_expectation.py:265`, `test_item55_mechanisms.py:169`.

The cause is that `module-coverage.json:2-14` leaves out most files that call `write_clause` directly (about 15, including the ones above, `test_identification_evidence`, `test_predicate_identity`, `test_live_references` and `test_relation_references`). The fix is to add these §36 ports to the ledger and add those files to the module group. That is a targeted selection, not a sweep.

**B3. The §43 `refs` repair in `Models.py:20415` adds a new risk (fix in the same re-freeze).** The old LTM0 column was never filled by live writes, so this field used to be `(-1, -1, -1)`. It now takes `ref[-1]` of each role reference. For a slot reference (`('slot', occ, role)`, written at `ClauseRow.py:523-524` via `:332`) or a constituent reference (`('constituent', i)`), that gives the role or index number. When that number is 0, `Understanding.py:217` raises; otherwise it puts a meaningless number in the field. The safest fix is to keep `(-1, -1, -1)` for stored rows, or map only `('sym', id)`.

### Checked and sound
1. **Scope:** storage-only. The relative occurrence sits beside the same edge and the Part handle is still returned (`ClauseRow.py:497-558`). No tags, READING, MTM or derivation changes.
2. **Postings and P5:** postings are absent from saved state and are rebuilt after the full owner loads (`Models.py:4428-4431`); `leaf_complete` is non-persistent. P5 has both approved Part-pair subcases with fresh copies and the actual owner (`test_learning_readiness.py:189-204`). Adjacency, the nudge and `symbolCentroid` are gone. Questions are still refused by kind.
3. **Evidence:** the occurrence keeps its pair and only a trust-derived Part edge reverses (`ClauseRow.py:464-475`, `Layers.py:9489-9493`). Explicit pairs stay explicit.
4. **Forgetting:** the conservative barrier is justified (§42). Every Part child plus its transitive closure is protected (`Forgetting.py:145-155`), and owned structure is snapshotted before either table advances (`:424-439`). Incoming references are frozen per pass, coarsening clears only `role_refs` (vectors are not rewritten), and `forget_pending` is saved and restored.
5. **§46 Part-read ports are legitimate, not a weakening.** `read_reference_bindings` (`ClauseRow.py:608-652`, which the diff does not touch) has no equality path, and the fixtures make no resolver choice. So the old assertions were certifying the binder that bind §10.4 retired, and they fail with or without 2b. The new checks match §10.4's "a kept sentence reading… never binds anything by itself." The selected refinement and content stay gated in `test_reference_arrivals.py` (kind == three, `head_reference`). §40 stays operand-scoped, and the §56 journal fixture is correct: `_run` uses a no-op boundary while a sentence run is already active, so the disabled consumer stays disabled.
6. **Other production repairs:** the iterative participant traversal keeps the same visit order. Recall isolation restores the caller's own history object and batch stamp on both success and exception. The ClauseJournal Part-only branch and the Queries fallback removal are sound.

### Non-blocking follow-ups
- **Narrow-idea port** (`test_sentence_retention.py:63-64`): both writes probably resolve to the same row, so line 63 likely compares a row with itself. A sharper check is "no `('ltm', …)` reference at all", with a replay against the pre-2b writer as §45 did for the Part counts.
- **§46 receipt wording:** say these three Part-read tests were already failing since §10.4, independent of 2b. That means published main carried failing tests.
- **`'where'` comparison:** dropping it from `test_10_packed…` (it was already gone from LTM0) is disclosed only in the fixture ledger. It belongs in the assertion ledger.
- **Other mode effects to record:** `ThoughtFeatures.py:61,92` now sees every stored record as assertive. An interrogative stored with nonzero evidence as an `observation` can now be accepted as a fact.
- **Dead code:** `_native_predicate_index` is always empty (`ClauseRow.py:288-295,344-346`).
- **Wording:** the README calls the first frozen failure a "native end-state case" while the request says "module failure". It is the same test in the module group.

Once B1–B3 are fixed, a re-freeze should rerun the three scoped groups with the added write-clause files. No learning claim is affected.
