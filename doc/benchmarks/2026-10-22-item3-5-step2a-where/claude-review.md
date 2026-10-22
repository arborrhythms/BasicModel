## Verdict: publish, no blockers

I reviewed source `d4f316411` read-only. I ran no tests and edited nothing.

### Scope: matches the claim
- **Production code changes only in the three named files** (`source-diff.patch`):
  - `bin/Layers.py:8765`: the buffer is removed.
  - `bin/Layers.py:9333`: checkpoint load drops the old column.
  - `bin/Layers.py:9356`: the column is removed from the load defaults.
  - Row export (former `:9852`), compaction columns (`:10094`) and reset (former `:10143`): `where` is removed.
  - `bin/ClauseRow.py`: the close-time copy (former `:561`) and the synthetic relation-row location (`:244`) are removed.
  - `bin/ReferenceContext.py`: the frame field, its `detached()` copy and its `from_row` read are removed.
- **`.when` is unchanged.** Its close write at `bin/ClauseRow.py:560` is still guarded by `row >= before_count`, and the load defaults, compaction, reset and `RowTime.migrate` are untouched. Metadata authentication, polarity, modes, refs, `adjacent_*` and leaf postings are not touched.

### Authorizations: exact
- The exact-value removal at `test/test_clause_storage.py:96` is what §16 authorized.
- The shape port at `:23` matches `proposed-shape-port.patch` and `shape-port-authorization.json` character for character. `store.when.shape == (8, 4)` is still asserted.
- In `test/test_independent_components.py:295–296`, the only change is dropping the `where=` argument. `when=` stays and no assertion changed.

### Restore behaviour: correct
- The old column is popped before row resizing, the default-filling loop and `super()._load_from_state_dict`. So `strict=True` never sees it as an unexpected key.
- `RowTime.migrate` only zeroes `.when` when `model_clock` is missing, which the new test does not trigger. So the exact `.when` checks really do run through restore and compaction.

### No missed live consumers
I searched every non-`torch.where` use of `.where`, `'where'`, `where=` and `[‘where’]` in `bin/` and `test/`. What remains belongs to other owners:
- the sparse `where` on `ConceptualAttentionLayer` (`ConceptForgetting.py:92`, `test_concept_evidence.py`)
- percept, SubSpace and Meronomy fields
- `What.where`
- `MemoryIndex._scope_contains`, which reads sidecar scope, not the column
- the Clause working field

Two other checks:
- `Models.py:20408` builds `SentenceEndState` from `clause.where`, not from the stored row's `where`.
- No caller passes enough positional arguments to `SituationFrame` to be shifted by the removed field; the most is five.

### Gate coverage: as claimed
- The selectors in `frozen-source.json` match the commands in `gate-runs.json`.
- `storage/result.json` shows both `[False]` and `[True]` cases of the new test and the time-isolation case, each passing.
- Totals are 25 + 85 + 113 = 223 passed, 7 skipped, no failures. `full_sweep` and `fresh_series` are both false.

### Notes (none block publication)
1. **The "legacy" case isn't an old-schema checkpoint.** `test/test_ltm0_location.py:34–35` adds `where` to a current-format state dict. That is enough here because the pop doesn't depend on anything else and happens first. But `README.md:25` ("a legacy checkpoint") would be more accurate as "a checkpoint carrying the retired column".
2. **One small behaviour change.** Before, the close overwrote `where` even when it re-witnessed an existing row; now nothing is written for those rows. Since the column is gone, this has no effect. I mention it only for the record.
3. **`SentenceEndState.where` still exists.** It is copied from the clause and kept in the bounded recall history (`Models.py:20418`). I read this as the percept/working field that §16 keeps. Alec may want to confirm that recall history doesn't count as LTM; that can wait for the 5.5 context-wholes spec.
4. **Not checked here:** my tools are restricted to `basicmodel/`, so I couldn't search the parent WikiOracle tree for uses of `SituationFrame.where` or `row()['where']`. A quick grep there before the `dbd9b52` successor bump would close this. I also relied on the git-status snapshot (only doc paths changed) rather than recomputing the frozen hash myself.

The 2b representation questions and the R10 arrival notes are out of scope, as you said. Nothing here makes a learning claim or reaches item 3.0.
