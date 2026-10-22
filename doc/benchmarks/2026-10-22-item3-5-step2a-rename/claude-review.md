**Review of eb6980e7d (rename) and the §14 cull preflight, 2026-10-22**

I only read files. I ran nothing and edited nothing.

## Finding 1: the rename is safe to publish, with one check I couldn't do

**My verdict:** the BasicModel rename is safe to publish. Nothing in this repository blocks it.

What I confirmed:
- **Module and class paths.** `bin/LTM1.py:13` defines `class LTM1` and `bin/Layers.py:8727` defines `class LTM0`. `bin/PartStore.py` is gone and nothing provides the old names. Every live import uses `from LTM1 import LTM1` (`Layers.py:5798`, `Layers.py:8799` and six test files).
- **No leftover old names.** There are no `TernaryTruthStore` or `PartStore` identifiers anywhere in `bin/`, `test/` or `data/`. That includes longer names containing them and their removed lines in `source-diff.patch`. The only remaining mentions are in dated historical docs, such as `todo.md:209` and older plans.
- **Class names don't leak into saved data.** None of the code that builds names from `__class__.__name__` applies to `LTM0` or `LTM1`. `Layer` is a plain `nn.Module`, not a codebook owner (`space_carrier.py:81`).
- **Checkpoint dependence: none found.**
  - The writers save state dicts (`embed.py:3013`, `util.py:2103`, `IndependentComponents.py:20`), not module instances.
  - In `checkpoint-class-paths.json`, the pickles reference only builtins, `collections`, `numpy` and `torch`. There are no project-module pickle references in any of the 91 files.
  - `old_name_hits` is `[]` and all eight registered phases are null.
- **Gate scope.** `gate-runs.json` runs exactly §2.0's two gates on this hash: 113 passed with 7 existing skips, and 85 passed. The pending-premise result is 0 bound / 8 open forward and 0 bound / 16 open reverse. The receipt doesn't promote any earlier evidence.

Two residual items. Neither blocks the BasicModel side:
- **WikiOracle is unchecked.** Your workspace limits me to `basicmodel/`, so I couldn't search the parent WikiOracle repository. Publication pairs with a WikiOracle commit, so check it for `PartStore` or `TernaryTruthStore` imports before publishing. If any exist, that is a blocker.
- **Most changed test files weren't run.** About 70 test files changed, but the gates only collect about 4 of them. For example, `test_forgetting.py`, `test_clause_storage.py`, `test_part_read.py`, `test_priming_diffusion.py` and `test_sentence_priming.py` all import `LTM1` and weren't run. The exact identifier-only source audit covers this, and the gate scope matches the spec. Running a collect-only pass over the changed tests would close the gap if Alec wants it; it isn't required.

## Finding 2: no listed cull meets §10's condition today

§10's own examples were wrong about the code, and that is my error, not Codex's. "`refs` (no live writer)", "`where` (no numerical consumer)", "negative stance is the exchanged lanes" and "a question is an open reference" all need the qualifications in §14. **None of the listed culls has a derived source in its readers today.** A passing rename gate approves no cull.

Corrections and additions to §14's code facts:

| Field | Correction or addition |
|---|---|
| `rel_type` | Accurate (`Layers.py:9324`). There is nothing to cull; keep the loud rejection. |
| `refs` | **Readers missing from §14:** concept protection at `ConceptForgetting.py:66`, forgetting scoring at `Forgetting.py:166`, the `row()` export at `Layers.py:9849`, and `point_of_row` via the predicate index at `ClauseRow.py:349`. The coarsening write is at `Forgetting.py:390`. **Citation fix:** `test_forgetting.py:139` only sets up the value; the check that it is cleared is at `:156`, and `:126` compares `refs` state. `test_clause_storage.py:95` asserts that `write_clause` leaves `refs` at -1, which supports "no close-time writer". |
| `where` | **This is a live close-time stamp, not a dead column.** `ClauseJournal.py:256-264` stamps it from `program.symbol_where`, then `ClauseRow.py:561` writes it. That is the same path that feeds `.when`. |
| `metadata_required` | It also has readers outside restore. `raw_meaning_of` returns None when the flag is set and the context is missing (`Layers.py:9133`). It also feeds `row()['metadata_complete']` (`Layers.py:9862`) and `Language.py:11865`. |
| Leaf postings | Accurate. On a standalone restore without an owner, `reindex_meanings` (`MemoryIndex.py:260`) can't regenerate the incomplete terms that the saved postings carry. |
| `adjacent_*` | **Function name is wrong.** The writer is `Models._commit_sentence` (`Models.py:20368`); there is no `_sentence_keep`. Otherwise accurate: `reindex_meanings` doesn't touch adjacency, so these columns are data, not a cache. |
| `polarity` | Needs one nuance. When no evidence is supplied, `append_meaning` does fold polarity into the lanes (`Layers.py:9504`). The bit is still lost when trust is 0. With explicit evidence (`:9507`), polarity is fully independent, which is what the probe shows. |
| `grammatical_mode` | There are three modes, including `unspecified` (`Layers.py:8743`), which the legacy migration assigns (`:9375`). An open reference can't produce `unspecified`. `accept_fact` reads the column (`:9406`). |

The probe's caveat holds: it shows rows the write interface accepts, not rows that live closes are known to produce. Removing the polarity or mode gap by rejecting such rows would be an admission change, and that needs Alec.

## Dispositions

**Already authorized, needing no decision.** Under §10's "each only where its readers already have the derived source", every listed field simply stays in place. `rel_type` is a no-op. The two R10 arrival notes stay in 2c unchanged.

**New decisions for Alec, with my recommendation for each:**
1. **Land the rename as its own commit.** Yes, once the WikiOracle import search comes back clean.
2. **`refs`:** retire it in 2b, when LTM1's tagged references replace it. Port the `test_forgetting` coarsening tests to the stored `role_refs` field then, not now.
3. **`metadata_required` and `semantic_fingerprint`:** keep them through 2b. They authenticate the sidecar data that 2b moves into LTM1, so revisit them once the sidecar is gone.
4. **Leaf postings and `leaf_complete`:** keep them. A cull needs an S10 port first: either an owner-only restore, or a specified wire view for standalone restore. Decide that within 2b.
5. **`adjacent_*`:** reclassify them as retained data, not a cache. Decide in 2b whether word-pair wholes become LTM1 rows or stay a justified column, as `order` did.
6. **`polarity`:** keep it as a justified column. Narrowing admission to fold polarity into the lanes would need its own spec.
7. **`grammatical_mode`:** keep it until 2b's question layout from §10's S2 exists, and decide what happens to legacy `unspecified` rows.
8. **`where`:** keep it as a justified column beside `.when` until the 5.5 context-wholes spec. This overrides §2.1's removal list, so it needs Alec's explicit approval.
9. **Amend §10's 2a text** so its examples match the code facts above. All these culls stay explicitly on step 2's line. Neither 2a nor line 2 is complete, and no learning claim is made.
