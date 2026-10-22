## Verdict: don't publish 58722725d. The production change is sound, and the failure is in the new test's setup.

The gate stays failed: 110 passed, 4 existing prerequisite skips, and 1 failure in the new test at line 145 (`test/test_reference_reconstruction.py:145`). The production hunk is correct, and I found nothing blocking in it. The failure is a defect in how the new test builds its next-sentence frame, not in production. Codex's diagnosis is correct and so is the proposed correction. Applying it still needs Alec's approval under the forced-order stop rule. I made no edits and ran no tests.

## The production change (`bin/ReferenceClose.py:795`)
- **What it does:** the one hunk drops `content*0 if anonymous`, so the own-object preview record keeps its site's content. That is the §15 R10 decision.
- **What it leaves alone:** the record is still anonymous, because `anonymous` is still derived from `forward_identity` at line 791. Prepare still sees it as unresolved and keeps the pending name; commit still opens the token with `kind=None` (lines 1054–1056). The record has no `kind` key, as the test checks at line 120.
- **Why it's consistent:** after the close, the token's value comes from `token_value`, which takes the element-wise maximum over the occurrences' rows (lines 116–138). That gives `a[0]`, the same content the preview record now shows. The asymmetry R10 was opened to fix is gone.
- **Effects:** the cost reads `record['value']` (`ReferenceReconstruction.py:27`), and the scorer reads `content`, which previously was also zeroed. So the change affects both the reconstruction cost and the resolver's coverage features. §14 declared exactly that. The scoped receipt shows the same-sentence features as `[1,1,0,…]`, logits 1 vs 0, and cost 0, where the earlier probe had a 0.5/0.5 tie.
- **Scope:** any later non-mint mention in the same close can now cover an anonymous named record, not only a repeat of the same name. That is the intended meaning of "the token exists before the close," and the existing exclusion rules still apply.

## Independent diagnosis of the failure
I confirmed it from the source:
- The root of the sentence written by `write_clause` has order 0 (`bare_frame.order = 0`). So `accessible` does not add the row itself (line 191, which needs order 1).
- Both sites are anonymous. They have no symbol and no Part edge, and `symbol_ids` is empty in both frames.
- That leaves `frame.activations` (line 233) as the only way to reach them, and `SituationFrame.from_row` always sets activations to `()`. The result is no records, no candidates, and an own object.
- Production keeps that metadata. `Models.py:20377–20379` passes `('symbols'/'activations', id(clause))`, built by `ClauseRow.py:574–580`, to `bind_observation_occurrence`, and `SituationAddress.read` puts it back on the frame.
- **Why the existing tests didn't catch this:** they also call bare `from_row`, but their antecedent is an order-1 mention row, which line 191 reaches without activations.
- **Precedent:** `test_bind_thought.keep` already builds frames as `replace(from_row(...), symbol_ids=..., activations=...)`, which is equivalent to the proposed fix.

## The proposed correction
`SituationAddress(store.occurrence_of(row), symbols=written.get(('symbols', id(kept)), ()), activations=written.get(('activations', id(kept)), ())).read(store)` is right:
- The key `id(kept)` matches production's key, which is also the prepared clause.
- It changes only how the test builds the frame; no assertion text changes.
- §13's "any other assertion change" therefore arguably doesn't apply. But the stop rule is triggered by the failure itself, so the decision is Alec's.

## Additional issues
1. **The diagnostic doesn't check line 147.** `frame_probe.py` records the next-sentence selection, features and cost, but not `later['individuals'][token]['value']`. From the source I expect it to pass: `individuals()` takes the maximum of `token_value(token)`, which is `a[0]` and already checked at line 141, with both site rows' points, which are `a[0]`. That remains unverified until the rerun.
2. **Optional, Alec's call:** adding `assert len(frame.activations) == 2` would pin the cause. It is a new assertion, so it shouldn't go in without approval.
3. **A no-op step at commit (minor, existing behaviour, not caused by R10).** Because site 1 selects `('new', 0)` and carries the name, `_arrival_plan` also plans an arrival for it (line 888). At commit, `token_of(occ1)` is still `None`, because uses are bound only after arrivals. So that arrival hits `continue` with `reference=None` and writes no audit entry. The test's count of 2 still holds, so no fix is needed for R10. It may be worth noting before step 2.
4. **An unchecked arrival in the corrected next-sentence read.** That read will also plan an arrival for the old occurrence, because its forward address is still pending. The arrival is preview-only and outside the cost's item sum, and no assertion checks it.

## Recommendation to Alec
- Keep 58722725d unpublished, with its failed receipt kept as failed.
- Approve the fixture-only frame correction, with every assertion unchanged, and decide on item 2.
- Then freeze the new source, rerun the same five scoped files once (no fresh series, no sweep), confirm that lines 145–151 pass (line 147 especially), and get a Claude review before publishing.

Step 2 should not start before then. None of this is a learning claim.
