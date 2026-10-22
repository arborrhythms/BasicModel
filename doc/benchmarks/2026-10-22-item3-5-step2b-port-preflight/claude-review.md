I reviewed this read-only: I didn't run any tests, edit anything or publish anything. P2 is correct as written. P1, P3, P4 and P5 each need an amendment. I also found five conflicts that §24 doesn't list.

## What's already covered and what isn't

**Already covered (don't ask again):** the standalone posting restore ports, the three adjacency-only test retirements, the two evidence tuples, the two-pass coarsening port with §19's snapshot port, and the role-0 read in `test_live_parent_pins_its_child`. Removing the `symbolCentroid` switch and its config entry is part of the nudge removal Alec already decided (§23.2).

**Still needs Alec, because each edits an existing test:** P1–P5, plus four of the five extra items below. P2 and P3 can be approved together as direct consequences of decisions already made; neither asks for a new design choice.

## P1 — stored mode

- **The proposed assertion would always pass.** `withdraw_origin` sets both evidence values to zero for every withdrawn row (`bin/Layers.py:9981–9982`). The question was also written with the default trust of 0, so it starts at `(0, 0)` (`Layers.py:9502–9504`). The leaf row at line 53 makes the same point already. Swapping in `evidence == (0., 0.)` drops the only check that is specific to the question.
- **Better port:**
  ```python
  inner = store.row(_index(store, inner_ref))
  assert inner["kind"] == "question" and inner["evidence"] == (0., 0.)
  ```
  This is not a stored-mode check. It reads the record kind, which 2b keeps. Withdrawal leaves that kind in place and only downgrades facts (`Layers.py:9978–9980`). It is also exactly what blocks fact authority: `accept_fact` checks it at `Layers.py:9409` and the reasoner at `bin/reasoning.py:163`.
- **§22 missed a production reader.** `reasoning.py:170` reads the stored mode (`fact.mode != "assertive"`). §22 only lists `raw_meaning_of` and `accept_fact`.
- **Open gap: what mode does a stored row report when read back?** `ConceptualMeaning` always carries a mode (default `"assertive"`, `bin/Meaning.py:153`).
  - I recommend the default `"assertive"` with no extra code. LTM1 rows already read back that way, and `test_selected_relation_meaning.py:401` checks it. With that choice, the check at `reasoning.py:170` becomes a no-op and should be deleted.
  - Choosing `"unspecified"` instead would make the reasoner skip every fact, and would change the chooser's mode inputs for every retrieved row.
- **No other test asserts a stored LTM0 mode.** `test_existence_metadata.py:100–104` depends on the mode check on the incoming meaning at write time (`Layers.py:9491`), which reads no stored column. That check must stay.

## P2 — LTM1 columns

The patch is correct. `LTM1.columns` (`bin/LTM1.py:14`) drives `row`, `retain` and the resize path in `_load_from_state_dict`. `test_part_store.py:22` derives its expected set from `columns`, so it passes unchanged. Putting `ref_kinds` right after `refs` matches how they pair.

Two things the implementation must handle (neither changes an assertion):
- `retain` clears every column except `refs` and `surprise` to 0 (`LTM1.py:130`), so 0 must not be a valid tag.
- A checkpoint saved before 2b has no `ref_kinds`, and the resize loop would raise `KeyError` (`LTM1.py:172–174`). Old rows need a load-time default of SYMBOL.

## P3 — `symbol_centroid`

The narrow port is the right call, and it is covered in substance by the nudge removal. With the nudge gone, `place` changes nothing: the cap step works out to `max(f, min(f, cap)) = f`. The rule that every assertion change goes to Alec still means he needs to nod at it.

The edit isn't safe on its own, though:
- **The snapshot check depends on the code being removed.** The certificate records `id(mereology._form_context)` (`test_operators_final_b.py:124`) and asserts `snapshot_counts == [2, 2]` (`:150`). `_form_context` comes from `centroid_forms()`'s cache (`bin/MereologicalCodes.py:51–58`, `:102`). If 2b deletes `centroid_forms`/`place` outright, or stops caching it, both checks break. 2b must keep a per-forward form snapshot, cached on the vocabulary revision, with the adjacency revision and the switch dropped from the cache key.
- **`test_review14_contracts.py:44` calls `centroid_audit()`.** Keep that method, or bring the call's removal to Alec.
- **Remove the config entry too.** `symbolCentroid` must also come out of `data/model.xml:504` and `data/model.xsd:1134`. Leaving them would be the unused switch again.

## P4 — the LTM0 `refs` inventory

The rule needs three additions:

1. **The old checks can't fail today.** No production code writes `LTM0.refs` (spec §1; it is only filled with −1 at `Layers.py:10140` and `Forgetting.py:390`). Every listed snapshot and restore comparison therefore compares columns of −1s. Ported to tagged LTM1 rows, they become real checks for the first time, so this is a strengthening, not a mechanical port. The rule should:
   - snapshot the LTM1 rows that existed beforehand, by handle (`refs` and `ref_kinds`);
   - list any new rows the operation is expected to write (relevant for `test_reference_arrivals.py:113` and `test_reference_close.py:199`);
   - treat a failure as a production finding that stops work.
2. **`test_selected_nested_meaning.py:78`** compares the forward, batch-eval and packed paths. It must compare tagged structure through the address mapping, not raw handle integers, because relation handles are allocated concept ids.
3. **`test_operators_round4a0.py` can't be ported mechanically** (the planting lines are `:98` and `:133`; the `refs` checks are `:103` and `:141`). These fixtures put an occurrence directly in role 0 with no derivation behind it. Under S2/§20 the only layout for that is an untyped occurrence-to-occurrence edge, which is ruled out. Their `role_refs[0]` assertions (`:102`, `:140`) hit the same problem. The repeated-edges test in `test_nested_retention.py:81–95` does too (role 0 is an occurrence when `twice=True`). Each fixture needs either a rewrite as a typed derivation, as §20 did for forgetting, or a decided rule for legacy role-0 occurrences. Either way that is a fixture-meaning decision for Alec.

Also:
- **`identity_measurement.py:197`** currently always reads −1 and reports "no reference". Pointing it at the reading row would change what the observer measures, so that needs to be disclosed.
- **`test_concept_forgetting.py:53/85`** can be ported by planting a reading row. Note that the new relation handle uses up a concept id, so check that the reclaim counts don't move.
- The emptiness and absence checks and the restore-equality checks are mechanical.

## P5 — corrupted postings in learning readiness

This is separate from the standalone authorization: it's the full-model `load_model` path. Removing the saved postings means the subcase can't run any more; it would end in `StopIteration` at `:190`, not a `ValueError`. Replacing it with a malformed `ref_kinds` is a new assertion.

Swapping it in place is fine if:
- the corruption goes into the exported `part_store` extras (S1), not `state_dict`;
- the rejection comes from LTM1's own restore validation and names `ref_kinds`;
- the corruption is a bad value as well as a bad shape. A reshape alone would only hit PyTorch's generic size error.

## Conflicts §24 doesn't list

1. **`test_truth_text.py:47`** asserts `store.row(row)['refs'].numel() == 3` on an LTM0 idea row. The inventory searched for `.refs` and missed reads of the `'refs'` key from `LTM0.row()` (`Layers.py:9850`). The fix is an absence port, the same as P4.
2. **`test_part_read.py:25–28`** is the full-owner stale-cache rebuild test, not a standalone one. It writes `state['posting_codes']`, which won't exist after 2b. Port: drop the stale-cache injection and keep the load, configure and `leaf_terms` assertions. It belongs in the same decision as P5.
3. **`test_accessible_mind.py:111` and `:419`** read `leaf_complete` in memory. They are unaffected if §23 is read literally (remove it from the checkpoint only, so it stays as a non-persistent buffer). That reading needs no decision. If the attribute is deleted, both assertions change.
4. **§11.1's end of minting at close time, if it lands in 2b (§3.2):** `test_sentence_identity.py:31–39` ("minted as a part argument") and `test_relation_closing.py:54–56` pin relation arguments to minted symbols. Neither the audit nor the authorization covers them. I haven't swept further.
5. **Probably already broken on published 2a, by reading the code only:** `test_selected_nested_meaning.py:81` calls `getattr(store, 'where')`, but 2a removed `LTM0.where` (`Layers.py:9333` drops it on load). This is unconfirmed; whether to run that one test is Alec's call.

## The decisions Alec needs to make

1. P1: approve the kind-plus-evidence form, the readback mode (`"assertive"` recommended), and deleting the stored-mode check at `reasoning.py:170`.
2. P2 and P3 together, with P3 conditional on keeping the cached form snapshot and on removing `symbolCentroid` from `model.xml` and `model.xsd`.
3. P4's rule with the three additions above, plus a fixture decision for the `round4a0` and repeated-edges role-0 occurrences.
4. P5 together with the `test_part_read.py:25–28` port.
5. Which round ends minting at close time; if it's 2b, a short audit of the LTM1-argument assertions first.
6. Whether to confirm the 2a `where` issue now.
