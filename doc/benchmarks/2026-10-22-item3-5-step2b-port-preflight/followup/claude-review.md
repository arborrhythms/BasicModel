## Verdict

This is nearly a sound package to put to Alec. P1–P3 are right as amended, and P4–P5 need only small tightening. Three items are not exact yet, and two of them would change semantics without anyone noticing:

1. **P7 needs a reader precedence rule.** Without one, `_derivation_symbol` would start returning the pending word, which breaks the antecedent the arrival tests depend on.
2. **P6's nested-terminal row has no decided place to record the occupation.**
3. **P6's full-inventory row needs a narrower inventory check**, and it should name the existing atomic-exhaustion test.

As instructed, I didn't edit, run or publish anything, and I didn't audit anything beyond the listed tests.

## 1. P1–P5

- **P1: sound.** `row()` exports `evidence` (`bin/Layers.py:9848`). The kind is still `question` after withdrawal, and that kind is what keeps fact authority away. Default `assertive` on read and deleting the mode conjunct at `bin/reasoning.py:170` are both correct. One wording fix: line 170 has no separate evidence guard (trust only signs the support at `:182`). Say "kind (`:163`), metadata (`:167`), shape, mask and content (`:172–176`) guards stay."
- **P2: sound.** One clarification: only occupied pre-2b rows (`refs > 0`) restore as SYMBOL. Empty rows get 0, which matches `retain` (`bin/LTM1.py:130`).
- **P3: sound.** The `[2, 2]` snapshot count depends on `id(_form_context)` staying stable, and the cache keyed on vocabulary revision (`bin/MereologicalCodes.py:51–58`, with the adjacency entry and switch dropped from the key) keeps that. `test_review14_contracts.py:44` only checks that nothing is mutated, so `centroid_audit()` returning `{}` is fine.
- **P4: sound, with two fixture points.**
  - The `round4a0` and repeated-edges rewrites are the right kind of change. But "a declared typed derivation" isn't exact enough to approve yet. Name the operation whose declared participant is the child. For repeated edges, also say what carries `(root, 2, child)`: the derivation's second argument, or the retained LTM0 `role_refs[2]`.
  - `records_scanned == 2` (`test_nested_retention.py:95`) is one of the literal counts the new rows may change. State it exactly.
- **P5: sound, with three exact additions.**
  - The bad value should be an out-of-range tag on an occupied row, which LTM1 can reject on its own. An OCC pointing at a missing target only fails the owner's check after both tables load, so it doesn't test LTM1's validation.
  - The shape check has to run before `super()._load_from_state_dict`. Otherwise `alloc.parts.load_state_dict(saved["part_store"])` (`bin/Models.py:4151`) raises PyTorch's generic `RuntimeError` instead.
  - Keep `pytest.raises(ValueError, match='ref_kinds')`.
- **`test_part_read.py:25–28`:** once the injection is gone, the `999` check can't fail. Recommend replacing the injection with an assertion that no `posting_*` key is in `state`. That carries the original stale-cache obligation honestly. Keep both `leaf_terms` checks.

## 2. P6: the close-time-mint audit

These follow §11.1, with these corrections:

| Test | Assessment |
| --- | --- |
| `referenced_phrase_is_minted…` | **Follows §11.1.** Assert exactly: `refs[1]` is tagged OCC to the existing constituent (`len(f.store) == 1` is unchanged); its reading row reaches edge `e` with `refs[e] == [chases, mouse]`, both tagged SYM; and `_csw_row_of(e) is None`. Compare the occurrence's point to `clause.children[0].point` at the existing tolerance. A narrowed point that differs is a stop. |
| `mm_grammar_full_inventory…` | **Needs correction.** `inventory(f.cs) == before` can't stay whole, because relation handles still advance `next_id`. Port it as: `rows` and `placement` unchanged (no seats); `next_id`, LTM0 and LTM1 deltas exactly as the declared layout says. "Retain separate atomic failure coverage" should name the existing test, `test_part_closing.py:47–55` (`complete_derivation_capacity_failure_has_no_prefix`), not imply a new one. |
| `new_relation_argument…unseatable…` | **Follows §11.1.** Make it exact: one write under zero order-1 seats succeeds. Drop the restore-caps rewrite, since a second write would add a second occurrence. The bare order-1 argument gets a new planned occurrence with no reading row, since it has no root. State that, and assert `csw_rows` unchanged. |
| `nested_terminal_mint…` | **Not exact: there is no place for the occupation.** Occupations are keyed by symbol and skip a bare word (`bin/PartClosing.py:265–271`). An OCC can't enter that map untagged (S6). The options are: drop the stored occupation, since the derivation edge already records the position; key it on the word, which changes the `:267` rule; or add a tagged key, which is new layout. This needs Alec. I recommend the first option, which means the `counts`/restore checks are explicitly retired, not ported. Separately: the reading `Part(OCC t, SYM word)` must not be indexed as a definition. Assert that `definitions.words` gains nothing. |
| `minted_surface_phrase…` | **Follows §11.1** if the layout is stated: a new occurrence `p` with reading `Part(OCC p, SYM word)`, an argument tagged OCC `p`, no definition edge, and no seat. |
| `three_slot…numerical_operands` | **Follows §11.1.** The two idea occurrences already exist, so it's a clean port. Change "exactly" to "at the existing `assert_close` tolerance". As written it tightens the check. |

The two related positive-symbol checks (`test_relation_closing.py:35`, `:93`) are correct as written. Two notes:
- The `relations()` sets now include reading rows. `relations('definition')` must still exclude them, or `definitions.numel() == 0` changes meaning.
- The comment at `:65–66` ("arguments now require native seats") goes out of date. `concept_rows=64` stays.

None of the fixtures has a determiner, so §12's object-at-determination path isn't involved.

## 3. P7

- **Representation port vs. changed antecedent.** `_derivation_symbol` (`bin/ReferenceClose.py:97–114`) reads the raw role 0 first and only falls back to MTM's `kind` when that is `None`. If "the derivation root" simply becomes "the reading row's root", a pending token reads back as the word `x`. Since the reading is never rewritten, it would still be `x` after the arrival. That breaks `test_reference_arrivals.py:106` (`== words['three']`), changes `:61`'s selection, and turns pending arguments into `SYM x` at `bin/PartClosing.py:119`. **Exact correction:** for an occurrence that has an MTM token, the token's `kind` wins, and a pending token returns `None`, not its word reading. This is an implementation consequence that keeps the current semantics; it isn't a test change.
- **The raw pending-word port at `:76`/`:184` is sound** under that rule. Raw reading == `SYM words['x']`, `role_refs[1:] == (None, None)`, and the MTM `kind is None` stays.
  - For `:184`, say whether `used`'s reading is its own `Part(OCC used, SYM x)` with the token link staying in MTM/bindings. That's my recommendation. The alternative, an OCC→OCC relation, would be new layout.
  - Add the strengthening that P4 already implies: snapshot the pending reading row before the arrival and check it is unchanged afterwards (`:112–113`).
- **The cycle fixture is sound.** Write `refs`, `ref_kinds` and `count` directly into the buffers, not through `load_state_dict`, in case restore checks for acyclicity. The 2b index rebuild that `leaves` triggers must not reject the cycle. `{2, 3}` stays as it is.
- **Writer consequence of §11.2.** Today `add(11, 2, symbol=10)` followed by `add(10, 3, symbol=11)` succeeds, because `11` passes as a symbol. 2b's writer must also reject an explicit `symbol=` handle that an existing row already references. Otherwise the public API can still build a cycle.
- **Spot check, not an audit.** `test_nested_retention.py:147–196` corrupts the `role_refs` in `_semantic_rows`. If 2b moves those references out of the sidecar, they need the same corruption-only port. If not, nothing changes.

## Already-authorized implementation consequences (no test decision)

- Remove `reasoning.py:170`'s mode conjunct.
- `ref_kinds`: zero is invalid, occupied pre-2b rows load as SYMBOL, and LTM1 validates value and shape before `super()`.
- Keep the vocabulary-cached form snapshot and `centroid_audit()`; remove `symbolCentroid` from `model.xml`/`model.xsd`.
- `leaf_complete` becomes a non-persistent buffer.
- Argument occurrences are planned and published in the close (S4/§11.1).
- Reading rows are never indexed as definitions (S6).
- `_derivation_symbol` gives MTM precedence for tokens.
- The writer rejects a reused or referenced `symbol=` handle (§11.2).

## Test authorizations Alec still needs

1. **P1:** the kind-plus-evidence assertion.
2. **P2 and P3:** the two column edits and the conjunct removal.
3. **P4:**
   - the absence port at `test_truth_text.py:47` and the `where` removal at `test_selected_nested_meaning.py:81`;
   - the by-handle snapshot rule and cross-path normalization;
   - disclosure that the `identity_measurement` observer changes what it measures;
   - planted reading rows in `test_concept_forgetting`;
   - the three typed-derivation fixture rewrites, with exact layouts.
4. **P5:** two corruption subcases replacing the one, and the `test_part_read` injection replaced by an absence check.
5. **P6:**
   - the six changed expectations as corrected above;
   - the nested-terminal occupation (a decision, not a port);
   - the two positive-symbol replacements.
6. **P7:** the root-reader routings, the `:76`/`:184` pending-reading ports plus the unchanged-reading snapshot, and the corruption-only cycle setup.

Item 3.0 is not reached.
