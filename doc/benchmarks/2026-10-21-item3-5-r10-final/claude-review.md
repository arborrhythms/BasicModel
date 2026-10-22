## Verdict: publish `23a41540f`

The source conforms to §§15 and 17, the applied fixture patch is exactly the one Alec approved, the production change is correct, and the evidence supports what the receipt claims and nothing more. I edited nothing and ran nothing.

## The fixture patch
- **It is the approved patch.** The hunks in `r10-final/source-diff.patch` match `r10/proposed-fixture-correction.patch` exactly. Only the git header lines differ.
- **The patch base is right.** Its pre-image blob is `4d3259c7f`, the same post-image the failed `58722725d` patch produced. So it applies to the failed source directly, and test and production are unchanged apart from these lines.
- **Only the test file changes.** The diff touches one import and replaces one frame-building line with three. No production or configuration file appears in it.
- **No assertion changed.** The 39 in `source-audit.json` is the file's 29 bare `assert` statements plus 10 `assert_close` calls; I counted them by hand. None is inside the changed hunk, and nothing was added. The delegated review's optional `len(frame.activations) == 2` assertion was correctly left out.
- **Line mapping is right.** In the failed source the frame line was 144, the selection check 145 and the value check 147. Three added lines move 145–151 to 148–154, so the value check is now line 150.

## It reads the frame the way production does
- `SituationAddress.read` (`bin/ReferenceContext.py:46-49`) is `from_row` plus the stored `symbol_ids` and `activations`. The only difference from the old fixture is the metadata that was missing.
- Production builds the same address (`bin/Layers.py:11009`). The metadata comes from `written_clauses[b][('symbols'/'activations', id(clause))]` (`bin/Models.py:20378-20379`), where `clause` is the clause passed to `write_clause` (`bin/ClauseRow.py:574-580`). The fixture uses `id(kept)`, which is the clause it passes to `write_clause` at line 132, so the keys match.

## Production (unchanged from `58722725d`)
- The single hunk at `bin/ReferenceClose.py:795` stores `content` as both content and value. Lines 790–791 still derive `anonymous` from `forward_identity`.
- Prepare still treats the anonymous selection as unresolved and keeps the pending name (lines 944–958, 968–969).
- Commit still opens the token with `kind=None` (line 1055).
- I agree with both earlier reviews that this is correct.

## The new gate's capture (`scoped-state/.../outcome.json`)
**Same sentence**
- Site 0 has an empty menu and becomes an anonymous own object.
- Site 1 has candidates `[('new', 0)]`, features `[1,1,0,0,0,0,0,0]`, logits `0/1` and probability `.7310586`. It is greedy, with no departure.
- The record's value is `a[0]`.

**Next sentence**
- It selects `('ltm','address-v1',-6527…)`, the introducing token, with the same features, logits and probability.
- The record's value is `a[0]`, which is what line 150 compares against `record['value']`.
- The record holds both occurrences, so the activations reached the menu.

**Other checks**
- `scoped.json` exits 0, and `scoped.log` shows 115 of 115 cases with `mtm_reidentification: 4` skips.
- 44+22+16+16+13 = 111 passed. That agrees with the failed run (12 passed plus 1 failed in the reconstruction file becomes 13).
- A pass means lines 148–154 all ran, including the read-isolation checks at 153–154.

## Notes (none block publication)
1. **The plan's `cost` is not the reference cost.** The same-sentence plan records `"cost": 1.0`. That is the resolver's diagnostic-only cost (`bin/ReferenceClose.py:787-789`): it adds 1 when a site's menu is empty, here site 0. It doesn't enter R. The registered reconstruction cost is asserted zero at lines 123 and 151.
2. **The expected step-2c arrival appears.** The next-sentence plan holds the preview-only arrival for the still-pending address, as §17's second note anticipated. The same-sentence plan holds arrivals for both joined sites. Both are outside the reference term and are already carried into 2c. Nothing to do now.
3. **Some receipt files aren't committed yet.** `gate-summary.json`, `frozen-source.json`, `scoped/`, `scoped-state/`, `scoped.json`, `scoped.log` and the review files are untracked, and `README.md` has uncommitted edits. That doesn't affect the frozen source hash, but they should go into the receipts commit with this review before both mains and the WikiOracle bump.
4. **No standing exception was used.** This correction rests on Alec's explicit §17 approval, not the new exception, and the README cites it that way. The failed `58722725d` receipt and its diagnostic stay a retained failure. Nothing from them counts as evidence for this pass.

## Evidence scope
The receipt claims one scoped run of five files on `23a41540f`. It makes no learning claim and ran no fresh series or sweep, and it promotes nothing from the failed run. Those claims match the retained files.

Step 2 can start after R10 is published.
