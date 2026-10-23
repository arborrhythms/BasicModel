**Verdict: LAND**

This review used only Read, Grep and Glob. I ran no code or tests and edited nothing. There are no blocking findings for `6a36b7ceeb5a5da9ae6019f62c59c16d27f7f1ce`, and no learning claim to assess.

### N1: the repair works and stays narrow
- **What changed.** The function now has one new rule: if `index.objects(identity)` is nonempty, it returns `None` (`bin/Spaces.py:16660-16672`). Otherwise it uses the §38 baseline's `interpret.lexical_form(identity)`. `n1-repair.patch` and the cumulative `source-diff.patch:1108-1120` agree on this.
- **Ordinary bound words keep their old behaviour.** In the baseline, a word with an object is not `no_object` (`bin/Interpret.py:47-51`), so `lexical_form` read `definitions.words(word)` (`Interpret.py:71-73`). For an ordinary bound word that is empty, so it returned `None`. The new rule returns `None` too. Bound words gain no surfaces, so terminals, readback, the rung-zero audit, `RadixLayer.reverse` candidates (`Layers.py:12663`) and `derive` (`MereologicalCodes.py:232`) see the same set of rows as before.
- **Objects and objectless words are unchanged.** Their `objects()` list is empty, so they take the same `lexical_form` path as before. This includes the `_field_pending` and synonym cases (`len(words) != 1` → `None`).
- **The dual-role fault is fixed.** That case is a word that is also another word's object. In the baseline, `words(word12) == (plus,)`, so its row spelled `plus` over a native code built from `12`'s parts. Now that row spells `None`. `derive` then uses its own part interval rather than pinning a definition form. This is the only row class whose output differs from the baseline.
- **The regression port is correct.** `test/test_step2b_storage.py:19` checks the word has no surface before the second definition. Line 22 checks it has none after, and would fail under the baseline (`b'plus'`). Line 23 checks the object still spells `b'12'`. Both native-code decode checks (`:24-26`) and both link checks (`:21`, `:27`) are verbatim. §65 declares this as a contract correction, not a setup fix. That is the right label.
- **Consumer coverage is complete.** Every test file that calls `word_surface_for_row` or `lexical_form` is in a selected group: `step2b_storage`, `definition_integrity`, `word_store`, `mixing_reconstruction_bank`, `tied_operator_reconstruction` and `echoic_decoder`. The four consumers I named are present (`frozen-source.json:97-100`). The two remaining callers (`operators_run_audit.py`, `objective_conflicts_probe.py`) are probes, not tests.

### §66: echoic fixture port
- **The failure is not caused by 2b.** The test patches in `reverse_binary_step`, `_reverse_of_binary_op`, `_finish_binary_inverse`, `generate_unary_step`, `generate_policy_logits`, `choose_generate`, `_bounded_binary_reconstruction`, `decoder_eligibility` and `readback_scores`. Their homes, `Language.py`, `Generative.py` and `SentenceUnderstanding.py`, do not appear in `source-diff.patch` at all. The `Layers.py` hunks touch only LTM0 (`:401-436`), not `Ops`. That is stronger evidence than the seven AST hashes. The test was therefore already failing on `bb0045d7a`. It only surfaced now because the file was brought into scope.
- **The port is minimal.** It changes the literal plus one comment (`test/test_echoic_decoder.py:110-111`). The kernel uses code directions only (`Layers.py:15145-15147`), and `[1,.5]` and `[2,1]` point the same way, so the composed root is identical. Directions, the policy, budget eight, the monkeypatch (`:128-135`) and every assertion (`:121-125`, `:137-141`) are unchanged. Nothing is forced or masked. The `[[2,1],[1,2]]` failure (`count [0,0]`) and the passing cube diagnostic are retained.
- **Minor wording note.** In this read-only pass I could not find the exact check that enforces the "declared `[0,1]` cube domain". My guess is that bank norms above 1 affect the eligibility residuals or the progress test. The port is valid either way, since production is unchanged and the assertions are verbatim. A one-line pointer in the receipt to the rejecting check would help.

### Gate evidence
`gate-summary.json` is bound to `6a36b7ce…` for every group:

| Group | Passed | Skipped | Completed / selected | Exit |
| --- | ---: | ---: | --- | ---: |
| Regressions | 85 | 0 | 85 / 85 | 0 |
| Native | 123 | 7 | 130 / 130 | 0 |
| Modules | 936 | 2 | 938 / 938 | 0 |
| Total | 1,144 | 9 | 1,153 | |

The report hashes are recorded and `full_sweep` and `fresh_series` are both false. The selected files total 92 (8 + 8 + 76), and every changed test file in the diff is among them. No new skip was added.

### Non-blocking notes
1. **A word whose only object is another word.** "plus" defined as naming `word12` now has an object row that does not spell. So "plus" has no emitting terminal through that object. The receipt states the rule ("returns None for a word with an object") but not this effect. This is the open question of who owns a dual-role row, which belongs in the identity spec.
2. **Earlier notes.** B1–B3, the §41 dispositions and my earlier non-blocking notes are declared as accepted and their implementation is unchanged. The observer key collision (old note 7) goes away with the narrow rule. The five indirect-consumer files stay with the step-2 end sweep.

Publication can go ahead as planned: push BasicModel main, bump the WikiOracle gitlink, then the §§34–35 identity review.
