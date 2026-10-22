Review item 3.5 R10 on frozen source 58722725d50ab5d009c98f99c27e93c39458dac0, relative to published a514a71e0. This is the actual Claude review requested by Alec. Read only; do not edit files or run tests. The review must retain the failed gate as failed.

Authority: doc/plans/2026-10-21-reconstruction-reads-the-bind.md sections 12–15. Alec's latest handoff says line 1 lands first at source 3a9db635e with receipts 63c600712 (done: BasicModel a514a71e0, WikiOracle b49960a), then R10 is a separate small commit: a named own-object preview record keeps its site's content before the kept close writes MTM; it stays anonymous and pending until arrival refinement. New repeated-name gate: forced order-one reading permitted by section 13, two nonempty mentions in a sentence, second binds first at zero reference cost with nonzero resolver features, matching the next-sentence mention. Five scoped files only; no fresh series or sweep. Receipt, Claude review, then publication. The standing rule says any failure under forced order stops for Claude and Alec; source and assertions have therefore not changed after this failure.

Read:
- bin/ReferenceClose.py preview, prepare, commit, accessible, token_value.
- test/test_reference_reconstruction.py, especially the sole new test at lines 91–153.
- doc/benchmarks/2026-10-21-item3-5-r10/source-diff.patch and source-audit.md.
- Its frozen-source.json, gate-summary.json, and scoped-state/test_named_token_keeps_its_content_before_and_after_the_close/outcome.json.
- Its frame_probe.py, frame-probe.json and frame-probe.log.
- bin/Models.py around 20340–20380, bin/ReferenceContext.py SituationAddress.read/SituationFrame.from_row, bin/ClauseRow.py write_clause's written activations, and the existing test_reference_close.close helper.

Results on the frozen hash: 110 passed, 4 pre-existing FutureWork prerequisite skips, 1 failed (the new test only). Within-sentence checks all pass: alias selected greedily, features [1,1,0,0,0,0,0,0], candidate logit 1 against dummy 0, registered reference cost 0, pool isolation and anonymous/pending state, commit count 2. Failure at line 145: the next frame has no candidates and the mention opens its own object.

Codex diagnosis and proposal, for independent scrutiny: my new fixture calls SituationFrame.from_row on the sentence root, omitting symbols and activations from the written close. Production stores both in SituationAddress and restores them when reading a frame. The read-only diagnostic restores the failed gate's initial RNG and executes the unchanged test to the same failure, then reads the same close through SituationAddress using its actual written metadata. It finds two activations, selects the original token and gives zero cost, with the same coverage features. It does not edit the gate or production. The failed gate is not promoted to a pass.

Proposed correction, not applied: construct the test's next frame through SituationAddress(store.occurrence_of(row), symbols=written.get(('symbols', id(kept)), ()), activations=written.get(('activations', id(kept)), ())).read(store), preserving every assertion and all production code. Then freeze the fixture-only change and rerun the same five-file scoped set once on that source. This needs disposition under Alec's explicit forced-order stop rule; no publication or step-2 coding has begun.

Please review the production change and regression, independently diagnose the failure, assess the concrete fixture correction, and report any additional issue. Give a clear publication verdict for the current frozen hash and a recommendation to Alec. Do not infer approval to edit from a diagnostic pass. No learning claim.
