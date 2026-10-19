# Review of BasicModel candidate `f7e74af04` (`codex/item4-repair`) against `9207f5e03`

**Verdict: the source is not approved for the landing sweep, and Item 4 is not complete.**

The repairs mostly hold when read against the source. Three findings block the sweep:

- **Repair G is wrong.** It reads the closing confidence (κ) from the wrong place.
- **Repair E changed Item 5's retention behaviour.** Omission no longer counts towards retention surprise.
- **The prime is on at the "zero" defaults.** That change was not decided and was not measured.

Two high-severity defects in how the S level identifies and reads symbols should be fixed in the same round. None of the unmet gates the author disclosed are waived.

How I reviewed: by reading only, with `git diff`/`git show`, Read and Grep. Nothing was run. Finding H1 depends on behaviour I worked out from the code, so it says what to check.

---

## Part 1 — New actionable code defects

### Blocking

**B1. κ is taken from the envelope's role position, not from the symbol (repair G is incorrect)**
- **Where:**
  - `bin/Layers.py:10630-10632`: the estimate's presence is `[σ(top symbol's logit), σ(−30), σ(−30)]`.
  - `bin/Models.py:20253` and `:20265`: that presence is passed into the document and row closings.
  - `bin/Meaning.py:68-74`: `ClosingImage.form` applies `presence[r]` to whichever key matched *observed* role `r`.
  - `bin/Layers.py:9671-9672` (`append_expectation_pair`): same path.
- **Impact:** At any nonzero row or document gain:
  - Observed roles 1 and 2 get κ≈9e-14, so they are never attenuated, however confident the predictor is.
  - Role 0 is attenuated using the confidence of an unrelated top-ranked symbol.
  - §9.3 passes only because the cat sits in role 0.
  - `test_item4_repair.py::test_closing_confidence_multiplies_gain_and_open_role_keeps_prime` supplies an even `torch.full((3,),c)` presence, which hides the defect.
  - Shipped g=0 makes it inert, but this is the mechanism being landed.
- **Repair:** Make κ per symbol: either from the bank prediction at the matched key, or explicitly 1 with ê carrying certainty (the spec's default). Never index it by observed role position. Add a native regression with a three-role relation row at g=1.

**B2. Retention surprise drops omission — a silent change to the carried Item 5 formula**
- **Where:**
  - `bin/Expectation.py:82-94`: the support is the *observed* symbols only.
  - Consumers: `bin/Layers.py:10477-10479` → `bin/ClauseRow.py:417` (native write) and `bin/Layers.py:9675`.
- **Impact:**
  - A symbol that was expected but never arrived has target 0, so it falls outside the support and its residual is ignored.
  - A row whose only surprise is an omission is scored about 0 and is forgotten first. Spec §4.1 says the residual "sets retention priority".
  - The baseline `expectation_surprise` weighted empty roles by expected presence (a soft union). The candidate removed that, which breaks the plan's carried boundary: "must not silently change Item 5's retention formula".
  - `test_retention_surprise_is_not_diluted_by_bank_padding` pads with ê=0.5 symbols that never arrived (real omissions) and asserts they don't count, so it locks the regression in.
- **Repair:** Weight each symbol by `max(o, ê)` (the soft union), i.e. `Σ w·r² / Σ w`. Symbols with ê≈0 and o=0 still drop out, so there is no dilution, and confident omissions count. Fix the test so confident padding raises surprise and ê≈0 padding leaves it unchanged.

**B3. The prime and φ change default behaviour at "zero" gains (decision needed before the sweep)**
- **Where:**
  - `bin/Models.py:20632-20644`: the compose prime is built from any pending D forecast. Nothing gates it on a gain or a weight.
  - `bin/Language.py:6945-6948`: it is added unscaled to the logits that pick the greedy derivation.
  - `bin/ModelCandidateAttention.py:181-199`: prime and P-z columns go into placement.
  - `bin/Language.py:6502` and `data/BasicModel.xml`: φ=0.5 replaces the landed uniform proposal (φ=1).
- **Impact:**
  - From the second sentence of every document, an untrained head (sigmoid≈0.5 everywhere) primes towards the bank centroid. That adds a bias in [−1, 1] to greedy compose choices under the shipped "defaults are zero" configuration.
  - The parity probe can't see this: it disables expectation and presents single sentences in eval (`zero_baseline_probe.py:17,31`).
  - So the §9.1 "exact equality" covers a configuration the sweep won't run.
  - φ=0.5 is Codex's value, not a decided one.
- **Repair:** Alec decides. Either:
  - add a model.xml prime weight (default 0) and set φ back to 1 until the §9.1 gate runs with expectation enabled, or
  - accept both, and re-run parity with expectation on, multi-sentence documents and training departures.

  The receipt's "Defaults are zero" (README, todo:37) should say that the prime and φ are not.

### High

**H1. S identifies symbols by value distance and ignores the native handle (to check)**
- **Where:**
  - `bin/SymbolField.py:38-41`: `identities` is unpacked but never used.
  - `bin/SymbolField.py:25-31`, `bin/Expectation.py:253,267-269,284`: identification is `cdist ≤ 1e-5` on STM values.
  - `bin/Meaning.py:55`: same in the closing.
- **Impact:**
  - The ≤1e-5 test itself is implemented correctly; the problem is what it compares.
  - STM rows are activated codes (`bin/Interpret.py:22-25`: code × evidence lanes). At g_P>0 they are also P-gained, via `bin/Models.py:19433-19434` → `leaf_evidence` → `Interpret`.
  - Any arrival whose evidence doesn't reproduce the codebook key exactly becomes "unknown": no S observation, no gain, and its loss target is 0.
  - The README's cold surface-word probe ("an unknown composed value") fits this. The §9.3 cat test uses a registered identity.
- **Repair:** Identify through the STM's native handles (`state[4]` ↔ `_bank_ids`/`_bank_addresses`) and read activation from `state[5]`/lanes. Keep value matching only for composed values that have no handle. Add a native probe in which an ordinary word with evidence (0.6, 0.2) is observed by S.

**H2. The compose gain assumes every arrival is for-lane**
- **Where:** `bin/Expectation.py:286-291`; called with no lanes at `bin/Language.py:6854,6864-6865,12464`.
- **Impact:** A symbol that arrived with against-evidence is scored as `(1,0)` against ê. That is exactly the `(1,0)`/`(0,1)` conflation plan §A forbids. The STM content view is then mis-attenuated.
- **Repair:** Pass the real lanes (SymbolField already keeps `source_lanes`) into `SymbolRead.gains`.

**H3. Every estimate row stores the whole bank as Python tuples**
- **Where:**
  - `bin/Layers.py:10451-10459`: `_expected_symbols` holds N×(D+2) floats.
  - `bin/ClauseRow.py:423` and `:496-501`: `_observed_symbol_activations` holds N pairs.
  - The semantic fingerprint is computed over these bindings.
- **Impact:** Each sentence costs O(N·D) memory, checkpoint size and hashing. At bank and LTM capacities of 1024, this goes from megabytes to gigabytes. Plan §C asked for this bound to be measured; it wasn't.
- **Repair:** Store ids and logits, and keys only where activation is nontrivial (or a shared bank snapshot per parameter version). Measure before Item 0.

### Medium

**M1. Retired-argument swallowing and single-value knobs remain** (against Alec's rule: no retired argument swallowing)
- Parameters accepted and never used:
  - `ClosingImage.form` `form_width`/`content_width`: `bin/Meaning.py:34` (callers `bin/Models.py:20248-20260`, which is dead computation).
  - `sentence_prediction_cost(gain=)`: `bin/Layers.py:10768`, passed at `bin/Models.py:19927`.
  - `word_distribution(teacher_forcing=)`: `bin/WordExpectation.py:26`.
  - `_prediction_inputs(replay=)`: `bin/Layers.py:10607`.
- A knob that exists only to reject other values:
  - `sentenceExpectationScope` enum with the single value `symbols`: `data/model.xsd:959-965`.
  - The matching constructor kwarg: `bin/Layers.py:10512-10515`.
- Dead code gated on the retired `'structured'` scope: `bin/Queries.py:1158`.
- **Repair:** Delete them and update the callers.

**M2. The migration fabricates handles**
- **Where:** `bin/Layers.py:9168` writes ids `0..k` and order `0`.
- **Impact:** These collide with real native ids and are exposed through `expectation_pair` → `identities`. They also contradict the unknown-order convention at `bin/Layers.py:9330` (−1). Otherwise, the migration does run once at load, after authentication, with fingerprints refreshed (`bin/Layers.py:9268-9288`), as required.
- **Repair:** Use −1 for both id and order.

**M3. The expectation-free placement view still carries prediction-derived inputs**
- **Where:**
  - `bin/ModelCandidateAttention.py:196-199`: P's z stays in `expectation_free=True`.
  - `bin/ModelCandidateAttention.py:90-99`: `_needs` still feeds the three-role forecast envelope as a structured "need", in parallel with the new prime.
- **Impact:** Plan §D says to omit "other prediction-derived chooser features". The envelope is also the source of B1.
- **Repair:** Either neutralise z (cold = 1) in the free view, or get a recorded decision that z counts as salience. Retire the envelope slot from `_needs`.

**M4. Two different retention-surprise formulas for the same row**
- **Where:** `bin/Layers.py:10479` (native write; ignores the observation's `non`-withheld marks) versus `bin/Layers.py:9673-9675` (honours them).
- **Repair:** Use one owner; fix together with B2.

**M5. A stale schema test is hidden by a skip**
- **Where:** `test/test_sentence_expectation.py:237` asserts that `structured`/`root` validate, which the new schema forbids.
- **Impact:** In `regression-2` it was skipped because lxml is missing. As a result, none of the new schema elements were validated in this round, and the test will fail wherever lxml is installed.

### Low
- **L1. Rare illegal draw.** The CDF draw can pick an ineligible trailing action when float rounding leaves `cumsum[-1] < 1` (`bin/Language.py:6571`, `bin/AttentionTraversal.py:54`). This would trip the `has_choice` assert.
- **L2. Undeclared threshold.** `ThoughtClosing.absence` hard-codes a 0.5 threshold that isn't a model.xml parameter (`bin/ThoughtClosing.py:81`).
- **L3. Dead state:**
  - `versions` is recorded but never checked (`bin/Layers.py:10680`).
  - `_inter_last_pred_root` is always None.
  - `model._word_surprise` has no production consumer (`bin/ModelAttention.py:309`).
  - The `ThoughtCredit.py:32` scope check is always true.
- **L4. Stale config and docs:**
  - `data/model.xml:304-305` (orphan comments, including "legacy gain … migration fallback").
  - `data/model.xml:315-319` (says structured scope and live sources).
  - `data/model.xsd:954` (`rowExpectationArmaScale`, unread).
  - `doc/GradientFlow.md:447` (`interLossWeight`).
- **L5. Double training signal.** Each order-0 arrival trains the shared head twice from different contexts: the word loss and the `expectation.symbols` loss. Confirm this is intended.

---

## Part 2 — Repairs A–K confirmed by reading

- **A, ownership.**
  - `SymbolExpectation` is the only owner. The aliases, ARMA/contrastive heads, `_discourse_arma_loss`, the `Spaces` import, the `negative_image`/`closing_gain` code and the WalkTrials forecast pair are all gone.
  - There is a head migration (`Layers.py:10792-10812`) and an estimate-row migration (`:9135-9171`).
  - Residues: M1, M2.
- **B, credit.**
  - Each owner's scale is the actual mixture 1/q × walk_count × walk_rounds (`Models.py:19463-19477`, `AttentionTraversal.py:57-64`, `Language.py:6737`).
  - Ties have no policy graph, and a positive finite correction is asserted (`SentenceCredit.py:123-124`).
  - Keep, credit and reader weights all use R+A (`SentenceCredit.py:93-133`). The trial returns its R+A cost (`Models.py:20820`).
  - The seed-23 cause (tied compose exposures) is consistent with this code.
- **C/D, S gain.**
  - The exact-match test is ≤1e-5.
  - Raw STM and leaf coefficients aren't mutated; a separate gained activation feeds content candidates only (`Language.py:6849-6877`).
  - Order 0 is gained once per consumer: S in the compose view; `_word_column` for the answer reader.
  - Caveats: H1, H2.
- **E.** Padding no longer dilutes the score, but see B2.
- **F.** The prime and S read are cleared in the per-sentence `finally` (`Models.py:20887-20889`) and in the run context's `finally` (`:13632-13635`). Both run before `_commit_sentence` and thought (`:21001`).
- **G.** κ is now explicit, but wrong (B1). The open role is spared attenuation and keeps its prime, as Alec decided (`Meaning.py:59-66`).
- **H.** Both attention paths use one `PerceptMoments`. Observation happens only in training (`ModelAttention.py:354-356`). The read happens before the update, with a prior snapshot.
- **I.** The non-negativity assertion covers the evidence-lane content; signed identity keys are untouched (`Expectation.py:76-78`).
- **J.** The four-result `select_logits` call passes through in `test/forced_math_grammar.py:44-48,136`. All 3-tuple callers in production code don't depend on the fourth result.
- **K.**
  - The inverse menu dispatches one operator per entry (`Generative.py:195-207`).
  - Choose and apply are compiled separately from the eager driver (`Models.py:21880-21886`).
  - CPU leaves the default device unset (`util.py:265-275`, `Optimizer.py:705-712`).
  - Compiled runs at nonzero gains were not exercised.
- **`non`.** Both lanes are cleared (`Language.py:2269`, `ClauseJournal.py:564`). The withheld mark is written (`ClauseJournal.py:497-508`). The inverse returns the retained witness (`DerivationReconstruction.py:23-30`).

---

## Part 3 — Disclosed unmet acceptance and learning gates (none waived)

1. **Forced ordinary-answer certificate:** skipped because the `copular_identity` checkpoint is missing. That is not a completed certificate.
2. **Initial-binding distribution:** fails (0.8780–1.1125 against a required 0.9–1.1). This is the carried Item 0 class; it stays open.
3. **§9.1:** g=0 parity holds only for consumer views with expectation disabled. With expectation on, it is not established (and B3 says it can't hold as built). Standing XOR and the math chain with every level on are not established.
4. **§9.2:** per-level falling residual and a flat collapse diagnostic are not established. Three arms have unexposed owners. R and D departures were not exercised.
5. **§9.3:** this is a mechanism test with a controlled head, not "a sentence repeated until predicted". The cold probe is correctly not counted.
6. **§9.6:** gate false. Seed 23 shuffled has the lower R, and no optimality oracle exists.
7. **Unchanged:** the original failed native run stays failed and is not regraded. Item 0 findings and the Item 5.5 learning track stay open. The clean-tree completion (commit, push, WikiOracle bump) is pending. No full sweep has run.

---

## Disposition

**Not approved to sweep.**

1. Fix B1 and B2, and get Alec's decision on B3 (prime weight and φ default).
2. Fix H1 and H2. Fix H3, or measure it and get it accepted.
3. Clear M1–M5. That includes making sure lxml is present for the sweep.
4. Then commit a new candidate for a short re-review, followed by the single full sweep.

Item 4 stays incomplete until the §9 gates in Part 3 are met on their own terms.
