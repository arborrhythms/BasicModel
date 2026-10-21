# Review of frozen source `575cab06c`, step 1b after review nine (§19)

I read §19 in full, along with §10.3 and §10.4, the source diff, the changed production files, the six failing tests, the forcing fixture, and these receipts: the gate summary, the source audit, C-pair, bind-departures, harness-recovery and both `bind_gate.py` copies. I edited and ran nothing. Every claim below comes from reading the code or the recorded receipts, not from tests passing.

**Verdict: do not land yet.** One mechanism defect within decision B needs a repair. The six red arrival tests are not production defects, but every one needs Alec to authorize a port. E's certificate needs a decision from Alec before it means what its receipt says.

## Accepted on inspection

- **A, keep and credit.** Closing plan costs are now diagnostic only (`Models.py:19896-19899`). `SentenceCredit.components` is R, E and the supplied answer only (`SentenceCredit.py:90-95`).
- **A, single writer.** The bind surrogate is p(a)·scale·ΔC(R + A), the same `score_function` the compose chooser uses (`Models.py:20877-20897`). It reaches backward only through `_sentence_reader_costs['expectation']`, and `_backward_training_loss` applies that to the expectation owner, which owns `components` and so the resolver policy (`Models.py:2550-2553`, `2653-2672`). The `loss = loss + credit` line and the late `registry.add` don't reach backward, because `total_loss` only seeds the AMP scaler. There is no double count.
- **A, no pathwise leak.** The features pass through `carrier()`, which detaches (`IdentityRows.py:32`). The only thing that gets a gradient is `policy`.
- **One-departure accounting.** Walk 3 excludes the other walks. The bind round is drawn from the greedy trial's decisions, and q = 1 for the next-ranked option, so scale = walk_count·walk_rounds is right.
- **Kept selections.** Admission reuses the chosen trial's scored plan (`Models.py:20288-20295`). The admit-time re-preview is discarded.
- **D, as written.** `select`/`choose` implement the zero-logit dummy and strict beat, with ties to the dummy (`ReferenceClose.py:406-433`). Own objects are unpriced, indefinites mint, and minted individuals make no entry (`:992-1006`). Gaps stay open. A dummy at an arrival leaves the token unchanged.
- **B, single refinement.** The merge direction, key, kind, earlier uses and summed count are right (`MediumTermMemory.py:75-89`). This close's uses bind after the merge, through `token_of`. The ported reproduction matches its retained original (`anonymous-bind-review-followup/diagnostics/test_remaining_findings.py`) apart from the `token_of` line.
- **Isolation.** Preview and prepare don't write the pool, and the compiled-boundary cases pass.
- **C.** The WORD-row guard (`ClauseJournal.py:27-31`, `Language.py:12594-12598`) leaves the OBJECT-ID lane unchanged. The bootstrap replay and both MM draws use byte-equal RNG states and pass with zero R and zero seats.
- **F.** No new detaches on MTM reads.
- **Evidence.** All 20 manifests match 827 files at the frozen hash. The forward-6 recovery is explicit and that draw was not replaced. The strict-certificate copy and the B port match their originals.

## Findings, in priority order

### 1. Mechanism defect within B: a later re-run on a joined site merges the whole antecedent token. Must fix.

**Where.** `_arrival_plan` re-runs bind for every historical site whose name appears anywhere in the close. Any word ID counts (`ReferenceClose.py:774-775`, `812-833`). It keys each site by `token = mtm.token_of(occurrence)`, which after an earlier merge is B's key, not the site's own. `commit` then calls `merge(token, selected)` or `bind(token, selected)` on all of B (`:1045-1048`).

**Failing scenario.**
1. `y is x plus two.` opens T for x.
2. `the box is red.` opens B under D.
3. `x is the box.` merges T into B. This is correct.
4. `the ball is blue.` opens D.
5. `x is the ball.` re-runs bind for the x site at T. Its token is now B. If the ball's occurrence is selected (a positive prime is enough, or any learned weight), `merge(B, D)` runs: the box's key is retired and its sentence-2 use now reads as the ball.

If the selected occurrence is unpooled instead, `bind(B, …)` appends it to the box's token and contaminates every use of the box. Re-selecting a member of B also adds 1 to `count` on every later mention of x. That matters for step 2's promotion threshold.

This contradicts B's text: "a later arrival that re-binds the site moves its occurrence to the new antecedent's token".

**Repair.**
- Carry the site occurrence in each arrival item.
- At commit, merge or bind the token only when `mtm.token_of(site) == site`, meaning the site is still its own pending object.
- Otherwise, move only that occurrence (`mtm.bind(new_token, site, …)`). Selecting the current owner should be a no-op.
- Where to move a joined site when the new candidate is unpooled is unspecified by B. That is a question for Alec; skipping the move is the safe default.

**Regression.** Run the scenario above and assert that B's key, members and count are unchanged after step 5.

### 2. Question for Alec, E: `open_names` now forces x open at the resolver seam

`ForcedGrammar.reference_choice` ranks "open" at 10, and a candidate can reach at most 3 without `named_bindings` (`test/forced_math_grammar.py:44-65`). Before, `named_bindings`' +30 overrode this. Now every greedy x site and every greedy x arrival is forced to the dummy.

Only an explore departure can bind x, because `choose`'s alternate bypasses `select`. On a premise, a departure can never be kept:
- A is 0 because there is no target.
- R doesn't read the bind, because the plan is applied only at admission.

**Evidence that this is what happened.** All 97 recorded departures are `tie:greedy` with all-zero logits, every draw reports 0 bound, and no bind credit ever produced an update.

**Consequences.**
- The correct-closure branch of E's `check()` is unreachable. The receipt's `closure_forced: false` is literally true, but it should also say that x's reference choice is forced open.
- The item-0 strict certificate keeps `open_names`, so it can never pass under any training. "Unforced" and "`open_names` kept" contradict each other.

**Recommendation.** Restrict `open_names` to the formation seam (`:160`) and leave the resolver live. At initialization D keeps x open anyway, so E's results should be unchanged. This is a fixture change and needs Alec's authorization.

There is a related question for Alec: with no delayed credit, arrival and premise binds receive no R + A signal at all. The only learning for name closure is generalization from question sentences and the supervised mention loss.

### 3. The six red cases: not production defects, ports need Alec's authorization

**Three count failures** (`test_reference_arrivals.py:92`, and `:227` for both the eager and compiled cases).
- The README attributes these to D. They are actually caused by B.
- The arriving x opened its own token T2 before this round too: the old `anonymous` path ran whenever the forced `select` returned `None` for T, whose kind is `None`.
- The old code then ran `bind(T2, occ3)`, which took occ3 out of T. That is exactly the defect B repaired, and it is what left T with count 2.
- Now T2 merges into T, giving `(T, occ3, T2)` with count 1 + 1 + 1 = 3. That is the correct state under B.
- **Port:** `count == 3`, or `count == len(occurrences)`.
- The first test stops at line 92, so lines 93-109 haven't run on this source. They must run after the port. I expect them to pass, but I haven't verified that.

**Three zero-score failures** (`:173`, `:247`, `:262`).
- Each test's fixture relies on argmax binding at logit 0.
- An empty request gets zero request coverage (`IdentityRows.py:70-73`). With policy `[0,1,1,1,0,…]` (`ReferenceClose.py:38`), the logit is prediction + prime, which is 0 here, so D correctly picks the dummy.
- What each test is named for (a bound use doesn't assign a kind; no pool entry for a native individual; an unpooled antecedent keeps its kind) is still valid.
- **Port:** drive the choice through the existing `select` seam (`force_named`), or give the fixture a positive feature. Keep every assertion.

### 4. Mechanism defect, medium: `answer_meaning` matches roles by index across nesting levels

**Where.** `answer_meaning` checks a site's `item['role']` against the root meaning's open and bound referent roles (`ReferenceClose.py:556-570`). But a request's role is relative to its immediate parent clause (`ClauseJournal.py:556-559`, `ReferenceClose.py:621-643`).

**Scenario.** In `what is x plus two ?`, x has role 0 inside `x plus two`. Its selected token is filled into the root's role-0 `what` slot. `BindingAnswers.cost` (`:86-99`) then scores x's identity against the answer. That gives partial credit (between 0.5 and 1) instead of 1, and an exact 0 for something like `what is x plus zero ?`. So x's bind is credited as if x answered the question.

Current gates only ask `what is y ?`, where y is a direct root role, so they don't exercise this.

**Repair.** Have `sites()` record where each site sits in the tree. Fill only sites that are a direct root role (`parent is clause`, or `clause.refs[role] == ('clause', i)` for that node).

`answer_loss` (`:533-547`) has the same pre-existing correspondence. It "remains" under A, so whether to align it is Alec's call.

### 5. Mechanism, low: `merge` drops the source token's kind when the antecedent has none

`merge` keeps `antecedent.kind` even when it is `None` (`MediumTermMemory.py:83-85`).

**Scenario.** T has kind three and is merged into a pooled anonymous token W (for example an own object from the same close). The kind is lost, `_derivation_symbol` returns `None`, and `matches(…, 'three')` fails.

**Repair.** Adopt `previous.kind` when `antecedent.kind is None`, following §10.4's `bind` rule. B's own kind is never overwritten.

### 6. Evidence and attribution notes

- **Carrier certificate.** `test_relation_closing.py::test_mixing_relation_keeps_admitted_object_ids…` passed only in dev run 07. Its six changed production files hash-match the frozen source, but there is no receipt on `575cab06c` itself. Either run that single test on the frozen hash (not a sweep), or Alec accepts the hash identity.
- **C's scope.** The guard means no leaf in the mixing reader can become an objectless-word site, because the WORD lane is absent there. That restores the pre-carrier behaviour and should be declared.
- **SentenceCredit caller test.** The test changed its input from `reference=` to `answer=` (`test_reference_close.py:215-217`). Its assertions are unchanged, but it now tests a different thing than its name says. That is beyond E and B, so Alec should acknowledge it.
- **`bind_gate.py` copy.** Its header says "only root path changes", but the bodies are byte-identical. The hash differs only because of the header, and the location comes from `__file__`. The header wording should be fixed.
- **Thought-face training.** `train_thought` still trains `choice_cost` locally (`ReferenceClose.py:498-513`). That is outside A's closing scope as written. Whether A's principle extends to thought-episode binds is a question for Alec.

## For Alec: exact questions

1. Authorize the six ports from finding 3: counts to 3, and the zero-score setups driven through `force_named` with assertions kept.
2. Restrict `open_names` to the formation seam in both the E certificate and the item-0 certificate (finding 2).
3. Decide whether arrival and premise binds need a delayed credit path, since R + A is structurally zero for them.
4. Decide finding 1's unpooled re-bind case, and whether to align `answer_loss` with finding 4.
5. Acknowledge the SentenceCredit caller-test change and the C scope.

## Landing verdict

**Not ready to land.** Repair findings 1, 4 and 5; they are small, local changes in `ReferenceClose.commit`/`answer_meaning` and `MediumTermMemory.merge`, each with a regression. Get Alec's rulings on questions 1-3. Then freeze a new hash and re-run the targeted gates:
- the twelve fresh draws, or a subset Alec names
- the regression file set
- the carrier certificate
- the bootstrap and MM pair

That is not a second full sweep. Publication and the naming sweep stay pending until then.
