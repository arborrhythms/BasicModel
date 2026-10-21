# Claude review of 3d453e6bb

# Review: frozen source `3d453e6bb`, item 3.5 §10.4

This was a read-only review. I ran nothing and edited nothing. My evidence is the diff, the affected source on this tree, and the retained receipts.

## Verdict

**Do not land.** There are four implementation defects that can be repaired inside §10.3/§10.4 (findings 2, 3, 5, 6). The reverse D2 failures also need a decision from Alec. The receipts misdiagnose them: the deciding term is A's closing reference credit, not the frame menu. A new hash then needs every gate re-run, following the user's rule. The bootstrap failure needs Alec's disposition separately.

## Prioritized findings

### 1. The reverse D2 failures are decided by A's reference credit, not by the missing candidate. Alec has to settle this.

The tables contradict the "missing candidate" diagnosis in `reverse-2-failure.md` and `reverse-3-failure.md`. In both failing streams, R ties exactly (.0360673778) on `x is three`. The **greedy** reading, which has the pending-x site (`forward=[[0, 12]]`), pays A = .408. The **explore** reading, which mints x or makes it a field, pays A = 0. The keep is "explore, by A" (`fresh-final-reverse-3-tables.md:15-16, 78-79`).

A's third component is `answer + reference` (`SentenceCredit.py:85-90`). The reference part is `plan['cost'] * components.weight` (`Models.py:19895-19902`). These premises have no answer target, so A here is essentially the closing plan cost. That cost works like this:

- **A named first mention always pays.** A site with no candidates pays `+1`, and so does a site whose selector returns None (`ReferenceClose.py:673-675`). This is the old price for a *failed* bind, even though §10.4 makes the anonymous token a successful outcome ("`missing_reason` is then not set").
- **The same-close arrival rerun also pays.** It adds a strictly positive `choice_cost` (`ReferenceClose.py:803`).
- **A nameless mint or field reading pays 0.** A mint with no choices adds nothing (`:661-662`). It has no forward name, so there is no arrival either. The new guard at `ClauseJournal.py:370` (`request['mode'] != 'mint'`) strips the forward name from every mint request; the draw-3 reading is exactly this case.

So whenever explore proposes a nameless reading of the first mention, A keeps it deterministically. This is a structural bias, not R noise. The same bias is visible in the forward arm: `fresh-final-forward-1` documents 1 and 6 keep explore at A = 0 over greedy at A = .2, with R tied. D1's `>= 2` threshold hides it there.

**Is there a repair inside the contract? No.**
- **Putting the order-0 `three` complement in the next frame's menu** would change §10.3's candidate boundary. Candidates are individuals plus order-1 occurrences, and "candidates still come from the frames in this round". It would also make a field seat an antecedent. And it leaves the bias in place.
- **Repricing the anonymous outcome** to match a first-mention mint is still not enough. The arrival rerun's `choice_cost` keeps the named reading dearer, and §10.4 requires the arrival to be "credited by the same `choice_cost`".
- **Draw 2's reading has no reference request at all** (an order-0 field mint). No bind-side rule can reach it.

The real cause is that the *operation* chooser (mint, field or named site) removes the site before `bind` runs. §10.4's claim that the certificate is "deterministic at any initialization" assumed the site exists.

**Required now:** have the receipts split A into its answer and reference terms, per site and per arrival, for both trials in streams 2 and 7, so the arithmetic is on record. Then put the question below to Alec.

### 2. An arrival refinement deletes another MTM token (implementation, high)

At `ReferenceClose.py:968-979`, when the arrival selects an accessible pooled candidate, `selected` is that candidate's token key. That key is the *introduction* of a different token B. `mtm.bind(T, selected)` then calls `forget(selected)` (`MediumTermMemory.py:50-51`). Because `token == occurrence` there, B's whole entry is deleted and every earlier use's pointer is dropped (`:67-70`).

The comment at `:984-986` protects only *this* close's uses. B's earlier uses become unlinked occurrences, and `individuals()` then splits them. This writes a merge (in fact a destructive one) into MTM, which breaks §10.3: "revision merges … are not written to MTM". It also undoes the intent of §17 Finding A.

- **Failure condition:** token B has an introduction b and a use u from an earlier close. A later arrival refines T and selects B. Afterwards `token_of(u) is None`.
- **Fix:** if the candidate already belongs to another token B ≠ T, never detach it. Refine only `T.kind` from the candidate's kind and leave B intact. Moving a candidate into T is only allowed when it is unpooled.
- **Regression:** the scenario above. Assert that B's entry and `token_of(u) == B` are unchanged, and that `T.kind` is set.

### 3. Arrival sites are not access-bounded (implementation, high)

`_arrival_plan` collects historical pending sites from the posting index. The only filters are name and stream bit (`ReferenceClose.py:746-767`). There is no frame-accessibility check and no document check. By contrast, `pending_names` does require `token_of(occurrence) in individuals` (`:626`).

In multi-document or epoch training, a pending `x` from an old document in the same stream gets refined, which is a **write** to MTM, by a later document's `x is 7`. The old reader had a similar reach, but it only *read*.

- **Fix:** keep a historical site only when its token is among this close's accessible individuals. That is the same test `pending_names` already applies.
- **Regression:** two documents in one stream, with the first document's frame inaccessible. Its token must be unchanged.

### 4. Revision, re-runs and multiple sites (implementation plus a contract clarification, medium)

- **Kind is first-wins.** `kind=kind if entry.kind is None else entry.kind` (`MediumTermMemory.py:57`). A "later arrival may re-bind (revision)" (§10.4), but it can only append content; it can never change the token's kind. With the production argmax `select` (`:406-408`), the first arrival's untrained pick fixes the kind permanently.
- **Every historical site is re-run at every later close that mentions the name.** Raw rows keep `_forward_references` forever, and nothing checks whether the token is already refined. As a result, A's reference credit and `count` grow with document history, and refreshing `when` shields the token from eviction.
- **Several sites on one token are each re-bound in the same close**, and each is charged. `test_bound_use_does_not_assign_a_kind_before_arrival` asserts `== 2`. If the two selections differ, they silently conflict: the first sets the kind and both contents are max-pooled.
- **Fix:** bind at most once per token per close. Alec needs to state whether revision overwrites the kind (see the question below).

### 5. Native arrival writes a working vector into LTM (implementation, medium-high)

When the arrival selects a native individual that has no located occurrence, `prepare` builds a synthetic order-1 row whose point is `menu[selected]['value']`, the max-pooled *working* vector. It is then admitted as a companion (`ReferenceClose.py:896-901`, `:808`).

That conflicts with §10.3 in two ways. A bind to a minted individual should write nothing, and working vectors are computed on read and never stored. The row also feeds `individuals()`'s exclusion scan (`:307-311`) and later frames.

- **Fix:** for a native candidate, set `T.kind = native` without materializing a row. This needs a narrow MTM kind-refinement method, or binding through T's own introduction.
- The assertion `occurrences[-1] == refinement['reference']` in `test_native_arrival_refines_an_existing_anonymous_token` locks in the current behaviour. Changing it is a port that needs authorization.

### 6. `bind_site` builds an invalid witness (implementation, confirmed by `witness-repro-04`)

`ReferenceClose.py:444-445` wraps every non-None identity as `('sym', identity)`. Since §10.3, `accessible()` sets `identity` to an MTM token or an occurrence (`:194, :243`), so the wrapper produces `('sym', ('ltm', …))`.

- **Fix:** mirror `occurrences()` (`:249`): wrap only integer identities and pass `ltm` addresses through unchanged.
- **Regression:** promote `diagnostics/test_witness_repro.py` into the suite.

### 7. Arrival credit only partly follows the ordinary path (medium)

**Same as ordinary sites:** scoring (`score`), selection (`select`), `choice_cost`, and the +1 for no selection.

**Different from ordinary sites:**
- `commit` records `_mentions` only for `plan['items']` (`:950-954`), so `resolved()` and `mention_loss` never train arrival menus.
- `answer_loss` iterates only over `items` (`:506`).
- The arrival audit records `missing_reason=None` whenever the token survives, even when nothing was selected (`:981-983`).

Either record arrival menus the same way, or declare the asymmetry. In both cases, fix the audit to key on `selected`.

### 8. Two changes that need declaring or approval (process, medium)

- **`ClauseJournal.py:370`:** mint requests now drop the forward name. §10.4 does not mention this, and it is material to the draw-3 reading (finding 1).
- **The decomposition fixture now patches `resolver.select`** (`forced_decomposition_grammar.py`). It applies the pre-declared `open_names` rule, but to a *new* closing seam. The user's instruction fixes the forcing scope, so this needs explicit declaration in the receipt and Alec's acknowledgement, not just a mention in the README.

### 9. Lower-priority issues

- `prepare` maps arrival constituents to admitted ones by index, guarded only by a count check (`:884-893`). An admission that reorders terms would materialize the wrong antecedent. Match by leaves instead.
- `ClauseRow.py:656`: `max()` over witnesses fails when no use has an integer location.
- `ClauseRow.py:615`: the `exclude` parameter is now unused.
- `ReferenceClose.py:101`: an integer that is also a row id now resolves to that row's derivation. Confirm that symbol ids and row ids come from disjoint allocators.
- `Models.py:20290`: `dict(scored, …)` shares the `arrivals` list, so `commit` mutates the scored trial record. This is harmless now, but take a copy.

### Areas you asked about that check out

- **Preview/prepare isolation:** holds. Trials only read the pool, and `menu` and `individuals` are local.
- **Older raw rows and the bank:** never rewritten. The kept close does admit *new* companion rows (arrival copies and lexical sites); only the native synthetic row in finding 5 is a violation.
- **Native versus occurrence kinds:** `_derivation_symbol`'s fallback to the token's `kind` makes PartClosing, PartRead and BindingAnswers read consistently. The witness in finding 6 is the one place where the two kinds get confused.

## Bootstrap zero-R failure

**Implementation finding (not 3.5 code):** the magnitudes in `bootstrap-bank-04` are exact unit norms over different supports. .09622504562 = 1/√108 and .08606629819 = 1/√135. So the decoder returns the first word's meaning spread over 135 active coordinates, 27 more than the bank row's 108. That accounts for the nonzero relative content term. It is a lead worth chasing in the decoder's normalization, not a verified cause.

Component reading is off and MTM is empty. However, `ClauseJournal.finish_clause` changed in this diff and runs regardless of component reading. The batch-equality replay was done against `823f157cd`, not `3d453e6bb`. The R value matches to 17 digits, but the receipt should extend `bootstrap_probe`'s comparison to this hash.

**Disposition (Alec's call):** this is a pre-existing failure in an XOR test unrelated to 3.5, and it depends on the initialization. Under the user's rule it is not waived, so the eight-file gate cannot be reported green.

## Exact questions for Alec

1. **Reverse arm.** At a first mention, the closing reference credit charges §10.4's bind path. Opening the anonymous token costs the old missing-bind price, and the same-close arrival rerun adds a positive `choice_cost`. A reading that mints x or makes it a field seat pays nothing. With R tied at initialization, A keeps the nameless reading whenever explore proposes one: 2 of 6 reverse draws here, plus streams in the forward arm. After that, no `bind` ever relates x to `three`. Which do you want?
   - **(a)** Extend the certificate's forcing to the operation choice at the named word, so x stays a named site. The live keep, assertions, budget and objective stay unchanged. **I recommend this:** it is the narrowest option, and it is the only one that also covers draw 2's field reading.
   - **(b)** Change how the closing credit prices the anonymous outcome and its refinement relative to a first-mention mint. This changes the objective, and on its own it does not reach draw 2.
   - **(c)** Treat reverse D2 under live operation choice as an item-0 learning certificate.

   Whichever you choose, should an explicit mint of a word keep its forward name (`ClauseJournal.py:370`)?

2. **Revision.** Should a later arrival overwrite a refined token's `kind` (true revision), or should the first refinement fix it and later arrivals only append?

3. **Bootstrap.** Should this landing fix the pre-existing XOR zero-R failure (the 108 versus 135 normalization lead), or split it into its own item with the eight-file gate recorded as failing?

## Required before re-freeze

1. Fix findings 2, 3, 5 and 6, each with the regression named above.
2. Bind once per token per close (finding 4).
3. Fix the arrival audit and either record arrival menus in mentions or declare the asymmetry (finding 7).
4. Declare the two changes in finding 8.
5. Add the A decomposition to the receipts (finding 1).
6. Get Alec's answers to the three questions.
7. On the new hash: all twelve fresh draws, the eight-file set, both decomposition arms and both MM gates. No earlier-source pass is promoted.
