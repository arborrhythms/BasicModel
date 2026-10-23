I couldn't fully clear the source06 arm. There is one fixture bug and one production blocker, and either alone sinks the chain. Your three suspicions are real, but (1) and (3) mostly follow from the production blocker rather than standing as separate defects. This is from reading the code and the -05/-06 reports only; I ran nothing.

## A. -06: the fixture loops `bind` and sentence 1 never closes

Sentence 1 ("y is x plus one."), certificate.json from line 17011 on:

- After `verb(S, '.')` the stack top is `[PREDICATE, -1]`. Then `surface(is, VP)` produces scope flags `0`.
- In `ClauseScope.apply` (`bin/ClauseScope.py:115-117`), a headed wrapper keeps only `SENTENCE` from its head, so `PREDICATE` is dropped.
- The fixture's phrase guard (`forced_decomposition_grammar.py:134-137`) therefore sees a bare operand. Its right labels still contain `x`, from the overlay word column. So it requests `bind` on the whole "is x plus one ." phrase.
- That phrase has order 0, and bind keeps its operand's stamp (`Language.py:12580`). In `MTMOverlay.apply` (`MTMOverlay.py:145-156`), `opening = bind & order==1` is false, so the stack entry and its column 5 don't change. The fixture's `bound` check (`:128-129`) stays false and it binds again.
- About 30 binds use up the closing budget (`Models.py:21706-21707, 21857`), so `equal(y, …)` never runs. That gives `(0,1)`/`(1,1)` = `[None,None,None]`.

This confirms your suspicion (2): the wrapper loses the phrase status. -05 failed differently: stamping the constants cost a fourth operation per word, which broke sentence 2 (`[sym, None, None]`).

**Fixture fix:** treat the top as bare only when the overlay source column is zero, i.e. `int(identity_stack[b,0,5]) == 0`. A pushed word has 0 there; every bind/lower that opens a token, and every VP or S, sets `column+1`. This replaces the `bound`, `determined` and `phrase` tests and adds no production mask.

## B. Production blocker: a bind to a resolved antecedent leaves the whole sentence open

Even with A fixed, this would fail the same way -05 did. In -05, doc 1's question only had `what` open, yet it ran just `query, conclude`.

1. Every order-1 leaf gets `forward_identity` (`ClauseJournal.py:92-95`). `close_member` copies it onto a bind request (`:403-405`).
2. `ReferenceClose.prepare` (`:987-992, 1007-1008`) sees `named` and opens `('referent', 0)` on the child. When the reference is an `ltm` token it also sets the child's refs to `-1`. It does this even when the bind chose x's resolved token.
3. In `write_clause`, `pending` → `incomplete` spreads up to the root (`ClauseRow.py:448-453`). So the root is never prepared or Part-closed (`:476-478`). It is stored as a retained-open LTM0 row with occurrence refs (`:510-530`). That is exactly -05's `(0,1)` reading: `('ltm',…), verb, ('ltm',…)`.
4. The fixture's query reads Part rows. `relation_evidence` matches Part refs, which are `('sym', …)`, against those `ltm` refs (`Queries.py:1042-1058`). Nothing matches, the components are empty, and the `ask` never fires.

**This is also why y has `kind None` (your suspicion 1).** Because the root wasn't prepared, y's lowering was planned only as its own root, not as a relation argument. PartClosing reserves no symbol for that (`PartClosing.py:143-153`). So `_derivation_symbol(occurrence)` returns None and `commit` opens y with `kind=None` (`ReferenceClose.py:1108-1125`). Then `prepare:978-979` treats y as unresolved, your §17 repair copies that onto the question's role 2, and `read_reference_bindings` leaves it unfilled (`ClauseRow.py:639-640`). Result: `[referent 0, referent 2]`.

**Smallest fix (`ReferenceClose.prepare`):** if a compose bind picked a token that has a kind (or a native int), don't treat it as `named`/open. Use that kind as the composition reference and as `refs`, so PartClosing plans the enclosing relation. Unresolved or anonymous binds keep today's open path. With that, y gets an argument symbol in sentence 1, and only `what` stays open in the question. This keeps today's layout, readers and minting, and fits §1.5 (resolved sentences compress). Before landing, check with `git log -L` whether the leaf `forward_identity` on resolved binds predates this line.

## C. Remaining kind/reader conflation (lower priority after B)

`kind is None` means two things: "anonymous pending" and "finished own object without a symbol". The readers that conflate them are `ReferenceClose.py:978`, `MTMOverlay.py:56` (which also stops later equalities updating the value) and `ClauseRow.py:639-643`. The expression reader only follows a token through its derivation symbol (`PartRead.py:205-211`).

**Fix:** add an explicit `pending` field on `Token`, set only by the anonymous `open`, cleared wherever `bind`/`merge` supply a kind. Those three readers should test it, and `read_reference_bindings` should fill a non-pending token from its value. This changes the checkpoint format (legacy: `pending = kind is None`), which §5.8 allows.

## D. Review of the §17 repair

It's correct, with these points:

- The loop at `ReferenceClose.py:1039` covers role 1 as well. Restrict it to roles (0, 2), matching `sites()` at `:676`.
- `opened` can get duplicate entries; dedupe them.
- It amplifies B/C: any resolved child that is wrongly unresolved now also opens the parent's role. Land it together with B.
- Add a regression where a bind to a resolved token leaves the parent role closed.

## Decisions to flag for Alec

1. **A bind with a location on an operand that isn't order 1.** The menu offers it (no word-order gate, per §5.3). When chosen, ClauseScope takes the token ref (`resolve`, `:72-89`) and STM takes the joined value. But the overlay doesn't open it, and `close_member` drops it (`order != 1`). That conflicts with "a recorded bind is authoritative". Either gate those locations or define what binding a phrase means. Choosing between them changes the contract.
2. **Repeated binds aren't excluded.** Bind on the same operand is idempotent, yet a learned chooser can repeat it until the budget runs out, like the fixture did. Excluding bind-after-bind on the same operand, like the inverse pairs, is a grammar change.
3. **Whether B fits his intent.** It decides whether a bound mention writes its derivation symbol into the Part row or stays a working reference. I read §5.1 and §1.5 as allowing it, but it's his design boundary.

One thing I couldn't explain: why doc 1 in -05 resolved y while doc 0 didn't.
