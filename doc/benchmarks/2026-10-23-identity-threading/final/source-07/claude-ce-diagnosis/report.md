I found the exact cause. It's a mechanism defect, not a decision boundary, and no Alec §1 decision reverses. Under §39 Codex can proceed with a dated record. Everything below comes from reading the source and retained captures only; I ran nothing.

## Cause

**1. Where it actually starts: the write boundary drops the nested dependency.** `bin/ClauseRow.py`, unchanged since `b38889eb7` (production.patch only touches `row_kind`, `written_rows` and the pending bit there).
- `:448-453`: `pending()` only looks at a node's own annotations. `incomplete()` also looks at its descendants.
- `:476-478`: an incomplete node is never PartClosed.
- `:510`: `retain_open = pending(value) or (value.relation is not None and incomplete(value))`.
  - The fused "x plus two" node has `relation=None` and `pending=False`, so it falls into the completed-idea branch at `:531-540`.
  - Since it was never prepared, `symbol is None`, so it is stored as an opaque idea: refs `(None,None,None)`, `_open_references: []`, evidence `(1,1)` (ce-memory-probe row -6601…).
- The root `y = ·` is retained open because of the descendant. At `:519-524` its role 2 names the child's occurrence, and `:541-544` copies only its own marks, which are empty.
- Result: row 3587… is an asserted, closed equality whose operand is a placeholder computed from x's zero own-object content (the plan's `individuals['new',0'].content` is all zeros). Nothing on either row records that x (-8215…) is still pending.

**2. The read boundary trusts it.** `bin/ThoughtBinding.py:84-102` accepts any `meaning.role_refs[answer]` from an equality frame. The frame came from the cued LTM0 path in `bin/Queries.py:956-985`, not from Parts. The follow-up `_query_relation='equal'` read over Parts (`:1033ff`, PartRead components) correctly found nothing, because no edge exists. So PartRead isn't the culprit; it is behaving correctly. Your hypothesis is right about the symptom, but the evidence is already gone before the query runs.

**3. Why the test used to pass: the earlier pass was vacuous.** In `ports-03`, the episode for "what is y ?" queried the bare `what` site (3292930159256856593, the leaf-0 bind occurrence, shown in its arrival-keep), with no equality and no complement. `Queries.py:959-964` returns empty for a lone unconstrained name. Since §17 the question correctly keeps `what = y` (657563931269123051: refs `(what-site, is, y)`). That exposes defect 1.

The §17/§19 propagation at `bin/ReferenceClose.py:1051-1067` is correct as restricted. Copying x's name onto root role 2 would be wrong, because role 2 is x+2, not x.

## Is there an existing check that certifies resolution? No.

- Row -6601… (fused, unresolved) and row -90059… (minted x, resolved) are identical in every resolution-relevant field: one slot, refs `None`, no open slots, evidence `(1,1)`, no MTM token. Telling them apart by `_producing_operation`, a seat or content would be the kind of inference §18C rules out.
- §19's `Token.pending` only covers identities. Opening an MTM token for a fused value would turn it into a bindable location, which changes compose-bind menus and parity.
- The dependency is known only at write time (`incomplete()`) and, briefly, in `ThoughtClosing.py:30-60`. That code already excludes 3587… and -6601… from x's own query, but never persists the fact.
- I rejected writing role 2 as `None` so that the existing `role_refs[answer] is None` test (`ThoughtBinding.py:90`) catches it. It drops the reference §2 says to keep, and it would make the declarative itself raise a role-2 root episode, minting the fused value in the x example.

So a small piece of metadata, written once at the existing write boundary and read at the existing answer boundary, is appropriate. It stays accurate under the current layout: when x later resolves, -6601… is never recomputed, so answering from it would stay wrong. Re-resolving it belongs to item 3.0.

## Minimal repair

1. **`ClauseRow.py:515-530` and `:543-544`.** In the retain_open loop, for each role whose `('clause', n)` child satisfies `incomplete(child) and not pending(child)` (an unresolved dependency strictly below the child), collect `(role, child_occurrence)`. Store it as one sidecar binding on the parent, e.g. `_unresolved_operands`.
   - Directly pending, named children are skipped; they already have open slots, so §17 is unchanged.
   - No tensor, column, row or derivation is added. The fused row stays exactly as stored today, and every row is written once.
2. **`ThoughtBinding.py:88-91`.** `continue` when `answer` is in the frame's `_unresolved_operands`. Equality edges are left alone: rewriting a region through y = x+2 is still valid; only taking an unresolved operand as the final answer is refused. The skip returns `goal`, and `ThoughtStream.py:149-151` returns that, so there is no fallback fill.
3. **`ThoughtFeatures.py:64-67`.** Add the new key to the excluded list so chooser inputs stay byte-identical (same contract as `test_section14.py:29`).
4. **Optional, same mechanism:** `ThoughtReferences.fill:188-197` imports `supplied.role_refs[role]` from a single frame for non-equality goals. Give it the same guard if you want to close this class of bug; this gate doesn't need it.

## Regressions

- **The gate:** `test_math_chain_section14.py:280` stays verbatim.
- **Write unit:** a direct `write_clause` of `y equal (fused ∘ pending x)` marks role 2 with the fused occurrence, and the fused and raw rows are unchanged. A resolved-x counterpart gets a Part edge and no mark. A direct pending child (§17 case) gets an open slot and no mark.
- **Integrate unit:**
  - A marked equality frame leaves the goal open, with no `_bound_roles`.
  - The reverse direction (answering y for the region -6601…) still binds.
- **Metadata unit:** `semantic_metadata` is identical with and without the mark.
- **Sidecar round-trip:** checkpoint save/load preserves the new key.
- **Rerun:** both pending-premise orders, the R10 alias and direct-role certificates, set85, the native set, and both decomposition arms, all on a new frozen hash. No result from 59b7c8b8 should be promoted.

## The decomposition cases now passing

In both fixtures x (or z) is bound to an already-resolved identity (§18B), so "x plus one" is complete. The root is PartClosed and answers come through the Part relation and component path. No mark is ever written, so the in-progress original-arm runs on this hash shouldn't change.

One caveat: if an explore trial picks the dummy for x (a pending own object under §19), that premise becomes incomplete and the guard refuses an answer through it. That outcome is correct, but it could change explore-suffix costs, which is why both arms need to rerun on the repaired hash.

## Decision boundary

This is not a new boundary. §2 ("an unresolved object leaves the event as an open reference") and the existing assertion (empty search, question still open, nothing minted) already decide it. Accepting a symbolic answer like "x plus two" would be a new decision for Alec, but it isn't a §1 reversal and isn't needed here. Codex should record this as spec §20, dated 2026-10-23, under §39.
