**Verdict: DO NOT LAND.**

The receipt is complete and consistent, and most of the identity line matches §1 and the documented repairs. One change in the live compose path is neither authorized nor documented, and the existing test meant to protect that behaviour doesn't exercise the path that changed.

I worked read-only. I traced the code; I did not execute anything, and no test or learning stage ran as part of this review.

## Necessary fix

**1. With the identity overlay on, compose no longer looks up the existing typed concept for an operand.**

- **Where:** `bin/ReferenceContext.py:343-346`. With `identity_state` supplied, `propose` returns the raw operand before calling `resolve_order` (line 370).
- **Why it reaches production:** `Models.py` passes `identity_state` and `reference_scope` together under `sentence_transactions`, so every production call to `prepare_operands` now takes this early return.
- **What changed:** before this line, at close time, any operand requested at an order other than one was replaced by the existing concept at that order (`value = where(order==1, value, typed_value)`). That covered the kind and generic determiners and the implicit S/VP/relation roles.
  - Now composition uses the word's own leaf value for these operands.
  - Stored references are mostly unaffected, because `resolve_lexical_references` still resolves them at capture.
  - The numbers are affected: the fused points, the values passed from one VP to the next, and the overlay's stored state for generic and kind sentences now come from a different operand.
- **Why the gates miss it:** `test_live_references.py::test_existing_kind_point_is_resolved_before_the_operator` calls `prepare_operands` without `identity_state`, so it still passes.
- **Why it isn't authorized:** §1.3 and §5.3 remove only the reference variants and keep the determiners' folds and stamped orders. Neither the spec nor the README mentions dropping this lookup.
- **Minimal repair:**
  - In the identity branch, keep the single typed option and still offer no reference alternatives. Apply `resolve_order` where `order != 1`, keep the raw value where `order == 1` (bind owns that order), and keep the existing `literal_ids` exemption.
  - Parametrize the kind-point test over `identity_state`. It should assert the kind's point and that exactly one unary option is offered.
  - Then freeze a new source and rerun all the scoped groups on it.

## Notes for a later round (not blocking)

- **`test_live_references` port** (`test/test_live_references.py:264-265`): it now asserts overlay handle 1 instead of `cid`. Handle 1 is the staged held occurrence, but nothing checks its origin. Add `entry.identity_origins[0, :2] == (2, occurrence[-1])`.
- **Decomposition mapping:**
  - Categories and rules are mapped by name, and every surviving row is confirmed bitwise equal, for both seeds (4199722147, 2428440506) in both arms.
  - The `tool_embedding`, `what_projection` and `apply_anchor` rows are mapped by position, which assumes bind is the last unary rule. It is: it's the last `<compose>` rule in `complete.grammar`.
  - The equality check reuses the same index map, so for those rows it confirms the copy rather than the mapping itself. A name-based assertion would close that gap.
- **`MediumTermMemory.token_of`'s new event fallback** (`:801-803`): when `open` or `bind` hits an occurrence that is some subject's event, `forget` deletes that event from every history. It doesn't trigger inside one close, because opens and binds run before `_commit_events`. A re-close at an event occurrence would lose history silently, though, so it should be guarded.
- **Captured events:** without `factored_refs`, an event's verb is the leaf word id (`MTMOverlay` stack column 1). `_participants` then drops it if `point()` is None. This doesn't affect G1–G3.

## Verified and acceptable

- **MTM:**
  - It holds current values, names with their readings, and event histories, each with separate ownership.
  - VP results replace the value; descriptions join it.
  - Every committed value is detached by `_value()`.
  - Checkpoint v2 is validated before it replaces the pool, and v1 migrates under its old convention.
  - Native individuals get an entry keyed by their occurrence; the bank is read but never written.
- **Compose bind:**
  - The menu is one action per staged frame location or valid overlay location, plus the dummy.
  - The dummy's logit stays exactly zero, both before and after priming, and `-inf` exclusions are preserved.
  - Credit reaches the bind both through the derivative and through the score function.
  - The value scored for reconstruction is the one captured at the bind (`record`).
- **Close and commit:**
  - Preview treats a recorded compose bind as authoritative and never chooses again.
  - Trial isolation, provisioning (pool restored in `finally`) and the stream guards (`:1547`, `:1607`) are correct.
- **No writes from MTM into concepts:**
  - `expose_form` and `absorb_form` are gone, and `IdentityRows.mint` has no production caller.
  - The remaining bank writes are close-time minting (`PartClosing._seat`) and word identity, both retained by decision.
  - The four new certificates check that training and later mention loss never allocate or modify a concept region, and that the loss still carries gradients.
- **Words as objects:** `Interpret.define` raises, `_definition` skips, and the N1 special case is removed.
- **Ordering and structure:**
  - Introductions are recovered beneath wrappers, and determined relations get their own occurrence-backed object.
  - `sites()` uses one chronological order for scoring and admission.
  - An event whose determined introduction is missing raises.
  - An unchanged dummy keeps its carried object without a seat.
  - Copied stores re-attach their index owner, with one observer per store.
- **Order-zero words:** preview and admission now agree, and word ids are excluded from the ambiguity check.
- **c/e and query guards:** the `_unresolved_operands` annotation is write-once on the parent's existing sidecar. It is excluded from `semantic_metadata`, and enforced in `integrate`, `fill` and `best_match`. The unmarked reverse endpoint and explicit reflexive witnesses remain valid.
- **The §18–19 bridges:**
  - Pending is now an explicit flag.
  - Resolved compose binds keep the current Part relation.
  - Pending names propagate only to participant roles 0 and 2 of relation clauses.
  - The transpose-then-surface port is a fixture-only change.
- **Item 3 boundary:** no rows, columns or time stamps are added. The occurrence mapping is temporary, so none of this implements the deferred layout.
- **Test ports:** every port matches its README row and `assertion-audit.json`. The tensor-literal edits are covered by the audit's note that `test-ports.patch` owns those checks, and no port weakens a contract except the minor `live_references` gap above.
- **Gate counts:** they reconcile exactly for hash `bf44bbc47`: 1,611 passed and 12 skipped out of 1,623 cases, across 132 files.
  - The skips are nine absent phase artifacts plus three opt-in slow MPS/Inductor tests. None counts as a pass, and no GPU execution is claimed.

**Limitations of the evidence:**
- The pending-premise certificate shows 0 bound, so there is no learning evidence.
- The derivative checks are unit-level overlay checks; the end-to-end chooser gradient check runs eager only.
- The retained-RNG replays did not restore the presentation corpus's private generator.
