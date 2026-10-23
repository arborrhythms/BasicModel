No §1 reversal is needed for any of the three failures. All three repairs are justified under §39. I worked read-only from the frozen source, the source-09 probes and the patches. I ran nothing, and the checks below are traced through the code and the retained probe frames, not executed.

## 1. Compose bind names an unwritten introduction (`test_math_chain`)

**Cause.** The overlay and the recovered clause disagree about what a determination over a relation is.

- **The overlay treats it as an object.** `MTMOverlay.apply` opens a handle for every determining operation, whether or not its operand is a relation (`bin/MTMOverlay.py:147-153`). It writes `choice.candidate` as the handle's value and records origin tag 3 at that column. The probe's `tag 3, address 7` is that write. Later journal rows then treat handle 24 as an identity: the equalities at actions 21 and 25 update it (`[24,24,0,·,-1,1]`), and action 43 binds to it.
- **Recovery never creates a site for it**, for four separate reasons:
  - `finish_clause` creates the determiner's request only when `region(description)` is not None (`bin/ClauseJournal.py:214-217`). A relation has no region.
  - `close_member` rejects any item with `has_point=False` (`:376`).
  - `recover()` unwraps the determiner and stamps the clause with the inner relation's identity, `identity_record(node)` (`:421-428`, `:732`), while the composition stays the determiner's. That is why the probe shows a `definite_unary` composition carrying action 5 and handle 0. Handle 24's identity is erased.
  - `retain_introductions` needs a `close_request` before it can recover a required handle (`:767-769`).
- **The defect already loses data without any error.** `_commit_events` skips an event whose subject handle has no token (`bin/ReferenceClose.py:1293-1299`). The probe shows handle 25 as an event subject (action 17) with no site either. So equality events on objects like this are already dropped from the MTM history (§1.5). I'm inferring that 25 was opened the same way (likely a bind with no antecedent chosen, over the relation); I couldn't read that journal row.

**Does a determined relation need its own occurrence-backed object site?** Yes. §2 says a determiner's result is an NP the chooser may bind, and the overlay, both equality updates and the later bind already treat it as one. Forbidding determination over relations would be a menu mask, which is not authorized.

**How to tell an introduction from an event**, using only the transaction record (journal columns: result, subject, object, verb, bind, update):

| Kind | Journal record |
|---|---|
| Introduction | `event[0] == h > 0`, `event[1] == 0`, `event[4] <= 0`, and origin `h-1` is tag 3 written at this action's own column. Determining rules, and an order-1 bind with no antecedent chosen (`handle == fresh`). |
| Use | `event[4] > 0` |
| Event | `event[1] > 0` (S or equality subject). Always recovered as a relation clause by the existing event branch, never as an own object. |

None of these tests uses `has_point`, content magnitude or a seat.

**Minimal repair:**
1. **`ClauseJournal.py:206-228`.** When the record above marks the action as an introduction and `region(description)` is None, still set `close_request`, with `content=frames[action, 2]` and `introduction=True`. That content is exactly the value the overlay opened (`choice.candidate`, `Models.py:21822-21831`), so it is §1.4's initial value, not a new meaning made at close. Points that have a region stay unchanged.
2. **`close_member`.** Accept `request['introduction']` regardless of `has_point`. Return the own object with:
   - `point` set to that content;
   - `children=(recover(description),)`, so the relation clause keeps its relation, references and children as its own row;
   - the introducing node's `identity_record`;
   - `subject_word_id=-1`.

   The parent event's role then references the object (`('clause', n)`). That matches the journal, where the subject is 24, not the relation. Part edges come from the composition, which doesn't change.
3. **`ReferenceClose.has_word` (`:463-470`).** Return True for `request.get('introduction')`. Without this, `own_object` stays False and `handles[24]` is None again: `definite_unary` has no `implementation=` in `complete.grammar`, so its `formal_name` is not `'lower'`. Keep `subject_word_id=-1`. Otherwise commit records the first leaf's word as the token's reading (`:1123-1125`), so "x" could later resolve to the relation object.
4. **Repeated wrappers.** Only one node per handle owns the tag-3 origin. A later bind wrapper that chooses no antecedent reuses the operand's handle (`MTMOverlay.py:145-146`). Today `retain_introductions` counts that wrapper as the owner, so the site may be sorted at the wrapper's later action, after a use in between. The site's identity and sort key must come from the introducing action. A second determination opens a fresh handle, so both tokens survive independently.
5. **Guard.** Once introductions are guaranteed, change the silent `continue` at `ReferenceClose.py:1298` to raise for tag-3 handles, so this can't recur unnoticed.

**Regressions to add (all existing assertions stay):**
- The captured shape: determination over `implies`, two equality updates, then a compose bind. Check the preview handle, the token's history of both events in journal order, the updated value, and the relation row unchanged.
- A bind with no antecedent chosen, over a relation (the handle-25 case).
- Determination of a determination: two tokens.
- A bind with no antecedent over a determined relation, then a later use, on both the direct and the re-admitted path.

## 2. Admission changed the selected clause structure (`test_gradient_factorization`)

The proposal is correct and narrow. Evidence:
- `Language.py:12649` mints only when `order > 0`. At order 0 admission never mints, so the comment's reason for keeping the source id in preview ("a trial may not mint") doesn't apply. The `-1` at admission is just the fall-through.
- The existing identity fallback already keeps source ids: when no order is ≥ 0, `Models.py:12839-12840` drops the references and `concept_ids` are used. Order 1 also keeps the source id (`:12636-12641`, and `test_lexical_reference_orders.py:47-49`).
- Keeping the word's own id cannot define a word as another word's object. `PartClosing._definition` returns on `word == symbol` and on word symbols (`:203-207`).
- §14 requires scoring and admission to enumerate one structure. The probe confirms this: the admitted children `[3,4]` and `[1,6]` are absent from the scored reading, and the keep decisions themselves are unremarkable (greedy kept, A=0).
- The direct callers' assertions (`test_operators_final.py:162`, `test_word_interpretation.py:257-258`) are at orders 1 and 2 and are unaffected.

**Caveat.** An ambiguous order-0 word (associations exist but `cid` is None) diverges the same way: the source id in preview, `-1` at admission. It would trip the same guard. I recommend making order 0 identical in both phases in this same change: keep the source id when there is no association, and use `-1` in both phases when ambiguous, per the docstring. Add a regression for each. Higher orders stay unchanged.

## 3. Old Difference fixture outside the decoder domain (`test_review11_contracts`)

This is a fixture-domain port, not a production change.
- The `[0,1]` operand check is in `Language.py:13481-13483` (`decoder_eligibility`). Neither it nor `Generative.inverse_menu` is in `production.patch`.
- The step-2b decoder-domain receipt already cites this check, and its §66 echoic port made the identical ×0.5 rescale, accepted with LAND.

I checked the scaled fixture by hand:
- **Inputs.** Event `[[.5,.5],0,0]`, bank `[[.5,0],[0,.5],[.5,1]]`.
- **First step.** Sum `[.5,0]+[0,.5]` and Difference `[.5,1]-[0,.5]` are both in domain. Stop stays ineligible because `[.5,.5]` has no unique exact single-code fit. So `alternatives[0,0]` holds, and the bias `[1,0,20]` with zero weights selects action 0.
- **Leaves.** Both are unique exact fits with zero residual, so `count == 2`.
- **Gradient assertion.** Unaffected, since the weights are zero.
- **Tolerance.** Its `clamp_min(1/width)` term is not scale-invariant, but it only loosens the bound, and these residuals are exactly 0.

Record it in the assertion audit as a change to the literals only.

## What this doesn't establish

This is a focused diagnosis of one failed frozen source. It is not a LAND or final review, and it makes no gate or learning claim. The replays are diagnostic only, because the corpus generator wasn't restored. Each repair needs its own frozen hash and the full scoped gates.
