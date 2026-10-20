# `bind` as a thought operation: the concrete design (Claude, 2026-10-19; for Codex, item 3.5 step 1)

Decided by Alec on 2026-10-19 ([todo 3.5](../../todo.md), [catalogue §3.3](../specs/2026-09-29-operator-catalogue.md#33-bind),
[thinking spec §1.1](../specs/2026-10-07-thinking.md#1-the-open-reference), which already
has `bind()` as a method `interpret()` calls and binding as two-stage). This
plan says what to build, in the code's own names. Nothing here is a new
scorer, a word class, or a language.

## 1. What exists (read 2026-10-19)

| piece | where | what it does |
|---|---|---|
| binding request | `ReferenceContext.reference_requests` (from the chooser's actions) → `ClauseJournal` `close_request` on the parent node → `Clause.reference_request` = `dict(mode, leaves, cue, content, role, predicate)` | only for operands of rules that declare `reference_roles`/`binding_spec` with mode `mint`/`bind`/`pronoun` at order 1: `lower` and `lower_unary` (`determiner_mode='mint'`), `definite_unary` (`'bind'`), `ContextualBindLayer` (`argin2` `'pronoun'`; vestigial, no reverse) |
| stage 1, bond at the word | `Interpret.bind(value, identity, scope, candidate_ids, …, mode)` | backward candidates among STM participants; offers mint (`-1`) and open (`0`) to the ordinary chooser; no word class, no draw |
| stage 2, resolution at the close | `ReferenceClose.preview(clause, frames, document, position)` → `prepare` → `commit` (`Models.py` ≈19622, 20336, 20350) | per order-1 constituent with a request: candidates = `accessible(frames)` (STM frames + `disc.situation_references(b)`) + `'new'`; features `explanation`, `prediction`, recency, candidate role, request role; `policy` (5 weights) + `cue_policy` over the form; argmax; cost trains the chooser and the cue weights; `commit` mints or binds, writes `store.refs[index, 0]`, learns `seen`/`cooccurrence`/`associations[(atom, head)]`/`predicates[head]`, metadata `_individual_context` (incl. `role`), audit with `missing_reason` (`'no_accessible_individual'`) |
| open slots | `ThoughtReferences.open_slots(meaning)`: `_open_references` binding (`('referent'|'relation', role)`), plus unfilled roles of a relation row | what a thought episode works on; `needs_episode` gates `run_selected_thought` |
| thought operations | `Language.ThoughtOperationSpec` (`semantic_id`, `operand_roles`, `result_role`, `forms`) declared in a grammar's `<thought>` list; `GrammaticalThoughtRegistry` joins `THOUGHT_EXECUTORS[semantic_id]`; `ThoughtOperationCandidate(operation, request, open_roles)`; the one chooser selects; `R + A` judges | `ask`, `query`, `part`, `whole`, `equal`, `not`, the faces of §2 |

What is missing: a **site** that is not a determiner; `bind` as something the
thought chooser can select; and the elided participant, which no code detects.

## 2. The operation

- **Declaration.** `semantic_id='bind'`, `operand_roles=('argin1',)`,
  `result_role='argout1'`, one unary structural form (the same shape as
  `not`), listed in the `<thought>` sections beside `ask`/`query`/…
  `THOUGHT_EXECUTORS['bind']` is the executor below. No chart rule id: it
  cannot fire in compose/generate.
- **Request.** The operand is the row's meaning with one open referent
  role `r` (`('referent', r)` in `open_slots`). `ThoughtOperationCandidate.open_roles=(r,)`.
- **Forward (executor).** Build `individuals` exactly as `preview` does —
  `components.candidate_frames(frames)` for the situation's frames plus
  the episode's own earlier constituents — and score the open role's
  content against them with **the same function**: refactor the body of
  `preview`'s per-constituent loop into `ReferenceClose.score(content,
  predicate, cue, role, individuals) -> (choices, features, logits)` and
  call it from both `preview` and the executor. Selection = argmax; the
  result's evidence pair = `(p_selected, 1 − p_selected)` from the softmax;
  the effect = `role_refs[r] ← identity` (so `('referent', r)` leaves
  `_open_references` and enters `_bound_roles`, `ClauseJournal.py:580`),
  written through `prepare`/`commit` at the close as today. If no
  candidate clears the resolver's own `'no_accessible_individual'`
  condition the operation returns the pair `(0, 0)` and the role stays
  open — thinking spec §1.1's forward reference.
- **Reverse.** Re-emit the request's `leaves` (the description or the
  determiner) when there are any; nothing when the site was elided — the
  no-leaf face of occurrence spec §12.
- **Credit.** The episode's `R + A` judge credits the choice *to bind*
  (the thought chooser); the resolver's existing cost credits *whom*
  (`policy`, `cue_policy`, the operation chooser through `explanation`).
  No third signal.
- **Called once per site.** Several open roles in one row are several
  `bind` candidates, one per role; the chooser orders them.

## 3. The three sites, concretely

1. **Definite phrase (today's cue).** Unchanged: `definite_unary` emits
   the request; when `preview` resolves it the close binds; when it leaves
   it open (`missing_reason`, or the chooser's `'open'` formation,
   `ClauseJournal.py:112`) the role is in `_open_references` and `bind`
   is a candidate in the episode. No new code beyond §2.
2. **Operand with no object of its own (the pronoun case).** In
   `Interpret.forward` (word face), when `index.objects(word)` is empty
   and the word is not `_field_pending`, do not return `None`: return the
   word's row as content with a request `mode='pronoun'`, `content` =
   empty (zeros), `cue` = the leaf's form (`object_atoms`/the WS form
   band), so the request reaches `reference_requests` without any rule's
   `reference_roles` — add a per-leaf request path in
   `ReferenceContext.reference_requests` (a `kind == 0` leaf whose
   interpretation flagged `no_object`). The resolver then scores it on
   `prediction`, recency, role and cue, which is what a pronoun has. This
   replaces `ContextualBindLayer`'s route; delete that layer when this
   lands (it has no reverse and returns its left operand).
3. **Open role in the verb's frame (the elided case).** At `commit`, key
   the learned association by role as well: `associations[(atom, head,
   role)]` with `role` from the context already written to
   `_individual_context`. Define `ReferenceClose.expected_roles(predicate)`:
   the roles `r` for which `Σ_atom associations[(atom, head, r)]` over the
   predicate's `support` heads is at least `recurrence` (the components'
   existing `recurrence=4`). At the close (`ClauseJournal` where
   `_open_references` is written), for a row whose predicate is present
   and whose `role_mask[r]` is `False` with `r ∈ expected_roles`, append
   `('referent', r)` to `_open_references`. That is the whole detector: a
   verb that has had a participant at role `r` four times and has none now
   has an open role there. It enters the episode like a question's open
   role; `bind` may fill it from the situation or leave it open. Nothing
   names a word, a class or a language.

## 4. Order of work and the gates

1. The `score` refactor of `preview` (no behaviour change; the existing
   reference-close certificates must hold unchanged).
2. The `bind` declaration, executor, reverse; the episode selects it on a
   row with an open referent role; `R + A` credit through the existing
   `thought_pair`.
3. Site 2, then delete `ContextualBindLayer` and its tests' assumptions.
4. Site 3: the role-keyed associations and `expected_roles`; the
   `_open_references` write at the close.

Gates, each a generated corpus from `test/constituent_corpus.py`'s
vocabulary, no word class named, the resolver's choice reported against
the generator's referent:

- **G1 definite**: *a X … the X* with a distractor individual; the bound
  reference is the generator's.
- **G2 pronoun**: *a X … PRO V* where PRO is a generated marker leaf with
  no object; resolution by prediction, recency, role, cue.
- **G3 elision**: a transitive verb seen with an object four times, then
  without; the close opens role 2; `bind` fills it from the situation or
  leaves it open, and the audit says which.
- The existing reference-close certificates (item 6 §16) unchanged; one
  sweep at the landing.

## 5. Not in this step

Minting by recurrence in LTM1 and the LTM1/LTM3 layout (3.5 step 2);
cross-document individuals (order-2 kinds); any learning claim.
