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

## 6. Identity without side structures (Alec, 2026-10-19/20; amends §1–§3)

Alec: only the major structures — percept, concept, symbol, STM, LTM —
may hold identity. `components.nouns` (signatures, witnesses), the
resolver's `seen` / `cooccurrence` / `associations` / `predicates` tables
and the **ICA event chart** are removed from the identity path. An LTM1
row holds **no embedded reference**: a reference would destroy the row's
order and concreteness. Generic (unminted) objects exist, and identifying
several generic references as one object *is* the binding problem, done
over **short-term concepts** until a concept is minted.

**Identity has two manifestations and no third:**

1. **Minted:** a concept row with its symbol at the shared index; its
   content is that row's vector, folded from its occurrences at each
   commit (what `absorb` computes today, written into the concept row);
   its occurrences are the LTM rows whose identity column names the
   symbol. Proper nouns are this from the first occurrence (step 1).
2. **Short-term:** the NP's fold point — the unminted centroid of the
   perceptual space its parts and wholes define — **with its `.where` and
   `.when` band**, held in STM for the situation. Verified 2026-10-20 on
   the production fold operators (`/tmp/centroid_probe.py` through
   `SentenceFixture`, width 8, untrained codes): identical description ⇒
   identical point; a narrower description is contained in the wider
   (meet equals the narrower); different descriptions separate; the
   determiner leaves no trace. Not verified, and not expected: identity of
   *individuals* — two cats introduced with the same words have one
   centroid, and an underspecified re-mention leans toward one candidate
   by the accident of the codes. So the location band is part of the
   short-term concept (the object file: Kahneman, Treisman & Gibbs 1992;
   Pylyshyn's indexes), and the remaining evidence is `bind`'s. The
   concept dissolves at the end of the situation unless recurrence mints
   it.

**What `bind` reads, from the major structures only:**
- content: the request's fold point against each candidate's point (a
  short-term concept in STM, or a minted concept row), in concept
  coordinates — containment first, then distance; no chart;
- location: the request's band against the candidate's band;
- expectation: the prime (a read of the symbol field) and the verb's
  participants read from LTM rows' identity columns (rebuildable cache if
  needed, like the definitions index);
- recency and role: the stream's row order and the recorded derivation;
- form: the word's percept and symbol;
- the combining weights are `bind`'s own parameters.

**Row shape (amends the two-truths amendment's storage note):** one
concept slot, the identity column, the address columns. The identity
column names minted symbols only; while a participant is short-term the
column is empty and its identity is the inference above. A fused LTM1
row's subject is therefore recoverable as a symbol once minted, and
before that only as the short-term concept the derivation points at.

**Before building:** rerun the probe on production codes (the [0, 1]
percept cube) as toy G0; the fixture's signed codes make the containment
test approximate.

### 6.1 The content term is the meet, and STM is an index (Alec, 2026-10-20)

Alec: a pronoun (a large region) and an individual (a specific one, many
defining parts and wholes) overlap, and as centroids they suffer the
bias §6 recorded; the **intersection** of their perceptual/conceptual
definitions does not. The probe agrees: distance leaned toward one
candidate by the codes' accident, containment through the meet did not.

- **Content term = two ratios from the meet** (coordinatewise minimum in
  the cube, the sigma/pi machinery): for request `q` and candidate `c`,
  `|q ∧ c| / |c|` (how much of the candidate the request covers) and
  `|q ∧ c| / |q|` (how much of the request the candidate covers). An
  underspecified request ("the cat"; a pronoun) contains every candidate
  of its kind, so content ties and location, recency and expectation
  decide — correct. A more specific request excludes candidates whose
  meet with it is only the kind. No distance anywhere in `bind`.
- **Form term = the same operation**: a form (a pronoun, a determiner)
  has a learned region; the test is the candidate's containment in it.
  This replaces the `cue_policy` matrix with a region per form, learned
  by exposure.
- **STM stays an index.** STM holds the addresses (`.where`, `.when`) of
  the most recent rows — eight is the right order (Cowan; Miller) — plus
  the symbol activation field the prime and the chooser already read. The
  candidate's full region is read from its LTM1 occurrence row's slot at
  bind time; the intersection is computed on read, never stored. A
  short-term concept is therefore nothing but a recent LTM1 row that STM
  points at, read through the meet; it needs no copy in STM and no new
  STM capability. (Holding eight recent activations in STM would
  duplicate the rows the close already wrote.)

### 6.2 Reinterpretation, and the migration as step 1b (Alec, 2026-10-20)

Alec: with the intersection computed on read, LTM stores the bound
versions, so a reinterpretation of identity would have to read the
surface text again. Not quite: the NP occurrence row keeps its fold point
and band raw and the sentence row keeps its fused point raw; the binding
is the identity column beside them. A reinterpretation re-runs `bind`
over the stored rows and rewrites the column (what `revise` does today,
with the occurrence points immutable). What cannot be recovered without
the text is the **parse**: only the fused derivation is kept. The one real
cost is the **minted concept row**: its content is a fold (meet/join) of
its occurrences, which cannot subtract an occurrence back out, so it is a
**rebuildable cache** over the rows that name the symbol, re-folded when
a binding is revised — like the definitions index over the DEF rows. Rows
item 5 has forgotten leave the fold as it stands.

**Step 1b — the migration (decided 2026-10-20; after step 1's landing,
before LTM1/LTM3):** keep the operation, its registry and reverse face,
the three site detectors, the kept-write contract and gates G1–G3;
replace `ReferenceClose.score`'s feature construction and the structures
it reads: (i) content = the two meet ratios `|q∧c|/|c|`, `|q∧c|/|q|` in
concept coordinates against the candidate's LTM1 row point or minted
concept row — the ICA event chart leaves the identity path; (ii) form =
containment of the candidate in the form's learned region — `cue_policy`
retired; (iii) expectation = the prime plus the verb's participants read
from LTM rows' identity columns with a rebuildable cache — the
role-keyed `associations`/`predicates`/`seen`/`cooccurrence` tables
retired; (iv) `components.nouns` (signatures, witnesses) retired — a
minted individual's content is its concept row, re-folded from its
occurrence rows; witnesses are a query; (v) STM = recent row addresses
plus the activation field, the intersection computed on read. Gates:
G1–G3 unchanged, the 18 reference-close cases ported to the new scorer,
both decomposition certificates in both arms, toy G0 on production
[0, 1] codes first.
