# The thought loop: audit and decisions (Claude, 2026-10-07)

Alec, 2026-10-07: "My feeling is that the thinking operators may not be
fully specified." An audit of the documents against the code confirms it.
The seven registered thought operators (`not`, `part`, `isPart`, `equal`,
`quantize`, `arma`, `what`; `bin/Queries.py` `_thought_executors`) are each
declared with operands, read/write scopes and evidence kinds, but no
document specifies the loop around them end to end, and the code has filled
the gaps in ways no one decided.

## 1. Findings (verified in code where marked)

1. **Thought runs only on questions** (verified: selected thought raises on a
   non-interrogative meaning, `Models.py` ≈6207; the query field is built
   only for interrogative clauses, ≈20534). Conclusions after declaratives,
   the production plan's between-sentence thinking, and the accessible-mind
   spec's absence inference (§2.6.5: conclude `not X` from an uncancelled
   image) have no entry point.
2. **Thought's choices are effectively untrained** (verified:
   `selectedThoughtPolicyWeight` defaults to 0.0, `data/model.xml:312`; only
   `MM_query_reasoning.xml` sets it, and that config leaves `answerSynthesis`
   off, so no episode opens in training). Untrained, the chooser runs the
   parsed request once and concludes. The `thought_pair` greedy/explore
   comparison exists (`WalkTrials.py`) but is documented only as "paired
   walks".
3. **Results never become operands.** Candidates bind operands only from the
   request, active and root meanings; a result reaches the chooser only as
   support scalars. No document specifies chaining.
4. **Thought writes nothing to LTM** (no write permission,
   `AccessibleMind.py`), against the production plan ("LTM retains ordinary
   structured thoughts"); round 4a-0 already addresses thought-made rows.
5. **Scalars.** `part`, `equal` and `isTrue` return scalar support, against
   catalogue rule 6 and now against the two lanes (two-truths §1.1).
6. **Stale or inconsistent documents.** `exist`, `true`, `lookup` described
   as live in QueryContracts, AccessibleMind.md, Reasoning.md, the 09-18
   spec and `complete.grammar`; `query` specified as the union of matching
   rows but returning the best row, with a knowing write the accessible-mind
   spec defers; the budget named `selectedThoughtBudget` in two documents
   and rejected by the code in favour of `attentionBudget`; the 09-18 spec's
   status line still "pending".
7. **Operator-level mismatches.** `not` negates only an operand of the
   question, never the expectation image; open-role `part` sets are not
   seeded into knowing; `isPart` at order 0 bypasses the taxonomy;
   `conceptualize` is unreachable; `arma` ignores its operand; the
   gain-setting act (§2.6.6) is unimplemented; `equal`'s choice of identity
   check over "two part questions" is unrecorded; thought faces have no
   specified inverse.

## 2. Decisions for Alec (with Claude's recommendations)

1. **When thought runs.** At every sentence closing, declarative and
   interrogative, within a per-sentence work budget; a question additionally
   opens a subgoal; anticipation (`expect`) runs between sentences.
2. **What thought may write.** Its conclusions become LTM rows of kind
   inference, addressed by 4a-0's rule (document/turn, ordinal, content),
   with an evidence pair and references to the rows they were drawn from,
   so forgetting can judge deducibility.
3. **Chaining.** Each result enters the serial stream as a slot the next
   operation can bind (working memory).
4. **Training.** The scheme of compose applied to thought: the greedy
   thought episode against one departure, credited by the owner-step cost of
   what follows — the answer where supplied, expectation's error on the next
   sentence everywhere — so thought that anticipates better is credited. The
   REINFORCE/EMA path retires into it.
5. **Results are evidence pairs or concepts, never scalars.**
6. **Inventory and names.** `query` (what), `expect` (arma),
   `symbolize`/`conceptualize` (quantize and its reverse), `part`/`whole`/
   `equal`, `not` (including over the expectation image), the gain act;
   catalogue questions 6–8 (`expect` kept in thought; a sentence absolute
   when it closes to one slot; `bind` as `interpret`'s choice for a word,
   kept for a participant with no word); stale documents corrected; one
   budget name.

Proposed sequence: the thought spec written while Codex builds 6.5 (which
does not depend on it); the thought step after 6.5 and before 6; forgetting
(5) depends on it for deducibility.

## 3. Alec's rulings on the open questions (2026-10-07)

- **Interpret and bind.** `interpret` is the decision to look up the object a
  word names; `bind` is the decision, for a pronoun or definite article, to
  identify that reference. Alec asks whether binding can be implicit.
  Claude: yes — one decision with two triggers: `interpret` chooses between
  minting a referent and binding an earlier occurrence; a determiner or
  pronoun forces the choice, recurrence within an active frame leaves it to
  the chooser, credited by what follows (object permanence, 2026-09-21;
  "identity is an expectation").
- **Expect.** Applied at every level, so a global setting, not a grammar
  operator (`<sentenceExpectation>` and the gain `g` already exist). `arma`
  leaves the thought inventory; the expectation's image reaches thought as
  chooser context, two-laned.
- **Exist.** A content-addressed search of LTM for an absolute truth: the
  row's presence, evidence pair and trust determine existence. Returns as a
  query form leaving the pair (never a scalar); `isTrue(P)` is `exist(P)`.
- **LTM relations.** part-of (is-a over extents included), equal, implies
  (if-then), and the **attitudes** (a subject's relation to a proposition:
  knows, believes, wants, doubts, said — the modal projection; two-truths:
  attitudes do not assert their argument). Before/after are `.when`
  containment queries; cause is an implication with a `.when` order.
- **Renames** sooner: in the thought step. `quantize` to future work (all
  concepts are symbolized).
- **ARMA** is the expectation (`InterSentenceLayer`: a Linear–Tanh–Linear
  predictor over the last 5 sentence representations and 2 residuals, MSE
  against the next sentence; a next-word CE predictor within the sentence).
  `what()` is still the recursive subgoal query (`continues=True`); union vs
  best row still to be ruled.
- **`isX` formalized.** For every relation X over concepts: `X(a, b)`
  conceptual, computed from codes (containment, residual, identity, region
  containment), leaving conceptual content and an evidence pair — the
  detailed answer; `isX(a, b)` symbolic, read from LTM and the taxonomy by
  references, leaving the pair and the witnessing rows — the hard answer.

## 4. Second rulings (Alec, 2026-10-07)

- **`query` rename now** (`what` → `query`; the recursive subgoal kept).
  Text in §5. Union vs best row still open.
- **Thinking must work soon**: Alec suggests "now or 5.5". Claude proposes
  its own item **6.2**, after 6.5 and before 6 (forgetting is the hard
  dependency; 5.5 is already large); spec written during 6.5.
- **Bind can be implicit** — decided (§3).
- **`isX : X :: conjunction : intersection`** — symbolic to conceptual, the
  duality the catalogue already has for the connectives (`conjunction`/
  `disjunction` over symbols, `intersection`/`union` over vectors).
  `isTrue` (symbolic: the row by reference, its pair and trust) and `exist`
  (conceptual: presence in conceptual space) are the same pair.
- **LTM relation kinds**: part, implies (if/then and temporal chaining),
  equals (definition), and the varying fourth, **Operator** — an arbitrary
  VP with a reference that needs expansion. Nothing missing: attitudes,
  comparison, possession are Operator rows; cause is implies with a `.when`
  order; is-a is part over extents; questions/estimates are record kinds.
  Keep explicit: `equals` is definitional; identity between occurrences of
  one individual is a reference (binding), not a row.

## 5. The `query` rename (Claude, for Codex, 2026-10-07)

Part of the next Codex text (6.5's head). *Corrected by §6:* `query` is the
LTM lookup, returning the best match; the subgoal push `what(Q)` becomes
`ask(Q)`. The rename `what` → `ask` applies throughout:
the thought executor's semantic id and domain, the `<thought>` sections of
every `.grammar`, `Grammar.thought_operations`, the model's public
`what()` API (kept as a deprecated alias for one release, raising with the
new name), tests, QueryContracts, AccessibleMind.md, Reasoning.md, the
catalogue (§3.6 marked done). Behaviour unchanged: the best-matching frame,
else a subgoal in the same controller (`continues`), pending Alec's ruling
on union vs best row. No measurement beyond the sweep.

## 6. Two operations, not one (Alec, 2026-10-07)

Alec: a `query` for a match over LTM (**return the best match**), and the
question that creates an answer used for thinking — pushed onto a stack.

Both exist, separated on 2026-09-15 (production plan): `query(X, Y)` the
lookup, kept distinct; `what(Q)` "the general conceptual-subgoal request",
preserving the question's deep structure through the boundary loop. The
push is specified (ThoughtHistory.md, 2026-09-20): a `descend` record on the
thought history (the episode's stack, replayed from `begin`/`thought`/
`descend`/`return`/`cutoff`/`finish` records, no second planner stack), one
shared work budget across levels, the drain at cutoff popping at most the
active depth of returns plus one finish; a completed interrogative sentence
is adapted into its meaning and `run_selected_thought` opens the root
episode. What went wrong is the name: the code registered the push as
`what`, and the catalogue's "`what`/`lookup` → `query`" would have merged
the two. Correction to §5: **`query` is the lookup (best match)**; the push
is renamed **`ask(Q)`** (Claude's proposal), with `descend`/`return` as its
records. `ask` as an operation is also the fix for finding 1 (thought runs
only on questions): a declarative closing can ask.

Inventory, as it stands: `query`, `ask`, `exist`/`isTrue`, `part`/`isPart`,
`equal`/`isEqual`, `implies`/`isImplied`, `not`, the gain act; `expect` a
setting; `quantize` future work.

## 7. A question is an open reference (Alec, 2026-10-07)

Alec: `ask()` should use a forward reference that then needs to be filled in;
the mechanism must also be implicit — "what is the capital of England?" is a
forward reference; an unresolved reference like *what* must be bound, can be
bound to anything that exists, so it gets bound to nothing; **that is the
marker of a question, not any stack manipulation.**

Consequences, replacing §6's "push":
- A **question is a row with an open reference** — a reference the binder
  could not resolve because nothing existing matches. Interrogative mode is
  derivable from the row, not a separate flag (the surface `?` and the
  wh-word are evidence for the open reference, not its definition).
- **`ask`** is the attempt to fill an open reference: by `query` over LTM
  (best match for the row with the slot open), by inference, or by later
  text. The answer *is* the binding. A nested question is a further open
  reference created while filling the first; they resolve in dependency
  order — the stack is the set of open references in working memory. The
  history's `descend`/`return` records remain the trace, not the marker.
- **Implicit questions.** Since binding is implicit (§4), any closing with a
  reference nothing matches opens one: cataphora when later text fills it,
  a question when thought must; at document end an unfilled reference is
  stored as a question row. This is the entry condition for thought at
  every closing, declarative or interrogative (finding 1 closed).
- A forward reference is an expectation of a referent ("identity is an
  expectation", Philosophy); its filling is credited like any choice by
  what follows.
- Existing machinery: open roles and null references in the store, the
  `question`/`estimate` record kinds, the open-role variants in the thought
  grammar; the determiner's `bind` mode with no antecedent is the explicit
  case. What changes: these become the definition of asking, and thought's
  operations bind to the open references rather than to a pushed meaning.

## 8. The renames (Alec, 2026-10-07: "sooner rather than later"), for Codex

Folded into item 6.2 as its first part (thinking spec §2.1). Each rename keeps the old
name for one release as an alias that raises with the new name; documents,
tests, grammars and the catalogue follow.

| old | new | note |
| --- | --- | --- |
| `what(Q)` (thought) | `ask(Q)` | the attempt to fill a row's open references (§7); no push |
| `what`/`lookup` (LTM) | `query` | the best match (Alec) |
| `chunk` | `synthesize`; its reverse `analyze` | catalogue §3.1, decided |
| `quantize` | — | retired to future work (all concepts are symbolized); `symbolize`/`conceptualize` remain future names |
| `arma`/`expect` (thought) | — | a global setting (`<sentenceExpectation>`, gain `g`); the image reaches the chooser as context |
| `true`, `exist` (scalar) | `isTrue(P)` / `exist(P)` | the pair, never a scalar; symbolic / conceptual faces (6.2 §2) |
| `selectedThoughtBudget`, `thinkingBudget` | `attentionBudget` | one budget name; the others rejected at load |

The `.grammar` `<thought>` sections list `ask`, `query`, `part`, `whole`,
`equal`, `not`; `arma` and `quantize` are removed from them. The thinking
spec (6.2) then adds `isTrue`/`exist`, `isPart`/`part`, `isEqual`/`equal`,
`isImplied`/`implies` as the two faces, and the gain act.
