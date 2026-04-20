# Learned SVO Role Identification from the Grammar

This document specifies what it would take to drive `TruthLayer.universality`
from grammar-derived Subject/Verb/Object roles rather than the positional
heuristic currently in use.

`TruthLayer.universality(subject, verb, object, lifting_layer, symbolic_space)`
implements the Golden Rule: it scores luminosity change under S/O reversal,
so `K(X,Y) + K(Y,X)` illuminates more than `K(X,Y)` alone for kind actions.
The method takes `(S, V, O)` concept tensors as inputs; the question is where
those three tensors come from.

## Current state

As of this pass, SVO is produced by a positional tap inside
`SyntacticLayer._compose_vector` in `bin/Language.py`: the first three active
leaf positions of a batch row are labelled subject/verb/object when every
row has at least three active leaves. For the canonical three-token
transitive corpus in `test/test_universality.py`
(`"the teacher helped the student"` without determiners) this produces
the right roles. It is wrong for realistic inputs: determiners shift the
roles, adjectives insert modifiers, word-order variation breaks it entirely,
and embedded clauses are invisible.

The plumbing downstream of `last_svo` is already correct for a learned tap
to drop into:

- `SyntacticLayer.last_svo: Optional[Tuple[Tensor, Tensor, Tensor]]`
- `SyntacticLayer.lifting_layer: LiftingLayer` (instantiated in `init_lifting`,
  called from `WordSpace._build_syntactic_layer`)
- `MentalModel.forward` reads both, calls `truth_layer.universality(...)`,
  and exposes the result via `self._universality_score`
- `truth_modulated_loss(universality_score=...)` folds the score into the
  training loss

Only the producer of `last_svo` needs to change.

## Target grammar shape

The binary grammar should be able to express:

    S  -> S VO
    VO -> V O
    V  -> v(V)      -- terminal category verb
    O  -> n(O)      -- terminal category noun

so that a parse of a three-content-word sentence produces a derivation tree
whose outer rule is `S -> S VO`, and whose `VO` child comes from `VO -> V O`.
The subject vector is arg-0 of the outer rule; the verb vector is arg-0 of
the inner rule; the object vector is arg-1 of the inner rule.

Longer inputs extend naturally:

    S   -> NP VP
    NP  -> DET N | N
    VP  -> V NP
    NP' -> DET N | ADJ N | DET ADJ N

and an SVO walker reads (subject = arg-0 of S, verb = head of VP, object =
head of VP's NP).

## Phase A - Grammar metadata (low risk, ~60 LoC)

### A.1 Extend `RuleDef`

```python
RuleDef = namedtuple(
    'RuleDef',
    ['tier', 'canonical', 'arity', 'method_name', 'lhs', 'rhs_symbols'],
)
```

- `lhs`: nonterminal name this rule reduces to (`'S'`, `'VO'`, `'NP'`, ...).
- `rhs_symbols`: tuple of nonterminal / terminal category names for each
  RHS slot (e.g. `('V', 'O')` for `VO -> V O`).

Existing rules default `lhs='S'`, `rhs_symbols=('S', ...)` for backwards
compatibility with the current single-tier S grammar.

### A.2 Multi-LHS `configure()`

`Grammar.configure()` hardcodes `for lhs in ('S',)` (see `bin/Language.py`
around the `configure` method). Replace with iteration over every key in
the XML `<grammar>` block, so each key becomes a distinct nonterminal
with its own rule list.

### A.3 RHS parser changes

`_parse_rule` currently parses function-call syntax (`lift(S, S)`,
`not(S)`). Add a bare-symbol-sequence form for typed rules:

    <S>S VO</S>           -- rhs_symbols=('S', 'VO'), method='merge'
    <VO>V O</VO>          -- rhs_symbols=('V', 'O'), method='merge'

Branch on whether the RHS contains `(` / `)`. The function-call form stays
for existing ops; the bare-symbol form is the new typed form.

### A.4 XML/XSD schema

`MentalModel.xml` and its XSD must allow non-S nonterminals as child
elements of `<grammar>`. Add each new nonterminal tag as an optional
element in the XSD. No migration needed for existing configs because
`<S>...</S>` stays valid.

## Phase B - Compose restructure (high risk, ~120 LoC)

### B.1 Problem: left-associative cascade can't produce `(S, (V, O))`

`_compose_vector` in `bin/Language.py` is a strict left-associative cascade:

    composed = leaf[0]
    for d:
        composed = rule_d(composed, leaf[d+1])

This bracketing is `(((leaf0, leaf1), leaf2), ...)`. To realise `S -> S VO`
where `VO` is itself `V O`, we need `(leaf1, leaf2)` to merge first into VO,
then `(leaf0, VO)` into S. Left-associative cascade cannot express this.
`_compose_to_target` has the same limitation - it pairs (0,1), (2,3),
etc. leftmost-first.

### B.2 Chart-like pair selection

Replace the cascade with per-step pair selection:

- At each composition step the rule head picks, differentiably, WHICH two
  adjacent leaves to merge plus WHICH rule to apply.
- A pair-scorer MLP takes the current hidden state plus a pair context
  `(left, right)` and outputs a softmax over `(N-1)` adjacent pairs.
- The selected pair is merged; the resulting slot replaces the pair in
  the live-leaf array. Total steps: `N-1`.
- Maintain a per-batch live-leaf count and an alive-mask `[B, N_max]`.

### B.3 Risks

- O(N^2) per depth step at train time (soft pair mixture over pairs x
  rules) vs O(1) in the current code. Acceptable for short inputs
  (N <= 16); may need chunking for longer.
- Hard argmax at eval, soft at train; convergence needs care.
- `decompose()` needs a matching inverse that reads the pair-selection trace.
- `subspace.add_word()` currently records a single position per rule firing.
  Pair-selection records `(pos_left, pos_right)` and a merged-slot position;
  `WordEncoding.encode(...)` already has `leaf1/leaf2/leaf3` fields for this.

### B.4 Why a cheaper hybrid doesn't work

One might keep the left-assoc cascade and let the rule head pick rules
whose LHS is only reachable after the inner merge has happened. For `N V N`:

- Step 1 wants `V N -> VO` (inner first).
- Step 2 wants `N VO -> S` (outer second).

The left-assoc cascade forces step 1 to merge `leaf[0]` with `leaf[1]`
(i.e. `N V` first), which is `N V` - not a rule in the target grammar.
No rule fires, compose stalls. Left-assoc is unsalvageable for standard
SVO. Phase B is required.

## Phase C - Category tensor alongside activation (~60 LoC)

### C.1 Per-slot category

Augment the data tensor `[B, N, D]` with a category tensor
`category: [B, N]` where `category[b, n]` is the nonterminal / terminal
category ID at that slot (or `-1` for padding). This rides alongside
activation through compose.

### C.2 Initialisation from POS

Two options for seeding category at input time:

- **(a) External tagger.** Run nltk POS tagging (already in
  `requirements.txt`) on input text at data load, map tags to our
  lexical categories (`N`, `V`, `ADJ`, `ADV`, `DET`, ...), store
  category indices alongside token indices. Clean; requires plumbing
  through `InputSpace`.
- **(b) Codebook-derived POS.** K-means on the `WordVectors` codebook to
  produce K lexical clusters; at runtime, look up each leaf's cluster as
  its category. Cheaper, co-trained. Quality depends on embedding geometry.

(a) is the clearer path, and nltk is already a dependency.

### C.3 Rule-compatibility masking

The rule head emits `rule_probs: [B, max_depth, num_rules]`. Extend to a
compatibility mask:

- For each candidate rule `r`, its `rhs_symbols` specifies the nonterminals
  the RHS must hold.
- At each merge step, compute `compat[b, r] = 1` iff the categories of the
  two candidate children match `r.rhs_symbols`, else `0`.
- Multiply into `rule_probs`, then renormalise. Incompatible rules drop out.
- On merge, set the merged slot's category to `r.lhs`.

### C.4 Gradient / training dynamics

A hard compat mask kills gradient through incompatible rules - semantically
correct (those rules cannot fire) but brittle early in training if category
assignment is wrong. Mitigations:

- Small epsilon on incompatible rules early; anneal to zero over training.
- Gumbel-softmax on the merged-slot category assignment to keep gradients
  flowing through category propagation; anneal temperature high-to-low.
- A designated always-compatible fallback rule (e.g. `chunk`) so compose
  never stalls on unknown pairs.

## Phase D - SVO extraction hook (~30 LoC)

Once C is working, SVO falls out of the derivation trace.

### D.1 Record the trace

In `_compose_vector`, each rule firing produces a tuple
`(rule_id, left_slot, right_slot, merged_slot)`. Accumulate per-batch into
`self._derivation_trace: List[List[Tuple[...]]]` over depth.

### D.2 Walk the trace to SVO

After compose completes, for each batch row:

1. Find the outermost `S -> S VO` firing (argmax rule at the final step,
   constrained to `S`-LHS rules).
2. Read its left arg as subject (slot pointing to an `S`-category leaf or
   subtree - the canonical S subject is a noun phrase whose head is the
   subject noun).
3. Find the `VO -> V O` firing that produced the right arg of step 1.
4. Read its left arg as verb, its right arg as object.

Each of S/V/O is the concept vector at the respective slot in `data`
(or the aggregate of the subtree slots if we want phrase heads rather than
single-word heads).

### D.3 Batch-level variability

Per-batch traces are natural (rule firings are already per-batch). SVO
extraction runs per batch and produces a per-batch mask of "has SVO"; rows
without the canonical shape leave their `last_svo` entry as None. The
`MentalModel` hook already handles a global `None`; extending to a batch
mask means `truth_layer.universality(s, v, o, ..., mask=...)` filters out
rows without valid SVO before computing luminosity deltas.

## Phase E - Training dynamics

### E.1 Soft-mixture / hard-gate interaction

Phase B's soft pair mixture and Phase C's hard category gate interact:
- Rule output is a soft mixture over compatible rules.
- Merge output is a soft mixture over pairs.
- Merged-slot category is a categorical argmax of rule LHS (hard).

If hard category assignment is wrong, subsequent compat masks are wrong.
Gumbel-softmax on the category assignment lets gradients backpropagate
through the "wrong" choice and correct it.

### E.2 Curriculum

- Start with short canonical inputs (3-content-word SVO, matching the
  test corpus) where the correct parse is unambiguous.
- Extend to 5-token `DET N V DET N` once short inputs produce stable SVO.
- Add modifiers (`ADJ`, `ADV`) and embedded clauses last.

### E.3 Universality loss weighting

`architecture.UniversalityWeight` currently defaults to 0.1. Once SVO is
learned rather than heuristic, raise toward 0.3-0.5 - the Golden Rule
signal is only stable when SVO extraction is reliable, and only then
should it dominate loss.

## Phase F - Test suite

### F.1 New unit tests

- `test_ruledef_typed.py`: `RuleDef` stores `lhs` / `rhs_symbols`; grammar
  exposes them; XML round-trip preserves them.
- `test_compose_chart.py`: chart-like compose reduces `N` leaves to `1` in
  `N-1` steps, preserves tensor dims, handles alive-mask.
- `test_category_propagation.py`: merging `(N, V)` under `VP -> V N` yields
  a slot with category `VP`; incompatible pairs zero the rule prob.

### F.2 Update `test/test_universality.py`

`_get_svo_and_luminosity` currently returns `(None, None)` with a comment
noting that the C-tier ternary-lift path is gone. Once Phase D is in place,
replace it with a derivation-trace walk that reads SVO from the model and
passes through `truth_layer.universality(...)`. The `xfail` markers stay:
untrained models will not score `kind > unkind`.

## Cost estimate

| Phase | LoC | Risk   | Depends on |
|-------|-----|--------|------------|
| A: Grammar metadata        | ~60  | low    | -          |
| B: Chart-like compose      | ~120 | **high** | A        |
| C: Category tensor         | ~60  | medium | A, B       |
| D: SVO hook                | ~30  | low    | C          |
| E: Training dynamics       | tuning | medium | D        |
| F: Tests                   | ~80  | low    | A, B, C, D |
| **Total**                  | ~350 |        |            |

Phase B is the load-bearing risk. Allow at least a week for the compose
restructure; everything else layers cleanly once B is stable.

## Interim: positional tap (what ships today)

Until the learned approach lands, `SyntacticLayer._compose_vector` uses a
positional heuristic: any batch row with at least three active leaves
labels positions 0/1/2 as subject/verb/object. This fires for the
`test_universality.py` canonical corpus and silently skips longer inputs
(`last_svo` stays `None`, universality stays `None`, the loss term
contributes zero).

Known limits of the positional tap:

- Wrong SVO for five-token sentences with determiners (picks the determiner
  as "subject").
- No role fidelity for non-canonical word order.
- No support for embedded clauses (only the top-level SVO is visible).

All three are addressed by Phases A-D.

## Appendix - hooks that Phase D builds on (already wired)

- `SyntacticLayer.last_svo: Optional[Tuple[Tensor, Tensor, Tensor]]`
  (set in `compose`, populated in `_compose_vector`).
- `SyntacticLayer.lifting_layer: LiftingLayer` (created in `init_lifting`,
  called from `WordSpace._build_syntactic_layer`).
- `MentalModel.forward` reads both, invokes
  `truth_layer.universality(s, v, o, lifting_layer, symbolicSpace)`,
  and stores `self._universality_score`.
- `truth_modulated_loss(universality_score=..., universality_weight=...)`
  integrates the score into the training loss.

These four interfaces stay stable when learned SVO replaces positional SVO.
The only change is how `last_svo` is produced.
