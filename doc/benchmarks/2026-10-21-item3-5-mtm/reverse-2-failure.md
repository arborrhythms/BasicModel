# Frozen-source reverse failure: conflicting retained equalities

Source: `eac2fe87b2e2f0d24ee16fe8250dec37e59abbc7`. The second of the
six declared fresh reverse initializations fails the unchanged assertion
in `test_forced_ordinary_pending_premise_in_both_orders[True]`, line 403:
no forward reference on a non-estimate row may still name x. This is a
failure of that semantic certificate, not a representation port. The six
fresh forward initializations passed their unchanged assertion.

The gate stopped at this failure for diagnosis. Before any source change,
the full first-draw capture was replayed with the same arrival capture and
keep probe. [Validation](reverse-2-replay-validation.json) confirms identical
initial parameters, buffers, all three saved RNG files, source manifest,
both trials' readings, costs, keep decisions, retained rows, Part store,
MTM and observed reads. Both runs fail at row 5 in document/stream 2.
The only serialization difference is the order of two native type sets
(`concept`, `object`); their members agree. The initial exact dictionary
comparison caught that difference; it was not ignored in the validation.

## The deciding readings

The first sentence is `x is three.`. Its trials tie at
R = 0.036067377775907516, A = 0; `tie:greedy` keeps the equality with
the pending x subject and the retained three complement. The complement
is row 4, address `8438255287335991528`; the equality is row 5, address
`1638543793312781037`. Before the second sentence, captured working reads
of row 5 fill x with that complement.

The second sentence is `y is x plus two.`. In document 2:

| Trial | R | E (excluded) | A | R + A | Keep |
|---|---:|---:|---:|---:|---|
| exploit / greedy | 0.036067377775907516 | 0.023622898384928703 | 0 | 0.036067377775907516 | discarded |
| explore | 0.032460641115903854 | 0.023622898384928703 | 0 | 0.032460641115903854 | kept, deciding `reconstruction` |

Greedy's child is `(-1, plus, two-child)` with a forward x subject.
Explore instead makes that child an equality with a forward x subject.
Its complement is another equality, containing a space/plus equality and
the two child. The full graphs, word IDs, open references, composition
trees and every document's two trials are in the
[first-draw keep capture](fresh-final-reverse-2-keep.jsonl) and
[16 document tables](reverse-2-keep-tables.md). Compose departure is 18.
No supplied answer target exists, and neither deciding trial has a
reference-closing request. Both batches leave MTM empty.

The kept explore graph writes a second x equality at row 31, address
`-7520361910242946377`, whose complement is row 30, address
`-5494138845512631668`. Thus row 5's later working read sees two distinct,
positively supported x complements in its own stream. The rule in
`ClauseRows.read_reference_bindings` (lines 680–684) deliberately leaves
multiple different candidates open. It neither drops the earlier three
row nor chooses a physical row as identity authority. The captured row 5
read consequently retains `[(0, 12)]` (x) and its open referent slot.

## Disposition

This is a surviving semantic failure on the frozen source. Its immediate
cause is the R-selected competing equality and the reader's ambiguity
rule. It is a reverse failure at the D2 name-scoped assertion, distinct
from the forward `matches(..., 'three')` failure discussed in plan §17.
No change to the assertion, forced scope, attention budget, objective,
keep rule or ambiguity rule is made. A contract/mechanism decision is
required before this gate can be closed or the source published.

After this diagnosis, predeclared reverse draws 3–5 continued on the same
source. Draws 3 and 4 passed; [draw 5 failed](reverse-5-failure.md) and
stopped the round at the arrival-reading decision boundary. This draw is
retained as a failure and is never replaced. The original failure and
exact failing replay remain under `fresh-final-reverse-2*` and
`reverse-2-replay*` respectively.
