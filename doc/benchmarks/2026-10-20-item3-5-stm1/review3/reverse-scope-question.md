# Reverse-order certificate: the unresolved scope question

The unchanged certificate fails at line 398 on reviewed close `6903713c7`.
`arrival-10` captures it; `arrival-14` restores its process RNGs and captured
source manifest. Initialized parameters/buffers, final parameters/RNG,
all LTM1 rows (including native addresses), occupations and staged sources
match exactly. The existing object-ID-set iteration permutes a few DEF
handles; their pairs and evidence are identical. The failure is identical.

| Question | Native evidence |
|---|---|
| Which row fails? | Row 5, stream 2, unverified constituent; not the pending `x` occurrence |
| Its unresolved reference | Role 2, word symbol 14 (`is`), `_forward_references = ((2, 14),)` |
| Its other endpoint | `('sym', 35)`; the live explore suffix produced this constituent |
| The actual pending `x` row | Row 6, stream 2, question; raw role 0 names word 12 (`x`) and complement is the occurrence of row 5 |
| What the certificate reads | Every non-estimate row, asserting no forward references of any name |
| What its forward arm already states | A live departure may open another name; arrival of `x` must not fill an unrelated variable |
| Architectural constraint | An unrelated unresolved reference cannot be silently filled, removed or retyped merely to satisfy a global assertion |

Proposed certificate correction for review: identify the actual pending
`x` occurrence(s) from their raw forward-reference name, then read them
through `meaning_of` and keep the **same** no-forward-reference assertion.
Preserve the live explore suffix and the independent bound-correct count.
This changes the scope of the assertion from the whole store to the
specified reference. It is **not applied**: Alec explicitly required the
assertions and forced scope to remain fixed, so that scope decision needs
confirmation rather than an unannounced narrowing.

Separate repair under way: the forward answer reader flattened a head
chain past composite symbol 24, even though that symbol had a retained
definition for the supplied target. `arrival-12` → `arrival-13` reproduces
that line-391 failure with identical LTM1 rows and initial/final parameters;
`arrival-15/failure-expression.json` shows the skipped intermediate identity.
The reader must test each head identity before descending further.
