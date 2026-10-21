# Twelve first draws on the third frozen source

Source `3d453e6bb6927e5da1492c25106a4ef107b87bca`. Twelve distinct initial-parameter fingerprints.
Six first draws per order; every initialization, generator state, both trials,
R/E/A and keep decision captured from the outset. No failed draw was replaced.
Assertions, forcing scope, zero budget, objective and keep law unchanged.

| Draw | Result | Full keep tables |
|---|---|---|
| fresh-final-forward-1 | passed | [both batches](fresh-final-forward-1-tables.md) |
| fresh-final-forward-2 | passed | [both batches](fresh-final-forward-2-tables.md) |
| fresh-final-forward-3 | passed | [both batches](fresh-final-forward-3-tables.md) |
| fresh-final-forward-4 | passed | [both batches](fresh-final-forward-4-tables.md) |
| fresh-final-forward-5 | passed | [both batches](fresh-final-forward-5-tables.md) |
| fresh-final-forward-6 | passed | [both batches](fresh-final-forward-6-tables.md) |
| fresh-final-reverse-1 | passed | [both batches](fresh-final-reverse-1-tables.md) |
| fresh-final-reverse-2 | failed | [both batches](fresh-final-reverse-2-tables.md) |
| fresh-final-reverse-3 | failed | [both batches](fresh-final-reverse-3-tables.md) |
| fresh-final-reverse-4 | passed | [both batches](fresh-final-reverse-4-tables.md) |
| fresh-final-reverse-5 | passed | [both batches](fresh-final-reverse-5-tables.md) |
| fresh-final-reverse-6 | passed | [both batches](fresh-final-reverse-6-tables.md) |

**10 passed, 2 failed.** All six forward draws pass. Reverse draws 2 and 3
fail unchanged D2 and exactly replay; [draw 2](reverse-2-failure.md) and
[draw 3](reverse-3-failure.md) retain the diagnoses. A kept first reading
has no pending x site and its order-0 three complement is absent from
the next frame’s bind menu. The failure is not relabeled or waived.

The eight-file set’s additional captured reverse failure is separate;
it is not one of these twelve predeclared initializations.
