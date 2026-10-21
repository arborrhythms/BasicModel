# Frozen-source fresh arrival gate: stopped

Source `eac2fe87b2e2f0d24ee16fe8250dec37e59abbc7`. Six fresh draws were declared per order.
Every completed draw retains its full first-run initialization and
both trial captures from the outset. No draw is replaced.

| Order / draw | Result | Correct rows / roots | Full paired readings |
|---|---|---|---|
| forward / 1 | passed | 8/8 | [fresh-final-forward-1](fresh-final-forward-1-tables.md) |
| forward / 2 | passed | 8/8 | [fresh-final-forward-2](fresh-final-forward-2-tables.md) |
| forward / 3 | passed | 8/8 | [fresh-final-forward-3](fresh-final-forward-3-tables.md) |
| forward / 4 | passed | 8/8 | [fresh-final-forward-4](fresh-final-forward-4-tables.md) |
| forward / 5 | passed | 8/8 | [fresh-final-forward-5](fresh-final-forward-5-tables.md) |
| forward / 6 | passed | 8/8 | [fresh-final-forward-6](fresh-final-forward-6-tables.md) |
| reverse / 1 | passed | 8/8 | [fresh-final-reverse-1](fresh-final-reverse-1-tables.md) |
| reverse / 2 | failed | failure; no success certificate | [fresh-final-reverse-2](fresh-final-reverse-2-tables.md) |
| reverse / 3 | passed | 8/8 | [fresh-final-reverse-3](fresh-final-reverse-3-tables.md) |
| reverse / 4 | passed | 8/8 | [fresh-final-reverse-4](fresh-final-reverse-4-tables.md) |
| reverse / 5 | failed | failure; no success certificate | [fresh-final-reverse-5](fresh-final-reverse-5-tables.md) |
| reverse / 6 | not run: stopped | — | — |

**11/12 completed: 9 passed, 2 failed; 1 not run.** All 11 completed initializations have distinct parameter fingerprints.

Reverse draws [2](reverse-2-failure.md) and [5](reverse-5-failure.md)
were exactly replayed before any production or assertion change.
Their failing replays are diagnostics, not additional fresh draws.
The gate stops for Alec’s contract/mechanism decision after draw 5.
The eight-file set, both decomposition arms and both eight-seat MM
gates are unrun on this source. No earlier-source result is promoted.
