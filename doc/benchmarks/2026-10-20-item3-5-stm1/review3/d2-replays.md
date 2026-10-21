# D2: reverse scope follows x across every non-estimate row

Source: `eb75de56ab62a04553553d62f8edb5c898d992cc`.
The forward assertion, forced scope, budget, graph count and departure
assertion are unchanged. Every raw pending-x row must also have no open
referent slot in its working read. Row 5 / word 14 (`is`) is unrelated
explore-suffix content; see closing-read-causes §6.2.

## Saved arrival-14

| Check | Result |
|---|---|
| Exact saved initialization replay | `d2-arrival-14-state`, full arrival capture |
| Unchanged x resolution and graph count | Passed; 6/8 roots carry three, required >= 2 |
| Native certificate | `d2-arrival-14/result.json`: 1 passed |

## Saved fresh-both-reverse-5

| Check | Result |
|---|---|
| Exact saved initialization replay | `d2-reverse-5-state`, full arrival capture |
| Unchanged x resolution and graph count | Passed; 7/8 roots carry three, required >= 2 |
| Native certificate | `d2-reverse-5/result.json`: 1 passed |

These two replays do not replace the twelve fresh checks on the later
§10.2 source. The prior failed runs are retained unchanged.
