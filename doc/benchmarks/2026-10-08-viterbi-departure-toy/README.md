# Viterbi-like principle for one departure credited at the end (2026-10-08)

Item 4.5, [spec §5.3](../../specs/2026-10-08-expectation-at-every-level.md).
Alec asked to verify that "the last step is optimal, and if we build a test
set that only requires one more layer at a time, we can escape local minima".

`toy.py`: a tabular chooser over a depth-D, branching-K derivation with a
terminal cost only (random in [0, 1] per leaf). Each episode: greedy path;
one departure, *where* uniform over the remaining rounds, *which* a mix
(`phi` uniform over the non-greedy alternatives, else the policy with greedy
excluded); greedy completion; score-function credit at the departure with
`C_greedy − C_explore`. Same episode budget per regime:

- **root**: every item is the full derivation from the root.
- **graded**: "one more layer at a time" — stage k starts at random states
  k rounds from the end (the last round alone, then the last two, …), the
  final stage from the root.
- **mixed**: the same subproblems, start depth drawn uniformly, unordered.

Result (200 random instances each; `results.txt`): fraction whose final
greedy path is optimal.

| D, K, episodes, phi | root | graded | mixed |
|---|---|---|---|
| 5, 3, 6000, 0.3 | 0.04 | **0.96** | 0.32 |
| 4, 3, 3000, 0.3 | 0.12 | **0.95** | 0.53 |
| 5, 3, 6000, 0.0 | 0.07 | **0.95** | 0.36 |
| 5, 3, 20000, 0.3 | 0.04 | **0.96** | 0.43 |

**Reading.** The credit of one departure plus greedy completion is exact
(Bellman's `Q(s, a) = V*(s')` read at the end) exactly when the greedy
completion from the departed state is already optimal. The last round always
satisfies that (its completion is empty), so it is learned first; graded data
then makes every new round's completion optimal before that round is
trained — backward induction carried by the data order, Viterbi's principle
of optimality. Root-only does not improve with 3.3× the budget (0.04 → 0.04):
completions off the greedy path are never learned, so early departures are
credited against unlearned suffixes and the policy settles in a local
minimum. Unordered subproblems help (low regret) but do not reach the
optimum reliably: the order matters, not only the content. `phi` matters
little here because the tabular policy starts near uniform.

**Limits.** Tabular: no generalization across states (the model's chooser
shares parameters, which can help or hurt completions it never visited).
Terminal cost only, Markov state. One seed family; no seed is pinned to pass.

## Follow-up: does a tunable softmax replace the departure? (Alec, 2026-10-08)

`softmax_toy.py`, `softmax_lr.py`, `softmax_results.txt`. Every round sampled
from `softmax(logits/τ)`, one derivation per episode, score-function credit
at every round against a running per-state value baseline; **twice** the
departure's episodes (a departure trial runs two derivations).

| rule (graded order) | optimal |
|---|---|
| one departure, φ = 0.3 (100 instances) | **0.98** |
| softmax, best of τ ∈ {1, 0.3, 0.1} × three learning rates (50 instances) | 0.54–0.66 |
| softmax, root-only, any τ | ≤ 0.12 |

The first sweep's collapse at τ ≤ 0.1 (0.01, 0.00) was a learning-rate
artefact (the gradient scales as 1/τ); tuned, every τ plateaus near 0.6.
Why the departure wins: its credit is a *paired* comparison under the same
input (greedy as the baseline, no estimated value), at *one* choice, with an
*optimal* (greedy) finish — the Viterbi condition. The sampled softmax
credits every round from one shared cost (the confound grows with depth) and
measures the value of its own noisy finish, not the optimal one.
