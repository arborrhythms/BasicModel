# Reverse failures: answer and reference terms of A

Frozen source `3d453e6bb6927e5da1492c25106a4ef107b87bca`.
All premise targets are None. Answer cost is absent (zero). A equals
the captured reference credit, which is plan cost times .1. Per-site
terms below are reconstructed from captured features and probabilities
using the unchanged choice_cost, and checked against the captured sum.

| Draw / document / batch | Trial | Site costs | Arrival costs | Plan | Answer | Reference = A | R | Keep |
|---|---|---|---|---:|---:|---:|---:|---|
| fresh-final-reverse-2 / 2 / 1 | exploit | [1.0] | [3.0794415416798357] | 4.079441547 | 0 | 0.407944173 | 0.036067378 | explore |
| fresh-final-reverse-2 / 2 / 1 | explore | [] | [] | 0.000000000 | 0 | 0.000000000 | 0.036067378 | explore |
| fresh-final-reverse-2 / 2 / 2 | exploit | [1.0] | [1.0] | 2.000000000 | 0 | 0.200000003 | 0.036067378 | tie:greedy |
| fresh-final-reverse-2 / 2 / 2 | explore | [1.0] | [1.0] | 2.000000000 | 0 | 0.200000003 | 0.036067378 | tie:greedy |
| fresh-final-reverse-3 / 7 / 1 | exploit | [1.0] | [3.0794415416798357] | 4.079441547 | 0 | 0.407944173 | 0.036067378 | explore |
| fresh-final-reverse-3 / 7 / 1 | explore | [0] | [] | 0.000000000 | 0 | 0.000000000 | 0.036067378 | explore |
| fresh-final-reverse-3 / 7 / 2 | exploit | [1.0] | [1.0] | 2.000000000 | 0 | 0.200000003 | 0.036067378 | tie:greedy |
| fresh-final-reverse-3 / 7 / 2 | explore | [1.0] | [1.0] | 2.000000000 | 0 | 0.200000003 | 0.036067378 | tie:greedy |

On the first sentence the named path costs 1 for its first bind plus
3.079441542 for the arrival choice (1 + log(8)), yielding A=.407944173.
The kept field/mint path has neither charge and A=0. R ties. Thus the
earlier operation/keep decision removes the named site; the missing
three candidate in the next frame then prevents recovery. These are
two stages of the failure, not a lost answer target or an MTM eviction.

No price, objective, keep, scope or assertion is changed by this analysis.
