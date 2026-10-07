# The credit loop in a toy (Claude, 2026-10-06)

Written after round 2d, before specifying round 2e (operators update plan
§21–§22), to test the reader rules against the loop's fixed point instead of
against the next campaign. Two operators; conjunction roots are four
independent directions (XOR-separable by an affine reader), disjunction roots
are sums of word vectors (XOR affinely inseparable, floor ¼); an affine reader
with Adam; a one-logit chooser credited by the score-function term with the
answer difference as advantage; a walk-first draw (half the sentences get a
compose departure, half tie); reconstruction ties so the keep is greedy;
400 epochs. `sim_independent_roots.py`: 20 seeds, disjunction roots
independent of conjunction's. `sim_correlated_roots.py`: 40 seeds, disjunction
roots correlated with conjunction's (closer to the kernels' geometry).

| Reader rule | Independent roots: flips / at 0 (20) | Correlated roots: flips / at 0 (40) |
| --- | ---: | ---: |
| Kept root only (round 2) | 12 / 12 — never from a disjunction start | — |
| Both roots ½/½ (2c, 2d) | 20 / 20, compromise .001–.028 | 20 / 19 (of 20), compromise to .071 |
| Policy-weighted (plan §22 as first drafted) | 20 / 20 | **29 / 30**: in 10 seeds the chooser commits to disjunction before the reader learns conjunction roots, the explore weight vanishes, no flip |
| Judge on both roots, presented reader on the kept root | 20 / 20 | **40 / 38**; the two misses flipped at epochs 343 and 390 |

The toy understates the real reader's compromise (2d: ½ weight gave 2/10 at
zero) and says nothing about counts; it ranks the rules by their fixed
points. The two-reader rule is round 2e (plan §22, rewritten).
