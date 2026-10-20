# Remaining one-successor certificate failure

The original comparison hit the obsolete three-result `select_logits`
wrapper. Forwarding its complete four-result tuple exposes two subsequent
issues: a resource defect, now repaired, and a functional certificate failure,
still open. Neither is classified as a learning success.

## Native execution and memory

The original compiled replay was interrupted after approximately 355 seconds;
its receipt is incomplete. The behavioral eager replay reached the 8 GiB
guard at the first question. `memory-probe.log` places this before any episode
snapshot. `allocation-stack-probe.log` shows memory growing from about 2 GiB
to 5.5 GiB within the initial `narrow_words` call. At the old work budgets
512/768, each round scored all reserved bracket slots and retained their
neural graphs even though only a few word brackets could occupy them.

`narrow_words` now declares the word-count bound to `attend`. Only that prefix
enters the neural scorer; the full masked menu is restored before selection.
The budget and table are unchanged. A native assertion forbids excluding a
legal action. Scoped tests compare the complete logits, selected actions,
probabilities, RNG state and parameter gradients; explicit compiled narrowing
also passes. The two certificates now complete in both gain configurations
below 2.2 GiB. The sparse audit repair is independently valid but was not the
cause of this native memory stop.

## Why the completed one-successor case still fails

Initialization is the original protocol's seed **4199722147**, derived from
fresh recorded entropy before either arm. No passing seed was selected.
`decomposition_keep_probe.py` replays precisely that initialization once,
observing the existing trial costs and field reads without changing choices,
parameters, budgets or assertions. `decomposition-keep/keep.jsonl` contains
all four batches. Constructor parameters compare exactly to the original run.
The diagnostic process exits normally; its questions still fail the proof
criterion, so its exit zero is not a certificate pass.

| Question row | Greedy R | Explore R | Greedy A | Explore A | Kept trial |
|---|---:|---:|---:|---:|---|
| 0 | .1839771569 | .1939771473 | 1.9219052792 | 1.0539722443 | Explore: 1.2479493618 < 2.1058824062 |
| 1 | .1839771569 | .2039771378 | 1.9219052792 | 1.1109412909 | Explore: 1.3149183989 < 2.1058824062 |

Row 0's candidate departure changes the word-read order from source positions
`[0,2,4,6]` to `[0,2,6,4]`. The question region consequently names a different
composed reference from the stored assignment. The fixture's checked request
selector finds no matching assignment; the resulting episode performs no
successor inference. Row 1 departs at compose round 49 and closes the open
reference, so it has no question episode. The immutable observer records zero
correct chains in the two rows. The all-level-enabled arm has the same failed
certificate; both arms pass the separate two-successor certificate. Compressed
native reports are in `decomposition-bounded-reports`.

The keep decision agrees with the decided R+A rule: A, not E, favors each
alternative. This is not a tie, bootstrap leakage or an expectation-only
regression. No change to the keep rule is proposed here. The forced fixture
currently constrains grammar operations and bindings while leaving the
native departure live; it cannot assume that one of two retained readings
will always remain its prescribed proof.

**Current-round blocker for review:** determine the conforming repair to this
forced mechanism certificate under thinking §14.1 and §14.11–14.13. Preserve
the live explore suffix, frozen observer, verifier and original assertion.
It must not be silently converted into a skip, probabilistic pass, new
training campaign, or carried learning gate. Gate 1 and landing remain open.
