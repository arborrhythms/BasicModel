# §6.8 bootstrap and the §7 gates

These are mechanism checks on the uncommitted step-1b candidate. No
learning result is regraded. The original failed runs remain in this
directory; a later passing case does not turn an earlier failed run green.

## 1. What supplies word membership now?

| Question | Answer |
|---|---|
| Durable source | The ordered pairs of the common Part store |
| An absolute close | Its existing semantic reference names the surviving Part root; LTM1 retains the narrowed concept |
| A complete relative close | The native Part root supplies its context |
| Leaves | Native lexical seats reached through the derivation; a native perceptual witness need not have a textual spelling |
| Inverse posting index | Its production numerical-unfold path is retired; postings rebuild from references and Part |
| Checkpoint cache | Rebuilt after the native owner and semantic extras load |
| Cycles | Visited once; reachable terminal leaves remain readable |
| Exact/four-meaning gate | Passes in `native-47`, `native-48` and `native-70`, with the original exactness and four-distinct-meanings assertions |

`test_part_read` in [native-50](native-50/report.html) changes the narrowed
slot without changing the membership read, poisons a serialized posting
cache and rebuilds it from Part, and checks termination on a cycle.
No working identity is reconstructed or written back into an existing
concept row. The LTM1 column cull and context-whole migration remain step 2.

## 2. Why did the paired reconstruction pay 0.1, and does it now tie?

The old LTM1-only read credited one word of a two-word sentence. One leaf
therefore had a nonzero context code and the other had no code. Min erased
the lone code; max retained it. The two legal walks consequently paid
different **content** reconstruction errors, despite identical byte
reconstruction and decoded lengths. The content error of 1 had weight
0.1. This was not a bootstrap-snapshot mismatch or an optimizer change.

| Run / initialization | Batch | Greedy R, four rows | Explore R, four rows | Result |
|---|---:|---|---|---|
| `native-43`, saved failure | 2 | 0.1, 0.1, 0, 0.1 | 0, 0, 0.1, 0 | Failed |
| `native-47`, exact replay of 43 | 1–3 | 0, 0, 0, 0 | 0, 0, 0, 0 | Paired case passes |
| `native-48`, unchanged random certificate | 1–3 | 0, 0, 0, 0 | 0, 0, 0, 0 | Paired case passes |
| `native-50`, replay after index and STM close repairs | 1–3 | 0, 0, 0, 0 | 0, 0, 0, 0 | Paired case passes |
| `native-61`, exact replay of 43 after Part-only diffusion | 1–3 | 0, 0, 0, 0 | 0, 0, 0, 0 | Paired case passes |
| `native-70`, unchanged random certificate | 1–3 | 0, 0, 0, 0 | 0, 0, 0, 0 | Paired case passes |

The [metrics](bootstrap-metrics.json) preserve each batch, both trials,
and expected/decoded lengths. The original initialization is
`regression-diagnostic-43/test_both_training_trials_share_bootstrap_snapshot-initial-rng.pt`.
The test's three epochs, one batch per epoch and exact-zero assertion
are unchanged. No passing seed was selected.

## 3. Which other §7 gates have current evidence?

| Gate | Current evidence |
|---|---|
| G1–G3 | All 32 generated cases pass in `native-70`; all 43 bind-thought cases pass |
| Reference close and mention supervision | Both files pass in `native-70`; the existing reference assertions remain |
| Cube and inverse contracts | 12 cube cases and 35 lift/lower cases pass in `native-70` |
| Native §14 initialization read | Native-70 completes 32 documents, 5,372 menus, 3,864 retained candidates; chooser-content ratio 1.0–1.0000000298023224 against unchanged [0.9, 1.1]. Form and expectation terms are held out under parity |
| Reverse-order premise resolution | Both orders pass in `native-70` at the saved `pending-32` initializations; ordinary copular interpretation is preserved |
| STM1 close | The live working STM dissolves after close; the ended field remains an output record. `test_clause_scope` passes |
| IntraSentenceLayer | No production/config occurrence; retired-name audit passes in `native-70` |
| Copular artifact | Still prerequisite-skipped |
| Original decomposition | Attempt 71: both lengths complete in both arms, all four failed; the missing type source is recorded in bind plan §6.9 |

The initialization report is retained under
[`native-70-reports/.../distribution.json`](native-70-reports/test_ordinary_initial_binding_distribution_includes_every_retained_candidate/distribution.json).
The premise stores are in `pending-70`; those runs replay the saved
initializations, not a seed chosen to satisfy the assertions.

## 4. What remains unresolved?

| Finding | Status |
|---|---|
| Expression read | §6.8's rule is decided. The current symbol data lacks the verb classification assumed for symbol 48; §6.9 asks which current symbol owner supplies it |
| Empty-search c/e certificate | Saved failure 61 is reproduced in 63 and repaired at the closing read; passes in native-70. Historical native-50 has no saved initialization and is not claimed as an exact replay |
| Whole §14 file | `native-70`: 12 passed, one copular prerequisite skip |
| Full landing sweep | Not run in this round |
| Publication | Not performed; both mains still hold step 1 |

The [closing cause record](closing-read-causes.md) distinguishes the
concrete definition/equality collision, the saved self-witness
counterexample, the rejected global guard and the closing-only repair.
The [current carry table](status.md) preserves artifact retirement,
step-2 row cull/recurrence, Item 3 priming strata/geometry, step-4 verb
groups, step-5 second difference, Item 0 and separate 5.5 learning gates.
