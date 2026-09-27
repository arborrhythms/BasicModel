# Item 7.5 September 27 review corrections — superseded draft

Nothing is committed. The two closing sections of the
[specification](../../specs/2026-09-26-one-operation-per-round.md) supersede the
[September 26 implementation receipt](../2026-09-26-item7-5/README.md).

**Historical draft:** this receipt measures the earlier whole-batch winner
transaction. Alec subsequently confirmed per sentence and amended the spec:
each sentence trains both derivations and commits its row-local winners before
the next sentence begins. That implementation and its validation are in the
[sentence-seal receipt](../2026-09-27-item7-5-seals/README.md). All numbers here
remain evidence for the archived draft, not the current implementation.

The [structured results](validation-summary.json),
[measurement comparison](reconstruction-comparison.json), and
[source archive](review-source.tar.gz) describe 667 runtime/configuration/test
files with manifest SHA-256
`d26b69d625856af785f159bac485da382d19d4de6ae65c68e3c1eeea6f8088a5`.
The affected checks, explicit gates and measurements match that source exactly.
Earlier red/intermediate attempts remain separately identified in the results;
their original logs and requests are preserved in each `raw-receipt.tar.gz`.

## Implemented corrections

- `architecture.composeTemperature` is finite, nonnegative, defaults to zero,
  and controls both training choices. The 90/10 mixture is removed. The model's
  untempered legal-action softmax supplies straight-through credit, including
  on the forced alternative round.
- Argmax compares raw logits; structural preference applies only to exact
  logit ties. The old probability-rounding claim is withdrawn.
- Evaluation runs exploit only, without exploration or an optimizer. A native
  repeat from the same input/runtime produces the same program twice.
- At zero temperature, explore replays exploit's prefix under the updated
  weights, changes the uniformly selected forced round, then uses argmax.
  Packed intermediate seals participate in their actual execution order.
- The draft winner transaction gives both trials the same staged Teacher
  addresses and questions, and enables their private observation/prediction
  objectives. A strictly lower explore loss retains explore's complete end
  state; otherwise exploit survives. Both optimizer updates survive and the
  public clock/training counter advance once. The retained discourse history
  becomes the context for subsequent prediction.
- Schema, Params, Language, STM and the root README describe the new selector.
  No DP, flattened-temperature pass, compose policy loss or separate unary
  layer was reintroduced.

The [delta from the reviewed source](changes-since-review.patch) includes the
new [mechanism tests](../../../test/test_compose_review.py). The original
reviewed source manifest is retained in [JSON](reviewed-source-manifest.json).

## Checks so far

The selector red probes preserve the old temperature/tie/evaluation failures.
The winner red probe preserves the lower-explore-loss failure. The first native
winner integration run caught a prefix-restoration ordering defect; the prefix
is now computed while exploit's word-layout metadata is still available.

- Focused winner/record/evaluation tests: **32 passed** after that fix.
- Affected suite: **132 passed, 13 skipped**, all 145 selected cases completed.
- Explicit compiler, packed operand/reconstruction, and two-epoch graph checks:
  **4 passed**.
- Unchanged XOR_grammar CLI gates: **2 failed**, both at WholeSpace's six-row
  fixed capacity during first-epoch reset/autobind.
- Unchanged MM gate: **passed** this run. Its historical **0.21757** failure
  remains visible in the prior receipt. The raw-forward MM gate does not test
  the paired `runBatch` driver or establish mature-checkpoint utility.
- Documentation links: **109 passed**.
- The unchanged depth-3 campaign **fails again** on this source: all 16 observed
  end states have depth one. The new source-matched full sweep has not yet been
  issued.

These are bounded CPU checks, not million-sentence model training. Workers
retain the existing 8 GiB/1,800-second limits; the full-sweep driver retains a
24 GiB aggregate reservation and at most three workers. It isolates the
relative-STM cases that previously exhausted a shared worker, with the same
case inventory, assertions and limits. No seeds or thresholds were selected.

## Measurements

The existing seed-42 serial and packed/single protocols are rerun unchanged,
under 8 GiB/1,200-second process limits. The timing wrapper is observational;
it adds no optimizer step, sentence, selection policy or training budget.

Warmed training (five batches after two warmups), CPU wall time per batch:

| Component | Seconds | Share |
|---|---:|---:|
| Exploit forward | 0.359613 | 2.57% |
| Explore forward | 0.364350 | 2.61% |
| Exploit backward | 6.587968 | 47.13% |
| Explore backward | 6.621047 | 47.37% |
| Runtime snapshots | 0.025297 | 0.18% |
| Runtime restores | 0.001249 | 0.009% |
| Other batch work | 0.018701 | 0.13% |
| Total | 13.978225 | 100% |

Forward timing covers `_what_or_think`, including understanding and answer
construction. Backward timing covers `_backward_training_loss`. Optimizer
steps, loss assembly, staging and housekeeping are in "Other". Snapshot timing
includes the pre-exploit and exploit-end snapshots; restore timing includes
the pre-explore restore and the exploit-end restore when exploit wins.
Epoch-tail work is excluded from this split and retained in the unchanged
throughput measure. [Raw per-batch timings](measurements/batch-timing.json)
also preserve both trial losses, the selected winner, and cold/warm labels.

The two backwards consumed **94.50%** of the warmed batch; both snapshots and
restores together consumed **0.19%** on this draft. The amended review decision
requires the sentence/scratch-state restructure for correct causal commits,
independently of this timing. These measurements neither gate that decision
nor describe its implementation.

Serial reconstruction means (no tuning):

| Phase | Reviewed September 26 | Current draft |
|---|---:|---:|
| Before training | 0.1040502079 | 0.1040863823 |
| During training | 0.1041770041 | 0.1069973588 |
| After training | 0.1039280705 | 0.1023077499 |

Warmed throughput is **0.1429868375 sentences/second**, including epoch tails.
The prior 0.1271711631 reading overlapped other CPU work; it is not a controlled
speed comparison. Explore wins six of seven batch comparisons in this draft;
that is a diagnostic of this short workload, not a learning acceptance result.

Packed/single evaluation retains identical initial parameter and dictionary
fingerprints. Exact reconstruction parity is **still absent**:

| Layout | Reviewed September 26 | Current draft |
|---|---:|---:|
| Packed | 0.7877992094 | 0.7846425250 |
| Single | 0.6835970134 | 1.1408610493 |

Both rows report truncation in the packed presentation and in both single
presentations. The second sentence in each row has matching byte cost between
layouts; the first sentences differ. These values are preserved without
changing budgets, thresholds, seeds or reconstruction weights. Removing
stochastic evaluation did not restore packed/single parity on this fixture.
