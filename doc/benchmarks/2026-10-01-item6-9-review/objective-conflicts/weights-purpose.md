# Weights, units and ownership

These are the effective values recorded from each initialized model, not changes
to either configuration. The native benchmark stopped at the memory guard before
any cost was measured; its weights and parameter/buffer ownership are available,
but its magnitudes and reach are not. Both cut arms remove only the trial answer
term. Batch-end answer training keeps its existing cut and weights.

| Term / coefficient | XOR_grammar | Native tied-answer benchmark | Where and purpose |
|---|---:|---:|---|
| Trial supplied answer, step 5a | 1 | 1 | Compare each trial's own answer against the supplied answer; intact graph |
| Trial supplied answer, cut arm | 0 | 0 | Omitted from trial comparison/training in this measurement arm |
| Trial reconstruction scale | .1, inactive | .5 | Tied byte reconstruction; XOR has reconstructInLoop=false |
| Trial legacy intra prediction | .1 | 0, retired path disabled | Predict the next conceptual input inside the sentence; XOR's active expectation term |
| Trial inter prediction | .1 | .1 | Predict the next ended sentence, when predecessor context exists |
| Inter content MSE / presence BCE / kind BCE | 1 / 1 / 1 | 1 / 1 / 1 | Internal structured prediction terms, then multiplied by inter weight |
| Trial inter contrastive | 0 | 0 | InfoNCE term disabled; temperature remains .1 |
| Trial grammar lesson | 1 | 1 | Enabled lesson terms only; none observed in the XOR run |
| Batch supplied answer | .9 | .5 | 1 − reconstructionScale, applied to lossOut |
| Batch reconstruction, lossIn | .1 | .5 | Full-percept D3 event cost for XOR; tied byte cost for native |
| Batch additional lossRev | .1 | .5 | Applied when present; XOR training value is zero, final evaluation reports D3 here too |
| Event what / where / when | .7 / .2 / .1 | .7 / .2 / .1 | Band MSE weights before the outer cost coefficient; absent bands contribute nothing |
| Batch embedding SBOW | .05 | .25 | Symbol bag-of-words embedding objective |
| Batch conceptual similarity | 0 | 0 | Conceptual SBOW disabled |
| Definition sparsity | 0 | 0 | Disabled; the returned value would already include its lambda |
| ARMA | 0 | 0 | Disabled expectation term |
| Batch legacy intra coefficient | .1 | 0 | XOR's intra term is already trained within trials; no batch replay is observed |
| Batch inter / contrastive | .1 / 0 | .1 / 0 | Sentence-trial reports are not replayed at batch end |
| Batch grammar lesson | 1 | 1 | Detached report after its trial-owned training |
| Output walk policy | 0 | 0 | Optional batch-end policy credit; never trial-owned |
| Expectation policy | 0 | 0 | Disabled |
| Selected thought policy | 0 | 0 | Disabled |
| Leaf distillation | 0 | 0 | Disabled |
| Pipeline auxiliary total | 1 | 1 | Each enabled registry term keeps its own recorded inner weight; none observed for XOR |
| Gate L1 lambda | 0 | 0 | Disabled gate sparsity; returned cost would already carry lambda |
| Concept readout L1 lambda, per stage | 0 | .01 | Optimizer-owned proximal update; its batch cost is a detached report with outer weight 1 |
| Luminosity / universality | .1 / .1 | .1 / .1 | Detached contextual multiplier on assembled batch cost when the truth store is nonempty |
| Truth falsity penalty | 0 | 0 | Disabled additive term |
| Truth balance | .1 | .1 | Function default; applies only to an eligible nonempty truth accumulator |

The numeric XOR trial answer is **raw MSE × 1**. Its batch answer is **raw MSE ×
.7 × .9 = .63**. Native supplied text answers use the event-band weights before
their outer coefficient (1 in a trial, .5 at batch end). No rescaling was made.

XOR's truth store takes the empty-store return, so the effective multiplier is
1 and no additive truth/balance cost is observed. The largest absolute residual
between the observed training total and the term-by-term accounting is
1.78e−8. Zero-valued terms and absent terms are distinct in the saved tables.

The byte candidate-assignment temperature is .1. It controls the assignment
distribution, not a multiplier on the loss. The native concept dictionaries are
buffers with requires_grad=false and contextual_rotation_only=true. Their .01
contextual learning rate is an update rule, not an objective coefficient; the
situation and expectation context weights are both zero. XOR's contextual rate
is zero and its object code dictionaries are trainable parameters.

The complete effective configurations, original instrumentation, per-batch
weights, parameter identities and raw/weighted term distributions are retained
in each arm's plan.json, weights.json, events.jsonl and manifests. The native
guard prevents any claim about the usual magnitude of its configured terms.
