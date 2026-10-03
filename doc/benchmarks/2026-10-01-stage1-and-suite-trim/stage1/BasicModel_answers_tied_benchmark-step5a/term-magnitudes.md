# Every observed training cost term

Raw and weighted units are separate; zeros remain. Missing terms have count 0. Per-trial distributions count active rows; batch distributions count batches.

| Term | N | Min | p10 | Median | Mean | p90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| band.None._run_batch_once.what.raw | 1 | 0.438982 | 0.438982 | 0.438982 | 0.438982 | 0.438982 | 0.438982 |
| band.None._run_batch_once.what.weighted | 1 | 0.3072874 | 0.3072874 | 0.3072874 | 0.3072874 | 0.3072874 | 0.3072874 |
| batch.accounting_residual | 1 | -2.514571e-08 | -2.514571e-08 | -2.514571e-08 | -2.514571e-08 | -2.514571e-08 | -2.514571e-08 |
| batch.raw.arma_loss | 0 | — | — | — | — | — | — |
| batch.raw.aux_total | 0 | — | — | — | — | — | — |
| batch.raw.csbow | 0 | — | — | — | — | — | — |
| batch.raw.defsp | 0 | — | — | — | — | — | — |
| batch.raw.expectation_policy_loss | 0 | — | — | — | — | — | — |
| batch.raw.gate_l1 | 0 | — | — | — | — | — | — |
| batch.raw.grammar_lesson | 0 | — | — | — | — | — | — |
| batch.raw.inter_contrastive | 0 | — | — | — | — | — | — |
| batch.raw.inter_loss | 0 | — | — | — | — | — | — |
| batch.raw.intra_loss | 0 | — | — | — | — | — | — |
| batch.raw.ld_loss | 0 | — | — | — | — | — | — |
| batch.raw.lossIn | 1 | 0.6871587 | 0.6871587 | 0.6871587 | 0.6871587 | 0.6871587 | 0.6871587 |
| batch.raw.lossOut | 1 | 0.3072874 | 0.3072874 | 0.3072874 | 0.3072874 | 0.3072874 | 0.3072874 |
| batch.raw.lossRev | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| batch.raw.output_policy_loss | 0 | — | — | — | — | — | — |
| batch.raw.readout_l1 | 1 | 0.01083333 | 0.01083333 | 0.01083333 | 0.01083333 | 0.01083333 | 0.01083333 |
| batch.raw.sbow | 0 | — | — | — | — | — | — |
| batch.raw.selected_pol_loss | 0 | — | — | — | — | — | — |
| batch.raw.totalLoss | 1 | 0.5080563 | 0.5080563 | 0.5080563 | 0.5080563 | 0.5080563 | 0.5080563 |
| batch.truth.balance_weight | 1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| batch.truth.luminosity_weight | 1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| batch.truth.truth_loss_weight | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| batch.truth.universality_weight | 1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| batch.weighted.lossIn | 1 | 0.3435794 | 0.3435794 | 0.3435794 | 0.3435794 | 0.3435794 | 0.3435794 |
| batch.weighted.lossOut | 1 | 0.1536437 | 0.1536437 | 0.1536437 | 0.1536437 | 0.1536437 | 0.1536437 |
| batch.weighted.lossRev | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| batch.weighted.readout_l1 | 1 | 0.01083333 | 0.01083333 | 0.01083333 | 0.01083333 | 0.01083333 | 0.01083333 |
| trial.exploit.raw.expectation_contrastive | 28 | 0 | 0 | 0 | 0 | 0 | 0 |
| trial.exploit.raw.expectation_inter | 28 | 0 | 0 | 0 | 0 | 0 | 0 |
| trial.exploit.raw.grammar_lesson | 0 | — | — | — | — | — | — |
| trial.exploit.raw.reconstruction | 28 | 0.5886263 | 0.6166085 | 0.7028254 | 0.759922 | 0.9064351 | 1.829042 |
| trial.exploit.raw.supplied_answer | 28 | 0.06293587 | 0.06341616 | 0.1779662 | 0.1459474 | 0.1890581 | 0.1957262 |
| trial.exploit.weighted.expectation | 28 | 0 | 0 | 0 | 0 | 0 | 0 |
| trial.exploit.weighted.reconstruction | 28 | 0.2943131 | 0.3083043 | 0.3514127 | 0.379961 | 0.4532176 | 0.9145209 |
| trial.exploit.weighted.supplied_answer | 28 | 0.06293587 | 0.06341616 | 0.1779662 | 0.1459474 | 0.1890581 | 0.1957262 |
| trial.exploit.weighted.total | 28 | 0.374299 | 0.4194908 | 0.5342265 | 0.5259084 | 0.6329486 | 1.036109 |
| trial.explore.raw.expectation_contrastive | 28 | 0 | 0 | 0 | 0 | 0 | 0 |
| trial.explore.raw.expectation_inter | 28 | 0 | 0 | 0 | 0 | 0 | 0 |
| trial.explore.raw.grammar_lesson | 0 | — | — | — | — | — | — |
| trial.explore.raw.reconstruction | 28 | 6.872745e-05 | 0.4124761 | 0.7017426 | 0.6871587 | 0.9064351 | 1.829042 |
| trial.explore.raw.supplied_answer | 28 | 0.0625 | 0.06338529 | 0.1779662 | 0.14363 | 0.1890581 | 0.1957262 |
| trial.explore.weighted.expectation | 28 | 0 | 0 | 0 | 0 | 0 | 0 |
| trial.explore.weighted.reconstruction | 28 | 3.436373e-05 | 0.2062381 | 0.3508713 | 0.3435794 | 0.4532176 | 0.9145209 |
| trial.explore.weighted.supplied_answer | 28 | 0.0625 | 0.06338529 | 0.1779662 | 0.14363 | 0.1890581 | 0.1957262 |
| trial.explore.weighted.total | 28 | 0.06297023 | 0.3087581 | 0.5342265 | 0.4872093 | 0.6329486 | 1.036109 |
