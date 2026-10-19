# §9.2/6 repaired-tree measurement

One fixed run, seeds 19 and 23, each graded and shuffled: six training
batches of two sentences, Adam 0.001, exactly the corpus and shuffle from
the original measurement. One held-out forward before and after. Four gains
are measured at 0.5 after the matched zero-expectation consumer check; the
configuration defaults remain zero. CPU, one thread, 600 seconds and 8 GiB.
No continuation, threshold tuning, seed replacement or added updates.

Report each row's departure owner, its actual R+A difference, non-tied
exposure, proposal correction, policy loss, gradient and parameter change.
Compose and narrowing share the reducer; candidate attention owns separate
parameters. An owner without non-tied exposure is unexposed. Record local
residuals, explore wins and raw carrier norms. This is a fresh diagnostic on
the repaired tree, not a regrade of the original failed run.

The original six-batch corpus supplies no exhaustive native derivation
oracle. Its R is not a derivation-optimality rate. Preserve that rate as
unestablished and its gate false; paired execution and parameter changes
cannot certify it. Neither can this short run clear Item 0 or Item 5.5.
