# Required relative-error baseline decision

Part 4c is not implemented. Spec §10.2 requires every cost term to be divided by a detached trivial predictor's error computed from its targets. Two live cases do not supply a positive denominator:

* `InterSentenceLayer._kind_loss` scores a single `idea`/`relation` target with BCE. A single target, or a batch containing only one kind, has empirical entropy zero. At logit zero its raw error is log(2), not zero. Exact division is undefined/infinite. Using float32 epsilon instead gives a cost about 5.81 million; that is a new effective priority, not equal magnitudes. `zero-baseline-counterexample.json` records the arithmetic.
* The native production fixture has `concept_readout_l1 = 0.01` on all four rungs (the production stage-1 weights receipt). This is an optimizer-owned proximal regularizer, with no prediction targets or target entropy/variance. Definition sparsity is likewise a structural penalty when enabled. A zero target prior has zero trivial-predictor error. A running loss scale, a fixed unit baseline, or an exception for regularizers is not the literal target-derived rule.

The pending question is which baseline/exception to use for zero-variance/zero-entropy target sets and targetless structural penalties. No epsilon floor, new priority, running normalizer, seed, or threshold has been chosen. Trial/batch Error ownership, normalization, and reconstruction-priority gradients remain unimplemented pending that rule. The original answer training remains intact.

This is a design question about the mandated formula, not a permission request to edit or test. Universal reconstruction was independently prototyped and measured; it remains incomplete for older reading paths, and only that unfinished prototype was restored out of the production tree. Its source patch and all failures remain available for continuation without repeating the measurements.

## Resolution — spec §10.2a, relayed by Alec on 2026-10-01

The question above is resolved. The uninformed predictor is uniform for categorical/binary terms (baseline log K), and the origin for squared errors (detached mean target squared norm on the same active entries). Use a ratio of means, no epsilon floor and no running loss scale. Targetless regularizers, including the proximal readout L1, retain their own strengths and are recorded separately from relative errors. An exactly all-zero squared-error target set is likewise a penalty.

This preserves relative error against an uninformed reference; it replaces the earlier empirical-entropy/centered-variance reference. A nearly solved categorical term is not rescaled upward as its loss falls. The rule does not estimate an irreducible noise floor or guarantee matching gradient norms. The squared-error denominator depends on the origin and on target energy; for learned-code targets that energy can change with training even though its gradient is detached. These are measurement limitations, not a remaining request for a baseline decision.

The scope clarification also resolves the failed universal prototype: require reconstruction through the understanding on meronomy with grammar, retain perceptual reconstruction without grammar, list old modes as exempt until trim item 12, and count sentences with no admitted surface candidate without assigning a reconstruction term. Production normalization and precedence are still pending implementation.

The two historical documentation-link failures were saved in `historical-links-before/`; their destinations were repaired from the item-8 move manifest with complete old/new document text in `historical-links-repair.json`. Both unchanged cases pass in `historical-links-after/` (8 GiB worker guard).
