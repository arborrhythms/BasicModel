# R2 diagnostic context

The objective diagnostic used `output/item6-identity-rst/recovery-factorial-off/final.ckpt`
on production revision `e38a19fbd`, before the forward charge. Its four inputs
are the first four recovery held-out clauses. `probe_objective.py` supplies
all-sum/STOP actions while retaining native legality, weights and cost terms.
This is a mechanism witness. An earlier all-lift witness kept the empty
determiner meaning and was rejected as evidence of meaning retention.

The first-word logit diagnostic used `output/item6-identity-rst/warmup/warmup.ckpt`.
Initial chooser weights were evaluated on that same warm-up input. Reading
can provision the percept rows that the warm-up has not yet seen; no optimizer
ran, and the diagnostic copy was discarded.

`probe_skip_non.py` supplies one native legal STOP departure at the first
meaning-removing non. The rest of the suffix runs natively at learning rate
zero, with conceptual admission frozen. No trainable parameter changes. The
fixture in `../validation/retained-trials.pt` retains both native programs and
all four clauses. `test/test_forward_discard.py` recomputes their forward
charge and the keep decision. It is the one new defect regression.

The charge uses relative L1 mass. Binary input coverage takes the coordinatewise
maximum absolute operand values; summing both operands would double-count
shared evidence. It preserves pole swaps but can miss equal-mass substitutions.
The content certificate therefore remains. The operator certificate still
requires exact word recovery and zero recomposition residual; its total now
includes real OR counterevidence discard. The paired-trial certificate checks
the shared context object directly, rather than using zero cost as a proxy.

The scoped recovery driver is `../run_recovery.py`, committed in `0cd221d74`.
It restores the accepted R/S/T warm-up and uses the existing corpus, native
presentation path and active-column grader. The existing restore adapter loads
metadata, then copies saved tensors back after the loader resynchronizes its
shared codebook. This does not certify native compiled checkpoint retention.
Closed norms and action programs are read from the same admitted held-out
reading that identification grades. No extra held-out replay is performed.

The older eight-branch attempt stopped intentionally during warm-up after the
scope correction: two-word 64/64 and four-word 26/64, no branch started. Its
partial output remains in `output/item6-identity-r2`. The completed scoped runs
are in `output/item6-r2-scoped`. Candidate commits identify source; there is
no source archive or integrity procedure for this measurement.
