# Retained positional-content development attempt

The new `test_forced_not_and_or_descend_and_snapshot_certificate` failure
revealed a defect in the first content charge. Two recovered word meanings
were reversed; their meaning bands were fully retained. Positional L1 error
charged 1.633 and 1.560 while the positive/negative band totals had exactly
zero error. The unchanged certificate correctly rejected the extra cost.

R now compares the sum of each positive and negative meaning coordinate
across the clause's leaves. This is the permitted meaning-band coverage
option: it retains multiplicity and prevents opposing leaves cancelling,
without charging orientation as lost content. The certificate assertion is
unchanged. The numerical regression also includes opposing leaves and a
permuted complete reconstruction.

The concurrent warm-up and sweep were interrupted before the source edit.
No experiment branch had started. Their source-matched partial receipts and
the first saved-checkpoint attention continuation are retained here. The
final experiment and attention diagnostic restart with the same seed,
checkpoint, and presentation budgets. No failed run is counted as complete.
