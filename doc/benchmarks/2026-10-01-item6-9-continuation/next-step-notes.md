# Preparation while step 5a validation runs

Historical preparation note, retained for the receipt. Steps 6 and 7 are now
implemented and checked; see README.md for the outcomes.

Step 6 must use the current sentence's ended slots and recorded operations
before `_commit_sentence` discards its trace. The existing
`_reconstruct_sentences` already reverses actual unary positions and binary
operations, with an actual leaf occurrence as a witness when one exists. Its
candidate search currently averages symmetric pairs. The next failing probe
will require one minimum-residual hard pair while checking that its parent
gradient still equals the existing soft residual mixture.

The XOR fixture's mixing binding has private grammar object rows, but no native
WORD symbol metadata. The reconstruction measurement must supply an explicit
numerical candidate bank, rather than republish the word symbols that caused
the reviewed item 7 regression. Candidate words should come from the known
object vocabulary; target text must only score the recovered words. The
existing byte training objective and its native bank stay unchanged.

The class and reconstruction gates should observe the final evaluation at the
open sentence boundary, retain only the resulting texts and unavailable flags,
and avoid another forward pass or retention of a completed grammar trace. The
reconstruction assertion becomes four of four word multisets, preserving
multiplicity and admitting only transpositions. `step6-probe-draft.py` is a
draft outside the collected source; it has not run yet.

Step 7 needs a per-slot unary history, carried explicitly through compiled
composition. It must follow push and binary compaction, reset a replaced slot,
and exclude an immediate inverse only at that slot. A private CSLang carry can
hold it without publishing native word identities or changing the public
compiled answer tuple. Parallel `OperationSelectionLayer.derive` also needs
the same history update. The legacy closing path needs inspection before
claiming full coverage.
