# Item 11b landing brief

Alec, September 23, 2026. Implement todo item 11b, then stop for review.

1. An order-0 concept reads the product of the towers' feature memberships
   with its own signed weights. Use the existing pi fold over PartSpace
   percept rows and WholeSpace property rows. A negative weight is a
   nonnegative exponent on the row's complement. Both symbols use
   pervasion: required memberships for the positive, their complements for
   the negative. No dual fold at the seam; never an existential tower fold.
2. Union only across occurrences inside the subject's extent at readout,
   where both arises. Keep runs as positions and word units as extents.
3. Delete PerceptRead, the atom projection, conceptEvidenceFloor and its
   calibration harness. Codes remain for similarity, retrieval and tied
   reconstruction as a consequence of the definition. Use the existing
   definition-sparsity penalty over feature weights; add no regularizer.
4. For P = "is a one", two-position extents give 11 true-only, 00
   false-only, and 01/10 both. Learn XOR as a conjunction over P's two
   symbols, unseeded from primitive input, pools 4 and 8, three runs each.
   Keep (A ∨ B) ∧ ¬(A ∧ B) as a composition check only.
5. The gate passes, model.xml has no floor, and unrelated controls read
   exactly zero at every scope, removing the 256-position accumulation
   limit. Preserve 11a priors and constant-signature segmentation. Report
   serial reconstruction against d4dc385 as before.
6. Move _prepare_part_learning out of getParameters to the sentence
   boundary beside promotion_observe. The getter is pure. Normalization
   of assigned provisional disjunctive rows only is accepted.
7. Replace the 11a Architecture seam, maps, projection and extent-fold
   duality with the membership read and Alec's formulation. Precision
   exists only for location. Two-truths §1.1 already points to 11b. Close
   the todo normalization residue.

Publish: failing probe, fix, affected files, one source-matched full
receipt, BasicModel commit/push, WikiOracle bump/push, then stop for review.
