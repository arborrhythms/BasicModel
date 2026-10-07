# Operators round 4a — meanings exist

Status: implementation and validation in progress; **uncommitted, awaiting Claude's review**.
Baseline: `cce3a4f7b6e72fab4a9a27fd70d3fa7602553814`. Claude's existing plan amendment and meaning toy are preserved.

Order-zero meaning is a detached recency mean of fixed sentence identities on the for pole, with zero against. Form kernels retain 3a behavior; meaning composition uses the bilattice. Interpretation exchanges meaning poles under negative activation. Expectation trains once on kept trial rows only.

The 3a store had occurrence addresses but no sentence content key. This candidate adds a content lookup alongside provenance: identified sentence word bytes define a stable digest, direct opaque rows fall back to their initial form, and that same key seeds the sparse identity. Occurrence IDs, membership and recency remain intact.

[Grammar certificates](semantic-gates.json) are exact for both four-word vocabularies. The [BasicModel diagnostic](semantic-basicmodel.json) exhausts every ordered word pair, including self, and every row of the configured local corpus census. It reports 0.03154% conjunction false membership, 3.69037% disjunction, and 1.75383% for each negation against case. Both for-pole negation cases are zero; there are no false negatives. These rates are diagnostics; exact membership uses postings. The corpus census is separate from a resident production store or a training run.

The measurement campaign will run exactly ten sum controls, ten shared class/reconstruction trainings, and ten MM_xor trainings, unseeded and without retries or replacement. All outcomes will be retained. No centroid or priming change belongs to this round.
