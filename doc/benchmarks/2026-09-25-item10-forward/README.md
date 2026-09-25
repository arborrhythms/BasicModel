# Item 10 final review correction: reconstruction from forward evidence

Review approved for publication by Alec on September 25, 2026.
This supersedes the saved-view inverse in the
[earlier resolution receipt](../2026-09-25-item10-resolution/README.md).

Item 10 remains **decided: dropped**. The normalized mode is deleted; its
[evaluation and source archive](../2026-09-24-item10/README.md) remain the
record of the rejection. Composed grammar operands exceed unit energy;
thirty alternating means preserve only a sixth of the spread; the geometric
floor has a dead gradient below and a steep slope above; convex bias emits
nonzero output for all-zero input. Reconstruction is worse and there is no
consistent speedup. Refine-before-raise keeps the accepted three-update
patience policy unchanged.

## Inverse and exclusions

The understanding's native reconstruction carrier contains only concept ids,
paired evidence, occurrence evidence, position spans and extents. No saved
percept event or input stack is used. `_reverse_body` resolves definitions by
concept id and calls `cs_percept_attribution` against the captured occurrence
evidence. Native row activity supplies radix decoding. The distributed codes
provide the continuous reconstruction score; code similarity does not decode
the surface. A later perception cannot replace the evidence or its geometry.

Parts are contained at the subject extent. A retained bracket of the group's
length locates its ordered constituents even when WholeSpace subdivides the
word into finer property runs. A larger bracket leaves placement unresolved.
Property descent distributes support only to written primitive memberships;
it cannot use a reconstruction gradient to invent an unseen member. Ambiguity
can still lose information: reconstruction is best effort from the forward.

Serial mode excludes field sigma, pi and not; parallel mode excludes grammar
lift and lower. Regression tests execute both modes with forbidden operations
poisoned. `XOR_exact.xml` drops the unexercised grammar block and explicitly
states that its lesson teaches primitive memberships and one name, never XOR.
The learned positive pole, order-0 located pi and order-1 sigma remain intact.

## Failing probes and corrections

The first probes rejected the stored native views and caught a parallel
forward dispatching the grammar. The first attribution inverse passed the
focused native runs but the CLI reconstructed **1/4**, below the fifty-percent
bar, and failed crisp output. Two smaller probes isolated missing placement
at the containing extent and inverse gradients entering absent primitive
memberships. Both are fixed. The part-only fixture now explicitly supplies
the ordered definitions it tests; the raw lesson helper supplies only the
named property and does not create word definitions.

The first full attempt caught a category-learning fixture that forced grammar
composition into parallel mode. It was interrupted and is diagnostic only.
The fixture now runs in serial mode and retains its category-assignment and
role-learning assertions; all three tests pass. The final receipt starts over
on that corrected source.

The test named `test_at_least_50_pct_inputs_reconstruct` had actually asserted
25%. Its assertion now enforces Alec's stated fifty-percent bar as
`2 * matches >= total`. The crisp-output assertion remains MSE < .05. Neither
gate selects a seed, changes the epoch count, nor weakens a threshold.

## Validation

The single final [full receipt](full-result.json.gz) completes **4,824 unique cases** with exit zero and no red outcomes. Every check below and the serial measurements match the same **648 source files**, SHA-256 of the sorted source map:

`bac8dbede6bc069132a4fb1e3ca75c683ace40614ea7e8e1832f9745a734b001`.

| Check | Passed | Skipped | Existing expected failure |
| --- | ---: | ---: | ---: |
| [affected-final](affected-final-summary.json) | 405 | 68 | 1 |
| [xor-final](xor-final-summary.json) | 2 | 0 | 0 |
| [explicit-final](explicit-final-summary.json) | 24 | 0 | 0 |
| [full](full-summary.json) | 4,502 | 321 | 1 |

The unseeded [CLI reconstruction gate](cli-gates.json) reaches **4/4** against the fifty-percent bar. Crisp output passes its unchanged MSE < .05 assertion. The [nine native learning runs](native-learning-runs.json) retain every declared run. The new serial exclusion check is marked slow and passes in the explicit selection. The full suite's expected failure is the existing cleared-cache word-overlap probe; no failing gate is reclassified.

[Serial reconstruction](serial/comparison.json) after training is **.0891618710**, against **.0923267286** at `d4dc385`, without a tolerance gate. Pi off/on and frozen priors agree. [Prior and segmentation measurements](serial/priors.json) preserve all eight rows over 256 bytes and all 160 runs across 32 sentences.

The [source archive](review-source.tar.gz), [hash map](review-source.json), [correction delta](source-delta.json) and [tracked diff](tracked-source.patch) make the proposed source reviewable. Earlier failing and interrupted attempts are diagnostics only. A concurrent affected-file attempt encountered a temporary test-config deletion; final checks ran sequentially after the full suite.

BasicModel remains at `606683a8e32aac66569669a5e607f64eeec3ae32` and WikiOracle at `8e7123a1c55ce10ce924d1f27b1e2037dc8890c7`. No commit, push, parent bump or item 9 work.


## Affected files

- [Models.py](../../../bin/Models.py): evidence capture, attribution inverse,
  activity-decoded reporting and mode dispatch.
- [Understanding.py](../../../bin/Understanding.py): owned conceptual field
  evidence and spatial brackets, addressed by concept id.
- [Spaces.py](../../../bin/Spaces.py): explicit attribution observations and
  native activity realization, with a continuous code score.
- [Layers.py](../../../bin/Layers.py): radix decoding by native activity.
- [PerceptProperties.py](../../../bin/PerceptProperties.py): reverse attribution
  restricted to written primitive memberships.
- [XOR_exact.xml](../../../data/XOR_exact.xml): removed grammar and corrected
  lesson/reconstruction contract.
- [Concept output tests](../../../test/test_concept_output.py),
  [CLI gates](../../../test/test_explicit_dimensions.py) and
  [category smoke](../../../test/test_category_em_smoke.py): inverse ownership,
  extent placement, mode exclusions and the fifty-percent reconstruction bar.
- [Architecture](../../Architecture.md), [Testing](../../Testing.md) and
  [todo](../../../todo.md): the accepted rule and completed review correction.

The earlier item 10 and 11c residue changes remain part of this uncommitted
proposal. Their archived measurements and deletion record are retained.

Final documentation-link verification: **84/84 passed**.
