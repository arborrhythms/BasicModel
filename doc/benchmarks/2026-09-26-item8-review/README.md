# Item 8 review corrections

Claude's feedback identifies two defects in the
[initial candidate](../2026-09-26-item8/README.md): catalog composition switches
the selection tensor, and a common-prefix overlap can reward a reconstruction
that omits positions. Both corrections are accepted. The functional chooser
and reconstruction trace now apply the structural tie rule to the committed
marginal for every catalog. The overlap helper returns zero for mismatched
row/position counts, retaining its existing threshold and strict expected
failure. The unchanged peer-pipeline fixture remains a regression control.
SelectedMeaning also explains the zero-initialized deterministic thought
policy and structural conclude transition.

Deterministic failing probes precede the fixes: both mixed-catalog paths chose
from raw scores, while missing/extra positions or a missing batch row could
score perfect overlap. Matching-shape full and partial overlap controls retain
their original scoring. The review's approximate one-in-a-hundred XPASS rate
has not been measured; no repeat or seed selection is needed to demonstrate
the prefix defect.

Before measurements, retain the initial protocol unchanged, including all
three diagnostic seeds and the incomplete seed-1 memory stop. Repeat only the
reviewed seed-42 seven-update reconstruction baseline and exact packed/single
parity once on the corrected source. Preserve the explicit XOR capacity
failures and missing-qualified-checkpoint skips. Run affected checks, then one
full default sweep on that same source, with 8 GiB per worker and at most
24 GiB aggregate. The user's review follow-up authorizes committing these
corrections after the receipt is reissued. No publication/push is requested.

The full held-out learning comparisons and growing structural-coverage study
remain pending the million-sentence FineWeb prerequisite. Correctness probes
remain unconditional. Learned utility and decreasing opaque use are unproven.

The [review-only patch](review-changes.patch) isolates these changes from the
initial candidate. [Contract checks](unchanged-contracts.json) verify that the
peer-pipeline fixture, XOR configurations/assertions, pytest strict setting,
and the overlap test's assertion/marker syntax remain unchanged.

The affected selection completes **140 cases: 124 passed, 15 opt-in skips and
one expected failure**, with no unexpected failures. All eleven new probe
cases pass after five failed before the fixes. The explicit gates retain two
XOR capacity failures and three missing-qualified-checkpoint skips. An earlier
selector typo collected no cases and remains in the diagnostic record.

The corrected-source reconstruction baseline exactly matches reviewed 9b:
**.10059066489338875 before, .09482414424419403 during, and
.09287650510668755 after training**. Packed and single runs match the original
candidate exactly in parameter/dictionary identities, complete sentence
artifacts and **.6839025616645813** byte reconstruction. The warmed serial
observation is **.49777 sentences/second**, including routing reads and
concurrent CPU work; it is not an isolated speed comparison. Routing again
records 30 sentence presentations and 288 structural operations with no opaque
candidate, so the zero shares establish no decline.

The initial parity wrapper lacked the legacy driver's import path. Both
processes exited before model construction; their logs and the original
wrapper are retained. Only those unexecuted parity controls were repeated
after the import-path correction; the completed serial baseline was retained.

The single [corrected-source full sweep](full/result.json.gz) passes with
**4,980 unique cases: 4,653 passed, 326 skipped and one expected failure**.
The overlap gate scores zero against the unchanged .8 threshold; it no longer
rewards an incomplete position prefix. This confirms the test correction, not
successful per-position reconstruction. The sweep takes **1,417.84 seconds**
and peaks at **7.19 GiB per worker**, with no memory stops or compile-cache
retries. Three workers each have an 8 GiB cap within the 24 GiB aggregate
reservation, with at most 32 cases from one file per batch.

The full sweep, [affected checks](affected/result.json.gz),
[explicit gates](xor-gates/result.json.gz), and repeated reconstruction
measurements match all **673 source files**, aggregate SHA-256
`aa88d5d6c5c898c6b9aab8e549bdf61bba6c4849ab51699fc0a8874892eba9c1`.
See the [validation summary](validation-summary.json),
[source manifest](source-manifest.json), [source archive](review-source.tar.gz),
[complete implementation patch](changes.patch), and
[reconstruction comparison](reconstruction-comparison.json).
The original three-seed diagnostics retain their earlier source identities;
the new summary lists those deltas explicitly instead of claiming a rerun.
Final documentation-link verification is recorded in the validation summary.

The review corrections are validated for the requested commit. The two
unchanged XOR capacity failures, incomplete seed-1 diagnostic, and unavailable
qualified learning comparisons remain open; this implementation does not
close item 8's empirical gates.
