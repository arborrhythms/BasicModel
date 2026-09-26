# Item 8 review candidate

The original measurements and test reports remain diagnostic evidence; the
[review correction receipt](../2026-09-26-item8-review/README.md) supersedes
its implementation and validation status.

Started from BasicModel `074093ec`, following [todo](../../../todo.md).
Uncommitted; stop for Claude's review before committing. Learned utility and
decreasing opaque use remain unproven. No qualifying FineWeb checkpoint has
been evaluated.

The [predeclared protocol](PROTOCOL.md) separates unconditional correctness
from quality evaluations requiring one million completed FineWeb training
sentences. It fixes seeds, independent training arms, held-out strata,
actual-work comparisons and reconstruction/discrimination controls before
results. The complete held-out curriculum, additional coverage catalogs and
independently trained qualified controls remain future empirical work.

The chooser now prefers structural implementations on exact score ties and
preserves better opaque scores. It adds no learned parameters or numeric
answer features. Soft probabilities and exploration are unchanged. Committed
programs supply sentence/operation routing counts; missing evidence cannot be
counted as zero. The current named structural inventory does not contain a
generic opaque candidate, so a measured zero cannot establish declining use.

Three new probes first failed for binary tie selection, missing thought tie
metadata and W=6 rejection. After the fix all three pass. Broader affected
checks found a compatibility adapter whose committed marginal intentionally
differs from raw scores; the correction preserves that existing choice unless
the actual catalog mixes structural and opaque implementations.

The inherited MM grammar assertion passed one ambient initialization before
the change. This is not evidence that the historical .2175724208 failure is
fixed. Both unchanged XOR_grammar CLI assertions initially fail on unsupported
W=6. Supporting that existing static width exposes the next blocker: the
six-row symbol inventory exhausts during the first epoch's reset/autobind,
after batch execution and before quality scoring completes. No corpus,
capacity, seed, epoch budget, quality assertion or expected-failure marker
was changed.

The remaining trace is `runEpoch` → `dispatch_per_row_reset` →
`_commit_autobind_from_stash` → `_maybe_autobind_meta` → `insert_meta` →
`insert_whole`: the legacy path allocates word and META rows in WholeSpace and
reaches row 6 at capacity 6. `XOR_grammar.xml` omits `propertyBasis`, whose
default is false; `MM_grammar.xml` explicitly uses the property inventory and
downstream concept ownership. Whether porting that ownership is appropriate
for the historical XOR fixture remains a review question, not a tested fix.

The [original arbitrary-symbol probes](archived-arithmetic-isolation-probes.py)
remain preserved from the September 17 archive. Current checks trap the exact
arithmetic module, retired code helpers and scratchpad entries after corpus
construction; run numeric and renamed surface inputs through native
understanding/reconstruction/output; and verify a real optimizer step.
Renamed relation checks preserve payloads while changing native references and
word bindings. These are isolation controls, not learned arithmetic evidence.
The current native fixture uses its normal data owner rather than the retired
math-specific fixture, and typed thought execution uses its current context.

## Measurements

The [serial baseline](baseline/serial-baseline.json.gz) exactly reproduces the
reviewed 9b values: before **.10059066489338875**, during
**.09482414424419403**, after **.09287650510668755**. The unchanged workload
has seven updates; five timed updates follow two warmups. Measured throughput
is **.55197 sentences/second**, including the post-batch routing read. Another
CPU diagnostic worker was active, so this is a warmed observation, not an
isolated speed comparison. No speedup is claimed.

The [packed](measurements/packed.json.gz) and
[single](measurements/single.json.gz) runs exactly match initial parameter and
dictionary hashes, all four sentence artifacts and mean byte cost
**.6839025616645813**. These remain reconstruction controls, not utility gates.

[Committed routing](baseline/routing.json.gz) contains 8 before-training,
14 training and 8 after-training sentence presentations, with respectively
76, 136 and 76 grammar operations. Every selected operation is structural:
both opaque shares are zero. No sentence lacks operations. The three phases
are not increasing-coverage training arms. There is no opaque candidate in
this catalog, so these zeros cannot satisfy the decreasing-share criterion.

All three declared MM seeds were attempted once with the unchanged .20 bar,
Adam .01 and the declared full 900-update diagnostic budget:

| Seed | Completed updates | Initial MSE | Final MSE | Minimum MSE | Status |
|---:|---:|---:|---:|---:|---|
| 0 | 900 | .25002918 | .01436063 | .01434910 | Below inherited bar |
| 1 | Unknown | — | — | — | Stopped by 8 GiB limit |
| 2 | 900 | .25050098 | .00025620 | 1.4655e-14 | Below inherited bar |

The raw [seed 0](measurements/xor-0.json.gz) and
[seed 2](measurements/xor-2.json.gz) records retain every loss and sampled
gradient norms. The [seed 1 process receipt](measurements/xor-1.process.json.gz)
records exit 137 after 215.41 seconds and a sampled footprint of 8.0035 GiB,
at which the unchanged guard stopped it. The harness wrote its numerical
summary only on completion, so its last completed update/loss is unavailable.
That incomplete run is not a pass, and the cause of the additional retained
memory has not been isolated. No seed was substituted, retried, or discarded.
The historical unseeded .2175724208 failure remains unresolved.

## Validation status

The final [affected checks](affected/result.json.gz) pass **108/108**: five
arithmetic isolation checks plus 103 documentation checks. The broader
207-case affected selection preceded the final context-only fixture port;
it had 204 passes, two opt-in skips and one failure from that fixture's retired
execution context. Existing
chooser, pipeline, generation, reconstruction, discrimination and readiness
checks retain their assertions.

The final [explicit gates](xor-gates/result.json.gz) complete **2 failures and
3 skips**. Both XOR_grammar assertions fail on the unchanged six-row capacity.
The three artifact quality evaluations skip for missing qualifying checkpoint
evidence; they are not passing quality results. No million-sentence checkpoint,
independent causal controls, growing structural catalogs, or learned-utility
result has been supplied by this candidate.

The single [full sweep](full/result.json.gz) completes all **4,968 unique
cases: 4,641 passed, 326 skipped and one strict XPASS**. Its exit code is 1.
It takes **1,419.71 seconds**, peaking at **4.82 GiB per worker**, with no
memory-limit stops or compile-cache retries. Three workers each have an 8 GiB
cap within the 24 GiB aggregate reservation; batches contain at most 32 cases
from one file.

The **strict XPASS** occurs in
`test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input`.
Its existing assertion unexpectedly passed, so pytest correctly reports a
failed sweep under the unchanged strict expected-failure marker. The reviewed
9b sweep recorded an expected failure for the same unchanged test. This
fixture uses a fresh unseeded model, and its overlap helper scores only the
common position prefix; it does not require matching reconstructed/input
lengths. That is a limitation to investigate, not an established explanation
of this run: the passing overlap and tensor shapes were not logged. No marker
was removed, assertion relaxed, or repeat attempted to change the outcome.
This candidate therefore has no green full-sweep claim and no new
reconstruction-quality claim.

The full sweep, final affected checks and explicit gates match all **673
source files**, aggregate SHA-256
`20e335f9dfa6bfb0e08d63af2e723e635dd0df39e061be3d328d76a7e56dbd73`.
The measurements precede only the final arithmetic fixture's context port;
their runtime/config sources match, and the single test-source delta is
retained explicitly. See the [validation summary](validation-summary.json),
[source manifest](source-manifest.json), [review source archive](review-source.tar.gz),
[implementation patch](changes.patch), [reconstruction comparison](reconstruction-comparison.json)
and [complete seed summary](xor-summary.json). Earlier failed probes and
intermediate corrections are archived under `diagnostics/` with their source
deltas. Final documentation-link verification is recorded in the validation
summary.
