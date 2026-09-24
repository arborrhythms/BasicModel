# Item 11b: containment at the extent and the raw smoke guard

This corrects `aa235d2` / `11dd53d` before 11c. A PartSpace literal is
contained in its subject's extent, whether it names one part or an ordered
group of parts. Its presence is shared by that subject's positions; other
positions cannot deny a part already present there. Absence still requires
complete observation of the subject. Separate subjects retain separate
evidence, and an ordered group cannot match across their boundaries.
WholeSpace properties retain pervasion at each position. The existing
signed folds and the union at extent readout are unchanged.

Raw analysis retains uniform carrier coordinates, clips their ends to each
input's last observed byte, and clears regions consisting only of padding.
For the smoke workload's eleven observed bytes in a 4,096-byte buffer, its
first raw region is `[0,11)`, not `[0,512)`. The native constant-signature
runs and word extents are unchanged.

The smoke guard again uses `data/MM_sparse_concept.xml`, runs a real
optimizer epoch, and checks each present word by concept id. Its extent,
retained positions, carrier and symbols must have positive evidence and
exactly zero negative evidence. The attended-field, recurrence and
identity corrections from the preceding landing remain in place. The
untracked 11c plan is excluded.

## Failing probes and correction

The [initial receipt](red-result.json.gz) fails all four cases on the
accepted runtime: a present single part gains negative evidence from other
positions; an ordered group spanning positions loses its positive evidence;
raw spans include padding; and the restored raw smoke cannot read its
present words. The [workers](red-workers.log) and
[source manifest](red-source-manifest.json) preserve those failures.

The first [targeted correction](targeted-result.json.gz) passes 26/26,
including all six unseeded native XOR runs. The restored smoke reads
`hello world`, `hello there`, `loving world` and `loving there` as `(1,0)`
at both extent and symbol scope ([workers](targeted-workers.log)).

Adding an empty-subject assertion then exposed a zero-width reduction in
the attended-field binder: both parametrizations fail in the
[empty-subject receipt](empty-red-result.json.gz), with
[workers](empty-red-workers.log) and
[source manifest](empty-red-source-manifest.json). An empty field now
contributes zero support to the existing binding calculation.

The final [affected receipt](affected-result.json.gz) passes **129/129**
with slow tests enabled, two CPU workers and one-case batches. It covers
membership reads, dual towers, per-turn binding, recurrent part definitions,
evidence scopes, primitive properties, sparse folds, codebook ownership and
end-to-end learning. The [restored smoke](smoke.log) reads each of its four
present words as `(1,0)` at extent and symbol scope; its assertions also
check retained position and carrier evidence. All [six fresh unseeded XOR
runs](xor-affected.json) pass. The [workers](affected-workers.log),
[driver](affected-driver.log) and
[source manifest](affected-source-manifest.json) preserve the final run.

## Measurements

The [preserved driver](../2026-09-23-item11b/run_measurements.py) replays
serial reconstruction with `conceptualPi` off and on, frozen primitive
memberships, and the prior/segmentation comparison. Seed 42 fixes this
measurement only. The [comparison](final-source/comparison.json) matches
accepted 11b exactly:

| Serial reconstruction mean | `d4dc385` | Current, off and on |
| --- | ---: | ---: |
| Before training | 0.1005932558 | 0.1005932558 |
| Five measured updates | 0.0927930698 | 0.0949083805 |
| After training | 0.0923267286 | 0.0924575571 |

The after-training mean remains 0.142% above the reference; the update
mean is 2.28% above it. Frozen primitive memberships reproduce all three
reference means exactly. The [prior comparison](final-source/priors.json)
retains all eight rows over 256 bytes with zero error and unchanged
segmentation for 32 sentences and 160 runs.

All four guarded probes exit zero. The [manifest](final-source/manifest.json)
matches the affected run's 639-file validated source map, SHA-256 of its
sorted JSON representation:
`523f8196273c5e715b4c8aaadc868520d39b768d16bf6b25c0be6d102dc63ad7`.

## Full validation

The [full receipt](full-result.json.gz) completes **4,798 unique cases:
4,467 passed, 330 skipped, one existing expected failure**, exit zero.
The expected failure remains
`test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input`.
All [six fresh unseeded XOR runs](xor-full.json) pass again, with MSE
`2.842170943040401e-14`. No assertion is waived and no compiler-cache retry
ran. Explicit slow coverage, including the raw smoke, is in the affected
receipt above.

The run uses ten one-thread CPU workers, eight-case, one-file batches and
8/20 GiB worker/aggregate caps. It completes in 1,092.1 seconds and peaks
at 12.39 GiB aggregate memory ([summary](full-summary.json),
[driver](full-driver.log)). All 639 files in the
[source manifest](full-source-manifest.json) match the affected receipt,
measurements and working tree byte for byte. Documentation is recorded
separately. Preserved raw logs and generated measurement XML retain their
original bytes.

[Receipt metadata](receipt-info.json) records implementation commit
`15c9bde`. All 639 committed source blobs match the validated snapshot
byte for byte.

All **80 documentation-link cases** pass after completing the receipt and
todo ([receipt](doc-links-result.json.gz)). Source and prose pass whitespace
checks; preserved raw logs and generated measurement XML are unchanged.
