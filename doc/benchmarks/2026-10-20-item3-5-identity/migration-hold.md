# Strict artifact migration — publication held

**Alec, 2026-10-20:** “Keep strict rejection; hold publication for a migration decision.”

The registered strict-load certificates remain failed. No artifact is rewritten,
no content is clipped or replaced with unknown, and no new training is used.
The read-only probe replaces the migration method with an inspection hook in a
temporary model. Its completion is not a successful strict restore or a gate pass.
The original files, native addresses, evidence and saved weights remain on disk.

## Can the retained stage3 artifact be loaded under the new identity contract?

| Measurement | Result |
|---|---|
| Original artifact | `output/stage4e-student-order-20261017/order.ckpt` |
| Strict restore | Fails: identity content requires native concept coordinates in [0, 1]. |
| Distinct retained identity occurrence rows inspected | 5 |
| Signed or greater-than-one occurrence rows | 5 |
| Exact current-form matches under the saved projection | 0 |
| Matches allowing one presence scalar under the saved projection (maximum coordinate error ≤ 1e-5) | 3 |
| Training steps | 0 |
| Disposition | Migration decision required; strict rejection retained; publication held. |

Diagnostics: [full rows and projection comparisons](checkpoint_probe-stage3.json), [probe log](checkpoint_probe-stage3.log).

## Can the retained np_definite artifact be loaded under the new identity contract?

| Measurement | Result |
|---|---|
| Original artifact | `output/np-definite-restored-20261018/trained.ckpt` |
| Strict restore | Fails: identity content requires native concept coordinates in [0, 1]. |
| Distinct retained identity occurrence rows inspected | 6 |
| Signed or greater-than-one occurrence rows | 5 |
| Exact current-form matches under the saved projection | 0 |
| Matches allowing one presence scalar under the saved projection (maximum coordinate error ≤ 1e-5) | 3 |
| Training steps | 0 |
| Disposition | Migration decision required; strict rejection retained; publication held. |

Diagnostics: [full rows and projection comparisons](checkpoint_probe-np_definite.json), [probe log](checkpoint_probe-np_definite.log).

## Can the retained np_description artifact be loaded under the new identity contract?

| Measurement | Result |
|---|---|
| Original artifact | `output/np-description-flow-20261018/trained.ckpt` |
| Strict restore | Fails: identity content requires native concept coordinates in [0, 1]. |
| Distinct retained identity occurrence rows inspected | 18 |
| Signed or greater-than-one occurrence rows | 17 |
| Exact current-form matches under the saved projection | 0 |
| Matches allowing one presence scalar under the saved projection (maximum coordinate error ≤ 1e-5) | 3 |
| Training steps | 0 |
| Disposition | Migration decision required; strict rejection retained; publication held. |

Diagnostics: [full rows and projection comparisons](checkpoint_probe-np_description.json), [probe log](checkpoint_probe-np_description.log).

## What is not recoverable from the retained stage-3 evidence?

The five distinct signed rows are 508, 515, 517, 520 and 537. Rows 508, 517
and 537 have a scaled match under the old form projection, with errors below
2.4e-7. That is diagnostic evidence for a possible migration, not an approved
conversion. Rows 515 and 520 instead contain signed relation-address values
(±0.065653212 across 232 coordinates), with no matching form projection,
no retained descriptive derivation or leaf postings, and zero native cached
concept content. Their native descriptive region cannot be recovered from
a relation address alone. The NP continuations retain these rows too.

A decision must specify the disposition of those unrecoverable occurrence
contents and the resulting continuity guarantees for the saved learner.
Until then this branch cannot claim artifact compatibility or a clean
landing gate. Step 1 stays published; step 1b is an unpublished candidate.
