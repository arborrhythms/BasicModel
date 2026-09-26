# Item 9b: occurrence positions and per-word reconstruction time

Status: committed in `8bc710a` after validation and Claude's acceptance.
The [protocol](PROTOCOL.md) preserves the unchanged measurement workload and
memory limits. This follows the accepted item 9 learning-evaluation machinery
and addresses the decision recorded in plan section 4f.

Claude's experiment showed that reducing the coordinate periods did not remove
the reconstruction-cost jump. The model had begun putting the same field
timestamp into every word event, and the reconstruction objective charged for
recovering that timestamp on every word. That changed the reported objective;
it did not demonstrate worse byte reconstruction. The production loss uses
separately weighted content, position and time means. A zero reverse prediction
against a unit sine/cosine time band adds `0.1 * 0.5 = 0.05` per comparison.

The correction keeps the advancing `when_time` band in events and captured
programs, while excluding it from per-word D3, leaf-distillation and detached
reverse objectives. General event comparisons can still score time. No new
field-level loss is introduced.

An input occurrence now carries its byte start in the registry's input range.
Its stored part row is the identity of a reusable percept, not the position
where it appeared. Input symbols use that same start. The symbol range reserves
one address per pole, used by thought-produced symbols, without extra slots for
occurrences. The existing ladder formula remains; its periods follow the smaller
registry. A part's end can be calculated from its start and byte length. The
band alone cannot supply a whole's end; this remains an explicit limitation.

The initial red selection exposed two fixture mistakes: the D3 seam belongs
to `BasicModel`, and native grammar retains whitespace units. Corrected probes
include starts `[0, 3, 4, 7, 8]` for `cat cat sat`. The native subjective-clock
check already passed before the patch. Coordinate and reconstruction-gradient
checks failed as expected; no seed, quality threshold or tolerance was changed.

The [corrected red probes](diagnostics/red-corrected/result.json.gz) and
[native occurrence probe](diagnostics/red-native/result.json.gz) ran against
the preceding review's runtime, before any fix. Their source maps and worker
logs are retained, together with the initial fixture failures. Existing range
tests now assert the new occurrence contract; the large-address test retains
its original 512-million-address scale without the removed slot multiplier.

## Measurements

The unchanged seed-42, CPU-eager MM_ladder driver runs seven optimizer updates;
the training mean uses its five timed updates after two warmups. This is a
measurement, not a seed-selected quality assertion.

| Source | Before training | Mean training | After training |
|---|---:|---:|---:|
| Reviewed September 25 follow-up | .1005906649 | .0949082300 | .0882481802 |
| Before this correction | .1505906619 | .1347484022 | .1362805218 |
| This correction | .1005906649 | .0948241442 | .0928765051 |

The [new baseline](measurements/serial-baseline.json.gz) removes the extra
initial .05 and exactly matches the earlier initial loss. Mean training cost
is 0.089% below the earlier mean, but final cost is **5.245% higher**. Claude's
prediction of recovery within 1% therefore holds for the initial and mean
values, **not the final value**. The correction changes the position inputs
and objective while retaining the transported timestamp; it does not recreate
the earlier training trajectory. These measurements do not isolate the cause
of the remaining final difference. No seed, update budget or assertion was
adjusted to remove it; .0928765051 is the new after-training baseline.

The [packed/single comparison](measurements/comparison.json.gz) is exact:
initial parameters, initial dictionary, all four sentence roots, reference
and recovered ideas, and sentence byte costs match. Both layouts give mean
byte cost **.6839025616645813**, unchanged from the prior receipt. The driver
warms identical vocabulary and gives the joining space to the first sentence
in both layouts. This is reconstruction parity, not learned success.
Occurrence locations and timestamps describe each layout's actual input;
the parity comparison does not require those bands to be identical across
packed and separate presentations.

Claude's supplied [control](claude-experiment/control.json.gz) and
[reduced-period run](claude-experiment/reduced_periods.json.gz) are archived
with the bisection outputs. Those are Claude's experiments, not new reruns in
this follow-up. The current baseline and parity measurements are new runs.
The [guarded runner](run_measurements.py) uses the existing item-10 drivers;
copies of those drivers and the unchanged parity configuration are retained
under `measurements/`.

## Verification

The [affected selection](affected/result.json.gz) passes **40 checks** and
skips **14 opt-in slow cases**. The [explicit slow selection](slow/result.json.gz)
passes **5/5**, including native packed reconstruction under three ambient
initializations and the real training-path check that D3 keeps the position
weight while excluding time. Peaks are 1.27 and 6.66 GiB per worker. All runs
use the unchanged 8 GiB process cap; concurrent reservations stay within 24 GiB.

The [final full sweep](full/result.json.gz) completes **4,948 unique cases:
4,621 passed, 326 skipped and one existing expected failure**, with no
unexpected failures. It takes 1442.23 seconds and peaks at
4.81 GiB per worker. There are no compile-cache retries or memory-limit
stops. Three workers use one file and at most 32 cases per batch, under the same
8 GiB cap each. The [full worker logs](full/workers.log.gz) and
[source manifest](full/source-manifest.json.gz) preserve the evidence.

The full and affected/slow checks and measurements share **669 source files**,
aggregate SHA-256
`61bca3d4143303c811ce3a12b1e27067bd418e833e26a376db2568dfb0363ecf`.
The [source map](source-manifest.json), [source archive](review-source.tar.gz),
[patch against the item 9 follow-up](follow-up.patch) and
[validation summary](validation-summary.json) identify this correction's eight
runtime/test files. The publication includes the preceding reviewed item 9/9b
work as well as this correction.
Final documentation-link verification passes **101/101**.

No maturity-qualified learning result is claimed. The million-sentence
FineWeb checkpoint requirement and its evaluations remain unchanged; no
qualified checkpoint has been supplied. The old slow reconstruction-cache
probe remains unverified and assigned to item 1. The global-coordinate
transport-noise question remains in FutureWork.

## Review acceptance

Claude accepted the correction on September 26, conditional on the full sweep
finishing green; the result above satisfies that condition. Alec subsequently
authorized the commit and requested a handoff for item 8. Claude reported 178
local passes across the 9b/parity/XOR/documentation selection, and 28 passes plus
the known-red `test_mm20m_grammar_free_derivation_roundtrip` in the occurrence
and round-trip selection. That test's zero exact recovery remains the item 6
recovery gate; it is not waived or counted as passing this landing. Claude
accepted the recorded after-training difference as the new short-run baseline.
These are the reviewer's reported local results; the source-matched runs above
are Codex's validation.

## Publication

Implementation `8bc710a8b68cd71e4331b31bd22d2a7c8fd03f57` [matches all 669 reviewed source blobs](committed-source-verification.json).
Item 9b is recorded under Done; item 9's empirical acceptance still requires
a qualified FineWeb checkpoint. Item 8 is the next implementation task. The
publication changes only review status and handoff bookkeeping after the full
sweep; no runtime, test or configuration source has changed.

Publication documentation links also pass **101/101**, including the committed-source verification link.
