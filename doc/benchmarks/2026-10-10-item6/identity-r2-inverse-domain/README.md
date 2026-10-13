# R2 repair: local inverse domain (2026-10-13)

Candidate on `codex/item6-r2-forward-discard`; previous R2 was rejected.
The inverse domain is shared with `DerivationReconstruction`. The forward
charge covers only each operation's own undefined meaning coordinates,
using its actual output, with the whole detached input mass as denominator.
It compensates the field objective's κ = 0 exemption; the free-byte path
has no such exemption. No operator-name exceptions or rule retirements.

## Are the accepted reconstruction certificates restored?

`test_operators_final_b.py` is restored to the accepted R/S/T version.
[Certificate log](diagnostic/certificates.log):

| Certificate | Required result | Observed |
|---|---|---|
| Exact OR / NOT reads | R = 0 on every row | Pass, unchanged assertion |
| Both training trials | Zero reconstruction cost | Pass, unchanged assertion |
| Exact OR forward discard | 0 | Pass |
| Local lower domain below an undefined ancestor | Defined locally; undefined inherited | Pass |

Search operators remain undefined in the **field** derivation reader. The
accepted exact OR certificate uses **free-byte** reconstruction; its forward
charge is zero because that objective does not exempt failed inverse
coordinates. This repair does not assert that OR has an analytic inverse.

## Does skipping one `non` still win?

The [native paired trial](diagnostic/skip-non.json) uses the retained R/S/T
checkpoint, one legal STOP departure and the native suffix; parameters do
not change. All other keep terms remain 1.85.

| Retained clause | Greedy discard / keep cost | Explore discard / keep cost | Keep | Final meaning norm |
|---|---:|---:|---|---:|
| the black cat runs . | 3 / 4.85 | 2 / 3.85 | explore | 0 |
| the white cat moves . | 3 / 4.85 | 2 / 3.85 | explore | 0 |
| the small cat waits . | 3 / 4.85 | 2 / 3.85 | explore | 0 |
| the large cat sleeps . | 3 / 4.85 | 2 / 3.85 | explore | 0 |

Local credit passes **4/4**, advantage **−1** each. Closed retention is still
unmet in this one-departure diagnostic.

## Is the observer failure resolved?

The logger discarded calls with detached logits. Production enables the
compose policy graph only at a sampled compose departure; narrowing-only
choices can leave every compose call detached. Key `"1"` identifies epoch 1.
The regression exercises this native `policy_grad=False` path without
pinning a seed. Logging now counts all training calls and reports gradient
calls separately; policy gradients and the certificate are unchanged.
[Before](diagnostic/observer-before.log), [after](diagnostic/observer-after.log):

| Check | Before | After |
|---|---|---|
| Detached compose call | `KeyError: '1'` | 1 observed call, 0 gradient calls |
| Existing ordinary-batch observer certificate | Failed in rejected R2 sweep | Pass, unchanged assertion |

## Does the corrected objective retain meaning and identify atoms?

Two gate-on branches only, from the accepted R/S/T stages-1–4 checkpoint.
Same seed 20261010, revision-3 corpus, learning rate .003, 144 training and
48 held-out documents per branch. The existing held-out pass observes
closed norms and kept programs. Identification requires an active column
with |cos| ≥ .9. No extra warm-up, identity branch or held-out replay.

| Branch | Nonzero closed meanings | Norm min / mean / max | Identification | Minted columns | Merged pairs | Kept programs |
|---|---:|---:|---:|---:|---:|---|
| Factorial | pending / 48 | pending | pending / 240 | pending | pending | pending |
| Confounded | pending / 48 | pending | pending / 240 | pending | pending | pending |

## What does the final sweep show?

Pending one full sweep after the two measurements. The supervisor retains
its memory and timeout bounds; the commit is the source record.

Carried, not remeasured: relation trace **2/4**, **43** unadmitted leaves,
two-word CE **.031999–.032067**, four-word CE **.016526–.016706** (both gates
**.01**), free readback **0/4**. Identity waits for nonzero closed meaning;
no rule is retired. The parts-of-speech pass follows this repair.
