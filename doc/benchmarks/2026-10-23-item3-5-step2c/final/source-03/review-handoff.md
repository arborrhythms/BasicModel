**Superseded:** source-03 completed with one captured failure; see [diagnosis](diagnosis.json). The corrected source-04 owns its gates anew.

# Step 2c: mediated review hand-off

For Alec to pass to Claude. Review is mediated through Alec; no direct
review request is sent by Codex. Publication of BasicModel main and the
WikiOracle bump, then step 2's one full sweep, follow the review.

**Frozen source:** `e4f61bd5b2ba4315f2e753119106c8a22fd17696`, on main.
**Published base:** BasicModel `148953f9b`, WikiOracle `175c101`, identity
source `65ab7fbe9` reviewed LAND in identity spec §34.
**Specification:** [step 2](../../../../plans/2026-10-21-step2-one-ltm.md),
§§11–13 and the code review/repair record §§73–81 under §39.
**Receipt:** [complete implementation and failure ledger](../../README.md).
**Evidence:** scoped checks are still running; final counts will be recorded
here before this hand-off is delivered. No review verdict is claimed.

This implements 2c on today's storage. Three distinct source documents
make a reading eligible for a concept. Three source sentences promote an
MTM token to a concept containing the union of its immutable mention rows;
the token keeps its key, kind, members and current VP value and gains its
symbol. Already seated non-word readings and native individuals need no
second seat. The actual source sentence is recorded at the close because
embedded occurrence addresses encode constituent positions; unavailable
legacy source evidence is not invented.

MTM expiry counts kept closes in each stream, refreshes from actual
mentions/refinements/events, and removes the document's entries at its
end. Several closes may share model time. Token retirement drops only
that identity; an event owned by another retained identity survives while
its row survives. Actual `forget(address)` remains row reclamation and
removes those event witnesses. A same-close capacity eviction is distinct
from a determined introduction that never closed; the latter still fails
loudly. No capacity or seat is reserved.

The definitions cache rebuild is certified after forgetting/restoration.
A standalone order-one composite gets a site through its declared head;
a surface-wrapped relation does not get an extra root site. Both carried
R10 arrival-plan notes are certified without a new choice or objective.
A captured unseated verb word with a unique DEF uses that reading for role
evidence; an ambiguous unavailable reading invents no evidence.

The [port ledger](test-port-ledger.json), [exact test patch](test-ports.patch)
and [assertion audit](assertion-audit.json) account for every port. Existing
Python assertions remain verbatim. The bank comparison excludes only
newly seated promotion rows and checks that every excluded row is a
promotion; every prior bank row and the existing LTM/Part snapshot checks remain exact.
The fixed-reading helper now returns its real conceptual owner, preserving
its trust/LTM-switch setup and all consumer assertions. No test retires.

Please review the complete [production patch](production.patch) and
[source audit](source-audit.json), with particular attention to source
counting, promotion identity continuity, expiry and event ownership, and
the standalone-site boundary. Today's derivation, readers, close-time
minting and three-slot rows stay. The LTM0/LTM1 layout and shared STM chunks
remain item 3's. There is no learning claim or full sweep in this receipt.
Item 3.0 has not been reached.
