# Development after Claude's review of 3d453e6bb

This is not the next frozen gate source. The complete twelve-draw,
eight-file, decomposition and MM evidence remains attributed to
[3d453e6bb](../2026-10-21-item3-5-anonymous-bind-v3/README.md), whose review
says not to land. No earlier-source pass is promoted.

The independent repairs here preserve every existing assertion:

* A thought bind keeps occurrence witnesses as addresses, wrapping only
  integer native identities as symbols.
* A native arrival without a retained occurrence refines the existing
  anonymous introduction through `mtm.bind(token, token, kind=native)`.
  It does not materialize the native candidate's working vector as a new
  LTM companion. The already-retained-occurrence path remains unchanged.
* An arrival with no selected candidate reports `no_selected_candidate`;
  the earlier anonymous token remains bound to its own address.
* Admission copies arrival records before adding commit audit fields, so
  the scored trial's record is not mutated.
* A historical posting can refine only a token represented by the current
  frame occurrence records. The guard uses records before the read-time
  split/merge pass, so a different working-group key does not hide an
  accessible token. Current-close sites are still considered once per site.
  The two native-arrival positive fixtures now include the pending site's
  frame alongside the native candidate's frame; all their assertions and
  capacities remain unchanged. An uncued same-stream site in a different
  document is covered separately.

The user explicitly requires one rerun per pending **site** and setting
kind only when it is None. Claude's suggested per-token deduplication and
kind overwriting are therefore not applied. Arrival choice_cost and chooser
credit stay unchanged. Arrivals currently do not enter the ordinary
delayed `_mentions` / answer-supervision paths; this asymmetry is declared,
not silently changed into a new objective or premise answer target.

At Alec's request, the forcing/objective decision is deferred to review
time, with the exact
[A decomposition](../2026-10-21-item3-5-anonymous-bind-v3/reference-cost-breakdown.md).
The destructive pooled-candidate transfer and bootstrap zero-R failure
also remain open. The proposed kind-only fix
for a pooled candidate needs care: preserving its kind alone does not
establish preservation of its distinctive occurrence content or identity.
No menu masking, field-seat identity, assertion change, forced keep,
forced-scope extension or objective change is made here.

| Development run | Result | Evidence |
| --- | --- | --- |
| `repairs-01` | 37 passed | Witness, native-arrival and audit repairs; the arrival, MTM and occurrence files. Before the frame-access guard. |
| `remaining-02` | 2 failed | Both the pooled-candidate ownership loss and uncued cross-document refinement reproduced before the guard. |
| `scope-03` | 20 passed | All 16 arrival cases plus four review regressions, with the frame-access guard. |
| `remaining-04` | 1 passed, 1 failed | The unchanged cross-document reproduction passes; the pooled candidate's earlier use still loses its token. |

These bounded runs retain their logs and source manifests, including both
red runs. They are development checks, not substitutes for gates on the
next final frozen source. The [review handoff](review-handoff.md) carries
the remaining questions and the exact source/evidence boundaries.
