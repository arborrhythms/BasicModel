# Step 2b — layout and ending close-time minting

Authority: [step-2 spec §§20–23 and §27](../../plans/2026-10-21-step2-one-ltm.md#27-claudes-review-of-the-2b-port-package--2026-10-22)
and Alec's 2026-10-22 handoff. P1–P7, the occurrence-only occupation
disposition and the listed implementation consequences are authorized.
The amended proposal and both preceding reviews remain in the sealed
[port-preflight receipt](../2026-10-22-item3-5-step2b-port-preflight/README.md).

Production base: published 2a **`0b030e3ebdeca87f9ebf80739237437c3c75ea8f`**.
Current review input: **`00bc915a98ebf072b87c0fb7bb9d54a831128b25`**, on
main. That commit preserves Claude's six incoming documentation edits
unchanged; it is not a 2b implementation freeze.

## Current stop: identifying a reading row

[Spec §28](../../plans/2026-10-21-step2-one-ltm.md#28-reading-row-discrimination--2026-10-22-reviewed-by-claude-for-alec)
records a new layout question. Reading rows and ordinary derivations can
have the same `(OCC, SYM)` tag pattern and can coalesce when their
arguments are identical. The two-tag layout does not persist which row
owns an occurrence's reading. Occupation reads, restore and staged
forgetting require that distinction.

The actual [Claude review](layout-preflight/claude-review.md) confirms
the gap and recommends a third `READING` value in the existing tag
column, subject to Alec. It also corrects the initial diagnosis: excluding
the current edge can avoid the claimed recursion in the small example,
and today's protection rule narrows the coarsening example. The decisive
issue is ambiguous or coalesced rows, not an observed failed execution.

The [request](layout-preflight/claude-review-request.md), complete stream,
stderr, model/run metadata and [source audit](layout-preflight/source-audit.json)
are retained. Review used `claude-opus-5-5` with read-only tools and exited
successfully. The copied runner's `review-input.json.scope` retains its
earlier assertion-port label; the saved request and source audit identify
this run's actual reading-layout scope. No test or production diagnostic
was executed.

## Source and port ledger

| Item | Current state |
| --- | --- |
| Frozen layout commit | Pending the §28 decision |
| Frozen mint-removal commit | Pending layout and its gates |
| Changed production/config files | None |
| Changed assertions | None |
| Fixture rewrites | None |
| Retired tests | None |
| Gates and failed runs on 2b | None; no gate has run |
| Publication / WikiOracle bump | Not performed |

Every implemented assertion port, fixture rewrite and retired test will
be listed here under its frozen source. Both frozen commits require the
eight regression files, the retained eight-file set and the files of
every changed module, with captures attributed to that hash. An existing
test or production failure still stops; no prior passing result is
promoted to this source. Claude's source review precedes publication.

The requested sequence remains layout with P1–P5/P7, then ending
close-time minting with P6. Work stays on main. The two arrival notes are
carried to 2c, and step 2's one full sweep remains at its end. Item 3.0
has not been reached. No learning claim.
