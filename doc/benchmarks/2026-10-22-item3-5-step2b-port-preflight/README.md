# Step 2b assertion-port preflight — 2026-10-22

**Status: stopped for Alec's two decisions in [spec §26](../../plans/2026-10-21-step2-one-ltm.md#26-follow-up-review-two-decisions-for-alec--2026-10-22).**
No 2b production/test edit, implementation freeze or gate run. The
handoff's assertion-change rule and §22's explicit stored-mode rule
require the stop; the same-round setup exception does not apply.

The requested repository handoff is complete. Main is published through
`c39ef9f3ab9dc334fbee3b6e83d520a256423005`, including `d5fc19791` and
`61860c849` and the verbatim commit of Claude's seven documents, with the
exact requested message. [repository-publication.json](repository-publication.json)
verifies the paths, bytes, ancestry and remote main. No WikiOracle bump
for documentation alone; WikiOracle remains `baa2e7f`. All work now stays
on main, with no new branch. The later preflight commits are local pending
the decisions, not a 2b publication.

Production and tests still match published 2a (`0b030e3eb`).
[source-audit.json](source-audit.json) records that boundary. No test or
training run was made in this preflight, so there is no new failed run or
gate result. Every earlier failed run remains at its original source.

## Review inputs and reports

| Review | Input | Disposition |
| --- | --- | --- |
| [Initial actual Claude review](claude-review.md) | `76de86f7a` | P2 correct; amendments needed to P1/P3/P4/P5 and additional conflicts found. |
| [Actual Claude follow-up](followup/claude-review.md) | `a9409901c` | P1–P3 sound; exact refinements to P4–P7 and one occupation decision for Alec. |

Both reviews used Claude Opus in read-only mode. Their requests, complete
streams, stderr and run/input metadata are retained. Neither review ran
tests, edited source or approved an implementation for publication.

[amended-port-proposal.md](amended-port-proposal.md) incorporates the
follow-up corrections and states the two requested decisions:

1. Authorize the corrected existing-test ports P1–P7, excluding the
   occupation choice. This includes the specified typed fixtures, exact
   retained structure comparisons, checkpoint-validation changes,
   occurrence/no-seat expectations and pending-word reading ports, while
   preserving the other semantic assertions, capacities and tolerances.
2. Decide the occurrence-only argument's occupation. Claude recommends
   reading its position from the typed derivation and retiring this
   case's redundant symbol-keyed count/restore checks. The alternative
   records an occupation on the word and changes the bare-word rule.

No port is applied. The original `proposed-fixed-ports.patch` remains as
the reviewed historical proposal; its evidence-only P1 is superseded by
`proposed-fixed-ports-amended.patch`. Both patches were checked for clean
application only, not applied or tested. `ltm0-refs-audit.json` preserves
the first direct-column inventory; later row-export/root-reader and
closing cases are listed in the amended proposal. This is a scoped audit,
not a claim that all possible ports have been found.

Already authorized changes are not requested again: standalone posting
restore ports; the two evidence tuples; two-pass coarsening and its
snapshot port; the named live-parent role-0 reader; and retirement of
`test_centroid_moves_without_losing_identity_and_caps_containment`,
`test_adjacent_wholes_rewitness_compact_and_checkpoint`, and
`test_absent_adjacent_wholes_leave_the_lower_form`. None has been applied.
Every final port/retirement must be individually listed in the later
implementation receipt.

The 2b design remains §§20–23. Close-time minting ends in 2b under §11.1;
that round assignment is not reopened. The two R10 arrival notes remain
in 2c, and step 2's only full sweep remains at its end. Item 3.0 is not
reached. Implementation, its frozen-hash scoped gates, Claude's source
review, push and WikiOracle bump remain outstanding.
