# Step 2b assertion-port preflight — 2026-10-22

Main is published through `c39ef9f3ab9dc334fbee3b6e83d520a256423005`:
both preflight commits and the requested verbatim commit of Claude's
seven documents. No WikiOracle bump for documentation alone. Production
and tests still match published 2a (`0b030e3eb`). No new branch was made.

[Spec §24](../../plans/2026-10-21-step2-one-ltm.md#24-test-port-audit-before-2b-implementation--2026-10-22)
accepts §§20–23 and identifies additional existing-test assertions before
implementation: stored mode, the LTM1 column count, an obsolete switch in
a mixed certificate, remaining LTM0 refs checks, and malformed saved
postings in a complete-model test. Each has a recommended disposition.

`proposed-fixed-ports.patch` is a proposal only, with exact edits for
P1–P3. `ltm0-refs-audit.json` records the direct old-column accesses for
P4. `source-audit.json` confirms that no production/test edit or run has
occurred. The authorizations already given for standalone restore,
adjacency-only retirements, evidence and staged coarsening are not being
requested again.

The handoff's assertion rule and §22's explicit stored-mode rule require
this stop for Claude and Alec. It is not a design reconsideration, a
failed gate or the new-test setup exception. The review and response will
be retained here. Implementation and all gates remain outstanding on main.
