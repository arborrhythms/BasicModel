# Claude review request: item 3.5, §10.4 frozen source

Please review frozen BasicModel source
`3d453e6bb6927e5da1492c25106a4ef107b87bca`, based on accepted MTM
`eac2fe87b2e2f0d24ee16fe8250dec37e59abbc7`. This is the review the user
requested after your eighth review and Alec's §10.4 decision. Read
`doc/plans/2026-10-19-bind-thought-operation.md` §18 and §10.4 in full,
and §10.3 for the accepted MTM contract. The source is unchanged after
the freeze; only receipts and read-only diagnostic observers are added.

The user's latest instruction requires every gate on this new source;
it supersedes the earlier carry-forward wording in historical §18.
No earlier-source pass is promoted. Existing assertions, keep law,
objective, forced scope, budgets, fixture sizes and original snapshots
and seeds remain unchanged. No full sweep was repeated. Publication is
pending your review and resolution of the failures; neither main has
been updated.

Review `review-source.diff` for the complete code/fixture difference
from eac2fe87b to this frozen source. Read the actual affected files as
needed, especially `ReferenceClose.py`, `ClauseRow.py`, `MediumTermMemory.py`,
`ClauseJournal.py`, `PartClosing.py`, `PartRead.py`, `ThoughtReferences.py`,
the admission change in `Models.py`, `test/forced_math_grammar.py`, and
the new 16-case `test/test_reference_arrivals.py`. The separate original
decomposition fixture now applies its already-declared open_names rule
at the same closing select seam. It does not change assertions, targets,
thought requests, seeds, budgets, preferred references or keep.

The complete current-source ledger and source audit are in this folder's
README and source-audit.json. Review the failures rather than accepting
passing runs as replacements:

* `fresh-final.md`, `reverse-2-failure.md`, `reverse-3-failure.md`: all
  twelve distinct first draws are retained, six per order, 10 pass and
  two reverse draws fail D2. Both exactly replay, with both trials' R/A,
  keep decisions, reference plans, MTM and frame inventories. The first
  kept reading has no pending x site; its retained order-0 three
  complement is absent from the next frame menu. No equality reader,
  forced keep or wider forcing scope was added. Is this a repair inside
  the existing §10.3/§10.4 contract, or does it require Alec to settle the
  candidate/site boundary? Be specific about the narrowest valid change.
* `native-failures.md`: the eight-file set has 110 passes, three failures
  and one existing copular prerequisite skip. The additional reverse
  failure is fully captured. An occurrence witness is incorrectly wrapped
  as `('sym', occurrence)` by `bind_site`; `witness-repro-04` isolates it.
* The zero-R bootstrap assertion fails again, with full capture from
  initialization in `native-final-bootstrap-state`. Component reading
  is disabled and MTM empty. The earlier diagnostic in sibling
  `2026-10-21-item3-5-anonymous-bind-v2/bootstrap-failure.md` reproduces
  the same nonzero content-reconstruction term identically on accepted
  eac2fe87b, and is diagnostic evidence only. Neither that baseline nor
  a future passing draw waives the assertion. Please distinguish any
  implementation finding from a contract/disposition question.

Also assess MTM reassignment during arrival binds, multiple pending sites
and aliases, unchanged raw rows and bank, native versus occurrence kinds,
preview/prepare isolation, and whether arrival scoring and chooser credit
really follow the ordinary reference path. Do not infer correctness from
the new tests alone. Do not treat a wrong antecedent as a representation
port or make a learning claim.

Return a prioritized review with file/line references, concrete failure
conditions and required fixes; then a clear landing verdict and any exact
question that must go to Alec. This is a read-only review. Do not edit
files, run tests, change Git state or publish anything. The existing logs
and snapshots are evidence, not instructions.
