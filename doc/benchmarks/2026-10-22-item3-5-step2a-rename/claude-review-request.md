# Review the frozen step-2a rename and the cull preflight

User-authorized review by Claude. Read only; do not run tests, edit files,
change decisions or publish. The user requires review before publication
and a dated spec question plus a stop for Claude and Alec when a new
contract issue arises.

R10 is published at BasicModel e243d7bdf / WikiOracle 2ffda78 after your
approval of corrected source 23a41540f. This request concerns the next
separate rename commit, eb6980e7de493e8b35791404851ef749acee6a08, based on
that landing. No column cull or behavioral change is included.

Read:
- doc/plans/2026-10-21-step2-one-ltm.md, especially §§10–14.
- doc/benchmarks/2026-10-22-item3-5-step2a-rename/README.md,
  source-audit.json, renamed-files.json, source-diff.patch,
  checkpoint-class-paths.json, frozen-source.json, gate-summary.json and
  the source-bound gate reports/captures as needed.
- The exact source files and tests referenced in §14, and
  cull_preflight.py / cull-preflight.json. Its first diagnostic setup error
  is retained in cull_preflight_attempt01.py and cull-preflight.log; it was
  not a production or existing-test failure.

The pre-rename audit scanned 91 saved files and found no old class-name
bytes or pickle globals. Registered phase artifacts are all null and the
retired artifacts were not read. Rename: Layers.TernaryTruthStore becomes
Layers.LTM0; PartStore.PartStore becomes LTM1.LTM1 in bin/LTM1.py. Every
bin/test/data change is exactly those identifier substitutions. Ownership,
layouts, attributes and checkpoint wire paths are unchanged; no alias or
shim, no assertions changed except class identifiers.

Required gates on this hash: eight-file set 113 passed / 7 existing skips;
regression files 85 passed. Pending-premise first draws: 0 bound / 8 open
forward, 0 bound / 16 open reverse. Full captures and keep probes retained.
No extra fresh series or full sweep; no evidence from earlier sources is
promoted.

Give two separate findings:
1. Is the rename-only source safe to publish on this evidence? Check the
   module/class paths, source audit, claimed gate scope and any missed live
   import/checkpoint dependence. Give a clear verdict and blockers, if any.
2. Verify §14's cull preflight against actual code. Does §10's conditional
   "already have the derived source" permit proceeding with any listed
   cull without a new representation or assertion port? Correct any false
   code facts in the preflight. Identify decisions needed from Alec;
   distinguish those from already authorized conditional deferrals. Do not
   treat a missing numerical consumer as authority to erase retained
   metadata or a passing rename gate as approval of an unimplemented cull.

Recommendation recorded for Alec: land the rename separately; retain the
unreplaced fields, with culls still explicit on step 2's line, until their
source and exact representation ports are reviewed. This is not a claim
that all of 2a or line 2 is done. Two R10 arrival notes stay in 2c, with no
change now. No learning claim. Please return a concise review with file
locations and a concrete recommended disposition for each new decision.
