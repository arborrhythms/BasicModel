# Check the amended 2b port dispositions

Read-only user-authorized Claude review: no edits, tests, experiments or
publication. Give recommendations, not authorization on Alec's behalf.

Your first review is retained at
`doc/benchmarks/2026-10-22-item3-5-step2b-port-preflight/claude-review.md`.
Read it and `amended-port-proposal.md` in that directory, then step-2 spec
§25 and only the cited code/tests needed to check the amendments.

This follow-up accepts your corrections. It closes two scope questions:
- §11.1 already ends close-time argument minting in 2b; do not ask Alec
  which round it belongs to again.
- The static `where` conflict is enough for an exact port proposal. No
  extra 2a run is needed before the required stop, and none has been run.

Please focus on:
1. Whether P1–P5 now preserve the intended obligations, including the
   kind-plus-evidence check, default mode without replacement state,
   cached form snapshot, existing-row graph snapshots, typed fixture
   meanings and owner-validated bad tags/shapes.
2. P6's limited close-time-mint audit: do the proposed OCC/content/no-seat
   expectations follow §11.1 for the six listed tests and two related
   positive-symbol checks? Identify any claim that would silently change
   another semantic assertion. Keep the native carrier certificate and
   all fixture capacities unchanged. No need for a broader test sweep.
3. P7: distinguish a root-reader representation port from a changed
   antecedent/content/pending assertion; assess the proposed raw pending
   word-reading port and the corruption-only cycle fixture while keeping
   normal writes acyclic.

Give a concise verdict on whether this is a sound port package to put to
Alec, list any remaining exact correction, and separate already-authorized
implementation consequences from additional test authorizations. Do not
reopen the settled owner, tags, address handle, mode removal, evidence,
postings/adjacency removal or staged-forgetting designs. Avoid running a
new audit outside the listed tests. No implementation is frozen or gated;
this is a proposed port package only, on main. Item 3.0 is not reached.
