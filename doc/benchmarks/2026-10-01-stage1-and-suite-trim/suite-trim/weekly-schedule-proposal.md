# Weekly slow-test proposal

Proposed to Alec: Sunday at 03:00 local time, using launchd. The proposed plist
is `org.wikioracle.basicmodel.slow-tests.proposed.plist` in this receipt. It is
**not installed or loaded**. The target uses the existing .venv, so it cannot
advance the held environment baseline.

The target selects every central or historical RUN_SLOW gate and the three
RUN_MPS_SLOW cases. Ordinary cases use 8 GiB per worker; only the two native
production objective arms use 24 GiB, one at a time. The aggregate reservation
is 24 GiB, and the per-worker deadline stays 30 minutes. Each case is attempted
once. A resource stop remains a process failure; the dispatcher continues only
unattempted cases. Attempt counts and completed pytest reports are distinct,
so full attempted coverage does not imply a passing run. The three MPS
cases switch to MPS themselves; the CUDA availability skips remain intact.

A dated directory in tmp/slow-tests contains the detailed bounded receipts and
a small record.json with date, commit, selected/attempted/completed counts, outcomes,
failures, duration, peaks and ceiling policy. latest.json points to the newest
record by copying its metadata, including incomplete/failed status. A complete
sweep warns if the record is missing, older than seven days incomplete, or failed.

The first complete weekly selection began on the immutable review-13 source
copy while part 4 continued. Its record is kept separately from the closing
candidate's full sweep and never substitutes for that sweep.
