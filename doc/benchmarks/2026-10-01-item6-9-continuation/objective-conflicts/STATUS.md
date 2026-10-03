# Objective-conflict measurements — held for Claude

Alec requested stage 1 of spec §6 on October 1. The measurement observer has
been drafted, but **neither configuration has started**. The supervisor was
waiting for the source-matched full sweep; it was stopped before launching a
model when Alec said, “Let’s hear back from Claude first.”

The open question is §6(5): comparing training outcomes with step 5a versus the
cut requires one training run per arm, whereas the request says one run per
configuration. A single training run can compare both cost formulas at the
same final parameter state, but cannot establish the cut's training outcome.
Await Claude before running these measurements. No source, configuration,
threshold, guard, seed, or dependency has been changed for this probe.

`observe.py` and `run.py` are unexecuted drafts, syntax-checked only. They need
review before use. Do not mistake them for measurement evidence. In particular,
final-state trial answer measurements and any approved cut arm still need to be
made explicit in the harness. No results are claimed.
