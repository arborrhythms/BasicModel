# Native graded corpus protocol (fixed before the run)

This is Item 4's bounded native learning measurement, separate from the earlier
mechanism toys and from Item 0 and 5.5 acceptance gates. It uses the real small
canonical BasicModel and the ordinary sentence-level paired optimizer, with
no grammar labels, teacher trees, identity warm-up or forced correct choices.

The same 12 training sentences occur in each arm. Graded order presents two-word
modifier phrases, then three-word clauses, then four-word expanded sentences.
Each stage adds one word and one intended bottom-up composition. The observed
rounds and end depths are recorded: the intended construction is not assumed to
have been learned. The shuffled arm permutes the same six paired batches once.
Two seeds (19 and 23) start each arm from the same initialization. Both use six
training batches, learning rate 0.001, the canonical gain and exploration settings,
and the same held-out pair before and after training. No extra training follows
a failed gate. The common model is 16-dimensional with a 64-row concept bank,
STM capacity 4 and input width 32. The independent intra-sentence auxiliary is
disabled, so Item 4's shared head owns E.

Report whole-observation R, local E, actual compose rounds and final depths;
report changes to the chooser and shared predictor parameters. A successful run
must execute both paired trials and update both owners. A positive learning gate
requires held-out R to decrease in each graded seed and finish below that seed's
shuffled control. Zero or tied costs do not establish recovery. A failed gate is
reported as failed; this small allowance is a diagnostic, not a mature learner.

Wall-clock cap: 600 seconds; process-tree memory cap: 8 GiB. The harness writes
partial progress after every batch and returns a nonzero status for execution
errors. A timeout is incomplete, never a pass. The result names the exact source
fingerprint and preserves the dataset/order and per-batch records.
