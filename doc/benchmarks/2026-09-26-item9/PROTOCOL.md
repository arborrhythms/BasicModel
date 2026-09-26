# Item 9: declared before the learning runs

Seeds **0, 1, 2**; **64 optimizer updates** per seed and condition; batch
size **2**; **16 validation batches** before and after. Use the first **256
cached FineWeb documents**, their native document split and complete sentences
of at most eight whitespace words, packed into the existing W32 native
configuration. No sentence is clipped. An exhausted dataset or capacity is a
failed run, not a shorter learning result. There are four independent training
conditions: ordered prediction, shuffled context, context-free prediction, and
reconstruction only. Initialization, corpus, optimizer count and reconstruction
objective are shared. No seed is selected from results.

The starting configuration is the September 21 packed native experiment.
Remove retired capacity knobs and reserve 32,768 PartSpace and 65,536 concept
rows at construction; keep its native widths and operators. The reconstruction
basis limit is 512, enough for this input's byte capacity; report every actual
basis truncation. This is a bounded native model, not the full production-width
training run. Use the ordinary eager tensor loop, with no checkpoint autoload,
autosave, or private substitute encoder. Every reconstructed byte must come
from the owned forward program and candidate bank.

The eager dispatcher executes the existing condition/body tensors directly,
as in the archived native erosion measurement. It changes no cell operations
or gradients and is not a compiled-throughput result. An initial partial
ordered seed-0 attempt using graph-dispatched nested loops is preserved as a
diagnostic, not a 64-update result; it was stopped for roughly 48 seconds of
loop overhead per training batch before inspecting its learning scores.

Shuffling breaks the pairing between a context and its next sentence. It draws
from a bounded bank of **earlier** native contexts, excludes the current
context, and uses no targets or future inputs. The first empty-bank prediction
has zero context. Choices are retained for replay of the same context; bank
tensors detach at optimizer boundaries. The context-free arm zeros both
context values and masks. Only the reconstruction-only arm disables the
prediction and residual-policy losses. All other optimizer settings stay the
same. Report context-bank coverage and update counts.

Score held-out feature MSE and presence BCE, both on current encodings and on
the same pretraining encodings. Compare the separately trained controls at
equal updates. Also report target variance so representation collapse cannot
be called improved prediction. The ordered arm must beat both trained controls
in every seed. Its final reconstruction byte cost and serial sealed-root
discrimination on the fixed XOR/FineWeb probes must be no worse than the
reconstruction-only arm, allowing only numerical tolerance (1e-6).

For the native ordered arm, ask the ordinary thought controller about the same
held-out observed meanings at gains zero and one, with 64 work units available
per question. Restore the same memory and controller state between conditions.
Record every answer, Brier error against the presented positive assertion,
thought-step count and actual work. Compare work only at matching answer error;
equal failure is a null result, never a reasoning advantage. Preserve failed
parse/unsupported-question counts. Repeat the existing three-seed parsed-text
semantic comparison on this source: related versus unrelated continuations,
and the controlled thought comparison. It remains a separate frozen-encoding
study, not evidence of native joint learning.

Re-run the seven-update serial baseline and packed/single reconstruction parity
before interpreting learning differences. Preserve all failures and nulls.
Do not extend runs, change seeds, alter thresholds or tune controls after
seeing their scores. A persistent null goes back to Alec for a decision.

Harness correction before interpreting scores: the first shuffled seed-0
and context-free seed-0 attempts stopped at a runtime reconstruction flag
that the harness incorrectly called candidate truncation. That flag also
includes unavailable inverses and malformed reverse programs. The complete
runs retain all such rows and report unavailable operator names and actual
per-sentence candidate-limit use separately. No learning threshold, seed,
update budget or runtime cell changes for this correction.

Each process is capped at 8 GiB. Total concurrent reservations cannot exceed
24 GiB. Freeze runtime source throughout each run and retain source hashes,
configuration, commands, capped-process diagnostics and raw per-batch results.
