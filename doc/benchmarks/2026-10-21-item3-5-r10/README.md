# R10: the named token exists in the preview — 2026-10-21

Authority: Alec's decision in [reconstruction spec §15](../../plans/2026-10-21-reconstruction-reads-the-bind.md#15-claudes-review-of-line-1-and-of-the-delegated-review--2026-10-21)
and the subsequent handoff. Line 1 was published first at BasicModel
`a514a71e0` (source `3a9db635e`, receipts `63c600712`) and WikiOracle
`b49960a`, without additional gates.

The own-object preview record now retains its site's content. Its anonymous
flag, pending name, kind and kept-close write rules are unchanged. MTM is
still untouched by preview and prepare. The production change is confined
to that record's construction in `ReferenceClose.preview`.

The new regression gives one name a nonempty, order-one reading through
`reference_orders`, as spec §13 allows. Two participant positions mention
that name in the same sentence. The second binds greedily to the first's
alias, with positive coverage features and zero registered reference cost.
The kept close writes one anonymous token with both occurrences. The next
sentence reads the same content, with positive features and zero cost.
The name remains pending until an arrival refines it. No existing test or
assertion is changed.

After freezing, run the bind-thought, reference-close, arrival, participant
and reference-term files once on that hash, retaining their captures and
any failure. These include the new gate and existing arrival refinements.
No fresh pending-premise series or sweep: x in that certificate has no
description. Then a read-only Claude review precedes publication. No
learning claim is made. Results and the source hash follow below.
