# Reverse draw 3: explicit mint leaves no pending arrival site

**Review correction:** the missing next-frame candidate is the later
failure to recover. A's reference credit first chooses the nameless mint:
R ties, but the named path costs 1 + (1 + log(8)), weighted by .1, while
the mint costs zero. The [complete cost decomposition](reference-cost-breakdown.md)
shows both trials in both failing streams, with no supplied answer target.
This operation/keep decision is the contract boundary identified by
[Claude's review](claude-review.md); scope, objective and keep stay unchanged.

Frozen source `3d453e6bb6927e5da1492c25106a4ef107b87bca`. The third
reverse first draw fails unchanged D2 at line 403: row 70 in stream 7,
occurrence `('ltm', 'address-v1', 7409830812677866703)`, remains pending
for x. This is a mechanism failure. No change follows it.

The [complete keep tables](fresh-final-reverse-3-tables.md) retain both
trials for both batches. In stream 7, the first sentence keeps explore:
R = .0360673778 for both; greedy A = .4079441726, explore A = 0. The
departure is 0. Explore applies `lower_unary` to x, producing an explicit
mint request with no forward identity. Commit opens a token whose kind
is 27. There is no pending x site for arrival bind to refine.

The next frame has Part root 28, no individual symbols, and activations
for the x token and the Part root. Its accessible record is only the
order-1 x occurrence `('ltm', 'address-v1', -8206164119154584367)`.
The order-0 `three` complement is retained but absent from the menu.
Both second-trial bind reruns therefore select None. A new anonymous x
token stays at kind None. Both second trials have R = .0360673778 and
A = .2000000030; tie:greedy keeps the failing reading.

`reverse3-replay-03` reproduces both batches exactly, including readings,
reference plans, R/E/A, keep, audit and MTM, and observes the frame
inventory without altering it. The [comparison](reverse3-replay-03-comparison.json)
records the equality. This is the same missing-candidate boundary as
[reverse draw 2](reverse-2-failure.md), reached through an explicit
order-1 mint rather than an order-0 field mint.

The eight-file set's separate reverse run also fails D2 after an explicit
mint of x in stream 2. Its first-draw capture is under `native-final-state`,
and [its two-trial tables](native-final-reverse-tables.md) are retained.
No equality reader, forced keep, wider forcing scope, new target or
assertion change is introduced to make these cases pass.
