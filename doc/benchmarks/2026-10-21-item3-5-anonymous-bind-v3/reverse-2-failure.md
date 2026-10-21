# Reverse draw 2: reference credit removes the site; the frame cannot recover it

**Review correction:** the missing candidate is the later failure to
recover. The earlier deciding mechanism is A's closing reference credit:
with R tied, the named path pays .407944173 while the field reading pays
zero and is kept. The [per-site and arrival breakdown](reference-cost-breakdown.md)
reconciles every term from the capture; answer cost is zero and all premise
targets are None. [Claude's review](claude-review.md) identifies that
operation/keep decision as the contract boundary. Neither the objective
nor forcing scope is changed.

Source `3d453e6bb6927e5da1492c25106a4ef107b87bca`. The second reverse
first draw fails the unchanged D2 assertion at line 403 of
`test_math_chain_section14.py`: occurrence row 46 still has forward name
`x` (12). This is a mechanism failure, not a representation port. No
production, fixture, assertion, budget, objective or keep-rule change
follows this result.

Full initialization, generator state, reads, failure store/expression and
both trials were captured on the first run. The [keep tables](fresh-final-reverse-2-tables.md)
show every document in both batches. `reverse-replay-01` reproduces the
failure with identical readings, programs, reference plans, field, costs,
keep decisions, word inventory, reference audit and MTM in both batches;
[the comparison](reverse-replay-01-comparison.json) records this. The
separate `reverse-frames-02` replay only observes the frame inventory.

The failing document is stream 2:

| Batch | Greedy R | Greedy A | Explore R | Explore A | Keep |
|---|---:|---:|---:|---:|---|
| `x is three` | .0360673778 | .4079441726 | .0360673778 | 0 | explore, by A |
| `y is x plus two` | .0360673778 | .2000000030 | .0360673778 | .2000000030 | tie:greedy |

The first kept reading explicitly mints a field-valued `x`; its `x is`
constituent has order 1, its `three` constituent has order 0, and neither
has a reference-close request. It creates no pending x site and no MTM
entry for this stream. The retained derivation is Part 23 = (22, 19):
22 contains `x is`, while 19 contains the actual `three` constituent.
Nothing in that equality is read as a binding.

The next frame has row 23, order 0, no native individual symbols, and
activations for the order-1 `x is` occurrence and the Part root. Its
`accessible()` result contains only occurrence
`('ltm', 'address-v1', -4296030677818230725)`, whose derivation is 22.
The retained `three` occurrence, row 10, is order 0 and is not an
activation. The Part root activation is a symbol address, not a located
LTM1 occurrence. Thus the bind menu has no candidate whose semantic head
is `three`; the forced open-name rule chooses no candidate, as it should
when its +30 named target is absent.

The second close opens the anonymous token at row 46,
`('ltm', 'address-v1', -2599378222445204962)`, with kind None. Its arrival
rerun also has no `three` candidate and leaves kind None. The row still
has its pending x marker and open referent when the certificate reads it.
This is not a stale membership, eviction, ambiguity reopening, or a
keep-tie choosing between a correct and incorrect second reading: both
second-trial menus lack the antecedent.

The second stage of the boundary is concrete: §10.4's anonymous-token path
works when there is a word site, but this kept first reading has no site;
§10.3's present frame activations do not expose its order-0 complement.
Changing the keep, forcing a wider grammar scope, reading an equality as
a binding, or treating a field seat as an individual would violate the
instructions. Those changes have not been made. The remaining declared
first draws and scoped gates continue on this same hash, with this
failure retained for Claude's completed review and Alec's disposition.
