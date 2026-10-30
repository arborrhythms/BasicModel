# Priming as location-based diffusion in the one conceptual space (Claude, 2026-10-26; design, with decisions for Alec)

Item 3, its own round after item 3.0 lands. This is the successor to the
stratified priming surface of [Orders §9](../Orders.md#9-per-order-conceptual-spaces-ramsified-over-one-symbol-tree-the-stratified-priming-surface-alec-2026-10-20-decided),
which [Orders §10](../Orders.md#10-orders-are-learned-definitions-raise-three-compositions-lower-alec-2026-10-21-decided-built-in-item-3)
withdrew on 2026-10-22. Claude's design list of 2026-10-26 called the
stratified surface "decided, not designed for code"; that was wrong, it
was withdrawn, and this file is the correction. Nothing here is part of
item 3.0.

## 1. The record

**Standing (Orders §10, 2026-10-22).** One conceptual space; every
order's concepts are placed by one objective, raise the meet ratio with
what the surface activates and lower it against negatives
([FutureWork, placement](../FutureWork.md#placement-from-the-registered-priming-surface-one-metric-for-sbow-diffusion-and-bind-alec-2026-10-20-proposed-with-the-objective-to-decide), decided); all symbols
in the one mereological tree; membership recorded as Part edges (the
LTM1 rows whose two tags are `SYMBOL`) and read by edge, never by
distance; the spread crosses orders. Withdrawn: the per-order cubes, the
"point in every order's space at once" read, `priming_row_orders` as a
read restriction.

**Alec (2026-10-26).** "The priming surface is the connectionist
algorithm over all symbols. It carries the activation of all symbols.
Initially, the activation comes from those symbols that are present in
the input, and it propagates that activation via diffusion. [The
geometry of centroids over the mereological towers] is not relevant to
the interconnections. What is relevant is whether the orders are
ramsified. One option for implementation, rather than a square matrix
over the interconnection weights of the concepts associated with those
symbols, is just the distribution of those vectors in conceptual space
(which gives rise to that pairwise matrix). I think that is most
efficient, and it expresses diffusion as location based (radially,
since amplitude over concepts is reserved for certainty)."

**Claude's reading.** The weights between symbols are not stored. The
placed meaning codes, `MereologicalCodes.placement [V, meaning_pairs]`,
are the network: the coupling of rows $i$ and $j$ is a function of their
two positions, and the diffusion step applies that coupling to the
activation without forming the $[V, V]$ matrix, as `Priming.geometric`
already does with `placement @ (placement[S].T @ r[S])`. The surface
value is certainty, the excess over neutral; the codes carry location
only, since `placement_step` returns unit-normalized codes, so no row's
amplitude enters the kernel. Under §10 the orders are ramsified in LTM
(LTM0 occurrences, LTM1 relations, the derived LTM2) and not in
conceptual space, so one kernel over one `placement` reaches every order,
and a kind is reached from its members exactly as far as placement has
put it beside them.

## 2. What exists, in code names

- `ConceptualSpace._priming_boosts [nStreams, nVectors]`, neutral `1.0`,
  preallocated; `_priming_surface(V, batch)` raises on any resize;
  `_priming_target()` is the canonical holder; the columns are
  `similarity_codebook` rows of every order (`_priming_dim()`,
  `_csw_row_of(symbol)`).
- `prime_seen(rows, bump, active_rows)`: decay `surface * d + (1 - d)`
  (`_priming_decay`, config `primingDecay` 0.9); `Priming.diffuse` along
  `_priming_edges()`, which is `Priming.part_edges(owner)`: the taxonomy
  rows (`parts.ref_kinds.eq(parts.SYMBOL).all(-1)`), rebuilt on
  `parts.symbol_revision` into `_priming_edge_src`, `_priming_edge_dst`,
  `_priming_edge_weight` (`[2 * partsCap]`), degree-normalized,
  conservative, with the feasible-fraction guard and the
  `_last_priming_diffusion` ledger (`primingSpread` 0.05, `_priming_spread`);
  then the bump `+1` at `rows`; clamp at 0; returns
  `relevance_weights(batch=B)`.
- `prime_desire(rows, valence, bump)`: signed bump, floor 0; returns
  `relevance_weights`.
- `Priming.geometric(owner, surface)`, read-only: `r = (surface -
  1).clamp_min(0)`, `S = r > 0`, `scores = (r @ A) @ A.T` with `A =
  placement`, zero at `S` and at rows with `placement_witnesses == 0`;
  the `geometricNeighbours` (8) rows with the largest score receive
  `geometricSpread` (0.05) of `r_S`'s total in proportion to score, taken
  from `S` in proportion to `r`; never written back.
- `relevance_weights(batch)` is `_bounded_priming(surface +
  geometric(surface))` (`primingMaxBoost` 2.0); `priming_weights` aliases
  it. Readers: `Models` (5618, 6691), `ModelAttention` (218),
  `SentenceAttention` (132), `project_priming_to_towers` (Spaces 16808),
  `_assemble_relevance_priority`, the reading scope.
- `placement_context(batch)` is `(_priming_boosts[batch] -
  1).clamp_min(0)` of the stored surface. Readers:
  `MereologicalCodes` (placement, 160; `collapse_audit`, 260),
  `MemoryConsolidation._fold` (190). The priming guard (Architecture):
  codes never read energy that was computed from codes; today it holds
  because `geometric` is never stored.
- `priming_row_orders()` has no reader.

## 3. The design

### 3.1 The kernel is the placement

The coupling is the metric placement raises and `bind`'s content term
reads, the meet ratio, so there is one metric (FutureWork, placement,
decided). It is applied as a field, never as a matrix. For one stream
with excess $r$ and active set $S = \{j : r_j > 0\}$:

```text
z   = amax over j in S of (r_j * A[j])                    # active field, [P]
c_i = sum_d min(A[i, d], z_d) / sum_d A[i, d]            # coverage of row i, [V]
```

with `A = placement` and `c_i = 0` for `i` in `S` and for rows with
`placement_witnesses == 0`. The active field is the union of the active
codes, each scaled by its certainty, the same representation the knowing
field uses (coordinatewise max); `c_i` is the fraction of row $i$'s code
the field covers, in $[0, 1]$, and depends on $i$ only through its
position relative to the field. Cost $O((|S| + V) P)$ per stream, with
$P$ = `meaning_pairs`; the $[V, V]$ matrix the positions give rise to is
never formed. Placement itself is unchanged: `placement_step` moves a
code toward the codes of the activated set and away from negatives, so
what placement learns is exactly what the diffusion reads.

### 3.2 The step, stored

Inside `prime_seen`, in place of the edge diffusion, between the decay
and the bump; and inside `prime_desire`, after its bump:

```text
r = (surface - 1).clamp_min(0);  S = r > 0                # per stream
c = coverage(A, r)                                       # 3.1
N = topk(c, geometricNeighbours)                         # bounded fan-out
budget = geometricSpread * r.sum(-1) * (c[N].sum(-1) > 0)
change[N] += budget * c[N] / c[N].sum(-1)
change[S] -= budget * r[S] / r[S].sum(-1)
fraction = feasible fraction, as diffuse computes it today
surface += change * fraction
```

Conservative: the stream's total is unchanged (summed in double), the
feasible-fraction guard prevents overdrawing a hub, and
`_last_priming_diffusion` keeps `before`, `after`, `budget_fraction` and
`neighbours` (the count of rows that received energy; `graph_edges` and
`degree` go). Because the surface persists through decay, energy spread
at one sentence is active at the next and spreads again: the connectionist
propagation Alec describes is the stored surface iterated once per
`prime_seen`, one hop per sentence, without a hop count.

### 3.3 The ledger placement reads

A second preallocated buffer `_priming_seen [nStreams, nVectors]`,
neutral `1.0`, beside `_priming_boosts`, written by `prime_seen` with the
decay and the bump only: no diffusion, no desire. `placement_context(batch)`
returns `(_priming_seen[batch] - 1).clamp_min(0)`; `collapse_audit` and
`_fold` read through it. The guard holds by construction, as it does
today, and the gate checks the ledger byte for byte. What placement reads
is the registered sentence with recency weighting, the window
co-occurrence of FutureWork's identity; second-order similarity comes
from the objective itself, since two words that never co-occur but share
a context code are both moved toward it.

### 3.4 Reads

`relevance_weights(batch)` returns `_bounded_priming(surface)` with no
read-time term; `Priming.geometric` is gone, its work done by 3.2. Every
consumer keeps its interface and now reads the propagated surface.
`prime_seen` and `prime_desire` return `relevance_weights(batch=B)` as
today.

### 3.5 Across orders

Nothing is added. A kind's row is placed by the same objective from the
contexts it occurs in, which include the sentences stating membership:
when "Felix is a cat" is registered, both rows are in the seen set and
each is placed toward the other at that close, so the location channel
reaches the kind from the member to the degree placement has learned.
Membership is read by edge (`ConceptIndex`, the LTM1 `SYMBOL`/`SYMBOL`
rows) and that is unchanged.

### 3.6 What retires

`Priming.part_edges`, `Priming.diffuse`, `Priming.geometric`,
`ConceptualSpace._priming_edges`, the `_priming_edge_src`,
`_priming_edge_dst`, `_priming_edge_weight` buffers with
`_priming_edge_count`, `_priming_edge_revision`, `_priming_edge_rebuilds`,
`_priming_spread` and the config `primingSpread`; `priming_row_orders()`
as housekeeping. The two diffusion knobs are `geometricSpread` and
`geometricNeighbours`; `primingDecay` and `primingMaxBoost` are
unchanged. Step 2 spec §12 decision 1 (taxonomy edges conduct, default
on) is superseded under D1.

## 4. Decisions for Alec, with recommendations

- **D1. Channels.** Location only (recommended), or location plus the
  taxonomy edge channel. The edge conducts a stated relation from the
  row's write; location conducts it after placement has moved the pair,
  at `placementRate` 0.1 per close in which both are active. For: one
  mechanism, no rebuild on `symbol_revision`, no `[2 * partsCap]` buffers,
  and a stated relation is also registered context. Against: a stated
  relation primes less on the first sentences after it is stated.
- **D2. Kernel.** Coverage by the active field in the meet ratio
  (recommended; the one metric with placement and `bind`), or the cosine
  `A (A^T r)` as today. Same cost; the cosine is a second metric.
- **D3. Stored or read-only.** Stored propagation with the seen ledger
  for placement (recommended), or the present read-only one-hop term on
  every read. For: computed once per sentence rather than on each of the
  several reads per forward; the surface carries the activation; the
  guard moves to the ledger and stays byte-exact. Against: one more
  `[nStreams, nVectors]` buffer.
- **D4. Fan-out.** Keep `geometricNeighbours` as the bound (recommended),
  or add a coverage radius with the count as a cap.

## 5. Gates (one freeze; sized to the change)

1. *Conservation.* On a fixture with three streams, two active,
   `before == after` in double within `1e-9 * V` for `prime_seen` and
   `prime_desire`.
2. *Guard.* `_priming_seen` is byte-equal before and after the diffusion
   part of `prime_seen`; `placement_context(b)` equals
   `(_priming_seen[b] - 1).clamp_min(0)`; `prime_desire` leaves
   `_priming_seen` unchanged.
3. *Kernel.* On a six-row fixture with hand-set `placement`, the coverage
   equals a direct computation; rows in `S` and unplaced rows receive 0;
   the `topk` set matches.
4. *Reach across orders.* A kind row placed inside the member's
   neighbourhood receives energy when the member is bumped; a far row does
   not.
5. *Persistence.* A row two hops from the bumped row receives energy only
   at the second `prime_seen`, scaled by the decay.
6. *Retirement.* No `_priming_edge_*` attribute exists; `Priming` exposes
   none of the three retired functions; two consecutive
   `relevance_weights` reads are equal and change nothing (the step 2
   read-guard test ported).
7. *Standing.* Step 2's gates unchanged on the hash: the placement
   objective, `collapse_audit`, the cost certificate; the cost
   certificate's allocation read shows no `[V, V]` tensor.

## 6. Ports and documents

Tests asserting taxonomy conduction, `graph_edges`, `degree` or
`membershipPriming`'s retirement port to the location step with
before/after text in the receipt. `primingSpread` leaves `model.xml` and
Params.md; `geometricSpread` and `geometricNeighbours` are documented as
the diffusion knobs. Architecture's "conceptual channel computed on read,
never stored" and the guard's wording are replaced exactly, naming the
ledger. Orders §10's "cross-order diffusion stands" gets one sentence:
by location, in the one space.

## 7. Where, and what this leaves alone

Its own round after item 3.0 lands; independent of the sorting round
(form side) and of the `intuit` and thought designs, which follow it.
Order-2 kinds and kind-level verbs sit on the one space and the Part
edges and are unaffected beyond that; their sections come with their
rounds. The symbolic heat path (`Taxonomy.build_semantic_heat`,
`_active_heat_set`, the `<attention>` and `<symbolicPriming>` gates) is a
separate consumer and is not touched.

## 8. Decided (Alec, 2026-10-26), and the design as amended

**The parallel (Alec).** "There is a nice parallel in higher-order
concepts and higher-order symbols both being defined in terms of their
zeroth-order constituents, whether geometric or distributional." The
symbol side already has it: a higher-order symbol's form is the centroid
of its members' forms, weighted by their evidence (Orders §9, Centroids;
`centroid_forms`, `MemoryConsolidation._fold`). This design gives the
concept side the same rule: a higher-order concept's place is the centroid
of its members' places, weighted by the fuzzy membership the Part edges
record. Order 0 is the base case on both sides, geometric for the form
and distributional for the place.

**D1, decided: location only, with the edges as weights, not
conductors.** Alec: "0 is dictating centroids on the fuzzy membership for
both symbols and concepts at all higher orders. ... It situates types in
token space, so is meaningful primarily for contiguous sets." The kernel
reads positions only; the taxonomy edges conduct no energy; they place
the higher-order rows. In code names:

- *Which rows.* A row with member rows in the codebook (a class the raise
  created, whose members are symbol rows) is a *derived* row:
  `placement_derived [V]` bool, preallocated, set when `ConceptIndex`
  records its first membership. Every other row, an order-0 symbol and an
  individual alike, is placed by context with `placement_step` as step 2
  built it: an individual's zeroth-order constituents are its occurrences,
  and their contexts are its context. `placement_step` skips derived rows.
- *The centroid.* `MereologicalCodes.derive_class_placements()`, once per
  sentence after the close's placement steps: for each derived row,
  `placement[c] = normalize(sum over members m of w_m * placement[m])`
  with `w_m` the member's net positive evidence on the membership row (the
  lanes, never netted below zero), and `placement_witnesses[c]` the count
  of members with `w_m > 0`. The membership list is the existing
  `Priming.part_edges` cache, kept and rebuilt on `parts.symbol_revision`
  into the `_priming_edge_*` buffers with direction (member, class) as
  `ConceptIndex` records the raise and weight `w_m`; definition rows
  (`DefinitionIndex`) are not membership and do not enter.
- *Contiguity.* The centroid is kept while every member with `w_m > 0`
  lies within the neighbourhood radius of §8 D4, `cos(placement[m],
  placement[c]) >= cos(radius)`; otherwise `placement_valid[c]` (bool,
  preallocated) is false and the row neither sends nor receives in the
  kernel. A discontinuous kind is then primed only by its own bumps, which
  is the "meaningful primarily for contiguous sets" of Alec's answer made
  exact. *Claude's call, to confirm:* the same radius serves the
  neighbourhood and contiguity, so one notion of "near" exists.

**D2, decided: the dot product.** Alec: "simple and fast for v. 1.0."
§3.1's coverage is replaced by today's form: `z = A[S].T @ r[S]` (`[P]`),
`s = A @ z` (`[V]`), zero at `S`, at unplaced rows and at invalid derived
rows; `cos_i = s_i / |z|`, which is the cosine between row $i$ and the
context direction since the codes are unit. Shares of the budget are in
proportion to `s`. The meet ratio stays placement's metric; one metric for
both is deferred, recorded here.

**D3, decided: stored, once per sentence, in the parallel pass.** Alec:
"this should already be happening this way." The edge diffusion is stored
today; the geometric term is computed on every read, several per forward,
and that is what changes. The one step runs in the `prime_seen` call of
`Models._prime_sentence_symbols(sid)`, batched over the streams, in the
order decay, bump, propagate, so the sentence's own words prime their
neighbours for the sentence's reading. The second call,
`Models._prime_seen_step` with the fired rows after the forward, bumps
without propagating, as does `prime_desire`; thought's serial bumps are
propagated at the next sentence's pass. `prime_seen(rows, bump,
active_rows, propagate=True)` carries the flag; `_prime_seen_step` and
`prime_desire` pass `False`. §3.2's "and inside `prime_desire`" is
withdrawn.

**D4, decided: radius with count, adaptive.** Alec: "If count is always
exceeded, shrink the radius (and vice versa)." Receivers are the rows with
`cos_i >= cos(radius)`, at most `geometricNeighbours` of them (the largest
`cos_i`; a fixed-shape `topk` over the masked cosines, so compiled shapes
do not vary). `_geometric_radius`, a preallocated `[1]` buffer in radians,
starts at `geometricRadius` (degrees, default 15, the angle
`recurrenceAngle` already uses) and adapts once per sentence after the
step from the mean receiver count over the active streams before the cap:
above the count, `radius *= 1 - geometricRadiusRate`; below it, `radius
*= 1 + geometricRadiusRate`; equal, unchanged; clamped to
`[geometricRadiusMin, geometricRadiusMax]` (degrees, defaults 1 and 90).
`geometricRadiusRate` default 0.02, so only a sustained excess or
shortfall moves it. The ledger records `radius` and `receivers`.

**The step, as amended (replaces §3.2).**

```text
Priming.propagate(owner, surface, active):                # once per sentence, [B, V]
  r = (surface - 1).clamp_min(0);  S = r > 0
  ok = placed & (~derived | valid)                        # senders and receivers
  z = (r * ok) @ A                                        # [B, P]
  s = z @ A.T;  s[S | ~ok] = 0;  cos = s / |z|
  mask = cos >= cos(radius);  count = mask.sum(-1)        # before the cap
  values, N = topk(cos * mask, geometricNeighbours)       # fixed shape; zeros receive nothing
  budget = geometricSpread * r.sum(-1) * (values.sum(-1) > 0)
  change[N] += budget * values / values.sum(-1)
  change[S] -= budget * r[S] / r[S].sum(-1)
  fraction = feasible fraction, as today
  surface += change * fraction
  adapt radius from count.mean over active streams; write the ledger
```

Conservation, the feasible-fraction guard, the ledger (`before`, `after`,
`budget_fraction`, `receivers`, `radius`) and the seen ledger
`_priming_seen` for placement (§3.3) stand.

**Retirements, as amended (replaces §3.6).** `Priming.diffuse` and
`Priming.geometric` retire into `Priming.propagate`; `part_edges` and the
`_priming_edge_*` buffers stay as the membership cache of D1;
`_priming_spread` and `primingSpread` go; `priming_row_orders()` as
housekeeping. Knobs: `geometricSpread`, `geometricNeighbours`,
`geometricRadius`, `geometricRadiusRate`, `geometricRadiusMin`,
`geometricRadiusMax`; `primingDecay` and `primingMaxBoost` unchanged.

**Gates, as amended.** §5 gate 3 checks the cosine against a direct
computation and that the radius mask and cap select as specified. Two
gates are added: *centroid*, a class with two near members sits at their
normalized weighted mean, `placement_witnesses` counts them, and it
receives when one member is bumped; *contiguity*, a class with two
members farther apart than the radius is invalid and neither sends nor
receives, while its members still do. One gate on the controller: a
fixture whose active rows have more rows within the radius than the count
shrinks it over ten sentences and one with fewer grows it, both staying
inside the clamp. §5 gate 2 adds that `_prime_seen_step` and
`prime_desire` leave the step's ledger untouched. The rest of §5 stands.

**Still Claude's calls, to confirm or correct before the Codex text.** The
same radius for contiguity (D1); the propagation in the pre-composition
call with decay, bump, propagate as the order (D3); the controller's
defaults (D4). The Codex text is written when item 3.0 lands.

## 9. Claude's review of Codex's item 2.9 round (frozen `b031b11ee9f2`) — 2026-10-27: approved for landing once two document repairs are applied

**Receipt.** `doc/benchmarks/2026-10-27-item2-9-priming/`. The frozen hash
recomputes from `frozen-source.json`'s inline `source_snapshot` (sha256 of
the compact canonical JSON, `sort_keys`, separators `(',', ':')`) to
`b031b11ee9f279bca43eaeee002703724073b4d834780f726d995249946e3332`, and
all 917 snapshot files hash equal to the live checkout
`/Users/arogers/github/WikiOracle/basicmodel`; none is missing or differs.
The snapshot covers `bin`, `test`, `data`, `Makefile`, `README.md`,
`pytest.ini`, `requirements.txt` and the six protocol files; `doc/` is
outside it, and `document-hashes.json`'s four documents hash equal to the
live tree. Every `gates/*/run.json` and the native smoke carry the one
hash; the smoke started before the freeze was written and ran on the same
source. Base `91974ced9b`, the item 3.0 landing, verified against the remote.
No commit, no publication, no `RUN_SLOW`, no seed (none in the diff of
`bin/` or `test/`; `summary.json` records `seed_pinning: false`), no resize
(the three `_priming_edge_*` slabs keep their `2 * partsCap` shape; the two
new bool columns and the two `[nStreams, nVectors]` ledgers are constructed
once; the pointer gates assert it).

**Gate counts.** 1873 gate executions passed, 47 skipped, 0 failed; 1771
distinct cases.

| group | passed | skipped | note |
|---|---:|---:|---|
| core (the item 3.0 "unchanged sets" list, 152 selectors, all run) | 1623 | 36 | 26 slow marks, 10 absent phase artifacts |
| priming (the 7 ported files + 4 consumers) | 146 | 4 | the 4 are the pre-existing slow tower-projection tests |
| placement (step 2's objective and `collapse_audit`) | 6 | 0 | |
| packing | 1 | 3 | 3 slow native packing cases |
| XOR standing gate | 8 | 0 | 6 native passes; the 2 CLI cases failed on the first run for a missing `.venv` in the isolated checkout, before model construction, and passed on the replay against the same hash (`gates/xor-cli-runtime-replay`) |
| decomposition, zero and enabled arms | 0 | 2 + 2 | slow; original seeds, snapshot hashes and the 512/768 budgets recorded in `decomposition-deferred.json`, not executed |
| snapshots and budgets | 89 | 0 | |

Alec deferred the slow marks on 2026-10-27 ("Skip the run-slow tests, we
will run them later"); the 26 + 4 + 3 + 4 deferred cases go into the
November sweep and are not claimed here. The five scoped groups and step 2's
placement gates are the ones the hand-off named; the only selectors beyond
the 152-entry scoped list are those groups' own files.

**§5 and §8 gates, by test.** Conservation (§5.1):
`test_priming_diffusion.py::test_location_outflow_preserves_energy_codes_rng_and_inactive_streams`
(three streams, two active, `before == after` within `1e-9 * V` in double;
inactive stream untouched; codes and RNG untouched). Guard (§5.2 and the §8
addition): `test_seen_ledger_guard_decay_bump_desire_and_fired_rows`:
`_priming_seen` byte-equal across `propagate`, `placement_context(b)` equal
to `(_priming_seen[b] - 1).clamp_min(0)` on all three streams,
`prime_desire` leaves the seen ledger, the step ledger and the radius
untouched and adds exactly its bump; `BasicModel._prime_seen_step` run as
the real method bumps the seen ledger and leaves the step ledger and radius
untouched; two consecutive `relevance_weights` reads equal and change
neither ledger. Kernel (§5.3 as amended): `test_six_row_cosine_mask_and_fixed_cap_match_direct_computation`
(cosine against a direct computation, rows in `S`, an unplaced row and a
derived invalid row at 0, the radius mask and the fixed-shape `topk` of 2
observed through a monkeypatched `topk`, shares in proportion, debit from
`S`). Reach across orders (§5.4): the class row receives when its member is
bumped in `test_class_centroid_weight_witnesses_cross_order_reach_and_context_skip`;
the row at 80 degrees in the conservation gate receives nothing.
Persistence (§5.5): `test_priming_energy.py::test_successive_primes_propagate_further`
(row 5, two hops from row 3 on the placed chain, receives only at the second
`prime_seen`, with `0.8 * 0.5` of the decayed excess). Retirement (§5.6):
`test_retired_functions_and_surface_methods_are_absent`,
`test_retired_spread_attribute_and_configuration_are_absent`,
`test_priming_energy.py::test_priming_row_order_read_is_retired`, listed in
`retirement-certificates.json`; `part_edges` and the `_priming_edge_*`
buffers retained as §8 D1 requires. Standing (§5.7): the placement group on
the hash; `test_cost_certificate_has_no_pairwise_inventory_tensor` records
all 79 tensor outputs of one `propagate` at `B = 3`, `V = 257`, `P = 7`
under a dispatch mode, none of shape `[V, V]`, every allocation at most
`max(B V, V P)` elements. Centroid (§8): the class sits at
`normalize(placement[a] + 0.5 placement[b])` after a `truth_minus = 0.5`
witness on b's membership, `placement_witnesses[c] == 2`, order 1,
`place_rows` skips it. Contiguity (§8): `test_noncontiguous_class_cannot_supply_field_or_receive_but_members_can`.
Controller (§8): `test_controller_uses_active_mean_before_cap_for_ten_sentences`
(5 in-radius rows against a count of 2 shrinks by `0.98` per sentence, 1
grows by `1.02`, 2 holds, over ten sentences; inactive streams do not dilute
the mean; the clamp holds at both ends; the buffer pointer is constant).
One step per sentence (§8 D3): `test_one_precomposition_step_and_one_centroid_close_per_sentence`
on a two-stream native forward records `sentence`, `step` (shape `[2, V]`,
both streams active, the registered words already bumped when the step
runs), `centroid`, in that order and once each. Construction errors fail
loud: `test_bad_geometric_construction_fails_loud` over the six knobs.
Checkpoints: `test_class_flags_roundtrip_with_fixed_shapes`.

**Code against §8, read in full.** `Priming.propagate` is the pseudocode:
`r = (surface - 1).clamp_min(0)`; `ok = (placement_witnesses > 0) &
(~placement_derived | placement_valid)`; `z = (r * ok) @ A`; `s = z @ A.T`
zeroed at `r > 0` and at `~ok`; `cos = s / |z|`; mask `cos >= cos(radius)`
(with `cos > 0` in addition, which only excludes exact zeros at a radius of
90 degrees); `count` before the cap; one fixed-shape `topk` of
`min(geometricNeighbours, V)`; `budget = geometricSpread * r.sum(-1) *
(mass > 0) * active`; receivers in proportion to the selected cosines,
donors in proportion to `r`; the feasible fraction as `diffuse` computed it;
totals in double on CPU; ledger `before`, `after`, `budget_fraction`,
`receivers`, `radius` (the radius used, before adaptation); then the
controller on the mean of `count` over active streams, clamped. `configure`
reads the six knobs once at construction into `_geometric_spread`,
`_geometric_neighbours`, `_geometric_radius_rate`, the min and max in
radians, and the `[1]` buffer `_geometric_radius`; `model.xml`, `model.xsd`
and Params.md carry them; `primingSpread` is gone from all three and from
`Models.py`. `Space.prime_seen(rows, bump, decay, active_rows,
propagate=True)` decays and bumps `_priming_boosts` and `_priming_seen`
alike, then propagates on the flag; `Models._prime_sentence_symbols` passes
`True`, both `_prime_seen_step` calls pass `False`, `prime_desire` bumps
`_priming_boosts` only. `placement_context` reads `_priming_seen`;
`relevance_weights` is `_bounded_priming(surface)`; the hard reset clears
both ledgers. `part_edges` caches `(member, class, (truth_plus -
truth_minus).clamp_min(0))` for the LTM1 `SYMBOL`/`SYMBOL` rows whose parent
is a `('part-base', class)` key of `relate_idx`, so definition rows and
occurrence chains never enter; it rebuilds on `parts.symbol_revision`, which
now also advances when a `SYMBOL`/`SYMBOL` row's evidence changes (`LTM1`
and the `LTM0` truth setter). `ConceptIndex.index_part_row` marks the class
row derived and invalid at its first membership and expires the cache.
`MereologicalCodes.derive_class_placements` runs once per close from
`Models` after every active stream's placement and promotion, lower orders
first, writes the normalized weighted centroid, the witness count and the
contiguity flag against `cos(_geometric_radius)`; `place_rows` skips derived
rows. The two flags are saved in `placement_extras` and the state dict;
version-1 payloads without them are reconstructed from the raise records
with no resize. Documents: Orders §10 has its one sentence; the step 2 spec
§12 decision 1 is marked superseded by append; Params.md has the six knobs.

**Observations, recorded, no repair.** (O1) A member without a placement
has a zero code, so its cosine to the centroid is 0 and the class is
invalid until every member with `w_m > 0` is placed; `placement_witnesses[c]`
counts the member regardless. This is the spec read strictly and is the
right v1 behaviour: an unplaced member is nowhere, so the class is not
contiguous. (O2) The decay is written `1 + (surface - 1) * d` instead of
`surface * d + 1 - d`, algebraically the same, so neutral stays exactly
`1.0` in float32; accepted. (O3) `Priming.configure` runs in every
`Space.__init__`, so every Space carries the radius buffer; only the
canonical concept target propagates, so nothing else reads it; accepted.
(O4) `BaseModel.priming_spread` survives as the gain of
`project_priming_to_towers` (`Models.py` 6729), a reader of the retired
`primingSpread` that §2 did not inventory; it now reads `geometricSpread`
at the same default, so the tower projection moves with the diffusion
budget. One knob is right for v1; the Params line is repair R2. (O5)
GradientFlow.md and Orders.md §§9–10 still name `primingSpread` and
`priming_row_orders` as history; they stay. (O6) Codex's receipt is honest
about the two environment failures in the XOR group and about the deferred
slow cases; the preliminary `development/core` preflight is excluded from
the gate results.

**Repairs, documents only, outside the frozen hash, no re-run.**

- **R1. Architecture.md: the dated record was deleted, not replaced
  exactly.** The round removed the paragraphs "Amended 2026-10-20 (Alec)",
  "Amended 2026-10-22 (Alec; Orders §10)" and "Guard against the feedback
  loop (Alec's question, 2026-10-20)" with Alec's quotation and the Levy and
  Goldberg 2014 and Qiu et al. 2018 grounding, and wrote three new
  paragraphs in their place. §6 asked for two exact replacements inside
  those paragraphs. Repair: restore the three paragraphs from `HEAD` word
  for word, then make these exact replacements inside them: in the
  2026-10-20 paragraph, "computed on read against the active field and
  STM's recent rows, never stored, and the one that placement must not
  read" becomes "computed once per sentence by `Priming.propagate` and
  stored in `_priming_boosts` (item 2.9, 2026-10-27; read-only until then),
  and the one that placement must not read"; in the 2026-10-22 paragraph,
  "(Claude's reading of the decision, to confirm when item 3 builds that
  channel). The loop guard is unchanged: placement reads the surface before
  the conceptual channel, the chooser and attention read it after." becomes
  "(confirmed and built in item 2.9). The loop guard keeps its form:
  placement reads the seen ledger, the chooser and attention the propagated
  surface (item 2.9)."; in the guard paragraph, "One surface, two read
  points: placement reads it before the conceptual channel is applied; the
  chooser and attention read it after." becomes "Two ledgers, one read point
  each (item 2.9, 2026-10-27): placement reads `_priming_seen`, written by
  the decay and the seen bump only; the chooser and attention read
  `_priming_boosts` after `Priming.propagate`." The old "Priming diffusion
  (`<primingSpread>`, ...)" paragraph stays replaced by the round's "Priming
  diffusion (`geometricSpread`, default 0.05)" paragraph, and the round's
  "One conceptual space, stored location diffusion (item 2.9, 2026-10-27)"
  paragraph stays after it, with its "Guard against the feedback loop"
  companion retitled "Guard, as built (item 2.9, 2026-10-27)" so the
  heading is not duplicated. The one-line "Amended 2026-10-20 (Alec)" lead
  sentence the round kept ("a spreading field, not a gate on attention")
  goes, since the restored paragraph carries that amendment.
- **R2. Params.md:** the `geometricSpread` row adds "Also the gain
  `BaseModel.priming_spread` hands to `project_priming_to_towers`, which read
  `primingSpread` before item 2.9."

Both edits regenerate `document-hashes.json`; the frozen hash is untouched
because `doc/` is outside the snapshot. No gate re-runs.

**Landing.** Approved on `b031b11ee9f2` once R1 and R2 are applied. Alec
commits, or tells Codex to: one BasicModel commit of the tree with the
receipt's generated evidence excluded as item 3.5 and 3.0 did; todo.md item
2.9 records the hash and this section; the WikiOracle submodule bump
follows. The deferred slow cases ride the November sweep. Item 2.8 is
independent and its spec is next; item 2.7's certificate still carries the
unforced 1-to-2 raise, which orders §70 reads as not supplied by either
existing path.

**Landed (2026-10-27).** BasicModel `17e9454763`, "Land item 2.9:
location-based priming diffusion and reviewed documents", on
`b031b11ee9f2` with the R1 and R2 document repairs of §9 applied.

**Superseded at item 2.5 round B (decided by Alec, 2026-10-28; thinking
spec §15).** The open-pass `intuit` step over the learned definition
manifold becomes the spreading mechanism, consolidating conceptual
expectation and priming into one operator; `Priming.propagate`, the
radius controller, the `geometric*` knobs and the seen ledger retire
then, while placement, the derived class centroids, `bind`'s content
term, MTM matching and the bank's selection keep reading positions.
This step runs as built until that round lands; item 2.7 does not touch
it.
