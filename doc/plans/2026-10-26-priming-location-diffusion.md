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
