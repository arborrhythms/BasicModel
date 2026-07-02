# Iterated Symbolic Loop — collapsing the SparseLayer families (DRAFT)

> **STATUS: DRAFT 2026-07-02 (design direction confirmed by Alec in the
> two-phase execution thread; NOT yet approved for execution).** Successor to
> [2026-07-02-two-phase-loops-sparse-relation.md](2026-07-02-two-phase-loops-sparse-relation.md)
> (P1–P5 executed). Alec's confirmed points: (1) the collapsed layer carries
> NO role-tags; (2) vine-as-wave iteration is the intended forward semantics;
> (3) self-edges are forbidden, longer cycles deliberately allowed — with the
> human-problems correspondence called out in the docs.

## Motivation

Strict ramsification (per-order families, dyadic capacity, order-$k$ reads
orders $< k$ only) buys guaranteed termination but costs three observed
defects in the landed two-phase implementation:

1. **Chain truncation.** A Gallistel vine link is `[whole=current,
   part=rest]`; at `symbolicOrder=1` the *rest* link is the SAME order as its
   parent, so its weighted edge is illegal and silently dropped — the tensor
   reading of a chain degenerates to `[whole-word, bias]` per link while only
   the discrete store keeps the sequence exactly. High-order vines need
   recursion depth the ramsified blocks cannot afford.
2. **Capacity fragmentation.** The dyadic split $N/2, N/4, \ldots$ starves
   long sequences: every link consumes a row at its own order's shrinking
   block, while flat order-1 links (Alec: "multiple order-1 relations") would
   share one large pool.
3. **Stratified one-shot composition trains poorly** (P4 gate observations):
   the snap's slot-mean readout is input-blind at init on small inventories,
   and the demux-fed pump attenuates early-stage gradients $\sim$100$\times$ —
   the single feedforward sweep through order blocks gives the symbolic side
   no chance to settle.

## Design

### One square untyped SparseLayer

- Rows = the whole concept inventory ($N$); columns = $[N \mid 1]$ — the
  concept activations plus the constant-1 EVERYTHING bias. **No role-tagged
  column blocks** (Alec, confirmed): a given concept can be a whole in one
  relation and a part in another; the edge set is exactly the binary relation
  *as a set* (the COO reading), and direction/order lives in (a) the discrete
  record store's ordered `[whole, part]` rows (sec 4c) and (b) the NESTING of
  reified rows (the vine), never in typed columns.
- **Self-edges are forbidden** (`add_edge(i, i)` raises): $x = \{x\}$ — the
  Quine atom — is the degenerate self-reference every classical paradox
  (liar, Russell, Grelling) is built on. **Cycles of length $\ge 2$ are
  deliberately allowed** — see "The human problems" below.
- The per-order symbol families, the role blocks, `cs_source_layout`'s
  stacked offsets, and order-keyed row allocation all retire. The
  `ConceptAllocator`, the discrete record store, `singleton_concept`, the
  chain builder, and the typed-intersection meta read-out carry over
  unchanged. `order_of` derivation survives only as bookkeeping (reporting /
  capacity policy), not as an edge-legality gate.

### Iteration replaces stratification (the wave)

$$a^{0} = \mathrm{snap}(\text{settled field}),\qquad
  a^{i+1} = \tanh\!\big(W\,[a^{i} \mid 1] + s\big),\quad i = 0..K-1$$

where $s$ clamps the snap rows (the order-0/grounded block) as a SOURCE term
each step. `symbolicOrder` is reinterpreted as the **iteration count $K$** —
a runtime dial, not a capacity partition. A vine of depth $d$ activates as a
**wave**: one link per iteration, so arbitrary-length sequences live in
finite flat storage and partial prefixes activate meaningfully for
recognition. Compile discipline mirrors the subsymbolic pump: fixed-$K$
unroll, with the P4 `snap_settle_qe` signal read per iteration as the settle
STATISTIC (no data-dependent control flow); adaptive early exit stays future
work.

### Cycles

The self-edge ban removes only the length-1 degenerate case; everything
interesting about the un-ramsified design lives in the cycles it keeps.

**Loops are a property of taxonomies, not a new taxonomy.** The taxonomy in
this system is the parts/wholes structure itself. A well-constructed
taxonomy is loop-free (a DAG: the vine and all reified structure), and its
activation under iteration is a passing wave, settled after depth-many
steps. Loops arise in **paradoxical or poorly constructed taxonomies** — a
concept reachable among its own parts/wholes through intermediaries — and,
deliberately, wherever sequential structure is wanted (below).

**Fuzzy membership with a temporal reading.** There is no "loop gain" as a
primitive: an edge is a **fuzzy set-membership degree** (how strongly this
part belongs to this whole), and iteration gives the memberships a
**TEMPORAL reading** — membership asserted at step $i$ contributes to what
is asserted at step $i+1$. That temporal reading is exactly what makes the
recursion vine an ordered list (sets are unordered; the nesting unrolls in
time), and it is also what a loop does: re-assert some of its own
memberships each step. Whether a re-entrant loop's activation fades or
persists is then a DERIVED matter — the composed membership degree around
the loop (attenuated by the tanh at each step). Weak composed membership:
the loop is a transient echo on the wave's tail. Persistent re-assertion
with grounded support: a reverberating assembly (Hebb) — sequence memory
and habit as structure, the functional payoff of allowing loops. And in the
LIMIT of large composed degrees the loop becomes self-asserting regardless
of anything outside it — large "loop gains" make loops **particularly
paradoxical** (and drive their codes to the tanh corners, the same
saturation drift the meronymic butterfly's docstring records). A negative
membership inside a loop can additionally produce oscillation rather than
settling — the dynamical image of the liar, and of functional alternation
(contrast, either/or search).

**Groundedness — Kripke's truth criterion as the diagnostic.** Kripke's
construction (Outline of a Theory of Truth, 1975) evaluates a language with
self-reference by iterating from the EMPTY valuation: whatever acquires a
value at the least fixed point is **grounded**; the liar and its relatives
never do — they are *ungrounded* rather than ill-formed. The engineering
translation is exact and operational, and it does not depend on any weight
policy:

- Iterating $a^0 = 0$ with the snap SOURCE clamped is the analogue of
  Kripke's ground-up evaluation: an activation is **grounded** iff it
  appears in this iteration — i.e. its support traces back to perception
  through the graph.
- A cycle whose activation persists (or would persist) with the source term
  removed is **ungrounded**: sustained by the loop itself, answering to
  nothing outside it. The two-run probe (source-on from zeros vs
  source-released) separates grounded assemblies from free-running loops at
  any weight sign.
- Paradoxical structure then shows up exactly as Kripke says it should: not
  as a crash, but as activation that either never settles (revision-style
  oscillation, Gupta–Belnap) or settles without ground. The QE settle
  signal reports the first; the groundedness probe reports the second.

**On weight positivity.** IF a run keeps weights non-negative, the update is
monotone on the presence lattice and the classical theorems apply verbatim
(Knaster–Tarski/Kleene least fixed point; Kripke's construction IS this
monotone iteration). We record that as an available analytical regime, NOT a
design commitment — signed weights are load-bearing in this design
(anti-presence; "sign = present vs anti-present"), negative feedback has
functional value, and the groundedness criterion above works either way.
If flagged loops prove disruptive in practice the escalation ladder is:
damping ($a^{i+1} \leftarrow (1-\lambda)a^{i} + \lambda\,u^{i+1}$), then a
symmetric-weight (Hopfield-energy) discipline on cyclic components — each a
policy dial, none required a priori.

### The human problems (call-out; Alec 2026-07-02)

Ramsification solves self-reference by outlawing it — a solution humans
plainly do not implement. Dropping it reproduces the human situation by
design:

- **Paradox = self-reference.** Every classical paradox (liar, Russell,
  Grelling) is self-reference; the self-edge ban excludes the degenerate
  direct case while longer self-reference-through-intermediaries stays
  expressible, exactly as natural language keeps the liar expressible.
  Kripke's groundedness, not a type discipline, is what separates the
  benign from the pathological uses.
- **Cycles within addiction and trauma.** An ungrounded self-asserting loop
  — memberships re-asserted by the cycle itself, insulated from the snap's
  source term — is the formal shape of rumination, addiction, and trauma
  re-entry: reachable states that do not settle and do not answer to
  perception. The SAME structure with grounded support and moderate
  composed membership is sequence memory and habit. Capability and
  pathology are one mechanism at different membership strengths and
  groundings, which is why no static prohibition can keep the first and
  exclude the second.
- The engineering posture follows: bound the iteration, keep the grounded
  source term on, observe settling and groundedness — and treat flagged
  ungrounded cycles as first-class phenomena to study, not errors to
  suppress.

## Migration sketch (to be expanded into tasks on approval)

1. Collapse `ConceptAllocator.layer(order)` to ONE square layer (no roles);
   `assign_row` allocates from the whole non-snap inventory; keep
   `order_slice(0)` as the snap block.
2. `_populate_concept_weights` v3: every sym constituent gets ONE untyped
   edge (row = the relation's row, col = the constituent's row); EVERYTHING
   $\to$ the bias column; raw refs stay reference-store-only; self-edge
   check. Min-support and the singleton exemption unchanged.
3. `cs_forward_content` v3: the clamped-source iteration above; decode
   unchanged (`a x softplus(atom)`).
4. Byte-identity: `symbolicOrder=0` untouched (all gated on
   `_sparse_active()`); the two-phase cutover site is unchanged — only the
   symbolic phase INSIDE it is replaced.
5. Tests: wave-activation unit tests (a depth-$d$ vine needs $d$ iterations;
   prefix activation at $i < d$); no-self-edge; cycle non-settling flagged by
   the QE signal; the KRIPKE GROUNDEDNESS PROBE (source-on-from-zeros vs
   source-released separates grounded assemblies from free-running loops);
   temporal-membership pins (weak composed membership around a loop decays;
   a strongly self-asserting loop persists and is flagged as ungrounded).
6. Experiments: the XOR driver configs rerun; the P4 gradient-attenuation and
   snap-blindness probes rerun against the iterated reading.

## Open questions (for the approval pass)

- Capacity policy for the relation pool (mint-order first-come vs salience).
- Whether the WEIGHTED reading should see the ordered pair at all (current
  answer: no — set-membership only; revisit if direction-sensitive weighting
  proves necessary for e.g. non-material implication strength).
- Negative-cycle policy default (flag-only vs damped).
- Whether `symbolicOrder` should be renamed (`symbolicIterations`?) at the
  config surface or kept for continuity.
