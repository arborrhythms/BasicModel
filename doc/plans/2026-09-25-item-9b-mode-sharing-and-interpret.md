# Item 9b: one conceptual structure across serial and parallel modes — the erosion gate, mode interleaving, and the `interpret` operator

**Status (Claude, 2026-09-25).** Written on Alec's instruction after the
2026-09-25 discussion of whether serial concepts also exist in the parallel
mind. Alec's position: yes, via the same structure; training serial
concepts helps develop the parallel objects; the concern is that parallel
mode erodes the conceptual boundaries of serial processing. Entries 1–2
record the decision and its reasons. Entry 5 (`interpret`) is **decided**
(Alec: add it in serial mode regardless). Entries 4b–4d were decided by Alec
the same day and written into Architecture, Philosophy, Language, Lexicon,
Spaces and the accessible-mind spec; entries 3–4 remain Claude's
proposals with two yes/no questions in entry 7. Codex implements after item 9's
parity landing; the todo line is added with that landing, since the todo is
in Codex's working tree at the time of writing. Nothing here reopens 11c
(`606683a`); the [mode exclusion](../Architecture.md#the-three-cognitive-operations-updated-for-11c)
recorded there is kept within a pass and relaxed only across passes (entry 4).

## 1. Decision: share the structure (decided in direction, Alec 2026-09-25)

The order-0 rows, their definitions, the two symbols per concept and the
native memberships are **one inventory** for both modes. Neither mode may
hold a private copy of a definition, and a checkpoint written in one mode
loads unchanged in the other. After 11c this is already the storage
situation (persistent concept ids; both modes read native perception); this
entry makes it a rule with a test: a concept learned in serial mode is read
in parallel mode by the same id with the same definition and poles, and
conversely.

## 2. Reasons: the psychological evidence

The human literature was checked before this was written. It says the two
modes share one conceptual structure, that the sharing erodes each side
somewhat, and that the erosion is mostly a loss of *online maintenance*, not
of stored structure. Each finding is paired with the mechanism it maps onto.

**Same structure.** Semantic memory has one amodal hub behind verbal and
nonverbal access; semantic dementia erodes a concept in naming and drawing
together (Lambon Ralph et al. 2017). A heard label lets an image suppressed
by continuous flash suppression break into awareness where an uninformative
cue does not, at a perceptual locus (Lupyan & Ward 2013): this is the
symbol's reverse pi against the field (11c entry 9), the *label feedback*
that entry 4 relies on. Two distinct labels let 9-month-olds individuate two
objects where tones and sounds do not (Xu 2002); words as "essence
placeholders" (Carey 2009) is the order-1 individual forming with object
permanence under a name (11c entry 11).

**Serial processing maintains crisp boundaries online.** Russian speakers'
category advantage at their blue boundary vanishes under verbal, not
spatial, interference (Winawer et al. 2007). Adults lose the ability to
combine geometry with a landmark while verbally shadowing but not while
clapping (Hermer-Vazquez, Spelke & Katsnelson 1999; contested by Ratliff &
Newcombe 2008). Aphasia selectively impairs low-dimensional categories,
things sharing one feature, in proportion to the naming deficit, while
family-resemblance categories survive (Lupyan & Mirman 2013; the language
network's role is contested by Benn et al. 2023). This is Alec's concern
confirmed in humans: with the serial system offline, crisp, single-feature
(pi-selected) categories degrade toward graded similarity. Weight the colour
and developmental results above the two contested ones.

**The reverse erosion.** Learned categories compress within-category
discrimination, acquired equivalence (Goldstone 1994). Verbalizing a face
impairs its later recognition, verbal overshadowing (Schooler &
Engstler-Schooler 1990; replicated in Alogna et al. 2014). So the serial
system erodes parallel detail as the parallel system erodes serial
boundaries. Each mode costs the other something; neither erosion is fatal.

**Meditators, the direct evidence for parallel-mode training.** Twenty
minutes of breath meditation reduced Stroop interference and led to more
*atypical* category exemplars than controls (Wenk-Sormaz 2005): boundary
loosening in the predicted direction. Three months of intensive practice
improved fine perceptual discrimination, persisting five months (MacLean et
al. 2010). The attentional blink shrinks after retreat (Slagter et al.
2007), and one-pointed concentration stabilizes binocular rivalry (Carter
et al. 2005). Deikman (1966) called this deautomatization. The
counterweight: after eight weeks of mindfulness training the narrative and
experiential modes of self-reference become neurally *dissociable* (Farb et
al. 2007). Training yields the ability to engage either mode; it does not
remove the narrative one.

**What follows.** Share the structure (entry 1). Expect erosion in both
directions and measure it (entry 3). Do not keep the modes as permanent
alternatives; interleave them, and let the label re-sharpen the field after
parallel passes, as a noting meditator names what arose (entry 4). Give the
serial loop one operator that writes the parallel field's objects from
words, so serial training develops parallel objects (entry 5).

## 3. The erosion gate (proposed; question 7b)

**Measure.** A categorical-perception index over the native memberships:

    CP = D_between − D_within

where `D` is the membership-level discrimination between two probes (the
L2 distance of their order-0 readings), *between* pairs straddle a concept
boundary and *within* pairs fall inside one concept, on a fixed probe set
drawn from the smoke workload (`MM_sparse_concept`) and the 20-document
FineWeb launch corpus. Two companions: **alternatives admitted per concept**
(the Wenk-Sormaz analogue: atypical alternatives) and **within-category
discrimination alone** (the MacLean analogue).

**Conditions.** (a) serial-only training, (b) parallel-only, (c)
interleaved per entry 4, each from the same initialization, three seeds.
The Winawer analogue is a fourth read, not a condition: the same probes
read with the symbol leg masked (symbols offline) and unmasked.

**Predictions the literature licenses.** Parallel-only lowers `CP` and
raises admitted alternatives while improving within-category
discrimination. Serial-only raises `CP` and compresses within-category
discrimination. Interleaved holds `CP` near serial-only while keeping the
within-category gain. Masking the symbol leg lowers `CP` in every
condition, most in serial-only.

**Gate.** Ordering claims only, holding across all three seeds; no
threshold numbers are preregistered and no seed is pinned to pass. A
violated ordering is a finding about the architecture, recorded, not tuned
away. Results go to [FutureWork](../FutureWork.md) beside the both-prompt
section.

## 4. Interleaving (proposed; question 7a)

**Rule.** Within a pass the 11c mode exclusion stands: the field's sigma, pi
and not run only in parallel passes; the grammar's lift and lower only in
serial passes. Across passes the modes **alternate on the same content**.
A parallel pass over the attended fields of the last `N` sentences follows
every `N` serial sentences, and each parallel pass ends with
**re-symbolization**: every word symbol of those sentences is read back
against the field (reverse pi of the symbol, 11c entry 9), which is the
reverse face of `interpret` (entry 5) and the mechanical form of label
feedback. That read-back is what keeps `CP` up after the parallel pass.

**Knob.** `<architecture><modeSchedule>` with values `serial` (today's
default), `parallel`, and `interleave:N`. Live, not gated, per the
live-wiring rule; documented once in Params. `serial` and `parallel`
reproduce today's two configurations exactly, so existing receipts are
unaffected. The first FineWeb session (item 0) runs `serial`; the erosion
gate (entry 3) runs all three.

## 4b. Attention: serial is focused, parallel is open (decided, Alec 2026-09-25)

**Alec's question (2026-09-25).** Attention is now an 8/8 combination of
PartSpace and WholeSpace. For parallel mode, could all known and active
percepts simply stimulate concepts, so that serial mode *is* the
focused-attention mode?

**Assessment: yes, and it is what the spec already says.** The 8/8 cap is
the conceptual slab width: `_order_caps` tapers `[base, base/2, …, 1]` from
`outputShape[0] = 8` tiles, inherited from the fixed-size fold machinery
that 11c deleted. After 11b the field is gathered sparsely — every
definition referencing an attended membership is a candidate — and then
truncated to `n0` by membership support. Removing the truncation in parallel
mode makes the field the whole inventory, which is the accessible-mind
spec's own definition of parallel knowing (§2.2: one value per concept over
the whole codebook). Serial mode keeps the 8/8 attended field: the current
word, its parts and wholes, and STM — focused attention.

**Consequences.** (i) Cost stays sparse: only definitions referencing active
memberships are read, so work scales with active percepts × referencing
definitions, not with `nVectors`. (ii) The field tensor becomes ragged per
turn (`[B, E, P, referenced rows]`); symbols, thought and checkpoints already
resolve by concept id (11b), so no addressing changes. (iii) The taper's
fixed slots remain only as the serial cap. (iv) Open attention admits more
*both* readings and more alternatives per concept — the erosion entry 3
measures — so parallel passes need the re-symbolization of entry 4 all the
more. (v) `subsymbolicLoop` retargeting in parallel mode becomes a choice
of what to symbolize, not what to read.

**Where the percepts occur (Alec, 2026-09-25).** With every perceptual
codebook item wired to order-0 concepts, the field must still know
*where* each percept occurs. The old 8/8 attention coupled `.what` and
`.where` into the concepts through the attended brackets. Two ways to
avoid that coupling: (A) a separate `.where` channel for conceptual space,
one location per concept; or (B) conceptual attention as **a single convex
`.where`**, one bracket for the whole attentive field.

**Recommendation: (B).** (A) would give concepts individual locations,
which 11c entry 4 decided against (no concept is individually located; all
live within one attentive field) and which the opacity note forbids
(`canonical_shape("ConceptualSpace") = (0, 0)`). (B) is that decision made
into the attention mechanism: the attentive field *is* one convex bracket
over the input; open attention in parallel mode is the bracket set to the
whole input, focused attention in serial mode is the bracket narrowed to
the current word's extent and its neighbourhood; the region-of-interest
move is narrowing the bracket, which is attention as non-affirming
exclusion (accessible mind §2.6.7). Inside the bracket the retained
occurrence pairs remain field coordinates, which is what the readout
unions over and where *both* is born (11c entry 5); *both* is then the
prompt to narrow the bracket (divide, 11c entry 6), and a support that is
discontiguous *within* the bracket is the prompt to raise order (sigma).
Convexity is what keeps the two prompts distinct. The 8/8 coupling
disappears: the caps no longer select which concepts exist in the field,
only how wide the bracket is. What (B) cannot express — attending to two
disjoint regions at once — is exactly what a higher-order concept is for,
so it is a feature, not a limitation.

## 4c. The field as a percept activation vector; where by bracket only (decided, Alec 2026-09-25)

**Alec (2026-09-25).** The 8/8 perceptual field can be emitted as 16
percept activations over *all* percepts, the rest zero; in parallel mode
none are zero. That coexists easily with the COO sparse store the concept
definitions already use. But it carries no `.where`: it says only, by
wire, which percepts were present. So: abandon `.where` for concepts, use
attention (the single convex bracket of 4b) to define the one space the
field is read in, and accept that a concept may be both present and
absent within that field — which is exactly what the *both* corner of the
tetralemma was built to handle.

**Assessment: yes.** Concepts never had a `.where` in principle (the
opacity note; 11c entry 4). What has one is a percept event and a symbol
occurrence. Under this proposal the conceptual side becomes three sparse
objects and nothing else:

- the **percept activation vector** `a ∈ [0,1]^{2·P}`: for every percept
  row, its presence and its observed complement, each **pooled over the
  occurrences inside the bracket** (the union at readout that 11b already
  performs per symbol, moved down to the seam); 16 nonzero pairs in serial
  mode, all active percepts in parallel mode;
- the **definition matrix** `W` (the existing COO store): one sparse row
  per order-0 concept over percept poles, read by the 11c folds (product
  of literals for `c⁺`, L2 union of opposites for `c⁻`, union over
  alternatives);
- the **symbol matrix** `W_σ` above it.

The bracket is the only location at the conceptual level, and it is one
convex interval. `cs_read_memberships`' `[B, E, P]` occurrence tensors,
the retained position pairs travelling through the field and checkpoints,
and the `n0` truncation all go; the where they carried stays in perception,
where the percept events keep theirs.

**What *both* means here.** If a property holds on one run inside the
bracket and fails on another, its percept pair pools to `(1, 1)` before
any concept reads it, so the concept reads *both*. That is not an error:
it is the field reporting that the bracket pools heterogeneous
occurrences. The two remedies are the two prompts already decided: narrow
the bracket (attention, 11c entry 6), or, if the heterogeneity is the
thing to name, raise order (sigma). Convexity (4b) is what keeps those two
remedies distinct.

**Two conditions.**

1. **Perception keeps exact co-location.** A pi at order 0 over pooled
   activations computes *co-presence within the bracket*, which equals
   co-location only when the bracket is one occurrence — the same
   `∃x A ∧ ∃x B ≠ ∃x (A ∧ B)` that rules out pi over symbols (11c entry 9),
   now at the seam. So conjunctions that must be exact are perception's:
   PartSpace's fused part (*A at 1 and B at 2*, one recurring unit),
   WholeSpace's pervading run and property intersections on a run. The
   conceptual intersection is bracket-scoped and may read *both*; a
   certified conjunction is one the field has narrowed to a single
   occurrence, or one perception has fused. XOR stays solvable and gets
   simpler: `10` and `01` are two fused parts, two distinct percepts, so
   the two cases are pi over those presences and sigma unions them; the
   *both* corner of "is a one" over the word bracket is the reading that
   prompts it.
2. **Attribution still reaches `.where`, through percepts.** The
   refine-before-raise gate of the 11c landing counts runs of the brackets
   supporting a *both* reading. That count moves to perception's side:
   reverse pi attributes the reading to percept rows by wire, and *their*
   events carry the brackets to count. Nothing conceptual needs a location
   for it.

**What it changes in landed code.** The 11b/11c seam read (per-extent,
per-position tensors and retained pairs in the conceptual field) is
replaced by the per-bracket pooled read; symbol readout and checkpoints
stop carrying position pairs; `_bind_attended_concepts`' `n0` truncation
goes (4b). The XOR gate, the exact-zero controls and the serial baseline
are the regression set. This is a change to the 11c landing and is
Codex's after Alec's yes.

## 4d. One where, one when, many whats (decided, Alec 2026-09-25)

**Alec (2026-09-25).** Remove `.when` as well. The precise form: a concept
*in a field* does get a `.where` and a `.when`, but only **one** of each —
the field's — together with a **multiplicity of `.what` perceptual codes**
that are parts or wholes of that (where, when) location.

**Assessment: yes, for the same reason as `.where`.** A concept row in the
store has neither coordinate; a field has exactly one of each; a percept
event and a symbol occurrence keep their own. So the three things are:

| Thing | `.where` | `.when` | `.what` |
|---|---|---|---|
| percept event, symbol occurrence | its own bracket | its own time | one code |
| a **field** (a reading; an LTM row is a sealed field) | one convex bracket | one interval | many codes: the parts and wholes inside the bracket |
| a **concept row** in the store | none | none | its definition over percept poles |

**What `.when` is for a field.** The field's time is its position in the
LTM chain (the row being written, the sentence) plus the exact clock
side-band that already rides beside `.when` (2026-07-04). A recalled row
enters thought with *its* (where, when): its bracket in the source
document and its chain position, which is the situation the predictor
anchors (object permanence by reference). A when-interval wider than one
row is an **episode**; the chain is how it is addressed.

**Temporal *both*.** A concept present at one time and absent at another
inside the field's interval reads *both*, exactly as across space, and the
two remedies are the same: narrow the interval (attend to one row), or
raise order when the change itself is the thing to name — a process, an
event, a habit. A contiguous temporal support refines; a discontiguous one
raises. Convexity in time is the chain interval, so 4b's argument carries
over unchanged.

**Tense.** The opacity note already makes tense and aspect the identity on
the concept, with the symbolic realisation owning the `.when` coordinate.
Under 4d that is exact: tense is the relation between the field's interval
and the utterance's, a relation between two brackets, never a property of
a concept row; `lift` extends a thing into a process by widening the
interval the field reads, not by stamping a time on the code.

**Consequences for LTM and symbols (Alec, 2026-09-25).** (i) **Every LTM
row picks up a `.where` and a `.when`**: the field's pair, written once by
the seal (item 7's row schema gains the two columns beside `refs`,
surprise and the `(c⁺, c⁻)` pair). (ii) **`.where` is one unique field
over all percepts.** Input positions have unique `.where`s by
construction; every **symbol occurrence** must also have a unique
`.where` in that same field, in addition to the input positions — a
symbol is a located percept (11c entry 7) and two occurrences never share
a location at one time. (iii) **LTM is the exception** (corrected, Alec 2026-09-25): the
**address of an LTM row is its `.when` alone** — the chain position is
unique — and its `.where` records **what it was looking at**; rows may
share a `.where` (the same document bracket seen again). (iv) **Symbols'
`.where` extends the `.where` of the pre-allocated part and whole
percepts (Alec).** `.where` is one address space: input positions first,
then the PartSpace part rows and WholeSpace whole rows at their
pre-allocated offsets, then the symbols continuing that range. This is the
global where-space slice registry that existed before 2026-06-04 (each
codebook's slice `[offset, offset + nVectors)`; the `where_offset` stubs
in `Codebook` and `Embedding` still mark where it was) revived on the
perceptual side **as an address space for locations only** — codebook
identity stays the row index, per the
[Codebook Uniqueness Contract](../Spaces.md#codebook-uniqueness-contract);
`.where` never again keys a codebook row; symbol occurrences produced by thought take their
symbols' offsets and are never fed to the external observation stream.
Claude's earlier "after the input's last position" proposal is withdrawn.

**What changes in landed code.** Nothing conceptual carried `.when`
already (`canonical_shape("ConceptualSpace") = (0, 0)`); the symbolic
layer's mux/demux of `.where`/`.when` around `execute` stays, since symbol
occurrences are percepts. What 4d adds to 4c is the rule that the field
records **one** (where, when) beside its activation vector, and that an
LTM row is a sealed field carrying that pair; the seal (item 7) writes it
once per row. No per-concept temporal coordinate is introduced anywhere.

## 4e. Fixed-capacity perceptual codebooks with where-space slices (proposed; question 7e)

**Alec's concern (2026-09-25).** `allocate_codebook_slice` was a good
optimization: it pre-allocated the perceptual codebooks so they did not
grow at runtime. Its retirement (2026-06-04, with `global_max_val` and
`reset_codebook_registry`) took the pre-allocation with it.

**What the tree does today.** The percept store grows **geometrically at
runtime**: `RadixLayer._grow_to` doubles capacity on insert or at the
post-step flush, `Codebook.grow_to` re-registers the `nn.Parameter` with
extended storage and copies the rows, and the owner's parameter list and
optimizer groups are migrated (`_migrate_optimizer_parameter`), with the
docstrings disagreeing on whether moments survive. WholeSpace's codebook
has the same `grow_to` path (`Models.py`, `eval_nanochat_grammar.py`).
Every growth changes a tensor shape, so under `torch.compile` it is a
recompile; the code admits this ("Inductor recompiles each time the vocab
grows"; the BPE artifact and `word_learning=0` exist to keep the cache
warm). The concept inventory, by contrast, already does it the old way:
physical rows pre-allocated, **logical** growth activates rows
(`grow_inventory`, `_active_inventory_rows`), parameter and optimizer
ownership preserved. The production PartSpace starts at `nVectors =
32,768` with `maxVectors = 1,048,576` and `nDim = 136`.

**Proposal.** Restore fixed capacity for every perceptual codebook — the
percept store, WholeSpace properties, the lexicon, the symbols — as the
concept inventory does now: physical capacity fixed at construction,
logical active-prefix growth only, admission past capacity a hard error
naming the knob (as `RadixLayer._capacity_exhausted` already does at
`maxVectors`). Where-space slices are then assigned at construction from
those capacities (entry 4d: input positions, then parts, then wholes,
then symbols), and a symbol's `.where` is stable for the life of the
model. Per the no-legacy rule, delete `RadixLayer._grow_to`,
`Codebook.grow_to`, the optimizer-migration path and `maxVectors`
(`nVectors` becomes the physical capacity). Shapes never change, so one
compile serves the run.

**Cost, so the choice is explicit.** The reason growth was made geometric
was memory at the ceiling:

| PartSpace capacity | `W` at `nDim` 136, fp32 | with two Adam moments |
|---:|---:|---:|
| 32,768 (today's start) | 17.8 MB | 53 MB |
| 262,144 | 143 MB | 428 MB |
| 1,048,576 (today's ceiling) | 570 MB | 1.7 GB |

The pilot never approaches the ceiling, and the full session can declare
its capacity in `model.xml` as it declares everything else; a run that
exhausts it stops with the knob named, which is already item 0's stop
condition for the order-0 inventory. The ConceptualSpace entry of
`nVectors = 1,048,576` at `nDim = 1032` (4.3 GB dense) should be checked
against what is physically allocated before the session; if it is dense,
the same fixed-capacity rule applies and the number needs choosing.

## 5. The `interpret` operator (decided, Alec 2026-09-25)

**What it is.** A serial-mode `<compose>` operator that takes a
**word-concept** and yields the **object-concept** it refers to:

    interpret_O1 = interpret.forward(interpret_I1)

Unary, like `not`; its second argument is implicit, the current attended
field. The word-concept is the code the word arrives as (accessible mind
§2.0: "a word arrives already projected"), a symbol occurrence, hence a
located percept in the field (11c entry 7). The object-concept is a row of
the shared inventory: the **order-1 particular** when the field contains an
occurrence the word addresses (*the cat*, *Felix*), otherwise the **kind**
(*cat*), per the word-form-to-order rule in
[Lexicon](../Lexicon.md#word-forms-and-concept-orders) and 11c entry 11.
Which it is, is the grammar's resolution (determiner, name, generic); the
operator does not read the surface form, and no word is anchored to it.

**Not a mode, and not optional (Alec, 2026-09-25).** Serial mode is
forced today; `interpret` is not something to force or to route. In serial
mode it is the **default translation from word to object for every
word**: each arriving word-concept is interpreted before it takes part in
composition, so lift, lower, part and the rest compose object-concepts,
not words. The chooser does not decide *whether* to interpret; what is
learned is only the resolution (particular or kind) that the grammatical
context supplies. This retires the chooser-routing test that an earlier
draft of this entry proposed.

**Unknown words create objects (Alec, 2026-09-25).** Some sequences of
letters correspond to no known object. Then `interpret` **mints a new
object-concept**: a provisional row of the shared inventory whose only
literal is the word occurrence that named it, no perceptual definition
yet. That is how object-concepts come to exist **without direct
experience** of them: by testimony, the essence-placeholder of Xu 2002 and
Carey 2009, and the mechanism the Architecture already assumes when it
says a kind over objects "cannot be witnessed until the mind has a video
feed; its membership rests on testimony". The new row's order follows the
same grammatical resolution as a known word's (a name yields an order-1
individual, a count noun a kind); its later definition is written when
percepts are witnessed with it, or asserted by the seal (*a wug is a
bird*). Admission follows the ordinary provisional rule, recurrence at the
boundary, so a one-off misspelling is forgotten and a recurring new word
becomes an object. The second occurrence of a new word resolves to the
row its first occurrence minted: object permanence by testimony.

**Why it bridges the modes.** The word-concept lives in the serial stream
(§2.4, a code). The object-concept lives in the field (§2.2/2.3). `interpret`
is the one place the serial loop **writes the parallel field's object rows
from words**, which is Alec's hypothesis that serial training develops
parallel objects, and the human evidence for it (Xu 2002; Carey 2009). Its
reverse is the label feedback of Lupyan & Ward. The mode exclusion is kept:
`interpret` is a serial operator; the field it reads and writes is native
perception plus the order-0 rows, never the parallel sigma/pi.

**Where `interpret` starts (Alec, 2026-09-25).** Known words are not
"translated": PartSpace **looks them up** — the word is the recurring unit
the fold ladder admitted, found by synthesis over its bytes. `interpret`
begins from that word row; it is never the byte-to-word step. If the
lookup needs a cache, or a hint about when to stop synthesizing over bytes
and return the word's code, that is an optimization concession under the
serial-mode flag, beside the concessions already there, and is listed with
the throughput levers in [FutureWork](../FutureWork.md#throughput-levers-for-the-serial-loop-item-1-candidates).

**Relation to what exists.** `create_word_object_meta` already mints an
`[object isa word]` triple `(A = word, B = object, C = meta)` at order 1 from
the host side (`Language.py`, the order-1 branch of the symbol tower). Per
the no-legacy rule, `interpret` **replaces** that host-side path with a
declared grammar operator: same triple, same shared inventory, but routed
by the chooser and reversible through the grammar. The two-truths seal
already writes asserted part rows *between the object concepts the words
resolve to*; `interpret` is the operator that produces those object
concepts, so the seal stops resolving them itself.

**Faces.** Compose: `interpret.forward(word) → object`. Generate:
`interpret_I1 = interpret.reverse(interpret_O1)`, object → word, which is
lexicalization through the owned spelling inverse, the step generation
already performs implicitly. Thought: not permitted by default; a per-model
allow-list may add it (a thought that names what is attended is the
noting meditator's move, and belongs with the re-symbolization of entry 4).

**Tests.** (1) After witnessing *the cat sat*, `interpret` on the word
*cat* addresses the order-1 particular whose events are that sentence's
occurrences. (2) On *cats are animals* it yields the kind, and the seal
writes the part row between the two object rows, as the two-truths contract
requires. (3) `interpret.reverse` on that object regenerates the word
through the spelling inverse. (4) **Every word is interpreted**: in a
serial sentence, each composed operand is an object row, never a raw
word-concept; no surface anchor exists for `interpret`. (5) **Unknown
word**: *the wug sat* mints one provisional object row with the word as its
only literal; *the wug flew* resolves to the same row; a later witnessed
percept, or the seal on *a wug is a bird*, writes its definition; a word
seen once is not admitted at the boundary. (6) **Parity across modes**: the
object row `interpret` produces in serial mode is the row the parallel
field binds for the same input (entry 1's test). (7)
`create_word_object_meta` and its callers are gone.

## 6. Documentation after landing

- [Architecture](../Architecture.md): the three-operations section gains
  `interpret` and the schedule; the mode-exclusion paragraph says "within a
  pass" and points here for alternation across passes.
- [Language](../Language.md): operator catalog entry for `interpret` with
  its three faces; `default.grammar` and `complete.grammar` carry the rule;
  the "concept events are opaque" passage cites `interpret` as the
  resolution step it describes.
- [Accessible mind spec §4](../specs/2026-09-20-accessible-mind-subsystems.md):
  `interpret`'s row in "which grammar may touch what": compose R 2.1, R W
  2.2 and 2.4, nothing else.
- [Lexicon](../Lexicon.md): word forms and concept orders names `interpret`
  as the resolution operator.
- [Params](../Params.md): `modeSchedule`.
- [Testing](../Testing.md) and a receipt under `doc/benchmarks/`.
- [FutureWork](../FutureWork.md): the erosion gate results and the
  literature paragraph of entry 2, cited from Philosophy.
- [todo](../../todo.md): item 9b line, added with the landing.

## 7. Questions (Alec's yes/no)

- **7a.** Accept `modeSchedule` = `serial | parallel | interleave:N` with
  re-symbolization closing every parallel pass?
- **7b.** Accept the erosion gate as ordering claims across three seeds,
  with no preregistered thresholds?
- **7e.** Restore fixed physical capacity for all perceptual codebooks
  with logical growth only, assign where-space slices at construction,
  and delete the geometric-growth and optimizer-migration paths and
  `maxVectors`?



## References

- Alogna et al. 2014, Registered replication report: Schooler & Engstler-Schooler (1990). *Perspectives on Psychological Science* 9, 556–578.
- Benn et al. 2023, The language network is not engaged in object categorization. *Cerebral Cortex* 33, 10380–10400. https://pmc.ncbi.nlm.nih.gov/articles/PMC10545444/
- Carey 2009, *The Origin of Concepts*. Oxford.
- Carter et al. 2005, Meditation alters perceptual rivalry in Tibetan Buddhist monks. *Current Biology* 15, R412–R413.
- Deikman 1966, De-automatization and the mystic experience. *Psychiatry* 29, 324–338.
- Farb et al. 2007, Attending to the present: mindfulness meditation reveals distinct neural modes of self-reference. *SCAN* 2, 313–322. https://academic.oup.com/scan/article/2/4/313/1676557
- Goldstone 1994, Influences of categorization on perceptual discrimination. *JEP: General* 123, 178–200.
- Hermer-Vazquez, Spelke & Katsnelson 1999, Sources of flexibility in human cognition: dual-task studies of space and language. *Cognitive Psychology* 39, 3–36.
- Lambon Ralph, Jefferies, Patterson & Rogers 2017, The neural and computational bases of semantic cognition. *Nature Reviews Neuroscience* 18, 42–55.
- Lupyan & Mirman 2013, Linking language and categorization: evidence from aphasia. *Cortex* 49, 1187–1194. https://www.sciencedirect.com/science/article/abs/pii/S0010945212001931
- Lupyan & Ward 2013, Language can boost otherwise unseen objects into visual awareness. *PNAS* 110, 14196–14201. https://www.pnas.org/doi/10.1073/pnas.1303312110
- MacLean et al. 2010, Intensive meditation training improves perceptual discrimination and sustained attention. *Psychological Science* 21, 829–839. https://journals.sagepub.com/doi/10.1177/0956797610371339
- Ratliff & Newcombe 2008, Is language necessary for human spatial reorientation? *Cognitive Psychology* 56, 142–163. https://www.sciencedirect.com/science/article/abs/pii/S0010028507000357
- Schooler & Engstler-Schooler 1990, Verbal overshadowing of visual memories. *Cognitive Psychology* 22, 36–71.
- Slagter et al. 2007, Mental training affects distribution of limited brain resources. *PLoS Biology* 5, e138.
- Wenk-Sormaz 2005, Meditation can reduce habitual responding. *Alternative Therapies in Health and Medicine* 11, 42–58. https://www.ncbi.nlm.nih.gov/pubmed/15819448
- Winawer et al. 2007, Russian blues reveal effects of language on color discrimination. *PNAS* 104, 7780–7785. https://www.pnas.org/doi/10.1073/pnas.0701644104
- Xu 2002, The role of language in acquiring object kind concepts in infancy. *Cognition* 85, 223–250. https://pubmed.ncbi.nlm.nih.gov/12169410/
