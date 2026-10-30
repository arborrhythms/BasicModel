# Item 2.8: order-0 concepts strictly sorted, two subspaces, one call (Claude, 2026-10-27; spec in code names, with decisions for Alec)

Successor to the interval rule and the "parts and wholes sorted with
respect to one another in shared coordinates" presumption, both superseded
by Alec's decision of 2026-10-26 ([FutureWork, "Decided (Alec, 2026-10-26):
two subspaces, sorted per subspace, one call"](../FutureWork.md), under the
sorting item of 2026-10-25). Its own round after item 2.9 lands; independent
of 2.9 in design, sequential in landing because both touch
`MereologicalCodes`. Nothing here touches priming, placement or the
meaning codes. The Codex text is written when Alec has answered §5.

## 1. The record

**Decided (Alec, 2026-10-26).** The percept width splits at construction
into an identity subspace (`WordIdentity`'s pair and length coordinates)
and a property subspace sized by config. Identity sorts by atom
containment, the join of atom codes; property sorts by extension over the
byte primitives, a type's code the union over its member bytes of the
byte's hashed bits weighted by `PrimitiveProperties.members`, so a subtype's
code is at most its supertype's. Location selects the set and never enters
a code. A part percept's property subspace is the meet of the types holding
over its bytes; a type's identity subspace is one. The concept form is the
join of the selected parts' forms in the identity subspace and the meet of
the selected whole types' codes in the property subspace. `Ops.partOf` over
the whole code is the one call. The certificate: within tower 0 as
respecified on 2026-10-25, within tower 1 by membership nesting, across
levels by at-most, and at order 0 every nested pair of concepts in a native
lesson's inventory strictly sorted, `containment_audit` becoming that
check. No cross-tower content comparison is claimed.

## 2. What exists, in code names

- *Layout.* `WordIdentity.resolve_identity_layout(config)` runs only for
  `dataType == 'embedding'` with `PartSpace.identityByParts` true: `width =
  identityPairDim + identityLengthDim` (64 + 32), `form_event = width +
  sum(canonical_shape('PartSpace'))` (the 4 + 4 `.where`/`.when` band),
  written into `InputSpace`, `PartSpace`, `WholeSpace` `nDim` and into
  `ConceptualSpace.nDim` as `form_event + complement`. `Codebook.enable_derived_codes`
  reads `percept_event_width = PartSpace.nDim` and `percept_width =
  percept_event_width - 8`; `MereologicalCodes` holds `percept_width`,
  `percept_event_width`, `code_width`, `meaning_pairs`, and `_forms [V,
  percept_width]`. `LTM0.percept_width` and its `percept_layout` buffer,
  `PrependedSymbols.symbolize`, `IdentityRows`, `DocumentCover` and
  `MemoryConsolidation._fold` read `percept_width` as one number. Numeric
  configurations never pass through `resolve_identity_layout` and have no
  `WordIdentity`.
- *Tower 0.* `perceptualSpace.subspace.what` (`MereologicalCodes._native(0)`),
  whose rows for atoms and words are fixed by `WordIdentity._fixed(row,
  value)` into `basis.fixed_row_mask`, `basis.fixed_row_codes` and `basis.W`
  (`Codebook` reads fixed rows through the mask at Spaces 2366 and 2411).
  `WordIdentity.width == store.dim`; `atom_code(atom)` sets `ones` of the
  `bit_stream(bytes_of(value), pair_dim)` bits for a pair atom and bit
  `pair_dim + k - 1` for the length-`k` atom; `form(raw)` is the
  coordinatewise max over `words[raw]`'s atom codes and `nested[raw]`'s
  forms; `part_atoms(raw)`; `revision` advances on a mint.
- *Tower 1.* `wholeSpace.subspace.what` (`_native(1)`), a monotonic
  `Codebook` whose `W` is an `nn.Parameter` in the same `nDim` coordinates,
  beside `basis.primitive_properties = PrimitiveProperties(nVectors)`:
  `members [rows, 256]` (parameter, clamped to `[0, 1]`), `observed`,
  `fixed_rows`, `fixed_members`, `define_word(row)` for the canonical
  `"word"` row of `_CANONICAL_PROPERTY_ROWS` (letter, digit, whitespace,
  punctuation, capital, control, high_byte, pad, word), `coefficients()`,
  `on_counts(counts, conjunctive=True)` (read at Spaces 20385 and 20462 for
  the carrier `weights @ rows`, and at Models 11469 for
  `_ar_whole_reference_presence`), `evidence_on_counts`, `teach`. The
  rows' content `W` is learned and bears no order to tower 0; that is the
  measured cross-tower failure of 2026-10-25.
- *Concept forms.* `MereologicalCodes.derive(rows)`: a retained row (a
  concept, an individual, a form region) reads `similarity_codebook.W[row,
  :percept_width]`, written by `IdentityRows` (96), `MemoryConsolidation._fold`
  (179) and `ConceptIndex` (71, inheritance); any other row reads
  `_forms[row]`, which `derive` fills from `_definitions(row)` (the
  `(tower, pid, net)` triples of `alloc.layer().features`, tower `(col // 2)
  % 2`, pole `col % 2`, group `feature_groups[(row, col)]`) through
  `_interval(edges)`: tower 0 gives the max over the parts' codes (the
  lower bound), tower 1 the min over `1 - d (1 - code)` (the upper bound);
  only `lower` is written, and an identity word's `_forms[row,
  :identity.width]` is then overwritten by `identity.form(raw)` through
  `centroid_forms()`, which calls `SymbolCentroid.place(forms, parts)`: it
  returns the forms unchanged and the postings-containment report
  (`centroid_audit`). `containment_audit()` builds the same postings over
  tower-0 pids of order-0 rows and reports `pairs`, `violating_pairs`,
  `coordinates`, `largest` on `form[contained] <= form[container]`.
- *The literals.* A word's `('ps', pid)` and `('ws', pid)` features are
  written at Spaces 15622 from the reader's `parts` and `wholes`; `ws`
  literals are admitted only when `wholeSpaces[0].subspace.what` carries
  `primitive_properties`. `.where` stays on `PerceptField.events`.
- *The one call.* `Ops.partOf(S1, S2)` is `(a <= b).all(-1)` after
  `_order_align`; `wholeOf` its dual. Consumers keep their interface: the
  `isPart` rule (Language 4104), `containment_audit`, `SymbolCentroid.place`,
  the free decoder's inverse.
- *Context wholes.* `compose_blocks` (Layers 2498) carries the
  leaf-weighted `MeaningKernels.centroid` of the operands' `[:form_width]`
  with the `whole` mean, and `reverse_blocks` and `DerivationReconstruction`
  (72) invert it exactly with `split_centroid`; `PrependedSymbols.symbolize`
  takes a name's form from its reading or the mean of its slots' forms;
  `_fold` writes an order-1 symbol's form as its members' centroid (Orders
  §9).

## 3. The design, in code names

### 3.1 The split, at construction

`resolve_identity_layout` reads `property = ps.setdefault('propertyDim',
D_p)` and `ones = ps.setdefault('propertyOnes', ps['identityOnes'])`
(defaults in §5 D1), sets `width = pair + length + property`, and writes
`form_event = width + 8` into the same sections as today. `MereologicalCodes`
gains `identity_width` (`pair + length`, or `percept_width` where no
identity exists), `property_width` (`property`, or 0), with `percept_width
= identity_width + property_width` unchanged as the sum every present
reader uses; `enable_derived_codes` passes the two widths (from
`PartSpace.identityPairDim + identityLengthDim` and `PartSpace.propertyDim`;
fixtures pass them explicitly). `WordIdentity.width` stays `pair +
length`; the identity bank `store.dim` becomes `identity_width +
property_width`, and the check at `WordIdentity.__init__` becomes `store.dim
== self.width + property_width`. A numeric configuration has
`property_width == 0` and is unchanged in every shape. Preallocated, never
resized; a checkpoint whose `percept_layout` or `MereologicalCodes` widths
differ fails loud as today.

### 3.2 Property codes: tower 1 derived from membership

`PrimitiveProperties` gains a fixed buffer `byte_bits [256,
property_width]`, row `b` the `ones` bits `bit_stream(bytes([b]),
property_width)[:ones]` set to 1 (computed once at construction, no
language named: bytes are bytes), and `codes(rows=None)`: `amax over b of
coefficients(rows)[:, b, None] * byte_bits[b]`, shape `[rows,
property_width]`, differentiable in `members` (the max passes gradient to
the attaining byte), monotone in membership: if `members[a] <= members[b]`
coordinatewise then `codes()[a] <= codes()[b]`. Hash collisions produce
ties, as in tower 0.

Tower 1's content rows become derived: `WholeSpace`'s property basis `W`
is no longer a parameter but a buffer `[nVectors, nDim]` refreshed from
`members` at `begin_forward` when `members._version` changes (the cache
pattern `MereologicalCodes.begin_forward` uses): `W[row, :identity_width]
= 1` (the saturated join), `W[row, identity_width:percept_width] =
codes()[row]`, the `.where`/`.when` band 0. The carrier reads at Spaces
20385 and 20462 (`weights @ rows`) and every other reader of `ws.W` are
unchanged in interface; gradient reaches `members` through the derived
rows as it reaches it through `on_counts` today. *Claude's reading, to
confirm (§5 D3):* this is what "tower 1 codes occupy both" with "a type's
code is the union ... weighted by `members`" means in code; nothing else
makes the property subspace monotone in membership.

### 3.3 Tower 0's property fill

A tower-0 row (an atom or a word) keeps its identity code in
`[:identity_width]` and gets `[identity_width:percept_width]` = the soft
meet of the types holding over its bytes: with `counts` the byte multiset of
`raw` (`counts_in_spans` on the word's own bytes, or a direct bincount; an
atom's bytes are the atom's), `p = on_counts(counts, conjunctive=True)`
(`[rows]`), `fill = amin over t of (1 - p_t (1 - codes()[t]))`, the
soft-meet form `_interval` already uses for tower 1, and `fill = 1` where
no type holds. `WordIdentity._fixed` writes `cat(identity_code, fill)`;
`PrimitiveProperties` gains `revision`, advanced by `teach` and by any
write to `members` (the clamp after the optimizer step at 198), and
`WordIdentity.refresh_property_fills()` rewrites the fill of every fixed
row when the revision it last saw differs, called from the same
`begin_forward` as 3.2, bounded by the fixed-row count. *Claude's call (§5
D2):* fill written into the fixed rows and refreshed on the revision,
rather than recomputed on every read, because the fixed rows are read by
compiled composition paths that must not call `on_counts`.

### 3.4 The concept form

`_interval(edges)` retires. `_subspace_bounds(edges)` returns `(join
[identity_width], meet [property_width])`: `join` the max over the tower-0
parts' rows restricted to `[:identity_width]` (today's lower bound on the
identity subspace; for an identity word this equals `identity.form(raw)` by
construction, so the `centroid_forms` overwrite goes), `meet` the min over
the tower-1 wholes' rows restricted to `[identity_width:percept_width]` of
`1 - d (1 - code)` with `d` the net evidence (today's upper bound on the
property subspace), and `meet = 1` with no wholes. `derive` writes
`_forms[row] = cat(join, meet)` for a non-retained row; retained rows keep
`W[row, :percept_width]`, and their writers (`IdentityRows`, `_fold`,
`ConceptIndex`) are untouched. The sort follows: if A's span nests in B's
then A's `ps` set is inside B's (the join rises) and A's `ws` set contains
B's (the meet falls or holds), so `partOf(form(A), form(B))`, strictly
where B has a part A lacks or A has a type B lacks.

### 3.5 Location selects, codes do not

Unchanged: the reader writes a word's `('ps', pid)` literals from its atoms
and nested parts and its `('ws', pid)` literals from the types
`on_counts` finds over its span (Spaces 15622; Models 11469), and `.where`
stays with the event. No `.where`, `.when` or span coordinate enters
`codes()`, `fill`, `join` or `meet`. The certificate reads nesting from the
recorded literals, not from spans.

### 3.6 The one call

`Ops.partOf` is unchanged: over the whole percept code the two per-subspace
verdicts are conjoined by `all(-1)`. It reads part-of where the identity
subspace separates the pair and is-a where the property subspace does. No
`isPart`/`isa` split; `wholeOf`, the `isPart` rule, the thought operation
and the free decoder's inverse keep their interfaces.

### 3.7 Context wholes

`compose_blocks`'s `centroid` and `reverse_blocks`'s `split_centroid` stay
the rule for composed wholes, over both subspaces alike, because
composition needs the exact inverse that reconstruction and the bounded
binary reconstruction read, and a join has none. The join-and-meet rule is
the form rule for order-0 concept rows (3.4). `PrependedSymbols.symbolize`
and `_fold` are unchanged. *Claude's call (§5 D4).* The "context-wholes form
ports" are therefore the tests whose expected order-0 forms assume one
undivided percept width, listed in §6, not the centroid kernel's tests.

### 3.8 What retires

`MereologicalCodes._interval`; the `centroid_forms` overwrite of an
identity word's form and `SymbolCentroid.place`'s report, whose
postings-containment check becomes the tower-0 part of `containment_audit`
(`centroid_audit` retires with it; `centroid_forms` stays only if a reader
beyond `derive` and the audit exists at build time, else retires). Each
retirement gets a certificate test asserting absence. `Ops.partOf`,
`wholeOf`, `on_counts`, `evidence_on_counts` and `WordIdentity.form` are
unchanged.

## 4. The certificate (`containment_audit`, the item's gate)

`containment_audit(*, before=None, after=None)` returns one dict with four
reports, each `pairs`, `violating_pairs`, `coordinates`, `largest`, and for
the order-0 report also `non_strict_pairs`:

1. *tower0*: for each kept word `raw` in `identity.words`, every atom in
   `part_atoms(raw)` satisfies `partOf(atom_code(atom), form(raw))`, every
   nested part `partOf(form(part), form(raw))`, and every word pair whose
   atom sets nest `partOf` on their forms; read through `WordIdentity` and
   the tower-0 rows those forms occupy (the 2026-10-25 respecification;
   `SymbolCentroid.place`'s postings loop moves here). Each word's distance
   from the all-ones corner reported as today.
2. *tower1*: every pair of property rows `(a, b)` with `(members[a] > 0)
   <= (members[b] > 0)` on the `observed` support satisfies
   `partOf(codes()[a], codes()[b])`.
3. *cross*: every tower-0 row's full code is at most each type holding
   over its bytes (`p_t > 0`), and every order-0 concept's form is at most
   each of its `ws` literals' rows; both hold by 3.3 and 3.4 and the audit
   reports them.
4. *order0*: over the order-0 rows of the inventory, every pair `(A, B)`
   with `ps(A) <= ps(B)` and `ws(B) <= ws(A)` (sets of pids with `net > 0`),
   `A != B`, satisfies `partOf(form(A), form(B))`; where `ps(B) - ps(A)` or
   `ws(A) - ws(B)` is nonempty the pair must also be strict, `(form(A) <
   form(B)).any()`; ties with equal sets are reported, not violations. Self
   pairs omitted. `before`/`after` snapshots as today.

The gate runs it on the `test_review13_subspaces` fixture extended with a
`PrimitiveProperties` tower and on one native lesson's inventory (the short
native lesson smoke the receipts already run, with the audit read after the
lesson) and requires zero violating and zero non-strict pairs in all four
reports. No cross-tower content comparison is made.

## 5. Decisions for Alec, with recommendations

- **D1. `propertyDim` and `propertyOnes` defaults.** Recommended: 64 and
  `identityOnes` (3). For: the property subspace then costs what the pair
  subspace costs, and a top type (the word row over every letter) saturates
  to the all-ones corner, which is the right code for a top. Alternative:
  256 with one bit per byte, the exact extension with no collisions, at
  the price of 256 more coordinates in every space's `nDim` and every
  compiled shape.
- **D2. Where the tower-0 fill lives.** Recommended: written into the
  fixed rows and refreshed on `PrimitiveProperties.revision` (3.3).
  Alternative: computed on each read, which puts `on_counts` inside the
  compiled composition paths.
- **D3. Tower 1's `W` derived from `members`.** My reading of the
  decision (3.2); confirm. Consequence: `ws.W` is a buffer, not a
  parameter; `test_review14_contracts` asserting `ws.W.requires_grad`
  ports to asserting that gradient reaches `members`.
- **D4. Composed wholes keep the centroid.** Recommended (3.7). The
  alternative, join and meet in `compose_blocks`, loses `split_centroid`'s
  exact inverse.
- **D5. Saved models.** The width changes the checkpoint layout, so the
  item 3.5 decomposition snapshots under `output/item3-5-identity-initial`
  (the seeds and hashes are in the 2.9 receipt's
  `decomposition-deferred.json`) will not load after 2.8. Recommended:
  regenerate them from their recorded seeds in the November sweep rather
  than carry a loader for the old layout (no legacy paths).

## 6. Ports, gates, round

*Ports (before/after text in the receipt):* `test_review13_subspaces`
(`enable_derived_codes(percept_width=3)` becomes explicit `identity_width`
and `property_width`; forms compared on the identity subspace; the
property subspace is one where no `ws` literal exists), `test_review14_contracts`
(the interval test reads the meet on the property subspace instead of
`expected = lower`; the `requires_grad` assertion per D3),
`test_operators_round2b` (the audit's four-report shape),
`test_sweep_form_audit`, `test_when_range_encoding`, `test_stage_3_parser_cleanup`,
`test_presentation_memory`, `test_forgetting`, `test_document_covers`,
`test_mereology` and `test_conceptual_introspection` wherever a width or an
order-0 form is assumed undivided; `operators_run_audit`,
`objective_conflicts_probe` and `attention_field_gates` for the audit's
shape. The `centroid` kernel tests are untouched under D4.

*Gates (one freeze):* (1) layout: `percept_width == identity_width +
property_width`, `percept_event_width == percept_width + 8`, every
section's `nDim` as `resolve_identity_layout` writes, a numeric
configuration unchanged, `propertyDim < propertyOnes` fails loud; (2)
`codes()` monotone on a four-row `PrimitiveProperties` fixture with nested
and weighted memberships, and differentiable into `members`; (3) tower 1
derived: `W[row]` equals `cat(ones, codes()[row], zeros)`, refreshed after
`teach`, pointer constant; (4) the fill: a word with holding types carries
their soft meet, a word with none carries ones, refreshed on the revision;
(5) the form: `derive(row)` equals `cat(join, meet)` on the fixture, the
retained rows unchanged, `identity.form(raw)` equal to the join for an
identity word; (6) the certificate of §4, zero violating and zero
non-strict pairs on the fixture and the native lesson, and a constructed
violation reported in `before`; (7) one call: a nested pair true, a pair the
identity subspace separates one way and the property subspace the other
false both ways, `wholeOf` dual; (8) retirements certified; (9) standing:
the five scoped groups and 2.9's priming and placement groups on the hash,
one native lesson smoke, cost: no `[V, V]` or `[rows, 256,
property_width]` tensor outside construction (`byte_bits` is `[256,
property_width]` once). No seed, no resize, no `RUN_SLOW`; the deferred slow
cases and D5's regeneration ride the November sweep.

*Exit:* the certificate green on one hash; FutureWork's sorting item marked
built here by append; Params.md rows for `propertyDim` and `propertyOnes`.

## 7. Alec's decisions (2026-10-27)

Alec answered §5 in conversation: D1 "256"; D2 "Yes"; D3 "Parts of
parts of … bytes. I don't see any alternative"; D4 "Sure"; D5 "Yes".
The D1 alternative presented was 256 coordinates with one coordinate per
byte, avoiding hash collisions. Thus `propertyDim=256`, `propertyOnes=1`,
and `byte_bits` is the byte identity matrix in that layout; the hashed
union remains the rule for other configured widths. These defaults
supersede §5 D1's 64/3 recommendation and §3.2's hashed-byte construction
for the exact 256/1 layout.

The other four recommendations stand: fixed tower-0 fills refreshed on
membership revision, tower-1 content derived from byte membership,
centroids and their exact inverse retained for composition, and old-width
saved models rejected with decomposition snapshots regenerated in the
November sweep. No compatibility loader and no slow runs in this round.

## 8. The one percept vector, stated plainly: what each row carries in each subspace (Claude, 2026-10-27; corrected the same day under §9)

There is one percept vector of `percept_width` coordinates per row, in
every tower and in the concept codebook alike. It has two subspaces:
identity, coordinates `[0, identity_width)`, and property, coordinates
`[identity_width, percept_width)`. A part's *content* has its support in
the identity subspace; a whole's *content* has its support in the
property subspace. The other subspace of each is not content: it holds
the bound that makes the one call `Ops.partOf` over the whole vector read
part-of and is-a correctly. Nothing of one tower is compared with content
of the other.

| row | identity coordinates | property coordinates |
|---|---|---|
| tower-0 atom (a pair or a length atom) | its atom code: `ones` bits of `bit_stream(pair)`, or the one length bit | the fill: soft meet of the types holding over the atom's own bytes, `amin_t (1 - p_t (1 - codes()[t]))`; all ones when no type holds |
| tower-0 word | the join (coordinatewise max) of its atoms' codes and its nested parts' forms, `WordIdentity.form(raw)` | the fill over the word's own bytes |
| tower-1 type | all ones (the top: the join of every instance the type admits) | its weighted byte extension `codes()[t]`, equal to `members[t]` under the 256/1 layout |
| order-0 concept, not retained | the join of its `ps` literals' identity codes | the soft meet of its `ws` literals' property codes weighted by net evidence; all ones with no `ws` literal |
| retained concept or individual | as its writer stored it (`IdentityRows`, `_fold`, `ConceptIndex`); unchanged by this item | the same |

The `.where`/`.when` band after `percept_width` and the meaning after
`percept_event_width` are untouched. The zeros `WordIdentity._storage_code`
pads at insertion are the row's initial value only; `_fixed` writes the
fill before the row is read, and `refresh_property_fills` rewrites every
fixed row's property coordinates when `PrimitiveProperties.revision`
changes.

Why the one call reads correctly:

- *part-of within tower 0*: atom $\le$ word on identity by the join; on
  property both carry fills of the same byte multiset or a sub-multiset,
  and a sub-multiset holds every type the multiset holds, so the smaller
  row's meet is at most the larger row's. The comparison is content on
  identity and a bound on property.
- *is-a within tower 1*: subtype $\le$ supertype on property by the
  monotone extension; both are ones on identity. The comparison is content
  on property and a tie on identity.
- *a part and a type*: no inequality between them is required, claimed
  or checked (Alec, §9: they have different support over the vector). It
  happens to hold by construction, since the part's identity is at most
  the type's ones and its fill is a meet over the types holding over it,
  but the audit's `cross` report is marked not applicable and no content
  of tower 0 is compared with content of tower 1.
- *nested order-0 concepts*: A's `ps` set inside B's raises the join; A's
  `ws` set containing B's, with at least B's weight on each shared type,
  lowers the meet. So `form(A)` $\le$ `form(B)`. Equal vectors are one
  point of the partial order (Alec, §9: mutual containment is allowed;
  parthood is a partial order, not a strict one); the audit reports ties
  and never fails them.

A row with content in both subspaces does not exist: that is the sense in
which parts and wholes have different support on the single percept
vector. Alec's clarification (§9) corrected two claims of this section's
first draft: no inequality between a part and a type is required, and
nesting need not be strict. §§3.4, 4.3, 4.4 and 6 read with §9 where they
say "strictly" or "across levels".

## 9. Alec's partial-order clarification (2026-10-27)

Alec clarified in conversation: "part does not have to be less than type
because they have different support over the vector" and "a given vector
does not have to have a comparison between parts and wholes (types)."
He also confirmed: "you can have mutual containment; that said, we are
aiming for parthood as a partial order only."

This clarification governs the proposed explanation in §8, including its
part-under-type and strict-nesting claims. It supersedes the cross-level
at-most requirement in §4.3
and the requirement in §§3.4, 4.4 and 6 that different literal sets must
produce strict inequality. The soft meets in §§3.3–3.4 stay unchanged.
`Ops.partOf` remains coordinatewise `<=`: incomparable vectors are
allowed; mutual containment implies equal vectors, even if their row
identities differ. Equal codes represent the same point of this partial
order, rather than requiring distinct labels to be strictly separated.

The certificate checks the order implied by the weighted inputs. For
tower 1 the premise is `coefficients[a] <= coefficients[b]`, rather than
containment of their nonzero masks: equal masks with unequal weights do
not imply containment in both directions. For order 0, the identity
premise remains `ps(A) <= ps(B)`; the property premise also requires each
positive `ws(B)` weight to be at most its matching `ws(A)` weight. This is
the monotonicity condition of the specified soft meet. Support-only pairs
that lack that weighted order are counted separately, not asserted as
inequalities. The `cross` report key remains with `applicable=false` and
zero pairs. Ties (`non_strict_pairs`, `equal_set_ties`) are observations,
not failures. The gate requires zero `violating_pairs` in the applicable
reports, retains the constructed `before` violation, and explicitly
checks incomparable vectors, mutual containment of equal vectors and the
partial-order laws. The other gates, budgets and deferred slow runs stand.

The new soft-meet read bounds its net evidence weight to `[0, 1]`.
Overlapping feature groups can contribute the same native type twice to
`_definitions`; its summed net evidence remains intact, but it is a
presence weight when read by the soft meet. The audit uses the same bounded
weight. The regression keeps a recorded net of 2 and verifies that its
form remains in the cube without changing either stored evidence or the
composition kernels.

## 10. Authorized boundary repair after the first freeze (2026-10-27)

The first frozen run (`be4514dedea8`) failed the native worked-document
core gate. Its receipt remains under
`doc/benchmarks/2026-10-27-item2-8-subspaces/`. Alec authorized repairs
where the failures are clear, followed by a new freeze.

The description-to-operand boundary in `GrammaticalThoughtRegistry`
summed occupied roles and divided by the square root of their count.
Captured native form coordinates reached `2 / sqrt(3) = 1.1547005`,
outside the unit cube even though the input coordinates were valid.
`_description_value` now uses the centroid on `[:percept_width]` at all
four description-fold sites. The event and meaning coordinates retain
their existing square-root normalization. The composition and inverse
kernels are unchanged. Deterministic two- and three-role regressions
exercise inline description and meaning operands, owned occurrence
resolution, the controller menu, and gradients through both slices.

The second freeze (`70f21c757906`) captured the persistence failure in a
smaller scorer test. Its retained trace shows a closed equality unchanged
after empty queries: no witnesses, no bound roles, zero evidence, but
`_producing_operation='query'`. The writer mistook the last attempted
operation for an actual binding and bypassed the semantic-chain writer.
The deterministic empty-query regression reproduces that rejection.
The repair distinguishes read provenance from a checked referent binding:
only explicit `_bound_roles` select the binding route. A canonical
unbound relation follows the existing semantic-chain writer. Unchanged
recalled chains still reuse their witnesses; unrecognized multi-role
results still fail at the existing guard. No binding is fabricated and
no evidence is added. The failed second receipt remains under
`doc/benchmarks/2026-10-27-item2-8-subspaces-r2/`.

**Validated on `555e63e2f521` (2026-10-27).** The short native lesson passed before
the final freeze. All 20 item-certificate tests and the requested
scoped groups pass on that source, including both formerly failing
core cases and the five persistence regressions. The slow cases
remain deferred; no seed was installed. The [final receipt](../benchmarks/2026-10-27-item2-8-subspaces-r3/README.md)
preserves the two failed freezes, exact commands, port texts,
retirement and allocation certificates, and a fresh source-hash
recompute. This tree remains uncommitted for review through Alec.

## 11. Claude's review of Codex's item 2.8 round (frozen `555e63e2f521`) — 2026-10-27: approved for landing

**Receipt.** `doc/benchmarks/2026-10-27-item2-8-subspaces-r3/`, base
`17e9454763` (the item 2.9 landing). The frozen hash recomputes from the
inline `source_snapshot` to
`555e63e2f521f97bce9e890a811211bd4afd044943f26936729ba5388e63c112`; all
919 snapshot files hash equal to the live checkout, no source file under
`bin/`, `test/` or `data/model.*` is outside the snapshot, and the four
documents in `document-hashes.json` match the tree. Every gate group and
the native smoke ran on that hash with `RUN_SLOW` and `BASIC_SEED` unset;
no seed in the diff, no resize (the layout is fixed at construction; the
pointer gates hold across refreshes; a checkpoint of another layout is
rejected by the new `_form_layout` buffer). The two failed freezes
(`be4514dedea8`, `70f21c757906`) are preserved with their captures, not
replaced.

**Gate counts.** 2053 passes, 48 skips, 0 failures across overlapping
groups: core 1628/36 (the same 26 slow and 10 absent-artifact skips as
2.9), ports 155/1, priming 146/4, placement 6, packing 1/3, XOR 8,
snapshots and budgets 89, subspaces 20, decomposition 0/2 + 0/2 deferred.
The 152-selector scoped list ran in full; the selectors beyond it are the
named groups' own files plus the eleven port files and the new
certificate. The two cases that failed the earlier freezes pass.

**The spec's gates, by test** (`test/test_form_subspaces.py`, 20 cases):
layout (1) `test_construction_resolves_the_split_and_preserves_numeric_layout`
and the three fail-loud cases; codes monotone and differentiable (2)
`test_exact_byte_union_is_monotone_and_differentiable_for_weighted_members`
and the hashed-layout parity test; tower 1 derived (3)
`test_tower1_refresh_preserves_storage_and_routes_gradient_to_members`;
the fill (4) `test_fixed_property_fills_refresh_on_membership_revision`;
the form (5) `test_forms_join_identity_and_meet_properties_and_preserve_retained_rows`;
the certificate (6) `test_containment_certificate_and_constructed_before_violation`,
`test_certificate_orders_weights_allows_ties_and_imposes_no_part_type_order`,
`test_certificate_requires_weighted_whole_nesting_for_a_soft_meet`, and the
native lesson `test_native_lesson_certificate` ("cat", "cats": zero
violating pairs in every applicable report, `partOf` true and strict on
the full forms); one call (7) `test_one_call_conjoins_both_subspaces_and_wholeof_is_dual`
(incomparable vectors, antisymmetry on vectors, transitivity); retirements
(8) `test_retirement_certificates` (`_interval`, `centroid_forms`,
`centroid_audit`, `SymbolCentroid.place`, `_centroid_cache`); standing
(9) the scoped groups above and
`test_cost_certificate_has_no_pairwise_or_three_axis_union_workspace`
(no `[V, V]`, no three-axis union workspace; `byte_bits` is the 256 by 256
identity once). Added under §9 and §10: the overlapping-feature-group
presence bound, the two description-fold cube regressions, and five
thought-persistence regressions in `test_thought_binding.py`.

**Code against §§3, 8 and 9.** `resolve_identity_layout` reads
`propertyDim` 256 and `propertyOnes` 1 and writes `width = pair + length +
property` into every section; numeric layouts are untouched.
`PrimitiveProperties(rows, property_width, ones)` holds `byte_bits` (the
identity matrix under 256/1, `bit_stream` bits otherwise), `codes()` (the
weighted byte union, equal to `coefficients()` under 256/1), `fill(raw)`
(the soft meet `amin_t (1 - p_t (1 - codes()[t]))` over `on_counts` of the
row's own bytes), `revision` (advanced by `teach`, the optimizer's writes
and `project`'s clamp, observed through tensor versions). The property
basis of `WholeSpace` becomes a buffer through `Codebook.enable_property_codes`:
`[ones | codes() | zeros]`, refreshed at `begin_forward` when the revision
moves or a differentiable forward needs a fresh graph; the LBG split that
wrote `W.data` is bypassed for a derived book; gradient reaches `members`
(asserted). `WordIdentity(..., property_width)` writes
`cat(identity_code, fill)` into the fixed rows and `refresh_property_fills`
rewrites them on the revision; `Models._snapshot_sentence_codes` calls the
book's `begin_forward` and the refresh once before the first lexical bank
of each sentence. `MereologicalCodes` carries `identity_width`,
`property_width`, `_form_layout`; `_subspace_bounds` returns the identity
join over tower-0 rows and the property soft meet over tower-1 rows with
net evidence clamped to `[0, 1]` (§9's presence bound), ones for a
missing tower; `derive` writes `cat(join, meet)` for non-retained rows and
leaves retained rows to their writers. `containment_audit` delegates to
`FormContainment.audit`: tower 0 as respecified on 2026-10-25, tower 1
under the weighted premise `coefficients[a] <= coefficients[b]`
(support-only pairs counted, not asserted), order 0 under `ps(A) <= ps(B)`
and weighted `ws(B) <= ws(A)` with ties reported, `cross` marked not
applicable. `Ops.partOf` is untouched. This is §8's table as corrected
under §9.

**The two authorized boundary repairs (§10).** Both are outside §3 and
both are real defects the wider form exposed. `GrammaticalThoughtRegistry._description_value`
replaces `sum / sqrt(n)` on the whole description with the centroid on
`[:percept_width]` and the old normalization after it; the captured value
`2 / sqrt(3)` was outside the cube. `ThoughtStream.write` now routes to the
checked-binding writer only when `_bound_roles` holds a `referent`; every
other reader of `_bound_roles` already filters on that kind, so the only
behaviour that changes is the empty-read case the second freeze captured
(a query with no witnesses, no bound roles and zero evidence), which now
takes the semantic-chain writer as it should; the malformed multi-role
guard and the recalled-chain reuse are asserted unchanged.

**One correction to my own 2.9 R2.** The wording I gave for Params.md put
the token `primingSpread` back into the file, which the landed 2.9
certificate `test_retired_spread_attribute_and_configuration_are_absent`
forbids; the landed tree `17e9454763` therefore carries one failing
certificate case. Codex's port in this round asserts the absence of the
parameter row (`| \`primingSpread\` |`) instead, which is what the
retirement means. The port is right and lands with 2.8.

**Observations, no repair this round.** (O1) `refresh_property_fills`
derives an atom's fill from `bytes_of(atom[1])`, so a boundary pair atom
such as the first pair of a word carries the marker byte, a mint atom
carries its position suffix, and a length atom carries a digit; the fill
is then ones for boundary pairs, the letter extension for inner pairs and
the digit extension for length atoms. Nothing compares an atom's property
coordinates (§9), and word rows and concept forms are computed from the
word's own bytes, so no gate or consumer depends on it; but the fill of
an atom should be computed from the pair's two characters without the
marker and a length or mint atom should carry ones. Carried to the next
round that opens `WordIdentity`. (O2) `fill` and `_subspace_bounds` use
the conjunctive `on_counts` read, so a type holds over a row only when it
holds over every present byte; a word with one non-letter byte has no
letter type over it and a fill of ones there. That is the decided read.
(O3) The soft meet is monotone only under weighted nesting, which is why
the audit's order-0 premise checks weights; a pair that nests by support
alone is counted in `support_pairs_without_weight_order` and not asserted.
(O4) D5 stands: the old-width decomposition snapshots are rejected and
regenerate from their recorded seeds in the November sweep.

**Landing.** Approved on `555e63e2f521`. One BasicModel commit of the
tree with generated evidence excluded as 2.9 and 3.0 did; todo.md item 2.8
records the hash and this section; push; bump WikiOracle. No tests, no
`RUN_SLOW`. The deferred slow cases and D5's regeneration ride the
November sweep.

**Landed (2026-10-27).** BasicModel `a8a09a6acb`, "Land item 2.8: identity
and property form subspaces", on `555e63e2f521`; the O1 atom-fill
observation of §11 is carried to the next round that opens `WordIdentity`.
