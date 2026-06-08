# Analysis / Synthesis Dual-Input Plan

Date: 2026-06-08
Status: Draft plan

## Goal

Reformulate the current subsymbolic and symbolic loops as an explicit
analysis/synthesis pair:

- analysis: `InputSpace.forward()` splits one source input into a perceptual
  branch and a conceptual/symbolic branch.
- recurrent integration: `PerceptualSpace` and `SymbolicSpace` both receive
  input-derived evidence, then the existing C-tier recurrence combines
  percepts, symbols, and concepts.
- synthesis: reverse reconstruction recombines the perceptual and conceptual
  branches back into the input surface with an `InvertibleLinearLayer`.
- configuration: the perceptual-side `<chunking>` knob becomes analytic-only
  `<analysis>`; any operation that synthesizes a surface migrates to the
  symbolic loop.

The immediate shape contract is:

```text
InputSpace.forward(input) -> (percepts_in, concepts_in)

percepts_in: [B, 1, N]
concepts_in: [B, N, 1]
```

The perceptual branch treats the input as a specifically presented field with
an eidetic, continuous, discriminative codebook over InputSpace. The
conceptual/symbolic branch treats the same input as a sequence of generalizable
positions, meanings, objects, or terms.

## Current Context

The current forward path is:

```text
_lex_embed_stem:
  InputSpace.forward(x) -> in_sub
  PerceptualSpace.embed_stem(in_sub)
  InputSpace.finalize_stem(in_sub, perceptualSpace)

_forward_body:
  PS_sub_stage0 = PerceptualSpace.forward(in_sub)
  for each stage:
    SS_sub = SymbolicSpace.forward(prevCS_forSS)
    CS_sub = ConceptualSpace.forward(contribution, SS_sub)
    ConceptualCombine(PS_t, SS_t, CS_t)
```

Important existing contracts:

- `InputSpace` is currently a pure raw lexer and returns one `SubSpace`.
- `PerceptualSpace` owns embedding/chunking/codebook work after the June 7
  peer-removal refactor; this plan renames the perceptual breakdown contract
  from `<chunking>` to `<analysis>` and reformulates the PS codebook as
  discriminative/dichotomizing rather than representational.
- `SymbolicSpace.forward()` currently receives only `CS_subspaceForSS`.
- `ConceptualCombine` already mixes `PS_t`, `SS_t`, and `CS_t` using an
  invertible 3-stream layer, so this plan should feed better PS/SS streams into
  that layer rather than replacing it.
- Reverse reconstruction currently seeds from concepts, walks body reverse,
  then runs `PerceptualSpace.reverse()` and `InputSpace.reverse()`.

## Design

### 1. Split InputSpace Output

Change the conceptual contract of `InputSpace.forward()` from one carrier to two
carriers:

```python
percepts_in, concepts_in = inputSpace.forward(inputData)
```

Planned representation:

- `percepts_in` is the raw surface as `[B, 1, N]`.
- `concepts_in` is the transposed/generalized view as `[B, N, 1]`.
- Both retain enough metadata to reconstruct the original source ordering,
  dtype, and padding/null mask.

Implementation decision to make later:

- Prefer a small `NamedTuple` or dataclass only if tuple plumbing becomes hard
  to read. The public contract should still unpack as two outputs.
- Keep a compatibility shim during migration only inside model orchestration,
  not as a permanent silent fallback.

### 2. Add Input to SymbolicSpace

Update the symbolic loop to accept direct input-derived evidence:

```python
SymbolicSpace.forward(CS_subspaceForSS, IS_concepts=None)
```

Stage behavior:

- Stage 0 reads `IS_concepts`.
- Later stages read recurrent `CS_subspaceForSS`.
- If a config wants repeated symbolic input injection, make that a later knob;
  the first implementation should mirror PS and read input once.

This means `_forward_body` becomes:

```text
percepts_in, concepts_in = staged input from InputSpace
PS_sub_stage0 = perceptualSpace.forward(percepts_in)

for t:
  if t == 0:
    SS_sub = ss.forward(prevCS_forSS, concepts_in)
  else:
    SS_sub = ss.forward(prevCS_forSS)
  CS_sub = cs.forward(contribution, SS_sub)
  ConceptualCombine(PS_t, SS_t, CS_t)
```

The symbolic input contribution should be additive or compositional inside
`SymbolicSpace`, not bolted onto `ConceptualCombine`. `ConceptualCombine` should
continue to see one `SS_t` stream.

### 3. Discriminative Perceptual Analysis

The current PS codebook often behaves like a nearest vector table. That is too
symbolic for the role Tibetan philosophy assigns to specifically characterized
entities. A specifically characterized percept should be subsymbolic: present as
this region, this timing, this intensity/shape, not as a remembered generic row.

So the subsymbolic loop can use a codebook, but not a representational content
codebook. Its PS codebook is **eidetic and continuous** because it discriminates
the given InputSpace field directly. It chunks by applying analytic functions
that return **dichotomies with no content**: boundaries, masks, partitions,
supports, and region relations. These dichotomies say "this side / that side"
or "this region / outside it"; they do not introduce a generic object or term
value.

The current `<chunking>` parameter should be replaced by `<analysis>`. The new
name is not cosmetic: perceptual analysis may segment, summarize, and index the
input, but it must not synthesize new surface content. Existing levers such as
`analyse`, `bpe`, and `radix` can remain implementation modes under the new
knob, but their perceptual-side contract is breakdown only. Any technique whose
main job is to spell out, complete, generate, or otherwise synthesize a surface
belongs in the symbolic loop.

Initial analysis contract:

```text
chunk bytes / span / region
  -> AnalyticDichotomy
       support / mask / boundary
       span or region metadata
       low-frequency measurements
       discriminative codebook identity
       no representational content identity
```

Start with simple, stable summaries:

- mean value over the region
- length / support size
- endpoint or boundary markers
- optional low-pass features once the mean path is green

For image-like inputs, analysis can return segmented regions. Those regions are
eidetic, continuous, and parametric, but they are not symbolic content. They are
extensional descriptions: the `.where` support plus analytic measurements over
that support. They can reconstruct or address a region, but they do not claim
to be a generic object.

For text, the initial version can reuse existing chunk metadata from the
`analyse`, `bpe`, and `radix` paths, but the perceptual side should keep only
the analytic segmentation and measurements. Any spelling, completion, or content
lookup moves to the symbolic loop. For numeric or image-like data, the same
interface can summarize spatial regions.

The plan should avoid host-side per-token synchronization in the training path.
Metadata used only for reports can stay lazy, following the existing deferred
text-reconstruction pattern.

### 4. Symbolic Content Codebook

The generally characterized side has one representational content codebook in
`SymbolicSpace`. It is heterogeneous: entries live in `.what` as descriptors,
not as separate tables for different kinds of generality. They are not mapped
through `.where`; instead, `.active`, `.where`, and `.when` are used together
to materialize selected `.what` descriptors as a tensor.

Descriptor rows can be high-frequency or low-frequency:

- HF descriptor: a local content descriptor such as a pixel, byte, glyph,
  edge, or similarly fine-grained value.
- LF descriptor: a coarse descriptor such as a frequency, mean, low-pass
  feature, region statistic, or functional summary.

Every descriptor row remains a possible `.what` value. `.active` selects or
weights rows from that codebook; `.where` places the selected content in an
extension; `.when` places it temporally when the representation has time. A
pixel descriptor and a frequency descriptor can therefore be materialized into
the same region at different resolutions without either row being permanently
bound to that region. `don-spyi` and `sgra-spyi` should be represented as
descriptor roles or facets inside this one symbolic codebook, not as separate
codebooks.

This supports both directions:

- analyze synthesized concepts: symbolic synthesis emits HF/LF descriptor
  candidates with `.active` / `.where` / `.when` placements, then analysis
  tests them against perceptual supports.
- synthesize analyzed concepts: perceptual analysis yields supports, then the
  symbolic loop chooses descriptor rows that can describe them.

But the SS content codebook is still symbolic. Moving between HF and LF
descriptor rows does not produce a specifically characterized entity. It only
moves among flavors or resolutions of generally characterized representation.
The PS codebook remains a different kind of codebook: discriminative/eidetic
over the presented field, not representational.

### 5. Extensional and Intensional SubSpaces

Subspaces should explicitly allow both extensional and intensional
descriptions:

- Extensional: direct supports, masks, boundaries, `.where`, `.when`, and
  measurements over a presented field. This is the subsymbolic,
  specifically-characterized side; its codebook selects discriminative analytic
  functions, not content.
- Intensional: codebook content, meanings, terms, categories, and other generic
  rows. This is the symbolic, generally-characterized side.

The current generic `SubSpace.materialize()` assumes a codebook prototype can be
read back directly from `.what` using `.active` when a codebook-bearing slot is
present. That is correct for symbolic intensional spaces but too narrow for
analytic percepts. Analytic percepts should materialize from extensional
support and measurement, not from generic `.what` content.

Target analytic materialization:

```python
self.analysis.materialize(self.active, self.where, self.when)
```

In words:

- `.analysis` is a discriminative function codebook or dichotomy selector, not
  a representational content codebook.
- `.active` indexes or weights the selected dichotomy/function.
- `.where` defines the region or support being reconstructed.
- `.when` defines the temporal support when present.
- `materialize()` returns the realized percept event or region descriptor for
  that selected analytic function at that region/time.

This may be too awkward to force into the current one-size `SubSpace` class.
Plan for two concrete subspace classes:

- `AnalyticSubSpace`: used by perceptual analysis; materialization is
  dichotomy/support-plus-measurement realization through a discriminative
  codebook whose rows select dichotomizing analytic functions rather than
  content prototypes.
- `SyntheticSubSpace`: used by symbolic/generative reconstruction;
  materialization is term/meaning driven and may spell out or complete surface
  structure.

The migration rule is simple: perceptual space owns analytic decomposition;
symbolic space owns synthetic completion.

### 6. Input Reconstruction as Synthesis

Reverse reconstruction should no longer treat `InputSpace.reverse()` as only a
single-stream decode. It should synthesize the original input from both branches:

```python
InputSpace.reverse(percepts_recon, concepts_recon)
```

Recombination layer:

```text
flatten(percepts_recon): [B, N]
flatten(concepts_recon): [B, N]
concat:                  [B, 2N]
InvertibleLinearLayer:   [B, 2N] -> [B, N] or square [B, 2N] -> [B, 2N]
```

Open engineering choice:

- For approximate reconstruction, a rectangular `InvertibleLinearLayer(2N, N)`
  is enough and matches the desired output width.
- For exact/perfect reconstruction, use a square `InvertibleLinearLayer(2N, 2N)`
  and thread or zero an `N`-wide augment, mirroring `ConceptualCombine`.

The first implementation should pick one mode explicitly and test it. Do not
hide exactness assumptions in comments.

### 7. Philosophy Documentation

Rename:

```text
basicmodel/doc/BuddhistParallels.md -> basicmodel/doc/Philosophy.md
```

Then update links in:

- `README.md`
- `basicmodel/doc/Spaces.md`
- `basicmodel/doc/Logic.md`
- relevant comments/tests that point readers to the old filename

Add sections at the top of `Philosophy.md` before the Buddhist material:

```text
# Philosophy

## Analytic / Synthetic (Kant)

## Epistemic Levels (Ramsey)

## Buddhist Epistemology
```

Content to add:

- Kant: analysis decomposes the given input into distinguishable conditions;
  synthesis recombines perceptual and conceptual conditions into an object of
  reconstruction or judgment.
- Ramsey: add the project-specific epistemic-level mapping here, but verify the
  exact Ramsey terminology before presenting it as scholarship.
- Buddhist epistemology: map reconstruction to a combination of:
  - `rang-mtshan`: specifically characterized particulars, grounded in
    eidetic continuous PS supports discriminated from InputSpace.
  - `spyi-mtshan`: generally characterized entities.
  - `don-spyi`: meaning-generality.
  - `sgra-spyi`: term/sound-generality.
  - one symbolic content codebook whose HF and LF descriptor rows can carry
    meaning-general and term-general roles; `.active`, `.where`, and `.when`
    materialize those descriptors in relation to perceptual supports.

The key bridge text:

```text
Input reconstruction combines specifically characterized perceptual supports
with generally characterized symbolic entities. The symbolic branch uses one
content codebook whose HF and LF descriptor rows can carry meaning-general and
term-general roles; the perceptual branch uses an eidetic continuous
discriminative codebook so it can ground determinate supports without turning
them into generic symbols.
```

## Implementation Phases

### Phase 0: Contract Tests

Add tests before changing behavior:

- `InputSpace.forward()` returns two outputs with exact shapes `[B,1,N]` and
  `[B,N,1]` for a tiny byte/raw fixture.
- `SymbolicSpace.forward()` accepts the optional direct input branch.
- Existing single-stream assumptions fail loudly at the model boundary rather
  than silently dropping the concept input.
- `InputSpace.reverse(percepts, concepts)` produces `[B,N]` / source-compatible
  reconstruction.
- `<analysis>` is accepted as the replacement for `<chunking>` in a tiny config,
  while legacy `<chunking>` is either rejected loudly or accepted only through a
  documented compatibility shim.
- perceptual analysis materializes from analytic dichotomies plus `.where` /
  `.when`, using a discriminative codebook rather than a representational
  content codebook.
- the symbolic codebook can store HF and LF descriptor rows in `.what`, then
  materialize them with `.active`, `.where`, and `.when` while preserving
  descriptor-role metadata for meaning-general and term-general use.

### Phase 1: InputSpace Split

Modify `InputSpace.forward()` and model staging only:

- produce `percepts_in`
- produce `concepts_in`
- preserve current host token metadata for PS embedding
- update `_lex_embed_stem` to pass only the percept branch into
  `PerceptualSpace.embed_stem`
- keep `finalize_stem` responsible for PS bookkeeping until the split is stable

Acceptance:

- tiny model forward still runs
- raw/byte shape tests pass
- no codebook behavior changes yet

### Phase 2: Symbolic Input Branch

Modify `SymbolicSpace.forward()`:

- accept `IS_concepts=None`
- on stage 0, convert `IS_concepts` into symbolic evidence
- combine it with the existing recurrent `CS_subspaceForSS` path
- keep `SymbolicSpace.perceptualSpace_ref` only for structural lexicon ownership

Acceptance:

- a test proves stage 0 symbolic output changes when `IS_concepts` changes
- a test proves stages after 0 can run without direct input
- current symbolic codebook / taxonomy tests remain green or are updated for the
  intentional contract shift

### Phase 3: Analysis Parameter and Discriminative Perceptual Chunking

Replace perceptual chunking with analytic decomposition:

- replace `<chunking>` with `<analysis>` in schema/config docs
- keep existing `analyse`, `bpe`, `radix`, and related levers only as analysis
  modes
- remove or migrate perceptual-side spell-out/synthesis behavior into the
  symbolic loop
- implement dichotomy outputs first: mask, support, boundary, and region
  relation
- implement mean/length/support measurements over those regions
- route `analyse`, `bpe`, and `radix` chunk outputs through the same
  characterization API
- keep exact byte recovery through existing inverse tables only as report or
  compatibility metadata, not as the meaning of the analytic percept
- add report-only metadata lazily

Acceptance:

- variable-length chunks map to stable fixed-width summaries
- image-like segmentation can return extensional region descriptors through the
  discriminative PS codebook without creating symbolic content rows
- analytic dichotomies can reconstruct/address a `.where` region when
  materialized
- perceptual analysis does not synthesize new surface content
- perceptual analysis does not create SS content descriptor rows
- no training-path per-token host sync regressions

### Phase 4: Extensional and Intensional SubSpaces

Split materialization behavior:

- introduce `AnalyticSubSpace` for PS-side descriptor-plus-region realization
- introduce `SyntheticSubSpace` or a symbolic equivalent for SS-side
  term/meaning-driven generation
- keep shared base behavior only where the semantics are genuinely identical
- update materialization callers to avoid assuming `.what[active]` is the whole
  event

Acceptance:

- analytic materialization calls into an analysis/dichotomy function bank with
  `(active, where, when)`, not into a representational content codebook
- synthetic materialization remains responsible for spelling out or completing
  surfaces
- symbolic materialization can use the `.what` content codebook together with
  `.active`, `.where`, and `.when`
- tests distinguish analytic decomposition from symbolic synthesis

### Phase 5: Symbolic Descriptor Codebook

Formalize the single generally characterized symbolic content codebook:

- represent each symbolic row as a descriptor with kind/role metadata, such as
  HF pixel/value descriptor, LF frequency/summary descriptor, `don-spyi`, or
  `sgra-spyi`
- define materialization so `.active` selects descriptor rows from `.what`,
  while `.where` and `.when` place the selected descriptor values in the output
  tensor
- allow vectors/activations to pass between descriptor families inside the same
  codebook
- keep the representational SS content codebook out of the subsymbolic PS loop
- use symbolic chunking here, where chunking can create a generic content row

Acceptance:

- an HF descriptor row and an LF descriptor row can be materialized at the same
  `.where` / `.when` placement
- meaning-general and term-general roles are metadata or descriptor roles in
  one SS codebook, not separate SS codebooks
- analyzed supports can be described by symbolic rows without becoming PS
  codebook content
- synthesized symbolic rows can be checked against perceptual supports
- tests prove the PS loop uses a discriminative/dichotomizing codebook, not a
  representational content codebook, in the configured path

### Phase 6: Synthesis Reverse

Modify reverse path:

- have `PerceptualSpace.reverse()` produce the percept reconstruction branch
- have `SymbolicSpace` or `ConceptualSpace.reverse()` expose the conceptual
  reconstruction branch
- call `InputSpace.reverse(percepts_recon, concepts_recon)`
- add the recombination `InvertibleLinearLayer`

Acceptance:

- direct `InputSpace.forward()` then `InputSpace.reverse()` roundtrip test
- model-level reverse reconstruction finite on `MM_xor` or a smaller fixture
- if perfect mode is claimed, assert the augment-threaded exact path; otherwise
  document that reconstruction is approximate and loss-trained

### Phase 7: Philosophy Rename and Link Cleanup

Doc-only changes:

- rename `BuddhistParallels.md` to `Philosophy.md`
- add Kant / Ramsey / Buddhist reconstruction sections
- update Markdown links and code comments that mention the old doc name

Acceptance:

- no broken Markdown links for the renamed doc
- README points to `Philosophy.md`
- the new philosophy section explicitly explains the
  `rang-mtshan` / `spyi-mtshan` / `don-spyi` / `sgra-spyi` mapping

## Risks

- Shape churn: many tests currently assert single-arg `InputSpace` and
  `SymbolicSpace` contracts. Treat failures as expected migration work, not
  regressions.
- Exactness ambiguity: `2N -> N` recombination cannot be a full bijection over
  arbitrary branch pairs. Perfect reconstruction needs a square layer plus an
  augment or a constrained analysis split.
- Codebook semantics: changing PS codebook from representational vector identity
  to discriminative regional analysis may affect nearest-neighbor assumptions in
  `radix`, `bpe`, and report rendering.
- Subspace semantics: splitting analytic and synthetic materialization reduces
  ambiguity but may touch many generic `SubSpace.materialize()` callers.
- Config migration: `<chunking>` has many existing callers/tests; the plan needs
  a deliberate compatibility policy rather than silent aliasing forever.
- Generality drift: allowing one SS codebook to hold both HF and LF descriptors
  can make symbolic descriptors feel "perceptual" unless the plan keeps PS
  discriminative, eidetic, continuous, and extensional.
- Philosophy rename blast radius: the old filename is referenced in docs,
  comments, and tests.

## Done Criteria

- Both PS and SS receive input-derived branches.
- `<analysis>` replaces perceptual `<chunking>`, and perceptual analysis is
  breakdown-only.
- Synthesis behavior has moved to the symbolic loop.
- The subsymbolic/perceptual loop uses an eidetic continuous codebook that is
  discriminative/dichotomizing on InputSpace.
- Perceptual chunking works through analytic dichotomies selected by a PS
  codebook, without turning those selections into symbolic content prototypes.
- A single SS content codebook stores HF and LF descriptor rows in `.what`,
  including any `don-spyi` / `sgra-spyi` roles, and materializes them with
  `.active`, `.where`, and `.when`.
- Analytic and synthetic subspace materialization are separate enough that PS
  cannot accidentally become symbolic through a representational content
  codebook.
- Reverse reconstruction uses both perceptual and conceptual branches.
- The recombination layer is explicit and tested.
- `Philosophy.md` replaces `BuddhistParallels.md` and documents Kant, Ramsey,
  and the Buddhist specifically/general-characterized reconstruction mapping.
