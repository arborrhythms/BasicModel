# MM_20M architecture backlog (expanded)

Four parked items that surfaced while chasing radix reconstruction + XOR.
None is started; each needs the work (and in two cases the *decision*) below.
Cross-refs: `doc/plans/2026-06-08-mm20m-xor-collapse.md` (XOR pipeline
collapse) and `doc/plans/2026-06-08-radix-mps-adam-step-zero.md` (the MPS
optimizer fix). Backlog ids are the in-session task numbers.

---

## 1. Head / endpoint dim sizing (Task #11) -- ARCHITECTURAL, needs a decision

### Problem
The OutputSpace head cannot represent more than one output value. `MM_20M.xml`
declares:

```xml
<OutputSpace>
  <nInput>8</nInput>  <nInputDim>1024</nInputDim>
  <nVectors>1</nVectors> <nDim>1</nDim>
  <nOutput>1</nOutput> <nOutputDim>1</nOutputDim>
</OutputSpace>
```

A codebook with `nVectors = 1` can only ever quantize to a single prototype
row. So even if the conceptual/symbolic tiers preserve a per-row distinction,
the head collapses all four XOR rows to one representable value -- the MSE
optimum is then the constant row-mean ($\frac{1}{4}(0+1+1+0) = 0.5$). This was
flagged during the XOR-collapse analysis as an independent contributor: it
caps the head's expressiveness regardless of upstream fixes.

### Current mechanism
Endpoint dims are resolved through `BaseModel._resolve_dim`
(`bin/Models.py:308`) and `canonical_shape` (`bin/architecture.py`, imported at
`Models.py:62`). The "respect-explicit" concern is that endpoint
(InputSpace / OutputSpace) widths are auto-derived / canonicalised rather than
honouring the explicitly configured `nVectors` / `nDim`, so a config author
cannot reliably size the head for the target output cardinality.

### Decision required
Pick the head representation for a low-cardinality / scalar output:

1. **Validate-and-require** -- enforce `nVectors $\geq$ n_classes` for a
   quantised classification head; fail loud on a 1-vector head used for a
   $>1$-class target.
2. **Regression head** -- when `nVectors = 1` (or codebook absent), make the
   head an *unquantised* linear map (`nInputDim $\to$ nOutputDim`) with a
   sigmoid/identity readout, so a scalar target is representable without a
   codebook.
3. **Respect-explicit** -- stop auto-forcing endpoint dims; honour the
   configured `nVectors`/`nDim` verbatim, and let the config author size the
   head (this is the narrower fix the task title names).

Recommendation: (2)+(3) together -- a config-honoured unquantised regression
head is the natural fit for binary/scalar supervised outputs (XOR, the
`predicted` column), while quantised heads stay available for symbolic outputs.

**DECISION (2026-06-09): (2)+(3) accepted.** Build the config-honoured
unquantised regression head; respect explicit endpoint dims. Scheduled in
the 2026-06-09 build batch.

### Work
- Decide (1)/(2)/(3) above.
- Add the chosen head path in OutputSpace construction; gate on
  `nVectors`/codebook presence.
- If validating, add the check to `validate_config`.
- Stop the endpoint-dim auto-force in `_resolve_dim` for explicitly-set
  endpoints (respect-explicit).

### Acceptance
- A binary-label task (XOR) has a head that can represent both classes.
- InputSpace/OutputSpace dims honour explicit config; no silent canonical
  override of an explicitly-set endpoint width.
- `make test` green.

---

## 2. Remove `codebook` from PS / SS in model.xsd (Task #13)

> **Decided (2026-06-09):** drop the `<codebook>` *element* (the config knob)
> from **both** PS and SS -- it is no longer optional. The codebook itself
> **stays in both spaces**; only the settable option is removed: **SS**
> codebook is mandatory (`quantize`, hardwired), and **PS** codebook is
> **integrated with `<chunking>`** (the chunking / radix store *is* the PS
> codebook). Scheduled in the 2026-06-09 build batch.

### Problem
`data/model.xsd` exposes `<codebook>` (`codebookModeEnum`: `none | quantize |
project`) on both **PerceptualSpace** (`model.xsd:504`) and **SymbolicSpace**
(`model.xsd:542`). After the modality re-architecture the PS/SS codebook is no
longer a free choice -- configs carry comments like *"PerceptualSpace codebook
is mandatory (modality re-architecture); was none"* and *"SymbolicSpace
codebook is mandatory ...; was none"*. The element is now vestigial /
misleading: a config can set an inert or self-contradictory value (e.g.
`codebook=none` on a tier where the codebook is structurally required).

### Work
- Remove the `<codebook>` element from the PS and SS complexTypes in
  `data/model.xsd` (lines 504 and 542). Keep `codebookModeEnum` only if a
  remaining tier still uses it; otherwise retire the type too.
- Remove the readers (`TheXMLConfig.space(section, "codebook")` for PS/SS) and
  fix the resulting behaviour to the single re-architecture-mandated mode.
- Update every config that sets `<codebook>` on PS/SS (`XOR_exact.xml`,
  `MM_20M.xml` comments, the smoke configs) -- delete the lines.
- Update the sweep harness: `test/_sweep_chunking.py` overrides
  `SWEEP_CODEBOOK` / `SWEEP_PS_CB` / `SWEEP_SS_CB` by string-substituting the
  `<codebook>` element; those substitutions must be removed or repointed.

### Caveats
- Confirm the "mandatory" target mode per tier (PS vs SS may differ -- SS in
  MM_20M ships `quantize`, PS ships `none`). The re-architecture intent must be
  pinned before deleting the knob, or this silently changes behaviour.
- Tied to Task #16: if SS stops snapping (decodes pre-snap $z$), the SS
  codebook semantics shift -- sequence #16 first or co-design.

### Acceptance
- `<codebook>` is not a settable option on PS/SS in the schema.
- No live config sets it; behaviour is fixed per the re-architecture.
- Schema validates; `make test` green.

---

## 3. SS reverse -- decode the continuous pre-snap $z$ (Task #16): recon + XOR together

> **Co-designed (2026-06-09)** with `2026-06-09-asymmetric-vq-symbolic-ss.md`:
> the asymmetric routing's *input $\to$ codebook* leg (recon gradient on the
> continuous code, no STE) is the same pre-snap-$z$ idea.

### Problem
`SymbolicSpace.forward` snaps the symbolic activation to the codebook: after
`act = l1_proximal(act)` (`bin/Spaces.py:15467`) the intrinsic snap
(`Codebook.forward`, "naming the closest point in concept space") quantises the
continuous activation $z$ to the nearest discrete prototype. The snap is
**lossy** -- it discards the continuous offset $z - \mathrm{prototype}(z)$.

Two objectives conflict on this snap:
- **Reconstruction / XOR** want the *continuous* $z$ (the fine-grained signal
  the head needs to separate non-snapped classes).
- **Symbolic discreteness** wants the snapped prototype (the named symbol).

Today the reverse path decodes from the snapped prototype, so the continuous
distinction is gone by the time the head / reconstruction reads it.

### Approach
Capture the **pre-snap continuous $z$** in the forward and thread it to the
reverse, so:
- forward still snaps (the symbolic state stays discrete), and
- `SymbolicSpace.reverse` decodes from $z$ (the continuous pre-snap value),
  not from $\mathrm{prototype}(z)$.

This is the VQ-VAE straight-through pattern applied to the *reconstruction*
leg: the symbol is the codeword, but the reconstruction (and the head's XOR
read) sees $z$. It lets reconstruction stay byte-perfect while the continuous
signal survives for the supervised head -- "recon + XOR together".

### Work
- In `SymbolicSpace.forward`, stash the pre-snap activation $z$ (before the
  `Codebook.forward` snap) as a forward-local (thread it, do not persist on the
  layer -- per the parallel-recurrence data-flow rule).
- In `SymbolicSpace.reverse` (and the `_reverse_body` SS leg), decode from the
  threaded $z$ when present; fall back to the snapped prototype otherwise.
- Ensure the head's symbolic read (if the head reads SS) sees $z$, not the
  prototype.
- Keep invertibility / the perfect-reconstruction round-trip intact.

### Dependencies
- Interacts with the **combine-output wiring** (the open XOR piece in
  `2026-06-08-mm20m-xor-collapse.md`): if the head reads the snapped SS, both
  this and the wiring are needed for XOR; verify which tier the head actually
  reads.
- Co-design with Task #2 (#13): removing the PS/SS codebook knob changes the
  snap's configurability.

### Acceptance
- SS reverse reconstructs from continuous pre-snap $z$; reconstruction stays
  byte-perfect on the smoke prompts.
- The continuous symbolic signal is available to the supervised head (no
  quantisation cliff between SS and the head).
- Perfect-reconstruction round-trip tests still pass.

---

## 4. BPE / MPHF reconstruction -- codebook-sharing refactor (deferred by user)

### Problem
`chunking=bpe` and `chunking=mphf` do not reconstruct the input the way
`radix` / `lexicon` do. The earlier "constant-collapse" report was a harness
artefact (the sweep defaulted to `lexer=raw`, an empty byte buffer; `bpe`/`mphf`
require `lexer=byte`). The *real* gap is that each chunking front end embeds
through its own representation -- `_embed_bpe` (`bin/Spaces.py:8956`),
`_embed_mphf` (`9796`), `_embed_radix` (`8807`), `_embed_lexicon` (`9720`),
`_embed_byte` (`9629`) -- and `bpe`/`mphf` chunks do not decode back through a
codebook surface shared with the byte/radix path. So the reverse (chunk
$\to$ bytes) is not uniform, and reconstruction quality differs by mode.

### Approach (the deferred refactor)
Share **one canonical codebook surface** -- the `byte_mode` codebook (the
authoritative surface-form codebook used by radix's PerceptStore) -- across all
chunking methods, so a `bpe`/`mphf` chunk decodes through the same byte
codebook on the reverse path that `radix` uses. The chunking front end becomes
purely a *segmentation* strategy over a shared embedding/codebook, not a
separate codebook owner.

### Work
- Make the byte/`PerceptStore` codebook the shared backing store; have
  `_embed_bpe` / `_embed_mphf` resolve their chunks to ids in that shared
  codebook rather than a private table.
- Implement the `bpe`/`mphf` reverse (chunk-id $\to$ surface bytes) against the
  shared codebook, mirroring `RadixLayer.reverse`.
- Verify reconstruction on the four smoke prompts for `bpe` and `mphf` matches
  `radix`/`lexicon` (byte-perfect, with `lexer=byte`).

### Status
**Scheduled (2026-06-09 batch).** Previously deferred ("bigger refactor, can
wait"); pulled into the 2026-06-09 build batch after the XOR / asymmetric-vq
work.

### Acceptance
- `bpe` and `mphf` reconstruct the smoke prompts byte-perfect under
  `lexer=byte`, equivalent to `radix`/`lexicon`.
- A single shared codebook backs all chunking modes; no per-mode private
  codebook for the surface form.

---

## Suggested sequencing

1. **#16 (SS pre-snap $z$)** + the combine-output wiring -- jointly unblock XOR
   while keeping reconstruction (these two are the live path).
2. **#11 (head sizing)** -- so the head can actually represent the XOR output
   once the upstream signal arrives.
3. **#13 (remove PS/SS codebook knob)** -- schema cleanup, co-designed with #16.
4. **BPE/MPHF codebook-sharing** -- the larger deferred refactor, last.
