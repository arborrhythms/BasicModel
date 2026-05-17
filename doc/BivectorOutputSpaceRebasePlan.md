# Bivector Plan: `MM_5M_bivector.xml` OutputSpace re-base fix + fail-fast checkpoint guard

> Status: **design only — not yet implemented** (deferred per owner; the
> separate `cudaMemcpyDtoH` brick work is tracked independently and is
> *not* part of this plan). The earlier "decouple bivectorOutput from
> the `codebook` flag" idea was based on a misread config (CS/SS are
> `<codebook>true`) and was fully reverted.

## Context

`python bin/train.py --model data/MM_5M_bivector.xml --data text --num-epochs 1 --batches 10`
crashes; `MM_5M.xml` (non-bivector) trains fine. Investigation (proven by
experiment) found **two independent issues**:

**Bug #1 — OutputSpace re-base shape mismatch (deterministic; the real blocker).**
`MM_5M_bivector.xml` has grammar rules (`intersection`/`union`) ⇒
`useGrammar="all"`, `conceptualOrder=3`. In `_build_pipelines_per_stage`
the **grammar path** ([Models.py:3891-3900](bin/Models.py)) sets each CS
stage `cs_out = [n_t, percept_dim+obj_percept]` (= width **10**) and
**ignores `<ConceptualSpace><nOutputDim>2</nOutputDim>`** — the *plain*
path ([3901-3907](bin/Models.py)) honors it via
`conceptOutputShape`/`_c_nOutputDim`, the grammar path does not. The
OutputSpace re-base ([Models.py:3995-3997](bin/Models.py)) sizes the head
from `conceptualSpaces[-1].outputShape[1]` (= stale **10**), but a
`bivectorOutput=true` CS actually emits a catuskoti `[B, V_C, 2]`
(`ProjectionBasis.forward`, [Spaces.py:2230-2232](bin/Spaces.py),
provably non-negative). `OutputSpace.forwardBegin`
([Spaces.py:5523](bin/Spaces.py)) then does
`x.reshape(B, -1, nInputDim=256*10=2560)` on a size-65536
(=128·256·**2**) event → `RuntimeError`. Confirmed deterministic: fresh
build (autoload off) ×4 all crash here.

Key fact (verified by trace + live probe): the declared CS `outputShape`
width is **read only by the OutputSpace re-base**, never for inter-stage
data flow (the recurrent cell hands off via `_subspaceForPS/SS` with
shape-flexible `forwardBegin` reshape; `GrammarMergeGlue` only halves N,
never touches D). And `ConceptualSpace.nOutputDim` is the **true emitted
feature width** in every config:

| config | term CS `.outputShape[1]` | term CS `.nOutputDim` | bivec |
|---|---|---|---|
| MM_5M_bivector (grammar+bivec, **bug**) | 10 (stale) | **2** (correct) | yes |
| MM_xor (grammar, non-bivec) | 10 | 10 | no |
| MM_xor_bivector (plain+bivec) | 2 | 2 | yes |

`.nOutputDim == .outputShape[1]` for every currently-passing config, and
differs **only** in the buggy case (where `.nOutputDim` is the correct
value).

**Bug #2 — flaky `normalize("bivector")` was a stale checkpoint, not a
code bug.** `ProjectionBasis.forward` is provably non-negative; fresh
builds never hit the assert. The original `-0.795` came from autoloading
the stale `MM_5M_bivector.ckpt`, whose architecture no longer matches the
config. `load_weights` ([Models.py:1346](bin/Models.py)) detects the
mismatch ([1418-1442](bin/Models.py)) but only **soft-warns**
(`TheMessage(...); return False`); the autoload caller
([Models.py:672](bin/Models.py), the *only* `load_weights` caller, gated
by `_t("autoload")`) ignores the return value, so the model proceeds on
fresh weights and crashes downstream. Desired: a **fail-fast guard**
instead of soft-warn + later crash. (Owner manages the `.ckpt`.)

## Fix (recommended approach)

### Fix #1 — OutputSpace re-base reads the true emitted width (Models.py:3995-3997)

One-semantic-line change: derive the head input width from the terminal
CS's **`nOutputDim`** (true emitted feature width — bivector-aware: 2 for
bivec, `==outputShape[1]` otherwise) instead of the stale declared
`outputShape[1]`. Keep `output_n = outputShape[0]` (the N axis is
correct; only the width is stale). Replace:

```python
_term_cs_shape = list(self.conceptualSpaces[-1].outputShape)
output_n = int(_term_cs_shape[0])
outputInputShape = [output_n, int(_term_cs_shape[1])]
```
with (read `nOutputDim`; update the now-accurate comment):
```python
_term_cs = self.conceptualSpaces[-1]
output_n = int(_term_cs.outputShape[0])
# Width = the terminal C stage's ACTUAL emitted feature width.
# nOutputDim is bivector-aware (2 under <bivectorOutput>; ==outputShape[1]
# otherwise); the grammar path's declared outputShape[1] is the stale
# percept width and would mis-size nInputDim for bivector CS.
outputInputShape = [output_n, int(_term_cs.nOutputDim)]
```

Rationale over "make the grammar path honor `<nOutputDim>`" (considered,
rejected as primary): the declared `outputShape` tuple is not used for
data flow (only this re-base reads it), so making it "truthful" adds a
broader edit to shared `_build_pipelines_per_stage` for **no functional
gain** beyond this 1-line consumption-site fix, with larger blast radius.
This fix is provably regression-safe (identical for every passing config;
correct for the bug).

### Fix #2 — fail-fast on autoload architecture mismatch (Models.py:1346, 1426, 672)

The legitimate non-strict cases (vocab grow via `_restore_vocab_extras`,
one-time bivector migration) are reconciled *before* the mismatch block
at [1418](bin/Models.py); benign stale **extra** keys (`unexpected`-only,
`strict=False`) do **not** enter the
`if mismatches or missing or fatal_unexpected:` block (they load fine
with the existing "Ignored N stale keys" message). So remaining
`mismatches` (shape drift) / `missing` (model expects keys the ckpt
lacks) at [1426](bin/Models.py) are genuine, crash-causing config drift.

- Add a `require_match: bool = False` param to
  `load_weights(self, path=None, strict=False)`
  ([Models.py:1346](bin/Models.py)).
- In the mismatch block ([1426-1442](bin/Models.py)): build the existing
  actionable `lines` message; if
  `require_match and (mismatches or missing or fatal_unexpected)` →
  `raise ValueError("\n".join(lines))` instead of
  `TheMessage(...); return False`. Non-autoload callers keep soft-warn.
- Autoload caller ([Models.py:672](bin/Models.py)):
  `self.load_weights(wpath, require_match=True)`.

Fails fast at the real detection point with the already-actionable
diagnostic ("correct the XML … or delete/move `<path>`"), preserves
vocab-grow / bivector-migration / benign-stale-key loads, only hardens
the autoload path (the single caller).

## Critical files

- `bin/Models.py` — Fix #1 re-base [3995-3997]; Fix #2 `load_weights`
  sig [1346], mismatch block [1426-1442], autoload caller [672].
- `bin/Spaces.py` — no change; `OutputSpace.__init__` (~[10243],
  `flat_in = inputShape[0]*inputShape[1]`) and `forwardBegin` ([5519])
  are the consumers Fix #1 makes consistent.
- New test: `test/test_bivector_grammar_outputspace_rebase.py`.
- Bug #2 test: extend an existing weights/checkpoint test (locate via
  `grep -rl "load_weights" test/`) or add a focused case.

## Verification

From `basicmodel/`, `BASICMODEL_DEVICE=cpu PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m pytest`.

1. **TDD red→green (Fix #1):** new test builds a grammar+bivector model
   (temp config = `MM_xor_bivector.xml` with `intersection`/`union`
   grammar + `conceptualOrder=3`, or `MM_5M_bivector.xml` with
   `autoload=false` so the owner ckpt is untouched) and asserts
   `outputSpace.nInputDim == terminal_CS.outputShape[0] * terminal_CS.nOutputDim`
   and a `torch.no_grad()` forward returns the 4-tuple (no reshape
   crash).
2. **TDD red→green (Fix #2):** `load_weights(path, require_match=True)`
   against a deliberately shape-mismatched state_dict **raises
   ValueError** (and `require_match=False` still soft-warns / returns
   False; a benign extra-key ckpt still loads).
3. **Real repro:** `python bin/train.py --model data/MM_5M_bivector.xml
   --data text --num-epochs 1 --batches 3` — with a fresh/matching ckpt
   (or `autoload=false`) now trains; with the stale ckpt + default
   autoload it now **fails fast** with the actionable message.
4. **Regression sweep:** bivector / invertibility / e2e xor·mm /
   phase2 then full `test/ -q`; confirm `MM_xor` and `MM_xor_bivector`
   unchanged.

Success: new tests green; `MM_5M_bivector` fresh trains; stale ckpt
fails fast; full sweep shows no new regressions.

## Risks

1. **Fix #1 blast radius:** mitigated — `.nOutputDim == .outputShape[1]`
   for every passing config; only the buggy path changes. Re-confirm via
   the sweep.
2. **Output loss:** in IR mode the head loss is zero-weighted (side
   channel), so Fix #1's width change does not perturb the trained
   objective; verify the forward 4-tuple + `lossIn` path still finite.
3. **Fix #2 over-strictness:** only the autoload caller passes
   `require_match=True`; migration / vocab-grow run *before* the guarded
   block; `unexpected`-only stale keys don't trip it. Pin all three.
4. **Owner-managed ckpt:** do not move/delete `data/MM_5M_bivector.ckpt`;
   the guard just makes existing drift fail fast and actionable.

---

## Design rationale & open question: TruthLayer × tetralemma

This records *why* the bivector matters and an unresolved design
question raised during review (kept here so the rationale travels with
the plan).

The bivector activation `[aP, aN]` (affirming / denying evidence, each in
`[0, 1]`) encodes the catuskoti / tetralemma four corners:

| corner | `aP` | `aN` | reading |
|---|---|---|---|
| TRUE | 1 | 0 | asserted |
| FALSE | 0 | 1 | denied |
| NEITHER | 0 | 0 | ignorance / no commitment |
| BOTH | 1 | 1 | contradiction |

Its value over a signed scalar is exactly that it keeps **contradiction**
(`BOTH ≈ aP·aN`) and **ignorance** (`NEITHER ≈ (1−aP)(1−aN)`) as
separate coordinates — the two corners a scalar `aP−aN` collapses to ~0.
(See `doc/BuddhistParallels.md` for the catuskoti mapping and
`test/test_conceptual_bivector.py` for the distinguishability contracts.)

**Open question (unresolved):** measuring truth-with-contradiction is
valuable *especially for the TruthLayer*, but it is unclear to what
degree the measurement of a given proposition *with respect to that
region of truth* actually utilizes the tetralemma. The crux: a
measurement "utilizes the tetralemma" **iff the measurement operator
itself is bivector-valued / paraconsistent**. If the TruthLayer scores a
proposition against the accumulated truth region via anything that
reduces to the signed projection (cosine / dot / `aP−aN` then
threshold), the four-valuedness is *stored but not utilized* — "the
region contradicts p" and "the region is silent on p" both read as low
truth, which is precisely the degeneracy the bivector was meant to
prevent.

**Concrete way to resolve it (grounded, not speculative):** trace the
TruthLayer DoT/measurement path (`bin/Language.py` `truth_modulated_loss`
/ `bin/Layers.py` `TruthLayer`, the `accumulateTruth` consumers) and
determine whether the proposition-vs-region comparison collapses the
bivector to a scalar before comparing, or branches on the contradiction
(`aP·aN`) and ignorance (`(1−aP)(1−aN)`) components. If it collapses,
the bivector is not utilized regardless of how it is stored — that is
the change point to evaluate (make the comparison/DoT update treat
"contradicts the region" distinctly from "region silent on p" — i.e.
paraconsistent vs. intuitionistic handling). This is investigation +
design, **out of scope for the two fixes above**, recorded here as a
deliberate follow-up.
