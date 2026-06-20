# PerceptualSpace `<bivectorOutput>` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `<bivectorOutput>true</bivectorOutput>` support to PerceptualSpace per spec B3 (`doc/specs/2026-04-24-lift-lower-bivector-design.md` §B3, line 908), mirroring the existing wiring in ConceptualSpace ([Spaces.py:7228-7236](../../bin/Spaces.py)) and SymbolicSpace ([Spaces.py:7679-7692](../../bin/Spaces.py)), and verify with `MM_xor_bivector.xml` end-to-end.

**Architecture:** PerceptualSpace's per-slot output activation is promoted from a bitonic scalar in `[-1, +1]` to a paired-index bivector `[aP, aN] ∈ [0, 1]²` via the canonical Q2 split `aP = max(0, x); aN = max(0, -x)` (spec §Q2, line 1405). The split happens at the activation boundary inside `PerceptualSpace.forward()`; the inverse `x = aP − aN` runs at the reverse boundary. PerceptualSpace's `.what` lexicon / Embedding stays untouched — only the activation tensor changes shape.

**Tech Stack:** Python 3.12, PyTorch, the project's XML config system (`TheXMLConfig`).

---

## Context — `<subsymbolicEnabled>` is documentation-only

The 2026-05-08 conceptual-loopback design (`doc/plans/2026-05-08-bivector-activation-conceptual-loopback-design.md` Stage 2) dropped the `<subsymbolicEnabled>` flag. The symbolic loopback (which conjoins `[P_event || S_event]` to feed ConceptualSpace via concat-on-last-axis at [Spaces.py:7396](../../bin/Spaces.py)) is now unconditional — gated only by `<ConceptualSpace><bivectorOutput>true</bivectorOutput></ConceptualSpace>` at [Models.py:1797-1820](../../bin/Models.py). No live reader for `subsymbolicEnabled` exists in the codebase. We add it to the XML as a documentation-only marker per the user's request.

The dim-alignment constraint introduced by the loopback: with both P and S bivector,
- `P.nOutputDim == S.nOutputDim == 2` (both produce per-slot bivector pairs)
- `C.nInputDim == P.nOutputDim == 2` (C's encoded input width = P's output width)
- Loopback widens C's input PiLayer by `S.nOutputDim = 2` → effective PiLayer input = 4
- `C.nDim` (codebook prototype width) is independent — stays at 6 in MM_xor_bivector
- `P.nOutput == S.nOutput == 8` (N axes must match for the concat — already holds)

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `bin/Spaces.py` | `PerceptualSpace` class | Modify `__init__`, `forward`, `reverse` |
| `data/MM_xor_bivector.xml` | E2E bivector test config | Modify `<PerceptualSpace>`, `<ConceptualSpace>`, `<architecture>` |
| `test/test_perceptual_bivector.py` | Unit + round-trip tests | Create |

---

## Task 1: Read `bivectorOutput` and `svdOrthogonalInit` flags in `PerceptualSpace.__init__`

**Files:**
- Modify: `bin/Spaces.py:6047` (`PerceptualSpace.__init__`)

- [ ] **Step 1.1: Add the flag reads at the top of `__init__`**

Insert immediately after the `naive = TheXMLConfig.get(...)` read at [Spaces.py:6054](../../bin/Spaces.py), so the flags are available for any what-basis builder running inside `super().__init__()`. Mirror the pattern from [ConceptualSpace.__init__:7228-7236](../../bin/Spaces.py).

```python
        naive = TheXMLConfig.get("architecture.naive")
        # Bivector regime flags (spec B3, doc/specs/2026-04-24-lift-lower-
        # bivector-design.md §B3). When ``bivectorOutput`` is true, the
        # per-slot percept activation is the catuskoti bivector
        # ``[aP, aN] = (max(0, x), max(0, -x))`` (Q2 promotion, spec
        # line 1405). ``svdOrthogonalInit`` is reserved for symmetry with
        # ConceptualSpace / SymbolicSpace; it is consulted only when a
        # Codebook is built on ``.what`` (codebook=true mode), not on the
        # default Embedding lexicon.
        try:
            self._bivector_output = bool(
                TheXMLConfig.space(section, "bivectorOutput"))
        except KeyError:
            self._bivector_output = False
        try:
            self._svd_orthogonal_init_cfg = bool(
                TheXMLConfig.space(section, "svdOrthogonalInit"))
        except KeyError:
            self._svd_orthogonal_init_cfg = False
```

- [ ] **Step 1.2: Verify the flags load**

Run from `basicmodel/`:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'bin')
from Config import TheXMLConfig
TheXMLConfig.load('data/MM_xor_bivector.xml')
print('PerceptualSpace.bivectorOutput =',
      TheXMLConfig.space('PerceptualSpace', 'bivectorOutput'))
"
```

Expected: `KeyError` (because the XML doesn't have the flag yet — that comes in Task 4). The default-fallback `False` path is exercised; this confirms the `try/except KeyError` wraps as expected.

---

## Task 2: Apply Q2 promotion at the `forward()` boundary

**Files:**
- Modify: `bin/Spaces.py:6780` (`PerceptualSpace.forward`)

The percept activation today is whatever `forwardEnd` writes through to `subspace.activation` (per-slot scalar derived from the codebook snap or, when codebook=false, from the post-attention event norm). Under bivector mode, replace that scalar with the paired-index `[aP, aN]` per slot.

- [ ] **Step 2.1: Add a helper `_q2_promote_activation` on `PerceptualSpace`**

Insert immediately after the existing `_slot_forward` method (around [Spaces.py:6778](../../bin/Spaces.py)):

```python
    def _q2_promote_activation(self, event: torch.Tensor) -> torch.Tensor:
        """Q2 bitonic-to-bivector promotion (spec §Q2, line 1405).

        Reduces the per-slot percept event to a signed scalar via signed-
        sum across the content dim, then splits onto the non-negative
        paired-index axes ``(aP, aN) = (max(0, x), max(0, -x))``.

        Args:
            event: ``[B, N, D_P]`` per-slot percept content.

        Returns:
            ``[B, N, 2]`` bivector activation, monotonic in ``[0, 1]^2``.
        """
        # Signed sum over the content dim: each percept's net signed
        # presence. For bitonic event values in ``[-1, +1]``, this lives
        # in ``[-D_P, +D_P]`` -- the magnitude carries activation
        # strength, the sign carries polarity.
        x = event.sum(dim=-1)
        aP = torch.relu(x)
        aN = torch.relu(-x)
        return torch.stack([aP, aN], dim=-1)
```

- [ ] **Step 2.2: Apply the promotion before `return vspace`**

Modify the end of `PerceptualSpace.forward` (around [Spaces.py:6900-6902](../../bin/Spaces.py)) — insert immediately before the final `return vspace`:

```python
        if self._bivector_output:
            # Q2 promotion: replace the legacy per-slot scalar activation
            # with the catuskoti bivector ``[B, N, 2]``. Downstream
            # ConceptualSpace consumes this as the left half of the
            # widened ``[P_event || S_event]`` PiLayer input (gated on
            # ConceptualSpace.bivectorOutput at Models.py:1797-1820).
            event = vspace.materialize(mode="event")
            if event is not None and event.dim() == 3:
                bivec = self._q2_promote_activation(event)
                vspace.activation.setW(bivec)

        return vspace
```

- [ ] **Step 2.3: Run the existing PerceptualSpace tests to confirm no regression**

```bash
cd basicmodel && .venv/bin/python -m pytest test/test_perceptualspace_bpe_forward.py -x -q
```

Expected: all PASS. The bivector path is gated on `self._bivector_output` and these tests don't set the flag, so they take the legacy path.

---

## Task 3: Apply inverse split at the `reverse()` boundary

**Files:**
- Modify: `bin/Spaces.py:6904` (`PerceptualSpace.reverse`)

- [ ] **Step 3.1: Add a helper `_q2_lower_activation` on `PerceptualSpace`**

Insert immediately after `_q2_promote_activation` from Task 2:

```python
    def _q2_lower_activation(self, bivec: torch.Tensor,
                             content_dim: int) -> torch.Tensor:
        """Inverse of `_q2_promote_activation`: bivector -> per-slot
        signed scalar -> broadcast to per-slot content vector.

        The forward Q2 promotion is many-to-one over the content dim
        (signed sum collapses ``D_P`` features to one scalar), so the
        reverse cannot recover the per-feature pattern. We broadcast the
        recovered scalar uniformly across the content dim; downstream
        ``reverseEnd`` / ``InvertibleLinearLayer.reverse`` handles any
        further structure recovery.

        Args:
            bivec: ``[B, N, 2]`` bivector activation.
            content_dim: target ``D_P`` for the output event.

        Returns:
            ``[B, N, D_P]`` event tensor with the recovered scalar
            broadcast across the content axis.
        """
        x = bivec[..., 0] - bivec[..., 1]              # [B, N]
        return x.unsqueeze(-1).expand(-1, -1, content_dim).contiguous()
```

- [ ] **Step 3.2: Apply the lowering at the start of `reverse`**

Modify the start of the non-Embedding branch of `PerceptualSpace.reverse` (around [Spaces.py:6913-6917](../../bin/Spaces.py)):

```python
        if self.invertible:
            vspace.normalize("percepts", target="event",
                             normalize=True, reverse=True)
        if self._bivector_output:
            # Lower the bivector activation back to a per-slot signed
            # scalar broadcast across the percept content dim, so the
            # downstream invertible reverse sees the pre-Q2 layout.
            bivec = vspace.activation.getW()
            if bivec is not None and bivec.dim() == 3 and bivec.shape[-1] == 2:
                event = self._q2_lower_activation(bivec, self.subspace.nDim)
                vspace.event.setW(event)
        y = self.reverseBegin(vspace, returnVectors=True)
```

- [ ] **Step 3.3: Confirm the Embedding-mode reverse path is untouched**

The `_reverse_text` branch ([Spaces.py:6910-6912](../../bin/Spaces.py)) returns early for `isinstance(self.subspace.what, Embedding)`. Bivector mode under MM_xor_bivector does NOT hit `_reverse_text` because the XOR data is not real text and the Embedding's reverse path is text-decoding — not exercised in bivector smoke tests. Document this as a caveat in a comment above the new bivector check:

```python
        # NOTE: When ``self.subspace.what`` is an Embedding (text mode),
        # `_reverse_text` returns earlier and bypasses this bivector
        # lowering. Mixing text-mode lexicon with bivectorOutput=true is
        # outside the current B3 scope; revisit when text-mode bivector
        # is needed.
```

---

## Task 4: Update `MM_xor_bivector.xml`

**Files:**
- Modify: `data/MM_xor_bivector.xml`

Three blocks change: `<PerceptualSpace>` (add the bivector flags + drop dims to 2), `<ConceptualSpace>` (drop `nInputDim` to 2 to match P's new output), and `<architecture>` (add the documentation-only `subsymbolicEnabled` marker).

- [ ] **Step 4.1: Update `<PerceptualSpace>` block**

Replace the existing block at [data/MM_xor_bivector.xml:61-73](../../data/MM_xor_bivector.xml) with:

```xml
  <PerceptualSpace>
    <nInput>8</nInput>
    <nInputDim>4</nInputDim>
    <nVectors>8</nVectors>
    <!-- nDim is the codebook prototype width; the P-tier output is the
         per-slot catuskoti bivector [B, N, 2], so nOutputDim is 2
         regardless of nDim. Mirrors the ConceptualSpace block below. -->
    <nDim>2</nDim>
    <nOutput>8</nOutput>
    <nOutputDim>2</nOutputDim>
    <nWhere>0</nWhere>
    <nWhen>0</nWhen>
    <hasAttention>false</hasAttention>
    <invertible>true</invertible>
    <codebook>false</codebook>
    <nonlinear>false</nonlinear>
    <bivectorOutput>true</bivectorOutput>
    <svdOrthogonalInit>true</svdOrthogonalInit>
  </PerceptualSpace>
```

- [ ] **Step 4.2: Update `<ConceptualSpace>` block**

In the existing block at [data/MM_xor_bivector.xml:75-91](../../data/MM_xor_bivector.xml), change `<nInputDim>4</nInputDim>` to `<nInputDim>2</nInputDim>` so it matches PerceptualSpace's new `nOutputDim=2`. The widening (gated on `<bivectorOutput>true`) adds `S.nOutputDim=2` on top, giving the C input PiLayer width = 4.

```xml
  <ConceptualSpace>
    <nInput>8</nInput>
    <nInputDim>2</nInputDim>   <!-- was 4; now matches P.nOutputDim=2 -->
    <nVectors>8</nVectors>
    <!-- nDim is the codebook prototype width; the C-tier output is the
         per-prototype catuskoti bivector [B, V_C, 2], so nOutputDim is 2
         regardless of nDim. -->
    <nDim>6</nDim>
    <nOutput>8</nOutput>
    <nOutputDim>2</nOutputDim>
    <hasAttention>false</hasAttention>
    <invertible>false</invertible>
    <codebook>true</codebook>
    <nonlinear>false</nonlinear>
    <bivectorOutput>true</bivectorOutput>
    <svdOrthogonalInit>true</svdOrthogonalInit>
  </ConceptualSpace>
```

- [ ] **Step 4.3: Add `<subsymbolicEnabled>` marker to `<architecture>` block**

Insert at the top of `<architecture>` (after `<conceptualOrder>` at [data/MM_xor_bivector.xml:16](../../data/MM_xor_bivector.xml)). The flag is documentation-only — no live reader exists. The actual loopback gate is `<ConceptualSpace><bivectorOutput>true</bivectorOutput></ConceptualSpace>`, already set in this config.

```xml
    <conceptualOrder>1</conceptualOrder>
    <!-- Documentation marker: the symbolic loopback (P_event || S_event
         concat into ConceptualSpace's PiLayer) is unconditional under
         the bivector regime per
         doc/plans/2026-05-08-bivector-activation-conceptual-loopback-design.md
         Stage 2. The actual gate is <ConceptualSpace><bivectorOutput>;
         no code reads this flag. -->
    <subsymbolicEnabled>true</subsymbolicEnabled>
    <useButterflies>false</useButterflies>
```

- [ ] **Step 4.4: Update the header comment to mention PerceptualSpace bivector**

Update the comment block at [data/MM_xor_bivector.xml:3-14](../../data/MM_xor_bivector.xml) to add the PerceptualSpace line:

```xml
  <!-- MM_xor_bivector: PerceptualSpace + ConceptualSpace + SymbolicSpace
       bivector regime on the XOR dataset. Exercises spec §B3 (Perceptual)
       alongside Stage 4-6 of the 2026-05-08 conceptual-loopback design:
         - PerceptualSpace.bivectorOutput=true: forward applies Q2
           promotion (max(0, x), max(0, -x)) per slot; activation
           shape becomes [B, N, 2].
         - ConceptualSpace.bivectorOutput=true: forward returns
           [B, V_C, 2]; reverse lifts via cached SVD pseudo-inverse.
         - SymbolicSpace.bivectorOutput=true: same shape on its
           output, mirror of the C-tier.
         - svdOrthogonalInit=true on both codebooks so the lift is
           well-conditioned from t=0.
         - useButterflies=false: butterfly schedule is incompatible
           with the [P||S] widening (Note A in the spec).
         - conceptualOrder=1: single C-S step, no per-stage halving.
         - subsymbolicEnabled marker (documentation-only; loopback is
           unconditional under bivector regime). -->
```

---

## Task 5: Add unit tests for the Q2 promote / lower helpers

**Files:**
- Create: `test/test_perceptual_bivector.py`

- [ ] **Step 5.1: Write the failing test file**

```python
"""PerceptualSpace bivector activation -- spec B3 acceptance.

Tests the Q2 promote / lower helpers and the round-trip property:
forward then reverse on a bitonic input recovers a signed-collapse
of the original (the per-feature pattern is lost by design -- see
PerceptualSpace._q2_lower_activation docstring).
"""
import sys
import unittest
from pathlib import Path

import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT / "bin"))


class TestQ2Promotion(unittest.TestCase):
    """Q2: scalar bitonic -> bivector monotonic split."""

    def setUp(self):
        from Config import TheXMLConfig
        TheXMLConfig.load(str(_PROJECT / "data/MM_xor_bivector.xml"))
        from Models import BasicModel
        self.model = BasicModel()
        self.percep = self.model.perceptualSpace

    def test_bivector_output_flag_is_true(self):
        self.assertTrue(self.percep._bivector_output)

    def test_q2_promote_positive_event_yields_aP_only(self):
        # Per-slot positive event sums to +D_P; aP = +D_P, aN = 0.
        event = torch.ones(2, 8, 2)            # [B=2, N=8, D_P=2]
        bivec = self.percep._q2_promote_activation(event)
        self.assertEqual(bivec.shape, (2, 8, 2))
        torch.testing.assert_close(bivec[..., 0],
                                   2.0 * torch.ones(2, 8))
        torch.testing.assert_close(bivec[..., 1],
                                   torch.zeros(2, 8))

    def test_q2_promote_negative_event_yields_aN_only(self):
        event = -1.0 * torch.ones(2, 8, 2)
        bivec = self.percep._q2_promote_activation(event)
        torch.testing.assert_close(bivec[..., 0],
                                   torch.zeros(2, 8))
        torch.testing.assert_close(bivec[..., 1],
                                   2.0 * torch.ones(2, 8))

    def test_q2_promote_zero_event_yields_zero_bivec(self):
        # NEITHER corner of the tetralemma: aP = aN = 0.
        event = torch.zeros(2, 8, 2)
        bivec = self.percep._q2_promote_activation(event)
        torch.testing.assert_close(bivec, torch.zeros(2, 8, 2))

    def test_q2_promote_mixed_signs_per_slot(self):
        # Slot 0: positive, slot 1: negative. No "BOTH" emitted from a
        # single signed-sum scalar -- BOTH only arises when the signed
        # scalar is overwritten upstream with both poles set.
        event = torch.zeros(1, 2, 2)
        event[0, 0, :] = 0.5      # signed sum = 1.0 -> [1, 0]
        event[0, 1, :] = -0.25    # signed sum = -0.5 -> [0, 0.5]
        bivec = self.percep._q2_promote_activation(event)
        torch.testing.assert_close(bivec[0, 0],
                                   torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(bivec[0, 1],
                                   torch.tensor([0.0, 0.5]))


class TestQ2Lowering(unittest.TestCase):
    """Inverse Q2: bivector -> per-slot scalar broadcast across D_P."""

    def setUp(self):
        from Config import TheXMLConfig
        TheXMLConfig.load(str(_PROJECT / "data/MM_xor_bivector.xml"))
        from Models import BasicModel
        self.model = BasicModel()
        self.percep = self.model.perceptualSpace

    def test_lower_recovers_signed_scalar_broadcast(self):
        # Bivector [aP=0.7, aN=0.2] -> scalar 0.5 -> broadcast to D_P=4.
        bivec = torch.tensor([[[0.7, 0.2]]])      # [B=1, N=1, 2]
        event = self.percep._q2_lower_activation(bivec, content_dim=4)
        self.assertEqual(event.shape, (1, 1, 4))
        torch.testing.assert_close(event,
                                   0.5 * torch.ones(1, 1, 4))

    def test_lower_then_promote_collapses_per_feature_detail(self):
        # The promote-lower round-trip is lossy: per-feature variation
        # within a slot is replaced by the broadcast scalar. Verify the
        # scalar value survives even though the pattern doesn't.
        event = torch.tensor([[[0.3, -0.1, 0.5, 0.0]]])    # signed sum = 0.7
        bivec = self.percep._q2_promote_activation(event)
        recovered = self.percep._q2_lower_activation(bivec, content_dim=4)
        # Recovered is uniform 0.7 across all features (vs. original
        # [0.3, -0.1, 0.5, 0.0]). Confirm the per-slot scalar matches.
        torch.testing.assert_close(recovered.sum(dim=-1) / 4,
                                   torch.tensor([[0.175]]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 5.2: Run the unit tests, expect PASS**

```bash
cd basicmodel && .venv/bin/python -m pytest test/test_perceptual_bivector.py -x -v
```

Expected: 6 tests PASS. If any fail with `AttributeError: '_bivector_output'` or `'_q2_promote_activation'`, Tasks 1–3 were not completed. If they fail with shape mismatches in `BasicModel()` construction, Task 4 (XML dim alignment) was not completed.

---

## Task 6: Add an end-to-end round-trip test for the full pipeline

**Files:**
- Modify: `test/test_perceptual_bivector.py` (add a new test class)

- [ ] **Step 6.1: Append round-trip integration test**

Append to `test/test_perceptual_bivector.py`:

```python
class TestPipelineRoundTrip(unittest.TestCase):
    """MM_xor_bivector end-to-end: input -> P -> C -> S -> reverse path
    runs without shape errors, and PerceptualSpace's activation tensor
    has the expected ``[B, N, 2]`` bivector shape."""

    def setUp(self):
        from Config import TheXMLConfig
        TheXMLConfig.load(str(_PROJECT / "data/MM_xor_bivector.xml"))
        from Models import BasicModel
        self.model = BasicModel()

    def test_perceptual_activation_is_bivector_after_forward(self):
        from Models import TheData
        TheData.load_xor()        # whatever the existing XOR loader is
        batch = TheData.get_batch(batch_size=4)
        # Run one forward pass through the full pipeline.
        out = self.model.forward(batch)
        # Inspect the percept activation tensor shape.
        act = self.model.perceptualSpace.subspace.activation.getW()
        self.assertIsNotNone(act, "PerceptualSpace.activation is empty")
        self.assertEqual(act.dim(), 3,
                         f"Expected [B, N, 2], got shape {tuple(act.shape)}")
        self.assertEqual(act.shape[-1], 2,
                         f"Expected last dim=2 (bivector), got {act.shape[-1]}")

    def test_bivector_components_are_non_negative(self):
        from Models import TheData
        TheData.load_xor()
        batch = TheData.get_batch(batch_size=4)
        self.model.forward(batch)
        act = self.model.perceptualSpace.subspace.activation.getW()
        self.assertTrue((act >= 0).all(),
                        f"Bivector components must be in [0, 1]^2; "
                        f"min observed = {act.min().item()}")

    def test_loopback_widened_input_to_conceptual(self):
        # ConceptualSpace's PiLayer input width should be P.nOutputDim
        # (2) + S.nOutputDim (2) = 4 under the loopback widening.
        cs = self.model.conceptualSpace
        self.assertEqual(cs._right_half_dim, 2,
                         f"Expected loopback widen=2 (S.nOutputDim), "
                         f"got {cs._right_half_dim}")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6.2: Adapt to actual data loader API if needed**

If `TheData.load_xor()` / `get_batch()` does not match the actual API in `bin/Models.py` / `bin/Data.py`, replace with the existing pattern used by `test/test_xor_grammar.py` or `test/test_mm_xor.py`. Read those test files first if the names above don't resolve.

```bash
cd basicmodel && head -60 test/test_mm_xor.py
```

- [ ] **Step 6.3: Run the round-trip tests**

```bash
cd basicmodel && .venv/bin/python -m pytest test/test_perceptual_bivector.py::TestPipelineRoundTrip -x -v
```

Expected: 3 tests PASS. Failures here typically mean either (a) the loopback widening saw a shape mismatch, or (b) the activation write in Task 2 wasn't reaching the final `vspace.activation`.

---

## Task 7: Smoke-run the full pipeline against `MM_xor_bivector.xml`

**Files:**
- None modified

- [ ] **Step 7.1: Run the existing MM_xor_bivector smoke test**

```bash
cd basicmodel && .venv/bin/python -m pytest test/test_mm_xor.py -x -v -k bivector
```

Expected: PASS. If a shape-mismatch error fires inside `ConceptualSpace.forward` or `_build_combined_input`, it means the loopback right-half is being computed against an N or D that no longer matches the perceptual event after Task 4's dim changes — re-check P.nOutput == S.nOutput == 8 and that ConceptualSpace.nInputDim was actually changed to 2 in the XML.

- [ ] **Step 7.2: Run the full bivector test suite to confirm no regressions**

```bash
cd basicmodel && .venv/bin/python -m pytest test/test_csbp.py test/test_conceptual_bivector.py test/test_bivector_basis.py test/test_perceptual_bivector.py -x -v
```

Expected: all PASS. The C-tier and S-tier bivector tests were green before this change; they should still be green because none of them depend on PerceptualSpace's activation shape (they construct ConceptualSpace / SymbolicSpace standalone).

- [ ] **Step 7.3: Manual check of the activation flow**

```bash
cd basicmodel && .venv/bin/python -c "
import sys; sys.path.insert(0, 'bin')
from Config import TheXMLConfig
TheXMLConfig.load('data/MM_xor_bivector.xml')
from Models import BasicModel
m = BasicModel()
print('P._bivector_output =', m.perceptualSpace._bivector_output)
print('P.subspace.nDim    =', m.perceptualSpace.subspace.nDim)
print('C.nInputDim        =', m.conceptualSpace.inputShape[1])
print('C._right_half_dim  =', m.conceptualSpace._right_half_dim)
print('Effective C input  =', m.conceptualSpace.inputShape[1] + m.conceptualSpace._right_half_dim)
"
```

Expected output:
```
P._bivector_output = True
P.subspace.nDim    = 2
C.nInputDim        = 2
C._right_half_dim  = 2
Effective C input  = 4
```

---

## Open Questions / Future Work

These items are explicitly out of scope for this plan but worth flagging:

1. **Per-element vs per-slot Q2.** The spec is ambiguous about whether the Q2 split applies per content feature (`[B, N, D_P] → [B, N, D_P, 2]`) or per slot (`[B, N, D_P] → [B, N, 2]`). This plan implements per-slot via signed-sum reduction, matching the B-summary table at [spec line 1212](../specs/2026-04-24-lift-lower-bivector-design.md) which says PerceptualSpace activation shape is `[B, N, 2 + nWhere + nWhen]`. If the per-element interpretation turns out to be required by some downstream consumer, swap `_q2_promote_activation` to skip the signed-sum reduction.

2. **Codebook-on-`.what` for PerceptualSpace.** The spec's B4 row says PerceptualSpace's prototype `.what` width is also 2 (mirroring SymbolicSpace, unlike ConceptualSpace's high-dim). For configs with `<codebook>true</codebook>`, this would mean replacing the lexicon Embedding with a 2-D Codebook. Out of scope for the XOR test path (which has `<codebook>false</codebook>`); revisit when a real-text bivector config is needed.

3. **Removing the `subsymbolicEnabled` XML marker.** After this lands, the marker exists in `MM_xor_bivector.xml` but has no live reader. A separate cleanup task can remove it from the XML schema entirely, matching the 2026-05-08 design Stage 2 acceptance criterion ("XML schema drops the flags").

4. **Reverse-path lossiness.** `_q2_lower_activation` recovers a per-slot scalar but broadcasts uniformly across `D_P`, losing the per-feature pattern of the original event. This is fundamentally the same lossiness as the C-tier bivector (which collapses `D_concept` features to 2). If the reverse path needs per-feature recovery, a higher-dim codebook on `.what` (item 2 above) would store the per-feature pattern in the prototype, recovered via the `Codebook.reverse(bivec, project=True)` SVD pseudo-inverse path.
