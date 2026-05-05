# Resolve and Luminosity Handoff

## Goals

1. Fix `SymbolicSpace.resolve()` — change `pos + neg` to `pos - neg`.
2. Replace `TruthLayer.luminosity()` with the area-overlap formula derived
   from the contiguity kernel.

---

## Background

### Why resolve() was wrong

`resolve()` collapses the [pos, neg] bivector to a scalar before the
symbolic codebook sees it.  The current implementation:

```python
# Spaces.py line ~7424, inside SymbolicSpace.resolve()
subspace.activation.setW(pos + neg)
```

`pos + neg` is the **unsigned** sum — it treats affirmation and negation as
equal contributors and discards polarity.  The correct value is the signed
**Degree of Truth**:

```
DoT = pos - neg  ∈ [-1, 1]
```

- pos=1, neg=0  →  DoT = +1  (certain affirmation)
- pos=0, neg=1  →  DoT = -1  (certain negation)
- pos = neg     →  DoT =  0  (balanced / unknown)

This DoT is exactly what the codebook should look up against and what the
TruthSet stores as the scalar truth value of each symbol.

### Design: symbols are resolved, excluded middle holds

Symbols are single-pole committed assertions.  The [pos, neg] bivector
exists as an intermediate representation inside SymbolicSpace during
grammatical composition; `resolve()` converts it to a plain signed scalar
before the codebook snap.  Every stored symbol is either affirming (DoT > 0)
or negating (DoT < 0) — contradiction is an inter-symbol relationship
computed by `luminosity()`, not an intra-symbol encoding.

### Why the luminosity formula changed

The existing `TruthLayer.luminosity()` (`Layers.py:4837`) computes:

```python
relu(min(positive_poles(truths))).norm()
```

This measures the brightness of the positive-pole conjunction but does not
account for overlapping regions with conflicting truth values.  The new
formula is:

```
luminosity = area - overlapArea × |t_A - t_B|
```

where:
- `area` ∈ [0, 1] — total normalised area of the truth field (sum of
  individual Gaussian region areas, each proportional to σ²).
- `overlapArea` ∈ [0, area] — kernel overlap between two propositions
  (Gaussian kernel, same as contiguity checks).
- `|t_A - t_B|` ∈ [0, 2] — difference in DoT between the two propositions.

**Range proof:**  overlapArea ≤ area by the containment constraint, so
`overlapArea × |t_A - t_B| ≤ area × 2`, giving
`luminosity ∈ [area − 2·area, area − 0] = [−area, area] ⊆ [−1, 1]`.

The formula is maximised (+1) when the truth field is fully committed with
no contradicting overlaps, and minimised (−1) only in the degenerate case
where two fully-coincident regions assert opposite truth values over the
entire space.

---

## Implementation Steps

### Step 1 — Fix resolve()

**File:** `bin/Spaces.py`
**Location:** `SymbolicSpace.resolve()`, the line that calls `setW`.

Search for the string `pos + neg` inside `resolve()` — it should be unique
and near line 7424 in the current file.

```python
# Before
subspace.activation.setW(pos + neg)

# After
subspace.activation.setW(pos - neg)
```

No other changes to `resolve()` are needed.  The call chain is already
correct: `resolve()` → `activation.setW(...)` → `activation.getW()` →
`_nearest_symbol_target()` → `activation.setW(snapped)`.

### Step 2 — Restore or inline kernel_overlap

`Basis.kernel_overlap` was removed on 2026-05-02 (see comment at
`Spaces.py:844–848`).  The reference implementation is in
`bin/etc/old.py:532` (`OldLogicLayer.kernel_overlap`).

Options (pick the simpler one):

**A.** Add a module-level helper in `Layers.py` near `TruthLayer`:

```python
def _gaussian_kernel_overlap(X, Y, sigma_x, sigma_y, eps=1e-8):
    """Gaussian kernel overlap: exp(-d² / 2(σx² + σy²)).

    Args:
        X: [N, D]  Y: [M, D]
        sigma_x: scalar or [N]   sigma_y: scalar or [M]
    Returns:
        [N, M] overlap matrix, values in (0, 1].
    """
    d2 = torch.cdist(X.unsqueeze(0), Y.unsqueeze(0)).squeeze(0) ** 2  # [N, M]
    sx = sigma_x if torch.is_tensor(sigma_x) else torch.full((X.shape[0],), sigma_x, device=X.device)
    sy = sigma_y if torch.is_tensor(sigma_y) else torch.full((Y.shape[0],), sigma_y, device=Y.device)
    denom = 2.0 * (sx.unsqueeze(1) ** 2 + sy.unsqueeze(0) ** 2) + eps
    return torch.exp(-d2 / denom)
```

**B.** Import from `etc.old` (not recommended — that module is a legacy dump).

### Step 3 — Add area() helper on TruthLayer

The area of a Gaussian region is proportional to σ_D (the D-dimensional
Gaussian integral ∝ σ^D, but for a unit-normalised 1-D scalar σ the area
is just σ² normalised to [0, 1]).

`TruthLayer` has `activeSigma` as a declared but unpopulated slot (see
`todo.md` Reasoning System section and `Spaces.py:521`).  Use a fallback
default σ when `activeSigma` is None:

```python
_DEFAULT_SIGMA = 0.1   # tunable; controls how spread each truth region is
```

Add a private helper:

```python
def _region_area(self, sigma=None) -> float:
    """Normalised area of a single Gaussian truth region.

    Returns sigma^2, clamped to [0, 1].  If sigma is None, uses
    activeSigma if populated, otherwise _DEFAULT_SIGMA.
    """
    if sigma is None:
        sigma = (self.activeSigma if self.activeSigma is not None
                 else _DEFAULT_SIGMA)
    if torch.is_tensor(sigma):
        sigma = float(sigma.mean().item())
    return min(float(sigma) ** 2, 1.0)
```

### Step 4 — Replace TruthLayer.luminosity()

**File:** `bin/Layers.py`
**Location:** `TruthLayer.luminosity()` starting at line 4837.

Keep the existing method signature (`pi_layer=None`) for backward
compatibility.  Replace the body:

```python
def luminosity(self, pi_layer=None, sigma=None) -> torch.Tensor:
    """Luminosity of the truth field.

    luminosity = area - overlapArea * |t_A - t_B|

    where:
      area       = total normalised Gaussian area of all stored truths.
      overlapArea = pairwise kernel overlap between each pair of truths,
                   summed and normalised by the number of pairs.
      |t_A - t_B| = absolute difference in DoT between each pair.

    Range: [-1, 1].  High = consistent committed field.
           Low = contradicting overlapping assertions.
    """
    n = self.count.item()
    if n == 0:
        return torch.tensor(0.0, device=self.truths.device)

    stored = self.truths[:n]                           # [n, D]
    if pi_layer is not None:
        stored = pi_layer.reverse(stored)

    # DoT for each stored truth: first component is the resolved scalar.
    # If truths are stored as plain scalars (1-D), use directly.
    # If stored as [pos, neg] bivectors (last dim == 2), compute pos-neg.
    if stored.shape[-1] == 2:
        dot = stored[..., 0] - stored[..., 1]          # [n]
    elif stored.shape[-1] == 1:
        dot = stored[..., 0]
    else:
        # Multi-dim: use the resolved activation norm with sign from mean.
        dot = stored.mean(dim=-1)                      # [n] fallback

    region_area = self._region_area(sigma)
    total_area = min(n * region_area, 1.0)

    if n == 1:
        return torch.tensor(float(total_area), device=stored.device)

    # Pairwise overlap and disagreement.
    # For large TruthSets this is O(n²); acceptable for max_truths=1024.
    X = stored.unsqueeze(1)                            # [n, 1, D]
    Y = stored.unsqueeze(0)                            # [1, n, D]
    d2 = ((X - Y) ** 2).sum(-1)                       # [n, n]
    sig = region_area ** 0.5                           # σ from area = σ²
    denom = 2.0 * 2.0 * sig ** 2 + 1e-8
    overlap = torch.exp(-d2 / denom)                   # [n, n]  ∈ (0,1]

    # |t_A - t_B| for each pair
    ta = dot.unsqueeze(1)                              # [n, 1]
    tb = dot.unsqueeze(0)                              # [1, n]
    disagreement = (ta - tb).abs()                     # [n, n] ∈ [0, 2]

    # Sum over upper triangle (exclude self-overlap on diagonal)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool,
                                  device=stored.device), diagonal=1)
    pairs = mask.sum().item()
    if pairs == 0:
        return torch.tensor(float(total_area), device=stored.device)

    overlap_area_sum = (overlap * mask.float()).sum() / pairs   # ∈ [0, 1]
    disagreement_mean = (disagreement * mask.float()).sum() / pairs  # ∈ [0, 2]

    penalty = overlap_area_sum * disagreement_mean
    lum = total_area - penalty.item()
    return torch.tensor(float(lum), device=stored.device)
```

The old `darkness()` method can be left in place — it remains a useful
diagnostic even if `luminosity()` no longer uses the positive-pole split.

---

## Tests to Add

Add to `test/` (new file `test_resolve_luminosity.py` or append to an
existing truth/symbolic test):

```python
def test_resolve_pos_minus_neg():
    """resolve() should write pos - neg, not pos + neg."""
    # Set up a minimal SymbolicSpace (or mock subspace) with a known bivector.
    # pos=0.8, neg=0.3  →  expected DoT = 0.5 (not 1.1)
    ...

def test_resolve_negation():
    """Negation: pos=0.0, neg=1.0  →  DoT = -1.0."""
    ...

def test_luminosity_no_overlap():
    """Two truths in different regions: penalty = 0, luminosity = area."""
    ...

def test_luminosity_full_overlap_max_disagreement():
    """Two fully-overlapping truths with DoT=+1 and DoT=-1:
    luminosity should be at its minimum (near -1 for area=1)."""
    ...

def test_luminosity_consistent_field():
    """All truths agree: penalty ≈ 0, luminosity ≈ area."""
    ...

def test_luminosity_range():
    """luminosity() always returns a value in [-1, 1]."""
    ...
```

---

## Acceptance Criteria

- `resolve()` passes `pos - neg` to `setW`, not `pos + neg`.
- A bivector with pos=1, neg=0 resolves to activation +1.0.
- A bivector with pos=0, neg=1 resolves to activation -1.0.
- `TruthLayer.luminosity()` returns values in [-1, 1] for any valid TruthSet.
- Existing tests pass (no regressions in symbolic forward / truth grounding).
