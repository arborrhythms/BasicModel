# Signal Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Git policy for this project:** the user (Alec) manages all commits personally. Treat every "Commit" step as **"pause, summarize the change, and wait for user direction"** — never run `git commit`, `git add`, `git push`, or any branch-mutating command without explicit per-step approval. Read-only `git status` / `git diff` is fine.
>
> **Worktree policy:** work directly in the main checkout. Do **not** create worktrees, branches, or stashes.

**Goal:** Replace `Chart`'s soft-superposition CKY parse-forest with a GPU-friendly signal-routing parser. Keep the legacy chart alongside, conditionalized via `WordSpace.routerKind` XML config.

**Architecture:** Per-layer slab `[B, N, D]` stays as the dataflow. Local MLPs emit per-position COPY/REDUCE action scores over all rules in a tier (`[B, N, R_copy]`, `[B, N-1, R_reduce]`). A structured DP picks the legal tiling — soft marginals during training, hard Viterbi at inference. A four-branch **trainable comparator-mixer** (`{keep, reduce-here, shift-from-right, pad}`) produces a length-N soft slab for the inverse / generate pass; a separate hard-compacted slab feeds the next layer's binary op so operand quality stays clean. The `chart_vec`/`chart_score` `[B, N+1, N+1, K, D]` allocation goes away entirely on the signal path.

**Tech Stack:** PyTorch 2.x, Python 3.12, ROCm-on-AMD-Strix-Halo runtime, existing `basicmodel/bin/util.py` `XMLConfig`, existing `pytest` test layout under `basicmodel/test/`.

**Reference design:** the conversation-thread design note (signal-based structured routing parser with comparator-mixer extension), and `basicmodel/doc/specs/2026-05-01-syntactic-layer-refactor.md` for the chart surface this replaces.

---

## File Structure

**Create:**

| Path | Responsibility |
|---|---|
| `basicmodel/bin/SignalRouter.py` | Scorers, structured DP, hard Viterbi, comparator-mixer, hard + soft compaction, `BinaryStructuredReductionLayer`, `UnaryStructuredLayer`. Self-contained module. |
| `basicmodel/test/test_signal_router_brute_force.py` | Brute-force enumerator over all legal COPY/REDUCE tilings for N ≤ 8, used as ground truth for DP/Viterbi tests. |
| `basicmodel/test/test_signal_router_dp.py` | Soft DP: shape, partition score, marginals match brute force. |
| `basicmodel/test/test_signal_router_viterbi.py` | Hard Viterbi: legality (no overlapping reduces, full coverage), best-score match brute force, multi-op dispatch. |
| `basicmodel/test/test_signal_router_scorer.py` | Local scorers: shapes, gradient reach, multi-op output dim. |
| `basicmodel/test/test_signal_router_comparator.py` | Four-branch comparator-mixer: gate softmax, branch coverage, gradient on each branch's gate. |
| `basicmodel/test/test_signal_router_compaction.py` | `compact_hard` provenance metadata, `compact_soft` length-N alignment, `span_start`/`span_end` propagation. |
| `basicmodel/test/test_signal_router_layer.py` | `BinaryStructuredReductionLayer` end-to-end: shape, gradient flow into op + scorer + comparator. |
| `basicmodel/test/test_signal_router_wordspace.py` | WordSpace dispatch: `routerKind="signal"` produces a `current_rules` dict with the same per-tier shape as `routerKind="chart"`. |

**Modify:**

| Path | Why |
|---|---|
| `basicmodel/bin/Language.py` | `Chart.compose` and `Chart.generate` dispatch on `routerKind` flag; legacy CKY path untouched on `routerKind="chart"`. Construct a `SignalRouter` subordinate when flag is `"signal"`. |
| Project XML config (whichever loads `WordSpace.chartTau` etc., per `TheXMLConfig.get("WordSpace.chartTau", ...)`) | Add `WordSpace.routerKind` (default `"chart"`) and `WordSpace.signal.temperature` (default `1.0`). Tasks reference reads via `TheXMLConfig.get(...)`. |

**Out of scope for this plan** (open questions documented at the end):

- Replacing `host_layer`/`GRAMMAR_LAYER_CLASSES` registry with a fully tensorized rule bank (the rule-axis tensor). The first cut **routes over op_id but still calls into existing per-rule modules through the registry**; the bulk-tensorization is a follow-up.
- Production-quality compaction kernels (the plan ships Python-loop compaction for correctness; profiling-driven kernel work is a follow-up).
- TruthLayer / SymbolicSpace integration changes — the signal router is wired strictly inside `WordSpace`'s parse path.

---

## Task 0: Config flag + skeleton dispatch

Establish the migration scaffold: the new code path exists, is selected via XML, and falls through to a clear `NotImplementedError` until later tasks fill it in. All existing tests must stay green on the default (`"chart"`) path.

**Files:**
- Create: `basicmodel/bin/SignalRouter.py`
- Modify: `basicmodel/bin/Language.py` (Chart.__init__ around line 815, Chart.compose around line 949, Chart.generate around line 984)
- Test: `basicmodel/test/test_signal_router_wordspace.py`

- [ ] **Step 0.1: Write the failing dispatch test**

```python
# basicmodel/test/test_signal_router_wordspace.py
"""Dispatch test: routerKind selects between chart and signal paths."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import pytest
import torch
import Language
from Language import Chart


def test_chart_default_routerkind_is_chart():
    chart = Chart(nInput=4, nOutput=4, max_depth=3, hidden_dim=16, feature_dim=4)
    assert chart.router_kind == "chart"


def test_chart_routerkind_signal_dispatches_to_signal_router():
    chart = Chart(nInput=4, nOutput=4, max_depth=3, hidden_dim=16, feature_dim=4,
                  router_kind="signal")
    assert chart.router_kind == "signal"
    # Skeleton: signal path raises NotImplementedError until later tasks land.
    with pytest.raises(NotImplementedError):
        chart.compose(torch.randn(1, 4, 4), word_space=None)
```

- [ ] **Step 0.2: Run the test to verify it fails**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_wordspace.py -v`
Expected: both tests FAIL — `Chart` does not yet accept `router_kind` and has no attribute `router_kind`.

- [ ] **Step 0.3: Create the SignalRouter module skeleton**

```python
# basicmodel/bin/SignalRouter.py
"""Signal-based structured routing parser.

Replaces the Chart's soft-superposition CKY forest with per-layer
COPY/REDUCE routing on the subspace tensor. See
basicmodel/doc/plans/2026-05-02-signal-router.md.

Selected via WordSpace.routerKind = "signal" in XML.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalRouter(nn.Module):
    """Top-level signal-routing parser. Owned by Chart when
    router_kind == "signal". Parallels Chart.compose / Chart.generate.

    Filled in by tasks 1-13 of the implementation plan.
    """

    def __init__(self, n_input, n_output, *, hidden_dim, feature_dim,
                 max_depth, temperature=1.0):
        super().__init__()
        self.n_input = int(n_input)
        self.n_output = int(n_output)
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        self.max_depth = int(max_depth)
        self.temperature = float(temperature)

    def compose(self, data, word_space, subspace=None):
        raise NotImplementedError(
            "SignalRouter.compose: pending plan task 9 wiring."
        )

    def generate(self, target, word_space, subspace=None):
        raise NotImplementedError(
            "SignalRouter.generate: pending plan task 12 wiring."
        )
```

- [ ] **Step 0.4: Add router_kind to Chart.__init__ and dispatch in compose/generate**

In `basicmodel/bin/Language.py`, modify `Chart.__init__` (around line 795) to accept and store `router_kind`, reading XML default. Add the dispatch branches in `Chart.compose` (around line 949) and `Chart.generate` (around line 984).

```python
# basicmodel/bin/Language.py — Chart.__init__ signature change
def __init__(self, nInput, nOutput=None, *, max_depth=12,
             hidden_dim=256, D_rule=32, chart_tau=None, w_max=8,
             unary_max_depth=2, feature_dim=None,
             router_kind=None):
    super().__init__()
    # ... existing body unchanged through chart_tau resolution ...

    # Router selection. XML-driven; falls back to "chart" for legacy.
    if router_kind is None:
        try:
            router_kind = str(TheXMLConfig.get(
                "WordSpace.routerKind", "chart"))
        except Exception:
            router_kind = "chart"
    if router_kind not in ("chart", "signal"):
        raise ValueError(
            f"WordSpace.routerKind must be 'chart' or 'signal', "
            f"got {router_kind!r}.")
    self.router_kind = router_kind

    # Lazy SignalRouter construction; only built when needed.
    self._signal_router = None

    # ... rest of existing __init__ unchanged ...
```

```python
# basicmodel/bin/Language.py — add a helper just below __init__
def _ensure_signal_router(self):
    if self._signal_router is None:
        from SignalRouter import SignalRouter
        try:
            temperature = float(TheXMLConfig.get(
                "WordSpace.signal.temperature", 1.0))
        except Exception:
            temperature = 1.0
        # Assigning an nn.Module to an attribute auto-registers it as a
        # submodule, so it is included in parameters() / state_dict().
        self._signal_router = SignalRouter(
            n_input=self.nInput,
            n_output=self.nOutput,
            hidden_dim=self.hidden_dim,
            feature_dim=self._pair_feature_dim,
            max_depth=self.max_depth,
            temperature=temperature,
        )
    return self._signal_router
```

```python
# basicmodel/bin/Language.py — Chart.compose dispatch (around line 968)
object.__setattr__(self, '_active_word_space', word_space)
try:
    if self.router_kind == "signal":
        router = self._ensure_signal_router()
        rules = router.compose(data, word_space, subspace=subspace)
        if word_space is not None:
            word_space.current_rules = rules
        return rules
    if self.training:
        composed, _svo = self._compose_chart_cky(
            data, word_space, subspace)
    else:
        composed, _svo = self._compose_chart_cky_viterbi(
            data, word_space, subspace)
    rules = self._collect_rule_selections(word_space)
    word_space.current_rules = rules
    self.last_composed = composed
    return rules
finally:
    object.__setattr__(self, '_active_word_space', None)
```

Apply the analogous `if self.router_kind == "signal"` branch to `Chart.generate` (around line 996), calling `router.generate(target, word_space, subspace=subspace)`.

- [ ] **Step 0.5: Run the new dispatch test**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_wordspace.py -v`
Expected: both tests PASS.

- [ ] **Step 0.6: Run the full test suite to confirm no regression on the chart path**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/ -x -q`
Expected: same pass/fail set as before this task — chart path is the default and is untouched. Any new failures must trace to Task 0 changes; fix before proceeding.

- [ ] **Step 0.7: Pause for user review and commit**

Summarize: SignalRouter module skeleton + Chart `router_kind` dispatch flag, default `"chart"`, signal path raises NotImplementedError. Wait for user direction on whether to commit or continue without.

---

## Task 1: Brute-force tiling enumerator (test fixture)

The DP and Viterbi tests in tasks 2-3 need an independent ground-truth to compare against. A brute-force enumerator over all legal `{COPY × R_copy, REDUCE × R_reduce}` tilings for small `N` is the cleanest reference.

**Files:**
- Create: `basicmodel/test/test_signal_router_brute_force.py`

- [ ] **Step 1.1: Write the brute-force enumerator with self-tests**

```python
# basicmodel/test/test_signal_router_brute_force.py
"""Brute-force enumerator over legal COPY/REDUCE tilings.

Used as ground truth for the structured-DP and Viterbi tests.
N <= 8 is the practical ceiling; runtime is exponential.
"""
from __future__ import annotations

import math
from typing import Iterator, List, Tuple

import pytest


# A tile is (kind, op_id) where kind in {"copy", "reduce"}.
Tile = Tuple[str, int]


def enumerate_tilings(
    n: int, r_copy: int, r_reduce: int
) -> Iterator[List[Tile]]:
    """Yield every legal tiling of length-n positions.

    A legal tiling covers positions 0..n-1 with non-overlapping tiles:
      - copy tile (length 1) at position t with op c in [0, r_copy)
      - reduce tile (length 2) at positions t, t+1 with op r in [0, r_reduce)
    """
    if n == 0:
        yield []
        return
    # Copy at position 0.
    for c in range(r_copy):
        for tail in enumerate_tilings(n - 1, r_copy, r_reduce):
            yield [("copy", c)] + tail
    # Reduce at positions 0,1.
    if n >= 2:
        for r in range(r_reduce):
            for tail in enumerate_tilings(n - 2, r_copy, r_reduce):
                yield [("reduce", r)] + tail


def score_tiling(
    tiling: List[Tile],
    copy_score,    # [N, R_copy] tensor or 2D list
    reduce_score,  # [N-1, R_reduce] tensor or 2D list
) -> float:
    """Sum the per-tile scalar scores along this tiling. Single-batch."""
    total = 0.0
    pos = 0
    for kind, op in tiling:
        if kind == "copy":
            total += float(copy_score[pos][op])
            pos += 1
        else:  # reduce
            total += float(reduce_score[pos][op])
            pos += 2
    return total


def best_tiling(
    copy_score, reduce_score, n: int, r_copy: int, r_reduce: int
) -> Tuple[List[Tile], float]:
    """Argmax legal tiling and its score."""
    best, best_score = None, -math.inf
    for t in enumerate_tilings(n, r_copy, r_reduce):
        s = score_tiling(t, copy_score, reduce_score)
        if s > best_score:
            best, best_score = t, s
    return best, best_score


def logsumexp_tilings(
    copy_score, reduce_score, n: int, r_copy: int, r_reduce: int
) -> float:
    """Brute-force partition function (log-sum-exp over all tilings)."""
    if n == 0:
        return 0.0
    scores = [
        score_tiling(t, copy_score, reduce_score)
        for t in enumerate_tilings(n, r_copy, r_reduce)
    ]
    m = max(scores)
    return m + math.log(sum(math.exp(s - m) for s in scores))


# ---- self-tests on the enumerator itself ----------------------------

def _count(n, rc, rr):
    return sum(1 for _ in enumerate_tilings(n, rc, rr))


def test_enumerate_count_n_zero():
    assert _count(0, 1, 1) == 1  # the empty tiling


def test_enumerate_count_n_one_no_reduce_possible():
    assert _count(1, 1, 1) == 1
    assert _count(1, 3, 5) == 3  # only copy, R_copy choices


def test_enumerate_count_n_two():
    # N=2: copy-copy (rc^2) + reduce (rr) = rc^2 + rr
    assert _count(2, 2, 3) == 2 * 2 + 3


def test_enumerate_count_n_three():
    # N=3: ccc (rc^3), Rc (rr*rc), cR (rc*rr)
    assert _count(3, 2, 3) == 2**3 + 3 * 2 + 2 * 3


def test_enumerate_no_overlapping_reduces():
    # No tiling at any N has two reduces touching the same position.
    for n in range(1, 6):
        for tiling in enumerate_tilings(n, 1, 1):
            covered = []
            pos = 0
            for kind, _ in tiling:
                if kind == "copy":
                    covered.append((pos, pos))
                    pos += 1
                else:
                    covered.append((pos, pos + 1))
                    pos += 2
            # flat coverage equals 0..n-1, no repeats
            flat = sorted(p for a, b in covered for p in range(a, b + 1))
            assert flat == list(range(n))
```

- [ ] **Step 1.2: Run the enumerator self-tests**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_brute_force.py -v`
Expected: all five tests PASS. (No SignalRouter code is exercised here.)

- [ ] **Step 1.3: Pause for user review and commit**

Summarize: brute-force tiling enumerator + helpers (`enumerate_tilings`, `score_tiling`, `best_tiling`, `logsumexp_tilings`) plus self-tests, no production code touched. Wait for user direction.

---

## Task 2: Multi-op soft DP

Generalize the binary copy/reduce soft DP to a vector of ops per action: `copy_score: [B, N, R_copy]`, `reduce_score: [B, N-1, R_reduce]`. Returns `logZ`, per-position copy marginals (summed over op axis), per-pair reduce marginals, and per-(position, op) marginals for the comparator regularizer.

**Files:**
- Modify: `basicmodel/bin/SignalRouter.py`
- Test: `basicmodel/test/test_signal_router_dp.py`

- [ ] **Step 2.1: Write failing DP tests**

```python
# basicmodel/test/test_signal_router_dp.py
"""Soft DP correctness against brute-force enumeration."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import math

import pytest
import torch

from SignalRouter import binary_tiling_soft_dp
from test_signal_router_brute_force import (
    enumerate_tilings, logsumexp_tilings, score_tiling,
)


def _rand_scores(B, N, rc, rr, seed=0):
    g = torch.Generator().manual_seed(seed)
    cs = torch.randn(B, N, rc, generator=g)
    rs = torch.randn(B, N - 1, rr, generator=g) if N >= 1 else \
        torch.empty(B, 0, rr)
    return cs, rs


def test_soft_dp_logZ_matches_brute_force_small_N():
    for N in (1, 2, 3, 4, 5):
        for rc, rr in ((1, 1), (2, 1), (1, 2), (3, 2)):
            cs, rs = _rand_scores(1, N, rc, rr, seed=N * 100 + rc * 10 + rr)
            out = binary_tiling_soft_dp(cs, rs)
            logZ_dp = out["logZ"][0].item()
            logZ_bf = logsumexp_tilings(
                cs[0].tolist(),
                rs[0].tolist() if N > 1 else [],
                N, rc, rr,
            )
            assert math.isclose(logZ_dp, logZ_bf, rel_tol=1e-5, abs_tol=1e-5), \
                f"N={N} rc={rc} rr={rr} dp={logZ_dp} bf={logZ_bf}"


def test_soft_dp_marginals_sum_consistency():
    # At every source position t in [0, N-2], exactly one of:
    #   - a copy fires at t (sum over ops)
    #   - a reduce fires at t (sum over ops, length-2 starting at t)
    #   - a reduce fires at t-1 (covers t)
    # the per-position "covered" mass = 1 in expectation.
    cs, rs = _rand_scores(2, 6, 3, 2, seed=42)
    out = binary_tiling_soft_dp(cs, rs)
    p_copy = out["copy_marginal"]      # [B, N]
    p_reduce = out["reduce_marginal"]  # [B, N-1]
    B, N = p_copy.shape
    for b in range(B):
        for t in range(N):
            covered = p_copy[b, t].item()
            if t < N - 1:
                covered += p_reduce[b, t].item()
            if t > 0:
                covered += p_reduce[b, t - 1].item()
            assert math.isclose(covered, 1.0, abs_tol=1e-4), \
                f"position t={t} covered_mass={covered}"


def test_soft_dp_per_op_marginals_sum_to_action_marginal():
    cs, rs = _rand_scores(1, 5, 3, 2, seed=7)
    out = binary_tiling_soft_dp(cs, rs)
    # copy_marginal_op: [B, N, R_copy]; copy_marginal: [B, N] = sum over ops.
    assert torch.allclose(
        out["copy_marginal_op"].sum(-1), out["copy_marginal"], atol=1e-5)
    assert torch.allclose(
        out["reduce_marginal_op"].sum(-1), out["reduce_marginal"], atol=1e-5)


def test_soft_dp_gradient_reaches_scores():
    cs = torch.randn(1, 4, 2, requires_grad=True)
    rs = torch.randn(1, 3, 2, requires_grad=True)
    out = binary_tiling_soft_dp(cs, rs)
    out["logZ"].sum().backward()
    assert cs.grad is not None and (cs.grad.abs().sum() > 0)
    assert rs.grad is not None and (rs.grad.abs().sum() > 0)
```

- [ ] **Step 2.2: Run to verify failure**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_dp.py -v`
Expected: all four tests FAIL — `binary_tiling_soft_dp` is not yet implemented.

- [ ] **Step 2.3: Implement the multi-op soft DP in SignalRouter.py**

```python
# basicmodel/bin/SignalRouter.py — append below the SignalRouter class

def binary_tiling_soft_dp(
    copy_score: torch.Tensor,
    reduce_score: torch.Tensor,
    temperature: float = 1.0,
):
    """Forward-backward over legal COPY/REDUCE tilings, multi-op.

    Args:
        copy_score:   [B, N, R_copy] per-(position, op) log-scores.
        reduce_score: [B, N-1, R_reduce] per-(adjacent-pair, op) log-scores.
        temperature:  scalar; scores divided by this before DP.

    Returns dict:
        logZ:               [B] log partition function.
        alpha:              [B, N+1] forward log-messages.
        beta:               [B, N+1] backward log-messages.
        copy_marginal:      [B, N]      P(copy fires at t)
        reduce_marginal:    [B, N-1]    P(reduce fires at t,t+1)
        copy_marginal_op:   [B, N, R_copy]    P(copy with op c at t)
        reduce_marginal_op: [B, N-1, R_reduce] P(reduce with op r at t)
    """
    B, N, R_copy = copy_score.shape
    if N == 0:
        zero = torch.zeros(B, device=copy_score.device, dtype=copy_score.dtype)
        return {
            "logZ": zero,
            "alpha": zero.unsqueeze(1),
            "beta": zero.unsqueeze(1),
            "copy_marginal": copy_score.new_zeros(B, 0),
            "reduce_marginal": copy_score.new_zeros(B, 0),
            "copy_marginal_op": copy_score.new_zeros(B, 0, R_copy),
            "reduce_marginal_op": copy_score.new_zeros(B, 0, 0),
        }

    R_reduce = reduce_score.shape[-1] if reduce_score.numel() > 0 else 0

    c = copy_score / temperature                          # [B, N, R_copy]
    r = reduce_score / temperature                        # [B, N-1, R_reduce]

    # Per-action log-sum-exp over op axis = action-level log-score.
    c_action = torch.logsumexp(c, dim=-1)                 # [B, N]
    r_action = (torch.logsumexp(r, dim=-1)
                if R_reduce > 0 and N > 1
                else copy_score.new_full((B, max(N - 1, 0)), -1e9))

    NEG_INF = -1e9

    alpha = copy_score.new_full((B, N + 1), NEG_INF)
    alpha[:, 0] = 0.0
    for t in range(N):
        alpha[:, t + 1] = torch.logaddexp(
            alpha[:, t + 1], alpha[:, t] + c_action[:, t])
        if t + 1 < N:
            alpha[:, t + 2] = torch.logaddexp(
                alpha[:, t + 2], alpha[:, t] + r_action[:, t])
    logZ = alpha[:, N]

    beta = copy_score.new_full((B, N + 1), NEG_INF)
    beta[:, N] = 0.0
    for t in reversed(range(N)):
        beta[:, t] = torch.logaddexp(
            beta[:, t], c_action[:, t] + beta[:, t + 1])
        if t + 1 < N:
            beta[:, t] = torch.logaddexp(
                beta[:, t], r_action[:, t] + beta[:, t + 2])

    # Action-level marginals.
    copy_log_marginal = alpha[:, :N] + c_action + beta[:, 1:N + 1] - logZ.unsqueeze(1)
    copy_marginal = copy_log_marginal.exp()
    if N > 1:
        reduce_log_marginal = (
            alpha[:, :N - 1] + r_action + beta[:, 2:N + 1] - logZ.unsqueeze(1))
        reduce_marginal = reduce_log_marginal.exp()
    else:
        reduce_marginal = copy_score.new_zeros(B, 0)

    # Per-(action, op) marginals: P(action fires at t) * softmax(op | action).
    op_post_copy = F.softmax(c, dim=-1)                   # [B, N, R_copy]
    copy_marginal_op = copy_marginal.unsqueeze(-1) * op_post_copy
    if N > 1 and R_reduce > 0:
        op_post_reduce = F.softmax(r, dim=-1)             # [B, N-1, R_reduce]
        reduce_marginal_op = reduce_marginal.unsqueeze(-1) * op_post_reduce
    else:
        reduce_marginal_op = copy_score.new_zeros(B, max(N - 1, 0), R_reduce)

    return {
        "logZ": logZ,
        "alpha": alpha,
        "beta": beta,
        "copy_marginal": copy_marginal,
        "reduce_marginal": reduce_marginal,
        "copy_marginal_op": copy_marginal_op,
        "reduce_marginal_op": reduce_marginal_op,
    }
```

- [ ] **Step 2.4: Run the DP tests**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_dp.py -v`
Expected: all four tests PASS. If `logZ_matches_brute_force_small_N` fails for `N=1`, the empty-`reduce_score` shape handling needs adjustment — fix in the function, not the test.

- [ ] **Step 2.5: Pause for user review and commit**

Summarize: multi-op `binary_tiling_soft_dp` with action and per-op marginals, validated against brute-force ground truth for `N ∈ [1, 5]`.

---

## Task 3: Multi-op hard Viterbi

Pick the single best legal tiling. Returns `copy_mask: [B, N, R_copy]` and `reduce_mask: [B, N-1, R_reduce]` one-hot tensors so the chosen op is named, not just the action.

**Files:**
- Modify: `basicmodel/bin/SignalRouter.py`
- Test: `basicmodel/test/test_signal_router_viterbi.py`

- [ ] **Step 3.1: Write failing Viterbi tests**

```python
# basicmodel/test/test_signal_router_viterbi.py
"""Hard Viterbi correctness against brute force, plus legality."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import math

import torch

from SignalRouter import binary_tiling_viterbi
from test_signal_router_brute_force import best_tiling


def _rand_scores(B, N, rc, rr, seed=0):
    g = torch.Generator().manual_seed(seed)
    cs = torch.randn(B, N, rc, generator=g)
    rs = torch.randn(B, max(N - 1, 0), rr, generator=g)
    return cs, rs


def test_viterbi_score_matches_brute_force_small_N():
    for N in (1, 2, 3, 4, 5, 6):
        for rc, rr in ((1, 1), (2, 1), (1, 2), (2, 3)):
            cs, rs = _rand_scores(1, N, rc, rr, seed=N + rc * 17 + rr * 31)
            out = binary_tiling_viterbi(cs, rs)
            score_dp = out["score"][0].item()
            _, score_bf = best_tiling(
                cs[0].tolist(),
                rs[0].tolist() if N > 1 else [],
                N, rc, rr,
            )
            assert math.isclose(score_dp, score_bf, rel_tol=1e-5, abs_tol=1e-5)


def test_viterbi_route_is_legal_no_overlapping_reduces():
    cs, rs = _rand_scores(3, 6, 2, 2, seed=99)
    out = binary_tiling_viterbi(cs, rs)
    cm = out["copy_mask"]      # [B, N, R_copy] one-hot, summed over R_copy = {0,1}
    rm = out["reduce_mask"]    # [B, N-1, R_reduce]
    B, N, _ = cm.shape
    for b in range(B):
        # Per-position coverage: each t is covered by copy@t OR reduce@t OR
        # reduce@t-1, exactly one.
        for t in range(N):
            covered = int(cm[b, t].sum().item())
            if t < N - 1:
                covered += int(rm[b, t].sum().item())
            if t > 0:
                covered += int(rm[b, t - 1].sum().item())
            assert covered == 1, f"b={b} t={t} covered={covered}"


def test_viterbi_one_hot_per_active_action():
    cs, rs = _rand_scores(2, 5, 3, 2, seed=11)
    out = binary_tiling_viterbi(cs, rs)
    cm = out["copy_mask"]
    rm = out["reduce_mask"]
    # Where the action fires, exactly one op is selected; elsewhere all zero.
    cm_fired = cm.sum(-1)
    rm_fired = rm.sum(-1)
    assert torch.all((cm_fired == 0) | (cm_fired == 1))
    assert torch.all((rm_fired == 0) | (rm_fired == 1))
```

- [ ] **Step 3.2: Run to verify failure**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_viterbi.py -v`
Expected: tests FAIL — `binary_tiling_viterbi` is not yet implemented.

- [ ] **Step 3.3: Implement multi-op Viterbi in SignalRouter.py**

```python
# basicmodel/bin/SignalRouter.py — append

def binary_tiling_viterbi(
    copy_score: torch.Tensor,
    reduce_score: torch.Tensor,
):
    """Argmax legal COPY/REDUCE tiling, multi-op.

    Args:
        copy_score:   [B, N, R_copy]
        reduce_score: [B, N-1, R_reduce]

    Returns:
        score:       [B] best-route score.
        copy_mask:   [B, N, R_copy] one-hot at chosen (position, op) for COPY.
        reduce_mask: [B, N-1, R_reduce] one-hot at chosen (position, op).
        action_kind: [B, N+1] long; backpointer kind at each step boundary.
        action_op:   [B, N+1] long; backpointer op at each step boundary.
    """
    B, N, R_copy = copy_score.shape
    R_reduce = reduce_score.shape[-1] if reduce_score.numel() > 0 else 0
    device = copy_score.device
    dtype = copy_score.dtype

    if N == 0:
        return {
            "score": torch.zeros(B, device=device, dtype=dtype),
            "copy_mask": copy_score.new_zeros(B, 0, R_copy),
            "reduce_mask": copy_score.new_zeros(B, 0, R_reduce),
            "action_kind": torch.zeros(B, 1, device=device, dtype=torch.long),
            "action_op": torch.zeros(B, 1, device=device, dtype=torch.long),
        }

    NEG_INF = -1e9

    # Argmax over op axis collapses to per-action best score + op id.
    c_best, c_argop = copy_score.max(dim=-1)    # [B, N], [B, N]
    if R_reduce > 0 and N > 1:
        r_best, r_argop = reduce_score.max(dim=-1)  # [B, N-1], [B, N-1]
    else:
        r_best = copy_score.new_full((B, max(N - 1, 0)), NEG_INF)
        r_argop = torch.zeros(B, max(N - 1, 0), device=device, dtype=torch.long)

    dp = copy_score.new_full((B, N + 1), NEG_INF)
    dp[:, 0] = 0.0
    back_kind = torch.full((B, N + 1), -1, device=device, dtype=torch.long)
    back_op = torch.zeros((B, N + 1), device=device, dtype=torch.long)

    for t in range(N):
        cand_copy = dp[:, t] + c_best[:, t]
        better = cand_copy > dp[:, t + 1]
        dp[:, t + 1] = torch.where(better, cand_copy, dp[:, t + 1])
        back_kind[:, t + 1] = torch.where(
            better, torch.zeros_like(back_kind[:, t + 1]), back_kind[:, t + 1])
        back_op[:, t + 1] = torch.where(better, c_argop[:, t], back_op[:, t + 1])

        if t + 1 < N:
            cand_reduce = dp[:, t] + r_best[:, t]
            better_r = cand_reduce > dp[:, t + 2]
            dp[:, t + 2] = torch.where(better_r, cand_reduce, dp[:, t + 2])
            back_kind[:, t + 2] = torch.where(
                better_r, torch.ones_like(back_kind[:, t + 2]),
                back_kind[:, t + 2])
            back_op[:, t + 2] = torch.where(
                better_r, r_argop[:, t], back_op[:, t + 2])

    copy_mask = copy_score.new_zeros(B, N, R_copy)
    reduce_mask = copy_score.new_zeros(B, max(N - 1, 0), R_reduce)

    # Backtrace per row. Python loop is fine for now; replace with batched
    # gather if profiling demands. The DP itself is the hot path on GPU.
    for b in range(B):
        t = N
        while t > 0:
            kind = int(back_kind[b, t].item())
            op = int(back_op[b, t].item())
            if kind == 0:  # copy
                copy_mask[b, t - 1, op] = 1.0
                t -= 1
            elif kind == 1:  # reduce
                reduce_mask[b, t - 2, op] = 1.0
                t -= 2
            else:
                raise RuntimeError(
                    f"Viterbi backtrace at b={b} t={t} has no valid backpointer "
                    f"(kind={kind}). DP message corrupt.")

    return {
        "score": dp[:, N],
        "copy_mask": copy_mask,
        "reduce_mask": reduce_mask,
        "action_kind": back_kind,
        "action_op": back_op,
    }
```

- [ ] **Step 3.4: Run the Viterbi tests**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_viterbi.py -v`
Expected: all three tests PASS.

- [ ] **Step 3.5: Pause for user review and commit**

Summarize: multi-op `binary_tiling_viterbi` returning per-(position, op) one-hot masks plus backpointers; legality (no overlap, full coverage) and score-vs-brute-force verified for N ∈ [1, 6].

---

## Task 4: Local placement scorers

Local MLPs that read the contextualized slab and produce per-(position, op) scalar fields. Multi-op output dim configurable per layer.

**Files:**
- Modify: `basicmodel/bin/SignalRouter.py`
- Test: `basicmodel/test/test_signal_router_scorer.py`

- [ ] **Step 4.1: Write failing scorer tests**

```python
# basicmodel/test/test_signal_router_scorer.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch

from SignalRouter import BinaryPlacementScorer


def test_scorer_shapes():
    B, N, D = 2, 5, 8
    R_copy, R_reduce = 3, 4
    s = BinaryPlacementScorer(d_model=D, r_copy=R_copy, r_reduce=R_reduce)
    h = torch.randn(B, N, D)
    reduced = torch.randn(B, N - 1, D)
    cs, rs = s(h=h, reduced=reduced)
    assert cs.shape == (B, N, R_copy)
    assert rs.shape == (B, N - 1, R_reduce)


def test_scorer_gradient_reaches_h_and_reduced():
    B, N, D = 1, 4, 6
    s = BinaryPlacementScorer(d_model=D, r_copy=2, r_reduce=2)
    h = torch.randn(B, N, D, requires_grad=True)
    reduced = torch.randn(B, N - 1, D, requires_grad=True)
    cs, rs = s(h=h, reduced=reduced)
    (cs.sum() + rs.sum()).backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    assert reduced.grad is not None and reduced.grad.abs().sum() > 0


def test_scorer_handles_n_one_no_reduce_pairs():
    B, N, D = 1, 1, 4
    s = BinaryPlacementScorer(d_model=D, r_copy=2, r_reduce=2)
    h = torch.randn(B, N, D)
    reduced = torch.empty(B, 0, D)
    cs, rs = s(h=h, reduced=reduced)
    assert cs.shape == (B, 1, 2)
    assert rs.shape == (B, 0, 2)
```

- [ ] **Step 4.2: Run to verify failure**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_scorer.py -v`
Expected: tests FAIL — `BinaryPlacementScorer` is not yet defined.

- [ ] **Step 4.3: Implement BinaryPlacementScorer**

```python
# basicmodel/bin/SignalRouter.py — append

class BinaryPlacementScorer(nn.Module):
    """Local MLPs producing per-(position, op) COPY/REDUCE scores."""

    def __init__(self, d_model: int, r_copy: int, r_reduce: int,
                 hidden: int = None):
        super().__init__()
        hidden = hidden if hidden is not None else d_model
        self.r_copy = int(r_copy)
        self.r_reduce = int(r_reduce)
        self.copy = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.r_copy),
        )
        self.reduce = nn.Sequential(
            nn.Linear(3 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.r_reduce),
        )

    def forward(self, *, h: torch.Tensor, reduced: torch.Tensor):
        """h: [B, N, D] context; reduced: [B, N-1, D] candidate ops."""
        copy_score = self.copy(h)                              # [B, N, R_copy]
        if reduced.shape[1] == 0:
            B = h.shape[0]
            reduce_score = h.new_zeros(B, 0, self.r_reduce)
            return copy_score, reduce_score
        h_left = h[:, :-1, :]
        h_right = h[:, 1:, :]
        feat = torch.cat([h_left, h_right, reduced], dim=-1)
        reduce_score = self.reduce(feat)                       # [B, N-1, R_reduce]
        return copy_score, reduce_score
```

- [ ] **Step 4.4: Run the scorer tests**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_scorer.py -v`
Expected: all three tests PASS.

- [ ] **Step 4.5: Pause for user review and commit**

Summarize: `BinaryPlacementScorer` with separate COPY and REDUCE MLPs producing per-op scalar fields, including the empty-reduce-pairs (`N=1`) edge.

---

## Task 5: Comparator-mixer (four-branch trainable gates)

Per output position, blend `{keep x_j, reduce-here r_j, shift-from-right x_{j+1}, pad}` with **trainable** gate weights produced by a small MLP. The DP marginals from Task 2 are exposed as a regularizer/prior in Task 12, but the actual slot composition is the comparator.

**Files:**
- Modify: `basicmodel/bin/SignalRouter.py`
- Test: `basicmodel/test/test_signal_router_comparator.py`

- [ ] **Step 5.1: Write failing comparator tests**

```python
# basicmodel/test/test_signal_router_comparator.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch

from SignalRouter import ComparatorMixer


def _gather_branches(x, reduced):
    """Build [B, N, 4, D] where branches per j are
       (keep=x_j, reduce=r_j, shift=x_{j+1}, pad=0)."""
    B, N, D = x.shape
    pad = x.new_zeros(B, 1, D)
    x_shift_right = torch.cat([x[:, 1:, :], pad], dim=1)     # x_{j+1} with last=pad
    r_padded = torch.cat([reduced, pad], dim=1)              # r_j; last=pad
    return torch.stack([x, r_padded, x_shift_right, pad.expand_as(x)], dim=2)


def test_comparator_output_shape():
    B, N, D = 2, 5, 6
    cm = ComparatorMixer(d_model=D)
    x = torch.randn(B, N, D)
    reduced = torch.randn(B, N - 1, D)
    branches = _gather_branches(x, reduced)
    h = torch.randn(B, N, D)
    y, gates = cm(h=h, branches=branches)
    assert y.shape == (B, N, D)
    assert gates.shape == (B, N, 4)
    assert torch.allclose(gates.sum(-1), torch.ones(B, N), atol=1e-5)


def test_comparator_gradient_into_each_branch_and_into_h():
    B, N, D = 1, 4, 5
    cm = ComparatorMixer(d_model=D)
    x = torch.randn(B, N, D, requires_grad=True)
    reduced = torch.randn(B, N - 1, D, requires_grad=True)
    h = torch.randn(B, N, D, requires_grad=True)
    branches = _gather_branches(x, reduced)
    y, _ = cm(h=h, branches=branches)
    y.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert reduced.grad is not None and reduced.grad.abs().sum() > 0
    assert h.grad is not None and h.grad.abs().sum() > 0


def test_comparator_temperature_sharpens_gates():
    B, N, D = 1, 4, 4
    torch.manual_seed(0)
    cm_hot = ComparatorMixer(d_model=D, temperature=10.0)
    cm_cold = ComparatorMixer(d_model=D, temperature=0.1)
    cm_cold.load_state_dict(cm_hot.state_dict())
    x = torch.randn(B, N, D)
    reduced = torch.randn(B, N - 1, D)
    h = torch.randn(B, N, D)
    branches = _gather_branches(x, reduced)
    _, g_hot = cm_hot(h=h, branches=branches)
    _, g_cold = cm_cold(h=h, branches=branches)
    # Cold (low T) ~ more peaked; max gate larger on average.
    assert g_cold.max(dim=-1).values.mean() > g_hot.max(dim=-1).values.mean()
```

- [ ] **Step 5.2: Run to verify failure**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_comparator.py -v`
Expected: tests FAIL — `ComparatorMixer` is not yet defined.

- [ ] **Step 5.3: Implement ComparatorMixer**

```python
# basicmodel/bin/SignalRouter.py — append

class ComparatorMixer(nn.Module):
    """Four-branch trainable comparator-mixer.

    Per output position j, builds
        y_j = sum_k gate_jk * branch_jk
    over branches k in (keep=x_j, reduce=r_j, shift=x_{j+1}, pad=0).

    Gate weights from a small MLP over local context h_j with optional
    bias contributions from sibling features. Softmax with configurable
    temperature gives a strict generalization of (a) the soft DP-driven
    blend (when the MLP learns to copy DP marginals into the gates) and
    (b) hard routing (when the MLP learns one-hots).
    """

    NUM_BRANCHES = 4  # keep, reduce, shift, pad

    def __init__(self, d_model: int, hidden: int = None,
                 temperature: float = 1.0):
        super().__init__()
        hidden = hidden if hidden is not None else d_model
        self.temperature = float(temperature)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.NUM_BRANCHES),
        )

    def forward(self, *, h: torch.Tensor, branches: torch.Tensor):
        """h: [B, N, D]; branches: [B, N, 4, D] in branch order
        (keep, reduce, shift, pad). Returns (y: [B, N, D], gates: [B, N, 4])."""
        gate_logits = self.gate_mlp(h) / self.temperature      # [B, N, 4]
        gates = F.softmax(gate_logits, dim=-1)
        y = (gates.unsqueeze(-1) * branches).sum(dim=2)        # [B, N, D]
        return y, gates
```

- [ ] **Step 5.4: Run the comparator tests**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_comparator.py -v`
Expected: all three tests PASS. The temperature test is sensitive to seeds; if it borders the threshold tighten by averaging over more positions or several seeds inside the test.

- [ ] **Step 5.5: Pause for user review and commit**

Summarize: `ComparatorMixer` four-branch softmax-gated blend with trainable temperature; gradient verified through every branch and through the context input.

---

## Task 6: Hard compaction with provenance

Walk the Viterbi route once, write the compacted next-layer slab, attach `src_left`, `src_right`, `action_kind`, `action_op`, and propagated `span_start`/`span_end`. Python-loop implementation; profiling-driven kernel work is a follow-up.

**Files:**
- Modify: `basicmodel/bin/SignalRouter.py`
- Test: `basicmodel/test/test_signal_router_compaction.py`

- [ ] **Step 6.1: Write failing compaction tests**

```python
# basicmodel/test/test_signal_router_compaction.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch

from SignalRouter import compact_hard, binary_tiling_viterbi


def _trivial_op(left, right):
    return left + right


def test_compact_hard_lengths_match_viterbi_route():
    B, N, D = 2, 6, 4
    torch.manual_seed(0)
    x = torch.randn(B, N, D)
    reduced = _trivial_op(x[:, :-1], x[:, 1:])
    copy_score = torch.randn(B, N, 1)
    reduce_score = torch.randn(B, N - 1, 1)
    route = binary_tiling_viterbi(copy_score, reduce_score)
    y, meta = compact_hard(
        x=x, reduced=reduced,
        copy_mask=route["copy_mask"], reduce_mask=route["reduce_mask"],
    )
    assert y.shape == (B, N, D)  # padded to input length
    for b in range(B):
        n_reduce = int(route["reduce_mask"][b].sum().item())
        n_copy = int(route["copy_mask"][b].sum().item())
        assert int(meta["lengths"][b].item()) == n_copy + n_reduce
        assert n_copy + 2 * n_reduce == N


def test_compact_hard_provenance_pointers():
    B, N, D = 1, 5, 3
    x = torch.arange(B * N * D).view(B, N, D).float()
    reduced = x[:, :-1] + x[:, 1:]
    # Force tiling: REDUCE(0,1), COPY(2), REDUCE(3,4) by hand-built masks.
    cm = torch.zeros(B, N, 1)
    cm[0, 2, 0] = 1.0
    rm = torch.zeros(B, N - 1, 1)
    rm[0, 0, 0] = 1.0
    rm[0, 3, 0] = 1.0
    y, meta = compact_hard(x=x, reduced=reduced,
                           copy_mask=cm, reduce_mask=rm)
    L = int(meta["lengths"][0].item())
    assert L == 3
    # Slot 0: REDUCE(0,1) -> src_left=0 src_right=1 action_kind=1
    assert int(meta["src_left"][0, 0].item()) == 0
    assert int(meta["src_right"][0, 0].item()) == 1
    assert int(meta["action_kind"][0, 0].item()) == 1
    # Slot 1: COPY(2) -> src_left=2 src_right=-1 action_kind=0
    assert int(meta["src_left"][0, 1].item()) == 2
    assert int(meta["src_right"][0, 1].item()) == -1
    assert int(meta["action_kind"][0, 1].item()) == 0
    # Slot 2: REDUCE(3,4) -> src_left=3 src_right=4 action_kind=1
    assert int(meta["src_left"][0, 2].item()) == 3
    assert int(meta["src_right"][0, 2].item()) == 4
    assert int(meta["action_kind"][0, 2].item()) == 1


def test_compact_hard_span_start_end_propagation():
    B, N, D = 1, 4, 2
    x = torch.randn(B, N, D)
    reduced = x[:, :-1] + x[:, 1:]
    cm = torch.zeros(B, N, 1)
    cm[0, 0, 0] = 1.0
    cm[0, 3, 0] = 1.0
    rm = torch.zeros(B, N - 1, 1)
    rm[0, 1, 0] = 1.0  # REDUCE(1,2)
    span_start = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    span_end   = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    _, meta = compact_hard(
        x=x, reduced=reduced, copy_mask=cm, reduce_mask=rm,
        span_start=span_start, span_end=span_end,
    )
    L = int(meta["lengths"][0].item())
    assert L == 3
    # Slot 0: COPY(0) -> [0,0]
    # Slot 1: REDUCE(1,2) -> [1,2]
    # Slot 2: COPY(3) -> [3,3]
    assert meta["span_start"][0, :L].tolist() == [0, 1, 3]
    assert meta["span_end"][0, :L].tolist() == [0, 2, 3]
```

- [ ] **Step 6.2: Run to verify failure**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_compaction.py -v`
Expected: tests FAIL — `compact_hard` is not yet defined.

- [ ] **Step 6.3: Implement compact_hard**

```python
# basicmodel/bin/SignalRouter.py — append

def compact_hard(
    *,
    x: torch.Tensor,                 # [B, N, D]
    reduced: torch.Tensor,           # [B, N-1, D]
    copy_mask: torch.Tensor,         # [B, N, R_copy]
    reduce_mask: torch.Tensor,       # [B, N-1, R_reduce]
    span_start: torch.Tensor = None,
    span_end: torch.Tensor = None,
):
    """Walk the hard route and write the compacted slab. Output is padded
    to length N so all batches share a tensor; per-row valid length is
    returned in metadata.
    """
    B, N, D = x.shape
    device = x.device
    dtype = x.dtype

    y = x.new_zeros(B, N, D)
    src_left = torch.full((B, N), -1, device=device, dtype=torch.long)
    src_right = torch.full((B, N), -1, device=device, dtype=torch.long)
    action_kind = torch.full((B, N), -1, device=device, dtype=torch.long)
    action_op = torch.full((B, N), -1, device=device, dtype=torch.long)

    have_spans = span_start is not None and span_end is not None
    if have_spans:
        next_span_start = torch.full((B, N), -1, device=device, dtype=torch.long)
        next_span_end = torch.full((B, N), -1, device=device, dtype=torch.long)
    lengths = torch.zeros(B, device=device, dtype=torch.long)

    cm_per_pos = copy_mask.sum(-1)        # [B, N]
    rm_per_pos = reduce_mask.sum(-1)      # [B, N-1]
    cm_op = copy_mask.argmax(-1)          # [B, N]
    rm_op = reduce_mask.argmax(-1) if reduce_mask.numel() > 0 else \
        torch.zeros_like(cm_op[:, :0])    # [B, N-1]

    for b in range(B):
        i = 0
        j = 0
        while i < N:
            do_reduce = (
                i < N - 1
                and float(rm_per_pos[b, i].item()) > 0.5
            )
            if do_reduce:
                y[b, j] = reduced[b, i]
                src_left[b, j] = i
                src_right[b, j] = i + 1
                action_kind[b, j] = 1
                action_op[b, j] = rm_op[b, i]
                if have_spans:
                    next_span_start[b, j] = span_start[b, i]
                    next_span_end[b, j] = span_end[b, i + 1]
                i += 2
                j += 1
            else:
                # Defensive: if neither copy nor reduce is selected here,
                # treat as copy of source (legality should prevent this).
                y[b, j] = x[b, i]
                src_left[b, j] = i
                src_right[b, j] = -1
                action_kind[b, j] = 0
                action_op[b, j] = cm_op[b, i] if cm_per_pos[b, i] > 0 else -1
                if have_spans:
                    next_span_start[b, j] = span_start[b, i]
                    next_span_end[b, j] = span_end[b, i]
                i += 1
                j += 1
        lengths[b] = j

    meta = {
        "lengths": lengths,
        "src_left": src_left,
        "src_right": src_right,
        "action_kind": action_kind,
        "action_op": action_op,
    }
    if have_spans:
        meta["span_start"] = next_span_start
        meta["span_end"] = next_span_end
    return y, meta
```

- [ ] **Step 6.4: Run the compaction tests**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_compaction.py -v`
Expected: all three tests PASS.

- [ ] **Step 6.5: Pause for user review and commit**

Summarize: `compact_hard` writes the next-layer slab plus full provenance — `lengths`, `src_left`/`src_right`, `action_kind`, `action_op`, propagated `span_start`/`span_end`. Lengths and pointers verified on a hand-built tiling.

---

## Task 7: Soft compaction (length-N) for the inverse pass

The inverse / generate pass needs gradient on the routing decision per position. Build a length-N alignment view of the slab using the soft DP marginals — every output slot is a fraction of the source-position copies and reductions. Differentiable through the marginals.

**Files:**
- Modify: `basicmodel/bin/SignalRouter.py`
- Test: extends `basicmodel/test/test_signal_router_compaction.py`

- [ ] **Step 7.1: Append failing soft-compaction tests**

```python
# basicmodel/test/test_signal_router_compaction.py — append

from SignalRouter import compact_soft


def test_compact_soft_returns_length_N_slab():
    B, N, D = 2, 5, 4
    x = torch.randn(B, N, D)
    reduced = x[:, :-1] + x[:, 1:]
    p_copy = torch.full((B, N), 1.0)               # all-copy marginals
    p_reduce = torch.zeros(B, N - 1)
    y_soft = compact_soft(
        x=x, reduced=reduced,
        copy_marginal=p_copy, reduce_marginal=p_reduce,
    )
    assert y_soft.shape == (B, N, D)
    # All-copy means y_soft == x.
    assert torch.allclose(y_soft, x, atol=1e-5)


def test_compact_soft_single_reduction_blends_neighbours():
    B, N, D = 1, 4, 3
    x = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]],
                     dtype=torch.float32)
    reduced = x[:, :-1] + x[:, 1:]
    # Force REDUCE@1 with mass 1; copies elsewhere.
    p_copy = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    p_reduce = torch.tensor([[0.0, 1.0, 0.0]])
    y_soft = compact_soft(
        x=x, reduced=reduced,
        copy_marginal=p_copy, reduce_marginal=p_reduce,
    )
    # Position 0: copy x_0
    assert torch.allclose(y_soft[0, 0], x[0, 0])
    # Position 1: r_1 = x_1 + x_2
    assert torch.allclose(y_soft[0, 1], x[0, 1] + x[0, 2])
    # Position 2: shifted x_3 (post-reduction)
    assert torch.allclose(y_soft[0, 2], x[0, 3])
    # Position 3: pad (zero) since the sequence shrunk
    assert torch.allclose(y_soft[0, 3], torch.zeros(D))


def test_compact_soft_gradient_flows_through_marginals():
    B, N, D = 1, 4, 3
    x = torch.randn(B, N, D)
    reduced = x[:, :-1] + x[:, 1:]
    p_copy = torch.full((B, N), 0.5, requires_grad=True)
    p_reduce = torch.full((B, N - 1), 0.25, requires_grad=True)
    y_soft = compact_soft(
        x=x, reduced=reduced,
        copy_marginal=p_copy, reduce_marginal=p_reduce,
    )
    y_soft.sum().backward()
    assert p_copy.grad is not None and p_copy.grad.abs().sum() > 0
    assert p_reduce.grad is not None and p_reduce.grad.abs().sum() > 0
```

- [ ] **Step 7.2: Run to verify failure**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_compaction.py -v`
Expected: the three new soft-compaction tests FAIL; the hard-compaction tests still PASS.

- [ ] **Step 7.3: Implement compact_soft**

```python
# basicmodel/bin/SignalRouter.py — append

def compact_soft(
    *,
    x: torch.Tensor,                  # [B, N, D]
    reduced: torch.Tensor,            # [B, N-1, D]
    copy_marginal: torch.Tensor,      # [B, N]
    reduce_marginal: torch.Tensor,    # [B, N-1]
):
    """Length-N soft compaction view: per output position j,
        y_j = (mu_no_reduce_left[j])         * x_j
            + (P[reduce-here at j])          * r_j
            + (P[reduce-to-the-left of j])   * x_{j+1}
            + (P[shrunk past j])             * 0
    where the partition treats positions independently to a first
    approximation. Sufficient for gradient on routing decisions; the
    hard slab from compact_hard remains the clean operand source.
    """
    B, N, D = x.shape

    # Probability that a reduction has fired strictly before position j.
    # Cumulative sum of reduce_marginal up to but not including j.
    if reduce_marginal.shape[1] == 0:
        cumshift = x.new_zeros(B, N)
    else:
        # cumshift[:, j] = sum_{k<j} reduce_marginal[:, k]
        cum = torch.cumsum(reduce_marginal, dim=1)             # [B, N-1]
        cumshift = torch.cat(
            [x.new_zeros(B, 1), cum], dim=1)                   # [B, N]

    # Per-position branch weights:
    # keep   = copy_marginal[j] * (1 - cumshift[j])
    # reduce = reduce_marginal[j]  (extended with 0 at j=N-1)
    # shift  = cumshift[j]   (clamp at 1 implicitly via marginals)
    # pad    = 1 - keep - reduce - shift
    keep = copy_marginal * (1.0 - cumshift.clamp(0.0, 1.0))
    if reduce_marginal.shape[1] == 0:
        reduce_w = x.new_zeros(B, N)
    else:
        pad_zero = x.new_zeros(B, 1)
        reduce_w = torch.cat([reduce_marginal, pad_zero], dim=1)
    shift = cumshift.clamp(0.0, 1.0)
    pad_w = (1.0 - keep - reduce_w - shift).clamp(min=0.0)

    pad_slab = x.new_zeros(B, 1, D)
    x_shift = torch.cat([x[:, 1:, :], pad_slab], dim=1)        # x_{j+1}, last=pad
    if reduce_marginal.shape[1] == 0:
        r_padded = x.new_zeros(B, N, D)
    else:
        r_padded = torch.cat([reduced, pad_slab], dim=1)        # r_j, last=pad

    y = (
        keep.unsqueeze(-1) * x
        + reduce_w.unsqueeze(-1) * r_padded
        + shift.unsqueeze(-1) * x_shift
        + pad_w.unsqueeze(-1) * pad_slab.expand_as(x)
    )
    return y
```

- [ ] **Step 7.4: Run the soft-compaction tests**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_compaction.py -v`
Expected: all six tests PASS (three hard, three soft).

- [ ] **Step 7.5: Pause for user review and commit**

Summarize: `compact_soft` produces a length-N differentiable slab from the soft DP marginals, paired with `compact_hard` that produces a clean compacted slab. The two slabs together are the dual-slab pattern.

---

## Task 8: BinaryStructuredReductionLayer

Compose Tasks 2-7: contextualize, compute reduce candidates, score, structured-route (soft DP + hard Viterbi), and produce both the hard and soft slabs. The op call uses `op_id` from the Viterbi `action_op` to select among rule modules — multi-op dispatch is via the existing per-rule registry callable, parametrized by the layer at construction.

**Files:**
- Modify: `basicmodel/bin/SignalRouter.py`
- Test: `basicmodel/test/test_signal_router_layer.py`

- [ ] **Step 8.1: Write failing layer tests**

```python
# basicmodel/test/test_signal_router_layer.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
import torch.nn as nn

from SignalRouter import BinaryStructuredReductionLayer


class _AddOp(nn.Module):
    """Trivial binary op: y = left + right."""
    def forward(self, left, right):
        return left + right


class _MulOp(nn.Module):
    def forward(self, left, right):
        return left * right


def test_layer_forward_shapes():
    B, N, D = 2, 5, 4
    layer = BinaryStructuredReductionLayer(
        d_model=D, ops=[_AddOp(), _MulOp()], r_copy=1)
    x = torch.randn(B, N, D)
    hard, soft, routing = layer(x)
    assert hard.shape == (B, N, D)
    assert soft.shape == (B, N, D)
    assert routing["copy_mask"].shape == (B, N, 1)
    assert routing["reduce_mask"].shape == (B, N - 1, 2)
    assert routing["lengths"].shape == (B,)
    assert routing["copy_marginal"].shape == (B, N)
    assert routing["reduce_marginal"].shape == (B, N - 1)


def test_layer_gradient_into_op_and_scorer():
    B, N, D = 1, 4, 3
    op = _AddOp()
    layer = BinaryStructuredReductionLayer(
        d_model=D, ops=[op], r_copy=1)
    x = torch.randn(B, N, D, requires_grad=True)
    hard, soft, _ = layer(x)
    (hard.sum() + soft.sum()).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    # Scorer parameters should also receive gradient.
    grads = [p.grad for p in layer.scorer.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_layer_n_one_degenerate_pass_through():
    B, N, D = 2, 1, 3
    layer = BinaryStructuredReductionLayer(
        d_model=D, ops=[_AddOp()], r_copy=1)
    x = torch.randn(B, N, D)
    hard, soft, routing = layer(x)
    assert hard.shape == (B, N, D)
    # No reduction possible at N=1; the row pass-through is the input.
    assert torch.allclose(hard, x)
```

- [ ] **Step 8.2: Run to verify failure**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_layer.py -v`
Expected: tests FAIL — `BinaryStructuredReductionLayer` is not yet defined.

- [ ] **Step 8.3: Implement BinaryStructuredReductionLayer**

```python
# basicmodel/bin/SignalRouter.py — append

class _IdentityContext(nn.Module):
    def forward(self, x):
        return x


class BinaryStructuredReductionLayer(nn.Module):
    """One layer: contextualize, score, route, compact (hard + soft).

    Args:
        d_model: feature dim.
        ops: sequence of binary nn.Modules; len(ops) = R_reduce. Each
             receives (left[B, N-1, D], right[B, N-1, D]) and returns
             [B, N-1, D]. The Viterbi route picks one op per reduce site.
        r_copy: number of copy "ops" (typically 1; >1 lets the router
             distinguish copy specializations like typed identities).
        context_net: optional contextualizer for h. Defaults to identity.
        temperature: comparator-mixer softmax temperature.
    """

    def __init__(self, *, d_model, ops, r_copy=1, context_net=None,
                 temperature=1.0):
        super().__init__()
        self.d_model = int(d_model)
        self.ops = nn.ModuleList(list(ops))
        self.r_reduce = len(self.ops)
        self.r_copy = int(r_copy)
        self.context_net = context_net if context_net is not None else _IdentityContext()
        self.scorer = BinaryPlacementScorer(
            d_model=self.d_model,
            r_copy=self.r_copy,
            r_reduce=self.r_reduce,
        )
        self.comparator = ComparatorMixer(
            d_model=self.d_model, temperature=temperature)

    def _stacked_reduced(self, x):
        """[B, N-1, R_reduce, D] candidate ops applied to each adjacent pair."""
        if x.shape[1] < 2:
            return x.new_zeros(x.shape[0], 0, self.r_reduce, x.shape[-1])
        left = x[:, :-1, :]
        right = x[:, 1:, :]
        per_op = [op(left, right) for op in self.ops]
        return torch.stack(per_op, dim=2)                      # [B, N-1, R, D]

    def _selected_reduced(self, stacked, route_op):
        """Gather the chosen reduction at each position from the stack.
        stacked: [B, N-1, R, D]; route_op: [B, N-1] long.
        """
        B, Nm1, _, D = stacked.shape
        if Nm1 == 0:
            return stacked.new_zeros(B, 0, D)
        idx = route_op.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(
            B, Nm1, 1, D)
        gathered = stacked.gather(dim=2, index=idx).squeeze(2)
        return gathered

    def _gather_branches(self, x, reduced_chosen):
        """Build [B, N, 4, D] in branch order (keep, reduce, shift, pad)."""
        B, N, D = x.shape
        pad_slab = x.new_zeros(B, 1, D)
        x_shift = torch.cat([x[:, 1:, :], pad_slab], dim=1)
        r_padded = torch.cat([reduced_chosen, pad_slab], dim=1) \
            if reduced_chosen.shape[1] > 0 else x.new_zeros(B, N, D)
        return torch.stack(
            [x, r_padded, x_shift, pad_slab.expand_as(x)], dim=2)

    def forward(self, x, *, span_start=None, span_end=None):
        B, N, D = x.shape
        if N <= 1:
            routing = {
                "copy_mask": x.new_zeros(B, N, self.r_copy),
                "reduce_mask": x.new_zeros(B, 0, self.r_reduce),
                "lengths": torch.full((B,), N, device=x.device, dtype=torch.long),
                "copy_marginal": x.new_zeros(B, N),
                "reduce_marginal": x.new_zeros(B, 0),
                "logZ": x.new_zeros(B),
                "degenerate": True,
            }
            return x, x, routing

        h = self.context_net(x)
        stacked_reduced = self._stacked_reduced(x)             # [B, N-1, R, D]
        # Per-pair "best reduction so far" candidate for the scorer.
        # Use mean-over-ops as a stable summary; the scorer also sees per-
        # op specialization through its own multi-output head.
        reduced_summary = stacked_reduced.mean(dim=2)          # [B, N-1, D]

        copy_score, reduce_score = self.scorer(
            h=h, reduced=reduced_summary)

        soft = binary_tiling_soft_dp(copy_score, reduce_score)
        hard = binary_tiling_viterbi(copy_score, reduce_score)

        # Hard slab: pick the chosen op per reduction, compact.
        reduce_op_id = hard["action_op"]                       # [B, N+1] long
        # Per-pair op id: gather from the reduce_mask one-hot.
        if hard["reduce_mask"].numel() > 0:
            reduce_op_per_pair = hard["reduce_mask"].argmax(-1)  # [B, N-1]
        else:
            reduce_op_per_pair = torch.zeros(
                B, 0, device=x.device, dtype=torch.long)
        chosen_reduced = self._selected_reduced(
            stacked_reduced, reduce_op_per_pair)                # [B, N-1, D]

        hard_slab, hard_meta = compact_hard(
            x=x, reduced=chosen_reduced,
            copy_mask=hard["copy_mask"], reduce_mask=hard["reduce_mask"],
            span_start=span_start, span_end=span_end,
        )

        # Soft slab: comparator-mixer over the four branches, with the
        # DP marginal-driven slab as a parallel artifact for inverse-pass
        # losses (see Task 12).
        branches = self._gather_branches(x, chosen_reduced)
        soft_slab, gates = self.comparator(h=h, branches=branches)

        marginal_slab = compact_soft(
            x=x, reduced=chosen_reduced,
            copy_marginal=soft["copy_marginal"],
            reduce_marginal=soft["reduce_marginal"],
        )

        routing = {
            "copy_mask": hard["copy_mask"],
            "reduce_mask": hard["reduce_mask"],
            "lengths": hard_meta["lengths"],
            "src_left": hard_meta["src_left"],
            "src_right": hard_meta["src_right"],
            "action_kind": hard_meta["action_kind"],
            "action_op": hard_meta["action_op"],
            "copy_score": copy_score,
            "reduce_score": reduce_score,
            "copy_marginal": soft["copy_marginal"],
            "reduce_marginal": soft["reduce_marginal"],
            "copy_marginal_op": soft["copy_marginal_op"],
            "reduce_marginal_op": soft["reduce_marginal_op"],
            "logZ": soft["logZ"],
            "gates": gates,
            "marginal_slab": marginal_slab,
        }
        if span_start is not None and span_end is not None:
            routing["span_start"] = hard_meta["span_start"]
            routing["span_end"] = hard_meta["span_end"]

        return hard_slab, soft_slab, routing
```

- [ ] **Step 8.4: Run the layer tests**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_layer.py -v`
Expected: all three tests PASS.

- [ ] **Step 8.5: Pause for user review and commit**

Summarize: `BinaryStructuredReductionLayer` ties scorer + soft DP + hard Viterbi + comparator-mixer + dual slab compaction (`compact_hard` clean operand path, `comparator`/`compact_soft` differentiable paths). End-to-end gradient verified.

---

## Task 9: WordSpace dispatch — minimal end-to-end signal compose

Wire `SignalRouter.compose` into the live `current_rules` contract. The first cut runs **one** `BinaryStructuredReductionLayer` over the input slab, produces per-tier rule-id selections from `action_op`, and populates `word_space.current_rules` with the same shape as the chart path. This is intentionally minimal — multi-tier stacking and full grammar parity are deferred to follow-up plans.

**Files:**
- Modify: `basicmodel/bin/SignalRouter.py`
- Test: extends `basicmodel/test/test_signal_router_wordspace.py`

- [ ] **Step 9.1: Append failing dispatch tests**

```python
# basicmodel/test/test_signal_router_wordspace.py — append

import torch.nn as nn


class _Stub(nn.Module):
    def forward(self, left, right):
        return left + right


class _StubWordSpace:
    def __init__(self):
        self.current_rules = {}
        self.generate_rules = {}
        self._compose_generation = 0

    def host_layer(self, tier, rule_name):
        return None


def test_signal_compose_populates_current_rules():
    chart = Chart(nInput=4, nOutput=4, max_depth=3, hidden_dim=16,
                  feature_dim=4, router_kind="signal")
    # Inject a tiny ops list onto the lazy SignalRouter so compose can run.
    router = chart._ensure_signal_router()
    router.attach_layer_ops(ops=[_Stub()], r_copy=1, tier="C")
    ws = _StubWordSpace()
    rules = chart.compose(torch.randn(2, 4, 4), word_space=ws)
    # Returned dict has the per-tier shape: {tier: list-of-rows-of-rule-ids}.
    assert isinstance(rules, dict)
    assert "C" in rules
    rows = rules["C"]
    assert len(rows) == 2  # batch
    for row in rows:
        # Each row is a list of rule_ids for the reductions on that batch.
        assert isinstance(row, list)
        for rid in row:
            assert isinstance(rid, int)
            assert 0 <= rid < 1  # only one op attached
```

- [ ] **Step 9.2: Run to verify failure**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_wordspace.py -v`
Expected: existing dispatch tests PASS, new test FAILS — `attach_layer_ops` and the populated `compose` are not yet implemented.

- [ ] **Step 9.3: Implement SignalRouter.compose with one binary layer**

```python
# basicmodel/bin/SignalRouter.py — modify SignalRouter

class SignalRouter(nn.Module):
    def __init__(self, n_input, n_output, *, hidden_dim, feature_dim,
                 max_depth, temperature=1.0):
        super().__init__()
        self.n_input = int(n_input)
        self.n_output = int(n_output)
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        self.max_depth = int(max_depth)
        self.temperature = float(temperature)
        self._binary_layer = None
        self._binary_tier = None

    def attach_layer_ops(self, *, ops, r_copy=1, tier="C"):
        """Build the binary reduction layer with the given ops list.

        Tier names follow the chart convention ("P", "C", "S", ...).
        Op order in the list defines op_id; collect_rule_selections will
        emit those ids in the resulting per-tier dict.
        """
        # Assigning an nn.Module to an attribute auto-registers it as a
        # submodule, so it is included in parameters() / state_dict().
        self._binary_layer = BinaryStructuredReductionLayer(
            d_model=self.feature_dim,
            ops=ops,
            r_copy=r_copy,
            temperature=self.temperature,
        )
        self._binary_tier = str(tier)

    def compose(self, data, word_space, subspace=None):
        if self._binary_layer is None:
            raise RuntimeError(
                "SignalRouter.compose called before attach_layer_ops().")
        hard_slab, _soft_slab, routing = self._binary_layer(data)
        rules = self._collect_rule_selections(routing)
        # Stash for inverse pass / inspection.
        self._last_routing = routing
        self._last_hard_slab = hard_slab
        return rules

    def generate(self, target, word_space, subspace=None):
        # Plan task 12 fills this in. Until then, return empty.
        return {}

    def _collect_rule_selections(self, routing):
        """Emit {tier: [[rule_id, ...] per batch row]} from routing."""
        action_kind = routing["action_kind"]    # [B, N]
        action_op = routing["action_op"]        # [B, N]
        lengths = routing["lengths"]            # [B]
        B = action_kind.shape[0]
        rows = []
        for b in range(B):
            row = []
            L = int(lengths[b].item())
            for j in range(L):
                if int(action_kind[b, j].item()) == 1:
                    row.append(int(action_op[b, j].item()))
            rows.append(row)
        return {self._binary_tier: rows}
```

- [ ] **Step 9.4: Run the dispatch tests**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_wordspace.py -v`
Expected: all dispatch tests PASS.

- [ ] **Step 9.5: Run the full test suite to confirm chart path still green**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/ -x -q`
Expected: same pre-Task-0 baseline pass/fail set. New regressions must trace to this task.

- [ ] **Step 9.6: Pause for user review and commit**

Summarize: `SignalRouter.compose` runs a single `BinaryStructuredReductionLayer` and emits a chart-compatible per-tier rule selection dict via `attach_layer_ops`. End-to-end signal path produces the structural contract the rest of the model expects.

---

## Task 10: Unary layer + XOR-grammar acceptance (AND, OR, NOT)

Validate the signal router against the XOR grammar's three primitives — AND, OR, NOT — end-to-end. AND and OR are binary (already supported by Task 8). NOT is unary, so this task adds a `UnaryStructuredLayer`. Per-position action choice in a unary layer is independent across positions (no overlap constraint), so no structured DP is needed — a per-position softmax over `R_copy + R_apply` actions is sufficient.

The acceptance test stacks one unary tier (NOT) in front of one binary tier (AND, OR) inside a single `SignalRouter` instance and runs a small fixture through it. **Multi-space parallel routers** (PerceptualSpace / ConceptualSpace / SymbolicSpace each owning their own router) remain deferred — see open-questions in Task 13.

**Files:**
- Modify: `basicmodel/bin/SignalRouter.py` (add `UnaryStructuredLayer`, `attach_unary_ops`, two-tier compose stacking)
- Test: `basicmodel/test/test_signal_router_unary.py`
- Test: `basicmodel/test/test_signal_router_xor_grammar.py`

- [ ] **Step 10.1: Write failing unary layer tests**

```python
# basicmodel/test/test_signal_router_unary.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
import torch.nn as nn

from SignalRouter import UnaryStructuredLayer


class _NegateOp(nn.Module):
    def forward(self, x):
        return -x


class _AbsOp(nn.Module):
    def forward(self, x):
        return x.abs()


def test_unary_layer_output_shape_unchanged():
    B, N, D = 2, 5, 4
    layer = UnaryStructuredLayer(d_model=D, ops=[_NegateOp(), _AbsOp()],
                                 r_copy=1)
    x = torch.randn(B, N, D)
    hard, soft, routing = layer(x)
    assert hard.shape == (B, N, D)
    assert soft.shape == (B, N, D)
    # action axis = R_copy + R_apply = 1 + 2 = 3
    assert routing["action_logits"].shape == (B, N, 3)
    assert routing["action_probs"].shape == (B, N, 3)
    assert torch.allclose(routing["action_probs"].sum(-1),
                          torch.ones(B, N), atol=1e-5)


def test_unary_layer_hard_one_hot_per_position():
    B, N, D = 1, 4, 3
    layer = UnaryStructuredLayer(d_model=D, ops=[_NegateOp()], r_copy=1)
    x = torch.randn(B, N, D)
    _, _, routing = layer(x)
    cm = routing["copy_mask"]      # [B, N, R_copy]
    am = routing["apply_mask"]     # [B, N, R_apply]
    fired = cm.sum(-1) + am.sum(-1)
    assert torch.all(fired == 1.0)


def test_unary_layer_gradient_into_op_and_input():
    B, N, D = 1, 4, 3
    op = _NegateOp()
    layer = UnaryStructuredLayer(d_model=D, ops=[op], r_copy=1)
    x = torch.randn(B, N, D, requires_grad=True)
    hard, soft, _ = layer(x)
    (hard.sum() + soft.sum()).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    grads = [p.grad for p in layer.scorer.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
```

- [ ] **Step 10.2: Run to verify failure**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_unary.py -v`
Expected: tests FAIL — `UnaryStructuredLayer` is not yet defined.

- [ ] **Step 10.3: Implement UnaryStructuredLayer**

```python
# basicmodel/bin/SignalRouter.py — append

class _UnaryPlacementScorer(nn.Module):
    """Per-position logits over (R_copy + R_apply) action choices."""
    def __init__(self, d_model: int, r_copy: int, r_apply: int,
                 hidden: int = None):
        super().__init__()
        hidden = hidden if hidden is not None else d_model
        self.r_copy = int(r_copy)
        self.r_apply = int(r_apply)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.r_copy + self.r_apply),
        )

    def forward(self, h: torch.Tensor):
        return self.mlp(h)                                 # [B, N, R_c + R_a]


class UnaryStructuredLayer(nn.Module):
    """One unary layer: contextualize, score, choose action per position.

    Action space per position: R_copy + R_apply choices, exactly one
    fires. No structured DP — positions are independent under unary.

    Hard slab: argmax-selected action per position.
    Soft slab: softmax-weighted blend over (copy of x_j) and (apply of
    each unary op to x_j).
    """

    def __init__(self, *, d_model, ops, r_copy=1, context_net=None,
                 temperature=1.0):
        super().__init__()
        self.d_model = int(d_model)
        self.ops = nn.ModuleList(list(ops))
        self.r_apply = len(self.ops)
        self.r_copy = int(r_copy)
        self.temperature = float(temperature)
        self.context_net = context_net if context_net is not None else _IdentityContext()
        self.scorer = _UnaryPlacementScorer(
            d_model=self.d_model, r_copy=self.r_copy, r_apply=self.r_apply,
        )

    def _stacked_applied(self, x):
        """[B, N, R_apply, D] each unary op applied to every position."""
        if self.r_apply == 0:
            B, N, D = x.shape
            return x.new_zeros(B, N, 0, D)
        per_op = [op(x) for op in self.ops]
        return torch.stack(per_op, dim=2)

    def forward(self, x):
        B, N, D = x.shape
        h = self.context_net(x)
        applied = self._stacked_applied(x)                 # [B, N, R_apply, D]

        action_logits = self.scorer(h) / self.temperature  # [B, N, R_c + R_a]
        action_probs = F.softmax(action_logits, dim=-1)

        # Soft slab: weighted blend over (copy x_j) and applied_op(x_j).
        # branches: [B, N, R_c + R_a, D]
        copy_branch = x.unsqueeze(2).expand(B, N, self.r_copy, D)
        if self.r_apply > 0:
            branches = torch.cat([copy_branch, applied], dim=2)
        else:
            branches = copy_branch
        soft_slab = (action_probs.unsqueeze(-1) * branches).sum(dim=2)

        # Hard slab: argmax over actions.
        action_id = action_logits.argmax(dim=-1)            # [B, N]
        is_copy = action_id < self.r_copy
        # Gather the selected branch per position.
        gather_idx = action_id.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1, D)
        hard_slab = branches.gather(dim=2, index=gather_idx).squeeze(2)

        # Decompose into copy_mask / apply_mask one-hots.
        copy_mask = x.new_zeros(B, N, self.r_copy)
        apply_mask = x.new_zeros(B, N, self.r_apply) if self.r_apply > 0 \
            else x.new_zeros(B, N, 0)
        flat_one_hot = F.one_hot(
            action_id, num_classes=self.r_copy + self.r_apply
        ).to(x.dtype)
        if self.r_copy > 0:
            copy_mask = flat_one_hot[..., :self.r_copy]
        if self.r_apply > 0:
            apply_mask = flat_one_hot[..., self.r_copy:]

        # Per-position rule_id emission for chart-compatible dict.
        # Apply ops are op_id 0..R_apply-1; copy ops are op_id 0..R_copy-1
        # (separate namespace via "kind" flag in routing).
        action_kind = torch.where(
            is_copy,
            torch.zeros_like(action_id),                    # 0 == copy
            torch.full_like(action_id, 2),                  # 2 == apply (unary)
        )
        action_op = torch.where(
            is_copy, action_id, action_id - self.r_copy)

        routing = {
            "action_logits": action_logits,
            "action_probs": action_probs,
            "copy_mask": copy_mask,
            "apply_mask": apply_mask,
            "action_kind": action_kind,
            "action_op": action_op,
            "lengths": torch.full((B,), N, device=x.device, dtype=torch.long),
        }
        return hard_slab, soft_slab, routing
```

- [ ] **Step 10.4: Run the unary tests**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_unary.py -v`
Expected: all three tests PASS.

- [ ] **Step 10.5: Write failing XOR-grammar acceptance test**

```python
# basicmodel/test/test_signal_router_xor_grammar.py
"""End-to-end acceptance: stack NOT (unary) in front of AND/OR (binary)
inside one SignalRouter and confirm the dispatch produces sensible
per-tier rule selections plus full-graph gradient flow.

The ops here are minimal float-tensor proxies for AND / OR / NOT;
plugging in real GRAMMAR_LAYER_CLASSES instances is a follow-up plan
(see Task 13 open questions).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
import torch.nn as nn

import Language
from Language import Chart


class _StubWordSpace:
    def __init__(self):
        self.current_rules = {}
        self.generate_rules = {}
        self._compose_generation = 0
    def host_layer(self, tier, rule_name):
        return None


class _AndOp(nn.Module):
    """Multiplicative AND: matches the existing pi-style conjunction."""
    def forward(self, left, right):
        return left * right


class _OrOp(nn.Module):
    """Additive OR clipped to [-1,1]: matches the existing sigma-style
    disjunction shape."""
    def forward(self, left, right):
        return (left + right).clamp(min=-1.0, max=1.0)


class _NotOp(nn.Module):
    """Sign flip; XOR-fixture truths live in {-1, 1}."""
    def forward(self, x):
        return -x


def test_xor_router_emits_per_tier_rule_dict():
    chart = Chart(nInput=4, nOutput=4, max_depth=3, hidden_dim=16,
                  feature_dim=4, router_kind="signal")
    router = chart._ensure_signal_router()
    router.attach_unary_ops(ops=[_NotOp()], r_copy=1, tier="C_unary")
    router.attach_layer_ops(ops=[_AndOp(), _OrOp()], r_copy=1, tier="C")
    ws = _StubWordSpace()
    rules = chart.compose(torch.randn(2, 4, 4), word_space=ws)
    assert "C_unary" in rules
    assert "C" in rules
    # Binary tier has 2 ops (AND, OR).
    for row in rules["C"]:
        for rid in row:
            assert rid in (0, 1), f"unexpected binary op id {rid}"
    # Unary tier has 1 op (NOT). When the router chooses APPLY, op_id is 0.
    for row in rules["C_unary"]:
        for rid in row:
            assert rid == 0, f"unexpected unary op id {rid}"


def test_xor_router_gradients_reach_all_three_ops():
    chart = Chart(nInput=4, nOutput=4, max_depth=3, hidden_dim=16,
                  feature_dim=4, router_kind="signal")
    router = chart._ensure_signal_router()
    not_op, and_op, or_op = _NotOp(), _AndOp(), _OrOp()
    # Wrap the tensorless ops in tiny linear shells so they have parameters
    # to receive gradient. Re-use real GrammarLayer instances would skip
    # this; for now use Linear(D, D, bias=False) before the op as the proxy
    # parameter source.
    class _ParamApply(nn.Module):
        def __init__(self, D, op, arity):
            super().__init__()
            self.proj = nn.Linear(D, D, bias=False)
            self.op = op
            self.arity = arity
        def forward(self, *args):
            if self.arity == 1:
                return self.op(self.proj(args[0]))
            return self.op(self.proj(args[0]), self.proj(args[1]))

    D = 4
    pnot = _ParamApply(D, not_op, 1)
    pand = _ParamApply(D, and_op, 2)
    por = _ParamApply(D, or_op, 2)
    router.attach_unary_ops(ops=[pnot], r_copy=1, tier="C_unary")
    router.attach_layer_ops(ops=[pand, por], r_copy=1, tier="C")

    ws = _StubWordSpace()
    x = torch.randn(2, 4, D, requires_grad=True)
    chart.compose(x, word_space=ws)
    # Back-prop a sum over the soft slab (most parameters are in the unary
    # tier on the soft path).
    loss = router._last_soft_slab.sum() + router._last_hard_slab.sum()
    loss.backward()
    # Every op's projection should see a gradient (each gets exercised in
    # the soft mixture even when not the hard pick).
    for name, p in [("not", pnot.proj), ("and", pand.proj), ("or", por.proj)]:
        assert p.weight.grad is not None and p.weight.grad.abs().sum() > 0, \
            f"no gradient reached the {name} op"
```

- [ ] **Step 10.6: Run to verify failure**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_xor_grammar.py -v`
Expected: tests FAIL — `attach_unary_ops` and the two-tier `compose` are not yet implemented.

- [ ] **Step 10.7: Add attach_unary_ops + two-tier compose stacking**

Modify `SignalRouter` to support a unary tier in front of the binary tier. Replace its `__init__`, add `attach_unary_ops`, and update `compose` and `_collect_rule_selections` to emit both tiers.

```python
# basicmodel/bin/SignalRouter.py — replace SignalRouter class

class SignalRouter(nn.Module):
    def __init__(self, n_input, n_output, *, hidden_dim, feature_dim,
                 max_depth, temperature=1.0):
        super().__init__()
        self.n_input = int(n_input)
        self.n_output = int(n_output)
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        self.max_depth = int(max_depth)
        self.temperature = float(temperature)
        self._unary_layer = None
        self._unary_tier = None
        self._binary_layer = None
        self._binary_tier = None
        self._last_routing = None
        self._last_unary_routing = None
        self._last_hard_slab = None
        self._last_soft_slab = None

    def attach_unary_ops(self, *, ops, r_copy=1, tier="C_unary"):
        self._unary_layer = UnaryStructuredLayer(
            d_model=self.feature_dim,
            ops=ops, r_copy=r_copy,
            temperature=self.temperature,
        )
        self._unary_tier = str(tier)

    def attach_layer_ops(self, *, ops, r_copy=1, tier="C"):
        self._binary_layer = BinaryStructuredReductionLayer(
            d_model=self.feature_dim,
            ops=ops, r_copy=r_copy,
            temperature=self.temperature,
        )
        self._binary_tier = str(tier)

    def compose(self, data, word_space, subspace=None):
        if self._binary_layer is None and self._unary_layer is None:
            raise RuntimeError(
                "SignalRouter.compose called before any attach_* call.")
        x = data
        rules = {}

        if self._unary_layer is not None:
            u_hard, u_soft, u_routing = self._unary_layer(x)
            self._last_unary_routing = u_routing
            rules[self._unary_tier] = self._collect_unary_rule_selections(u_routing)
            # Pass the soft (differentiable) slab to the binary tier so
            # gradient reaches the unary ops; hard slab is also valid and
            # is what the chart_kind="signal" downstream consumer would
            # use in production. Choice is documented as an open question.
            x = u_soft

        if self._binary_layer is not None:
            hard_slab, soft_slab, routing = self._binary_layer(x)
            self._last_routing = routing
            self._last_hard_slab = hard_slab
            self._last_soft_slab = soft_slab
            rules[self._binary_tier] = self._collect_binary_rule_selections(routing)
        else:
            self._last_hard_slab = x
            self._last_soft_slab = x

        return rules

    def generate(self, target, word_space, subspace=None):
        # Plan task 12 fills this in. Until then, return empty.
        return {}

    def _collect_unary_rule_selections(self, routing):
        """Per-batch list of unary op_ids that fired (apply, not copy)."""
        kind = routing["action_kind"]      # [B, N]; 0=copy, 2=apply
        op = routing["action_op"]          # [B, N]
        B = kind.shape[0]
        rows = []
        for b in range(B):
            row = [int(op[b, j].item())
                   for j in range(kind.shape[1])
                   if int(kind[b, j].item()) == 2]
            rows.append(row)
        return rows

    def _collect_binary_rule_selections(self, routing):
        kind = routing["action_kind"]      # [B, N]
        op = routing["action_op"]
        lengths = routing["lengths"]
        B = kind.shape[0]
        rows = []
        for b in range(B):
            row = []
            L = int(lengths[b].item())
            for j in range(L):
                if int(kind[b, j].item()) == 1:
                    row.append(int(op[b, j].item()))
            rows.append(row)
        return rows
```

- [ ] **Step 10.8: Run the XOR-grammar acceptance test**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_xor_grammar.py -v`
Expected: both tests PASS.

- [ ] **Step 10.9: Re-run the dispatch tests to confirm no regression on the binary-only path**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_wordspace.py basicmodel/test/test_signal_router_unary.py basicmodel/test/test_signal_router_xor_grammar.py -v`
Expected: every test in this batch PASSES.

- [ ] **Step 10.10: Pause for user review and commit**

Summarize: `UnaryStructuredLayer` (per-position softmax, no DP) plus two-tier stacking inside `SignalRouter` so a single `compose` runs NOT (unary) then AND/OR (binary). XOR-grammar acceptance test confirms per-tier rule selections and gradient reaching all three ops.

---

## Task 11: Routing regularization losses

Three losses, exposed as standalone functions consumers can sum into their training objective:

1. **Copy penalty** — discourages all-COPY collapse.
2. **Length penalty** — encourages reduction depth.
3. **Comparator-DP KL** — pulls comparator gates toward the soft-DP marginals when `lambda_dp_prior > 0`. Provides the "DP marginals as prior on the comparator" coupling discussed in Task 5.

**Files:**
- Modify: `basicmodel/bin/SignalRouter.py`
- Test: `basicmodel/test/test_signal_router_losses.py`

- [ ] **Step 11.1: Write failing loss tests**

```python
# basicmodel/test/test_signal_router_losses.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch

from SignalRouter import (
    copy_penalty, length_penalty, comparator_dp_kl,
)


def test_copy_penalty_zero_when_no_copies():
    routing = {"copy_marginal": torch.zeros(2, 5)}
    loss = copy_penalty([routing], lambda_copy=1.0)
    assert float(loss) == 0.0


def test_copy_penalty_grows_with_copy_mass():
    a = {"copy_marginal": torch.full((2, 5), 0.2)}
    b = {"copy_marginal": torch.full((2, 5), 0.9)}
    assert float(copy_penalty([b], 1.0)) > float(copy_penalty([a], 1.0))


def test_length_penalty_grows_with_lengths():
    a = {"lengths": torch.tensor([2, 3])}
    b = {"lengths": torch.tensor([5, 5])}
    assert float(length_penalty([b], 1.0)) > float(length_penalty([a], 1.0))


def test_comparator_dp_kl_zero_when_gates_match_marginals():
    B, N = 1, 4
    p_copy = torch.tensor([[0.4, 0.3, 0.5, 0.6]])
    p_reduce = torch.tensor([[0.3, 0.0, 0.0]])  # extended below
    # Build target distribution over 4 branches per position.
    cum = torch.cumsum(p_reduce, dim=1)
    cumshift = torch.cat([torch.zeros(B, 1), cum], dim=1)        # [B, N]
    keep = p_copy * (1.0 - cumshift.clamp(0.0, 1.0))
    reduce_w = torch.cat([p_reduce, torch.zeros(B, 1)], dim=1)
    shift = cumshift.clamp(0.0, 1.0)
    pad = (1.0 - keep - reduce_w - shift).clamp(min=1e-8)
    target = torch.stack([keep, reduce_w, shift, pad], dim=-1)
    target = target / target.sum(-1, keepdim=True)
    routing = {
        "gates": target.clone(),
        "copy_marginal": p_copy,
        "reduce_marginal": p_reduce,
    }
    kl = comparator_dp_kl([routing], lambda_dp_prior=1.0)
    assert float(kl) < 1e-4


def test_losses_are_differentiable():
    p_copy = torch.full((1, 4), 0.5, requires_grad=True)
    p_reduce = torch.full((1, 3), 0.25, requires_grad=True)
    gates = torch.full((1, 4, 4), 0.25, requires_grad=True)
    routing = {
        "copy_marginal": p_copy,
        "reduce_marginal": p_reduce,
        "gates": gates,
        "lengths": torch.tensor([3]),
    }
    loss = (copy_penalty([routing], 1e-3)
            + length_penalty([routing], 1e-4)
            + comparator_dp_kl([routing], 1e-2))
    loss.backward()
    assert p_copy.grad is not None and p_copy.grad.abs().sum() > 0
    assert gates.grad is not None and gates.grad.abs().sum() > 0
```

- [ ] **Step 11.2: Run to verify failure**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_losses.py -v`
Expected: all five tests FAIL.

- [ ] **Step 11.3: Implement the loss functions**

```python
# basicmodel/bin/SignalRouter.py — append

def copy_penalty(route_traces, lambda_copy: float = 1e-3):
    if lambda_copy == 0.0:
        return torch.tensor(0.0)
    total = 0.0
    seen = False
    for r in route_traces:
        if "copy_marginal" in r:
            total = total + r["copy_marginal"].mean()
            seen = True
    if not seen:
        return torch.tensor(0.0)
    return lambda_copy * total


def length_penalty(route_traces, lambda_len: float = 1e-4):
    if lambda_len == 0.0:
        return torch.tensor(0.0)
    total = 0.0
    seen = False
    for r in route_traces:
        if "lengths" in r:
            total = total + r["lengths"].float().mean()
            seen = True
    if not seen:
        return torch.tensor(0.0)
    return lambda_len * total


def comparator_dp_kl(route_traces, lambda_dp_prior: float = 0.0):
    """KL(comparator gates || target built from soft DP marginals).

    The target per output position j is the four-branch distribution
    that compact_soft uses internally:
        keep   = p_copy[j]   * (1 - cumshift[j])
        reduce = p_reduce[j] (extended with 0 at j=N-1)
        shift  = cumshift[j] (cumulative reduce mass strictly before j)
        pad    = remainder
    This term encodes "comparator gates should agree with the structured
    DP marginals to first order" without forcing equality.
    """
    if lambda_dp_prior == 0.0:
        return torch.tensor(0.0)
    total = 0.0
    seen = False
    for r in route_traces:
        gates = r.get("gates", None)
        p_copy = r.get("copy_marginal", None)
        p_reduce = r.get("reduce_marginal", None)
        if gates is None or p_copy is None or p_reduce is None:
            continue
        seen = True
        B, N, K = gates.shape
        if p_reduce.shape[1] == 0:
            cumshift = gates.new_zeros(B, N)
            reduce_w = gates.new_zeros(B, N)
        else:
            cum = torch.cumsum(p_reduce, dim=1)
            cumshift = torch.cat([gates.new_zeros(B, 1), cum], dim=1)
            reduce_w = torch.cat([p_reduce, gates.new_zeros(B, 1)], dim=1)
        keep = p_copy * (1.0 - cumshift.clamp(0.0, 1.0))
        shift = cumshift.clamp(0.0, 1.0)
        pad = (1.0 - keep - reduce_w - shift).clamp(min=1e-8)
        tgt = torch.stack([keep, reduce_w, shift, pad], dim=-1)
        tgt = tgt / tgt.sum(-1, keepdim=True).clamp(min=1e-8)
        log_gates = (gates.clamp(min=1e-8)).log()
        kl = (tgt * (tgt.clamp(min=1e-8).log() - log_gates)).sum(-1).mean()
        total = total + kl
    if not seen:
        return torch.tensor(0.0)
    return lambda_dp_prior * total
```

- [ ] **Step 11.4: Run the loss tests**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_losses.py -v`
Expected: all five tests PASS.

- [ ] **Step 11.5: Pause for user review and commit**

Summarize: three regularization losses (`copy_penalty`, `length_penalty`, `comparator_dp_kl`), each gated by its own lambda and each verified to be zero / non-zero / differentiable as appropriate.

---

## Task 12: Generate / inverse pass scaffolding

Mirror the chart's `generate` contract on the signal path. The inverse pass walks the soft slab + comparator gates from the most recent `compose` and emits a `generate_rules` dict. First-cut emission scheme: per output position, pick the highest-probability **non-pad** branch and translate its (kind, op_id) into a generate rule id. This is intentionally simple — a learned inverse expander is a follow-up plan.

**Files:**
- Modify: `basicmodel/bin/SignalRouter.py`
- Test: extends `basicmodel/test/test_signal_router_wordspace.py`

- [ ] **Step 12.1: Append failing generate test**

```python
# basicmodel/test/test_signal_router_wordspace.py — append

def test_signal_generate_emits_rules_dict_after_compose():
    chart = Chart(nInput=4, nOutput=4, max_depth=3, hidden_dim=16,
                  feature_dim=4, router_kind="signal")
    router = chart._ensure_signal_router()
    router.attach_layer_ops(ops=[_Stub()], r_copy=1, tier="C")
    ws = _StubWordSpace()
    target = torch.randn(2, 4, 4)
    chart.compose(target, word_space=ws)
    g = chart.generate(target, word_space=ws)
    assert isinstance(g, dict)
    assert "C" in g
    rows = g["C"]
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row, list)
        for rid in row:
            assert isinstance(rid, int)
            assert 0 <= rid < 1
```

- [ ] **Step 12.2: Run to verify failure**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_wordspace.py::test_signal_generate_emits_rules_dict_after_compose -v`
Expected: FAILS — `SignalRouter.generate` returns `{}` until this task.

- [ ] **Step 12.3: Implement generate**

Replace the stub `SignalRouter.generate` with:

```python
# basicmodel/bin/SignalRouter.py — modify SignalRouter.generate

    def generate(self, target, word_space, subspace=None):
        if self._binary_layer is None:
            raise RuntimeError(
                "SignalRouter.generate called before attach_layer_ops().")
        # Run an inside pass if compose hasn't been called or target differs.
        # First cut: assume compose was just called and reuse routing.
        routing = getattr(self, "_last_routing", None)
        if routing is None:
            self.compose(target, word_space, subspace=subspace)
            routing = self._last_routing

        # Pick the top non-pad branch per position; map to generate rule id.
        gates = routing["gates"]                 # [B, N, 4]  (keep, reduce, shift, pad)
        # Mask out the pad branch.
        masked = gates.clone()
        masked[..., 3] = -1.0
        top_branch = masked.argmax(dim=-1)       # [B, N]

        # action_op from the hard route names op_ids per slot of the
        # compacted (length-L) slab; use it as the op id source.
        action_kind = routing["action_kind"]     # [B, N]
        action_op = routing["action_op"]         # [B, N]
        lengths = routing["lengths"]
        B = gates.shape[0]
        rows = []
        for b in range(B):
            row = []
            L = int(lengths[b].item())
            for j in range(L):
                if int(action_kind[b, j].item()) == 1:
                    row.append(int(action_op[b, j].item()))
            rows.append(row[::-1])  # generation pops in reverse
        return {self._binary_tier: rows}
```

- [ ] **Step 12.4: Run the generate test**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_wordspace.py -v`
Expected: all WordSpace dispatch tests PASS.

- [ ] **Step 12.5: Pause for user review and commit**

Summarize: signal path now satisfies both `compose` and `generate` contracts — `generate_rules` is the reverse-order list of reductions from the last `compose`. A learned inverse expander remains a follow-up.

---

## Task 13: Full-suite regression sweep + plan handoff

Confirm the chart path is bit-identical for `routerKind="chart"` and that the signal path runs end-to-end on the small fixture without disturbing existing tests. Capture the open questions for the follow-up plan.

**Files:**
- Modify: `basicmodel/doc/plans/2026-05-02-signal-router.md` (this file — append the open-questions section).

- [ ] **Step 13.1: Run the full pytest suite**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/ -x -q`
Expected: same pre-Task-0 baseline pass/fail set. Any regression must trace to this plan; fix before proceeding.

- [ ] **Step 13.2: Run only the signal-router tests as a sanity batch**

Run: `basicmodel/.venv/bin/python -m pytest basicmodel/test/test_signal_router_*.py -v`
Expected: all signal-router tests PASS.

- [ ] **Step 13.3: Append open-questions section to this plan**

Add the following to the bottom of this plan document so the follow-up plan(s) can pick up cleanly.

```markdown
## Open Questions / Follow-up Plans

These were called out as out-of-scope during planning but should be the basis
for follow-up plans before the signal router replaces the chart in
production.

### Multi-space parallel routers
The first cut runs one unary tier (NOT) + one binary tier (AND/OR) inside
a single SignalRouter (Tasks 9, 10). Production grammars run separate
routers in parallel inside PerceptualSpace (P), ConceptualSpace (C), and
SymbolicSpace (S), each owning its own grammar and its own router state.
Decide:
- One SignalRouter per Space, parallel — clean isolation, redundant DP work.
- One SignalRouter shared, with a space-axis embedding mixed into the scorer
  input — single DP, single set of params, cross-space leakage risk.

### Tensorized rule bank
Op dispatch is currently a `nn.ModuleList[BinaryOp]` and `op_id` indexes it.
For full GPU efficiency the rule weights should be a single stacked tensor,
with the chosen `op_id` selecting via gather rather than Python dispatch.
This is the "rule-axis tensor" piece.

### Learned inverse expander
generate() emits the reverse of the last compose route. A learned expander
that consumes the soft slab and predicts where to insert reductions is the
proper inverse-pass model.

### Replacement of compact_hard's Python loop
Current `compact_hard` is a `for b in range(B):` loop over batches and a
while loop over positions. Acceptable for correctness; for production
training, replace with prefix-sum + scatter so the operation is one fused
GPU kernel.

### Soft-vs-hard slab routing for downstream layers
Right now `BinaryStructuredReductionLayer` returns both `hard_slab` and
`soft_slab`. Once integrated into a stack, the calling code has to pick
which slab feeds the next layer's op call (clean operands → hard) vs.
which feeds the routing scorer + inverse pass (gradient → soft).
Document the convention as part of the stacking plan.
```

- [ ] **Step 13.4: Pause for user review and commit**

Summarize: full test suite green on chart and signal paths, open questions documented for the follow-up plans (multi-tier stacking, tensorized rule bank, learned inverse expander, batched compaction kernel, slab routing convention).

---

## Open Questions / Follow-up Plans

These were called out as out-of-scope during planning but should be the basis
for follow-up plans before the signal router replaces the chart in
production.

### Multi-space parallel routers
The first cut runs one unary tier (NOT) + one binary tier (AND/OR) inside
a single SignalRouter (Tasks 9, 10). Production grammars run separate
routers in parallel inside PerceptualSpace (P), ConceptualSpace (C), and
SymbolicSpace (S), each owning its own grammar and its own router state.
Decide:
- One SignalRouter per Space, parallel — clean isolation, redundant DP work.
- One SignalRouter shared, with a space-axis embedding mixed into the scorer
  input — single DP, single set of params, cross-space leakage risk.

### Tensorized rule bank
Op dispatch is currently a `nn.ModuleList[BinaryOp]` and `op_id` indexes it.
For full GPU efficiency the rule weights should be a single stacked tensor,
with the chosen `op_id` selecting via gather rather than Python dispatch.
This is the "rule-axis tensor" piece.

### Learned inverse expander
generate() emits the reverse of the last compose route. A learned expander
that consumes the soft slab and predicts where to insert reductions is the
proper inverse-pass model.

### Replacement of compact_hard's Python loop
Current `compact_hard` is a `for b in range(B):` loop over batches and a
while loop over positions. Acceptable for correctness; for production
training, replace with prefix-sum + scatter so the operation is one fused
GPU kernel.

### Soft-vs-hard slab routing for downstream layers
Right now `BinaryStructuredReductionLayer` returns both `hard_slab` and
`soft_slab`. Once integrated into a stack, the calling code has to pick
which slab feeds the next layer's op call (clean operands → hard) vs.
which feeds the routing scorer + inverse pass (gradient → soft).
Document the convention as part of the stacking plan.

### Real GRAMMAR_LAYER_CLASSES integration
Task 10's XOR acceptance test uses minimal float-tensor proxies (`_AndOp`,
`_OrOp`, `_NotOp`). A follow-up should wire the actual `IntersectionLayer`,
`UnionLayer`, `NotLayer` from `basicmodel/bin/Layers.py:GRAMMAR_LAYER_CLASSES`
into the SignalRouter and confirm the existing parametrized folds work
inside the new dispatch.
