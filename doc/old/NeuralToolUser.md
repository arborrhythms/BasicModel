# NeuralToolUser Plan

Date: 2026-06-12

> **SUPERSEDED (2026-06-14): the training route is now per-layer
> soft-superposition, not the hard executor + REINFORCE below.** After the
> executor landed, the design was simplified to its essence: drop the
> Viterbi hard route and the straight-through from the forward *value*, and
> run the structured layers as a **pure sum-product superposition at a
> temperature** $t \in [0,1]$ ($0$ = sharp/deterministic, $1$ = flat). The
> chooser is then in the gradient path **directly** — no argmax, no
> `.detach()`, no policy log-prob, no advantage. Training is still a
> two-pass over each sentence, but as two ordinary trials: pass A at $t=0$
> (recorded), pass B at `<exploreTemperature>` (exploration, trimmed from
> the batch error, no batch-count increment) — temperature sampling to
> escape a sharp local commitment, **not** `loss_A + loss_B`. A hard tree is
> recovered on demand by pinning $t=0$ and reading the argmax routing trace.
> The live design and contract are in
> [doc/Language.md](../Language.md) → "Soft-superposition route (the
> `<learning>` two-pass)". Code: `superposition_scale` +
> `superposition_temperature` on `BinaryStructuredReductionLayer` /
> `UnaryStructuredLayer`, `BasicModel._set_superposition_temperature`, the
> `runBatch(superposition_temperature=…)` arg, and the `runEpoch` two-pass.
> Tests: `test_soft_superposition.py`, `test_two_pass_driver.py`.
>
> Removed with the pivot: `neural_tool_user_two_pass`, `_output_task_loss`,
> `_neural_tool_user_policy_term` (Models.py) and `perturb_route_scores`
> (Language.py). The hard-executor machinery described below
> (`NeuralToolUser` / `parse_greedy` / `parse_explore` / `two_pass_loss` /
> `cross_product_*` / `_BinaryStepperAdapter` / `_ntu_*` route store / the
> `<neuralToolUser>` compose branch) still exists but is **legacy / off the
> live path**, pending a removal decision. `MLPTransformChooser`, the
> `TransformChooser` abstraction, and the three cognitive operations are
> unaffected — they sit under the soft-superposition route just as they did
> under the executor. Everything below this note is retained as the design
> history that led here.
>
> **Status (2026-06-13): migration steps 1–3 landed (behavior-preserving).**
> `TransformChooser` (base) and `AnchorDotTransformChooser` are in
> `bin/Language.py`; `UnaryStructuredLayer` and
> `BinaryStructuredReductionLayer` delegate their placement scoring to the
> chooser. The anchor-dot chooser reproduces the original inline einsums
> exactly (parity pinned by `test/test_transform_chooser_parity.py`).
>
> Design choice for basin safety: the default chooser is **stateless** —
> the `copy_anchor` / `apply_anchor` / `reduce_anchor` Parameters stay
> owned by the layers and are passed in at call time, so the state_dict
> keys are unchanged and the scoring is byte-identical (the full suite is
> green). The future `MLPTransformChooser` owns its own params — a
> deliberate new-params cutover behind a config flag, which starts a fresh
> basin anyway.
>
> **Status (2026-06-13): the executor + training engine landed (off the
> live path).** Per the user-refined design (settled in conversation):
> - `cross_product_action_dist` / `cross_product_route_logprob` — the
>   per-level distribution is normalized JOINTLY over the (op × location)
>   cross-product; a route's log-prob/entropy are read from it. The route
>   stays the valid `binary_tiling_viterbi` tiling (already fires several
>   compatible non-overlapping reduces per level → "simultaneous ops").
> - `NeuralToolUser` — pass 1 greedy (Viterbi, saves route+scores); pass 2
>   replays to a random divergence level L and forces a DIFFERENT op at one
>   fired location (op excluded + renormalized → divergence guaranteed when
>   ≥2 ops are legal; temperature 0..1 shapes which alternative). Skips
>   levels with <2 legal ops.
> - `two_pass_loss` — single-point advantage at L:
>   `loss = mean(loss_A+loss_B) + λ·mean(adv_A·logp_a + adv_B·logp_b) − λ_e·entropy@L`,
>   both log-probs off ONE live dist at L. Verified: `d(loss)/d(logp_b) =
>   adv_b` (below-baseline route's op pushed up). `parallel_prime` = the
>   existing §5 tower-attention priming (no new tensor).
> All tested in isolation (`test_cross_product_route.py`,
> `test_neural_tool_user.py`).
>
> **Live cutover LANDED (config-gated, default off).** New
> `<architecture><neuralToolUser>` boolean (model.xsd). When true,
> `LanguageLayer.compose`'s binary stage runs the executor instead of the
> soft-DP fold loop: `_BinaryStepperAdapter` bridges the real
> `BinaryStructuredReductionLayer` (reusing its `_stacked_reduced` /
> `chooser` / `_selected_reduced` / `compact_hard`), `parse_greedy` folds
> the slab, and the route + per-level cross-product distributions are saved
> on the router's route store (`_ntu_route` / `_ntu_dist`); fired rule-ids
> come from the route. In NTU mode `rule_probs` is the detached route
> summary (the hard scatter — the chooser is trained by the two-pass policy
> loss at the model/train level, not via rule_probs). Flag off → the
> soft-DP path is byte-identical (state_dict / basin unchanged). Tests:
> `test_neural_tool_user_cutover.py`.
>
> **The three cognitive operations LANDED (config-gated / dark by default;
> Architecture.md).** (1) granularity = the folds themselves; (2)
> subsymbolic order = the CS→PS loop, with **`<symbolicComposition>`**
> (default off) re-feeding the prior pass's symbolic carrier
> (`cs._subspaceForSS`) to PartSpace at t>0 so Sigma composes higher-order
> symbols ([Models.py](../bin/Models.py) per-stage body; reuses the
> wide↔deep regroup), and the **analysis-side top-k attention gate**
> (`WholeSpace._topk_priming_mask`) keeping the top-k analysed positions by
> the priming over the codes — dark unless an intent is set; (3) symbolic
> order = the serial grammatical loop. Tests: `test_symbolic_composition.py`,
> `test_analysis_attention_gate.py`. Flag-off / intent-off is byte-identical.
>
> **`MLPTransformChooser` LANDED (config-gated, default anchordot).** The
> contextual cutover chooser: a learned MLP over per-candidate context
> (slot/pair state, candidate op output, learned tool embedding, sinusoidal
> position) producing the same per-(op, location) logit shapes as the
> anchor-dot scorer, so it drops into the layers, `cross_product_action_dist`
> and the executor. Selected by `<architecture><transformChooser>mlp` (set
> on the router before the structured layers are built, via the
> `make_transform_chooser` factory). It OWNS params (tool embeddings + MLP),
> so enabling it changes the state_dict and starts a fresh basin — a
> deliberate cutover; default `anchordot` is byte-identical (no new params).
> Tests: `test_mlp_transform_chooser.py` (unit, factory, layer selection,
> end-to-end config→router).
>
> **Two-pass training driver LANDED.** _(SUPERSEDED — this REINFORCE driver
> and `neural_tool_user_two_pass` / `_neural_tool_user_policy_term` were
> removed; see the top note. The live two-pass is soft-superposition, gated
> on `<learning>` alone, no policy term.)_ Without it the chooser never learns
> — the live cutover runs only the greedy pass.
> `BasicModel.neural_tool_user_two_pass(x, task_loss_fn, ...)` runs the
> forward TWICE: pass A greedy (no temperature) and pass B explore
> (temperature). Pass B is enabled by a transient `_ntu_explore` toggle on
> the router — `compose` then replays pass-A's saved route
> (`_ntu_route[tier]`) and diverges via `parse_explore`, stashing
> `logp_a`/`logp_b`/`entropy@L`. The driver assembles
> `loss = loss_A + loss_B + λ·(adv_A·logp_a + adv_B·logp_b) − λ_e·entropy`;
> the advantages are detached (so only the two task-loss *scalars* matter
> for credit assignment) and the chooser gradient rides the live
> log-probs. Proven: the policy log-prob backprops to the chooser params
> (anchors and MLP) and the advantage moves `logp` the right way
> (`test_neural_tool_user_cutover.py`, `test_neural_tool_user.py`).
>
> **Wired into training behind `<learning>` (default off).** _(SUPERSEDED —
> the `<learning>` + `<neuralToolUser>` coupling and the policy term below
> were dropped; the live two-pass is gated on `<learning>` ALONE and runs two
> soft-superposition trials via `runEpoch` — see the top note.)_ No callback /
> no `runBatch` restructure: when `<architecture><learning>true` and
> `<neuralToolUser>` is on, each TRAINING batch runs the parser a SECOND
> time with `<exploreTemperature>` (default 0.5) and adds the policy term.
> Pass A is `runBatch`'s normal greedy forward (its `lossOut` is the task
> signal); `_neural_tool_user_policy_term` runs the explore forward,
> computes pass B's output loss, and returns
> `adv_A·logp_a + adv_B·logp_b` to add to the batch loss. Default off → the
> single forward (byte-identical). `BasicModel.neural_tool_user_two_pass`
> remains as the standalone callback-driven API. Tests:
> `test_two_pass_driver.py` (flag parse, runBatch step runs).
>
> **Remaining (genuinely future):** the soft-codebook option for
> codebook-bearing tools (softmax-over-similarity, anneal to hard).

## Summary

`NeuralToolUser` is a proposal to replace the current soft-superposed
grammar router with a hard-routing neural tool user over STM.

The model keeps `GrammarLayer` classes as the transform/tool
implementations:

```text
NotLayer, UnionLayer, IntersectionLayer, LiftLayer, LowerLayer, QueryLayer, ...
```

The new component is a `TransformChooser`: a neural controller that chooses
which operation to call and where to call it.

```text
[F_i, where] = TransformChooser(context)
output       = F_i(STM, where)
```

The design goal is to support variable-length hard reductions without needing
to keep all derivations in a shared soft tensor shape.

## Motivation

The current signal router has a useful differentiable surrogate:

```text
copy/reduce scores -> soft DP marginals + hard Viterbi route
```

This works while reductions can be represented as masked or padded slabs.
However, a general neural tool user should be able to run hard parses whose
intermediate structures diverge. Once a reduction changes the live length or
tree shape, soft superposition becomes increasingly artificial.

The proposed replacement is:

1. Run one or more hard parses.
2. Train every executed transform by its ordinary differentiable task loss.
3. Train the chooser from a route-level policy signal, without storing full
   routes.

This gives the parser a path toward arbitrary tools, stochastic exploration,
and richer context-conditioned routing.

## Current Compatibility Target

The current chooser behavior is anchor-dot scoring inside:

```text
UnaryStructuredLayer
BinaryStructuredReductionLayer
```

Conceptually:

```text
candidate = tool(slot_or_pair_state)
logit     = dot(candidate, learned_anchor_for_tool)
```

This must be preserved first. The first `TransformChooser` implementation
should reproduce the current anchor-dot behavior:

```python
class AnchorDotTransformChooser(nn.Module):
    def score_unary(self, x, applied, *, context):
        # Existing behavior:
        # copy_score  = <x, copy_anchor>
        # apply_score = <applied_tool_output, apply_anchor>
        ...

    def score_binary(self, x, reduced, *, context):
        # Existing behavior:
        # copy_score   = <x, copy_anchor>
        # reduce_score = <reduced_tool_output, reduce_anchor>
        ...
```

This makes the refactor behavior-preserving before introducing a more
expressive chooser.

## Proposed Architecture

Keep the transform/tool implementation separate from the choice policy:

```text
GrammarLayer        = transform/tool implementation
TransformChooser    = scores and samples tool/location candidates
NeuralToolUser      = executes hard parse loops using a chooser and tools
```

`GrammarLayer` remains responsible for tensor semantics:

```python
candidate = tool(left, right)
candidate = tool(slot)
```

`TransformChooser` is responsible for routing:

```python
logits = chooser(context)
action = sample_or_argmax(logits, legal_mask)
```

The action contains:

```python
{
    "kind": "copy" | "unary" | "binary" | "stop",
    "tool_id": int | None,
    "where": int | tuple[int, int],
}
```

## MLP TransformChooser

The intended cutover chooser is an MLP or small contextual network:

```text
logit(tool, where) =
    MLP(
        slot_or_pair_state,
        tool_embedding,
        position_or_span_embedding,
        parse_depth,
        parallel_prime,
        route_history
    )
```

Inputs:

- `slot_or_pair_state`: the STM slot or adjacent pair being considered.
- `tool_embedding`: learned identity vector for the candidate
  `GrammarLayer` rule/operator.
- `position_or_span_embedding`: where the action would apply.
- `parse_depth`: current reduction depth or loop step.
- `parallel_prime`: optional priming state from the parallel pass.
- `route_history`: compact summary of prior choices, not a full route log.
- `legal_mask`: grammar legality and arity constraints.
- `noninterference_mask`: optional constraint preventing overlapping writes.

The chooser returns:

```python
ChoiceResult(
    action=action,
    log_prob=log_prob,
    entropy=entropy,
    logits=logits,
)
```

Only `log_prob` and `entropy` need to be accumulated during execution. Full
routes do not need to be stored.

## Two-Hard-Parse Training

For each input, run two hard parses:

```text
route A: suggested / lower-temperature chooser
route B: perturbed / higher-temperature chooser
```

Both routes execute normally:

```python
out_A, stats_A = neural_tool_user.parse(input, mode="suggested")
out_B, stats_B = neural_tool_user.parse(input, mode="explore")

loss_A = task_loss(out_A)
loss_B = task_loss(out_B)
```

Each parse accumulates:

```python
stats.log_prob_sum
stats.entropy_sum
stats.step_count
```

No full route storage is required.

## Training Objective

Train the executed computations with ordinary differentiable loss:

```python
loss_exec = loss_A + loss_B
```

This trains:

```text
GrammarLayer parameters
STM representations
state encoders
downstream reconstruction / prediction layers
```

Train the chooser with a route-level policy signal:

```python
baseline = 0.5 * (loss_A.detach() + loss_B.detach())

adv_A = loss_A.detach() - baseline
adv_B = loss_B.detach() - baseline

loss_choose = (
    adv_A * stats_A.log_prob_sum
  + adv_B * stats_B.log_prob_sum
)

loss_entropy = -(
    stats_A.entropy_sum + stats_B.entropy_sum
)

loss = (
    loss_exec
  + lambda_choose * loss_choose
  + lambda_entropy * loss_entropy
)
```

Because training minimizes `loss`, a route with lower-than-baseline loss gets
its log-probability increased. A route with higher-than-baseline loss gets its
log-probability decreased.

This is the central credit assignment rule:

```text
The execution loss trains the tools.
The route-level log-prob loss trains the chooser.
```

## Perturbation Strategy

Do not initially force route B to differ from route A.

Instead, make difference likely through:

- higher sampling temperature,
- action-space noise,
- dropout in the chooser,
- ergodic perturbation in chooser layers,
- exploration entropy.

If the two routes are identical, the example still trains the executed
computation. The chooser receives little comparison signal, which is acceptable.
If identical routes are too common, increase exploration.

## ErgodicLayer Integration

`ErgodicLayer` can be used inside the chooser MLP to inject adaptive parameter
noise:

```text
W_eff = bias * W + var * noise
```

This is useful for exploration, but it does not by itself solve discrete
choice credit assignment. The chooser still needs the accumulated
`log_prob_sum` policy loss above.

Recommended initial use:

```text
suggested route: low or no ergodic noise
explore route:   higher temperature and/or ergodic noise
```

## Legal and Noninterfering Actions

The chooser should not need to learn basic legality from scratch.

At each step, construct masks for:

- arity: unary vs binary vs stop,
- live STM positions,
- grammar tier/category legality,
- noninterference between simultaneous actions,
- maximum parse depth or step budget.

For the first implementation, choose one action per step. Later, allow top-k
noninterfering actions per step.

Top-k hard selection is still a discrete operation. If it is trained by the
same policy-loss mechanism, the chooser only needs the summed log-probability
of the selected actions.

## Codebook and Non-Smooth Tools

Most `GrammarLayer` math is differentiable enough for execution training:

- `lift` / `lower` use Pi/Sigma-style transforms.
- non-monotonic `union` / `intersection` use smooth RadMax/RadMin kernels.
- monotonic min/max variants are differentiable almost everywhere but can
  route sparse gradients to winning operands.

Codebook selection is different. Nearest-row quantization uses hard
`argmin`/`argmax` with straight-through or selected-row gradient. In those
paths:

```text
the selected code rows can train;
the selection boundary is not ordinary backprop.
```

For fully neural tool choice over codebook-bearing transforms, add an optional
soft codebook mode:

```python
soft_code = softmax(sim(x, codebook) / tau) @ codebook
```

Then anneal to hard or use hard-forward / soft-backward where needed.

## Migration Plan

1. Introduce a `TransformChooser` interface.
2. Implement `AnchorDotTransformChooser` to reproduce the current scorer.
3. Refactor `UnaryStructuredLayer` and `BinaryStructuredReductionLayer` so
   they call the chooser for logits instead of computing logits inline.
4. Add a hard parse executor that accumulates:

   ```text
   log_prob_sum
   entropy_sum
   step_count
   ```

5. Add two-hard-parse training mode:

   ```text
   loss_exec + lambda_choose * loss_choose + lambda_entropy * loss_entropy
   ```

6. Add `MLPTransformChooser` with contextual features:

   ```text
   slot_or_pair_state
   tool_embedding
   position/span_embedding
   parse_depth
   parallel_prime
   route_history
   ```

7. Keep anchor-dot chooser as the default until parity tests pass.
8. Cut over configs to `MLPTransformChooser` behind an XML/config flag.

## Minimal Interfaces

Candidate context:

```python
@dataclass
class ToolChoiceContext:
    stm: torch.Tensor
    slot_or_pair_state: torch.Tensor
    candidate_outputs: torch.Tensor | None
    tool_ids: torch.Tensor
    tool_embeddings: torch.Tensor
    positions: torch.Tensor
    parse_depth: torch.Tensor
    parallel_prime: torch.Tensor | None
    route_history: torch.Tensor | None
    legal_mask: torch.Tensor
```

Chooser result:

```python
@dataclass
class ChoiceResult:
    action_kind: torch.Tensor
    action_tool: torch.Tensor
    action_where: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    logits: torch.Tensor
```

Route stats:

```python
@dataclass
class RouteStats:
    log_prob_sum: torch.Tensor
    entropy_sum: torch.Tensor
    step_count: torch.Tensor
```

## Tests

Compatibility tests:

- Anchor-dot chooser produces the same logits as current unary scoring.
- Anchor-dot chooser produces the same logits as current binary scoring.
- Existing `rule_probs` gradient-to-router tests still pass in compatibility
  mode.

Hard-route tests:

- A hard parse executes without soft DP.
- Two hard parses train both execution losses.
- `log_prob_sum` receives gradient into chooser parameters.
- Lower-loss route increases probability relative to the two-route baseline.
- Identical routes do not crash and still train execution loss.

Codebook tests:

- Hard codebook paths still train selected rows only.
- Optional soft codebook path sends gradient to multiple rows.

## Open Questions

- What is the first compact `route_history` representation?
- Should `parallel_prime` be a direct tensor, a projection, or a cached
  summary?
- Should route B use temperature noise, ergodic parameter noise, or both?
- Should top-k noninterfering actions wait until one-action-per-step parity is
  stable?
- Should `rule_probs` remain as a downstream summary, or should the
  `RouteStats`/chooser state replace it for intra-sentence prediction?

