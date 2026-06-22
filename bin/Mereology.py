"""Mereology mixin -- the family of contemplative-awareness measures.

Houses the back-projection machinery (`hoc_shape`, `_walk_reverse`,
`_derivation_path`, `_leaf_path_trust`, etc.) and the five scalar
measures that share it:

* `Contiguous()` -- one-pointedness via `Ops.corner_overlap`.
* `Continuous()` -- empirical ε-δ continuity via `Ops.epsilon_delta`.
* `Peaceful()`   -- TruthLayer-luminosity uniformity (placeholder).
* `Area()`       -- sum of leaf hyperrectangle volumes.
* `Luminosity()` -- totalArea − pairwise(overlap × DoT_disagreement).

The mixin is pure (no `__init__`, no state of its own); it accesses
model-owned attributes (`self.wholeSpace`, `self.conceptualSpace`,
`self.symbolSpace`, `self.subsymbolicOrder`) via ``self``.  Mix in by
inheriting *first*: ``class BaseModel(Mereology, nn.Module): ...``.

See ``doc/research/three-surfaces.md`` and the
``2026-05-04-introspective-subsymbolic-handoff`` plan family for the
geometric foundations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from Layers import Ops, GRAMMAR_LAYER_CLASSES, CONTIGUITY_PRESERVING_OPS
from util import TheXMLConfig


# ---------------------------------------------------------------------------
# Higher-order-concept shape descriptors -- consumed by hoc_shape /
# Contiguous / Continuous (see basicmodel/doc/research/three-surfaces.md
# and the 2026-05-04 spec at plans/.../sparkling-peach.md). These are
# pure data records; the math lives on Mereology.hoc_shape.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleSpec:
    """One node in the derivation path: rule_name + arity + space_role.

    Immutable record consumed by ``Mereology.hoc_shape`` when walking
    the reverse derivation. Space-role is one of 'subsymbolic' / 'CS' /
    'SS'; arity is 1 for unary ops and 2 for binary fold operators.
    """
    rule_name: str
    arity:     int
    space_role:      str


@dataclass(frozen=True)
class StepInfo:
    """One layer-level reverse executed during the hoc_shape walk.

    Captures the rule that fired (rule_name, arity, space_role), whether
    its reverse preserves contiguity (per
    ``Layers.CONTIGUITY_PRESERVING_OPS``), the binary-fanout branch
    (``''`` for unary, ``'left'`` / ``'right'`` for binary), and the
    per-layer top-K active-neuron cap that bounds downstream
    contiguity computation.
    """
    rule_name:      str
    arity:          int
    space_role:           str
    contiguous:     bool
    branch:         str
    active_indices: 'torch.Tensor'   # [B, K_cap] long
    active_count:   'torch.Tensor'   # [B] long
    K_cap:          int


@dataclass(frozen=True)
class HoCShape:
    """Result of ``Mereology.hoc_shape``.

    leaves -- one ``[B, V, D_C1]`` tensor per leaf in the derivation
              tree. List length is 1 for default-only / unary-only
              paths; doubles for each binary op along the path.
    mask   -- ``[B, V]`` bool. True where the input symbolic vector
              had a nonzero per-position norm (Mereology applies no
              magnitude threshold; truth recording/acceptance is
              governed by the continuous ``truthCriterion`` bar
              elsewhere, and the old ``truthMinMagnitude`` knob is
              retired).
    per_step -- list of StepInfo, DFS pre-order, one entry per
              layer-level reverse executed.
    """
    leaves:    list
    mask:      'torch.Tensor'
    per_step:  list


class Mereology:
    """Contemplative-awareness measure family + back-projection
    machinery, factored out of `BaseModel` so the spatial / mereological
    primitives live in one place.

    Pure mixin: no `__init__`, no state.  Mix into a concrete model
    by inheriting *first*: ``class BaseModel(Mereology, nn.Module):``.

    Required host attributes (all set by ``BaseModel.__init__``):
      * ``self.wholeSpace``  -- carrier of the higher-order
        bivector activation that drives every measure.
      * ``self.conceptualSpace`` (and optionally ``self.wholeSpaces`` /
        ``self.conceptualSpaces`` for staged models) -- back-projection
        targets.
      * ``self.symbolSpace``      -- chart / grammar host for
        ``host_layer`` lookups in ``_lookup_host_layer``.
      * ``self.subsymbolicOrder`` -- number of stages for the default
        derivation path.
    """

    # -- Contemplative Awareness Characterizations ---------------------

    def Contiguous(self) -> float:
        """Continuous mereological-contiguity measure in ``[-1, +1]``.

        One-Pointedness / Shamatha / Focused Attention. The forward
        pass produces higher-order symbols at WholeSpace; this
        method back-projects each active higher-order symbol through
        every layer of the derivation (via ``hoc_shape``) to expose
        the C(1) constituent regions, then runs pairwise
        ``Ops.corner_overlap`` over trustworthy leaves to score
        whether the constituents share at least one corner along
        any dimension.

        Return value:
          ``+1.0`` -- every (trustworthy) leaf pair overlaps; the
                     higher-order concept is fully contiguous.
          ``-1.0`` -- no trustworthy leaf pair overlaps; the
                     constituents are mereologically disjoint.
          ``0.0``  -- unknown: no trustworthy leaves to decide on
                     (every back-projection traversed at least one
                     lossy op like intersection / union /
                     conjunction / disjunction; the pseudo-inverse
                     erases the geometry needed for the test).
          intermediate -- average pairwise measure, weighted by
                          per-pair trust.

        A single trustworthy leaf returns ``+1.0`` (trivially one-
        pointed); a single untrustworthy leaf returns ``0.0``.
        """
        sym = getattr(self, 'wholeSpace', None)
        if sym is None or not hasattr(sym, 'subspace'):
            return 0.0
        try:
            sym_act = sym.subspace.materialize()
        except Exception:
            return 0.0
        if sym_act is None or not torch.is_tensor(sym_act) or sym_act.numel() == 0:
            return 0.0

        shape = self.hoc_shape(sym_act)
        leaves = shape.leaves
        if not leaves:
            return 0.0

        leaf_trust = self._leaf_path_trust(shape)
        if not leaf_trust:
            return 0.0

        if len(leaves) == 1:
            return 1.0 if leaf_trust[0] else 0.0

        # Pairwise corner-overlap over the .what bivector ([..., :2]).
        total_weight = 0.0
        weighted_sum = 0.0
        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                if not (leaf_trust[i] and leaf_trust[j]):
                    continue
                a = leaves[i][..., :2]
                b = leaves[j][..., :2]
                if not (torch.is_tensor(a) and torch.is_tensor(b)):
                    continue
                if a.shape[-1] != 2 or b.shape[-1] != 2:
                    continue
                m = Ops.corner_overlap(a, b)
                # Aggregate over the [B, V, ...] leading dims.
                # Mean over only the masked-active positions where
                # both leaves had extent; positions where corner_
                # overlap returned 0 (degenerate) contribute 0,
                # which already biases the average toward neutral.
                weighted_sum += float(m.mean().item())
                total_weight += 1.0

        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight

    def _leaf_path_trust(self, shape) -> list:
        """For each leaf in ``shape.leaves``, AND-fold the
        ``contiguous`` flag along the steps that produced it.

        ``shape.per_step`` is a flat DFS pre-order list. For binary
        nodes, the spec emits the per-branch StepInfo immediately
        before that branch's subtree steps. We traverse the flat
        list and reconstruct per-leaf trust by emitting one bool
        per leaf in the same order ``_walk_reverse`` produced them.

        For default-only / unary-only paths (``len(leaves) == 1``),
        this collapses to ``all(s.contiguous for s in per_step)``.
        """
        leaves = shape.leaves
        per_step = shape.per_step
        if not leaves:
            return []
        if len(leaves) == 1:
            return [all(bool(s.contiguous) for s in per_step)]

        # Walk per_step entries; for each, push the cumulative AND
        # onto the stack for the duration of its subtree. When we
        # complete a leaf-equivalent branch, emit.
        if not per_step:
            return [True] * len(leaves)

        idx = [0]
        out = []

        def rebuild(prev_contig: bool):
            """Mirror _walk_reverse's recursion shape using per_step.

            Walks the DFS pre-order ``per_step`` list using the shared
            ``idx`` cursor and emits one boolean per reconstructed
            leaf into ``out``. Maintains a cumulative-AND of contiguity
            flags down the path so the emitted leaf flag matches
            "every step on the path preserved contiguity".
            """
            if idx[0] >= len(per_step):
                # No more steps: this is a leaf.
                out.append(prev_contig)
                return
            step = per_step[idx[0]]
            cumulative = prev_contig and bool(step.contiguous)
            idx[0] += 1
            if int(step.arity) == 1 and step.branch == '':
                # Unary: descend into the single subtree.
                rebuild(cumulative)
            elif int(step.arity) == 2:
                # Binary: this StepInfo is the LEFT branch; the
                # next StepInfo at the same depth (after the left
                # subtree) is the RIGHT branch.
                rebuild(cumulative)
                if idx[0] < len(per_step):
                    right_step = per_step[idx[0]]
                    if int(right_step.arity) == 2 and right_step.branch == 'right':
                        cumulative_r = prev_contig and bool(right_step.contiguous)
                        idx[0] += 1
                        rebuild(cumulative_r)
                    else:
                        # Mismatched tree -- emit a degenerate leaf.
                        out.append(cumulative)
            else:
                # Unknown arity; treat as unary.
                rebuild(cumulative)

        try:
            rebuild(True)
        except Exception:
            return [True] * len(leaves)

        # Pad / truncate to match leaves length.
        if len(out) < len(leaves):
            out.extend([True] * (len(leaves) - len(out)))
        return out[:len(leaves)]

    def hoc_shape(self, symbolic_vector, max_active_per_layer=None):
        """Reverse-convolve a higher-order symbol tensor through its
        derivation tree; return the C(1) constituent bivectors as a
        list of ``[B, V, D_C1]`` tensors (one per leaf of the
        derivation tree) plus the per-step contiguity flags and per-
        step top-K active-neuron caps.

        The walk is fully vectorized over the ``[B, V]`` leading
        dims -- every active position is reverse-projected in
        parallel through the same sequence of layer reverses.

        Forward composition stacks layer by layer; each reverse step
        is a "convolution" of the parent's ``[pos, neg]`` bivector
        shape with the layer's reverse kernel. Unary ops fold one
        parent into one operand; binary ops fan out one parent into
        two (one for ``left``, one for ``right``). Per-step
        contiguity (the bool flag on each ``StepInfo``) tracks
        whether the layer's reverse preserves connectedness -- only
        ``pi`` / ``sigma`` / ``lift`` / ``lower`` / ``not`` do; all
        others use a lossy pseudo-inverse.

        Args:
            symbolic_vector: ``[B, V, D]`` activation tensor at the
                highest conceptual order. Typically obtained via
                ``self.wholeSpace.subspace.materialize()`` so the
                ``.active`` mask is already applied.
            max_active_per_layer: optional int. Caps how many top-K
                positions per layer carry into
                ``per_step.active_indices``. Falls back to
                ``architecture.maxActivePerLayer`` config (default 8).

        Returns:
            HoCShape -- a small dataclass with:
              * ``leaves``: list of ``[B, V, D_C1]`` tensors. Length
                ``1`` for default-only / no-binary-op runs;
                ``2^k_binary`` for k_binary binary ops along the path.
              * ``mask``: ``[B, V]`` bool -- True where the input
                position had nonzero norm (Mereology applies no
                magnitude threshold; the old ``truthMinMagnitude`` knob
                is retired -- truth recording/acceptance is governed by
                the continuous ``truthCriterion`` bar elsewhere). Leaves
                carry full ``[B, V, ...]`` shape
                regardless; consumers AND-fold via the mask.
              * ``per_step``: list of ``StepInfo``, one per layer-
                level reverse executed during the walk, in
                outer-to-inner traversal order. Each StepInfo
                records ``(rule_name, arity, space_role, contiguous,
                branch, active_indices, active_count, K_cap)``.

            Empty leaves list when no position is active.
        """
        if symbolic_vector is None or not torch.is_tensor(symbolic_vector):
            empty_mask = torch.zeros(0, dtype=torch.bool)
            return HoCShape(leaves=[], mask=empty_mask, per_step=[])
        if symbolic_vector.numel() == 0 or symbolic_vector.dim() < 2:
            empty_mask = torch.zeros(symbolic_vector.shape[:-1] if symbolic_vector.dim() >= 1 else (0,),
                                     dtype=torch.bool, device=symbolic_vector.device)
            return HoCShape(leaves=[], mask=empty_mask, per_step=[])

        # Step 1: per-position activity mask. Mereology applies no
        # magnitude threshold (a magnitude threshold is not how truths are
        # accepted -- see the content-aware ``truthCriterion`` path that
        # governs both truth recording and learned-relation acceptance; the
        # old ``truthMinMagnitude`` knob is retired). The threshold
        # collapses to 0.0 here, so any position with nonzero norm counts
        # as active and the all-zero / NULL-padded inputs still fold to an
        # empty shape. ``threshold`` is
        # threaded into ``_walk_reverse`` / ``_build_step`` (per-step
        # activity gate) so it stays a named local here.
        threshold = 0.0
        norms = symbolic_vector.norm(dim=-1)
        mask = norms > threshold
        if not bool(mask.any().item()):
            return HoCShape(leaves=[], mask=mask, per_step=[])

        # Step 2: derivation path (outer-to-inner). Same path for all
        # (B, V) per the spec's row-0 canonical-path convention.
        path = self._derivation_path()

        # Step 3: K cap.
        if max_active_per_layer is None:
            try:
                max_active_per_layer = int(
                    TheXMLConfig.get("architecture.maxActivePerLayer",
                                     default=8) or 8)
            except (KeyError, TypeError, ValueError):
                max_active_per_layer = 8
        K_cap = max(1, int(max_active_per_layer))

        # Step 4: vectorized recursive descent.
        leaves, per_step = self._walk_reverse(symbolic_vector, path, K_cap, threshold)
        return HoCShape(leaves=leaves, mask=mask, per_step=per_step)

    def _derivation_path(self):
        """Build the outer-to-inner rule sequence for hoc_shape.

        Default-only mode: synthesize a fixed alternating
        ``[sigma(SS), pi(CS)] * subsymbolicOrder`` path -- no chart was
        consulted (Phase 1.5 fast-path bypass keeps current_rules
        empty), so every position followed the per-space_role default
        unary rule.

        Chart-driven mode: read row 0 of
        ``self.symbolSpace.generate_rules`` per space_role (SS then CS then
        subsymbolic in pipeline-reverse order) and concatenate. Same
        canonical-path convention as ``_row_zero_rules`` at Language.py:2247.
        """
        n_stages = max(1, int(getattr(self, 'subsymbolicOrder', 1) or 1))
        path = []

        ss = getattr(self, 'symbolSpace', None)
        gen_rules = getattr(ss, 'generate_rules', None) if ss is not None else None
        chart_populated = bool(gen_rules) and any(
            v for v in gen_rules.values() if v
        ) if isinstance(gen_rules, dict) else False

        if not chart_populated:
            # Default-only path: ['sigma' (SS), 'pi' (CS)] * n_stages.
            for _ in range(n_stages):
                path.append(RuleSpec(rule_name='sigma', arity=1, space_role='SS'))
                path.append(RuleSpec(rule_name='pi',    arity=1, space_role='CS'))
            return path

        # Chart-driven: walk SS-space_role rules then CS-space_role rules in
        # reverse-pipeline order. Use row 0 as canonical.
        try:
            from Language import TheGrammar
            for space_role in ('SS', 'CS', 'subsymbolic'):
                per_space_role = gen_rules.get(space_role) if gen_rules else None
                if not per_space_role:
                    continue
                row_zero = (per_space_role[0]
                            if per_space_role and isinstance(per_space_role[0], list)
                            else per_space_role)
                for rule_id in row_zero:
                    try:
                        rd = TheGrammar.rules[int(rule_id)]
                        if rd.method_name is None:
                            continue
                        path.append(RuleSpec(
                            rule_name=rd.method_name,
                            arity=int(rd.arity),
                            space_role=str(space_role)))
                    except (IndexError, AttributeError, ValueError, TypeError):
                        continue
        except Exception:
            pass

        if not path:
            # Fallback: default-only synthesis if chart parsing failed.
            for _ in range(n_stages):
                path.append(RuleSpec(rule_name='sigma', arity=1, space_role='SS'))
                path.append(RuleSpec(rule_name='pi',    arity=1, space_role='CS'))
        return path

    def _walk_reverse(self, parent_tensor, path, K_cap, threshold):
        """Recursive vectorized descent through the derivation tree.

        Returns:
            (leaves, per_step) where:
              * leaves: list of ``[B, V, D']`` tensors -- one per leaf.
              * per_step: list of StepInfo -- one per layer-level
                reverse executed (DFS order: pre-order on each
                subtree).
        """
        if not path:
            return ([parent_tensor], [])

        head = path[0]
        rest = path[1:]
        contig = head.rule_name in CONTIGUITY_PRESERVING_OPS
        layer = self._lookup_host_layer(head.space_role, head.rule_name)

        if layer is None:
            # No host layer wired -- skip this rule (treat as identity).
            return self._walk_reverse(parent_tensor, rest, K_cap, threshold)

        if head.arity == 1:
            child = self._call_reverse_shape_adaptive(layer, parent_tensor)
            if child is None:
                # Reverse failed (lossy / not invertible) -- skip step.
                return self._walk_reverse(parent_tensor, rest, K_cap, threshold)
            step = self._build_step(head, contig, child, K_cap, threshold,
                                    branch='')
            sub_leaves, sub_steps = self._walk_reverse(child, rest, K_cap, threshold)
            return (sub_leaves, [step, *sub_steps])

        if head.arity == 2:
            try:
                pair = layer.generate(parent_tensor)
            except Exception:
                return self._walk_reverse(parent_tensor, rest, K_cap, threshold)
            if not isinstance(pair, tuple) or len(pair) != 2:
                # Not a (left, right) pair -- treat as unary.
                child = pair if torch.is_tensor(pair) else parent_tensor
                step = self._build_step(head, contig, child, K_cap, threshold,
                                        branch='')
                sub_leaves, sub_steps = self._walk_reverse(child, rest, K_cap, threshold)
                return (sub_leaves, [step, *sub_steps])
            left, right = pair
            step_l = self._build_step(head, contig, left,  K_cap, threshold,
                                      branch='left')
            step_r = self._build_step(head, contig, right, K_cap, threshold,
                                      branch='right')
            leaves_L, steps_L = self._walk_reverse(left,  rest, K_cap, threshold)
            leaves_R, steps_R = self._walk_reverse(right, rest, K_cap, threshold)
            return (leaves_L + leaves_R,
                    [step_l, *steps_L, step_r, *steps_R])

        # Unsupported arity -- skip.
        return self._walk_reverse(parent_tensor, rest, K_cap, threshold)

    @staticmethod
    def _call_reverse_shape_adaptive(layer, x):
        """Call ``layer.reverse(x)`` with per-position / flatten
        adaptation so a layer whose ``nInput`` differs from
        ``x.shape[-1]`` still works.

        Some host layers (e.g. WholeSpace.sigma in BasicModel)
        are per-position: ``nInput == per-position dim``, and
        ``layer.reverse(x: [B, V, D])`` returns ``[B, V, D']``.

        Others (e.g. ConceptualSpace.pi in BasicModel.xml) are
        flatten-based: ``nInput == V * D``, and
        ``layer.reverse(x: [B, V*D])`` returns ``[B, V*D']``. For
        those we flatten the per-position input on call, reverse,
        and reshape the output back to ``[B, V, D']``.

        Returns the per-position output tensor, or ``None`` when
        reverse fails (lossy layer / dim cannot be reconciled).
        """
        if not torch.is_tensor(x):
            return None
        nInput = getattr(layer, 'nInput', None)
        if nInput is None:
            try:
                return layer.reverse(x)
            except Exception:
                return None
        # Per-position match.
        if nInput == x.shape[-1]:
            try:
                return layer.reverse(x)
            except Exception:
                return None
        # Flatten match.
        if x.dim() >= 3:
            B = x.shape[0]
            V = x.shape[1]
            D = x.shape[-1]
            if V * D == nInput:
                flat = x.reshape(B, V * D)
                try:
                    out = layer.reverse(flat)
                except Exception:
                    return None
                if not torch.is_tensor(out):
                    return None
                # Reshape back to per-position.
                if out.shape[-1] % V == 0:
                    D_out = out.shape[-1] // V
                    return out.reshape(B, V, D_out)
                return out
        return None

    def _lookup_host_layer(self, space_role, rule_name):
        """Resolve a (space_role, rule_name) to a layer instance.

        Tries ``symbolSpace.host_layer`` first (chart-registered host
        layers, including WholeSpace.sigma / ConceptualSpace.pi /
        LiftLayer / LowerLayer). Falls back to the parameter-free
        ``GRAMMAR_LAYER_CLASSES[rule_name]()`` instance for ops that
        aren't on the host registry (e.g. NotLayer when it isn't
        attached as a builtin).
        """
        ss = getattr(self, 'symbolSpace', None)
        if ss is not None and hasattr(ss, 'host_layer'):
            try:
                lyr = ss.host_layer(space_role, rule_name)
                if lyr is not None:
                    return lyr
            except Exception:
                pass
        cls = GRAMMAR_LAYER_CLASSES.get(rule_name)
        if cls is None:
            return None
        try:
            return cls()
        except TypeError:
            return None

    @staticmethod
    def _build_step(head, contig, tensor, K_cap, threshold, branch):
        """Construct a StepInfo for one reverse call.

        Vectorized top-K active selection over the [B, V, D] tensor:
        rank positions by L2 norm, take top-K per batch row, count
        how many are above threshold.
        """
        # tensor: [B, V, D] (or [B, V, ...] -- norm collapses last dim).
        if tensor.dim() < 2:
            empty = torch.zeros(0, dtype=torch.long, device=tensor.device)
            return StepInfo(
                rule_name=head.rule_name, arity=int(head.arity),
                space_role=str(head.space_role), contiguous=bool(contig),
                branch=str(branch),
                active_indices=empty.unsqueeze(0),
                active_count=torch.zeros(1, dtype=torch.long, device=tensor.device),
                K_cap=int(K_cap))
        norms = tensor.norm(dim=-1)         # [B, V]
        if norms.dim() < 2:
            norms = norms.unsqueeze(0)
        B, V = norms.shape[0], norms.shape[1] if norms.dim() >= 2 else 1
        K_eff = min(int(K_cap), V)
        if K_eff <= 0:
            empty = torch.zeros((B, 0), dtype=torch.long, device=tensor.device)
            cnt = torch.zeros(B, dtype=torch.long, device=tensor.device)
            return StepInfo(
                rule_name=head.rule_name, arity=int(head.arity),
                space_role=str(head.space_role), contiguous=bool(contig),
                branch=str(branch),
                active_indices=empty, active_count=cnt, K_cap=int(K_cap))
        topk = torch.topk(norms, K_eff, dim=-1)
        idx = topk.indices                   # [B, K_eff] long
        vals = topk.values                   # [B, K_eff] float
        above = (vals > threshold).long()    # [B, K_eff]
        count = above.sum(dim=-1)            # [B]
        return StepInfo(
            rule_name=head.rule_name, arity=int(head.arity),
            space_role=str(head.space_role), contiguous=bool(contig),
            branch=str(branch),
            active_indices=idx, active_count=count, K_cap=int(K_cap))

    def _reverse_one_subsymbolic_order(self, x, stage_idx):
        """Back-project ``x`` through one subsymbolicOrder stage:
        WholeSpace.sigma.reverse -> ConceptualSpace.pi.reverse.

        Legacy helper retained for callers that pre-date the
        ``hoc_shape`` rewrite. New callers should go through
        ``hoc_shape`` (which honors derivation history, top-K caps,
        and per-step contiguity tracking).
        """
        try:
            sym_stages = getattr(self, 'wholeSpaces', None)
            con_stages = getattr(self, 'conceptualSpaces', None)
            if sym_stages is not None and stage_idx < len(sym_stages):
                sym = sym_stages[stage_idx]
                con = con_stages[stage_idx]
            else:
                sym = self.wholeSpace
                con = self.conceptualSpace
        except (AttributeError, IndexError):
            return None
        try:
            sigma = getattr(sym, 'sigma', None)
            if sigma is None:
                return None
            y = sigma.reverse(x) if hasattr(sigma, 'reverse') else x
            pi = getattr(con, 'pi', None)
            if pi is None:
                return y
            return pi.reverse(y) if hasattr(pi, 'reverse') else y
        except Exception:
            return None

    def Continuous(self) -> float:
        """Simplicity (Continuity / Open Awareness) -- empirical
        ε-δ continuity measure on the back-projected leaf
        hyperrectangles.

        Definition: ``f`` is continuous at ``x`` iff
        ``∀ε > 0, ∃δ > 0`` with ``‖x' − x‖ < δ ⇒ ‖f(x') − f(x)‖ < ε``.
        Adapted to a finite set of leaf hyperrectangles
        (``hoc_shape.leaves``): for each pair of leaves the input-
        side and output-side bivectors define boxes; the ratio of
        their union diameters is the empirical ε / δ for that pair.

        Continuity holds when the worst-case ratio across
        trustworthy pairs stays bounded by a configured target
        (default 1.0 -- non-expansion). The measure is the
        ``tanh(γ · (target − worst_ratio))`` squash so the return
        value lies in ``[-1, +1]`` with the same sign convention as
        ``Contiguous()``:

          ``+1`` -- continuous: every leaf pair's output-spread
                    stayed within the configured factor of the
                    input-spread.
          ``-1`` -- discontinuous: at least one trustworthy pair
                    blew up beyond the target.
          ``0``  -- unknown: no trustworthy pairs (all leaves
                    traversed at least one lossy reverse op, so
                    the back-projection geometry is unreliable;
                    same convention as ``Contiguous()``).
          intermediate -- saturated tanh of the deviation from
                          target.

        Theoretical alignment: the per-layer LDU
        ``_d_effective`` clamp to ``[ε, 1.0]`` already bounds each
        layer's spectral norm; the empirical ratio measures how
        tight that bound stays under the actual derivation walk.
        Sample-based, not analytic: a +1 result means "no observed
        violations on this sample", not "globally continuous".

        Knobs (read once per call from XML or instance attrs):
          ``architecture.continuityRatioTarget`` (default 1.0).
          ``architecture.continuitySharpness``   (default 1.0).
          ``architecture.continuityNorm``        (default 'inf'; 'l2' available).
        """
        sym = getattr(self, 'wholeSpace', None)
        if sym is None or not hasattr(sym, 'subspace'):
            return 0.0
        try:
            sym_act = sym.subspace.materialize()
        except Exception:
            return 0.0
        if sym_act is None or not torch.is_tensor(sym_act) or sym_act.numel() == 0:
            return 0.0

        shape = self.hoc_shape(sym_act)
        leaves = shape.leaves
        if not leaves:
            return 0.0

        leaf_trust = self._leaf_path_trust(shape)
        if not leaf_trust:
            return 0.0

        # Single leaf: trivially continuous if trustworthy.
        if len(leaves) < 2:
            return 1.0 if leaf_trust[0] else 0.0

        out_boxes = self._stack_leaf_boxes(leaves)              # [K, *, 2]
        if out_boxes is None:
            return 0.0
        in_boxes = self._stack_input_boxes(sym_act, out_boxes.shape)  # [K, *, 2]
        if in_boxes is None:
            return 0.0

        # Read knobs from config (with sensible defaults).
        try:
            ratio_target = float(TheXMLConfig.get(
                "architecture.continuityRatioTarget", default=1.0) or 1.0)
        except (KeyError, TypeError, ValueError):
            ratio_target = 1.0
        try:
            sharpness = float(TheXMLConfig.get(
                "architecture.continuitySharpness", default=1.0) or 1.0)
        except (KeyError, TypeError, ValueError):
            sharpness = 1.0
        try:
            norm_kind = str(TheXMLConfig.get(
                "architecture.continuityNorm", default='inf') or 'inf')
        except (KeyError, TypeError, ValueError):
            norm_kind = 'inf'
        if norm_kind not in ('inf', 'l2'):
            norm_kind = 'inf'

        ratios = Ops.epsilon_delta(in_boxes, out_boxes, norm=norm_kind)  # [K, K]

        # Mask: only trustworthy pairs above the diagonal.
        K = ratios.shape[0]
        leaf_trust_t = torch.tensor(leaf_trust, dtype=torch.bool,
                                    device=ratios.device)
        pair_trust = leaf_trust_t.unsqueeze(0) & leaf_trust_t.unsqueeze(1)
        triu = torch.triu(torch.ones(K, K, dtype=torch.bool,
                                     device=ratios.device), diagonal=1)
        mask = pair_trust & triu

        if not bool(mask.any().item()):
            return 0.0

        worst = float(ratios[mask].max().item())
        # Squash to [-1, +1].
        return math.tanh(sharpness * (ratio_target - worst))

    @staticmethod
    def _stack_leaf_boxes(leaves):
        """Stack a list of leaf tensors into a ``[K, ..., 2]``
        bivector tensor for ``Ops.epsilon_delta``.

        Each leaf is ``[B, V, D]`` with the ``.what`` bivector at
        ``[..., :2]``; we slice and stack along a new K dim.
        Returns ``None`` if any leaf is missing the bivector slice.
        """
        slices = []
        for leaf in leaves:
            if not torch.is_tensor(leaf) or leaf.shape[-1] < 2:
                return None
            slices.append(leaf[..., :2])
        try:
            return torch.stack(slices, dim=0)
        except Exception:
            return None

    @staticmethod
    def _stack_input_boxes(sym_act, target_shape):
        """Build an input-side bivector tensor matching
        ``target_shape`` (the output-side shape from
        ``_stack_leaf_boxes``).

        For the binary-fanout case every leaf shares the same
        input parent (the higher-order symbolic vector), so we
        broadcast ``sym_act[..., :2]`` along the K dim. Returns
        ``None`` if shapes don't reconcile.
        """
        if not torch.is_tensor(sym_act) or sym_act.shape[-1] < 2:
            return None
        in_slice = sym_act[..., :2]           # [B, V, 2]
        K = target_shape[0]
        try:
            tail = list(target_shape[1:])     # [*leaf_dims..., 2]
            if list(in_slice.shape) == tail:
                return in_slice.unsqueeze(0).expand(K, *tail)
            # Allow leading-dim broadcast (B == 1 etc.).
            return in_slice.unsqueeze(0).expand(K, *tail)
        except Exception:
            return None

    # -- Area / Luminosity (Phase 1b replacements) ---------------------

    def Area(self, x2=None) -> float:
        """Hyperrectangle volume of the higher-order symbolic vector's
        back-projected leaf set.  Companion to :meth:`Contiguous` /
        :meth:`Continuous`.

        Unary form (``x2 is None``):
          1. ``sym_act = self.wholeSpace.subspace.materialize()``
          2. ``shape = self.hoc_shape(sym_act); leaves = shape.leaves``
          3. Filter leaves by :meth:`_leaf_path_trust`.
          4. For each trustworthy leaf, take ``leaf[..., :2]`` and
             feed :func:`Ops.hyperrectangle_volume` (treating the
             flat ``[B, V, 2]`` as ``[B, V, 1, 2]`` -- one axis per
             slot).  Sum across leaves.
          5. Persist per-step record onto
             ``self.conceptualSpace.subspace.knowing``.

        Binary form (``x2`` given):
          Subtract ``x2`` from ``sym_act`` first, then run
          :meth:`hoc_shape` on the difference.  Per the user's
          monotone-mapping optimisation, the resulting leaves are the
          difference regions and the volume sums into the binary
          ``Area``.

        Returns ``0.0`` when no trustworthy leaves are available
        (matches the ``Contiguous`` / ``Continuous`` no-trust
        convention).  Otherwise returns a Python float in ``[0, 1]``.
        """
        sym = getattr(self, 'wholeSpace', None)
        if sym is None or not hasattr(sym, 'subspace'):
            return 0.0
        try:
            sym_act = sym.subspace.materialize()
        except Exception:
            return 0.0
        if sym_act is None or not torch.is_tensor(sym_act) or sym_act.numel() == 0:
            return 0.0

        if x2 is not None and torch.is_tensor(x2):
            try:
                target = sym_act - x2
            except Exception:
                target = sym_act
        else:
            target = sym_act

        shape = self.hoc_shape(target)
        leaves = shape.leaves
        if not leaves:
            return 0.0
        leaf_trust = self._leaf_path_trust(shape)
        if not leaf_trust:
            return 0.0

        total = 0.0
        leaf_vols = []
        for leaf, trust in zip(leaves, leaf_trust):
            if not trust:
                leaf_vols.append(0.0)
                continue
            if not torch.is_tensor(leaf) or leaf.shape[-1] < 2:
                leaf_vols.append(0.0)
                continue
            # ``leaf[..., :2]`` is per-position bivector.  Treat each
            # position as one axis: reshape to ``[..., 1, 2]`` so
            # ``hyperrectangle_volume`` returns one volume per leading
            # element.  Aggregate via mean over the [B, V] dims so
            # different leaf populations are comparable.
            box = leaf[..., :2].unsqueeze(-2)
            vol = Ops.hyperrectangle_volume(box)
            try:
                v = float(vol.mean().item())
            except Exception:
                v = 0.0
            leaf_vols.append(v)
            total += v
        # Clamp to [0, 1]; the leaf-volume normalisation can drift slightly
        # above 1 with degenerate inputs.
        if total < 0.0:
            total = 0.0
        if total > 1.0:
            total = 1.0
        self._record_knowing(leaves=leaves, leaf_trust=leaf_trust,
                             leaf_vols=leaf_vols, area=total,
                             luminosity=None)
        return total

    def Luminosity(self, x2=None, truth_layer=None) -> float:
        """Self-consistency of the back-projected leaf set.

            Luminosity = totalArea − sharedArea × |DoT_disagreement|

        Unary form (``x2 is None``, ``truth_layer is None``):
          ``totalArea = self.Area()``.  For every trustworthy leaf
          pair, compute :func:`Ops.hyperrectangle_overlap_volume` on
          the ``[B, V, 2]`` slot-bivector slice, and the absolute DoT
          disagreement (``|dot_i − dot_j|`` with ``dot = pos − neg``
          reduced over the slot).  Sum the pairwise penalty and
          subtract from ``totalArea``.  Result clamped to ``[-1, 1]``.

        Unary form with ``truth_layer``:
          Cumulative-vs-rest fold over the layer's stored truths.
          Each truth is decoded back to concept-space (via
          :meth:`WholeSpace.decode_to_concept`) and treated as a
          single-leaf bivector.  Initialise running region from
          ``(t0, t1)``; for each next truth ``t_i`` compute
          ``lum(running, t_i)`` and update
          ``running ← element-wise union`` (``Ops.union``).  Returns
          the running aggregate.

        Binary form (``x2`` given):
          Apply the monotone-mapping optimisation on
          ``sym_act − x2`` and aggregate as in the unary form.

        Returns ``0.0`` for the same edge cases as :meth:`Contiguous`
        / :meth:`Continuous`.
        """
        if truth_layer is not None:
            return self._luminosity_truth_fold(truth_layer)

        sym = getattr(self, 'wholeSpace', None)
        if sym is None or not hasattr(sym, 'subspace'):
            return 0.0
        try:
            sym_act = sym.subspace.materialize()
        except Exception:
            return 0.0
        if sym_act is None or not torch.is_tensor(sym_act) or sym_act.numel() == 0:
            return 0.0

        if x2 is not None and torch.is_tensor(x2):
            try:
                target = sym_act - x2
            except Exception:
                target = sym_act
        else:
            target = sym_act

        shape = self.hoc_shape(target)
        leaves = shape.leaves
        if not leaves:
            return 0.0
        leaf_trust = self._leaf_path_trust(shape)
        if not leaf_trust:
            return 0.0

        # totalArea: sum of trustworthy leaves' volumes.
        leaf_vols = []
        leaf_dots = []
        leaf_boxes = []
        total_area = 0.0
        for leaf, trust in zip(leaves, leaf_trust):
            if not trust or not torch.is_tensor(leaf) or leaf.shape[-1] < 2:
                leaf_vols.append(0.0)
                leaf_dots.append(0.0)
                leaf_boxes.append(None)
                continue
            box = leaf[..., :2].unsqueeze(-2)        # [B, V, 1, 2]
            vol = Ops.hyperrectangle_volume(box)
            v = float(vol.mean().item())
            leaf_vols.append(v)
            total_area += v
            # DoT = pos - neg, mean over leading dims.
            dot = (leaf[..., 0] - leaf[..., 1]).mean()
            leaf_dots.append(float(dot.item()))
            leaf_boxes.append(box)
        total_area = max(0.0, min(1.0, total_area))

        # Pairwise penalty.
        penalty = 0.0
        n = len(leaves)
        for i in range(n):
            if not leaf_trust[i] or leaf_boxes[i] is None:
                continue
            for j in range(i + 1, n):
                if not leaf_trust[j] or leaf_boxes[j] is None:
                    continue
                shared = Ops.hyperrectangle_overlap_volume(
                    leaf_boxes[i], leaf_boxes[j])
                try:
                    s = float(shared.mean().item())
                except Exception:
                    s = 0.0
                disagree = abs(leaf_dots[i] - leaf_dots[j])
                penalty += s * disagree

        lum = total_area - penalty
        if lum > 1.0:
            lum = 1.0
        if lum < -1.0:
            lum = -1.0
        self._record_knowing(leaves=leaves, leaf_trust=leaf_trust,
                             leaf_vols=leaf_vols, area=total_area,
                             luminosity=lum)
        return lum

    def _luminosity_truth_fold(self, truth_layer) -> float:
        """Catuṣkoṭi luminosity of the truth store (thin delegator).

        The computation lives on :meth:`TruthLayer.luminosity`
        (MeronomySpec §3, rev 2026-06-10b): per conceptual dimension the
        stored signed references split into true/false pole coverage
        ``(T_k, F_k)`` and the measure is
        ``mean_k[(T_k − F_k) − min(T_k, F_k)]`` in ``[-1, 1]`` — total
        area weighted by sign minus sign-conflict regions, computed
        directly over the codes. The former per-row
        ``decode_to_concept`` pullback and sequential cumulative fold
        are retired; the ``sym`` handle is still passed for signature
        compatibility and ignored by the layer.
        """
        if truth_layer is None:
            return 0.0
        return truth_layer.luminosity(sym=getattr(self, 'wholeSpace', None))

    def _record_knowing(self, *, leaves, leaf_trust, leaf_vols, area,
                        luminosity):
        """Persist a per-pass record onto
        ``self.conceptualSpace.subspace.knowing``.

        Captures area / luminosity plus the top-2 leaves' intersection
        and union (via :meth:`Ops.intersection` / :meth:`Ops.union`).
        Silently skips when the conceptualSpace or its subspace is
        absent (e.g. fixture models built without a ConceptualSpace).
        """
        cs = getattr(self, 'conceptualSpace', None)
        sub = getattr(cs, 'subspace', None) if cs is not None else None
        if sub is None:
            return
        # Pick the top-2 leaves by volume for intersection / union.
        ranked = sorted(
            ((vol, idx) for idx, (vol, t) in enumerate(zip(leaf_vols, leaf_trust)) if t),
            reverse=True)
        intersection = None
        union = None
        if len(ranked) >= 2:
            i = ranked[0][1]
            j = ranked[1][1]
            li = leaves[i]
            lj = leaves[j]
            if (torch.is_tensor(li) and torch.is_tensor(lj)
                    and li.shape == lj.shape and li.shape[-1] >= 2):
                a = li[..., :2]
                b = lj[..., :2]
                try:
                    intersection = Ops.intersection(a, b, monotonic=False)
                    union = Ops.union(a, b, monotonic=False)
                except Exception:
                    intersection = None
                    union = None
        if sub.knowing is None:
            sub.knowing = []
        sub.knowing.append({
            'step':        len(sub.knowing),
            'area':        area,
            'luminosity':  luminosity,
            'intersection': intersection,
            'union':        union,
        })

    def isIsomorphic(self, x1=None, x2=None) -> float:
        """Mereological-isomorphism measure between two bivector inputs.

            ``y = sharedPartsAndWholes(x1, x2) / totalPartsAndWholes(x1, x2)``

        Both inputs are back-projected via :meth:`hoc_shape` into their
        C(1) constituent leaves (parts). The (x1, x2) whole-vs-whole pair
        plus each (leaves1[i], leaves2[i]) part-vs-part pair contributes
        one hyperrectangle overlap volume to the numerator
        ``sharedPartsAndWholes`` and one union volume to the denominator
        ``totalPartsAndWholes``. Untrustworthy leaf pairs (any step on
        the derivation path used a lossy reverse) are dropped, matching
        the :meth:`Contiguous` / :meth:`Continuous` no-trust convention.

          ``1.0`` -- parts and wholes coincide; x1 and x2 are
                     mereologically isomorphic.
          ``0.0`` -- disjoint regions, or no trustworthy pairs to
                     decide on.
          intermediate -- Jaccard / IoU of the combined parts-and-
                          wholes decomposition.

        Args:
            x1: optional ``[B, V, D]`` bivector. Defaults to
                ``self.wholeSpace.subspace.materialize()``.
            x2: ``[B, V, D]`` bivector to compare against. Required
                for a non-zero return.
        """
        if x1 is None:
            sym = getattr(self, 'wholeSpace', None)
            if sym is None or not hasattr(sym, 'subspace'):
                return 0.0
            try:
                x1 = sym.subspace.materialize()
            except Exception:
                return 0.0
        if (x1 is None or not torch.is_tensor(x1) or x1.numel() == 0 or
                x2 is None or not torch.is_tensor(x2) or x2.numel() == 0):
            return 0.0
        if x1.shape[-1] < 2 or x2.shape[-1] < 2:
            return 0.0

        shape1 = self.hoc_shape(x1)
        shape2 = self.hoc_shape(x2)
        leaves1, leaves2 = shape1.leaves, shape2.leaves
        trust1 = self._leaf_path_trust(shape1) if leaves1 else []
        trust2 = self._leaf_path_trust(shape2) if leaves2 else []

        # Whole-vs-whole pair, then matched part-vs-part pairs.
        pairs = []
        if x1.shape == x2.shape:
            pairs.append((x1[..., :2].unsqueeze(-2),
                          x2[..., :2].unsqueeze(-2)))
        for i in range(min(len(leaves1), len(leaves2))):
            if i < len(trust1) and not trust1[i]:
                continue
            if i < len(trust2) and not trust2[i]:
                continue
            li, lj = leaves1[i], leaves2[i]
            if not (torch.is_tensor(li) and torch.is_tensor(lj)):
                continue
            if li.shape[-1] < 2 or lj.shape[-1] < 2:
                continue
            if li.shape != lj.shape:
                continue
            pairs.append((li[..., :2].unsqueeze(-2),
                          lj[..., :2].unsqueeze(-2)))

        if not pairs:
            return 0.0

        shared_parts_and_wholes = 0.0
        total_parts_and_wholes = 0.0
        for a, b in pairs:
            try:
                v_a = float(Ops.hyperrectangle_volume(a).mean().item())
                v_b = float(Ops.hyperrectangle_volume(b).mean().item())
                v_shared = float(
                    Ops.hyperrectangle_overlap_volume(a, b).mean().item())
            except Exception:
                continue
            shared_parts_and_wholes += v_shared
            total_parts_and_wholes += (v_a + v_b - v_shared)

        if total_parts_and_wholes <= 0.0:
            return 0.0
        return max(0.0, min(1.0,
                            shared_parts_and_wholes / total_parts_and_wholes))

    def Peaceful(self):
        """One Taste (Emotional Symmetry / Balance).

        Letting attachment to feelings within conceptual space be
        uniformly 1, so that instead of adapting weight space to our
        thoughts we adapt our feelings equanimously to our sensory space.
        Requires emotional symmetry.

        Characterisation -- balance dissonance and consonance:
          * Feelings (vedana / valence annotations) should not be removed
            -- that is the nihilist's mistake.  Instead they must be
            *appropriate*: consonant with reality.
          * Appropriateness manifests when the objects that are loved are
            either real (grounded in PartSpace with trust > 0) or
            when the representations are at least 5-dimensional (which
            limits the dissonance that arises from reification of
            low-dimensional abstractions).
          * The loss landscape should be symmetric w.r.t. positive and
            negative valence -- no bias toward pleasant or unpleasant
            content in the gradient signal.

        Computationally, Peaceful() should measure the balance between
        dissonance and consonance across the TruthLayer and verify that
        the model does not preferentially attend to or avoid any
        particular valence.
        """
        raise NotImplementedError
