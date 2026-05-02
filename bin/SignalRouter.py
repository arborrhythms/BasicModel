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


class _BinaryGrammarOpAdapter(nn.Module):
    """Adapt a GrammarLayer with a `.compose(left, right)` method into a
    plain binary callable for the SignalRouter's `BinaryStructuredReductionLayer`.

    The CKY chart calls `gl.compose(left, right)` on `[..., D]` pairs;
    `BinaryStructuredReductionLayer` calls `op(left, right)` on
    `[B, N-1, D]` pairs. The two contracts agree element-wise; this
    adapter just forwards.
    """

    def __init__(self, gl):
        super().__init__()
        self.gl = gl

    def forward(self, left, right):
        return self.gl.compose(left, right)


class SignalRouter(nn.Module):
    """Top-level signal-routing parser. Owned by Chart when
    router_kind == "signal". Parallels Chart.compose / Chart.generate.

    Multi-tier: a unary layer and/or a binary layer can be attached per
    tier (e.g., 'P', 'C', 'S'). On compose, tiers run in sorted order;
    within each tier, unary fires first then binary, with the soft slab
    of the previous step feeding the next so gradient reaches every op.
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
        self._unary_layers = nn.ModuleDict()
        self._binary_layers = nn.ModuleDict()
        # Parallel arrays of global rule_ids per attached layer; keyed by
        # tier. Local op_id (the index inside the layer's ModuleList) maps
        # to the corresponding global rule_id at this list position.
        self._unary_rule_ids = {}
        self._binary_rule_ids = {}
        # Per-compose cache for generate / inspection.
        self._last_input = None
        self._last_output = None
        self._last_tier_routings = {}

    def attach_unary_ops(self, *, ops, rule_ids=None, r_copy=1, tier="S"):
        tier = str(tier)
        layer = UnaryStructuredLayer(
            d_model=self.feature_dim,
            ops=ops, r_copy=r_copy,
            temperature=self.temperature,
        )
        self._unary_layers[tier] = layer
        if rule_ids is None:
            rule_ids = list(range(len(ops)))
        else:
            rule_ids = [int(r) for r in rule_ids]
        if len(rule_ids) != len(ops):
            raise ValueError(
                f"attach_unary_ops: len(rule_ids)={len(rule_ids)} != "
                f"len(ops)={len(ops)} for tier {tier!r}")
        self._unary_rule_ids[tier] = rule_ids

    def attach_layer_ops(self, *, ops, rule_ids=None, r_copy=1, tier="S"):
        tier = str(tier)
        layer = BinaryStructuredReductionLayer(
            d_model=self.feature_dim,
            ops=ops, r_copy=r_copy,
            temperature=self.temperature,
        )
        self._binary_layers[tier] = layer
        if rule_ids is None:
            rule_ids = list(range(len(ops)))
        else:
            rule_ids = [int(r) for r in rule_ids]
        if len(rule_ids) != len(ops):
            raise ValueError(
                f"attach_layer_ops: len(rule_ids)={len(rule_ids)} != "
                f"len(ops)={len(ops)} for tier {tier!r}")
        self._binary_rule_ids[tier] = rule_ids

    def compose(self, data, word_space, subspace=None):
        if not self._unary_layers and not self._binary_layers:
            raise RuntimeError(
                "SignalRouter.compose called before attach_layer_ops() / "
                "attach_unary_ops().")
        x = data
        rules = {}
        self._last_tier_routings = {}
        all_tiers = sorted(set(self._unary_layers.keys())
                           | set(self._binary_layers.keys()))

        for tier in all_tiers:
            B = x.shape[0]
            tier_routing = {}
            tier_rules_per_row = [[] for _ in range(B)]

            unary_layer = self._unary_layers[tier] if tier in self._unary_layers else None
            if unary_layer is not None:
                u_hard, u_soft, u_routing = unary_layer(x)
                tier_routing["unary"] = u_routing
                rid_table = self._unary_rule_ids[tier]
                kind = u_routing["action_kind"]
                op = u_routing["action_op"]
                for b in range(B):
                    for j in range(kind.shape[1]):
                        if int(kind[b, j].item()) == 2:
                            tier_rules_per_row[b].append(
                                rid_table[int(op[b, j].item())])
                # Propagate soft slab so gradient reaches unary ops at
                # later tiers / through the binary stage of this tier.
                x = u_soft

            binary_layer = self._binary_layers[tier] if tier in self._binary_layers else None
            if binary_layer is not None:
                # Recursive reduction: iterate the binary layer up to
                # (N-1) times so the slab folds down to a single S start
                # state. Each round's marginal_slab feeds the next; the
                # leading position (index 0) accumulates the fully-folded
                # state. The shape stays [B, N, D] in soft form (right-
                # tail positions get pad-weighted as reductions fire);
                # the canonical [B, 1, D] root state is x[:, 0:1, :]
                # after the final round.
                rid_table = self._binary_rule_ids[tier]
                max_rounds = max(0, x.shape[1] - 1)
                round_routings = []
                for _ in range(max_rounds):
                    b_hard, b_soft, b_routing = binary_layer(x)
                    round_routings.append(b_routing)
                    kind = b_routing["action_kind"]
                    op = b_routing["action_op"]
                    lengths = b_routing["lengths"]
                    B_now = kind.shape[0]
                    for b in range(B_now):
                        L = int(lengths[b].item())
                        for j in range(L):
                            if int(kind[b, j].item()) == 1:
                                tier_rules_per_row[b].append(
                                    rid_table[int(op[b, j].item())])
                    x = b_soft
                if round_routings:
                    # Last round's routing is the canonical "binary"
                    # diagnostic; the full sequence is in "binary_rounds".
                    tier_routing["binary"] = round_routings[-1]
                    tier_routing["binary_rounds"] = round_routings

            self._last_tier_routings[tier] = tier_routing
            rules[tier] = tier_rules_per_row

        # Canonical S start state: leading position of the final slab,
        # shape [B, 1, D]. Downstream consumers that want the single
        # parsed state read from here.
        if x.shape[1] >= 1:
            self._last_root_state = x[:, 0:1, :]
        else:
            self._last_root_state = x
        # Length-N expansion of the root state, for shape-compat write-
        # back into subspace.event so downstream layers don't need
        # adapting to a [B, 1, D] input.
        self._last_input = data
        self._last_output = self._last_root_state.expand(
            -1, data.shape[1], -1).contiguous()
        return rules

    def generate(self, target, word_space, subspace=None):
        if not self._unary_layers and not self._binary_layers:
            raise RuntimeError(
                "SignalRouter.generate called before attach_layer_ops() / "
                "attach_unary_ops().")
        if not self._last_tier_routings:
            self.compose(target, word_space, subspace=subspace)
        # Generate emits the compose-order list reversed per row, so that
        # the inverse pass pops the last-applied rule first. Tier order is
        # also reversed (innermost first).
        compose_rules = self._compose_rules_from_routings()
        all_tiers = sorted(compose_rules.keys(), reverse=True)
        return {tier: [row[::-1] for row in compose_rules[tier]]
                for tier in all_tiers}

    def _compose_rules_from_routings(self):
        rules = {}
        for tier, tier_routing in self._last_tier_routings.items():
            tier_rules_per_row = None
            if "unary" in tier_routing:
                rid_table = self._unary_rule_ids[tier]
                r = tier_routing["unary"]
                kind = r["action_kind"]
                op = r["action_op"]
                B = kind.shape[0]
                tier_rules_per_row = [[] for _ in range(B)]
                for b in range(B):
                    for j in range(kind.shape[1]):
                        if int(kind[b, j].item()) == 2:
                            tier_rules_per_row[b].append(
                                rid_table[int(op[b, j].item())])
            if "binary" in tier_routing:
                rid_table = self._binary_rule_ids[tier]
                r = tier_routing["binary"]
                kind = r["action_kind"]
                op = r["action_op"]
                lengths = r["lengths"]
                B = kind.shape[0]
                if tier_rules_per_row is None:
                    tier_rules_per_row = [[] for _ in range(B)]
                for b in range(B):
                    L = int(lengths[b].item())
                    for j in range(L):
                        if int(kind[b, j].item()) == 1:
                            tier_rules_per_row[b].append(
                                rid_table[int(op[b, j].item())])
            if tier_rules_per_row is not None:
                rules[tier] = tier_rules_per_row
        return rules

    # -- backwards-compat shims for diagnostics / older tests -----------
    @property
    def _last_routing(self):
        # Returns the binary routing of the highest-tier (last-run) tier
        # that has one, mirroring the pre-multi-tier API.
        for tier in sorted(self._last_tier_routings.keys(), reverse=True):
            tr = self._last_tier_routings[tier]
            if "binary" in tr:
                return tr["binary"]
        return None

    @property
    def _last_unary_routing(self):
        for tier in sorted(self._last_tier_routings.keys(), reverse=True):
            tr = self._last_tier_routings[tier]
            if "unary" in tr:
                return tr["unary"]
        return None

    @property
    def _last_hard_slab(self):
        return self._last_output

    @property
    def _last_soft_slab(self):
        return self._last_output

    def _collect_binary_rule_selections(self, routing):
        kind = routing["action_kind"]
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
    neg_inf_b = copy_score.new_full((B,), NEG_INF)

    # alpha as a list of [B] tensors; avoids autograd-hostile in-place
    # writes into a stacked [B, N+1] tensor.
    alpha_list = [neg_inf_b for _ in range(N + 1)]
    alpha_list[0] = copy_score.new_zeros(B)
    for t in range(N):
        alpha_list[t + 1] = torch.logaddexp(
            alpha_list[t + 1], alpha_list[t] + c_action[:, t])
        if t + 1 < N:
            alpha_list[t + 2] = torch.logaddexp(
                alpha_list[t + 2], alpha_list[t] + r_action[:, t])
    alpha = torch.stack(alpha_list, dim=1)
    logZ = alpha[:, N]

    beta_list = [neg_inf_b for _ in range(N + 1)]
    beta_list[N] = copy_score.new_zeros(B)
    for t in reversed(range(N)):
        beta_list[t] = torch.logaddexp(
            beta_list[t], c_action[:, t] + beta_list[t + 1])
        if t + 1 < N:
            beta_list[t] = torch.logaddexp(
                beta_list[t], r_action[:, t] + beta_list[t + 2])
    beta = torch.stack(beta_list, dim=1)

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

    c_best, c_argop = copy_score.max(dim=-1)              # [B, N], [B, N]
    if R_reduce > 0 and N > 1:
        r_best, r_argop = reduce_score.max(dim=-1)
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

    for b in range(B):
        t = N
        while t > 0:
            kind = int(back_kind[b, t].item())
            op = int(back_op[b, t].item())
            if kind == 0:
                copy_mask[b, t - 1, op] = 1.0
                t -= 1
            elif kind == 1:
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


# BinaryPlacementScorer was removed in favor of anchor-based scoring on
# the rule's own output. See BinaryStructuredReductionLayer.copy_anchor /
# reduce_anchor and the einsum-based scoring in its forward(). Same goes
# for _UnaryPlacementScorer / UnaryStructuredLayer below.


class ComparatorMixer(nn.Module):
    """Four-branch trainable comparator-mixer.

    Per output position j, builds
        y_j = sum_k gate_jk * branch_jk
    over branches k in (keep=x_j, reduce=r_j, shift=x_{j+1}, pad=0).

    Gate weights from a small MLP over local context h_j; softmax with
    configurable temperature gives a strict generalization of (a) the
    soft DP-driven blend (when the MLP learns to copy DP marginals into
    the gates) and (b) hard routing (when the MLP learns one-hots).
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
    rm_op = (reduce_mask.argmax(-1) if reduce_mask.numel() > 0
             else torch.zeros_like(cm_op[:, :0]))

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
        cum = torch.cumsum(reduce_marginal, dim=1)             # [B, N-1]
        cumshift = torch.cat(
            [x.new_zeros(B, 1), cum], dim=1)                   # [B, N]

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
        # Anchor-based scoring (Stern et al. 2017 / Vaswani et al. 2017
        # style): the placement score is an inner product between the
        # rule's own output and a per-rule learnable anchor vector.
        # The same gradient that trains the rule trains its anchor; no
        # separate scorer MLP / second-optimizer pathology.
        self.copy_anchor = nn.Parameter(torch.randn(self.r_copy, self.d_model) * 0.02)
        self.reduce_anchor = nn.Parameter(torch.randn(self.r_reduce, self.d_model) * 0.02)
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
        if reduced_chosen.shape[1] > 0:
            r_padded = torch.cat([reduced_chosen, pad_slab], dim=1)
        else:
            r_padded = x.new_zeros(B, N, D)
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

        # Anchor-based scoring (replaces the old scorer MLP):
        #   copy_score[b, n, c]   = <x[b, n, :],            copy_anchor[c, :]>
        #   reduce_score[b, p, r] = <stacked_reduced[..., r, :], reduce_anchor[r, :]>
        # Each rule's anchor is part of its own parameter set, so the
        # placement score is a derived quantity of the rule's own
        # computation -- one optimizer, one graph, no separate scorer.
        copy_score = torch.einsum('bnd,cd->bnc', x, self.copy_anchor)
        if stacked_reduced.shape[1] > 0 and self.r_reduce > 0:
            reduce_score = torch.einsum(
                'bnrd,rd->bnr', stacked_reduced, self.reduce_anchor)
        else:
            reduce_score = x.new_zeros(B, max(N - 1, 0), self.r_reduce)

        soft = binary_tiling_soft_dp(copy_score, reduce_score)
        hard = binary_tiling_viterbi(copy_score, reduce_score)

        if hard["reduce_mask"].numel() > 0:
            reduce_op_per_pair = hard["reduce_mask"].argmax(-1)  # [B, N-1]
        else:
            reduce_op_per_pair = torch.zeros(
                B, 0, device=x.device, dtype=torch.long)

        # Hardened op-selection: forward uses one-hot argmax (sparse
        # commitment to a single op per pair); backward uses the soft
        # softmax over reduce_score so the scorer still receives gradient
        # via straight-through.
        if reduce_score.numel() > 0:
            op_soft = F.softmax(reduce_score, dim=-1)            # [B, N-1, R]
            op_hard = F.one_hot(
                op_soft.argmax(-1), num_classes=op_soft.shape[-1]
            ).to(op_soft.dtype)
            op_weights = op_hard + op_soft - op_soft.detach()
            chosen_reduced = (op_weights.unsqueeze(-1) * stacked_reduced).sum(dim=2)
        else:
            chosen_reduced = self._selected_reduced(
                stacked_reduced, reduce_op_per_pair)            # [B, N-1, D]

        hard_slab, hard_meta = compact_hard(
            x=x, reduced=chosen_reduced,
            copy_mask=hard["copy_mask"], reduce_mask=hard["reduce_mask"],
            span_start=span_start, span_end=span_end,
        )

        # Hardened DP marginals: forward uses the Viterbi route's hard
        # masks (one-hot copy at chosen positions, one-hot reduce at
        # chosen pairs) so the slab becomes structurally sparse — most
        # positions are either kept-as-is or reduced cleanly, with pad
        # at consumed slots. Backward uses the soft DP marginals as the
        # gradient surrogate. This is the "hardening" the user asked
        # for: forward decisions commit, gradient still flows back.
        copy_hard_action = hard["copy_mask"].sum(-1)             # [B, N], 0 or 1
        if hard["reduce_mask"].numel() > 0:
            reduce_hard_action = hard["reduce_mask"].sum(-1)     # [B, N-1]
        else:
            reduce_hard_action = soft["reduce_marginal"]
        copy_marginal_st = (
            copy_hard_action + soft["copy_marginal"] - soft["copy_marginal"].detach()
        )
        reduce_marginal_st = (
            reduce_hard_action + soft["reduce_marginal"]
            - soft["reduce_marginal"].detach()
        )
        marginal_slab = compact_soft(
            x=x, reduced=chosen_reduced,
            copy_marginal=copy_marginal_st,
            reduce_marginal=reduce_marginal_st,
        )

        # Comparator-mixer kept as a secondary trainable gate over the
        # four structural branches; available in the routing dict for
        # the Task 11 DP-prior regularizer and inverse-pass diagnostics.
        branches = self._gather_branches(x, chosen_reduced)
        _comp_slab, gates = self.comparator(h=h, branches=branches)
        soft_slab = marginal_slab

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


# _UnaryPlacementScorer was removed -- see UnaryStructuredLayer's
# copy_anchor / apply_anchor and the einsum-based scoring in its forward.


class UnaryStructuredLayer(nn.Module):
    """One unary layer: contextualize, score, choose action per position.

    Action space per position: R_copy + R_apply choices, exactly one
    fires. No structured DP -- positions are independent under unary.

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
        # Anchor-based scoring: per-(action, op) learnable anchor; score
        # is the inner product of the candidate output with the anchor.
        self.copy_anchor = nn.Parameter(torch.randn(self.r_copy, self.d_model) * 0.02)
        self.apply_anchor = nn.Parameter(torch.randn(self.r_apply, self.d_model) * 0.02)

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

        # Anchor-based scoring (replaces the old scorer MLP):
        #   copy_score[b, n, c]  = <x[b, n, :],            copy_anchor[c, :]>
        #   apply_score[b, n, a] = <applied[b, n, a, :], apply_anchor[a, :]>
        copy_score = torch.einsum('bnd,cd->bnc', x, self.copy_anchor)
        if self.r_apply > 0:
            apply_score = torch.einsum(
                'bnad,ad->bna', applied, self.apply_anchor)
        else:
            apply_score = x.new_zeros(B, N, 0)
        action_logits = torch.cat([copy_score, apply_score], dim=-1) / self.temperature
        action_soft = F.softmax(action_logits, dim=-1)
        # Hardened: forward uses argmax one-hot, backward gets the
        # softmax gradient via straight-through.
        action_hard = F.one_hot(
            action_soft.argmax(-1), num_classes=action_soft.shape[-1]
        ).to(action_soft.dtype)
        action_probs = action_hard + action_soft - action_soft.detach()

        # Soft slab: weighted blend over (copy x_j) and applied_op(x_j).
        copy_branch = x.unsqueeze(2).expand(B, N, self.r_copy, D)
        if self.r_apply > 0:
            branches = torch.cat([copy_branch, applied], dim=2)
        else:
            branches = copy_branch
        soft_slab = (action_probs.unsqueeze(-1) * branches).sum(dim=2)

        # Hard slab: argmax over actions.
        action_id = action_logits.argmax(dim=-1)            # [B, N]
        is_copy = action_id < self.r_copy
        gather_idx = action_id.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1, D)
        hard_slab = branches.gather(dim=2, index=gather_idx).squeeze(2)

        flat_one_hot = F.one_hot(
            action_id, num_classes=self.r_copy + self.r_apply
        ).to(x.dtype)
        copy_mask = flat_one_hot[..., :self.r_copy] if self.r_copy > 0 \
            else x.new_zeros(B, N, 0)
        apply_mask = flat_one_hot[..., self.r_copy:] if self.r_apply > 0 \
            else x.new_zeros(B, N, 0)

        # action_kind: 0 == copy, 2 == apply (unary). action_op is the
        # local op id within its kind's namespace.
        action_kind = torch.where(
            is_copy,
            torch.zeros_like(action_id),
            torch.full_like(action_id, 2),
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
