

import math, os, warnings
from collections import namedtuple
from contextlib import contextmanager, nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
try:
    from torchviz import make_dot
except ImportError:
    make_dot = None
from sklearn.decomposition import PCA
import torch.optim as optim
from torch.profiler import profile as torch_profile, ProfilerActivity, schedule as profiler_schedule
from functools import partial
from datetime import datetime
import util
from util import TheDevice, TheMessage
from visualize import Report, TheReport
from util import ProjectPaths, compile, TheXMLConfig, init_config, init_compile_backend
from embed import WordVectors, PretrainModel
from data import Data, TheData
from Layers import Layer, PiLayer, SigmaLayer, ButterflyStage  # Import custom layers from Model.py
from Layers import LinearLayer, InvertibleLinearLayer, AttentionLayer, AssociationLayer, MapppingLayer, LiftingLayer, LoweringLayer, ChunkLayer
from Layers import ColumnUsageTracker, LiftingLayer, CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon
from Layers import SortingLayer, TruthLayer, InterSentenceLayer, SparsityRegularizer, SmoothingRegularizer, ImpenetrableLayer
from parse import quick_parser
from collections import namedtuple as _namedtuple


from Layers import Layer, PiLayer, SigmaLayer # Import custom layers from Model.py
from Layers import LinearLayer, AttentionLayer
from Layers import ColumnUsageTracker, LiftingLayer, CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon
from Layers import Error, TheError

from Spaces import ActiveEncoding, WhereEncoding, WhenEncoding, WhatEncoding, EventEncoding, WordEncoding
from Spaces import Basis, Tensor, Codebook, Embedding
from Spaces import SubSpace, WordSubSpace, Space, InputSpace, PerceptualSpace, ModalSpace, ConceptualSpace, SymbolicSpace, OutputSpace

class Grammar:
    """Single-tier (S) grammar rule catalog (post-rewrite, 2026-04-19).

    The C (conceptual) tier has been merged into S: all compositional
    operations (not, part, intersection, union, lift, lower) are now S-tier
    productions.  The P (perceptual) tier has been removed from the
    grammar.  The C/P query helpers (conceptual(), perceptual(),
    c_methods, p_methods) are retained for backward compatibility with
    ConceptualSyntacticLayer / PerceptualSyntacticLayer, which now
    always receive an empty rule set.

    Owns the rule definitions parsed from XML config.  All learnable
    parameters and rule execution live on the tier-specific SyntacticLayer
    subclasses (SymbolicSyntacticLayer, ConceptualSyntacticLayer,
    PerceptualSyntacticLayer).
    """

    RuleDef = _namedtuple('RuleDef', ['tier', 'canonical', 'arity', 'method_name'])

    def __init__(self):
        self.rules = []
        self.rule_table = {}
        self._configured = False
        self.interpretation = 0.5
        self.thought_free = False

    # -- Rule catalog --------------------------------------------------

    def __len__(self):
        self._ensure_configured()
        return len(self.rules)

    def __getitem__(self, idx):
        self._ensure_configured()
        return self.rules[idx].canonical

    def arity(self, rule_id):
        return self.rules[rule_id].arity

    def method_name(self, rule_id):
        return self.rules[rule_id].method_name

    def tier(self, rule_id):
        return self.rules[rule_id].tier

    def binary_rules(self):
        return [i for i in range(len(self.rules)) if self.rules[i].arity == 2]

    # -- Configuration from XML ----------------------------------------

    def configure(self, grammar_dict):
        self.rules = []
        self._configured = True
        for lhs in ('S',):
            raw = grammar_dict.get(lhs, [])
            if isinstance(raw, str):
                raw = [raw]
            for rhs_text in raw:
                rhs = rhs_text.strip()
                rule_def = self._parse_rule(lhs, rhs)
                self.rules.append(rule_def)
        self.rule_table = {idx: rule.canonical for idx, rule in enumerate(self.rules)}

    def rule_by_id(self, rule_id):
        """Return the canonical production string for a rule_id (0-based)."""
        return self.rule_table[rule_id]

    def _parse_rule(self, lhs, rhs):
        if '(' in rhs:
            func_name = rhs[:rhs.index('(')]
            args_str = rhs[rhs.index('(') + 1:rhs.rindex(')')]
            args = [a.strip() for a in args_str.split(',') if a.strip()]
            arity = len(args)
            canonical = f"{lhs} -> {rhs}"
            return self.RuleDef(lhs, canonical, arity, func_name)
        if rhs == 'epsilon':
            return self.RuleDef(lhs, f"{lhs} -> epsilon", 0, None)
        if rhs == 'S':
            return self.RuleDef(lhs, f"{lhs} -> {rhs}", 1, None)
        raise ValueError(f"Cannot parse grammar rule: {lhs} -> {rhs}")

    _NOOP_GRAMMAR = {'S': 'not(S)'}

    def _ensure_configured(self):
        if self._configured:
            return
        cfg = None
        try:
            candidate = TheXMLConfig.get("WordSpace.language.grammar")
            if isinstance(candidate, dict):
                cfg = candidate
        except (KeyError, AttributeError):
            pass
        if cfg is None:
            cfg = self._NOOP_GRAMMAR
        self.configure(cfg)
        try:
            interp = TheXMLConfig.get("WordSpace.language.interpretation")
            self.interpretation = float(interp)
        except (KeyError, AttributeError, TypeError, ValueError):
            pass

    # -- Rule queries --------------------------------------------------

    def symbolic(self):
        self._ensure_configured()
        return [i for i, r in enumerate(self.rules) if r.tier == 'S']

    def symbolic_transition(self):
        self._ensure_configured()
        for i, r in enumerate(self.rules):
            if r.tier == 'S' and r.method_name is None and r.arity == 1:
                return i
        return None

    @property
    def s_methods(self):
        """Set of method names available on the S (symbolic) tier."""
        return {r.method_name for r in self.rules if r.tier == 'S' and r.method_name}

    def _s_rule_ids(self):
        """Return dict of method_name -> rule_id for S-tier operational rules."""
        result = {}
        for i, r in enumerate(self.rules):
            if r.tier == 'S' and r.method_name is not None:
                result[r.method_name] = i
        return result

    # _conceptual_forward, _symbolic_forward, forward, reverse -- moved to
    # specialized SyntacticLayer subclasses (ConceptualSyntacticLayer,
    # SymbolicSyntacticLayer, PerceptualSyntacticLayer).  Grammar retains
    # only rule catalog, project(), and *Forward/*Reverse operations.

    # composeSyntax, _compose_conceptual, _compose_symbolic -- removed.
    # Soft superposition is now inlined in _conceptual_forward and _symbolic_forward.

    # -- C-tier operations live on SyntacticLayer / ConceptualSyntacticLayer
    # as *Forward / *Reverse method pairs.  See _RULE_METHODS dispatch.
TheGrammar = Grammar()

class SyntacticLayer(Layer):
    """Per-space rule prediction layer for the recursive grammar.

    Each instance handles a subset of the Grammar's rules (one cognitive
    space's rules).  Uses a weight-tied recursive architecture with depth
    embeddings.

    **This layer only predicts rules and generates word tuples.**  It does
    not execute operations on representations -- that is done by the owning
    space's ``projectXxx()`` method, which knows the native representation
    type (activations, vectors, etc.).

    Args:
        nInput:    activation width (number of symbol/concept/percept slots).
        nOutput:   same as nInput.
        rules:     list of global Grammar rule IDs this layer handles
                   (e.g. [1,2,3,4,5] for the symbolic space).
        transition_rule: optional global rule ID for the transition rule
                   (e.g. 6 for S->C).  Included in prediction but signals
                   hand-off to the next space.
        max_depth: maximum derivation depth.
        hidden_dim: width of the shared derivation hidden state.
        grammar:   Grammar instance.
        tau:       Gumbel-softmax temperature.
    """

    # Transition bias scale: (1 - interpretation) * TRANSITION_SCALE is added
    # to the transition rule's logit. The transition rule (S->C or C->P) acts
    # as NOP -- "stop deriving this tier, pass through."
    # Low interpretation -> transition dominates -> no reductions (episodic).
    # High interpretation -> grammar rules fire -> composition (semantic).
    TRANSITION_SCALE = 10.0

    def __init__(self, nInput, nOutput, rules, transition_rule=None,
                 max_depth=12, hidden_dim=256, grammar=None, tau=1.0):
        super().__init__(nInput, nOutput)
        # Store grammar as non-Module attribute to avoid circular nn.Module
        # reference (Grammar owns SyntacticLayers, SyntacticLayers reference
        # Grammar). Using object.__setattr__ bypasses nn.Module.__setattr__
        # which would register it as a submodule.
        if grammar is None:
            grammar = Grammar()
        object.__setattr__(self, 'grammar', grammar)
        self.rules           = list(rules)
        self.transition_rule = transition_rule
        # Build the full set of rule IDs this layer predicts over
        self.all_rules = list(rules)
        if transition_rule is not None and transition_rule not in self.all_rules:
            self.all_rules.append(transition_rule)
        self.num_rules  = len(self.all_rules)
        # Map from local index -> global rule ID
        self.rule_index = {rid: i for i, rid in enumerate(self.all_rules)}
        # Local index of the transition rule (for interpretation bias)
        self.transition_index = (self.rule_index.get(transition_rule)
                                 if transition_rule is not None else None)
        self.max_depth  = max_depth
        self.hidden_dim = hidden_dim
        self.tau        = tau

        # Rule prediction network (weight-tied across depths)
        self.input_proj       = LinearLayer(nInput, hidden_dim)
        self.derivation_layer = LinearLayer(hidden_dim, hidden_dim)
        self.rule_head        = LinearLayer(hidden_dim, self.num_rules)
        self.depth_embed      = nn.Embedding(max_depth, hidden_dim)
        self.activation_fn    = nn.GELU()

        # Xavier initialization so logits start in a numerically stable range.
        # LinearLayer defaults to torch.randn which gives std=1.0; for large
        # dims this produces huge activations that saturate softmax/gumbel.
        for layer in [self.input_proj, self.derivation_layer, self.rule_head]:
            nn.init.xavier_normal_(layer.W)
        nn.init.normal_(self.depth_embed.weight, std=0.02)

        # Register child layers for ergodic dispatch
        self.layers = [self.input_proj, self.derivation_layer, self.rule_head]

    # -- Basis-delegated rule execution ----------------------------

    def _basis(self, subspace):
        """Return the Basis from a SubSpace (or None)."""
        return subspace.basis if subspace is not None else None

    def _mono(self, subspace):
        """True if this subspace uses monotonic logic."""
        b = self._basis(subspace)
        return b is None or b.monotonic

    @staticmethod
    def _expand_mask(mask, feature_dim):
        """Expand a concept-axis mask ``[K]`` to a storage mask ``[feature_dim]``.

        When ``feature_dim == 2 * K`` the input is interpreted as bivector
        storage and the mask is repeated so paired poles ``(2k, 2k+1)`` stay
        co-masked. When ``feature_dim == K`` the mask is used as-is.
        Returns a tensor on the caller's device/dtype.
        """
        if mask is None:
            return None
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, dtype=torch.float32)
        mask = mask.to(dtype=torch.float32)
        K = mask.shape[-1]
        if feature_dim == 2 * K:
            return mask.repeat_interleave(2)
        if feature_dim == K:
            return mask
        # Fallback: if neither matches, broadcast / truncate conservatively.
        if feature_dim < K:
            return mask[:feature_dim]
        return torch.cat([mask, mask.new_zeros(feature_dim - K)], dim=-1)

    def _apply_mask(self, out, mask, subspace=None):
        """Apply a mask either along the feature axis (default) or the
        position axis (when ``subspace`` is provided and ``mask`` aligns
        with ``out.shape[-2]``).

        Feature-axis path: element-wise multiply along the last dim.
        Position-axis path: zero the corresponding rows on
        ``subspace._active`` so ``SubSpace.materialize()`` gating propagates
        the mask downstream. Returns ``out`` unchanged in this case.
        No-op when ``mask is None``.
        """
        if mask is None or not torch.is_tensor(out):
            return out
        if (subspace is not None
                and torch.is_tensor(mask)
                and out.ndim >= 2
                and mask.shape[-1] == out.shape[-2]
                and getattr(subspace, "_active", None) is not None):
            active = subspace._active
            # mask aligns with the N (position) axis of _active = [..., N, M].
            # Append a singleton trailing dim for M; broadcasting handles
            # any leading batch dims automatically.
            m = mask.to(device=active.device, dtype=active.dtype).unsqueeze(-1)
            subspace._active = active * m
            return out
        m = self._expand_mask(mask, out.shape[-1])
        if m is None:
            return out
        return out * m.to(device=out.device, dtype=out.dtype)

    # -- Forward/Reverse dispatch ------------------------------------
    #
    # C-tier ops (invertible): not, intersection, union, lift, lower
    # S-tier ops (lossy, no inverse): equals, part, true, non, swap
    # P-tier ops (invertible): chunk
    #
    # _RULE_METHODS maps rule name -> (forwardName, reverseName|None, binary)

    _RULE_METHODS = {
        'union':        ('unionForward',        'unionReverse',        True),
        'intersection': ('intersectionForward', 'intersectionReverse', True),
        'not':          ('notForward',          'notReverse',          False),
        'equals':       ('equalsForward',       None,                  True),
        'part':         ('partForward',         None,                  True),
        'chunk':        ('chunkForward',        'chunkReverse',        True),
        'true':         ('trueForward',         None,                  False),
        'non':          ('nonForward',          None,                  False),
    }

    def project(self, grammar, rule_id, left, right=None, third=None, subspace=None,
                mask=None):
        """Execute a grammar rule forward. Subclasses override for parametric rules.

        ``mask`` (optional concept-axis Mask of shape ``[K]``) is forwarded
        to the dispatched operator. ``None`` preserves legacy behavior.
        """
        method_name = grammar.rules[rule_id].method_name
        if method_name is None:
            return left  # transition -- pass through

        if method_name in self._RULE_METHODS:
            fn_name, _, binary = self._RULE_METHODS[method_name]
            fn = getattr(self, fn_name)
            if binary:
                if right is not None:
                    return fn(left, right, subspace, mask=mask)
                return left
            return fn(left, subspace, mask=mask)

        return left

    def reverse_project(self, grammar, rule_id, result, right=None, subspace=None,
                        mask=None):
        """Execute a grammar rule inverse. Returns best-effort recovery of left operand."""
        method_name = grammar.rules[rule_id].method_name
        if method_name is None:
            return result

        if method_name in self._RULE_METHODS:
            _, rev_name, binary = self._RULE_METHODS[method_name]
            if rev_name is None:
                return result  # lossy op -- no inverse
            fn = getattr(self, rev_name)
            if binary:
                return fn(result, right, subspace, mask=mask)
            return fn(result, subspace, mask=mask)

        return result

    # -- C-tier: invertible operations -----------------------------

    def notForward(self, left, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.negation(left, monotonic=self._mono(subspace))
        else:
            out = -left
        return self._apply_mask(out, mask, subspace=subspace)

    def notReverse(self, result, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.negation_inverse(result, monotonic=self._mono(subspace))
        else:
            out = -result
        return self._apply_mask(out, mask, subspace=subspace)

    def intersectionForward(self, left, right, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.conjunction(left, right, monotonic=self._mono(subspace))
        else:
            out = torch.min(left, right)
        return self._apply_mask(out, mask, subspace=subspace)

    def intersectionReverse(self, result, right, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.conjunction_inverse(result, right, monotonic=self._mono(subspace))
        else:
            out = result
        return self._apply_mask(out, mask, subspace=subspace)

    def unionForward(self, left, right, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.disjunction(left, right, monotonic=self._mono(subspace))
        else:
            out = torch.max(left, right)
        return self._apply_mask(out, mask, subspace=subspace)

    def unionReverse(self, result, right, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.disjunction_inverse(result, right, monotonic=self._mono(subspace))
        else:
            out = result
        return self._apply_mask(out, mask, subspace=subspace)

    # -- P-tier: chunk (invertible) --------------------------------

    def chunkForward(self, left, right, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.disjunction(left, right, monotonic=True)
        elif right is None:
            out = left
        else:
            out = torch.max(left, right)
        return self._apply_mask(out, mask, subspace=subspace)

    def chunkReverse(self, result, right, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.disjunction_inverse(result, right, monotonic=True)
        else:
            out = result
        return self._apply_mask(out, mask, subspace=subspace)

    # -- S-tier: lossy operations (no inverse) ---------------------

    def equalsForward(self, left, right, subspace, mask=None):
        """S -> equals(S, S): agreement score via concept-level mutual parthood.

        When called from the S-tier with a wired SymbolicSpace back-reference,
        reverse-project both operands from S to C via the owning
        SymbolicSpace's PiLayer, then delegate to the C-tier Basis.equal
        (mutual parthood, scalar=True) on the bitonic concept subspace.
        Otherwise fall back to the local subspace basis or elementwise min.

        Under a mask, agreement is computed only on the selected dims.
        """
        if mask is not None:
            m = self._expand_mask(mask, left.shape[-1])
            m = m.to(device=left.device, dtype=left.dtype)
            denom = m.sum().clamp(min=1.0)
            agree = 1.0 - ((left - right).abs() * m).sum(dim=-1) / denom
            agree = agree.clamp(0.0, 1.0)
            while agree.ndim < right.ndim:
                agree = agree.unsqueeze(-1)
            return self._apply_mask(agree * right, mask, subspace=subspace)

        sym_space = getattr(self, "_symbolic_space", None)
        concept_space = getattr(sym_space, "conceptualSpace", None) if sym_space else None
        concept_basis = None
        if concept_space is not None:
            c_sub = getattr(concept_space, "subspace", None)
            concept_basis = getattr(c_sub, "basis", None) if c_sub else None
        pi = getattr(sym_space, "layer", None) if sym_space else None

        # Reverse-project only when the operand actually looks like a per-symbol
        # vector (last dim == PiLayer's nOutput). Activations over symbol
        # indices [B, K] fall through to the local subspace basis.
        pi_output_dim = getattr(pi, "nOutput", None) if pi is not None else None
        if (concept_basis is not None
                and pi is not None
                and hasattr(pi, "reverse")
                and pi_output_dim is not None
                and left.shape[-1] == pi_output_dim
                and right.shape[-1] == pi_output_dim):
            left_c = pi.reverse(left)
            right_c = pi.reverse(right)
            score = concept_basis.equal(left_c, right_c, monotonic=False, scalar=True)
            while score.ndim < right.ndim:
                score = score.unsqueeze(-1)
            return score * right

        b = self._basis(subspace)
        if b is not None:
            score = b.equal(left, right, monotonic=self._mono(subspace), scalar=True)
            while score.ndim < right.ndim:
                score = score.unsqueeze(-1)
            return score * right
        return torch.min(left, right)

    def partForward(self, left, right, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            score = b.part(left, right, monotonic=self._mono(subspace), scalar=True)
            while score.ndim < right.ndim:
                score = score.unsqueeze(-1)
            out = score * right
        else:
            out = torch.min(left, right)
        return self._apply_mask(out, mask, subspace=subspace)

    # -- S-tier trinity: true / false / non as partition of unity --
    # For x  in  [-1, 1]:  true(x) + false(x) + non(x) = 1
    #   true(x)  = max(0, x)     "I commit: yes"
    #   false(x) = max(0, -x)    "I commit: no"
    #   non(x)   = 1 - |x|       "I commit: indeterminate"
    # Inputs are clamped to [-1, 1] defensively so the partition holds
    # regardless of upstream producer conventions.

    def trueForward(self, left, subspace, mask=None):
        left = torch.clamp(left, -1.0, 1.0)
        b = self._basis(subspace)
        if b is not None:
            out = b.pos(left)
        else:
            out = torch.relu(left)
        return self._apply_mask(out, mask, subspace=subspace)

    def falseForward(self, left, subspace, mask=None):
        """Positive rectification of the negation. The 'no' commitment.

        Partitions with trueForward/nonForward: true + false + non = 1.
        """
        left = torch.clamp(left, -1.0, 1.0)
        b = self._basis(subspace)
        if b is not None:
            out = b.pos(-left)
        else:
            out = torch.relu(-left)
        return self._apply_mask(out, mask, subspace=subspace)

    def nonForward(self, left, subspace, mask=None):
        """Triangular residual: 1 - |x|. The 'indeterminate' commitment.

        Completes the S-tier trinity partition of unity. Replaces the
        earlier sigmoid/zero response which was incompatible with
        true + false + non = 1.
        """
        left = torch.clamp(left, -1.0, 1.0)
        out = 1.0 - left.abs()
        return self._apply_mask(out, mask, subspace=subspace)

    def conjunctionForward(self, left, right, subspace, mask=None):
        """S-tier sentence-level AND. Hadamard conjunction on bitonic activations.

        Distinct from C-tier intersection which composes concepts; this
        composes propositions. Delegates to Basis.conjunction when available
        (which respects sign agreement); falls back to torch.minimum.
        """
        b = self._basis(subspace)
        if b is not None:
            out = b.conjunction(left, right, monotonic=self._mono(subspace))
        else:
            out = torch.minimum(left, right)
        return self._apply_mask(out, mask, subspace=subspace)

    def disjunctionForward(self, left, right, subspace, mask=None):
        """S-tier sentence-level OR. Hadamard disjunction on bitonic activations.

        Distinct from C-tier union which composes concepts; this composes
        propositions. Delegates to Basis.disjunction when available
        (which respects sign agreement); falls back to torch.maximum.
        """
        b = self._basis(subspace)
        if b is not None:
            out = b.disjunction(left, right, monotonic=self._mono(subspace))
        else:
            out = torch.maximum(left, right)
        return self._apply_mask(out, mask, subspace=subspace)

    # -- Rule #2: S-tier slot selectors (what / where / when) -----
    # Parameter-free axis projections. Each zeros non-selected column
    # blocks while preserving shape. The C -> S boundary demux has
    # already put the content in the canonical [what|where|when]
    # layout (see SubSpace.demux); these selectors just mask the
    # non-selected blocks when the activation tensor is vector-shaped.
    #
    # When compose() passes [B, N] scalar norms (non-vector mode) the
    # block structure isn't accessible, so selectors degenerate to
    # identity -- the grammar's axis semantics still hold because the
    # selected vs non-selected dimensions are carried by the
    # subspace's modality tensors rather than the [B, N] activation.

    def _split_widths(self, subspace):
        if subspace is None:
            return None, None, None
        nWhat = getattr(subspace, 'nWhat', None)
        nWhere = getattr(subspace, 'nWhere', 0)
        nWhen = getattr(subspace, 'nWhen', 0)
        return nWhat, nWhere, nWhen

    def whatForward(self, left, subspace, mask=None):
        """Axis selector: keep what-block, zero where/when-blocks."""
        if left.ndim < 3:
            return self._apply_mask(left, mask, subspace=subspace)  # scalar mode -- no columns
        nWhat, nWhere, nWhen = self._split_widths(subspace)
        if nWhat is None or (nWhere == 0 and nWhen == 0):
            return self._apply_mask(left, mask, subspace=subspace)
        out = torch.zeros_like(left)
        out[..., :nWhat] = left[..., :nWhat]
        return self._apply_mask(out, mask, subspace=subspace)

    def whereForward(self, left, subspace, mask=None):
        """Axis selector: keep where-block, zero what/when-blocks."""
        if left.ndim < 3:
            return self._apply_mask(left, mask, subspace=subspace)
        nWhat, nWhere, nWhen = self._split_widths(subspace)
        if nWhat is None or nWhere == 0:
            return self._apply_mask(torch.zeros_like(left), mask, subspace=subspace)
        out = torch.zeros_like(left)
        out[..., nWhat:nWhat + nWhere] = left[..., nWhat:nWhat + nWhere]
        return self._apply_mask(out, mask, subspace=subspace)

    def whenForward(self, left, subspace, mask=None):
        """Axis selector: keep when-block, zero what/where-blocks."""
        if left.ndim < 3:
            return self._apply_mask(left, mask, subspace=subspace)
        nWhat, nWhere, nWhen = self._split_widths(subspace)
        if nWhat is None or nWhen == 0:
            return self._apply_mask(torch.zeros_like(left), mask, subspace=subspace)
        out = torch.zeros_like(left)
        out[..., nWhat + nWhere:] = left[..., nWhat + nWhere:]
        return self._apply_mask(out, mask, subspace=subspace)

    # -- forward: predict rules ------------------------------------

    def forward(self, x):
        """Predict rule distributions and build word tuples.

        Args:
            x: [B, N] activation vector from the space's subspace.

        Returns dict:
            rule_logits:     [B, max_depth, num_rules]  (local indices)
            rule_probs:      [B, max_depth, num_rules]
            predicted_rules: [B, max_depth]             (global rule IDs)
            words:           list of (batch, vector, rule) tuples
        """
        B, N = x.shape

        if self.num_rules == 0:
            empty_logits = torch.zeros(B, self.max_depth, 0,
                                       device=x.device, dtype=x.dtype)
            empty_predicted = torch.zeros(B, self.max_depth,
                                          device=x.device, dtype=torch.long)
            return {
                "rule_logits":     empty_logits,
                "rule_probs":      empty_logits,
                "predicted_rules": empty_predicted,
                "words":           [[] for _ in range(B)],
            }

        h = self.input_proj.forward(x)
        h = self.activation_fn(h)

        depth_ids = torch.arange(self.max_depth, device=x.device)
        depth_vecs = self.depth_embed(depth_ids)

        all_logits = []
        all_probs  = []

        # Transition bias: (1 - interpretation) * scale on the transition
        # rule logit. The transition rule (S->C or C->P) is the NOP -- "stop
        # deriving, pass through." Low interpretation biases toward it.
        interp = self.grammar.interpretation if self.grammar is not None else 0.5
        transition_bias = (1.0 - interp) * self.TRANSITION_SCALE

        for d in range(self.max_depth):
            h = h + depth_vecs[d]
            h = self.derivation_layer.forward(h)
            h = self.activation_fn(h)
            logits = self.rule_head.forward(h)  # [B, num_rules]

            # Bias the transition rule logit. Detach the bias so it
            # doesn't flow gradients -- interpretation is a hyperparameter,
            # the grammar shouldn't learn to predict NOP.
            if self.transition_index is not None and transition_bias > 0:
                logits = logits.clone()
                logits[:, self.transition_index] = (
                    logits[:, self.transition_index].detach() + transition_bias
                )

            if self.training:
                probs = F.gumbel_softmax(logits, tau=self.tau, hard=False)
            else:
                probs = F.softmax(logits, dim=-1)

            all_logits.append(logits)
            all_probs.append(probs)

        rule_logits = torch.stack(all_logits, dim=1)
        rule_probs  = torch.stack(all_probs, dim=1)

        local_predicted = rule_logits.argmax(dim=-1)
        global_predicted = torch.tensor(
            [[self.all_rules[local_predicted[b, d].item()]
              for d in range(self.max_depth)]
             for b in range(B)],
            device=x.device, dtype=torch.long
        )

        active_positions = self._active_positions(x)
        words = self._generate_derivation(global_predicted, active_positions)

        return {
            "rule_logits":     rule_logits,
            "rule_probs":      rule_probs,
            "predicted_rules": global_predicted,
            "words":           words,
        }

    # -- helpers ----------------------------------------------------

    def _active_positions(self, x):
        """Extract per-batch lists of active (nonzero) positions."""
        B = x.shape[0]
        positions = []
        for b in range(B):
            active = torch.nonzero(x[b], as_tuple=False).squeeze(-1)
            positions.append(active.tolist())
        return positions

    def _generate_derivation(self, predicted_rules, active_positions):
        """Build word tuples from predicted rules and active positions."""
        B = predicted_rules.shape[0]
        all_words = []
        for b in range(B):
            rules     = predicted_rules[b].tolist()
            positions = active_positions[b]
            n = len(positions)
            if n == 0:
                continue
            if n == 1:
                terminal = self._find_terminal_rule()
                all_words.append((b, positions[0], terminal))
                continue
            pos_idx = 0
            for rule_id in rules:
                if pos_idx >= n - 1:
                    break
                arity = self.grammar.arity(rule_id)
                if arity != 2:
                    binary = [r for r in self.rules if self.grammar.arity(r) == 2]
                    rule_id = binary[0] if binary else rule_id
                all_words.append((b, positions[pos_idx], rule_id))
                pos_idx += 1
            terminal = self._find_terminal_rule()
            all_words.append((b, positions[-1], terminal))
        return all_words

    def _find_terminal_rule(self):
        """Find the terminal (arity 0) rule in this layer's rule set."""
        for r in self.all_rules:
            if self.grammar.arity(r) == 0:
                return r
        if self.transition_rule is not None:
            return self.transition_rule
        return self.all_rules[0]

    # -- reverse: deterministic tree-walk --------------------------

    def reverse(self, words, nVectors, batch_size):
        """Decode derivation to recover the activation vector."""
        activation = torch.zeros(batch_size, nVectors, device=TheDevice.get())
        for b, v, r in words:
            activation[b, v] = 1.0
        return activation

    # -- utilities -------------------------------------------------

    def set_tau(self, tau):
        """Anneal the Gumbel-softmax temperature."""
        self.tau = tau
class PerceptualSyntacticLayer(SyntacticLayer):
    """P-tier SyntacticLayer: BPE-style chunk merging.

    Owns the chunk codebook and iterative merge loop.  Repeatedly merges
    the top two active positions while the codebook score exceeds the
    entropic threshold.  Stops at word boundaries (whitespace).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_layer = None  # created lazily on first P-tier data

    def _chunk_rule_id(self):
        """Return the rule_id for the P-tier chunk rule, or None."""
        for r in self.all_rules:
            if self.grammar.rules[r].method_name == 'chunk':
                return r
        return None

    def _ensure_chunk_layer(self, nDim):
        """Lazily create the chunk codebook when we first see P-tier data.

        BPE mode and the minimum pair frequency are read from
        ``PerceptualSpace`` config.  The BPE target vocabulary size is
        derived from the codebook's ``nVectors`` (falling back to
        ``chunkTargetVocabSize`` for legacy configs, then 1024).

        Legacy defaults (``chunkBPE=false``) keep the whitespace-boundary
        behavior unchanged; ``chunkBPE=true`` switches the layer into
        greedy longest-match BPE with a learned merge table that grows
        during ``train_step``.
        """
        if self.chunk_layer is None:
            def _pcfg(key, default):
                try:
                    return TheXMLConfig.space("PerceptualSpace", key)
                except (KeyError, TypeError, ValueError):
                    return default
            bpe = bool(_pcfg("chunkBPE", False))
            # Derive target vocab from codebook nVectors; fall back to
            # legacy chunkTargetVocabSize, then 1024.
            try:
                target_vocab = int(_pcfg("nVectors", 0))
            except (TypeError, ValueError):
                target_vocab = 0
            if target_vocab <= 0:
                try:
                    target_vocab = int(_pcfg("chunkTargetVocabSize", 1024))
                except (TypeError, ValueError):
                    target_vocab = 1024
            try:
                min_pair_freq = int(_pcfg("chunkMinPairFrequency", 2))
            except (TypeError, ValueError):
                min_pair_freq = 2
            self.chunk_layer = ChunkLayer(
                nDim,
                bpe=bpe,
                target_vocab_size=target_vocab,
                min_pair_frequency=min_pair_freq,
            ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')

    def compose(self, data, subspace, grammar):
        """Apply P-tier chunk merging.

        In byte mode (subspace._byte_indices set): hard-merge at whitespace
        boundaries first, then run learned BPE, then compact to dense word slots.

        Args:
            data: [B, N, D] percept tensor
            subspace: SubSpace for word recording
            grammar: Grammar instance for subspace access
        Returns:
            data with chunk merges applied (and compacted in byte mode)
        """
        subspace.word = []
        chunk_rid = self._chunk_rule_id()
        if chunk_rid is None or data.ndim != 3:
            return data

        self._ensure_chunk_layer(data.shape[-1])
        cb = self.chunk_layer

        # Byte mode: hard-merge at whitespace boundaries
        byte_indices = getattr(subspace, '_byte_indices', None)
        if byte_indices is not None:
            data, span_meta = cb.hard_merge_spans(data, byte_indices)
            subspace._byte_span_meta = span_meta

        # Learned BPE loop -- boundary check delegates to ChunkLayer
        while True:
            any_merged = False
            pairs = subspace.top_two_of_stack(data)
            for b, (pos1, pos2) in enumerate(pairs):
                if pos1 < 0 or pos2 < 0:
                    continue
                if cb.is_word_boundary(data, b, pos2,
                                       subspace=subspace, byte_indices=byte_indices):
                    continue
                v1, v2 = data[b, pos1], data[b, pos2]
                should, chunk_id = cb.should_merge(v1, v2)
                if not should:
                    continue
                merged, _ = cb.encode(v1, v2)
                data = data.clone()
                data[b, pos1] = merged
                data[b, pos2] = 0.0
                subspace.word.append((b, pos1, pos2, chunk_rid))
                any_merged = True
            if not any_merged:
                break

        # Byte mode: compact sparse -> dense word slots
        if byte_indices is not None:
            nWordSlots = getattr(subspace, '_nWordSlots', data.shape[1])
            where_enc = getattr(subspace, 'whereEncoding', None)
            data, compact_map = cb.compact(data, nWordSlots, span_meta, where_enc)
            subspace._compact_map = compact_map

        return data

    def decompose(self, data, subspace, grammar):
        """Reverse P-tier chunk merges using recorded 4-tuple words.

        In byte mode: un-compacts first, then undoes BPE merges.

        Args:
            data: [B, N, D] tensor (compacted in byte mode)
            subspace: SubSpace with recorded words
            grammar: Grammar instance (unused, kept for interface consistency)
        Returns:
            data with chunk merges undone
        """
        # Byte mode: un-compact dense word slots back to sparse byte positions
        compact_map = getattr(subspace, '_compact_map', None)
        if compact_map is not None and self.chunk_layer is not None:
            byte_indices = getattr(subspace, '_byte_indices', None)
            nByteSlots = byte_indices.shape[1] if byte_indices is not None else data.shape[1]
            data = self.chunk_layer.uncompact(data, compact_map, nByteSlots)

        # Undo BPE merges
        words = subspace.get_words()
        for word in reversed(words):
            if len(word) != 4:
                continue
            b, pos1, pos2, rule_id = word
            if self.chunk_layer is None:
                continue
            merged = data[b, pos1]
            best_sim = -1.0
            best_k = 0
            for k in range(self.chunk_layer.nChunks):
                sim = F.cosine_similarity(
                    merged.unsqueeze(0),
                    self.chunk_layer.merge[k].unsqueeze(0), dim=-1)
                if sim.item() > best_sim:
                    best_sim = sim.item()
                    best_k = k
            v1, v2 = self.chunk_layer.decode(best_k)
            data = data.clone()
            data[b, pos1] = v1
            data[b, pos2] = v2
        return data
class ConceptualSyntacticLayer(SyntacticLayer):
    """C-tier SyntacticLayer: deterministic not + soft-weighted composition.

    Rule application order:
      1. not(C) -- flips negative concepts to positive (mean < 0).
      2. Soft superposition -- remaining rules weighted by predicted probs.

    Owns lift/lower layers.
    """

    _RULE_METHODS = {
        **SyntacticLayer._RULE_METHODS,
        'lift':  ('liftForward',  'liftReverse',  True),
        'lower': ('lowerForward', 'lowerReverse', True),
    }

    def init_conceptual_params(self, concept_dim):
        """Initialize C-tier learnable parameters."""
        self.lifting_layer = LiftingLayer(16, concept_dim)
        self.lowering_layer = LoweringLayer(concept_dim)
        self._symbolic_space = None  # set by BasicModel after init

    # -- C-tier projected ops: lift/lower via PiLayer ----------------

    def _cs_layer(self):
        """PiLayer for concept->symbol projection (ss.layer)."""
        if self._symbolic_space is not None:
            return getattr(self._symbolic_space, 'layer', None)
        return None

    def liftForward(self, left, right, subspace, mask=None):
        """Projected conjunction through symbolic space, back to concept space."""
        cs = self._cs_layer()
        if cs is not None:
            s_a = cs.forward(left)
            s_b = cs.forward(right)
            out = cs.reverse(s_a * s_b)
        else:
            out = left * right
        return self._apply_mask(out, mask, subspace=subspace)

    def liftReverse(self, result, right, subspace, mask=None):
        """Recover first operand: s_a = result / s_b, then PiLayer.reverse."""
        cs = self._cs_layer()
        if cs is not None:
            s_res = cs.forward(result)
            s_b = cs.forward(right)
            s_a = s_res / (s_b + epsilon)
            out = cs.reverse(s_a)
        else:
            out = result / (right + epsilon)
        return self._apply_mask(out, mask, subspace=subspace)

    def lowerForward(self, left, right, subspace, mask=None):
        """Projected disjunction through symbolic space, back to concept space.

        Rescale the sum by 1/2 so it stays in ``(-1, 1)`` -- the operand
        domain that ``PiLayer.reverse`` requires.  Both ``s_a`` and
        ``s_b`` are tanh outputs in ``(-1, 1)``; their unscaled sum lies
        in ``(-2, 2)`` which hits the hard clamp inside ``_to_mult`` and
        silently saturates the backward pass, yielding a badly
        conditioned gradient that leaks into every optimizer step via
        the soft rule mixture.  Training instability on deep chains
        (pairwise compose over conceptualOrder iterations) traces back
        to this saturation.  Dividing by 2 is information-preserving
        (the reverse scales by 2) and keeps the operand strictly inside
        the PiLayer domain.
        """
        cs = self._cs_layer()
        if cs is not None:
            s_a = cs.forward(left)
            s_b = cs.forward(right)
            out = cs.reverse((s_a + s_b) / 2)
        else:
            out = (left + right) / 2
        return self._apply_mask(out, mask, subspace=subspace)

    def lowerReverse(self, result, right, subspace, mask=None):
        """Recover first operand from the rescaled lower forward.

        Given ``result = cs.reverse((s_a + s_b) / 2)`` we have
        ``s_res = cs.forward(result) = (s_a + s_b) / 2``, so
        ``s_a = 2 * s_res - s_b``.  For a valid forward pair both
        ``s_a`` and ``s_b`` lie in ``(-1, 1)`` by construction, so
        ``2 * s_res - s_b`` also lies in ``(-1, 1)`` and the PiLayer
        reverse does not clamp.
        """
        cs = self._cs_layer()
        if cs is not None:
            s_res = cs.forward(result)
            s_b = cs.forward(right)
            s_a = 2 * s_res - s_b
            out = cs.reverse(s_a)
        else:
            out = 2 * result - right
        return self._apply_mask(out, mask, subspace=subspace)

    def project(self, grammar, rule_id, left, right=None, third=None, subspace=None,
                mask=None):
        """Execute a rule. Lift/lower are in _RULE_METHODS via super()."""
        return super().project(grammar, rule_id, left, right, third,
                               subspace=subspace, mask=mask)

    def reverse_project(self, grammar, rule_id, result, right=None, subspace=None,
                        mask=None):
        """Inverse dispatch -- delegates to super()."""
        return super().reverse_project(grammar, rule_id, result, right,
                                       subspace=subspace, mask=mask)

    def compose(self, data, subspace, grammar, target_count=None):
        """Apply C-tier composition.

        Args:
            data: [B, N, D] concept tensor
            subspace: SubSpace for word recording
            grammar: Grammar instance for rule execution
            target_count: If set, use pairwise reduction to this token count
                          (hierarchical mode). None uses cascading accumulator.
        Returns:
            (composed_data, svo_or_None) -- svo is set if ternary lift fired
        """
        subspace.word = []
        self.last_svo = None   # reset per-compose
        self.last_rule_probs = None  # per-depth composable rule probs
        self.last_composable_rules = None  # global rule IDs for columns
        c_rules = grammar._s_rule_ids()
        not_rid = c_rules.get('not')

        # Snapshot codebook indices before any modifications (for decompose)
        basis = getattr(subspace, 'basis', None)
        cb = basis.getW() if basis is not None else None
        if cb is not None and data.shape[-1] == cb.shape[-1]:
            B0, N0, D0 = data.shape
            self._leaf_cb_indices = (
                data.detach().reshape(-1, D0) @ cb.T
            ).argmax(dim=-1).reshape(B0, N0)
        else:
            self._leaf_cb_indices = None

        # Phase 1: deterministic not at top-of-stack
        tops = subspace.top_of_stack(data)
        for b, pos in enumerate(tops):
            if pos < 0:
                continue
            vec = data[b, pos]

            # -- not: negate via Basis.negation (bitonic: -x, self-inverse)
            if not_rid is not None:
                if vec.mean() < 0:
                    data = data.clone()
                    data[b, pos] = self.notForward(vec.unsqueeze(0).unsqueeze(0),
                                                    subspace).squeeze(0).squeeze(0)
                    subspace.add_word(b, pos, not_rid)

        # Dispatch: hierarchical pairwise reduction or cascading accumulator
        if target_count is not None:
            return self._compose_to_target(data, subspace, grammar, target_count,
                                           c_rules, not_rid)

        # Phase 2: soft-weighted composition via SyntacticLayer
        B, N, D = data.shape

        # Guard: skip soft superposition if data dims don't match SyntacticLayer
        expected_n = self.input_proj.nInput
        if N != expected_n:
            return data, self.last_svo

        # Derive [B, N] activation for SyntacticLayer
        activation = torch.norm(data, dim=-1) / math.sqrt(D)

        # Get rule probabilities from SyntacticLayer
        out = super().forward(activation)
        rule_probs = out['rule_probs']  # [B, max_depth, num_rules]

        # Identify composable rules (exclude not -- already applied in Phase 1)
        exclude = {'not'}
        composable_local = []
        composable_global = []
        for local_idx, global_id in enumerate(self.all_rules):
            if grammar.rules[global_id].method_name not in exclude:
                composable_local.append(local_idx)
                composable_global.append(global_id)

        if not composable_global:
            return data, self.last_svo

        # Need at least one binary+ rule for cascading to combine anything;
        # unary rules just return left, consuming leaves without merging.
        has_binary = any(grammar.arity(gid) >= 2 for gid in composable_global)
        if not has_binary:
            return data, self.last_svo

        # Build per-batch active positions
        active_positions = [subspace.active_positions(b, data) for b in range(B)]
        max_leaves = max((len(p) for p in active_positions), default=0)
        if max_leaves == 0:
            return data, self.last_svo

        # Record terminal words for each leaf (transition rule + codebook index)
        cb_indices = self._leaf_cb_indices
        t_rid = self.transition_rule if self.transition_rule is not None else composable_global[0]
        if cb_indices is not None:
            for b in range(B):
                for i, pos in enumerate(active_positions[b]):
                    if i < max_leaves:
                        subspace.add_word(b, pos, t_rid, order=-1,
                                          leaf1=cb_indices[b, pos].item())

        # Extract leaf vectors via masks
        masks = torch.zeros(B, max_leaves, N, device=data.device)
        for b in range(B):
            for i, pos in enumerate(active_positions[b]):
                if i < max_leaves:
                    masks[b, i, pos] = 1.0
        leaf_vecs = masks.unsqueeze(-1) * data.unsqueeze(1)  # [B, L, N, D]

        composed = leaf_vecs[:, 0, :, :]  # start with first leaf
        self.last_composable_rules = composable_global
        depth_probs = []  # collect per-depth renormalized probs

        d = 0
        leaf_idx = 1  # next leaf to consume
        while d < self.max_depth and leaf_idx < max_leaves:
            left = composed
            right = leaf_vecs[:, leaf_idx, :, :]

            # Check if a ternary rule can fire (needs one more leaf)
            has_third = leaf_idx + 1 < max_leaves

            results = []
            for global_id in composable_global:
                a = grammar.arity(global_id)
                if a == 3 and has_third:
                    third = leaf_vecs[:, leaf_idx + 1, :, :]
                    result = self.project(grammar, global_id, left, right, third, subspace=subspace)
                elif a == 2:
                    result = self.project(grammar, global_id, left, right, subspace=subspace)
                else:
                    result = self.project(grammar, global_id, left, subspace=subspace)
                results.append(result)

            results = torch.stack(results, dim=1)  # [B, n_composable, N, D]

            # Extract and renormalize probabilities for composable subset
            probs_d = rule_probs[:, d, :][:, composable_local]  # [B, n_composable]
            probs_d = probs_d / (probs_d.sum(dim=-1, keepdim=True) + 1e-8)
            depth_probs.append(probs_d.detach())                # [B, n_composable]

            # Hard selection in eval mode (exact for decompose); soft mixture in training
            best = probs_d.argmax(dim=-1)  # [B]
            if self.training:
                probs_d = probs_d.unsqueeze(-1).unsqueeze(-1)   # [B, n_composable, 1, 1]
                composed = (probs_d * results).sum(dim=1)       # [B, N, D]
            else:
                # Select argmax rule output per batch element
                idx = best.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
                idx = idx.expand(-1, 1, results.shape[2], results.shape[3])
                composed = results.gather(1, idx).squeeze(1)    # [B, N, D]

            # Record argmax rule as word
            best_global = composable_global[best[0].item()]
            for b in range(B):
                if d < len(active_positions[b]):
                    subspace.add_word(b, active_positions[b][min(d, len(active_positions[b]) - 1)],
                                      composable_global[best[b].item()])

            # Advance: ternary rules consume 2 leaves, others consume 1
            best_arity = grammar.arity(best_global)
            leaf_idx += (2 if best_arity == 3 and has_third else 1)
            d += 1

        if depth_probs:
            self.last_rule_probs = torch.stack(depth_probs, dim=1)  # [B, depths, n_composable]
        return composed, self.last_svo

    def _compose_to_target(self, data, subspace, grammar, target_count,
                           c_rules, not_rid):
        """Reduce active tokens to target_count via independent pairwise grammar reductions.

        Used by the hierarchical forward loop. Each round pairs adjacent active
        positions, applies soft-weighted grammar rules, and zeros consumed tokens
        until the active count reaches target_count.

        Only binary-or-greater rules participate in the pairwise reduce.
        A unary rule by definition cannot merge two operands -- including
        one would make its per-pair output ignore ``right``, and if its
        probability dominates the soft mixture (which happens at init
        for any fresh SyntacticLayer with a biased softmax prior) the
        reduce degenerates to ``composed ~= left`` and ``right``'s
        content is silently discarded.  ``not`` is also excluded since
        it's already applied in Phase 1 of ``compose``.
        """
        B, N, D = data.shape

        # Identify composable rules (exclude not -- already applied in
        # Phase 1 -- AND any rule whose arity is < 2, which can't merge
        # a pair at all; see docstring).
        exclude = {'not'}
        composable_local = []
        composable_global = []
        for local_idx, global_id in enumerate(self.all_rules):
            if grammar.rules[global_id].method_name in exclude:
                continue
            if grammar.arity(global_id) < 2:
                continue
            composable_local.append(local_idx)
            composable_global.append(global_id)

        if not composable_global:
            return data, self.last_svo

        # Get rule probabilities from SyntacticLayer
        activation = torch.norm(data, dim=-1) / math.sqrt(D)
        expected_n = self.input_proj.nInput
        if N == expected_n:
            out = super().forward(activation)
            rule_probs = out['rule_probs']  # [B, max_depth, num_rules]
        else:
            # Dims don't match SyntacticLayer -- use uniform probs
            rule_probs = torch.ones(B, self.max_depth, len(self.all_rules),
                                    device=data.device) / len(self.all_rules)

        # Build per-batch active positions
        active = [subspace.active_positions(b, data) for b in range(B)]

        # Record terminal words for each active leaf (codebook indices from pre-Phase1)
        cb_indices = self._leaf_cb_indices
        t_rid = self.transition_rule if self.transition_rule is not None else composable_global[0]
        if cb_indices is not None:
            for b in range(B):
                for pos in active[b]:
                    subspace.add_word(b, pos, t_rid, order=-1,
                                      leaf1=cb_indices[b, pos].item())

        d = 0

        while d < self.max_depth:
            max_active = max(len(a) for a in active)
            if max_active <= target_count:
                break

            new_data = data.clone()
            for b in range(B):
                positions = active[b]
                new_positions = []
                i = 0
                while i < len(positions) - 1 and (len(positions) - i + len(new_positions)) > target_count:
                    left_pos, right_pos = positions[i], positions[i + 1]
                    left = data[b:b+1, left_pos:left_pos+1, :]
                    right = data[b:b+1, right_pos:right_pos+1, :]

                    results = []
                    for gid in composable_global:
                        a = grammar.arity(gid)
                        if a >= 2:
                            r = self.project(grammar, gid, left, right, subspace=subspace)
                        else:
                            r = self.project(grammar, gid, left, subspace=subspace)
                        results.append(r)
                    results = torch.stack(results, dim=1)  # [1, n_composable, 1, D]

                    probs_d = rule_probs[b:b+1, min(d, rule_probs.shape[1]-1), :]
                    probs_d = probs_d[:, composable_local]
                    probs_d = probs_d / (probs_d.sum(dim=-1, keepdim=True) + 1e-8)

                    best_local = probs_d.argmax(dim=-1)[0].item()
                    if self.training:
                        composed = (probs_d.unsqueeze(-1).unsqueeze(-1) * results).sum(dim=1)
                    else:
                        composed = results[:, best_local]

                    new_data[b, left_pos] = composed[0, 0]
                    new_data[b, right_pos] = 0.0  # zero out consumed

                    best_rid = composable_global[best_local]
                    subspace.add_word(b, left_pos, best_rid, order=d)
                    new_positions.append(left_pos)
                    i += 2

                while i < len(positions):
                    new_positions.append(positions[i])
                    i += 1
                active[b] = new_positions

            data = new_data
            d += 1

        # Return a tensor with the ORIGINAL N shape so downstream
        # (including the reverse path through ``conceptualSpace.reverse``
        # and ``perceptualSpace.reverse``) keeps the slot-axis width it
        # was built for.  Surviving active positions hold the reduced
        # content; consumed positions are left zero.  Compact this down
        # to ``[B, target_count, D]`` at the caller if needed.
        result = torch.zeros(B, N, D, device=data.device)
        for b in range(B):
            for pos in active[b]:
                result[b, pos] = data[b, pos]

        return result, self.last_svo

    def decompose(self, data, subspace, grammar):
        """Reconstruct pre-compose tensor from symbolic word record.

        Terminal words (order == -1) carry codebook indices of the original
        leaf vectors.  Reconstruction looks up each leaf from the codebook
        and places it at its recorded position, producing the exact
        pre-compose tensor without any cached tensors.

        Args:
            data: tensor (same shape as compose output, used for shape/device)
            subspace: SubSpace with recorded words
            grammar: Grammar instance (unused, kept for API compat)
        Returns:
            [B, N, D] tensor with leaf vectors at their original positions
        """
        words = subspace.get_words()
        basis = getattr(subspace, 'basis', None)
        cb = basis.getW() if basis is not None else None
        if cb is None:
            return data  # no codebook -- fall back to identity

        result = torch.zeros_like(data)
        for word in words:
            if word[WordEncoding.ORDER] != -1:
                continue  # skip rule words -- only terminals carry leaves
            b = word[WordEncoding.BATCH]
            pos = word[WordEncoding.VECTOR]
            cb_idx = word[WordEncoding.LEAF1]
            if cb_idx >= 0:
                result[b, pos] = cb[cb_idx]
        return result
class SymbolicSyntacticLayer(SyntacticLayer):
    """S-tier SyntacticLayer: soft-weighted composition on 2D activations.

    All S-tier rules (true, non, swap, equals, part, transition) are applied
    fractionally using learned rule probabilities.

    Owns swap parameters (Sinkhorn-normalised soft permutation).
    """

    def init_swap(self, symbol_dim, n_symbol_slots):
        """Initialize swap and non parameters."""
        swap_size = max(symbol_dim, n_symbol_slots, 1)
        self.swap_marker = nn.Parameter(torch.randn(swap_size) * 0.01)
        self.swap_logits = nn.Parameter(torch.zeros(3, 3))
        self._swap_sinkhorn_iters = 5
        self.non_threshold = nn.Parameter(torch.tensor(0.0))
        # Set by WordSpace._build_symbolic_layer so equalsForward can
        # reverse-project S operands back to C and delegate to Basis.equal.
        self._symbolic_space = None

    def _swap_soft_perm(self):
        M = self.swap_logits
        for _ in range(self._swap_sinkhorn_iters):
            M = M - M.logsumexp(dim=-1, keepdim=True)
            M = M - M.logsumexp(dim=-2, keepdim=True)
        return M.exp()

    def swapForward(self, left, right, subspace=None, mask=None):
        """Soft permutation via Sinkhorn-normalised logits."""
        P = self._swap_soft_perm()
        marker = self.swap_marker.to(left.device)
        if left.ndim == 3:
            m = marker.unsqueeze(0).unsqueeze(0).expand_as(left)
        elif left.ndim == 2:
            m = marker[:left.shape[-1]].unsqueeze(0).expand_as(left)
        else:
            m = marker
        if right is None:
            right = left
        stack = torch.stack([left, right, m], dim=0)
        out = torch.einsum('ij,j...->i...', P, stack)
        return self._apply_mask(out[0], mask, subspace=subspace)

    _RULE_METHODS = {
        **SyntacticLayer._RULE_METHODS,
        'swap':        ('swapForward',        None, True),
        # Rule #1: trinity + coordination (S-tier only)
        'false':       ('falseForward',       None, False),
        'conjunction': ('conjunctionForward', None, True),
        'disjunction': ('disjunctionForward', None, True),
        # Rule #2: symbol demux slot selectors (S-tier only)
        'what':        ('whatForward',        None, False),
        'where':       ('whereForward',       None, False),
        'when':        ('whenForward',        None, False),
        # Rule #3: query (contradiction marker at accumulation point)
        'query':       ('queryForward',       None, True),
    }

    # Rule #3: Norm-drop threshold. If a new rule-application result
    # would reduce the accumulator's norm below this fraction of its
    # current value, the accumulation point interprets it as symbolic
    # contradiction and emits a query word + preserves the existing
    # accumulator instead of absorbing the cancelling contribution.
    # Tuning note: start at 0.1 (90% reduction) per plan; too tight
    # emits spurious queries on legitimate near-cancellations, too
    # loose lets real contradictions collapse silently.
    _QUERY_NORM_DROP_RATIO = 0.1

    def queryForward(self, left, right, subspace=None, mask=None):
        """Query: return the preserved accumulator operand.

        The query marker is pushed onto WordSubSpace at the
        accumulation point (see `compose()`), not by this forward.
        When the parse tree is re-evaluated downstream, `queryForward`
        returns the first operand -- the accumulator state that was
        preserved when the cancelling contribution arrived. The second
        operand (the dropped symbol) exists only in the parse-tree
        record and is unused here.
        """
        return self._apply_mask(left, mask, subspace=subspace)

    def project(self, grammar, rule_id, left, right=None, subspace=None, mask=None):
        """Execute a rule via _RULE_METHODS dispatch."""
        return super().project(grammar, rule_id, left, right,
                               subspace=subspace, mask=mask)

    def compose(self, data, subspace, grammar):
        """Apply S-tier soft-weighted composition.

        Args:
            data: [B, N] or [B, N, D] symbol activation tensor
            subspace: SubSpace for word recording
            grammar: Grammar instance for rule execution
        Returns:
            composed symbol activations, same shape as input
        """
        subspace.word = []
        if data.ndim == 3:
            # 3D vector mode: extract norms for grammar, scale vectors by result
            norms = data.norm(dim=-1)                    # [B, N]
            composed_norms = self.compose(norms, subspace, grammar)  # [B, N]
            scale = composed_norms / (norms + 1e-8)      # [B, N]
            return data * scale.unsqueeze(-1)             # [B, N, D]

        B, N = data.shape

        # Guard: skip soft superposition if data dims don't match SyntacticLayer
        expected_n = self.input_proj.nInput
        if N != expected_n:
            return data

        # Get rule probabilities from SyntacticLayer
        out = super().forward(data)
        rule_probs = out['rule_probs']  # [B, max_depth, num_rules]
        all_rules = self.all_rules

        # Build per-batch active positions
        active_positions = [subspace.active_positions(b, data) for b in range(B)]
        max_leaves = max((len(p) for p in active_positions), default=0)
        if max_leaves == 0:
            return data

        # Extract leaf activations via masks
        masks = torch.zeros(B, max_leaves, N, device=data.device)
        for b in range(B):
            for i, pos in enumerate(active_positions[b]):
                if i < max_leaves:
                    masks[b, i, pos] = 1.0
        leaf_acts = masks * data.unsqueeze(1)  # [B, L, N]

        composed = leaf_acts[:, 0, :]  # start with first leaf

        # Rule #3 state: track the rule applied at the previous step
        # per batch row, so a query push at the norm-drop site has a
        # referent for "what was the preserved accumulator's rule".
        # -1 = no prior rule (accumulator is still a raw leaf).
        last_rule_per_batch = [-1 for _ in range(B)]
        query_rid = None
        for _idx, _gid in enumerate(all_rules):
            if grammar.rules[_gid].method_name == 'query':
                query_rid = _gid
                break

        for d in range(min(self.max_depth, max(max_leaves - 1, 1))):
            if d + 1 >= max_leaves:
                break
            left = composed
            right = leaf_acts[:, d + 1, :]

            results = []
            for rule_id in all_rules:
                a = grammar.arity(rule_id)
                if a == 2:
                    result = self.project(grammar, rule_id, left, right, subspace=subspace)
                else:
                    result = self.project(grammar, rule_id, left, subspace=subspace)
                results.append(result)

            results = torch.stack(results, dim=1)  # [B, num_rules, N]
            probs_d = rule_probs[:, d, :]           # [B, num_rules]

            # -- Rule #3: norm-drop detection at the accumulation point --
            # Any symbolic contradiction (A  and  not A, true(A)  and  false(A),
            # axis-restricted variants) manifests as a significant drop
            # in the accumulator norm when the candidate is mixed in.
            # We detect the symptom here, push a `query` marker onto the
            # word-stream buffer, and preserve the prior accumulator.
            # See plan: "Rule #3 -- Query at S-tier".
            candidate = (probs_d.unsqueeze(-1) * results).sum(dim=1)  # [B, N]
            prev_norm = left.norm(dim=-1)         # [B]
            cand_norm = candidate.norm(dim=-1)    # [B]
            drop_threshold = self._QUERY_NORM_DROP_RATIO * prev_norm
            # Only fire when there was a real accumulator to cancel
            # (prev_norm > 1e-6), and the candidate norm is below the
            # drop threshold.
            query_mask = (prev_norm > 1e-6) & (cand_norm < drop_threshold)
            if query_mask.any():
                # For batches where query fires: preserve the old accumulator.
                # For batches where it does not: use the candidate mixture.
                mask = query_mask.unsqueeze(-1).expand_as(candidate)  # [B, N]
                composed = torch.where(mask, left, candidate)
                # Push query marker onto the word-stream buffer for
                # each batch row that tripped the check. The leaves
                # record the rule identities of the preserved side
                # (left_rule_id) and the incoming rule that would have
                # caused the cancellation (right_rule_id).
                best_for_push = probs_d.argmax(dim=-1)  # [B]
                word_sub = getattr(self, 'word_subspace', None)
                for b in range(B):
                    if not bool(query_mask[b].item()):
                        continue
                    left_rid = last_rule_per_batch[b]
                    right_rid = int(best_for_push[b].item())
                    right_gid = all_rules[right_rid] if right_rid < len(all_rules) else -1
                    if query_rid is not None and word_sub is not None:
                        word_sub.push(b, query_rid,
                                      leaves=(left_rid, right_gid, -1))
                    # Preserve the prior rule identity for this batch
                    # row -- the accumulator did not advance.
            else:
                composed = candidate

            # Record argmax rule as word
            best = probs_d.argmax(dim=-1)  # [B]
            for b in range(B):
                if d < len(active_positions[b]):
                    subspace.add_word(b, active_positions[b][d], all_rules[best[b].item()])
                    # Track last advancing rule per batch for future
                    # query-push referents -- only update for rows
                    # that did not trip the query mask this step.
                    if not bool(query_mask[b].item()):
                        last_rule_per_batch[b] = all_rules[best[b].item()]

        return composed

    def decompose(self, data, subspace, grammar):
        """Reverse S-tier operations using recorded word tuples.

        Args:
            data: [B, N] or [B, N, D] tensor (same shape as compose output)
            subspace: SubSpace with recorded words
            grammar: Grammar instance for rule info
        Returns:
            data with grammar operations undone (best-effort)
        """
        words = subspace.get_words()
        for word in reversed(words):
            if len(word) < 3:
                continue
            b = word[WordEncoding.BATCH]
            pos = word[WordEncoding.VECTOR]
            rule_id = word[WordEncoding.RULE]
            rule = grammar.rules[rule_id]
            if rule.method_name in ('non', 'union', 'intersection',
                                    'lift', 'lower'):
                pass  # Non-invertible
            elif rule.method_name is not None:
                pass  # Not cleanly invertible
        return data


class PoSStack:
    """Ordered push/pop stack of PoS vectors, fixed dim per row."""

    def __init__(self, dim):
        self._dim = dim
        self._entries = []

    def push(self, vec):
        assert vec.shape == (self._dim,), (
            f"PoSStack dim={self._dim}, got vec shape {tuple(vec.shape)}"
        )
        self._entries.append(vec)

    def pop(self):
        return self._entries.pop()

    def depth(self):
        return len(self._entries)

    def flatten(self):
        if not self._entries:
            return torch.zeros(0)
        return torch.cat(self._entries, dim=0)


class ReconstructionStack:
    """Tuple stack of (rule_id, word_id) entries for surface reconstruction.

    Temporary placeholder until generation-from-meaning is solved. Not
    consumed by the rule predictor or sentence prediction.
    """

    def __init__(self):
        self._entries = []

    def push(self, rule_id, word_id):
        self._entries.append((int(rule_id), int(word_id)))

    def peek(self):
        return self._entries[-1]

    def pop(self):
        return self._entries.pop()

    def depth(self):
        return len(self._entries)


class WordSpace(Space):
    """Service space that owns the word-stream buffer, the SyntacticLayers,
    the truth store, and the inter-sentence discourse substrate.

    Runtime-parallel to PerceptualSpace / ConceptualSpace / SymbolicSpace
    but functionally a buffer + composition dispatcher. WordSpace owns
    the SyntacticLayers directly; home spaces receive ``wordSpace`` as
    a per-call parameter on ``forward(vspace, wordSpace=...)`` /
    ``reverse(vspace, wordSpace=...)`` and reach the layers via the
    explicit per-tier methods ``forwardPercepts`` / ``forwardConcepts``
    / ``forwardSymbols`` (and the matching ``reverse*`` variants). The
    layers push their word records into ``self.subspace`` (a
    ``WordSubSpace``) via a back-reference set at construction time, so
    ConceptualSpace can read a muxed view of machine state that
    includes percepts, symbols, and words.

    One unified constructor builds everything: WordSubSpace, all three
    SyntacticLayers, TruthLayer, and (conditionally) DiscourseSpace.
    XML config drives the truth-store capacity and discourse-prediction
    gating.

    Per-sentence lifecycle: BasicModel calls ``clear_sentence()`` at
    sentence boundaries to rewind the buffer.

    Subclasses ``Space`` for the universal training contract
    (``getParameters`` / ``paramUpdate`` / ``set_sigma``), but
    bypasses ``Space.__init__`` because there is no factory-style
    input/output/codebook shape tuple -- the subspace is a
    ``WordSubSpace`` built from the symbolic peer's column layout
    and all children are registered directly into ``self.layers`` /
    ``self.params`` so the inherited training-contract walks still
    work.
    """

    name = "Words"
    config_section = "WordSpace"

    def __init__(self, perceptualSpace, conceptualSpace, symbolicSpace,
                 nPercepts, nConcepts, nSymbols,
                 concept_dim, symbol_dim):
        # Bypass Space.__init__ -- WordSpace doesn't fit the factory
        # style. Call nn.Module directly and populate the Space-contract
        # fields by hand.
        nn.Module.__init__(self)

        # 1. Grammar must be configured before any SyntacticLayer
        # construction can resolve rule sets / transition rules.
        TheGrammar._configured = False
        TheGrammar._ensure_configured()
        grammar = TheGrammar

        # 2. Size WordSubSpace from SymbolicSpace's subspace column
        # layout so downstream consumers of wordSpace.read() concat
        # cleanly with peer tensors.
        sub = symbolicSpace.subspace
        nWhere = int(getattr(sub, 'nWhere', 0) or 0)
        nWhen  = int(getattr(sub, 'nWhen',  0) or 0)
        nWhat  = int(getattr(sub, 'nWhat',  0) or 0)
        muxed  = int(getattr(sub, 'muxedSize', nWhat + nWhere + nWhen)
                     or (nWhat + nWhere + nWhen))
        self.subspace = WordSubSpace(
            nDim=muxed, nWhat=nWhat, nWhere=nWhere, nWhen=nWhen,
            max_depth=256, max_arity=3, batch=1,
        )

        # 3. Space-contract fields.
        self.layers = nn.ModuleList()
        self.params = []
        self.wordSpace = None                        # no parent wordSpace
        self.nDim = muxed
        self.nWhat = nWhat
        self.nWhere = nWhere
        self.nWhen = nWhen
        self.muxedSize = muxed
        self.inputShape  = [0, muxed]
        self.outputShape = [0, muxed]
        self.spaceShape  = [0, muxed]

        # 4. Layer slots (filled below).
        self.perceptualSyntacticLayer = None
        self.conceptualSyntacticLayer = None
        self.symbolicSyntacticLayer = None

        # 5. Build the three SyntacticLayers, each of which back-wires
        # the home space's ``wordSpace`` routing pointer.
        if perceptualSpace is not None:
            self._build_perceptual_layer(perceptualSpace, nPercepts, grammar)
        if conceptualSpace is not None:
            self._build_conceptual_layer(
                conceptualSpace, nConcepts, grammar, concept_dim)
        if symbolicSpace is not None:
            self._build_symbolic_layer(
                symbolicSpace, nSymbols, grammar, symbol_dim)

        # 6. TruthLayer -- shared truth store for symbolic activations.
        # Lives on WordSpace so SymbolicSpace doesn't have to carry it
        # alongside its already heavy pi/sort/codebook machinery.
        try:
            max_truths = int(TheXMLConfig.get("WordSpace.truthMaxEntries"))
        except (KeyError, TypeError, ValueError):
            max_truths = 1024
        self.truth_layer = TruthLayer(symbol_dim, max_truths=max_truths)
        if self.truth_layer not in self.layers:
            self.layers.append(self.truth_layer)
        for p in self.truth_layer.parameters():
            if all(p is not q for q in self.params):
                self.params.append(p)

        # 6b. PoS codebook -- 64 prototypes x 4 dims, direct addressing.
        # Not registered in self.layers (no training loop integration yet);
        # the VectorQuantize backend provides the nn.Module bookkeeping.
        self.pos_codebook = Codebook()
        self.pos_codebook.create(
            nInput=0,       # input-side width unused for direct addressing
            nVectors=64,    # nPoS
            nDim=4,         # nPoSDim
            customVQ=True,
            monotonic=False,
            passThrough=False,
        )

        # 6c. PoS stack -- push/pop store for PoS vectors during parsing.
        self.pos_stack = PoSStack(dim=4)  # matches pos_codebook nDim

        # 6c'. Reconstruction stack -- (rule_id, word_id) entries for surface
        # reconstruction. Placeholder until generation-from-meaning is solved.
        self.reconstruction_stack = ReconstructionStack()

        # 6d. Rule predictor -- nonlinear head over the flattened PoS stack.
        # Task 4.2: emits softmax logits over TheGrammar.rule_table, the
        # authoritative rule-id space (includes START/S/P productions);
        # len(symbolic()) would be only the S-tier subset and would under-size
        # the output.
        #
        # Option A (per task notes): torch.nn stdlib Sequential with a Tanh
        # nonlinearity -- no new layer type added to Layers.py. Stash
        # in_features on the WordSpace because Sequential has no such attr.
        n_rules = len(TheGrammar.rule_table)
        self.n_rules = n_rules
        max_depth = int(nPercepts)
        pos_dim = 4  # matches pos_codebook / pos_stack nDim
        rule_in_features = max_depth * pos_dim
        self._rule_predictor_in_features = rule_in_features
        self.rule_predictor = nn.Sequential(
            nn.Linear(rule_in_features, rule_in_features),
            nn.Tanh(),
            nn.Linear(rule_in_features, n_rules),
        )
        for p in self.rule_predictor.parameters():
            if all(p is not q for q in self.params):
                self.params.append(p)

        # 7. DiscourseSpace -- optional inter-sentence substrate.
        # Gated on <architecture><training><sentencePrediction>; tasks
        # without inter-sentence structure (XOR, MNIST) leave it off.
        # The contrastive loss has no learnable parameters; the three
        # training keys that survive are ``sentenceContextWindow``
        # (recent buffer depth used for the attractive centroid),
        # ``sentenceCentroidHistory`` (older centroids used for the
        # repulsive force), and ``sentenceLambda`` (repulsive scale).
        self.discourse = None
        if bool(TheXMLConfig.training("sentencePrediction", False)):
            try:
                n_sym_rows = int(symbolicSpace.outputShape[0])
            except (AttributeError, IndexError, TypeError):
                n_sym_rows = int(getattr(symbolicSpace, 'nVectors', 0) or 0)
            if n_sym_rows > 0 and muxed > 0:
                context_window = int(TheXMLConfig.training(
                    "sentenceContextWindow", 12) or 12)
                centroid_history = int(TheXMLConfig.training(
                    "sentenceCentroidHistory", 3) or 3)
                sentence_lambda = float(TheXMLConfig.training(
                    "sentenceLambda", 1.01) or 1.01)
                self.discourse = InterSentenceLayer(
                    n_symbols=n_sym_rows,
                    max_depth=int(getattr(
                        self.subspace, 'max_depth', 256) or 256),
                    n_dim=muxed,
                    context_window=context_window,
                    centroid_history=centroid_history,
                    lam=sentence_lambda,
                    concept_dim=int(concept_dim),
                )
                self.layers.append(self.discourse)
                for p in self.discourse.parameters():
                    if all(p is not q for q in self.params):
                        self.params.append(p)

    # -- PoS helpers --------------------------------------------------
    def pos_lookup(self, active_symbols):
        """Return the 4-dim PoS vector for the given active-symbol pattern.

        Uses nearest-neighbor lookup against the pos_codebook weight matrix
        (cosine distance) so the result is always deterministic and doesn't
        require running a full VQ forward pass (which updates codebook state).

        Args:
            active_symbols: 1-D tensor of shape [N], typically resolved
                activations from SymbolicSpace.resolve().

        Returns:
            Tensor of shape (4,) -- the matching PoS prototype row.
        """
        w = self.pos_codebook.getW()  # [64, 4]
        # Project active_symbols to a query vector by taking the weighted
        # mean of codebook rows, then snap to the nearest row.
        # For a pure lookup, we map from symbol-index space to a scalar
        # by summing activation * codebook row for each symbol slot
        # (mod number of codebook rows).
        n_sym = active_symbols.shape[0]
        n_cb = w.shape[0]
        # Compute a soft query: sum active_symbols[i] * w[i % n_cb]
        indices = torch.arange(n_sym, device=w.device) % n_cb
        query = (active_symbols.to(w.device).unsqueeze(-1) * w[indices]).sum(0)  # [4]
        # Nearest-neighbor snap: argmax cosine similarity
        query_norm = query / (query.norm() + 1e-8)
        w_norm = w / (w.norm(dim=-1, keepdim=True) + 1e-8)
        idx = (w_norm @ query_norm).argmax()
        return w[idx]

    # -- rule predictor ------------------------------------------------
    def predict_rule(self):
        """Emit softmax logits over the rule table from the full PoS stack.

        Reads ``self.pos_stack.flatten()`` as a 1-D tensor of length
        ``depth * pos_dim``.  Zero-pads up to
        ``self._rule_predictor_in_features`` when the stack is shallower
        than ``max_depth``; truncates (keeping the most recent frames) when
        the stack has overflowed, so the head always sees a fixed-width
        window of the top ``max_depth`` PoS vectors.  Returns a tensor of
        shape ``(n_rules,)`` suitable for ``torch.softmax`` / CE loss.
        """
        assert self.rule_predictor[-1].out_features == len(TheGrammar.rule_table), (
            "Grammar reconfigured after WordSpace construction; rule_predictor stale"
        )
        flat = self.pos_stack.flatten()
        target_len = self._rule_predictor_in_features
        numel = flat.numel()
        # Pick a device/dtype anchor that follows the rule_predictor, so
        # padding lives on the same device whether or not the stack was
        # previously populated.
        first_param = next(self.rule_predictor.parameters())
        if numel < target_len:
            pad = torch.zeros(
                target_len - numel,
                device=flat.device if numel > 0 else first_param.device,
                dtype=flat.dtype if numel > 0 else first_param.dtype,
            )
            if numel == 0:
                flat = pad
            else:
                flat = torch.cat([flat, pad])
        elif numel > target_len:
            # Stack deeper than the configured max_depth window: keep the
            # most recent frames (tail slice) so the predictor always sees
            # the top of the stack.
            flat = flat[numel - target_len:]
        return self.rule_predictor(flat.unsqueeze(0)).squeeze(0)

    def predict_rule_hard(self):
        """Return argmax rule_id for inference.

        Detached from autograd (wraps predict_rule in no_grad). If gradients
        through the argmax path are ever needed (e.g., REINFORCE baseline,
        Gumbel-argmax), call predict_rule().argmax() directly instead.
        """
        with torch.no_grad():
            return int(self.predict_rule().argmax().item())

    # -- reconstruction stack -----------------------------------------
    def record_derivation(self, rule_id, word_id):
        """Record a (rule_id, word_id) derivation step on the reconstruction stack.

        Placeholder surface until generation-from-meaning is solved. The
        stack is not consumed by the rule predictor or sentence prediction.
        """
        self.reconstruction_stack.push(rule_id, word_id)

    # -- truth-modulated loss -----------------------------------------
    def truth_modulated_loss(self, total_loss, symbolic_space,
                             symbol_acts=None, universality_score=None,
                             luminosity_weight=0.1, universality_weight=0.1,
                             truth_loss_weight=0.0,
                             allow_excluded_middle=1,
                             allow_contradiction=0,
                             balance_weight=0.1):
        """Apply the WordSpace-owned TruthLayer modulation to a loss.

        The transform has two parts:

        1. **Multiplicative modulation** -- penalize irrational and
           unkind propositions by scaling ``total_loss`` by
           ``(1 + lum_w * (1 - lum_norm) + u_w * (1 - u_norm))``,
           where ``lum_norm = luminosity(symbolic_space.layer).clamp(0, 1)``
           and ``u_norm = universality_score.clamp(-1, 1)`` (or 0
           when the caller has no universality score cached yet).

        2. **Additive falsity penalty** -- when
           ``truth_loss_weight > 0`` and the caller provides
           committed symbol activations, add
           ``truth_loss_weight * falsity_penalty(symbol_acts, basis)``
           using ``symbolic_space.subspace.basis``.  ``symbol_acts``
           should be the last entry of the model's ``symbol_states``
           cache -- the post-pi activations from the final Sigma-Pi
           iteration.  Both operands of the
           disjunction then live in symbol space by construction
           (stored truths were also recorded from symbol-space
           activations in ``SymbolicSpace.forwardEnd``).

        Returns ``total_loss`` unchanged when the TruthLayer is
        absent or empty (bootstrap case with no truths recorded
        yet).  The caller is responsible for only invoking this in
        train mode -- the method itself has no ``train`` flag.

        All inputs that reach outside WordSpace (``symbolic_space``,
        ``symbol_acts``, ``universality_score``, the three weights)
        are passed explicitly so WordSpace never needs a back-
        reference to the model.
        """
        if self.truth_layer is None or len(self.truth_layer) == 0:
            return total_loss

        lum = self.truth_layer.luminosity(symbolic_space.layer)
        lum_norm = lum.clamp(0, 1)
        if universality_score is not None:
            u_norm = universality_score.clamp(-1, 1)
        else:
            u_norm = torch.tensor(0.0, device=total_loss.device)

        total_loss = total_loss * (1 + luminosity_weight * (1 - lum_norm)
                                     + universality_weight * (1 - u_norm))

        if truth_loss_weight > 0 and symbol_acts is not None:
            basis = getattr(
                getattr(symbolic_space, 'subspace', None), 'basis', None)
            if basis is not None:
                truth_penalty = self.truth_layer.falsity_penalty(
                    symbol_acts, basis)
                total_loss = total_loss + truth_loss_weight * truth_penalty

        # Quaternary-corner balance penalty: discourages forbidden corners
        # (N, B) on committed symbol activations. Runs whenever the knobs
        # select a non-permissive corner and symbol_acts are provided.
        # Under the current SymbolicSpace layout each row is
        # [pos_pole, neg_pole, where..., when...] -- slice the leading
        # bivector before passing to the paired-index penalty so that
        # positional-template dims don't spuriously register as N/B.
        # See basicmodel/doc/BuddhistParallels.md and doc/Spaces.md.
        wants_balance = (int(allow_excluded_middle) == -1
                         or int(allow_contradiction) == 0)
        if (balance_weight > 0 and wants_balance
                and symbol_acts is not None
                and torch.is_tensor(symbol_acts)
                and symbol_acts.shape[-1] >= 2):
            bivector = symbol_acts[..., :2]
            balance = self.truth_layer.tetralemma_balance_penalty(
                bivector,
                allow_excluded_middle=int(allow_excluded_middle),
                allow_contradiction=int(allow_contradiction))
            total_loss = total_loss + balance_weight * balance

        return total_loss

    # -- wiring -------------------------------------------------------
    def attach_codebook_host(self, host):
        """Wire the host space to WordSubSpace for push() gating.

        Typically the ``SymbolicSpace`` instance. Stored as a non-Module
        back-reference so that ``push()`` can be called from compose().
        Rule-identity vectors in the word buffer are written as zeros
        (the empty-slot sentinel) because the learnable rule_codebook
        has been removed; only the parse-tree ledger retains rule_id.
        """
        self.subspace.attach_codebook_host(host)

    def attach_layer(self, kind, layer):
        """Register a pre-built SyntacticLayer as this WordSpace's
        ``<kind>SyntacticLayer``.

        Sets ``layer.word_subspace`` as a back-reference so compose()
        can push onto the shared buffer, appends the layer to
        ``self.layers`` for ``Space.paramUpdate`` delegation, and
        merges its parameters into ``self.params`` for the curated
        ``Space.getParameters`` walk.
        """
        if layer is None:
            return
        attr = f'{kind}SyntacticLayer'
        if not hasattr(self, attr):
            raise ValueError(
                f"WordSpace: unknown syntactic kind {kind!r}; "
                f"expected one of 'perceptual', 'conceptual', 'symbolic'")
        setattr(self, attr, layer)
        layer.word_subspace = self.subspace
        if layer not in self.layers:
            self.layers.append(layer)
        for p in layer.parameters():
            if all(p is not q for q in self.params):
                self.params.append(p)

    # -- private factory helpers: build + wire SyntacticLayers --------
    def _resolve_hidden_dim(self, n_slots):
        try:
            configured = int(TheXMLConfig.get("WordSpace.syntacticHiddenDim"))
            if configured > 0:
                return configured
        except (KeyError, TypeError, ValueError):
            pass
        return min(256, max(64, n_slots * 4))

    def _build_perceptual_layer(self, space, n_slots, grammar):
        layer = PerceptualSyntacticLayer(
            nInput=n_slots, nOutput=n_slots,
            rules=[],
            transition_rule=None,
            max_depth=max(n_slots - 1, 1),
            hidden_dim=self._resolve_hidden_dim(n_slots),
            grammar=grammar,
        )
        self.attach_layer('perceptual', layer)
        space.attach_wordSpace(self)
        return layer

    def _build_conceptual_layer(self, space, n_slots, grammar, concept_dim):
        layer = ConceptualSyntacticLayer(
            nInput=n_slots, nOutput=n_slots,
            rules=[],
            transition_rule=None,
            max_depth=max(n_slots - 1, 1),
            hidden_dim=self._resolve_hidden_dim(n_slots),
            grammar=grammar,
        )
        layer.init_conceptual_params(concept_dim)
        self.attach_layer('conceptual', layer)
        space.attach_wordSpace(self)
        return layer

    def _build_symbolic_layer(self, space, n_slots, grammar, symbol_dim):
        layer = SymbolicSyntacticLayer(
            nInput=n_slots, nOutput=n_slots,
            rules=grammar.symbolic(),
            transition_rule=grammar.symbolic_transition(),
            max_depth=max(n_slots - 1, 1),
            hidden_dim=self._resolve_hidden_dim(n_slots),
            grammar=grammar,
        )
        layer.init_swap(symbol_dim, n_slots)
        layer._symbolic_space = space
        self.attach_codebook_host(space)
        self.attach_layer('symbolic', layer)
        space.attach_wordSpace(self)
        return layer

    # -- per-tier composition methods ---------------------------------
    def forwardPercepts(self, data, subspace):
        """P-tier compose. Side effect: word-emitting pushes onto the
        buffer. Returns the composed activation.
        """
        layer = getattr(self, 'perceptualSyntacticLayer', None)
        if layer is None:
            return data
        return layer.compose(data, subspace, TheGrammar)

    def forwardConcepts(self, data, subspace, target_count=None):
        """C-tier compose. ``ConceptualSyntacticLayer.compose`` may
        return ``(data, svo)`` when a ternary lift fires; we preserve
        that tuple contract so callers (ConceptualSpace.forward) can
        stash the SVO on themselves for the ``last_svo`` property.

        ``target_count`` routes into the pairwise reduction path in
        ``ConceptualSyntacticLayer._compose_to_target``.  Pairwise
        reduction slices each slot to ``[1, 1, D]`` before invoking
        the grammar's binary rules, which degenerates the per-slot
        PiLayer to a pure ``D->D`` map and lets the two operands'
        content actually merge into one slot.  The cascading default
        (``target_count=None``) keeps full ``[B, N, D]`` shapes and so
        cannot move information across the slot axis -- fine for
        sparse-representation use but useless whenever the two
        operands live in different slots.
        """
        layer = getattr(self, 'conceptualSyntacticLayer', None)
        if layer is None:
            return data, None
        result = layer.compose(data, subspace, TheGrammar, target_count=target_count)
        if isinstance(result, tuple):
            return result
        return result, None

    def forwardSymbols(self, data, subspace):
        """S-tier compose. Includes the Rule #2 demux side effect: the
        muxed [B, N, D] symbol tensor gets split into what/where/when
        modality slots before compose runs, so slot selectors see
        axis-separated state.
        """
        layer = getattr(self, 'symbolicSyntacticLayer', None)
        if layer is None:
            return data
        if data.ndim == 3 and data.shape[-1] == subspace.muxedSize:
            subspace.demux(data)
        return layer.compose(data, subspace, TheGrammar)

    def reversePercepts(self, data, subspace):
        layer = getattr(self, 'perceptualSyntacticLayer', None)
        if layer is None:
            return data
        return layer.decompose(data, subspace, TheGrammar)

    def reverseConcepts(self, data, subspace):
        layer = getattr(self, 'conceptualSyntacticLayer', None)
        if layer is None:
            return data
        return layer.decompose(data, subspace, TheGrammar)

    def reverseSymbols(self, data, subspace):
        layer = getattr(self, 'symbolicSyntacticLayer', None)
        if layer is None:
            return data
        return layer.decompose(data, subspace, TheGrammar)

    # -- buffer access + lifecycle ------------------------------------
    def read(self):
        """Return the fixed-width stack tensor for ConceptualSpace to
        concat with percepts and symbols.
        """
        return self.subspace.read()

    def clear_sentence(self):
        """Reset the stack at sentence boundaries."""
        self.subspace.clear()

    def get_blocks(self, b=0):
        """Return the parse-tree ledger for batch row `b`."""
        return self.subspace.get_blocks(b)

    def ensure_batch(self, batch):
        """Resize the underlying buffer to match a new batch size."""
        self.subspace.ensure_batch(batch)
