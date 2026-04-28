

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
from Layers import Layer, PiLayer, SigmaLayer  # Import custom layers from Model.py
from Layers import LinearLayer, InvertibleLinearLayer, AttentionLayer, AssociationLayer, MapppingLayer, ChunkLayer
from Layers import ColumnUsageTracker, CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon, Ops
from Layers import SortingLayer, TruthLayer, LiftingLayer, InterSentenceLayer, SparsityRegularizer, SmoothingRegularizer, ImpenetrableLayer
from util import parse
from collections import namedtuple as _namedtuple


from Layers import Layer, PiLayer, SigmaLayer # Import custom layers from Model.py
from Layers import LinearLayer, AttentionLayer
from Layers import ColumnUsageTracker, CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon
from Layers import Error, TheError

from Spaces import ActiveEncoding, WhereEncoding, WhenEncoding, WhatEncoding, EventEncoding, WordEncoding
from Spaces import Basis, Tensor, Codebook, Embedding
from Spaces import SubSpace, WordSubSpace, Space, InputSpace, PerceptualSpace, ModalSpace, ConceptualSpace, SymbolicSpace, OutputSpace


class Grammar:
    """Single-tier (S) grammar rule catalog (post-rewrite, 2026-04-19).

    The C (conceptual) tier has been merged into S: all compositional
    operations (not, part, intersection, union, lift, lower) are now S-tier
    productions.  The P (perceptual) tier has been removed from the
    grammar.

    Owns the rule definitions parsed from XML config.  All learnable
    parameters and rule execution live on a single unified
    ``SyntacticLayer`` instance owned by ``WordSpace``.
    """

    # lhs:          nonterminal this rule reduces to ('S', 'VO', 'NP', 'VP', ...).
    # rhs_symbols:  typed-form RHS category tuple (e.g. ('V', 'O') for VO -> V O).
    #               None for legacy function-call / epsilon / passthrough rules.
    RuleDef = _namedtuple(
        'RuleDef',
        ['tier', 'canonical', 'arity', 'method_name', 'lhs', 'rhs_symbols'],
    )

    def __init__(self):
        self.rules = []
        self.rules_upward = []
        self.rules_downward = []
        # Step 6: Layer-2.5 reverse productions, derived mechanically
        # from rules_upward at load time.  Each entry is
        # ``(args_tuple, reverse_op_name, (lhs,))``.
        self.reverse_rules = []
        self.rule_table = {}
        self._configured = False
        self.interpretation = 0.5
        self.thought_free = False
        # Start symbol -- name of the nonterminal that marks a complete
        # derivation. SyntacticLayer.compose tests row b's top-of-stack
        # category against this name; matches signal a soft sentence
        # boundary and trigger soft_reset(b) in the outer loop. Configurable
        # via <start>S</start> in the language XML; falls back to "S"
        # (the historical default) when unset.
        self.start_symbol = "S"

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
        """Configure rules from an XML-derived dict.

        Accepts two shapes:
          (a) flat: {'S': ['not(S)'], ...}  — legacy upward-only.
          (b) split: {'upward': {'S': [...], 'VO': [...]},
                      'downward': {'S': ['C']}}
        """
        self.rules_upward = []
        self.rules_downward = []
        self._configured = True

        if ('upward' in grammar_dict or 'downward' in grammar_dict):
            up = grammar_dict.get('upward', {}) or {}
            dn = grammar_dict.get('downward', {}) or {}
            self._fill_rule_list(self.rules_upward, up)
            self._fill_rule_list(self.rules_downward, dn)
        else:
            # Legacy flat form — treat as upward.
            self._fill_rule_list(self.rules_upward, grammar_dict)

        # Canonical union so callers reading `g.rules` see upward first,
        # then downward. Upward rule IDs stay stable for existing code.
        self.rules = list(self.rules_upward) + list(self.rules_downward)
        self.rule_table = {idx: rule.canonical
                           for idx, rule in enumerate(self.rules)}
        # Step 6 parity: derive Layer-2.5 reverse rules from upward
        # productions even on the legacy XML path so consumers of
        # ``self.reverse_rules`` work uniformly across load paths.
        self.reverse_rules = self._derive_reverse_rules(self.rules_upward)

    def _fill_rule_list(self, target, rules_dict):
        # New syntax: <rule>head = body</rule> — head may be a comma-
        # separated tuple of categories (for multi-output downward rules
        # like `S,S = intersection_inv(VO)`). Body is a function call
        # (`f(A, B)`), bare-symbol sequence (`A B`), or a single category
        # (`C` / `A`). Rules in this form arrive under the 'rule' key
        # because that's the XML element name used.
        rule_entries = rules_dict.get('rule', None)
        if rule_entries is not None:
            if isinstance(rule_entries, str):
                rule_entries = [rule_entries]
            for entry in rule_entries:
                if '=' not in entry:
                    raise ValueError(
                        f"<rule> requires 'head = body' syntax, got: {entry!r}")
                lhs_raw, body = entry.split('=', 1)
                lhs = ','.join(p.strip() for p in lhs_raw.split(',') if p.strip())
                target.append(self._parse_rule(lhs, body.strip()))

        # Legacy syntax: <S>body</S> with nonterminal as tag. Kept for
        # backward compat with tests and older XMLs. 'S' stays implicitly
        # first so existing rule-id ordering is stable.
        keys = [k for k in rules_dict.keys() if k != 'rule']
        if 'S' in keys:
            keys = ['S'] + [k for k in keys if k != 'S']
        for lhs in keys:
            raw = rules_dict.get(lhs, [])
            if isinstance(raw, str):
                raw = [raw]
            for rhs_text in raw:
                rhs = rhs_text.strip()
                target.append(self._parse_rule(lhs, rhs))

    def rule_by_id(self, rule_id):
        """Return the canonical production string for a rule_id (0-based)."""
        return self.rule_table[rule_id]

    def _parse_rule(self, lhs, rhs):
        # All rules are S-tier post-2026-04-19 merge; `tier` is retained
        # for Grammar.symbolic()/symbolic_transition()/s_methods routing.
        # New typed categories live on the `lhs` field instead.
        tier = 'S'
        if '(' in rhs:
            func_name = rhs[:rhs.index('(')]
            args_str = rhs[rhs.index('(') + 1:rhs.rindex(')')]
            args = [a.strip() for a in args_str.split(',') if a.strip()]
            arity = len(args)
            canonical = f"{lhs} -> {rhs}"
            return self.RuleDef(tier, canonical, arity, func_name,
                                lhs, tuple(args))
        if rhs == 'epsilon':
            return self.RuleDef(tier, f"{lhs} -> epsilon", 0, None,
                                lhs, ())
        if rhs == lhs:
            return self.RuleDef(tier, f"{lhs} -> {rhs}", 1, None,
                                lhs, (rhs,))
        # Bare-symbol-sequence form: '<VO>V O</VO>' or '<S>S VO</S>'.
        # RHS is a whitespace-separated sequence of nonterminal / terminal
        # category names. method_name='merge' signals the typed compose
        # path (Phase B chart-like pair selection) vs. the legacy function
        # dispatch. arity = number of RHS slots (0 is already handled by
        # the epsilon branch; unary passthrough is handled by rhs == lhs).
        parts = rhs.split()
        if parts and all(p.isidentifier() for p in parts):
            arity = len(parts)
            # 'C' is the pseudo-terminal used in downward productions:
            # 'S -> C' means "emit the codebook atom that best matches the
            # current deep state." It dispatches through emit_head, not
            # merge, because there's nothing to combine — just a lookup.
            if len(parts) == 1 and parts[0] == 'C':
                method = 'emit_head'
            else:
                method = 'merge'
            return self.RuleDef(tier, f"{lhs} -> {rhs}", arity, method,
                                lhs, tuple(parts))
        raise ValueError(f"Cannot parse grammar rule: {lhs} -> {rhs}")

    # -- grammar.cfg loader (Step 6) -----------------------------------
    #
    # Step 6 of the lift / lower / bivector refactor introduces a
    # text-based rule table at ``data/grammar.cfg`` in explicit-op form.
    # Each production's RHS literally names the Ops method to dispatch
    # on at rule-application time.  See parent plan §Step 6 lines
    # 405–620 and ``data/grammar.cfg`` for the format spec.
    #
    # Sections are bracketed (``[upward]`` / ``[downward]``) and default
    # to ``[upward]`` when no header has been seen.  The parser is
    # line-oriented, comment-prefixed (``#``), with no third-party
    # dependency.  Reverse productions (Layer 2.5) are derived
    # mechanically from the upward rules at load time and exposed via
    # ``self.reverse_rules`` as
    # ``[(args_tuple, op_name + 'Reverse', (lhs,)), ...]``.

    _CFG_SECTION_UPWARD   = 'upward'
    _CFG_SECTION_DOWNWARD = 'downward'

    def load_from_cfg(self, path):
        """Configure rules from a ``data/grammar.cfg``-style file.

        File format (line-oriented):
            ``# ...``               line comment
            ``[section]``           section header (upward / downward)
            ``LHS = body``          rule; body is ``op(args)`` or a single
                                    category for PROJECT, or ``epsilon``
            blank lines             ignored

        After loading:
            ``self.rules_upward``   forward productions and post-hoc S-ops
            ``self.rules_downward`` downward / generative productions
            ``self.reverse_rules``  Layer 2.5 reverse productions, derived
                                    mechanically from rules_upward
            ``self.rules``          upward + downward
        """
        with open(path, 'r') as fh:
            lines = fh.read().splitlines()
        sections = self._parse_cfg_lines(lines)
        self._apply_cfg_sections(sections)

    def _parse_cfg_lines(self, lines):
        """Group raw cfg lines by section.  Returns a dict
        ``{section_name: [(lhs, rhs), ...]}``.  Default section is
        ``upward``; rules appearing before any header land there.
        """
        sections = {
            self._CFG_SECTION_UPWARD:   [],
            self._CFG_SECTION_DOWNWARD: [],
        }
        current = self._CFG_SECTION_UPWARD
        for raw in lines:
            # Strip line comments and surrounding whitespace.  Inline
            # comments (``#`` mid-line) are also stripped — the rule body
            # never legitimately contains a ``#``.
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                name = line[1:-1].strip().lower()
                if name not in sections:
                    raise ValueError(
                        f"grammar.cfg: unknown section [{name}]; "
                        f"expected one of {sorted(sections)}"
                    )
                current = name
                continue
            if '=' not in line:
                raise ValueError(
                    f"grammar.cfg: rule requires 'LHS = body' syntax; "
                    f"got {line!r}"
                )
            lhs_raw, body = line.split('=', 1)
            lhs = ','.join(p.strip() for p in lhs_raw.split(',') if p.strip())
            sections[current].append((lhs, body.strip()))
        return sections

    def _apply_cfg_sections(self, sections):
        """Populate ``self.rules_*`` from parsed cfg sections and
        derive the reverse rule table.
        """
        self.rules_upward = [
            self._parse_rule(lhs, body)
            for lhs, body in sections[self._CFG_SECTION_UPWARD]
        ]
        self.rules_downward = [
            self._parse_rule(lhs, body)
            for lhs, body in sections[self._CFG_SECTION_DOWNWARD]
        ]
        self.rules = list(self.rules_upward) + list(self.rules_downward)
        self.rule_table = {idx: rule.canonical
                           for idx, rule in enumerate(self.rules)}
        self.reverse_rules = self._derive_reverse_rules(self.rules_upward)
        self._configured = True

    @staticmethod
    def _derive_reverse_rules(forward_rules):
        """Mechanically derive Layer-2.5 reverse productions from
        forward Layer-1 productions.

        Pattern (parent plan §Step 6 lines 562–568):
            forward:  LHS = op(arg1, arg2)
            reverse:  arg1, arg2 = opReverse(LHS)

        Returns a list of tuples
        ``(args_tuple, reverse_op_name, (lhs,))``.

        Self-inverse / exact-inverse ops (``not``, ``project``,
        ``negation``) get ``opReverse`` = same op name (their forward
        body is its own inverse).  Everything else gets the
        ``<op>Reverse`` suffix.

        PROJECT rules surface as ``projectReverse`` for symmetry with
        the parent plan's Layer 2.5 table.  Two shapes count as
        PROJECT:
          * ``method_name is None`` (transition / epsilon / X -> X
            pass-through);
          * ``method_name == 'merge'`` with a single RHS slot — the
            cfg form ``LHS = single_category`` (e.g. ``S = NP``,
            ``NP = N``) which the existing ``_parse_rule`` classifies
            as a unary merge but is semantically a typed projection.
        """
        SELF_INVERSE = {'not'}
        reverses = []
        for rule in forward_rules:
            op = rule.method_name
            args = rule.rhs_symbols or ()
            lhs = rule.lhs
            is_project = (op is None) or (op == 'merge' and len(args) == 1)
            if is_project:
                reverses.append((args, 'projectReverse', (lhs,)))
                continue
            if op in SELF_INVERSE:
                reverses.append((args, op, (lhs,)))
                continue
            reverses.append((args, op + 'Reverse', (lhs,)))
        return reverses

    _NOOP_GRAMMAR = {'S': 'not(S)'}

    def _ensure_configured(self):
        if self._configured:
            return
        # <start>S</start> in WordSpace.language: the canonical name of
        # the start nonterminal. Used by SyntacticLayer.compose to detect
        # sentence completion (top-of-stack category == start_symbol)
        # and signal soft_reset to the outer doc-streaming loop. Falls
        # back to "S" (historical default) when unset.
        try:
            start_raw = TheXMLConfig.get("WordSpace.language.start")
            if start_raw is not None:
                self.start_symbol = str(start_raw).strip() or "S"
        except (KeyError, AttributeError):
            pass
        cfg = None
        try:
            candidate = TheXMLConfig.get("WordSpace.language.grammar")
            if isinstance(candidate, dict):
                cfg = candidate
        except (KeyError, AttributeError):
            pass
        # New (Step 6) path: prefer ``WordSpace.language.grammarCfg`` —
        # a string path to a ``data/grammar.cfg`` file in explicit-op
        # form.  Resolved relative to ``ProjectPaths.PROJECT_DIR`` if
        # not absolute.  Falls through to the XML grammar (legacy path)
        # when absent or unreadable.
        cfg_path = None
        try:
            cfg_path = TheXMLConfig.get("WordSpace.language.grammarCfg")
        except (KeyError, AttributeError):
            pass
        if cfg_path:
            resolved = (cfg_path if os.path.isabs(cfg_path)
                        else os.path.join(ProjectPaths.PROJECT_DIR, cfg_path))
            if os.path.exists(resolved):
                self.load_from_cfg(resolved)
                try:
                    interp = TheXMLConfig.get("WordSpace.language.interpretation")
                    self.interpretation = float(interp)
                except (KeyError, AttributeError, TypeError, ValueError):
                    pass
                return
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

    @property
    def categories(self):
        """Ordered tuple of unique derivation labels across all rules.

        Derived from both ``lhs`` (including comma-split multi-output
        heads) and ``rhs_symbols``. Used to size the category codebook
        on ``WordSpace``, so every label has its own learned embedding.
        """
        self._ensure_configured()
        names = set()
        for rule in self.rules:
            for cat in str(rule.lhs).split(','):
                cat = cat.strip()
                if cat:
                    names.add(cat)
            for sym in (rule.rhs_symbols or ()):
                if sym:
                    names.add(sym)
        return tuple(sorted(names))

    def _s_rule_ids(self):
        """Return dict of method_name -> rule_id for S-tier operational rules."""
        result = {}
        for i, r in enumerate(self.rules):
            if r.tier == 'S' and r.method_name is not None:
                result[r.method_name] = i
        return result

    # All compositional rules live on the unified SyntacticLayer class
    # as *Forward / *Reverse method pairs.  See _RULE_METHODS dispatch.
TheGrammar = Grammar()

class SyntacticLayer(Layer):
    """Unified rule-prediction and rule-execution layer for the grammar.

    Post-refactor (2026-04-19) this single class owns every grammar
    rule (union, intersection, not, lift, lower, equals, part, true, false,
    non, conjunction, disjunction, swap, what, where, when, query, absorb).
    The prior per-tier subclasses
    (``PerceptualSyntacticLayer`` / ``ConceptualSyntacticLayer`` /
    ``SymbolicSyntacticLayer``) have been merged into this class.

    Uses a weight-tied recursive architecture with depth embeddings for
    rule prediction and dispatches rule bodies through ``_RULE_METHODS``.

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
                 max_depth=12, hidden_dim=256, grammar=None, tau=1.0,
                 feature_dim=None):
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

        # Phase B: pair-scorer MLP. Scores adjacent live-leaf pairs to
        # pick a merge site under the chart compose path. ``feature_dim``
        # is the last-axis width D of leaf vectors at compose time (falls
        # back to nInput when callers don't distinguish -- unit tests do
        # set ``D == nInput``, but real pipelines have D = symbol_dim
        # which is independent of n_slots). ``hidden_dim`` is the width
        # of the rule-prediction hidden state that conditions the score.
        fd = nInput if feature_dim is None else feature_dim
        self._pair_feature_dim = fd
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 2 * fd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Xavier initialization so logits start in a numerically stable range.
        # LinearLayer defaults to torch.randn which gives std=1.0; for large
        # dims this produces huge activations that saturate softmax/gumbel.
        for layer in [self.input_proj, self.derivation_layer, self.rule_head]:
            nn.init.xavier_normal_(layer.W)
        nn.init.normal_(self.depth_embed.weight, std=0.02)

        # Register child layers for ergodic dispatch. pair_scorer is
        # kept off this list because it's an nn.Sequential without the
        # LinearLayer set_sigma/W contract the ergodic loop expects.
        self.layers = [self.input_proj, self.derivation_layer, self.rule_head]

        # Per-compose caches. compose() resets these each call; pre-init
        # here so read sites (e.g. MentalModel universality hook) can read
        # safely on layers whose compose hasn't fired.
        self.last_svo = None
        self.last_rule_probs = None
        self.last_composable_rules = None
        # LiftingLayer is instantiated by init_lifting() from
        # WordSpace._build_syntactic_layer. Pre-init so universality
        # read sites can test for None without a missing-attr trap.
        self.lifting_layer = None
        # Per-batch list of (rule_id, left, right, merged_slot,
        # merged_category) tuples appended by chart compose. Reset at
        # compose start; consumed by SVO extraction and downward head
        # emission.
        self._derivation_trace = None
        # Category machinery (Phase C). Populated lazily on first chart
        # compose; tests may seed categories directly via _seed_category.
        self._category_names = None
        self._category_index = None
        self._last_category = None

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
    # C-tier/SigmaPi ops (invertible): not, intersection, union, lift, lower
    # S-tier ops (lossy, no inverse): equals, part, true, non, swap, absorb
    #
    # _RULE_METHODS maps rule name -> (forwardName, reverseName|None, binary)

    # Unified rule table (post-subclass-merge, 2026-04-19). The three
    # former subclasses (PerceptualSyntacticLayer, ConceptualSyntacticLayer,
    # SymbolicSyntacticLayer) have been collapsed into this class.
    _RULE_METHODS = {
        # Sugar absorption. S-tier and lossy: the right operand is discarded.
        'absorb':       ('absorbForward',       None,                  True),
        # C-tier invertible concept ops
        'union':        ('unionForward',        'unionReverse',        True),
        'intersection': ('intersectionForward', 'intersectionReverse', True),
        'not':          ('notForward',          'notReverse',          False),
        # Lift / lower are now in-space binary algebra (PiLayer round-trip
        # removed when the three subclasses merged).
        'lift':         ('liftForward',         'liftReverse',         True),
        'lower':        ('lowerForward',        'lowerReverse',        True),
        # S-tier lossy ops
        'equals':       ('equalsForward',       None,                  True),
        'part':         ('partForward',         None,                  True),
        'true':         ('trueForward',         None,                  False),
        'false':        ('falseForward',        None,                  False),
        'non':          ('nonForward',          None,                  False),
        'conjunction':  ('conjunctionForward',  None,                  True),
        'disjunction':  ('disjunctionForward',  None,                  True),
        'swap':         ('swapForward',         None,                  True),
        'what':         ('whatForward',         None,                  False),
        'where':        ('whereForward',        None,                  False),
        'when':         ('whenForward',         None,                  False),
        'query':        ('queryForward',        None,                  True),
    }

    # Step 6: Direct ``Ops`` dispatch table used by the explicit-op
    # grammar.cfg path.  Each entry maps a grammar op-name to the bare
    # ``Ops`` callable the dispatcher invokes when the rule is
    # interpreted **without** Basis/subspace/mask context (the
    # "stateless" view).  When grammar.cfg provides the rule and the
    # caller has no subspace context, ``dispatch_ops`` below routes
    # through this table; when the caller has subspace context, the
    # existing ``_RULE_METHODS`` path is preferred (it handles masks
    # and codebook-search inverses correctly).
    #
    # Entries here mirror the parent plan §Step 6 Layer-1 / Layer-2
    # tables: ``intersection`` / ``union`` route to the unified
    # ``Ops.lower`` / ``Ops.lift`` mode-dispatchers, ``not`` to
    # ``Ops.negation``, etc.  Ops with no stateless implementation
    # (``absorb``, ``query``, ``swap``, ``what`` / ``where`` / ``when``)
    # are absent and must go through the
    # ``_RULE_METHODS`` path.
    #
    # Step 5 deferral 3 (binary rule application): the cfg dispatcher
    # uses option (a) — explicit two-operand calls into the unified
    # ``Ops.lower`` / ``Ops.lift`` mode-dispatchers — so it stays
    # bit-equivalent to the legacy ``_RULE_METHODS`` path.  The
    # alternative (option (b): pack the constituents into a single
    # N-element tensor and call ``space.sigma.forward(packed, binary=True)``
    # / ``space.sigma.forward(...)`` to use the Step-5 STE top-2 selector)
    # is feasible per-rule but trades the explicit binary algebra for
    # auto-pruning; defer that to a follow-up commit so the Step 6
    # acceptance sweep stays a clean equivalence check.  Option (c)
    # (per-rule annotation) becomes practical once a per-rule flag is
    # added to grammar.cfg; out of scope here.
    _OPS_METHODS = {
        'lift':         lambda x, y=None, **kw: Ops.lift(x, y, **kw),
        'lower':        lambda x, y=None, **kw: Ops.lower(x, y, **kw),
        'union':        lambda x, y=None, **kw: Ops.lift(x, y, mode='OR',
                                                         **{k: v for k, v in kw.items()
                                                            if k != 'mode'}),
        'intersection': lambda x, y=None, **kw: Ops.lower(x, y, mode='AND',
                                                          **{k: v for k, v in kw.items()
                                                             if k != 'mode'}),
        'not':          lambda x, **kw: Ops.negation(x, **kw),
        'conjunction':  lambda x, y, **kw: Ops.conjunction(x, y, **kw),
        'disjunction':  lambda x, y, **kw: Ops.disjunction(x, y, **kw),
        'non':          lambda x, **kw: Ops.non(x, **kw),
        'true':         lambda x, **kw: torch.relu(torch.clamp(x, -1.0, 1.0)),
        'false':        lambda x, **kw: torch.relu(-torch.clamp(x, -1.0, 1.0)),
        'part':         lambda x, y, **kw: Ops.part(x, y, **kw),
        'equals':       lambda x, y, **kw: Ops.equal(x, y, **kw),
    }

    @classmethod
    def dispatch_ops(cls, op_name, *args, **kwargs):
        """Resolve a grammar op-name to its bare ``Ops`` callable and
        invoke it (Step 6 explicit-op grammar dispatch, stateless view).

        Used by callers that have a rule table loaded from
        ``data/grammar.cfg`` and want to apply the named op without
        going through layer context (Basis / subspace / mask).  For
        ops that need layer context (``absorb``, ``query``, ``swap``,
        ``what`` / ``where`` / ``when``)
        callers must use ``project()`` instead — those entries are not
        present in ``_OPS_METHODS`` and a ``KeyError`` here signals
        that the layer-context dispatch path is required.

        Args:
            op_name: grammar method name (``'lift'``, ``'lower'``, ...).
            *args, **kwargs: forwarded to the resolved Ops callable.

        Returns:
            Whatever the Ops callable returns — typically a tensor.
        """
        return cls._OPS_METHODS[op_name](*args, **kwargs)

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
            out = b.negationReverse(result, monotonic=self._mono(subspace))
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
            out = b.conjunctionReverse(result, right, monotonic=self._mono(subspace))
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
            out = b.disjunctionReverse(result, right, monotonic=self._mono(subspace))
        else:
            out = result
        return self._apply_mask(out, mask, subspace=subspace)

    # -- Sugar absorption -----------------------------------------------
    # `absorb(S, S)` is a binary left-pass: returns the left operand and
    # discards the right. The right operand is the "sugar" that should be
    # absorbed into the surrounding constituent without contributing to it.
    # Unary `identity(S)` was removed as a grammar rule because it is just
    # the transition / PROJECT behavior under another name.

    def absorbForward(self, left, right, subspace, mask=None):
        # Left-pass: return left unchanged; right is sugar that the rule
        # predictor opted to discard. mask is applied to the surviving
        # output so per-cell gating still flows through.
        return self._apply_mask(left, mask, subspace=subspace)

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
        """Triangular residual: 1 - |clamp(x, -1, 1)|. The 'indeterminate'
        commitment.

        Completes the S-tier trinity partition of unity:
        ``true(x) + false(x) + non(x) = 1``. Always uses the bitonic
        formula -- the partition only holds under bitonic semantics, so
        the rule is independent of the subspace's monotonic flag.
        """
        out = Ops.non(left, monotonic=False)
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
        out = Ops.what(left, *self._split_widths(subspace))
        return self._apply_mask(out, mask, subspace=subspace)

    def whereForward(self, left, subspace, mask=None):
        """Axis selector: keep where-block, zero what/when-blocks."""
        out = Ops.where(left, *self._split_widths(subspace))
        return self._apply_mask(out, mask, subspace=subspace)

    def whenForward(self, left, subspace, mask=None):
        """Axis selector: keep when-block, zero what/where-blocks."""
        out = Ops.when(left, *self._split_widths(subspace))
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

    # -- lift / lower: in-space algebra (post PiLayer round-trip removal) --

    def liftForward(self, left, right, subspace, mask=None):
        """In-space lift: elementwise product. Post-merge (2026-04-19)
        the old PiLayer round-trip (forward to S, multiply, reverse to C)
        collapses to the in-space body because the caller already has the
        forwarded operands.
        """
        return self._apply_mask(
            Ops.lower(left, right, mode='AND', kind='smooth'),
            mask, subspace=subspace)

    def liftReverse(self, result, right, subspace, mask=None):
        """Recover first operand from the elementwise product."""
        return self._apply_mask(Ops.liftReverse(result, right), mask, subspace=subspace)

    def lowerForward(self, left, right, subspace, mask=None):
        """In-space lower: arithmetic mean."""
        return self._apply_mask(
            Ops.lift(left, right, mode='OR', kind='smooth'),
            mask, subspace=subspace)

    def lowerReverse(self, result, right, subspace, mask=None):
        """Recover first operand from the mean."""
        return self._apply_mask(Ops.lowerReverse(result, right), mask, subspace=subspace)

    # -- swap: Sinkhorn-normalised soft permutation ------------------

    def init_swap(self, symbol_dim, n_symbol_slots):
        """Initialize swap and non parameters. Called unconditionally on
        the unified SyntacticLayer at construction time."""
        swap_size = max(symbol_dim, n_symbol_slots, 1)
        self.swap_marker = nn.Parameter(torch.randn(swap_size) * 0.01)
        self.swap_logits = nn.Parameter(torch.zeros(3, 3))
        self._swap_sinkhorn_iters = 5
        self.non_threshold = nn.Parameter(torch.tensor(0.0))

    def init_lifting(self, concept_dim, nVerbs=16):
        """Instantiate the LiftingLayer used by TruthLayer.universality.

        The codebook holds ``nVerbs`` learned verb matrices ([D, D])
        that act on concept vectors of dimension ``concept_dim``. The
        layer is the endpoint for Golden-Rule scoring: ``forward_transitive_svo``
        routes (subject, verb, object) through the verb codebook so the
        TruthLayer can measure luminosity change under S/O reversal.
        Registered as a submodule so its parameters participate in training.
        """
        try:
            configured = int(TheXMLConfig.get("architecture.VerbCodebookSize"))
            if configured > 0:
                nVerbs = configured
        except (KeyError, TypeError, ValueError):
            pass
        self.lifting_layer = LiftingLayer(nVerbs=nVerbs, nDim=concept_dim)

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
            D = left.shape[-1]
            m = marker[:D].unsqueeze(0).unsqueeze(0).expand_as(left)
        elif left.ndim == 2:
            m = marker[:left.shape[-1]].unsqueeze(0).expand_as(left)
        else:
            m = marker
        if right is None:
            right = left
        stack = torch.stack([left, right, m], dim=0)
        out = torch.einsum('ij,j...->i...', P, stack)
        return self._apply_mask(out[0], mask, subspace=subspace)

    # -- Rule #3: query (contradiction marker at accumulation point) --
    # Norm-drop threshold. If a new rule-application result would reduce
    # the accumulator's norm below this fraction of its current value,
    # the accumulation point interprets it as symbolic contradiction and
    # emits a query word + preserves the existing accumulator instead of
    # absorbing the cancelling contribution.
    _QUERY_NORM_DROP_RATIO = 0.1

    def queryForward(self, left, right, subspace=None, mask=None):
        """Query: return the preserved accumulator operand.

        The query marker is pushed onto WordSubSpace at the accumulation
        point (see `compose()`), not by this forward. When the parse tree
        is re-evaluated downstream, `queryForward` returns the first
        operand -- the accumulator state that was preserved when the
        cancelling contribution arrived.
        """
        return self._apply_mask(left, mask, subspace=subspace)

    # -- compose / decompose: unified driver ------------------------

    def compose(self, data, subspace, grammar, target_count=None):
        """Apply composition to a batch of activations or vectors.

        Handles both 3D ``[B, N, D]`` (vector mode) and 2D ``[B, N]``
        (activation mode, used by the old S-tier path). In 2D mode the
        grammar acts on norms; the returned tensor has the same shape.

        When ``target_count`` is supplied (3D only), pairwise reduction
        drives the active token count down to that value.

        Returns ``(composed, svo_or_None)`` -- ``svo`` is reserved for
        ternary lift output; currently always ``None``.
        """
        subspace.word = []
        self.last_svo = None
        self.last_rule_probs = None
        self.last_composable_rules = None
        B_guess = data.shape[0] if torch.is_tensor(data) and data.ndim >= 2 else 1
        self._derivation_trace = [[] for _ in range(B_guess)]

        if data.ndim == 2:
            composed = self._compose_activation(data, subspace, grammar)
            return composed, self.last_svo

        return self._compose_vector(data, subspace, grammar, target_count)

    def _compose_activation(self, data, subspace, grammar):
        """2D path: [B, N] activation mode. Used for S-tier compose on
        norms and for the 2D-input branch of the unified compose.
        """
        B, N = data.shape

        # Guard: skip soft superposition if data dims don't match
        expected_n = self.input_proj.nInput
        if N != expected_n or self.num_rules == 0:
            return data

        out = self.forward(data)
        rule_probs = out['rule_probs']  # [B, max_depth, num_rules]
        all_rules = self.all_rules

        active_positions = [subspace.active_positions(b, data) for b in range(B)]
        max_leaves = max((len(p) for p in active_positions), default=0)
        if max_leaves == 0:
            return data

        masks = torch.zeros(B, max_leaves, N, device=data.device)
        for b in range(B):
            for i, pos in enumerate(active_positions[b]):
                if i < max_leaves:
                    masks[b, i, pos] = 1.0
        leaf_acts = masks * data.unsqueeze(1)  # [B, L, N]

        composed = leaf_acts[:, 0, :]

        last_rule_per_batch = [-1 for _ in range(B)]
        query_rid = None
        for _gid in all_rules:
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
                    result = self.project(grammar, rule_id, left, right,
                                          subspace=subspace)
                else:
                    result = self.project(grammar, rule_id, left,
                                          subspace=subspace)
                results.append(result)

            results = torch.stack(results, dim=1)  # [B, num_rules, N]
            probs_d = rule_probs[:, d, :]          # [B, num_rules]

            # Rule #3: norm-drop detection at the accumulation point
            candidate = (probs_d.unsqueeze(-1) * results).sum(dim=1)  # [B, N]
            prev_norm = left.norm(dim=-1)         # [B]
            cand_norm = candidate.norm(dim=-1)    # [B]
            drop_threshold = self._QUERY_NORM_DROP_RATIO * prev_norm
            query_mask = (prev_norm > 1e-6) & (cand_norm < drop_threshold)
            # One sync per tensor per d-step, not per b-row.
            query_mask_list = query_mask.tolist()
            best = probs_d.argmax(dim=-1)
            best_list = best.tolist()
            if query_mask.any():
                mask_exp = query_mask.unsqueeze(-1).expand_as(candidate)
                composed = torch.where(mask_exp, left, candidate)
                best_for_push = probs_d.argmax(dim=-1)
                best_push_list = best_for_push.tolist()
                word_sub = getattr(self, 'word_subspace', None)
                for b in range(B):
                    if not query_mask_list[b]:
                        continue
                    left_rid = last_rule_per_batch[b]
                    right_rid = best_push_list[b]
                    right_gid = all_rules[right_rid] if right_rid < len(all_rules) else -1
                    if query_rid is not None and word_sub is not None:
                        word_sub.push(b, query_rid,
                                      leaves=(left_rid, right_gid, -1))
            else:
                composed = candidate

            for b in range(B):
                if d < len(active_positions[b]):
                    best_gid = all_rules[best_list[b]]
                    subspace.add_word(b, active_positions[b][d], best_gid)
                    if not query_mask_list[b]:
                        last_rule_per_batch[b] = best_gid

        return composed

    def _pair_scorer(self, hidden, pairs, alive):
        """Score adjacent-leaf pairs for chart-like merge.

        hidden [B, H], pairs [B, P, 2, D], alive [B, N] -> logits [B, P].
        Dead pairs get -inf so softmax routes zero probability to them.

        ``pairs`` can be either (a) full-width [B, N-1, 2, D] (tests pass
        this; dead pairs are zero-filled) or (b) compacted [B, P_live, 2, D]
        from ``_live_pairs`` (live-pipeline; P_live = #alive - 1 per batch,
        padded to the max across batches). The shared mask derives
        alive-pair count per batch row and masks tail slots as dead.

        Runtime feature dim: chart compose can be invoked with D ranging
        from the symbol-codebook bivector width (2) to the muxed symbol
        width (nDim + nWhere + nWhen). The pair-scorer was sized with
        ``feature_dim`` at construction; if the live tensor arrives with
        a different D, rebuild the MLP lazily rather than fail in the
        ``nn.Linear`` shape check.
        """
        B, P, _, D = pairs.shape
        H = hidden.shape[-1]
        if D != self._pair_feature_dim:
            self.pair_scorer = nn.Sequential(
                nn.Linear(H + 2 * D, H),
                nn.GELU(),
                nn.Linear(H, 1),
            ).to(hidden.device)
            self._pair_feature_dim = D
        h = hidden.unsqueeze(1).expand(B, P, H)
        flat = pairs.reshape(B, P, 2 * D)
        feat = torch.cat([h, flat], dim=-1)
        logits = self.pair_scorer(feat.reshape(B * P, -1)).reshape(B, P)
        alive_counts = alive.to(torch.long).sum(dim=1)  # [B]
        num_pairs = (alive_counts - 1).clamp_min(0)     # [B]
        idx = torch.arange(P, device=alive.device).unsqueeze(0)
        pair_alive = idx < num_pairs.unsqueeze(1)       # [B, P]
        logits = logits.masked_fill(~pair_alive, float('-inf'))
        return logits

    def _live_pairs(self, live, alive):
        """Extract adjacent-pair tensors from live-leaf ordering.

        Returns (pair_tensor [B, P_max, 2, D], list[list[(l, r)]],
                 pair_indices [B, P_max, 2] long, pair_alive [B, P_max] bool).

        ``batch_pairs`` (Python list) is kept for consumers that drive
        per-row ``subspace.add_word()`` calls (which mutate a Python list
        and so must already cross the host boundary). ``pair_indices`` and
        ``pair_alive`` are tensor views of the same data, used by callers
        that can stay on-device — e.g. the category-compat fill.
        """
        B, N, D = live.shape
        # One sync to bring the alive mask to Python; inner scan is free.
        alive_list = alive.tolist()
        batch_pairs = []
        max_P = 0
        for b in range(B):
            positions = [i for i in range(N) if alive_list[b][i]]
            pairs = list(zip(positions[:-1], positions[1:]))
            batch_pairs.append(pairs)
            max_P = max(max_P, len(pairs))
        pair_tensor = torch.zeros(B, max_P, 2, D, device=live.device)
        pair_indices = torch.zeros(B, max_P, 2, dtype=torch.long, device=live.device)
        pair_alive = torch.zeros(B, max_P, dtype=torch.bool, device=live.device)
        for b, pairs in enumerate(batch_pairs):
            for p, (l, r) in enumerate(pairs):
                pair_tensor[b, p, 0] = live[b, l]
                pair_tensor[b, p, 1] = live[b, r]
                pair_indices[b, p, 0] = l
                pair_indices[b, p, 1] = r
                pair_alive[b, p] = True
        return pair_tensor, batch_pairs, pair_indices, pair_alive

    def _ensure_category_table(self, grammar):
        if getattr(self, '_category_names', None) is not None:
            return
        names = set()
        for rule in grammar.rules:
            # rule.lhs is a string; multi-output rules store it as a
            # comma-joined tuple ('S,S' for `S,S = intersection_inv(VO)`).
            for cat in str(rule.lhs).split(','):
                cat = cat.strip()
                if cat:
                    names.add(cat)
            for sym in (rule.rhs_symbols or ()):
                names.add(sym)
        ordered = ['?'] + sorted(n for n in names if n)
        self._category_names = ordered
        self._category_index = {n: i for i, n in enumerate(ordered)}

    def _seed_category(self, category):
        self._last_category = category.clone()

    def _apply_rules_to_pairs(self, pair_tensor, composable_global,
                              grammar, subspace):
        """For each (pair, rule) compute the merged vector. Returns
        merged[B, P, R, D] where R=len(composable_global).
        """
        B, P, _, D = pair_tensor.shape
        R = len(composable_global)
        merged = torch.zeros(B, P, R, D, device=pair_tensor.device)
        for p in range(P):
            left = pair_tensor[:, p, 0, :].unsqueeze(1)
            right = pair_tensor[:, p, 1, :].unsqueeze(1)
            for r, gid in enumerate(composable_global):
                result = self.project(grammar, gid, left, right,
                                      subspace=subspace)
                merged[:, p, r] = result.squeeze(1)
        return merged

    def _extract_svo_from_trace(self, grammar, original_data):
        """Walk self._derivation_trace → last_svo (subject, verb, object).

        Looks for the outermost `S -> S VO` firing and its matching
        inner `VO -> V O` firing; subject = arg-0 of outer, verb/object =
        arg-0/arg-1 of inner. Leaves last_svo None if no batch row has
        the canonical shape.
        """
        if self._derivation_trace is None:
            return
        B, N, D = original_data.shape
        s_list, v_list, o_list = [], [], []
        any_valid = False
        zero = torch.zeros(1, D, device=original_data.device)
        for b in range(B):
            trace = self._derivation_trace[b]
            outer = None
            for entry in reversed(trace):
                rule_id = entry[0]
                rule = grammar.rules[rule_id]
                if rule.lhs == 'S' and rule.rhs_symbols == ('S', 'VO'):
                    outer = entry
                    break
            if outer is None:
                s_list.append(zero); v_list.append(zero); o_list.append(zero)
                continue
            left_slot, right_slot = outer[1], outer[2]
            vo_entry = None
            for entry in trace:
                rule_id = entry[0]
                rule = grammar.rules[rule_id]
                if (rule.lhs == 'VO' and rule.rhs_symbols == ('V', 'O')
                        and entry[3] == right_slot):
                    vo_entry = entry
                    break
            if vo_entry is None:
                s_list.append(zero); v_list.append(zero); o_list.append(zero)
                continue
            s_list.append(original_data[b, left_slot:left_slot + 1])
            v_list.append(original_data[b, vo_entry[1]:vo_entry[1] + 1])
            o_list.append(original_data[b, vo_entry[2]:vo_entry[2] + 1])
            any_valid = True
        if any_valid:
            self.last_svo = (
                torch.stack(s_list, dim=0),
                torch.stack(v_list, dim=0),
                torch.stack(o_list, dim=0),
            )

    def emit_head(self, state, codebook):
        """Downward `S -> C`: emit the codebook atom that best matches `state`.

        Uses scalar projection onto each unit-normalized atom to measure
        how much of that atom lives in `state`. Returns:

          best_idx      [B]    -- codebook row index of the best match.
          contained     [B, D] -- projection * atom_unit (the slice of
                                  the atom that is actually in `state`).
          residual      [B, D] -- state - contained.

        One step of "look at the remaining meaning, emit the atom it is
        most richly part of, subtract its contribution."
        """
        W = codebook.getW()
        W_norm = W / W.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scores = state @ W_norm.T
        scores = scores.clamp_min(0.0)
        best_idx = scores.argmax(dim=-1)
        best_atom_unit = W_norm[best_idx]
        scalar = scores.gather(1, best_idx.unsqueeze(-1))
        contained = scalar * best_atom_unit
        residual = state - contained
        return best_idx, contained, residual

    def _compose_vector_chart(self, data, subspace, grammar):
        """Phase B chart compose. Differentiable pair + rule selection.

        Populates self._derivation_trace with 5-tuples
        (rule_id, left_slot, right_slot, merged_slot, merged_category_id).
        Returns (composed [B, N, D], None). SVO left to Phase D walker.
        """
        B, N, D = data.shape
        live = data.clone()
        alive = torch.zeros(B, N, dtype=torch.bool, device=data.device)
        active_positions = [subspace.active_positions(b, data)
                            for b in range(B)]
        for b, positions in enumerate(active_positions):
            for p in positions:
                alive[b, p] = True

        activation = torch.norm(live, dim=-1) / math.sqrt(D)
        expected_n = self.input_proj.nInput
        if N != expected_n or self.num_rules == 0:
            return live, None

        out = self.forward(activation)
        rule_probs_per_depth = out['rule_probs']
        hidden_per_depth = out.get('hidden', None)

        # Upward rules only in chart compose; downward is emit-time.
        up_rules = [i for i, r in enumerate(grammar.rules)
                    if r in grammar.rules_upward]
        composable_global = [gid for gid in up_rules
                             if grammar.arity(gid) >= 2]
        if not composable_global:
            return live, None
        composable_local = [self.all_rules.index(gid)
                            for gid in composable_global
                            if gid in self.all_rules]
        if not composable_local:
            return live, None
        composable_global = [self.all_rules[i] for i in composable_local]

        # Category machinery (Task 7 will populate; Task 5 just keeps space).
        self._ensure_category_table(grammar)
        if self._last_category is not None:
            category = self._last_category.to(data.device).long()
        else:
            category = torch.full((B, N), 0, dtype=torch.long,
                                  device=data.device)

        rule_lhs_ids = []
        rule_rhs_ids = []
        for gid in composable_global:
            lhs = grammar.rules[gid].lhs
            rhs = grammar.rules[gid].rhs_symbols or ()
            # Upward rules are single-output; if a multi-output rule
            # (comma-joined lhs) ever reaches chart compose, tag the
            # merged slot with the first output category.
            lhs_cat = lhs.split(',', 1)[0].strip() if ',' in lhs else lhs
            rule_lhs_ids.append(self._category_index.get(lhs_cat, 0))
            if len(rhs) >= 2:
                rule_rhs_ids.append(
                    (self._category_index.get(rhs[0], 0),
                     self._category_index.get(rhs[1], 0)))
            else:
                rule_rhs_ids.append((0, 0))

        # Pre-build per-rule category tensors once (no per-step rebuild).
        rule_lhs_req = torch.tensor(
            [r[0] for r in rule_rhs_ids],
            device=data.device, dtype=torch.long)                  # [R]
        rule_rhs_req = torch.tensor(
            [r[1] for r in rule_rhs_ids],
            device=data.device, dtype=torch.long)                  # [R]
        no_rhs_mask = torch.tensor(
            [grammar.rules[g].rhs_symbols is None
             for g in composable_global],
            device=data.device, dtype=torch.bool)                  # [R]

        depth_probs = []
        for step in range(min(self.max_depth, N - 1)):
            pair_tensor, pair_positions, pair_indices, pair_alive = (
                self._live_pairs(live, alive))
            if pair_tensor.shape[1] == 0:
                break

            if hidden_per_depth is not None and hidden_per_depth.ndim >= 2:
                if hidden_per_depth.ndim == 3:
                    hidden = hidden_per_depth[:, min(step, hidden_per_depth.shape[1] - 1), :]
                else:
                    hidden = hidden_per_depth
            else:
                hidden = torch.zeros(B, self.hidden_dim, device=data.device)

            pair_logits = self._pair_scorer(hidden, pair_tensor, alive)
            pair_probs = torch.softmax(pair_logits, dim=-1)

            rule_probs_step = rule_probs_per_depth[:, step, :][:, composable_local]
            rule_probs_step = rule_probs_step / (
                rule_probs_step.sum(dim=-1, keepdim=True) + 1e-8)
            depth_probs.append(rule_probs_step.detach())

            merged = self._apply_rules_to_pairs(
                pair_tensor, composable_global, grammar, subspace)

            # Compat mask: typed rules require (lhs_req, rhs_req) to match
            # the pair's category IDs; legacy function-call rules (no
            # rhs_symbols) are always compatible. Category id 0 ("?")
            # is a wildcard -- unseeded leaves match any typed rule.
            #
            # Vectorized: pair_indices [B, P, 2] gives (ls, rs) per pair on
            # device; gather category at those indices; broadcast against
            # the pre-built rule_{lhs,rhs}_req [R] tensors. No host sync,
            # no Python triple loop.
            P_here, R_here = pair_tensor.shape[1], len(composable_global)
            ls_idx = pair_indices[:, :, 0]                         # [B, P]
            rs_idx = pair_indices[:, :, 1]                         # [B, P]
            cl = category.gather(1, ls_idx)                        # [B, P]
            cr = category.gather(1, rs_idx)                        # [B, P]
            cl_e = cl.unsqueeze(-1)                                # [B, P, 1]
            cr_e = cr.unsqueeze(-1)
            lhs_match = (cl_e == 0) | (cl_e == rule_lhs_req.view(1, 1, -1))
            rhs_match = (cr_e == 0) | (cr_e == rule_rhs_req.view(1, 1, -1))
            typed_compat = lhs_match & rhs_match                   # [B, P, R]
            compat = (no_rhs_mask.view(1, 1, -1) | typed_compat).to(data.dtype)
            # Zero out padded (dead) pair slots so they contribute nothing.
            compat = compat * pair_alive.unsqueeze(-1).to(data.dtype)
            # Per-step sync (one host roundtrip, not per-cell): break the
            # depth loop when no pair-rule combo is compatible across the
            # whole batch. Without this we'd run nan-degenerate iterations
            # to max_depth. Far cheaper than the prior triple loop.
            if compat.sum() == 0:
                break

            joint = (pair_probs.unsqueeze(-1)
                     * rule_probs_step.unsqueeze(1)
                     * compat)
            joint = joint / joint.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)

            if self.training:
                merged_vec = (joint.unsqueeze(-1) * merged).sum(dim=(1, 2))
            else:
                flat_idx = joint.reshape(B, -1).argmax(dim=-1)
                pair_idx = flat_idx // R_here
                rule_idx_local = flat_idx % R_here
                # `merged` may live on a different device than the default
                # (e.g. test_mental_model loads CPU data into a model whose
                # default device is MPS). Pin all index tensors to merged's
                # device explicitly so cross-device gather doesn't fire.
                row_idx = torch.arange(B, device=merged.device)
                merged_vec = merged[
                    row_idx,
                    pair_idx.to(merged.device),
                    rule_idx_local.to(merged.device)]

            # Trace + alive/live update -- always uses argmax for the
            # discrete trace even in training (gradient still flows
            # through merged_vec via the soft mixture).
            best_flat = joint.reshape(B, -1).argmax(dim=-1)
            best_pair = (best_flat // R_here).tolist()
            best_rule_local = (best_flat % R_here).tolist()

            # Collect per-batch updates; build an out-of-place live
            # replacement after the loop so autograd sees a fresh tensor
            # node each depth step instead of an in-place mutation chain.
            #
            # ``add_word`` calls were per-cell scalar (each appending to
            # ``subspace.word`` directly) before §6c. The brick-
            # vectorization handoff redirects them to a per-cell tensor
            # scatter into ``subspace.word_records`` / ``word_count``;
            # the chart path flushes those records onto ``subspace.word``
            # before returning so legacy consumers (``decompose``,
            # ``reconstruct``, the SVO walker) see the usual word list.
            # Collect the per-row args here, then dispatch one vector
            # ``add_word`` call per depth step after the row loop.
            live_mask = torch.zeros(B, N, device=live.device, dtype=live.dtype)
            aw_b, aw_left, aw_rule, aw_right = [], [], [], []
            for b in range(B):
                if not pair_positions[b] or best_pair[b] >= len(pair_positions[b]):
                    continue
                left_slot, right_slot = pair_positions[b][best_pair[b]]
                rule_gid = composable_global[best_rule_local[b]]
                merged_cat = rule_lhs_ids[best_rule_local[b]]
                live_mask[b, left_slot] = 1.0
                alive[b, right_slot] = False
                category[b, left_slot] = merged_cat
                self._derivation_trace[b].append(
                    (int(rule_gid), int(left_slot), int(right_slot),
                     int(left_slot), int(merged_cat)))
                aw_b.append(b)
                aw_left.append(int(left_slot))
                aw_rule.append(int(rule_gid))
                aw_right.append(int(right_slot))
            if aw_b:
                dev = data.device
                subspace.add_word(
                    torch.tensor(aw_b, device=dev, dtype=torch.long),
                    torch.tensor(aw_left, device=dev, dtype=torch.long),
                    torch.tensor(aw_rule, device=dev, dtype=torch.long),
                    order=step,
                    leaf1=torch.tensor(aw_left, device=dev, dtype=torch.long),
                    leaf2=torch.tensor(aw_right, device=dev, dtype=torch.long),
                )
            # merged_vec [B, D] broadcast to [B, N, D]; mask selects
            # the left_slot position per batch row. live_mask has no
            # grad_fn (zeros_like + mask assignment), so the version
            # bumps don't propagate into the autograd tape.
            live_mask = live_mask.unsqueeze(-1)
            merged_broadcast = merged_vec.unsqueeze(1).expand(-1, N, -1)
            live = live * (1.0 - live_mask) + merged_broadcast * live_mask

        if depth_probs:
            self.last_rule_probs = torch.stack(depth_probs, dim=1)
        self.last_composable_rules = composable_global
        self._extract_svo_from_trace(grammar, data)
        # Soft-reset signal: a per-source-row "this tick's parse derivation
        # reduced to start_symbol" flag. Only meaningful when the chart has
        # collapsed every leaf into a single live slot whose category is
        # the start nonterminal. Aggregated across the K windows of each
        # source row (any window completing the derivation marks the row
        # as having seen a sentence boundary this tick) so the outer loop's
        # soft_reset(batch=b_source) fires at most once per source row per
        # tick. Host-side only (no GPU sync) — drained by
        # WordSpace.drain_sentence_completed() after runBatch returns.
        self._signal_sentence_completed(subspace, alive, category, grammar)
        # Materialize the tick's tensor word buffer onto ``subspace.word``
        # so consumers that read words directly after compose (tests,
        # standalone calls) see the chart entries. Under runBatch this
        # flush happens before the per-tick End() clear, so legacy
        # consumers in the reverse pass observe the same word list as
        # the pre-§6c scalar-add_word path. The runEpoch outer loop's
        # ``flush_word_buffers`` after this remains correct and idempotent
        # (word_count is already zero post-flush).
        if hasattr(subspace, 'flush_word_buffer'):
            subspace.flush_word_buffer()
        return live, self.last_svo

    def _signal_sentence_completed(self, subspace, alive, category, grammar):
        """Mark source rows whose chart compose reduced to start_symbol.

        Looks up ``WordSpace.start_symbol`` via the grammar, scans
        ``alive`` for rows with exactly one live leaf whose category id
        matches the start id, and writes True into the per-source-row
        ``WordSpace._sentence_completed`` slot. Per-source-row mapping
        ``b_source = b_flat // K`` is recovered from
        ``wordSpace._row_K()``.
        """
        ws = getattr(subspace, 'wordSpace', None)
        if ws is None:
            return
        start_name = getattr(grammar, 'start_symbol', 'S')
        cat_index = getattr(self, '_category_index', None)
        if cat_index is None:
            return
        start_id = cat_index.get(start_name)
        if start_id is None:
            return
        # alive: [B_flat, N] bool; category: [B_flat, N] long.
        live_count = alive.long().sum(dim=1)         # [B_flat]
        # Boolean per row: exactly one live leaf AND its category == start.
        single_live = (live_count == 1)
        # Pull the position of the (single) live leaf for those rows.
        # argmax over the bool mask returns the first True index (which
        # is the only one when single_live is True).
        pos = alive.long().argmax(dim=1)             # [B_flat]
        leaf_cat = category.gather(1, pos.unsqueeze(1)).squeeze(1)  # [B_flat]
        completed_flat = single_live & (leaf_cat == int(start_id))  # [B_flat]
        if not bool(completed_flat.any().item()):
            return
        K = ws._row_K()
        completed_list = completed_flat.tolist()
        sc = getattr(ws, '_sentence_completed', None)
        if sc is None or len(sc) == 0:
            return
        for b_flat, done in enumerate(completed_list):
            if not done:
                continue
            b_source = b_flat // max(1, K)
            if 0 <= b_source < len(sc):
                sc[b_source] = True

    def _compose_vector(self, data, subspace, grammar, target_count=None):
        """3D path: [B, N, D] vector mode. Phase 1 deterministic not at
        top-of-stack, then either ``_compose_to_target`` pairwise reduction
        or Phase 2 cascading soft-weighted composition with query/norm-drop
        detection.
        """
        s_rules = grammar._s_rule_ids()
        not_rid = s_rules.get('not')

        # Phase B dispatch: when chart compose is enabled via
        # <WordSpace.chartCompose>true</chartCompose>, take the chart
        # path. target_count reductions stay on the legacy pairwise path.
        try:
            use_chart = bool(util.TheXMLConfig.get("WordSpace.chartCompose", False))
        except (KeyError, AttributeError, TypeError):
            use_chart = False
        if use_chart and target_count is None:
            return self._compose_vector_chart(data, subspace, grammar)

        # Snapshot codebook indices before any modifications (for decompose)
        basis = getattr(subspace, 'basis', None)
        cb = basis.getW() if basis is not None else None
        if cb is not None and data.shape[-1] == cb.shape[-1]:
            B0, N0, D0 = data.shape
            # ``cb`` may be 2D [V, D0] or 3D [batch, V, D0]; ``mT`` swaps the
            # last two dims for matmul in both cases without the ``.T``
            # deprecation on non-2D tensors.
            self._leaf_cb_indices = (
                data.detach().reshape(-1, D0) @ cb.mT
            ).argmax(dim=-1).reshape(B0, N0)
        else:
            self._leaf_cb_indices = None

        # Phase 1: deterministic not at top-of-stack
        tops = subspace.top_of_stack(data)
        for b, pos in enumerate(tops):
            if pos < 0:
                continue
            vec = data[b, pos]
            if not_rid is not None:
                if vec.mean() < 0:
                    data = data.clone()
                    data[b, pos] = self.notForward(
                        vec.unsqueeze(0).unsqueeze(0),
                        subspace).squeeze(0).squeeze(0)
                    subspace.add_word(b, pos, not_rid)

        if target_count is not None:
            return self._compose_to_target(data, subspace, grammar,
                                           target_count, s_rules, not_rid)

        # Phase 2: soft-weighted cascading composition
        B, N, D = data.shape

        expected_n = self.input_proj.nInput
        if N != expected_n or self.num_rules == 0:
            return data, self.last_svo

        activation = torch.norm(data, dim=-1) / math.sqrt(D)

        out = self.forward(activation)
        rule_probs = out['rule_probs']  # [B, max_depth, num_rules]

        # Composable rules: exclude not (already applied in Phase 1)
        exclude = {'not'}
        composable_local = []
        composable_global = []
        for local_idx, global_id in enumerate(self.all_rules):
            if grammar.rules[global_id].method_name not in exclude:
                composable_local.append(local_idx)
                composable_global.append(global_id)

        if not composable_global:
            return data, self.last_svo

        has_binary = any(grammar.arity(gid) >= 2 for gid in composable_global)
        if not has_binary:
            return data, self.last_svo

        active_positions = [subspace.active_positions(b, data) for b in range(B)]
        max_leaves = max((len(p) for p in active_positions), default=0)
        if max_leaves == 0:
            return data, self.last_svo

        cb_indices = self._leaf_cb_indices
        t_rid = self.transition_rule if self.transition_rule is not None else composable_global[0]
        if cb_indices is not None:
            # One sync for the whole index tensor; inner lookups are free.
            cb_indices_list = cb_indices.tolist()
            for b in range(B):
                cb_row = cb_indices_list[b]
                for i, pos in enumerate(active_positions[b]):
                    if i < max_leaves:
                        subspace.add_word(b, pos, t_rid, order=-1,
                                          leaf1=cb_row[pos])

        masks = torch.zeros(B, max_leaves, N, device=data.device)
        for b in range(B):
            for i, pos in enumerate(active_positions[b]):
                if i < max_leaves:
                    masks[b, i, pos] = 1.0
        leaf_vecs = masks.unsqueeze(-1) * data.unsqueeze(1)  # [B, L, N, D]

        # Positional SVO tap removed 2026-04-20: SVO is now derived from
        # the chart-compose derivation trace by _extract_svo_from_trace.
        # Legacy cascade leaves self.last_svo None (set by compose()).

        composed = leaf_vecs[:, 0, :, :]
        self.last_composable_rules = composable_global
        depth_probs = []

        # query rid for contradiction marker (if present in grammar)
        query_rid = None
        for _gid in self.all_rules:
            if grammar.rules[_gid].method_name == 'query':
                query_rid = _gid
                break
        last_rule_per_batch = [-1 for _ in range(B)]

        d = 0
        leaf_idx = 1
        while d < self.max_depth and leaf_idx < max_leaves:
            left = composed
            right = leaf_vecs[:, leaf_idx, :, :]
            has_third = leaf_idx + 1 < max_leaves

            results = []
            for global_id in composable_global:
                a = grammar.arity(global_id)
                if a == 3 and has_third:
                    third = leaf_vecs[:, leaf_idx + 1, :, :]
                    result = self.project(grammar, global_id, left, right,
                                          third, subspace=subspace)
                elif a == 2:
                    result = self.project(grammar, global_id, left, right,
                                          subspace=subspace)
                else:
                    result = self.project(grammar, global_id, left,
                                          subspace=subspace)
                results.append(result)

            results = torch.stack(results, dim=1)  # [B, n_composable, N, D]

            probs_d = rule_probs[:, d, :][:, composable_local]
            probs_d = probs_d / (probs_d.sum(dim=-1, keepdim=True) + 1e-8)
            depth_probs.append(probs_d.detach())

            best = probs_d.argmax(dim=-1)  # [B]

            # Phase 2 candidate mixture
            if self.training:
                probs_bcast = probs_d.unsqueeze(-1).unsqueeze(-1)
                candidate = (probs_bcast * results).sum(dim=1)  # [B, N, D]
            else:
                idx = best.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                idx = idx.expand(-1, 1, results.shape[2], results.shape[3])
                candidate = results.gather(1, idx).squeeze(1)   # [B, N, D]

            # Query / norm-drop: detect symbolic contradiction at the
            # accumulation point and preserve the prior accumulator.
            # Norm collapses the [N, D] axes to a scalar per batch row.
            prev_norm = left.reshape(B, -1).norm(dim=-1)
            cand_norm = candidate.reshape(B, -1).norm(dim=-1)
            drop_threshold = self._QUERY_NORM_DROP_RATIO * prev_norm
            query_mask = (prev_norm > 1e-6) & (cand_norm < drop_threshold)
            # One sync per tensor per d-step, not per b-row.
            query_mask_list = query_mask.tolist()
            best_list = best.tolist()
            if query_mask.any():
                mask_exp = query_mask.view(B, 1, 1).expand_as(candidate)
                composed = torch.where(mask_exp, left, candidate)
                word_sub = getattr(self, 'word_subspace', None)
                for b in range(B):
                    if not query_mask_list[b]:
                        continue
                    left_rid = last_rule_per_batch[b]
                    right_gid = composable_global[best_list[b]]
                    if query_rid is not None and word_sub is not None:
                        word_sub.push(b, query_rid,
                                      leaves=(left_rid, right_gid, -1))
            else:
                composed = candidate

            # Record argmax rule as word
            best_global = composable_global[best_list[0]]
            for b in range(B):
                if d < len(active_positions[b]):
                    best_gid_b = composable_global[best_list[b]]
                    subspace.add_word(
                        b,
                        active_positions[b][min(d, len(active_positions[b]) - 1)],
                        best_gid_b)
                    if not query_mask_list[b]:
                        last_rule_per_batch[b] = best_gid_b

            best_arity = grammar.arity(best_global)
            leaf_idx += (2 if best_arity == 3 and has_third else 1)
            d += 1

        if depth_probs:
            self.last_rule_probs = torch.stack(depth_probs, dim=1)
        return composed, self.last_svo

    def _compose_to_target(self, data, subspace, grammar, target_count,
                           s_rules, not_rid):
        """Reduce active tokens to target_count via pairwise grammar reductions."""
        B, N, D = data.shape

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

        activation = torch.norm(data, dim=-1) / math.sqrt(D)
        expected_n = self.input_proj.nInput
        if N == expected_n and self.num_rules > 0:
            out = self.forward(activation)
            rule_probs = out['rule_probs']
        else:
            rule_probs = torch.ones(B, self.max_depth, max(len(self.all_rules), 1),
                                    device=data.device) / max(len(self.all_rules), 1)

        active = [subspace.active_positions(b, data) for b in range(B)]

        cb_indices = self._leaf_cb_indices
        t_rid = self.transition_rule if self.transition_rule is not None else composable_global[0]
        if cb_indices is not None:
            # One sync for the whole index tensor; inner lookups are free.
            cb_indices_list = cb_indices.tolist()
            for b in range(B):
                cb_row = cb_indices_list[b]
                for pos in active[b]:
                    subspace.add_word(b, pos, t_rid, order=-1,
                                      leaf1=cb_row[pos])

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
                    results = torch.stack(results, dim=1)

                    probs_d = rule_probs[b:b+1, min(d, rule_probs.shape[1]-1), :]
                    probs_d = probs_d[:, composable_local]
                    probs_d = probs_d / (probs_d.sum(dim=-1, keepdim=True) + 1e-8)

                    best_local = probs_d.argmax(dim=-1)[0].item()
                    if self.training:
                        composed = (probs_d.unsqueeze(-1).unsqueeze(-1) * results).sum(dim=1)
                    else:
                        composed = results[:, best_local]

                    new_data[b, left_pos] = composed[0, 0]
                    new_data[b, right_pos] = 0.0

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

        result = torch.zeros(B, N, D, device=data.device)
        for b in range(B):
            for pos in active[b]:
                result[b, pos] = data[b, pos]

        return result, self.last_svo

    def decompose(self, data, subspace, grammar):
        """Reconstruct pre-compose tensor from the symbolic word record.

        Terminal words (order == -1) carry codebook indices of the original
        leaf vectors.  Reconstruction looks up each leaf from the codebook
        and places it at its recorded position, producing the exact
        pre-compose tensor without any cached tensors.  For 2D activation
        input (or when no codebook is available) the data is returned
        as-is -- the only surviving rule-word inverses (not, union,
        intersection, lift, lower) were already absorbed into the leaf
        representations in compose.
        """
        basis = getattr(subspace, 'basis', None)
        cb = basis.getW() if basis is not None else None
        if cb is None or data.ndim < 3 or data.shape[-1] != cb.shape[-1]:
            # No usable codebook -- fall through to the S-tier-style rule
            # reversal pass (currently a no-op; rules are lossy).
            words = subspace.get_words()
            for word in reversed(words):
                if len(word) < 3:
                    continue
                rule_id = word[WordEncoding.RULE]
                rule = grammar.rules[rule_id]
                if rule.method_name in ('non', 'union', 'intersection',
                                        'lift', 'lower'):
                    pass  # Non-invertible in this degraded path
                elif rule.method_name is not None:
                    pass  # Not cleanly invertible
            return data

        words = subspace.get_words()
        result = torch.zeros_like(data)
        for word in words:
            if word[WordEncoding.ORDER] != -1:
                continue
            b = word[WordEncoding.BATCH]
            pos = word[WordEncoding.VECTOR]
            cb_idx = word[WordEncoding.LEAF1]
            if cb_idx >= 0:
                result[b, pos] = cb[cb_idx]
        return result

    # -- utilities -------------------------------------------------

    def set_tau(self, tau):
        """Anneal the Gumbel-softmax temperature."""
        self.tau = tau

    def flush_word_buffer(self, subspace):
        """Drain the tick's tensor word buffer into ``subspace.word``.

        Outer-loop hook for the brick-vectorization handoff §6c (Path
        B). The chart path now materializes before compose returns, so
        the outer doc-streaming loop's post-brick call is normally
        idempotent (``word_count`` is already zero). Keeping this wrapper
        preserves the documented call site for any path that still writes
        buffered entries via the vector-typed ``subspace.add_word``
        overload.

        Thin wrapper -- the buffer state lives on the SubSpace so per-
        subspace word lists stay independent. The method exists on
        SyntacticLayer to match the call site documented in the plan
        (``wordSpace.syntacticLayer.flush_word_buffer(subspace)``).
        """
        if subspace is None:
            return
        flush = getattr(subspace, 'flush_word_buffer', None)
        if flush is not None:
            flush()


class CategoryStack:
    """Per-row push/pop stack of derivation-state embeddings.

    Holds one learned embedding per category slot pushed during parsing
    (e.g. S, VO). Consumed by the rule predictor MLP as a flattened
    window over recent stack frames.

    Storage is one Python list per batch row (``self._entries[b]``).
    The spec proposed a ``[B, max_depth, dim]`` tensor backing, but
    in-place ``__setitem__`` on a non-grad tensor breaks autograd
    propagation back to the pushed vec — and the rule-predictor
    gradient test depends on that propagation. List-of-lists preserves
    autograd through ``torch.cat`` in ``flatten`` while still giving
    per-row isolation under microbatch ``B*K`` rows.
    """

    def __init__(self, dim, batch=1, max_depth=64):
        self._dim = int(dim)
        self._batch = int(batch)
        self._max_depth = int(max_depth)
        self._entries = [[] for _ in range(self._batch)]

    def ensure_batch(self, batch):
        batch = int(batch)
        if batch == self._batch:
            return
        self._batch = batch
        self._entries = [[] for _ in range(self._batch)]

    def push(self, b, vec):
        assert vec.shape == (self._dim,), (
            f"CategoryStack dim={self._dim}, got vec shape {tuple(vec.shape)}"
        )
        assert len(self._entries[b]) < self._max_depth, (
            f"CategoryStack overflow at row {b}: max_depth={self._max_depth}"
        )
        self._entries[b].append(vec)

    def pop(self, b):
        return self._entries[b].pop()

    def depth(self, b):
        return len(self._entries[b])

    def flatten(self, b):
        if not self._entries[b]:
            return torch.zeros(0)
        return torch.cat(self._entries[b], dim=0)

    def clear_rows(self, start, end):
        """Empty rows ``[start, end)``. Used by per-row hard reset."""
        for b in range(int(start), min(int(end), self._batch)):
            self._entries[b] = []

class ReconstructionStack:
    """Per-row tuple stack of (rule_id, word_id). Tensor-backed for B>1.

    Storage is ``[B, max_depth, 2] long`` with a ``[B] long`` top index.
    Push-only in production today; peek/pop kept for tests and future
    generation-from-meaning consumers. Not consumed by the rule
    predictor or sentence prediction.
    """

    def __init__(self, batch=1, max_depth=64):
        self._batch = int(batch)
        self._max_depth = int(max_depth)
        self._entries = torch.zeros(self._batch, self._max_depth, 2,
                                    dtype=torch.long)
        self._top = torch.zeros(self._batch, dtype=torch.long)

    def ensure_batch(self, batch):
        batch = int(batch)
        if batch == self._batch:
            return
        self._batch = batch
        self._entries = torch.zeros(batch, self._max_depth, 2,
                                    dtype=torch.long)
        self._top = torch.zeros(batch, dtype=torch.long)

    def push(self, b, rule_id, word_id):
        idx = int(self._top[b].item())
        assert idx < self._max_depth, (
            f"ReconstructionStack overflow at row {b}: max_depth={self._max_depth}"
        )
        self._entries[b, idx, 0] = int(rule_id)
        self._entries[b, idx, 1] = int(word_id)
        self._top[b] += 1

    def peek(self, b):
        idx = int(self._top[b].item()) - 1
        return (int(self._entries[b, idx, 0].item()),
                int(self._entries[b, idx, 1].item()))

    def pop(self, b):
        self._top[b] -= 1
        idx = int(self._top[b].item())
        return (int(self._entries[b, idx, 0].item()),
                int(self._entries[b, idx, 1].item()))

    def clear_rows(self, start, end):
        """Reset rows ``[start, end)`` to empty stacks. Per-row hard reset."""
        s, e = int(start), min(int(end), self._batch)
        if e <= s:
            return
        self._entries[s:e].zero_()
        self._top[s:e] = 0

    def depth(self, b):
        return int(self._top[b].item())

class WordSpace(Space):
    """Service space that owns the word-stream buffer, the SyntacticLayer,
    the truth store, and the inter-sentence discourse substrate.

    Runtime-parallel to PerceptualSpace / ConceptualSpace / SymbolicSpace
    but functionally a buffer + composition dispatcher. WordSpace owns a
    single unified ``SyntacticLayer``; home spaces receive ``wordSpace``
    as a per-call parameter on ``forward(vspace, wordSpace=...)`` /
    ``reverse(vspace, wordSpace=...)`` and reach the layer via
    ``forwardSymbols`` / ``reverseSymbols``. The layer pushes its word
    records into ``self.subspace`` (a ``WordSubSpace``) via a
    back-reference set at construction time, so ConceptualSpace can read
    a muxed view of machine state that includes percepts, symbols, and
    words.

    One unified constructor builds everything: WordSubSpace, the
    SyntacticLayer, TruthLayer, and (conditionally) DiscourseSpace.
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

        # 4. Unified SyntacticLayer slot (filled below). Post subclass-
        # merge (2026-04-19) there is a single SyntacticLayer that owns
        # every compositional rule and is called from SymbolicSpace.
        self.syntacticLayer = None
        # 5. Build the SyntacticLayer anchored at SymbolicSpace.  The
        # perceptual and conceptual spaces also get a ``wordSpace``
        # back-reference so they can route through the shared buffer,
        # but only the symbolic space's compose() fires the layer.
        if perceptualSpace is not None:
            perceptualSpace.attach_wordSpace(self)
        if conceptualSpace is not None:
            conceptualSpace.attach_wordSpace(self)
        if symbolicSpace is not None:
            self._build_syntactic_layer(
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

        # 6b. Category codebook -- learned embedding per derivation label.
        # The first len(TheGrammar.categories) rows are reserved one-per-
        # label (category_index maps 'S' -> 0, 'VO' -> 1, ... in sorted
        # order); extra capacity is kept for the legacy pos_lookup path
        # (nearest-neighbor over activations). Not registered in
        # self.layers (no training-loop integration yet); the
        # VectorQuantize backend provides the nn.Module bookkeeping.
        #
        # Step 6: capacity is now max(64, len(categories)) so a richer
        # grammar.cfg (Layer 1 productions add VO, NP, VP, AP, MP, PP,
        # DEF, HAS plus the closed-class terminals) cannot overflow the
        # per-label slot reservation.  Legacy XML grammars stay at 64
        # since their category set is small.
        pos_dim = 4  # embedding width; also the category stack vector dim
        category_capacity = max(64, len(TheGrammar.categories))
        self.category_codebook = Codebook()
        self.category_codebook.create(
            nInput=0,           # input-side width unused for direct addressing
            nVectors=category_capacity,
            nDim=pos_dim,
            customVQ=True,
            monotonic=False,
            passThrough=False,
        )
        self.category_index = {
            name: idx for idx, name in enumerate(TheGrammar.categories)
        }
        # 6c. Category stack -- push/pop store for category-embedding
        # vectors during parsing. One frame per reduction step.
        self.category_stack = CategoryStack(dim=pos_dim)

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
        # pos_dim already bound above (category_codebook / category_stack dim).
        rule_in_features = max_depth * pos_dim
        # When nPercepts=0 (minimal test configs with no PerceptualSpace),
        # rule_in_features is 0; nn.Linear(0, 0) would emit a "zero-element
        # tensor init is a no-op" UserWarning. Widen to 1 feature so init
        # is well-defined. predict_rule pads the flattened stack to the
        # same target_len, so the head stays consistent with the stack.
        self._rule_predictor_in_features = max(1, rule_in_features)
        # Hidden dim is bottlenecked: the legacy square form
        # ``Linear(in, in)`` ballooned to ~17M params at in=4096
        # (~80% of the model).  A 256-wide bottleneck keeps the
        # capacity-vs-rule-count ratio healthy at the rule-counts
        # currently in use (a few dozen) while shrinking the layer
        # ~16x.  Caps at the input width so tiny test configs (where
        # in_features < 256) don't gain spurious capacity.
        rule_hidden = min(self._rule_predictor_in_features, 256)
        self.rule_predictor = nn.Sequential(
            nn.Linear(self._rule_predictor_in_features, rule_hidden),
            nn.Tanh(),
            nn.Linear(rule_hidden, max(1, n_rules)),
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

        # -- pipeline-carried per-batch state -----------------------------
        # batch / svo_dim track the per-row state allocations below.
        # ensure_batch() resizes them in step.
        self.batch = 1
        self.svo_dim = int(symbol_dim)

        # last_svo: (subject, verb, object) snapshot from the most recent
        # chart-compose trace. Stored as [B, 3, svo_dim] + a [B] bool valid
        # mask so each batch row is independent. Written via set_last_svo;
        # cleared by clear_last_svo (also at Reset on sentence boundary).
        # Registered as buffers so .to(device) moves them with the module.
        self.register_buffer(
            "_last_svo", torch.zeros(self.batch, 3, self.svo_dim))
        self.register_buffer(
            "_svo_valid", torch.zeros(self.batch, dtype=torch.bool))

        # STM-residual: fires once per sentence per row on the first
        # stm_residual(b) call; arm_stm() / Reset() re-arm. Buffer for
        # device-tracking parity with the SVO state above.
        self.register_buffer(
            "_stm_fired", torch.zeros(self.batch, dtype=torch.bool))
        self.stm_residual_scale = float(
            TheXMLConfig.training("sentencePrimingScale", 0.05) or 0.05)

        # Per-source-row sentence-completed signal driven by
        # SyntacticLayer.compose: True for row b when this tick's parse
        # derivation reduced to Grammar.start_symbol. Outer doc-streaming
        # loop drains via drain_sentence_completed() after each runBatch
        # and dispatches soft_reset(batch=b) for True rows. Host-side
        # list (no GPU sync); resized to B by ensure_microbatch.
        self._sentence_completed = [False] * self.batch

    # -- per-row last_svo accessors ---------------------------------------
    def set_last_svo(self, b, subj, verb, obj):
        """Write the SVO triple for batch row ``b``."""
        self._last_svo[b, 0] = subj
        self._last_svo[b, 1] = verb
        self._last_svo[b, 2] = obj
        self._svo_valid[b] = True

    def get_last_svo(self, b):
        """Return ``(subj, verb, obj)`` tensors for batch row ``b``."""
        e = self._last_svo[b]
        return e[0], e[1], e[2]

    def svo_valid(self, b):
        """True iff set_last_svo has fired for row ``b`` since last clear."""
        return bool(self._svo_valid[b].item())

    def clear_last_svo(self, b=None):
        """Clear the SVO valid mask for row ``b`` (or all rows when None)."""
        if b is None:
            self._svo_valid.zero_()
        else:
            self._svo_valid[b] = False

    # -- per-row STM-fired accessors --------------------------------------
    def stm_fired(self, b):
        """True iff stm_residual(b) has fired since last arm."""
        return bool(self._stm_fired[b].item())

    def mark_stm_fired(self, b):
        """Mark row ``b`` as having fired its STM residual this sentence."""
        self._stm_fired[b] = True

    def arm_stm(self, b=None):
        """Re-arm row ``b`` (or all rows when None) for the next sentence."""
        if b is None:
            self._stm_fired.zero_()
        else:
            self._stm_fired[b] = False

    def stm_residual(self, b=0):
        """Discourse prediction bias applied once per sentence per row.

        Reads ``self.discourse.predict()`` and returns
        ``discourse.prime(predicted, confidence, scale)``, or ``None`` when
        discourse is unavailable, not yet built, or the bias already fired
        this sentence for row ``b``.  ``arm_stm(b)`` / ``Reset()`` re-arms.

        ``b`` defaults to 0 for back-compat with single-row callers; the
        Task 9 cutover threads the row index from the body iteration.
        """
        if self.stm_fired(b):
            return None
        self.mark_stm_fired(b)
        disc = self.discourse
        if disc is None:
            return None
        pred, conf = disc.predict()
        return disc.prime(pred, conf, self.stm_residual_scale)

    def stm_residual_microbatch(self, B, K, expected_dim=None):
        """Vectorized STM residual for the microbatch body.

        For each source row ``b`` in ``[0, B)``: if ``_stm_fired[b]`` is
        False, this call contributes one discourse-bias term that broadcasts
        across all ``K`` windows derived from that source row.  Sources
        already fired contribute zero.  After the call, every source that
        contributed is marked fired.

        Returns a ``[B*K, concept_dim]`` tensor, or ``None`` when discourse
        is unavailable, every source row has already fired this sentence,
        or the priming vector does not live in the caller's basis width.

        The call site (ConceptualSpace.forward) broadcasts the result over
        the ``N`` axis via ``bias.unsqueeze(1)``.
        """
        BK = int(B) * int(K)
        not_fired = ~self._stm_fired  # [B] bool, no host sync
        # Removed: `if not bool(not_fired.any().item()): return None` early-out.
        # The gate at the bottom multiplies the bias by ``not_fired``, so when
        # every row has already fired the returned tensor is all-zero and the
        # caller's `y = y + bias.unsqueeze(1)` is a no-op anyway. The early-out
        # was a per-batch host sync that blocked CUDA-graph capture. We pay
        # one extra `disc.predict()` call per tick when no rows fire (cheap;
        # one matmul) in exchange for no GPU->CPU sync.
        disc = self.discourse
        if disc is None:
            return None
        pred, conf = disc.predict()
        if pred is None or conf is None:
            return None
        bias_full = disc.prime(pred, conf, self.stm_residual_scale)
        if bias_full is None:
            return None
        # predict() returns 1D ``[s_dim]`` for B=1 layers and 2D ``[B*K,
        # s_dim]`` for batched layers; prime() preserves that rank.  Lift
        # 1D to a single-row 2D tensor so downstream broadcasting is uniform.
        if bias_full.ndim == 1:
            bias_full = bias_full.unsqueeze(0)
        # Expand a single-row discourse output up to the body batch when
        # discourse and body have diverged (legacy non-microbatch paths
        # leave discourse at its construction batch).
        if bias_full.shape[0] != BK:
            if bias_full.shape[0] == 1:
                bias_full = bias_full.expand(BK, -1).contiguous()
            else:
                # Mismatched and not broadcastable: keep semantics safe by
                # skipping the bias rather than mis-broadcasting.
                return None
        if expected_dim is not None and int(bias_full.shape[-1]) != int(expected_dim):
            # Hierarchical/butterfly stages may operate in a packed state
            # basis that is narrower than the global DiscourseSpace concept
            # projection. Do not inject a residual across incompatible bases.
            return None
        # Gate per source row: each source's bias broadcasts to its K
        # windows; sources already fired are masked to zero.
        gate = not_fired.repeat_interleave(int(K)).to(bias_full.device)
        bias_full = bias_full * gate.to(bias_full.dtype).unsqueeze(-1)
        # Mark sources that contributed.
        self._stm_fired = self._stm_fired | not_fired
        return bias_full

    # -- Category helpers ---------------------------------------------
    def category_lookup(self, category):
        """Return the learned codebook embedding for a derivation label.

        One entry per label: ``category_index['S'] = 0``,
        ``category_index['VO'] = 1``, etc. Looks up directly by index —
        no activation-similarity snap — so this is the canonical path
        for "embedding for category X" when the label is already known
        from the grammar.

        Args:
            category: string category name ('S', 'VO', ...) or int
                row index into ``category_codebook``.

        Returns:
            Tensor of shape ``(pos_dim,)`` — the codebook row.
        """
        if isinstance(category, str):
            idx = self.category_index[category]
        else:
            idx = int(category)
        return self.category_codebook.getW()[idx]

    # -- PoS helpers (legacy) -----------------------------------------
    def pos_lookup(self, active_symbols):
        """Activation-similarity nearest-neighbor lookup into the codebook.

        Legacy path kept so ``SymbolicSpace.forward`` (and its tests) can
        map an active-symbol pattern to a codebook row without knowing
        the grammar category up-front. New code that already has the
        category name should use ``category_lookup(name)`` instead.

        Args:
            active_symbols: 1-D tensor of shape [N], typically resolved
                activations from SymbolicSpace.resolve().

        Returns:
            Tensor of shape (pos_dim,) -- the matching prototype row.
        """
        w = self.category_codebook.getW()  # [nVectors, pos_dim]
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
    def predict_rule(self, b=0):
        """Emit softmax logits over the rule table from row ``b``'s category stack.

        Reads ``self.category_stack.flatten(b)`` as a 1-D tensor of
        length ``depth * pos_dim``. Zero-pads up to
        ``self._rule_predictor_in_features`` when the stack is shallower
        than ``max_depth``; truncates (keeping the most recent frames)
        when the stack has overflowed, so the head always sees a fixed-
        width window of the top ``max_depth`` category embeddings.
        Returns a tensor of shape ``(n_rules,)`` suitable for
        ``torch.softmax`` / CE loss.
        """
        assert self.rule_predictor[-1].out_features == len(TheGrammar.rule_table), (
            "Grammar reconfigured after WordSpace construction; rule_predictor stale"
        )
        flat = self.category_stack.flatten(b)
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

    def predict_rule_hard(self, b=0):
        """Return argmax rule_id for inference.

        Detached from autograd (wraps predict_rule in no_grad). If gradients
        through the argmax path are ever needed (e.g., REINFORCE baseline,
        Gumbel-argmax), call predict_rule(b).argmax() directly instead.
        """
        with torch.no_grad():
            return int(self.predict_rule(b).argmax().item())

    # -- reconstruction stack -----------------------------------------
    def record_derivation(self, rule_id, word_id, b=0):
        """Record a (rule_id, word_id) derivation step on row ``b``'s reconstruction stack.

        Placeholder surface until generation-from-meaning is solved. The
        stack is not consumed by the rule predictor or sentence prediction.
        """
        self.reconstruction_stack.push(b, rule_id, word_id)

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
           where ``lum_norm = luminosity(symbolic_space.sigma).clamp(0, 1)``
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
        ``syntacticLayer``.

        The ``kind`` argument is retained for backward compatibility
        with external callers but is no longer meaningful; every
        invocation targets the single unified layer.  Sets
        ``layer.word_subspace`` as a back-reference so compose() can
        push onto the shared buffer, appends the layer to
        ``self.layers`` for ``Space.paramUpdate`` delegation, and
        merges its parameters into ``self.params`` for the curated
        ``Space.getParameters`` walk.
        """
        if layer is None:
            return
        self.syntacticLayer = layer
        layer.word_subspace = self.subspace
        if layer not in self.layers:
            self.layers.append(layer)
        for p in layer.parameters():
            if all(p is not q for q in self.params):
                self.params.append(p)

    # -- private factory helper: build + wire the SyntacticLayer -----
    def _resolve_hidden_dim(self, n_slots):
        try:
            configured = int(TheXMLConfig.get("WordSpace.syntacticHiddenDim"))
            if configured > 0:
                return configured
        except (KeyError, TypeError, ValueError):
            pass
        return min(256, max(64, n_slots * 4))

    def _build_syntactic_layer(self, space, n_slots, grammar, symbol_dim):
        """Build the unified SyntacticLayer anchored at the symbolic space.

        Post subclass-merge (2026-04-19): ``rules`` comes from
        ``grammar.symbolic()`` (all compositional rules live on S),
        ``init_swap`` is always called because every instance now
        supports the full rule set, and there is a single
        ``self.syntacticLayer`` attribute instead of three per-tier
        slots.
        """
        layer = SyntacticLayer(
            nInput=n_slots, nOutput=n_slots,
            rules=grammar.symbolic(),
            transition_rule=grammar.symbolic_transition(),
            max_depth=max(n_slots - 1, 1),
            hidden_dim=self._resolve_hidden_dim(n_slots),
            grammar=grammar,
            feature_dim=symbol_dim,
        )
        layer.init_swap(symbol_dim, n_slots)
        layer.init_lifting(symbol_dim)
        self.attach_codebook_host(space)
        self.attach_layer('syntactic', layer)
        space.attach_wordSpace(self)
        return layer

    # -- composition dispatch ----------------------------------------
    def forwardSymbols(self, data, subspace):
        """Compose via the unified SyntacticLayer.

        Dispatches to ``self.syntacticLayer.compose(data, subspace, grammar)``.
        Includes the Rule #2 demux side effect: the muxed [B, N, D] symbol
        tensor gets split into what/where/when modality slots before
        compose runs, so slot selectors see axis-separated state.
        Returns the composed tensor (stripping the SVO slot of the
        compose tuple so callers keep the pre-merge tensor contract).
        """
        layer = self.syntacticLayer
        if layer is None:
            return data
        if data.ndim == 3 and data.shape[-1] == getattr(subspace, 'muxedSize', -1):
            subspace.demux(data)
        result = layer.compose(data, subspace, TheGrammar)
        if isinstance(result, tuple):
            return result[0]
        return result

    def reverseSymbols(self, data, subspace):
        """Reverse-compose via the unified SyntacticLayer."""
        layer = self.syntacticLayer
        if layer is None:
            return data
        return layer.decompose(data, subspace, TheGrammar)

    def reconstruct(self, state, codebook_space, max_tokens=1):
        """Run the downward grammar on a deep state.

        MVP: emit_head on ``S -> C`` exactly once, returning the head atom
        index and the residual. ``max_tokens > 1`` is reserved for later
        expansion (e.g. NP VP templates that consume the residual).

        ``codebook_space`` is any space whose ``subspace.basis`` exposes a
        ``getW() -> [V, D]`` codebook (SymbolicSpace for internal atoms,
        InputSpace for word-vocab heads that can be decoded back to text).

        Returns a dict:
          'heads':      list[int] of length 1
          'contained':  [B, D] tensor -- atom contribution
          'residual':   [B, D] tensor -- leftover meaning after emission
          'state':      [B, D] tensor -- original input state (echo)
        """
        layer = self.syntacticLayer
        if layer is None or state is None:
            return {'heads': [], 'residual': state, 'state': state}
        cb = codebook_space.subspace.basis
        # No codebook wired (e.g. a passthrough/non-VQ SymbolicSpace):
        # emit a trivial "closest-head" = 0 for every batch row so
        # callers (MentalModel._predicted_head) get a well-formed list
        # instead of crashing on cb.getW() == None. The residual equals
        # the input state (no atom subtracted).
        if cb is None or getattr(cb, 'getW', lambda: None)() is None:
            B = state.shape[0]
            return {
                'heads': [0] * B,
                'contained': torch.zeros_like(state),
                'residual': state,
                'state': state,
            }
        idx, contained, residual = layer.emit_head(state, cb)
        return {
            'heads': idx.tolist(),
            'contained': contained,
            'residual': residual,
            'state': state,
        }

    # -- buffer access + lifecycle ------------------------------------
    def read(self):
        """Return the fixed-width stack tensor for ConceptualSpace to
        concat with percepts and symbols.
        """
        return self.subspace.read()

    def clear_sentence(self):
        """Reset the stack at sentence boundaries."""
        self.subspace.clear()

    def Reset(self, batch=None, hard=True):
        """Per-document teardown called by the outer doc-streaming loop.

        ``batch`` (optional int): clear per-row state only for source
        row ``batch``. ``None`` clears every row. When ``batch`` is set,
        the SyntacticLayer per-cell state at rows ``batch*K..(batch+1)*K``
        is cleared via the underlying stack helpers; the per-source-row
        ``_stm_fired[batch]`` is re-armed.

        ``hard`` (default True): True is the document boundary (full
        wipe of stack, SVO, STM, discourse). False is a sentence-internal
        soft reset — see ``soft_reset(batch=b)`` for the structured entry
        point. The base ``Reset`` cascades both flags down to the layer
        chain so child layers can opt in.
        """
        super().Reset(batch=batch, hard=hard)
        if not hard:
            # Soft reset (sentence boundary): callers should use
            # soft_reset(batch=b) directly. Treat a soft Reset as a
            # request to soft_reset every row in batch=None scope so the
            # cascade stays well-defined for callers that pass hard=False.
            if batch is None:
                B = int(self._stm_fired.shape[0])
                for b in range(B):
                    self.soft_reset(batch=b)
            else:
                self.soft_reset(batch=batch)
            return
        if batch is None:
            self.clear_sentence()
            # Re-arm STM residual on every row so the next sentence fires
            # once per row; drop the stale per-row SVO so composed-chart
            # readers don't carry it across sentence boundaries.
            self.arm_stm()
            self.clear_last_svo()
            self.clear_sentence_completed()
            return
        # Per-row hard reset. The body-side parse stack lives at
        # [B*K, ...] so the row's K cells span indices [batch*K, (batch+1)*K).
        # Use the existing per-row helpers where they exist; fall back to
        # a localized clear when not.
        K = self._row_K()
        bk_start, bk_end = int(batch) * K, (int(batch) + 1) * K
        # Stack: clear the K cells owned by this source row.
        sub = getattr(self, 'subspace', None)
        if sub is not None and hasattr(sub, 'clear_rows'):
            sub.clear_rows(bk_start, bk_end)
        elif sub is not None:
            # Fallback: full subspace clear keeps semantics safe even if
            # the per-row helper isn't available; the next forward will
            # reseed per-row state from the input.
            sub.clear()
        if hasattr(self, 'category_stack') and self.category_stack is not None:
            if hasattr(self.category_stack, 'clear_rows'):
                self.category_stack.clear_rows(bk_start, bk_end)
        if (hasattr(self, 'reconstruction_stack')
                and self.reconstruction_stack is not None):
            if hasattr(self.reconstruction_stack, 'clear_rows'):
                self.reconstruction_stack.clear_rows(bk_start, bk_end)
        # Per-source-row STM / SVO / sentence-complete signal.
        self.arm_stm(int(batch))
        self.clear_last_svo(int(batch))
        self.clear_sentence_completed(int(batch))

    def soft_reset(self, batch=None):
        """Re-arm sentence-internal state for row ``batch`` (or all rows).

        Soft reset fires when the parse derivation for a row reaches the
        configured ``Grammar.start_symbol`` — the structural sentence
        boundary. Clears every per-sentence working buffer so the next
        sentence starts fresh, while preserving the document-scoped
        carryover that bridges sentences:
          * **Cleared**: parse stack, category stack, reconstruction
            stack, ``_last_svo[batch]``, ``_stm_fired[batch]`` (re-armed);
            sentence-completed flag.
          * **Preserved**: ``InterSentenceLayer`` discourse history (the
            centroid ring buffer + the predictive bias) — this is the
            inter-sentence prior, accumulating across true sentences
            within a document.
          * **Preserved**: codebook EMA, learned weights — those are
            training-time state, not per-sentence context.

        Differs from a hard reset (``Reset(batch=b, hard=True)``) by
        leaving discourse history alone; hard reset wipes that too.
        """
        if batch is None:
            self.arm_stm()
            self.clear_last_svo()
            self.clear_sentence_completed()
            # Reset every row's parse-side working state. clear_sentence
            # zeroes the WordSubSpace stack; the category and
            # reconstruction stacks fan out to the same row count.
            self.clear_sentence()
            if (hasattr(self, 'category_stack')
                    and self.category_stack is not None
                    and hasattr(self.category_stack, 'clear_rows')):
                self.category_stack.clear_rows(0, self.batch)
            if (hasattr(self, 'reconstruction_stack')
                    and self.reconstruction_stack is not None
                    and hasattr(self.reconstruction_stack, 'clear_rows')):
                self.reconstruction_stack.clear_rows(0, self.batch)
            return
        b = int(batch)
        self.arm_stm(b)
        self.clear_last_svo(b)
        self.clear_sentence_completed(b)
        # Per-row clear over the K cells [b*K, (b+1)*K) that own this
        # source row in the body's flattened microbatch view.
        K = self._row_K()
        bk_start, bk_end = b * K, (b + 1) * K
        sub = getattr(self, 'subspace', None)
        if sub is not None and hasattr(sub, 'clear_rows'):
            sub.clear_rows(bk_start, bk_end)
        if (hasattr(self, 'category_stack')
                and self.category_stack is not None
                and hasattr(self.category_stack, 'clear_rows')):
            self.category_stack.clear_rows(bk_start, bk_end)
        if (hasattr(self, 'reconstruction_stack')
                and self.reconstruction_stack is not None
                and hasattr(self.reconstruction_stack, 'clear_rows')):
            self.reconstruction_stack.clear_rows(bk_start, bk_end)

    def _row_K(self):
        """Per-source-row K (microbatch window count) inferred from state.

        ``_stm_fired`` is sized [B] (per source row); the body-side
        ``self.batch`` is sized [B*K]. The ratio recovers K. Returns 1
        when no microbatch has been allocated yet.
        """
        try:
            B = int(self._stm_fired.shape[0])
        except (AttributeError, IndexError, TypeError):
            return 1
        if B <= 0:
            return 1
        return max(1, int(self.batch) // B)

    def clear_sentence_completed(self, batch=None):
        """Clear the sentence-completed signal for row ``batch`` (or all).

        ``_sentence_completed`` is a host-side ``list[bool]`` of length B
        that ``SyntacticLayer.compose`` appends into when a row's
        derivation reduces to ``Grammar.start_symbol``. The outer
        doc-streaming loop drains it after each ``runBatch``.
        """
        # Lazy-init so callers (and Reset) work before the first compose
        # has populated the list.
        if not hasattr(self, '_sentence_completed') or self._sentence_completed is None:
            try:
                B = int(self._stm_fired.shape[0])
            except (AttributeError, IndexError, TypeError):
                B = 1
            self._sentence_completed = [False] * B
            return
        if batch is None:
            for i in range(len(self._sentence_completed)):
                self._sentence_completed[i] = False
            return
        b = int(batch)
        if 0 <= b < len(self._sentence_completed):
            self._sentence_completed[b] = False

    def drain_sentence_completed(self):
        """Return-and-clear the per-row sentence-completed signal.

        Outer loop pattern (post-runBatch):
          ``for b in wordSpace.drain_sentence_completed(): soft_reset(b)``

        Returns a ``list[int]`` of source-row indices whose derivation
        completed during the last compose; the underlying buffer is then
        cleared so the next tick starts from a clean slate.
        """
        if not hasattr(self, '_sentence_completed') or self._sentence_completed is None:
            return []
        completed = [
            i for i, v in enumerate(self._sentence_completed) if v]
        for i in completed:
            self._sentence_completed[i] = False
        return completed

    def get_blocks(self, b=0):
        """Return the parse-tree ledger for batch row `b`."""
        return self.subspace.get_blocks(b)

    def ensure_batch(self, batch):
        """Resize the underlying buffer + per-batch stacks to a new batch size.

        ensure_batch is the single fan-out point for every per-row buffer
        WordSpace owns: the WordSubSpace event, the CategoryStack /
        ReconstructionStack stacks, and the Task-2 ``last_svo`` /
        ``_stm_fired`` tensors.  Reallocates fresh storage; per-row state
        is zeroed.
        """
        batch = int(batch)
        if batch == self.batch:
            # Cascade still runs in case callers grew their own state
            # without going through the WordSpace.batch counter.
            self.subspace.ensure_batch(batch)
            self.category_stack.ensure_batch(batch)
            self.reconstruction_stack.ensure_batch(batch)
            return
        self.batch = batch
        self.subspace.ensure_batch(batch)
        self.category_stack.ensure_batch(batch)
        self.reconstruction_stack.ensure_batch(batch)
        # Keep the new buffers on the existing device so .to(device)
        # invariants survive the resize.
        device = self._last_svo.device
        self._last_svo = torch.zeros(batch, 3, self.svo_dim, device=device)
        self._svo_valid = torch.zeros(batch, dtype=torch.bool, device=device)
        self._stm_fired = torch.zeros(batch, dtype=torch.bool, device=device)

    def ensure_microbatch(self, B, K):
        """Resize per-row state for the microbatch AR pipeline.

        Body-side state (subspace, stacks, last_svo, svo_valid) is sized
        to B*K — each window has its own row inside the body's flattened
        view. _stm_fired stays at B because STM firing is a per-source-row
        once-per-sentence event shared across all K windows of that row.
        Discourse buffers (InterSentenceLayer) also stay at B: discourse
        history accumulates across sentences within one source stream,
        and all K windows of a stream share that history (the post-body
        snapshot collapses K to mirror legacy last-cursor semantics).
        """
        BK = int(B) * int(K)
        self.ensure_batch(BK)
        device = self._stm_fired.device
        if self._stm_fired.shape[0] != int(B):
            self._stm_fired = torch.zeros(int(B), dtype=torch.bool, device=device)
        if self.discourse is not None and hasattr(self.discourse, 'ensure_batch'):
            self.discourse.ensure_batch(int(B))
        # _sentence_completed: per-source-row host bool, drained by the
        # outer doc-streaming loop after each runBatch. Resized in step
        # with the source-row count B so soft-reset signaling tracks the
        # current microbatch shape.
        if (not hasattr(self, '_sentence_completed')
                or self._sentence_completed is None
                or len(self._sentence_completed) != int(B)):
            self._sentence_completed = [False] * int(B)
