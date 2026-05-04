

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
from Layers import CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon, Ops
from Layers import SortingLayer, TruthLayer, LiftingLayer, InterSentenceLayer, SparsityRegLayer, SmoothingRegLayer, ImpenetrableLayer, ContiguousLayer
from util import parse
from collections import namedtuple as _namedtuple


from Layers import Layer, PiLayer, SigmaLayer # Import custom layers from Model.py
from Layers import LinearLayer, AttentionLayer
from Layers import CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon
from Layers import Error, TheError

from Spaces import ActiveEncoding, WhereEncoding, WhenEncoding, WhatEncoding, EventEncoding, WordEncoding
from Spaces import Basis, Tensor, Codebook, Embedding
from Spaces import SubSpace, WordSubSpace, Space, InputSpace, PerceptualSpace, ModalSpace, ConceptualSpace, SymbolicSpace, OutputSpace


def grammar_uses(rule_name):
    """Return True iff any rule body in the configured grammar invokes
    ``rule_name`` as a function call.

    Note: previously consumed by ConceptualSpace's grammar-driven wiring
    inference (DNF auto-wrap), which has been removed. NegationLayer is
    no longer auto-wired — wire composite C-tier wrappers explicitly via
    the ``layer`` kwarg on ConceptualSpace. This helper remains available
    for runtime grammar inspection.

    Reads the parsed XML grammar at WordSpace.language.grammar and the
    optional grammarCfg path; scans rule bodies (string leaves) for the
    substring ``rule_name(``. Returns False on any read error or when
    no grammar is configured.
    """
    needle = f"{rule_name}("
    try:
        cfg = TheXMLConfig.get("WordSpace.language.grammar")
    except (KeyError, AttributeError):
        cfg = None

    def _scan(node):
        if isinstance(node, str):
            return needle in node
        if isinstance(node, dict):
            return any(_scan(v) for v in node.values())
        if isinstance(node, (list, tuple)):
            return any(_scan(v) for v in node)
        return False

    if cfg is not None and _scan(cfg):
        return True

    try:
        cfg_path = TheXMLConfig.get("WordSpace.language.grammarCfg")
    except (KeyError, AttributeError):
        cfg_path = None
    if cfg_path:
        resolved = (cfg_path if os.path.isabs(cfg_path)
                    else os.path.join(ProjectPaths.PROJECT_DIR, cfg_path))
        try:
            with open(resolved, "r") as fh:
                if needle in fh.read():
                    return True
        except (FileNotFoundError, OSError):
            pass
    return False


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

    # Maps the new tier-bucket section names to the RuleDef.tier
    # field. Each space tier (PerceptualSpace, ConceptualSpace,
    # SymbolicSpace) reads its own subset by tier when filtering for
    # which rules are licensed in its forward path.
    _TIER_SECTIONS = {
        'percepts': 'P',
        'concepts': 'C',
        'symbols':  'S',
    }

    def configure(self, grammar_dict):
        """Configure rules from an XML-derived dict.

        Accepts these shapes:
          (a) flat: {'S': ['not(S)'], ...}  — legacy compose-only.
          (b) named sections: {'compose': {...}, 'generate': {...}}
              with `op.forward(args)` / `op.reverse(arg)` rule bodies.
          (c) tier-scoped sections: {'compose': {'symbols': {...},
                                                 'concepts': {...},
                                                 'percepts': {...}},
                                     'generate': {...same shape...}}
              Each tier's rules carry tier='S' / 'C' / 'P' on the
              RuleDef, so each space can filter to the rules licensed
              for it. A space "can conduct any/all of the operations"
              -- runtime gating is independent of tier tagging; the
              tags are an inductive-bias hint, not a hard restriction.
        """
        self.rules_upward = []
        self.rules_downward = []
        self._configured = True

        has_named = any(k in grammar_dict
                        for k in ('compose', 'generate'))
        if has_named:
            up = grammar_dict.get('compose') or {}
            dn = grammar_dict.get('generate') or {}
            self._fill_section(self.rules_upward, up)
            self._fill_section(self.rules_downward, dn)
        else:
            # Legacy flat form — treat as parse.
            self._fill_section(self.rules_upward, grammar_dict)

        # Canonical union so callers reading `g.rules` see upward first,
        # then downward. Upward rule IDs stay stable for existing code.
        self.rules = list(self.rules_upward) + list(self.rules_downward)
        self.rule_table = {idx: rule.canonical
                           for idx, rule in enumerate(self.rules)}
        # Step 6 parity: derive Layer-2.5 reverse rules from upward
        # productions even on the legacy XML path so consumers of
        # ``self.reverse_rules`` work uniformly across load paths.
        self.reverse_rules = self._derive_reverse_rules(self.rules_upward)
        self._bump_rule_table_version()

    def _fill_section(self, target, section_dict):
        """Read a parse / generate section, dispatching to per-tier
        rule lists when `<symbols>` / `<concepts>` / `<percepts>`
        sub-sections are present, or to the cross-tier reader otherwise.

        Tier-bucket detection is non-destructive: a section with both a
        `<rule>` directly and tier sub-sections will read both, with
        the direct rules tagged tier='S' (the default).
        """
        if not isinstance(section_dict, dict):
            return
        # Tier sub-sections.
        for tier_key, tier_letter in self._TIER_SECTIONS.items():
            tier_block = section_dict.get(tier_key)
            if tier_block:
                self._fill_rule_list(target, tier_block, tier=tier_letter)
        # Direct rules (no tier wrapper) -> default tier 'S'.
        direct_keys = [k for k in section_dict.keys()
                       if k not in self._TIER_SECTIONS]
        if direct_keys:
            direct = {k: section_dict[k] for k in direct_keys}
            self._fill_rule_list(target, direct, tier='S')

    def _fill_rule_list(self, target, rules_dict, tier='S'):
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
                target.append(self._parse_rule(lhs, body.strip(), tier=tier))

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
                target.append(self._parse_rule(lhs, rhs, tier=tier))

    def rule_by_id(self, rule_id):
        """Return the canonical production string for a rule_id (0-based)."""
        return self.rule_table[rule_id]

    def _parse_rule(self, lhs, rhs, tier='S'):
        # `tier` may be 'S' (symbols, default), 'C' (concepts), or
        # 'P' (percepts). Set by `_fill_section` from <symbols> /
        # <concepts> / <percepts> sub-sections under <parse> /
        # <generate>. Used by space-tier filters at runtime to gate
        # which rules apply in each space's forward path.
        if '(' in rhs:
            func_name = rhs[:rhs.index('(')]
            args_str = rhs[rhs.index('(') + 1:rhs.rindex(')')]
            args = [a.strip() for a in args_str.split(',') if a.strip()]
            arity = len(args)
            # Accept the explicit-direction forms `op.forward(args)` and
            # `op.reverse(arg)`. Strip the `.forward` / `.reverse`
            # suffix to recover the bare op-name; direction is implicit
            # from which section (parse / generate) the rule sits in.
            #
            # Sanity-check: `.forward` is expected in <parse>, `.reverse`
            # in <generate>. The parser doesn't know the section here
            # (it's per-rule), so we silently accept either suffix.
            if func_name.endswith('.forward'):
                func_name = func_name[:-len('.forward')]
            elif func_name.endswith('.reverse'):
                func_name = func_name[:-len('.reverse')]
            # Note: `pi` / `sigma` and other layer-name forms remain
            # as-authored in `method_name`. They resolve to the
            # semantic op name (`intersection` / `union`) at dispatch
            # time via SyntacticLayer's _METHOD_ALIASES lookup; that
            # keeps `RuleDef.method_name` faithful to what the XML
            # said while letting `_RULE_METHODS` stay keyed on the
            # semantic names.
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

    # Canonical section names: compose / generate, aligning with
    # `SyntacticLayer.compose` / `generate` method names. The
    # ``parse`` / ``upward`` / ``downward`` aliases were removed
    # 2026-05-01.
    _CFG_SECTION_UPWARD   = 'compose'
    _CFG_SECTION_DOWNWARD = 'generate'
    _CFG_SECTION_ALIASES  = {}

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
        ``parse``; rules appearing before any header land there.
        Legacy section names ``upward`` / ``downward`` are accepted
        as aliases for ``parse`` / ``generate``.
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
                # Translate legacy section aliases.
                name = self._CFG_SECTION_ALIASES.get(name, name)
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
        self._bump_rule_table_version()

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

    # ---- Rule probability gating (Pattern A, body-only) ------------
    #
    # In useGrammar="all", each operator rule has a learned firing
    # probability that gates the corresponding bottom-up layer
    # (intersection -> Pi, union -> Sigma, not -> propositional NEG).
    # When useGrammar != "all" (or the SyntacticLayer is dormant), the
    # probability is the Python float 1.0 — call sites use ``p is 1.0``
    # as a structural fast path that compiles down to a direct layer
    # call with no spurious mul-by-1 graph nodes.
    #
    # ``_fired_bodies`` enforces single-application per derivation:
    # once a rule body has fired, ``rule_probability`` returns 0 for
    # subsequent calls on the same body until ``reset_derivation`` is
    # invoked. This prevents pathological multi-NOT or multi-OR stacks
    # without splitting S/C into typed tiers.

    def rule_probability(self, body):
        """Probability that rule with given ``body`` fires at the
        current parse step. Returns a Python float in dormant mode so
        call sites can skip the gate via ``p is 1.0`` or ``p is 0.0``.

        ``body`` is the rule's RHS string as it appears in the XML
        (e.g. ``"intersection(C, C)"`` or ``"not(S)"``). Bodies are
        globally unique across the grammar so we don't need the LHS.

        Dormant defaults preserve existing pipeline behavior:
          - fold operators (intersection, union)  -> 1.0 (always fire)
          - negation-like ops (not, non)          -> 0.0 (don't fire)
          - Contiguous(S)                         -> 1.0 if
            ``thought_free`` else 0.0 (gated by Shamatha mode)

        These defaults match what the bottom-up Pi/Sigma layers do
        today (always run) and what the previously-absent NEG layer
        did (nothing). In ``useGrammar="all"`` mode, learned predictors
        in ``_learned_rule_probs`` override the dormant defaults.
        """
        fired = getattr(self, "_fired_bodies", None)
        if fired is not None and body in fired:
            return 0.0
        learned = getattr(self, "_learned_rule_probs", None)
        if learned is not None and body in learned:
            return learned[body]
        if body.startswith("Contiguous("):
            return 1.0 if self.thought_free else 0.0
        if body.startswith("not(") or body.startswith("non("):
            return 0.0
        return 1.0

    def note_rule_fired(self, body):
        """Mark a rule body as having fired in the current derivation."""
        fired = getattr(self, "_fired_bodies", None)
        if fired is None:
            self._fired_bodies = set()
            fired = self._fired_bodies
        fired.add(body)

    def reset_derivation(self):
        """Clear the per-derivation single-application bookkeeping."""
        self._fired_bodies = set()

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

    # ---- Soft-superposition chart: packed-rule-table machinery -----
    #
    # `softChartCompose=true` on WordSpace activates a CKY-style inside
    # pass in `SyntacticLayer._compose_chart_cky`. That path needs the
    # rule catalog as flat tensors rather than a Python list of RuleDef
    # so per-(span, rule) candidates can be enumerated as one bmm-shape
    # op. `build_rule_table_packed` rebuilds those tensors from
    # `self.rules`, applying the marker-compilation step described in
    # the floating-blossom spec: sugar rules (e.g. `absorb`) are NOT
    # given their own row -- they are folded into per-operand
    # `marker_mask` flags on the productive rules they license.
    #
    # `_rule_table_version` is bumped whenever the catalog changes
    # (configure / load_from_cfg / future add/remove). Consumers (e.g.
    # SyntacticLayer.rule_embed / rule_bias / marker_bias parameters)
    # cache by version and rebuild on mismatch.

    SUGAR_METHODS = frozenset({'absorb'})

    def _ensure_packed_table(self, device=None):
        """Return the packed rule-table dict, building it lazily.

        Keys (all torch tensors, shape leading dim R = number of
        productive rows; sugar rows are dropped):
            'lhs'         Long[R]      LHS category id (0 if missing)
            'rhs_left'    Long[R]      first RHS category id (0 = wildcard)
            'rhs_right'   Long[R]      second RHS category id (0 if unary)
            'arity'       Long[R]      1 or 2
            'marker_mask' Bool[R, 2]   per-operand: True iff sugar may absorb here
            'global_id'   Long[R]      back-reference to self.rules index
        """
        self._ensure_configured()
        cached = getattr(self, '_rule_table_packed_cache', None)
        if cached is not None and cached.get('device') == device:
            return cached['table']
        # Build a category index mirroring SyntacticLayer._ensure_category_table.
        names = set()
        for rule in self.rules:
            for cat in str(rule.lhs).split(','):
                cat = cat.strip()
                if cat:
                    names.add(cat)
            for sym in (rule.rhs_symbols or ()):
                if sym:
                    names.add(sym)
        ordered = ['?'] + sorted(n for n in names if n)
        cat_index = {n: i for i, n in enumerate(ordered)}

        # Step 1: identify sugar rules and the LHS categories they target.
        # A sugar rule's lhs is the host category whose operand may be
        # absorbed -- e.g. `S = absorb(S, S)` says "an S operand may be
        # marked sugar inside a productive rule that produces S."
        sugar_lhs = set()
        for rule in self.rules:
            if rule.method_name in self.SUGAR_METHODS:
                sugar_lhs.add(rule.lhs.split(',', 1)[0].strip()
                              if ',' in rule.lhs else rule.lhs)

        # Step 2: build the packed table from productive rules only.
        lhs_l, rl_l, rr_l, ar_l, mm_l, gid_l = [], [], [], [], [], []
        for gid, rule in enumerate(self.rules):
            if rule.method_name in self.SUGAR_METHODS:
                continue  # marker-compiled away, not a freestanding row
            if rule.arity not in (1, 2):
                continue  # arity-0 (epsilon) and arity-3 not in chart
            lhs_cat = (rule.lhs.split(',', 1)[0].strip()
                       if ',' in rule.lhs else rule.lhs)
            rhs = rule.rhs_symbols or ()
            rl = cat_index.get(rhs[0], 0) if len(rhs) >= 1 else 0
            rr = cat_index.get(rhs[1], 0) if len(rhs) >= 2 else 0
            # Marker compilation: an operand of category X can be absorbed
            # if a sugar rule's lhs matches X (the sugar rule licenses
            # X-shaped sugar at this slot). Unary rules: only slot 0.
            mm_left = bool(rhs) and rhs[0] in sugar_lhs and rule.arity == 2
            mm_right = (len(rhs) >= 2 and rhs[1] in sugar_lhs
                        and rule.arity == 2)
            lhs_l.append(cat_index.get(lhs_cat, 0))
            rl_l.append(rl)
            rr_l.append(rr)
            ar_l.append(rule.arity)
            mm_l.append([mm_left, mm_right])
            gid_l.append(gid)

        device = device or torch.device('cpu')
        if lhs_l:
            table = {
                'lhs':         torch.tensor(lhs_l, dtype=torch.long, device=device),
                'rhs_left':    torch.tensor(rl_l,  dtype=torch.long, device=device),
                'rhs_right':   torch.tensor(rr_l,  dtype=torch.long, device=device),
                'arity':       torch.tensor(ar_l,  dtype=torch.long, device=device),
                'marker_mask': torch.tensor(mm_l,  dtype=torch.bool, device=device),
                'global_id':   torch.tensor(gid_l, dtype=torch.long, device=device),
            }
        else:
            table = {
                'lhs':         torch.empty(0, dtype=torch.long, device=device),
                'rhs_left':    torch.empty(0, dtype=torch.long, device=device),
                'rhs_right':   torch.empty(0, dtype=torch.long, device=device),
                'arity':       torch.empty(0, dtype=torch.long, device=device),
                'marker_mask': torch.empty((0, 2), dtype=torch.bool, device=device),
                'global_id':   torch.empty(0, dtype=torch.long, device=device),
            }
        # Stash the category index alongside so SyntacticLayer can read
        # the same view rather than rebuilding it.
        table['_cat_index'] = cat_index
        table['_cat_names'] = ordered
        self._rule_table_packed_cache = {'device': device, 'table': table}
        return table

    @property
    def rule_table_version(self):
        """Counter bumped on every catalog change. Chart layers read this
        to decide whether to rebuild rule_embed / rule_bias rows."""
        return getattr(self, '_rule_table_version_counter', 0)

    def _bump_rule_table_version(self):
        self._rule_table_version_counter = self.rule_table_version + 1
        # Drop the packed cache so the next read rebuilds.
        self._rule_table_packed_cache = None

TheGrammar = Grammar()


# =====================================================================
# Chart -- soft-superposition CKY parser owned by WordSpace.
#
# Spec: doc/specs/2026-05-01-syntactic-layer-refactor.md
#
# Owns chart-specific params, runs inside / outside passes, dispatches
# per-cell rule applications through ``word_space.host_layer(tier,
# rule_name)`` (when a registry is wired) so the chart's grammar choice
# directly fires the host space's parametrized fold.
#
# Per Q10.3: the Chart reads from ``TheGrammar`` (module-level singleton);
# there is no per-call grammar parameter and no stored grammar reference.
#
# Per Q10.5: ``compose`` / ``generate`` branch on ``self.training``.
# Training -> soft inside / outside passes (logsumexp / softmax mixing,
# gradient flows broadly). Eval -> Viterbi argmax IN PLACE OF soft
# mixing (one committed derivation per row).
# =====================================================================
class Chart(nn.Module):
    """Soft-superposition CKY chart parser. Owned by WordSpace.

    Holds chart parameters, runs inside / outside passes, dispatches
    per-cell rule applications through ``word_space.host_layer(tier,
    rule_name)`` to fire the host space's parametrized folds directly.
    """

    # Norm-drop threshold for the legacy 2D-activation accumulation
    # path. Kept here for symmetry with the old SyntacticLayer field;
    # the chart inside/outside passes don't rely on it.
    _QUERY_NORM_DROP_RATIO = 0.1

    def __init__(self, nInput, nOutput=None, *, max_depth=12,
                 hidden_dim=256, D_rule=32, chart_tau=None, w_max=8,
                 unary_max_depth=2, feature_dim=None,
                 router_kind=None):
        super().__init__()
        nOutput = nOutput if nOutput is not None else nInput
        self.nInput = int(nInput)
        self.nOutput = int(nOutput)
        self.max_depth = int(max_depth)
        self.hidden_dim = int(hidden_dim)
        self.D_rule = int(D_rule)
        self.w_max = int(w_max)
        self.unary_max_depth = int(unary_max_depth)
        # XML override for chart_tau when caller didn't pin one. Read at
        # construction time (Q10.3: Chart configures itself from XML once).
        if chart_tau is None:
            try:
                chart_tau = float(TheXMLConfig.get(
                    "WordSpace.chartTau", 1.0))
            except Exception:
                chart_tau = 1.0
        self.chart_tau = float(chart_tau)

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

        # Rule prediction network (weight-tied across depths). Mirrors
        # the legacy SyntacticLayer.forward; renamed predict_rules below.
        self.input_proj = LinearLayer(self.nInput, self.hidden_dim)
        self.derivation_layer = LinearLayer(
            self.hidden_dim, self.hidden_dim)
        # rule_head is sized to the max grammar rule count; dispatch
        # filters by tier through the per-space SyntacticLayer.
        # During Step 1 the head is sized to the configured grammar's
        # rule count; we rebuild lazily when grammar versions change.
        n_rules = max(1, len(getattr(TheGrammar, 'rule_table', [])) or 1)
        self.num_rules = n_rules
        self.rule_head = LinearLayer(self.hidden_dim, self.num_rules)
        self.depth_embed = nn.Embedding(self.max_depth, self.hidden_dim)
        self.activation_fn = nn.GELU()

        # Pair-scorer MLP. Same contract as the legacy version.
        fd = self.nInput if feature_dim is None else int(feature_dim)
        self._pair_feature_dim = fd
        self.pair_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim + 2 * fd, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

        # Xavier init on linear weights.
        for layer in (self.input_proj, self.derivation_layer,
                      self.rule_head):
            nn.init.xavier_normal_(layer.W)
        nn.init.normal_(self.depth_embed.weight, std=0.02)

        # Soft-chart machinery (rule-shaped Parameters built lazily).
        self._soft_chart_built = False
        self._soft_chart_version = -1
        self._compat_score_mod = None
        self._unary_compat_mod = None
        self._lex_cat_scorer = None

        # Per-call chart state. Reset each compose; pre-init for read
        # safety on layers whose compose hasn't fired yet.
        self._chart_score = None
        self._chart_vec = None
        # POS side-channel: per-cell probability simplex over the
        # category axis, populated by _chart_inside. See doc/Language.md
        # "POS side-channel" for the contract.
        self._chart_pos = None
        self._outside_score = None
        self._outside_vec = None
        self._derivation_trace = None
        self.last_svo = None
        self.last_rule_probs = None
        self.last_composable_rules = None

        # Category machinery.
        self._category_names = None
        self._category_index = None
        self._last_category = None

        # Rule executor. When None, the Chart falls back to the
        # typed-GrammarLayer facade registry for per-rule math (so the
        # Chart works in isolation). Step 7 wires WordSpace's
        # host_layer dispatch in here.
        self._rule_executor = None
        # Active wordSpace handle for the current compose / generate
        # call. Set by `compose` / `generate` for the duration of the
        # call so `_apply_rule_forward` can look up host layers via
        # the registry.
        object.__setattr__(self, '_active_word_space', None)

    def _ensure_signal_router(self):
        """Lazy-build the SignalRouter when router_kind == 'signal'.

        Assigning an nn.Module to an attribute auto-registers it as a
        submodule, so it is included in parameters() / state_dict().
        """
        if self._signal_router is None:
            from SignalRouter import SignalRouter as _SignalRouter
            try:
                temperature = float(TheXMLConfig.get(
                    "WordSpace.signal.temperature", 1.0))
            except Exception:
                temperature = 1.0
            self._signal_router = _SignalRouter(
                n_input=self.nInput,
                n_output=self.nOutput,
                hidden_dim=self.hidden_dim,
                feature_dim=self._pair_feature_dim,
                max_depth=self.max_depth,
                temperature=temperature,
            )
        return self._signal_router

    # ------------------------------------------------------------------
    # Public API (Q10.3 / Q10.5 surface).
    # ------------------------------------------------------------------
    def predict_rules(self, x):
        """Predict rule distributions and build word tuples.

        Args:
            x: ``[B, N]`` activation vector (per-position scalar norms).

        Returns dict matching the legacy ``SyntacticLayer.forward``:
            rule_logits:     ``[B, max_depth, num_rules]``
            rule_probs:      ``[B, max_depth, num_rules]``
            predicted_rules: ``[B, max_depth]`` (global rule IDs)
        """
        B, N = x.shape

        if self.num_rules == 0:
            empty_logits = torch.zeros(B, self.max_depth, 0,
                                       device=x.device, dtype=x.dtype)
            empty_predicted = torch.zeros(B, self.max_depth,
                                          device=x.device,
                                          dtype=torch.long)
            return {
                "rule_logits":     empty_logits,
                "rule_probs":      empty_logits,
                "predicted_rules": empty_predicted,
            }

        # Pad / truncate x to the rule predictor's input width so callers
        # at varying N can reuse one Chart.
        expected_n = self.input_proj.nInput
        if N < expected_n:
            pad = torch.zeros(B, expected_n - N, device=x.device,
                              dtype=x.dtype)
            x_in = torch.cat([x, pad], dim=-1)
        elif N > expected_n:
            x_in = x[..., :expected_n]
        else:
            x_in = x

        h = self.input_proj.forward(x_in)
        h = self.activation_fn(h)
        depth_ids = torch.arange(self.max_depth, device=x.device)
        depth_vecs = self.depth_embed(depth_ids)

        all_logits = []
        all_probs = []
        for d in range(self.max_depth):
            h = h + depth_vecs[d]
            h = self.derivation_layer.forward(h)
            h = self.activation_fn(h)
            logits = self.rule_head.forward(h)
            if self.training:
                probs = F.gumbel_softmax(logits, tau=1.0, hard=False)
            else:
                probs = F.softmax(logits, dim=-1)
            all_logits.append(logits)
            all_probs.append(probs)

        rule_logits = torch.stack(all_logits, dim=1)
        rule_probs = torch.stack(all_probs, dim=1)
        predicted = rule_logits.argmax(dim=-1)
        return {
            "rule_logits":     rule_logits,
            "rule_probs":      rule_probs,
            "predicted_rules": predicted,
        }

    def compose(self, data, word_space, subspace=None):
        """Run the inside pass over ``data``; populate per-tier rule
        selections on ``word_space.current_rules``.

        Args:
            data: ``[B, N, D]`` token vectors to parse.
            word_space: WordSpace instance owning this chart. Receives
                the rule selections and reads ``host_layer(tier, rule)``
                during dispatch.
            subspace: optional subspace handle threaded into per-rule
                math (legacy compatibility).

        Returns:
            dict[str, list[int]]: per-tier rule-id lists. Side effects:
            populates ``word_space.current_rules`` and refreshes
            ``self._chart_score`` / ``self._chart_vec``.
        """
        # Stash the wordSpace ref so per-rule dispatch
        # (`_apply_rule_forward`) can look up host layers via
        # `word_space.host_layer(tier, rule_name)`. Cleared at end.
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

    def generate(self, target, word_space, subspace=None):
        """Run the outside pass + Viterbi backtrace over ``target``;
        populate ``word_space.generate_rules``.

        Args:
            target: ``[B, N, D]`` parent vectors to invert.
            word_space: WordSpace instance.
            subspace: optional subspace handle.

        Returns:
            dict[str, list[int]]: per-tier generate-rule-id lists.
        """
        object.__setattr__(self, '_active_word_space', word_space)
        try:
            if self.router_kind == "signal":
                router = self._ensure_signal_router()
                rules = router.generate(target, word_space, subspace=subspace)
                if word_space is not None:
                    word_space.generate_rules = rules
                return rules
            # The outside pass needs an inside pass result; in the
            # legacy path the inside pass runs before generate as part
            # of compose. Generate-only callers (testing reverse paths)
            # trigger an inside pass here too.
            if self._chart_score is None or self._chart_vec is None:
                _ = self.compose(target, word_space, subspace)
            rules = self._collect_generate_selections(word_space)
            word_space.generate_rules = rules
            return rules
        finally:
            object.__setattr__(self, '_active_word_space', None)

    # ------------------------------------------------------------------
    # Rule executor wiring (Step 7 wires this from WordSpace).
    # ------------------------------------------------------------------
    def attach_rule_executor(self, executor):
        """Install the per-rule math callable.

        Signature:
            executor(method_name, left, right, marker_mask, subspace)
                -> tensor

        When set, ``_apply_rule_forward`` dispatches through this
        callable instead of the typed-GrammarLayer facade fallback.
        """
        self._rule_executor = executor

    # ------------------------------------------------------------------
    # Rule-selection emission. The chart fills per-tier lists by walking
    # its derivation trace; the per-space SyntacticLayer then pops one
    # rule per forward step. A rule fires at most once per forward
    # (Q5).
    # ------------------------------------------------------------------
    def _collect_rule_selections(self, word_space):
        """Collect per-tier per-row rule selections from the chart's
        derivation trace. Returns
        ``dict[tier_name -> list[list[rule_id]]]`` -- the outer list
        indexes batch row, the inner list indexes step.

        Suitable for ``word_space.current_rules``. Per-space syntactic
        layers read row 0 by default; per-row dispatch (firing
        different rules on different rows) is a follow-on.

        Tier resolution: each grammar rule's authored ``tier`` is
        consulted when available; otherwise the rule's ``lhs`` category
        first letter is used as a fallback hint.
        """
        per_tier = {}
        traces = self._derivation_trace
        if traces is None or not traces:
            return per_tier
        n_rows = len(traces)
        for row_idx, trace in enumerate(traces):
            trace = trace or []
            for entry in trace:
                rule_id = int(entry[0])
                tier = self._tier_for_rule(rule_id)
                rows = per_tier.setdefault(tier, [])
                while len(rows) < n_rows:
                    rows.append([])
                rows[row_idx].append(rule_id)
        # Pad short rows with empty lists so all tiers expose a uniform
        # n_rows shape (downstream cursor logic indexes by row).
        for tier, rows in per_tier.items():
            while len(rows) < n_rows:
                rows.append([])
        return per_tier

    def _collect_generate_selections(self, word_space):
        """Collect per-tier per-row generate-rule selections. Mirror of
        ``_collect_rule_selections``; reverses each row's trace so
        downward generation pops the last-applied rule first.
        """
        per_tier = {}
        traces = self._derivation_trace
        if traces is None or not traces:
            return per_tier
        n_rows = len(traces)
        for row_idx, trace in enumerate(traces):
            trace = list(trace or [])
            for entry in reversed(trace):
                rule_id = int(entry[0])
                tier = self._tier_for_rule(rule_id)
                rows = per_tier.setdefault(tier, [])
                while len(rows) < n_rows:
                    rows.append([])
                rows[row_idx].append(rule_id)
        for tier, rows in per_tier.items():
            while len(rows) < n_rows:
                rows.append([])
        return per_tier

    def _tier_for_rule(self, rule_id):
        """Map a global rule_id to its host-space tier ('P', 'C', 'S').
        Falls back to 'C' when the grammar's tier metadata is missing.
        """
        try:
            rule = TheGrammar.rules[rule_id]
        except (IndexError, AttributeError):
            return 'C'
        # Grammar.rule has a `tier` attr in some configs; otherwise
        # derive from the rule's lhs category name.
        tier = getattr(rule, 'tier', None)
        if tier in ('P', 'C', 'S'):
            return tier
        lhs = str(getattr(rule, 'lhs', '') or '')
        first = lhs.split(',', 1)[0].strip()
        if first.startswith('P'):
            return 'P'
        if first.startswith('C'):
            return 'C'
        return 'S'

    # ------------------------------------------------------------------
    # Chart internals: lazy build / category table.
    # ------------------------------------------------------------------
    def _category_names_count(self):
        table = TheGrammar._ensure_packed_table()
        return len(table['_cat_names'])

    def _ensure_category_table(self):
        if self._category_names is not None:
            return
        names = set()
        for rule in TheGrammar.rules:
            for cat in str(rule.lhs).split(','):
                cat = cat.strip()
                if cat:
                    names.add(cat)
            for sym in (rule.rhs_symbols or ()):
                names.add(sym)
        ordered = ['?'] + sorted(n for n in names if n)
        self._category_names = ordered
        self._category_index = {n: i for i, n in enumerate(ordered)}

    def _ensure_soft_chart_built(self, D, device):
        version = getattr(TheGrammar, 'rule_table_version', 0)
        if (self._soft_chart_built
                and self._soft_chart_version == version
                and self._compat_score_mod is not None
                and self._compat_score_mod.lin1.in_features
                    == 2 * D + self.D_rule):
            return
        # (Re)build the shared modules if D / D_rule changed or first call.
        rebuild_compat = (
            self._compat_score_mod is None
            or self._compat_score_mod.lin1.in_features
                != 2 * D + self.D_rule)
        if rebuild_compat:
            self._compat_score_mod = _CompatScore(
                D, self.D_rule).to(device)
            self._unary_compat_mod = _UnaryCompat(
                D, self.D_rule).to(device)
        cats = self._category_names_count()
        if (self._lex_cat_scorer is None
                or self._lex_cat_scorer.in_features != D
                or self._lex_cat_scorer.out_features != cats):
            self._lex_cat_scorer = nn.Linear(D, cats).to(device)
            nn.init.xavier_normal_(self._lex_cat_scorer.weight)
            nn.init.zeros_(self._lex_cat_scorer.bias)
        # Rebuild rule-table-shaped parameters.
        table = TheGrammar._ensure_packed_table(device=device)
        R = int(table['lhs'].shape[0])
        if R > 0:
            embed = nn.Parameter(torch.randn(
                R, self.D_rule, device=device) * 0.02)
        else:
            embed = nn.Parameter(torch.zeros(
                0, self.D_rule, device=device))
        # register_parameter will overwrite an existing same-named param.
        self.register_parameter('_rule_embed', embed)
        self.register_parameter(
            '_rule_bias', nn.Parameter(
                torch.zeros(R, device=device)))
        self.register_parameter(
            '_marker_bias', nn.Parameter(
                torch.zeros(R, 2, device=device)))
        self._soft_chart_built = True
        self._soft_chart_version = version

    # ------------------------------------------------------------------
    # Per-rule math dispatch.
    # ------------------------------------------------------------------
    def _apply_rule_forward(self, method_name, left, right, marker_mask,
                            subspace=None):
        """Dispatch rule's forward semantics. Marker operands are zeroed
        before the rule fires so a sugar operand contributes nothing.

        Dispatch order (Step 7 of the 2026-05-01 refactor):
          1. host_layer dispatch via the active wordSpace's registry —
             fires the host space's parametrized GrammarLayer (e.g.
             PiLayer-backed IntersectionLayer). The rule's authored
             tier (RuleDef.tier) selects the registry shard.
          2. external rule_executor (test injection point, optional).
          3. typed-GrammarLayer facade lookup (parameter-free, used
             when no host layer exists for this rule — lift, lower,
             swap, etc., on tiers that don't own a parametrized fold).
        """
        keep = (~marker_mask).to(left.dtype)
        kL = keep[..., 0:1]
        kR = keep[..., 1:2]
        l_eff = left * kL
        r_eff = right * kR
        # 1. host_layer dispatch.
        ws = self._active_word_space
        if ws is not None:
            tier = self._tier_for_method(method_name)
            host = ws.host_layer(tier, method_name) if tier else None
            if host is not None:
                arity = getattr(host, 'arity', 1)
                if arity == 2 and hasattr(host, 'compose'):
                    return host.compose(l_eff, r_eff)
                return host.forward(l_eff)
        # 2. external rule_executor.
        if self._rule_executor is not None:
            return self._rule_executor(
                method_name, l_eff, r_eff, marker_mask, subspace)
        # 3. Direct GrammarLayer class lookup (Step 8: replaces the
        # retired `_GrammarOpFacade._registry`). Construct a fresh
        # parameter-free instance each call; the parametrized ops
        # (swap) only fire via host_layer dispatch with the right
        # constructor args.
        try:
            from Layers import GRAMMAR_LAYER_CLASSES
        except Exception:
            return l_eff
        cls = GRAMMAR_LAYER_CLASSES.get(method_name)
        if cls is None:
            return l_eff
        try:
            inst = cls()
        except TypeError:
            return l_eff
        try:
            arity = getattr(inst, 'arity', 1)
            if arity == 2 and hasattr(inst, 'compose'):
                return inst.compose(l_eff, r_eff)
            return inst.forward(l_eff)
        except Exception:
            return l_eff

    def _tier_for_method(self, method_name):
        """Return the host tier ('P' / 'C' / 'S') for ``method_name``
        by scanning ``TheGrammar.rules`` for a rule whose method_name
        matches. The first match wins; if a method_name appears at
        multiple tiers, callers should pick one canonical tier.
        """
        if not method_name:
            return None
        for rule in TheGrammar.rules:
            mn = getattr(rule, 'method_name', None)
            if mn == method_name:
                t = getattr(rule, 'tier', None)
                if t in ('P', 'C', 'S'):
                    return t
        return None

    # ------------------------------------------------------------------
    # POS side-channel helpers.
    # ------------------------------------------------------------------
    def _apply_codebook_pos_seed(self, data, lex_log_probs, word_space,
                                 subspace, B, N, C):
        """Override the learned lexical scorer with codebook-stored POS
        tags at positions where a tagged atom is the nearest codebook
        match.

        Returns a possibly-updated ``lex_log_probs: [B, N, C]`` where
        positions with a known atom category have a hard one-hot
        log-distribution and other positions retain
        ``log_softmax(_lex_cat_scorer(data))``.

        No-op (returns input unchanged) when the symbolic codebook
        carries no per-atom tags or the input shape doesn't permit
        atom resolution. Always preserves gradient flow on un-tagged
        positions through ``_lex_cat_scorer``.
        """
        if word_space is None:
            return lex_log_probs
        sym_subspace = None
        try:
            sym_space = getattr(word_space, 'symbolicSpace', None)
            if sym_space is not None:
                sym_subspace = getattr(sym_space, 'subspace', None)
        except Exception:
            sym_subspace = None
        # Subspace handed in directly is the active S-tier subspace.
        # Prefer it when available so the codebook path tracks the
        # subspace currently being parsed.
        candidate = subspace if subspace is not None else sym_subspace
        if candidate is None:
            return lex_log_probs
        what = getattr(candidate, 'what', None)
        if what is None:
            return lex_log_probs
        cat_ids = getattr(what, 'category_ids', None)
        if cat_ids is None:
            return lex_log_probs
        try:
            W = what.getW()
        except Exception:
            return lex_log_probs
        if W is None or not torch.is_tensor(W) or W.ndim < 2:
            return lex_log_probs
        # Resolve each [B, N, D] data row to its nearest codebook
        # row via dot product, then gather the per-atom category id.
        # When the dot product is too low (< 0.1), we treat the atom
        # as not a codebook match and skip the override for that
        # position. This preserves _lex_cat_scorer gradients on
        # positions whose data isn't a confident codebook lookup.
        device = data.device
        dtype = data.dtype
        try:
            W_ = W.to(device=device, dtype=dtype)
            cat_ids_ = cat_ids.to(device=device, dtype=torch.long)
        except Exception:
            return lex_log_probs
        D_data = int(data.shape[-1])
        D_cb = int(W_.shape[-1])
        D_min = min(D_data, D_cb)
        if D_min <= 0:
            return lex_log_probs
        flat = data[..., :D_min].reshape(-1, D_min)
        cb = W_[:, :D_min]
        sims = flat @ cb.T  # [B*N, V]
        best_sim, best_idx = sims.max(dim=-1)  # [B*N]
        best_cat = cat_ids_.gather(0, best_idx.clamp(min=0,
                                                    max=cat_ids_.shape[0] - 1))
        # Mask: position has a tagged atom (cat > 0 == not '?') AND
        # the dot product is non-trivial (atom is plausibly nearest).
        thresh = 0.1
        valid = (best_cat > 0) & (best_sim > thresh)
        if not bool(valid.any()):
            return lex_log_probs
        # Rebuild lex_log_probs with one-hot overrides at valid rows.
        flat_log = lex_log_probs.reshape(-1, C).clone()
        cat_clamped = best_cat.clamp(min=0, max=C - 1)
        NEG = torch.full_like(flat_log, -1e9)
        rows = torch.arange(flat_log.shape[0], device=device)
        # Build a one-hot log-distribution for the override rows.
        override = NEG.clone()
        override[rows, cat_clamped] = 0.0
        flat_log = torch.where(
            valid.unsqueeze(-1).expand_as(flat_log), override, flat_log)
        return flat_log.reshape(B, N, C)

    def _compute_chart_pos(self):
        """Convert the chart's log-score tensor to a probability simplex
        over the category axis.

        Returns ``chart_pos: [B, N+1, N+1, C]`` where each row sums to 1
        across the trailing axis. Returns None if the chart hasn't run
        yet.
        """
        if self._chart_score is None:
            return None
        # softmax over the C axis turns log-scores into a per-cell
        # POS distribution. Cells with all NEG_INF (unreached cells)
        # become uniform — harmless for downstream consumers since
        # those cells aren't used in extraction either way.
        return F.softmax(self._chart_score, dim=-1)

    # ------------------------------------------------------------------
    # Inside pass (training: soft; eval: Viterbi). Q10.5.
    # ------------------------------------------------------------------
    def _compose_chart_cky(self, data, word_space, subspace=None):
        """Soft-mode CKY inside pass. Logsumexp / softmax-weighted
        scatter at every cell. Used during training.
        """
        return self._chart_inside(data, word_space, subspace, hard=False)

    def _compose_chart_cky_viterbi(self, data, word_space, subspace=None):
        """Hard-mode CKY inside pass. Argmax scatter at every cell.
        Used at eval (Q10.5).
        """
        return self._chart_inside(data, word_space, subspace, hard=True)

    def _chart_inside(self, data, word_space, subspace, *, hard):
        """Unified inside pass; ``hard=True`` switches per-cell scatter
        from logsumexp/softmax mixing to argmax+gather.
        """
        B, N, D = data.shape
        device = data.device
        dtype = data.dtype

        gv = getattr(TheGrammar, 'rule_table_version', 0)
        if gv != self._soft_chart_version:
            self._category_names = None
            self._category_index = None
        self._ensure_category_table()
        self._ensure_soft_chart_built(D, device)

        table = TheGrammar._ensure_packed_table(device=device)
        R = int(table['lhs'].shape[0])
        C = self._category_names_count()

        empty_score = torch.full((B, N + 1, N + 1, C), -1e30,
                                 device=device, dtype=dtype)
        empty_vec = torch.zeros((B, N + 1, N + 1, C, D),
                                device=device, dtype=dtype)
        if R == 0 or N == 0:
            self._chart_score = empty_score
            self._chart_vec = empty_vec
            self._derivation_trace = [[] for _ in range(B)]
            return data, None

        arity = table['arity']
        bin_idx = torch.nonzero(arity == 2, as_tuple=False).squeeze(-1)
        un_idx = torch.nonzero(arity == 1, as_tuple=False).squeeze(-1)
        R_bin = int(bin_idx.numel())
        R_un = int(un_idx.numel())

        rhs_left = table['rhs_left']
        rhs_right = table['rhs_right']
        lhs = table['lhs']
        mmask = table['marker_mask'].to(device=device)

        chart_score = empty_score
        chart_vec = empty_vec

        # Lexical fill (w=1).
        # POS side-channel — Tier 3 (learned scorer) is the baseline.
        lex_logits = self._lex_cat_scorer(data)
        lex_log_probs = F.log_softmax(lex_logits, dim=-1)
        # POS side-channel — Tier 1 (codebook lookup): when an input
        # position resolves to a tagged codebook atom in the symbolic
        # space's `what` codebook, override the learned scorer with a
        # one-hot over the atom's stored category. Tier 2 (`pos_lookup`)
        # is the runtime fallback; here we keep the implementation
        # focused on the codebook path because it's the durable seed
        # source per the plan.
        lex_log_probs = self._apply_codebook_pos_seed(
            data, lex_log_probs, word_space, subspace, B, N, C)
        i_diag = torch.arange(N, device=device)
        chart_score = chart_score.clone()
        chart_vec = chart_vec.clone()
        chart_score[:, i_diag, i_diag + 1, :] = lex_log_probs
        chart_vec[:, i_diag, i_diag + 1, :, :] = (
            data.unsqueeze(2).expand(B, N, C, D))

        bin_global = (table['global_id'][bin_idx].cpu().tolist()
                      if R_bin > 0 else [])
        un_global = (table['global_id'][un_idx].cpu().tolist()
                     if R_un > 0 else [])
        bin_methods = [TheGrammar.rules[g].method_name for g in bin_global]
        un_methods = [TheGrammar.rules[g].method_name for g in un_global]
        if R_bin > 0:
            rE_bin = self._rule_embed[bin_idx]
            rB_bin = self._rule_bias[bin_idx]
            mB_bin = self._marker_bias[bin_idx]
            mmask_bin = mmask[bin_idx].to(dtype=dtype)
            mmask_bin_bool = mmask[bin_idx]
            rl_bin = rhs_left[bin_idx]
            rr_bin = rhs_right[bin_idx]
            lhs_bin = lhs[bin_idx]
        if R_un > 0:
            rE_un = self._rule_embed[un_idx]
            rB_un = self._rule_bias[un_idx]
            rl_un = rhs_left[un_idx]
            lhs_un = lhs[un_idx]

        NEG_INF = -1e30
        w_max = min(int(self.w_max), N)

        for w in range(2, w_max + 1):
            P = N - w + 1
            if P <= 0 or R_bin == 0:
                continue
            i_range = torch.arange(P, device=device)
            offsets = torch.arange(1, w, device=device)
            Sp = int(offsets.numel())

            i_idx = i_range.unsqueeze(1).expand(P, Sp)
            k_idx = i_idx + offsets.unsqueeze(0).expand(P, Sp)
            j_idx = i_idx + w
            i_flat = i_idx.reshape(-1)
            k_flat = k_idx.reshape(-1)
            j_flat = (i_flat + w)
            i2 = i_flat.unsqueeze(1).expand(-1, R_bin)
            k2 = k_flat.unsqueeze(1).expand(-1, R_bin)
            j2 = j_flat.unsqueeze(1).expand(-1, R_bin)
            bL = rl_bin.unsqueeze(0).expand(P * Sp, R_bin)
            bR = rr_bin.unsqueeze(0).expand(P * Sp, R_bin)

            left = chart_vec[:, i2, k2, bL, :]
            right = chart_vec[:, k2, j2, bR, :]
            score_left = chart_score[:, i2, k2, bL]
            score_right = chart_score[:, k2, j2, bR]

            rE = rE_bin.view(1, 1, R_bin, -1).expand(B, P * Sp, R_bin, -1)
            mm = mmask_bin_bool.view(1, 1, R_bin, 2).expand(
                B, P * Sp, R_bin, 2)

            merged_per_rule = []
            for r_local in range(R_bin):
                l_r = left[:, :, r_local, :]
                r_r = right[:, :, r_local, :]
                mm_r = mm[:, :, r_local, :]
                merged_per_rule.append(self._apply_rule_forward(
                    bin_methods[r_local], l_r, r_r, mm_r,
                    subspace=subspace))
            merged_vec = torch.stack(merged_per_rule, dim=2)
            compat = self._compat_score_mod(left, right, rE, mm)
            marker_prior = (mB_bin * mmask_bin).sum(-1)

            # POS side-channel — Mechanism 1 (RHS POS compatibility).
            # Gate (rule, pair) combinations by whether the operand
            # spans' POS distributions match the rule's typed RHS.
            # Wildcards (rhs_*[r] == 0) keep score 1.0; tagged
            # categories produce an exponential penalty when the
            # operand's POS distribution doesn't agree.
            #
            # `pos_left[b, p, c]` is the probability that the left
            # span has category c, derived from the chart's per-cell
            # log-score softmaxed over C.
            pos_chart = F.softmax(chart_score, dim=-1)
            pos_left_bp = pos_chart[:, i2, k2, :]  # [B, P*Sp, R_bin, C]
            pos_right_bp = pos_chart[:, k2, j2, :]  # [B, P*Sp, R_bin, C]
            # Gather along C: select the probability corresponding to
            # each rule's expected RHS category. Indices are the
            # rule-broadcast tensors `rl_bin` / `rr_bin`.
            rl_bp = rl_bin.view(1, 1, R_bin, 1).expand(B, P * Sp, R_bin, 1)
            rr_bp = rr_bin.view(1, 1, R_bin, 1).expand(B, P * Sp, R_bin, 1)
            p_l = pos_left_bp.gather(-1, rl_bp).squeeze(-1)
            p_r = pos_right_bp.gather(-1, rr_bp).squeeze(-1)
            # Wildcard handling: for rules with rhs_*[r] == 0 ('?'),
            # the gate is unconditionally 1.0. This preserves
            # backward-compat with grammars whose rules don't carry
            # typed RHS yet.
            wild_l = (rl_bin == 0).view(1, 1, R_bin).expand_as(p_l)
            wild_r = (rr_bin == 0).view(1, 1, R_bin).expand_as(p_r)
            p_l = torch.where(wild_l, torch.ones_like(p_l), p_l)
            p_r = torch.where(wild_r, torch.ones_like(p_r), p_r)
            rhs_compat = (p_l * p_r).clamp(min=1e-9)
            rhs_compat_log = torch.log(rhs_compat)

            cand_score = (score_left + score_right
                          + rB_bin.view(1, 1, R_bin)
                          + compat
                          + marker_prior.view(1, 1, R_bin)
                          + rhs_compat_log)
            cand_score = cand_score.view(B, P, Sp, R_bin)
            tau = max(float(self.chart_tau), 1e-3)
            if tau != 1.0:
                cand_score = cand_score / tau
            merged_vec = merged_vec.view(B, P, Sp, R_bin, D)

            rule_to_A = F.one_hot(lhs_bin, num_classes=C).to(dtype)
            log_mask = torch.where(rule_to_A > 0,
                                   torch.zeros_like(rule_to_A),
                                   torch.full_like(rule_to_A, NEG_INF))
            scored_per_A = (cand_score.unsqueeze(-1)
                            + log_mask.view(1, 1, 1, R_bin, C))

            if hard:
                # Viterbi: per (B, P, C) take the argmax over (Sp, R_bin)
                # candidates. Decompose the flat argmax into split / rule
                # axes, gather merged_vec via advanced indexing, mask
                # off categories that no rule reaches.
                flat = scored_per_A.reshape(B, P, Sp * R_bin, C)
                new_score_w, best_idx = flat.max(dim=2)
                sp_idx = best_idx // max(R_bin, 1)
                r_idx = best_idx % max(R_bin, 1)
                b_ax = torch.arange(B, device=device).view(B, 1, 1)
                p_ax = torch.arange(P, device=device).view(1, P, 1)
                new_vec_w = merged_vec[b_ax, p_ax, sp_idx, r_idx]
                finite = (new_score_w > NEG_INF / 2).unsqueeze(-1)
                new_vec_w = new_vec_w * finite.to(dtype)
            else:
                new_score_w = torch.logsumexp(
                    scored_per_A.reshape(B, P, Sp * R_bin, C), dim=2)
                weights = (scored_per_A
                           - new_score_w.view(B, P, 1, 1, C)).exp()
                new_vec_w = (weights.unsqueeze(-1)
                             * merged_vec.unsqueeze(-2)).sum(dim=(2, 3))

            old_score = chart_score[:, i_range, i_range + w, :]
            old_vec = chart_vec[:, i_range, i_range + w, :, :]
            if hard:
                # In Viterbi mode the cell holds whichever side has the
                # higher score; ties favor the new width-w finding.
                pick_new = (new_score_w >= old_score)
                combined_score = torch.where(
                    pick_new, new_score_w, old_score)
                combined_vec = torch.where(
                    pick_new.unsqueeze(-1), new_vec_w, old_vec)
            else:
                stacked = torch.stack([old_score, new_score_w], dim=0)
                combined_score = torch.logsumexp(stacked, dim=0)
                old_w_n = (old_score - combined_score).exp()
                new_w_n = (new_score_w - combined_score).exp()
                combined_vec = (old_w_n.unsqueeze(-1) * old_vec
                                + new_w_n.unsqueeze(-1) * new_vec_w)
            chart_score = chart_score.clone()
            chart_vec = chart_vec.clone()
            chart_score[:, i_range, i_range + w, :] = combined_score
            chart_vec[:, i_range, i_range + w, :, :] = combined_vec

            # Bounded unary closure.
            if R_un > 0:
                for _ in range(int(self.unary_max_depth)):
                    bL_un = rl_un.unsqueeze(0).expand(P, R_un)
                    iP = i_range.unsqueeze(1).expand(P, R_un)
                    jP = (i_range + w).unsqueeze(1).expand(P, R_un)
                    child = chart_vec[:, iP, jP, bL_un, :]
                    child_score = chart_score[:, iP, jP, bL_un]

                    rE_un_b = rE_un.view(1, 1, R_un, -1).expand(
                        B, P, R_un, -1)
                    score_inc = (
                        self._unary_compat_mod(child, rE_un_b)
                        + rB_un.view(1, 1, R_un))
                    new_un_score = child_score + score_inc
                    tau_u = max(float(self.chart_tau), 1e-3)
                    if tau_u != 1.0:
                        new_un_score = new_un_score / tau_u
                    zero_right = torch.zeros_like(child)
                    mm_un = torch.zeros(
                        B, P, R_un, 2, dtype=torch.bool, device=device)
                    new_un_per_rule = []
                    for r_local in range(R_un):
                        c_r = child[:, :, r_local, :]
                        mm_r = mm_un[:, :, r_local, :]
                        new_un_per_rule.append(self._apply_rule_forward(
                            un_methods[r_local], c_r,
                            zero_right[:, :, r_local, :],
                            mm_r, subspace=subspace))
                    new_un_vec = torch.stack(new_un_per_rule, dim=2)

                    rule_to_A_un = F.one_hot(
                        lhs_un, num_classes=C).to(dtype)
                    log_mask_un = torch.where(
                        rule_to_A_un > 0,
                        torch.zeros_like(rule_to_A_un),
                        torch.full_like(rule_to_A_un, NEG_INF))
                    scored_un = (new_un_score.unsqueeze(-1)
                                 + log_mask_un.view(1, 1, R_un, C))
                    if hard:
                        new_un_per_A, sel_idx = scored_un.max(dim=2)
                        b_ax = torch.arange(B, device=device).view(B, 1, 1)
                        p_ax = torch.arange(P, device=device).view(1, P, 1)
                        new_un_vec_per_A = new_un_vec[b_ax, p_ax, sel_idx]
                        finite = (new_un_per_A > NEG_INF / 2).unsqueeze(-1)
                        new_un_vec_per_A = new_un_vec_per_A * finite.to(dtype)
                    else:
                        new_un_per_A = torch.logsumexp(scored_un, dim=2)
                        weights_un = (
                            scored_un - new_un_per_A.view(B, P, 1, C)).exp()
                        new_un_vec_per_A = (weights_un.unsqueeze(-1)
                                            * new_un_vec.unsqueeze(-2)).sum(
                                                dim=2)

                    old_score_u = chart_score[:, i_range, i_range + w, :]
                    old_vec_u = chart_vec[:, i_range, i_range + w, :, :]
                    if hard:
                        pick_new = (new_un_per_A >= old_score_u)
                        combined_u = torch.where(
                            pick_new, new_un_per_A, old_score_u)
                        combined_vec_u = torch.where(
                            pick_new.unsqueeze(-1),
                            new_un_vec_per_A, old_vec_u)
                    else:
                        stacked_u = torch.stack(
                            [old_score_u, new_un_per_A], dim=0)
                        combined_u = torch.logsumexp(stacked_u, dim=0)
                        ow = (old_score_u - combined_u).exp()
                        nw = (new_un_per_A - combined_u).exp()
                        combined_vec_u = (
                            ow.unsqueeze(-1) * old_vec_u
                            + nw.unsqueeze(-1) * new_un_vec_per_A)
                    chart_score = chart_score.clone()
                    chart_vec = chart_vec.clone()
                    chart_score[:, i_range, i_range + w, :] = combined_u
                    chart_vec[:, i_range, i_range + w, :, :] = combined_vec_u

        self._chart_score = chart_score
        self._chart_vec = chart_vec
        # POS side-channel: per-cell probability simplex over the
        # category axis. Surfaced for downstream consumers (syntax
        # tree dumper, debugger, future losses).
        self._chart_pos = F.softmax(chart_score, dim=-1)

        # Outside pass mirrors the inside pass's hard / soft mode.
        outside_score, outside_vec = self._compose_chart_outside(
            chart_score, chart_vec, B, N, D, dtype, device,
            bin_idx, R_bin,
            rl_bin if R_bin > 0 else None,
            rr_bin if R_bin > 0 else None,
            lhs_bin if R_bin > 0 else None,
            mmask_bin_bool if R_bin > 0 else None,
            rE_bin if R_bin > 0 else None,
            rB_bin if R_bin > 0 else None,
            mB_bin if R_bin > 0 else None,
            mmask_bin if R_bin > 0 else None,
            bin_methods, NEG_INF, hard=hard)
        self._outside_score = outside_score
        self._outside_vec = outside_vec

        start_name = getattr(TheGrammar, 'start_symbol', 'S')
        cat_idx = self._category_index or {}
        start_id = cat_idx.get(start_name, None)
        try:
            collapse_mode = str(util.TheXMLConfig.get(
                "WordSpace.chartCollapse", "root")).strip().lower()
        except Exception:
            collapse_mode = "root"
        if start_id is not None and N > 0:
            root_vec = chart_vec[:, 0, N, start_id, :]
            if collapse_mode == "broadcast":
                composed = root_vec.unsqueeze(1).expand(B, N, D).contiguous()
            else:
                composed = torch.zeros(B, N, D,
                                       device=data.device, dtype=data.dtype)
                composed[:, 0, :] = root_vec
        else:
            composed = data

        self._signal_sentence_completed_chart(
            subspace, chart_score, word_space)

        # Always extract a Viterbi trace; it's the canonical derivation
        # for both rule-selection collection and (in eval mode) the
        # legacy SVO walker.
        self._derivation_trace = self._viterbi_extract(
            chart_score, chart_vec, B, N)

        # POS side-channel — Mechanism 3 (rule-prediction conditioning).
        # Walk the parsed trace and push each merge's LHS category
        # embedding onto WordSpace.category_stack so subsequent
        # WordSpace.predict_rule(b) calls see the parse history.
        # When no wordSpace / category_stack is wired, this is a no-op.
        self._populate_category_stack(word_space, B)

        return composed, self.last_svo

    def _populate_category_stack(self, word_space, B):
        """Push the Viterbi trace's LHS category embeddings onto
        ``word_space.category_stack`` so the rule predictor sees
        the parse history. Idempotent: clears the stack rows for
        the active batch range first.
        """
        if word_space is None:
            return
        cstack = getattr(word_space, 'category_stack', None)
        if cstack is None:
            return
        codebook = getattr(word_space, 'category_codebook', None)
        if codebook is None:
            return
        try:
            W = codebook.getW()
        except Exception:
            return
        if W is None or not torch.is_tensor(W):
            return
        # Reset rows for this batch range; safe under repeated calls.
        if hasattr(cstack, 'clear_rows'):
            try:
                cstack.clear_rows(0, B)
            except Exception:
                pass
        if hasattr(cstack, 'ensure_batch'):
            try:
                cstack.ensure_batch(B)
            except Exception:
                pass
        traces = self._derivation_trace
        if traces is None:
            return
        n_cb = int(W.shape[0])
        for b in range(B):
            if b >= len(traces):
                break
            row = traces[b] or []
            for entry in row:
                # entry: (gid, i, k, merged_slot, merged_category)
                cat_id = int(entry[4]) if len(entry) >= 5 else 0
                if cat_id < 0 or cat_id >= n_cb:
                    continue
                vec = W[cat_id]
                try:
                    cstack.push(b, vec.detach())
                except Exception:
                    # Stack may overflow on long parses; ignore.
                    break

    # ------------------------------------------------------------------
    # Outside pass (mirrors hard / soft mode).
    # ------------------------------------------------------------------
    def _compose_chart_outside(self, chart_score, chart_vec,
                               B, N, D, dtype, device,
                               bin_idx, R_bin, rl_bin, rr_bin, lhs_bin,
                               mmask_bin_bool, rE_bin, rB_bin, mB_bin,
                               mmask_bin, bin_methods, NEG_INF, *, hard):
        C = self._category_names_count()

        outside_score = torch.full((B, N + 1, N + 1, C), NEG_INF,
                                   device=device, dtype=dtype)
        outside_vec = torch.zeros((B, N + 1, N + 1, C, D),
                                  device=device, dtype=dtype)
        if N == 0 or R_bin == 0:
            return outside_score, outside_vec

        start_name = getattr(TheGrammar, 'start_symbol', 'S')
        cat_idx = self._category_index or {}
        start_id = cat_idx.get(start_name, None)
        outside_score = outside_score.clone()
        if start_id is not None:
            outside_score[:, 0, N, start_id] = 0.0

        tau = max(float(self.chart_tau), 1e-3)
        marker_prior = (mB_bin * mmask_bin).sum(-1)
        rB_view = rB_bin.view(1, 1, R_bin)
        mp_view = marker_prior.view(1, 1, R_bin)

        w_max_local = min(int(self.w_max), N)
        for w in range(w_max_local, 1, -1):
            P = N - w + 1
            if P <= 0:
                continue
            i_range = torch.arange(P, device=device)
            offsets = torch.arange(1, w, device=device)
            Sp = int(offsets.numel())

            parent_score = outside_score[:, i_range, i_range + w, :]
            parent_vec = outside_vec[:, i_range, i_range + w, :, :]
            if not torch.isfinite(parent_score).any() and (
                    parent_score.max().item() <= NEG_INF / 2):
                continue

            i_idx = i_range.unsqueeze(1).expand(P, Sp)
            k_idx = i_idx + offsets.unsqueeze(0).expand(P, Sp)
            i_flat = i_idx.reshape(-1)
            k_flat = k_idx.reshape(-1)
            j_flat = (i_flat + w)
            i2 = i_flat.unsqueeze(1).expand(-1, R_bin)
            k2 = k_flat.unsqueeze(1).expand(-1, R_bin)
            j2 = j_flat.unsqueeze(1).expand(-1, R_bin)
            bL = rl_bin.unsqueeze(0).expand(P * Sp, R_bin)
            bR = rr_bin.unsqueeze(0).expand(P * Sp, R_bin)

            left_in_score = chart_score[:, i2, k2, bL]
            right_in_score = chart_score[:, k2, j2, bR]
            left_in_vec = chart_vec[:, i2, k2, bL, :]
            right_in_vec = chart_vec[:, k2, j2, bR, :]

            lhs_index = lhs_bin.view(1, 1, R_bin).expand(B, P, R_bin)
            parent_score_per_rule = parent_score.gather(-1, lhs_index)
            parent_vec_per_rule = parent_vec.gather(
                -2, lhs_index.unsqueeze(-1).expand(B, P, R_bin, D))
            parent_score_per_rule = parent_score_per_rule.unsqueeze(2).expand(
                B, P, Sp, R_bin).reshape(B, P * Sp, R_bin)
            parent_vec_per_rule = parent_vec_per_rule.unsqueeze(2).expand(
                B, P, Sp, R_bin, D).reshape(B, P * Sp, R_bin, D)

            rE = rE_bin.view(1, 1, R_bin, -1).expand(
                B, P * Sp, R_bin, -1)
            mm = mmask_bin_bool.view(1, 1, R_bin, 2).expand(
                B, P * Sp, R_bin, 2)
            compat = self._compat_score_mod(
                left_in_vec, right_in_vec, rE, mm)

            push_left_score = (parent_score_per_rule + right_in_score
                               + rB_view + compat + mp_view)
            push_right_score = (parent_score_per_rule + left_in_score
                                + rB_view + compat + mp_view)
            if tau != 1.0:
                push_left_score = push_left_score / tau
                push_right_score = push_right_score / tau

            outside_score = outside_score.clone()
            outside_vec = outside_vec.clone()
            for r in range(R_bin):
                Br = int(rl_bin[r].item())
                Cr = int(rr_bin[r].item())
                old_l_score = outside_score[:, i_flat, k_flat, Br]
                old_l_vec = outside_vec[:, i_flat, k_flat, Br, :]
                add_l_score = push_left_score[:, :, r].view(B, P * Sp)
                add_l_vec = parent_vec_per_rule[:, :, r, :].view(
                    B, P * Sp, D)
                if hard:
                    pick_new = (add_l_score >= old_l_score)
                    comb_l = torch.where(pick_new, add_l_score, old_l_score)
                    outside_score[:, i_flat, k_flat, Br] = comb_l
                    outside_vec[:, i_flat, k_flat, Br, :] = torch.where(
                        pick_new.unsqueeze(-1), add_l_vec, old_l_vec)
                else:
                    stk_l = torch.stack([old_l_score, add_l_score], dim=0)
                    comb_l = torch.logsumexp(stk_l, dim=0)
                    w_old_l = (old_l_score - comb_l).exp()
                    w_new_l = (add_l_score - comb_l).exp()
                    outside_score[:, i_flat, k_flat, Br] = comb_l
                    outside_vec[:, i_flat, k_flat, Br, :] = (
                        w_old_l.unsqueeze(-1) * old_l_vec
                        + w_new_l.unsqueeze(-1) * add_l_vec)
                old_r_score = outside_score[:, k_flat, j_flat, Cr]
                old_r_vec = outside_vec[:, k_flat, j_flat, Cr, :]
                add_r_score = push_right_score[:, :, r].view(B, P * Sp)
                add_r_vec = parent_vec_per_rule[:, :, r, :].view(
                    B, P * Sp, D)
                if hard:
                    pick_new = (add_r_score >= old_r_score)
                    comb_r = torch.where(pick_new, add_r_score, old_r_score)
                    outside_score[:, k_flat, j_flat, Cr] = comb_r
                    outside_vec[:, k_flat, j_flat, Cr, :] = torch.where(
                        pick_new.unsqueeze(-1), add_r_vec, old_r_vec)
                else:
                    stk_r = torch.stack([old_r_score, add_r_score], dim=0)
                    comb_r = torch.logsumexp(stk_r, dim=0)
                    w_old_r = (old_r_score - comb_r).exp()
                    w_new_r = (add_r_score - comb_r).exp()
                    outside_score[:, k_flat, j_flat, Cr] = comb_r
                    outside_vec[:, k_flat, j_flat, Cr, :] = (
                        w_old_r.unsqueeze(-1) * old_r_vec
                        + w_new_r.unsqueeze(-1) * add_r_vec)

        return outside_score, outside_vec

    # ------------------------------------------------------------------
    # Sentence-completion signaling.
    # ------------------------------------------------------------------
    def _signal_sentence_completed_chart(self, subspace, chart_score,
                                         word_space):
        ws = word_space if word_space is not None else (
            getattr(subspace, 'wordSpace', None) if subspace else None)
        if ws is None:
            return
        start_name = getattr(TheGrammar, 'start_symbol', 'S')
        cat_idx = self._category_index or {}
        start_id = cat_idx.get(start_name, None)
        if start_id is None:
            return
        Bf, N1, _, _C = chart_score.shape
        N = N1 - 1
        if N < 1:
            return
        root = chart_score[:, 0, N, :]
        argmax = root.argmax(dim=-1)
        completed_flat = (argmax == int(start_id))
        if not bool(completed_flat.any().item()):
            return
        K = ws._row_K() if hasattr(ws, '_row_K') else 1
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

    # ------------------------------------------------------------------
    # Viterbi backtrace (also used by reverse / generate path).
    # ------------------------------------------------------------------
    def _viterbi_extract(self, chart_score, chart_vec, B, N):
        device = chart_score.device
        table = TheGrammar._ensure_packed_table(device=device)
        R = int(table['lhs'].numel())
        if R == 0 or N == 0:
            return [[] for _ in range(B)]
        rhs_left = table['rhs_left']
        rhs_right = table['rhs_right']
        lhs = table['lhs']
        arity = table['arity']
        global_id = table['global_id']
        marker_mask_full = table['marker_mask'].to(device=device)
        start_name = getattr(TheGrammar, 'start_symbol', 'S')
        cat_idx = self._category_index or {}
        start_id = cat_idx.get(start_name, 0)

        rule_bias_all = self._rule_bias.detach()
        rule_embed_all = self._rule_embed.detach()
        marker_bias_all = self._marker_bias.detach()

        traces = [[] for _ in range(B)]
        for b in range(B):
            stack = [(0, N, int(start_id))]
            while stack:
                i, j, A = stack.pop()
                if j - i <= 1:
                    continue
                if j - i > self.w_max:
                    continue
                best = None
                best_score = -float('inf')
                for r in range(R):
                    if int(arity[r].item()) != 2:
                        continue
                    if int(lhs[r].item()) != A:
                        continue
                    Br = int(rhs_left[r].item())
                    Cr = int(rhs_right[r].item())
                    mm_r = marker_mask_full[r]
                    rE_r = rule_embed_all[r].view(1, -1)
                    rB_r = float(rule_bias_all[r].item())
                    mP_r = float(
                        (marker_bias_all[r]
                         * mm_r.to(marker_bias_all.dtype)
                         ).sum().item())
                    for k in range(i + 1, j):
                        s_left = chart_score[b, i, k, Br].item()
                        s_right = chart_score[b, k, j, Cr].item()
                        l_vec = chart_vec[b, i, k, Br].view(1, -1)
                        r_vec = chart_vec[b, k, j, Cr].view(1, -1)
                        compat = float(self._compat_score_mod(
                            l_vec, r_vec, rE_r, mm_r.view(1, 2)).item())
                        cand = s_left + s_right + rB_r + compat + mP_r
                        if cand > best_score:
                            best_score = cand
                            best = (r, k, Br, Cr)
                if best is None:
                    continue
                r, k, Br, Cr = best
                gid = int(global_id[r].item())
                # Trace tuple: (gid, i, k, j, A). j is the parent
                # span's right edge — recorded so syntax-tree
                # reconstruction (write_syntax_tree) can recover the
                # full span without re-walking the stack.
                traces[b].append((gid, i, k, j, A))
                stack.append((i, k, Br))
                stack.append((k, j, Cr))
        return traces


# ---------------------------------------------------------------------
# Chart sub-modules: scoring heads and (deprecated) compose helpers.
# Kept at module scope rather than nested in Chart so they survive
# pickling and are reachable by tests by import name.
# ---------------------------------------------------------------------
class _CompatScore(nn.Module):
    """Scores how plausible (left, right, rule) is as a merge.

    Bounded by ``compat_scale * tanh(...)`` so the per-cell softmax
    over rules cannot saturate to one-hot during training.
    """

    def __init__(self, D, D_rule, hidden=None, compat_scale=2.0):
        super().__init__()
        h = hidden or max(D, 64)
        self.lin1 = nn.Linear(2 * D + D_rule, h)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(h, 1)
        self.compat_scale = float(compat_scale)
        nn.init.xavier_normal_(self.lin1.weight)
        nn.init.xavier_normal_(self.lin2.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, left, right, rule_embed, marker_mask):
        keep = (~marker_mask).to(left.dtype)
        kL = keep[..., 0:1]
        kR = keep[..., 1:2]
        x = torch.cat([left * kL, right * kR, rule_embed], dim=-1)
        raw = self.lin2(self.act(self.lin1(x))).squeeze(-1)
        return self.compat_scale * torch.tanh(raw)


class _UnaryCompat(nn.Module):
    """Score for a unary closure step. Bounded the same way as
    ``_CompatScore``.
    """

    def __init__(self, D, D_rule, hidden=None, compat_scale=2.0):
        super().__init__()
        h = hidden or max(D, 64)
        self.lin1 = nn.Linear(D + D_rule, h)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(h, 1)
        self.compat_scale = float(compat_scale)
        nn.init.xavier_normal_(self.lin1.weight)
        nn.init.xavier_normal_(self.lin2.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, child, rule_embed):
        x = torch.cat([child, rule_embed], dim=-1)
        raw = self.lin2(self.act(self.lin1(x))).squeeze(-1)
        return self.compat_scale * torch.tanh(raw)


# =====================================================================
# Per-space SyntacticLayer (2026-05-01 refactor).
#
# Spec: doc/specs/2026-05-01-syntactic-layer-refactor.md §4.
#
# Each PerceptualSpace / ConceptualSpace / SymbolicSpace owns one of
# these. Holds the parametrized GrammarLayer instances for its tier's
# rules and dispatches `forward` / `reverse` based on the rule choice
# the chart wrote into ``word_space.current_rules`` /
# ``generate_rules`` (Q4 / Q10.1).
#
# Temporarily named ``SpaceSyntacticLayer`` while the legacy module-
# global ``SyntacticLayer`` is still in use (Q3: keep the name
# ``SyntacticLayer`` post-refactor; rename happens in Step 9 of the
# implementation order, when the legacy class is deleted).
# =====================================================================
class SpaceSyntacticLayer(Layer):
    """Per-space dispatcher.

    Construction:
        SpaceSyntacticLayer(tier='C', word_space=word_space,
                            host_layers={'intersection': pi_layer})

    Each entry in ``host_layers`` is registered with ``word_space`` at
    construction. The space's ``forward()`` and ``reverse()`` delegate
    here; ``forward()`` reads ``word_space.current_rules[tier]``,
    advances a per-tier cursor, and dispatches to the appropriate
    layer's ``compose`` (binary) or ``forward`` (unary). ``reverse()``
    mirrors via ``word_space.generate_rules[tier]`` and ``layer.generate``.

    The cursor resets at the start of each new ``word_space.compose()``
    / ``word_space.generate()`` call via the generation counters on
    WordSpace (Q10.1).
    """

    def __init__(self, tier, word_space, host_layers, default_rule=None):
        super().__init__(0, 0)
        self.tier = str(tier)
        # Stash host_layers in two parallel structures: ModuleList for
        # nn.Module bookkeeping (so optimizer scans see the parameters)
        # and a name-keyed dict for O(1) dispatch lookup.
        layers_list = [layer for layer in host_layers.values()
                       if layer is not None]
        self.layers = nn.ModuleList(layers_list)
        self._by_name = {name: layer for name, layer in host_layers.items()
                         if layer is not None}
        self.default_rule = default_rule
        # Register each host_layer with the wordSpace's host_layer
        # registry so the chart can dispatch into them.
        for rule_name, layer in self._by_name.items():
            word_space.register_host_layer(self.tier, rule_name, layer)
        self._cursor_compose = 0
        self._cursor_generate = 0
        self._cursor_compose_gen = -1
        self._cursor_generate_gen = -1
        # Stash the wordSpace as a non-Module attribute to avoid the
        # circular nn.Module ownership trap (wordSpace owns the chart;
        # chart's host_layer registry references this layer's children;
        # this layer references wordSpace).
        object.__setattr__(self, '_word_space', word_space)

    # -- cursor management ---------------------------------------------
    def _next_rule_name(self, *, direction):
        """Pop the next rule name for ``direction`` ('compose' or
        'generate'). Resets the cursor when wordSpace has bumped its
        generation counter for this direction.

        Reads ``word_space.current_rules`` / ``generate_rules`` as
        ``dict[tier, list[list[int]]]`` (per-row, per-step). For now
        we use row 0 as the canonical sequence; per-row dispatch (where
        rows fire different rules at the same step) is a follow-on.

        Returns the rule's ``method_name`` (string) or ``None`` /
        ``self.default_rule`` when no chart rule is available. The
        method name is the key used in ``self._by_name``.
        """
        ws = self._word_space
        if direction == 'compose':
            rules = ws.current_rules
            gen = ws._compose_generation
            if gen != self._cursor_compose_gen:
                self._cursor_compose = 0
                self._cursor_compose_gen = gen
            cursor = self._cursor_compose
        else:
            rules = ws.generate_rules
            gen = ws._generate_generation
            if gen != self._cursor_generate_gen:
                self._cursor_generate = 0
                self._cursor_generate_gen = gen
            cursor = self._cursor_generate
        per_tier = rules.get(self.tier) if rules else None
        per_step = self._row_zero_rules(per_tier)
        if cursor < len(per_step):
            rule_id = per_step[cursor]
            if direction == 'compose':
                self._cursor_compose = cursor + 1
            else:
                self._cursor_generate = cursor + 1
            try:
                method_name = TheGrammar.rules[int(rule_id)].method_name
            except (IndexError, AttributeError, ValueError, TypeError):
                method_name = None
            return method_name or self.default_rule
        # Fallback: legacy / no-chart configurations supply a default
        # rule (the per-space "natural" fold, e.g. 'intersection' for C).
        return self.default_rule

    @staticmethod
    def _row_zero_rules(per_tier):
        """Extract row 0's rule sequence from a per-row container.

        Tolerates both legacy ``list[int]`` (flat) and the multi-row
        ``list[list[int]]`` shape so callers using either contract
        keep working during the migration window.
        """
        if not per_tier:
            return []
        # Multi-row: list of lists.
        if isinstance(per_tier[0], list):
            return per_tier[0]
        # Flat list of ints (legacy).
        return per_tier

    # -- forward / reverse dispatch ------------------------------------
    #
    # The per-space dispatch takes a subspace and operates on the
    # subspace's tier-appropriate field:
    #   * S tier: the .what content (symbol activations)
    #   * P / C tier: the .event content (percept / concept activations)
    #
    # Rule choices come from word_space.current_rules / generate_rules
    # (populated by the chart). Cursor advances one step per call.
    def forward(self, subspace):
        """Fire one fold step on ``subspace`` per the chart's rule choice.

        Materializes the subspace's tier-appropriate field, applies
        the chosen rule's GrammarLayer.forward, writes the result back
        into the same field. Returns the (possibly-mutated) subspace.

        Falls back to ``self.default_rule`` when the chart hasn't
        written a rule for this tier.

        On the signal-router path the chart has already executed the
        derivation tensorially (router.compose folded the slab via the
        op modules and wrote the [B, 1, D] root state back into
        subspace.event). The legacy per-rule unary fold here would
        double-apply the op (and crash on truly-binary ops like
        ConjunctionLayer / DisjunctionLayer that don't expose a unary
        forward). Skip it.
        """
        ws = self._word_space
        if (ws is not None and getattr(ws, 'chart', None) is not None
                and getattr(ws.chart, 'router_kind', 'chart') == 'signal'):
            # Still advance the cursor so reverse() pops in sync.
            self._next_rule_name(direction='compose')
            return subspace
        rule_name = self._next_rule_name(direction='compose')
        if rule_name is None:
            return subspace
        layer = self._by_name.get(rule_name)
        if layer is None:
            return subspace
        # Binary (arity-2) rules are executed inside the chart's compose
        # via host_layer.compose(left, right). Calling .forward(x) on
        # them as a unary post-chart fold would crash (no `right` arg).
        # Only fire unary rules here.
        if int(getattr(layer, 'arity', 1)) != 1:
            return subspace
        x = self._read_subspace(subspace)
        if x is None:
            return subspace
        y = layer.forward(x)
        self._write_subspace(subspace, y)
        return subspace

    def reverse(self, subspace):
        """Inverse of ``forward``: fire one fold-reverse step on the
        subspace's tier-appropriate field. No-op when the rule isn't
        invertible.

        Skipped on the signal-router path — the chart's generate()
        handles inverse routing tensorially.
        """
        ws = self._word_space
        if (ws is not None and getattr(ws, 'chart', None) is not None
                and getattr(ws.chart, 'router_kind', 'chart') == 'signal'):
            self._next_rule_name(direction='generate')
            return subspace
        rule_name = self._next_rule_name(direction='generate')
        if rule_name is None:
            return subspace
        layer = self._by_name.get(rule_name)
        if layer is None or not getattr(layer, 'invertible', False):
            return subspace
        # Binary rules' inverse is handled by the chart's generate via
        # host_layer.generate(parent); skip here.
        if int(getattr(layer, 'arity', 1)) != 1:
            return subspace
        y = self._read_subspace(subspace)
        if y is None:
            return subspace
        x = layer.reverse(y)
        self._write_subspace(subspace, x)
        return subspace

    # -- subspace I/O per tier ------------------------------------------
    def _read_subspace(self, subspace):
        """Read the tier-appropriate tensor from ``subspace``.

        S tier: subspace.what (the symbol content). Codebook / Tensor
        bases expose ``getW()``; if ``what`` doesn't carry per-position
        symbol activations, fall back to materialize() so the layer
        sees the muxed event.

        P / C tier: subspace.materialize() (the muxed event tensor).
        """
        if subspace is None:
            return None
        if self.tier == 'S':
            what = getattr(subspace, 'what', None)
            if what is not None and hasattr(what, 'getW'):
                try:
                    w = what.getW()
                    if torch.is_tensor(w) and w.ndim >= 2:
                        return w
                except Exception:
                    pass
        if hasattr(subspace, 'materialize'):
            return subspace.materialize()
        return getattr(subspace, 'event', None)

    def _write_subspace(self, subspace, tensor):
        """Write ``tensor`` back into the tier-appropriate field.

        S tier: subspace.what.setW(tensor) when available; otherwise
        falls through to event so downstream consumers still see the
        update.

        P / C tier: subspace.set_event(tensor).
        """
        if subspace is None or tensor is None:
            return
        if self.tier == 'S':
            what = getattr(subspace, 'what', None)
            if what is not None and hasattr(what, 'setW'):
                try:
                    what.setW(tensor)
                    return
                except Exception:
                    pass
        if hasattr(subspace, 'set_event'):
            subspace.set_event(tensor)


# =====================================================================
# Grammar-op class registry for lazy host_layer construction (Q10.4).
# Maps rule method_name -> GrammarLayer class. Per-space syntactic
# layers consult this when the grammar references a rule whose host
# parametrized layer isn't already owned by the space.
# =====================================================================
def _grammar_layer_classes():
    """Return the rule_name -> GrammarLayer class registry.

    Step 8 of the 2026-05-01 refactor: read from the canonical module-
    level ``GRAMMAR_LAYER_CLASSES`` dict on Layers.py instead of
    rebuilding it. Returns ``{}`` if Layers.py isn't yet importable
    (Layers.py imports from Language.py at module load).
    """
    try:
        from Layers import GRAMMAR_LAYER_CLASSES
    except ImportError:
        return {}
    return dict(GRAMMAR_LAYER_CLASSES)


def build_space_syntactic_layer(space, word_space, *, tier,
                                builtin_layers=None,
                                default_rule=None):
    """Construct a per-space SpaceSyntacticLayer.

    Args:
        space: the host Space (PerceptualSpace / ConceptualSpace /
            SymbolicSpace). The constructed layer is stored on
            ``space.syntacticLayer`` and registered in the wordSpace's
            host_layer registry.
        word_space: the WordSpace coordinator. Owns the host_layer
            registry and the chart.
        tier: tier name ('P' / 'C' / 'S') used as the registry key.
        builtin_layers: dict[rule_name -> GrammarLayer instance] for
            rules backed by an already-constructed parametrized layer
            (e.g. {'intersection': space.pi}). These instances are
            registered as-is so their existing weights participate in
            training.
        default_rule: rule name to fall back to when the chart hasn't
            written a rule choice for this tier (legacy / no-chart
            configurations). Pass ``None`` to passthrough on missing
            choice.
    """
    builtin_layers = dict(builtin_layers or {})
    host_layers = dict(builtin_layers)
    cls_registry = _grammar_layer_classes()
    for rule in TheGrammar.rules:
        rule_tier = getattr(rule, 'tier', None)
        if rule_tier != tier:
            continue
        mn = getattr(rule, 'method_name', None)
        if not mn or mn in host_layers:
            continue
        cls = cls_registry.get(mn)
        if cls is None:
            continue
        try:
            host_layers[mn] = cls()
        except TypeError:
            # Some GrammarLayer wrappers (IntersectionLayer / UnionLayer)
            # require a parametrized inner layer at construction. Without
            # one, the host space's existing instance should already be
            # in builtin_layers; if it isn't, skip rather than fail.
            continue
    layer = SpaceSyntacticLayer(
        tier=tier, word_space=word_space,
        host_layers=host_layers, default_rule=default_rule)
    space.syntacticLayer = layer
    return layer


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
        # Soft-chart machinery (Delta 2). Lazily built on first
        # _compose_chart_cky call; cached version comes from
        # `Grammar.rule_table_version`. Note: rule-shaped Parameters
        # (_rule_embed / _rule_bias / _marker_bias) are NOT pre-set
        # here -- nn.Module's register_parameter rejects names that
        # already exist as plain attributes. Read sites should use
        # `getattr(self, '_rule_embed', None)`.
        #
        # When `WordSpace.softChartCompose` is true in the loaded XML
        # config, eagerly build the soft chart at construction time so
        # the parameters are visible to the optimizer's parameter scan
        # (which runs before the first forward pass). Without this,
        # the lazy-build path registers _rule_bias / _rule_embed /
        # _marker_bias only on first compose, after the optimizer has
        # already collected parameters -- the chart params then never
        # get gradient updates.
        self._soft_chart_built = False
        self._soft_chart_version = -1
        self._compat_score_mod = None
        self._unary_compat_mod = None
        self._lex_cat_scorer = None
        self.w_max = 8
        self.unary_max_depth = 2
        self.D_rule = 32
        # Softmax temperature on the chart's per-cell rule mixture.
        # τ = 1.0 reproduces standard softmax. Larger τ keeps the
        # mixture softer (multiple rules contribute meaningfully);
        # smaller τ sharpens toward one-hot. Setting τ ≥ 2.0 early in
        # training and annealing toward 1.0 prevents the saturation
        # observed when `compat_score` learns to produce large rule-
        # discriminating logits. See doc/research/2017-jang-gumbel-
        # softmax.md. Configurable via WordSpace.chartTau in the XML.
        try:
            from util import TheXMLConfig as _TheXMLConfig
            self.chart_tau = float(_TheXMLConfig.get("WordSpace.chartTau", 1.0))
        except Exception:
            self.chart_tau = 1.0
        # Surface-3 wiring (see doc/research/three-surfaces.md): roster
        # of GrammarLayer instances that have registered themselves
        # with this SyntacticLayer as their gate authority.
        self._registered_grammar_layers = []
        try:
            from util import TheXMLConfig as _TheXMLConfig
            _eager_chart = bool(_TheXMLConfig.get(
                "WordSpace.softChartCompose", False))
        except Exception:
            _eager_chart = False
        if _eager_chart and grammar is not None:
            try:
                grammar._ensure_configured()
                D_eager = (feature_dim if feature_dim is not None
                           else nInput)
                self._ensure_soft_chart_built(
                    grammar, int(D_eager),
                    torch.device('cpu'))
            except Exception:
                # If grammar isn't yet configured at this point, fall
                # back to lazy build. The optimizer-gradient gap
                # warning belongs to the caller.
                pass
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
        # Install ourselves as the class-level chart authority for
        # GrammarLayer at the very end of __init__ (after every other
        # attribute is set), so any GrammarLayer that auto-registers
        # via this authority sees a fully-constructed SyntacticLayer.
        # With the default model.xml grammar (pi / sigma / not in
        # their natural tiers), rule_probability for those ops returns
        # 1.0 (or the dormant default) so `gated_run` shortcuts to
        # passthrough -- the registration is a true no-op until a
        # call site decides to consult `should_run_rule`.
        try:
            from Layers import GrammarLayer as _GrammarLayer
            _GrammarLayer.set_chart_authority(self)
        except Exception:
            pass

    def register_grammar_layer(self, layer):
        """Add a GrammarLayer instance to this authority's roster.
        Idempotent on repeated registration."""
        if layer not in self._registered_grammar_layers:
            self._registered_grammar_layers.append(layer)

    def should_run_rule(self, rule_name):
        """Return the firing probability for ``rule_name`` per the
        grammar's `rule_probability` lookup.

        Synthesizes a `body` string matching the convention of
        `Grammar.rule_probability` so dormant defaults like
        ``not(...) -> 0.0`` and ``Contiguous(...) -> thought_free``
        still apply. Used by `GrammarLayer.gated_run` to gate
        parameterized folds. Returns 1.0 when no grammar is wired.
        """
        if self.grammar is None or not rule_name:
            return 1.0
        # Construct a body that the rule_probability prefix-checks
        # will recognize. Arity is unknown here, so use the unary form
        # with a single placeholder; rule_probability's startswith()
        # checks only inspect "<name>(", which matches.
        body = f"{rule_name}(S)"
        try:
            return float(self.grammar.rule_probability(body))
        except Exception:
            return 1.0

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

    # -- Forward/Reverse rule dispatch (REMOVED 2026-05-01) ----------
    # The legacy `*Forward` / `*Reverse` rule methods, the
    # `_RULE_METHODS` / `_OPS_METHODS` / `_METHOD_ALIASES` dispatch
    # tables, and `project` / `reverse_project` / `dispatch_ops` /
    # `_resolve_method` were removed. Per-rule math now lives on
    # the `GRAMMAR_LAYER_CLASSES` GrammarLayer subclasses
    # (Layers.py); the chart's `Chart._apply_rule_forward` and the
    # per-space `SpaceSyntacticLayer.forward` route through them.

    # -- compose / decompose / chart helpers (REMOVED 2026-05-01) ---
    # The legacy `compose`, `_compose_vector`, `_compose_to_target`,
    # `_compose_activation`, `_compose_vector_chart`,
    # `_compose_chart_cky`, `_compose_chart_outside`,
    # `_viterbi_extract`, `_signal_sentence_completed*`,
    # `_pair_scorer`, `_live_pairs`, `_apply_rules_to_pairs`,
    # `_ensure_soft_chart_built`, `_ensure_category_table`,
    # `_seed_category`, `_category_names_count`,
    # `_extract_svo_from_trace`, `emit_head`, `generate`,
    # `decompose` were all removed. Chart parsing lives on the
    # `Chart` class (top of this file). The `WordSpace` owns a
    # `Chart` instance; the chart is wired into the model pipeline
    # via `Pipeline.ChartCompose` / `ChartGenerate`.

    # -- utilities -------------------------------------------------

    def decompose(self, *args, **kwargs):
        """Back-compat alias for ``generate``. Renamed 2026-05-01."""
        return self.generate(*args, **kwargs)

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
        # 4a. Chart + host-layer registry. Per the 2026-05-01 syntactic-
        # layer refactor (doc/specs/2026-05-01-syntactic-layer-refactor.md):
        # WordSpace owns a Chart that runs CKY inside / outside passes
        # and writes per-(tier, step) rule selections into
        # ``current_rules`` / ``generate_rules``. Each per-space
        # SyntacticLayer registers its parametrized layers via
        # ``register_host_layer``; the chart consults
        # ``host_layer(tier, rule_name)`` to fire host-owned folds.
        self._host_layer_registry = {}
        self.current_rules = {}
        self.generate_rules = {}
        # Bumped on each compose / generate. Per-space SyntacticLayers
        # compare against this to know when to reset their per-tier
        # cursor (Q10.1).
        self._compose_generation = 0
        self._generate_generation = 0
        chart_hidden = self._resolve_hidden_dim(nSymbols)
        self.chart = Chart(
            nInput=nSymbols, nOutput=nSymbols,
            max_depth=max(nSymbols - 1, 1),
            hidden_dim=chart_hidden,
            feature_dim=symbol_dim,
        )
        self.layers.append(self.chart)
        for p in self.chart.parameters():
            if all(p is not q for q in self.params):
                self.params.append(p)
        # 5. Build the SyntacticLayer anchored at SymbolicSpace.  The
        # perceptual and conceptual spaces also get a ``wordSpace``
        # back-reference so they can route through the shared buffer,
        # but only the symbolic space's compose() fires the layer.
        if perceptualSpace is not None:
            perceptualSpace.attach_wordSpace(self)
            self._attach_per_space_syntactic_layer(
                perceptualSpace, tier='P')
        if conceptualSpace is not None:
            conceptualSpace.attach_wordSpace(self)
            self._attach_per_space_syntactic_layer(
                conceptualSpace, tier='C')
        if symbolicSpace is not None:
            self._build_syntactic_layer(
                symbolicSpace, nSymbols, grammar, symbol_dim)
            self._attach_per_space_syntactic_layer(
                symbolicSpace, tier='S')

        # 5b. Signal-router grammar wiring. When `WordSpace.routerKind ==
        # "signal"`, the chart's CKY paths are bypassed; the SignalRouter
        # needs explicit op modules attached to its per-tier scorers
        # before compose() can fire. We wire from the host_layer registry
        # populated in step 5 above.
        if self.chart.router_kind == "signal":
            self._wire_signal_router_grammar_ops()

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
        # persistent=False: runtime scratch state, not learned weights.
        # Excluding from state_dict avoids load-time shape mismatches when
        # the live model rebuilds at batch=1 and ensure_batch() resizes later.
        self.register_buffer(
            "_last_svo", torch.zeros(self.batch, 3, self.svo_dim),
            persistent=False)
        self.register_buffer(
            "_svo_valid", torch.zeros(self.batch, dtype=torch.bool),
            persistent=False)

        # STM-residual: fires once per sentence per row on the first
        # stm_residual(b) call; arm_stm() / Reset() re-arm. Runtime scratch.
        self.register_buffer(
            "_stm_fired", torch.zeros(self.batch, dtype=torch.bool),
            persistent=False)
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

    # -- chart compose / generate (2026-05-01 refactor) ---------------
    def compose(self, input_vectors, subspace=None):
        """Run the chart's inside pass; populate ``self.current_rules``.

        Idempotent within a forward pass: each per-space SyntacticLayer
        resets its per-tier cursor to 0 and pops one rule per fold step.
        """
        self._compose_generation += 1
        self.current_rules = self.chart.compose(
            input_vectors, self, subspace=subspace) or {}
        return self.current_rules

    def generate(self, target_vectors, subspace=None):
        """Run the chart's outside pass + Viterbi backtrace; populate
        ``self.generate_rules``.
        """
        self._generate_generation += 1
        self.generate_rules = self.chart.generate(
            target_vectors, self, subspace=subspace) or {}
        return self.generate_rules

    def register_host_layer(self, tier, rule_name, layer):
        """Register ``layer`` as the parametrized GrammarLayer for
        ``(tier, rule_name)``. The chart's per-cell rule dispatch reads
        the registry to fire the host space's owned fold.

        Per-space SyntacticLayers call this at construction.
        """
        if not rule_name:
            return
        self._host_layer_registry[(tier, rule_name)] = layer

    def host_layer(self, tier, rule_name):
        """Return the registered GrammarLayer for ``(tier, rule_name)``,
        or ``None``. The chart treats ``None`` as "rule has no host
        parametrized layer", routing to the generic fallback (Ops /
        typed-GrammarLayer facade) instead.
        """
        return self._host_layer_registry.get((tier, rule_name))

    def _wire_signal_router_grammar_ops(self):
        """Pull host_layer instances out of the registry and attach them
        to the signal router, grouped by (tier, arity).

        Walked on WordSpace construction when ``Chart.router_kind ==
        "signal"``. The chart's ``_ensure_signal_router()`` builds the
        SignalRouter the first time we ask for it; we then call
        ``attach_unary_ops`` / ``attach_layer_ops`` per tier with the
        live grammar's parametrized fold modules.

        Binary GrammarLayers (IntersectionLayer, UnionLayer, ...) expose
        their pair-wise math via ``.compose(left, right)``; unary ones
        (NotLayer, NonLayer, ...) via ``.forward(x)``. The signal
        router's ``BinaryStructuredReductionLayer`` calls ``op(left,
        right)``, so binary ops get wrapped with
        ``_BinaryGrammarOpAdapter``.
        """
        from SignalRouter import _BinaryGrammarOpAdapter

        router = self.chart._ensure_signal_router()

        # Group: (tier, arity) -> list of (rule_id, layer, rule_name)
        by_tier_arity = {}
        for rule_id in range(len(TheGrammar.rules)):
            rule = TheGrammar.rules[rule_id]
            tier = rule.tier
            arity = int(rule.arity)
            if arity not in (1, 2):
                continue
            rule_name = rule.method_name
            if not rule_name:
                continue
            layer = self._host_layer_registry.get((tier, rule_name))
            if layer is None:
                # Fallback 1: same rule registered under a different
                # tier. The grammar may declare a rule at the symbolic
                # tier whose host_layer was set up at the conceptual
                # tier (e.g. IntersectionLayer wraps a PiLayer that
                # ConceptualSpace owns).
                for (other_tier, other_rule), other_layer in (
                        self._host_layer_registry.items()):
                    if other_rule == rule_name:
                        layer = other_layer
                        break
            if layer is None:
                # Fallback 2: instantiate a fresh module from the
                # GRAMMAR_LAYER_CLASSES facade. Parametrized wrappers
                # (IntersectionLayer, UnionLayer) require an inner
                # pi/sigma layer; skip if we can't construct one.
                from Layers import GRAMMAR_LAYER_CLASSES
                cls = GRAMMAR_LAYER_CLASSES.get(rule_name)
                if cls is None:
                    continue
                try:
                    layer = cls()
                except TypeError:
                    continue
            by_tier_arity.setdefault((tier, arity), []).append(
                (rule_id, layer, rule_name))

        # Attach per (tier, arity).
        for (tier, arity), entries in sorted(by_tier_arity.items()):
            ops = []
            rule_ids = []
            for rule_id, layer, _name in entries:
                if arity == 2:
                    ops.append(_BinaryGrammarOpAdapter(layer))
                else:
                    ops.append(layer)
                rule_ids.append(rule_id)
            if arity == 1:
                router.attach_unary_ops(ops=ops, rule_ids=rule_ids, tier=tier)
            else:
                router.attach_layer_ops(ops=ops, rule_ids=rule_ids, tier=tier)

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

    def _attach_per_space_syntactic_layer(self, space, *, tier):
        """Build the per-space SpaceSyntacticLayer for ``space`` (Step 4
        of doc/specs/2026-05-01-syntactic-layer-refactor.md).

        Gathers the space's already-constructed parametrized layers
        (PiLayer / SigmaLayer / NotLayer / ContiguousLayer) and passes
        them into ``build_space_syntactic_layer`` as ``builtin_layers``
        so their existing weights stay live. Other rules the configured
        grammar references for this tier get lazy-constructed
        GrammarLayer wrappers.
        """
        builtin_layers = {}
        # Inner instance probes: use try/except rather than getattr-with-
        # defaults per the project's no-defensive-getattr stance.
        if tier == 'P':
            sigma = getattr(space, 'sigma', None)
            if sigma is not None:
                # Wrap PiLayer / SigmaLayer in their grammar-facing
                # GrammarLayer adapters so the chart's ``layer.compose``
                # / ``layer.generate`` contract is satisfied.
                from Layers import UnionLayer
                builtin_layers['union'] = UnionLayer(sigma)
            default_rule = 'union'
        elif tier == 'C':
            pi = getattr(space, 'pi', None)
            if pi is not None:
                from Layers import IntersectionLayer
                builtin_layers['intersection'] = IntersectionLayer(pi)
            default_rule = 'intersection'
        elif tier == 'S':
            sigma = getattr(space, 'sigma', None)
            if sigma is not None:
                from Layers import UnionLayer
                builtin_layers['union'] = UnionLayer(sigma)
            negation = getattr(space, 'propositional_negation', None)
            if negation is not None:
                builtin_layers['not'] = negation
            contiguous = getattr(space, '_contiguous_layer', None)
            if contiguous is not None:
                builtin_layers['Contiguous'] = contiguous
            default_rule = 'union'
        else:
            default_rule = None
        layer = build_space_syntactic_layer(
            space, self, tier=tier,
            builtin_layers=builtin_layers,
            default_rule=default_rule)
        # Register the new layer's parameters with the WordSpace param
        # list so the optimizer scan sees the lazily-constructed
        # GrammarLayer instances. The space already owns the built-in
        # parametrized fold layers (PiLayer / SigmaLayer / NotLayer /
        # ContiguousLayer) in its own params list, so register only
        # the *new* lazy-constructed wrappers (their parameters won't
        # already be in self.params).
        for p in layer.parameters():
            if all(p is not q for q in self.params):
                self.params.append(p)
        return layer

    def _build_syntactic_layer(self, space, n_slots, grammar, symbol_dim):
        """Build the legacy SyntacticLayer stub anchored at the symbolic
        space.

        Post-2026-05-01 refactor: chart parsing lives on ``self.chart``
        (a ``Chart`` instance); per-rule math lives on
        ``GRAMMAR_LAYER_CLASSES`` GrammarLayer subclasses; per-space
        dispatch lives on ``SpaceSyntacticLayer``. This call still
        constructs a legacy SyntacticLayer because consumers
        (`Models._universality_score` reads `last_svo` / `lifting_layer`
        off it) tolerate its presence as a passthrough -- those fields
        default to ``None`` and the consumers gate on that.
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
        self.attach_codebook_host(space)
        self.attach_layer('syntactic', layer)
        space.attach_wordSpace(self)
        return layer

    # -- composition dispatch ----------------------------------------
    def forwardSymbols(self, data, subspace):
        """Demux the muxed symbol tensor into the subspace's modality
        slots (Rule #2 axis commitment side effect).

        Post-2026-05-01 refactor: the actual symbolic composition runs
        on the chart (via ``ChartCompose`` in the pipeline + per-space
        ``SyntacticLayer.forward`` dispatch). This helper retains the
        demux side effect that downstream slot selectors depend on.
        """
        if data.ndim == 3 and data.shape[-1] == getattr(subspace, 'muxedSize', -1):
            subspace.demux(data)
        return data

    def reverseSymbols(self, data, subspace):
        """No-op pass-through: chart-driven generation handles the
        symbol-side reverse via ``ChartGenerate`` + per-space
        ``SyntacticLayer.reverse`` dispatch.
        """
        return data

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
