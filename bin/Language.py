

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
from util import ProjectPaths, compile, TheXMLConfig, init_config, init_compile_backend, autocast_compute_dtype
from embed import WordVectors, PretrainModel
from data import Data, TheData
from Layers import Layer, PiLayer, SigmaLayer  # Import custom layers from Model.py
from Layers import LinearLayer, InvertibleLinearLayer, AttentionLayer, AssociationLayer, MapppingLayer, ChunkLayer
from Layers import CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon, Ops
from Layers import SortingLayer, TruthLayer, LiftingLayer, InterSentenceLayer, SparsityRegLayer, SmoothingRegLayer, ImpenetrableLayer
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
    """Multi-tier grammar rule catalog (P / C / S / L).

    Tiers tag each rule with the space (or pseudo-space) that
    dispatches it:
      - ``P`` (perceptual) -- PerceptualSpace's SyntacticLayer.
      - ``C`` (conceptual) -- ConceptualSpace's SyntacticLayer.
        Bivector pre-codebook activation ``[B, V, 2]``.
      - ``S`` (symbolic)   -- SymbolicSpace's SyntacticLayer.
        Post-codebook activation: a scalar ``[B, V]`` per
        prototype. S-tier ops (``conjunction``, ``disjunction``,
        ``not``, ``lift``, ``lower``, ``part``, ``equals``,
        ``query``, ``true``, ``false``, ``swap``, ``non``) are
        monotonic functions on that scalar.
      - ``L`` (logical)    -- pure logical primitives
        (``intersection``, ``union``). 2026-05-05 directive: these
        are lattice min/max on bivector activation; not owned by
        any single space. The chart binds an L-tier op at whichever
        space the operands live in (``intersection(C, C)`` binds at
        C; ``intersection(S, S)`` would bind at S) -- the L tag is
        layer-side classification, not a routing tier.

    Owns the rule definitions parsed from XML config. All learnable
    parameters and rule execution live on a single unified
    ``SyntacticLayer`` instance owned by ``WordSpace``.
    """

    # lhs:          nonterminal this rule reduces to ('S', 'VO', 'NP', 'VP', ...).
    # rhs_symbols:  typed-form RHS category tuple (e.g. ('V', 'O') for VO -> V O).
    #               None for legacy function-call / epsilon / passthrough rules.
    # width_min, width_max: per-rule depth-band gates (Step 1 of the
    # 2026-05-04 perf plan).  When set (defaults: 0 = no minimum,
    # 0 = no maximum), the chart's _chart_inside skips this rule at
    # cells whose width falls outside [width_min, width_max].  Saves
    # per-cell rule-enumeration work for rules that are structurally
    # impossible at most widths (e.g., S = lift(NP, VP) only fires at
    # the root span; NP = N only at width 1).
    RuleDef = _namedtuple(
        'RuleDef',
        ['tier', 'canonical', 'arity', 'method_name', 'lhs', 'rhs_symbols',
         'width_min', 'width_max'],
    )
    RuleDef.__new__.__defaults__ = (0, 0)  # width_min=0, width_max=0 default

    def __init__(self):
        """Initialize an empty rule catalog; XML configuration happens lazily.

        Rules are populated on first access via ``_ensure_configured``.
        Holds tier-tagged rule lists, the upward / downward / reverse
        derivations, and the default start symbol ``"S"``.
        """
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
        """Total rule count after lazy configuration."""
        self._ensure_configured()
        return len(self.rules)

    def __getitem__(self, idx):
        """Return the canonical name string for rule ``idx``."""
        self._ensure_configured()
        return self.rules[idx].canonical

    def arity(self, rule_id):
        """Return the arity (1 or 2) of rule ``rule_id``."""
        return self.rules[rule_id].arity

    def method_name(self, rule_id):
        """Return the Python method name implementing rule ``rule_id``."""
        return self.rules[rule_id].method_name

    def tier(self, rule_id):
        """Return the tier tag ('P' / 'C' / 'S' / 'L') of rule ``rule_id``."""
        return self.rules[rule_id].tier

    def binary_rules(self):
        """Return the list of rule_ids that have arity 2."""
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
        # 'L' (logical): tier marker for ops that are pure logical
        # primitives (lattice min/max on bivector activation), not
        # owned by any single space. Per the 2026-05-05 directive,
        # ``intersection`` and ``union`` carry layer.tier='L' as
        # semantic metadata; the dispatcher still binds them at
        # whichever space's tier the grammar rule names (e.g.
        # ``intersection(C, C)`` binds at C, ``intersection(S, S)``
        # at S) -- the L tag is for classification, not for routing.
        'logical':  'L',
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
        """Parse ``<rule>`` entries from ``rules_dict`` and append to ``target``.

        Handles both the canonical ``<rule>head = body</rule>`` form
        (with optional ``width="MIN..MAX"`` gate) and the legacy
        ``<S>body</S>`` form. Each parsed rule is tagged with the
        supplied tier letter.
        """
        # New syntax: <rule>head = body</rule> — head may be a comma-
        # separated tuple of categories (for multi-output downward rules
        # like `S,S = intersection_inv(VO)`). Body is a function call
        # (`f(A, B)`), bare-symbol sequence (`A B`), or a single category
        # (`C` / `A`). Rules in this form arrive under the 'rule' key
        # because that's the XML element name used.
        # Optional attribute: width="MIN..MAX" gates the rule to cells
        # whose width falls in [MIN, MAX]. MIN/MAX may be 0 (no bound),
        # plain integers, or 'N' (means: equals chart's full input N --
        # signals "root span only" when both ends are 'N'). When the
        # XML element has attributes the parser delivers it as a dict
        # with '_' holding the text; bare strings have no width set.
        rule_entries = rules_dict.get('rule', None)
        if rule_entries is not None:
            if isinstance(rule_entries, str) or isinstance(rule_entries, dict):
                rule_entries = [rule_entries]
            for entry in rule_entries:
                if isinstance(entry, dict):
                    text = str(entry.get('_', '')).strip()
                    width_raw = entry.get('width', None)
                else:
                    text = str(entry)
                    width_raw = None
                if '=' not in text:
                    raise ValueError(
                        f"<rule> requires 'head = body' syntax, got: {text!r}")
                lhs_raw, body = text.split('=', 1)
                lhs = ','.join(p.strip() for p in lhs_raw.split(',') if p.strip())
                rule = self._parse_rule(lhs, body.strip(), tier=tier)
                # Apply width gate if specified.
                if width_raw is not None:
                    w_min, w_max = self._parse_width_attr(str(width_raw))
                    rule = rule._replace(
                        width_min=int(w_min), width_max=int(w_max))
                target.append(rule)

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

    @staticmethod
    def _parse_width_attr(text):
        """Parse a ``width="MIN..MAX"`` attribute into (min, max) ints.

        Accepted forms:
          ``"3..5"``     → (3, 5)
          ``"5"``        → (5, 5)  (single value: exact width)
          ``"3.."``      → (3, 0)  (no upper bound; 0 means open)
          ``"..5"``      → (0, 5)  (no lower bound)
          ``"N..N"``     → (-1, -1) (both equal full chart N; resolved
                                     to the actual N at runtime; sentinel
                                     -1 means 'use the live N from data.shape')
        Anything else falls back to (0, 0) = no gate.
        """
        s = str(text).strip()
        if not s:
            return (0, 0)
        if '..' in s:
            lo_s, hi_s = s.split('..', 1)
        else:
            lo_s = hi_s = s
        def _one(v):
            """Coerce one bound from the width attribute to an int sentinel.

            Empty / unparseable -> 0 (no bound). ``'N'`` -> -1 sentinel
            meaning "use the live N from data.shape". Else integer.
            """
            v = v.strip()
            if not v:
                return 0
            if v.upper() == 'N':
                return -1
            try:
                return int(v)
            except ValueError:
                return 0
        return (_one(lo_s), _one(hi_s))

    def _parse_rule(self, lhs, rhs, tier='S'):
        """Parse one ``lhs = rhs`` rule string into a ``RuleDef`` namedtuple.

        ``rhs`` can be a function call (``f(A, B)``), a bare-symbol
        sequence (``A B``), or a single category. Accepts the explicit-
        direction suffixes ``.forward`` / ``.reverse`` on the function
        name. ``tier`` is the per-rule routing tag.
        """
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
        """Lazily configure the grammar from XML on first use.

        Resolves the start symbol, looks up the inline XML grammar or
        a ``grammarCfg`` file path, and dispatches to ``configure`` or
        ``load_from_cfg``. Subsequent calls are no-ops via the
        ``_configured`` guard.
        """
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
        """Return rule_ids whose tier is 'S' (symbolic-tier rules)."""
        self._ensure_configured()
        return [i for i, r in enumerate(self.rules) if r.tier == 'S']

    def symbolic_transition(self):
        """Return rule_id of the unary tier-S transition rule, or None.

        Used by the symbolic head to find the unary-transition rule
        when the grammar exposes one (typically the no-op identity).
        """
        self._ensure_configured()
        for i, r in enumerate(self.rules):
            if r.tier == 'S' and r.method_name is None and r.arity == 1:
                return i
        return None

    @property
    def s_methods(self):
        """Set of method names available on the S (symbolic) tier.

        Excludes rules without a method_name (e.g. pure transitions).
        """
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
        wmin_l, wmax_l = [], []
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
            wmin_l.append(int(getattr(rule, 'width_min', 0) or 0))
            wmax_l.append(int(getattr(rule, 'width_max', 0) or 0))

        device = device or torch.device('cpu')
        if lhs_l:
            table = {
                'lhs':         torch.tensor(lhs_l, dtype=torch.long, device=device),
                'rhs_left':    torch.tensor(rl_l,  dtype=torch.long, device=device),
                'rhs_right':   torch.tensor(rr_l,  dtype=torch.long, device=device),
                'arity':       torch.tensor(ar_l,  dtype=torch.long, device=device),
                'marker_mask': torch.tensor(mm_l,  dtype=torch.bool, device=device),
                'global_id':   torch.tensor(gid_l, dtype=torch.long, device=device),
                'width_min':   torch.tensor(wmin_l, dtype=torch.long, device=device),
                'width_max':   torch.tensor(wmax_l, dtype=torch.long, device=device),
            }
        else:
            table = {
                'lhs':         torch.empty(0, dtype=torch.long, device=device),
                'rhs_left':    torch.empty(0, dtype=torch.long, device=device),
                'rhs_right':   torch.empty(0, dtype=torch.long, device=device),
                'arity':       torch.empty(0, dtype=torch.long, device=device),
                'marker_mask': torch.empty((0, 2), dtype=torch.bool, device=device),
                'global_id':   torch.empty(0, dtype=torch.long, device=device),
                'width_min':   torch.empty(0, dtype=torch.long, device=device),
                'width_max':   torch.empty(0, dtype=torch.long, device=device),
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
# SignalRouter -- inlined from bin/SignalRouter.py (2026-05-11 module
# consolidation). Selected via WordSpace.routerKind = 'signal' in XML.
# Replaces the Chart's soft-superposition CKY forest with per-layer
# COPY/REDUCE routing on the subspace tensor. Owned by Chart, lazily
# built via Chart._ensure_signal_router.
# =====================================================================
class _BinaryGrammarOpAdapter(nn.Module):
    """Adapt a GrammarLayer with a `.compose(left, right)` method into a
    plain binary callable for the SignalRouter's `BinaryStructuredReductionLayer`.

    The CKY chart calls `gl.compose(left, right)` on `[..., D]` pairs;
    `BinaryStructuredReductionLayer` calls `op(left, right)` on
    `[B, N-1, D]` pairs. The two contracts agree element-wise; this
    adapter just forwards.
    """

    def __init__(self, gl):
        """Wrap a GrammarLayer ``gl`` so it can act as a binary callable."""
        super().__init__()
        self.gl = gl

    def forward(self, left, right):
        """Forward ``(left, right)`` to the wrapped grammar layer's compose."""
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
        """Initialize empty unary / binary ModuleDicts; ops are attached later.

        ``feature_dim`` is the slab D; ``temperature`` divides logits in
        the inner soft DP. ``max_depth`` caps the number of binary
        reduction rounds; the actual cap is min(N-1, max_depth).
        """
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
        """Attach a unary tier; ops fire per-position with one selection.

        ``rule_ids`` parallels ``ops`` and maps local op_id to the
        grammar's global rule_id (defaults to identity range). Mutates
        ``self._unary_layers[tier]`` and ``self._unary_rule_ids[tier]``.
        """
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
        """Attach a binary tier; ops reduce adjacent pairs via Viterbi DP.

        ``rule_ids`` parallels ``ops`` and maps local op_id to grammar
        global rule_id. Mutates ``self._binary_layers[tier]`` and
        ``self._binary_rule_ids[tier]``.
        """
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
        """Run tiered unary then recursive binary reductions; return rule list.

        ``data`` is ``[B, N, D]``. For each tier in sorted order, unary
        fires per position then binary reduces adjacent pairs until N
        collapses to a single S-state. Returns ``{tier: list[list[rule_id]]}``
        and caches the root state + length-N expansion on ``self``.
        """
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
        """Reverse-pass mirror: emit the compose-order rule list reversed.

        If compose has not yet been called for ``target``, run it now.
        Tier order is reversed (innermost first) and each row's rule
        sequence is reversed so the inverse pass pops last-applied first.
        """
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
        """Rebuild per-row compose-order rule lists from cached routings.

        Walks ``self._last_tier_routings`` and translates each routing's
        ``(action_kind, action_op)`` tensors back into global rule_ids
        via the per-tier ``_unary_rule_ids`` / ``_binary_rule_ids`` tables.
        """
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
        """Extract per-row binary-op id lists from a single routing dict.

        Returns ``[B][L_b]`` lists of *local* op ids (no rule_id mapping),
        skipping non-reduce action kinds. Helper for diagnostics.
        """
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
        """Build the gate MLP (d_model -> hidden -> 4 branches).

        ``hidden`` defaults to ``d_model``. ``temperature`` divides
        gate logits before softmax; lower = harder routing.
        """
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
    """Default context net for the structured layers: pass-through.

    Used when no explicit contextualizer is supplied so the routing
    score sees raw ``x``.
    """
    def forward(self, x):
        """Return ``x`` unchanged."""
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
        """Wire ops list, per-rule anchor params, and the comparator mixer.

        Builds learnable ``copy_anchor`` ``[r_copy, D]`` and
        ``reduce_anchor`` ``[r_reduce, D]`` for anchor-based placement
        scoring (no separate scorer MLP). ``context_net`` defaults to
        identity if None.
        """
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
        """Build [B, N, 4, D] in branch order (keep, reduce, shift, pad).

        ``reduced_chosen`` is the per-pair reduction result, length N-1;
        it gets pad-extended to N at the last position so all four
        branches share an N-length axis for the comparator mixer.
        """
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
        """Score, route via Viterbi, compact; return (hard, soft, routing).

        Returns:
            hard_slab: [B, N, D] argmax-selected per-position action.
            soft_slab: [B, N, D] DP-marginal-weighted blend (gradient surrogate).
            routing:   dict with masks, scores, marginals, lengths, gates.
        Degenerate N<=1 returns the input twice with a stub routing.
        """
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
        """Wire ops list and per-(copy/apply) anchor params.

        Action space is ``R_copy + R_apply`` choices per position with
        anchor-based scoring; positions are independent (no DP). Builds
        ``copy_anchor`` ``[r_copy, D]`` and ``apply_anchor`` ``[r_apply, D]``.
        """
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
        """Score, choose per-position action, return (hard, soft, routing).

        Hard slab argmax-selects one branch per position; soft slab is
        the softmax-weighted blend over (copy_branch + applied_ops).
        Straight-through gradient connects the hard-forward / soft-backward.
        """
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
    """Penalty proportional to mean copy_marginal across route traces.

    Encourages the router to commit to non-copy actions (apply / reduce)
    rather than passing positions through unchanged. Returns scalar 0
    if ``lambda_copy`` is zero or no trace carries a copy_marginal.
    """
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
    """Penalty proportional to mean post-reduction length across traces.

    Pressures the router toward shorter derivations (more reduces).
    Returns scalar 0 if ``lambda_len`` is zero or no trace carries a
    lengths tensor.
    """
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

# -- End inlined SignalRouter section -------------------------------



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
                 hidden_dim=256, D_rule=32, chart_tau=None, w_max=None,
                 unary_max_depth=2, feature_dim=None,
                 router_kind=None):
        """Build the CKY chart parser owned by WordSpace.

        Resolves ``w_max`` (span width bound) and ``chart_tau`` from
        XML when not explicitly set. ``router_kind`` selects between
        the legacy CKY chart and the SignalRouter alternative.
        Allocates rule-embedding params and unary / binary scoring nets.
        """
        super().__init__()
        nOutput = nOutput if nOutput is not None else nInput
        self.nInput = int(nInput)
        self.nOutput = int(nOutput)
        self.max_depth = int(max_depth)
        self.hidden_dim = int(hidden_dim)
        self.D_rule = int(D_rule)
        # w_max bounds CKY span widths (and Viterbi recursion depth) so
        # the inside pass is O(B * N * w_max^2 * R) rather than O(B * N^3 * R).
        # XML override: <WordSpace><wMax>N</wMax></WordSpace>. The default
        # is 8 (preserves the legacy perf envelope of MM_5M-style configs
        # where N=1024 byte tokens × full N would be O(B*N^3*R) ≈ 2 GFLOPS
        # per row). Configs that need to parse full sentences (where the
        # chart's contribution actually matters) should set <wMax> to N
        # explicitly. The chart logs a one-time warning when it's asked
        # to extract a derivation for a span wider than w_max so silent
        # truncation is visible.
        if w_max is None:
            try:
                w_max_xml = TheXMLConfig.get("WordSpace.wMax", 0)
                w_max = int(w_max_xml) if int(w_max_xml) > 0 else 8
            except Exception:
                w_max = 8
        self.w_max = int(w_max) if int(w_max) > 0 else 8
        self._w_max_warned = False
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

        # Sparse-MoE-style rule gating (Shazeer et al. 2017).
        #   chartTopK      -- per (cell, split) keep only the top-K rules
        #                     by (score + Gaussian noise); zero the rest.
        #                     0 (default) disables; uses all R rules.
        #   chartNoiseEps  -- Gaussian noise scale on per-rule scores so
        #                     low-probability rules occasionally enter
        #                     the top-K and receive gradient.
        # Tracking for the load-balance loss is owned by Chart and read
        # by Models.ModelLoss when loadBalanceWeight > 0.
        try:
            self.chart_top_k = int(TheXMLConfig.get(
                "WordSpace.chartTopK", 0) or 0)
        except Exception:
            self.chart_top_k = 0
        try:
            self.chart_noise_eps = float(TheXMLConfig.get(
                "WordSpace.chartNoiseEps", 0.0) or 0.0)
        except Exception:
            self.chart_noise_eps = 0.0
        # Per-word stem mode. When true, the stem runs each word
        # through an individual P->C->S->C round trip and pushes
        # ideas onto ConceptualSpace.stm; the chart fires at C
        # over the STM buffer in the body.
        #
        # Default False (opt-in): the path is incompatible with
        # butterfly mode (``<useButterflies>true``) because the
        # per-word loop slices the perceptual event to N=1 slots
        # and the butterfly pack requires even N. Once butterfly
        # compatibility is added (either by pair-slicing or by
        # routing per-word through a non-butterfly path), the
        # default can flip to True and the legacy chart-at-stem
        # launch site (``Models._chart_compose``) can retire.
        try:
            self.per_word_stem = bool(TheXMLConfig.get(
                "WordSpace.perWordStem", False))
        except Exception:
            self.per_word_stem = False
        try:
            self.iterations_per_word = int(TheXMLConfig.get(
                "WordSpace.iterationsPerWord", 1) or 1)
        except Exception:
            self.iterations_per_word = 1
        # Load-balance state: per-rule activation count from the most
        # recent inside pass. Tensor [R_bin] of int counts; rebuilt each
        # _chart_inside call when chart_top_k > 0. None when sparse
        # gating is disabled.
        self._rule_load_count = None

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
        # Grammar reference for chart-authority rule gating. Populated
        # by ``WordSpace.__init__`` after the chart is built. The
        # ``register_grammar_layer`` / ``should_run_rule`` pair below
        # services GrammarLayer.gated_run via the
        # ``GrammarLayer._chart_authority`` slot.
        self.grammar = None
        self._registered_grammar_layers = []

    def register_grammar_layer(self, layer):
        """Add a GrammarLayer instance to this chart's roster.
        Idempotent on repeated registration."""
        if layer not in self._registered_grammar_layers:
            self._registered_grammar_layers.append(layer)

    def should_run_rule(self, rule_name):
        """Return the firing probability for ``rule_name`` per the
        grammar's ``rule_probability`` lookup.

        Synthesizes a body string matching ``Grammar.rule_probability``
        prefix-checks (``"<name>("``) so dormant defaults (e.g.
        ``not(...) -> 0.0``) still apply. Used by
        ``GrammarLayer.gated_run`` to gate parameterized folds. Returns
        1.0 when no grammar is wired.
        """
        if self.grammar is None or not rule_name:
            return 1.0
        body = f"{rule_name}(S)"
        try:
            return float(self.grammar.rule_probability(body))
        except Exception:
            return 1.0

    def _ensure_signal_router(self):
        """Lazy-build the SignalRouter when router_kind == 'signal'.

        Assigning an nn.Module to an attribute auto-registers it as a
        submodule, so it is included in parameters() / state_dict().
        """
        if self._signal_router is None:
            try:
                temperature = float(TheXMLConfig.get(
                    "WordSpace.signal.temperature", 1.0))
            except Exception:
                temperature = 1.0
            self._signal_router = SignalRouter(
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
        """Lazily build the chart's category-name ordering and reverse index.

        Walks every rule's lhs and rhs symbols, dedupes, prepends a
        ``'?'`` placeholder at index 0, and stores both the ordered
        list and a name -> index map for fast lookups.
        """
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
        """Lazily allocate the soft-chart scoring modules for the given D.

        Rebuilds when ``D`` / ``D_rule`` change or when the grammar's
        ``rule_table_version`` increments (rule catalog edit). Allocates
        the compat-score MLP, unary-compat module, and per-rule
        embedding / bias tensors on ``device``.
        """
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

    # Fixed tier classification (overrides per-rule authored ``tier``
    # attributes; 2026-05-12). Drives ``_tier_for_method`` and the
    # parameter-free fallback's substrate wiring.
    #   P (subsymbolic) — fires through P.sigma / C.pi via the
    #     LiftLayer / LowerLayer codebook-gate mechanism.
    #   C (local)       — runs on the C-tier idea tensors directly;
    #     no codebook lookup needed.
    #   S (symbolic)    — needs codebook lookup (parthood / equality
    #     / query) for its semantics.
    _RULE_TIER = {
        # P-tier: subsymbolic gate through substrate sigma / pi.
        'lift':         'P',
        'lower':        'P',
        # C-tier: local ops over the operand tensors.
        'union':        'C',
        'intersection': 'C',
        'conjunction':  'C',
        'disjunction':  'C',
        'not':          'C',
        'non':          'C',
        'swap':         'C',
        'copy':         'C',
        'true':         'C',
        'false':        'C',
        # S-tier: codebook-lookup-dependent.
        'query':        'S',
        'equals':       'S',
        'part':         'S',
    }

    def _apply_rule_forward(self, method_name, left, right, marker_mask,
                            subspace=None):
        """Dispatch rule's forward semantics. Marker operands are zeroed
        before the rule fires so a sugar operand contributes nothing.

        Dispatch order:
          1. host_layer dispatch via the active wordSpace's registry —
             fires the host space's parametrized GrammarLayer (e.g.
             PiLayer-backed IntersectionLayer).
          2. external rule_executor (test injection point, optional).
          3. typed-GrammarLayer fallback. Constructs the layer with
             substrate refs when it accepts them (lift / lower) so the
             substrate sigma / pi fires instead of the static lattice
             kernel. Local C-tier ops (union, intersection, ...) and
             S-tier ops (query, equals, part) instantiate without
             refs — they don't need them.

        Errors are surfaced (logged + raised), not swallowed. The
        previous bare ``except Exception: return l_eff`` silently
        no-op'd every dispatch failure, hiding wiring bugs.
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
        # 3. Typed-GrammarLayer fallback with substrate wiring.
        from Layers import GRAMMAR_LAYER_CLASSES
        cls = GRAMMAR_LAYER_CLASSES.get(method_name)
        if cls is None:
            # The rule fired but no GrammarLayer class exists for it.
            # Surface as a warning; return the left operand as a
            # degraded continuation so the chart can still complete.
            warnings.warn(
                f"_apply_rule_forward: no GrammarLayer for "
                f"method_name={method_name!r}; returning left operand.",
                stacklevel=2)
            return l_eff
        inst = self._instantiate_grammar_layer(method_name, cls, ws)
        arity = getattr(inst, 'arity', 1)
        if arity == 2 and hasattr(inst, 'compose'):
            return inst.compose(l_eff, r_eff)
        return inst.forward(l_eff)

    def _instantiate_grammar_layer(self, method_name, cls, ws):
        """Construct a GrammarLayer instance with appropriate substrate
        refs for its tier. Returns a fresh instance per call.

        P-tier (lift / lower) requires the substrate Spaces so the
        chart's gate fires through ``P.sigma`` / ``C.pi`` instead of
        the static lattice fallback. C-tier and S-tier ops are
        parameter-free.
        """
        tier = self._RULE_TIER.get(method_name)
        if tier == 'P' and ws is not None:
            if method_name == 'lift':
                return cls(symbolicSpace=getattr(ws, 'symbolicSpace', None),
                           perceptualSpace=getattr(ws, 'perceptualSpace', None))
            if method_name == 'lower':
                return cls(symbolicSpace=getattr(ws, 'symbolicSpace', None),
                           conceptualSpace=getattr(ws, 'conceptualSpace', None))
        # C / S tier and standalone (no wordSpace) callers: parameter-free.
        return cls()

    def _tier_for_method(self, method_name):
        """Return the host tier ('P' / 'C' / 'S') for ``method_name``.

        Uses the fixed classification (``_RULE_TIER``) so the tier of
        each operator is stable across grammars and doesn't depend on
        the authored ``tier`` attribute on individual rules.
        """
        if not method_name:
            return None
        return self._RULE_TIER.get(method_name)

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
        # Lazy-allocate the learned per-atom POS buffer once C is
        # known. Idempotent for the same C; no-op when the codebook
        # doesn't support category tagging.
        try:
            if hasattr(what, 'ensure_category_logits'):
                what.ensure_category_logits(C, device=W.device)
        except Exception:
            pass
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
        thresh = 0.1
        nearest_valid = best_sim > thresh
        # Stash the (best_idx, mask) so Chart.compose can run an EMA
        # update on category_logits after the inside pass settles.
        # Cleared at the next call.
        try:
            self._last_seed_atom_idx = best_idx.detach().clone()
            self._last_seed_atom_valid = nearest_valid.detach().clone()
        except Exception:
            self._last_seed_atom_idx = None
            self._last_seed_atom_valid = None

        flat_log = lex_log_probs.reshape(-1, C).clone()

        # Tier 1a: hard one-hot override from durable category_ids
        # (closed-class function-word seeds set externally; usually
        # absent in current configs but supported).
        valid_hard = (best_cat > 0) & nearest_valid
        if bool(valid_hard.any()):
            cat_clamped = best_cat.clamp(min=0, max=C - 1)
            NEG = torch.full_like(flat_log, -1e9)
            rows = torch.arange(flat_log.shape[0], device=device)
            override = NEG.clone()
            override[rows, cat_clamped] = 0.0
            flat_log = torch.where(
                valid_hard.unsqueeze(-1).expand_as(flat_log),
                override, flat_log)

        # Tier 1b: soft prior from learned category_logits[V, C].
        # Adds the per-atom learned log-distribution to the lex
        # scorer's log-probs at positions whose nearest codebook
        # match is confident. This is the "learn POS through parsing"
        # path: chart updates category_logits via EMA after Viterbi,
        # next forward sees the prior. No-op until the buffer has
        # accumulated nontrivial mass (any nonzero entry).
        cat_logits = getattr(what, 'category_logits', None)
        if (cat_logits is not None
                and torch.is_tensor(cat_logits)
                and cat_logits.shape[1] == C
                and bool(nearest_valid.any())
                and bool(cat_logits.abs().sum() > 0)):
            cl = cat_logits.to(device=device, dtype=flat_log.dtype)
            # log-softmax so the row is a normalised log-distribution
            # before we add it as a prior.
            cl_logp = F.log_softmax(cl, dim=-1)
            atom_logp = cl_logp.index_select(
                0, best_idx.clamp(min=0, max=cl.shape[0] - 1))
            # Soft additive prior on positions with a confident nearest
            # match; positions without one keep the bare lex scorer.
            prior_scale = 1.0
            additive = torch.where(
                nearest_valid.unsqueeze(-1).expand_as(atom_logp),
                prior_scale * atom_logp,
                torch.zeros_like(atom_logp))
            flat_log = flat_log + additive
            # Re-normalise so log-probs stay a valid log-distribution.
            flat_log = F.log_softmax(flat_log, dim=-1)

        return flat_log.reshape(B, N, C)

    def _update_codebook_pos_logits_from_chart(
            self, chart_score, B, N, word_space, subspace):
        """EMA-update the symbolic codebook's category_logits[V, C]
        using the chart's per-leaf POS distribution.

        The seed atom indices were stashed by `_apply_codebook_pos_seed`
        as ``self._last_seed_atom_idx`` (shape [B*N], codebook row of
        the nearest match per token position) and
        ``self._last_seed_atom_valid`` (bool [B*N], True iff the dot
        product passed the confidence threshold). For each valid leaf,
        we take the chart's per-cell POS softmax at (i, i+1) and EMA
        it into category_logits[atom_idx].

        No-op when the codebook has no category_logits buffer or when
        no positions passed the seed threshold. Runs without gradient.
        """
        if word_space is None:
            return
        sym_space = getattr(word_space, 'symbolicSpace', None)
        if sym_space is None:
            return
        sym_subspace = getattr(sym_space, 'subspace', None)
        candidate = subspace if subspace is not None else sym_subspace
        if candidate is None:
            return
        what = getattr(candidate, 'what', None)
        if what is None or not hasattr(what, 'update_category_logits'):
            return
        cat_logits = getattr(what, 'category_logits', None)
        if cat_logits is None:
            return
        atom_idx = getattr(self, '_last_seed_atom_idx', None)
        atom_valid = getattr(self, '_last_seed_atom_valid', None)
        if (atom_idx is None or atom_valid is None
                or chart_score is None or N <= 0):
            return
        if int(atom_valid.sum().item()) == 0:
            return
        # chart_pos at the leaves: [B, N, C], softmax over C.
        diag = torch.arange(N, device=chart_score.device)
        leaf_logits = chart_score[:, diag, diag + 1, :]  # [B, N, C]
        leaf_pos = F.softmax(leaf_logits, dim=-1)
        flat_pos = leaf_pos.reshape(B * N, -1)
        # Restrict to seed-valid positions and update those atoms.
        idx_flat = atom_idx.reshape(-1)
        valid_flat = atom_valid.reshape(-1).to(torch.bool)
        if idx_flat.shape[0] != flat_pos.shape[0]:
            return
        valid_rows = torch.nonzero(valid_flat, as_tuple=False).squeeze(-1)
        if valid_rows.numel() == 0:
            return
        sel_atoms = idx_flat.index_select(0, valid_rows)
        sel_pos = flat_pos.index_select(0, valid_rows)
        what.update_category_logits(sel_atoms, sel_pos, ema=0.05)

    def load_balance_loss(self, weight=1.0):
        """Coefficient-of-variation² penalty over per-rule activation
        counts, scaled by ``weight``.  Encourages the noisy top-K
        gating (Shazeer et al. 2017) to spread mass across rules
        instead of collapsing onto a few.

        Returns 0.0 (Python float, not tensor) when sparse gating is
        disabled or no inside pass has run yet.  Tensor when active.
        Caller adds it to the model's training loss.
        """
        load = getattr(self, '_rule_load_count', None)
        if load is None or load.numel() == 0 or float(weight) <= 0.0:
            return 0.0
        load_f = load.to(torch.float32)
        mean = load_f.mean()
        if float(mean.item()) <= 0.0:
            return 0.0
        cv2 = ((load_f - mean) ** 2).mean() / (mean ** 2 + 1e-8)
        return cv2 * float(weight)

    def reset_load_count(self):
        """Clear per-rule activation counts.  Call between batches so
        the load-balance loss reflects only the current batch's
        gating distribution."""
        self._rule_load_count = None

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
        # Allocate chart buffers in autocast's compute dtype when active,
        # so in-place writes from autocast-promoted ops (lex_cat_scorer,
        # rule_embed @ data, etc.) don't trip the dtype-mismatch error
        # at chart_score[...] = lex_log_probs.  Falls back to data.dtype
        # when autocast is off.
        dtype = autocast_compute_dtype(device, fallback=data.dtype)

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
        chart_score[:, i_diag, i_diag + 1, :] = lex_log_probs.to(dtype)
        chart_vec[:, i_diag, i_diag + 1, :, :] = (
            data.unsqueeze(2).expand(B, N, C, D).to(dtype))

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

            # Per-rule width gating: zero out rules whose declared
            # width-band [width_min, width_max] excludes the current
            # cell width w.  width_min/width_max == 0 mean "no bound";
            # width_min/width_max == -1 (sentinel from <rule width="N..">)
            # mean "= live N", resolved against the data's N.
            wmin_t = table['width_min'][bin_idx]
            wmax_t = table['width_max'][bin_idx]
            if int(wmin_t.abs().sum().item()) > 0 or int(wmax_t.abs().sum().item()) > 0:
                # Resolve -1 sentinels against the live N.
                wmin_eff = torch.where(wmin_t < 0,
                    torch.full_like(wmin_t, int(N)), wmin_t)
                wmax_eff = torch.where(wmax_t < 0,
                    torch.full_like(wmax_t, int(N)), wmax_t)
                # 0 means "no bound": replace with sentinels that always pass.
                wmin_eff = torch.where(wmin_eff == 0,
                    torch.zeros_like(wmin_eff), wmin_eff)
                wmax_eff = torch.where(wmax_eff == 0,
                    torch.full_like(wmax_eff, int(10**9)), wmax_eff)
                # Mask: True where this rule is allowed at width w.
                allowed = (wmin_eff <= int(w)) & (int(w) <= wmax_eff)  # [R_bin]
                if not bool(allowed.all().item()):
                    block = (~allowed).view(1, 1, 1, R_bin).expand_as(cand_score)
                    cand_score = cand_score.masked_fill(block, NEG_INF)

            # Noisy top-K rule gating (Shazeer et al. 2017,
            # "Outrageously Large Neural Networks").  When
            # chart_top_k > 0, add Gaussian noise to per-rule scores
            # at each (cell, split) position, then mask all rules
            # outside the top-K with NEG_INF.  The downstream softmax
            # / logsumexp paths see only the kept rules (others have
            # weight 0), which cuts effective per-cell rule-execution
            # cost from R_bin to chart_top_k while keeping rare-rule
            # gradients alive via the noise.  Disabled at eval (no
            # exploration during Viterbi).
            if (self.chart_top_k > 0
                    and self.chart_top_k < R_bin
                    and self.training
                    and not hard):
                noisy = cand_score
                if self.chart_noise_eps > 0.0:
                    noisy = noisy + torch.randn_like(noisy) * float(
                        self.chart_noise_eps)
                # top-K over the rule axis at each (B, P, Sp).
                _, top_idx = noisy.topk(
                    int(self.chart_top_k), dim=-1)
                keep_mask = torch.zeros_like(noisy)
                keep_mask.scatter_(-1, top_idx, 1.0)
                cand_score = cand_score.masked_fill(
                    keep_mask == 0, NEG_INF)
                # Update per-rule load count (used by load-balance loss).
                # Sum across all (B, P, Sp) cells -- how many candidate
                # slots picked each rule. Detached: this is bookkeeping,
                # not a differentiable signal here.
                with torch.no_grad():
                    incr = keep_mask.sum(dim=(0, 1, 2)).long().detach()
                if (self._rule_load_count is None
                        or self._rule_load_count.numel() != R_bin):
                    self._rule_load_count = incr.to(torch.long)
                else:
                    self._rule_load_count = (
                        self._rule_load_count.to(incr.device) + incr)

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
            # Casts: torch.logsumexp may upcast to fp32 even when inputs
            # are bf16 (autocast bf16 keeps reductions in fp32 for
            # numerical stability), so the resulting combined_* tensors
            # may not match chart_*'s dtype. Cast on the scatter.
            chart_score[:, i_range, i_range + w, :] = combined_score.to(
                chart_score.dtype)
            chart_vec[:, i_range, i_range + w, :, :] = combined_vec.to(
                chart_vec.dtype)

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
                    chart_score[:, i_range, i_range + w, :] = combined_u.to(
                        chart_score.dtype)
                    chart_vec[:, i_range, i_range + w, :, :] = combined_vec_u.to(
                        chart_vec.dtype)

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

        # POS learning hook: EMA-update the symbolic codebook's
        # category_logits[V, C] from the chart's per-leaf POS distribution.
        # Each leaf cell (i, i+1) carries a softmax over C categories;
        # we attribute that distribution to the codebook atom whose
        # vector is the nearest match for that leaf's input embedding
        # (the same `best_idx` computed in _apply_codebook_pos_seed).
        # Disabled in training mode: the soft-mode chart's per-cell
        # mixtures are ambiguous; EMA writes only run during eval/Viterbi
        # so the seed reflects definite parses, matching the user's
        # "store only definite assertions" rule.
        try:
            if (not hard) and self.training:
                pass  # soft mode in training: don't write seeds.
            else:
                self._update_codebook_pos_logits_from_chart(
                    chart_score, B, N, word_space, subspace)
        except Exception:
            # POS learning is advisory; never let it break the forward.
            pass

        # POS side-channel — Mechanism 3 (rule-prediction conditioning).
        # Walk the parsed trace and push each merge's LHS category
        # embedding onto WordSpace.category_stack so subsequent
        # WordSpace.predict_rule(b) calls see the parse history.
        # When no wordSpace / category_stack is wired, this is a no-op.
        self._populate_category_stack(word_space, B)

        # SVO extraction for the universality (Golden Rule) test.
        # Walks the derivation trace looking for ``S = lift(NP, VP)``
        # over ``VP = intersection(V, O)`` and stashes the operand
        # tensors on ``self.last_svo``. ``None`` when the trace
        # doesn't contain that signature.
        self.extract_svo()

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
        """Outside pass: compute outside scores / vectors over the chart.

        Mirrors the inside pass but drives top-down from the start
        symbol. ``hard`` selects between Viterbi (eval) and soft-DP
        (training). Returns ``(outside_score, outside_vec)`` shaped
        ``[B, N+1, N+1, C]`` and ``[B, N+1, N+1, C, D]``.
        """
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
                # Casts on every scatter: logsumexp / arithmetic on bf16
                # operands often returns fp32 under autocast, so the dst
                # dtype mismatch trips index_put_. Funnel everything
                # through outside_*.dtype for safety.
                if hard:
                    pick_new = (add_l_score >= old_l_score)
                    comb_l = torch.where(pick_new, add_l_score, old_l_score)
                    outside_score[:, i_flat, k_flat, Br] = comb_l.to(
                        outside_score.dtype)
                    outside_vec[:, i_flat, k_flat, Br, :] = torch.where(
                        pick_new.unsqueeze(-1), add_l_vec, old_l_vec).to(
                        outside_vec.dtype)
                else:
                    stk_l = torch.stack([old_l_score, add_l_score], dim=0)
                    comb_l = torch.logsumexp(stk_l, dim=0)
                    w_old_l = (old_l_score - comb_l).exp()
                    w_new_l = (add_l_score - comb_l).exp()
                    outside_score[:, i_flat, k_flat, Br] = comb_l.to(
                        outside_score.dtype)
                    outside_vec[:, i_flat, k_flat, Br, :] = (
                        w_old_l.unsqueeze(-1) * old_l_vec
                        + w_new_l.unsqueeze(-1) * add_l_vec).to(
                        outside_vec.dtype)
                old_r_score = outside_score[:, k_flat, j_flat, Cr]
                old_r_vec = outside_vec[:, k_flat, j_flat, Cr, :]
                add_r_score = push_right_score[:, :, r].view(B, P * Sp)
                add_r_vec = parent_vec_per_rule[:, :, r, :].view(
                    B, P * Sp, D)
                if hard:
                    pick_new = (add_r_score >= old_r_score)
                    comb_r = torch.where(pick_new, add_r_score, old_r_score)
                    outside_score[:, k_flat, j_flat, Cr] = comb_r.to(
                        outside_score.dtype)
                    outside_vec[:, k_flat, j_flat, Cr, :] = torch.where(
                        pick_new.unsqueeze(-1), add_r_vec, old_r_vec).to(
                        outside_vec.dtype)
                else:
                    stk_r = torch.stack([old_r_score, add_r_score], dim=0)
                    comb_r = torch.logsumexp(stk_r, dim=0)
                    w_old_r = (old_r_score - comb_r).exp()
                    w_new_r = (add_r_score - comb_r).exp()
                    outside_score[:, k_flat, j_flat, Cr] = comb_r.to(
                        outside_score.dtype)
                    outside_vec[:, k_flat, j_flat, Cr, :] = (
                        w_old_r.unsqueeze(-1) * old_r_vec
                        + w_new_r.unsqueeze(-1) * add_r_vec).to(
                        outside_vec.dtype)

        return outside_score, outside_vec

    # ------------------------------------------------------------------
    # Sentence-completion signaling.
    # ------------------------------------------------------------------
    def _signal_sentence_completed_chart(self, subspace, chart_score,
                                         word_space):
        """Set per-row sentence-completed flags when the root reaches start_symbol.

        Reads the root cell's argmax category for every row and sets
        ``word_space._sentence_completed[b_source]`` when it equals the
        start symbol's category id. Handles the K microbatch axis by
        mapping each flat row b_flat back to its source row b_source.
        """
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
        """Walk the Viterbi backtrace and emit one derivation trace per row.

        Returns ``[B]`` lists of ``(rule_id, i, k, j, cat_id)`` tuples
        in DFS pre-order. Logs one warning when the root span exceeds
        ``w_max`` (the silent-truncation failure mode).
        """
        device = chart_score.device
        table = TheGrammar._ensure_packed_table(device=device)
        R = int(table['lhs'].numel())
        if R == 0 or N == 0:
            return [[] for _ in range(B)]
        # One-time warning when the root span exceeds w_max -- this is
        # the silent-truncation failure mode (chart can't reach the
        # full-sentence root, returns empty trace, looks like the chart
        # didn't fire). Set <wMax> in XML to N to fix.
        if N > self.w_max and not getattr(self, '_w_max_warned', False):
            import logging
            logging.getLogger(__name__).warning(
                "Chart.w_max=%d but input N=%d; root span (0, %d) exceeds "
                "w_max so derivation traces will be empty. Set "
                "<WordSpace><wMax>%d</wMax></WordSpace> to fix (cost is "
                "O(N * w_max^2 * R) per row).",
                self.w_max, N, N, N)
            self._w_max_warned = True
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

    # ------------------------------------------------------------------
    # SVO extraction for the universality (Golden Rule) test.
    # ------------------------------------------------------------------
    def extract_svo(self):
        """Walk the Viterbi trace; find a top-level
        ``S = lift(NP, VP)`` and the matching
        ``VP = intersection(V, O)``; populate ``self.last_svo``
        with the (subject, verb, object) operand tensors.

        Sets ``self.last_svo`` to a 3-tuple ``(subject, verb, obj)``
        each ``[B, 1, D]``, or ``None`` when no row's derivation
        contains the SVO signature. Per-row miss is filled with the
        zero vector so the tensor stays full-shape; callers that
        need a per-row mask should consult ``self._svo_row_mask``
        (``[B]`` bool, True iff that row had a valid extraction).

        The 2026-05-05 grammar rewrite makes the object an explicit
        ``O`` nonterminal projected from ``NP`` (rather than the
        positional ``V NP`` merge), so the chart's parent rule_id
        at the VP cell directly identifies which child is the
        object -- no positional heuristic needed.
        """
        traces = self._derivation_trace
        chart_vec = self._chart_vec
        cat_idx = self._category_index or {}
        if traces is None or chart_vec is None or not cat_idx:
            self.last_svo = None
            self._svo_row_mask = None
            return None

        np_id = cat_idx.get('NP')
        vp_id = cat_idx.get('VP')
        v_id  = cat_idx.get('V')
        o_id  = cat_idx.get('O')
        if None in (np_id, vp_id, v_id, o_id):
            self.last_svo = None
            self._svo_row_mask = None
            return None

        rules = TheGrammar.rules

        def _matches(gid, lhs, method, rhs):
            """True iff rule ``gid`` has the given lhs/method/rhs signature.

            Used by ``extract_svo`` to locate the canonical
            ``S = lift(NP, VP)`` / ``VP = intersection(V, O)`` rules in
            the derivation trace.
            """
            try:
                r = rules[gid]
            except (IndexError, AttributeError):
                return False
            if str(r.lhs).strip() != lhs:
                return False
            if r.method_name != method:
                return False
            return tuple(r.rhs_symbols or ()) == rhs

        B = chart_vec.shape[0]
        D = chart_vec.shape[-1]
        device = chart_vec.device
        dtype = chart_vec.dtype
        subj = torch.zeros(B, 1, D, device=device, dtype=dtype)
        verb = torch.zeros(B, 1, D, device=device, dtype=dtype)
        obj  = torch.zeros(B, 1, D, device=device, dtype=dtype)
        row_mask = torch.zeros(B, dtype=torch.bool, device=device)

        for b in range(B):
            trace = traces[b] or []
            # Index by (i, j, A) for VP-cell lookup.
            trace_idx = {(int(e[1]), int(e[3]), int(e[4])): e for e in trace}
            s_entry = next(
                (e for e in trace
                 if _matches(int(e[0]), 'S', 'lift', ('NP', 'VP'))),
                None)
            if s_entry is None:
                continue
            i, k, j = int(s_entry[1]), int(s_entry[2]), int(s_entry[3])
            vp_entry = trace_idx.get((k, j, vp_id))
            if vp_entry is None:
                continue
            if not _matches(int(vp_entry[0]), 'VP', 'intersection',
                            ('V', 'O')):
                continue
            m = int(vp_entry[2])
            subj[b, 0] = chart_vec[b, i, k, np_id]
            verb[b, 0] = chart_vec[b, k, m, v_id]
            obj[b,  0] = chart_vec[b, m, j, o_id]
            row_mask[b] = True

        if not row_mask.any().item():
            self.last_svo = None
            self._svo_row_mask = row_mask
            return None
        self.last_svo = (subj, verb, obj)
        self._svo_row_mask = row_mask
        return self.last_svo


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
        """Build the 2-layer compat MLP and init weights via Xavier normal.

        ``hidden`` defaults to ``max(D, 64)``. ``compat_scale`` bounds
        the output magnitude so the per-cell softmax cannot saturate.
        """
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
        """Return a bounded compat score for ``(left, right, rule)`` merges.

        Zeros out marker-mask slots in left / right, concatenates with
        the rule embed, runs through the 2-layer MLP, and returns
        ``compat_scale * tanh(raw)``.
        """
        keep = (~marker_mask).to(left.dtype)
        kL = keep[..., 0:1]
        kR = keep[..., 1:2]
        x = torch.cat([left * kL, right * kR, rule_embed], dim=-1)
        raw = self.lin2(self.act(self.lin1(x))).squeeze(-1)
        return self.compat_scale * torch.tanh(raw)


class _UnaryCompat(nn.Module):
    """Score for a unary closure step. Bounded the same way as
    ``_CompatScore``.

    Used by the chart's unary closure pass to score ``(child, rule)``
    pairs. Two-layer MLP with tanh-bounded output.
    """

    def __init__(self, D, D_rule, hidden=None, compat_scale=2.0):
        """Build the 2-layer unary MLP and init weights via Xavier normal."""
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
        """Return ``compat_scale * tanh(MLP([child, rule_embed]))``."""
        x = torch.cat([child, rule_embed], dim=-1)
        raw = self.lin2(self.act(self.lin1(x))).squeeze(-1)
        return self.compat_scale * torch.tanh(raw)


# =====================================================================
# Per-space SyntacticLayer (2026-05-01 refactor; legacy class retired
# 2026-05-08).
#
# Spec: doc/specs/2026-05-01-syntactic-layer-refactor.md §4.
#
# Each PerceptualSpace / ConceptualSpace / SymbolicSpace owns one of
# these. Holds the parametrized GrammarLayer instances for its tier's
# rules and dispatches `forward` / `reverse` based on the rule choice
# the chart wrote into ``word_space.current_rules`` /
# ``generate_rules`` (Q4 / Q10.1).
# =====================================================================
class SyntacticLayer(Layer):
    """Per-space dispatcher.

    Construction:
        SyntacticLayer(tier='C', word_space=word_space,
                            host_layers={'pi': pi_layer},
                            host_space=concept_space)

    Each entry in ``host_layers`` is registered with ``word_space`` at
    construction. The space's ``forward()`` and ``reverse()`` delegate
    here; ``forward()`` reads ``word_space.current_rules[tier]``,
    advances a per-tier cursor, and dispatches to the appropriate
    layer's ``compose`` (binary) or ``forward`` (unary). ``reverse()``
    mirrors via ``word_space.generate_rules[tier]`` and ``layer.generate``.

    The cursor resets at the start of each new ``word_space.compose()``
    / ``word_space.generate()`` call via the generation counters on
    WordSpace (Q10.1).

    Per the 2026-05-07 rollback: there is no ``default_rule`` parameter.
    The grammar XML drives which rules fire — when the chart hasn't
    populated rules for this tier the dispatch is a no-op.
    """

    def __init__(self, tier, word_space, host_layers, host_space=None):
        """Register host layers with the WordSpace dispatch table.

        ``host_layers`` is a name -> Layer mapping; each is registered
        under ``(tier, rule_name)`` on the WordSpace so the chart can
        dispatch into the right parametrized fold. WordSpace / host_space
        are stashed via ``object.__setattr__`` to avoid the nn.Module
        ownership cycle (WordSpace owns the chart -> chart references
        this layer -> this layer references WordSpace).
        """
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
        # Register each host_layer with the wordSpace's host_layer
        # registry so the chart can dispatch into them.
        for rule_name, layer in self._by_name.items():
            word_space.register_host_layer(self.tier, rule_name, layer)
        self._cursor_compose = 0
        self._cursor_generate = 0
        self._cursor_compose_gen = -1
        self._cursor_generate_gen = -1
        # Stash the wordSpace and host_space as non-Module attributes
        # to avoid the circular nn.Module ownership trap (wordSpace
        # owns the chart; chart's host_layer registry references this
        # layer's children; this layer references wordSpace).
        object.__setattr__(self, '_word_space', word_space)
        # ``host_space`` is the per-tier Space (Perceptual / Conceptual /
        # Symbolic) that owns this dispatcher. When the chart fires
        # ``pi`` / ``sigma`` and the host space exposes
        # ``_pi_reverse`` / ``_sigma_reverse`` (two-pass ergodic mode
        # routes through ``pi2`` / ``sigma2``), reverse() delegates
        # there instead of the layer's bare ``reverse``.
        object.__setattr__(self, '_host_space', host_space)

    # -- cursor management ---------------------------------------------
    def _next_rule_name(self, *, direction):
        """Pop the next rule name for ``direction`` ('compose' or
        'generate'). Resets the cursor when wordSpace has bumped its
        generation counter for this direction.

        Reads ``word_space.current_rules`` / ``generate_rules`` as
        ``dict[tier, list[list[int]]]`` (per-row, per-step). For now
        we use row 0 as the canonical sequence; per-row dispatch (where
        rows fire different rules at the same step) is a follow-on.

        Returns the rule's ``method_name`` (string) or ``None`` when
        no chart rule is available (no code-level fallback — the grammar
        XML is the sole source of truth). The method name is the key
        used in ``self._by_name``.
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
            return method_name
        # No chart rule available -- post-2026-05-07 rollback removed
        # the ``default_rule`` code-level fallback. The grammar XML is
        # the sole source of truth; callers handle ``None`` as a no-op.
        return None

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

        Per the 2026-05-07 rollback, when the chart hasn't written a
        rule for this tier, dispatch is a no-op (no code-level
        fallback).

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
        x = self._read_subspace(subspace, layer=layer)
        if x is None:
            return subspace
        y = layer.forward(x)
        self._write_subspace(subspace, y, layer=layer)
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
        y = self._read_subspace(subspace, layer=layer)
        if y is None:
            return subspace
        # Two-pass ergodic adapter: when ``pi`` / ``sigma`` fires and
        # the host space exposes a tier-specific ``_pi_reverse`` /
        # ``_sigma_reverse`` (which routes through pi2/sigma2 in
        # two-pass ergodic mode), delegate there. Other unary rules
        # (not, etc.) keep going through ``layer.reverse``.
        host = getattr(self, '_host_space', None)
        if host is not None and rule_name == 'pi' and hasattr(host, '_pi_reverse'):
            x = host._pi_reverse(y)
        elif host is not None and rule_name == 'sigma' and hasattr(host, '_sigma_reverse'):
            x = host._sigma_reverse(y)
        else:
            x = layer.reverse(y)
        self._write_subspace(subspace, x, layer=layer)
        return subspace

    # -- subspace I/O per tier ------------------------------------------
    def _read_subspace(self, subspace, layer=None):
        """Read the per-position tensor from ``subspace`` for op dispatch.

        Routes through ``subspace.materialize()`` so the
        ``.active`` mask is applied and the op sees the live
        per-position activation -- never the underlying codebook
        weights ``W``. ``getW()`` on any ``.what`` / ``.where`` /
        ``.when`` Basis returns the codebook (a global lookup
        table); operating on that would mutate the codebook itself
        instead of the per-position activations the op is meant to
        transform.

        When ``layer.reads_activation`` is True (e.g.
        ``IntersectionLayer`` / ``UnionLayer``), the read source
        switches to ``materialize(mode='activation')`` -- the
        ``[B, V, 2]`` bivector activation -- because those ops
        operate on the activation poles, not the muxed event.

        Tier distinction is irrelevant here: every tier's per-
        position read goes through ``materialize()``. Tier-specific
        slicing of the muxed event (e.g. operating on the ``.what``
        bivector only) is the op's responsibility.
        """
        if subspace is None:
            return None
        # Activation-reading ops (IntersectionLayer / UnionLayer at
        # C-tier) read the bivector activation directly.
        if layer is not None and getattr(layer, 'reads_activation', False):
            if hasattr(subspace, 'materialize'):
                try:
                    return subspace.materialize(mode='activation')
                except Exception:
                    pass
        if hasattr(subspace, 'materialize'):
            return subspace.materialize()
        return getattr(subspace, 'event', None)

    def _write_subspace(self, subspace, tensor, layer=None):
        """Write the op's output back into ``subspace``.

        Default path: ``set_event(tensor)`` writes the muxed event
        and invalidates the cached materialize.

        Activation-writing path: when ``layer.reads_activation`` is
        True, the op produced a bivector activation
        ``[..., 2]`` -- write it via ``set_activation`` so the
        ``.activation`` Basis is updated and downstream
        ``materialize(mode='activation')`` reads the new value.
        """
        if subspace is None or tensor is None:
            return
        if layer is not None and getattr(layer, 'reads_activation', False):
            if hasattr(subspace, 'set_activation'):
                try:
                    subspace.set_activation(tensor)
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
                                builtin_layers=None):
    """Construct a per-space SyntacticLayer.

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

    Per the 2026-05-07 rollback there is no ``default_rule`` parameter;
    the grammar XML is the sole source of truth for which rule fires.
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
    layer = SyntacticLayer(
        tier=tier, word_space=word_space,
        host_layers=host_layers, host_space=space)
    space.syntacticLayer = layer
    return layer



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
        """Allocate per-row empty lists sized for ``batch`` rows.

        ``dim`` is the embedding dim; ``max_depth`` caps per-row depth
        (push asserts on overflow).
        """
        self._dim = int(dim)
        self._batch = int(batch)
        self._max_depth = int(max_depth)
        self._entries = [[] for _ in range(self._batch)]

    def ensure_batch(self, batch):
        """Reset per-row state to ``batch`` empty stacks.

        Cheap no-op when ``batch`` already matches the current width.
        """
        batch = int(batch)
        if batch == self._batch:
            return
        self._batch = batch
        self._entries = [[] for _ in range(self._batch)]

    def push(self, b, vec):
        """Append ``vec`` to row ``b``'s stack; assert shape + depth bound."""
        assert vec.shape == (self._dim,), (
            f"CategoryStack dim={self._dim}, got vec shape {tuple(vec.shape)}"
        )
        assert len(self._entries[b]) < self._max_depth, (
            f"CategoryStack overflow at row {b}: max_depth={self._max_depth}"
        )
        self._entries[b].append(vec)

    def pop(self, b):
        """Pop and return row ``b``'s top embedding."""
        return self._entries[b].pop()

    def depth(self, b):
        """Return current stack depth for row ``b``."""
        return len(self._entries[b])

    def flatten(self, b):
        """Concatenate row ``b``'s stack into a single 1D tensor.

        Empty rows return a zero-length tensor. Uses ``torch.cat`` so
        autograd flows back through any param-bearing entries.
        """
        if not self._entries[b]:
            return torch.zeros(0)
        return torch.cat(self._entries[b], dim=0)

    def clear_rows(self, start, end):
        """Empty rows ``[start, end)``. Used by per-row hard reset.

        Drops all pushed embeddings for the row range; subsequent
        ``depth(b)`` returns 0. Does not free the backing list slot.
        """
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
        """Allocate ``[B, max_depth, 2]`` long storage plus per-row top index."""
        self._batch = int(batch)
        self._max_depth = int(max_depth)
        self._entries = torch.zeros(self._batch, self._max_depth, 2,
                                    dtype=torch.long)
        self._top = torch.zeros(self._batch, dtype=torch.long)

    def ensure_batch(self, batch):
        """Reallocate backing tensors if ``batch`` width changed."""
        batch = int(batch)
        if batch == self._batch:
            return
        self._batch = batch
        self._entries = torch.zeros(batch, self._max_depth, 2,
                                    dtype=torch.long)
        self._top = torch.zeros(batch, dtype=torch.long)

    def push(self, b, rule_id, word_id):
        """Push ``(rule_id, word_id)`` onto row ``b``'s stack.

        Asserts the depth bound and advances the row's top index.
        """
        idx = int(self._top[b].item())
        assert idx < self._max_depth, (
            f"ReconstructionStack overflow at row {b}: max_depth={self._max_depth}"
        )
        self._entries[b, idx, 0] = int(rule_id)
        self._entries[b, idx, 1] = int(word_id)
        self._top[b] += 1

    def peek(self, b):
        """Return the top (rule_id, word_id) tuple for row ``b`` without popping."""
        idx = int(self._top[b].item()) - 1
        return (int(self._entries[b, idx, 0].item()),
                int(self._entries[b, idx, 1].item()))

    def pop(self, b):
        """Pop and return the top (rule_id, word_id) tuple from row ``b``."""
        self._top[b] -= 1
        idx = int(self._top[b].item())
        return (int(self._entries[b, idx, 0].item()),
                int(self._entries[b, idx, 1].item()))

    def clear_rows(self, start, end):
        """Reset rows ``[start, end)`` to empty stacks. Per-row hard reset.

        Zeros the (rule_id, word_id) backing tensor slice and the
        per-row top index; no-op when the range is empty / past end.
        """
        s, e = int(start), min(int(end), self._batch)
        if e <= s:
            return
        self._entries[s:e].zero_()
        self._top[s:e] = 0

    def depth(self, b):
        """Return current stack depth for row ``b`` (number of entries)."""
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
        """Build the chart, grammar layer, truth store, and per-tier dispatch.

        WordSpace bypasses the Space factory because its construction
        crosses tier boundaries (it needs references to Perceptual /
        Conceptual / Symbolic spaces). Detects the default-only grammar
        case so compose/generate can skip the CKY pass entirely.
        Mutates ``self`` to install ``syntacticLayer``, ``chart``,
        ``truthLayer``, the host_layer registry, and per-row buffers.
        """
        # Bypass Space.__init__ -- WordSpace doesn't fit the factory
        # style. Call nn.Module directly and populate the Space-contract
        # fields by hand.
        nn.Module.__init__(self)

        # Back-references to the three Spaces. Used post-2026-05-12 by
        # the grammar's lift/lower wiring to pass perceptual / conceptual
        # references to LiftLayer / LowerLayer at construction time, so
        # those layers can route the substrate sigma/pi after gating.
        # ``object.__setattr__`` bypasses nn.Module's submodule
        # registration so we don't create cycles (each Space is already
        # a direct child of the Model).
        object.__setattr__(self, 'perceptualSpace', perceptualSpace)
        object.__setattr__(self, 'conceptualSpace', conceptualSpace)
        object.__setattr__(self, 'symbolicSpace', symbolicSpace)

        # 1. Grammar must be configured before any SyntacticLayer
        # construction can resolve rule sets / transition rules.
        TheGrammar._configured = False
        TheGrammar._ensure_configured()
        grammar = TheGrammar

        # 1a. Detect the default-only case (every operational rule is
        # the unary pi / sigma fold registered as the per-tier
        # default). When true, ``compose`` / ``generate`` skip the
        # CKY-style inside / outside pass entirely; per-space
        # SyntacticLayer dispatch falls through to its registered
        # default rule, which fires PiLayer.forward / SigmaLayer.forward
        # exactly once per step -- mathematically identical to the
        # legacy bare ``self.pi(x)`` / ``self.sigma.forward(x)`` call
        # sites. Implicit non-operational rules (epsilon, X -> X
        # passthrough whose method_name is None) don't disqualify the
        # bypass.
        self._grammar_is_default_only = all(
            r.method_name is None or (
                r.method_name in ('pi', 'sigma') and r.arity == 1)
            for r in grammar.rules
        )

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

        # 4. Per-space SyntacticLayer dispatch lives on each space
        # (``space.syntacticLayer``); WordSpace itself no longer owns a
        # central SyntacticLayer instance.
        # 4a. Chart + host-layer registry. Per the 2026-05-01 syntactic-
        # layer refactor (doc/specs/2026-05-01-syntactic-layer-refactor.md):
        # WordSpace owns a Chart that runs CKY inside / outside passes
        # and writes per-(tier, step) rule selections into
        # ``current_rules`` / ``generate_rules``. Each per-space
        # SyntacticLayer registers its parametrized layers via
        # ``register_host_layer``; the chart consults
        # ``host_layer(tier, rule_name)`` to fire host-owned folds.
        self._host_layer_registry = {}
        # Initialize current_rules / generate_rules from the grammar
        # XML's per-tier natural folds. Per the 2026-05-07 rollback,
        # the grammar XML is the sole source of truth -- with no
        # ``default_rule`` code-level fallback the per-stage Spaces'
        # syntacticLayer dispatch must always have rules to fire.
        # ``compose`` / ``generate`` overwrite these on call.
        self.current_rules = self._default_compose_rules()
        self.generate_rules = self._default_generate_rules()
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
        # 4b. Hand the chart the grammar reference and install it as the
        # GrammarLayer chart authority. The chart's
        # ``register_grammar_layer`` / ``should_run_rule`` pair services
        # GrammarLayer.gated_run; the prior single-instance
        # SyntacticLayer that held this responsibility was retired.
        self.chart.grammar = grammar
        try:
            from Layers import GrammarLayer as _GrammarLayer
            _GrammarLayer.set_chart_authority(self.chart)
        except Exception:
            pass

        # 5. Per-space SyntacticLayer attachment. The perceptual
        # and conceptual spaces also get a ``wordSpace`` back-reference
        # so they can route through the shared buffer, but only the
        # symbolic space's compose() fires the chart.
        # Post-split: grammar's canonical home is S; the
        # SyntacticLayers at P and C are retained as backward-compat
        # dispatchers that no-op for grammars omitting per-tier
        # rules. They're not the architectural locus of grammar after
        # the split (S is), but the mechanism stays in place so
        # legacy configs continue to function and so any future
        # P/C-tier rule (e.g. for lift/lower at concept_dim) can
        # still fire through the chart's per-tier dispatch.
        if perceptualSpace is not None:
            perceptualSpace.attach_wordSpace(self)
            self._attach_per_space_syntactic_layer(
                perceptualSpace, tier='P')
        if conceptualSpace is not None:
            conceptualSpace.attach_wordSpace(self)
            self._attach_per_space_syntactic_layer(
                conceptualSpace, tier='C')
        if symbolicSpace is not None:
            symbolicSpace.attach_wordSpace(self)
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
        # Category / part-of-speech codebook.  This is the WordSpace's
        # ONLY codebook -- distinct from SymbolicSpace's symbol-prototype
        # codebook on ``SymbolicSpace.subspace.what.W``.
        #
        # Stores learned ``pos_dim``-wide embeddings for grammar
        # nonterminals AND POS terminals (S, NP, VP, AP, MP, PP, N, V,
        # ADJ, ADV, DET, P, O, '?', plus headroom).  Keyed by name via
        # ``self.category_index: dict[str, int]``; row ``i`` is the
        # embedding for category ``ordered_categories[i]``.
        #
        # Read by:
        #   * the chart's per-leaf POS scorer (Language.py: ``_chart_pos``,
        #     ``_apply_codebook_pos_seed``)
        #   * the rule predictor's input stack (the parsing-history
        #     vectors pushed onto ``self.category_stack``)
        # NOT used for symbol quantization -- that runs against the
        # symbolic codebook on SymbolicSpace.subspace.what.
        pos_dim = 4  # embedding width; also the category stack vector dim
        category_capacity = max(64, len(TheGrammar.categories))
        self.category_codebook = Codebook()
        self.category_codebook.create(
            nInput=0,           # input-side width unused for direct addressing
            nVectors=category_capacity,
            nDim=pos_dim,
            customVQ=True,
            monotonic=False,
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
        """Write the SVO triple for batch row ``b``.

        Stamps the three vectors into the ``[B, 3, svo_dim]`` buffer
        and marks the row's valid flag True. Subsequent reads via
        ``get_last_svo`` until ``clear_last_svo`` fires.
        """
        self._last_svo[b, 0] = subj
        self._last_svo[b, 1] = verb
        self._last_svo[b, 2] = obj
        self._svo_valid[b] = True

    def get_last_svo(self, b):
        """Return ``(subj, verb, obj)`` tensors for batch row ``b``.

        Caller is responsible for checking ``svo_valid(b)`` first;
        otherwise the returned vectors may be stale or zero.
        """
        e = self._last_svo[b]
        return e[0], e[1], e[2]

    def svo_valid(self, b):
        """True iff set_last_svo has fired for row ``b`` since last clear.

        Cheap CPU sync via ``.item()``; do not call inside tight loops.
        """
        return bool(self._svo_valid[b].item())

    def clear_last_svo(self, b=None):
        """Clear the SVO valid mask for row ``b`` (or all rows when None).

        Does not zero the underlying tensor -- subsequent ``get_last_svo``
        will return stale data unless re-stamped by ``set_last_svo``.
        """
        if b is None:
            self._svo_valid.zero_()
        else:
            self._svo_valid[b] = False

    # -- per-row STM-fired accessors --------------------------------------
    def stm_fired(self, b):
        """True iff stm_residual(b) has fired since last arm.

        Single-shot per sentence: re-armed by ``arm_stm`` or ``Reset``.
        """
        return bool(self._stm_fired[b].item())

    def mark_stm_fired(self, b):
        """Mark row ``b`` as having fired its STM residual this sentence.

        Subsequent ``stm_residual`` calls for ``b`` return None until
        ``arm_stm`` re-arms the row.
        """
        self._stm_fired[b] = True

    def arm_stm(self, b=None):
        """Re-arm row ``b`` (or all rows when None) for the next sentence.

        Resets the per-row single-shot flag so the next ``stm_residual``
        call can fire and stamp a new prediction bias.
        """
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

        Fast paths (no chart inside pass):
          * ``_grammar_is_default_only`` — every rule is the unary pi /
            sigma fold; rule selection is fully determined by the
            grammar XML so the chart inside pass adds no information.
          * ``useGrammar='none'`` — per-space SyntacticLayer dispatch
            still reads ``current_rules``, but the chart inside pass
            is skipped (MM_5M, the legacy BasicModel, etc.).

        Per the 2026-05-07 rollback both fast paths populate
        ``current_rules`` from the grammar XML's per-tier forward
        rules so per-space dispatch always finds a rule (no
        ``default_rule`` code-level fallback).
        """
        self._compose_generation += 1
        if self._grammar_is_default_only:
            self.current_rules = self._default_compose_rules()
            return self.current_rules
        # useGrammar='none' guard: skip the chart inside pass entirely.
        # Cached on first call to avoid an XMLConfig.get per forward.
        ug = getattr(self, '_use_grammar_cached', None)
        if ug is None:
            try:
                ug = str(util.TheXMLConfig.get(
                    "WordSpace.useGrammar", default="none")).lower()
            except Exception:
                ug = "none"
            self._use_grammar_cached = ug
        if ug == "none":
            self.current_rules = self._default_compose_rules()
            return self.current_rules
        self.current_rules = self.chart.compose(
            input_vectors, self, subspace=subspace) or {}
        return self.current_rules

    def generate(self, target_vectors, subspace=None):
        """Run the chart's outside pass + Viterbi backtrace; populate
        ``self.generate_rules``.

        Default-only and useGrammar='none' fast paths mirror ``compose``.
        """
        self._generate_generation += 1
        if self._grammar_is_default_only:
            self.generate_rules = self._default_generate_rules()
            return self.generate_rules
        ug = getattr(self, '_use_grammar_cached', None)
        if ug is None:
            try:
                ug = str(util.TheXMLConfig.get(
                    "WordSpace.useGrammar", default="none")).lower()
            except Exception:
                ug = "none"
            self._use_grammar_cached = ug
        if ug == "none":
            self.generate_rules = self._default_generate_rules()
            return self.generate_rules
        self.generate_rules = self.chart.generate(
            target_vectors, self, subspace=subspace) or {}
        return self.generate_rules

    # Method names that count as the per-tier "natural fold". The
    # default-only / useGrammar='none' fast paths fire only these from
    # the grammar XML so the per-space dispatch doesn't accidentally
    # invoke compositional operators (not, intersection, lift, ...)
    # that were authored for the chart's selection pass and are
    # disabled under useGrammar='none'. Maps the OLD ``default_rule``
    # semantics ('pi' for C, 'sigma' for P / S) onto the grammar XML
    # without re-introducing a code-level fallback: when the grammar
    # XML lacks ``C = pi(C)`` / ``S = sigma(S)`` / ``P = sigma(P)``
    # entries the dispatch is correctly a no-op for that tier.
    _NATURAL_FOLD_METHODS = ('pi', 'sigma')

    # Map a rule's LHS nonterminal to a per-tier letter for dispatch
    # routing. The legacy flat-format grammar parser tags every rule
    # ``tier='S'`` regardless of LHS; the canonical's LHS is
    # authoritative. Restrict to the three Space tiers; non-tier
    # nonterminals (NP, VP, ...) get filtered out by the caller.
    _LHS_TIER_MAP = {'P': 'P', 'C': 'C', 'S': 'S'}

    def _default_compose_rules(self):
        """Per-tier rule IDs for the default-only / useGrammar='none'
        fast path (forward direction).

        Returns ``dict[tier, list[list[int]]]`` listing each tier's
        forward natural-fold rule_ids from ``TheGrammar.rules``
        (``method_name`` in :data:`_NATURAL_FOLD_METHODS`,
        ``canonical`` not containing ``.reverse``). Cached after
        first call -- the grammar is fixed at construction time.
        """
        cache = getattr(self, '_default_compose_rules_cache', None)
        if cache is not None:
            return cache
        per_tier = {}
        for i, r in enumerate(TheGrammar.rules):
            mn = getattr(r, 'method_name', None)
            if mn not in self._NATURAL_FOLD_METHODS:
                continue
            canonical = getattr(r, 'canonical', '') or ''
            if '.reverse' in canonical:
                continue
            # The legacy flat-format parser tags every rule
            # ``tier='S'``; the canonical's LHS nonterminal is
            # authoritative for which Space tier should dispatch.
            tier = self._LHS_TIER_MAP.get(getattr(r, 'lhs', None))
            if tier is None:
                continue
            per_tier.setdefault(tier, []).append([i])
        merged = {tier: [[rid for row in rows for rid in row]]
                  for tier, rows in per_tier.items()}
        self._default_compose_rules_cache = merged
        return merged

    def _default_generate_rules(self):
        """Per-tier rule IDs for the default-only / useGrammar='none'
        fast path (reverse direction).

        Returns ``dict[tier, list[list[int]]]`` listing each tier's
        reverse natural-fold rule_ids (``method_name`` in
        :data:`_NATURAL_FOLD_METHODS`, ``canonical`` containing
        ``.reverse``). The dispatched ``method_name`` is shared with
        the matching forward rule so ``SyntacticLayer._by_name``
        resolves to the same parametrized layer either way.

        Falls back to the per-tier forward rule_ids when no explicit
        ``.reverse`` rules are listed for that tier. Legacy
        flat-format grammar blocks (``<S>sigma(S)</S>`` etc., without
        a ``<generate>`` section) still get the per-tier reverse
        dispatch because dispatch reads ``method_name`` rather than
        the canonical string.
        """
        cache = getattr(self, '_default_generate_rules_cache', None)
        if cache is not None:
            return cache
        forward_per_tier = {}
        reverse_per_tier = {}
        for i, r in enumerate(TheGrammar.rules):
            mn = getattr(r, 'method_name', None)
            if mn not in self._NATURAL_FOLD_METHODS:
                continue
            tier = self._LHS_TIER_MAP.get(getattr(r, 'lhs', None))
            if tier is None:
                continue
            canonical = getattr(r, 'canonical', '') or ''
            if '.reverse' in canonical:
                reverse_per_tier.setdefault(tier, []).append([i])
            else:
                forward_per_tier.setdefault(tier, []).append([i])
        per_tier = {}
        all_tiers = set(forward_per_tier) | set(reverse_per_tier)
        for tier in all_tiers:
            per_tier[tier] = (
                reverse_per_tier.get(tier)
                or forward_per_tier.get(tier))
        merged = {tier: [[rid for row in rows for rid in row]]
                  for tier, rows in per_tier.items()}
        self._default_generate_rules_cache = merged
        return merged

    def gate_l1_loss(self, lam=0.0):
        """L1 penalty on every per-rule ``raw_gate`` Parameter owned by
        this WordSpace's per-space SyntacticLayers.

        Post-2026-05-12: LiftLayer and LowerLayer no longer carry a
        learnable ``raw_gate`` -- VP's codebook activation now supplies
        the gate via elementwise multiplication. So this loss is a
        no-op for the rewritten lift/lower layers. The method is kept
        for back-compat with any GrammarLayer subclass that still
        owns a ``raw_gate`` parameter (none in tree today). Returns
        ``None`` when no such parameter is found.
        """
        lam = float(lam or 0.0)
        if lam <= 0.0:
            return None
        total = None
        for (_tier, _name), layer in self._host_layer_registry.items():
            raw = getattr(layer, 'raw_gate', None)
            if raw is None or not torch.is_tensor(raw):
                continue
            term = torch.tanh(raw).abs().sum()
            total = term if total is None else total + term
        if total is None:
            return None
        return lam * total

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
                             balance_weight=0.1,
                             model=None):
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

        # Luminosity is now a Mereology measure on the model itself.
        # When the caller supplies a `model` reference we delegate to
        # `model.Luminosity(truth_layer=...)`; otherwise (legacy path
        # without a model handle) we fall back to a neutral 0.0 score
        # so the multiplicative modulation degenerates to the
        # universality-only term -- preserving training stability for
        # callers that haven't been migrated yet.
        if model is not None and hasattr(model, 'Luminosity'):
            lum_val = float(model.Luminosity(truth_layer=self.truth_layer))
        else:
            lum_val = 0.0
        lum = torch.tensor(lum_val, device=total_loss.device,
                           dtype=total_loss.dtype)
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
        """Build the per-space SyntacticLayer for ``space`` (Step 4
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
                # SigmaLayer is the unary multiplicative OR-fold and is
                # registered under rule_name "sigma". Grammar XML's
                # ``P = sigma(P)`` rule fires SigmaLayer.forward(x).
                builtin_layers['sigma'] = sigma
        elif tier == 'C':
            pi = getattr(space, 'pi', None)
            if pi is not None:
                # PiLayer is the unary multiplicative AND-fold and is
                # registered under rule_name "pi". Grammar XML's
                # ``C = pi(C)`` rule fires PiLayer.forward(x).
                builtin_layers['pi'] = pi
        elif tier == 'S':
            sigma = getattr(space, 'sigma', None)
            if sigma is not None:
                builtin_layers['sigma'] = sigma
            negation = getattr(space, 'propositional_negation', None)
            if negation is not None:
                builtin_layers['not'] = negation
            # FusionLayer / ContiguousLayer were retired 2026-05-04:
            # the operator was a duplicate of DisjunctionLayer at
            # S-tier. Existing XML grammars referencing
            # ``Fusion(S, S)`` / ``Contiguous(S)`` should migrate to
            # ``disjunction(S, S)``.
            # Lift / Lower wiring: post-2026-05-12 refactor, lift and
            # lower are elementwise-gate-then-substrate-sigma/pi
            # operators. The gate is ``VP_c * NP_c`` at C-tier; the
            # substrate sigma (P.sigma, reconfigured at concept_dim)
            # is applied for lift, the substrate pi (C.pi) for lower.
            # The dedicated ``space.sigma_S`` / ``space.pi_S`` LDU
            # layers at sym_dim were retired -- they're redundant
            # with the substrate's existing sigma/pi.
            grammar_S_methods = {
                r.method_name for r in TheGrammar.rules
                if r.tier == 'S' and r.method_name is not None}
            if 'lift' in grammar_S_methods:
                from Layers import LiftLayer
                perceptualSpace = getattr(self, 'perceptualSpace', None)
                builtin_layers['lift'] = LiftLayer(
                    symbolicSpace=space,
                    perceptualSpace=perceptualSpace)
            if 'lower' in grammar_S_methods:
                from Layers import LowerLayer
                conceptualSpace = getattr(self, 'conceptualSpace', None)
                builtin_layers['lower'] = LowerLayer(
                    symbolicSpace=space,
                    conceptualSpace=conceptualSpace)
            # Mereological grammar layers: ``part`` / ``equals`` /
            # ``query`` are now pure-geometric operations on the
            # SymbolicSpace bivector codebook (clipped cosine
            # projection on the non-negative paired-index cone, per
            # Architecture.md §"Monotonicity of the bivector chain").
            # The standalone ``MereologicalTree`` sidecar that
            # previously stored explicit parent/equality links has
            # been retired -- the codebook IS the meronymic structure.
            # The ``<architecture><mereologicalTreeSize>`` XML knob
            # is correspondingly retired (any value is silently
            # ignored).
            if 'part' in grammar_S_methods:
                from Layers import PartLayer
                builtin_layers['part'] = PartLayer()
            if 'equals' in grammar_S_methods:
                from Layers import EqualsLayer
                builtin_layers['equals'] = EqualsLayer()
            if 'query' in grammar_S_methods:
                from Layers import QueryLayer
                builtin_layers['query'] = QueryLayer()
        layer = build_space_syntactic_layer(
            space, self, tier=tier,
            builtin_layers=builtin_layers)
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

    # -- composition dispatch ----------------------------------------
    def forwardSymbols(self, data, subspace):
        """Demux the muxed symbol tensor into the subspace's modality
        slots (Rule #2 axis commitment side effect).

        Post-2026-05-01 refactor: the actual symbolic composition runs
        on the chart (via ``ChartCompose`` in the pipeline + per-space
        ``SyntacticLayer.forward`` dispatch). This helper retains the
        demux side effect that downstream slot selectors depend on.

        Per the 2026-05-07 rollback, demux is skipped when there are
        no aux axes to split (nWhere == 0 and nWhen == 0). In that
        configuration the muxed event IS the .what content; routing
        it through ``set_what`` would clobber the codebook's transient
        slot and shadow the prototype Parameter from downstream
        ``getW()`` consumers (e.g. ``_nearest_symbol_target``).
        """
        if data.ndim == 3 and data.shape[-1] == getattr(subspace, 'muxedSize', -1):
            has_aux = (
                getattr(subspace, 'nWhere', 0) > 0
                or getattr(subspace, 'nWhen', 0) > 0)
            if has_aux:
                subspace.demux(data)
        return data

    def reverseSymbols(self, data, subspace):
        """No-op pass-through: chart-driven generation handles the
        symbol-side reverse via ``ChartGenerate`` + per-space
        ``SyntacticLayer.reverse`` dispatch.
        """
        return data

    def reconstruct(self, state, codebook_space, max_tokens=1):
        """Downward-generation MVP. The ``emit_head`` kernel lived on
        the legacy ``SyntacticLayer`` class (retired 2026-05-08); the
        Chart-based generation surface that replaces it doesn't yet
        expose a one-shot head emission. Until it does, this method
        returns the empty-emission stub callers (``Models._predicted_head``)
        already gate on.
        """
        return {'heads': [], 'residual': state, 'state': state}

    # -- buffer access + lifecycle ------------------------------------
    def read(self):
        """Return the fixed-width stack tensor for ConceptualSpace to
        concat with percepts and symbols.
        """
        return self.subspace.read()

    def clear_sentence(self):
        """Reset the stack at sentence boundaries.

        Forwards to the subspace's clear hook; called by ``BasicModel``
        on sentence boundary signals so the next sentence's parse
        starts with an empty word buffer.
        """
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
        """Return the parse-tree ledger for batch row ``b``.

        Forwards to ``self.subspace.get_blocks``. The ledger records the
        per-cell rule applications fired during the last compose, used
        for derivation-trace debugging.
        """
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
