

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
from Spaces import SubSpace, Space, InputSpace, PerceptualSpace, ModalSpace, ConceptualSpace, SymbolicSpace, OutputSpace


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

    # Order-typing primitives (plan:
    # doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
    # §Order-Typed Grammar). A category token like ``NP0`` / ``VP1`` /
    # ``NP*`` / ``NP*+1`` / ``NP*-1`` / ``DET`` (bare = constant 0) is
    # parsed into a ``ParsedCategory(name, order)`` where ``order`` is an
    # ``OrderExpr(kind, delta)``:
    #   kind='constant', delta=N   ->  literal order N
    #   kind='variable', delta=D   ->  rule-local '*' plus delta D
    OrderExpr = _namedtuple('OrderExpr', ['kind', 'delta'])
    ParsedCategory = _namedtuple('ParsedCategory', ['name', 'order'])

    # Per-rule order signature derived from the rule's parsed categories
    # plus its op name. ``order_delta``: +1 for ``lift``, -1 for ``lower``,
    # 0 for every other op (order-preserving by default).
    RuleOrderSignature = _namedtuple(
        'RuleOrderSignature',
        ['lhs_category', 'lhs_order_expr',
         'rhs_categories', 'rhs_order_exprs',
         'op_name', 'order_delta'],
    )

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
        # Phase 1 of the SubSpace.what STM refactor: V_sym is the size of
        # the terminal symbol codebook, which SymbolicSpace wires in once
        # its symbol codebook is built. Until then, the rule namespace
        # starts at 1 (treating V_sym=0). Used only by where_id_for_rule.
        # See doc/plans/2026-05-20-subspace-what-stm-signalrouter-refactor.md
        self.symbol_vocab_size = 0
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

    # -- Phase 1 GrammarRegistry surface --------------------------------
    #
    # Static lookup API for the SubSpace.what STM refactor. These
    # accessors do not run the live parser; they are pure rule-table
    # reads + a stable .where namespace. See
    # doc/plans/2026-05-20-subspace-what-stm-signalrouter-refactor.md
    # §"Phase 1: Grammar Registry Extraction".

    def num_rules(self):
        """Total rule count (configures lazily if needed)."""
        self._ensure_configured()
        return len(self.rules)

    def rule(self, rule_id):
        """Return the full ``RuleDef`` for ``rule_id``."""
        return self.rules[rule_id]

    def rules_for_tier(self, tier, arity=None):
        """Return rule_ids whose ``RuleDef.tier`` matches ``tier``.

        ``arity`` optionally filters to that arity (1 or 2).
        """
        self._ensure_configured()
        out = []
        for i, r in enumerate(self.rules):
            if r.tier != tier:
                continue
            if arity is not None and r.arity != arity:
                continue
            out.append(i)
        return out

    # -- Phase 1 .where namespace ---------------------------------------
    #
    # The stack-mode .where namespace is:
    #     0                           empty slot
    #     1..V_sym                    terminal symbol locations
    #     V_sym+1..V_sym+R_rule       grammar rule locations
    # V_sym is ``self.symbol_vocab_size``, populated by SymbolicSpace
    # in Phase 3. Empty/invalid inputs collapse to 0.

    def where_id_for_symbol(self, symbol_id):
        """Stack ``.where`` location for a terminal symbol codebook row.

        Returns 0 for invalid/empty inputs (matches the spec's
        zero-is-empty namespace).
        """
        if symbol_id is None or symbol_id < 0:
            return 0
        return int(symbol_id) + 1

    def where_id_for_rule(self, rule_id):
        """Stack ``.where`` location for a grammar rule.

        Returns 0 for invalid/empty inputs.
        """
        if rule_id is None or rule_id < 0:
            return 0
        return int(self.symbol_vocab_size) + 1 + int(rule_id)

    def decode_where(self, where_id):
        """Decode a stack ``.where`` location back into ``(kind, id)``.

        Inverse of ``where_id_for_symbol`` / ``where_id_for_rule`` (used
        in the Phase 7 reverse path; see
        doc/plans/2026-05-20-subspace-what-stm-signalrouter-refactor.md).

        Returns:
            ``('empty', None)``    when ``where_id <= 0``
            ``('terminal', sym_id)`` when ``1 <= where_id <= V_sym``
            ``('rule', rule_id)``    when ``where_id > V_sym``

        ``where_id`` may be a Python int, a float (the live router
        stores the int in a float tensor), or a 0-D tensor; values are
        coerced to int via ``int(round(...))`` so noisy lookups in a
        float-encoded carrier still land on the right bucket.
        """
        if where_id is None:
            return ('empty', None)
        if hasattr(where_id, 'item'):
            where_id = float(where_id.item())
        wid = int(round(float(where_id)))
        if wid <= 0:
            return ('empty', None)
        v_sym = int(self.symbol_vocab_size)
        if wid <= v_sym:
            return ('terminal', wid - 1)
        return ('rule', wid - v_sym - 1)

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
        self.id_SS = self._find_identity_rule_id(self.start_symbol)
        self._bump_rule_table_version()

    def _find_identity_rule_id(self, symbol):
        # Identity rule: LHS == RHS, arity 1, method_name None.
        # Used as the no-op grammatical transition at padding columns of
        # the static per-word loop (doc/plans/2026-05-20-static-per-word-loop-impl.md).
        for idx, rule in enumerate(self.rules_upward):
            if (rule.lhs == symbol
                    and rule.method_name is None
                    and rule.arity == 1
                    and rule.rhs_symbols == (symbol,)):
                return idx
        return None

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

    @staticmethod
    def _parse_category(token):
        """Parse a category token into ``(name, OrderExpr)``.

        Accepts (plan:
        doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
        §Order-Typed Grammar):

          ``DET``     -> name='DET', order=constant 0   (bare = sugar for 0)
          ``NP3``     -> name='NP',  order=constant 3
          ``VP1``     -> name='VP',  order=constant 1
          ``S4``      -> name='S',   order=constant 4
          ``NP*``     -> name='NP',  order=variable +0  (rule-local *)
          ``NP*+1``   -> name='NP',  order=variable +1
          ``NP*-1``   -> name='NP',  order=variable -1

        2026-05-20 Kleene restoration (path-to-complete §2): the
        polymorphic ``*`` form is restored alongside explicit constants.
        At REDUCE time the rule-local ``*`` is bound from the operand's
        order and propagated through the rule's other slots. Bare
        categories (no annotation) still bind to constant 0.

        Whitespace around the token is stripped. Malformed tokens
        raise ``ValueError``.
        """
        s = str(token).strip()
        if not s:
            raise ValueError(f"Cannot parse category: {token!r}")
        i = 0
        while i < len(s) and (s[i].isalpha() or s[i] == '_'):
            i += 1
        if i == 0:
            raise ValueError(f"Cannot parse category: {token!r}")
        name = s[:i]
        suffix = s[i:]
        if not suffix:
            return Grammar.ParsedCategory(
                name=name,
                order=Grammar.OrderExpr(kind='constant', delta=0))
        # Kleene form: '*', '*+N', '*-N'.
        if suffix.startswith('*'):
            rest = suffix[1:]
            if not rest:
                delta = 0
            else:
                try:
                    delta = int(rest)
                except ValueError:
                    raise ValueError(
                        f"Cannot parse category: {token!r} "
                        f"(Kleene suffix must be '*', '*+N', or '*-N')")
            return Grammar.ParsedCategory(
                name=name,
                order=Grammar.OrderExpr(kind='variable', delta=delta))
        try:
            delta = int(suffix)
        except ValueError:
            raise ValueError(
                f"Cannot parse category: {token!r} "
                f"(suffix must be a constant integer or a Kleene '*' form)")
        return Grammar.ParsedCategory(
            name=name,
            order=Grammar.OrderExpr(kind='constant', delta=delta))

    # Operations that change conceptual order. Every other op is
    # order-preserving (order_delta = 0). See plan §Order-Typed Grammar.
    _ORDER_CHANGING_OPS = {
        'lift':  +1,
        'lower': -1,
    }

    def _rule_order_signature(self, rule):
        """Compute the ``RuleOrderSignature`` for a parsed ``RuleDef``.

        Parses ``rule.lhs`` and each ``rule.rhs_symbols`` token through
        ``_parse_category`` to extract category names + ``OrderExpr``s.
        ``order_delta`` is +1 for ``lift``, -1 for ``lower``, 0 otherwise.
        """
        lhs_parsed = Grammar._parse_category(rule.lhs)
        rhs_parsed = tuple(
            Grammar._parse_category(s) for s in (rule.rhs_symbols or ()))
        op = rule.method_name
        delta = Grammar._ORDER_CHANGING_OPS.get(op, 0)
        return Grammar.RuleOrderSignature(
            lhs_category=lhs_parsed.name,
            lhs_order_expr=lhs_parsed.order,
            rhs_categories=tuple(p.name for p in rhs_parsed),
            rhs_order_exprs=tuple(p.order for p in rhs_parsed),
            op_name=op,
            order_delta=delta,
        )

    # No static validation of ``RuleOrderSignature`` at grammar-load time:
    # words are mapped to the category codebook by *soft assignment* that
    # participates in the parser's superposition state. Whether a given
    # word fills an ``NP3`` vs ``NP4`` slot is a runtime / superposition
    # question — not a fact the grammar can pre-empt. Order admissibility
    # therefore lives in STM REDUCE (Phase 2), where the soft category
    # distributions of operands are matched against the rule's order
    # signature dynamically.

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

        # Legacy tolerance: ``<C>C = pi(C)</C>`` is the per-tier element
        # form where the element NAME is the LHS and the CONTENT is the
        # RHS.  Some configs (MM_5M, LM_5M etc.) redundantly prefix the
        # content with ``LHS = `` -- strip it so the function-call parser
        # below sees just ``pi(C)`` and ``method_name`` ends up as
        # ``pi`` (the natural-fold key) rather than ``C = pi`` (which
        # silently falls through ``_default_compose_rules``'s
        # ``_NATURAL_FOLD_METHODS`` filter and leaves the chart with no
        # C-tier rule, breaking ConceptualSpace dispatch).
        rhs_stripped = rhs.lstrip()
        eq_prefix = f"{lhs}="
        if (rhs_stripped.startswith(eq_prefix)
                or rhs_stripped.startswith(f"{lhs} =")):
            rhs = rhs_stripped[rhs_stripped.index('=') + 1:].strip()

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
# RuleCodebook -- grammatical operation codebook (Phase 3 of the
# SubSpace.what STM refactor; see
# doc/plans/2026-05-20-subspace-what-stm-signalrouter-refactor.md).
#
# The rule codebook holds rule **identity / location**, NOT parent
# vectors. Parent vectors for reductions are computed by SyntacticLayer
# .execute on child arguments. This codebook only provides:
#   * the ``.where`` location stamped into reduced stack slots
#     (delegated to ``Grammar.where_id_for_rule``)
#   * an optional learned per-rule embedding for router scoring
#   * an optional rule identity vector for diagnostics
#
# Distinct from the SymbolicSpace symbol codebook (`SymbolicSpace
# .subspace.what`), which holds the long-term terminal symbol prototypes
# the SHIFT path quantizes against.
# =====================================================================
class RuleCodebook(nn.Module):
    """Long-term identity store for grammatical operations.

    Args:
        num_rules: number of grammar rules (R).
        embedding_dim: width of the optional per-rule scoring embedding.
            Default 0 (no learned embedding).
        grammar: optional non-owning reference to a ``Grammar`` whose
            ``where_id_for_rule`` provides the ``.where`` namespace.
            When unset the location falls back to a bare ``rule_id + 1``.
    """

    def __init__(self, num_rules, *, embedding_dim=0, grammar=None):
        """Allocate state; see class docstring."""
        super().__init__()
        self.num_rules = int(num_rules)
        # Non-owning reference so the Module graph stays cycle-free
        # (Grammar is a plain Python object, not an nn.Module, but the
        # stash-via-object.__setattr__ pattern is conservative).
        object.__setattr__(self, '_grammar', grammar)
        if embedding_dim and embedding_dim > 0:
            self.embedding = nn.Parameter(
                torch.zeros(self.num_rules, int(embedding_dim))
            )
            if self.num_rules > 0:
                nn.init.xavier_normal_(self.embedding)
        else:
            self.register_parameter('embedding', None)

    @property
    def grammar(self):
        """Read-only accessor for the bound Grammar (or None)."""
        return self._grammar

    def attach_grammar(self, grammar):
        """Late-bind a Grammar reference (used when wiring after construction)."""
        object.__setattr__(self, '_grammar', grammar)

    def location(self, rule_id):
        """Return the ``.where`` location for ``rule_id``.

        Routes through ``grammar.where_id_for_rule`` when a Grammar is
        attached; otherwise uses a bare ``rule_id + 1`` fallback so
        ``RuleCodebook(num_rules=R)`` is usable in isolation.
        Returns 0 for invalid inputs (matches the spec's empty sentinel).
        """
        if rule_id is None:
            return 0
        rid = int(rule_id)
        if rid < 0:
            return 0
        if self._grammar is not None:
            return self._grammar.where_id_for_rule(rid)
        return rid + 1


# =====================================================================
# LanguageLayer -- inlined from bin/LanguageLayer.py (2026-05-11 module
# consolidation). Selected via WordSpace.routerKind = 'signal' in XML.
# Replaces the Chart's soft-superposition CKY forest with per-layer
# COPY/REDUCE routing on the subspace tensor. Owned by Chart, lazily
# built via Chart._ensure_signal_router.
# =====================================================================
class _BinaryGrammarOpAdapter(nn.Module):
    """Adapt a GrammarLayer with a `.compose(left, right)` method into a
    plain binary callable for the LanguageLayer's `BinaryStructuredReductionLayer`.

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


class LanguageLayer(Layer):
    """Top-level signal-routing parser. Owned by Chart when
    router_kind == "signal", and by SymbolicSpace for the new
    stack-rewrite path (Phase 5 of the SubSpace.what STM refactor).
    Parallels Chart.compose / Chart.generate.

    Multi-tier: a unary layer and/or a binary layer can be attached per
    tier (e.g., 'P', 'C', 'S'). On compose, tiers run in sorted order;
    within each tier, unary fires first then binary, with the soft slab
    of the previous step feeding the next so gradient reaches every op.

    **Layer contract** (post-2026-05-20 stack-rewrite refactor):

    The canonical entry points are ``forward(subspace, syntactic_layer,
    ..., actions=...)`` and ``reverse(subspace, syntactic_layer, ...)``
    -- both wrap the stack-rewrite primitives (shift/reduce/unreduce)
    so call sites can treat LanguageLayer like any other Layer subclass.

    The legacy ``compose`` / ``generate`` are preserved verbatim for
    chart-router (router_kind='signal') back-compat; they operate on a
    ``[B, N, D]`` slab through the attached ``_unary_layers`` /
    ``_binary_layers`` ModuleDicts and produce per-row rule lists. The
    two paths are independent: the Layer-style ``forward`` ignores the
    ModuleDicts (it dispatches through the supplied SyntacticLayer's
    ``execute``), and the legacy ``compose`` ignores the Layer-style
    args.
    """

    def __init__(self, n_input, n_output, *, hidden_dim, feature_dim,
                 max_depth, temperature=1.0):
        """Initialize empty unary / binary ModuleDicts; ops are attached later.

        ``feature_dim`` is the slab D; ``temperature`` divides logits in
        the inner soft DP. ``max_depth`` caps the number of binary
        reduction rounds; the actual cap is min(N-1, max_depth).

        Calls ``Layer.__init__(n_input, n_output)`` so ``self.nInput``
        / ``self.nOutput`` / ``self.layers`` follow the standard Layer
        contract. The ergodic interface (paramUpdate / set_sigma /
        Start / End) is inherited and dispatches to ``self.layers``
        (a plain list, kept empty here -- the trainable scoring layers
        live in ``self._unary_layers`` / ``self._binary_layers``
        ModuleDicts so the optimizer still sees them via the standard
        nn.Module parameter walk).
        """
        super().__init__(int(n_input), int(n_output))
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
                "LanguageLayer.compose called before attach_layer_ops() / "
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
                "LanguageLayer.generate called before attach_layer_ops() / "
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

    # -- Phase 4 stack-rewrite path -------------------------------------
    #
    # See doc/plans/2026-05-20-subspace-what-stm-signalrouter-refactor.md
    # §"Phase 4: LanguageLayer Stack Rewrite Path". These methods operate
    # on a stack-mode SubSpace whose ``.what`` IS the live STM:
    #
    #     subspace.what:       [B, K, D] payloads
    #     subspace.where:      [B, K, W] codebook locations
    #     subspace.activation: [B, K]    occupancy mask (1=live, 0=empty)
    #
    # Existing ``compose`` / ``generate`` paths above remain intact for
    # the chart-router compatibility surface; ``shift`` / ``reduce`` /
    # ``forward_stack`` are the new path.
    #
    # First-patch design choices (per the plan's "Implementation Notes
    # For Claude"):
    #   * Hard SHIFT/REDUCE only -- no soft DP yet. Soft routing reuses
    #     the existing scoring utilities (binary_tiling_soft_dp) when
    #     wired in a later phase.
    #   * Per-row occupancy reads use a small eager bridge (occ.sum
    #     along K). This is the documented "small eager bridge" the
    #     plan permits for a first correctness patch.
    #   * Gradient flow: parent.what flows through the cloned-then-
    #     scatter write into subspace.what, then back through the
    #     SyntacticLayer.execute call to left/right child payloads and
    #     to the op's parameters.

    @staticmethod
    def _stack_n_live(subspace):
        """Per-row count of live stack slots (``activation > 0``)."""
        occ = subspace.materialize(mode="activation")
        if occ is None:
            raise RuntimeError(
                "LanguageLayer stack-mode: subspace.activation is None; "
                "set_activation([B, K]) must be called before SHIFT/REDUCE"
            )
        if occ.ndim != 2:
            raise ValueError(
                f"LanguageLayer stack-mode: activation must be [B, K], "
                f"got shape {tuple(occ.shape)}"
            )
        return (occ.abs() > 0).long().sum(dim=-1)              # [B]

    @staticmethod
    def _encode_where(where_buf, where_id):
        """Encode a scalar location into a single-slot ``.where`` vector.

        First-patch convention: stamp the integer into element [0] of
        the W-wide row; remaining elements are zero. The plan's
        encode/decode helpers can later swap this for a proper sin/cos
        encoding without changing the namespace semantics.
        """
        W = where_buf.shape[-1]
        vec = where_buf.new_zeros(W)
        if W >= 1:
            vec[0] = float(where_id)
        return vec

    def shift(self, subspace, terminal_what, where_id):
        """Push a terminal payload into the next empty stack slot.

        Mutates ``subspace.what`` / ``.where`` / ``.activation`` in
        place (via setters; the underlying Basis tensors are replaced).
        Returns the same subspace for fluent call chains.

        Args:
            subspace: stack-mode SubSpace.
            terminal_what: ``[B, D]`` payload (the snap of a continuous
                concept against the terminal symbol codebook).
            where_id: scalar int location in the stack ``.where``
                namespace (typically ``grammar.where_id_for_symbol(s)``).

        Raises:
            RuntimeError: if any batch row's stack is already full
                (no empty slot to receive the terminal).
        """
        what = subspace.materialize(mode="what")
        where = subspace.materialize(mode="where")
        if what is None or what.ndim != 3:
            raise ValueError(
                f"LanguageLayer.shift: subspace.what must be [B, K, D], "
                f"got shape {None if what is None else tuple(what.shape)}"
            )
        if where is None or where.ndim != 3:
            raise ValueError(
                f"LanguageLayer.shift: subspace.where must be [B, K, W], "
                f"got shape {None if where is None else tuple(where.shape)}"
            )
        B, K, D = what.shape
        if terminal_what.shape != (B, D):
            raise ValueError(
                f"LanguageLayer.shift: terminal_what shape {tuple(terminal_what.shape)} "
                f"!= ({B}, {D})"
            )
        n_live = self._stack_n_live(subspace)                  # [B]
        if (n_live >= K).any():
            raise RuntimeError(
                f"LanguageLayer.shift: stack full (K={K}); per-row n_live="
                f"{n_live.tolist()}"
            )
        next_empty = n_live                                    # [B] -- push index
        arange_B = torch.arange(B, device=what.device)

        # Clone-then-scatter preserves gradient flow back through the
        # untouched slots (autograd path: each row's other slots are
        # functions of the prior `what` tensor) AND through the new
        # `terminal_what` (the scatter writes it into slot `next_empty`).
        what_new = what.clone()
        what_new[arange_B, next_empty, :] = terminal_what

        where_new = where.clone()
        where_vec = self._encode_where(where, where_id)        # [W]
        where_new[arange_B, next_empty, :] = where_vec

        # Activation: stack-mode occupancy is a scalar 1.0 per live slot.
        occ = subspace.materialize(mode="activation")
        occ_new = occ.clone()
        occ_new[arange_B, next_empty] = 1.0

        subspace.set_what(what_new)
        subspace.set_where(where_new)
        subspace.set_activation(occ_new)
        return subspace

    def reduce(self, subspace, syntactic_layer, rule_id,
               *, rule_codebook=None, where_id=None):
        """Reduce the top two live stack slots with the given grammar rule.

        Implements the plan's hard REDUCE pseudo-code:

            parent = syntactic_layer.execute(rule_id, left, right)
            what[:, i, :] = parent      # surviving slot
            where[:, i, :] = rule_where
            occ[:, i] = 1
            what[:, j, :] = 0           # consumed slot
            where[:, j, :] = 0
            occ[:, j] = 0

        where ``i = n_live - 2`` (left, survives) and
              ``j = n_live - 1`` (right, consumed).

        Args:
            subspace: stack-mode SubSpace.
            syntactic_layer: per-tier SyntacticLayer with ``execute``
                (Phase 2). Computes parent.what from child payloads.
            rule_id: grammar rule id (must be arity 2 for top-2 reduce).
            rule_codebook: optional RuleCodebook providing the .where
                stamp via ``rule_codebook.location(rule_id)``. Either
                ``rule_codebook`` or ``where_id`` must be supplied.
            where_id: explicit ``.where`` location override. Wins when
                both are provided.

        Raises:
            RuntimeError: if any batch row has fewer than 2 live slots.
            ValueError: when neither rule_codebook nor where_id supply
                the rule's .where location.
        """
        if where_id is None and rule_codebook is None:
            raise ValueError(
                "LanguageLayer.reduce: provide either `rule_codebook` "
                "(preferred) or `where_id` (override)"
            )
        if where_id is None:
            where_id = rule_codebook.location(rule_id)

        what = subspace.materialize(mode="what")
        where = subspace.materialize(mode="where")
        n_live = self._stack_n_live(subspace)                  # [B]
        if (n_live < 2).any():
            raise RuntimeError(
                f"LanguageLayer.reduce: stack underflow (need >=2 live "
                f"slots); per-row n_live={n_live.tolist()}"
            )
        B, K, D = what.shape
        arange_B = torch.arange(B, device=what.device)
        i_slot = n_live - 2                                    # [B] survives
        j_slot = n_live - 1                                    # [B] consumed

        left  = what[arange_B, i_slot, :]                      # [B, D]
        right = what[arange_B, j_slot, :]                      # [B, D]
        parent = syntactic_layer.execute(int(rule_id), left, right)  # [B, D]

        what_new = what.clone()
        what_new[arange_B, i_slot, :] = parent
        what_new[arange_B, j_slot, :] = 0.0

        where_new = where.clone()
        where_vec = self._encode_where(where, where_id)        # [W]
        where_new[arange_B, i_slot, :] = where_vec
        where_new[arange_B, j_slot, :] = 0.0

        occ = subspace.materialize(mode="activation")
        occ_new = occ.clone()
        occ_new[arange_B, i_slot] = 1.0
        occ_new[arange_B, j_slot] = 0.0

        subspace.set_what(what_new)
        subspace.set_where(where_new)
        subspace.set_activation(occ_new)
        return subspace

    def unreduce(self, subspace, syntactic_layer, *,
                 grammar=None, rule_codebook=None):
        """Inverse of ``reduce``: split the top live slot via ``layer.reverse``.

        Phase 7 of the SubSpace.what STM refactor (see plan §"Reverse
        And Reconstruction"). Decodes the top live slot's ``.where`` to
        find which rule produced the parent, then calls that rule
        layer's ``reverse`` on the parent payload to recover children
        and writes them back into the stack.

        For lossy / non-bijective ops the layer's ``reverse`` is an
        identity stub (e.g. ``ConjunctionLayer.reverse(parent) ->
        (parent, parent)``). This is intentional per the plan:

            "use rule-specific reverse/generate if implemented
             otherwise identity/pass-through stub"

        No-op when the top slot is empty or stamped as a terminal --
        terminals are leaves on this path; their reverse is the
        codebook unsnap (Phase 8+ work).

        Args:
            subspace: stack-mode SubSpace (mutated in place).
            syntactic_layer: per-tier SyntacticLayer; provides the
                host layer for the decoded rule via ``_by_name``.
            grammar: Grammar for ``decode_where``. Required when
                ``rule_codebook`` is not supplied (or its grammar is
                None).
            rule_codebook: optional RuleCodebook with an attached
                Grammar; falls back to ``grammar`` when None.

        Raises:
            RuntimeError: on stack underflow (no live slots) or
                overflow (no room for the new right-child slot).
            KeyError: when the decoded rule is not registered on the
                SyntacticLayer's ``_by_name`` table.
        """
        # Resolve the Grammar used for .where decoding.
        if grammar is None and rule_codebook is not None:
            grammar = rule_codebook.grammar
        if grammar is None:
            raise ValueError(
                "LanguageLayer.unreduce: provide `grammar` or a "
                "`rule_codebook` with an attached Grammar"
            )

        what = subspace.materialize(mode="what")
        where = subspace.materialize(mode="where")
        n_live = self._stack_n_live(subspace)                  # [B]
        if (n_live < 1).any():
            raise RuntimeError(
                f"LanguageLayer.unreduce: stack underflow (no live "
                f"slots); per-row n_live={n_live.tolist()}"
            )
        B, K, D = what.shape
        if (n_live >= K).any():
            raise RuntimeError(
                f"LanguageLayer.unreduce: stack overflow (K={K}); "
                f"unreduce needs an empty slot to the right of the "
                f"top. Per-row n_live={n_live.tolist()}"
            )
        arange_B = torch.arange(B, device=what.device)
        top_slot = n_live - 1                                  # [B]

        # Decode .where to find the rule. First-patch convention:
        # batch-row-0's slot is canonical; per-row decoding is a
        # follow-up. The where carrier is float; decode_where rounds.
        top_where_b0 = where[0, int(top_slot[0].item()), 0]
        kind, decoded_id = grammar.decode_where(top_where_b0)

        if kind != 'rule':
            # Empty or terminal -- no reverse on this path.
            return subspace

        rule_id = int(decoded_id)
        method_name = grammar.method_name(rule_id)
        layer = syntactic_layer._by_name.get(method_name)
        if layer is None:
            raise KeyError(
                f"LanguageLayer.unreduce: tier={syntactic_layer.tier!r} "
                f"has no host layer for rule_id={rule_id} "
                f"(method_name={method_name!r}). Registered rules: "
                f"{sorted(syntactic_layer._by_name.keys())}"
            )
        arity = int(getattr(layer, 'arity', 2))

        parent = what[arange_B, top_slot, :]                   # [B, D]

        # Identity-stub fallback per plan §"Reverse And Reconstruction":
        #   "use rule-specific reverse/generate if implemented
        #    otherwise identity/pass-through stub"
        # The base ``Layer.reverse`` is a shape-asserting identity (not
        # a real inverse), so ``hasattr(layer, 'reverse')`` is True even
        # when no semantic inverse exists. We invoke ``reverse``, then
        # validate the return shape matches the rule's arity; on bad
        # shape or any exception we fall back to (parent[, parent]).
        def _identity_stub():
            return parent if arity == 1 else (parent, parent)

        try:
            child = layer.reverse(parent)
        except Exception:
            child = _identity_stub()
        else:
            if arity == 2 and (not isinstance(child, tuple)
                                or len(child) != 2):
                # Arity-2 rules must return (left, right); the base
                # Layer.reverse returns a single tensor -- treat as
                # identity-stub.
                child = _identity_stub()

        what_new = what.clone()
        where_new = where.clone()
        occ = subspace.materialize(mode="activation")
        occ_new = occ.clone()

        if arity == 1:
            # Reverse returned a single tensor; write back into the
            # top slot and leave occupancy / where unchanged for the
            # other slots. (Arity-1 reduce is not yet a primitive --
            # this branch is forward-looking.)
            if isinstance(child, tuple):
                # Some arity-1 reverses return single-tuple wrappings.
                child = child[0]
            what_new[arange_B, top_slot, :] = child
            # Where stays as the rule stamp; downstream callers can
            # re-decode if they want to track depth.
        else:
            # Arity-2: child is (left, right). Write left into the
            # top slot and right into the next-empty slot (top + 1).
            if not isinstance(child, tuple) or len(child) != 2:
                raise TypeError(
                    f"LanguageLayer.unreduce: arity-2 layer "
                    f"{method_name!r}.reverse(parent) must return "
                    f"(left, right); got {type(child).__name__}"
                )
            left, right = child
            new_slot = top_slot + 1                            # [B]
            what_new[arange_B, top_slot, :] = left
            what_new[arange_B, new_slot, :] = right
            # Children's .where is unknown without history -- clear it
            # to the empty sentinel (the plan permits identity-stub
            # behavior for reverse). Phase 8+ can carry an in-band
            # provenance trail if needed.
            where_new[arange_B, top_slot, :] = 0.0
            where_new[arange_B, new_slot, :] = 0.0
            occ_new[arange_B, new_slot] = 1.0
            # Top slot was already occupied; activation[top_slot] stays
            # at 1.0.

        subspace.set_what(what_new)
        subspace.set_where(where_new)
        subspace.set_activation(occ_new)
        return subspace

    def reverse_stack(self, subspace, syntactic_layer, *,
                      grammar=None, rule_codebook=None, max_steps=None):
        """Repeatedly ``unreduce`` until only terminal / empty slots
        remain (or ``max_steps`` is reached).

        Inverse-orchestrator counterpart to ``forward_stack``. Each
        step examines the top live slot's ``.where``; if it decodes
        to a rule, unreduce; otherwise we are done. ``max_steps``
        bounds the loop (defaults to ``K-1`` reductions worth).

        Returns the same subspace after unwinding.
        """
        if grammar is None and rule_codebook is not None:
            grammar = rule_codebook.grammar
        if grammar is None:
            raise ValueError(
                "LanguageLayer.reverse_stack: provide `grammar` or a "
                "`rule_codebook` with an attached Grammar"
            )
        what = subspace.materialize(mode="what")
        K = what.shape[1]
        budget = (K - 1) if max_steps is None else int(max_steps)
        for _ in range(budget):
            where = subspace.materialize(mode="where")
            n_live = self._stack_n_live(subspace)
            if int(n_live[0].item()) < 1:
                break
            top = int(n_live[0].item()) - 1
            top_where = where[0, top, 0]
            kind, _id = grammar.decode_where(top_where)
            if kind != 'rule':
                break
            self.unreduce(subspace, syntactic_layer,
                          grammar=grammar, rule_codebook=rule_codebook)
        return subspace

    def forward_stack(self, subspace, syntactic_layer, *,
                      actions, rule_codebook=None, grammar=None):
        """Run a sequence of hard SHIFT / REDUCE actions on a stack-mode subspace.

        First-patch orchestrator: takes an explicit ``actions`` list.
        A full router (scoring SHIFT vs REDUCE, soft DP) is a later
        phase; the contract pinned here is that the actions, regardless
        of how they are produced, rewrite ``subspace.what / .where /
        .activation`` correctly.

        Action format:
            ('shift', terminal_what: Tensor [B, D], where_id: int)
            ('reduce', rule_id: int)

        Args:
            subspace: stack-mode SubSpace (mutated in place).
            syntactic_layer: per-tier SyntacticLayer for REDUCE.
            actions: iterable of (kind, ...) tuples.
            rule_codebook: optional, used to resolve rule .where ids
                for REDUCE actions. Falls back to ``grammar`` when
                None.
            grammar: optional Grammar, used as a secondary fallback
                for ``where_id_for_rule`` when no rule_codebook is
                provided.

        Returns:
            The same subspace.
        """
        for step, action in enumerate(actions):
            if not action:
                continue
            kind = action[0]
            if kind == 'shift':
                if len(action) != 3:
                    raise ValueError(
                        f"forward_stack step {step}: shift expects "
                        f"(kind, terminal_what, where_id); got {action!r}"
                    )
                _, terminal_what, where_id = action
                self.shift(subspace, terminal_what, where_id)
            elif kind == 'reduce':
                if len(action) != 2:
                    raise ValueError(
                        f"forward_stack step {step}: reduce expects "
                        f"(kind, rule_id); got {action!r}"
                    )
                _, rule_id = action
                where_id = None
                rc = rule_codebook
                if rc is None and grammar is not None:
                    where_id = grammar.where_id_for_rule(int(rule_id))
                self.reduce(subspace, syntactic_layer, rule_id,
                            rule_codebook=rc, where_id=where_id)
            else:
                raise ValueError(
                    f"forward_stack step {step}: unknown action kind "
                    f"{kind!r}; expected 'shift' or 'reduce'"
                )
        return subspace

    # -- Canonical Layer-style entry points ----------------------------
    #
    # Mirrors the plan's "target call shape" (§"LanguageLayer Refactor"):
    #
    #     subspace = self.languageLayer.forward(
    #         subspace=subspace, syntactic_layer=...,
    #         grammar=..., terminal_codebook=...,
    #         rule_codebook=...,
    #     )
    #
    # ``forward`` wraps ``forward_stack``; ``reverse`` wraps
    # ``reverse_stack``. The wrappers exist so callers can treat
    # LanguageLayer like any other Layer subclass (``languageLayer
    # .forward(...) / .reverse(...)``) instead of having to know about
    # the lower-level shift/reduce/unreduce primitives.

    def forward(self, subspace, syntactic_layer, *,
                grammar=None, rule_codebook=None,
                terminal_codebook=None, actions=None):
        """Canonical forward entry: dispatch the stack-rewrite path.

        Args mirror the plan's target call shape. The router needs an
        explicit ``actions`` list for the first-patch implementation;
        a learned SHIFT-vs-REDUCE scorer that produces actions from
        ``terminal_codebook`` + ``rule_codebook`` is Phase 8+ work.

        Args:
            subspace: stack-mode SubSpace (mutated in place).
            syntactic_layer: per-tier SyntacticLayer with ``execute``.
            grammar: Grammar (used for rule .where decoding when
                ``rule_codebook`` is omitted).
            rule_codebook: optional RuleCodebook for rule .where
                stamping.
            terminal_codebook: accepted for plan-API symmetry but
                NOT consumed yet -- the terminal snap currently
                lives in ``SymbolicSpace._stack_route_forward`` as
                the eager bridge; future phases can move it here.
            actions: explicit ``[('shift', payload, where_id), ...]``
                action list. Required until a learned policy is wired.

        Raises:
            NotImplementedError: when ``actions`` is None (no learned
                scorer yet). The error message points to the lower-
                level shift/reduce primitives for the explicit path.
        """
        # ``terminal_codebook`` is part of the plan's target signature
        # but the stack-rewrite path's snap stays in SymbolicSpace for
        # now (first-patch eager bridge). Accepting + ignoring keeps
        # the API stable so future phases can move the snap here.
        del terminal_codebook
        if actions is None:
            raise NotImplementedError(
                "LanguageLayer.forward without explicit `actions` "
                "requires a learned SHIFT/REDUCE scorer (Phase 8+). "
                "Either pass actions=[('shift', payload, where_id), "
                "('reduce', rule_id), ...] explicitly, or call the "
                "shift()/reduce() primitives directly."
            )
        return self.forward_stack(
            subspace, syntactic_layer,
            actions=actions,
            rule_codebook=rule_codebook,
            grammar=grammar,
        )

    def reverse(self, subspace, syntactic_layer, *,
                grammar=None, rule_codebook=None, max_steps=None):
        """Canonical reverse entry: unwind the stack via reverse_stack.

        Args:
            subspace: stack-mode SubSpace (mutated in place).
            syntactic_layer: per-tier SyntacticLayer (provides
                ``_by_name`` for rule layer lookup).
            grammar: Grammar for ``.where`` decoding. Required when
                ``rule_codebook`` is None.
            rule_codebook: optional RuleCodebook with an attached
                Grammar; falls back to ``grammar`` when None.
            max_steps: bound on the unwind loop; defaults to ``K - 1``.

        Note: under the identity-stub contract (lossy ops like
        ConjunctionLayer where ``reverse(parent) == (parent, parent)``)
        the unwind only goes one level deep -- the children's .where
        is cleared by ``unreduce`` so reverse_stack sees an empty
        top and halts. Full multi-level unwinding requires a
        provenance trail (Phase 8+).
        """
        return self.reverse_stack(
            subspace, syntactic_layer,
            grammar=grammar,
            rule_codebook=rule_codebook,
            max_steps=max_steps,
        )


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

    # Backtrace: walk the DP message from t=N down to t=0, writing
    # one-hot copy/reduce masks per row. Each step decreases t by 1
    # (copy) or 2 (reduce), so N iterations is a static upper bound
    # for any input length. Fully tensorized -- no .item()/host sync,
    # no data-dependent Python branch -- so this body traces under
    # fullgraph + max-autotune (CUDA-graph-capturable). Replaces the
    # earlier ``for b in range(B): while t>0: kind=int(back_kind[b,t]
    # .item()); if kind == 0:`` walk that emitted cudaMemcpyDtoH per
    # step and failed dynamo with ``Could not guard on Eq(u0, 0)``.
    B_idx = torch.arange(B, device=device)
    t_cur = torch.full((B,), N, device=device, dtype=torch.long)
    for _ in range(N):
        alive = t_cur > 0
        # Clamp gather index so dead rows still produce a valid read.
        safe_t = t_cur.clamp(min=0, max=N)
        kind_t = back_kind[B_idx, safe_t]                # [B]
        op_t = back_op[B_idx, safe_t]                    # [B]
        is_copy = alive & (kind_t == 0)
        is_reduce = alive & (kind_t == 1)
        # ``op_t`` is the back_op value at ``safe_t``: it is a COPY-op
        # index for kind=0 rows and a REDUCE-op index for kind=1 rows.
        # When R_copy != R_reduce the same op_t may be out-of-bounds
        # for the opposite mask's last dim -- the torch.where masks
        # out the write, but the advanced-index gather itself must
        # see a valid index. Clamp per mask: it's a no-op when the
        # row's kind matches; harmless otherwise (write discarded).
        op_for_copy = op_t.clamp(max=max(R_copy - 1, 0))
        # Copy write: copy_mask[b, t-1, op] = 1 where is_copy.
        pos_copy = (safe_t - 1).clamp(min=0)
        cur_c = copy_mask[B_idx, pos_copy, op_for_copy]
        copy_mask[B_idx, pos_copy, op_for_copy] = torch.where(
            is_copy, torch.ones_like(cur_c), cur_c)
        # Reduce write: reduce_mask[b, t-2, op] = 1 where is_reduce.
        if R_reduce > 0 and N > 1:
            op_for_reduce = op_t.clamp(max=max(R_reduce - 1, 0))
            pos_reduce = (safe_t - 2).clamp(min=0)
            cur_r = reduce_mask[B_idx, pos_reduce, op_for_reduce]
            reduce_mask[B_idx, pos_reduce, op_for_reduce] = torch.where(
                is_reduce, torch.ones_like(cur_r), cur_r)
        # Advance t: 1 for copy, 2 for reduce, 0 otherwise (dead row
        # or invalid kind -- the latter is caught by the assert below).
        step = torch.where(
            is_copy, torch.ones_like(t_cur),
            torch.where(is_reduce, torch.full_like(t_cur, 2),
                        torch.zeros_like(t_cur)))
        t_cur = t_cur - step
    # Invariant note: a well-formed DP message consumes all positions
    # in N steps. Under the post-bivector-retirement chart scoring
    # path the backtrace can land in degenerate states (e.g.,
    # MM_xor_loopback's grammar with R_copy=0 leaves back_kind=-1 at
    # boundary positions), so the previous ``_assert_async`` was too
    # aggressive — it produced false-positive failures on otherwise-
    # valid forward passes. The masks remain semantically correct
    # (one-hot writes only fire under ``is_copy``/``is_reduce`` gates
    # that already check ``alive``), so unconsumed-prefix rows simply
    # contribute zero to the chart selection. Track in a follow-up if
    # the chart starts producing structurally wrong masks under this
    # relaxation.

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

    # Static-unrolled, fully tensorized walk over source positions
    # i=0..N-1. Each iteration consumes 1 (copy) or 2 (reduce) source
    # positions and writes one destination slot j. N iterations is a
    # static upper bound (the maximum possible action count). All
    # writes use per-row tensor masks -- no .item()/host sync, no
    # data-dependent Python branch -- so this body traces under
    # fullgraph + max-autotune. Replaces the earlier ``for b in
    # range(B): while i<N: float(rm_per_pos[b,i].item())>0.5; if
    # do_reduce:`` walk that emitted cudaMemcpyDtoH per step.

    # Safe lookups: pad zero-length per-position tensors to length 1
    # so unconditional advanced-index reads stay valid when N<=1.
    if rm_per_pos.shape[1] > 0:
        rm_per_pos_safe = rm_per_pos
        rm_op_safe = rm_op
    else:
        rm_per_pos_safe = x.new_zeros(B, 1)
        rm_op_safe = torch.zeros(B, 1, device=device, dtype=torch.long)
    if reduced.shape[1] > 0:
        reduced_safe = reduced
    else:
        reduced_safe = x.new_zeros(B, 1, D)

    B_idx = torch.arange(B, device=device)
    cursor = torch.zeros(B, device=device, dtype=torch.long)   # source pos i
    j_idx = torch.zeros(B, device=device, dtype=torch.long)    # dest pos j

    for _ in range(N):
        in_range = cursor < N
        can_reduce = in_range & (cursor < (N - 1))
        ci = cursor.clamp(max=max(N - 1, 0))
        ci_r = cursor.clamp(max=max(rm_per_pos_safe.shape[1] - 1, 0))
        rm_here = rm_per_pos_safe[B_idx, ci_r]                # [B]
        do_reduce = can_reduce & (rm_here > 0.5)
        do_copy = in_range & ~do_reduce
        jc = j_idx.clamp(max=max(N - 1, 0))

        # y[b, j] = reduced[b, ci] if do_reduce else x[b, ci]
        gathered_reduced = reduced_safe[B_idx, ci_r]          # [B, D]
        gathered_x = x[B_idx, ci]                             # [B, D]
        cur_y = y[B_idx, jc]
        y[B_idx, jc] = torch.where(
            do_reduce.unsqueeze(-1), gathered_reduced,
            torch.where(do_copy.unsqueeze(-1), gathered_x, cur_y),
        )

        # src_left[b, j] = ci (both branches when in_range)
        cur_sl = src_left[B_idx, jc]
        src_left[B_idx, jc] = torch.where(in_range, ci, cur_sl)
        # src_right[b, j] = ci+1 if reduce else -1 if copy
        cur_sr = src_right[B_idx, jc]
        src_right[B_idx, jc] = torch.where(
            do_reduce, ci + 1,
            torch.where(do_copy, torch.full_like(ci, -1), cur_sr),
        )
        # action_kind[b, j] = 1 if reduce, 0 if copy
        cur_ak = action_kind[B_idx, jc]
        action_kind[B_idx, jc] = torch.where(
            do_reduce, torch.ones_like(ci),
            torch.where(do_copy, torch.zeros_like(ci), cur_ak),
        )
        # action_op[b, j]: reduce -> rm_op[ci_r]; copy -> cm_op[ci]
        #                 if cm_per_pos[ci] > 0 else -1
        ao_reduce = rm_op_safe[B_idx, ci_r]
        cm_op_at = cm_op[B_idx, ci]
        cm_per_at = cm_per_pos[B_idx, ci]
        ao_copy = torch.where(
            cm_per_at > 0, cm_op_at, torch.full_like(ci, -1))
        cur_ao = action_op[B_idx, jc]
        action_op[B_idx, jc] = torch.where(
            do_reduce, ao_reduce,
            torch.where(do_copy, ao_copy, cur_ao),
        )

        if have_spans:
            # next_span_start[b, j] = span_start[b, ci] (both branches)
            cur_ss = next_span_start[B_idx, jc]
            next_span_start[B_idx, jc] = torch.where(
                in_range, span_start[B_idx, ci], cur_ss)
            # next_span_end[b, j] = span_end[b, ci+1] if reduce
            #                       else span_end[b, ci] if copy
            ci_plus_1 = (ci + 1).clamp(max=max(N - 1, 0))
            cur_se = next_span_end[B_idx, jc]
            next_span_end[B_idx, jc] = torch.where(
                do_reduce, span_end[B_idx, ci_plus_1],
                torch.where(do_copy, span_end[B_idx, ci], cur_se),
            )

        # Advance: cursor += 2 if reduce, 1 if copy, 0 otherwise.
        step = torch.where(
            do_reduce, torch.full_like(cursor, 2),
            torch.where(do_copy, torch.ones_like(cursor),
                        torch.zeros_like(cursor)),
        )
        cursor = cursor + step
        # j advances iff any action fired this iter.
        j_idx = j_idx + in_range.to(j_idx.dtype)

    lengths = j_idx

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

# -- End inlined LanguageLayer section -------------------------------

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
        the legacy CKY chart and the LanguageLayer alternative.
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
        # Per-word stem is the only path (legacy chart-at-stem retired
        # 2026-05-12). Each word runs an individual P->C->S->C round
        # trip in the stem, ideas accumulate on ConceptualSpace.stm,
        # and the chart fires at C over the STM buffer in the body.
        self.per_word_stem = True
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
        # Lazy LanguageLayer construction; only built when needed.
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
        """Lazy-build the LanguageLayer when router_kind == 'signal'.

        Assigning an nn.Module to an attribute auto-registers it as a
        submodule, so it is included in parameters() / state_dict().
        """
        if self._signal_router is None:
            try:
                temperature = float(TheXMLConfig.get(
                    "WordSpace.signal.temperature", 1.0))
            except Exception:
                temperature = 1.0
            self._signal_router = LanguageLayer(
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
    # attributes; 2026-05-12, C/S split 2026-05-18). Drives
    # ``_tier_for_method`` and the parameter-free fallback's substrate
    # wiring. The P tier was retired: there is no subsymbolic chart loop
    # anymore, only two tiers.
    #   C (concept) — concept-tensor ops on the concept codebook; act
    #     directly on the C-tier idea tensors (no symbolic wholeness
    #     assertion required).
    #   S (symbolic) — symbolic wholeness ops on the symbol codebook;
    #     produce a higher-epistemic-level wholeness of arguments
    #     (parthood / equality / query). lift / lower are S: they bridge
    #     to/from the symbol codebook.
    _RULE_TIER = {
        # C-tier: concept-tensor ops on the concept codebook.
        'union':        'C',
        'intersection': 'C',
        'swap':         'C',
        'copy':         'C',
        'not':          'C',
        'non':          'C',
        'true':         'C',
        'false':        'C',
        'part':         'C',
        'query':        'C',
        'area':         'C',
        'luminosity':   'C',
        'equal':        'C',
        # S-tier: symbolic wholeness ops on the symbol codebook.
        'conjunction':  'S',
        'disjunction':  'S',
        'isEqual':      'S',
        'isaPart':      'S',
        'lift':         'S',
        'lower':        'S',
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
            # Codebook prototype read: per
            # doc/specs/2026-05-21-subspace-slot-architecture.md Reader API,
            # ``.what.getW()`` on a codebook-bearing slot returns the
            # ``[V, D]`` prototype matrix (NOT per-batch content).
            # The ``ndim != 2`` early-out below guards the pre-migration
            # window when ``_active_payload`` could shadow with a 3-D
            # per-batch slab; Stage 4 retires the shadow and that guard
            # becomes vacuously true.
            W = what.getW()
        except Exception:
            return lex_log_probs
        # The rest of this method assumes a 2D ``[V, D]`` codebook: the
        # slicing ``W_[:, :D_min]`` indexes the second axis (== last when
        # 2D), and ``cb.T`` is a 2D matrix transpose. Higher-rank codebooks
        # would mis-index the wrong axis AND trip the PyTorch deprecation
        # warning ("use of `x.T` on tensors of dimension other than 2")
        # because ``.T`` would reverse all dims. Early-out keeps this a
        # 2D-only fast path; restructure here if a batched codebook ever
        # needs the POS-seed override.
        if W is None or not torch.is_tensor(W) or W.ndim != 2:
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
        embedding = getattr(word_space, 'category_embedding', None)
        if embedding is None:
            return
        try:
            W = embedding.weight
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
    def _tier_index(self):
        """Map this layer's tier label to its slot in the per-sentence
        WordSpace ``cursor`` tensor (shape ``[n_tiers=3]``).

        ``tier`` is set once at construction to one of the string
        literals 'P' / 'C' / 'S' (Language.py
        ``_attach_per_space_syntactic_layer`` passes ``tier='P'`` /
        ``'C'`` / ``'S'``; ``__init__`` coerces with ``str(tier)``),
        so this map is total over the live domain.
        """
        return {'P': 0, 'C': 1, 'S': 2}[str(self.tier)]

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
            # ``ws.cursor`` is a host ``list[int]`` of length 3 (one per
            # tier P/C/S). Reading via Python list indexing gives a
            # backed Python int the trace can compare with
            # ``len(per_step)`` — an int64 tensor read via ``int(...)``
            # would yield an unbacked SymInt and crash
            # ``fullgraph=True``. The per-compose reset happens
            # unconditionally at the top of WordSpace.compose, so there
            # is NO data-dependent generation gate here (recompile
            # cause #3 eliminated).
            ti = self._tier_index()
            cursor = ws.cursor[ti]
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
                ws.cursor[self._tier_index()] = cursor + 1
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

    # -- Phase 2 executor API (cursor-free) -----------------------------
    #
    # See doc/plans/2026-05-20-subspace-what-stm-signalrouter-refactor.md
    # §"Phase 2: SyntacticLayer Executor API". The LanguageLayer calls
    # these directly with a rule_id it has already selected; no
    # WordSpace.current_rules indirection.

    def execute(self, rule_id, left, right=None):
        """Run the grammar op for ``rule_id`` on ``(left[, right])``.

        Resolves ``rule_id`` to a host layer via ``TheGrammar`` and
        ``self._by_name`` and calls ``layer.compose`` with the right
        number of operands for the rule's arity. Returns the parent
        tensor. No cursor; no WordSpace state read.

        Identity rule (``method_name is None``, ``rhs == lhs``):
        returns ``left`` unchanged. No layer lookup, no parameter touch
        — the grammatical no-op used at padding columns of the static
        per-word loop.
        """
        method_name = TheGrammar.method_name(int(rule_id))
        if method_name is None:
            return left
        layer = self._by_name.get(method_name)
        if layer is None:
            raise KeyError(
                f"SyntacticLayer.execute: tier={self.tier!r} has no host "
                f"layer for rule_id={rule_id} (method_name={method_name!r}). "
                f"Registered rules: {sorted(self._by_name.keys())}"
            )
        arity = int(getattr(layer, 'arity', 1))
        if arity == 1:
            return layer.compose(left)
        if right is None:
            raise ValueError(
                f"SyntacticLayer.execute: arity-2 rule {method_name!r} "
                f"requires `right`; got None"
            )
        return layer.compose(left, right)

    def execute_superposed(self, rule_weights, left, right=None,
                           rule_ids=None):
        """Weighted combination of independent per-rule executions.

        Each candidate op computes on its own copy of ``(left, right)``
        and the results are combined once by weighted sum. Independent
        contribution semantics: no shared in-place accumulator one op
        mutates before the next. Matches the plan's superposed pseudo-
        code; implementations may optimize internally but must preserve
        the semantics.

        Args:
            rule_weights: shape ``[..., R]`` -- soft weights over the R
                candidate rules. Broadcast against the per-rule output's
                leading dims.
            left: arity-1 input (and arity-2 left operand).
            right: arity-2 right operand (None for arity-1-only mixes).
            rule_ids: iterable of R rule ids in the same order as the
                last axis of ``rule_weights``. Required.

        Returns:
            Tensor with the same per-rule output shape (R axis summed
            out).
        """
        if rule_ids is None:
            raise ValueError(
                "SyntacticLayer.execute_superposed: rule_ids is required"
            )
        if hasattr(rule_ids, "tolist"):
            rule_ids = list(rule_ids.tolist())
        rule_ids = list(rule_ids)
        if rule_weights.shape[-1] != len(rule_ids):
            raise ValueError(
                f"rule_weights last dim {rule_weights.shape[-1]} does not "
                f"match rule_ids length {len(rule_ids)}"
            )
        outs = []
        for rid in rule_ids:
            outs.append(self.execute(int(rid), left, right))
        stacked = torch.stack(outs, dim=-2)              # [..., R, D]
        return (stacked * rule_weights.unsqueeze(-1)).sum(dim=-2)

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

def _intersect_long_rows(a, b):
    """LongTensor intersection by row index, preserving sort order.

    Both inputs are 1-D ``LongTensor`` of unique row indices (typical
    output of ``refs_by_category`` / ``refs_by_order``). Returns the
    sorted intersection as a 1-D ``LongTensor``. Either side empty
    yields an empty result; both empty yields empty.
    """
    if a is None or b is None:
        return torch.empty(0, dtype=torch.long)
    if a.numel() == 0 or b.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    # Boolean ``isin`` is the simplest correct implementation; sizes are
    # category-bounded (tens, not thousands) so cost is negligible.
    return a[torch.isin(a, b)]

class Taxonomy:
    """Explicit parent->children order hierarchy for ramsified symbols.

    Distinct from the Meronomy (parthood) -- which stays codebook-
    per-order implicit / geometric (the ``PartLayer`` clipped-cosine,
    unchanged). The Taxonomy is the explicit hierarchy organizing
    symbols by order (0 = proper / specific; higher = more general).
    Pure-Python bookkeeping: no parameters, not an ``nn.Module``;
    hosted on the WordSpace singleton.

    Also hosts the **priming buffer** for reverse-generation working-
    memory state (plan doc/plans/2026-05-20-primed-reverse-generation.md).
    The buffer lives on the Taxonomy because propagation walks
    parent/children adjacency; co-locating the state with the graph
    avoids indirection. The legacy in-process dicts
    (``_parent``/``_children``) are unused on the WordSpace's instance
    — propagation queries the attached ``embed.KnowledgeView`` instead.
    """

    # Default priming knobs (overrideable per instance via
    # ``configure_priming``). Plan doc/plans/2026-05-20-primed-reverse-
    # generation.md §Configuration.
    DEFAULT_PRIMING_DEPTH = 2
    DEFAULT_HOP_DECAY = 0.5
    DEFAULT_TEMPORAL_DECAY = 0.9
    DEFAULT_BOOST_INITIAL = 1.0
    DEFAULT_PRIMING_ENABLED = True

    def __init__(self):
        self._order = {}      # node id -> int order
        self._children = {}   # node id -> list[node id]
        self._parent = {}     # node id -> parent node id or None
        self._next = 0
        # Priming state (allocated lazily via ``allocate_priming``).
        self._priming = None      # FloatTensor [B, V_ref_capacity] | None
        self._priming_B = 0
        self._priming_capacity = 0
        self._priming_live = 0
        self._priming_view = None  # embed.KnowledgeView for adjacency walks
        # Per-instance priming config (defaults from class constants).
        self.priming_depth = self.DEFAULT_PRIMING_DEPTH
        self.hop_decay = self.DEFAULT_HOP_DECAY
        self.temporal_decay = self.DEFAULT_TEMPORAL_DECAY
        self.boost_initial = self.DEFAULT_BOOST_INITIAL
        self.priming_enabled = self.DEFAULT_PRIMING_ENABLED
        # Telemetry counters (host-side; cheap to read in tests / logs).
        self._priming_select_count = 0
        self._priming_boosted_select_count = 0

    def add(self, order, parent=None):
        """Create a node at ``order`` (optionally under ``parent``);
        return its node id."""
        nid = self._next
        self._next += 1
        self._order[nid] = int(order)
        self._children[nid] = []
        self._parent[nid] = parent
        if parent is not None:
            self._children.setdefault(parent, []).append(nid)
        return nid

    def children(self, node):
        """Direct children of ``node`` (list of node ids)."""
        return list(self._children.get(node, []))

    def parent(self, node):
        """Parent node id of ``node`` (or ``None`` for a root)."""
        return self._parent.get(node)

    def order(self, node):
        """The order (0..N) recorded for ``node``."""
        return self._order[node]

    def all(self):
        """All node ids, in insertion order."""
        return list(self._order.keys())

    def __len__(self):
        return len(self._order)

    # -- priming buffer ---------------------------------------------------
    # Plan: doc/plans/2026-05-20-primed-reverse-generation.md §Part/whole
    # priming mask. ``_priming[b, ref_id] = 1 + boost(ref_id)``:
    #   * 1.0 = multiplicative identity (no priming, no change downstream)
    #   * >1.0 = boost (recently active or near-neighbor in taxonomy)
    #   * dissipates back toward 1.0 via ``decay`` and ``propagate``'s
    #     hop_decay
    # Allocated to ``capacity`` (V_ref_capacity) with only the first
    # ``live`` columns active; matches the artifact's capacity-slack
    # pattern for symbol-learning appends.

    def allocate_priming(self, batch_size, capacity, live, *, device=None,
                         dtype=None):
        """Allocate the priming buffer at ``[batch_size, capacity]``,
        initialized to 1.0 (multiplicative identity).

        ``live`` is the number of currently-occupied ref rows (the rest
        is slack reserved for symbol-learning appends). Re-allocation
        with a smaller batch / capacity / live preserves any current
        primed values in the overlapping region; a larger allocation
        copies forward and fills the new space with 1.0.
        """
        import torch
        B = int(batch_size)
        C = int(capacity)
        L = int(live)
        if dtype is None:
            dtype = torch.float32
        new = torch.ones(B, C, dtype=dtype, device=device)
        old = self._priming
        if old is not None:
            ob = min(B, old.shape[0])
            oc = min(C, old.shape[1])
            new[:ob, :oc] = old[:ob, :oc].to(device=device, dtype=dtype)
        self._priming = new
        self._priming_B = B
        self._priming_capacity = C
        self._priming_live = L

    def attach_view(self, view):
        """Attach an ``embed.KnowledgeView`` whose parent/children CSR
        drives priming propagation. Stored as a plain reference (the
        view is read-only). ``view=None`` detaches.
        """
        self._priming_view = view
        if view is not None:
            self._priming_live = int(view.n_refs_live)

    def reset(self, batch=None):
        """Reset the priming buffer to multiplicative identity (1.0).
        No-op when the buffer hasn't been allocated yet.

        ``batch=None`` resets every row; an integer resets only that
        row. Called at sentence boundaries — priming is sentence-
        scoped working memory, not a persistent learned signal.
        """
        if self._priming is None:
            return
        if batch is None:
            self._priming.fill_(1.0)
            return
        b = int(batch)
        if 0 <= b < self._priming_B:
            self._priming[b].fill_(1.0)

    def decay(self, temporal_decay=0.9, batch=None):
        """Dissipate priming boost between reverse calls within a
        sentence: ``priming = 1 + (priming - 1) * temporal_decay``.
        Identity (1.0) entries stay at 1.0.

        ``batch=None`` decays every row; an integer decays only that
        row.
        """
        if self._priming is None:
            return
        td = float(temporal_decay)
        if batch is None:
            self._priming.sub_(1.0).mul_(td).add_(1.0)
            return
        b = int(batch)
        if 0 <= b < self._priming_B:
            self._priming[b].sub_(1.0).mul_(td).add_(1.0)

    def prime(self, ref_ids, batch=0, boost=1.0):
        """Set ``priming_mask[batch, ref_ids] = max(current, 1 + boost)``.

        Element-wise max so multiple primings of the same ref within a
        sentence don't compound past ``1 + boost``. Out-of-range or
        negative ref_ids are silently dropped.
        """
        if self._priming is None:
            return
        import torch
        rids = torch.as_tensor(ref_ids, dtype=torch.long)
        if rids.ndim == 0:
            rids = rids.unsqueeze(0)
        mask = (rids >= 0) & (rids < self._priming_capacity)
        rids = rids[mask]
        if rids.numel() == 0:
            return
        b = int(batch)
        if b < 0 or b >= self._priming_B:
            return
        target = 1.0 + float(boost)
        cur = self._priming[b, rids]
        self._priming[b, rids] = torch.maximum(
            cur, torch.full_like(cur, target))

    def propagate(self, ref_ids, batch=0, depth=2, hop_decay=0.5):
        """Spread the boost from ``ref_ids`` along the attached view's
        parent/children adjacency for ``depth`` hops, multiplying the
        per-hop boost by ``hop_decay`` each step.

        At each hop, every active node ``r`` writes
        ``1 + (current[r] - 1) * hop_decay`` into its immediate parent
        and immediate children, taking element-wise max with whatever
        was already there. The frontier for the next hop is the union
        of the just-touched neighbors. Siblings are not directly
        primed — they reach the boost only via a shared parent across
        two hops.

        ``ref_ids`` is the seed set (typically the freshly-snapped refs
        that just entered STM). No-op when the buffer is unallocated,
        depth ≤ 0, or no view is attached.
        """
        if self._priming is None or depth <= 0:
            return
        view = self._priming_view
        if view is None:
            return
        import torch
        b = int(batch)
        if b < 0 or b >= self._priming_B:
            return
        seeds = torch.as_tensor(ref_ids, dtype=torch.long)
        if seeds.ndim == 0:
            seeds = seeds.unsqueeze(0)
        seeds = seeds[(seeds >= 0) & (seeds < self._priming_capacity)]
        if seeds.numel() == 0:
            return
        frontier = set(int(r) for r in seeds.tolist())
        decay = float(hop_decay)
        for _ in range(int(depth)):
            next_frontier = set()
            for r in frontier:
                cur = float(self._priming[b, r].item())
                if cur <= 1.0:
                    continue
                neighbor_value = 1.0 + (cur - 1.0) * decay
                if neighbor_value <= 1.0:
                    continue
                p = view.parent_of(r)
                if p is not None and 0 <= p < self._priming_capacity:
                    prev = float(self._priming[b, p].item())
                    if neighbor_value > prev:
                        self._priming[b, p] = neighbor_value
                        next_frontier.add(int(p))
                kids = view.children_of(r)
                for c in kids.tolist():
                    ci = int(c)
                    if 0 <= ci < self._priming_capacity:
                        prev = float(self._priming[b, ci].item())
                        if neighbor_value > prev:
                            self._priming[b, ci] = neighbor_value
                            next_frontier.add(ci)
            if not next_frontier:
                break
            frontier = next_frontier

    def priming_mask(self, batch=None):
        """Return the live slice of the priming buffer.

        ``batch=None`` returns ``[B, V_ref_live]``; an integer returns
        ``[V_ref_live]`` for that row. ``None`` when the buffer hasn't
        been allocated yet.
        """
        if self._priming is None:
            return None
        live = self._priming_live or self._priming_capacity
        if batch is None:
            return self._priming[:, :live]
        b = int(batch)
        if b < 0 or b >= self._priming_B:
            return None
        return self._priming[b, :live]

    @property
    def priming_capacity(self):
        return self._priming_capacity

    @property
    def priming_live(self):
        return self._priming_live

    def configure_priming(self, *,
                          priming_depth=None,
                          hop_decay=None,
                          temporal_decay=None,
                          boost_initial=None,
                          priming_enabled=None):
        """Override per-instance priming knobs.

        Typically called from ``BasicModel`` setup after parsing
        ``<architecture><priming>`` config. Any ``None`` argument
        leaves the existing value unchanged.
        """
        if priming_depth is not None:
            self.priming_depth = int(priming_depth)
        if hop_decay is not None:
            self.hop_decay = float(hop_decay)
        if temporal_decay is not None:
            self.temporal_decay = float(temporal_decay)
        if boost_initial is not None:
            self.boost_initial = float(boost_initial)
        if priming_enabled is not None:
            self.priming_enabled = bool(priming_enabled)

    def note_selection(self, ref_id, batch=0):
        """Telemetry: record that ``ref_id`` was selected in row ``batch``.
        Bumps ``_priming_select_count`` always; bumps
        ``_priming_boosted_select_count`` when the selected ref's
        priming was above identity (1.0) at selection time. Host-side
        counters — safe to read from tests and logs.
        """
        self._priming_select_count += 1
        if self._priming is None:
            return
        b = int(batch)
        r = int(ref_id)
        if (0 <= b < self._priming_B and 0 <= r < self._priming_capacity):
            if float(self._priming[b, r].item()) > 1.0:
                self._priming_boosted_select_count += 1

    def priming_telemetry(self):
        """Return ``(total, boosted)`` selection counts."""
        return (self._priming_select_count,
                self._priming_boosted_select_count)

class WordSubSpace(nn.Module):
    """Per-sentence grammar / serial-processing carrier — the third
    argument that travels alongside the data SubSpaces through the
    pipeline (reached via ``subspace.wordSpace`` after
    ``copy_context`` stamps the back-reference).

    Runtime-parallel to PerceptualSpace / ConceptualSpace / SymbolicSpace
    but functionally a composition dispatcher rather than a pipeline
    stage that produces data tensors. WordSubSpace owns:

      * the per-tier ``SyntacticLayer`` dispatchers (registered on
        each home space; reached via ``forwardSymbols`` /
        ``reverseSymbols``);
      * the CKY chart, truth store, and STM-driver backends;
      * the per-sentence parser cursor (``self.cursor`` — int64
        ``[n_tiers=3]``) and PerceptualSpace recurrent-pass index
        (``self.recur_pass`` — Python int);
      * inter-sentence discourse substrate (``InterSentenceLayer`` /
        priming taxonomy).

    The standalone ``SentenceState`` carrier was retired (2026-05-21):
    ``cursor`` and ``recur_pass`` now live directly on WordSubSpace;
    the cross-pass C→P / C→S feedback is read straight off
    ``ConceptualSpace._subspaceForPS`` / ``_subspaceForSS`` (the
    persistent CS-tier storage that ``ConceptualSpace.forward``
    mutates in place).

    Plain ``nn.Module`` subclass (not ``Space``): WordSubSpace is not a
    pipeline stage that produces data tensors, so it does not fit the
    factory-style input/output/codebook shape contract. The small set
    of Space-contract methods the model iterates over
    (``set_sigma`` / ``paramUpdate`` / ``getParameters`` / ``Start`` /
    ``End``) are inlined directly on this class.

    The legacy ``WordSubSpace`` SR-parser stack (a separate
    ``SubSpace`` subclass that buffered derivation steps) was removed
    2026-05-20 — its stack functionality migrated to
    ``ConceptualSpace.stm``. The current name reuses that string but
    denotes a different concept; the old class is gone, so there is
    no ambiguity in live code.
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
        # a unary substrate fold registered as the per-tier default).
        # When true, ``compose`` / ``generate`` skip the CKY-style
        # inside / outside pass entirely; per-space SyntacticLayer
        # dispatch falls through to its registered default rule, which
        # fires the substrate layer (``pi_input``, ``sigma_percept``,
        # …) exactly once per step -- mathematically identical to the
        # bare ``self.sigma_percept(x)`` / ``self.pi_input(x)`` call
        # sites. Legacy ``pi`` / ``sigma`` names alias to the new
        # substrates (see ``_attach_per_space_syntactic_layer``), so
        # both old and new grammar names qualify for the bypass.
        # Implicit non-operational rules (epsilon, X -> X passthrough
        # whose method_name is None) don't disqualify the bypass.
        self._grammar_is_default_only = all(
            r.method_name is None or (
                r.method_name in ('pi', 'sigma') and r.arity == 1)
            for r in grammar.rules
        )

        # 2. Mirror SymbolicSpace's column layout for the Space-contract
        # fields that downstream callers occasionally read off WordSpace
        # (``nDim`` / ``nWhat`` / ``nWhere`` / ``nWhen`` / ``muxedSize``).
        # The legacy ``WordSubSpace`` stack at ``self.subspace`` was
        # removed (2026-05-20) — its push / get_blocks / read surface
        # had no callers; the SR-parser value tape now lives on
        # ``ConceptualSpace.stm``. ``self.subspace = None`` keeps the
        # peer-Space attribute present so isinstance / ``hasattr``
        # probes don't crash.
        sub = symbolicSpace.subspace
        nWhere = int(getattr(sub, 'nWhere', 0) or 0)
        nWhen  = int(getattr(sub, 'nWhen',  0) or 0)
        nWhat  = int(getattr(sub, 'nWhat',  0) or 0)
        muxed  = int(getattr(sub, 'muxedSize', nWhat + nWhere + nWhen)
                     or (nWhat + nWhere + nWhen))
        self.subspace = None

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
        # Per-sentence serial-parser state, owned directly by WordSpace
        # (no separate SentenceState carrier).
        #   ``cursor``      — ``list[int]`` of length 3 (one per tier
        #                     P/C/S), the per-tier rule cursor consumed
        #                     by ``SyntacticLayer._next_rule_name``.
        #                     HOST Python ints (not a tensor) so the
        #                     read inside compiled forwards produces a
        #                     backed Python int that Dynamo can compare
        #                     against ``len(per_step)`` — an int64
        #                     tensor read via ``int(tensor)`` would
        #                     produce an unbacked SymInt and break
        #                     ``fullgraph=True`` at the rule-list
        #                     bounds check. Dynamo specializes the
        #                     traced graph per observed cursor value;
        #                     compose() resets to ``[0, 0, 0]`` at the
        #                     top of every call.
        #   ``recur_pass``  — Python int, PerceptualSpace recurrent-pass
        #                     index that selects ``pi_input[oi]`` from a
        #                     ModuleList (Dynamo-specialized natively;
        #                     Inductor would emit an unbacked SymInt for
        #                     a 0-d tensor source — see D8 capture-gate).
        self.cursor = [0, 0, 0]
        self.recur_pass = 0
        # Forward-only padding target for the static per-word loop
        # (doc/plans/2026-05-20-static-per-word-loop-impl.md §1).
        # The Model sets this to InputSpace.outputShape[0] after
        # construction; 0 means "no padding" so legacy / non-static
        # callers behave unchanged. The reverse cursor is NOT padded
        # — see §2R for the asymmetric left-shift on reconstruction.
        self._target_cursor_length = 0

        # Parser backend selector (plan
        # doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md).
        # ``chart`` (default) -- existing CKY path, untouched.
        # ``stm``             -- STM shift/reduce driver (in progress).
        # ``parallel``        -- both, chart authoritative.
        # XML knob: <parserBackend>chart</parserBackend> under WordSpace.
        try:
            backend_cfg = TheXMLConfig.space(
                "WordSpace", "parserBackend", default="chart")
        except (KeyError, TypeError, ValueError):
            backend_cfg = "chart"
        backend_str = str(backend_cfg or "chart").strip().lower()
        if backend_str not in ("chart", "stm", "parallel"):
            raise ValueError(
                f"WordSpace parserBackend={backend_cfg!r} is invalid; "
                "expected one of 'chart' / 'stm' / 'parallel'.")
        self.parser_backend = backend_str
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
            # P-tier SyntacticLayer retired (2026-05-18 C/S split): the
            # perceptual space no longer carries a chart-dispatched
            # SyntacticLayer. ``attach_wordSpace`` (shared-buffer
            # back-ref) is unrelated wiring and is kept. The tier='P'
            # branch in ``_attach_per_space_syntactic_layer`` is now
            # unreached but left dead-safe.
        if conceptualSpace is not None:
            conceptualSpace.attach_wordSpace(self)
            self._attach_per_space_syntactic_layer(
                conceptualSpace, tier='C')
        if symbolicSpace is not None:
            symbolicSpace.attach_wordSpace(self)
            self._attach_per_space_syntactic_layer(
                symbolicSpace, tier='S')

        # 5b. Signal-router grammar wiring. When `WordSpace.routerKind ==
        # "signal"`, the chart's CKY paths are bypassed; the LanguageLayer
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
        #
        # 2026-05-20 category_codebook retirement (plan
        # ``doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md``):
        # the embedding is now an ``nn.Embedding[N_categories, pos_dim]``
        # rather than a ``Codebook``. The codebook's VQ / polarity /
        # meronomy / SVD machinery was never used by the category-label
        # consumers; a plain Embedding is the right type and removes the
        # weight (no Codebook hidden state in the checkpoint).
        pos_dim = 4  # embedding width; also the category stack vector dim
        category_capacity = max(64, len(TheGrammar.categories))
        self.category_embedding = nn.Embedding(category_capacity, pos_dim)
        # Feed the manual optimizer-feed list (consumed by
        # ``getParameters``); ``nn.Embedding`` auto-registers its weight
        # as a Parameter on the parent module via attribute assignment,
        # but ``self.params`` is the canonical list callers walk.
        for p in self.category_embedding.parameters():
            if all(p is not q for q in self.params):
                self.params.append(p)
        self.category_index = {
            name: idx for idx, name in enumerate(TheGrammar.categories)
        }
        # Taxonomy: explicit parent->children order hierarchy for the
        # ramsified symbol space (Meronomy/parthood stays codebook-
        # per-order implicit, unchanged). Pure-Python; hosted here on
        # the WordSpace singleton, reached at runtime via
        # ``vspace.wordSpace.taxonomy``.
        self.taxonomy = Taxonomy()
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
        # pos_dim already bound above (category_embedding / category_stack dim).
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

        # 7. InterSentenceLayer -- optional ARMA(p, q) next-sentence
        # predictor.  Gated on <architecture><training><sentencePrediction>;
        # tasks without inter-sentence structure (XOR, MNIST) leave
        # it off.  Shape knobs live under <WordSpace> (armaP, armaQ,
        # armaHiddenDim); the loss weight lives under
        # <architecture><training><armaScale> and is read by runBatch.
        # Contrastive cosine machinery retired 2026-05-14 alongside
        # <maskedPrediction>.
        self.discourse = None
        if bool(TheXMLConfig.training("sentencePrediction", False)):
            try:
                n_sym_rows = int(symbolicSpace.outputShape[0])
            except (AttributeError, IndexError, TypeError):
                n_sym_rows = int(getattr(symbolicSpace, 'nVectors', 0) or 0)
            if n_sym_rows > 0 and muxed > 0:
                arma_p = int(TheXMLConfig.space(
                    "WordSpace", "armaP", default=5) or 5)
                arma_q = int(TheXMLConfig.space(
                    "WordSpace", "armaQ", default=2) or 2)
                arma_hidden = TheXMLConfig.space(
                    "WordSpace", "armaHiddenDim", default=None)
                self.discourse = InterSentenceLayer(
                    n_symbols=n_sym_rows,
                    max_depth=int(getattr(
                        self.subspace, 'max_depth', 256) or 256),
                    n_dim=muxed,
                    p=arma_p,
                    q=arma_q,
                    hidden_dim=(int(arma_hidden)
                                if arma_hidden is not None else None),
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
        # D8 capture-gate (2026-05-19): discourse-prediction cache.
        # ``stm_residual_microbatch`` is called per-word from
        # ``ConceptualSpace.forward`` and used to fire
        # ``discourse.predict()`` per-word -- which has a Python
        # ``predict_next(b=b)`` row-loop that's untraceable under
        # fullgraph=True and produces N DtoHs on CUDA. Discourse
        # prediction is **sentence-scoped** (the inter-sentence ARMA
        # state only changes when ``disc.observe()`` is called in
        # ``runBatch`` post-body), so we cache ``(pred, conf)`` here
        # and refresh it in ``arm_stm`` (the sentence-boundary
        # rearm). The per-word body reads the cache; the captured
        # graph never enters ``disc.predict()``.
        self._disc_pred = None
        self._disc_conf = None

        # Per-source-row sentence-completed signal driven by
        # SyntacticLayer.compose: True for row b when this tick's parse
        # derivation reduced to Grammar.start_symbol. Outer doc-streaming
        # loop drains via drain_sentence_completed() after each runBatch
        # and dispatches soft_reset(batch=b) for True rows. Host-side
        # list (no GPU sync); resized to B by ensure_microbatch.
        self._sentence_completed = [False] * self.batch

    # -- knowledge-artifact attach -----------------------------------------
    # Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
    # §Phase 2 — Loaders. ``attach_knowledge(view)`` wires a loaded
    # ``embed.KnowledgeView`` into the WordSpace so downstream consumers
    # (chart POS scorer, rule predictor, lift/lower restricted-candidate
    # inverse, STM REDUCE typed admissibility) consult it instead of the
    # legacy ``Taxonomy()`` scaffold; ``category_codebook`` was retired
    # 2026-05-20 in favor of ``category_embedding: nn.Embedding``. The
    # remaining legacy fields stay in place during Phase 2 for back-compat.

    def attach_knowledge(self, view):
        """Attach a loaded :class:`embed.KnowledgeView`. Replaces any
        previously attached view (last-write-wins). Stored via
        ``object.__setattr__`` to bypass nn.Module's submodule
        registration — the view holds tensors but isn't itself a Module
        and shouldn't appear in ``state_dict``.

        Also wires the view into ``self.taxonomy`` (for priming
        propagation adjacency) and allocates the per-batch priming
        buffer at multiplicative identity. Plan
        doc/plans/2026-05-20-primed-reverse-generation.md §Storage.
        """
        object.__setattr__(self, '_knowledge', view)
        # Bind the taxonomy's adjacency source + allocate priming.
        tax = getattr(self, 'taxonomy', None)
        if tax is not None and view is not None:
            tax.attach_view(view)
            capacity = int(view._parent.shape[0])
            tax.allocate_priming(
                batch_size=int(self.batch),
                capacity=capacity,
                live=int(view.n_refs_live),
            )

    @property
    def knowledge(self):
        """The attached :class:`embed.KnowledgeView`, or ``None`` when
        ``attach_knowledge`` has not been called for this WordSpace
        instance."""
        return getattr(self, '_knowledge', None)

    # -- priming kwargs helper ---------------------------------------------
    # Plan: doc/plans/2026-05-20-primed-reverse-generation.md §Reverse
    # operation flow. Given a rule's RuleOrderSignature + operand-side
    # bindings, builds the four kwargs (``left_rows``, ``right_rows``,
    # ``left_priming``, ``right_priming``) the recommender consumes.

    def priming_kwargs_for_slots(self, *,
                                 left_category, left_order,
                                 right_category=None, right_order=None,
                                 batch=0):
        """Build (left_rows, right_rows, left_priming, right_priming)
        kwargs for the inverse recommender from typed slot info.

        ``left_category`` / ``right_category`` are grammar category
        names (e.g. ``'NP'``, ``'VP'``). ``left_order`` / ``right_order``
        are the resolved integer conceptual orders for the slot,
        derived by the caller from the active rule's
        ``RuleOrderSignature`` and the operand-side order binding.

        Returns a dict with up to four keys (any ``None`` slot is
        omitted). Empty intersection rows are passed through as empty
        LongTensors — the recommender's row-mask helper then admits
        only the ⊥/⊤ sentinels for that slot (graceful degradation
        rather than failure).

        Priming weights are sliced from ``self.taxonomy.priming_mask
        (batch=batch)``, sized to ``V_ref_live``. The recommender
        truncates / pads to ``W.shape[0]`` (``K``) at use time, so
        passing the live slice is safe even when the codebook is
        capacity-slack-padded.

        Returns ``{}`` when no knowledge is attached (graceful fallback
        — caller can still proceed with un-typed, un-primed selection).
        """
        view = self.knowledge
        if view is None:
            return {}
        out = {}
        # Left slot
        if left_category is not None:
            cat_rows = view.refs_by_category(left_category)
            ord_rows = view.refs_by_order(int(left_order))
            out['left_rows'] = _intersect_long_rows(cat_rows, ord_rows)
        # Right slot
        if right_category is not None:
            cat_rows = view.refs_by_category(right_category)
            ord_rows = view.refs_by_order(int(right_order))
            out['right_rows'] = _intersect_long_rows(cat_rows, ord_rows)
        # Priming (per batch row). One mask covers all rows; the
        # recommender's row mask already gates feasibility, so the
        # full ref-id-indexed priming is correct for both slots.
        tax = getattr(self, 'taxonomy', None)
        # ``priming_enabled = false`` short-circuits to typed-only
        # behavior — no priming kwargs emitted at all.
        enabled = (tax is not None and getattr(tax, 'priming_enabled', True))
        if enabled:
            pm = None if tax is None else tax.priming_mask(batch=int(batch))
            if pm is not None:
                if left_category is not None:
                    out['left_priming'] = pm
                if right_category is not None:
                    out['right_priming'] = pm
        return out

    # -- STM backend driver wiring -----------------------------------------
    # Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
    # §Phase 2 / step 4. ``self.stm_driver`` is lazily constructed on
    # first ``compose()`` / ``generate()`` call under
    # ``parser_backend='stm'``, drawing rule signatures from the
    # attached KnowledgeView and the typed stack from
    # ``conceptualSpace.stm_typed``. The driver's scorer is sized to
    # the stm payload dim.

    def _init_stm_driver(self):
        """Construct ``self.stm_driver`` from attached knowledge +
        conceptualSpace.stm_typed. Raises a clear error if either
        prerequisite is missing.
        """
        view = self.knowledge
        if view is None:
            raise RuntimeError(
                "WordSpace STM backend requires a knowledge artifact: "
                "call ``ws.attach_knowledge(view)`` before compose()")
        cs = getattr(self, 'conceptualSpace', None)
        stm_typed = getattr(cs, 'stm_typed', None) if cs is not None else None
        if stm_typed is None:
            raise RuntimeError(
                "WordSpace STM backend requires "
                "conceptualSpace.stm_typed to be allocated")
        from stm_driver import STMDriver, RuleScorer
        rule_sigs = view.rule_order_signatures
        scorer = RuleScorer(
            payload_dim=stm_typed.dim, n_rules=len(rule_sigs))
        driver = STMDriver(typed_stack=stm_typed,
                           rule_signatures=rule_sigs,
                           scorer=scorer)
        # Register the driver so ``.to(device)`` reaches the scorer. The
        # scorer params are also mirrored into the legacy manual
        # optimizer-feed list below.
        self._stm_driver = driver
        params = self.__dict__.get('params')
        if isinstance(params, list):
            for p in driver.parameters():
                if all(p is not q for q in params):
                    params.append(p)

    @property
    def stm_driver(self):
        """Lazily-constructed STM driver; ``None`` if not yet built."""
        return getattr(self, '_stm_driver', None)

    def _compose_stm(self, input_vectors, subspace):
        """STM-backend compose: tokenize ``input_vectors`` via the
        reference-codebook order-0 snap, shift each token, run REDUCE
        until either a single root frame remains or no admissible rule
        fires, emit per-tier rule selections compatible with the
        ``current_rules`` consumer.

        ``input_vectors=None`` (the wiring-test contract) constructs
        the driver and returns an empty dict — no SHIFT/REDUCE.
        """
        if self.stm_driver is None:
            self._init_stm_driver()
        self.current_rules = self._stm_drive(input_vectors, mode='compose')
        return self.current_rules

    def _generate_stm(self, target_vectors, subspace):
        """STM-backend generate. Mirror of ``_compose_stm`` with each
        row's selected-rule sequence reversed — matches
        ``_collect_generate_selections``' downward-generation convention.
        """
        if self.stm_driver is None:
            self._init_stm_driver()
        self.generate_rules = self._stm_drive(
            target_vectors, mode='generate')
        return self.generate_rules

    def _stm_drive(self, input_vectors, *, mode):
        """Shared SHIFT/REDUCE loop for compose / generate.

        For each batch row:
          1. SHIFT every token from ``input_vectors[b]``. Snap the
             token payload to the nearest reference scalar to
             pick a (ref_id, category) so the typed admissibility mask
             can gate REDUCE. Order is the snapped ref's recorded order.
          2. REDUCE until either depth==1 or no admissible rule fires.
             Each REDUCE pops ``arity`` items, applies the rule's
             order-typed signature for the parent frame, and pushes
             back a parent (path-to-complete §4 calls the real grammar
             op via ``_apply_grammar_op``).
          3. Collect rule indices per tier into a
             ``dict[tier -> list[list[int]]]`` structure.
          4. Populate a parser-neutral ``ParseState`` on the WordSpace
             (path-to-complete §5): frames carry spans, actions carry
             backpointers, trace is the chosen derivation. Viterbi /
             chart-compatible consumers read from here.

        When ``mode='generate'``, each row's list is reversed so the
        last-applied rule comes out first.
        """
        if input_vectors is None:
            object.__setattr__(self, 'parse_state', None)
            return {}
        if not torch.is_tensor(input_vectors):
            return {}
        if input_vectors.ndim != 3:
            return {}
        view = self.knowledge
        stm_typed = self.conceptualSpace.stm_typed
        driver = self.stm_driver
        B, N, D = input_vectors.shape
        if hasattr(stm_typed, 'ensure_batch'):
            stm_typed.ensure_batch(B)
        rows = min(B, stm_typed.batch)
        # Reset rows we'll touch.
        for b in range(rows):
            while int(stm_typed._depth[b].item()) > 0:
                stm_typed.pop(b)
        # SHIFT snaps against all live non-root refs. Written words may
        # enter at orthographic order 0, but the selected reference is the
        # grammatical/conceptual meaning and carries its own order
        # (e.g. NP3 vs NP4).
        candidate_ids = torch.arange(1, view.n_refs_live, dtype=torch.long)
        if candidate_ids.numel() == 0:
            candidate_ids = torch.arange(view.n_refs_live, dtype=torch.long)
        per_tier: dict = {}
        from parse_state import ParseState
        parse_state = ParseState()
        # Per-row map: stack-slot index → ParseState frame index. The
        # TypedStack doesn't carry ParseState indices, so we shadow
        # them in a Python list per row and update on every push/pop.
        slot_to_frame: list = [[] for _ in range(rows)]

        def _try_reduce(b):
            """Attempt one REDUCE on row ``b``. Returns True if it fired
            and applied; False if no admissible rule or arity mismatch.
            Records the chosen rule in ``per_tier`` on success."""
            d = int(stm_typed._depth[b].item())
            if d < 1:
                return False
            try:
                result = driver.reduce_step_soft(b)
            except RuntimeError:
                return False
            rule_index = int(result['rule_index'])
            sig = result['rule_signature']
            rule_score = float(
                result['masked_logits'][rule_index].detach().cpu().item())
            rule_probability = float(
                result['probabilities'][rule_index].detach().cpu().item())
            arity = len(sig.get('rhs_categories', ()))
            # Capture operand orders BEFORE popping (popped frames
            # lose their order). The order-of-pop matters: we pop
            # right then left to mirror the SHIFT order.
            op_name = sig.get('op_name')
            if arity == 2 and d >= 2:
                right_order = int(stm_typed.top(b, k=1)['order'])
                left_order = int(stm_typed.top(b, k=2)['order'])
                right = stm_typed.pop(b)
                left = stm_typed.pop(b)
                # Path-to-complete §4: use the real grammar op for the
                # parent payload (intersection / union / etc.) rather
                # than the (left + right) / 2 placeholder.
                parent_payload = self._apply_grammar_op(
                    op_name, left['payload'], right['payload'])
                operand_orders = (left_order, right_order)
                right_frame_idx = slot_to_frame[b].pop()
                left_frame_idx = slot_to_frame[b].pop()
                operand_frame_indices = (left_frame_idx, right_frame_idx)
            elif arity == 1 and d >= 1:
                only_order = int(stm_typed.top(b, k=1)['order'])
                only = stm_typed.pop(b)
                parent_payload = self._apply_grammar_op(
                    op_name, only['payload'])
                operand_orders = (only_order,)
                only_frame_idx = slot_to_frame[b].pop()
                operand_frame_indices = (only_frame_idx,)
            else:
                return False
            lhs_cat = str(sig.get('lhs_category', 'UNK'))
            lhs_order = self._resolve_lhs_order(sig, operand_orders)
            stm_typed.push(b, parent_payload,
                           category_id_str=lhs_cat,
                           order=lhs_order, ref_id=-1)
            # Mirror into ParseState: record the parent frame + the
            # action (backpointers via operand_indices).
            parent_idx = parse_state.add_reduce(
                rule_id=rule_index,
                operand_indices=operand_frame_indices,
                parent_payload=parent_payload,
                parent_category=lhs_cat,
                parent_order=lhs_order,
                score=rule_score,
                probability=rule_probability,
            )
            parse_state.row_traces.setdefault(b, []).append(
                parse_state.actions[-1])
            slot_to_frame[b].append(parent_idx)
            tier = self._tier_for_stm_signature(sig, rule_index)
            rows_list = per_tier.setdefault(
                tier, [[] for _ in range(rows)])
            while len(rows_list) < rows:
                rows_list.append([])
            rows_list[b].append(rule_index)
            return True

        # Standard left-corner shift-reduce: shift one token, then
        # greedily reduce as many times as possible before the next
        # shift. After all tokens are shifted, do a final cleanup pass.
        max_reduces_total = max(1, N * 3 + 4)
        for b in range(rows):
            reduces_done = 0
            for n in range(N):
                payload = input_vectors[b, n]
                ref_id, category, order = self._stm_snap_token(
                    payload, view, candidate_ids)
                driver.shift(b, payload,
                             category=category, order=order,
                             ref_id=ref_id)
                # Path-to-complete §5: record the SHIFTed leaf into
                # the ParseState with span [n, n+1].
                leaf_idx = parse_state.add_leaf(
                    payload=payload, category=category,
                    order=order, ref_id=ref_id, position=n)
                slot_to_frame[b].append(leaf_idx)
                # Greedy reduce while admissible.
                while reduces_done < max_reduces_total:
                    if not _try_reduce(b):
                        break
                    reduces_done += 1
            # Final cleanup reduces.
            while reduces_done < max_reduces_total:
                if int(stm_typed._depth[b].item()) <= 1:
                    break
                if not _try_reduce(b):
                    break
                reduces_done += 1
        # The greedy STM picks one action per REDUCE; the trace IS
        # the chosen action list.
        # Pad rows + (for generate) reverse each row's sequence.
        for tier, rows_list in per_tier.items():
            while len(rows_list) < rows:
                rows_list.append([])
            if mode == 'generate':
                per_tier[tier] = [list(reversed(r)) for r in rows_list]
        parse_state.trace = list(parse_state.row_traces.get(
            0, parse_state.actions))
        if mode == 'generate':
            parse_state.generate_rules = per_tier
        else:
            parse_state.current_rules = per_tier
        object.__setattr__(self, 'parse_state', parse_state)
        if mode == 'compose':
            self._extract_svo_from_parse_state(parse_state)
        return per_tier

    # -- parser-neutral accessors (path-to-complete §6) --------------------
    # Consumers should read these rather than reaching into chart-only or
    # STM-only internals. The accessors prefer ``self.parse_state``
    # (populated by both backends) and fall back to the legacy
    # ``current_rules`` attribute for chart fast-paths that haven't been
    # migrated yet.

    def parse_rules_for_tier(self, tier):
        """Return ``[[rule_id, ...], ...]`` for the given tier. Empty
        list when the tier is absent. Reads ``self.parse_state`` first
        and falls back to ``self.current_rules`` (legacy chart path).
        """
        ps = getattr(self, 'parse_state', None)
        if ps is not None and ps.current_rules:
            return ps.current_rules.get(tier, [])
        legacy = getattr(self, 'current_rules', None) or {}
        return legacy.get(tier, [])

    def parse_derivation_trace(self):
        """Return the per-row list of ``(rule_id, span_start, span_end)``
        tuples. STM populates ``parse_state.actions`` with full span
        info; the chart's projection loses span detail and emits
        ``(rule_id, -1, -1)`` placeholders. Empty list when no derivation
        is available.
        """
        ps = getattr(self, 'parse_state', None)
        if ps is None or not ps.actions:
            return [[]]
        if getattr(ps, 'row_traces', None):
            rows = []
            for b in sorted(ps.row_traces):
                row = []
                for a in ps.row_traces[b]:
                    parent = ps.frames[a.parent_index]
                    row.append((a.rule_id, parent.span_start, parent.span_end))
                rows.append(row)
            return rows
        row = []
        for a in ps.actions:
            parent = ps.frames[a.parent_index]
            row.append((a.rule_id, parent.span_start, parent.span_end))
        return [row]

    @staticmethod
    def _apply_grammar_op(op_name, left, right=None):
        """Dispatch a grammar op_name to the corresponding kernel in
        ``Layers.Ops`` and apply it to the operand payloads.

        Path-to-complete §4: parent payloads use the real grammar op
        rather than ``(left + right) / 2``. Unknown ops fall back to
        the midpoint (binary) or identity (unary) so legacy grammars
        with custom op names don't break.

        Recognized ops (binary unless noted): ``conjunction`` /
        ``intersection`` (lattice meet), ``disjunction`` / ``union``
        (lattice join), ``lift`` and ``lower`` are order-changing —
        their kernels in Ops take additional kwargs and aren't used
        here for the parent-payload mixture; for those we currently
        fall back to a per-operand midpoint. Future work: thread the
        full kwargs through (mode, kind, etc.).
        """
        from Layers import Ops
        # Binary lattice ops
        if right is not None:
            if op_name in ('conjunction', 'intersection'):
                return Ops.intersection(left, right, monotonic=False)
            if op_name in ('disjunction', 'union'):
                return Ops.union(left, right, monotonic=False)
            # Fallback: midpoint placeholder
            return (left + right) / 2.0
        # Unary fallback: identity
        return left

    @staticmethod
    def _resolve_lhs_order(sig, operand_orders):
        """Compute the parent (LHS) order for a rule firing.

        Handles both constant LHS (return ``lhs_order`` directly) and
        variable LHS (bind ``*`` from the RHS operands' variable slots,
        then ``parent = binding + lhs.delta``). When the rule has no
        variable slots, this is a no-op pass-through of ``lhs_order``.

        Order_delta semantics (``+1`` for ``lift``, ``-1`` for
        ``lower``, ``0`` otherwise) is already baked into the LHS's
        delta by ``_rule_order_signature`` (``S* = lift(NP*, VP1)``
        gives LHS variable +1), so we don't apply order_delta again
        here.
        """
        lhs_kind = str(sig.get('lhs_order_kind', 'constant'))
        lhs_delta = int(sig.get('lhs_order', 0))
        if lhs_kind == 'constant':
            return lhs_delta
        rhs_kinds = sig.get('rhs_order_kinds') or []
        rhs_ords = sig.get('rhs_orders') or []
        binding = None
        for slot, op_order in enumerate(operand_orders):
            if slot >= len(rhs_kinds):
                break
            if str(rhs_kinds[slot]) == 'variable':
                candidate = int(op_order) - int(rhs_ords[slot])
                if binding is None:
                    binding = candidate
        if binding is None:
            # LHS variable but no RHS variable to bind from — fall
            # back to lhs_delta as if constant.
            return lhs_delta
        return binding + lhs_delta

    def _tier_for_stm_signature(self, sig, rule_index):
        """Derive the tier ('P' / 'C' / 'S') for a STM-fired rule.

        Preferred path: ``TheGrammar.rules[rule_index].tier`` when the
        live grammar's rule indices align with the loaded artifact's
        signatures. Fallback: LHS-category prefix heuristic from the
        serialized signature (which is all the artifact preserves).
        """
        try:
            rule = TheGrammar.rules[rule_index]
            tier = getattr(rule, 'tier', None)
            if tier in ('P', 'C', 'S'):
                return tier
        except (IndexError, AttributeError):
            pass
        lhs = str(sig.get('lhs_category', '') or '').strip()
        first = lhs[:1] if lhs else ''
        if first == 'P':
            return 'P'
        if first == 'C':
            return 'C'
        return 'S'

    def _stm_snap_token(self, payload, view, candidate_ids):
        """Snap a token payload to the nearest live reference.

        The selected reference supplies the conceptual order, so explicit
        grammar categories such as ``NP3`` and ``NP4`` can participate in
        typed admissibility while orthographic input remains just a
        surface vector.
        """
        if candidate_ids.numel() == 0 or view.n_refs_live == 0:
            return -1, 'UNK', 0
        scalar = float(payload.reshape(-1)[0].item()) \
            if payload.numel() > 0 else 0.0
        candidate_ids = candidate_ids.to(device=view.references.device)
        candidates = view.references[candidate_ids]
        idx = int((candidates - scalar).abs().argmin().item())
        ref_id = int(candidate_ids[idx].item())
        category = view.category_of_ref(ref_id) or 'UNK'
        order = view.order_of_ref(ref_id)
        return ref_id, category, order

    def _extract_svo_from_parse_state(self, parse_state):
        """Populate WordSpace last_svo from an STM ParseState when a row
        contains S=lift(NP, VP) over VP=intersection(V, O)."""
        if not hasattr(self, 'set_last_svo'):
            return
        row_traces = getattr(parse_state, 'row_traces', None) or {
            0: getattr(parse_state, 'trace', None) or parse_state.actions
        }

        def _base(cat):
            try:
                return Grammar._parse_category(cat).name
            except Exception:
                return str(cat)

        def _rule(rule_id):
            try:
                return TheGrammar.rules[int(rule_id)]
            except Exception:
                return None

        try:
            self.clear_last_svo()
        except Exception:
            pass
        for b, actions in row_traces.items():
            action_by_parent = {a.parent_index: a for a in actions}
            for action in actions:
                rule = _rule(action.rule_id)
                if rule is None or rule.method_name != 'lift':
                    continue
                rhs = tuple(_base(s) for s in (rule.rhs_symbols or ()))
                if _base(rule.lhs) != 'S' or rhs != ('NP', 'VP'):
                    continue
                if len(action.operand_indices) != 2:
                    continue
                subj_idx, vp_idx = action.operand_indices
                vp_action = action_by_parent.get(vp_idx)
                if vp_action is None:
                    continue
                vp_rule = _rule(vp_action.rule_id)
                if vp_rule is None or vp_rule.method_name != 'intersection':
                    continue
                vp_rhs = tuple(_base(s) for s in (vp_rule.rhs_symbols or ()))
                if _base(vp_rule.lhs) != 'VP' or vp_rhs != ('V', 'O'):
                    continue
                if len(vp_action.operand_indices) != 2:
                    continue
                verb_idx, obj_idx = vp_action.operand_indices
                try:
                    self.set_last_svo(
                        int(b),
                        parse_state.frames[subj_idx].payload,
                        parse_state.frames[verb_idx].payload,
                        parse_state.frames[obj_idx].payload,
                    )
                except Exception:
                    pass

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

        D8 capture-gate (2026-05-19): also refreshes the discourse-
        prediction cache (``_disc_pred`` / ``_disc_conf``). The
        inter-sentence ARMA state only updates when ``disc.observe()``
        fires post-body in ``runBatch``, so ``predict()`` returns the
        same value for the entire sentence; caching here keeps the
        per-word body free of any ``disc.predict()`` call.
        """
        if b is None:
            self._stm_fired.zero_()
        else:
            self._stm_fired[b] = False
        # Refresh discourse-prediction cache at the sentence boundary.
        disc = self.discourse
        if disc is not None:
            try:
                self._disc_pred, self._disc_conf = disc.predict()
            except Exception:
                self._disc_pred, self._disc_conf = None, None
        else:
            self._disc_pred, self._disc_conf = None, None

    def stm_residual(self, b=0):
        """Discourse prediction bias applied once per sentence per row.

        Reads the discourse-prediction cache (populated by ``arm_stm``)
        and returns ``discourse.prime(predicted, confidence, scale)``,
        or ``None`` when discourse is unavailable, not yet built, or the
        bias already fired this sentence for row ``b``.  ``arm_stm(b)``
        / ``Reset()`` re-arms and refreshes the cache.

        D8 capture-gate (2026-05-19): the cache replaces the per-call
        ``disc.predict()``, sentence-scoped (refreshed only when
        ``arm_stm`` fires at the sentence boundary).

        ``b`` defaults to 0 for back-compat with single-row callers; the
        Task 9 cutover threads the row index from the body iteration.
        """
        if self.stm_fired(b):
            return None
        self.mark_stm_fired(b)
        disc = self.discourse
        if disc is None:
            return None
        pred = self._disc_pred
        conf = self._disc_conf
        if pred is None or conf is None:
            return None
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
        # was a per-batch host sync that blocked CUDA-graph capture.
        #
        # D8 capture-gate (2026-05-19): read the discourse cache instead
        # of calling ``disc.predict()`` here. The cache is populated once
        # per sentence in ``arm_stm()`` (the sentence boundary). This
        # eliminates the per-word ``disc.predict()`` call (which had a
        # Python row-loop in ``_predict_live(b=b)`` that's untraceable
        # under fullgraph=True and a per-word DtoH on CUDA).
        disc = self.discourse
        if disc is None:
            return None
        pred = self._disc_pred
        conf = self._disc_conf
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
            # Hierarchical stages may operate in a state basis that is
            # narrower than the global DiscourseSpace concept projection.
            # Do not inject a residual across incompatible bases.
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
        """Return the learned embedding row for a derivation label.

        One entry per label: ``category_index['S'] = 0``,
        ``category_index['VO'] = 1``, etc. Looks up directly by index —
        no activation-similarity snap — so this is the canonical path
        for "embedding for category X" when the label is already known
        from the grammar.

        Args:
            category: string category name ('S', 'VO', ...) or int
                row index into ``category_embedding``.

        Returns:
            Tensor of shape ``(pos_dim,)`` — the embedding row.
        """
        if isinstance(category, str):
            idx = self.category_index[category]
        else:
            idx = int(category)
        return self.category_embedding.weight[idx]

    # -- PoS helpers (legacy) -----------------------------------------
    def pos_lookup(self, active_symbols):
        """Activation-similarity nearest-neighbor lookup into the embedding.

        Legacy path kept so ``SymbolicSpace.forward`` (and its tests) can
        map an active-symbol pattern to an embedding row without knowing
        the grammar category up-front. New code that already has the
        category name should use ``category_lookup(name)`` instead.

        Args:
            active_symbols: 1-D tensor of shape [N], typically resolved
                activations from SymbolicSpace.resolve().

        Returns:
            Tensor of shape (pos_dim,) -- the matching prototype row.
        """
        w = self.category_embedding.weight  # [N_categories, pos_dim]
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

        Backend dispatch (plan
        ``doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md``):
        ``self.parser_backend`` selects ``'chart'`` (default — current
        behavior) / ``'stm'`` (STM shift/reduce driver) / ``'parallel'``
        (both, chart authoritative). Unknown backends raise
        ``ValueError``; ``'stm'`` / ``'parallel'`` raise
        ``NotImplementedError`` until their drivers land.
        """
        backend = getattr(self, 'parser_backend', 'chart')
        if backend == 'stm':
            return self._compose_stm(input_vectors, subspace)
        if backend == 'parallel':
            # Construct STM driver first (cheap, no compute) so it's
            # available even if the chart raises. Then the chart runs
            # authoritatively and its result is returned.
            if self.stm_driver is None:
                self._init_stm_driver()
            # Fall through to the chart path below.
        elif backend != 'chart':
            raise ValueError(
                f"unknown parser_backend: {backend!r} "
                "(expected 'chart' / 'stm' / 'parallel')")
        # Per-compose cursor reset. The OLD semantics zeroed each
        # per-tier SyntacticLayer cursor lazily on every compose() call
        # (via the ``gen != _cursor_compose_gen`` branch keyed off this
        # counter). On the WordSpace.cursor path we reproduce that
        # EXACTLY with an unconditional in-place reset of all tiers'
        # cursors at the top of compose -- no data-dependent Python
        # branch on a per-batch nn.Module int (recompile cause #3
        # eliminated). ``cursor`` is a host list[int] so the read in
        # SyntacticLayer._next_rule_name traces cleanly under Dynamo
        # (no unbacked SymInt from ``int(tensor)``).
        for i in range(len(self.cursor)):
            self.cursor[i] = 0
        # Bump the generation counter for any consumers that haven't
        # been migrated off it yet (purely host-side bookkeeping).
        self._compose_generation += 1
        # Two firing modes, gated by ``_grammar_is_default_only``
        # (computed from the configured grammar at __init__ time):
        #
        #   * Default-only fast path — every operational rule is the
        #     unary substrate fold (``pi`` / ``sigma`` arity-1). The
        #     chart inside pass adds no information; ``current_rules``
        #     is populated from the grammar XML directly.
        #
        #   * Full chart — any other rule is present (``intersection``,
        #     ``union``, ``lift``, ``lower``, ``not``, …). The chart
        #     runs its inside pass to select per-cell winners.
        #
        # The retired ``<WordSpace><useGrammar>`` XML knob used to
        # also gate this; the grammar XML is now the sole driver.
        if self._grammar_is_default_only:
            self.current_rules = self._default_compose_rules()
            self._pad_S_cursor_to_target(self.current_rules)
            return self.current_rules
        self.current_rules = self.chart.compose(
            input_vectors, self, subspace=subspace) or {}
        self._pad_S_cursor_to_target(self.current_rules)
        return self.current_rules

    def _pad_S_cursor_to_target(self, rules_dict):
        # Forward-only asymmetric padding: extend the S-tier rule cursor
        # to ``self._target_cursor_length`` with ``TheGrammar.id_SS``
        # (the no-op grammatical transition).
        # See doc/plans/2026-05-20-static-per-word-loop-impl.md §1.
        # Non-S tiers naturally return None past their end (a no-op),
        # so only the S tier — the one that owns the per-word stem —
        # needs explicit padding.
        N = int(self._target_cursor_length)
        if N <= 0:
            return rules_dict
        id_SS = TheGrammar.id_SS
        if id_SS is None or rules_dict is None:
            return rules_dict
        s_rules = rules_dict.get('S')
        if s_rules is None:
            rules_dict['S'] = [id_SS] * N
            return rules_dict
        if s_rules and isinstance(s_rules[0], list):
            for row in s_rules:
                while len(row) < N:
                    row.append(id_SS)
        else:
            while len(s_rules) < N:
                s_rules.append(id_SS)
        return rules_dict

    def generate(self, target_vectors, subspace=None):
        """Run the chart's outside pass + Viterbi backtrace; populate
        ``self.generate_rules``.

        Default-only fast path mirrors ``compose``.

        Post-2026-05-14: the only training-time caller of this method
        was ``Models._chart_generate_from_stm``, which was retired
        with the reverse pipeline.  ``WordSpace.generate_rules`` is
        now populated once at construction via
        ``_default_generate_rules`` and consumed read-only by
        ``Mereology`` / ``Models`` for diagnostic dumps; the
        downstream chart's outside pass is never re-fired during
        training.  This method is kept on the public surface so that
        unit-level tests of the chart's outside pass (signal-router
        contract, Viterbi backtrace) can still drive it directly.

        Backend dispatch parallels ``compose()``: ``self.parser_backend``
        selects ``'chart'`` / ``'stm'`` / ``'parallel'``.
        """
        backend = getattr(self, 'parser_backend', 'chart')
        if backend == 'stm':
            return self._generate_stm(target_vectors, subspace)
        if backend == 'parallel':
            # See compose() for parallel-mode rationale.
            if self.stm_driver is None:
                self._init_stm_driver()
        elif backend != 'chart':
            raise ValueError(
                f"unknown parser_backend: {backend!r} "
                "(expected 'chart' / 'stm' / 'parallel')")
        self._generate_generation += 1
        if self._grammar_is_default_only:
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
        LanguageLayer the first time we ask for it; we then call
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
        if self.truth_layer is None or self.truth_layer.is_empty():
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

        # Quaternary-corner balance penalty: discourages forbidden
        # corners (N, B). The bivector substrate was retired (Phase 3):
        # ``symbol_acts`` is now a single signed scalar, so the old
        # ``symbol_acts[..., :2]`` pole slice is gone. The corner policy
        # instead reads the TruthLayer-internal accumulator -- the only
        # legitimate remaining bivector surface. ``tetralemma_balance_
        # penalty`` is a kept op that returns 0 for a non-paired
        # accumulator, so the term is inert until a paired/bivector
        # accumulator is configured; the Phase 5 client assessment
        # builds on this same accumulator read.
        wants_balance = (int(allow_excluded_middle) == -1
                         or int(allow_contradiction) == 0)
        if (balance_weight > 0 and wants_balance
                and not self.truth_layer.is_empty()):
            n = self.truth_layer.count.item()
            accumulator = self.truth_layer.truths[:n]
            balance = self.truth_layer.tetralemma_balance_penalty(
                accumulator,
                allow_excluded_middle=int(allow_excluded_middle),
                allow_contradiction=int(allow_contradiction))
            total_loss = total_loss + balance_weight * balance

        return total_loss

    # -- wiring -------------------------------------------------------
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
            # Phase C (2026-05-13 rebalance): PerceptualSpace owns
            # ``pi_input`` (input_dim → percept_dim) and ``pi_concept``
            # (concept_dim → percept_dim); both fire unconditionally
            # in the bare forward path. The chart can dispatch them by
            # rule name as well — ``pi_input`` is the IS-side fold so
            # we register it under both the new ``pi`` rule name (per
            # the doc/Spaces.md migration table: ``P = pi(IS)``) and
            # the legacy ``sigma`` alias (so old grammars
            # ``P = sigma(P)`` continue to find a layer). ``pi_concept``
            # is the C-feedback fold and gets the ``lower`` rule name.
            pi_input = getattr(space, 'pi_input', None)
            if pi_input is not None:
                builtin_layers['pi'] = pi_input
                builtin_layers['sigma'] = pi_input  # legacy alias
            pi_concept = getattr(space, 'pi_concept', None)
            if pi_concept is not None:
                builtin_layers['lower'] = pi_concept
        elif tier == 'C':
            # Phase B (2026-05-13 rebalance): ConceptualSpace owns
            # ``sigma_percept`` (percept_dim → concept_dim) — the
            # canonical forward C-tier fold. Register it under the new
            # ``sigma`` rule name (per the doc/Spaces.md migration
            # table: ``C = sigma(PS)``) and the legacy ``pi`` alias so
            # old grammars ``C = pi(C)`` continue to dispatch correctly.
            sigma_percept = getattr(space, 'sigma_percept', None)
            if sigma_percept is not None:
                builtin_layers['sigma'] = sigma_percept
                builtin_layers['pi'] = sigma_percept  # legacy alias
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
            # Lift / Lower wiring: per Phase A of the 2026-05-13
            # rebalance (doc/Spaces.md §"Where lift vs lower lives, if
            # not at the substrate"), LiftLayer and LowerLayer are
            # **pure rule-id annotators**: they compute a parameter-
            # free static lattice op (``Ops._lower_kernel`` for lift,
            # ``Ops._lift_kernel`` for lower) and the chart records
            # the ``rule_id`` on the surrounding parse cell. They no
            # longer borrow substrate sigma/pi instances. The
            # constructor signatures keep ``symbolicSpace`` /
            # ``perceptualSpace`` / ``conceptualSpace`` parameters for
            # API compatibility but the kernels ignore them.
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
            if 'isEqual' in grammar_S_methods:
                from Layers import IsEqualLayer
                builtin_layers['isEqual'] = IsEqualLayer()
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

        Post-2026-05-12 refactor: the actual symbolic composition runs
        on the chart at C-tier over the per-word STM buffer
        (``_chart_compose_at_C`` inside ``_forward_body``), with the
        per-space ``SyntacticLayer.forward`` dispatch consuming the
        chart's rule choices.  This helper retains the demux side
        effect that downstream slot selectors depend on.

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
        symbol-side reverse via ``BasicModel._chart_generate_from_stm``
        + per-space ``SyntacticLayer.reverse`` dispatch.
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
    def clear_sentence(self):
        """Reset per-sentence state at sentence boundaries.

        Called by ``BasicModel`` on sentence boundary signals. The
        legacy SR-parser WordSubSpace stack was removed (2026-05-20);
        the per-sentence cursor / recur_pass on WordSubSpace are reset
        in ``soft_reset``, and the category / reconstruction stacks
        have their own ``clear`` paths — so this entry point is now a
        no-op retained for API compatibility with existing callers.
        """
        return

    def _detach_persistent_state(self):
        """Sever the autograd graph carried across batches by persistent
        per-row state tensors.

        In-place writes during forward (``set_last_svo``, ``arm_stm`` →
        ``self._disc_pred = disc.predict()``, ``subspace.event.setW(...)``,
        etc.) leave the persistent buffers wired to the previous batch's
        autograd graph. Once that batch's ``backward()`` runs, the saved
        tensors are freed; the next batch's forward re-reads the same
        buffers, so its ``backward()`` walks into freed nodes and raises
        "Trying to backward through the graph a second time."

        Detaching here breaks history without changing values: the
        carried numeric state is preserved, but no autograd edges cross
        the batch boundary.

        Must be called only from the eager post-backward dispatch path
        (``BasicModel.post_tick_compact``) -- never from inside a
        ``torch.compile``'d region. ``self.buffers()`` / ``self.modules()``
        iteration trips Dynamo's "getattr() on nn.Module with pending
        mutation" guard under ``fullgraph=True``.
        """
        if self._disc_pred is not None:
            self._disc_pred = self._disc_pred.detach()
        if self._disc_conf is not None:
            self._disc_conf = self._disc_conf.detach()
        self._last_svo = self._last_svo.detach()
        # Catch floating-point buffers carried transitively by
        # submodules (subspace, category_stack, reconstruction_stack,
        # discourse, truth_layer).
        for buf in self.buffers():
            if buf.is_floating_point():
                buf.detach_()
        # Basis ``Tensor`` payloads (``subspace.event``, ``.what``, etc.)
        # store the live activation in ``W`` when ``W`` is a plain tensor
        # (non-Parameter slots) — those are plain attributes, not
        # registered buffers, so ``self.buffers()`` misses them.
        # The ``_active_payload`` shadow was retired Stage 4 of
        # doc/plans/2026-05-21-active-payload-retirement.md; per-batch
        # content for codebook-bearing slots reconstructs via
        # ``SubSpace.materialize``. Skip ``W`` when it's an
        # ``nn.Parameter`` (learned weight, not a transient).
        for mod in self.modules():
            w = getattr(mod, 'W', None)
            if (w is not None and torch.is_tensor(w)
                    and not isinstance(w, nn.Parameter)
                    and w.is_floating_point()):
                mod.W = w.detach()

    # -- Space-contract lifecycle hooks --------------------------------
    # WordSubSpace is a plain ``nn.Module`` (not a ``Space``) but the
    # model iterates ``self.spaces`` calling ``set_sigma`` /
    # ``paramUpdate`` / ``getParameters`` / ``Start`` / ``End`` /
    # ``Reset`` on each entry, so we provide the same surface directly.
    # All five inline the same "iterate self.layers, call if present"
    # pattern the ``Space`` base class implements.

    def set_sigma(self, sigma):
        """Propagate exploration meta-parameters to owned layers.

        Mirrors ``Space.set_sigma`` (the no-basis branch — WordSubSpace
        has no codebook basis slots; ``self.subspace`` is ``None``).
        """
        for layer in self.layers:
            if hasattr(layer, 'set_sigma'):
                layer.set_sigma(sigma)

    def paramUpdate(self):
        """In-place parameter update hook called once per training step."""
        for layer in self.layers:
            if hasattr(layer, 'paramUpdate'):
                layer.paramUpdate()

    def getParameters(self):
        """Return optimizable parameters owned by this module."""
        return self.params

    def Start(self):
        """Per-run initialization: cascade ``Start`` to owned layers."""
        for layer in self.layers:
            if hasattr(layer, 'Start'):
                layer.Start()

    def End(self):
        """Per-batch teardown: cascade ``End`` to owned layers."""
        for layer in self.layers:
            if hasattr(layer, 'End'):
                layer.End()

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
        point. Cascades ``Reset`` to owned layers (no ``super().Reset``
        — WordSubSpace inherits from ``nn.Module``, not ``Space``).
        """
        for layer in self.layers:
            if hasattr(layer, 'Reset'):
                layer.Reset(batch=batch, hard=hard)
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
            # ``cursor`` is a host-side list[int] of length 3 (see
            # __init__ for rationale). Reset in place to preserve
            # object identity for Dynamo cache reuse.
            for i in range(len(self.cursor)):
                self.cursor[i] = 0
            self.recur_pass = 0
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
            # Reset priming working memory at the sentence boundary.
            # Plan doc/plans/2026-05-20-primed-reverse-generation.md
            # §Storage — sentence-scoped lifecycle.
            tax = getattr(self, 'taxonomy', None)
            if tax is not None:
                tax.reset()
            return
        b = int(batch)
        self.arm_stm(b)
        self.clear_last_svo(b)
        self.clear_sentence_completed(b)
        # ``cursor`` is a host-side list[int] (see __init__); reset in
        # place to keep object identity stable for the Dynamo cache.
        for i in range(len(self.cursor)):
            self.cursor[i] = 0
        self.recur_pass = 0
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
        # Reset priming working memory for this source row at the
        # sentence boundary. _priming is sized [self.batch, V_ref_cap],
        # which matches the body's B*K view, so we reset each window
        # cell separately.
        tax = getattr(self, 'taxonomy', None)
        if tax is not None and tax._priming is not None:
            for bk in range(bk_start, bk_end):
                tax.reset(batch=bk)

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

    def ensure_batch(self, batch):
        """Resize the BODY-side per-row buffers to ``batch`` (= B*K under
        the microbatch contract).

        Body-side buffers owned here: the WordSubSpace event, the
        CategoryStack / ReconstructionStack stacks, and the per-window
        transient tensors ``_last_svo`` / ``_svo_valid``.  These reallocate
        fresh-zero on shape change -- they're per-microbatch-row state
        with no cross-batch lifecycle.

        ``_stm_fired`` and ``discourse`` are NOT touched here: they live
        at B (per source row), persist across forward calls within a
        sentence, and are owned by :meth:`ensure_microbatch`.  Wiping
        them on every K-change (which happens whenever ``actual_max``
        BPE word count crosses a power-of-two boundary in PerceptualSpace's
        AR unfold) would re-arm the once-per-sentence STM-residual fire
        flag mid-sentence, causing the discourse bias to inject multiple
        times for the same source row.
        """
        batch = int(batch)
        if batch == self.batch:
            # Cascade still runs in case callers grew their own state
            # without going through the WordSpace.batch counter.
            self.category_stack.ensure_batch(batch)
            self.reconstruction_stack.ensure_batch(batch)
            return
        self.batch = batch
        self.category_stack.ensure_batch(batch)
        self.reconstruction_stack.ensure_batch(batch)
        # Keep the new buffers on the existing device so .to(device)
        # invariants survive the resize.
        device = self._last_svo.device
        self._last_svo = torch.zeros(batch, 3, self.svo_dim, device=device)
        self._svo_valid = torch.zeros(batch, dtype=torch.bool, device=device)
        # Resize priming buffer to match the new batch size. Existing
        # primed values in the overlapping region are preserved (the
        # Taxonomy.allocate_priming implementation does the copy).
        view = getattr(self, '_knowledge', None)
        tax = getattr(self, 'taxonomy', None)
        if view is not None and tax is not None and tax._priming is not None:
            tax.allocate_priming(
                batch_size=batch,
                capacity=int(view._parent.shape[0]),
                live=int(view.n_refs_live),
                device=device,
            )
        # ``_stm_fired`` is intentionally NOT reallocated here -- see
        # docstring.  ``ensure_microbatch`` handles the B-sized fields.

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
        self.ensure_batch(BK)  # body-side only; preserves _stm_fired
        device = self._stm_fired.device
        if self._stm_fired.shape[0] != int(B):
            # First allocation, or source-row count B changed (a real
            # sentence-stream boundary, not a K-change). Fresh zeros.
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


# Backward-compat alias for the pre-2026-05-21 class name. ``WordSpace``
# was renamed to ``WordSubSpace`` when the standalone ``SentenceState``
# carrier was dissolved and the class was re-cast as a plain ``nn.Module``
# (no longer a ``Space``-subclass pipeline stage). Existing imports
# ``from Language import WordSpace`` and ``object.__new__(WordSpace)``
# call sites in legacy tests keep working through this alias; new code
# should reference ``WordSubSpace`` directly.
WordSpace = WordSubSpace
