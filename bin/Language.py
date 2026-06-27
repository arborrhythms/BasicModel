

import itertools, math, os, re, warnings
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
from Optimizer import Adam
from visualize import Report, TheReport
from util import ProjectPaths, compile, TheXMLConfig, init_config, init_compile_backend, autocast_compute_dtype
from embed import WordVectors, PretrainModel
from data import Data, TheData
from Layers import Layer, PiLayer, SigmaLayer  # Import custom layers from Model.py
from Layers import LinearLayer, InvertibleLinearLayer, AttentionLayer, AssociationLayer, MapppingLayer, ChunkLayer
from Layers import CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon, Ops
from Layers import SortingLayer, TruthLayer, RelativeTruthStore, TernaryTruthStore, LiftingLayer, InterSentenceLayer, SparsityRegLayer, SmoothingRegLayer, ImpenetrableLayer
from util import parse
from collections import namedtuple as _namedtuple

# Per doc/plans/2026-05-29-grammar-file-refactor.md §5: GrammarLayer
# stays in Layers.py (PiLayer / SigmaLayer / EqualLayer / TrueLayer /
# FalseLayer / SwapLayer / CopyLayer / AreaLayer / LuminosityLayer /
# IsaPartLayer also derive from it and stay). The grammar rule operator
# classes (NotLayer, NonLayer, IntersectionLayer, UnionLayer, LiftLayer,
# LowerLayer, SymbolizeLayer, ConjunctionLayer, DisjunctionLayer,
# IsEqualLayer, PartLayer, QueryLayer) physically live in this module
# below, after the Grammar singleton.
from Layers import GrammarLayer
from Layers import (
    EqualLayer, TrueLayer, FalseLayer, SwapLayer, CopyLayer,
    AreaLayer, LuminosityLayer, IsaPartLayer,
)
from Layers import (
    SurfaceSchema, T1_UNARY_AFFIX, T2_BINARY_INFIX,
    T3_BINARY_DIRECTIONAL, T4_BINARY_JUXTAPOSE, T5_BINARY_ELISION,
)

from Spaces import ActiveEncoding, WhereEncoding, WhenEncoding, WhatEncoding, EventEncoding, WordEncoding
from Spaces import Basis, Tensor, Codebook, Embedding
from Spaces import SubSpace, Space, PerceptualSpace, InputSpace, PartSpace, ModalSpace, ConceptualSpace, WholeSpace, OutputSpace

import xml.etree.ElementTree as _ET
from pathlib import Path as _Path

_GRAMMAR_DIR = _Path(__file__).parent.parent / "data"


from dataclasses import dataclass, field as _dc_field


@dataclass
class RoutingState:
    """First-class per-sentence routing decision produced by
    ``SymbolSubSpace.compose``.

    This is the ADDITIVE companion to the long-standing ``current_rules``
    dict: many consumers depend on ``current_rules`` staying exactly
    ``dict[space_role, list[list[int]]]`` (the per-row, per-step rule ids read
    by ``SyntacticLayer._next_rule_name``, the SS-space_role dispatch in
    ``Models``, and ``Spaces``' stack-route path), so ``RoutingState`` is
    stored ALONGSIDE it (on ``SymbolSubSpace.routing_state``) and never
    replaces it. It carries the same information in two extra forms the
    intra-sentence predictor needs:

    Fields
    ------
    rules_by_space_role : dict[space_role, list[list[int]]]
        The exact ``current_rules`` dict (same object), kept here so a
        single ``RoutingState`` is a self-contained snapshot.
    selected_rules : list[int]
        Flat list of the selected rule-ids for the canonical row (row 0
        of every space_role, concatenated in sorted-space_role order). Row 0 is the
        canonical sequence convention already used by
        ``SyntacticLayer._next_rule_name`` (per-row dispatch is a
        follow-on). Used to build ``rule_probs`` and for diagnostics.
    rule_probs : torch.Tensor | None
        Dense ``[B, n_rules]`` float distribution over the grammar's
        rule vocabulary (``n_rules == len(TheGrammar.rule_table)``). This
        is the rule-conditioning signal the intra-sentence predictor
        consumes (``ConceptualSpace._intra_routing_for_predict`` ->
        ``IntraSentenceLayer.routing``). FIRST CUT (see
        ``SymbolSubSpace._synthesize_rule_probs``): mass is scattered onto
        the SELECTED rule-ids and L1-normalized per row, so this encodes
        WHICH rules fired (not yet the gradient-bearing soft marginals
        fragmented in ``LanguageLayer._last_space_role_routings`` -- that is a
        documented future upgrade at the ``_synthesize_rule_probs``
        seam). ``None`` when no grammar/router has fired.
    """
    rules_by_space_role: dict = _dc_field(default_factory=dict)
    selected_rules: list = _dc_field(default_factory=list)
    rule_probs: object = None


class MetaSymbolCategoryLearner:
    """Pending role-evidence table for MetaSymbol category learning.

    Long-term state stays compact: a learned MetaSymbol keeps only its
    committed ``meta_pos -> category_id`` assignment on ``WholeSpace``. While
    the assignment is still unstable, this learner holds a bounded sparse row
    of accumulated grammatical role evidence for that MetaSymbol. Category
    centroids live in role-participation space, not MetaSymbol embedding space.
    """

    def __init__(self, n_roles, *, max_pending=4096, min_mass=4.0,
                 min_confidence=0.70, min_margin=0.10, stable_updates=2,
                 evidence_decay=1.0, prototype_ema=0.10):
        self.n_roles = int(n_roles)
        self.max_pending = int(max_pending)
        self.min_mass = float(min_mass)
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.stable_updates = int(stable_updates)
        self.evidence_decay = float(evidence_decay)
        self.prototype_ema = float(prototype_ema)
        self.pending = {}
        self.step = 0

    def _role_tensor(self, role_vec):
        if not torch.is_tensor(role_vec):
            role_vec = torch.tensor(role_vec, dtype=torch.float32)
        vec = role_vec.detach().float().reshape(-1).cpu()
        if vec.numel() != self.n_roles:
            return None
        return vec

    @staticmethod
    def _profile(evidence):
        mass = float(evidence.sum().item())
        if mass <= 0.0:
            return evidence.clone(), 0.0
        return evidence / mass, mass

    def _evict_if_needed(self):
        if self.max_pending <= 0:
            self.pending.clear()
            return
        if len(self.pending) < self.max_pending:
            return
        victim, _row = min(
            self.pending.items(),
            key=lambda kv: (float(kv[1].get("mass", 0.0)),
                            int(kv[1].get("last", 0))))
        self.pending.pop(victim, None)

    def pending_role(self, meta_pos, *, device=None, dtype=None):
        row = self.pending.get(int(meta_pos))
        if row is None:
            return None
        profile, mass = self._profile(row["evidence"])
        if mass <= 0.0:
            return None
        if dtype is None:
            dtype = torch.float32
        return profile.to(device=device, dtype=dtype)

    def score_assignment(self, ws, profile):
        """Return ``(best, confidence, margin)`` for a role-space profile."""
        role_table = getattr(ws, "_category_role", None)
        vq = getattr(ws, "_category_vq", None)
        codebook = role_table
        if codebook is None and vq is not None:
            codebook = getattr(vq, "codebook", None)
        if codebook is None or not torch.is_tensor(codebook):
            return None, 0.0, 0.0
        cb = codebook.detach().float().cpu()
        if cb.dim() != 2 or cb.shape[1] != profile.numel() or cb.shape[0] == 0:
            return None, 0.0, 0.0
        dist = ((cb - profile.reshape(1, -1)) ** 2).sum(dim=-1)
        best = int(torch.argmin(dist).item())
        best_dist = float(dist[best].item())
        confidence = 1.0 / (1.0 + best_dist)
        if dist.numel() == 1:
            margin = float("inf")
        else:
            ordered = torch.sort(dist).values
            margin = float((ordered[1] - ordered[0]).item())
        return best, confidence, margin

    def _assign_profile(self, ws, profile):
        idx = ws.assign_category(profile.reshape(1, -1))
        if idx is None:
            return None
        cat_id = int(idx.reshape(-1)[0])
        ws.update_category_role(
            torch.tensor([cat_id], dtype=torch.long),
            profile.reshape(1, -1),
            ema=self.prototype_ema)
        return cat_id

    def _commit(self, ws, meta_pos, category_id):
        assign = getattr(ws, "_category_assign", None)
        if assign is None:
            assign = {}
            object.__setattr__(ws, "_category_assign", assign)
        assign[int(meta_pos)] = int(category_id)
        self.pending.pop(int(meta_pos), None)
        return int(category_id)

    def observe(self, ws, meta_pos, role_vec):
        """Update category evidence for one MetaSymbol.

        Returns the committed category id when the symbol is already learned
        or becomes learned on this observation. Returns ``None`` while the
        symbol remains in the pending table.
        """
        vec = self._role_tensor(role_vec)
        if vec is None or vec.sum().item() <= 0.0:
            return None
        meta_pos = int(meta_pos)
        self.step += 1

        assign = getattr(ws, "_category_assign", None) or {}
        committed = assign.get(meta_pos)
        if committed is not None:
            profile, _mass = self._profile(vec)
            ws.update_category_role(
                torch.tensor([int(committed)], dtype=torch.long),
                profile.reshape(1, -1),
                ema=self.prototype_ema)
            return int(committed)

        row = self.pending.get(meta_pos)
        if row is None:
            if self.max_pending <= 0:
                return None
            self._evict_if_needed()
            row = {
                # On vec's device: vec is the model-device role tensor, and the
                # in-place ``row["evidence"].add_(vec)`` below would otherwise
                # mismatch (default-device zeros vs vec on mps/cuda). The
                # CPU-pinned test suite never exercised this.
                "evidence": torch.zeros(self.n_roles, dtype=torch.float32,
                                        device=vec.device),
                "mass": 0.0,
                "best": None,
                "stable": 0,
                "last": self.step,
            }
            self.pending[meta_pos] = row

        if self.evidence_decay < 1.0:
            row["evidence"].mul_(max(0.0, self.evidence_decay))
        row["evidence"].add_(vec)
        profile, mass = self._profile(row["evidence"])
        row["mass"] = mass
        row["last"] = self.step

        best = self._assign_profile(ws, profile)
        scored_best, confidence, margin = self.score_assignment(ws, profile)
        if scored_best is not None:
            best = scored_best
        if best is None:
            return None

        if row["best"] == int(best):
            row["stable"] = int(row.get("stable", 0)) + 1
        else:
            row["best"] = int(best)
            row["stable"] = 1

        if (mass >= self.min_mass
                and confidence >= self.min_confidence
                and margin >= self.min_margin
                and int(row["stable"]) >= self.stable_updates):
            return self._commit(ws, meta_pos, int(best))
        return None


def load_grammar(filename):
    """Load a ``.grammar`` XML file from ``data/`` and return a
    ``Grammar.configure()``-compatible dict.

    The .grammar format reuses the same inline rule syntax as the
    legacy ``<grammar>...</grammar>`` block:

        <?xml version="1.0"?>
        <grammar name="default">
          <compose>
            <rule>S = lift(NP, VP)</rule>
            <rule>S = intersection(S, S)</rule>
            ...
          </compose>
          <generate>
            <rule>S = not.reverse(S)</rule>
            ...
          </generate>
        </grammar>

    Each rule's category (head / argument labels), arity (argument
    count), function name, and return-value count are inferred from the
    body. Space-role, invertibility, and any other class-level metadata come
    from the rule's ``GrammarLayer`` subclass via ``GRAMMAR_LAYER_CLASSES``
    (set later in ``Grammar.load_from_grammar_file``).

    Compact ordered-category sugar is expanded later in
    ``load_from_grammar_file``: e.g. ``S45`` in a rule body means
    concrete alternatives ``S4`` and ``S5`` in the loaded rule table.
    """
    path = _GRAMMAR_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Grammar file not found: {path}")
    root = _ET.parse(path).getroot()
    return _grammar_xml_to_dict(root)


_COMPACT_ORDER_SET_RE = re.compile(r'\b([A-Z][A-Z_]*)([0-9]{2,})\b')


def _expand_compact_order_sets_in_rule(rule):
    """Expand compact ordered-category sugar in one rule string.

    In ``.grammar`` files, a multi-digit order suffix denotes a small
    set of concrete orders: ``S45`` expands to ``S4`` and ``S5``. The
    expansion is source-level sugar only; runtime rule signatures still
    carry exact single orders. Repeated uses of the same suffix in one
    rule are correlated, so ``S45 = not(NOT_S45)`` expands pairwise
    rather than as a Cartesian product.
    """
    if not isinstance(rule, str):
        return [rule]
    matches = list(_COMPACT_ORDER_SET_RE.finditer(rule))
    if not matches:
        return [rule]

    order_sets = []
    seen = set()
    for match in matches:
        digits = match.group(2)
        if digits not in seen:
            seen.add(digits)
            order_sets.append(digits)

    choices = [tuple(dict.fromkeys(digits)) for digits in order_sets]
    expanded = []
    for combo in itertools.product(*choices):
        selected = dict(zip(order_sets, combo))

        def repl(match):
            prefix, digits = match.groups()
            return f"{prefix}{selected[digits]}"

        expanded.append(_COMPACT_ORDER_SET_RE.sub(repl, rule))
    return expanded


def _expand_compact_order_sets(cfg):
    """Expand compact order-set sugar under every ``rule`` list."""
    if isinstance(cfg, list):
        out = []
        for item in cfg:
            out.extend(_expand_compact_order_sets(item))
        return out
    if not isinstance(cfg, dict):
        return cfg
    out = {}
    for key, value in cfg.items():
        if key == 'rule':
            rules = value if isinstance(value, list) else [value]
            expanded_rules = []
            for rule in rules:
                if isinstance(rule, dict) and '_' in rule:
                    for expanded in _expand_compact_order_sets_in_rule(
                            rule.get('_')):
                        expanded_rule = dict(rule)
                        expanded_rule['_'] = expanded
                        expanded_rules.append(expanded_rule)
                else:
                    expanded_rules.extend(
                        _expand_compact_order_sets_in_rule(rule))
            out[key] = expanded_rules
        else:
            out[key] = _expand_compact_order_sets(value)
    return out


def _start_item_text(item):
    """Surface text of one ``<start>`` entry.

    Tolerates the attribute-bearing ``{'_': text, 'name': ...}`` dict that
    ``_grammar_xml_to_dict`` now emits for ``<start name=...>`` (so the
    ``relative_truth`` / ``absolute_truth`` / ``everything`` roles survive
    parse), as well as the bare-string legacy shape.
    """
    if isinstance(item, dict):
        return str(item.get('_', '')).strip()
    return str(item).strip()


def _starts_by_name(start_raw):
    """Map each start *symbol* (after compact-order expansion) to the
    ``name`` attribute of the ``<start>`` it came from (or ``None``).

    Lets the loader partition WholeSpace starts into ``relative_truth``
    / ``absolute_truth`` role sets for relative-rule detection (R1.3).
    """
    if start_raw is None:
        return {}
    items = start_raw if isinstance(start_raw, list) else [start_raw]
    out = {}
    for item in items:
        text = _start_item_text(item)
        if not text:
            continue
        name = item.get('name') if isinstance(item, dict) else None
        for expanded in _expand_compact_order_sets_in_rule(text):
            for sym in (p.strip() for p in expanded.split() if p.strip()):
                out.setdefault(sym, name)
    return out


def _start_patterns_from_raw(start_raw, default='S'):
    """Parse one or more ``<start>`` entries into concrete patterns.

    Each pattern is a tuple of category tokens. Compact order-set sugar
    is allowed here too: ``S45`` becomes ``("S4",)`` and ``("S5",)``,
    while ``S45 REL S45`` becomes the two correlated patterns
    ``("S4", "REL", "S4")`` and ``("S5", "REL", "S5")``.
    """
    if start_raw is None:
        raw_items = [default]
    elif isinstance(start_raw, list):
        raw_items = start_raw
    else:
        raw_items = [start_raw]

    patterns = []
    seen = set()
    for raw in raw_items:
        text = _start_item_text(raw)
        if not text:
            continue
        for expanded in _expand_compact_order_sets_in_rule(text):
            pattern = tuple(part.strip() for part in expanded.split()
                            if part.strip())
            if pattern and pattern not in seen:
                seen.add(pattern)
                patterns.append(pattern)
    if not patterns:
        patterns = [(str(default).strip() or 'S',)]
    return tuple(patterns)


def _primary_start_symbol(patterns, default='S'):
    """Pick the primary atomic start symbol from parsed start patterns."""
    for pattern in patterns:
        if len(pattern) == 1:
            return pattern[0]
    if patterns and patterns[0]:
        return patterns[0][0]
    return default


def _grammar_xml_to_dict(node):
    """Convert an ``ElementTree`` grammar node into the nested-dict
    shape that ``Grammar.configure()`` consumes.

    Element nodes with children become dicts; leaf element nodes contribute
    their stripped text. Repeated tags (most importantly ``<rule>``) merge
    into a list under that tag, matching the shape ``TheXMLConfig`` emits
    for the legacy inline ``<grammar>...</grammar>`` block. Attributes on
    ``<rule>`` leaves are preserved as ``{"_": text, **attrs}``.
    """
    result = {}
    for child in node:
        tag = child.tag
        if len(list(child)) > 0:
            value = _grammar_xml_to_dict(child)
        else:
            text = (child.text or '').strip()
            value = {'_': text, **child.attrib} if (
                tag in ('rule', 'start') and child.attrib) else text
        if tag in result:
            if not isinstance(result[tag], list):
                result[tag] = [result[tag]]
            result[tag].append(value)
        else:
            result[tag] = value
    return result


# --------------------------------------------------------------------
# Stage 3 (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md): the
# chart and STM shift-reduce parsers are retired in favour of the
# signal router (``LanguageLayer``). Configs that still set retired
# knobs must error loudly per the project's fail-loud rule -- silent
# acceptance would let stale settings drift into the new pipeline.
# --------------------------------------------------------------------
_RETIRED_CHART_KNOBS = (
    "parserBackend",
    "routerKind",
    "chartTau",
    "chartTopK",
    "chartNoiseEps",
)
# ``<wMax>`` (the legacy STM-capacity alias) is retired too: STM depth now
# comes from ``<stmCapacity>`` (or the DEFAULT_CAPACITY=8 fallback), never
# ``<wMax>``.
_RETIRED_STM_KNOBS = ("wMax",)


def _assert_retired_chart_knobs_absent():
    """Raise ``ValueError`` if any retired knob lives in the loaded XML
    config under ``<SymbolSpace>``.

    Stage 3 retires ``<parserBackend>``, ``<routerKind>``, ``<chartTau>``,
    ``<chartTopK>``, ``<chartNoiseEps>`` in favour of the signal router;
    ``<wMax>`` is retired in favour of ``<stmCapacity>`` (STM depth). A loud
    failure here catches legacy XML files that still set them. Called from
    ``SymbolSubSpace.__init__`` after grammar configuration.
    """
    try:
        ss_section = TheXMLConfig.get("SymbolSpace", None)
    except (KeyError, AttributeError):
        ss_section = None
    if not isinstance(ss_section, dict):
        return
    offending = [k for k in (_RETIRED_CHART_KNOBS + _RETIRED_STM_KNOBS)
                 if k in ss_section]
    if offending:
        joined = ", ".join(f"<{k}>" for k in offending)
        raise ValueError(
            f"SymbolSpace XML config carries retired knob(s): {joined}. "
            f"The chart/router knobs (parserBackend, routerKind, chartTau, "
            f"chartTopK, chartNoiseEps) were retired in favour of the signal "
            f"router (LanguageLayer); <wMax> was retired in favour of "
            f"<stmCapacity> (STM depth). Remove the offending element(s) "
            f"from your config."
        )


def grammar_uses(rule_name):
    """Return True iff any rule body in the configured grammar invokes
    ``rule_name`` as a function call.

    Note: previously consumed by ConceptualSpace's grammar-driven wiring
    inference (DNF auto-wrap), which has been removed. NegationLayer is
    no longer auto-wired — wire composite CS-space_role wrappers explicitly via
    the ``layer`` kwarg on ConceptualSpace. This helper remains available
    for runtime grammar inspection.

    Reads the parsed XML grammar at SymbolSpace.language.grammar; scans
    rule bodies (string leaves) for the substring ``rule_name(``.
    Returns False on any read error or when no grammar is configured.
    """
    needle = f"{rule_name}("
    try:
        cfg = TheXMLConfig.get("SymbolSpace.language.grammar")
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
    return False


class Grammar:
    """Multi-space_role grammar rule catalog (subsymbolic / CS / SS).

    Space-roles tag each rule with the space that dispatches it:
      - ``subsymbolic`` (perceptual) -- PartSpace's SyntacticLayer.
      - ``CS`` (conceptual) -- ConceptualSpace's SyntacticLayer.
        Bivector pre-codebook activation ``[B, V, 2]``. The lattice
        primitives ``intersection`` / ``union`` bind here (lattice
        min/max on bivector activation).
      - ``SS`` (symbolic)   -- WholeSpace's SyntacticLayer.
        Post-codebook activation: a scalar ``[B, V]`` per
        prototype. SS-space_role ops (``conjunction``, ``disjunction``,
        ``not``, ``lift``, ``lower``, ``part``, ``equals``,
        ``query``, ``true``, ``false``, ``swap``, ``non``) are
        monotonic functions on that scalar.

    Owns the rule definitions parsed from XML config. All learnable
    parameters and rule execution live on a single unified
    ``SyntacticLayer`` instance owned by ``SymbolSpace``.
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
        ['space_role', 'canonical', 'arity', 'method_name', 'lhs', 'rhs_symbols',
         'width_min', 'width_max', 'query'],
    )
    RuleDef.__new__.__defaults__ = (0, 0, False)

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
        Holds space_role-tagged rule lists, the upward / downward / reverse
        derivations, and the default start symbol ``"S"``.
        """
        self.rules = []
        self.rules_upward = []
        self.rules_downward = []
        # PartSpace meronymic rule tables. Populated only when a
        # grammar file carries a ``<PartSpace>`` section (Phase 8b,
        # doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md).
        # These are kept SEPARATE from the symbolic tables above: the
        # existing symbolic parser reads ``self.rules`` (== the WS table),
        # so adding a PS section never perturbs symbolic rule ids. PS rules
        # carry space_role 'subsymbolic' and are consumed by the PS analyzer phases.
        self.ps_rules_upward = []
        self.ps_rules_downward = []
        self.ps_rules = []
        # Step 6: Layer-2.5 reverse productions, derived mechanically
        # from rules_upward at load time.  Each entry is
        # ``(args_tuple, reverse_op_name, (lhs,))``.
        self.reverse_rules = []
        self.rule_table = {}
        self._configured = False
        self.interpretation = 0.5
        self.thought_free = False
        # Phase 1 of the SubSpace.what STM refactor: V_sym is the size of
        # the terminal symbol codebook, which WholeSpace wires in once
        # its symbol codebook is built. Until then, the rule namespace
        # starts at 1 (treating V_sym=0). Used only by where_id_for_rule.
        # See doc/plans/2026-05-20-subspace-what-stm-signalrouter-refactor.md
        self.symbol_vocab_size = 0
        # Start patterns -- accepted completed derivation shapes.
        # ``start_symbol`` remains the primary single-category start for
        # legacy identity-rule and reset code; ``start_patterns`` can also
        # carry unreduced accepted forms such as ("S4", "REL", "S4").
        # Configurable via one or more <start>...</start> entries; falls
        # back to "S" (the historical default) when unset.
        self.start_symbol = "S"
        self.start_patterns = (("S",),)
        # Space-scoped starts (Phase R1.1,
        # doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
        # decision 7 / §4.4). ``WholeSpace.start`` configures the
        # symbolic parse starts; ``start_symbol`` / ``start_patterns``
        # above are the back-compat *alias* of them (id_SS,
        # is_start_pattern, reset and relative-rule code key off the
        # symbolic start). ``PartSpace.start`` configures the
        # analyzer root (``U``) -- a separate namespace the symbolic
        # parser never reads.
        self.ws_start_symbol = "S"
        self.ws_start_patterns = (("S",),)
        self.ps_start_symbol = None
        self.ps_start_patterns = ()
        # WS starts partitioned by ``<start name=...>``: a relative truth
        # is a binary-predicate end-state (the isEqual / isPart family),
        # an absolute truth collapses to a single idea. Consumed by
        # ``_relative_start_categories`` (R1.3).
        self.ws_relative_starts = frozenset()
        self.ws_absolute_starts = frozenset()
        # Task 6a (doc/plans/2026-05-29-stm-serial-parallel-modes.md §7):
        # cache of rule_ids that produce a RELATIVE truth (the
        # ``part`` / ``isEqual`` predicate family). Lazily computed by
        # ``_relative_rule_id_set`` and invalidated on every rule-table
        # bump. ``None`` == not yet computed.
        self._relative_rule_ids_cache = None

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

    def space_role(self, rule_id):
        """Return the space_role tag ('subsymbolic' / 'CS' / 'SS') of rule ``rule_id``."""
        return self.rules[rule_id].space_role

    def binary_rules(self):
        """Return the list of rule_ids that have arity 2."""
        return [i for i in range(len(self.rules)) if self.rules[i].arity == 2]

    # -- WholeSpace / PartSpace rule views --------------------
    #
    # The symbolic parser reads ``self.rules`` (and rules_upward /
    # rules_downward). These read-only aliases name that table the
    # WholeSpace table, mirroring ``ps_rules`` for the PartSpace
    # meronymic table. See
    # doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md.

    @property
    def ws_rules(self):
        """WholeSpace rule table (alias of the canonical ``rules``)."""
        return self.rules

    @property
    def ws_rules_upward(self):
        """WholeSpace compose (synthesis) rules."""
        return self.rules_upward

    @property
    def ws_rules_downward(self):
        """WholeSpace generate (analysis) rules."""
        return self.rules_downward

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

    def rules_for_space_role(self, space_role, arity=None):
        """Return rule_ids whose ``RuleDef.space_role`` matches ``space_role``.

        ``arity`` optionally filters to that arity (1 or 2).
        """
        self._ensure_configured()
        out = []
        for i, r in enumerate(self.rules):
            if r.space_role != space_role:
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
    # V_sym is ``self.symbol_vocab_size``, populated by WholeSpace
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

    # Maps the new space_role-bucket section names to the RuleDef.space_role
    # field. Each space space_role (PartSpace, ConceptualSpace,
    # WholeSpace) reads its own subset by space_role when filtering for
    # which rules are licensed in its forward path.
    _SPACE_ROLE_SECTIONS = {
        'percepts': 'subsymbolic',
        'concepts': 'CS',
        'symbols':  'SS',
    }

    def configure(self, grammar_dict):
        """Configure rules from an XML-derived dict.

        Accepts these shapes:
          (a) flat: {'S': ['not(S)'], ...}  — legacy compose-only.
          (b) named sections: {'compose': {...}, 'generate': {...}}
              with `op.forward(args)` / `op.reverse(arg)` rule bodies.
          (c) space_role-scoped sections: {'compose': {'symbols': {...},
                                                 'concepts': {...},
                                                 'percepts': {...}},
                                     'generate': {...same shape...}}
              Each space_role's rules carry space_role='SS' / 'CS' /
              'subsymbolic' on the
              RuleDef, so each space can filter to the rules licensed
              for it. A space "can conduct any/all of the operations"
              -- runtime gating is independent of space_role tagging; the
              tags are an inductive-bias hint, not a hard restriction.
        """
        self.rules_upward = []
        self.rules_downward = []
        self.ps_rules_upward = []
        self.ps_rules_downward = []
        # Introspection query ops declared in the grammar's <Queries>/<queries>
        # section (parse-NOPs: they build no structure). Registered here so the
        # model can reason over what it knows; the truth-grounded reasoner
        # (bin/reasoning.py) implements them as exist / equal / part / query /
        # quantize / wholes / parts.
        self.query_ops = []
        self._configured = True

        # PS / Symbolic-sectioned form (Phase 8b,
        # doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md):
        # a grammar nests <Synthesize>/<Analyze> under <PartSpace> and
        # <compose>/<generate> under <Symbolic>. PartSpace rules go to the
        # separate ps_* tables tagged space_role 'subsymbolic'; Symbolic rules
        # go to the canonical symbolic tables (so symbolic rule ids are
        # unperturbed by the presence of a PS section). A file with neither
        # wrapper is the legacy form and loads as the symbolic table.
        #
        # Section vocabulary: <PartSpace> nests <Synthesize> (parts -> whole)
        # and <Analyze> (whole -> parts) -- the mereological framing; <Symbolic>
        # nests <compose> / <generate> (the symbolic rules); a top-level
        # <Queries> declares the introspection ops.
        ps_block = grammar_dict.get('PartSpace')
        ws_block = grammar_dict.get('Symbolic')
        if ps_block is not None or ws_block is not None:
            if isinstance(ps_block, dict):
                self._fill_section(self.ps_rules_upward,
                                   ps_block.get('Synthesize') or {},
                                   default_space_role='subsymbolic')
                self._fill_section(self.ps_rules_downward,
                                   ps_block.get('Analyze') or {},
                                   default_space_role='subsymbolic')
            if isinstance(ws_block, dict):
                self._fill_section(self.rules_upward,
                                   ws_block.get('compose') or {})
                self._fill_section(self.rules_downward,
                                   ws_block.get('generate') or {})
        # Parse the top-level <Queries> section into op signatures.
        q_block = grammar_dict.get('Queries')
        if isinstance(q_block, dict):
            q = q_block.get('query')
            if isinstance(q, str):
                q = [q]
            if isinstance(q, list):
                self.query_ops = [str(x).strip() for x in q if str(x).strip()]
        if ps_block is None and ws_block is None:
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
        self.ps_rules = list(self.ps_rules_upward) + list(self.ps_rules_downward)
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

    def is_start_pattern(self, categories):
        """Return True iff ``categories`` is an accepted start pattern."""
        self._ensure_configured()
        pattern = tuple(str(c).strip() for c in categories if str(c).strip())
        return pattern in set(self.start_patterns)

    def _fill_section(self, target, section_dict, default_space_role='SS'):
        """Read a parse / generate section, dispatching to per-space_role
        rule lists when `<symbols>` / `<concepts>` / `<percepts>`
        sub-sections are present, or to the cross-space_role reader otherwise.

        Space-role-bucket detection is non-destructive: a section with both a
        `<rule>` directly and space_role sub-sections will read both, with the
        direct rules tagged ``default_space_role`` ('SS' for a WholeSpace /
        legacy section, 'subsymbolic' for a PartSpace section).
        """
        if not isinstance(section_dict, dict):
            return
        # Space-role sub-sections.
        for space_role_key, space_role_letter in self._SPACE_ROLE_SECTIONS.items():
            space_role_block = section_dict.get(space_role_key)
            if space_role_block:
                self._fill_rule_list(target, space_role_block, space_role=space_role_letter)
        # Direct rules (no space_role wrapper) -> the section's default space_role.
        direct_keys = [k for k in section_dict.keys()
                       if k not in self._SPACE_ROLE_SECTIONS]
        if direct_keys:
            direct = {k: section_dict[k] for k in direct_keys}
            self._fill_rule_list(target, direct, space_role=default_space_role)

    def _fill_rule_list(self, target, rules_dict, space_role='SS'):
        """Parse ``<rule>`` entries from ``rules_dict`` and append to ``target``.

        Handles both the canonical ``<rule>head = body</rule>`` form
        (with optional ``width="MIN..MAX"`` gate) and the legacy
        ``<S>body</S>`` form. Each parsed rule is tagged with the
        supplied space_role letter.
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
                    query_raw = entry.get('query', None)
                else:
                    text = str(entry)
                    width_raw = None
                    query_raw = None
                if '=' not in text:
                    raise ValueError(
                        f"<rule> requires 'head = body' syntax, got: {text!r}")
                lhs_raw, body = text.split('=', 1)
                lhs = ','.join(p.strip() for p in lhs_raw.split(',') if p.strip())
                rule = self._parse_rule(lhs, body.strip(), space_role=space_role)
                # Apply width gate if specified.
                if width_raw is not None:
                    w_min, w_max = self._parse_width_attr(str(width_raw))
                    rule = rule._replace(
                        width_min=int(w_min), width_max=int(w_max))
                if query_raw is not None:
                    rule = rule._replace(
                        query=self._parse_bool_attr(query_raw))
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
                target.append(self._parse_rule(lhs, rhs, space_role=space_role))

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
    def _parse_bool_attr(value):
        """Parse loose XML boolean attribute values."""
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        return text in ('1', 'true', 'yes', 'y', 'on')

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

    def _parse_rule(self, lhs, rhs, space_role='SS'):
        """Parse one ``lhs = rhs`` rule string into a ``RuleDef`` namedtuple.

        ``rhs`` can be a function call (``f(A, B)``), a bare-symbol
        sequence (``A B``), or a single category. Accepts the explicit-
        direction suffixes ``.forward`` / ``.reverse`` on the function
        name. ``space_role`` is the per-rule routing tag.
        """
        # `space_role` may be 'SS' (symbols, default), 'CS' (concepts), or
        # 'subsymbolic' (percepts). Set by `_fill_section` from <symbols> /
        # <concepts> / <percepts> sub-sections under <parse> /
        # <generate>. Used by space-space_role filters at runtime to gate
        # which rules apply in each space's forward path.

        # Legacy tolerance: ``<C>C = pi(C)</C>`` is the per-space_role element
        # form where the element NAME is the LHS and the CONTENT is the
        # RHS.  Some configs (MM_20M, LM_5M etc.) redundantly prefix the
        # content with ``LHS = `` -- strip it so the function-call parser
        # below sees just ``pi(C)`` and ``method_name`` ends up as
        # ``pi`` (the natural-fold key) rather than ``C = pi`` (which
        # silently falls through ``_default_compose_rules``'s
        # ``_NATURAL_FOLD_METHODS`` filter and leaves the chart with no
        # CS-space_role rule, breaking ConceptualSpace dispatch).
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
            return self.RuleDef(space_role, canonical, arity, func_name,
                                lhs, tuple(args))
        if rhs == 'epsilon':
            return self.RuleDef(space_role, f"{lhs} -> epsilon", 0, None,
                                lhs, ())
        if rhs == lhs:
            return self.RuleDef(space_role, f"{lhs} -> {rhs}", 1, None,
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
            return self.RuleDef(space_role, f"{lhs} -> {rhs}", arity, method,
                                lhs, tuple(parts))
        raise ValueError(f"Cannot parse grammar rule: {lhs} -> {rhs}")

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
    # without splitting SS/CS into typed space_roles.

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

    def load_from_grammar_file(self, filename):
        """Configure rules from a ``data/<filename>.grammar`` XML file.

        Delegates parsing to the module-level :func:`load_grammar` (which
        returns a ``configure()``-compatible nested-dict) and then runs
        the standard ``configure()`` path. Once that's done, each
        ``RuleDef``'s space_role is overwritten with the space_role declared on its
        rule's ``GrammarLayer`` subclass -- the .grammar file lists the
        rule body only (``<rule>S = lift(NP, VP)</rule>``); category,
        arity, function name, and return-value count come from the body
        and space_role / invertibility come from the layer class.

        If the grammar file contains ``<start>...</start>``, it sets the
        accepted start patterns and primary start nonterminal for this
        grammar. An implicit identity rule for the current start symbol
        is added when one isn't already present; it's the no-op
        grammatical transition the static per-word loop's cursor
        bookkeeping relies on (see ``_find_identity_rule_id``).
        """
        cfg = load_grammar(filename)
        if isinstance(cfg, dict):
            # Space-scoped starts: <start> nested under <PartSpace>
            # configures the analyzer root; nested under <Symbolic>
            # configures the symbolic parse. A top-level <start> (legacy /
            # unsectioned form) configures the symbolic start unless the
            # Symbolic section declares its own.
            ps_block = cfg.get('PartSpace')
            ws_block = cfg.get('Symbolic')
            ps_start_raw = (ps_block.get('start')
                            if isinstance(ps_block, dict) else None)
            ws_start_raw = (ws_block.get('start')
                            if isinstance(ws_block, dict) else None)
            top_start_raw = cfg.pop('start', None)
            if ws_start_raw is None:
                ws_start_raw = top_start_raw
            self._configure_starts(ps_start_raw, ws_start_raw)
            # Strip the nested start keys so they can't be mistaken for
            # rules downstream (configure() reads compose/generate only).
            if isinstance(ps_block, dict):
                ps_block.pop('start', None)
            if isinstance(ws_block, dict):
                ws_block.pop('start', None)
        cfg = _expand_compact_order_sets(cfg)
        cfg = self._ensure_identity_rule(cfg)
        self.configure(cfg)
        self._reassign_space_roles_from_layer_classes()

    def _configure_starts(self, ps_start_raw, ws_start_raw):
        """Set space-scoped starts from the raw ``<start>`` blocks.

        ``ws_start_raw`` configures the WholeSpace starts and the
        back-compat global alias (``start_symbol`` / ``start_patterns``);
        the symbolic start is what ``id_SS`` / ``is_start_pattern`` /
        reset and relative-rule detection key off. ``ps_start_raw``
        configures the analyzer root (``U``) -- a separate namespace.
        """
        if ws_start_raw is not None:
            self.ws_start_patterns = _start_patterns_from_raw(
                ws_start_raw, default=self.ws_start_symbol)
            self.ws_start_symbol = _primary_start_symbol(
                self.ws_start_patterns, default=self.ws_start_symbol)
            names = _starts_by_name(ws_start_raw)
            self.ws_relative_starts = frozenset(
                s for s, n in names.items() if n == 'relative_truth')
            self.ws_absolute_starts = frozenset(
                s for s, n in names.items() if n == 'absolute_truth')
        else:
            self.ws_start_patterns = ((self.ws_start_symbol,),)
        # The symbolic start IS the global start (back-compat alias).
        self.start_patterns = self.ws_start_patterns
        self.start_symbol = self.ws_start_symbol
        if ps_start_raw is not None:
            self.ps_start_patterns = _start_patterns_from_raw(
                ps_start_raw, default='U')
            self.ps_start_symbol = _primary_start_symbol(
                self.ps_start_patterns, default='U')

    def _ensure_identity_rule(self, cfg):
        """Return ``cfg`` with a start-symbol identity in <compose> if
        none is present (handles bare-dict, ``{'rule': [...]}``, and
        named-section ``{'compose': {...}}`` shapes)."""
        identity_body = f"{self.start_symbol} = {self.start_symbol}"

        def has_identity(section):
            if not isinstance(section, dict):
                return False
            raw = section.get('rule')
            if isinstance(raw, str):
                return raw.replace(' ', '') == identity_body.replace(' ', '')
            if isinstance(raw, list):
                return any(
                    (isinstance(r, str)
                     and r.replace(' ', '') == identity_body.replace(' ', ''))
                    for r in raw)
            for k, v in section.items():
                if k == 'rule':
                    continue
                if isinstance(v, dict) and has_identity(v):
                    return True
            return False

        def add_identity(section):
            raw = section.get('rule')
            if raw is None:
                section['rule'] = [identity_body]
            elif isinstance(raw, str):
                section['rule'] = [identity_body, raw]
            elif isinstance(raw, list):
                section['rule'] = [identity_body] + list(raw)
            return section

        if not isinstance(cfg, dict):
            cfg = {'compose': {'rule': [identity_body]}}
            return cfg
        # PS/Symbolic-sectioned form: the identity rule is a symbolic no-op
        # transition, so it belongs in the Symbolic compose section.
        if 'PartSpace' in cfg or 'Symbolic' in cfg:
            ws = cfg.get('Symbolic')
            ws = dict(ws) if isinstance(ws, dict) else {}
            cfg['Symbolic'] = ws
            compose = ws.get('compose')
            if compose is None:
                ws['compose'] = {'rule': [identity_body]}
            elif isinstance(compose, dict) and not has_identity(compose):
                ws['compose'] = add_identity(dict(compose))
            return cfg
        if 'compose' in cfg or 'generate' in cfg:
            compose = cfg.get('compose')
            if compose is None:
                cfg['compose'] = {'rule': [identity_body]}
            elif isinstance(compose, dict) and not has_identity(compose):
                cfg['compose'] = add_identity(dict(compose))
            return cfg
        if not has_identity(cfg):
            cfg = add_identity(dict(cfg))
        return cfg

    def _reassign_space_roles_from_layer_classes(self):
        """Replace each rule's ``space_role`` with the value declared on its
        ``GrammarLayer`` subclass.

        The .grammar file format leaves space_role off the rule body; per the
        2026-05-29 refactor the layer class is the source of truth for
        space_role (and other class-level metadata such as ``invertible``).
        Rules whose ``method_name`` isn't registered in
        ``GRAMMAR_LAYER_CLASSES`` keep the space_role the parser inferred from
        the section header (default 'SS').
        """
        registry = GRAMMAR_LAYER_CLASSES

        def fixup(rules):
            out = []
            for rule in rules:
                cls = registry.get(_dispatch_method_name_for_rule(rule))
                if cls is not None:
                    new_space_role = getattr(cls, 'space_role', rule.space_role) or rule.space_role
                    if new_space_role != rule.space_role:
                        rule = rule._replace(space_role=new_space_role)
                out.append(rule)
            return out

        self.rules_upward = fixup(self.rules_upward)
        self.rules_downward = fixup(self.rules_downward)
        self.rules = list(self.rules_upward) + list(self.rules_downward)
        self.rule_table = {idx: rule.canonical
                           for idx, rule in enumerate(self.rules)}
        self._bump_rule_table_version()

    def _ensure_configured(self):
        """Lazily configure the grammar from XML on first use.

        Resolves the start symbol, then dispatches by precedence:
          1. ``<grammar>name.grammar</grammar>`` -- string body whose
             text matches ``*.grammar``: loaded via
             :py:meth:`load_from_grammar_file`.
          2. ``<grammar>...</grammar>`` -- inline XML grammar dict
             (legacy explicit form).
        Subsequent calls are no-ops via the ``_configured`` guard.
        """
        if self._configured:
            return
        # <start>...</start> in SymbolSpace.language: accepted completed
        # derivation shapes. The primary single-category start is kept
        # in ``start_symbol`` for legacy identity/reset code.
        try:
            start_raw = TheXMLConfig.get("SymbolSpace.language.start")
            self.start_patterns = _start_patterns_from_raw(
                start_raw, default="S")
            self.start_symbol = _primary_start_symbol(
                self.start_patterns, default="S")
            # The inline-XML grammar has no PS/WS sections; its start is
            # the symbolic start (mirror into the ws_* alias). A later
            # load_from_grammar_file (the <grammar>NAME.grammar</grammar>
            # dispatch) overrides this with the file's space-scoped starts.
            self.ws_start_patterns = self.start_patterns
            self.ws_start_symbol = self.start_symbol
            names = _starts_by_name(start_raw)
            self.ws_relative_starts = frozenset(
                s for s, n in names.items() if n == 'relative_truth')
            self.ws_absolute_starts = frozenset(
                s for s, n in names.items() if n == 'absolute_truth')
        except (KeyError, AttributeError):
            self.start_symbol = "S"
            self.start_patterns = (("S",),)
            self.ws_start_symbol = "S"
            self.ws_start_patterns = (("S",),)

        # Defensive: warn loudly if a deprecated <useGrammar> tag still
        # sits in the XML (the knob was retired 2026-05-13 but configs
        # that survived from before that may still carry it).
        try:
            legacy_use = TheXMLConfig.get("SymbolSpace.language.useGrammar")
            if legacy_use is not None:
                warnings.warn(
                    "<useGrammar> is deprecated; use "
                    "<grammar>NAME.grammar</grammar> instead. Falling "
                    "back to default.grammar.",
                    DeprecationWarning, stacklevel=2)
                self.load_from_grammar_file("default.grammar")
                try:
                    interp = TheXMLConfig.get(
                        "SymbolSpace.language.interpretation")
                    self.interpretation = float(interp)
                except (KeyError, AttributeError, TypeError, ValueError):
                    pass
                return
        except (KeyError, AttributeError):
            pass

        candidate = None
        try:
            candidate = TheXMLConfig.get("SymbolSpace.language.grammar")
        except (KeyError, AttributeError):
            candidate = None

        # New path: ``<grammar>NAME.grammar</grammar>`` -- string body
        # whose text resolves to a ``.grammar`` file in ``data/``.
        if isinstance(candidate, str):
            name = candidate.strip()
            if name.endswith(".grammar"):
                self.load_from_grammar_file(name)
                try:
                    interp = TheXMLConfig.get(
                        "SymbolSpace.language.interpretation")
                    self.interpretation = float(interp)
                except (KeyError, AttributeError, TypeError, ValueError):
                    pass
                return

        cfg = candidate if isinstance(candidate, dict) else None
        if cfg is None:
            cfg = self._NOOP_GRAMMAR
        self.configure(cfg)
        try:
            interp = TheXMLConfig.get("SymbolSpace.language.interpretation")
            self.interpretation = float(interp)
        except (KeyError, AttributeError, TypeError, ValueError):
            pass

    # -- Rule queries --------------------------------------------------

    def symbolic(self):
        """Return rule_ids whose space_role is 'SS' (symbolic-space_role rules)."""
        self._ensure_configured()
        return [i for i, r in enumerate(self.rules) if r.space_role == 'SS']

    def symbolic_transition(self):
        """Return rule_id of the unary space_role-SS transition rule, or None.

        Used by the symbolic head to find the unary-transition rule
        when the grammar exposes one (typically the no-op identity).
        """
        self._ensure_configured()
        for i, r in enumerate(self.rules):
            if r.space_role == 'SS' and r.method_name is None and r.arity == 1:
                return i
        return None

    @property
    def s_methods(self):
        """Set of method names available on the SS (symbolic) space_role.

        Excludes rules without a method_name (e.g. pure transitions).
        """
        return {r.method_name for r in self.rules if r.space_role == 'SS' and r.method_name}

    @property
    def categories(self):
        """Ordered tuple of unique derivation labels across all rules.

        Derived from both ``lhs`` (including comma-split multi-output
        heads) and ``rhs_symbols``. Used to size the category codebook
        on ``SymbolSpace``, so every label has its own learned embedding.
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
        """Return dict of method_name -> rule_id for SS-space_role operational rules."""
        result = {}
        for i, r in enumerate(self.rules):
            if r.space_role == 'SS' and r.method_name is not None:
                result[r.method_name] = i
        return result

    # -- Relative-truth rule marker (Task 6a) --------------------------
    #
    # doc/plans/2026-05-29-stm-serial-parallel-modes.md §7. A RELATIVE
    # truth is a binary predicate over two ideas (the ``part`` /
    # ``isEqual`` family — the ``relative_truth`` starts of the
    # role-collapsed grammars; ``REL_T`` in the archived transitional
    # ``test/fixtures/transitional_pos.grammar``); its
    # serial sentence-boundary reduce must STOP at the depth-3 end-state
    # ``[predicate, idea1, idea2]`` rather than collapsing to a single
    # idea (which is the correct end-state for an ABSOLUTE truth). The
    # marker below lets the reduce site ask "is this rule_id relative?".

    # Op-name signal (Phase R1.3,
    # doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
    # §6). The role-collapsed grammar names the relative-truth family
    # ``isEqual`` / ``isPart`` (each query-dispatched). The retired
    # ``queryPart`` / ``assertPart`` / ``part`` op names are folded into
    # ``isPart`` and no longer appear here; the transitional grammar's
    # ``queryPart`` / ``assertPart`` forward rules stay relative via the
    # ``lhs == REL_T`` start signal below.
    _RELATIVE_OP_NAMES = frozenset({'isEqual', 'isPart'})

    def _relative_start_categories(self):
        """Return the set of category symbols that head a RELATIVE start.

        Grammar-driven primary signal for ``is_relative_rule``: the
        WholeSpace starts tagged ``<start name="relative_truth">`` (the
        role-collapsed ``isEqual_O1`` / ``isPart_O1`` outputs), retained
        through parse on ``ws_relative_starts``. For grammars that do not
        name their starts (the inline-XML path, or a bare
        ``<start>REL_T</start>``), a single-symbol ``"REL_T"`` start
        pattern is treated as the relative start (back-compat fallback).
        Absolute-only grammars (MM_xor / MM_20M) expose no relative start
        and carry none of the relative ops -> nothing relative, the
        conservative correct answer.
        """
        if self.ws_relative_starts:
            return set(self.ws_relative_starts)
        cats = set()
        for pattern in (self.start_patterns or ()):
            if len(pattern) == 1 and pattern[0] == "REL_T":
                cats.add(pattern[0])
        return cats

    def _relative_rule_id_set(self):
        """Cached set of rule_ids that produce a RELATIVE truth.

        A rule is relative iff EITHER
          * its ``lhs`` is a relative start category (primary,
            grammar-driven -- see ``_relative_start_categories``), OR
          * its ``method_name`` is in :data:`_RELATIVE_OP_NAMES`
            (fallback op-name set).
        Computed once per rule-table version; invalidated by
        ``_bump_rule_table_version``.
        """
        self._ensure_configured()
        cache = self._relative_rule_ids_cache
        if cache is not None:
            return cache
        rel_starts = self._relative_start_categories()
        ids = set()
        for i, r in enumerate(self.rules):
            lhs = getattr(r, 'lhs', None)
            mn = getattr(r, 'method_name', None)
            if (lhs in rel_starts) or (mn in self._RELATIVE_OP_NAMES):
                ids.add(i)
        self._relative_rule_ids_cache = ids
        return ids

    def is_relative_rule(self, rule_id):
        """Return True iff ``rule_id`` produces a RELATIVE truth.

        See :meth:`_relative_rule_id_set`. Out-of-range ids and ids that
        cannot be coerced to int return False (conservative: an unknown
        rule is not relative).
        """
        try:
            rid = int(rule_id)
        except (TypeError, ValueError):
            return False
        return rid in self._relative_rule_id_set()

    # All compositional rules live on the unified SyntacticLayer class
    # as *Forward / *Reverse method pairs.  See _RULE_METHODS dispatch.

    # ---- Soft-superposition chart: packed-rule-table machinery -----
    #
    # `softChartCompose=true` on SymbolSpace activates a CKY-style inside
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
    # (configure / future add/remove). Consumers (e.g.
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
        # Task 6a: the relative-rule set is derived from ``self.rules``
        # + ``start_patterns``; both can change on a rule-table bump
        # (configure / space_role-reassign / legacy load), so invalidate.
        self._relative_rule_ids_cache = None

TheGrammar = Grammar()

# =====================================================================
# Grammar rule operator classes -- moved from Layers.py per
# doc/plans/2026-05-29-grammar-file-refactor.md §5. All derive from
# the GrammarLayer base class which stays in Layers.py (alongside
# PiLayer / SigmaLayer and the unmoved subsymbolic-computation
# grammar layers).
# =====================================================================

class NotLayer(GrammarLayer):
    """Parameter-free propositional negation on the bivalent symbol bivector.

    Implements the grammar rule ``S = not(S)``. Operates on the
    materialized muxed event tensor ``[B, V, nWhat + nWhere + nWhen]``;
    the ``.what`` bivector ``[pos, neg]`` lives at ``[..., :2]``
    (nWhat == 2 by convention) and any nWhere / nWhen channels follow.
    Negation swaps the leading 2 dims of the last axis to ``[neg, pos]``
    at every ``(B, V)`` position; nWhere / nWhen pass through unchanged.

    Contradictions are preserved: a position with both ``pos`` and
    ``neg`` high stays contradictory after the swap (new
    ``pos = old neg`` and new ``neg = old pos`` are still both high).
    Contrast with bitonic ``-x`` negation, which collapses
    contradictions onto opposite-sign components.

    The dispatcher hands NotLayer the materialized tensor (with the
    ``.active`` mask applied) -- never the codebook ``W``. Forward
    returns the muxed tensor with only the bivector channels swapped;
    the dispatcher writes back via ``set_event``.

    Shape-preserving and self-inverse.
    """
    rule_name  = "not"
    arity      = 1
    invertible = True
    space_role       = 'CS'

    def __init__(self):
        """Initialize NotLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)

    def forward(self, x):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        self._check_bivector_shape(x)
        bivector = x[..., :2].flip(dims=(-1,))
        rest     = x[..., 2:]
        if rest.shape[-1] == 0:
            return bivector
        return torch.cat([bivector, rest], dim=-1)

    def reverse(self, y):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return self.forward(y)

class NonLayer(GrammarLayer):
    """Non-affirming negation (indeterminacy) on a bivector.

    For a ``[pos, neg]`` bivector at each axis, ``non`` returns
    ``[1 - pos, 1 - neg]`` per pole independently. This is the
    pole-wise complement: a position fully affirmed
    (``pos = 1, neg = 0``) becomes ``[0, 1]`` (pure negation);
    indeterminate (``pos = 0, neg = 0``) becomes ``[1, 1]`` (full
    contradiction); contradictory (``pos = 1, neg = 1``) becomes
    ``[0, 0]`` (full indeterminacy). The four corners of the
    tetralemma exchange via this map:

        affirm    [1,0] <-> [0,1] negate
        unknown   [0,0] <-> [1,1] contradict

    Self-inverse on each pole independently
    (``non(non(x)) = 1 - (1 - x) = x``), shape-preserving. Operates
    on the leading bivector slice ``[..., :2]`` of the muxed event;
    nWhere / nWhen channels at ``[..., 2:]`` pass through unchanged
    (same convention as ``NotLayer``).
    """
    rule_name  = "non"
    arity      = 1
    invertible = True
    space_role       = 'CS'

    def __init__(self):
        """Initialize NonLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)

    def forward(self, x):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        self._check_bivector_shape(x)
        bivector = 1.0 - x[..., :2]
        rest     = x[..., 2:]
        if rest.shape[-1] == 0:
            return bivector
        return torch.cat([bivector, rest], dim=-1)

    def reverse(self, y):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return self.forward(y)

class IntersectionLayer(GrammarLayer):
    """``intersection(C, C)`` -- per-pole "min toward zero" on
    a bivector activation tensor.

    Runtime ``space_role='CS'``: the operator is a lattice-min
    primitive that binds at the conceptual space_role. The dispatcher
    feeds it the bivector activation -- ``[B, V, 2]`` per position,
    ``[pos, neg]`` poles -- via ``reads_activation = True``. The
    operands' upstream space_role (CS vs SS codebook activation) is
    determined by the chart binding, not by this layer.

    Math via ``Ops.intersection`` (a public alias of
    ``Ops._conjunction_kernel``):
        monotonic=False (default) -> RadMin: same-sign min
            magnitude, zero passthrough. The pole closer to
            zero wins per channel.
        monotonic=True            -> strict lattice min.

    Lossy: min collapses dominated operands. ``decompose`` returns
    ``(parent, parent)`` as the best-effort identity recovery
    without auxiliary structure.

    Stage 6 (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):
    butterfly cascade mode (``butterfly=True, N=N``) lifts the
    binary fold to a cross-STM aggregator. At each cascade level
    the per-pair op takes the packed ``[a, b]`` (each ``[B, M, D]``),
    computes the intersection kernel element-wise across the two
    halves (RadMin in radial mode, lattice min in monotonic mode),
    broadcasts the result back into the packed ``2D`` form, and
    applies the per-node weight. After ``log2(N)`` levels every
    output position holds the cross-position intersection of all
    inputs, weighted by the cascade's per-node parameters.
    Identity-on-constant-input is preserved (min is idempotent on
    the diagonal); general inputs lose information (lossy fold).
    """
    rule_name        = "intersection"
    arity            = 2
    invertible       = False
    lossy            = True
    space_role             = 'CS'
    reads_activation = True

    def __init__(self, monotonic=False, nInput=0, nOutput=0,
                 butterfly=False, N=None):
        """Initialize IntersectionLayer; allocate state for the class contract.

        See class docstring for invariants.

        ``butterfly`` / ``N`` (Stage 6): when ``butterfly=True``, the
        layer becomes a cross-STM cascade aggregator. ``N`` is the
        per-position slot count (power of two); ``nInput == nOutput``
        defines D, the per-slot feature width.
        """
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)
        self.monotonic = bool(monotonic)

    # -- Butterfly per-pair op (Stage 6) ------------------------------
    def _butterfly_pair_op(self, x_pair, W_node):
        """Intersection per-pair op for the butterfly cascade.

        Split ``x_pair: [B, M, 2D]`` into the two halves ``a, b``
        (each ``[B, M, D]``), apply the intersection kernel element-
        wise (RadMin radial / lattice min monotonic), broadcast the
        result into the ``2D`` packed form, and weight by
        ``W_node: [M, 2D, 2D]``.

        At identity init (``W_node = I``) the output is
        ``cat([min(a,b), min(a,b)])`` -- idempotent on constant input.
        """
        D = self._butterfly_D
        a = x_pair[..., :D]
        b = x_pair[..., D:]
        if self.monotonic:
            m = torch.minimum(a, b)
        else:
            m = Ops._radmin(a, b)
        packed = torch.cat([m, m], dim=-1)
        return torch.einsum('bmi,mij->bmj', packed, W_node)

    def _butterfly_pair_op_reverse(self, y_pair, W_inv_node):
        """Reverse of ``_butterfly_pair_op``; approximate (lossy).

        The forward broadcasts the per-pair fold into both halves
        and applies the per-node weight. The reverse un-weights with
        ``W_inv_node`` then averages the two halves and re-broadcasts
        -- the natural ``(parent, parent)`` pseudo-inverse adapted to
        the cascade's packed-pair form.
        """
        unweighted = torch.einsum('bmi,mij->bmj', y_pair, W_inv_node)
        D = self._butterfly_D
        a_rec = unweighted[..., :D]
        b_rec = unweighted[..., D:]
        avg = 0.5 * (a_rec + b_rec)
        return torch.cat([avg, avg], dim=-1)

    def forward(self, left, right=None):
        """Forward pass.

        Non-butterfly mode (default): binary ``forward(left, right)``
        applies the intersection kernel directly on the operand pair.

        Butterfly mode (``self.butterfly``): unary ``forward(x)``
        runs the cascade over the per-position axis of ``x: [B, N, D]``;
        ``right`` is ignored.

        See class docstring for the operation this layer applies.
        """
        if self.butterfly:
            return self._butterfly_forward(left)
        return Ops.intersection(left, right, monotonic=self.monotonic)

    def reverse(self, parent, basis=None,
                left_rows=None, right_rows=None,
                left_priming=None, right_priming=None):
        """Reverse pass; inverse of ``forward``.

        Non-butterfly mode:
          * ``basis is None`` (default) -- lossy ``(parent, parent)``
            pseudo-inverse (kept for callers that don't have a
            codebook handy).
          * ``basis`` supplied (a Codebook / Basis with ``getW()``) --
            mereology-guided recommender via
            :py:meth:`Ops.conjunctionReverse`: walks ``W = basis.getW()``
            for an operand pair ``(x1, x2)`` such that
            ``intersection(x1, x2) ≈ parent``.

        Butterfly mode: cross-STM cascade reverse via the per-pair
        pseudo-inverse (broadcast averaged form); ``basis`` is ignored.

        Callers (signal-router dispatch, chart reverse) are expected
        to pass the relevant Basis (typically ``WholeSpace.
        subspace.what``) at the call site -- no back-ref is stored on
        the layer. See class docstring for the inversion contract.

        ``left_rows`` / ``right_rows`` (optional ``LongTensor``):
            typed/heat candidate restriction for x1 / x2 selection;
            forwarded to :py:meth:`Ops.conjunctionReverse`.
            Default ``None`` = current behavior (all rows eligible).

        ``left_priming`` / ``right_priming`` (optional ``FloatTensor``):
            soft boost-above-unity priming for the inverse recommender;
            forwarded to :py:meth:`Ops.conjunctionReverse`.
            Default ``None`` = identity (byte-identical to prior behavior).
        """
        if self.butterfly:
            return self._butterfly_reverse(parent)
        if basis is not None:
            # Mereology recommender; falls back internally to
            # (parent, parent) when W is empty.
            W = basis.getW() if hasattr(basis, 'getW') else None
            if W is not None:
                return Ops.conjunctionReverse(
                    parent, parent, W, monotonic=self.monotonic,
                    left_rows=left_rows, right_rows=right_rows,
                    left_priming=left_priming, right_priming=right_priming)
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        if self.butterfly:
            # In butterfly mode the binary op is the cross-STM cascade
            # on a single packed tensor; compose still expects the
            # binary signature for chart parsing, so we concatenate
            # ``left`` and ``right`` along the position axis and run
            # the cascade. Caller responsibility: shapes must match
            # the configured N (typically left and right are halves).
            x = torch.cat([left, right], dim=-2)
            return self._butterfly_forward(x)
        return Ops.intersection(left, right, monotonic=self.monotonic)

    def generate(self, parent, basis=None,
                 left_rows=None, right_rows=None,
                 left_priming=None, right_priming=None):
        """Drive the reverse / generation pass.

        ``basis`` (optional Codebook/Basis) forwarded to ``reverse``;
        see its docstring for the recommender vs lossy-fallback
        semantics.

        ``left_rows`` / ``right_rows`` / ``left_priming`` /
        ``right_priming``: forwarded verbatim to ``reverse``; see that
        method's docstring for semantics. Default ``None`` = no
        restriction / identity priming (byte-identical to prior behavior).
        """
        return self.reverse(parent, basis=basis,
                            left_rows=left_rows, right_rows=right_rows,
                            left_priming=left_priming, right_priming=right_priming)

class UnionLayer(GrammarLayer):
    """``union(C, C)`` -- per-pole "max toward zero" (max
    magnitude, away from zero) on a bivector activation tensor.

    Runtime ``space_role='CS'`` -- counterpart to ``IntersectionLayer``.
    Same dispatch contract: feeds on bivector activation
    ``[B, V, 2]`` via the ``reads_activation = True`` flag. The
    operands' upstream space_role is determined by the chart binding,
    not by this layer.

    Math via ``Ops.union`` (a public alias of
    ``Ops._disjunction_kernel``):
        monotonic=False (default) -> RadMax: same-sign max
            magnitude with zero passthrough.
        monotonic=True            -> strict lattice max.

    Lossy with ``(parent, parent)`` pseudo-inverse on reverse.

    Stage 6 (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):
    butterfly cascade mode is the cross-STM dual of ``IntersectionLayer``
    -- per-pair op computes the union kernel element-wise across
    the two halves, broadcasts, and weights by the per-node matrix.
    See ``IntersectionLayer`` for the cascade-shape contract.
    """
    rule_name        = "union"
    arity            = 2
    invertible       = False
    lossy            = True
    space_role             = 'CS'
    reads_activation = True

    def __init__(self, monotonic=False, nInput=0, nOutput=0,
                 butterfly=False, N=None):
        """Initialize UnionLayer; allocate state for the class contract.

        See class docstring for invariants.

        ``butterfly`` / ``N`` (Stage 6): see ``IntersectionLayer``.
        """
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)
        self.monotonic = bool(monotonic)

    # -- Butterfly per-pair op (Stage 6) ------------------------------
    def _butterfly_pair_op(self, x_pair, W_node):
        """Union per-pair op for the butterfly cascade.

        Split ``x_pair: [B, M, 2D]`` into the two halves ``a, b``,
        apply the union kernel element-wise (RadMax radial / lattice
        max monotonic), broadcast into ``2D``, weight by ``W_node``.
        """
        D = self._butterfly_D
        a = x_pair[..., :D]
        b = x_pair[..., D:]
        if self.monotonic:
            m = torch.maximum(a, b)
        else:
            m = Ops._radmax(a, b)
        packed = torch.cat([m, m], dim=-1)
        return torch.einsum('bmi,mij->bmj', packed, W_node)

    def _butterfly_pair_op_reverse(self, y_pair, W_inv_node):
        """Reverse of ``_butterfly_pair_op``; lossy ``(parent, parent)``
        analogue adapted to the packed-pair form."""
        unweighted = torch.einsum('bmi,mij->bmj', y_pair, W_inv_node)
        D = self._butterfly_D
        a_rec = unweighted[..., :D]
        b_rec = unweighted[..., D:]
        avg = 0.5 * (a_rec + b_rec)
        return torch.cat([avg, avg], dim=-1)

    def forward(self, left, right=None):
        """Forward pass.

        Non-butterfly: binary ``forward(left, right)`` -> ``Ops.union``.
        Butterfly: unary ``forward(x)`` -> cross-STM cascade; ``right``
        ignored.

        See class docstring for the operation this layer applies.
        """
        if self.butterfly:
            return self._butterfly_forward(left)
        return Ops.union(left, right, monotonic=self.monotonic)

    def reverse(self, parent, basis=None,
                left_rows=None, right_rows=None,
                left_priming=None, right_priming=None):
        """Reverse pass; inverse of ``forward``.

        Non-butterfly mode:
          * ``basis is None`` (default) -- lossy ``(parent, parent)``
            pseudo-inverse.
          * ``basis`` supplied (a Codebook / Basis with ``getW()``) --
            mereology-guided recommender via
            :py:meth:`Ops.disjunctionReverse`: walks ``W = basis.getW()``
            for an operand pair ``(x1, x2)`` such that
            ``union(x1, x2) ≈ parent``.

        Butterfly mode: cross-STM cascade reverse; ``basis`` is ignored.

        Callers pass the relevant Basis (typically ``WholeSpace.
        subspace.what``) at the call site -- no back-ref is stored on
        the layer. See class docstring for the inversion contract.

        ``left_rows`` / ``right_rows`` (optional ``LongTensor``):
            typed/heat candidate restriction for x1 / x2 selection;
            forwarded to :py:meth:`Ops.disjunctionReverse`.
            Default ``None`` = current behavior (all rows eligible).

        ``left_priming`` / ``right_priming`` (optional ``FloatTensor``):
            soft boost-above-unity priming for the inverse recommender;
            forwarded to :py:meth:`Ops.disjunctionReverse`.
            Default ``None`` = identity (byte-identical to prior behavior).
        """
        if self.butterfly:
            return self._butterfly_reverse(parent)
        if basis is not None:
            W = basis.getW() if hasattr(basis, 'getW') else None
            if W is not None:
                return Ops.disjunctionReverse(
                    parent, parent, W, monotonic=self.monotonic,
                    left_rows=left_rows, right_rows=right_rows,
                    left_priming=left_priming, right_priming=right_priming)
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        if self.butterfly:
            x = torch.cat([left, right], dim=-2)
            return self._butterfly_forward(x)
        return Ops.union(left, right, monotonic=self.monotonic)

    def generate(self, parent, basis=None,
                 left_rows=None, right_rows=None,
                 left_priming=None, right_priming=None):
        """Drive the reverse / generation pass.

        ``basis`` (optional Codebook/Basis) forwarded to ``reverse``.

        ``left_rows`` / ``right_rows`` / ``left_priming`` /
        ``right_priming``: forwarded verbatim to ``reverse``; see that
        method's docstring for semantics. Default ``None`` = no
        restriction / identity priming (byte-identical to prior behavior).
        """
        return self.reverse(parent, basis=basis,
                            left_rows=left_rows, right_rows=right_rows,
                            left_priming=left_priming, right_priming=right_priming)


# ===========================================================================
# Grammar-op GrammarLayer subclasses (Surface 3 facade, 2026-05-01).
#
# Each class below names one grammar operation (`rule_name`) and exposes
# the canonical GrammarLayer interface (forward / reverse / compose /
# decompose, plus `gated_run` from the base class). The math kernel
# delegates to the corresponding `SyntacticLayer.*Forward` /
# `*Reverse` method so semantics are byte-identical to the existing
# `_RULE_METHODS` dispatch path. The benefit is uniform surface:
# `isinstance(x, GrammarLayer)` and `x.rule_name` work for every op,
# the chart's per-rule per-cell dispatch can route through these
# subclasses without a separate `_RULE_METHODS` table, and
# `_chart_authority`'s gating applies uniformly via `gated_run`.
#
# The subclasses are stateless wrappers that look up the method by
# name on a SyntacticLayer instance passed in at call time -- they
# don't own a SyntacticLayer reference (which would be a circular
# ownership: SyntacticLayer constructs them via the chart's eager
# build, and they'd point back at SyntacticLayer). Pattern:
#
#     out = LiftLayer().forward(left, right, layer=syntactic_layer,
#                               subspace=subspace)
#
# Existing `_RULE_METHODS` dispatch in SyntacticLayer.project still
# works -- this is an addition, not a replacement. A follow-up can
# migrate the chart to dispatch through these classes.

# =====================================================================
# Grammar-op GrammarLayer subclasses (Step 8 of the 2026-05-01 syntactic-
# layer refactor). Each class is a self-contained GrammarLayer with
# direct Ops-based math. Replaces the prior `_GrammarOpFacade` /
# SyntacticLayer-dispatch pattern.
#
# `_GrammarOpFacade._registry` is retired: the chart now consults
# `wordSpace.host_layer(space_role, rule_name)` first (Step 7) and falls
# back to a hardcoded class lookup `GRAMMAR_LAYER_CLASSES` declared
# at module scope below.
# =====================================================================

# --- Event modality helpers (modality re-architecture, Phase 3) ----------
# CS-space_role grammar ops operate on the muxed event [what | where | when]
# (architecture.canonical_shape("ConceptualSpace") == (2, 2)). These split /
# reassemble the event and extend/retract the .when span. Content-only
# operands (width == the op's .what content width) bypass these and take the
# legacy content fold, so the WS-space_role route stays content-only.
_EVENT_WHEN_WIDTH = 2


def _split_event(x, content_width, when_width=_EVENT_WHEN_WIDTH):
    """Split a muxed event into (what, where, when) given the .what width."""
    what = x[..., :content_width]
    rest = x[..., content_width:]
    when = rest[..., -when_width:]
    where = rest[..., :-when_width]
    return what, where, when


def _event_when_encoding(when_width=_EVENT_WHEN_WIDTH):
    from Spaces import WhenRangeEncoding
    return WhenRangeEncoding(n_when=when_width)      # default period (_WHEN_PERIOD)


def _when_extent(when, when_width=_EVENT_WHEN_WIDTH):
    """The event DURATION (extent) decoded from a .when bracket tail (0 for an
    instant). Under the 2026-06-16 bracket redesign the magnitude is duration,
    not tense; tense is the interval-vs-now relation (see WhenRangeEncoding)."""
    enc = _event_when_encoding(when_width)
    _center, ext = enc.decode(when.detach())
    fe = ext.reshape(-1)
    return float(fe[0]) if fe.numel() else 0.0


def _lift_when(when, when_width=_EVENT_WHEN_WIDTH):
    """Advance the event one tense step toward the FUTURE (the
    verb-advances-future rule of spec Section 5), preserving the event duration.
    Under the 2026-06-16 .when bracket redesign tense is the interval position
    relative to now, so LIFT shifts the event-time CENTER by +_WHEN_TENSE_STEP
    ticks (``shift_time``) rather than rescaling a magnitude."""
    from Spaces import _WHEN_TENSE_STEP
    enc = _event_when_encoding(when_width)
    return enc.shift_time(when, +_WHEN_TENSE_STEP)


def _lower_when(when, when_width=_EVENT_WHEN_WIDTH):
    """Inverse of _lift_when: retreat the event one tense step toward the PAST
    (-_WHEN_TENSE_STEP ticks), preserving the event duration."""
    from Spaces import _WHEN_TENSE_STEP
    enc = _event_when_encoding(when_width)
    return enc.shift_time(when, -_WHEN_TENSE_STEP)


def _make_lex_gate(n_in, rank, seed, bias=1.4722):
    """Build the §2 lexical-mask projection (word code -> gate rank
    space) WITHOUT touching the global RNG (GrammarOpsPass §2).

    ``nn.Linear``'s constructor draws its kaiming init from the global
    generator; grammar operators are built inside seeded fixtures, so
    that draw would shift every later seeded tensor. ``skip_init``
    materializes the module without an init draw; the deterministic
    init (small weight, bias at atanh(0.9): near-identity gates that
    already differ per word) comes from a LOCAL generator.

    ``bias`` (default ``atanh(0.9)``): the constant bias fill. The adverb
    eigenvalue edit (``LiftLayer._adv_edit``) passes ``bias=0.0`` so an
    untrained edit is ``tanh(0) ~= 0`` -- the residual barely perturbs the
    sigma fold until training shapes it.
    """
    lex = torch.nn.utils.skip_init(nn.Linear, int(n_in), int(rank))
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    with torch.no_grad():
        lex.weight.copy_(
            torch.randn(lex.weight.shape, generator=gen) * 0.05)
        lex.bias.fill_(float(bias))
    return lex


def _rotate_where(where, theta=0.6):
    """Rotate the 2-dim .where block by a fixed angle -- the prepositional
    relation's deterministic, invertible effect on the phrase's spatial/
    relational extent (placeholder until per-marker gating is learned; spec
    Section 5 'PREPOSITION modifies .where')."""
    if where.shape[-1] < 2:
        return where
    import math as _math
    c, s = _math.cos(theta), _math.sin(theta)
    x, y = where[..., 0:1], where[..., 1:2]
    rot = torch.cat([c * x - s * y, s * x + c * y], dim=-1)
    if where.shape[-1] > 2:
        rot = torch.cat([rot, where[..., 2:]], dim=-1)
    return rot


class LiftLayer(GrammarLayer):
    """``lift(idea_a, idea_b)`` -- binary CS-space_role grammar op (sigma-style
    synthesis over a STM pair).

    Stage 4 (2026-05-27, doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):
    LiftLayer becomes a first-class binary ``GrammarLayer`` subclass
    dispatched by the signal router over STM pairs. The previous
    substrate-borrow pattern (reaching into a PartSpace-owned
    sigma fold via a gating round-trip through the symbol codebook)
    is retired -- LiftLayer owns its own internal SigmaLayer for the
    pairwise math and is fully self-contained.

    Math (sigma-style binary fold, additive log-domain):

        s = atanh(idea_a) + atanh(idea_b)      (clamped)
        y = tanh(W @ s + b)

        # reverse / generate (balanced split):
        s_hat = W^-1 @ atanh(y) - b            (LDU inverse)
        half  = s_hat / 2
        return tanh(half), tanh(half)

    Inherits ``compose`` / ``generate`` semantics from ``SigmaLayer``'s
    binary-tensor-op contract (``SigmaLayer.compose`` / ``.generate``)
    -- the internal SigmaLayer carries the LDU-invertible
    parameterisation.  ``reverse(parent)`` returns a ``(left, right)``
    pair via the balanced split.

    The construction is self-contained: no PartSpace /
    ConceptualSpace / WholeSpace references.  Pass ``nInput`` /
    ``nOutput`` to size the internal SigmaLayer; the back-compat
    keyword arguments ``wholeSpace`` / ``perceptualSpace`` /
    ``conceptualSpace`` are still accepted (for legacy call sites in
    ``Language.SymbolSubSpace._attach_per_space_syntactic_layer``) but
    only their codebook dimension is consulted to determine the
    operand width when ``nInput`` is not provided.

    Lexical mask (GrammarOpsPass §2, author sign-off 2026-06-11;
    supersedes the 2026-06-10 "declared validity mask" note): the
    masks are LEXICAL, not validity -- syntactic validity is the
    grammar files' job (GrammarOpsPass §1), and ``lift`` stays total
    over its grammar-licensed operands. The per-call ``gate`` low-rank
    slicing (``_d_effective``) is lexical modulation of this SHARED
    operator matrix: one verb operator, a per-word gate selecting its
    slice -- walking vs running without a weight matrix per verb.
    ``lexical_gate(code)`` is the producer: one learned projection per
    operator from the word's code to the gate rank space (tanh-bounded
    per the ``_d_effective`` convention), trained end-to-end with the
    composition losses. ``gate=None`` (the default everywhere) is
    byte-identical to the un-gated baseline.
    """
    rule_name  = "lift"
    arity      = 2
    invertible = True
    space_role       = 'CS'
    event_aware = True          # operates on the muxed event (extends .when)

    def __init__(self, nInput=None, nOutput=None, *,
                 wholeSpace=None, perceptualSpace=None,
                 conceptualSpace=None,
                 invertible=True, nonlinear=True,
                 butterfly=False, N=None,
                 force_verb_spectrum=False,
                 force_adverb_eig_edit=False):
        """Initialize LiftLayer.

        ``nInput`` / ``nOutput`` size the internal SigmaLayer; both
        default to the symbol codebook width when ``wholeSpace`` is
        supplied (back-compat with the per-space syntactic-layer
        construction path).  ``invertible`` / ``nonlinear`` are passed
        through to the internal SigmaLayer.

        ``butterfly`` / ``N`` (Stage 5): when True, allocate a butterfly
        cascade on the GrammarLayer base alongside the inner sigma. The
        binary ``forward(left, right)`` retains its existing semantics;
        the new ``forward_butterfly(x)`` form (or the GrammarLayer
        ``_butterfly_forward`` directly) runs the cascade over a
        ``[B, N, D]`` per-position tensor. The cascade's per-pair op
        delegates to the internal SigmaLayer's ``_butterfly_pair_op``.

        Back-references via ``object.__setattr__`` bypass nn.Module
        submodule tracking so the Space refs don't create module-tree
        cycles (the Spaces are registered under the top-level Model).
        """
        # Resolve operand width: prefer explicit ``nInput`` / ``nOutput``;
        # else fall back to the symbol codebook width for the legacy
        # ``LiftLayer(wholeSpace=ws)`` call shape.
        if nInput is None:
            if wholeSpace is not None:
                nInput = int(wholeSpace.subspace.what.nDim)
            else:
                nInput = 0
        if nOutput is None:
            nOutput = nInput
        super().__init__(int(nInput), int(nOutput),
                         butterfly=butterfly, N=N)
        # .what content width -- splits a muxed CS-space_role event operand
        # ([what | where | when]) from a content-only operand.
        object.__setattr__(self, '_content_width', int(nInput))
        # Back-references kept for back-compat with the
        # ``LiftLayer(wholeSpace=...)`` legacy construction path.
        # They are NOT consulted by ``forward`` / ``reverse`` -- the
        # binary fold runs entirely through ``self._sigma``.
        object.__setattr__(self, 'wholeSpace', wholeSpace)
        object.__setattr__(self, 'perceptualSpace', perceptualSpace)
        object.__setattr__(self, 'conceptualSpace', conceptualSpace)
        # Internal SigmaLayer -- additive log-domain pairwise fold.
        # LDU-invertible so ``reverse`` is exact at the spatial level.
        # The inner sigma is used both for the binary forward(left, right)
        # path and (via its ``_butterfly_pair_op`` method) for the
        # butterfly per-pair op when the cascade is enabled.
        if int(nInput) > 0:
            self._sigma = SigmaLayer(
                nInput=int(nInput), nOutput=int(nOutput),
                invertible=invertible, nonlinear=nonlinear)
            self.layers.append(self._sigma)
            # Lexical-mask projection (GrammarOpsPass §2): the word's
            # code -> the inner LDU's gate rank space. One projection
            # per operator, shared across the vocabulary; the per-verb
            # difference rides entirely in this embedding->gate map.
            # Init near-identity (small weight, bias at atanh(0.9)) so
            # untrained gates barely perturb the shared matrix while
            # already differing per word. Initialized GLOBAL-RNG-NEUTRAL
            # (skip_init + local generator): constructing the operator
            # must not shift the seeded draws of fixtures built after it.
            self._lex_gate = _make_lex_gate(
                int(nInput), min(int(nInput), int(nOutput)), seed=0x11F7)
            # Eigen mechanisms (both eig-based, confirmed 2026-06-20): the VERB is
            # the eig-spectrum OPERATOR (<verbSpectrum>, below); the ADVERB is the
            # eigenmodifier (<adverbEigEdit>). What was removed is the verbEigEdit
            # RESIDUAL (a sparse edit on top of the symmetric sigma fold) -- it is
            # now the adverb.
            #
            # ADVERB sparse eigenvalue edit. When <adverbEigEdit> is on, or
            # AdverbLayer forces this helper, an adverb modifies a composed VP
            # through a sparse eigenvalue edit of the verb, masked by the VP's
            # OWN eigen-signature (no learned mask). The only per-adverb
            # parameter is the edit projection δ_adv(ADV_code). Plain LiftLayer
            # still builds it ONLY when flagged so flag-off is byte-identical.
            # Zero-init -> edit ~0.
            self._adverb_eig_edit = (
                bool(force_adverb_eig_edit)
                or bool(TheXMLConfig.get(
                    "architecture.adverbEigEdit", default=False)))
            if self._adverb_eig_edit:
                self._adv_edit = _make_lex_gate(
                    int(nInput), int(nOutput), seed=0x5EED, bias=0.0)
                with torch.no_grad():
                    self._adv_edit.weight.zero_()
            else:
                self._adv_edit = None
            self._adverb_purchase = None

            # VERB eig-spectrum OPERATOR (Stage 1; doc/old/2026-06-20-idea-
            # decoder.md "VP parameterization"). When <verbSpectrum> is on, the
            # verb acts on the NP as Q·diag(e^w)·Qᵀ -- here the first increment is
            # the exp-diagonal (Q = I) in atanh-space: VP(NP) = tanh(e^{w_v} ⊙
            # atanh(NP)), with w_v the verb's SPARSE log-eigenvalues from the verb
            # code (soft-thresholded, zero-init). INVERTIBLE by construction
            # (e^w>0): unapply applies e^{-w_v}. Sparse -> most eigs are identity
            # (w=0 -> e^0=1), so the verb is a non-destructive spectral reshaping.
            # (Follow-ons: NP-conditional Q via Householder-from-NP; verb
            # identification on reverse.) Built ONLY when flagged, or when
            # VerbLayer forces this helper, so plain LiftLayer stays flag-off
            # byte-identical.
            self._verb_spectrum = (
                bool(force_verb_spectrum)
                or bool(TheXMLConfig.get(
                    "architecture.verbSpectrum", default=False)))
            if self._verb_spectrum:
                self._verb_spec = _make_lex_gate(
                    int(nInput), int(nOutput), seed=0x5B17, bias=0.0)
                with torch.no_grad():
                    self._verb_spec.weight.zero_()
            else:
                self._verb_spec = None
        else:
            # Zero-width construction (parameter-free harness probe).
            self._sigma = None
            self._lex_gate = None
            self._adverb_eig_edit = False
            self._adv_edit = None
            self._adverb_purchase = None
            self._verb_spectrum = False
            self._verb_spec = None

    # -- Butterfly per-pair op delegation (Stage 5) -----------------
    def _butterfly_pair_op(self, x_pair, W_node):
        """Delegate per-pair op to the internal SigmaLayer.

        The cascade weight ``W_node`` is supplied by the LiftLayer's
        own ``butterfly_W`` (inherited from GrammarLayer); the inner
        sigma contributes only the atanh / tanh nonlinearity around
        the einsum. This keeps LiftLayer's invertibility contract on
        the cascade while reusing sigma's per-pair kernel.
        """
        if self._sigma is None:
            raise RuntimeError(
                "LiftLayer._butterfly_pair_op: no internal sigma "
                "(zero-width construction).")
        return self._sigma._butterfly_pair_op(x_pair, W_node)

    def _butterfly_pair_op_reverse(self, y_pair, W_inv_node):
        """Reverse of ``_butterfly_pair_op``; delegate to inner sigma."""
        if self._sigma is None:
            raise RuntimeError(
                "LiftLayer._butterfly_pair_op_reverse: no internal sigma.")
        return self._sigma._butterfly_pair_op_reverse(y_pair, W_inv_node)

    def forward_butterfly(self, x):
        """Run the butterfly cascade forward over a ``[B, N, D]`` tensor.

        Distinct entry point from the binary ``forward(left, right)``
        so the existing binary-fold callers (chart / signal-router
        pair dispatch) are not surprised by the cascade semantics.
        """
        if not self.butterfly:
            raise RuntimeError(
                "LiftLayer.forward_butterfly: butterfly mode not enabled "
                "at construction.")
        return self._butterfly_forward(x)

    def reverse_butterfly(self, y):
        """Inverse of ``forward_butterfly``."""
        if not self.butterfly:
            raise RuntimeError(
                "LiftLayer.reverse_butterfly: butterfly mode not enabled.")
        return self._butterfly_reverse(y)

    def lexical_gate(self, code):
        """Per-word gate over the shared operator matrix (GrammarOpsPass
        §2: one matrix, many verbs).

        ``code``: the word's lexical embedding (its codebook row),
        ``[D]`` or ``[..., D]`` -- leading dims are flattened and
        averaged (a graded blend when several words prime one call);
        a wider row (muxed event) is sliced to the content width.
        Returns a tanh-bounded ``[rank]`` gate for ``forward`` /
        ``reverse`` ``gate=``. ``None`` code (or a zero-width layer)
        returns ``None``, so callers can pass the result through
        unconditionally.
        """
        if code is None or getattr(self, '_lex_gate', None) is None:
            return None
        cw = self._content_width
        v = code.reshape(-1, code.shape[-1])
        if cw and v.shape[-1] > cw:
            v = v[..., :cw]
        v = v.to(self._lex_gate.weight.dtype).mean(dim=0)
        return torch.tanh(self._lex_gate(v))

    def forward(self, left, right, gate=None):
        """Sigma-style binary fold over the STM pair ``(left, right)``.

        Delegates to ``SigmaLayer.compose`` which packs the operands
        in atanh-domain (``a + b``), applies the inner linear
        transform, and returns ``tanh(W @ (atanh(a) + atanh(b)) + b)``.

        ``gate`` (optional ``[rank]``, see ``lexical_gate``): lexical
        slice of the shared operator; ``None`` is byte-identical to
        the un-gated baseline.
        """
        if self._sigma is None:
            raise RuntimeError(
                "LiftLayer was constructed without operand-width "
                "information; cannot run forward. Pass nInput / "
                "nOutput, or supply a wholeSpace whose subspace "
                "carries a non-empty codebook.")
        cw = self._content_width
        if cw and left.shape[-1] > cw:
            # Muxed event: fold the .what content via the inner sigma; pass
            # the left (subject) operand's .where through; extend the
            # result's .when span > 1 with the center advanced.
            l_what, l_where, _l_when = _split_event(left, cw)
            r_what, _r_where, r_when = _split_event(right, cw)
            # The verb is the lift operator itself; no eig-edit is applied here
            # (the eig-based VERB edit was removed). An ADVERB, when present,
            # modifies this composed VP via ``apply_adverb``.
            content = self._sigma.compose(l_what, r_what, gate=gate)
            return torch.cat([content, l_where, _lift_when(r_when)], dim=-1)
        return self._sigma.compose(left, right, gate=gate)

    def apply_adverb(self, vp_content, adv_what):
        """ADVERB eigenmodifier: an adverb modifies a VP by a sparse eigenvalue
        edit of the verb. (The eig-based VERB edit was removed -- the verb is the
        lift operator; the ADVERB is the eigenmodifier: "ADV modifies the eigs of
        VP.") Invoked by the adverb operator over a composed VP; NOT called by
        ``lift.forward`` (the verb no longer eig-edits).

        No-op (returns ``vp_content`` unchanged) unless ``<adverbEigEdit>`` or
        ``AdverbLayer`` built the edit projection. When on:

        * ``δ_adv`` -- the adverb's per-eigenfeature edit from the ADV code, made
          SPARSE by soft-thresholding (an adverb touches few eigs);
        * ``p_vp`` -- the mask = the VP's OWN normalized eigen-signature (its
          active eigenfeatures); the adverb edits the eigenvalues the verb
          actually uses and preserves the rest;
        * the edit is the masked residual ``a2 = atanh(vp) + p_vp ⊙ δ_adv``.
          ``adverb_purchase`` is stashed for introspection (the firewall delta)."""
        if not getattr(self, '_adverb_eig_edit', False) or self._adv_edit is None:
            return vp_content
        a = torch.atanh(vp_content.clamp(-1 + epsilon, 1 - epsilon))
        # δ_adv from the ADV code, soft-thresholded for sparsity (few eigs).
        delta = torch.tanh(self._adv_edit(adv_what.to(self._adv_edit.weight.dtype)))
        tau = 0.1
        delta = torch.sign(delta) * torch.clamp(delta.abs() - tau, min=0.0)
        # p_vp: the VP-eigen mask = the verb's normalized eigen-signature in
        # [0, 1]; no learned parameter (the adverb edits the verb's active eigs).
        amax = a.abs().amax(dim=-1, keepdim=True)
        p_vp = a.abs() / (amax + epsilon)
        with torch.no_grad():
            num = (p_vp * a).norm(dim=-1)
            den = a.norm(dim=-1) + epsilon
            self._adverb_purchase = (num / den).detach()
        return torch.tanh(a + p_vp * delta)

    def _verb_spectrum_w(self, verb_what):
        """The verb's SPARSE log-eigenvalues w_v from the verb code,
        soft-thresholded so most eigs are 0 (-> e^0 = 1 -> identity)."""
        w = self._verb_spec(verb_what.to(self._verb_spec.weight.dtype))
        tau = 0.1
        return torch.sign(w) * torch.clamp(w.abs() - tau, min=0.0)

    def apply_verb(self, np_what, verb_what):
        """VERB eig-spectrum OPERATOR (Stage 1): act on the NP as the verb's
        diagonal spectrum in atanh-space -- VP(NP) = tanh(e^{w_v} ⊙ atanh(NP)),
        with w_v the verb's SPARSE log-eigenvalues (first increment: Q = I).
        INVERTIBLE by construction (e^w > 0; see ``unapply_verb``). Most eigs are
        identity (w=0 -> e^0=1), so the verb is a non-destructive spectral
        reshaping along the few eigs it touches. No-op unless ``<verbSpectrum>``
        or ``VerbLayer`` built the projection."""
        if not getattr(self, '_verb_spectrum', False) or self._verb_spec is None:
            return np_what
        a = torch.atanh(np_what.clamp(-1 + epsilon, 1 - epsilon))
        return torch.tanh(torch.exp(self._verb_spectrum_w(verb_what)) * a)

    def unapply_verb(self, vp_what, verb_what):
        """Inverse of ``apply_verb`` GIVEN the verb: NP = tanh(e^{-w_v} ⊙
        atanh(VP)). Exact round-trip with ``apply_verb`` for the same verb code.
        (Recovering the verb from the result alone -- the operator-identification
        reverse -- is a follow-on increment.)"""
        if not getattr(self, '_verb_spectrum', False) or self._verb_spec is None:
            return vp_what
        a = torch.atanh(vp_what.clamp(-1 + epsilon, 1 - epsilon))
        return torch.tanh(torch.exp(-self._verb_spectrum_w(verb_what)) * a)

    def reverse(self, parent, gate=None, basis=None,
                left_rows=None, right_rows=None,
                left_priming=None, right_priming=None):
        """Split ``parent`` back into a ``(left, right)`` pair.

        The ``.what`` content is recovered by the mereology-guided NEAREST-
        PROTOTYPE recommender (``Ops.liftReverseAll`` -> ``disjunctionReverse``
        -> ``_binary_op_recommend``) when a ``basis`` codebook ``W`` is present:
        it returns REAL stored constituents drawn from the codebook, so two
        distinct operands are recovered by recognition (non-destructive prototype
        match, storing no partition). With no basis it falls back to
        ``SigmaLayer.generate`` -- the partition-blind balanced split
        ``tanh(s/2), tanh(s/2)`` -- which stays the inverse for default-only /
        XOR grammars that carry no codebook. ``.where`` is copied to both
        children and ``.when`` retracted (exact). ``gate``: the forward gate; the
        LDU inverse uses ``1/(d * gate)`` automatically.
        """
        if self._sigma is None:
            raise RuntimeError(
                "LiftLayer was constructed without operand-width "
                "information; cannot run reverse.")
        W = basis.getW() if (basis is not None
                             and hasattr(basis, 'getW')) else None

        def _split_what(p_what):
            if W is not None and hasattr(W, 'shape') and int(W.shape[0]) > 0:
                return Ops.liftReverseAll(
                    p_what, W=W, left_rows=left_rows, right_rows=right_rows,
                    left_priming=left_priming, right_priming=right_priming)
            return self._sigma.generate(p_what, gate=gate)

        cw = self._content_width
        if cw and parent.shape[-1] > cw:
            p_what, p_where, p_when = _split_event(parent, cw)
            lc, rc = _split_what(p_what)
            back = _lower_when(p_when)
            return (torch.cat([lc, p_where, back], dim=-1),
                    torch.cat([rc, p_where, back], dim=-1))
        return _split_what(parent)

    def compose(self, left, right, gate=None):
        """Binary GrammarLayer compose entry -- routes to ``forward``."""
        return self.forward(left, right, gate=gate)

    def generate(self, parent, gate=None):
        """Binary GrammarLayer generate entry -- routes to ``reverse``."""
        return self.reverse(parent, gate=gate)


class VerbLayer(LiftLayer):
    """``verb(np, verb)`` -- sparse verb-conditioned spectral operator.

    The operator applies the verb's log-eigenvalue spectrum to the NP in
    atanh-space via :meth:`LiftLayer.apply_verb`. It forces the spectrum
    projection on even when the legacy ``<verbSpectrum>`` helper flag is off,
    so the grammar rule is live by construction. Zero-init means an untrained
    verb is the identity.
    """
    rule_name = "verb"
    arity = 2
    invertible = True
    space_role = 'CS'
    event_aware = True
    # The unreduce dispatcher must supply the verb operand to invert
    # apply_verb; without it there is no faithful inverse, so it skips this
    # op's reverse rather than fabricating a split (see reverse()).
    reverse_required_kwargs = ('verb_what',)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("force_verb_spectrum", True)
        super().__init__(*args, **kwargs)

    def forward(self, left, right, gate=None):
        if self._sigma is None:
            raise RuntimeError(
                "VerbLayer was constructed without operand-width "
                "information; cannot run forward.")
        cw = self._content_width
        if cw and left.shape[-1] > cw:
            l_what, l_where, _l_when = _split_event(left, cw)
            r_what, _r_where, r_when = _split_event(right, cw)
            content = self.apply_verb(l_what, r_what)
            return torch.cat([content, l_where, _lift_when(r_when)], dim=-1)
        return self.apply_verb(left, right)

    def reverse(self, parent, verb_what=None, basis=None, gate=None, **kwargs):
        """Exact inverse of :meth:`forward` -- REQUIRES the verb operand.

        ``unapply_verb`` strips the verb's log-eigenvalue spectrum from the
        parent to recover the NP. Without ``verb_what`` there is no faithful
        reconstruction, so this RAISES rather than fabricating a split (the
        old ``(left, left)`` pseudo-split was a placeholder). Callers that
        cannot supply the operand must not invoke reverse -- the unreduce
        dispatcher gates on ``reverse_required_kwargs``.
        """
        if self._sigma is None:
            raise RuntimeError(
                "VerbLayer was constructed without operand-width "
                "information; cannot run reverse.")
        if verb_what is None:
            raise RuntimeError(
                "VerbLayer.reverse requires the verb operand (verb_what) to "
                "invert apply_verb; refusing to fabricate a split.")
        return self.unapply_verb(parent, verb_what), verb_what


class AdverbLayer(LiftLayer):
    """``adverb(vp, adv)`` -- sparse adverb eigenmodifier.

    The operator applies :meth:`LiftLayer.apply_adverb` to a VP and an adverb
    code. It forces the adverb edit projection on, independent of the legacy
    ``<adverbEigEdit>`` helper flag. Zero-init means an untrained adverb is a
    no-op.
    """
    rule_name = "adverb"
    arity = 2
    invertible = False
    lossy = True
    space_role = 'CS'
    event_aware = True
    # No faithful inverse exists; the unreduce dispatcher skips this op's
    # reverse entirely rather than calling it (which would raise).
    reverse_dispatchable = False

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("force_adverb_eig_edit", True)
        super().__init__(*args, **kwargs)

    def forward(self, left, right, gate=None):
        if self._sigma is None:
            raise RuntimeError(
                "AdverbLayer was constructed without operand-width "
                "information; cannot run forward.")
        cw = self._content_width
        if cw and left.shape[-1] > cw:
            l_what, l_where, l_when = _split_event(left, cw)
            r_what, _r_where, _r_when = _split_event(right, cw)
            content = self.apply_adverb(l_what, r_what)
            return torch.cat([content, l_where, l_when], dim=-1)
        return self.apply_adverb(left, right)

    def reverse(self, parent, basis=None, gate=None, **kwargs):
        """Adverb editing is lossy (``invertible = False``) -- there is no
        faithful inverse, so reverse RAISES rather than fabricating one (the
        old ``(parent, parent)`` was a placeholder). The unreduce dispatcher
        gates on ``reverse_dispatchable = False`` and must not invoke this.
        """
        raise NotImplementedError(
            "AdverbLayer is lossy (invertible=False); apply_adverb has no "
            "exact inverse, so reverse is unavailable.")


class LowerLayer(GrammarLayer):
    """``lower(idea_a, idea_b)`` -- binary CS-space_role grammar op (pi-style
    lowering over a STM pair).

    Stage 4 (2026-05-27, doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):
    LowerLayer becomes a first-class binary ``GrammarLayer`` subclass
    dispatched by the signal router over STM pairs. The previous
    substrate-borrow pattern (reaching into a ConceptualSpace-owned
    pi fold via a gating round-trip through the symbol codebook) is
    retired -- LowerLayer owns its own internal PiLayer for the
    pairwise math and is fully self-contained.

    Math (pi-style binary fold, multiplicative log-domain):

        ell = log_mult(idea_a) + log_mult(idea_b)
        y   = tanh((W @ ell + b) / 2)

        # reverse / generate (balanced split):
        ell_hat = W^-1 @ (atanh-style transform of y) - b
        half    = ell_hat / 2
        return from_mult(exp(half)), from_mult(exp(half))

    Inherits ``compose`` / ``generate`` semantics from ``PiLayer``'s
    binary-tensor-op contract (``PiLayer.compose`` / ``.generate``)
    -- the internal PiLayer carries the LDU-invertible
    parameterisation.

    The construction is self-contained: no Space references.  See
    ``LiftLayer`` for the keyword-arg compatibility shim.

    Lexical mask (GrammarOpsPass §2; supersedes the 2026-06-10
    "declared validity mask" note): see the ``LiftLayer`` docstring --
    the same lexical-modulation contract applies to ``lower``
    (``lexical_gate(code)`` producer; ``gate=`` threading to the
    inner PiLayer; ``gate=None`` byte-identical to baseline).
    """
    rule_name  = "lower"
    arity      = 2
    invertible = True
    space_role       = 'CS'
    event_aware = True          # operates on the muxed event (retracts .when)

    def __init__(self, nInput=None, nOutput=None, *,
                 wholeSpace=None, perceptualSpace=None,
                 conceptualSpace=None,
                 invertible=True, nonlinear=True,
                 butterfly=False, N=None):
        """Initialize LowerLayer; symmetric to ``LiftLayer.__init__``
        but with an internal PiLayer (multiplicative log-domain
        fold) instead of a SigmaLayer.

        ``butterfly`` / ``N`` (Stage 5): when True, allocate a
        butterfly cascade on the GrammarLayer base. The per-pair op
        delegates to the internal PiLayer's ``_butterfly_pair_op``.
        """
        if nInput is None:
            if wholeSpace is not None:
                nInput = int(wholeSpace.subspace.what.nDim)
            else:
                nInput = 0
        if nOutput is None:
            nOutput = nInput
        super().__init__(int(nInput), int(nOutput),
                         butterfly=butterfly, N=N)
        object.__setattr__(self, '_content_width', int(nInput))
        object.__setattr__(self, 'wholeSpace', wholeSpace)
        object.__setattr__(self, 'perceptualSpace', perceptualSpace)
        object.__setattr__(self, 'conceptualSpace', conceptualSpace)
        if int(nInput) > 0:
            self._pi = PiLayer(
                nInput=int(nInput), nOutput=int(nOutput),
                invertible=invertible, nonlinear=nonlinear)
            self.layers.append(self._pi)
            # Lexical-mask projection (GrammarOpsPass §2); see LiftLayer.
            self._lex_gate = _make_lex_gate(
                int(nInput), min(int(nInput), int(nOutput)), seed=0x10E7)
        else:
            self._pi = None
            self._lex_gate = None

    # -- Butterfly per-pair op delegation (Stage 5) -----------------
    def _butterfly_pair_op(self, x_pair, W_node):
        """Delegate per-pair op to the internal PiLayer."""
        if self._pi is None:
            raise RuntimeError(
                "LowerLayer._butterfly_pair_op: no internal pi "
                "(zero-width construction).")
        return self._pi._butterfly_pair_op(x_pair, W_node)

    def _butterfly_pair_op_reverse(self, y_pair, W_inv_node):
        """Reverse of ``_butterfly_pair_op``; delegate to inner pi."""
        if self._pi is None:
            raise RuntimeError(
                "LowerLayer._butterfly_pair_op_reverse: no internal pi.")
        return self._pi._butterfly_pair_op_reverse(y_pair, W_inv_node)

    def forward_butterfly(self, x):
        """Run the butterfly cascade forward over ``[B, N, D]``."""
        if not self.butterfly:
            raise RuntimeError(
                "LowerLayer.forward_butterfly: butterfly mode not enabled.")
        return self._butterfly_forward(x)

    def reverse_butterfly(self, y):
        """Inverse of ``forward_butterfly``."""
        if not self.butterfly:
            raise RuntimeError(
                "LowerLayer.reverse_butterfly: butterfly mode not enabled.")
        return self._butterfly_reverse(y)

    # Per-word gate producer (GrammarOpsPass §2); same contract as
    # LiftLayer.lexical_gate (one learned projection per operator).
    lexical_gate = LiftLayer.lexical_gate

    def forward(self, left, right, gate=None):
        """Pi-style binary fold (multiplicative log-domain) over the
        STM pair ``(left, right)``.

        Delegates to ``PiLayer.compose`` which packs the operands in
        log-mult domain (``log_mult(a) + log_mult(b)``), applies the
        inner linear transform, and returns
        ``tanh((W @ (log_mult(a) + log_mult(b)) + b) / 2)``.

        ``gate`` (optional ``[rank]``, see ``lexical_gate``): lexical
        slice of the shared operator; ``None`` is byte-identical to
        the un-gated baseline.
        """
        if self._pi is None:
            raise RuntimeError(
                "LowerLayer was constructed without operand-width "
                "information; cannot run forward. Pass nInput / "
                "nOutput, or supply a wholeSpace whose subspace "
                "carries a non-empty codebook.")
        cw = self._content_width
        if cw and left.shape[-1] > cw:
            # Muxed event: fold the .what content via the inner pi; pass the
            # left operand's .where through; RETRACT the result's .when span
            # back toward a unit point (the inverse of LIFT).
            l_what, l_where, _l_when = _split_event(left, cw)
            r_what, _r_where, r_when = _split_event(right, cw)
            content = self._pi.compose(l_what, r_what, gate=gate)
            return torch.cat([content, l_where, _lower_when(r_when)], dim=-1)
        return self._pi.compose(left, right, gate=gate)

    def reverse(self, parent, gate=None, basis=None,
                left_rows=None, right_rows=None,
                left_priming=None, right_priming=None):
        """Split ``parent`` back into a ``(left, right)`` pair.

        The ``.what`` content is recovered by the mereology-guided NEAREST-
        PROTOTYPE recommender (``Ops.lowerReverseAll`` -> ``conjunctionReverse``
        -> ``_binary_op_recommend``) when a ``basis`` codebook ``W`` is present
        (real stored constituents, non-destructive prototype match); with no
        basis it falls back to the partition-blind balanced log-mult split
        (``PiLayer.generate``). ``.where`` is copied to both children and
        ``.when`` lifted (exact). ``gate``: the forward gate; the LDU inverse
        uses ``1/(d * gate)`` automatically.
        """
        if self._pi is None:
            raise RuntimeError(
                "LowerLayer was constructed without operand-width "
                "information; cannot run reverse.")
        W = basis.getW() if (basis is not None
                             and hasattr(basis, 'getW')) else None

        def _split_what(p_what):
            if W is not None and hasattr(W, 'shape') and int(W.shape[0]) > 0:
                return Ops.lowerReverseAll(
                    p_what, W=W, left_rows=left_rows, right_rows=right_rows,
                    left_priming=left_priming, right_priming=right_priming)
            return self._pi.generate(p_what, gate=gate)

        cw = self._content_width
        if cw and parent.shape[-1] > cw:
            p_what, p_where, p_when = _split_event(parent, cw)
            lc, rc = _split_what(p_what)
            back = _lift_when(p_when)
            return (torch.cat([lc, p_where, back], dim=-1),
                    torch.cat([rc, p_where, back], dim=-1))
        return _split_what(parent)

    def compose(self, left, right, gate=None):
        """Binary GrammarLayer compose entry -- routes to ``forward``."""
        return self.forward(left, right, gate=gate)

    def generate(self, parent, gate=None):
        """Binary GrammarLayer generate entry -- routes to ``reverse``."""
        return self.reverse(parent, gate=gate)

class PrepositionLayer(GrammarLayer):
    """preposition(P, X) -- marker-headed phrase packaging (binary, CS-space_role).

    Packages a learned surface marker P (that / to / in / because / when)
    with a phrase X (NP / VP / S). Transparent to X's content: forward(P, X)
    returns X unchanged so a downstream lift / intersection reads the
    phrase. The marker is recorded through the base-class absorb / emit
    machinery, NOT folded into content -- PREPOSITION does not decide the
    final relation; that is learned from how the marker-headed phrase
    participates downstream (spec "Operation 1: PREPOSITION").

    Starts PERMISSIVE: any [B, N, D] content is accepted as X. Per-marker
    argument gating (NP-only `in` vs S-only `that`) is a learned hook for
    later (bound_markers participation), not built here. reverse is the
    structural (X, X) split: the content side recovers the phrase exactly;
    the marker side is realized by emit from the bound marker.
    """
    rule_name = "preposition"; arity = 2
    invertible = True; lossy = False; space_role = 'CS'; reads_activation = False
    event_aware = True          # operates on the muxed event (modifies .where)

    def __init__(self, nInput=0, nOutput=0, butterfly=False, N=None):
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)
        object.__setattr__(self, '_content_width', int(nInput))
    def forward(self, left, right):
        # P (left) is the marker (absorbed); X (right) is the phrase. On a
        # muxed CS-space_role event, PREPOSITION modifies X's .where (the spatial /
        # relational extent the marker imposes), leaving .what / .when. With
        # no content-width info it stays a pass-through (legacy contract).
        cw = self._content_width
        if cw and right.shape[-1] > cw:
            x_what, x_where, x_when = _split_event(right, cw)
            return torch.cat([x_what, _rotate_where(x_where), x_when], dim=-1)
        return right                       # P is the marker (absorbed), X passes through
    def reverse(self, parent):
        cw = self._content_width
        if cw and parent.shape[-1] > cw:
            p_what, p_where, p_when = _split_event(parent, cw)
            x = torch.cat([p_what, _rotate_where(p_where, theta=-0.6), p_when], dim=-1)
            return x, x
        return parent, parent              # (marker_placeholder, phrase); emit realizes the marker
    def compose(self, left, right):
        return self.forward(left, right)
    def generate(self, parent):
        return self.reverse(parent)

class ContextualBindLayer(GrammarLayer):
    """bind(BIND, VP) -- contextual missing-NP marker (binary, CS-space_role).

    Surface LIFT(BIND, VP) is reinterpreted at parse time as
    LIFT(resolved_ref, VP) (spec "Operation 2"). compose(BIND_marker, VP)
    resolves an accessible participant already constructed in the current
    parse and returns its vector, so the enclosing lift sees a real NP in
    the left slot. Two context modes (stashed via set_bind_context before
    dispatch; a plain attribute, not an nn.Module child):

      * slab=[B, N, D]  -- the live constituent slab from the fold
        (Task 2.4). Resolution is vectorized NEAREST-LEFT: for each adjacent
        pair p the missing NP is constituent p-1 (the most recently built
        phrase before the BIND operand). Position 0 has no left context ->
        the marker passes through. This is the locality branch of
        bind_resolver, expressed as a tensor roll so it runs inside the
        parallel fold.
      * participants=[Participant] + licensing -- the ranked path
        (bind_resolver.resolve_bind): want=>subject-control,
        persuade=>object-control + locality + learned participation. Used by
        fixtures / unit tests and as the licensing refinement over locality.

    No context, or no candidate => the marker passes through unchanged --
    BIND never invents a binding.
    """
    rule_name = "bind"; arity = 2
    invertible = False; lossy = True; space_role = 'CS'; reads_activation = False

    def __init__(self, nInput=0, nOutput=0, butterfly=False, N=None):
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)
        object.__setattr__(self, '_bind_context', None)
    def set_bind_context(self, *, slab=None, participants=None, licensing=None):
        object.__setattr__(self, '_bind_context',
                           {'slab': slab, 'participants': participants, 'licensing': licensing})
    def clear_bind_context(self):
        object.__setattr__(self, '_bind_context', None)
    def forward(self, left, right):
        ctx = self._bind_context
        if ctx and ctx.get('slab') is not None:
            slab = ctx['slab']                              # [B, N, D] live constituents
            # nearest-left participant for each pair p: constituent p-1;
            # position 0 keeps itself (no left context -> passthrough).
            prior = torch.cat([slab[:, :1, :], slab[:, :-1, :]], dim=1)  # [B, N, D]
            return prior[:, :-1, :]                          # [B, N-1, D], aligned to pairs
        if ctx and ctx.get('participants'):
            from bind_resolver import resolve_bind
            vec, _chosen = resolve_bind(ctx['participants'], licensing=ctx.get('licensing'))
            if vec is not None:
                return vec.expand_as(left) if vec.shape != left.shape else vec
        return left
    def reverse(self, parent):
        return parent, parent              # lossy: contextual binding is not recoverable
    def compose(self, left, right):
        return self.forward(left, right)
    def generate(self, parent):
        return self.reverse(parent)

class _WhenOpMixin:
    """Shared helpers for unary ops that rewrite the .when tail of a
    materialized muxed event [B, V, nWhat + nWhere + nWhen]. Modifies ONLY
    the trailing nWhen (=2) columns; .what / .where pass through. Builds a
    matching WhenRangeEncoding to interpret / rewrite the phasor. Tense
    operates on the VP/event .when BEFORE the subject LIFT (spec note:
    equivalent post-LIFT)."""
    _WHEN_WIDTH = 2
    def _when_encoding(self):
        from Spaces import WhenRangeEncoding
        return WhenRangeEncoding(n_when=self._WHEN_WIDTH)   # default period (_WHEN_PERIOD)
    def _split_when(self, x):
        w = self._WHEN_WIDTH
        if x.shape[-1] < w:
            raise ValueError(f"{type(self).__name__}: event width {x.shape[-1]} < "
                             f".when width {w}; is nWhen enabled?")
        return x[..., :-w], x[..., -w:]

class TenseLayer(_WhenOpMixin, GrammarLayer):
    """tense(X) -- move the event .when along TIME relative to ``now`` (unary,
    CS-space_role; 2026-06-16 .when bracket redesign). Tense is the interval-vs-now
    relation, NOT a magnitude: PRESENT is identity (event stays at its time);
    PAST shifts the event-time CENTER -step ticks (toward the past); FUTURE shifts
    it +step ticks (toward the future). The shift is an exact phase rotation that
    PRESERVES the event duration (WhenRangeEncoding.shift_time), so an event at
    ``now`` becomes ``now-step`` (past) / ``now+step`` (future). ``reverse``
    applies the inverse shift (invertible round-trip, modulo the period wrap).
    Selected per-instance via set_op before dispatch."""
    rule_name = "tense"; arity = 1
    invertible = True; lossy = False; space_role = 'CS'; reads_activation = False
    # _DELTA is the TENSE step applied to the event-time CENTER (in clock ticks):
    # PRESENT = no shift, PAST = -step (toward past), FUTURE = +step (toward
    # future). step == _WHEN_TENSE_STEP (1.0 tick).
    from Spaces import _WHEN_TENSE_STEP as _STEP
    _DELTA = {"PRESENT": 0.0, "PAST": -_STEP, "FUTURE": +_STEP}
    del _STEP
    def __init__(self, nInput=0, nOutput=0, butterfly=False, N=None):
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)
        object.__setattr__(self, '_op', "PRESENT")
    def set_op(self, tense):
        if tense not in self._DELTA: raise ValueError(f"unknown tense {tense!r}")
        object.__setattr__(self, '_op', tense)
    def forward(self, x):
        head, when = self._split_when(x); delta = self._DELTA[self._op]
        if delta == 0.0: return x
        return torch.cat([head, self._when_encoding().shift_time(when, delta)], dim=-1)
    def reverse(self, y):
        head, when = self._split_when(y); delta = self._DELTA[self._op]
        if delta == 0.0: return y
        return torch.cat([head, self._when_encoding().shift_time(when, -delta)], dim=-1)
    def compose(self, x):     return self.forward(x)
    def generate(self, parent): return self.reverse(parent)

class AspectLayer(_WhenOpMixin, GrammarLayer):
    """aspect(X) -- a no-op (identity).

    Under the 2026-06-16 .when bracket redesign the magnitude carries event
    DURATION again, so an aspect that reshapes duration (PERFECT / PROGRESSIVE)
    *could* return -- but that op is NOT redesigned here, so AspectLayer stays a
    no-op: ``forward`` / ``compose`` / ``reverse`` / ``generate`` are all identity
    (return the input unchanged). The class is KEPT (not deleted) so the grammar's
    ``aspect`` rule dispatch stays intact; ``set_op`` still validates the kind for
    back-compat callers (e.g. MorphologyLayer) but has no numerical effect. A
    duration-reshaping aspect (operating on the bracket extent) is a documented
    growth path."""
    rule_name = "aspect"; arity = 1
    # No-op: nothing is lost (identity), so it is trivially invertible.
    invertible = True; lossy = False; space_role = 'CS'; reads_activation = False
    _KINDS = ("SIMPLE", "PERFECT", "PROGRESSIVE")
    def __init__(self, nInput=0, nOutput=0, butterfly=False, N=None):
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)
        object.__setattr__(self, '_op', "SIMPLE"); object.__setattr__(self, '_eps', 0.25)
    def set_op(self, kind, eps=0.25):
        if kind not in self._KINDS: raise ValueError(f"unknown aspect kind {kind!r}")
        object.__setattr__(self, '_op', kind); object.__setattr__(self, '_eps', float(eps))
    def forward(self, x):      return x           # no-op (duration/aspect retired)
    def reverse(self, parent): return parent      # no-op
    def compose(self, x):      return self.forward(x)
    def generate(self, parent): return self.reverse(parent)

class MorphologyLayer(GrammarLayer):
    """morphology(X) -- decompose a surface word form into a base lemma +
    role-neutral morphological features, routing tense/aspect onto the event
    .when (unary, CS-space_role).

    Delegates ANALYSIS to ``surface_morphology.analyze`` and the tense/aspect
    .when math to ``TenseLayer`` / ``AspectLayer`` (NOT re-derived here -- the
    Phase-4 ops are reused). The surface token is supplied out-of-band via
    ``set_token`` (mirroring ``TenseLayer.set_op``); with no token set, forward
    is a pass-through.

    Setting ``.what`` to the lemma's concept needs the lexicon (a model-level
    resource) and is the documented growth path; this layer routes the
    tense/aspect features the ``.when`` ops consume. No global POS inventory --
    ``features`` are morphological annotations, not parts of speech.
    """
    rule_name = "morphology"; arity = 1
    invertible = True; lossy = True; space_role = 'CS'; reads_activation = False
    event_aware = True          # routes the analyzed tense/aspect onto .when

    def __init__(self, nInput=0, nOutput=0, butterfly=False, N=None):
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)
        object.__setattr__(self, '_token', None)
        object.__setattr__(self, '_tense', TenseLayer())
        object.__setattr__(self, '_aspect', AspectLayer())

    def set_token(self, token):
        """Stash the surface token ``analyze()`` decomposes (out-of-band,
        like ``TenseLayer.set_op``)."""
        object.__setattr__(self, '_token', token)

    def _analyze(self):
        import surface_morphology
        if not self._token:
            return None, {}
        return surface_morphology.analyze(self._token)

    def forward(self, x):
        _lemma, feats = self._analyze()
        if not feats:
            return x                        # plain / unknown token: pass through
        out = x
        self._tense.set_op(feats.get("tense", "PRESENT"))
        out = self._tense.compose(out)      # delegate the .when rotation
        for asp in feats.get("aspect", []):
            self._aspect.set_op(asp)
            out = self._aspect.compose(out)  # delegate the .when reshape
        return out

    def reverse(self, y):
        _lemma, feats = self._analyze()
        if not feats:
            return y
        out = y
        for asp in reversed(feats.get("aspect", [])):
            self._aspect.set_op(asp)
            out = self._aspect.generate(out)
        self._tense.set_op(feats.get("tense", "PRESENT"))
        out = self._tense.generate(out)
        return out

    def compose(self, x):     return self.forward(x)
    def generate(self, parent): return self.reverse(parent)


class SymbolizeLayer(GrammarLayer):
    """``symbolize(percept, symbol)`` -- binary CS-space_role grammar op that
    binds a perceptual idea to a semantic idea, creating a META node in
    the WS taxonomy.

    Stage 9 (2026-05-27, doc/plans/2026-05-27-perceptstore-meta-taxonomy-
    reentrancy.md): SymbolizeLayer is registered with the signal router
    as a binary reduce op at space_role 'CS'. Enforced at sentence-parse
    boundaries (the parser dispatches it for every word + object pairing
    in a sentence's parse). Off-parse, it fires opportunistically when
    CS state's quantization-to-symbol pairs with the next STM slot's
    quantization.

    Originally named ``MetaLayer``; renamed 2026-05-28 to better reflect
    the operation ("symbolize a percept into a symbol") rather than the
    artifact produced (a META node in the taxonomy). The META node
    concept survives unchanged on the WholeSpace side (see
    ``WholeSpace.insert_meta`` / ``taxonomy``).

    Semantic contract (forward):

        forward(left, right):
          - left:  CS-space_role vector derived from a PS percept.
          - right: CS-space_role vector derived from an WS symbol.
          1. Identify the percept_id by nearest-row search in
             ``PartSpace.percept_store.codebook``.
          2. Identify the symbol_idx by nearest-row search in
             ``WholeSpace.subspace.what.getW()``.
          3. Call ``WholeSpace.insert_meta(ps_idx, ws_idx,
             fused_vec=(left + right) / 2)``. The call is idempotent on
             the pair: a fresh allocation on first sight, EMA-blend on
             subsequent calls.
          4. Return the META node's WS.codebook vector.

    Semantic contract (reverse):

        reverse(parent):
          parent is a META vector.
          1. Find the nearest WS row to ``parent``.
          2. If the row is a META node, walk
             ``WholeSpace.taxonomy_children`` to recover the
             ``(ps_idx, ws_idx)`` children.
          3. Return ``(PS.codebook[ps_row], WS.codebook[ws_row])`` as
             the ``(left, right)`` recovery.
          4. If the nearest row is not a META, fall back to the
             balanced split ``(parent / 2, parent / 2)`` (analogous to
             LiftLayer.reverse's split convention; the discrete
             recovery is approximate when no META binding exists for
             the parent's identity).

    Fallback: when the wired ``PartSpace`` lacks a
    ``percept_store`` (legacy lexicon mode), SymbolizeLayer cannot
    identify the percept_id structurally; forward then returns the
    no-op average ``(left + right) / 2`` without registering a META
    node. This keeps the layer harmless when not in radix mode.

    Numerical guard: ``NaN`` / ``Inf`` in ``left`` or ``right`` raises
    immediately per the project's "fail loud on numerical divergence"
    policy. Silent ``nan_to_num`` would let divergence propagate into
    the WS codebook through the META row seed.

    The construction back-references ``wholeSpace`` /
    ``perceptualSpace`` via ``object.__setattr__`` to bypass nn.Module
    submodule tracking (the Spaces are registered under the top-level
    Model; reattaching them here would create a module-tree cycle).
    """
    rule_name  = "symbolize"
    arity      = 2
    invertible = True
    space_role       = 'CS'

    def __init__(self, nInput=None, nOutput=None, *,
                 wholeSpace=None, perceptualSpace=None,
                 conceptualSpace=None,
                 butterfly=False, N=None):
        """Initialize SymbolizeLayer.

        ``nInput`` / ``nOutput`` size the layer's nominal input / output
        widths; both default to the symbol codebook width when
        ``wholeSpace`` is supplied. The layer is parameter-free
        (all gradient flows through the WS / PS codebooks owned by
        the wired Spaces); the GrammarLayer base allocates a butterfly
        cascade if requested but the binary forward / reverse path
        does not consult it.

        Back-references kept for the discrete-identity lookups in
        ``forward`` / ``reverse``: PS percept_id (nearest-row in
        ``perceptualSpace.percept_store.codebook``), WS symbol_idx
        (nearest-row in ``wholeSpace.subspace.what.getW()``), and
        the META insert via ``wholeSpace.insert_meta``.
        """
        if nInput is None:
            if wholeSpace is not None:
                nInput = int(wholeSpace.subspace.what.nDim)
            else:
                nInput = 0
        if nOutput is None:
            nOutput = nInput
        super().__init__(int(nInput), int(nOutput),
                         butterfly=butterfly, N=N)
        object.__setattr__(self, 'wholeSpace', wholeSpace)
        object.__setattr__(self, 'perceptualSpace', perceptualSpace)
        object.__setattr__(self, 'conceptualSpace', conceptualSpace)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _finite_or_raise(name, tensor):
        """Fail loud on NaN / Inf in operand tensors.

        Per the project's "fail loud" policy (MEMORY:
        feedback_fail_loud_on_divergence): silent ``nan_to_num`` would
        propagate divergence into the WS codebook row holding the META
        seed. Raise immediately so the stack trace surfaces.
        """
        if not torch.is_tensor(tensor):
            return
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"SymbolizeLayer: operand '{name}' contains NaN/Inf. "
                f"Numerical divergence must surface, not be silently "
                f"propagated into the WS codebook via the META seed. "
                f"finite={int(torch.isfinite(tensor).sum().item())}/"
                f"{int(tensor.numel())}.")

    @staticmethod
    def _nearest_row(codebook, query):
        """Return the row idx of ``codebook`` nearest to ``query``.

        ``codebook``: ``[V, D]``. ``query``: ``[D]`` (or shape that
        squeezes to ``[D]``). Empty codebook returns ``None``.
        """
        if codebook is None:
            return None
        if codebook.dim() < 2 or codebook.shape[0] == 0:
            return None
        # Collapse query to a 1-D vector and move to codebook's
        # device/dtype.
        q = query.detach().to(codebook.device, codebook.dtype).reshape(-1)
        if q.shape[0] != codebook.shape[1]:
            return None
        diffs = codebook - q.unsqueeze(0)
        sq = (diffs * diffs).sum(dim=1)
        return int(torch.argmin(sq).item())

    def _ps_store(self):
        """Resolve the PerceptStore (or None if not in radix mode)."""
        ps_space = getattr(self, 'perceptualSpace', None)
        if ps_space is None:
            return None
        return getattr(ps_space, 'percept_store', None)

    def _ws_codebook(self):
        """Resolve the WS codebook tensor (or None)."""
        ws = getattr(self, 'wholeSpace', None)
        if ws is None:
            return None
        sub = getattr(ws, 'subspace', None)
        if sub is None:
            return None
        cb = getattr(sub, 'what', None)
        if cb is None:
            return None
        return cb.getW()

    # ------------------------------------------------------------------
    # Forward / reverse / compose / generate
    # ------------------------------------------------------------------

    def forward(self, left, right):
        """Bind ``(left, right)`` into a META node and return its vector.

        See the class docstring for the full contract. Returns the
        META row's WS.codebook vector (post-insert / EMA-update).
        Falls back to ``(left + right) / 2`` when no PerceptStore is
        wired (legacy lexicon mode).
        """
        # Fail loud on NaN/Inf in either operand before any
        # store access -- a divergent operand would silently seed a
        # divergent META row.
        self._finite_or_raise('left', left)
        self._finite_or_raise('right', right)
        ws = getattr(self, 'wholeSpace', None)
        ps_store = self._ps_store()
        # Legacy / no-PerceptStore fallback: cannot resolve percept_id
        # structurally, so return the average without registering a
        # META node. This keeps SymbolizeLayer harmless in lexicon mode.
        if ws is None or ps_store is None:
            return (left + right) / 2.0
        ps_codebook = getattr(ps_store, 'codebook', None)
        ws_W = self._ws_codebook()
        if ps_codebook is None or ws_W is None:
            return (left + right) / 2.0
        # Identify the PS percept_id by nearest-row in PS.codebook.
        ps_row = self._nearest_row(ps_codebook, left)
        # Identify the WS symbol_idx by nearest-row in WS.codebook.
        ws_row = self._nearest_row(ws_W, right)
        if ps_row is None or ws_row is None:
            # Empty stores: nothing to bind. Fall back to the average.
            return (left + right) / 2.0
        # Resolve nearest-row hits to positive-int positions, lazy-
        # binding if a row hasn't been seen before (pre-Stage-3 rows in
        # the WS codebook may pre-date the position counter).
        ps_pos = ws.ensure_ps_position(ps_row)
        ws_pos = ws.ensure_ws_position(ws_row, kind="ws")
        # Fused vec = combine(left, right). Stage 9 spec: start with the
        # simple average; learnable combine is a future revision.
        fused = (left + right) / 2.0
        # Cast / shape the fused vec to match WS.codebook.
        fused_for_insert = fused.detach().to(ws_W.device, ws_W.dtype)
        if fused_for_insert.dim() > 1:
            fused_for_insert = fused_for_insert.reshape(-1)
        if fused_for_insert.shape[0] != ws_W.shape[1]:
            # Width mismatch: caller passed a non-codebook-shaped vec.
            # Fall back to the no-op average.
            return fused
        # Delegate to WholeSpace.insert_meta. Idempotent on the
        # pair: first call allocates a fresh META row; subsequent calls
        # return the cached META position and EMA-update the stored vec.
        meta_pos = ws.insert_meta(ps_pos, ws_pos,
                                  fused_vec=fused_for_insert)
        meta_row = int(ws._ws_pos_to_row[meta_pos])
        # LBG (Linde-Buzo-Gray) accumulation + split-trigger on the WS
        # row this binding pulls on. The pull direction is the WS
        # operand ``right`` (a CS-space_role vector derived from a WS symbol);
        # accumulated displacement variance > threshold triggers a row
        # split, creating a new WS row + META binding so the codebook
        # grows organically as training reveals sub-clusters within
        # what was a single symbol.
        right_for_lbg = right.detach().to(ws_W.device, ws_W.dtype)
        if right_for_lbg.dim() > 1:
            right_for_lbg = right_for_lbg.reshape(-1)
        if right_for_lbg.shape[0] == ws_W.shape[1]:
            ws.record_lbg_pull(ws_pos, right_for_lbg)
            ws.maybe_split_lbg(ws_pos)
        # Return the (possibly EMA-updated) META row's vector. We read
        # the *current* codebook so callers see the EMA-blended state.
        ws_W_current = self._ws_codebook()
        return ws_W_current[meta_row]

    def reverse(self, parent):
        """Recover the ``(left, right)`` pair from a META vector.

        Walks WS.codebook nearest-match to find the META row, then
        looks up the children via the taxonomy. Returns ``(ps_vec,
        ws_vec)`` for the children's codebook rows. Falls back to the
        balanced split ``(parent / 2, parent / 2)`` when the nearest
        WS row is not a registered META node (no surface-bytes path
        to recover).
        """
        self._finite_or_raise('parent', parent)
        ws = getattr(self, 'wholeSpace', None)
        ps_store = self._ps_store()
        ws_W = self._ws_codebook()
        # Fallback: no Spaces wired or no PS store -- balanced split.
        if ws is None or ps_store is None or ws_W is None:
            half = parent / 2.0
            return half, half
        # Nearest WS row to parent.
        nearest_row = self._nearest_row(ws_W, parent)
        if nearest_row is None:
            half = parent / 2.0
            return half, half
        nearest_pos = ws._ws_row_to_pos.get(nearest_row)
        if nearest_pos is None:
            # Row isn't bound to a position (pre-Stage-3 WS row); no
            # taxonomy entry to walk.
            half = parent / 2.0
            return half, half
        children = ws.taxonomy_children(nearest_pos)
        if not children:
            # Nearest match wasn't a META node -- nothing to walk.
            half = parent / 2.0
            return half, half
        # Separate children by kind: "ps" child + "ws" child.
        ps_child = None
        ws_child = None
        for child in children:
            ci = int(child)
            kind = ws._pos_kind.get(ci)
            if kind == "ps" and ps_child is None:
                ps_child = ci
            elif kind == "ws" and ws_child is None:
                ws_child = ci
        if ps_child is None or ws_child is None:
            # Malformed META (missing a side); fall back.
            half = parent / 2.0
            return half, half
        ps_row = ws._ps_pos_to_row.get(ps_child)
        ws_row = ws._ws_pos_to_row.get(ws_child)
        if ps_row is None or ws_row is None:
            half = parent / 2.0
            return half, half
        ps_row = int(ps_row)
        ws_row = int(ws_row)
        # Range-check the rows; a stale taxonomy entry could carry an
        # out-of-bounds row idx, which would crash the index op.
        ps_codebook = getattr(ps_store, 'codebook', None)
        if (ps_codebook is None
                or ps_row >= ps_codebook.shape[0]
                or ws_row >= ws_W.shape[0]):
            half = parent / 2.0
            return half, half
        left = ps_codebook[ps_row]
        right = ws_W[ws_row]
        return left, right

    def compose(self, left, right):
        """Binary GrammarLayer compose entry -- routes to ``forward``."""
        return self.forward(left, right)

    def generate(self, parent):
        """Binary GrammarLayer generate entry -- routes to ``reverse``."""
        return self.reverse(parent)

class ConjunctionLayer(GrammarLayer):
    """``S -> conjunction(S, S)`` -- monotonic min on the
    post-codebook scalar activation.

    Symbolic-space_role conjunction is the AND of two **codebook
    activation patterns**. Per the 2026-05-05 directive,
    WholeSpace's ``materialize(mode='activation')`` returns
    the **post-codebook** activation -- a ``[B, V]`` *scalar*
    strength per prototype (``effective_activation()``: the
    bivector ``[pos, neg]`` reduced via ``max(pos, neg)`` and
    gated by modal presence). Conjunction over two such patterns
    asks "which prototypes are active in *both* operands".

    Because the post-codebook activation is non-negative scalar,
    the natural composition kernel is the **monotonic** lattice
    min: ``torch.minimum(x, y)``. RadMin (the bivector kernel)
    would be wrong here -- there's no negative pole to manage.
    The class hard-codes ``monotonic=True`` and forwards to
    ``Ops.intersection`` so the kernel collapses to ``torch.min``
    via ``_lower_kernel(kind='strict')``.

    Distinct from ``IntersectionLayer`` (CS-space_role): IntersectionLayer
    operates on a bivector ``[..., 2]`` activation (concept-space_role
    pre-codebook) and supports both RadMin and lattice-min;
    ConjunctionLayer operates on a *scalar* ``[B, V]`` post-
    codebook activation and is strictly monotonic.

    Lossy with ``(parent, parent)`` pseudo-inverse on reverse.

    Stage 6 (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):
    butterfly cascade mode applies the monotonic min cross-STM
    pair-wise (per-pair op = ``torch.minimum`` on the two halves,
    broadcast, weight). See ``IntersectionLayer`` for the cascade-
    shape contract; this op is hard-coded monotonic so the radial
    branch is never taken.
    """
    rule_name        = "conjunction"
    arity            = 2
    invertible       = False
    lossy            = True
    space_role             = 'SS'
    reads_activation = True

    def __init__(self, nInput=0, nOutput=0, butterfly=False, N=None):
        """Initialize ConjunctionLayer; allocate state for the class contract.

        See class docstring for invariants.

        ``butterfly`` / ``N`` (Stage 6): see ``IntersectionLayer``.
        """
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)

    # -- Butterfly per-pair op (Stage 6) ------------------------------
    def _butterfly_pair_op(self, x_pair, W_node):
        """Conjunction per-pair op for the butterfly cascade.

        Element-wise ``torch.minimum`` on the two halves (monotonic
        only -- post-codebook activations are non-negative scalar),
        broadcast into the packed ``2D`` form, weight by ``W_node``.
        """
        D = self._butterfly_D
        a = x_pair[..., :D]
        b = x_pair[..., D:]
        m = torch.minimum(a, b)
        packed = torch.cat([m, m], dim=-1)
        return torch.einsum('bmi,mij->bmj', packed, W_node)

    def _butterfly_pair_op_reverse(self, y_pair, W_inv_node):
        """Reverse of ``_butterfly_pair_op``; lossy ``(parent, parent)``
        adapted to the packed-pair form."""
        unweighted = torch.einsum('bmi,mij->bmj', y_pair, W_inv_node)
        D = self._butterfly_D
        a_rec = unweighted[..., :D]
        b_rec = unweighted[..., D:]
        avg = 0.5 * (a_rec + b_rec)
        return torch.cat([avg, avg], dim=-1)

    def forward(self, left, right=None):
        # Post-codebook activation is monotonic-only -- no negative
        # pole to manage, so RadMin would be wrong.
        """Forward pass.

        Non-butterfly: binary ``forward(left, right)`` -> monotonic
        intersection (lattice min).
        Butterfly: unary ``forward(x)`` -> cross-STM monotonic-min
        cascade; ``right`` ignored.

        See class docstring for the operation this layer applies.
        """
        if self.butterfly:
            return self._butterfly_forward(left)
        return Ops.intersection(left, right, monotonic=True)

    def reverse(self, parent, basis=None, left_rows=None, right_rows=None,
                left_priming=None, right_priming=None):
        """Reverse pass; inverse of ``forward``.

        ``basis`` supplied (a Codebook/Basis with ``getW()``) -> the mereology
        recommender :py:meth:`Ops.conjunctionReverse` recovers an operand pair
        ``(x1, x2)`` with ``intersection(x1, x2) ~= parent`` from the codebook
        rows -- EXACT on a discrete vocabulary (the XOR reconstruction path).
        The AND-fold is many-to-one, so ``basis is None`` (no codebook handy)
        falls back to the lossy ``(parent, parent)`` pseudo-inverse. Mirrors
        ``IntersectionLayer.reverse``; the reconstruction driver passes
        ``basis=space_role_basis`` at ``LanguageLayer.unreduce``.
        """
        if self.butterfly:
            return self._butterfly_reverse(parent)
        if basis is not None:
            W = basis.getW() if hasattr(basis, 'getW') else None
            if W is not None:
                return Ops.conjunctionReverse(
                    parent, parent, W, monotonic=True,
                    left_rows=left_rows, right_rows=right_rows,
                    left_priming=left_priming, right_priming=right_priming)
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        if self.butterfly:
            x = torch.cat([left, right], dim=-2)
            return self._butterfly_forward(x)
        return Ops.intersection(left, right, monotonic=True)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)

class DisjunctionLayer(GrammarLayer):
    """``S -> disjunction(S, S)`` -- monotonic max on the
    post-codebook scalar activation.

    Symbolic-space_role disjunction is the OR of two **codebook
    activation patterns**: ``[B, V]`` post-codebook scalar
    activation (see ``ConjunctionLayer`` for the activation-
    semantics rationale). The natural composition kernel is the
    monotonic lattice max ``torch.maximum(x, y)``; the class
    hard-codes ``monotonic=True`` and forwards to ``Ops.union``,
    which collapses to ``torch.max`` via
    ``_lift_kernel(kind='strict')``.

    Distinct from ``UnionLayer`` (CS-space_role): UnionLayer operates on
    a bivector ``[..., 2]`` activation and supports both RadMax
    and lattice-max; DisjunctionLayer operates on a scalar
    ``[B, V]`` post-codebook activation and is strictly monotonic.

    Lossy with ``(parent, parent)`` pseudo-inverse on reverse.

    Stage 6 (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):
    butterfly cascade mode applies the monotonic max cross-STM
    pair-wise (per-pair op = ``torch.maximum``). See
    ``ConjunctionLayer`` / ``IntersectionLayer`` for the cascade-
    shape contract.
    """
    rule_name        = "disjunction"
    arity            = 2
    invertible       = False
    lossy            = True
    space_role             = 'SS'
    reads_activation = True

    def __init__(self, nInput=0, nOutput=0, butterfly=False, N=None):
        """Initialize DisjunctionLayer; allocate state for the class contract.

        See class docstring for invariants.

        ``butterfly`` / ``N`` (Stage 6): see ``IntersectionLayer``.
        """
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)

    # -- Butterfly per-pair op (Stage 6) ------------------------------
    def _butterfly_pair_op(self, x_pair, W_node):
        """Disjunction per-pair op for the butterfly cascade.

        Element-wise ``torch.maximum`` on the two halves (monotonic
        only -- post-codebook activations are non-negative scalar),
        broadcast into the packed ``2D`` form, weight by ``W_node``.
        """
        D = self._butterfly_D
        a = x_pair[..., :D]
        b = x_pair[..., D:]
        m = torch.maximum(a, b)
        packed = torch.cat([m, m], dim=-1)
        return torch.einsum('bmi,mij->bmj', packed, W_node)

    def _butterfly_pair_op_reverse(self, y_pair, W_inv_node):
        """Reverse of ``_butterfly_pair_op``; lossy ``(parent, parent)``
        adapted to the packed-pair form."""
        unweighted = torch.einsum('bmi,mij->bmj', y_pair, W_inv_node)
        D = self._butterfly_D
        a_rec = unweighted[..., :D]
        b_rec = unweighted[..., D:]
        avg = 0.5 * (a_rec + b_rec)
        return torch.cat([avg, avg], dim=-1)

    def forward(self, left, right=None):
        """Forward pass.

        Non-butterfly: binary ``forward(left, right)`` -> monotonic
        union (lattice max).
        Butterfly: unary ``forward(x)`` -> cross-STM monotonic-max
        cascade; ``right`` ignored.

        See class docstring for the operation this layer applies.
        """
        if self.butterfly:
            return self._butterfly_forward(left)
        return Ops.union(left, right, monotonic=True)

    def reverse(self, parent, basis=None, left_rows=None, right_rows=None,
                left_priming=None, right_priming=None):
        """Reverse pass; inverse of ``forward``.

        ``basis`` supplied (a Codebook/Basis with ``getW()``) -> the mereology
        recommender :py:meth:`Ops.disjunctionReverse` recovers an operand pair
        ``(x1, x2)`` with ``union(x1, x2) ~= parent`` from the codebook rows --
        EXACT on a discrete vocabulary (the XOR reconstruction path). The
        OR-fold is many-to-one, so ``basis is None`` falls back to the lossy
        ``(parent, parent)`` pseudo-inverse. Mirrors ``UnionLayer.reverse``.
        """
        if self.butterfly:
            return self._butterfly_reverse(parent)
        if basis is not None:
            W = basis.getW() if hasattr(basis, 'getW') else None
            if W is not None:
                return Ops.disjunctionReverse(
                    parent, parent, W, monotonic=True,
                    left_rows=left_rows, right_rows=right_rows,
                    left_priming=left_priming, right_priming=right_priming)
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        if self.butterfly:
            x = torch.cat([left, right], dim=-2)
            return self._butterfly_forward(x)
        return Ops.union(left, right, monotonic=True)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


def _argmax_prototype(x):
    """Per-batch top-1 prototype index from a ``[B, V, D]`` muxed
    event tensor.

    Computes the L2 norm of the ``.what`` bivector slice
    ``[..., :2]`` only (not the full muxed event) so the ranking
    reflects symbol identity / presence, not the nWhere / nWhen
    positional channels. When the input has last_dim < 2 the full
    last dim is used (degenerate, single-channel fallback).

    Returns a ``[B]`` long tensor where each entry is the position
    (codebook prototype index) with the largest ``.what`` L2 norm
    in that batch row -- the most-active prototype for that
    operand. Used by ``PartLayer`` / ``IsEqualLayer`` / ``QueryLayer``
    to map continuous activations to discrete codebook indices for
    mereological-tree bookkeeping.
    """
    if not torch.is_tensor(x):
        return torch.zeros(0, dtype=torch.long)
    if x.dim() < 2:
        return torch.zeros(1, dtype=torch.long, device=x.device)
    # Slice .what bivector when available; else use whole last-dim.
    what = x[..., :2] if x.shape[-1] >= 2 else x
    norms = what.norm(dim=-1)            # [B, V]
    if norms.dim() < 2:
        norms = norms.unsqueeze(0)
    return norms.argmax(dim=-1)          # [B]


def _parthood_geometric(left, right):
    """Clipped cosine parthood on per-batch dominant bivector activations.

    Replaces the explicit ``MereologicalTree`` lookup for
    ``PartLayer`` / ``IsEqualLayer`` / ``QueryLayer`` after the
    "codebook IS the meronymic tree" unification: parthood is
    expressed by codebook geometry on the bivector cone (see
    ``Architecture.md`` §"Monotonicity of the bivector chain").

    For per-batch dominant slot bivectors ``a = left[b, argmax_left]``
    and ``b = right[b, argmax_right]``, returns the clipped cosine
    similarity

        part(a, b) = max(0, a · b) / (|a| * |b|)

    which is the canonical mereological projection on the
    non-negative paired-index cone. The return is a ``[B]`` tensor
    in ``[0, 1]`` -- 1 means "fully a part of", 0 means disjoint.
    """
    a_idx = _argmax_prototype(left)             # [B]
    b_idx = _argmax_prototype(right)            # [B]
    B = int(left.shape[0])
    a_vec = left[torch.arange(B, device=left.device), a_idx]    # [B, K]
    b_vec = right[torch.arange(B, device=right.device), b_idx]  # [B, K]
    # Restrict to bivector head [pos, neg]; if last_dim < 2 use full.
    K = min(2, a_vec.shape[-1])
    a_biv = a_vec[..., :K]
    b_biv = b_vec[..., :K]
    dot = (a_biv * b_biv).sum(dim=-1)
    na = a_biv.norm(dim=-1)
    nb = b_biv.norm(dim=-1)
    return (dot.clamp(min=0.0) / (na * nb + 1e-9))               # [B]

class IsEqualLayer(GrammarLayer):
    """``S -> isEqual(S, S)`` -- symbolic identity assertion.

    SS-space_role identity: ``isEqual(A, B)`` asserts that A and B name the
    same concept by producing a single parent symbol that represents
    the wholeness of its arguments — a higher-epistemic-level
    assertion that cannot be expressed at the subsymbolic level.
    Compare with the CS-space_role ``equal`` which performs the geometric
    identity check on concept bivectors directly.

    Post-MereologicalTree retirement: equality is expressed purely
    geometrically. The codebook is now the meronymic structure; an
    asserted equality between ``A`` and ``B`` shows up as
    bivector co-location on the cone (mutual parthood
    ``part(A, B) ≈ 1`` AND ``part(B, A) ≈ 1``), which the codebook
    learns through training. No explicit equivalence-class table
    is stored.

    The forward returns ``torch.maximum(left, right)`` -- the
    lattice join under the bivector cone's max-as-disjunction
    interpretation -- so the chart's CKY consumer sees a single
    parent vector (semantics unchanged from the tree-backed
    version; the difference is only the absence of the tree write).

    Lossy with ``(parent, parent)`` pseudo-inverse on reverse.
    """
    rule_name        = "isEqual"
    arity            = 2
    invertible       = False
    lossy            = True
    space_role             = 'SS'
    reads_activation = False

    def __init__(self, tree=None):
        """Initialize IsEqualLayer.

        ``tree`` is accepted but ignored for backward compatibility
        with the chart's lazy-build call sites; the
        ``MereologicalTree`` has been retired.
        """
        super().__init__(0, 0)

    def forward(self, left, right):
        """Lattice join on the bivector cone (max element-wise)."""
        return torch.maximum(left, right)

    def reverse(self, parent):
        """Reverse pass; inverse of ``forward``.

        Lossy ``(parent, parent)`` pseudo-inverse -- the max-fold
        is not bijective.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class IsPartLayer(GrammarLayer):
    """``S -> isPart(S, S)`` -- symbolic parthood assertion.

    The mereological analogue of :class:`IsEqualLayer`: an SS-space_role
    *assertive* relation that states "A is part of B" as a single parent
    symbol, a higher-epistemic-level assertion than the CS-space_role geometric
    ``part`` test. Per decision 6 of the role-collapsed grammar spec,
    ``isPart`` is one relation dispatched by ``query``: assertive here,
    answer-producing (``queryPart``) when ``query="true"`` -- folding in
    the retired ``assertPart`` / ``queryPart`` operator names.

    Forward returns ``right`` -- the encompassing parent -- so the CKY
    consumer sees a single parent vector; the parthood relationship
    between ``left`` and ``right`` is carried by codebook geometry (the
    codebook IS the meronymic tree, see :class:`PartLayer`). Lossy with
    the ``(parent, parent)`` pseudo-inverse on reverse.
    """
    rule_name        = "isPart"
    arity            = 2
    invertible       = False
    lossy            = True
    space_role             = 'SS'
    reads_activation = False

    def __init__(self, tree=None):
        """``tree`` accepted but ignored (MereologicalTree retired)."""
        super().__init__(0, 0)

    def forward(self, left, right):
        """Pass the encompassing parent ``right`` through to the CKY
        consumer (the parthood geometry is learned in the codebook)."""
        return right

    def reverse(self, parent):
        """Lossy ``(parent, parent)`` pseudo-inverse -- ``isPart(A, B)``
        does not preserve A's identity in the parent vector."""
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class PartLayer(GrammarLayer):
    """``S -> part(S, S)`` -- mereological part-of on the bivector
    codebook.

    Post-MereologicalTree retirement: parthood is expressed
    geometrically by codebook position on the non-negative
    paired-index bivector cone (see ``Architecture.md``
    §"Monotonicity of the bivector chain"). The codebook IS the
    meronymic tree: A is part of B iff the clipped cosine
    projection of A's prototype onto B's prototype is high. The
    codebook learns this geometry through training on the rule
    composition; no separate adjacency table is stored.

    The forward returns ``right`` -- the encompassing parent --
    so the chart's CKY consumer sees a single parent vector
    (semantics unchanged from the tree-backed version; the
    difference is only the absence of the tree write).

    Lossy with ``(parent, parent)`` pseudo-inverse on reverse.
    """
    rule_name        = "part"
    arity            = 2
    invertible       = False
    lossy            = True
    space_role             = 'CS'
    reads_activation = False

    def __init__(self, tree=None):
        """Initialize PartLayer.

        ``tree`` is accepted but ignored for backward compatibility
        with the chart's lazy-build call sites; the
        ``MereologicalTree`` has been retired.
        """
        super().__init__(0, 0)

    def forward(self, left, right):
        """Pass the encompassing parent ``right`` through to the
        CKY consumer. The parthood relationship between ``left``
        and ``right`` is captured by the codebook geometry that
        learns under training -- no explicit tree write.
        """
        return right

    def reverse(self, parent):
        """Reverse pass; inverse of ``forward``.

        Lossy ``(parent, parent)`` pseudo-inverse -- ``part(A, B)``
        does not preserve A's identity in the parent vector.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class AssertPartLayer(PartLayer):
    """Assertive parthood relation; grammar-level alias for ``part``."""
    rule_name = "assertPart"


def _truth_bivector_like(score, template):
    """Broadcast a scalar truth score into the bivector shape of template."""
    pos = score.to(device=template.device, dtype=template.dtype)
    neg = torch.zeros_like(pos)
    truth = torch.stack([pos, neg], dim=-1)        # [B, 2]
    if template.dim() < 3:
        return truth
    V = template.shape[1]
    rest_dim = template.shape[-1] - 2
    if rest_dim > 0:
        tail = torch.zeros(truth.shape[0], rest_dim,
                           dtype=template.dtype, device=template.device)
        truth = torch.cat([truth, tail], dim=-1)
    return truth.unsqueeze(1).expand(-1, V, -1)


class QueryLayer(GrammarLayer):
    """``S -> query(S, S)`` -- mereological-truth query "is A part
    of B?" answered geometrically against the bivector codebook.

    Post-MereologicalTree retirement: the answer is the clipped
    cosine parthood between the per-batch dominant bivector
    activations of ``left`` and ``right`` (see
    ``_parthood_geometric``). Returns a continuous truth value
    in ``[0, 1]`` rather than the prior tree-lookup boolean --
    the codebook geometry IS the meronymic structure, so
    parthood is *always* defined for any two symbols (no
    "unknown" state). Returns a ``[B, V, 2]`` truth bivector
    broadcast across the V dimension:

        part(A, B) ≈ 1  -> [pos=1, neg=0] (full affirmation)
        part(A, B) ≈ 0  -> [pos=0, neg=0] (disjoint / no overlap)

    Lossy with ``(parent, parent)`` pseudo-inverse on reverse.
    """
    rule_name        = "query"
    arity            = 2
    invertible       = False
    lossy            = True
    space_role             = 'CS'
    reads_activation = False

    def __init__(self, tree=None):
        """Initialize QueryLayer.

        ``tree`` is accepted but ignored for backward compatibility
        with the chart's lazy-build call sites; the
        ``MereologicalTree`` has been retired in favour of
        codebook geometry.
        """
        super().__init__(0, 0)

    def forward(self, left, right):
        """Geometric parthood query: returns a per-batch ``[B, V, K]``
        truth bivector broadcast across V, with the bivector head
        carrying ``[pos=parthood, neg=0]``.
        """
        parthood = _parthood_geometric(left, right)    # [B] in [0, 1]
        return _truth_bivector_like(parthood, right)

    def reverse(self, parent):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class QueryPartLayer(QueryLayer):
    """Interrogative parthood relation; grammar-level alias for ``query``."""
    rule_name = "queryPart"


class QueryEqualLayer(QueryLayer):
    """Interrogative equality relation answered as mutual parthood."""
    rule_name = "queryEqual"

    def forward(self, left, right):
        equal = (
            _parthood_geometric(left, right)
            * _parthood_geometric(right, left))
        return _truth_bivector_like(equal, right)


def _dispatch_method_name_for_rule(rule):
    """Return the runtime GrammarLayer op for a parsed rule.

    The grammar can keep one relation name while using ``query="true"``
    to request answer-producing semantics.
    """
    method = getattr(rule, 'method_name', None)
    if getattr(rule, 'query', False) and method == 'isEqual':
        return 'queryEqual'
    if getattr(rule, 'query', False) and method == 'isPart':
        return 'queryPart'
    return method


class ExistLayer(GrammarLayer):
    """Existential truth wrapper for absolute-truth start forms."""
    rule_name        = "exist"
    arity            = 1
    invertible       = False
    lossy            = False
    space_role             = 'SS'
    reads_activation = False

    def __init__(self):
        super().__init__(0, 0)

    def forward(self, x):
        return x

    def reverse(self, parent):
        return parent

    def compose(self, x):
        return self.forward(x)

    def generate(self, parent):
        return self.reverse(parent)


# Hardcoded module-level lookup -- replaces the retired
# -- Conceptual introspection (2026-05-12) ----------------------------
#
# Per-stage introspective grammar operations that read mental content
# and produce scalar / vector annotations the network can condition on
# at subsequent conceptual orders. Design source:
# doc/plans/2026-05-04-conceptual-introspection-handoff.md
#
# Each is implemented as a fully differentiable function of the input
# activation (no learned parameters of its own). Plug them in by
# registering an op-class entry in `GRAMMAR_LAYER_CLASSES` and a
# `<rule>area(S)</rule>`-style entry in the model's grammar XML.

def area_op(x, sigma=None):
    """Normalised Gaussian region area: ``min(sigma**2, 1)``.

    Args:
        x: ``[..., D]`` activation (the proposition whose extent we
            measure). Used only for device / dtype routing -- the area
            depends solely on `sigma`.
        sigma: scalar or tensor (mean reduced). When None falls back to
            `_DEFAULT_SUBSYMBOLIC_SIGMA`.

    Returns:
        Scalar tensor on `x.device` / `x.dtype` in [0, 1].
    """
    if sigma is None:
        sigma = _DEFAULT_SUBSYMBOLIC_SIGMA
    if torch.is_tensor(sigma):
        s = sigma.float().mean()
    else:
        s = float(sigma)
        s = (x.new_tensor(s) if torch.is_tensor(x) else torch.tensor(s))
    return torch.clamp(s.pow(2), max=1.0)


def luminosity_op(x_a, x_b, sigma=None):
    """Pairwise luminosity ``area − overlapArea * |t_A − t_B|`` ∈ [-1, 1].

    Both inputs are bivector-tail activations ``[..., D]`` (D >= 2);
    the first two components carry the [pos, neg] poles. Returns a
    scalar consistent with `Mereology.Luminosity`'s pairwise term.
    """
    if sigma is None:
        sigma = _DEFAULT_SUBSYMBOLIC_SIGMA
    sigma_f = float(sigma) if not torch.is_tensor(sigma) else float(
        sigma.float().mean().item())
    a_flat = x_a.reshape(-1, x_a.shape[-1])
    b_flat = x_b.reshape(-1, x_b.shape[-1])
    overlap = _gaussian_kernel_overlap(
        a_flat, b_flat, sigma_f, sigma_f).mean()
    if x_a.shape[-1] >= 2 and x_b.shape[-1] >= 2:
        dot_a = (x_a[..., 0] - x_a[..., 1]).mean()
        dot_b = (x_b[..., 0] - x_b[..., 1]).mean()
    else:
        dot_a = x_a.mean()
        dot_b = x_b.mean()
    disagree = (dot_a - dot_b).abs()
    area = area_op(x_a, sigma_f).to(device=overlap.device, dtype=overlap.dtype)
    lum = area - overlap * disagree
    return torch.clamp(lum, min=-1.0, max=1.0)


def isa_part_op(child, parent, sigma=None):
    """One-step kernel overlap ``K(child, parent) ∈ (0, 1]`` per the
    plan's "is the child contained in the parent at this conceptual
    order" semantics.
    """
    if sigma is None:
        sigma = _DEFAULT_SUBSYMBOLIC_SIGMA
    sigma_f = float(sigma) if not torch.is_tensor(sigma) else float(
        sigma.float().mean().item())
    c_flat = child.reshape(-1, child.shape[-1])
    p_flat = parent.reshape(-1, parent.shape[-1])
    overlap = _gaussian_kernel_overlap(
        c_flat, p_flat, sigma_f, sigma_f)
    return overlap.mean()

# Method-name -> GrammarLayer subclass mapping. Moved from Layers.py
# per the 2026-05-29 grammar refactor (§5); most concrete subclasses
# now live above in this module, the remaining ones (Equal/True/False/
# Swap/Copy/Area/Luminosity/IsaPart) are imported at the top from Layers.
GRAMMAR_LAYER_CLASSES = {
    'not':          NotLayer,
    'non':          NonLayer,
    'intersection': IntersectionLayer,
    'union':        UnionLayer,
    'lift':         LiftLayer,
    'verb':         VerbLayer,
    'adverb':       AdverbLayer,
    'lower':        LowerLayer,
    'preposition':  PrepositionLayer,
    'bind':         ContextualBindLayer,
    'tense':        TenseLayer,
    'aspect':       AspectLayer,
    'morphology':   MorphologyLayer,
    'conjunction':  ConjunctionLayer,
    'disjunction':  DisjunctionLayer,
    'isEqual':      IsEqualLayer,
    'isPart':       IsPartLayer,
    'equal':        EqualLayer,
    'part':         PartLayer,
    'assertPart':   AssertPartLayer,
    'true':         TrueLayer,
    'false':        FalseLayer,
    'swap':         SwapLayer,
    'copy':         CopyLayer,
    'query':        QueryLayer,
    'queryEqual':   QueryEqualLayer,
    'queryPart':    QueryPartLayer,
    'exist':        ExistLayer,
    'area':         AreaLayer,
    'luminosity':   LuminosityLayer,
    'isaPart':      IsaPartLayer,
}


# ----------------------------------------------------------------------
# Corpus-scale connective supervision (Phase R5)
# ----------------------------------------------------------------------
# doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-
# grammar.md decision 8 + §5.5 + §8 R5. ``A AND B`` and ``A OR B`` are
# surface-indiscriminable, so the role-collapsed grammar does NOT give
# them distinct categories; they are discriminated by the slot-0
# OPERATOR SUPERPOSITION over {conjunction, disjunction}. The three
# helpers below supply the truth / consequence signal that makes that
# superposition load-bearing: a differentiable operator-superposition
# (the gradient analogue of
# ``perceptual_analyzer.soft_operator_compose``) plus an MSE loss
# against the observed consequence ``y``, so supervising on a corpus
# of (operands, consequence) pairs drives the slot-0 distribution to
# the connective whose truth table matches -- even when the operands
# (the surface) are identical between the AND and OR corpora.
def soft_connective_compose(dist, a, b, op_names, classes=None):
    """Tensor-weighted operator superposition over ``op_names`` -- the
    differentiable analogue of ``perceptual_analyzer.soft_operator_compose``.

    ``dist`` is a 1-D weight tensor aligned with ``op_names`` (gradient
    flows to it, unlike the float-coerced ``soft_operator_compose``). A
    one-hot ``dist`` reduces to that operator's hard compose, preserving
    the typed grammar as the limit.
    """
    if classes is None:
        classes = GRAMMAR_LAYER_CLASSES
    out = None
    for k, name in enumerate(op_names):
        y = classes[name]().compose(a, b)
        contrib = dist[k] * y
        out = contrib if out is None else out + contrib
    return out


def connective_truth_loss(logits, a, b, y, op_names, classes=None):
    """MSE between the slot-0 operator superposition's composed
    prediction and the observed consequence ``y``. ``logits`` are the
    (learnable) slot-0 operator logits; ``softmax(logits)`` is the
    superposition.
    """
    dist = torch.softmax(logits, dim=-1)
    pred = soft_connective_compose(dist, a, b, op_names, classes=classes)
    return ((pred - y) ** 2).mean()


def learn_connective_distribution(a, b, y,
                                  op_names=("conjunction", "disjunction"),
                                  steps=500, lr=0.1, seed=0, classes=None):
    """Fit the slot-0 operator superposition to a truth/consequence corpus.

    Minimizes :func:`connective_truth_loss` over ``steps`` Adam updates on
    the operator logits, then returns the learned distribution as
    ``{op_name: weight}``. The operands ``a`` / ``b`` are the surface (the
    same for AND and OR); ``y`` is the consequence that discriminates them.
    """
    torch.manual_seed(int(seed))
    logits = torch.zeros(len(op_names), requires_grad=True)
    opt = Adam([logits], lr=lr)
    for _ in range(int(steps)):
        opt.zero_grad()
        loss = connective_truth_loss(logits, a, b, y, op_names, classes=classes)
        loss.backward()
        opt.step()
    dist = torch.softmax(logits.detach(), dim=-1)
    return {name: float(dist[k]) for k, name in enumerate(op_names)}


# Per-operator SurfaceSchema assignment (doc/plans/2026-05-30-subsymbolic
# -analyzer-terminal-emitter.md, "Per-operator schema" table). Operators
# not listed keep the GrammarLayer default (T4 BINARY_JUXTAPOSE). The
# template is a class attribute so it is shared by every instance of an
# operator -- conjunction / disjunction / isEqual all reference the one
# T2 singleton (they are surface-indiscriminable, discriminated by the
# slot-0 operator vector). Assigning here (rather than per-class) keeps
# the table in one place next to the layer registry.
_OPERATOR_SURFACE_SCHEMAS = {
    # Unary affixes (T1): the operator owns a learned marker.
    'not':          T1_UNARY_AFFIX,
    'non':          T1_UNARY_AFFIX,
    'query':        T1_UNARY_AFFIX,
    'queryEqual':   T1_UNARY_AFFIX,
    'exist':        T1_UNARY_AFFIX,
    # Binary infix (T2): one INFIX/CIRCUM marker slot that may select
    # which op fires; order free.
    'conjunction':  T2_BINARY_INFIX,
    'disjunction':  T2_BINARY_INFIX,
    'isEqual':      T2_BINARY_INFIX,
    'equal':        T2_BINARY_INFIX,
    'union':        T2_BINARY_INFIX,
    'intersection': T2_BINARY_INFIX,
    # Binary directional (T3): (position, marker) co-varies with a
    # recorded order bit -- the part / possessive family. ``isPart`` is
    # the role-collapsed relation name (query-dispatched) that supersedes
    # ``assertPart`` / ``queryPart``.
    'isPart':       T3_BINARY_DIRECTIONAL,
    'part':         T3_BINARY_DIRECTIONAL,
    'queryPart':    T3_BINARY_DIRECTIONAL,
    'assertPart':   T3_BINARY_DIRECTIONAL,
    # lift / lower / verb / adverb: T3/T4 -- modifier marker or bare; the
    # bare (T4) default is kept so they round-trip without a marker.
    'lift':         T4_BINARY_JUXTAPOSE,
    'verb':         T4_BINARY_JUXTAPOSE,
    'adverb':       T4_BINARY_JUXTAPOSE,
    'lower':        T4_BINARY_JUXTAPOSE,
    # PREPOSITION (T3): a learned PRE marker (that / to / in / because /
    # when) heads the phrase; the marker does NOT select the op (it is
    # recorded, not folded into content). Directional: the marker side and
    # the content side are distinguishable by recorded order.
    'preposition':  T3_BINARY_DIRECTIONAL,
    # BIND (T5): contextual missing-NP marker. The surface BIND token is
    # elided -- it carries no content of its own; at parse time the left
    # slot is resolved to an accessible participant (nearest-left / control
    # licensing), so the realized phrase has no overt marker (spec
    # "Operation 2").
    'bind':         T5_BINARY_ELISION,
    # tense / aspect (T1): unary .when ops; each owns a learned affix
    # marker (the inflectional ending / auxiliary) realized over the head.
    'tense':        T1_UNARY_AFFIX,
    'aspect':       T1_UNARY_AFFIX,
    'morphology':   T1_UNARY_AFFIX,
    # Surface elision policies (T5): copy keeps the survivor (order id),
    # swap keeps the survivor with order swapped. Retired from the
    # symbolic grammar; kept as the absorb/emit elision primitives.
    'copy':         T5_BINARY_ELISION,
    'swap':         T5_BINARY_ELISION,
}
for _op_name, _op_schema in _OPERATOR_SURFACE_SCHEMAS.items():
    _op_cls = GRAMMAR_LAYER_CLASSES.get(_op_name)
    if _op_cls is not None:
        _op_cls.surface_schema = _op_schema


def _bind_moved_ops_singletons():
    """Bind ``Ops.<grammar_op>`` to ``_OpHandle`` instances for the grammar
    rule operator classes that moved here from Layers.py per the
    2026-05-29 grammar-file-refactor (\xa75).

    Mirrors ``Layers._bind_ops_singletons`` for the moved subset
    (negation / non / conjunction / disjunction / lift / lower / part).
    Called once at module load. The ``equal`` binding remains in
    Layers.py because EqualLayer stays there.
    """
    from Layers import _OpHandle
    bindings = (
        ('negation',    Ops._negation_kernel,    NotLayer),
        ('non',         Ops._non_kernel,         NonLayer),
        ('conjunction', Ops._conjunction_kernel, ConjunctionLayer),
        ('disjunction', Ops._disjunction_kernel, DisjunctionLayer),
        ('lift',        Ops._lift_kernel,        LiftLayer),
        ('lower',       Ops._lower_kernel,       LowerLayer),
        ('part',        Ops._part_kernel,        PartLayer),
    )
    for name, kernel, cls in bindings:
        try:
            inst = cls()
        except TypeError:
            continue
        setattr(Ops, name, _OpHandle(kernel, inst))


_bind_moved_ops_singletons()


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
# Distinct from the WholeSpace symbol codebook (`WholeSpace
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
# consolidation). Stage 3 (2026-05-27): promoted to the canonical
# parser; constructed directly on ``SymbolSubSpace.languageLayer``. The
# CKY chart and its ``_ensure_signal_router`` lazy bridge retired
# alongside the chart class.
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


def sentence_relative_mask(word_subspace, B, device=None):
    """Per-row ``[B]`` bool: True where the current sentence is a RELATIVE
    truth (the ``part`` / ``isEqual`` predicate family), read from
    ``word_subspace.current_rules`` + ``TheGrammar``.

    Host-side (a cheap dict lookup); conservative -- returns all-False on
    ANY uncertainty. Shared by ``BasicModel._sentence_relative_mask`` (the
    reduce site) and the hoisted ``ConceptualSpace.Reset`` relation-learning
    hook (both need the same signal off the compiled forward).
    """
    false_mask = torch.zeros(B, dtype=torch.bool, device=device)
    if word_subspace is None:
        return false_mask
    current_rules = getattr(word_subspace, 'current_rules', None)
    if not current_rules:
        return false_mask
    if not TheGrammar._relative_rule_id_set():
        return false_mask

    def _row_is_relative(rule_ids):
        for rid in (rule_ids or ()):
            if TheGrammar.is_relative_rule(rid):
                return True
        return False

    s_rules = None
    for key in ('SS',):
        if key in current_rules:
            s_rules = current_rules[key]
            break
    if not s_rules:
        return false_mask
    try:
        n_outer = len(s_rules)
    except TypeError:
        return false_mask
    if n_outer > 0 and not isinstance(s_rules[0], (list, tuple)):
        shared = _row_is_relative(s_rules)
        return torch.full((B,), bool(shared), dtype=torch.bool, device=device)
    if n_outer == B:
        flags = [_row_is_relative(row) for row in s_rules]
        return torch.tensor(flags, dtype=torch.bool, device=device)
    if n_outer == 1:
        shared = _row_is_relative(s_rules[0])
        return torch.full((B,), bool(shared), dtype=torch.bool, device=device)
    return false_mask


class LanguageLayer(Layer):
    """Top-level signal-routing parser. The canonical parser as of
    Stage 3 (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md,
    2026-05-27). Constructed directly on
    ``SymbolSubSpace.languageLayer``; the CKY chart and its lazy
    ``_ensure_signal_router`` bridge retired with the Chart class.

    Multi-space_role: a unary layer and/or a binary layer can be attached per
    space_role (e.g., 'subsymbolic', 'CS', 'SS'). On compose, space_roles run in sorted order;
    within each space_role, unary fires first then binary, with the soft slab
    of the previous step feeding the next so gradient reaches every op.

    **Layer contract** (post-2026-05-20 stack-rewrite refactor):

    The canonical entry points are ``forward(subspace, syntactic_layer,
    ..., actions=...)`` and ``reverse(subspace, syntactic_layer, ...)``
    -- both wrap the stack-rewrite primitives (shift/reduce/unreduce)
    so call sites can treat LanguageLayer like any other Layer subclass.

    ``compose`` / ``generate`` are the SymbolSubSpace-facing entry points;
    they operate on a ``[B, N, D]`` slab through the attached
    ``_unary_layers`` / ``_binary_layers`` ModuleDicts and produce
    per-row rule lists. The two paths are independent: the Layer-style
    ``forward`` ignores the ModuleDicts (it dispatches through the
    supplied SyntacticLayer's ``execute``), and ``compose`` ignores the
    Layer-style
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
        # Conceptual reduction order (T = subsymbolicOrder). Used as the
        # recursive-reduction round floor in ``compose`` (plan \xa76: the
        # per-space_role compose loop is collapsed into a single reduction space_role,
        # so the bound is ``max(subsymbolic_order, N-1)`` rather than the
        # number of declared space_roles). Defaults to 1; the host space sets it
        # when it can reach the model's subsymbolicOrder. ``max(1, N-1) ==
        # N-1`` so the default reproduces the pre-collapse round count.
        self.subsymbolic_order = 1
        # Placement-chooser kind for the structured layers built by
        # ``attach_unary_ops`` / ``attach_layer_ops``. "anchordot" (default)
        # = the stateless behavior-preserving scorer (no params); "mlp" =
        # the contextual MLPTransformChooser (owns params, new basin). The
        # host sets it from ``<architecture><transformChooser>`` BEFORE the
        # ops are attached (the layers pick the chooser at construction).
        self.transform_chooser = "anchordot"
        self._unary_layers = nn.ModuleDict()
        self._binary_layers = nn.ModuleDict()
        # Parallel arrays of global rule_ids per attached layer; keyed by
        # space_role. Local op_id (the index inside the layer's ModuleList) maps
        # to the corresponding global rule_id at this list position.
        self._unary_rule_ids = {}
        self._binary_rule_ids = {}
        # Per-compose cache for generate / inspection.
        self._last_input = None
        self._last_output = None
        self._last_space_role_routings = {}

    def _chooser_role_cats(self):
        """Category-context width for structured-layer MLP choosers.

        The role count comes from the same :func:`compute_role_vocabulary` the
        WholeSpace category codebook uses, so the chooser's context block and
        the codebook's role vectors share a width. Anchor-dot also uses the
        category vector, but as a layer-level score prior, so it needs no MLP
        input widening.
        """
        if str(getattr(self, "transform_chooser", "anchordot")) != "mlp":
            return 0
        if not bool(TheXMLConfig.get("architecture.categoryCodebook",
                                     default=True)):
            return 0
        try:
            return int(compute_role_vocabulary(TheGrammar)[2])
        except Exception:
            return 0

    def attach_unary_ops(self, *, ops, rule_ids=None, op_names=None,
                         op_space_roles=None, r_copy=1, space_role="SS"):
        """Attach a unary space_role; ops fire per-position with one selection.

        ``rule_ids`` parallels ``ops`` and maps local op_id to the
        grammar's global rule_id (defaults to identity range).
        ``op_names`` and ``op_space_roles`` parallel ``ops`` and carry the
        per-rule method-name and space_role tag ('CS' / 'SS' / ...) declared
        in the .grammar file. Both are optional; they enable downstream
        consumers (space_role-gated scoring, diagnostics) to look at each op
        without crawling back through ``TheGrammar``. Mutates
        ``self._unary_layers[space_role]`` and ``self._unary_rule_ids[space_role]``.
        """
        space_role = str(space_role)
        layer = UnaryStructuredLayer(
            d_model=self.feature_dim,
            ops=ops, r_copy=r_copy,
            temperature=self.temperature,
            chooser=getattr(self, "transform_chooser", "anchordot"),
            n_role_cats=self._chooser_role_cats(),
            op_names=op_names,
        )
        layer.op_names = list(op_names) if op_names is not None else None
        layer.op_space_roles = list(op_space_roles) if op_space_roles is not None else None
        self._unary_layers[space_role] = layer
        if rule_ids is None:
            rule_ids = list(range(len(ops)))
        else:
            rule_ids = [int(r) for r in rule_ids]
        if len(rule_ids) != len(ops):
            raise ValueError(
                f"attach_unary_ops: len(rule_ids)={len(rule_ids)} != "
                f"len(ops)={len(ops)} for space_role {space_role!r}")
        self._unary_rule_ids[space_role] = rule_ids

    def attach_layer_ops(self, *, ops, rule_ids=None, op_names=None,
                         op_space_roles=None, r_copy=1, space_role="SS"):
        """Attach a binary space_role; ops reduce adjacent pairs via Viterbi DP.

        ``rule_ids`` parallels ``ops`` and maps local op_id to grammar
        global rule_id. ``op_names`` and ``op_space_roles`` carry the per-rule
        method-name and space_role tag declared in the .grammar file; when
        supplied, the binary layer uses them to (a) gate scores so a
        CS-space_role op only fires at CS-space_role positions and an SS-space_role op only
        at SS-space_role positions, and (b) update each position's space_role after a
        ``lift`` (CS->SS) or ``lower`` (SS->CS) reduce. Both are optional;
        when omitted the layer falls back to ungated behaviour
        (backward-compat). Mutates ``self._binary_layers[space_role]`` and
        ``self._binary_rule_ids[space_role]``.
        """
        space_role = str(space_role)
        layer = BinaryStructuredReductionLayer(
            d_model=self.feature_dim,
            ops=ops, op_space_roles=op_space_roles, op_names=op_names,
            r_copy=r_copy,
            temperature=self.temperature,
            chooser=getattr(self, "transform_chooser", "anchordot"),
            n_role_cats=self._chooser_role_cats(),
        )
        self._binary_layers[space_role] = layer
        if rule_ids is None:
            rule_ids = list(range(len(ops)))
        else:
            rule_ids = [int(r) for r in rule_ids]
        if len(rule_ids) != len(ops):
            raise ValueError(
                f"attach_layer_ops: len(rule_ids)={len(rule_ids)} != "
                f"len(ops)={len(ops)} for space_role {space_role!r}")
        self._binary_rule_ids[space_role] = rule_ids

    def compose(self, data, word_space, subspace=None):
        """Run space_roleed unary then recursive binary reductions; return rule list.

        ``data`` is ``[B, N, D]``. For each space_role in sorted order, unary
        fires per position then binary reduces adjacent pairs until N
        collapses to a single S-state. Returns ``{space_role: list[list[rule_id]]}``
        and caches the root state + length-N expansion on ``self``.
        """
        if not self._unary_layers and not self._binary_layers:
            raise RuntimeError(
                "LanguageLayer.compose called before attach_layer_ops() / "
                "attach_unary_ops().")
        x = data
        rules = {}
        self._last_space_role_routings = {}
        all_space_roles = sorted(set(self._unary_layers.keys())
                           | set(self._binary_layers.keys()))

        # Category conditioning: build the per-slot category context ONCE from
        # terminal-slot identities and thread it into the FIRST space_role's
        # scoring only. Later rounds/space_roles fold composed slots whose
        # MetaSymbol identity no longer maps 1:1 to a percept, so their
        # category is neutral.
        cat_e = None
        _ss = getattr(word_space, 'wholeSpace', None)
        if (_ss is not None
                and getattr(_ss, 'category_codebook_enabled', None) is not None
                and _ss.category_codebook_enabled()):
            cat_e = self._build_category_context(x, _ss)
        terminal_space_role = all_space_roles[0] if all_space_roles else None

        for space_role in all_space_roles:
            B = x.shape[0]
            space_role_routing = {}
            space_role_rules_per_row = [[] for _ in range(B)]

            unary_layer = self._unary_layers[space_role] if space_role in self._unary_layers else None
            if unary_layer is not None:
                u_hard, u_soft, u_routing = unary_layer(
                    x, cat_ctx=(cat_e if space_role == terminal_space_role else None))
                space_role_routing["unary"] = u_routing
                rid_table = self._unary_rule_ids[space_role]
                kind = u_routing["action_kind"]
                op = u_routing["action_op"]
                for b in range(B):
                    for j in range(kind.shape[1]):
                        if int(kind[b, j].item()) == 2:
                            space_role_rules_per_row[b].append(
                                rid_table[int(op[b, j].item())])
                # Propagate soft slab so gradient reaches unary ops at
                # later space_roles / through the binary stage of this space_role.
                x = u_soft

            binary_layer = self._binary_layers[space_role] if space_role in self._binary_layers else None
            if binary_layer is not None:
                # Recursive reduction: iterate the binary layer up to
                # (N-1) times so the slab folds down to a single S start
                # state. Each round's marginal_slab feeds the next; the
                # leading position (index 0) accumulates the fully-folded
                # state. The shape stays [B, N, D] in soft form (right-
                # tail positions get pad-weighted as reductions fire);
                # the canonical [B, 1, D] root state is x[:, 0:1, :]
                # after the final round.
                #
                rid_table = self._binary_rule_ids[space_role]
                # plan \xa76: with CS and SS collapsed into one reduction space_role,
                # fold for ``max(subsymbolicOrder, N-1)`` rounds. Extra rounds
                # on an already-folded slab are safe no-ops (the layer
                # returns the degenerate path for N<=1).
                max_rounds = max(self.subsymbolic_order, x.shape[1] - 1)
                round_routings = []
                for _round_i in range(max_rounds):
                    b_hard, b_soft, b_routing = binary_layer(
                        x, cat_ctx=(cat_e if (space_role == terminal_space_role
                                              and _round_i == 0) else None))
                    round_routings.append(b_routing)
                    kind = b_routing["action_kind"]
                    op = b_routing["action_op"]
                    lengths = b_routing["lengths"]
                    B_now = kind.shape[0]
                    for b in range(B_now):
                        L = int(lengths[b].item())
                        for j in range(L):
                            if int(kind[b, j].item()) == 1:
                                space_role_rules_per_row[b].append(
                                    rid_table[int(op[b, j].item())])
                    x = b_soft
                if round_routings:
                    # Last round's routing is the canonical "binary"
                    # diagnostic; the full sequence is in "binary_rounds".
                    space_role_routing["binary"] = round_routings[-1]
                    space_role_routing["binary_rounds"] = round_routings

            self._last_space_role_routings[space_role] = space_role_routing
            rules[space_role] = space_role_rules_per_row

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

        # MetaSymbol category role observation (Phase 1; doc/Language.md
        # "Participation Categories as the Chooser's Syntactic-Category
        # Context"). Gated: only when the terminal WholeSpace has the category
        # codebook enabled. Captures the FIRST binary space_role's round-0 reduces
        # (the only round whose slab positions map 1:1 to the original
        # percepts) as per-row (left_pos, right_pos, method) tuples, stashed on
        # the WS for the autobind hook (which holds pid_2d) to attribute to
        # MetaSymbols. Off -> not computed (byte-identical).
        ws = getattr(word_space, 'wholeSpace', None)
        if (ws is not None
                and getattr(ws, 'category_codebook_enabled', None) is not None
                and ws.category_codebook_enabled()):
            ws._category_role_obs = self._collect_round0_role_obs()

        return rules

    def _build_category_context(self, x, ws):
        """Per-slot category role vector ``[B, N, n_roles]`` for grammar
        routing, or ``None`` when unavailable.

        Pure reads only (no E-step at score time): each terminal slot position
        -> percept id (stashed by the autobind hook earlier this step) -> PS
        position -> MetaSymbol -> assigned centroid -> role vector. Slots with
        no percept, no binding, or no centroid yet map to ``-1`` and become a
        zero (neutral) row via :meth:`category_role_of`. The position index
        aligns with the round-0 slab (== original percept positions; the same
        correspondence Phase 1's role observation relies on)."""
        last_pid = getattr(ws, '_category_last_pid', None)
        n_roles = int(getattr(ws, '_category_n_roles', 0) or 0)
        row_to_pos = getattr(ws, '_ps_row_to_pos', None)
        if not last_pid or n_roles == 0 or row_to_pos is None:
            return None
        B, N = int(x.shape[0]), int(x.shape[1])
        ctx = x.new_zeros(B, N, n_roles)
        for b in range(min(B, len(last_pid))):
            prow = last_pid[b]
            for n in range(min(N, len(prow))):
                pid = int(prow[n])
                if pid < 0:
                    continue
                ps_pos = row_to_pos.get(pid)
                if ps_pos is None:
                    continue
                meta_pos = ws.taxonomy_parent(ps_pos)
                if meta_pos is None:
                    continue
                role = ws.category_role_for_meta(
                    int(meta_pos), device=x.device, dtype=x.dtype)
                if role is not None:
                    ctx[b, n, :] = role.reshape(-1)[:n_roles]
        return ctx

    def _collect_round0_role_obs(self):
        """Round-0 reduces of the first binary space_role as per-row
        ``(left_pos, right_pos, method)`` tuples for Phase-1 category learning.

        Only round 0 has slab positions == original percept positions, so the
        operand positions index ``pid_2d`` directly in the autobind hook.
        Returns a list of B lists (empty when no binary space_role fired). Host-side
        bookkeeping; only runs when the category codebook is enabled."""
        for space_role in sorted(self._binary_layers.keys()):
            tr = (self._last_space_role_routings or {}).get(space_role) or {}
            rounds = tr.get("binary_rounds")
            if not rounds:
                continue
            r0 = rounds[0]
            kind = r0.get("action_kind")
            op = r0.get("action_op")
            sl = r0.get("src_left")
            sr = r0.get("src_right")
            if kind is None or op is None or sl is None or sr is None:
                continue
            rid_table = (getattr(self, "_binary_rule_ids", {}) or {}).get(space_role) or []
            kind_h = kind.tolist()
            op_h = op.tolist()
            sl_h = sl.tolist()
            sr_h = sr.tolist()
            obs = [[] for _ in range(len(kind_h))]
            for b in range(len(kind_h)):
                row_kind = kind_h[b]
                for j in range(len(row_kind)):
                    if int(row_kind[j]) != 1:          # 1 == fired reduce
                        continue
                    o = int(op_h[b][j])
                    if o < 0 or o >= len(rid_table):
                        continue
                    try:
                        method = TheGrammar.rules[rid_table[o]].method_name
                    except (IndexError, AttributeError, TypeError):
                        method = None
                    if not method:
                        continue
                    obs[b].append(
                        (int(sl_h[b][j]), int(sr_h[b][j]), str(method)))
            return obs          # first binary space_role only (positions == percepts)
        return []

    def generate(self, target, word_space, subspace=None):
        """Reverse-pass mirror: emit the compose-order rule list reversed.

        If compose has not yet been called for ``target``, run it now.
        Space-role order is reversed (innermost first) and each row's rule
        sequence is reversed so the inverse pass pops last-applied first.
        """
        if not self._unary_layers and not self._binary_layers:
            raise RuntimeError(
                "LanguageLayer.generate called before attach_layer_ops() / "
                "attach_unary_ops().")
        if not self._last_space_role_routings:
            self.compose(target, word_space, subspace=subspace)
        # Generate emits the compose-order list reversed per row, so that
        # the inverse pass pops the last-applied rule first. Space-role order is
        # also reversed (innermost first).
        compose_rules = self._compose_rules_from_routings()
        all_space_roles = sorted(compose_rules.keys(), reverse=True)
        return {space_role: [row[::-1] for row in compose_rules[space_role]]
                for space_role in all_space_roles}

    def _compose_rules_from_routings(self):
        """Rebuild per-row compose-order rule lists from cached routings.

        Walks ``self._last_space_role_routings`` and translates each routing's
        ``(action_kind, action_op)`` tensors back into global rule_ids
        via the per-space_role ``_unary_rule_ids`` / ``_binary_rule_ids`` tables.
        """
        rules = {}
        for space_role, space_role_routing in self._last_space_role_routings.items():
            space_role_rules_per_row = None
            if "unary" in space_role_routing:
                rid_table = self._unary_rule_ids[space_role]
                r = space_role_routing["unary"]
                kind = r["action_kind"]
                op = r["action_op"]
                B = kind.shape[0]
                space_role_rules_per_row = [[] for _ in range(B)]
                for b in range(B):
                    for j in range(kind.shape[1]):
                        if int(kind[b, j].item()) == 2:
                            space_role_rules_per_row[b].append(
                                rid_table[int(op[b, j].item())])
            if "binary" in space_role_routing:
                rid_table = self._binary_rule_ids[space_role]
                r = space_role_routing["binary"]
                kind = r["action_kind"]
                op = r["action_op"]
                lengths = r["lengths"]
                B = kind.shape[0]
                if space_role_rules_per_row is None:
                    space_role_rules_per_row = [[] for _ in range(B)]
                for b in range(B):
                    L = int(lengths[b].item())
                    for j in range(L):
                        if int(kind[b, j].item()) == 1:
                            space_role_rules_per_row[b].append(
                                rid_table[int(op[b, j].item())])
            if space_role_rules_per_row is not None:
                rules[space_role] = space_role_rules_per_row
        return rules

    # -- backwards-compat shims for diagnostics / older tests -----------
    @property
    def _last_routing(self):
        # Returns the binary routing of the highest-space_role (last-run) space_role
        # that has one, mirroring the pre-multi-space_role API.
        for space_role in sorted(self._last_space_role_routings.keys(), reverse=True):
            tr = self._last_space_role_routings[space_role]
            if "binary" in tr:
                return tr["binary"]
        return None

    @property
    def _last_unary_routing(self):
        for space_role in sorted(self._last_space_role_routings.keys(), reverse=True):
            tr = self._last_space_role_routings[space_role]
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
            syntactic_layer: per-space_role SyntacticLayer with ``execute``
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

        # CS-space_role ops operate on the muxed event [what | where | when] so
        # LIFT/LOWER can alter the .when span and PREPOSITION can modify the
        # .where (spec Section 5 / 6.4). The WS-space_role stack route stays
        # content-only (WS carries no where/when). Space-role comes off the
        # per-space_role syntactic layer.
        # Only event-aware ops (LIFT / LOWER / PREPOSITION) receive the muxed
        # event; content-only CS-space_role ops (intersection / union / ...) keep the
        # .what operand so their content-sized folds are unaffected.
        is_c_space_role = str(getattr(syntactic_layer, 'space_role', '')) == 'CS'
        _op = None
        if is_c_space_role:
            _mname = TheGrammar.method_name(int(rule_id))
            _op = syntactic_layer._by_name.get(_mname) if _mname else None
        _event_op = bool(getattr(_op, 'event_aware', False))
        when = subspace.materialize(mode="when") if _event_op else None
        use_event = (_event_op and when is not None
                     and when.ndim == 3 and when.shape[-1] > 0)
        if use_event:
            def _ev(slot):
                return torch.cat([what[arange_B, slot, :],
                                  where[arange_B, slot, :],
                                  when[arange_B, slot, :]], dim=-1)
            parent = syntactic_layer.execute(int(rule_id), _ev(i_slot), _ev(j_slot))
            p_what, p_where, p_when = _split_event(
                parent, what.shape[-1], when_width=when.shape[-1])
        else:
            left  = what[arange_B, i_slot, :]                  # [B, D]
            right = what[arange_B, j_slot, :]                  # [B, D]
            parent = syntactic_layer.execute(int(rule_id), left, right)  # [B, D]
            p_what, p_where, p_when = parent, None, None

        what_new = what.clone()
        what_new[arange_B, i_slot, :] = p_what
        what_new[arange_B, j_slot, :] = 0.0

        where_new = where.clone()
        if p_where is not None:
            # Op-modified .where (e.g. PREPOSITION) wins over the rule stamp.
            where_new[arange_B, i_slot, :] = p_where
        else:
            where_vec = self._encode_where(where, where_id)    # [W]
            where_new[arange_B, i_slot, :] = where_vec
        where_new[arange_B, j_slot, :] = 0.0

        occ = subspace.materialize(mode="activation")
        occ_new = occ.clone()
        occ_new[arange_B, i_slot] = 1.0
        occ_new[arange_B, j_slot] = 0.0

        subspace.set_what(what_new)
        subspace.set_where(where_new)
        if p_when is not None and when is not None:
            when_new = when.clone()
            when_new[arange_B, i_slot, :] = p_when             # op-altered .when
            when_new[arange_B, j_slot, :] = 0.0
            subspace.set_when(when_new)
        subspace.set_activation(occ_new)
        # Symbolic-priming forward-commit at the CS-space_role reduction
        # (plan doc/plans/2026-06-06-symbolic-heat-retrieval.md §Grammar
        # reduction) is DEFERRED here: the stack-mode ``subspace`` carries
        # only content (.what/.where/.when) at slots i_slot/j_slot — it has
        # no ref-id channel, and ``forward_stack``'s ('reduce', rule_id)
        # action supplies no operand ref_ids. The reduced children are not
        # snapped to codebook ref ids at this site, so there is nothing to
        # prime without inventing a content->ref nearest-row lookup (which
        # would add a host sync + dense scan on the training path). The
        # plan explicitly permits deferring row priming for unsnapped
        # parents (update only the semantic carrier z from the idea vector).
        # When a caller does have committed child ref_ids at a reduction,
        # ``SymbolSubSpace._commit_priming(b, ref_id)`` is the gated API to
        # prime them (see test_symbolic_heat_retrieval.py
        # ::TestReduceCommitPrimesChildRefs).
        return subspace

    @staticmethod
    def _recover_selected_row(vec, W, cand_rows=None, *, tol=1e-4):
        """Best-effort recover the W-row id the recommender selected.

        The mereology recommender (``Ops._binary_op_recommend``) returns
        selected operand VECTORS drawn verbatim from the augmented codebook
        ``[⊥, W..., ⊤]``; this maps such a vector back to its row id in ``W``.
        Used by ``unreduce``'s reverse self-priming (Phase 3b CAPSTONE; plan
        doc/plans/2026-06-06-symbolic-heat-retrieval.md §Reverse-path
        responsibilities). Search is restricted to ``cand_rows`` (the small
        per-slot candidate subset) when supplied, else the full ``W``.

        Returns the integer row id of the nearest W row within ``tol`` (L2),
        or ``None`` when ``vec`` matched no candidate within tolerance — which
        includes the ⊥ / ⊤ sentinel picks (zeros / ones), since those are not
        real codebook rows and (by construction) sit far from learned rows.
        Pure host-side; only invoked under the priming_enabled gate.
        """
        try:
            if W is None or not hasattr(W, 'shape') or W.shape[0] == 0:
                return None
            v = vec.reshape(-1)[:W.shape[1]].to(W.device, W.dtype)
            if cand_rows is not None:
                rows = torch.as_tensor(
                    cand_rows, dtype=torch.long, device=W.device).reshape(-1)
                rows = rows[(rows >= 0) & (rows < W.shape[0])]
                if rows.numel() == 0:
                    rows = torch.arange(W.shape[0], device=W.device)
            else:
                rows = torch.arange(W.shape[0], device=W.device)
            sub = W[rows]                                  # [|rows|, D]
            d = torch.linalg.vector_norm(
                sub - v.unsqueeze(0), dim=-1)              # [|rows|]
            j = int(torch.argmin(d).item())
            if float(d[j].item()) <= tol:
                return int(rows[j].item())
            return None
        except Exception:
            return None

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
            syntactic_layer: per-space_role SyntacticLayer; provides the
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
            # Post-2026-05-29 grammar-file-refactor (\xa75): rule's class
            # space_role may differ from this syntactic_layer's space_role (e.g.
            # intersection lives in IntersectionLayer with space_role='CS', so it
            # binds on ConceptualSpace's syntactic layer rather than the
            # WholeSpace one). Fall back to a fresh GRAMMAR_LAYER_CLASSES
            # instance so the dispatch path still completes; the identity-
            # stub reverse logic below will run on it.
            cls = GRAMMAR_LAYER_CLASSES.get(method_name)
            if cls is not None:
                try:
                    layer = cls()
                except TypeError:
                    layer = None
        if layer is None:
            raise KeyError(
                f"LanguageLayer.unreduce: space_role={syntactic_layer.space_role!r} "
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

        # 2026-05-29: pass the space_role-local Basis (codebook) as an
        # explicit arg so binary reverses (UnionLayer /
        # IntersectionLayer) can use the mereology-guided recommender
        # (``Ops.disjunctionReverse`` / ``Ops.conjunctionReverse``)
        # to recover an actual operand pair instead of returning the
        # lossy ``(parent, parent)`` pseudo-inverse. The layer's
        # ``reverse`` accepts ``basis`` as a keyword and falls back to
        # ``(parent, parent)`` when ``basis`` is None or has no W.
        # Passing the Basis (rather than its W tensor) keeps the door
        # open for richer codebook methods on reverse without changing
        # the call site.
        space_role_basis = getattr(subspace, 'what', None)

        # Phase 3b CAPSTONE (plan doc/plans/2026-06-06-symbolic-heat-retrieval
        # .md §Reverse-path responsibilities, §Phase 3): heat-biased candidate
        # restriction for the binary recommender (Intersection / Union only).
        #
        # GATED + DEFAULT-OFF BYTE-IDENTITY: ``reverse_kwargs`` stays EMPTY
        # unless ALL of (a) ``subspace.symbolSpace`` exists, (b) the host
        # layer is Intersection/Union (arity-2 recommender ops), AND (c) the
        # owning space's ``attention_mode != 'off'``. Every current config is
        # attention=off, so this whole block is dormant and the call below is
        # exactly ``layer.reverse(parent, basis=space_role_basis)`` as before — no
        # observable change on the live generation path. The broad ``except``
        # collapses any heat-path failure back to a plain reverse so a bug
        # here can NEVER break generation. Decoding is row-0-canonical
        # (``where[0, ...]`` above), so the query / order are taken at batch
        # row 0 to match. Lift/Lower use the algebraic inverse, not the
        # recommender, so they are intentionally NOT wired here.
        reverse_kwargs = {}
        ss = getattr(subspace, 'symbolSpace', None)
        if (ss is not None and arity == 2
                and isinstance(layer, (IntersectionLayer, UnionLayer))):
            space_role = str(getattr(syntactic_layer, 'space_role', ''))
            space = (getattr(ss, 'wholeSpace', None) if space_role == 'SS'
                     else getattr(ss, 'conceptualSpace', None) if space_role == 'CS'
                     else None)
            mode = (str(getattr(space, 'attention_mode', 'off'))
                    if space is not None else 'off')
            if mode != 'off':
                try:
                    # NOTE: ``grammar.rule(rule_id)`` returns the ``RuleDef``
                    # that ``_rule_order_signature`` consumes (it reads
                    # ``.lhs`` / ``.rhs_symbols``). ``grammar.rule_by_id``
                    # returns the canonical PRODUCTION STRING in this codebase,
                    # which would AttributeError here — the broad ``except``
                    # below would swallow it into a plain reverse, silently
                    # disabling the heat path. Use the RuleDef accessor.
                    rule_def = (grammar.rule(rule_id)
                                if hasattr(grammar, 'rule')
                                else grammar.rules[rule_id])
                    sig = grammar._rule_order_signature(rule_def)
                    cats = getattr(sig, 'rhs_categories', None)
                    if (cats is not None and len(cats) >= 2
                            and cats[0] is not None and cats[1] is not None):
                        # Intersection/Union are order-preserving, so the
                        # operands share the parent's order. Read the parent's
                        # per-slot order from the SymbolSubSpace order buffer at
                        # the row-0 canonical top slot. Guard the second-axis
                        # index: ``_order``'s width (the SymbolSubSpace STM depth)
                        # need not equal the language-layer stack width, so an
                        # out-of-range (or zeroed) slot yields order 0.
                        # A zeroed order can make ``refs_by_category ∩
                        # refs_by_order`` empty, which triggers the plan-
                        # sanctioned untyped content+heat fallback inside
                        # ``retrieval_candidates_for_slot`` (plan §Candidate
                        # generation fallback) -- the category filter is dropped
                        # for that slot, never the heat bias. When the typed
                        # set IS non-empty, hot-type-invalid-row exclusion still
                        # holds. The path never crashes.

                        # Phase 5: read retrieval scalar knobs from config
                        # (plan §Configuration).  All keys are optional; fall
                        # back to the sensible defaults below when absent.
                        # The space section name is "WholeSpace" for space_role
                        # 'SS' and "ConceptualSpace" for space_role 'CS'.
                        #
                        # ROBUSTNESS (critical): a knob read must NEVER disable
                        # the heat path. ``TheXMLConfig`` is a process-wide
                        # singleton whose ``_data`` is mutated by other code /
                        # tests; under some orderings it can be left WITHOUT an
                        # ``architecture`` section (or with a duplicated scalar
                        # key), in which case a bare ``TheXMLConfig.space(...,
                        # default)`` would still raise -- and that raise, caught
                        # by the broad ``except`` below, would silently bypass
                        # the heat path (ON==OFF). ``_cfg_knob`` localizes any
                        # such config-read failure and degrades it to the
                        # supplied default, so a config-state issue yields
                        # DEFAULT KNOBS, not a bypassed capstone. (The space()
                        # lookup is itself hardened against a missing
                        # <architecture> section; this is belt-and-suspenders
                        # for genuinely unexpected config-read errors.)
                        def _cfg_knob(_sec, _key, _default):
                            try:
                                return TheXMLConfig.space(_sec, _key, _default)
                            except Exception:
                                return _default
                        _cfg_sec = ('WholeSpace' if space_role == 'SS'
                                    else 'ConceptualSpace')
                        _r_alpha = float(_cfg_knob(
                            _cfg_sec, 'retrievalAlpha', 1.0))
                        _r_beta = float(_cfg_knob(
                            _cfg_sec, 'retrievalBeta', 0.5))
                        # gamma / delta default to 0.0 for primer mode so
                        # behavior is byte-identical to pre-Phase-5 when
                        # mode=='primer'.  For second-order / low-rank modes
                        # the caller must set non-zero values in the XML;
                        # code defaults remain 0.0 (no carrier contribution).
                        _r_gamma = float(_cfg_knob(
                            _cfg_sec, 'retrievalGamma', 0.0))
                        _r_delta = float(_cfg_knob(
                            _cfg_sec, 'retrievalDelta', 0.0))
                        _r_topk_content = int(_cfg_knob(
                            _cfg_sec, 'retrievalTopKContent', 64))
                        _r_topk_heat = int(_cfg_knob(
                            _cfg_sec, 'retrievalTopKHeat', 64))
                        _r_outer_topk = int(_cfg_knob(
                            _cfg_sec, 'retrievalOuterTopK', 32))
                        # CRITICAL: when mode=='primer', gamma and delta must
                        # be 0 to preserve byte-identity with pre-Phase-5.
                        if mode == 'primer':
                            _r_gamma = 0.0
                            _r_delta = 0.0

                        slot0 = int(top_slot[0].item())
                        order_buf = ss._order
                        if 0 <= slot0 < int(order_buf.shape[1]):
                            parent_order = int(order_buf[0, slot0].item())
                        else:
                            parent_order = 0
                        q = parent[0]  # row-0 canonical query
                        left = ss.retrieval_candidates_for_slot(
                            q, space_role_basis, cats[0], parent_order, batch=0,
                            topk_content=_r_topk_content,
                            topk_heat=_r_topk_heat,
                            alpha=_r_alpha, beta=_r_beta,
                            mode=mode, gamma=_r_gamma, delta=_r_delta,
                            outer_topk=_r_outer_topk)
                        right = ss.retrieval_candidates_for_slot(
                            q, space_role_basis, cats[1], parent_order, batch=0,
                            topk_content=_r_topk_content,
                            topk_heat=_r_topk_heat,
                            alpha=_r_alpha, beta=_r_beta,
                            mode=mode, gamma=_r_gamma, delta=_r_delta,
                            outer_topk=_r_outer_topk)
                        if left:
                            if left.get('rows') is not None:
                                reverse_kwargs['left_rows'] = left['rows']
                            if left.get('priming') is not None:
                                reverse_kwargs['left_priming'] = left['priming']
                        if right:
                            if right.get('rows') is not None:
                                reverse_kwargs['right_rows'] = right['rows']
                            if right.get('priming') is not None:
                                reverse_kwargs['right_priming'] = (
                                    right['priming'])
                except Exception:
                    # Any failure -> plain reverse (never break generation).
                    reverse_kwargs = {}

        # Reverse only when a FAITHFUL inverse is available. Two ways an op
        # opts out: (1) it declares ``reverse_dispatchable = False`` because
        # no inverse exists at all (a lossy op, e.g. AdverbLayer); (2) it
        # declares ``reverse_required_kwargs`` naming operands the reverse
        # path could not recover (VerbLayer needs ``verb_what``). In either
        # case do NOT invoke reverse -- fabricating a split would corrupt the
        # reconstruction. Fall back to the identity stub (the sanctioned
        # "no inverse" pass-through; generation must never break).
        _required = getattr(layer, 'reverse_required_kwargs', ())
        _can_reverse = (getattr(layer, 'reverse_dispatchable', True)
                        and all(k in reverse_kwargs for k in _required))
        if not _can_reverse:
            child = _identity_stub()
        else:
            try:
                child = layer.reverse(parent, basis=space_role_basis,
                                      **reverse_kwargs)
            except TypeError:
                # Backward-compat for layer reverses that don't accept the
                # basis kwarg yet (NotLayer, NonLayer, base GrammarLayer,
                # ...). The retry call must also be guarded: subclasses of
                # the base GrammarLayer that don't override .reverse inherit
                # ``raise NotImplementedError`` from the Layer base (when
                # ``self.butterfly`` is False), which must degrade to the
                # identity stub, not propagate.
                try:
                    child = layer.reverse(parent)
                except Exception:
                    child = _identity_stub()
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

            # Phase 3b CAPSTONE — reverse self-priming (plan
            # doc/plans/2026-06-06-symbolic-heat-retrieval.md §Reverse-path
            # responsibilities steps 7-8; plan-test 10). After a HEAT-STEERED
            # binary pick (reverse_kwargs was used), prime the selected
            # operand rows so subsequent reverse steps in the same sentence
            # see them as hot. BEST-EFFORT: gated on priming_enabled and fully
            # guarded — any difficulty is silently skipped (it must never
            # break generation). The recommender returns operand VECTORS, not
            # ids, so each selected row id is recovered by matching the
            # returned vector against the candidate ``rows`` subset of
            # ``W = space_role_basis.getW()`` (a small set), row-0 canonical.
            if reverse_kwargs:
                try:
                    tax = getattr(ss, 'taxonomy', None)
                    if (tax is not None
                            and getattr(tax, 'priming_enabled', False)
                            and space_role_basis is not None
                            and hasattr(space_role_basis, 'getW')):
                        W_rec = space_role_basis.getW()
                        if W_rec is not None:
                            for vec, side in ((left[0], 'left_rows'),
                                              (right[0], 'right_rows')):
                                cand = reverse_kwargs.get(side, None)
                                rid = self._recover_selected_row(
                                    vec, W_rec, cand)
                                if rid is not None and rid >= 0:
                                    tax.note_selection(rid, batch=0)
                                    ss._commit_priming(0, rid)
                except Exception:
                    # Never let a priming-lifecycle bug break generation.
                    pass
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
            # Dissipate priming between reverse calls (Phase 3b).
            # No-op today: the priming buffer is unallocated or all-1.0
            # until forward heat updates land in Phase 4.  Guard ensures
            # a missing symbolSpace / taxonomy is silently skipped.
            ss = getattr(subspace, 'symbolSpace', None)
            tax = getattr(ss, 'taxonomy', None) if ss is not None else None
            if tax is not None:
                tax.decay(temporal_decay=getattr(tax, 'temporal_decay', 0.9))
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
            syntactic_layer: per-space_role SyntacticLayer for REDUCE.
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
            syntactic_layer: per-space_role SyntacticLayer with ``execute``.
            grammar: Grammar (used for rule .where decoding when
                ``rule_codebook`` is omitted).
            rule_codebook: optional RuleCodebook for rule .where
                stamping.
            terminal_codebook: accepted for plan-API symmetry but
                NOT consumed yet -- the terminal snap currently
                lives in ``WholeSpace._stack_route_forward`` as
                the eager bridge; future phases can move it here.
            actions: explicit ``[('shift', payload, where_id), ...]``
                action list. Required until a learned policy is wired.

        Raises:
            NotImplementedError: when ``actions`` is None (no learned
                scorer yet). The error message points to the lower-
                level shift/reduce primitives for the explicit path.
        """
        # ``terminal_codebook`` is part of the plan's target signature
        # but the stack-rewrite path's snap stays in WholeSpace for
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
            syntactic_layer: per-space_role SyntacticLayer (provides
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


def _masked_softmax_lastdim(scores: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim that is NaN-safe for fully-masked rows.

    Standard ``F.softmax`` returns NaN for a row that is entirely
    ``-inf`` (every entry masked out). Here such a row is a
    structurally-impossible action whose action-level marginal is
    already 0, so the per-op posterior is multiplied by 0 downstream --
    any finite value is correct. This returns a 0 posterior on
    fully-dead rows and the ordinary softmax elsewhere, keeping the op
    posterior finite without altering the live (non-dead) rows or
    silencing a genuine divergence. Gradient-safe: dead rows carry no
    gradient (they are constant 0), live rows get the usual softmax
    gradient.
    """
    if scores.numel() == 0:
        return scores
    # A row is "dead" when its max over ops is non-finite (all -inf).
    row_max = scores.amax(dim=-1, keepdim=True)            # [..., 1]
    dead = ~torch.isfinite(row_max)                        # [..., 1] bool
    # Replace dead rows with zeros so softmax is finite (uniform) there;
    # zero out that uniform afterwards so the posterior is exactly 0.
    safe_scores = torch.where(dead, torch.zeros_like(scores), scores)
    post = F.softmax(safe_scores, dim=-1)
    post = torch.where(dead, torch.zeros_like(post), post)
    return post


def superposition_scale(temperature):
    """Score scale for the soft-superposition temperature (the parser's
    differentiable route, replacing Viterbi + straight-through under
    ``<learning>``). ``temperature`` in [0, 1] maps to ``1 - t``: at ``t=0``
    the scores pass through unchanged (the chooser's own softmax -- the
    sharp/deterministic exploit pass), at ``t=1`` the scores are zeroed so
    the superposition is uniform (the flat-random explore pass). Multiplying
    the scores by ``(1-t)`` keeps the chooser in the gradient path with the
    gradient scaled by ``(1-t)`` -- full at ``t=0``, vanishing at ``t=1``
    (a fully random route carries no preference to learn)."""
    return 1.0 - min(1.0, max(0.0, float(temperature or 0.0)))


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
    # A row can be ENTIRELY masked out (all -inf) when space_role-gating forbids
    # every op for that position/pair (see BinaryStructuredReductionLayer's
    # space_role mask). For such a row softmax(-inf, ...) is NaN, but the row's
    # ACTION marginal is exactly 0 (structurally impossible), so the
    # per-op product MUST be 0 -- not NaN. Use a masked softmax that
    # yields a 0 posterior on fully-dead rows (any finite value works
    # since it is multiplied by a 0 marginal; 0 keeps it tidy). This is
    # the correct value for a 0 x undefined structural cell, NOT silent
    # gating of a genuine numerical divergence (the action marginals are
    # the live, fail-loud-checked quantities).
    op_post_copy = _masked_softmax_lastdim(c)             # [B, N, R_copy]
    copy_marginal_op = copy_marginal.unsqueeze(-1) * op_post_copy
    if N > 1 and R_reduce > 0:
        op_post_reduce = _masked_softmax_lastdim(r)       # [B, N-1, R_reduce]
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
        # The gate MLP is sized to d_model (content .what width). A muxed
        # CS-space_role event h is [B, N, muxedSize] with where/when columns beyond
        # d_model; the routing decision reads content, so slice h to the gate's
        # input width. The branches (and the mixed output y) stay full-width,
        # so where/when ride through the mix untouched.
        d_in = self.gate_mlp[0].in_features
        h_gate = h[..., :d_in] if h.shape[-1] > d_in else h
        gate_logits = self.gate_mlp(h_gate) / self.temperature  # [B, N, 4]
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


class TransformChooser(nn.Module):
    """Routing policy: scores tool/location candidates for a structured
    layer.

    The plan separates the transform/tool IMPLEMENTATION (the
    ``GrammarLayer`` ops) from the CHOICE POLICY (which op to apply,
    where). ``UnaryStructuredLayer`` / ``BinaryStructuredReductionLayer``
    delegate their placement scoring here; the soft-DP / Viterbi route and
    the op execution stay on the layer.

    The default chooser (:class:`AnchorDotTransformChooser`) reproduces the
    layers' inline anchor-dot scoring exactly. ``MLPTransformChooser`` swaps
    in a contextual network behind a config flag. Subclasses implement
    ``score_unary`` / ``score_binary``.
    """

    def score_unary(self, x_score, applied_score, copy_anchor, apply_anchor,
                    cat_ctx=None):
        raise NotImplementedError

    def score_binary(self, x_score, reduced_score, copy_anchor, reduce_anchor,
                     cat_ctx=None):
        raise NotImplementedError


class AnchorDotTransformChooser(TransformChooser):
    """Anchor-dot placement scorer -- the scorer-level byte-identical default.

    Reproduces the original inline scoring (Stern et al. 2017 /
    Vaswani et al. 2017 style): the placement score is the inner product
    between a candidate's output and a per-rule learnable anchor. The scorer
    ignores ``cat_ctx``; the category-role prior is applied by the owning
    layer after scoring, so with ``categoryCodebook`` enabled (now the
    default) the END-TO-END route is no longer byte-identical even though
    this scorer is.

    Deliberately STATELESS: the ``copy_anchor`` / ``apply_anchor`` /
    ``reduce_anchor`` Parameters stay OWNED BY THE LAYER and are passed in
    at call time. Moving them into this submodule would rename their
    state_dict keys (``layer.copy_anchor`` -> ``layer.chooser.copy_anchor``)
    and risk a pinned basin; a param-less chooser adds no keys and keeps
    the scoring byte-identical. (The future MLPTransformChooser owns its own
    params -- a deliberate new-params cutover, behind a config flag.)
    """

    def score_unary(self, x_score, applied_score, copy_anchor, apply_anchor,
                    cat_ctx=None):
        """Return ``(copy_score, apply_score)`` for the unary layer.

        ``copy_score[b,n,c]  = <x_score[b,n,:],       copy_anchor[c,:]>``
        ``apply_score[b,n,a] = <applied_score[b,n,a,:], apply_anchor[a,:]>``

        ``cat_ctx`` is NOT consumed here -- the anchor-dot scorer stays
        stateless. The labelled-role category prior is added by the OWNING
        layer AFTER scoring (``_category_apply_prior``). Because
        ``categoryCodebook`` now defaults ON, that prior shifts routing by
        default, so the END-TO-END route is no longer byte-identical to the
        bare anchor-dot for category-bearing grammars (this scorer still is).
        """
        # Device safety (MPS): the anchors are the owning layer's Parameters; if
        # a device move missed them (e.g. choosers built lazily after the model's
        # .to(device)), align to the input's device. No-op when co-located.
        copy_anchor = copy_anchor.to(x_score.device)
        apply_anchor = apply_anchor.to(x_score.device)
        copy_score = torch.einsum('bnd,cd->bnc', x_score, copy_anchor)
        r_apply = int(apply_anchor.shape[0])
        if r_apply > 0 and applied_score.shape[2] > 0:
            apply_score = torch.einsum(
                'bnad,ad->bna', applied_score, apply_anchor)
        else:
            apply_score = x_score.new_zeros(
                x_score.shape[0], x_score.shape[1], r_apply)
        return copy_score, apply_score

    def score_binary(self, x_score, reduced_score, copy_anchor, reduce_anchor,
                     cat_ctx=None):
        """Return ``(copy_score, reduce_score)`` for the binary layer.

        ``copy_score[b,n,c]   = <x_score[b,n,:],            copy_anchor[c,:]>``
        ``reduce_score[b,p,r] = <reduced_score[b,p,r,:], reduce_anchor[r,:]>``

        ``cat_ctx`` is NOT consumed here -- the anchor-dot scorer stays
        stateless. The labelled-role category prior is added by the OWNING
        layer AFTER scoring (``_category_reduce_prior``). Because
        ``categoryCodebook`` now defaults ON, that prior shifts routing by
        default, so the END-TO-END route is no longer byte-identical to the
        bare anchor-dot for category-bearing grammars (this scorer still is).
        """
        # Device safety (MPS): align the owning layer's anchor Parameters to the
        # input's device in case a device move missed them. No-op when co-located.
        copy_anchor = copy_anchor.to(x_score.device)
        reduce_anchor = reduce_anchor.to(x_score.device)
        copy_score = torch.einsum('bnd,cd->bnc', x_score, copy_anchor)
        r_reduce = int(reduce_anchor.shape[0])
        if reduced_score.shape[1] > 0 and r_reduce > 0:
            reduce_score = torch.einsum(
                'bnrd,rd->bnr', reduced_score, reduce_anchor)
        else:
            reduce_score = x_score.new_zeros(
                x_score.shape[0], max(x_score.shape[1] - 1, 0), r_reduce)
        return copy_score, reduce_score


class MLPTransformChooser(TransformChooser):
    """Contextual MLP placement scorer -- the expressive cutover chooser.

    Replaces the anchor-dot single inner product with a learned MLP over
    per-candidate CONTEXT: the slot / pair state, the candidate op's output,
    a learned tool-identity embedding, and a sinusoidal position encoding.
    It produces the same per-(op, location) logit shapes as
    ``AnchorDotTransformChooser`` -- ``copy_score`` / ``apply_score`` (unary)
    and ``copy_score`` / ``reduce_score`` (binary) -- so it is a drop-in for
    the structured grammar layers.

    UNLIKE the anchor-dot chooser this OWNS parameters (the tool embeddings
    + the MLP), so enabling it CHANGES the state_dict and starts a fresh
    basin -- a deliberate cutover behind ``<transformChooser>mlp`` (default
    anchordot). The copy/apply/reduce anchors passed by the layer are
    ignored; the MLP conditions on context instead.

    Sizing (one chooser per layer): ``n_copy`` copy ops then ``n_op``
    apply/reduce ops, one tool-embedding row each (copy rows first).
    """

    def __init__(self, *, d_model, n_copy, n_op, embed_dim=8, pos_dim=8,
                 hidden=None, n_role_cats=0):
        super().__init__()
        self.d_model = int(d_model)
        self.n_copy = int(n_copy)
        self.n_op = int(n_op)
        self.embed_dim = int(embed_dim)
        self.pos_dim = int(pos_dim)
        # MetaSymbol Category codebook: per-slot syntactic-category context
        # width fed alongside the slot/cand/tool/pos features. 0 = no MLP input
        # widening. >0 widens the first Linear by ``n_role_cats``; ``_score``
        # then concatenates the per-slot role vector (zeros when no
        # ``cat_ctx`` is supplied at call time).
        self.n_role_cats = int(n_role_cats)
        hidden = int(hidden) if hidden is not None else max(8, self.d_model)
        self.tool_embedding = nn.Parameter(
            torch.randn(max(1, self.n_copy + self.n_op), self.embed_dim) * 0.02)
        in_dim = (2 * self.d_model + self.embed_dim
                  + self.n_role_cats + self.pos_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def _pos_emb(self, n, device, dtype):
        """Sinusoidal positional encoding ``[n, pos_dim]``."""
        if n <= 0:
            return torch.zeros(0, self.pos_dim, device=device, dtype=dtype)
        pos = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)   # [n,1]
        half = max(1, self.pos_dim // 2)
        k = torch.arange(half, device=device, dtype=dtype)
        div = torch.exp(-math.log(10000.0) * k / float(half))           # [half]
        ang = pos * div.unsqueeze(0)                                    # [n, half]
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if pe.shape[-1] < self.pos_dim:                                # pad odd
            pe = torch.cat(
                [pe, pe.new_zeros(n, self.pos_dim - pe.shape[-1])], dim=-1)
        return pe[:, :self.pos_dim]

    def _score(self, slot, cand, tool_rows, pos, cat_ctx=None):
        """Score ``R`` candidates at each of ``Npos`` locations.

        ``slot`` / ``cand``: ``[B, Npos, D]`` (broadcast over R) or ``cand``
        ``[B, Npos, R, D]``; ``tool_rows``: ``[R, embed_dim]``; ``pos``:
        ``[Npos, pos_dim]``. ``cat_ctx``: optional per-slot category
        role vector ``[B, Npos, n_role_cats]`` broadcast over R; ``None`` (or
        ``n_role_cats == 0``) feeds a zero block / no block. Returns
        ``[B, Npos, R]``.
        """
        B, Npos = slot.shape[0], slot.shape[1]
        R = int(tool_rows.shape[0])
        if R == 0 or Npos == 0:
            return slot.new_zeros(B, Npos, R)
        slot_e = slot.unsqueeze(2).expand(B, Npos, R, self.d_model)
        cand_e = (cand if cand.dim() == 4
                  else cand.unsqueeze(2).expand(B, Npos, R, self.d_model))
        tool_e = tool_rows.view(1, 1, R, self.embed_dim).expand(
            B, Npos, R, self.embed_dim)
        pos_e = pos.view(1, Npos, 1, self.pos_dim).expand(
            B, Npos, R, self.pos_dim)
        if self.n_role_cats > 0:
            # Category context block: the per-slot role vector, broadcast over
            # candidates and placed between the tool and position blocks (the
            # in_dim layout the first Linear was sized for). Missing context
            # (terminal codebook off, composed slot) -> a zero block.
            if cat_ctx is None:
                cat_e = slot.new_zeros(B, Npos, R, self.n_role_cats)
            else:
                cat_e = cat_ctx.to(slot.dtype).unsqueeze(2).expand(
                    B, Npos, R, self.n_role_cats)
            feat = torch.cat([slot_e, cand_e, tool_e, cat_e, pos_e], dim=-1)
        else:
            feat = torch.cat([slot_e, cand_e, tool_e, pos_e], dim=-1)
        # Cast to the slot dtype so the computed path and the degenerate
        # zero fallbacks agree even under autocast (the MLP may emit
        # bf16/fp16; the fallbacks keep the input dtype).
        return self.mlp(feat).squeeze(-1).to(slot.dtype)              # [B,Npos,R]

    def score_unary(self, x_score, applied_score, copy_anchor, apply_anchor,
                    cat_ctx=None):
        B, N, D = x_score.shape
        pos = self._pos_emb(N, x_score.device, x_score.dtype)
        copy_rows = self.tool_embedding[:self.n_copy]
        copy_score = self._score(x_score, x_score, copy_rows, pos,
                                 cat_ctx=cat_ctx)                      # copy=slot
        r_apply = applied_score.shape[2] if applied_score.dim() == 4 else 0
        if r_apply > 0:
            apply_rows = self.tool_embedding[self.n_copy:self.n_copy + r_apply]
            apply_score = self._score(x_score, applied_score, apply_rows, pos,
                                      cat_ctx=cat_ctx)
        else:
            apply_score = x_score.new_zeros(B, N, 0)
        return copy_score, apply_score

    def score_binary(self, x_score, reduced_score, copy_anchor, reduce_anchor,
                     cat_ctx=None):
        B, N, D = x_score.shape
        pos = self._pos_emb(N, x_score.device, x_score.dtype)
        copy_rows = self.tool_embedding[:self.n_copy]
        copy_score = self._score(x_score, x_score, copy_rows, pos,
                                 cat_ctx=cat_ctx)
        if (N >= 2 and reduced_score.dim() == 4
                and reduced_score.shape[1] > 0 and reduced_score.shape[2] > 0):
            pair_slot = 0.5 * (x_score[:, :-1] + x_score[:, 1:])       # [B,N-1,D]
            # Pair the operands' category rows the same way the slot is paired
            # (the spec's left/right operand reduction; avg here). None keeps
            # the feature block neutral.
            pair_cat = (0.5 * (cat_ctx[:, :-1] + cat_ctx[:, 1:])
                        if cat_ctx is not None else None)
            pos_pair = self._pos_emb(N - 1, x_score.device, x_score.dtype)
            r_reduce = int(reduced_score.shape[2])
            # The reduce op-axis must equal n_op (the construction-time
            # r_reduce) so the R_reduce layout is consistent with the
            # degenerate branch below and with the cross-product / Viterbi
            # readers -- fail loud on any future contract drift.
            assert r_reduce == self.n_op, (
                f"reduced_score op-axis {r_reduce} != n_op {self.n_op}")
            reduce_rows = self.tool_embedding[
                self.n_copy:self.n_copy + r_reduce]
            reduce_score = self._score(
                pair_slot, reduced_score, reduce_rows, pos_pair,
                cat_ctx=pair_cat)
        else:
            reduce_score = x_score.new_zeros(B, max(N - 1, 0), self.n_op)
        return copy_score, reduce_score


def make_transform_chooser(kind, *, d_model, n_copy, n_op, n_role_cats=0):
    """Factory: build the placement chooser for a structured layer.

    ``kind`` ``"anchordot"`` (default) -> the stateless behavior-preserving
    scorer (no params, basin unchanged); ``"mlp"`` -> the contextual
    :class:`MLPTransformChooser` (owns params; a deliberate new-basin
    cutover behind ``<transformChooser>``).

    ``n_role_cats``: the MetaSymbol category-context width fed to the MLP
    chooser; 0 leaves the MLP feature set unchanged. Anchor-dot uses category
    context via the structured layer's labelled-role prior.
    """
    k = str(kind or "anchordot").strip().lower()
    if k == "mlp":
        return MLPTransformChooser(
            d_model=d_model, n_copy=n_copy, n_op=n_op, n_role_cats=n_role_cats)
    # Accept exactly the values the <transformChooser> XSD enum allows, so
    # the factory and schema validation agree on the legal set.
    if k != "anchordot":
        raise ValueError(
            f"<transformChooser> must be 'anchordot' or 'mlp' (got {kind!r}).")
    return AnchorDotTransformChooser()


def compute_role_vocabulary(grammar):
    """Enumerate the role-collapsed grammar's operator roles for the
    MetaSymbol Category codebook (doc/Language.md "Participation Categories as
    the Chooser's Syntactic-Category Context").

    A ROLE is one operator argument slot. Input role ``(method, pos)`` renders
    ``<method>_I<pos+1>`` (operand position ``pos`` fed INTO the operator);
    output role ``(method,)`` renders ``<method>_O1`` (the result produced BY
    the operator) -- the same naming as ``bin/participation.py``. Roles are
    read off the live UPWARD (compose) rules; a rule with no ``method_name``
    (epsilon / passthrough projection) declares no role and is skipped.

    Returns ``(roles, role_index, n_roles)``: ``roles`` a deterministically
    ordered list of role-name strings (sorted inputs, then sorted outputs),
    ``role_index`` mapping role-name -> int column, ``n_roles == len(roles)``.
    Enumerate ONCE at build from the live rule set (not the .grammar text) --
    ``method_name is None`` projections drop out, so the file is not the
    source of truth.
    """
    rules = (getattr(grammar, 'rules_upward', None)
             or getattr(grammar, 'rules', None) or [])
    in_roles = set()
    out_roles = set()
    for r in rules:
        method = getattr(r, 'method_name', None)
        if not method:
            continue
        method = str(method)
        rhs = getattr(r, 'rhs_symbols', None) or []
        for pos in range(len(rhs)):
            in_roles.add((method, pos))
        out_roles.add(method)
    roles = [f"{m}_I{pos + 1}" for (m, pos) in sorted(in_roles)]
    roles += [f"{m}_O1" for m in sorted(out_roles)]
    role_index = {name: i for i, name in enumerate(roles)}
    return roles, role_index, len(roles)


def _role_index_for_categories():
    try:
        return compute_role_vocabulary(TheGrammar)[1]
    except Exception:
        return {}


def _role_column(role_index, method, suffix):
    if not method:
        return None
    return role_index.get(f"{method}_{suffix}")

class BinaryStructuredReductionLayer(nn.Module):
    """One layer: contextualize, score, route, compact (hard + soft).

    Args:
        d_model: feature dim.
        ops: sequence of binary nn.Modules; len(ops) = R_reduce. Each
             receives (left[B, N-1, D], right[B, N-1, D]) and returns
             [B, N-1, D]. The Viterbi route picks one op per reduce site.
        op_space_roles: optional list of space_role tags ('CS' / 'SS' / ...)
             parallel to ``ops``. When supplied alongside ``op_names``,
             enables per-position space_role-gated scoring: a rule only scores
             at a pair whose left and right operands are both at the
             rule's space_role. ``lift`` flips the carried operand's space_role from
             CS to SS and ``lower`` flips SS to CS (plan \xa76).
        op_names: optional list of method names parallel to ``ops``;
             used to identify the lift / lower ops for space_role flipping.
        r_copy: number of copy "ops" (typically 1; >1 lets the router
             distinguish copy specializations like typed identities).
        context_net: optional contextualizer for h. Defaults to identity.
        temperature: comparator-mixer softmax temperature.

    """

    def __init__(self, *, d_model, ops, r_copy=1, context_net=None,
                 temperature=1.0, op_space_roles=None, op_names=None,
                 chooser="anchordot", n_role_cats=0):
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
        # Placement scoring delegated to the TransformChooser.
        # ``chooser="anchordot"`` (default) is the stateless scorer that
        # keeps the anchors owned here (state_dict unchanged,
        # byte-identical); ``"mlp"`` builds the contextual
        # MLPTransformChooser (owns params -> new basin), sized to this
        # layer's r_copy copy ops + r_reduce reduce ops.
        self.chooser = make_transform_chooser(
            chooser, d_model=self.d_model,
            n_copy=self.r_copy, n_op=self.r_reduce, n_role_cats=n_role_cats)
        self.comparator = ComparatorMixer(
            d_model=self.d_model, temperature=temperature)
        # Per-rule name / space_role metadata retained for op identification
        # (e.g. lift / lower). Space-role-gated masking has been removed.
        self.op_names = list(op_names) if op_names is not None else None
        self.op_space_roles = (
            list(op_space_roles) if op_space_roles is not None else None)

    def _category_reduce_prior(self, cat_ctx):
        """Role-frequency prior for binary ops from labelled operand slots."""
        if cat_ctx is None or self.op_names is None or cat_ctx.shape[1] < 2:
            return None
        role_index = _role_index_for_categories()
        if not role_index:
            return None
        left = cat_ctx[:, :-1, :]
        right = cat_ctx[:, 1:, :]
        priors = []
        for name in self.op_names:
            c1 = _role_column(role_index, name, "I1")
            c2 = _role_column(role_index, name, "I2")
            score = left.new_zeros(left.shape[0], left.shape[1])
            if c1 is not None and c1 < left.shape[-1]:
                score = score + left[..., c1]
            if c2 is not None and c2 < right.shape[-1]:
                score = score + right[..., c2]
            priors.append(0.5 * score)
        if not priors:
            return None
        return torch.stack(priors, dim=-1)

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

    # CS-side execution stage: this applies the chosen reductions to the
    # concept tensors (``op(left, right)`` over self.ops in
    # _stacked_reduced), the counterpart to SymbolSubSpace.compose's
    # WS-side analysis. See SymbolSubSpace.compose docstring for the split.
    def forward(self, x, *, span_start=None, span_end=None, cat_ctx=None):
        """Score, route via Viterbi, compact; return (hard, soft, routing).

        Returns:
            hard_slab: [B, N, D] argmax-selected per-position action.
            soft_slab: [B, N, D] DP-marginal-weighted blend (gradient surrogate).
            routing:   dict with masks, scores, marginals, lengths, gates,
                       optionally ``span_start`` / ``span_end``.
        Degenerate N<=1 returns the input twice with a stub routing.

        ``cat_ctx``: optional per-slot category role vector ``[B, N,
        n_role_cats]``. It feeds the chooser where supported and always
        contributes the layer-level labelled-role prior.
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
        # Wire contextual BIND to live parse state: stash the current
        # constituent slab on any op that resolves a missing NP against the
        # constructed left-context. Applied before _stacked_reduced so the
        # op's compose(left, right) over all pairs sees the live slab.
        for _op in self.ops:
            _gl = getattr(_op, 'gl', _op)          # unwrap _BinaryGrammarOpAdapter
            if hasattr(_gl, 'set_bind_context'):
                _gl.set_bind_context(slab=x)
        stacked_reduced = self._stacked_reduced(x)             # [B, N-1, R, D]

        # Anchor-based scoring (replaces the old scorer MLP):
        #   copy_score[b, n, c]   = <x[b, n, :],            copy_anchor[c, :]>
        #   reduce_score[b, p, r] = <stacked_reduced[..., r, :], reduce_anchor[r, :]>
        # Each rule's anchor is part of its own parameter set, so the
        # placement score is a derived quantity of the rule's own
        # computation -- one optimizer, one graph, no separate scorer.
        # Anchors are sized to d_model (content .what width); a muxed CS-space_role
        # event carries where/when columns beyond d_model, so score on the
        # content slice (the reduce ops above still transform the full event).
        x_score = x[..., :self.d_model] if x.shape[-1] > self.d_model else x
        sr_score = (stacked_reduced[..., :self.d_model]
                    if stacked_reduced.shape[-1] > self.d_model
                    else stacked_reduced)
        copy_score, reduce_score = self.chooser.score_binary(
            x_score, sr_score, self.copy_anchor, self.reduce_anchor,
            cat_ctx=cat_ctx)
        cat_prior = self._category_reduce_prior(cat_ctx)
        if cat_prior is not None and cat_prior.shape == reduce_score.shape:
            reduce_score = reduce_score + cat_prior.to(
                device=reduce_score.device, dtype=reduce_score.dtype)

        # Soft-superposition temperature (the parser's differentiable route
        # under <learning>; doc/Language.md "weighted deduction"). When set,
        # the FORWARD value is the pure sum-product superposition at this
        # temperature -- NO Viterbi route and NO straight-through; the
        # chooser is in the gradient path directly. ``0`` = the chooser's own
        # (sharp/deterministic) softmax, ``1`` = uniform; the scores are
        # scaled by ``1-t`` (superposition_scale). Default None keeps the
        # legacy hard-Viterbi + straight-through forward (byte-identical).
        # Viterbi is still computed for the routing / tree read-off (that
        # read-off is outside the gradient path in both modes).
        _st = getattr(self, 'superposition_temperature', None)
        if _st is not None:
            _sc = superposition_scale(_st)
            cs_dp, rs_dp = copy_score * _sc, reduce_score * _sc
        else:
            cs_dp, rs_dp = copy_score, reduce_score
        soft = binary_tiling_soft_dp(cs_dp, rs_dp)
        hard = binary_tiling_viterbi(copy_score, reduce_score)

        if hard["reduce_mask"].numel() > 0:
            reduce_op_per_pair = hard["reduce_mask"].argmax(-1)  # [B, N-1]
        else:
            reduce_op_per_pair = torch.zeros(
                B, 0, device=x.device, dtype=torch.long)

        # Hardened op-selection: forward uses one-hot argmax (sparse
        # commitment to a single op per pair); backward uses the soft
        # softmax over reduce_score so the scorer still receives gradient
        # via straight-through. A fully space_role-masked pair (all -inf) gets a
        # 0 soft posterior (NaN-safe; see _masked_softmax_lastdim) -- its
        # reduce marginal is 0 downstream so the chosen reduction is
        # dropped from the slab, but the slab must stay FINITE (a NaN here
        # would propagate into the next round's folded slab and poison the
        # whole DP).
        if reduce_score.numel() > 0:
            if _st is not None:
                # Pure soft op superposition at temperature (no straight-through).
                op_weights = _masked_softmax_lastdim(rs_dp)     # [B, N-1, R]
            else:
                op_soft = _masked_softmax_lastdim(reduce_score)     # [B, N-1, R]
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
        if _st is not None:
            # Pure soft superposition: the forward value IS the differentiable
            # sum-product marginal at this temperature (no hard route, no
            # straight-through). This is the parser's training-time route.
            copy_marginal_st = soft["copy_marginal"]
            reduce_marginal_st = soft["reduce_marginal"]
        else:
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
                 temperature=1.0, chooser="anchordot", n_role_cats=0,
                 op_names=None):
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
        # Placement scoring delegated to the TransformChooser.
        # ``chooser="anchordot"`` (default) is the stateless scorer
        # (anchors owned here, state_dict unchanged); ``"mlp"`` builds the
        # contextual MLPTransformChooser (owns params -> new basin), sized
        # to r_copy copy + r_apply ops.
        self.chooser = make_transform_chooser(
            chooser, d_model=self.d_model,
            n_copy=self.r_copy, n_op=self.r_apply, n_role_cats=n_role_cats)
        self.op_names = list(op_names) if op_names is not None else None

    def _category_apply_prior(self, cat_ctx):
        """Role-frequency prior for unary ops from the labelled input slot."""
        if cat_ctx is None or self.op_names is None:
            return None
        role_index = _role_index_for_categories()
        if not role_index:
            return None
        priors = []
        for name in self.op_names:
            col = _role_column(role_index, name, "I1")
            if col is None or col >= cat_ctx.shape[-1]:
                priors.append(cat_ctx.new_zeros(cat_ctx.shape[0], cat_ctx.shape[1]))
            else:
                priors.append(cat_ctx[..., col])
        if not priors:
            return None
        return torch.stack(priors, dim=-1)

    def _stacked_applied(self, x):
        """[B, N, R_apply, D] each unary op applied to every position."""
        if self.r_apply == 0:
            B, N, D = x.shape
            return x.new_zeros(B, N, 0, D)
        per_op = [op(x) for op in self.ops]
        return torch.stack(per_op, dim=2)

    def forward(self, x, cat_ctx=None):
        """Score, choose per-position action, return (hard, soft, routing).

        Hard slab argmax-selects one branch per position; soft slab is
        the softmax-weighted blend over (copy_branch + applied_ops).
        Straight-through gradient connects the hard-forward / soft-backward.

        ``cat_ctx``: optional per-position category role vector ``[B, N,
        n_role_cats]``. It feeds the chooser where supported and always
        contributes the layer-level labelled-role prior.
        """
        B, N, D = x.shape
        h = self.context_net(x)
        applied = self._stacked_applied(x)                 # [B, N, R_apply, D]

        # Anchor-based scoring (replaces the old scorer MLP):
        #   copy_score[b, n, c]  = <x[b, n, :],            copy_anchor[c, :]>
        #   apply_score[b, n, a] = <applied[b, n, a, :], apply_anchor[a, :]>
        # The routing anchors are sized to d_model (= content .what width). A
        # muxed CS-space_role event is [B, N, muxedSize] with where/when columns beyond
        # d_model that the anchors don't score, so take the content slice for
        # the inner products. The ops above still see (and transform) the full
        # event; only the scalar routing scores read content.
        x_score = x[..., :self.d_model] if D > self.d_model else x
        applied_score = (applied[..., :self.d_model]
                         if applied.shape[-1] > self.d_model else applied)
        copy_score, apply_score = self.chooser.score_unary(
            x_score, applied_score, self.copy_anchor, self.apply_anchor,
            cat_ctx=cat_ctx)
        cat_prior = self._category_apply_prior(cat_ctx)
        if cat_prior is not None and cat_prior.shape == apply_score.shape:
            apply_score = apply_score + cat_prior.to(
                device=apply_score.device, dtype=apply_score.dtype)
        # Soft-superposition temperature (the differentiable route under
        # <learning>; same contract as the binary layer). When set, the
        # forward is the pure softmax superposition at this temperature
        # (scores scaled by 1-t) -- no argmax / straight-through; the chooser
        # is in the gradient path. Default None keeps the legacy
        # argmax-forward / soft-backward straight-through (byte-identical).
        _st = getattr(self, 'superposition_temperature', None)
        if _st is not None:
            _sc = superposition_scale(_st)
            action_logits = torch.cat(
                [copy_score, apply_score], dim=-1) * _sc / self.temperature
            action_probs = F.softmax(action_logits, dim=-1)
        else:
            action_logits = torch.cat(
                [copy_score, apply_score], dim=-1) / self.temperature
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

        # Hard slab: argmax over actions. Read the UNSCALED scores (not the
        # _sc-scaled action_logits) so the routing read-off is temperature-
        # stable and matches the binary layer's unscaled Viterbi read-off.
        # At t=1 superposition_scale is 0, which would zero action_logits and
        # collapse the argmax to all-copy; argmax is invariant to the positive
        # 1/temperature factor, so this is byte-identical on the default path.
        action_id = torch.cat(
            [copy_score, apply_score], dim=-1).argmax(dim=-1)   # [B, N]
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
# Chart -- RETIRED 2026-05-27 (Stage 3 of doc/plans/2026-05-26-two-loop-
# pi-sigma-substrate.md).
#
# The soft-superposition CKY chart parser and its inside / outside
# passes, packed rule-table machinery, load-balance bookkeeping, top-K
# gating, and POS side-channel are gone. The signal router
# (``LanguageLayer`` -- see above) is the canonical parser. SymbolSubSpace
# constructs ``self.languageLayer`` directly; rule-firing probability is
# served by ``SymbolSubSpace.should_run_rule`` via the grammar's
# ``rule_probability`` lookup. The retired XML knobs
# (``parserBackend``, ``routerKind``, ``chartTau``, ``chartTopK``,
# ``chartNoiseEps``) raise a loud ``ValueError`` at config load time --
# see ``_assert_retired_chart_knobs_absent`` above.
# =====================================================================


# =====================================================================
# Per-space SyntacticLayer (2026-05-01 refactor; legacy class retired
# 2026-05-08).
#
# Spec: doc/specs/2026-05-01-syntactic-layer-refactor.md §4.
#
# Each PartSpace / ConceptualSpace / WholeSpace owns one of
# these. Holds the parametrized GrammarLayer instances for its space_role's
# rules and dispatches `forward` / `reverse` based on the rule choice
# the chart wrote into ``word_space.current_rules`` /
# ``generate_rules`` (Q4 / Q10.1).
# =====================================================================
class SyntacticLayer(Layer):
    """Per-space dispatcher.

    Construction:
        SyntacticLayer(space_role='CS', word_space=word_space,
                            host_layers={'pi': pi_layer},
                            host_space=concept_space)

    Each entry in ``host_layers`` is registered with ``word_space`` at
    construction. The space's ``forward()`` and ``reverse()`` delegate
    here; ``forward()`` reads ``word_space.current_rules[space_role]``,
    advances a per-space_role cursor, and dispatches to the appropriate
    layer's ``compose`` (binary) or ``forward`` (unary). ``reverse()``
    mirrors via ``word_space.generate_rules[space_role]`` and ``layer.generate``.

    The cursor resets at the start of each new ``word_space.compose()``
    / ``word_space.generate()`` call via the generation counters on
    SymbolSpace (Q10.1).

    Per the 2026-05-07 rollback: there is no ``default_rule`` parameter.
    The grammar XML drives which rules fire — when the chart hasn't
    populated rules for this space_role the dispatch is a no-op.
    """

    def __init__(self, space_role, word_space, host_layers, host_space=None):
        """Register host layers with the SymbolSpace dispatch table.

        ``host_layers`` is a name -> Layer mapping; each is registered
        under ``(space_role, rule_name)`` on the SymbolSpace so the chart can
        dispatch into the right parametrized fold. SymbolSpace / host_space
        are stashed via ``object.__setattr__`` to avoid the nn.Module
        ownership cycle (SymbolSpace owns the chart -> chart references
        this layer -> this layer references SymbolSpace).
        """
        super().__init__(0, 0)
        self.space_role = str(space_role)
        # Stash host_layers in two parallel structures: ModuleList for
        # nn.Module bookkeeping (so optimizer scans see the parameters)
        # and a name-keyed dict for O(1) dispatch lookup.
        layers_list = [layer for layer in host_layers.values()
                       if layer is not None]
        self.layers = nn.ModuleList(layers_list)
        self._by_name = {name: layer for name, layer in host_layers.items()
                         if layer is not None}
        # Register each host_layer with the symbolSpace's host_layer
        # registry so the chart can dispatch into them.
        for rule_name, layer in self._by_name.items():
            word_space.register_host_layer(self.space_role, rule_name, layer)
        self._cursor_compose = 0
        self._cursor_generate = 0
        self._cursor_compose_gen = -1
        self._cursor_generate_gen = -1
        # Stash the symbolSpace and host_space as non-Module attributes
        # to avoid the circular nn.Module ownership trap (symbolSpace
        # owns the chart; chart's host_layer registry references this
        # layer's children; this layer references symbolSpace).
        object.__setattr__(self, '_word_space', word_space)
        # ``host_space`` is the per-space_role Space (Perceptual / Conceptual /
        # Symbolic) that owns this dispatcher. When the chart fires
        # ``pi`` / ``sigma`` and the host space exposes
        # ``_pi_reverse`` / ``_sigma_reverse`` (two-pass ergodic mode
        # routes through ``pi2`` / ``sigma2``), reverse() delegates
        # there instead of the layer's bare ``reverse``.
        object.__setattr__(self, '_host_space', host_space)

    # -- cursor management ---------------------------------------------
    def _space_role_index(self):
        """Map this layer's space_role label to its slot in the per-sentence
        SymbolSpace ``cursor`` tensor (shape ``[n_space_roles=3]``).

        ``space_role`` is set once at construction to one of the string
        literals 'subsymbolic' / 'CS' / 'SS' (Language.py
        ``_attach_per_space_syntactic_layer`` passes ``space_role='subsymbolic'`` /
        ``'CS'`` / ``'SS'``; ``__init__`` coerces with ``str(space_role)``),
        so this map is total over the live domain.
        """
        return {'subsymbolic': 0, 'CS': 1, 'SS': 2}[str(self.space_role)]

    def _next_rule_name(self, *, direction):
        """Pop the next rule name for ``direction`` ('compose' or
        'generate'). Resets the cursor when symbolSpace has bumped its
        generation counter for this direction.

        Reads ``word_space.current_rules`` / ``generate_rules`` as
        ``dict[space_role, list[list[int]]]`` (per-row, per-step). For now
        we use row 0 as the canonical sequence; per-row dispatch (where
        rows fire different rules at the same step) is a follow-on.

        Returns the rule's ``method_name`` (string) or ``None`` when
        no chart rule is available (no code-level fallback — the grammar
        XML is the sole source of truth). The method name is the key
        used in ``self._by_name``.
        """
        ss = self._word_space
        if direction == 'compose':
            rules = ss.current_rules
            # ``ss.cursor`` is a host ``list[int]`` of length 3 (one per
            # space_role subsymbolic/CS/SS). Reading via Python list indexing gives a
            # backed Python int the trace can compare with
            # ``len(per_step)`` — an int64 tensor read via ``int(...)``
            # would yield an unbacked SymInt and crash
            # ``fullgraph=True``. The per-compose reset happens
            # unconditionally at the top of SymbolSpace.compose, so there
            # is NO data-dependent generation gate here (recompile
            # cause #3 eliminated).
            ti = self._space_role_index()
            cursor = ss.cursor[ti]
        else:
            rules = ss.generate_rules
            gen = ss._generate_generation
            if gen != self._cursor_generate_gen:
                self._cursor_generate = 0
                self._cursor_generate_gen = gen
            cursor = self._cursor_generate
        per_space_role = rules.get(self.space_role) if rules else None
        per_step = self._row_zero_rules(per_space_role)
        if cursor < len(per_step):
            rule_id = per_step[cursor]
            if direction == 'compose':
                ss.cursor[self._space_role_index()] = cursor + 1
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
    def _row_zero_rules(per_space_role):
        """Extract row 0's rule sequence from a per-row container.

        Tolerates both legacy ``list[int]`` (flat) and the multi-row
        ``list[list[int]]`` shape so callers using either contract
        keep working during the migration window.
        """
        if not per_space_role:
            return []
        # Multi-row: list of lists.
        if isinstance(per_space_role[0], list):
            return per_space_role[0]
        # Flat list of ints (legacy).
        return per_space_role

    # -- Phase 2 executor API (cursor-free) -----------------------------
    #
    # See doc/plans/2026-05-20-subspace-what-stm-signalrouter-refactor.md
    # §"Phase 2: SyntacticLayer Executor API". The LanguageLayer calls
    # these directly with a rule_id it has already selected; no
    # SymbolSpace.current_rules indirection.

    def execute(self, rule_id, left, right=None):
        """Run the grammar op for ``rule_id`` on ``(left[, right])``.

        Resolves ``rule_id`` to a host layer via ``TheGrammar`` and
        ``self._by_name`` and calls ``layer.compose`` with the right
        number of operands for the rule's arity. Returns the parent
        tensor. No cursor; no SymbolSpace state read.

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
            # Post-2026-05-29 grammar-file-refactor (\xa75): the rule may
            # bind at a different space_role's syntactic layer than self
            # (intersection / union / lift / lower carry the CS-space_role class
            # space_role so they register on ConceptualSpace rather than
            # WholeSpace; an SS-space_role execute that hits one of those
            # rule_ids needs the layer even though _by_name doesn't have
            # it). Fall back to a fresh GRAMMAR_LAYER_CLASSES instance for
            # the dispatch; parameterized layers that need an inner pi /
            # sigma won't instantiate (TypeError) and we re-raise the
            # original KeyError so the failure mode stays loud.
            cls = GRAMMAR_LAYER_CLASSES.get(method_name)
            if cls is not None:
                try:
                    layer = cls()
                except TypeError:
                    layer = None
        if layer is None:
            raise KeyError(
                f"SyntacticLayer.execute: space_role={self.space_role!r} has no host "
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
    # subspace's space_role-appropriate field:
    #   * SS space_role: the .what content (symbol activations)
    #   * subsymbolic / CS space_role: the .event content (percept / concept activations)
    #
    # Rule choices come from word_space.current_rules / generate_rules
    # (populated by the chart). Cursor advances one step per call.
    def forward(self, subspace):
        """Fire one fold step on ``subspace`` per the chart's rule choice.

        Materializes the subspace's space_role-appropriate field, applies
        the chosen rule's GrammarLayer.forward, writes the result back
        into the same field. Returns the (possibly-mutated) subspace.

        Per the 2026-05-07 rollback, when the chart hasn't written a
        rule for this space_role, dispatch is a no-op (no code-level
        fallback).

        Stage 3 (chart retirement): the signal router has already
        executed the derivation tensorially (languageLayer.compose
        folded the slab via the op modules and wrote the [B, 1, D] root
        state back into subspace.event). The legacy per-rule unary fold
        here would double-apply the op (and crash on truly-binary ops
        like ConjunctionLayer / DisjunctionLayer that don't expose a
        unary forward). When a non-default grammar is wired, skip it.
        """
        ss = self._word_space
        if (ss is not None
                and getattr(ss, 'languageLayer', None) is not None
                and not getattr(ss, '_grammar_is_default_only', True)):
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
        subspace's space_role-appropriate field. No-op when the rule isn't
        invertible.

        Stage 3 (chart retirement): on the signal-router path the
        router's ``generate`` handles inverse routing tensorially,
        so the per-rule reverse here is a cursor-advance no-op.
        """
        ss = self._word_space
        if (ss is not None
                and getattr(ss, 'languageLayer', None) is not None
                and not getattr(ss, '_grammar_is_default_only', True)):
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
        # the host space exposes a space_role-specific ``_pi_reverse`` /
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

    # -- subspace I/O per space_role ------------------------------------------
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

        Space-role distinction is irrelevant here: every space_role's per-
        position read goes through ``materialize()``. Space-role-specific
        slicing of the muxed event (e.g. operating on the ``.what``
        bivector only) is the op's responsibility.
        """
        if subspace is None:
            return None
        # Activation-reading ops (IntersectionLayer / UnionLayer at
        # CS-space_role) read the bivector activation directly.
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

def build_space_syntactic_layer(space, word_space, *, space_role,
                                builtin_layers=None):
    """Construct a per-space SyntacticLayer.

    Args:
        space: the host Space (PartSpace / ConceptualSpace /
            WholeSpace). The constructed layer is stored on
            ``space.syntacticLayer`` and registered in the symbolSpace's
            host_layer registry.
        word_space: the SymbolSpace coordinator. Owns the host_layer
            registry and the chart.
        space_role: space_role name ('subsymbolic' / 'CS' / 'SS') used as the registry key.
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
        rule_space_role = getattr(rule, 'space_role', None)
        if rule_space_role != space_role:
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
        space_role=space_role, word_space=word_space,
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
    hosted on the SymbolSpace singleton.

    Also hosts the **priming buffer** for reverse-generation working-
    memory state (plan doc/plans/2026-05-20-primed-reverse-generation.md).
    The buffer lives on the Taxonomy because propagation walks
    parent/children adjacency; co-locating the state with the graph
    avoids indirection. The legacy in-process dicts
    (``_parent``/``_children``) are unused on the SymbolSpace's instance
    — propagation queries the attached ``embed.KnowledgeView`` instead.
    """

    # Default priming knobs (overrideable per instance via
    # ``configure_priming``). Plan doc/plans/2026-05-20-primed-reverse-
    # generation.md §Configuration.
    DEFAULT_PRIMING_DEPTH = 2
    DEFAULT_HOP_DECAY = 0.5
    DEFAULT_TEMPORAL_DECAY = 0.9
    DEFAULT_BOOST_INITIAL = 1.0
    # Class default True preserves the historical retrieval-helper gate for
    # bare Taxonomy() instances (priming_kwargs_for_slots /
    # retrieval_candidates_for_slot).  PRODUCTION SymbolSubSpace taxonomies are
    # set off-by-default by SymbolSubSpace.attach_knowledge, which calls
    # configure_priming(priming_enabled=<symbolicPriming>) (default False)
    # right where it allocates the priming buffer — so forward heat production
    # is off (zero training-path cost) unless <symbolicPriming> is set.
    # Plan doc/plans/2026-06-06-symbolic-heat-retrieval.md.
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

    def prime_with_weights(self, weights, batch=None):
        """Merge boost-above-unity row weights into the priming buffer
        (element-wise max; GrammarOpsPass §5).

        The single intent's per-tower boosts enter the WS retrieval
        plumbing here: the merged buffer flows to the inverse
        recommender through ``priming_kwargs_for_slots`` — no new
        mechanism, one new producer. ``weights``: ``[V]`` (``>= 1.0``
        by convention; sized to ``min(V, capacity)``, the rest stays).
        ``batch=None`` merges into every row, an integer into one.
        No-op when the buffer is unallocated (dark by construction);
        the merge dissipates through ``decay`` / ``reset`` exactly like
        every other priming write (sentence-scoped working memory).
        """
        if self._priming is None or weights is None:
            return
        import torch
        w = torch.as_tensor(weights, dtype=self._priming.dtype,
                            device=self._priming.device).reshape(-1)
        n = min(int(w.shape[0]), self._priming_capacity)
        if n <= 0:
            return
        if batch is None:
            self._priming[:, :n] = torch.maximum(
                self._priming[:, :n], w[:n].unsqueeze(0))
            return
        b = int(batch)
        if 0 <= b < self._priming_B:
            self._priming[b, :n] = torch.maximum(
                self._priming[b, :n], w[:n])

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

    # -- derived heat: content+heat retrieval (plan 2026-06-06-symbolic-heat-retrieval) --

    def heat_mask(self, batch=0):
        """Return r = max(_priming - 1, 0) over the LIVE rows.

        Mirrors ``priming_mask`` slicing:
          * ``batch=None`` -> ``[B, V_live]``
          * an integer    -> ``[V_live]``
        Returns ``None`` when ``_priming`` is ``None`` AND when an integer
        ``batch`` is out of range (same as ``priming_mask``; the two cases
        are indistinguishable to callers).
        Default ``batch=0`` per plan
        ``doc/plans/2026-06-06-symbolic-heat-retrieval.md`` §API additions.
        """
        if self._priming is None:
            return None
        live = self._priming_live or self._priming_capacity
        if batch is None:
            return (self._priming[:, :live] - 1.0).clamp(min=0.0)
        b = int(batch)
        if b < 0 or b >= self._priming_B:
            return None
        return (self._priming[b, :live] - 1.0).clamp(min=0.0)

    def topk_heat(self, k, batch=0, rows=None):
        """Return ref-ids (LongTensor) of the up-to-k hottest LIVE rows.

        Only rows with r > 0 are eligible.  Result is sorted by heat
        descending.  ``rows`` (optional LongTensor of ref-ids) restricts
        the candidate set to those ids (still intersected with live +
        r > 0).  Returns an empty LongTensor when ``_priming`` is None,
        ``k <= 0``, or nothing is hot.

        See plan
        ``doc/plans/2026-06-06-symbolic-heat-retrieval.md`` §API additions.
        """
        import torch
        if self._priming is None or int(k) <= 0:
            return torch.zeros(0, dtype=torch.long)
        empty = torch.zeros(0, dtype=torch.long, device=self._priming.device)
        b = int(batch)
        if b < 0 or b >= self._priming_B:
            return empty
        live = self._priming_live or self._priming_capacity
        r = (self._priming[b, :live] - 1.0).clamp(min=0.0)
        # Candidate ids: arange(live) filtered to r > 0.
        cand_ids = torch.where(r > 0)[0]           # LongTensor of live indices
        if rows is not None:
            # Restrict to the caller-supplied ids that are also live.
            rows_t = torch.as_tensor(rows, dtype=torch.long, device=self._priming.device)
            valid_rows = rows_t[(rows_t >= 0) & (rows_t < live)]
            # Intersect: keep only cand_ids that appear in valid_rows.
            mask = torch.isin(cand_ids, valid_rows)
            cand_ids = cand_ids[mask]
        if cand_ids.numel() == 0:
            return empty
        cand_r = r[cand_ids]
        actual_k = min(int(k), cand_ids.numel())
        topk_vals, topk_local = torch.topk(cand_r, actual_k, largest=True, sorted=True)
        return cand_ids[topk_local]

    def _active_heat_set(self, batch, rows, topk):
        """Private helper: compute (ids, r_vals) for the active hot set.

        ``ids``   — LongTensor of ref-ids in the active set S.
        ``r_vals``— FloatTensor of heat values at those ids.
        Returns ``(ids, r_vals)``; both tensors are on the priming
        buffer's device.

        Used internally by ``build_semantic_heat`` and ``build_outer_heat``
        to avoid logic duplication.
        Plan ``doc/plans/2026-06-06-symbolic-heat-retrieval.md`` §Core
        representation.
        """
        import torch
        b = int(batch)
        if b < 0 or b >= self._priming_B:
            empty_ids = torch.zeros(0, dtype=torch.long, device=self._priming.device)
            empty_r   = torch.zeros(0, dtype=self._priming.dtype, device=self._priming.device)
            return empty_ids, empty_r
        device = self._priming.device
        live = self._priming_live or self._priming_capacity
        r = (self._priming[b, :live] - 1.0).clamp(min=0.0)
        # Start with all hot live ids.
        hot_ids = torch.where(r > 0)[0]
        if rows is not None:
            rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
            valid_rows = rows_t[(rows_t >= 0) & (rows_t < live)]
            mask = torch.isin(hot_ids, valid_rows)
            hot_ids = hot_ids[mask]
        if topk is not None and int(topk) > 0 and hot_ids.numel() > 0:
            actual_k = min(int(topk), hot_ids.numel())
            cand_r = r[hot_ids]
            _, local_top = torch.topk(cand_r, actual_k, largest=True, sorted=True)
            hot_ids = hot_ids[local_top]
        r_vals = r[hot_ids]
        return hot_ids, r_vals

    def build_semantic_heat(self, codebook_rows, batch=0, rows=None, topk=None):
        """Return z = A_S^T r_S, shape [D].

        S = live rows with r > 0 for ``batch``, optionally restricted to
        ``rows`` and/or limited to the top-``topk`` hottest (via
        ``topk_heat`` logic, delegated to ``_active_heat_set``).
        A_S = ``codebook_rows[S]``; r_S = heat at S.
        z = r_S @ A_S.

        Returns ``zeros([D])`` when S is empty or ``_priming`` is None.
        Preserves ``codebook_rows.device`` and ``codebook_rows.dtype``.

        Plan ``doc/plans/2026-06-06-symbolic-heat-retrieval.md``
        §Core representation, §API additions.
        """
        import torch
        D = codebook_rows.shape[1]
        zero = torch.zeros(D, device=codebook_rows.device,
                           dtype=codebook_rows.dtype)
        if self._priming is None:
            return zero
        b = int(batch)
        if b < 0 or b >= self._priming_B:
            return zero
        ids, r_vals = self._active_heat_set(batch, rows, topk)
        if ids.numel() == 0:
            return zero
        # Move ids/r_vals to codebook device; cast r_vals to codebook dtype.
        ids = ids.to(device=codebook_rows.device)
        r_vals = r_vals.to(device=codebook_rows.device,
                           dtype=codebook_rows.dtype)
        A_S = codebook_rows[ids]       # [|S|, D]
        return r_vals @ A_S            # [D]

    def build_outer_heat(self, codebook_rows, batch=0, rows=None, topk=None,
                         low_rank=True):
        """Return the active outer-product factor(s) for S.

        U = diag(sqrt(r_S)) @ A_S = sqrt(r_S)[:, None] * A_S, shape [|S|, D].

        ``low_rank=True``  -> return U  (shape [|S|, D]).
        ``low_rank=False`` -> return dense C = U^T @ U  (shape [D, D]).

        Empty S:
          ``low_rank=True``  -> [0, D]
          ``low_rank=False`` -> zeros([D, D])

        Preserves ``codebook_rows.device`` and ``codebook_rows.dtype``.

        Plan ``doc/plans/2026-06-06-symbolic-heat-retrieval.md``
        §Core representation (``low-rank`` mode), §API additions.
        """
        import torch
        D = codebook_rows.shape[1]
        if self._priming is None or int(batch) < 0 or int(batch) >= self._priming_B:
            if low_rank:
                return torch.zeros(0, D, device=codebook_rows.device,
                                   dtype=codebook_rows.dtype)
            return torch.zeros(D, D, device=codebook_rows.device,
                               dtype=codebook_rows.dtype)
        ids, r_vals = self._active_heat_set(batch, rows, topk)
        if ids.numel() == 0:
            if low_rank:
                return torch.zeros(0, D, device=codebook_rows.device,
                                   dtype=codebook_rows.dtype)
            return torch.zeros(D, D, device=codebook_rows.device,
                               dtype=codebook_rows.dtype)
        ids = ids.to(device=codebook_rows.device)
        r_vals = r_vals.to(device=codebook_rows.device,
                           dtype=codebook_rows.dtype)
        A_S = codebook_rows[ids]                          # [|S|, D]
        sqrt_r = r_vals.sqrt()[:, None]                   # [|S|, 1]
        U = sqrt_r * A_S                                   # [|S|, D]
        if low_rank:
            return U
        return U.t() @ U                                   # [D, D]

class ObjectSubSpace(nn.Module):
    """Durable PartSpace meronymic-analysis carrier -- the PS
    analogue of :class:`SymbolSubSpace`.

    doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md
    ("Carrier State"): SymbolSubSpace stores taxonomic state for symbolic
    parsing; ObjectSubSpace stores meronymic state for perceptual
    analysis -- spans, part ids, parent / child links, route ids /
    scores, depth, and the replay metadata ``reverse()`` needs to
    re-realize the surface. It is a durable state HOLDER, not a parser:
    trainable routing modules live on the LanguageLayer-like router, and
    the transient stack-mode SubSpace view (``.what`` / ``.where`` /
    ``.activation``) used while invoking the shared router is a separate
    adapter (Phase 5), not this carrier.

    All tensors keep fixed physical capacity; the live count is
    ``_depth`` and :meth:`live_mask` derives which slots are active --
    exactly as SymbolSubSpace's typed STM does. ``push`` / ``pop`` /
    ``update`` keep every parallel buffer in sync.

    Durable buffers (row ``b``, slot ``d`` in ``[0, _depth[b])``)::

        _buffer       [B, cap, percept_dim]  PS span / part vector
        _part_id      [B, cap]   PS codebook row id, or -1 (byte fallback)
        _span_start   [B, cap]   inclusive byte/atom start, or -1
        _span_end     [B, cap]   exclusive byte/atom end, or -1
        _span_where   [B, cap, 2] endpoint-sum spatial key phase(s)+phase(e)
        _parent_id    [B, cap]   derivation parent slot, or -1
        _left_id      [B, cap]   left child slot, or -1
        _right_id     [B, cap]   right child slot, or -1
        _route_id     [B, cap]   selected meronymic operation id, or -1
        _route_score  [B, cap]   local route confidence / score
        _depth        [B]        logical live depth

    ``_route_id`` is PS-only: it is the meronymic route used for surface
    replay. The WS operator identity lives in ``.what`` slot 0 (see the
    Phase-2 contract), not here.

    Marker-route metadata (absorb/emit replay -- "route-metadata on
    ObjectSubSpace" in the codification)::

        _marker_ps_id    [B, cap]    bound marker PS row id, or -1
        _marker_span     [B, cap, 2] marker sub-span endpoint-sum key
        _order_bit       [B, cap]    recorded order (0=id, 1=swap) for T3
        _marker_position [B, cap]    PRE/INFIX/SUF/CIRCUM code, or -1
    """

    # Marker-position codes for _marker_position; -1 == unset/marker-free.
    MARKER_POS = {'PRE': 0, 'INFIX': 1, 'SUF': 2, 'CIRCUM': 3}

    # field name -> (buffer attr, kind) for the generic update/clear paths.
    # kind: 'vec' (the payload), 'long', 'float', 'pair' (a 2-vector).
    _LONG_FIELDS = (
        'part_id', 'span_start', 'span_end', 'parent_id', 'left_id',
        'right_id', 'route_id', 'marker_ps_id', 'order_bit',
        'marker_position')
    _FLOAT_FIELDS = ('route_score',)
    _PAIR_FIELDS = ('span_where', 'marker_span')
    _LONG_DEFAULTS = {
        'part_id': -1, 'span_start': -1, 'span_end': -1, 'parent_id': -1,
        'left_id': -1, 'right_id': -1, 'route_id': -1, 'marker_ps_id': -1,
        'order_bit': 0, 'marker_position': -1}

    def __init__(self, percept_dim, capacity=8, batch=1):
        """Allocate the fixed-capacity parallel buffers; see class docstring."""
        super().__init__()
        self.percept_dim = int(percept_dim)
        self.capacity = int(capacity)
        self.max_depth = int(capacity)
        cap, dim_p, B = self.capacity, self.percept_dim, int(batch)

        def reg_long(name, fill):
            self.register_buffer(
                name, torch.full((B, cap), fill, dtype=torch.long),
                persistent=False)

        self.register_buffer(
            '_buffer', torch.zeros(B, cap, dim_p), persistent=False)
        for fld in self._LONG_FIELDS:
            reg_long('_' + fld, self._LONG_DEFAULTS[fld])
        self.register_buffer('_route_score', torch.zeros(B, cap),
                             persistent=False)
        self.register_buffer('_span_where', torch.zeros(B, cap, 2),
                             persistent=False)
        self.register_buffer('_marker_span', torch.zeros(B, cap, 2),
                             persistent=False)
        self.register_buffer('_depth', torch.zeros(B, dtype=torch.long),
                             persistent=False)

    @property
    def batch(self):
        """Physical row count (grown by :meth:`ensure_batch`)."""
        return self._buffer.shape[0]

    def depth(self, b):
        """Live span count on row ``b``."""
        return int(self._depth[b].item())

    def live_mask(self):
        """``[B, cap]`` bool mask: True for live slots (slot < _depth[b])."""
        idx = torch.arange(self.capacity, device=self._depth.device)
        return idx.unsqueeze(0) < self._depth.unsqueeze(1)

    def push(self, b, vec, *, part_id=-1, span_start=-1, span_end=-1,
             span_where=None, parent_id=-1, left_id=-1, right_id=-1,
             route_id=-1, route_score=0.0, marker_ps_id=-1,
             marker_span=None, order_bit=0, marker_position=-1):
        """Append one meronymic span to row ``b``; keep every parallel
        buffer in sync; increment ``_depth``. Returns the slot written.
        Raises ``AssertionError`` on overflow past ``max_depth``.
        """
        d = int(self._depth[b].item())
        if d >= self.max_depth:
            raise AssertionError(
                f"ObjectSubSpace overflow at row {b}: "
                f"max_depth={self.max_depth}")
        self._buffer[b, d] = vec.to(
            device=self._buffer.device, dtype=self._buffer.dtype)
        self._part_id[b, d] = int(part_id)
        self._span_start[b, d] = int(span_start)
        self._span_end[b, d] = int(span_end)
        self._parent_id[b, d] = int(parent_id)
        self._left_id[b, d] = int(left_id)
        self._right_id[b, d] = int(right_id)
        self._route_id[b, d] = int(route_id)
        self._route_score[b, d] = float(route_score)
        self._marker_ps_id[b, d] = int(marker_ps_id)
        self._order_bit[b, d] = int(order_bit)
        self._marker_position[b, d] = int(marker_position)
        if span_where is not None:
            self._span_where[b, d] = torch.as_tensor(
                span_where, device=self._span_where.device,
                dtype=self._span_where.dtype)
        if marker_span is not None:
            self._marker_span[b, d] = torch.as_tensor(
                marker_span, device=self._marker_span.device,
                dtype=self._marker_span.dtype)
        self._depth[b] = d + 1
        return d

    def update(self, b, slot, **fields):
        """Update fields of an existing live slot WITHOUT changing depth.

        Used by the analyzer to write the chosen route id / child links /
        route metadata back after a route is selected. ``vec`` updates the
        payload; any of the long / float / pair field names is accepted.
        Unknown field names raise ``KeyError``; a non-live slot raises
        ``IndexError``.
        """
        d = int(self._depth[b].item())
        if not (0 <= slot < d):
            raise IndexError(
                f"ObjectSubSpace.update: slot {slot} not live "
                f"(depth {d}) at row {b}")
        for name, value in fields.items():
            if name == 'vec':
                self._buffer[b, slot] = value.to(
                    device=self._buffer.device, dtype=self._buffer.dtype)
            elif name in self._LONG_FIELDS:
                getattr(self, '_' + name)[b, slot] = int(value)
            elif name in self._FLOAT_FIELDS:
                getattr(self, '_' + name)[b, slot] = float(value)
            elif name in self._PAIR_FIELDS:
                buf = getattr(self, '_' + name)
                buf[b, slot] = torch.as_tensor(
                    value, device=buf.device, dtype=buf.dtype)
            else:
                raise KeyError(
                    f"ObjectSubSpace.update: unknown field {name!r}")

    def get(self, b, slot):
        """Return a dict snapshot of slot ``(b, slot)``'s parallel state."""
        out = {'vec': self._buffer[b, slot].clone()}
        for fld in self._LONG_FIELDS:
            out[fld] = int(getattr(self, '_' + fld)[b, slot].item())
        out['route_score'] = float(self._route_score[b, slot].item())
        out['span_where'] = self._span_where[b, slot].clone()
        out['marker_span'] = self._marker_span[b, slot].clone()
        return out

    def top(self, b, k=1):
        """Peek the k-th span from the top (k=1 is the most recent)."""
        d = int(self._depth[b].item())
        if d < k:
            raise AssertionError(
                f"ObjectSubSpace.top: row {b} has {d} spans, asked k={k}")
        return self.get(b, d - k)

    def pop(self, b):
        """Pop the top span from row ``b``, clear its slot, decrement
        ``_depth``, and return its snapshot dict."""
        d = int(self._depth[b].item())
        if d <= 0:
            raise AssertionError(
                f"ObjectSubSpace underflow at row {b}: stack is empty")
        slot = d - 1
        out = self.get(b, slot)
        self._clear_slot(b, slot)
        self._depth[b] = slot
        return out

    def _clear_slot(self, b, slot):
        """Reset one slot's parallel buffers to their defaults."""
        self._buffer[b, slot] = 0
        for fld in self._LONG_FIELDS:
            getattr(self, '_' + fld)[b, slot] = self._LONG_DEFAULTS[fld]
        self._route_score[b, slot] = 0
        self._span_where[b, slot] = 0
        self._marker_span[b, slot] = 0

    def clear(self, b=None):
        """Reset row ``b`` (or all rows when ``None``) to empty."""
        rows = range(self.batch) if b is None else [b]
        for r in rows:
            for slot in range(self.capacity):
                self._clear_slot(r, slot)
            self._depth[r] = 0

    def ensure_batch(self, batch):
        """Grow the row dimension to ``batch``, preserving live state in
        existing rows (fresh rows start empty). Mirrors
        ``SymbolSubSpace._ensure_stm_batch``."""
        batch = int(batch)
        prev = self._buffer.shape[0]
        if batch <= prev:
            return
        dev = self._buffer.device

        def grow(buf, fill):
            new = torch.full(
                (batch,) + tuple(buf.shape[1:]), fill,
                dtype=buf.dtype, device=dev)
            new[:prev] = buf
            return new

        self._buffer = grow(self._buffer, 0)
        for fld in self._LONG_FIELDS:
            setattr(self, '_' + fld,
                    grow(getattr(self, '_' + fld), self._LONG_DEFAULTS[fld]))
        self._route_score = grow(self._route_score, 0)
        self._span_where = grow(self._span_where, 0)
        self._marker_span = grow(self._marker_span, 0)
        self._depth = grow(self._depth, 0)


class SymbolSubSpace(SubSpace):
    """Per-sentence grammar / serial-processing carrier — the third
    argument that travels alongside the data SubSpaces through the
    pipeline (reached via ``subspace.symbolSpace`` after
    ``copy_context`` stamps the back-reference).

    Runtime-parallel to PartSpace / ConceptualSpace / WholeSpace
    but functionally a composition dispatcher rather than a pipeline
    stage that produces data tensors. SymbolSubSpace owns:

      * the per-space_role ``SyntacticLayer`` dispatchers (registered on
        each home space; reached via ``forwardSymbols`` /
        ``reverseSymbols``);
      * the CKY chart and truth store;
      * the per-sentence parser cursor (``self.cursor`` — Python ints,
        one per space_role) and PartSpace recurrent-pass index
        (``self.recur_pass`` — Python int);
      * the **typed STM stack** (payload frames + per-frame category /
        order / ref_id metadata) — formerly held by ``TypedStack`` at
        ``ConceptualSpace._stm_typed``; the ``ShortTermMemory`` Layer
        on ``ConceptualSpace`` reads/writes these buffers (Phase D of
        doc/specs/2026-05-21-wordsubspace-stm-layer-refactor.md);
      * inter-sentence discourse substrate (``InterSentenceLayer`` /
        priming taxonomy).

    The standalone ``SentenceState`` carrier was retired (2026-05-21):
    ``cursor`` and ``recur_pass`` now live directly on SymbolSubSpace;
    the cross-pass C→P / C→S feedback is read straight off
    ``ConceptualSpace._subspaceForPS`` / ``_subspaceForWS`` (the
    persistent CS-space_role storage that ``ConceptualSpace.forward``
    mutates in place).

    Real ``SubSpace`` subclass (2026-05-21 SymbolSubSpace/STM Layer
    refactor): SymbolSubSpace IS the data carrier the STM driver acts on.
    It inherits the SubSpace slot machinery but is not a pipeline
    ``Space`` and does not produce data tensors of its own; the inherited
    ``.event`` / ``.what`` / ``.where`` / ``.when`` slots stay empty,
    while the parallel-tensor typed STM stack (``_buffer`` / ``_category``
    / ``_order`` / ``_ref_id`` / ``_depth``) carries the parse state.
    """

    name = "Words"
    config_section = "SymbolSpace"

    def __init__(self, perceptualSpace, conceptualSpace, wholeSpace,
                 nPercepts, nConcepts, nSymbols,
                 concept_dim, symbol_dim):
        """Build the chart, grammar layer, truth store, per-space_role dispatch,
        and the typed STM stack data.

        SymbolSpace IS a SubSpace (2026-05-21 refactor) and bypasses the
        Space factory because its construction crosses space_role boundaries
        (it needs references to Perceptual / Conceptual / Symbolic
        spaces). Detects the default-only grammar case so compose /
        generate can skip the CKY pass entirely. Mutates ``self`` to
        install ``syntacticLayer``, ``chart``, ``truthLayer``, the
        host_layer registry, per-row buffers, and the typed STM stack.
        """
        # 1. Mirror WholeSpace's column layout for the Space-contract
        # fields that downstream callers occasionally read off SymbolSpace
        # (``nDim`` / ``nWhat`` / ``nWhere`` / ``nWhen`` / ``muxedSize``).
        sub = wholeSpace.subspace
        nWhere = int(getattr(sub, 'nWhere', 0) or 0)
        nWhen  = int(getattr(sub, 'nWhen',  0) or 0)
        nWhat  = int(getattr(sub, 'nWhat',  0) or 0)
        muxed  = int(getattr(sub, 'muxedSize', nWhat + nWhere + nWhen)
                     or (nWhat + nWhere + nWhen))

        # 2. Initialise as a real SubSpace. The slot Bases stay empty
        # — SymbolSubSpace is a data carrier (typed STM stack), not a
        # pipeline space that produces tensors via ``.what`` / ``.event``.
        # Pass encodings sized to mirror WholeSpace's band so encoding
        # nDim == self.nWhen / self.nWhere downstream readers expect (the
        # 2026-06-06 uniform-band convention gives WS (2, 2) so the
        # default WhereEncoding(0,0) / WhenEncoding(0,0) would drift).
        from Spaces import WhereEncoding as _WhereEncoding
        from Spaces import WhenEncoding as _WhenEncoding
        SubSpace.__init__(
            self,
            inputShape=[0, muxed], outputShape=[0, muxed],
            nInputDim=muxed, nOutputDim=muxed,
            whereEncoding=_WhereEncoding(0, nWhere, nWhen) if nWhere else None,
            whenEncoding=_WhenEncoding(n_when=nWhen) if nWhen else None,
        )
        # Restamp nWhat / nWhere / nWhen to mirror WholeSpace's column
        # layout (downstream callers read these to size projections).
        self.nWhat = nWhat
        self.nWhere = nWhere
        self.nWhen = nWhen
        self.muxedSize = muxed
        self.nDim = muxed
        self.spaceShape = [0, muxed]

        # Back-references to the three Spaces. Used post-2026-05-12 by
        # the grammar's lift/lower wiring to pass perceptual / conceptual
        # references to LiftLayer / LowerLayer at construction time, so
        # those layers can route the substrate sigma/pi after gating.
        # ``object.__setattr__`` bypasses nn.Module's submodule
        # registration so we don't create cycles (each Space is already
        # a direct child of the Model).
        object.__setattr__(self, 'perceptualSpace', perceptualSpace)
        object.__setattr__(self, 'conceptualSpace', conceptualSpace)
        object.__setattr__(self, 'wholeSpace', wholeSpace)

        # 3. Grammar must be configured before any SyntacticLayer
        # construction can resolve rule sets / transition rules.
        TheGrammar._configured = False
        TheGrammar._ensure_configured()
        grammar = TheGrammar

        # 3-op. Insert the grammar's operations into the WholeSpace
        # codebook so the operator-prefixed parse tree's operation nodes
        # are codebook-resolvable (doc/plans/2026-05-30-subsymbolic-
        # analyzer-terminal-emitter.md, Phase 2 amended 2026-06-02): every
        # node of the prefixed syntactic tree (operations + terminal
        # symbols) exists in the codebook, while the computed ideas live in
        # STM. Operators are tagged "op" (distinct from meaning-bearing
        # symbols) and never written into the STM idea space. No-op when
        # the WS .what is not a Codebook.
        if wholeSpace is not None:
            wholeSpace.insert_operations(grammar)

        # 3a. Detect the default-only case (every operational rule is
        # a unary substrate fold registered as the per-space_role default).
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

        # 4. Space-contract fields. SubSpace.__init__ already set
        # ``self.symbolSpace = None``; the rest are SymbolSubSpace-specific.
        self.layers = nn.ModuleList()
        self.params = []

        # Slice B / conceptualize (Alec 2026-06-21): the SYMBOL CODEBOOK on this
        # carrier's ``.what``. When symbol_tower is on, ``SS.subspace.what`` holds
        # the symbol codebook (symbols 1:1 with concepts); CS reads it as the SS
        # bind leg (tower-symmetric with PS/WS ``.what``). The other slot Bases
        # stay empty -- this object is also the grammar/STM stack, which rides on
        # ``.event``, not ``.what``, so the codebook does not disturb it. Default
        # off -> ``.what`` stays empty -> byte-identical.
        if bool(TheXMLConfig.get("architecture.symbolTower", default=False)):
            from Spaces import Codebook as _Codebook
            _sym_cb = _Codebook()
            _sym_cb.use_dot_product = False
            _sym_cb.create(1, int(nSymbols), int(symbol_dim), customVQ=True)
            self.what = _sym_cb
            for _p in _sym_cb.parameters():
                if all(_p is not _q for _q in self.params):
                    self.params.append(_p)

        # 4. Per-space SyntacticLayer dispatch lives on each space
        # (``space.syntacticLayer``); SymbolSpace itself no longer owns a
        # central SyntacticLayer instance.
        # 4a. Chart + host-layer registry. Per the 2026-05-01 syntactic-
        # layer refactor (doc/specs/2026-05-01-syntactic-layer-refactor.md):
        # SymbolSpace owns a Chart that runs CKY inside / outside passes
        # and writes per-(space_role, step) rule selections into
        # ``current_rules`` / ``generate_rules``. Each per-space
        # SyntacticLayer registers its parametrized layers via
        # ``register_host_layer``; the chart consults
        # ``host_layer(space_role, rule_name)`` to fire host-owned folds.
        self._host_layer_registry = {}
        # Initialize current_rules / generate_rules from the grammar
        # XML's per-space_role natural folds. Per the 2026-05-07 rollback,
        # the grammar XML is the sole source of truth -- with no
        # ``default_rule`` code-level fallback the per-stage Spaces'
        # syntacticLayer dispatch must always have rules to fire.
        # ``compose`` / ``generate`` overwrite these on call.
        self.current_rules = self._default_compose_rules()
        self.generate_rules = self._default_generate_rules()
        # First-class routing decision (ADDITIVE companion to
        # ``current_rules``; never replaces it -- see ``RoutingState``).
        # Built in ``compose`` (both the default-only fast path and the
        # full-router path) right after ``current_rules`` is set. Carries
        # the dense ``[B, n_rules]`` ``rule_probs`` the intra-sentence
        # predictor consumes. Initialized here from the default fold so
        # ``routing_state`` is always a valid object (its ``rule_probs``
        # is None until the first ``compose`` with a known batch size).
        self.routing_state = self._build_routing_state(
            self.current_rules, batch_size=None)
        # Bumped on each compose / generate. Per-space SyntacticLayers
        # compare against this to know when to reset their per-space_role
        # cursor (Q10.1).
        self._compose_generation = 0
        self._generate_generation = 0
        # Per-sentence serial-parser state, owned directly by SymbolSpace
        # (no separate SentenceState carrier).
        #   ``cursor``      — ``list[int]`` of length 3 (one per space_role
        #                     subsymbolic/CS/SS), the per-space_role rule cursor consumed
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
        #   ``recur_pass``  — Python int, PartSpace recurrent-pass
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

        # Stage 3 (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):
        # the signal router (``LanguageLayer``) is the canonical parser.
        # The CKY ``Chart`` class and the STM shift-reduce path retire
        # here. ``self.languageLayer`` is constructed directly on the
        # SymbolSubSpace -- the chart's lazy ``_ensure_signal_router``
        # indirection is gone.
        _assert_retired_chart_knobs_absent()
        chart_hidden = self._resolve_hidden_dim(nSymbols)
        try:
            _signal_temperature = float(TheXMLConfig.get(
                "SymbolSpace.signal.temperature", 1.0))
        except Exception:
            _signal_temperature = 1.0
        self.languageLayer = LanguageLayer(
            n_input=nSymbols, n_output=nSymbols,
            hidden_dim=chart_hidden,
            feature_dim=symbol_dim,
            max_depth=max(nSymbols - 1, 1),
            temperature=_signal_temperature,
        )
        # The signal router's grammar reference (read by per-rule gating
        # and the diagnostics that used to call ``chart.grammar``).
        self.languageLayer.grammar = grammar
        self.layers.append(self.languageLayer)
        for p in self.languageLayer.parameters():
            if all(p is not q for q in self.params):
                self.params.append(p)
        # Stage 3: the chart was the GrammarLayer ``_chart_authority``
        # gating per-rule firing via ``should_run_rule``. The chart is
        # retired; SymbolSubSpace itself now serves as the authority --
        # it owns the live grammar and exposes the same
        # ``register_grammar_layer`` / ``should_run_rule`` surface.
        self._registered_grammar_layers = []
        try:
            from Layers import GrammarLayer as _GrammarLayer
            _GrammarLayer.set_chart_authority(self)
        except Exception:
            pass

        # 5. Per-space SyntacticLayer attachment. The perceptual
        # and conceptual spaces also get a ``symbolSpace`` back-reference
        # so they can route through the shared buffer, but only the
        # symbolic space's compose() fires the chart.
        # Post-split: grammar's canonical home is S; the
        # SyntacticLayers at the subsymbolic and CS space_roles are retained
        # as backward-compat dispatchers that no-op for grammars omitting
        # per-space_role rules. They're not the architectural locus of grammar
        # after the split (SS is), but the mechanism stays in place so
        # legacy configs continue to function and so any future
        # subsymbolic/CS-space_role rule (e.g. for lift/lower at concept_dim) can
        # still fire through the chart's per-space_role dispatch.
        if perceptualSpace is not None:
            perceptualSpace.attach_symbolSpace(self)
            # subsymbolic-space_role SyntacticLayer retired (2026-05-18 CS/SS split): the
            # perceptual space no longer carries a chart-dispatched
            # SyntacticLayer. ``attach_symbolSpace`` (shared-buffer
            # back-ref) is unrelated wiring and is kept. The space_role='subsymbolic'
            # branch in ``_attach_per_space_syntactic_layer`` is now
            # unreached but left dead-safe.
        if conceptualSpace is not None:
            conceptualSpace.attach_symbolSpace(self)
            self._attach_per_space_syntactic_layer(
                conceptualSpace, space_role='CS')
        if wholeSpace is not None:
            wholeSpace.attach_symbolSpace(self)
            self._attach_per_space_syntactic_layer(
                wholeSpace, space_role='SS')

        # 5b. Signal-router grammar wiring. The LanguageLayer needs
        # explicit op modules attached to its per-space_role scorers before
        # compose() can fire. We wire from the host_layer registry
        # populated in step 5 above. Stage 3 (chart retirement): this
        # always runs -- the signal router is the canonical parser.
        self._wire_signal_router_grammar_ops()

        # 6. TruthLayer -- shared truth store for symbolic activations.
        # Lives on SymbolSpace so WholeSpace doesn't have to carry it
        # alongside its already heavy pi/sort/codebook machinery.
        try:
            max_truths = int(TheXMLConfig.get("SymbolSpace.truthMaxEntries"))
        except (KeyError, TypeError, ValueError):
            max_truths = 1024
        self.truth_layer = TruthLayer(symbol_dim, max_truths=max_truths)
        if self.truth_layer not in self.layers:
            self.layers.append(self.truth_layer)
        for p in self.truth_layer.parameters():
            if all(p is not q for q in self.params):
                self.params.append(p)

        # 6a-bis. The SECOND truth set (GrammarOpsPass §6; sign-off
        # 2026-06-11): relative truths — relations between ideas,
        # stored as UNCOLLAPSED (np1, vp, np2) triples and consumed
        # only by the reasoning loop. A sibling of the absolute store
        # by construction, so luminosity/coverage never has to mask.
        #
        # LTM consolidation (doc/specs/mereological-order-raising.md, Alec
        # 2026-06-18): when ``<ltmConsolidation>`` is on, the discourse LTM and
        # this RelativeTruthStore are COMBINED into ONE unified
        # ``TernaryTruthStore`` (``self.ltm_store``); the RTS is then RETIRED
        # (not constructed) so a consolidated config has a single reasoning
        # home. When the gate is OFF the legacy RelativeTruthStore is built
        # exactly as before (flag-off byte-identical -- the ~19 standalone-RTS
        # tests pass a store= explicitly and still construct the class). The
        # unified store rides at the FULL idea/event width (``muxed`` -- the
        # end-state payload width the InterSentenceLayer records), and slices
        # the CONTENT band (``symbol_dim`` -- what nearest_ws_row /
        # _conform_idea_vec reason over) on read.
        _ltm_on = bool(TheXMLConfig.get(
            "architecture.ltmConsolidation", default=False))
        # Capacity: reuse the discourse LTM knob (same source the
        # InterSentenceLayer reads), falling back to the TruthLayer cap so a
        # config that sets neither still gets a sensible bound.
        _ltm_cap = int(TheXMLConfig.space(
            "SymbolSpace", "ltmCapacity", default=max_truths) or max_truths)
        self.relative_store = None
        self.ltm_store = None
        if _ltm_on:
            # Assigned as an ATTRIBUTE (registers as a submodule -> its buffers
            # ride the state_dict) but deliberately NOT appended to
            # ``self.layers`` and NOT added to ``self.params``: it has only
            # buffers (no trainable params) and staying out of ``self.layers``
            # keeps it OUT of the Reset cascade (which iterates ``self.layers``
            # and only fires objects with a capital-``Reset``), so the
            # persistent LTM survives every document-boundary Reset.
            self.ltm_store = TernaryTruthStore(
                muxed, capacity=_ltm_cap, content_width=symbol_dim)
        else:
            self.relative_store = RelativeTruthStore(
                symbol_dim, max_triples=max_truths)

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
        # Category / part-of-speech codebook.  This is the SymbolSpace's
        # ONLY codebook -- distinct from WholeSpace's symbol-prototype
        # codebook on ``WholeSpace.subspace.what.W``.
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
        # symbolic codebook on WholeSpace.subspace.what.
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
        # the SymbolSpace singleton, reached at runtime via
        # ``vspace.symbolSpace.taxonomy``.
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
        # len(symbolic()) would be only the SS-space_role subset and would under-size
        # the output.
        #
        # Option A (per task notes): torch.nn stdlib Sequential with a Tanh
        # nonlinearity -- no new layer type added to Layers.py. Stash
        # in_features on the SymbolSpace because Sequential has no such attr.
        n_rules = len(TheGrammar.rule_table)
        self.n_rules = n_rules
        max_depth = int(nPercepts)
        # pos_dim already bound above (category_embedding / category_stack dim).
        rule_in_features = max_depth * pos_dim
        # When nPercepts=0 (minimal test configs with no PartSpace),
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
        # it off.  Shape knobs live under <SymbolSpace> (armaP, armaQ,
        # armaHiddenDim); the loss weight lives under
        # <architecture><training><armaScale> and is read by runBatch.
        # Contrastive cosine machinery retired 2026-05-14 alongside
        # <maskedPrediction>.
        self.discourse = None
        if bool(TheXMLConfig.training("sentencePrediction", False)):
            try:
                n_sym_rows = int(wholeSpace.outputShape[0])
            except (AttributeError, IndexError, TypeError):
                n_sym_rows = int(getattr(wholeSpace, 'nVectors', 0) or 0)
            if n_sym_rows > 0 and muxed > 0:
                arma_p = int(TheXMLConfig.space(
                    "SymbolSpace", "armaP", default=5) or 5)
                arma_q = int(TheXMLConfig.space(
                    "SymbolSpace", "armaQ", default=2) or 2)
                arma_hidden = TheXMLConfig.space(
                    "SymbolSpace", "armaHiddenDim", default=None)
                # LTM chain capacity (Task 7, plan §8) — same read
                # pattern as armaP/armaQ; bounds the per-row STM
                # end-state chain on the InterSentenceLayer.
                ltm_capacity = int(TheXMLConfig.space(
                    "SymbolSpace", "ltmCapacity", default=1024) or 1024)
                # Pre-existing minor: SymbolSubSpace IS a SubSpace (no
                # nested ``self.subspace``); use object.__getattribute__
                # to avoid nn.Module.__getattr__ raising on the missing
                # attribute, then fall through to the documented 256
                # default that ``getattr`` would have hit.
                _ss_sub = self.__dict__.get('subspace', None)
                self.discourse = InterSentenceLayer(
                    n_symbols=n_sym_rows,
                    max_depth=int(getattr(_ss_sub, 'max_depth', 256) or 256),
                    n_dim=muxed,
                    p=arma_p,
                    q=arma_q,
                    hidden_dim=(int(arma_hidden)
                                if arma_hidden is not None else None),
                    concept_dim=int(concept_dim),
                    ltm_capacity=ltm_capacity,
                )
                # L_inter weight (Task 8, plan §9): read the new
                # <architecture><training><interLossWeight> knob (default
                # 0.1, mirroring <intraLossWeight>) and gate the inter-level
                # predictor's loss accumulation with it. The model also reads
                # this weight (see BasicModel) to scale the consumed term.
                _inter_w = float(
                    TheXMLConfig.training("interLossWeight", 0.1) or 0.1)
                self.discourse.set_inter_loss_weight(_inter_w)
                # InfoNCE next-idea contrastive term (<interContrastiveWeight> /
                # <interContrastiveTemp>). 0.0 weight -> MSE-only (byte-identical).
                self.discourse.set_inter_contrastive(
                    float(TheXMLConfig.training("interContrastiveWeight", 0.0)
                          or 0.0),
                    float(TheXMLConfig.training("interContrastiveTemp", 0.1)
                          or 0.1))
                self.layers.append(self.discourse)
                # ``self.discourse.parameters()`` now also enumerates the
                # inter-level ``_inter_predictor`` (registered as a submodule
                # via attribute assignment) and ``cast``, so this single loop
                # exposes every InterSentenceLayer param to the optimizer.
                for p in self.discourse.parameters():
                    if all(p is not q for q in self.params):
                        self.params.append(p)
                # LTM consolidation FU3 (Change 2, 2026-06-18): when the
                # gate is on AND the unified ``ltm_store`` was built, WIRE the
                # discourse AR predictor to read its recency window from the
                # GLOBAL store instead of its per-row deque (and stop the
                # deque append at observe -- the Models observe site's
                # store-append is the single source). Gate OFF (or no
                # ltm_store) leaves ``_ltm_store=None`` -> the legacy deque
                # path, byte-identical.
                if _ltm_on and self.ltm_store is not None:
                    self.discourse._ltm_store = self.ltm_store
                    self.discourse._ltm_consolidation = True

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
        # derivation matches a configured Grammar start pattern. Outer
        # doc-streaming loop drains via drain_sentence_completed() after
        # each runBatch and dispatches soft_reset(batch=b). Host-side
        # list (no GPU sync); resized to B by ensure_microbatch.
        self._sentence_completed = [False] * self.batch

        # -- typed STM stack (Phase D of the 2026-05-21 SymbolSubSpace /
        # STM Layer refactor) ---------------------------------------------
        # The parallel-tensor stack carrying per-frame ``category`` /
        # ``order`` / ``ref_id`` metadata alongside the vector payload.
        # Formerly ``TypedStack`` lived at ``ConceptualSpace._stm_typed``;
        # it is now SymbolSubSpace's own data. The ``ShortTermMemory`` Layer
        # on ``ConceptualSpace`` (Phase E) reads / writes these buffers.
        # Capacity defaults to ConceptualSpace's stm_capacity (XML
        # ``<stmCapacity>`` -- see ConceptualSpace.__init__), else 8.
        try:
            stm_capacity = int(getattr(conceptualSpace, 'stm_capacity', 0))
        except (TypeError, ValueError):
            stm_capacity = 0
        if stm_capacity <= 0:
            stm_capacity = 8
        self._stm_capacity = int(stm_capacity)
        self._stm_payload_dim = int(concept_dim)
        # TypedStack-equivalent public attributes (mirror its old API
        # so legacy callers and tests still see ``.max_depth`` / ``.dim``
        # on the stack carrier).
        self.max_depth = self._stm_capacity
        self.dim = self._stm_payload_dim
        cap = self._stm_capacity
        dim_p = self._stm_payload_dim
        # Float payload + long-typed metadata, all sized to ``self.batch``
        # (which ``ensure_batch`` / ``ensure_microbatch`` grow on demand).
        self.register_buffer(
            '_buffer',
            torch.zeros(self.batch, cap, dim_p),
            persistent=False)
        self.register_buffer(
            '_category',
            torch.full((self.batch, cap), -1, dtype=torch.long),
            persistent=False)
        self.register_buffer(
            '_order',
            torch.zeros((self.batch, cap), dtype=torch.long),
            persistent=False)
        self.register_buffer(
            '_ref_id',
            torch.full((self.batch, cap), -1, dtype=torch.long),
            persistent=False)
        self.register_buffer(
            '_depth',
            torch.zeros(self.batch, dtype=torch.long),
            persistent=False)
        # Parallel host-side string-form category names. Populated when
        # ``push`` is given ``category_id_str``; left ``None`` when the
        # int id is the primary form.
        self._category_names = [
            [None] * cap for _ in range(self.batch)
        ]

        # -- legacy idea-stack buffers (Phase E completion of doc/specs/
        # 2026-05-21-wordsubspace-stm-layer-refactor.md) --------------------
        # The CKY-compose chart pushes unquantized CS-space_role activations
        # ("ideas") via the ``ShortTermMemory`` Layer; spec §"Removed
        # Public Surfaces" calls for that Layer to be data-free. The
        # idea-stack data therefore lives on SymbolSubSpace alongside the
        # typed STM (separate parallel buffers; the chart's push doesn't
        # carry typed metadata so they cannot share a row). The
        # ``ShortTermMemory`` Layer proxies ``push`` / ``peek`` /
        # ``snapshot`` / ``push_step`` / ``push_window_batch`` /
        # ``push_step_masked`` / ``size`` / ``is_full`` / ``is_empty`` /
        # ``clear`` / ``ensure_batch`` / ``ensure_capacity`` to the
        # ``_idea_*`` methods below via the back-reference attached at
        # ``SymbolSubSpace.__init__`` tail.
        self._idea_capacity = cap
        self._idea_max_depth_host = 0
        self.register_buffer(
            '_idea_buffer',
            torch.zeros(self.batch, cap, dim_p),
            persistent=False)
        self.register_buffer(
            '_idea_depth',
            torch.zeros(self.batch, dtype=torch.long),
            persistent=False)

        # Attach this SymbolSubSpace to conceptualSpace.stm so the
        # ShortTermMemory Layer can route its data-accessor methods to
        # our idea-stack buffers (Phase E completion).
        stm_layer = getattr(conceptualSpace, 'stm', None)
        if stm_layer is not None and hasattr(stm_layer, 'attach_word_subspace'):
            stm_layer.attach_word_subspace(self)

    # -- typed STM stack API (formerly TypedStack methods) -------------------
    # Mirrors the public surface of the retired ``typed_stack.TypedStack``.
    # Callers pass the row index ``b`` (host-side int); all four parallel
    # buffers stay in sync. The ``ShortTermMemory`` Layer on
    # ``ConceptualSpace`` invokes these via
    # ``ss.push(...)`` / ``ss.pop(b)`` / ``ss.top(b)`` etc.

    def conceptualize(self, order, part=None, whole=None,
                      word_parts=None, word_whole=None, key=None, parts=None,
                      concept_ids=None):
        """Form a concept on the ConceptualSpace symbol tables (Alec 2026-06-21).

        A concept is a FLEXIBLE combination of two percepts; this is the unified
        dispatch over the three orders (doc/old/2026-06-21-higher-order-symbolic-
        composition.md sections 2b / 4b / 4c). The SS subspace owns this method;
        the CS owns the ``_sym_*`` relation tables it mutates -- the duality
        (``CS.forward`` processes ``SS.subspace``). Host-side; the caller runs it
        in the eager island (``symbol_tower`` relaxes fullgraph).

          order 0 -> ``[part, whole]``      : ``relate(part, whole)`` -- one
                     part-percept tied to one whole-percept (constituents carry
                     ``.where`` / ``.when``).
          order 1 -> ``[object isa word]``  : ``create_word_object_meta(word_parts,
                     word_whole, key)`` -> ``(A=word, B=object, C=meta)``; the
                     constituents' ``.where`` / ``.when`` = 0 (abstract).
          order 2 -> higher-order object    : ``synthesize_higher_order(parts)`` --
                     collapse the over-collected many into one superset.
          order 3 -> sequence chain         : ``conceptualize_chain(concept_ids)`` --
                     a tail-recursive ``[whole, part]`` list (Gallistel unitization
                     of behavior) for learning indefinitely long sequences.

        Constituents are stored BY REFERENCE (codebook index, or ``('sym', id)``
        for sub-symbols) -- never duplicate codes (section 4c). Letters/bytes are
        snapped to the percept codebook by the caller; other data is referenced by
        its ``.where`` boundary, unsnapped. Returns the concept id (order 0/2) or
        the ``(A, B, C)`` triple (order 1), or ``None`` on missing inputs.
        """
        cs = getattr(self, 'conceptualSpace', None)
        if cs is None:
            return None
        if order == 0:
            if part is None or whole is None:
                return None
            return cs.relate(part, whole)
        if order == 1:
            if word_parts is None or word_whole is None:
                return None
            return cs.create_word_object_meta(word_parts, word_whole, key=key)
        if order == 2:
            if not parts:
                return None
            return cs.synthesize_higher_order(parts)
        if order >= 3:
            if not concept_ids:
                return None
            return cs.conceptualize_chain(concept_ids)
        return None

    def _commit_priming(self, b, ref_id):
        """Gated forward-commit heat update for a single committed ref.

        Primes ``ref_id`` and propagates the boost along the taxonomy
        adjacency for the freshly-committed word/percept/idea. Plan
        ``doc/plans/2026-06-06-symbolic-heat-retrieval.md`` §Forward-path
        responsibilities (word/percept commit; CS-space_role grammar reduction).

        TRAINING-PATH ZERO-COST GUARANTEE: the ``priming_enabled`` check
        (False by default — set from ``<symbolicPriming>`` via
        ``configure_priming``) short-circuits FIRST, before any
        ``prime``/``propagate``. ``propagate`` is the expensive host-side
        graph walk (``.item()`` per node); it must never run when the
        feature is off. With ``<symbolicPriming>`` absent/false this method
        is a guaranteed no-op (a single ``getattr`` + boolean test, no
        host sync, no tensor mutation).
        """
        tax = getattr(self, 'taxonomy', None)
        # Sentinel False: a missing/absent taxonomy attribute must default to
        # production-off (no forward heat).  Contrast with Taxonomy's class
        # constant DEFAULT_PRIMING_ENABLED=True, which is the historical gate
        # for bare Taxonomy() retrieval helpers — not for missing attributes.
        if (tax is not None and getattr(tax, 'priming_enabled', False)
                and tax._priming is not None and int(ref_id) >= 0):
            rid = int(ref_id)
            tax.prime([rid], batch=b, boost=tax.boost_initial)
            tax.propagate([rid], batch=b, depth=tax.priming_depth,
                          hop_decay=tax.hop_decay)

    def push(self, b, vec, *, category_id=None, category_id_str=None,
             order=0, ref_id=-1):
        """Push one frame onto row ``b``'s typed STM stack.

        ``category_id`` (int) and / or ``category_id_str`` (str) may be
        provided. At least one must be set. When only the string form
        is given, the int slot defaults to -1 and the integer-keyed
        admissibility paths can't be used until a codebook lookup fills
        in the id.
        """
        if category_id is None and category_id_str is None:
            raise ValueError(
                "SymbolSubSpace.push: provide category_id or category_id_str")
        d = int(self._depth[b].item())
        assert d < self.max_depth, (
            f"SymbolSubSpace STM overflow at row {b}: "
            f"max_depth={self.max_depth}")
        self._buffer[b, d] = vec.to(
            device=self._buffer.device, dtype=self._buffer.dtype)
        self._category[b, d] = (
            int(category_id) if category_id is not None else -1)
        self._order[b, d] = int(order)
        self._ref_id[b, d] = int(ref_id)
        self._category_names[b][d] = category_id_str
        self._depth[b] = d + 1
        # Forward-commit symbolic-priming heat (gated; no-op unless
        # <symbolicPriming> is enabled — guard is priming_enabled-FIRST).
        # Plan doc/plans/2026-06-06-symbolic-heat-retrieval.md §Word/percept
        # commit.
        self._commit_priming(b, ref_id)

    def pop(self, b):
        """Pop the top frame from row ``b`` and return its metadata.

        Returns a dict with ``payload``, ``category`` (int),
        ``category_str`` (Optional[str]), ``order``, ``ref_id``.
        """
        d = int(self._depth[b].item())
        assert d > 0, (
            f"SymbolSubSpace STM underflow at row {b}: stack is empty")
        top_slot = d - 1
        out = {
            'payload':  self._buffer[b, top_slot].clone(),
            'category': int(self._category[b, top_slot].item()),
            'category_str': self._category_names[b][top_slot],
            'order':    int(self._order[b, top_slot].item()),
            'ref_id':   int(self._ref_id[b, top_slot].item()),
        }
        self._buffer[b, top_slot] = 0
        self._category[b, top_slot] = -1
        self._order[b, top_slot] = 0
        self._ref_id[b, top_slot] = -1
        self._category_names[b][top_slot] = None
        self._depth[b] = top_slot
        return out

    def top(self, b, k=1):
        """Peek at the k-th frame from the top on row ``b`` without popping
        (k=1 is the most recent; k=2 the one beneath it; etc.).
        """
        d = int(self._depth[b].item())
        assert d >= k, (
            f"SymbolSubSpace.top: row {b} has {d} items, asked for k={k}")
        slot = d - k
        return {
            'payload':  self._buffer[b, slot].clone(),
            'category': int(self._category[b, slot].item()),
            'category_str': self._category_names[b][slot],
            'order':    int(self._order[b, slot].item()),
            'ref_id':   int(self._ref_id[b, slot].item()),
        }

    def reduce_admissibility(self, b, rule_signatures):
        """Build the admissibility mask for row ``b``'s current stack top.

        Reads the top two items (or top one for unary REDUCEs at
        depth==1) and matches against each rule signature via
        :func:`embed.admissibility_mask`. Returns a length-
        ``len(rule_signatures)`` ``BoolTensor``.

        Convention: with depth ``d``, the "left operand" is slot
        ``d-2`` (second from top) and the "right operand" is slot
        ``d-1`` (top). When ``d == 1``, only ``left`` is set --
        producing a unary admissibility check.
        """
        from embed import admissibility_mask as _admissibility_mask
        d = int(self._depth[b].item())
        if d == 0:
            return torch.zeros(len(rule_signatures), dtype=torch.bool)
        right_slot = d - 1
        left_slot = d - 2 if d >= 2 else d - 1
        if d == 1:
            return _admissibility_mask(
                rule_signatures,
                left_cat=self._category_names[b][left_slot]
                or str(int(self._category[b, left_slot].item())),
                left_order=int(self._order[b, left_slot].item()),
            )
        return _admissibility_mask(
            rule_signatures,
            left_cat=self._category_names[b][left_slot]
            or str(int(self._category[b, left_slot].item())),
            left_order=int(self._order[b, left_slot].item()),
            right_cat=self._category_names[b][right_slot]
            or str(int(self._category[b, right_slot].item())),
            right_order=int(self._order[b, right_slot].item()),
        )

    def _ensure_stm_batch(self, batch):
        """Grow the typed-STM row dimension to ``batch``, preserving
        existing live stack state. Called from ``ensure_batch`` so the
        STM (and the parallel idea-stack buffers) stays in lockstep
        with the rest of SymbolSubSpace's per-row buffers.
        """
        batch = int(batch)
        if batch <= self._buffer.shape[0]:
            return
        device = self._buffer.device
        cap = self._stm_capacity
        dim_p = self._stm_payload_dim
        prev = self._buffer.shape[0]
        new_buffer = torch.zeros(
            batch, cap, dim_p,
            dtype=self._buffer.dtype, device=device)
        new_category = torch.full(
            (batch, cap), -1,
            dtype=self._category.dtype, device=device)
        new_order = torch.zeros(
            (batch, cap),
            dtype=self._order.dtype, device=device)
        new_ref_id = torch.full(
            (batch, cap), -1,
            dtype=self._ref_id.dtype, device=device)
        new_depth = torch.zeros(
            batch, dtype=self._depth.dtype, device=device)
        new_buffer[:prev] = self._buffer
        new_category[:prev] = self._category
        new_order[:prev] = self._order
        new_ref_id[:prev] = self._ref_id
        new_depth[:prev] = self._depth
        self._buffer = new_buffer
        self._category = new_category
        self._order = new_order
        self._ref_id = new_ref_id
        self._depth = new_depth
        self._category_names.extend(
            [[None] * cap for _ in range(batch - prev)])
        # Mirror the resize onto the legacy idea-stack buffers (Phase E
        # completion of the 2026-05-21 refactor). Fresh allocation
        # zero-fills new rows; existing live state is preserved.
        idea_cap = self._idea_capacity
        new_idea_buf = torch.zeros(
            batch, idea_cap, dim_p,
            dtype=self._idea_buffer.dtype, device=device)
        new_idea_depth = torch.zeros(
            batch, dtype=self._idea_depth.dtype, device=device)
        new_idea_buf[:prev] = self._idea_buffer
        new_idea_depth[:prev] = self._idea_depth
        self._idea_buffer = new_idea_buf
        self._idea_depth = new_idea_depth

    # -- idea-stack methods (Phase E completion: chart's
    # ``ShortTermMemory.push`` / ``peek`` / etc. proxy here) ---------------

    def idea_push(self, b, idea):
        """Untyped push onto row ``b`` of the idea stack. Mirrors the
        retired ``ShortTermMemory.push(b, idea)``.

        Newest-at-slot-0 convention: the new idea lands at slot 0 and the
        existing occupants shift RIGHT (slots ``[0, depth)`` -> ``[1,
        depth+1)``). Overflow RAISES (capacity is managed by the caller /
        the rolling-window ``_stm_shift_and_push``; ``push`` itself is the
        strict-bound primitive).
        """
        depth = int(self._idea_depth[b].item())
        if depth >= self._idea_capacity:
            raise RuntimeError(
                f"SymbolSubSpace.idea_push: row {b} is at capacity "
                f"({self._idea_capacity}); reduce before pushing further.")
        if depth > 0:
            self._idea_buffer[b, 1:depth + 1] = self._idea_buffer[
                b, 0:depth].clone()
        self._idea_buffer[b, 0] = idea
        self._idea_depth[b] = depth + 1
        if depth + 1 > self._idea_max_depth_host:
            self._idea_max_depth_host = depth + 1

    def idea_push_step(self, ideas):
        """Vectorised single-step push: shape ``[B, D]``.

        Newest-at-slot-0: shift every row's stack RIGHT by one slot, then
        write the new idea to slot 0.
        """
        B, D = ideas.shape
        buf = self._idea_buffer
        cap = int(buf.shape[1])
        if cap > 1:
            buf[:, 1:cap] = buf[:, 0:cap - 1].clone()
        buf[:, 0] = ideas
        self._idea_depth = self._idea_depth + 1
        self._idea_max_depth_host = self._idea_max_depth_host + 1

    def idea_push_step_masked(self, ideas, gate_b_1):
        """Masked variant of ``idea_push_step`` (newest-at-slot-0).

        For gated rows, shift RIGHT and write the new idea at slot 0; for
        un-gated rows the buffer + depth are left unchanged.
        """
        B, D = ideas.shape
        buf = self._idea_buffer
        cap = int(buf.shape[1])
        gate_b = gate_b_1.view(B)
        gate_bool = gate_b.bool()
        gate_col = gate_bool.view(B, 1, 1)
        shifted = buf.clone()
        if cap > 1:
            shifted[:, 1:cap] = buf[:, 0:cap - 1]
        shifted[:, 0] = ideas
        self._idea_buffer = torch.where(gate_col, shifted, buf)
        self._idea_depth = self._idea_depth + gate_bool.long()

    def idea_push_window_batch(self, ideas):
        """Push ``W`` consecutive ideas onto every batch row in one shot.
        ``ideas`` shape ``[B, W, D]`` (position 0 is the OLDEST of the
        window, position ``W-1`` the newest).

        Newest-at-slot-0: shift the existing stack RIGHT by ``W`` and write
        the window REVERSED into slots ``[0, W)`` so the window's newest
        (position ``W-1``) lands at slot 0.
        """
        B, W, D = ideas.shape
        if W == 0:
            return
        buf = self._idea_buffer
        cap = int(buf.shape[1])
        if cap > W:
            buf[:, W:cap] = buf[:, 0:cap - W].clone()
        # Reverse the window along its position axis so the newest window
        # position (W-1) maps to slot 0.
        buf[:, 0:W] = torch.flip(ideas, dims=[1])
        self._idea_depth = self._idea_depth + W
        self._idea_max_depth_host = self._idea_max_depth_host + int(W)

    def idea_pop(self, b):
        """Pop and return the top idea for row ``b``, or ``None`` when empty.

        Newest-at-slot-0: the top (newest) is slot 0; pop it and shift the
        remaining occupants LEFT (slots ``[1, depth)`` -> ``[0, depth-1)``).
        """
        depth = int(self._idea_depth[b].item())
        if depth == 0:
            return None
        idea = self._idea_buffer[b, 0].clone()
        if depth > 1:
            self._idea_buffer[b, 0:depth - 1] = self._idea_buffer[
                b, 1:depth].clone()
        self._idea_buffer[b, depth - 1].zero_()
        self._idea_depth[b] = depth - 1
        return idea

    def idea_peek(self, b, n=0):
        """Return the ``n``-th item from top of row ``b``, or ``None``.

        Newest-at-slot-0: ``n`` counts back from the newest, so the n-th
        item lives directly at slot ``n``.
        """
        depth = int(self._idea_depth[b].item())
        if depth <= n:
            return None
        return self._idea_buffer[b, n]

    def idea_snapshot(self, detach=False):
        """Return ``[B, max_depth, D]`` slice of the live idea buffer.

        Newest-at-slot-0 convention: the returned slab is NEWEST-FIRST
        along the slot axis (slot 0 = most-recent idea; the oldest live
        slot is at index ``depth-1``). The slice bounds are unchanged.
        """
        B = int(self._idea_buffer.shape[0])
        if B == 0:
            return None
        max_depth = int(self._idea_max_depth_host)
        if max_depth == 0:
            return None
        if max_depth > self._idea_capacity:
            max_depth = self._idea_capacity
        snap = self._idea_buffer[:, :max_depth, :]
        if detach:
            snap = snap.detach().clone()
        else:
            snap = snap.clone()
        return snap

    def idea_size(self, b):
        """Current depth (number of occupied slots) for row ``b``."""
        return int(self._idea_depth[b].item())

    def idea_is_full(self, b):
        """True when row ``b`` is at capacity."""
        return self.idea_size(b) >= self._idea_capacity

    def idea_is_empty(self, b):
        """True when row ``b`` has no occupants."""
        return self.idea_size(b) == 0

    def idea_clear(self, b=None):
        """Clear row ``b`` (or all rows when ``b`` is ``None``)."""
        if b is None:
            self._idea_buffer.zero_()
            self._idea_depth.zero_()
            self._idea_max_depth_host = 0
            return
        b = int(b)
        if b < 0 or b >= int(self._idea_buffer.shape[0]):
            return
        self._idea_buffer[b].zero_()
        self._idea_depth[b] = 0
        self._idea_max_depth_host = int(self._idea_depth.max().item())

    def idea_ensure_capacity(self, capacity):
        """Grow the per-slot capacity to at least ``capacity`` (grow-only)."""
        capacity = int(capacity)
        if capacity <= self._idea_capacity:
            return
        device = self._idea_buffer.device
        B = int(self._idea_buffer.shape[0])
        new_buf = torch.zeros(
            B, capacity, self._stm_payload_dim, device=device,
            dtype=self._idea_buffer.dtype)
        old_cap = int(self._idea_buffer.shape[1])
        if old_cap > 0:
            new_buf[:, :old_cap, :] = self._idea_buffer
        self._idea_buffer = new_buf
        self._idea_capacity = capacity

    def idea_ensure_batch(self, batch):
        """Resize idea-stack buffers to ``batch`` rows (fresh allocation).

        Idempotent. Used by ``ShortTermMemory.ensure_batch`` (the chart's
        consumer surface), which delegates here.
        """
        batch = int(batch)
        if int(self._idea_buffer.shape[0]) == batch:
            return
        device = self._idea_buffer.device
        self._idea_buffer = torch.zeros(
            batch, self._idea_capacity, self._stm_payload_dim, device=device)
        self._idea_depth = torch.zeros(batch, dtype=torch.long, device=device)
        self._idea_max_depth_host = 0

    # -- WS-side constituent stack (MeronomySpec §6 rev 2026-06-10c/11;
    # MeronomyPlan Stage 7) -------------------------------------------
    # The serial-mode ANALYSIS workspace: symbolic constituents under
    # analysis, word codes at the leaves -- the dual of the PS-side
    # ``_idea_*`` stack (which is structurally unchanged and holds
    # semantic referent content). Same newest-at-slot-0 convention and
    # capacity discipline. Allocated lazily on first use: parallel mode
    # never touches it (the duals engage only in serial mode), so the
    # stack stays dark for every existing path.

    def _ensure_constituent_stack(self, code_dim=None):
        """Lazily allocate the constituent stack beside the idea stack."""
        if getattr(self, '_constituent_buffer', None) is not None:
            return
        B = int(self._idea_buffer.shape[0])
        D = int(code_dim if code_dim is not None else self._stm_payload_dim)
        cap = int(self._idea_capacity)
        device = self._idea_buffer.device
        self._constituent_buffer = torch.zeros(B, cap, D, device=device)
        self._constituent_depth = torch.zeros(
            B, dtype=torch.long, device=device)

    def constituent_depth_of(self, b):
        """Current analysis-stack depth for row ``b`` (0 when dark)."""
        if getattr(self, '_constituent_buffer', None) is None:
            return 0
        return int(self._constituent_depth[b].item())

    def constituent_peek(self, b, n=0):
        """The n-th most recent constituent (newest at 0)."""
        depth = self.constituent_depth_of(b)
        if n >= depth:
            raise IndexError(
                f"constituent_peek({b}, {n}): depth is {depth}")
        return self._constituent_buffer[b, n]

    def constituent_push(self, b, code):
        """One ANALYSIS write: push a symbolic constituent (newest at 0).

        Overflow raises, mirroring ``idea_push`` -- capacity is the
        workspace's Miller cap, managed by the caller.
        """
        code = torch.as_tensor(code)
        self._ensure_constituent_stack(code_dim=int(code.shape[-1]))
        depth = int(self._constituent_depth[b].item())
        cap = int(self._constituent_buffer.shape[1])
        if depth >= cap:
            raise RuntimeError(
                f"SymbolSubSpace.constituent_push: row {b} is at capacity "
                f"({cap}); split/shift before pushing further.")
        if depth > 0:
            self._constituent_buffer[b, 1:depth + 1] = \
                self._constituent_buffer[b, 0:depth].clone()
        self._constituent_buffer[b, 0] = code
        self._constituent_depth[b] = depth + 1

    def constituent_pop(self, b):
        """Pop and return the newest constituent (slot 0)."""
        depth = self.constituent_depth_of(b)
        if depth == 0:
            raise RuntimeError(
                f"SymbolSubSpace.constituent_pop: row {b} is empty")
        top = self._constituent_buffer[b, 0].clone()
        if depth > 1:
            self._constituent_buffer[b, 0:depth - 1] = \
                self._constituent_buffer[b, 1:depth].clone()
        self._constituent_buffer[b, depth - 1] = 0
        self._constituent_depth[b] = depth - 1
        return top

    def constituent_split(self, b, left, right):
        """The binary SPLIT move (the serial form of π; spec §6 rev c).

        Replaces the top constituent with its two parts -- ONE move,
        one stack written (the single-writer mutex counts this as the
        tick's move). ``left`` lands newest (slot 0), ``right`` beneath
        it, mirroring left-to-right analysis order.
        """
        self.constituent_pop(b)
        self.constituent_push(b, torch.as_tensor(right))
        self.constituent_push(b, torch.as_tensor(left))

    def constituent_clear(self):
        """Sentence-boundary reset of the analysis stack (idempotent)."""
        if getattr(self, '_constituent_buffer', None) is None:
            return
        self._constituent_buffer.zero_()
        self._constituent_depth.zero_()

    # -- knowledge-artifact attach -----------------------------------------
    # Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
    # §Phase 2 — Loaders. ``attach_knowledge(view)`` wires a loaded
    # ``embed.KnowledgeView`` into the SymbolSpace so downstream consumers
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
            # Master switch for symbolic-priming heat (forward working
            # memory). Default false => taxonomy.priming_enabled False =>
            # the gated forward-commit path (push / reduce) is a no-op with
            # zero training-path cost. Coerced to bool the same way
            # ``hasAttention`` is (the XML parser already yields a Python
            # bool for ``xs:boolean`` leaves). Plan
            # doc/plans/2026-06-06-symbolic-heat-retrieval.md §A.
            symbolic_priming = bool(
                TheXMLConfig.get("architecture.symbolicPriming",
                                 default=False))
            tax.configure_priming(priming_enabled=symbolic_priming)

    @property
    def knowledge(self):
        """The attached :class:`embed.KnowledgeView`, or ``None`` when
        ``attach_knowledge`` has not been called for this SymbolSpace
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

    def retrieval_candidates_for_slot(self, query, basis, category, order,
                                      batch=0, topk_content=64, topk_heat=64,
                                      *, alpha=1.0, beta=0.5,
                                      mode='primer', gamma=0.0, delta=0.0,
                                      outer_topk=32):
        """Heat+content candidate union and boosted row-weights for ONE
        inverse-recommender slot.  Returns a dict the caller maps onto
        ``left_*`` or ``right_*`` recommender kwargs.  Plan
        ``doc/plans/2026-06-06-symbolic-heat-retrieval.md`` §Candidate
        generation / §Recommender changes / §Phase 5.

        Parameters
        ----------
        query : Tensor
            Query vector.  Any shape ending in ``D`` (the codebook column
            dimension).  Extra leading dims are collapsed with
            ``.reshape(-1)[:D]``, so passing a ``[1, D]`` or ``[B, D]``
            slice is safe.
        basis : object
            Object with a ``getW()`` method returning the ``[K, D]``
            symbolic codebook.  ``K ≈ V_live`` (live ref rows).
        category : str
            Grammar/POS category name passed to
            ``KnowledgeView.refs_by_category``.
        order : int
            Conceptual order passed to ``KnowledgeView.refs_by_order``.
        batch : int
            Which priming batch row to use.  Default 0.
        topk_content : int
            How many content-nearest rows to include in the union candidate
            set.  Default 64.
        topk_heat : int
            How many hottest priming rows to include.  Default 64.
        alpha : float
            Weight on cosine similarity in the boosted priming exponent.
            Default 1.0.  See §Retrieval score.
        beta : float
            Weight on ``log1p(r_i)`` (taxonomic heat) in the exponent.
            Default 0.5.  Logarithmic scale prevents a hot ancestor from
            swamping content proximity.
        mode : str
            Retrieval mode (one of ``'off'``, ``'primer'``, ``'second-order'``,
            ``'low-rank'``).  Default ``'primer'``.  When ``mode`` is
            ``'off'`` or ``'primer'`` (or ``gamma == delta == 0``), NO carrier
            terms are added — weight is exactly
            ``exp(alpha*sim + beta*log1p(r))``, preserving byte-identical
            output with pre-Phase-5 behavior.  When ``mode`` is
            ``'second-order'`` or ``'low-rank'``, first- and second-order
            carrier contributions are added for candidate rows.
        gamma : float
            Weight on the first-order carrier term ``dot(a_i, z)`` where
            ``z = A_S^T r_S`` (plan §Retrieval score).  Default 0.0.
            Only active when mode is ``'second-order'`` or ``'low-rank'``.
        delta : float
            Weight on the second-order carrier term ``dot(a_i, Cq)`` where
            ``Cq`` is computed without materializing a ``[V,V]`` tensor
            (plan §Retrieval score).  Default 0.0.  For ``'low-rank'``
            mode: ``Cq = U^T(Uq)``; for ``'second-order'``: ``Cq = C_dense @ q``
            (only when D is small).  Only active when mode is
            ``'second-order'`` or ``'low-rank'``.
        outer_topk : int
            Top-k active rows for ``build_semantic_heat`` /
            ``build_outer_heat`` carrier builders.  Default 32.

        Returns
        -------
        dict
            ``{'rows': LongTensor, 'priming': FloatTensor[K],
               'diagnostics': {...}}`` — or ``{}`` on graceful fallback.

            * ``rows``     — candidate union (up to topk_content + topk_heat),
              further intersected with the typed admissible set when non-empty.
            * ``priming``  — ``[K]`` float weight tensor; 1.0 = identity for
              non-candidate rows; ``exp(alpha*sim + beta*log1p(r))`` for
              candidate rows (+ optional carrier terms when active).
              Drop-in for ``left_priming`` / ``right_priming``.
            * ``diagnostics`` — counts and fallback label for debugging.

        Notes
        -----
        Graceful fallback:  returns ``{}`` when ``self.knowledge`` is ``None``,
        ``basis`` is ``None`` or has no ``getW()``, or ``getW()`` returns
        ``None``.  This mirrors ``priming_kwargs_for_slots``'s ``{}`` fallback.

        Device consistency:  ``q`` is moved to ``A.device`` before any
        computation.  The returned ``rows`` and ``priming`` live on the same
        device as ``A``.

        No [V,V] allocation:  ``z`` and ``Cq`` are ``[D]`` vectors; carrier
        gathers are over the small candidate set ``C``.  A ``[V,V]`` dense
        matrix is NEVER formed, even in second-order mode (where ``C_mat`` is
        ``[D,D]``, not ``[V,V]``).

        Size of ``priming``:  sized to ``A.shape[0]`` (K), which equals
        ``V_live`` for a fully-allocated codebook.  If the taxonomy priming
        buffer has a different live count, the returned vector still has K
        elements (heat gathered from positions within-range, zeros elsewhere).
        The recommender truncates / pads ``priming`` to ``W.shape[0]`` at
        use time, consistent with ``priming_kwargs_for_slots``.
        """
        import torch
        import torch.nn.functional as F

        # --- Graceful fallback guards (mirror priming_kwargs_for_slots) ------
        view = self.knowledge
        if view is None:
            return {}
        if basis is None:
            return {}
        getW_fn = getattr(basis, 'getW', None)
        if getW_fn is None:
            return {}
        A = getW_fn()
        if A is None:
            return {}

        K, D = A.shape  # [K, D] codebook

        # --- Reduce query to a single [D] vector on A's device ---------------
        q = query
        if not isinstance(q, torch.Tensor):
            q = torch.as_tensor(q, dtype=A.dtype)
        q = q.reshape(-1)[:D].to(A.device, A.dtype)

        # --- Step 1: Content scores — cosine similarity [K] ------------------
        # Single matrix-vector proximity; NOT [V,V].
        sim = F.cosine_similarity(q.reshape(1, -1), A, dim=1)  # [K]
        k_c = min(topk_content, K)
        if k_c > 0:
            _, topk_idx = torch.topk(sim, k=k_c)
            C_content = topk_idx  # LongTensor [k_c]
        else:
            C_content = torch.empty(0, dtype=torch.long, device=A.device)

        # --- Step 2: Heat candidates -----------------------------------------
        tax = getattr(self, 'taxonomy', None)
        enabled = (tax is not None and getattr(tax, 'priming_enabled', True))
        if enabled and tax is not None:
            C_heat = tax.topk_heat(topk_heat, batch=batch)
            C_heat = C_heat.to(A.device)
        else:
            C_heat = torch.empty(0, dtype=torch.long, device=A.device)

        # --- Step 3: Typed admissible set ------------------------------------
        cat_rows = view.refs_by_category(category)
        ord_rows = view.refs_by_order(int(order))
        C_typed = _intersect_long_rows(cat_rows, ord_rows)
        if C_typed.numel() > 0:
            C_typed = C_typed.to(A.device)

        # --- Step 4: Union + typed mask + fallback ---------------------------
        # union = unique(C_content ∪ C_heat)
        if C_content.numel() > 0 and C_heat.numel() > 0:
            union = torch.unique(torch.cat([C_content, C_heat]))
        elif C_content.numel() > 0:
            union = C_content
        elif C_heat.numel() > 0:
            union = C_heat
        else:
            union = torch.empty(0, dtype=torch.long, device=A.device)

        fallback = 'none'
        if C_typed.numel() > 0:
            C = _intersect_long_rows(union, C_typed)
        else:
            C = union

        if C.numel() == 0:
            # Three-level fallback (plan §Candidate generation)
            if C_typed.numel() > 0:
                C = C_typed
                fallback = 'typed_only'
            elif C_content.numel() > 0:
                C = C_content
                fallback = 'content_only'
            else:
                C = torch.empty(0, dtype=torch.long, device=A.device)
                fallback = 'sentinel'

        # --- Step 5: Boosted row-weight vector [K] ---------------------------
        # Initialize to 1.0 (multiplicative identity; non-candidates unchanged)
        priming = torch.ones(K, dtype=torch.float32, device=A.device)

        if C.numel() > 0:
            # Gather heat mask (r = max(p - 1, 0)) for indexed rows in C
            if enabled and tax is not None:
                heat_vec = tax.heat_mask(batch=batch)  # [V_live] or None
            else:
                heat_vec = None

            # Gather cosine scores at candidate ids
            # C may contain ids >= K if taxonomy live > codebook rows; clip.
            valid_mask = C < K
            C_valid = C[valid_mask]
            if C_valid.numel() > 0:
                sim_C = sim[C_valid]   # [|C_valid|] content scores

                # Heat at each candidate: r_i = heat_vec[i] if available
                if heat_vec is not None and C_valid.numel() > 0:
                    in_range = C_valid < heat_vec.numel()
                    r_C = torch.zeros(C_valid.numel(),
                                      dtype=torch.float32, device=A.device)
                    if in_range.any():
                        r_C[in_range] = heat_vec[C_valid[in_range]].to(
                            dtype=torch.float32, device=A.device)
                else:
                    r_C = torch.zeros(C_valid.numel(),
                                      dtype=torch.float32, device=A.device)

                # Base exponent: exp(alpha * sim_i + beta * log1p(r_i))
                exponent = (alpha * sim_C.to(torch.float32)
                            + beta * torch.log1p(r_C))

                # --- Phase 5: optional carrier terms (plan §Phase 5) ---------
                # Only active when mode is 'second-order' or 'low-rank' AND
                # at least one of gamma/delta is non-zero.  When mode is
                # 'off' or 'primer', this block is entirely skipped and
                # the output is byte-identical to pre-Phase-5 behavior.
                # No [V,V] tensor is ever formed: z and Cq are [D] vectors;
                # all gathers are over C_valid (the small candidate set).
                use_carriers = (mode in ('second-order', 'low-rank')
                                and (gamma != 0.0 or delta != 0.0))
                if use_carriers:
                    k_outer = outer_topk if outer_topk > 0 else None

                    # gamma term: z = A_S^T r_S  (plan §Core representation)
                    if gamma != 0.0 and tax is not None:
                        z = tax.build_semantic_heat(
                            A, batch=batch, topk=k_outer)  # [D]
                        z = z.to(dtype=torch.float32, device=A.device)
                        carrier_z = (A[C_valid].to(dtype=torch.float32,
                                                    device=A.device)
                                     @ z)   # [|C_valid|]
                        exponent = exponent + gamma * carrier_z

                    # delta term: Cq via low-rank or dense path
                    if delta != 0.0 and tax is not None:
                        if mode == 'low-rank':
                            # Cq = U^T (U q)  — no [D,D] or [V,V] materialized
                            U = tax.build_outer_heat(
                                A, batch=batch, topk=k_outer,
                                low_rank=True)          # [k, D]
                            U = U.to(dtype=torch.float32, device=A.device)
                            q_f32 = q.to(dtype=torch.float32, device=A.device)
                            Cq = U.t() @ (U @ q_f32)   # [D]
                        else:
                            # 'second-order': dense C_mat [D, D] — only safe
                            # when D is small (plan: dense C only for small D)
                            C_mat = tax.build_outer_heat(
                                A, batch=batch, topk=k_outer,
                                low_rank=False)         # [D, D]
                            C_mat = C_mat.to(dtype=torch.float32,
                                             device=A.device)
                            q_f32 = q.to(dtype=torch.float32, device=A.device)
                            Cq = C_mat @ q_f32          # [D]
                        carrier_C = (A[C_valid].to(dtype=torch.float32,
                                                    device=A.device)
                                     @ Cq)              # [|C_valid|]
                        exponent = exponent + delta * carrier_C

                boost = torch.exp(exponent)
                priming[C_valid] = boost

        # --- Step 6: Return dict ---------------------------------------------
        diagnostics = {
            'n_content':    int(C_content.numel()),
            'n_heat':       int(C_heat.numel()),
            'n_typed':      int(C_typed.numel()),
            'n_candidates': int(C.numel()),
            'fallback':     fallback,
        }
        return {
            'rows':        C,
            'priming':     priming,
            'diagnostics': diagnostics,
        }

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
        if torch.compiler.is_exporting():
            # Host bookkeeping -- skip under torch.export. ``_svo_valid`` is not
            # a registered buffer, so it lifts as a CONSTANT and this in-place
            # ``zero_()`` trips executorch edge-lowering ("Constant ... is
            # mutated in the forward method. Pls register it as buffer"). The
            # mask only gates the NEXT cycle's get_last_svo; it never feeds the
            # exported head prediction, so skipping it during export is a no-op
            # on the traced output (normal/compiled runs are unaffected).
            return
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

        Legacy path kept so ``WholeSpace.forward`` (and its tests) can
        map an active-symbol pattern to an embedding row without knowing
        the grammar category up-front. New code that already has the
        category name should use ``category_lookup(name)`` instead.

        Args:
            active_symbols: 1-D tensor of shape [N], typically resolved
                activations from WholeSpace.resolve().

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
            "Grammar reconfigured after SymbolSpace construction; rule_predictor stale"
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
        """Run the signal router's compose pass; populate
        ``self.current_rules``.

        Idempotent within a forward pass: each per-space SyntacticLayer
        resets its per-space_role cursor to 0 and pops one rule per fold step.

        Fast paths (no router compose):
          * ``_grammar_is_default_only`` — every rule is the unary pi /
            sigma fold; rule selection is fully determined by the
            grammar XML so the router adds no information.

        Stage 3 (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):
        the CKY ``Chart`` and STM shift-reduce parsers retire; the
        signal router (``self.languageLayer``) is the canonical parser.
        The ``<parserBackend>`` and ``<routerKind>`` knobs that gated
        the legacy paths raise loudly at config load.

        WS-analysis / CS-execution split
        --------------------------------
        Conceptually this is the WS-side analysis stage: a soft
        superposition over the taxonymic codebook that selects, per
        space_role, a hard rule list (the returned ``current_rules`` dict,
        ``{space_role: [rule_id, ...]}``). The CS-side execution stage --
        actually applying the chosen reductions (lift / lower / union /
        intersection / swap / quantize / not) to the concept tensors --
        runs in ``ConceptualSpace.forward`` (and the WholeSpace
        stack-route path) and the per-space_role ``SyntacticLayer`` cursors
        during reverse. Only lift / lower / union / intersection consult
        the codebook (inverse-recommended via ``Ops.disjunctionReverse``
        / ``Ops.conjunctionReverse``); swap / quantize / not are
        tensor-only.

        Implementation caveat (read before trusting the split as a code
        boundary): on the DEFAULT-ONLY fast path the split holds
        cleanly -- ``compose`` emits ``current_rules`` from the grammar
        XML and runs NO tensor reduction, and the per-space_role
        ``SyntacticLayer.forward`` cursors execute the unary pi / sigma
        fold (CS-side). On the FULL-ROUTER path, however,
        ``LanguageLayer.compose`` currently does BOTH: it selects the
        rules AND folds the slab tensorially through the op modules
        (``BinaryStructuredReductionLayer.forward`` -> ``op(left,
        right)``), caching the [B, 1, D] root in ``_last_root_state``.
        The per-space_role ``SyntacticLayer.forward`` / ``reverse`` then
        deliberately SKIP re-execution on that path (guarded by
        ``not _grammar_is_default_only``) precisely because re-running
        the ops would double-apply them. So in the full-router case the
        WS-analysis and CS-execution stages are co-located inside
        ``LanguageLayer.compose`` rather than separated across the
        modules named above. See plan task 5 follow-up.
        """
        # Per-compose cursor reset. The OLD semantics zeroed each
        # per-space_role SyntacticLayer cursor lazily on every compose() call
        # (via the ``gen != _cursor_compose_gen`` branch keyed off this
        # counter). On the SymbolSpace.cursor path we reproduce that
        # EXACTLY with an unconditional in-place reset of all space_roles'
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
        #     router adds no information; ``current_rules`` is
        #     populated from the grammar XML directly.
        #
        #   * Full router — any other rule is present (``intersection``,
        #     ``union``, ``lift``, ``lower``, ``not``, …). The signal
        #     router runs its compose pass to select per-position
        #     copy / reduce ops.
        #
        # The retired ``<SymbolSpace><useGrammar>`` XML knob used to
        # also gate this; the grammar XML is now the sole driver.
        # Batch size for the dense ``rule_probs``: the per-word /
        # boundary fire passes the STM snapshot ``[B, N, D]`` as
        # ``input_vectors``, so its leading dim is the predictor's B.
        # ``None`` when the operand is not a [B, ...] tensor (the
        # ``RoutingState`` then falls back to inferring B from the rule
        # rows / leaves ``rule_probs=None`` -- see ``_build_routing_state``).
        b_hint = (int(input_vectors.shape[0])
                  if torch.is_tensor(input_vectors) and input_vectors.dim() >= 1
                  else None)
        # A5 fullgraph fix: thread the compose operand's device into the
        # RoutingState build so the dense ``rule_probs`` allocation avoids
        # the non-proxyable ``TheDevice.get()`` inside the traced forward.
        dev_hint = (input_vectors.device
                    if torch.is_tensor(input_vectors) else None)
        if self._grammar_is_default_only:
            self.current_rules = self._default_compose_rules()
            self._pad_S_cursor_to_target(self.current_rules)
            # ADDITIVE: build the first-class RoutingState alongside the
            # (unchanged) ``current_rules`` dict so the intra-sentence
            # predictor can read a dense ``[B, n_rules]`` ``rule_probs``.
            self.routing_state = self._build_routing_state(
                self.current_rules, batch_size=b_hint, device=dev_hint)
            return self.current_rules
        self.current_rules = self.languageLayer.compose(
            input_vectors, self, subspace=subspace) or {}
        self._pad_S_cursor_to_target(self.current_rules)
        # ADDITIVE: same RoutingState build on the full-router path.
        self.routing_state = self._build_routing_state(
            self.current_rules, batch_size=b_hint, device=dev_hint)
        return self.current_rules

    def _pad_S_cursor_to_target(self, rules_dict):
        # Forward-only asymmetric padding: extend the SS-space_role rule cursor
        # to ``self._target_cursor_length`` with ``TheGrammar.id_SS``
        # (the no-op grammatical transition).
        # See doc/plans/2026-05-20-static-per-word-loop-impl.md §1.
        # Non-SS space_roles naturally return None past their end (a no-op),
        # so only the SS space_role — the one that owns the per-word stem —
        # needs explicit padding.
        N = int(self._target_cursor_length)
        if N <= 0:
            return rules_dict
        id_SS = TheGrammar.id_SS
        if id_SS is None or rules_dict is None:
            return rules_dict
        s_rules = rules_dict.get('SS')
        if s_rules is None:
            rules_dict['SS'] = [id_SS] * N
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
        """Run the signal router's reverse pass; populate
        ``self.generate_rules``.

        Default-only fast path mirrors ``compose``.

        Stage 3 (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):
        the signal router (``self.languageLayer``) is the canonical
        parser. The retired chart and STM shift-reduce dispatch paths
        are gone.

        Reverse mirror of the WS-analysis / CS-execution split (see
        ``compose``): this populates ``self.generate_rules`` (the hard
        reverse rule list per space_role). On the default-only path the
        CS-side execution stage is the per-space_role ``SyntacticLayer.reverse``
        cursors, which pop ``generate_rules`` last-applied-first and run
        each rule's inverse fold. On the full-router path
        ``LanguageLayer.generate`` handles the inverse routing
        tensorially and ``SyntacticLayer.reverse`` is a cursor-advance
        no-op (same co-location caveat as ``compose``).
        """
        self._generate_generation += 1
        if self._grammar_is_default_only:
            self.generate_rules = self._default_generate_rules()
            return self.generate_rules
        self.generate_rules = self.languageLayer.generate(
            target_vectors, self, subspace=subspace) or {}
        return self.generate_rules

    # Method names that count as the per-space_role "natural fold". The
    # default-only / useGrammar='none' fast paths fire only these from
    # the grammar XML so the per-space dispatch doesn't accidentally
    # invoke compositional operators (not, intersection, lift, ...)
    # that were authored for the chart's selection pass and are
    # disabled under useGrammar='none'. Maps the OLD ``default_rule``
    # semantics ('pi' for CS, 'sigma' for subsymbolic / SS) onto the grammar XML
    # without re-introducing a code-level fallback: when the grammar
    # XML lacks ``C = pi(C)`` / ``S = sigma(S)`` / ``P = sigma(P)``
    # entries the dispatch is correctly a no-op for that space_role.
    _NATURAL_FOLD_METHODS = ('pi', 'sigma')

    # Map a rule's LHS nonterminal to a per-space_role value for dispatch
    # routing. The legacy flat-format grammar parser tags every rule
    # ``space_role='SS'`` regardless of LHS; the canonical's LHS is
    # authoritative. Restrict to the three Space space_roles; non-space_role
    # nonterminals (NP, VP, ...) get filtered out by the caller.
    # KEYS are LHS nonterminal categories ('P' / 'C' / 'S'); VALUES are
    # the space_role labels.
    _LHS_SPACE_ROLE_MAP = {'P': 'subsymbolic', 'C': 'CS', 'S': 'SS'}

    # -- RoutingState construction (Task: rule-conditioned predictor) --

    @staticmethod
    def _flatten_selected_rules(rules_by_space_role):
        """Flat row-0 rule-id list across space_roles (sorted-space_role order).

        Row 0 is the canonical sequence convention already used by
        ``SyntacticLayer._next_rule_name`` (per-row dispatch is a
        follow-on). Tolerates both the multi-row ``list[list[int]]`` and
        the legacy flat ``list[int]`` per-space_role shapes via
        ``SyntacticLayer._row_zero_rules``.
        """
        if not rules_by_space_role:
            return []
        flat = []
        for space_role in sorted(rules_by_space_role.keys()):
            per_space_role = rules_by_space_role.get(space_role)
            row0 = SyntacticLayer._row_zero_rules(per_space_role)
            for rid in row0:
                try:
                    flat.append(int(rid))
                except (TypeError, ValueError):
                    continue
        return flat

    def _synthesize_rule_probs(self, rules_by_space_role, batch_size, device=None):
        """Build the dense ``[B, n_rules]`` rule distribution.

        ``device`` (A5 fullgraph fix): the compose operand's device,
        threaded down to the ``torch.zeros`` allocations so neither builder
        calls ``TheDevice.get()`` inside the traced forward (DeviceHandle is
        not Dynamo-proxyable). ``None`` falls back to ``TheDevice.get()``.

        Dispatches between two builders:

        * SOFT path (``_synthesize_rule_probs_soft``): when the signal
          router (``self.languageLayer``) ran ``compose`` and cached its
          per-space_role SOFT marginals in ``_last_space_role_routings``, aggregate
          those differentiable marginals into a global ``[B, n_rules]``
          tensor. This keeps a graph back to the router's anchor
          scorers, so the intra-sentence predictor's ``routing_proj``
          bias backprops predictor-loss -> ``rule_probs`` -> router
          marginals -> router scorer params (``copy_anchor`` /
          ``apply_anchor`` / ``reduce_anchor``).
        * HARD path (``_synthesize_rule_probs_hard``): the default-only /
          ``useGrammar='none'`` fast path never calls
          ``LanguageLayer.compose`` (so ``_last_space_role_routings`` is
          empty); there is no router to train. Fall back to the
          (detached) hard scatter -- unit mass onto the SELECTED rule-ids,
          L1-normalized per row.

        Branch on whether ``self.languageLayer._last_space_role_routings`` is
        populated.
        """
        ll = getattr(self, 'languageLayer', None)
        space_role_routings = getattr(ll, '_last_space_role_routings', None) if ll is not None else None
        if space_role_routings:
            soft = self._synthesize_rule_probs_soft(
                space_role_routings, batch_size, device=device)
            if soft is not None:
                return soft
            # Soft aggregation declined (no usable marginals / shapes did
            # not line up): fall through to the hard scatter so the
            # predictor still gets a (detached) conditioning signal.
        return self._synthesize_rule_probs_hard(
            rules_by_space_role, batch_size, device=device)

    def _synthesize_rule_probs_soft(self, space_role_routings, batch_size,
                                    device=None):
        """Aggregate the router's SOFT per-space_role marginals into a dense,
        GRADIENT-BEARING ``[B, n_rules]`` rule distribution.

        ``space_role_routings`` is ``self.languageLayer._last_space_role_routings``:
        ``{space_role: {"unary": u_routing, "binary": last_round_routing,
        "binary_rounds": [...]}}`` cached by ``LanguageLayer.compose``.

        Differentiable aggregation (per space_role)
        -------------------------------------
        * unary: ``apply_counts = action_probs[:, :, R_copy:].sum(dim=1)``
          -> ``[B, R_apply]`` expected count per APPLY op. Column ``a``
          maps to ``_unary_rule_ids[space_role][a]``. (``action_probs`` is the
          straight-through softmax -- gradient-bearing; the COPY columns
          ``[:, :, :R_copy]`` are dropped: copy ops carry no distinct
          rule_ids.)
        * binary: ``reduce_counts = reduce_marginal_op.sum(dim=1)`` ->
          ``[B, R_reduce]`` expected count per REDUCE op, SUMMED over ALL
          reduction rounds (``binary_rounds``) to match the hard
          scatter's accumulate-across-rounds semantics. Column ``r`` maps
          to ``_binary_rule_ids[space_role][r]``.

        The per-op count COLUMNS are scattered to their global rule_id
        via ``index_add`` (differentiable w.r.t. the source counts -- NO
        in-place index assignment, NO ``.detach()`` / ``.item()`` on the
        counts). Rows with any mass are L1-normalized; zero-mass rows stay
        zero (the additive bias is then just ``routing_proj``'s bias).

        Returns the live differentiable ``[B, n_rules]`` tensor, or
        ``None`` when no usable marginals exist (caller then falls back to
        the hard scatter). FAIL LOUD: a non-finite marginal or assembled
        ``rule_probs`` raises (no silent NaN gating).
        """
        ll = self.languageLayer
        n_rules = int(len(TheGrammar.rule_table))
        if n_rules <= 0:
            return None

        # A5: device threaded from the compose operand (avoids the
        # non-proxyable ``TheDevice.get()`` inside the traced forward).
        if device is None:
            device = TheDevice.get()
        # Collect (rule_id_index_tensor, count_columns) contributions per
        # space_role, then index_add them onto a single zeros base so the graph
        # back to action_probs / reduce_marginal_op is preserved.
        contributions = []          # list of (idx [K] long, src [B, K] float)
        B_resolved = batch_size

        def _check_finite(name, t):
            if not torch.isfinite(t).all():
                raise ValueError(
                    f"SymbolSubSpace._synthesize_rule_probs_soft: non-finite "
                    f"{name} marginal (fail-loud per numerical policy).")

        for space_role, space_role_routing in space_role_routings.items():
            if not isinstance(space_role_routing, dict):
                continue

            # --- Unary apply ops ---------------------------------------
            u_routing = space_role_routing.get("unary")
            if isinstance(u_routing, dict):
                action_probs = u_routing.get("action_probs")
                if torch.is_tensor(action_probs) and action_probs.dim() == 3:
                    unary_layer = ll._unary_layers.get(space_role) \
                        if hasattr(ll._unary_layers, 'get') \
                        else (ll._unary_layers[space_role]
                              if space_role in ll._unary_layers else None)
                    r_copy = int(getattr(unary_layer, 'r_copy', 0)) \
                        if unary_layer is not None else 0
                    apply_slice = action_probs[:, :, r_copy:]   # [B, N, R_apply]
                    if apply_slice.shape[-1] > 0:
                        _check_finite("unary action_probs", apply_slice)
                        apply_counts = apply_slice.sum(dim=1)    # [B, R_apply]
                        rid_table = ll._unary_rule_ids.get(space_role, []) \
                            if hasattr(ll._unary_rule_ids, 'get') \
                            else ll._unary_rule_ids[space_role]
                        idx, cols = self._map_op_columns(
                            apply_counts, rid_table, n_rules)
                        if idx is not None:
                            contributions.append((idx, cols))
                            B_resolved = B_resolved or cols.shape[0]

            # --- Binary reduce ops (summed over all rounds) ------------
            rounds = space_role_routing.get("binary_rounds")
            if not rounds:
                one = space_role_routing.get("binary")
                rounds = [one] if isinstance(one, dict) else []
            reduce_total = None     # [B, R_reduce]
            for r_routing in rounds:
                if not isinstance(r_routing, dict):
                    continue
                rmo = r_routing.get("reduce_marginal_op")
                if not (torch.is_tensor(rmo) and rmo.dim() == 3):
                    continue
                if rmo.shape[1] == 0 or rmo.shape[2] == 0:
                    continue
                _check_finite("binary reduce_marginal_op", rmo)
                round_counts = rmo.sum(dim=1)                    # [B, R_reduce]
                reduce_total = (round_counts if reduce_total is None
                                else reduce_total + round_counts)
            if reduce_total is not None:
                rid_table = ll._binary_rule_ids.get(space_role, []) \
                    if hasattr(ll._binary_rule_ids, 'get') \
                    else ll._binary_rule_ids[space_role]
                idx, cols = self._map_op_columns(
                    reduce_total, rid_table, n_rules)
                if idx is not None:
                    contributions.append((idx, cols))
                    B_resolved = B_resolved or cols.shape[0]

        if not contributions:
            return None

        # Resolve B from the contributions if the caller's hint was None.
        B = int(B_resolved) if B_resolved else int(contributions[0][1].shape[0])
        if B <= 0:
            return None

        # Differentiable scatter: zeros base + index_add of each per-op
        # count column at its global rule_id. index_add is autograd-safe
        # w.r.t. the source (the count columns carry the router graph).
        probs = torch.zeros(B, n_rules, device=device)
        for idx, cols in contributions:
            if int(cols.shape[0]) != B:
                # Batch mismatch between space_roles' marginals: skip the soft
                # path entirely rather than fabricate an alignment.
                return None
            probs = probs.index_add(1, idx.to(device), cols.to(device))

        # L1-normalize rows with mass; zero rows stay zero. Build the
        # normalized tensor functionally (no in-place row assignment) so
        # the graph to the count columns is preserved.
        row_sums = probs.sum(dim=1, keepdim=True)               # [B, 1]
        denom = torch.where(row_sums > 0, row_sums,
                            torch.ones_like(row_sums))
        probs = probs / denom

        if not torch.isfinite(probs).all():
            raise ValueError(
                "SymbolSubSpace._synthesize_rule_probs_soft produced a "
                "non-finite rule_probs tensor (fail-loud per numerical "
                "policy).")
        return probs

    def _map_op_columns(self, counts, rid_table, n_rules):
        """Map per-op count columns ``counts`` ``[B, R]`` to their global
        rule_id indices for a differentiable ``index_add``.

        ``rid_table`` is the per-space_role ``op_id -> rule_id`` list
        (``_unary_rule_ids`` / ``_binary_rule_ids``). Returns
        ``(idx [K] long, cols [B, K] float)`` selecting only the op
        columns whose rule_id is in ``[0, n_rules)``; ``(None, None)``
        when no column maps. ``cols`` is a (differentiable) column
        gather of ``counts`` -- it keeps the graph to the marginals.
        """
        if not torch.is_tensor(counts) or counts.dim() != 2:
            return None, None
        R = int(counts.shape[1])
        keep_cols = []
        keep_rids = []
        for op_id in range(min(R, len(rid_table))):
            try:
                rid = int(rid_table[op_id])
            except (TypeError, ValueError):
                continue
            if 0 <= rid < n_rules:
                keep_cols.append(op_id)
                keep_rids.append(rid)
        if not keep_cols:
            return None, None
        col_idx = torch.tensor(keep_cols, dtype=torch.long,
                               device=counts.device)
        cols = counts.index_select(1, col_idx)                  # [B, K]
        idx = torch.tensor(keep_rids, dtype=torch.long)
        return idx, cols

    def _synthesize_rule_probs_hard(self, rules_by_space_role, batch_size,
                                    device=None):
        """Build the dense ``[B, n_rules]`` rule distribution (HARD scatter).

        DETACHED fallback for the default-only / ``useGrammar='none'``
        fast path (no router to train): scatter unit mass onto the
        SELECTED rule-ids and L1-normalize per row, so each
        ``rule_probs[b]`` is a distribution over the rules that FIRED on
        row ``b``. This conditions the intra-sentence predictor on WHICH
        rules fired. The gradient-bearing soft-marginal aggregation lives
        in ``_synthesize_rule_probs_soft`` (used whenever the router ran).

        Per-row handling
        ----------------
        * Full-router path: ``rules_by_space_role[space_role]`` is ``B`` rows (one
          per batch element, see ``LanguageLayer.compose``); each row's
          own selected ids are scattered onto that row.
        * Default-only / single-canonical-row path: a single row 0 is the
          canonical sequence; it is BROADCAST across all ``B`` rows (the
          predictor then sees one shared distribution -- matching the
          "row 0 is canonical" convention).

        Zero-fire rows (no selected ids) are left as an all-zero row (the
        additive bias then contributes only ``routing_proj``'s bias term;
        documented). Returns ``None`` when ``n_rules <= 0`` or no usable
        batch size can be resolved.

        FAIL LOUD: a non-finite value in the assembled tensor raises
        (never silently sanitized) per the project's numerical policy.
        """
        n_rules = int(len(TheGrammar.rule_table))
        if n_rules <= 0:
            return None

        # Resolve B: prefer the caller's hint (the compose operand's
        # leading dim). Fall back to the rule-row count when the operand
        # batch dim was unavailable but the rows are per-row.
        per_space_role_rows = None
        max_rows = 0
        if rules_by_space_role:
            for space_role in rules_by_space_role:
                rows = rules_by_space_role[space_role]
                if isinstance(rows, list) and rows and isinstance(rows[0], list):
                    max_rows = max(max_rows, len(rows))
                    per_space_role_rows = per_space_role_rows or {}
        B = batch_size if batch_size is not None else (max_rows or None)
        if B is None or int(B) <= 0:
            return None
        B = int(B)

        # A5: device threaded from the compose operand (avoids the
        # non-proxyable ``TheDevice.get()`` inside the traced forward).
        if device is None:
            device = TheDevice.get()
        probs = torch.zeros(B, n_rules, device=device)

        # Per-row scatter. For each batch row, accumulate the row's own
        # rule-ids if the per-space_role container is per-row (len == B or the
        # row index exists), else fall back to row 0 (broadcast canonical).
        canonical_flat = self._flatten_selected_rules(rules_by_space_role)
        for b in range(B):
            row_ids = []
            for space_role in sorted(rules_by_space_role.keys()) if rules_by_space_role else []:
                per_space_role = rules_by_space_role.get(space_role)
                if (isinstance(per_space_role, list) and per_space_role
                        and isinstance(per_space_role[0], list)):
                    # Per-row container: use this row when present, else
                    # the canonical row 0 (broadcast).
                    row = per_space_role[b] if b < len(per_space_role) else per_space_role[0]
                    for rid in row:
                        try:
                            row_ids.append(int(rid))
                        except (TypeError, ValueError):
                            continue
                else:
                    # Legacy flat list: shared across rows.
                    for rid in (per_space_role or []):
                        try:
                            row_ids.append(int(rid))
                        except (TypeError, ValueError):
                            continue
            if not row_ids:
                # Per-row container existed but this row fired nothing AND
                # there is no per-row fallback -> use the canonical flat.
                row_ids = canonical_flat
            for rid in row_ids:
                if 0 <= rid < n_rules:
                    probs[b, rid] += 1.0

        # L1-normalize each row that has any mass; zero rows stay zero
        # (documented: the additive bias is then just routing_proj's bias).
        # Branchless so the compiled forward carries no data-dependent guard:
        # a zero row has ``probs == 0`` and ``row_sum == 0``, and
        # ``0 / clamp_min(tiny) == 0``; non-zero rows have integer mass
        # ``>= 1`` that ``clamp_min(tiny)`` leaves untouched. Equivalent to
        # the old ``if nz.any(): probs[nz] /= row_sums[nz]``.
        row_sums = probs.sum(dim=1, keepdim=True)
        probs = probs / row_sums.clamp_min(torch.finfo(probs.dtype).tiny)

        # FAIL LOUD on a non-finite rule-prob tensor -- but gate behind
        # MODEL_DEBUG: ``isfinite().all()`` is a data-dependent host sync
        # (graph break) under torch.compile, and ``util.MODEL_DEBUG`` is a
        # constant the tracer folds away when off, so the check leaves the
        # compiled forward entirely. Divergence still surfaces under
        # MODEL_DEBUG runs and via the eager finite-loss guard.
        if util.MODEL_DEBUG and not torch.isfinite(probs).all():
            raise ValueError(
                "SymbolSubSpace._synthesize_rule_probs produced a non-finite "
                "rule_probs tensor (fail-loud per numerical policy).")
        return probs

    def _build_routing_state(self, rules_by_space_role, batch_size, device=None):
        """Assemble the ``RoutingState`` companion for ``current_rules``.

        ``rules_by_space_role`` is the (unchanged) ``current_rules`` dict;
        ``batch_size`` is the predictor's B (the compose operand's
        leading dim) or ``None``. Builds ``selected_rules`` (flat row-0
        ids) and the dense ``rule_probs`` ``[B, n_rules]`` (or ``None``).

        ``device`` (A5 fullgraph fix): the compose operand's device,
        threaded so the dense ``rule_probs`` is allocated WITHOUT calling
        ``TheDevice.get()`` inside the traced forward (that returns a
        ``DeviceHandle`` Dynamo cannot proxy -> a fullgraph graph break).
        ``None`` (the eager ``__init__`` seed call) falls back to
        ``TheDevice.get()``.
        """
        rules_by_space_role = rules_by_space_role or {}
        selected = self._flatten_selected_rules(rules_by_space_role)
        rule_probs = self._synthesize_rule_probs(
            rules_by_space_role, batch_size, device=device)
        return RoutingState(
            rules_by_space_role=rules_by_space_role,
            selected_rules=selected,
            rule_probs=rule_probs)

    def _default_compose_rules(self):
        """Per-space_role rule IDs for the default-only / useGrammar='none'
        fast path (forward direction).

        Returns ``dict[space_role, list[list[int]]]`` listing each space_role's
        forward natural-fold rule_ids from ``TheGrammar.rules``
        (``method_name`` in :data:`_NATURAL_FOLD_METHODS`,
        ``canonical`` not containing ``.reverse``). Cached after
        first call -- the grammar is fixed at construction time.
        """
        cache = getattr(self, '_default_compose_rules_cache', None)
        if cache is not None:
            return cache
        per_space_role = {}
        for i, r in enumerate(TheGrammar.rules):
            mn = getattr(r, 'method_name', None)
            if mn not in self._NATURAL_FOLD_METHODS:
                continue
            canonical = getattr(r, 'canonical', '') or ''
            if '.reverse' in canonical:
                continue
            # The legacy flat-format parser tags every rule
            # ``space_role='SS'``; the canonical's LHS nonterminal is
            # authoritative for which Space space_role should dispatch.
            space_role = self._LHS_SPACE_ROLE_MAP.get(getattr(r, 'lhs', None))
            if space_role is None:
                continue
            per_space_role.setdefault(space_role, []).append([i])
        merged = {space_role: [[rid for row in rows for rid in row]]
                  for space_role, rows in per_space_role.items()}
        self._default_compose_rules_cache = merged
        return merged

    def _default_generate_rules(self):
        """Per-space_role rule IDs for the default-only / useGrammar='none'
        fast path (reverse direction).

        Returns ``dict[space_role, list[list[int]]]`` listing each space_role's
        reverse natural-fold rule_ids (``method_name`` in
        :data:`_NATURAL_FOLD_METHODS`, ``canonical`` containing
        ``.reverse``). The dispatched ``method_name`` is shared with
        the matching forward rule so ``SyntacticLayer._by_name``
        resolves to the same parametrized layer either way.

        Falls back to the per-space_role forward rule_ids when no explicit
        ``.reverse`` rules are listed for that space_role. Legacy
        flat-format grammar blocks (``<S>sigma(S)</S>`` etc., without
        a ``<generate>`` section) still get the per-space_role reverse
        dispatch because dispatch reads ``method_name`` rather than
        the canonical string.
        """
        cache = getattr(self, '_default_generate_rules_cache', None)
        if cache is not None:
            return cache
        forward_per_space_role = {}
        reverse_per_space_role = {}
        for i, r in enumerate(TheGrammar.rules):
            mn = getattr(r, 'method_name', None)
            if mn not in self._NATURAL_FOLD_METHODS:
                continue
            space_role = self._LHS_SPACE_ROLE_MAP.get(getattr(r, 'lhs', None))
            if space_role is None:
                continue
            canonical = getattr(r, 'canonical', '') or ''
            if '.reverse' in canonical:
                reverse_per_space_role.setdefault(space_role, []).append([i])
            else:
                forward_per_space_role.setdefault(space_role, []).append([i])
        per_space_role = {}
        all_space_roles = set(forward_per_space_role) | set(reverse_per_space_role)
        for space_role in all_space_roles:
            per_space_role[space_role] = (
                reverse_per_space_role.get(space_role)
                or forward_per_space_role.get(space_role))
        merged = {space_role: [[rid for row in rows for rid in row]]
                  for space_role, rows in per_space_role.items()}
        self._default_generate_rules_cache = merged
        return merged

    def clear_grammar_cache(self):
        """Erase forward-derived grammar/routing traces before idea reverse.

        ``reconstructFromIdea`` uses this to avoid replaying the parse tree
        that comprehension left behind. The next ``generate`` call must infer a
        fresh reverse rule path from its target idea snapshot.
        """
        self.current_rules = {}
        self.generate_rules = {}
        self.recur_pass = 0
        for i in range(len(self.cursor)):
            self.cursor[i] = 0
        self.routing_state = self._build_routing_state(
            {}, batch_size=None)
        ll = getattr(self, 'languageLayer', None)
        if ll is not None:
            ll._last_space_role_routings = {}
            ll._last_output = None
            ll._last_root_state = None

    def gate_l1_loss(self, lam=0.0):
        """L1 penalty on every per-rule ``raw_gate`` Parameter owned by
        this SymbolSpace's per-space SyntacticLayers.

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
        for (_space_role, _name), layer in self._host_layer_registry.items():
            raw = getattr(layer, 'raw_gate', None)
            if raw is not None and torch.is_tensor(raw):
                term = torch.tanh(raw).abs().sum()
                total = term if total is None else total + term
            # Adverb eigenvalue edit: an L1 pull on the per-adverb edit
            # projection encourages a sparse expression over the eigs (the
            # primary sparsity is the in-forward soft-threshold; this is the
            # additional training pressure). Only present when <adverbEigEdit>
            # built it, so this is naturally gated.
            adv_edit = getattr(layer, '_adv_edit', None)
            if adv_edit is not None and hasattr(adv_edit, 'weight'):
                term = torch.tanh(adv_edit.weight).abs().sum()
                total = term if total is None else total + term
        if total is None:
            return None
        return lam * total

    def register_host_layer(self, space_role, rule_name, layer):
        """Register ``layer`` as the parametrized GrammarLayer for
        ``(space_role, rule_name)``. The chart's per-cell rule dispatch reads
        the registry to fire the host space's owned fold.

        Per-space SyntacticLayers call this at construction.
        """
        if not rule_name:
            return
        self._host_layer_registry[(space_role, rule_name)] = layer

    def host_layer(self, space_role, rule_name):
        """Return the registered GrammarLayer for ``(space_role, rule_name)``,
        or ``None``. The chart treats ``None`` as "rule has no host
        parametrized layer", routing to the generic fallback (Ops /
        typed-GrammarLayer facade) instead.
        """
        return self._host_layer_registry.get((space_role, rule_name))

    # -- Chart-authority surface (Stage 3) --
    #
    # The chart's ``register_grammar_layer`` / ``should_run_rule`` pair
    # serviced ``GrammarLayer.gated_run`` via ``_chart_authority``. The
    # chart is retired; SymbolSubSpace inherits the responsibility.

    def register_grammar_layer(self, layer):
        """Add a GrammarLayer instance to the chart-authority roster.
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
        grammar = getattr(self.languageLayer, 'grammar', None)
        if grammar is None or not rule_name:
            return 1.0
        body = f"{rule_name}(S)"
        try:
            return float(grammar.rule_probability(body))
        except Exception:
            return 1.0

    def _wire_signal_router_grammar_ops(self):
        """Attach grammar-rule layers to the signal router, grouped by
        (space_role, arity), driven by the loaded rule list.

        Post-2026-05-29 grammar-file-refactor (\xa75): the rule list is
        the canonical source of (name, space_role, arity) — the .grammar XML
        the model config points to is decoded into ``TheGrammar.rules``
        by ``Grammar.load_from_grammar_file``; this method consumes the
        decoded list, resolves a layer for each entry, and calls
        ``attach_unary_ops`` / ``attach_layer_ops`` per (space_role, arity).
        Per-rule space_role and op-name metadata is now propagated to the
        attached layers so the binary reduction layer can do
        space_role-respecting position gating (plan \xa76).

        Binary GrammarLayers (IntersectionLayer, UnionLayer, ...) expose
        their pair-wise math via ``.compose(left, right)``; unary ones
        (NotLayer, NonLayer, ...) via ``.forward(x)``. The signal
        router's ``BinaryStructuredReductionLayer`` calls ``op(left,
        right)``, so binary ops get wrapped with
        ``_BinaryGrammarOpAdapter``.
        """
        router = self.languageLayer

        # Group: (space_role, arity) -> list of (rule_id, layer, rule_name)
        #
        # Only compose rules (rules_upward) participate in the
        # forward-direction wiring: each one becomes an op the router
        # can dispatch in ``compose()``. Generate rules (rules_downward
        # — bodies like ``S, S = lower(S)``) describe the *reverse*
        # operation of the same op, not a separate op. Their arity counts
        # the LHS multi-output -- e.g. ``S, S = lower(S)`` has body arity
        # 1 but the underlying layer is binary, so attaching it again as
        # a unary op would call ``LowerLayer.forward(x)`` with one
        # argument and trip the layer's binary kernel.
        by_space_role_arity = {}
        n_upward = len(TheGrammar.rules_upward)
        for rule_id in range(n_upward):
            rule = TheGrammar.rules_upward[rule_id]
            space_role = rule.space_role
            arity = int(rule.arity)
            if arity not in (1, 2):
                continue
            rule_name = rule.method_name
            if not rule_name:
                continue
            layer = self._resolve_rule_layer(
                space_role, _dispatch_method_name_for_rule(rule))
            if layer is None:
                continue
            by_space_role_arity.setdefault((space_role, arity), []).append(
                (rule_id, layer, rule_name))

        # plan \xa76: collapse CS and SS into a single reduction space_role in CS.
        # Instead of attaching one layer per (space_role, arity), aggregate every
        # entry of a given arity ACROSS space_roles into a single attach call under
        # the conceptual reduction space_role 'CS'. ``op_space_roles`` / ``op_names`` still
        # carry the .grammar-declared metadata, so the merged binary layer
        # (BinaryStructuredReductionLayer) keeps each op's original space_role
        # tag for lift/lower (CS<->SS) role identification; ops, rule_ids,
        # op_names and op_space_roles are concatenated in lockstep so op order
        # stays aligned with rule_id order. Group by arity only, with a
        # stable (space_role, original-order) ordering for determinism.
        REDUCE_SPACE_ROLE = 'CS'
        by_arity = {}
        for (space_role, arity), entries in sorted(by_space_role_arity.items()):
            for rule_id, layer, name in entries:
                if arity == 2:
                    op = _BinaryGrammarOpAdapter(layer)
                else:
                    op = layer
                by_arity.setdefault(arity, {
                    "ops": [], "rule_ids": [], "op_names": [],
                    "op_space_roles": [],
                })
                bucket = by_arity[arity]
                bucket["ops"].append(op)
                bucket["rule_ids"].append(rule_id)
                bucket["op_names"].append(name)
                # preserve the op's *original* space_role letter as the role label
                bucket["op_space_roles"].append(space_role)
        # Placement-chooser cutover: pick the chooser kind BEFORE the
        # structured layers are built (they choose at construction).
        # ``<architecture><transformChooser>`` -- default "anchordot"
        # (stateless, basin unchanged); "mlp" builds the contextual
        # MLPTransformChooser (owns params, deliberate new basin).
        router.transform_chooser = str(TheXMLConfig.get(
            "architecture.transformChooser", default="anchordot"))
        for arity, bucket in sorted(by_arity.items()):
            if arity == 1:
                router.attach_unary_ops(
                    ops=bucket["ops"], rule_ids=bucket["rule_ids"],
                    op_names=bucket["op_names"], op_space_roles=bucket["op_space_roles"],
                    space_role=REDUCE_SPACE_ROLE)
            else:
                router.attach_layer_ops(
                    ops=bucket["ops"], rule_ids=bucket["rule_ids"],
                    op_names=bucket["op_names"], op_space_roles=bucket["op_space_roles"],
                    space_role=REDUCE_SPACE_ROLE)

        # plan \xa76: plumb the conceptual-reduction round floor onto the
        # router. subsymbolicOrder lives on the model (Models.py), not on
        # SymbolSubSpace, and is not cleanly reachable here within ~2 attribute
        # hops -- so leave the LanguageLayer default of 1. ``max(1, N-1) ==
        # N-1`` reproduces the pre-collapse round count, making the floor a
        # harmless no-op until/unless a host wires subsymbolicOrder through.
        _co = getattr(self, 'subsymbolicOrder', None)
        if _co is not None:
            router.subsymbolic_order = int(_co)

    def _resolve_rule_layer(self, space_role, rule_name):
        """Return a Layer instance for ``(space_role, rule_name)`` from the host
        registry, falling back to a fresh GRAMMAR_LAYER_CLASSES instance.

        Lookup order:
          1. Exact ``(space_role, rule_name)`` match in ``_host_layer_registry``
             -- the canonical wiring set up when the host space (e.g.
             ConceptualSpace) constructs its parametrized fold.
          2. Same ``rule_name`` registered under any other space_role --
             IntersectionLayer's PiLayer is owned by ConceptualSpace; the
             grammar may name it at the symbolic space_role.
          3. Fresh ``GRAMMAR_LAYER_CLASSES[rule_name]()`` instance --
             parameter-free ops (NotLayer, NonLayer, ...) that don't need
             a learned host module.
        Returns None if no resolution succeeds.
        """
        layer = self._host_layer_registry.get((space_role, rule_name))
        if layer is not None:
            return layer
        for (_other_space_role, other_rule), other_layer in (
                self._host_layer_registry.items()):
            if other_rule == rule_name:
                return other_layer
        cls = GRAMMAR_LAYER_CLASSES.get(rule_name)
        if cls is None:
            return None
        try:
            return cls()
        except TypeError:
            return None

    # -- truth-modulated loss -----------------------------------------
    def truth_modulated_loss(self, total_loss, symbolic_space,
                             symbol_acts=None, universality_score=None,
                             luminosity_weight=0.1, universality_weight=0.1,
                             truth_loss_weight=0.0,
                             allow_excluded_middle=1,
                             allow_contradiction=0,
                             balance_weight=0.1,
                             model=None):
        """Apply the SymbolSpace-owned TruthLayer modulation to a loss.

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
           activations in ``WholeSpace.forwardEnd``).

        Returns ``total_loss`` unchanged when the TruthLayer is
        absent or empty (bootstrap case with no truths recorded
        yet).  The caller is responsible for only invoking this in
        train mode -- the method itself has no ``train`` flag.

        All inputs that reach outside SymbolSpace (``symbolic_space``,
        ``symbol_acts``, ``universality_score``, the three weights)
        are passed explicitly so SymbolSpace never needs a back-
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
            configured = int(TheXMLConfig.get("SymbolSpace.syntacticHiddenDim"))
            if configured > 0:
                return configured
        except (KeyError, TypeError, ValueError):
            pass
        return min(256, max(64, n_slots * 4))

    def _attach_per_space_syntactic_layer(self, space, *, space_role):
        """Build the per-space SyntacticLayer for ``space`` (Step 4
        of doc/specs/2026-05-01-syntactic-layer-refactor.md).

        Gathers the space's already-constructed parametrized layers
        (PiLayer / SigmaLayer / NotLayer / ContiguousLayer) and passes
        them into ``build_space_syntactic_layer`` as ``builtin_layers``
        so their existing weights stay live. Other rules the configured
        grammar references for this space_role get lazy-constructed
        GrammarLayer wrappers.
        """
        builtin_layers = {}
        # Inner instance probes: use try/except rather than getattr-with-
        # defaults per the project's no-defensive-getattr stance.
        if space_role == 'subsymbolic':
            # Phase C (2026-05-13 rebalance): PartSpace owns
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
        elif space_role == 'CS':
            # Phase B (2026-05-13 rebalance): ConceptualSpace owns
            # ``sigma_percept`` (percept_dim → concept_dim) — the
            # canonical forward CS-space_role fold. Register it under the new
            # ``sigma`` rule name (per the doc/Spaces.md migration
            # table: ``C = sigma(PS)``) and the legacy ``pi`` alias so
            # old grammars ``C = pi(C)`` continue to dispatch correctly.
            sigma_percept = getattr(space, 'sigma_percept', None)
            if sigma_percept is not None:
                builtin_layers['sigma'] = sigma_percept
                builtin_layers['pi'] = sigma_percept  # legacy alias
            # Stage 4 (2026-05-27 doc/plans/2026-05-26-two-loop-pi-
            # sigma-substrate.md): LiftLayer / LowerLayer are binary
            # GrammarLayer ops at the CS space_role (STM-pair composition).
            # When the active grammar declares them inside <concepts>
            # (rule.space_role == 'CS'), build them here so they bind at the
            # ConceptualSpace SyntacticLayer and flow through to the
            # signal router as CS-space_role reduce ops via
            # ``_wire_signal_router_grammar_ops``.
            #
            # Sized to the symbol codebook width via the back-reference
            # to wholeSpace (``symbol_dim == concept_dim`` invariant
            # holds post-bivector retirement; either dim works).
            grammar_C_methods = {
                r.method_name for r in TheGrammar.rules
                if r.space_role == 'CS' and r.method_name is not None}
            wholeSpace = getattr(self, 'wholeSpace', None)
            perceptualSpace = getattr(self, 'perceptualSpace', None)
            if 'lift' in grammar_C_methods:
                from Layers import LiftLayer
                builtin_layers['lift'] = LiftLayer(
                    wholeSpace=wholeSpace)
            if 'verb' in grammar_C_methods:
                builtin_layers['verb'] = VerbLayer(
                    wholeSpace=wholeSpace)
            if 'adverb' in grammar_C_methods:
                builtin_layers['adverb'] = AdverbLayer(
                    wholeSpace=wholeSpace)
            if 'lower' in grammar_C_methods:
                from Layers import LowerLayer
                builtin_layers['lower'] = LowerLayer(
                    wholeSpace=wholeSpace)
            # Stage 9 (2026-05-27 doc/plans/2026-05-27-perceptstore-meta-
            # taxonomy-reentrancy.md): SymbolizeLayer (originally
            # MetaLayer, renamed 2026-05-28) is the binary CS-space_role grammar
            # op that binds a perceptual idea to a semantic idea,
            # creating a META node in the WS taxonomy. Needs BOTH the
            # wholeSpace (for ``insert_meta`` + WS codebook nearest-
            # match) AND the perceptualSpace (for the PerceptStore
            # codebook nearest-match that identifies the percept_id).
            if 'symbolize' in grammar_C_methods:
                from Layers import SymbolizeLayer
                builtin_layers['symbolize'] = SymbolizeLayer(
                    wholeSpace=wholeSpace,
                    perceptualSpace=perceptualSpace)
        elif space_role == 'SS':
            # Pi/Sigma swap (analysis/synthesis plan Phase 3, rev.
            # 2026-06-09): the WS-owned fold is ``self.pi`` (top-down
            # analysis). Register it under the new ``pi`` rule name AND the
            # legacy ``sigma`` alias -- the same alias idiom the subsymbolic/CS space_roles
            # use above -- so existing grammars (``S = sigma(S)``,
            # model.xml's default) keep dispatching the WS fold. The
            # grammar-DSL token migration is Phase-4 (knob split) work.
            fold = getattr(space, 'pi', None)
            if fold is not None:
                builtin_layers['pi'] = fold
                builtin_layers['sigma'] = fold  # legacy alias
            negation = getattr(space, 'propositional_negation', None)
            if negation is not None:
                builtin_layers['not'] = negation
            # FusionLayer / ContiguousLayer were retired 2026-05-04:
            # the operator was a duplicate of DisjunctionLayer at
            # SS-space_role. Existing XML grammars referencing
            # ``Fusion(S, S)`` / ``Contiguous(S)`` should migrate to
            # ``disjunction(S, S)``.
            # Lift / Lower wiring: per Stage 4 of doc/plans/
            # 2026-05-26-two-loop-pi-sigma-substrate.md, LiftLayer
            # and LowerLayer are first-class binary GrammarLayer ops
            # at the CS space_role with their own internal SigmaLayer /
            # PiLayer (no substrate borrow).  This SS-space_role branch is
            # kept for back-compat with XML grammars that still
            # declare ``<S>lift(S, S)</S>`` / ``<S>lower(S, S)</S>``
            # -- in that case rule.space_role == 'SS' and the wiring picks
            # the layer up under the SS-space_role host registry.  The
            # canonical home (post-Stage 4) is the CS-space_role branch
            # above.
            grammar_S_methods = {
                r.method_name for r in TheGrammar.rules
                if r.space_role == 'SS' and r.method_name is not None}
            # Lift / Lower stay explicit: they are parametrized ops that
            # need host wiring (``wholeSpace=space``), so the generic
            # ``cls()`` below would mis-build them. They are wired first
            # and the loop's ``if name in builtin_layers`` guard skips
            # them.
            if 'lift' in grammar_S_methods:
                from Layers import LiftLayer
                builtin_layers['lift'] = LiftLayer(
                    wholeSpace=space)
            if 'verb' in grammar_S_methods:
                builtin_layers['verb'] = VerbLayer(
                    wholeSpace=space)
            if 'adverb' in grammar_S_methods:
                builtin_layers['adverb'] = AdverbLayer(
                    wholeSpace=space)
            if 'lower' in grammar_S_methods:
                from Layers import LowerLayer
                builtin_layers['lower'] = LowerLayer(
                    wholeSpace=space)
            # All other SS-space_role ops: instantiate the parameter-free ops
            # generically from the module-local GRAMMAR_LAYER_CLASSES
            # registry rather than a hardcoded per-op special-case chain.
            # The registry maps every op name to its canonical class
            # (defined in THIS file), so this both (a) preserves the ops
            # the old chain wired (``isEqual`` / ``part`` / ``query``)
            # and (b) picks up any other parameter-free op a grammar
            # declares at the symbolic (SS) space_role (``queryPart`` / ``assertPart``
            # / ``not`` / ``swap`` / ``copy`` / ...) that the old chain
            # silently missed.
            #
            # Mereological grammar layers (``part`` / ``isEqual`` /
            # ``query`` / ...) are pure-geometric operations on the
            # WholeSpace codebook (clipped cosine projection on the
            # non-negative paired-index cone, per Architecture.md
            # §"Monotonicity of the bivector chain"); the standalone
            # ``MereologicalTree`` sidecar was retired -- the codebook IS
            # the meronymic structure.
            for name in grammar_S_methods:
                if name in builtin_layers:
                    # Already wired (lift / lower above; sigma / not from
                    # the substrate). Don't clobber the parametrized
                    # instance with a fresh parameter-free one.
                    continue
                cls = GRAMMAR_LAYER_CLASSES.get(name)
                if cls is None:
                    # 'merge' / unknown -- not a registry op.
                    continue
                try:
                    builtin_layers[name] = cls()
                except TypeError:
                    # Op needs constructor params (host-wired elsewhere,
                    # e.g. via _resolve_rule_layer / the CS-space_role branch).
                    # Skip rather than force; only the narrow
                    # constructor-arity TypeError is swallowed -- any
                    # other exception propagates (fail loud).
                    continue
        layer = build_space_syntactic_layer(
            space, self, space_role=space_role,
            builtin_layers=builtin_layers)
        # Register the new layer's parameters with the SymbolSpace param
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
        on the chart at CS-space_role over the per-word STM buffer
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
        legacy SR-parser SymbolSubSpace stack was removed (2026-05-20);
        the per-sentence cursor / recur_pass on SymbolSubSpace are reset
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
    # SymbolSubSpace is a plain ``nn.Module`` (not a ``Space``) but the
    # model iterates ``self.spaces`` calling ``set_sigma`` /
    # ``paramUpdate`` / ``getParameters`` / ``Start`` / ``End`` /
    # ``Reset`` on each entry, so we provide the same surface directly.
    # All five inline the same "iterate self.layers, call if present"
    # pattern the ``Space`` base class implements.

    def set_sigma(self, sigma):
        """Propagate exploration meta-parameters to owned layers.

        Mirrors ``Space.set_sigma`` (the no-basis branch — SymbolSubSpace
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
        — SymbolSubSpace inherits from ``nn.Module``, not ``Space``).
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
            # zeroes the SymbolSubSpace stack; the category and
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
        derivation matches a configured ``Grammar.start_patterns`` entry.
        The outer doc-streaming loop drains it after each ``runBatch``.
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
          ``for b in symbolSpace.drain_sentence_completed(): soft_reset(b)``

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

        Body-side buffers owned here: the SymbolSubSpace event, the
        CategoryStack / ReconstructionStack stacks, and the per-window
        transient tensors ``_last_svo`` / ``_svo_valid``.  These reallocate
        fresh-zero on shape change -- they're per-microbatch-row state
        with no cross-batch lifecycle.

        ``_stm_fired`` and ``discourse`` are NOT touched here: they live
        at B (per source row), persist across forward calls within a
        sentence, and are owned by :meth:`ensure_microbatch`.  Wiping
        them on every K-change (which happens whenever ``actual_max``
        BPE word count crosses a power-of-two boundary in PartSpace's
        AR unfold) would re-arm the once-per-sentence STM-residual fire
        flag mid-sentence, causing the discourse bias to inject multiple
        times for the same source row.
        """
        batch = int(batch)
        if batch == self.batch:
            # Cascade still runs in case callers grew their own state
            # without going through the SymbolSpace.batch counter.
            self.category_stack.ensure_batch(batch)
            self.reconstruction_stack.ensure_batch(batch)
            self._ensure_stm_batch(batch)
            return
        self.batch = batch
        self.category_stack.ensure_batch(batch)
        self.reconstruction_stack.ensure_batch(batch)
        self._ensure_stm_batch(batch)
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


# Plain-attr writes that the held SymbolSubSpace coordinator reads back
# INTERNALLY must forward through ``SymbolSpace.__setattr__`` -- otherwise an
# external ``symbolSpace.recur_pass = t`` would land on the wrapper while the
# coordinator's own ``self.recur_pass`` read never sees it.
_SYMBOLSPACE_FORWARD_WRITES = {
    'recur_pass', 'serial_mode', 'normalizer',
    '_per_sentence_initialized', '_target_cursor_length',
}


class SymbolSpace(PerceptualSpace):
    """The unified grammar/symbol container (2026-06-21 SymbolSpace refactor,
    Stage 3).

    OWNS the ``SymbolSubSpace`` coordinator (the typed-STM stack + grammar
    dispatch carrier) and is the home for the per-space_role SyntacticLayers and (Stage
    4) the symbol tables. It is a transparent CONTAINER: every ``symbolSpace.X``
    call site keeps working by FORWARDING to the held coordinator -- reads fall
    through ``__getattr__``, and the plain-attr writes the coordinator reads back
    internally (``_SYMBOLSPACE_FORWARD_WRITES`` + anything it already owns)
    forward through ``__setattr__``. Registered submodules / parameters stay on
    SymbolSpace.

    It SUBCLASSES ``PerceptualSpace`` (so ``isinstance(symbolSpace,
    PerceptualSpace)`` holds -- SS is a peer perceptual tower alongside
    PartSpace/WholeSpace, the third ``.what``/``.where`` carrier) but constructs
    via ``nn.Module.__init__``, deliberately SKIPPING ``Space.__init__``'s
    object/what/where/when VQ-basis construction this coordinator must not own.
    The eight Space-contract members defined on BOTH ``Space`` and the held
    ``SymbolSubSpace`` (``Reset`` / ``Start`` / ``End`` / ``paramUpdate`` /
    ``set_sigma`` / ``getParameters`` / ``attach_knowledge`` / ``knowledge``) are
    EXPLICITLY overridden below to delegate to the coordinator -- the inherited
    ``Space`` versions would otherwise SHADOW it (``__getattr__`` only catches
    *missing* attrs), breaking e.g. ``ss.Reset()``'s STM re-arm. Transparent
    forwarding is byte-identical to the pre-refactor behaviour where
    ``m.symbolSpace`` *was* the coordinator.
    """

    config_section = "SymbolSpace"

    def __init__(self, perceptualSpace, conceptualSpace, wholeSpace,
                 nPercepts, nConcepts, nSymbols, concept_dim, symbol_dim):
        nn.Module.__init__(self)
        self.subspace = SymbolSubSpace(
            perceptualSpace=perceptualSpace,
            conceptualSpace=conceptualSpace,
            wholeSpace=wholeSpace,
            nPercepts=nPercepts,
            nConcepts=nConcepts,
            nSymbols=nSymbols,
            concept_dim=concept_dim,
            symbol_dim=symbol_dim,
        )
        # SymbolSubSpace.__init__ pointed the home spaces' ``.symbolSpace``
        # back-ref at ITSELF (the coordinator); re-point them at THIS container so
        # ``perceptualSpace.symbolSpace is model.symbolSpace`` holds (the pipeline
        # carry contract). ``attach_symbolSpace`` uses object.__setattr__ -> no
        # nn.Module cycle.
        for _sp in (perceptualSpace, conceptualSpace, wholeSpace):
            if _sp is not None and hasattr(_sp, 'attach_symbolSpace'):
                _sp.attach_symbolSpace(self)

    # ``forward_symbol`` (the SymbolSpace->WholeSpace reach for the SS bind leg)
    # was RETIRED 2026-06-21: the leg is now CS-MEDIATED. ConceptualSpace.
    # _build_symbol_leg reads the order-raising codes and syncs them onto
    # SS.subspace.what, so SymbolSpace no longer reaches WholeSpace -- CS, which
    # sees all three towers in its forward, does it. See Spaces.py
    # ConceptualSpace._build_symbol_leg + bind_streams.

    # -- attribute forwarding to the held coordinator --------------------
    def __getattr__(self, name):
        # nn.Module.__getattr__ resolves registered submodules / params /
        # buffers and real instance attrs first; everything else (the
        # coordinator's API + Space-contract fields) forwards to the held
        # SymbolSubSpace. Guarded so a lookup before ``subspace`` is
        # registered raises cleanly instead of recursing.
        try:
            return super().__getattr__(name)
        except AttributeError:
            sub = self.__dict__.get('_modules', {}).get('subspace')
            if sub is not None:
                return getattr(sub, name)
            raise

    def __setattr__(self, name, value):
        # Submodules / parameters register on SymbolSpace. Plain-attr writes the
        # coordinator already owns (or in the explicit forward set) forward to it
        # so the coordinator's internal reads observe the external write.
        if not isinstance(value, (nn.Module, nn.Parameter)):
            sub = self.__dict__.get('_modules', {}).get('subspace')
            if (sub is not None and name != 'subspace'
                    and (name in _SYMBOLSPACE_FORWARD_WRITES
                         or (hasattr(sub, name)
                             and not hasattr(type(self), name)))):
                # Forward DATA writes the subspace owns -- but NOT assignments to
                # names SymbolSpace itself defines (its override methods /
                # properties), so e.g. a test spying ``ss.Reset = fn`` replaces
                # SymbolSpace's Reset rather than looping through the subspace.
                setattr(sub, name, value)
                return
        super().__setattr__(name, value)

    # -- Space-method OVERLAP: explicit overrides delegating to the subspace --
    # These 8 members are defined on BOTH ``Space`` and the SymbolSubSpace; the
    # inherited ``Space`` versions would shadow the subspace's (``__getattr__``
    # only catches MISSING attrs), so we override them to run the subspace's
    # implementation. (Step 2 of the decomposition migrates the grammar half UP
    # into this Space; these delegations shrink as that lands.)
    def Reset(self, *a, **k):
        return self.subspace.Reset(*a, **k)

    def Start(self, *a, **k):
        return self.subspace.Start(*a, **k)

    def End(self, *a, **k):
        return self.subspace.End(*a, **k)

    def paramUpdate(self, *a, **k):
        return self.subspace.paramUpdate(*a, **k)

    def set_sigma(self, *a, **k):
        return self.subspace.set_sigma(*a, **k)

    def getParameters(self, *a, **k):
        return self.subspace.getParameters(*a, **k)

    def attach_knowledge(self, *a, **k):
        return self.subspace.attach_knowledge(*a, **k)

    @property
    def knowledge(self):
        return self.subspace.knowledge

    # -- the CS coupling interface: forward / reverse ---------------------
    # CS interacts with SymbolSpace ONLY through these (the grammar compose /
    # generate dispatch over a CS-space_role STM snapshot CS provides). Default-only
    # grammars run inline (traceable under fullgraph=True); full-router grammars
    # run in a @torch.compiler.disable eager island (their per-row rule-id
    # bookkeeping is data-dependent host control flow dynamo cannot guard on; the
    # products are host dicts read before captured regions, so the island changes
    # no numerics). Moved here from Models._chart_compose_at_C / _ss_*_eager.
    def forward(self, snap):
        """Grammar COMPOSE over the STM snapshot (populates current_rules)."""
        if snap is None:
            return
        if getattr(self, '_grammar_is_default_only', True):
            self.compose(snap)
        else:
            self._compose_eager(snap)

    def reverse(self, snap):
        """Grammar GENERATE over the STM snapshot (populates generate_rules)."""
        if snap is None:
            return
        if getattr(self, '_grammar_is_default_only', True):
            self.generate(snap)
        else:
            self._generate_eager(snap)

    @torch.compiler.disable
    def _compose_eager(self, snap):
        self.compose(snap)

    @torch.compiler.disable
    def _generate_eager(self, snap):
        self.generate(snap)

    # -- CS -> SS: the .forward()-mediated symbol leg --------------------------
    def forward_concept_to_symbol(self, concept_sub):
        """CS -> SS: the symbol (representation) leg of a concept -- the
        reverse/decode direction of the row-aligned concept<->symbol
        dictionary, in a sparse autoencoder ``PS/WS -(W)-> CS -> SS -(edges)->
        percept``.

        The concept arrives THROUGH THE ARGUMENT (the dataflow rule: Spaces are
        operators; cross-space interaction goes only through ``forward``). The
        symbol is the ROW-ALIGNED view of the concept -- symbol ``i`` is the
        ``.what`` view of concept ``i`` -- so the leg is built from the
        concept's OWN materialized codes, NEVER from the WholeSpace meta
        codebook or a stashed ``_model_symbolSpace`` pointer (the reach the
        retired ``ConceptualSpace._build_symbol_leg`` made). SymbolSpace syncs
        the first ``N`` rows of the codebook IT OWNS (``subspace.what``) to the
        concept codes (a space writing its own codebook -- allowed) under
        ``no_grad``, so the decode / symbol-attention paths read the row-aligned
        symbol view; the returned leg is detached, so no gradient reaches the
        symbol codebook (the grad-bearing concept content is the scatter-add).
        Returns the ``[B, N, D]`` symbol-leg SubSpace, or ``None`` for an
        empty / degenerate concept (-> ``bind_streams`` fills a zero leg),
        matching the old leg's ``None`` contract.
        """
        if concept_sub is None or concept_sub.is_empty():
            return None
        event = concept_sub.materialize()
        if event is None or event.dim() < 2:
            return None
        sym_event = event.detach()
        if sym_event.dim() == 2:
            sym_event = sym_event.unsqueeze(0)
        N = int(sym_event.shape[-2])
        D = int(sym_event.shape[-1])
        # Sync SS's OWN .what codebook (row-aligned, first-N rows) to the
        # per-batch-mean concept codes. Self-write only -- no cross-space reach.
        cb = getattr(getattr(self, "subspace", None), "what", None)
        W = cb.getW() if (cb is not None and hasattr(cb, "getW")) else None
        if W is not None and torch.is_tensor(W) and int(W.shape[0]) > 0:
            rows = min(N, int(W.shape[0]))
            cw = min(D, int(W.shape[-1]))
            if rows > 0 and cw > 0:
                with torch.no_grad():
                    W[:rows, :cw] = sym_event[:, :rows, :cw].mean(dim=0).to(
                        W.device, W.dtype)
        sub = SubSpace(inputShape=(N, D), outputShape=(N, D),
                       nInputDim=D, nOutputDim=D)
        sub.copy_context(concept_sub)
        sub.set_event(sym_event)
        return sub


# The historical ``SymbolSpace = SymbolSubSpace`` alias (retired Phase G of
# doc/specs/2026-05-21-wordsubspace-stm-layer-refactor.md) is now a REAL
# container Space (``SymbolSpace`` above) that OWNS the SymbolSubSpace. The XML
# config section name ``<SymbolSpace>`` is preserved unchanged.
