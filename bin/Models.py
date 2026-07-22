"""Top-level model assembly, data loading, and experiment reporting.

``BasicModel`` composes the custom layers from ``Model.py`` into a set of
spaces that move between raw inputs, percepts, concepts, symbols, syntax,
and outputs.  The same module also carries the project utilities used to
load datasets, resolve config paths, plot results, and save reports.
"""

import copy
import logging
import math, os, warnings
import time
from pathlib import Path
from collections import namedtuple
from contextlib import contextmanager, nullcontext
import numpy as np
warnings.filterwarnings(
    "ignore",
    message="Initializing zero-element tensors is a no-op",
    category=UserWarning,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as _torch_mp
# DataLoader workers transfer tensors to the main process via shared
# memory.  PyTorch's default 'file_descriptor' strategy opens one FD per
# transferred storage; with prefetch_factor > 0 and many batches per
# epoch, this exhausts the per-process FD limit on Linux (default 1024).
# 'file_system' uses /tmp shm files instead -- no FD pressure.  See
# https://pytorch.org/docs/stable/multiprocessing.html#sharing-strategies.
try:
    _torch_mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass  # already set or unsupported on this platform

# Raise the soft FD limit to the hard ceiling so prefetch_factor /
# multi-worker DataLoaders do not hit "Too many open files" on Linux
# defaults (RLIMIT_NOFILE soft=1024).  No-op on platforms without
# resource (e.g. Windows) or when the soft limit is already saturated.
try:
    import resource as _resource
    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    if _soft < _hard:
        _resource.setrlimit(_resource.RLIMIT_NOFILE, (_hard, _hard))
except (ImportError, ValueError, OSError):
    pass
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
TheDevice = util.TheDevice
TheMessage = util.TheMessage

from visualize import Report, TheReport
from util import ProjectPaths, XMLConfig, compile, TheXMLConfig, init_config, init_compile_backend, amp_context, init_model_amp, init_device, TheDevice
from architecture import canonical_shape
import util as _util
from embed import WordVectors, PretrainModel, _random_unit_ball
from Optimizer import (Adam, SparseAdam, RowLocalAdam, MultiOptimizer,
                       preflight_finite_gradients)
from checkpoint_migrations import (
    LEGACY_WHOLE_STRUCTURE_KEY,
    OPTIMIZER_PARAM_NAMES_KEY,
    build_optimizer_param_manifest,
    infer_legacy_optimizer_param_manifest,
    migrate_wholespace_checkpoint,
    remap_optimizer_state_by_name,
    stamp_checkpoint_schema,
)
from data import Data, TheData

from Layers import Layer, PiLayer, SigmaLayer  # Import custom layers from Model.py
from Layers import ConceptualCombine
from Layers import LinearLayer
from Layers import LiftingLayer, CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon
from Layers import Error, TheError
from Layers import TernaryTruthStore
from Layers import Ops, GRAMMAR_LAYER_CLASSES, CONTIGUITY_PRESERVING_OPS
from Mereology import Mereology
from dataclasses import dataclass, field
from typing import List

from Spaces import ActiveEncoding, WhereEncoding, WhenEncoding, WhatEncoding, EventEncoding
from Spaces import Basis, Tensor, Codebook, Embedding
from Spaces import SubSpace, Space, InputSpace, PartSpace, ModalSpace, ConceptualSpace, WholeSpace, OutputSpace, ShortTermMemory
from Spaces import ReadingAttention, GlobalAttention
# ``normalize_codebook_mode`` moved onto ``Space`` as a staticmethod
# (2026-05-21) so the parsing logic stays namespaced; callers below
# read it as ``Space.normalize_codebook_mode(...)``.
from Language import SymbolSubSpace, SymbolSpace
from util import parse
# -- Inlined from Pipeline.py (2026-05-11 module consolidation) -------
# Previously in basicmodel/bin/Pipeline.py; inlined here when the body
# Sequential refactor reduced Pipeline.py to a handful of classes that
# Models.py was the sole consumer of. SubsymbolicTee dropped as dead
# code (zero construction sites).

# Process-global dedupe set for advisory pipeline-stage exceptions.
# Key: (exception_type_name, file, line). Value: count.
_PIPELINE_EXC_SEEN: dict = {}
_DEDUPE_FLUSH_EVERY = 1000

# fullgraph=True is the strict no-graph-break gate (see
# ``enable_compiled_step``). ``BASIC_FULLGRAPH=0`` relaxes it so dynamo
# segments the graph at each break and logs them all -- used to enumerate
# remaining breaks while migrating host-side symbol creation onto
# pre-allocated codebooks (insert-not-grow). Defaults to the strict gate.
ENUM_FULLGRAPH = os.environ.get("BASIC_FULLGRAPH", "1") != "0"


def _checkpoint_host_copy(value):
    """Detach a pure-Python checkpoint sidecar from live mutable state.

    Structural dictionaries are intentionally not ``nn.Module`` state, but
    several of them contain tensors (sparse edge values, LBG accumulators,
    promotion evidence).  Copy tensors to CPU so the sidecar is portable and
    recurse through the few container types used by the relation stores.
    """
    if torch.is_tensor(value):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {
            copy.deepcopy(k): _checkpoint_host_copy(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_checkpoint_host_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_host_copy(v) for v in value)
    if isinstance(value, set):
        return {_checkpoint_host_copy(v) for v in value}
    if isinstance(value, frozenset):
        return frozenset(_checkpoint_host_copy(v) for v in value)
    return copy.deepcopy(value)


def _mps_rng_state_or_none():
    """Return the Metal RNG state when this PyTorch build exposes it."""
    get_state = getattr(getattr(torch, "mps", None), "get_rng_state", None)
    if not callable(get_state):
        return None
    try:
        return get_state().detach().to("cpu").clone()
    except (RuntimeError, AttributeError):
        # CPU-only hosts and older macOS runtimes may expose the Python symbol
        # while having no usable MPS generator.
        return None


def _grammar_rows_for_report(rows):
    """Normalize legacy and batched rule selections for text reporting.

    Signal-router selections are normally ``list[list[int]]`` (one rule
    path per batch row), while the serial SS cursor may expose its shared
    path as ``list[int]``.  Reporting must accept both without mistaking
    every rule id in a shared path for an iterable batch row.
    """
    if torch.is_tensor(rows):
        rows = rows.detach().cpu().tolist()
    if rows is None:
        return []
    if isinstance(rows, (int, np.integer)):
        return [[int(rows)]]
    rows = list(rows)
    if not rows:
        return []
    if isinstance(rows[0], (list, tuple)):
        return [list(row) for row in rows]
    return [rows]


def _grammar_row_preview(row, limit=64):
    """Return ``(head, omitted, tail)`` for a bounded rule-path preview."""
    row = list(row)
    limit = max(2, int(limit))
    if len(row) <= limit:
        return row, 0, []
    n_head = limit // 2
    n_tail = limit - n_head
    return row[:n_head], len(row) - limit, row[-n_tail:]


class ReverseAdapter(nn.Module):
    """Wrap a module so its reverse() method is called via forward().

    Retained for backwards-compat (tests reference it); the live
    BasicModel reverse path no longer uses it -- explicit
    ``module.reverse(x)`` calls inside ``reverse`` do the job.
    """

    def __init__(self, wrapped):
        super().__init__()
        if isinstance(wrapped, nn.Module):
            self.wrapped = wrapped
        else:
            object.__setattr__(self, "_wrapped_nonmodule", wrapped)
            self.wrapped = None

    def forward(self, subspace):
        target = self.wrapped if self.wrapped is not None else self._wrapped_nonmodule
        return target.reverse(subspace)


class GrammarMergeGlue(nn.Module):
    """Progressive-bottleneck glue for ``useGrammar == 'all'``.

    Average-merge of adjacent pairs along the N axis, halving N per
    stage. Caches pairwise differences (``_merge_diff``) for exact
    reverse. ``is_last=True`` makes the glue a pass-through (deepest
    conceptual stage doesn't halve again).
    """

    def __init__(self, stage_idx: int, initial_n: int, is_last: bool):
        super().__init__()
        self.stage_idx = int(stage_idx)
        self.initial_n = int(initial_n)
        self.is_last = bool(is_last)
        self._merge_diff = None

    def forward(self, subspace):
        if subspace.is_empty():
            return subspace
        if self.is_last:
            return subspace
        x = subspace.materialize()
        left = x[:, 0::2, :]
        right = x[:, 1::2, :]
        self._merge_diff = left - right
        subspace.set_event((left + right) / 2)
        return subspace

    def reverse(self, subspace):
        if subspace.is_empty():
            return subspace
        if self.is_last:
            return subspace
        diff = self._merge_diff
        assert diff is not None, (
            "GrammarMergeGlue.reverse called without prior forward")
        y = subspace.materialize()
        left = y + diff / 2
        right = y - diff / 2
        B, N_half, D = left.shape
        expanded = torch.zeros(B, N_half * 2, D, device=y.device, dtype=y.dtype)
        expanded[:, 0::2, :] = left
        expanded[:, 1::2, :] = right
        subspace.set_event(expanded)
        self._merge_diff = None
        return subspace


class _BodyStage(nn.Module):
    """Body-stage record with non-owning Conceptual/Whole Space references.

    Those Spaces are already structurally owned by their model ModuleLists.
    Registering them again in a ModuleDict duplicated every Parameter and
    checkpoint key. Only the optional merge operator is a child here.
    """

    def __init__(self, cs, ws, merge=None):
        super().__init__()
        object.__setattr__(self, '_cs', cs)
        object.__setattr__(self, '_ws', ws)
        if merge is not None:
            self.merge = merge

    def __contains__(self, key):
        return key in ('cs', 'ws') or (
            key == 'merge' and 'merge' in self._modules
        )

    def __getitem__(self, key):
        if key == 'cs':
            return self._cs
        if key == 'ws':
            return self._ws
        if key == 'merge' and 'merge' in self._modules:
            return self._modules['merge']
        raise KeyError(key)


# FlattenKWrapper retired 2026-05-11: K-axis flatten/restore is now
# handled inline by ``BasicModel._flatten_k`` / ``_restore_k`` around
# each stem/body/head boundary in the pipeline methods.

# -- End inlined-from-Pipeline section -------------------------------


class Normalizer:
    """Thin wrapper over TheData's min/max scaling.

    Spaces hold a reference to an instance of this class (set during
    model construction) and call ``self.normalizer.{normalize,denormalize}``
    instead of reaching into the TheData global. Keeps the forward/reverse
    contract free of module-level data coupling.
    """

    def __init__(self, source):
        """Hold the underlying TheData-like object as ``_source``."""
        self._source = source

    def normalize(self, x, which="input"):
        """Forward to ``_source.normalize``; ``which`` selects input/output bounds."""
        return self._source.normalize(x, which=which)

    def denormalize(self, x, which="input"):
        """Forward to ``_source.denormalize``; ``which`` selects input/output bounds."""
        return self._source.denormalize(x, which=which)


# Higher-order-concept shape descriptors (RuleSpec / StepInfo / HoCShape)
# moved to bin/Mereology.py alongside the measure family that consumes
# them (Contiguous / Continuous / Peaceful / Area / Luminosity).


class BaseModel(Mereology, nn.Module):
    """Shared training, plotting, and persistence infrastructure for all models.

    Inherits the contemplative-awareness measure family
    (``Contiguous`` / ``Continuous`` / ``Peaceful`` / ``Area`` /
    ``Luminosity``) and the back-projection machinery (`hoc_shape`,
    `_walk_reverse`, `_derivation_path`, `_leaf_path_trust`, etc.)
    from :class:`Mereology`.  See ``bin/Mereology.py``.
    """
    name           = "BaseModel"
    spaces         = []
    reversible    = False
    plot           = False
    _optimizer     = None
    checkpoint_every_batches = 0
    _training_step_count = 0
    # Two-pass learning pass B (the explore trial) sets this True so the
    # per-sentence side effects (model clock, discourse ARMA rings, LTM
    # end-state chain, step/checkpoint counter) do NOT fire a second time
    # on the same sentence -- B trains the chooser but must not advance the
    # sequence state pass A already committed. See runBatch(exploration_trial).
    _exploration_trial = False
    # Class-level defaults for ``serial`` / ``symbolicOrder`` so that
    # BasicModel()
    # constructed directly (without going through ``init_config``)
    # still answers both attributes. ``init_config`` overrides these
    # via instance attributes when XML is loaded. ``serial`` selects the
    # per-word grammatical path; ``symbolicOrder`` is an order / loop
    # budget for symbolic processing. The split replaces the retired
    # ``conceptualMode`` serial|parallel enum (2026-06-13).
    serial = False
    symbolicOrder = 0
    # PS/WS concept binding at the serial word boundary. ``mixing`` keeps the
    # historical learned ConceptualCombine path for every existing config;
    # ``aligned`` is an explicit opt-in that preserves locations and consumes
    # the complete non-raw fold ladder from both towers.
    concept_binding = "mixing"
    # Class-level default for ``syntacticOrder`` (doc/specs/orders.md, NEW
    # 2026-06-19): the parse-tree DEPTH cap for the serial grammatical
    # reduction. 0 = unbounded (the reduction runs to its single-S fixpoint;
    # byte-identical). Overridden per instance via init_config when set.
    syntacticOrder = 0
    # Model-level trust in incoming assertions/testimony. It scales
    # per-truth DegreeOfTruth values before they enter TruthLayer/LTM and
    # attenuates persisted STM description trust.
    trust = 1.0
    # Scale applied to the DiscourseSpace contrastive loss. The
    # inter-sentence DiscourseSpace lives on ``self.symbolSpace``
    # (``self.symbolSpace.discourse``) rather than directly on the
    # model. Callers that need it should read through
    # ``symbolSpace``; ``<training><sentencePrediction>false`` in
    # config leaves ``symbolSpace.discourse`` as ``None``.
    # Class-level defaults for ARMA loss weight + priming scale.
    # Pre-2026-05-14 contrastive sentence-loss knobs retired alongside
    # <maskedPrediction>; the single ARMA MSE replaces the contrastive
    # + predictive cosine pair.
    arma_scale = 0.1
    sentence_priming_scale = 0.05
    # Inter-sentence end-state prediction loss weight (Task 8, plan §9).
    # ``InterSentenceLayer.consume_inter_loss`` returns a per-sentence-mean
    # MSE which ``runBatch`` weights by this before adding to ``TheError``.
    inter_loss_weight = 0.1
    # The InfoNCE next-idea contrastive term weight + the prediction-trial
    # fraction; both default 0.0 -> the feature is off -> byte-identical.
    inter_contrastive_weight = 0.0
    prediction_trial_ratio = 0.0

    @staticmethod
    def load_config(config_path=None):
        """Load model settings from an XML config file.

        Delegates to XMLConfig._parse_xml().  Returns a dict of dicts;
        missing fields are filled by create_from_config() using model.xml.
        """
        if config_path is None:
            config_path = os.path.join(ProjectPaths.PROJECT_DIR, "model.xml")
        return XMLConfig._parse_xml(config_path)

    @staticmethod
    def _unit_interval(value, default=1.0):
        """Coerce ``value`` into ``[0, 1]``; non-finite/malformed -> default."""
        try:
            out = float(value)
        except (TypeError, ValueError):
            out = float(default)
        if not math.isfinite(out):
            out = float(default)
        return max(0.0, min(1.0, out))

    def _effective_incoming_trust(self, trust):
        """Signed incoming DoT scaled by the model's testimony trust."""
        try:
            raw = float(trust)
        except (TypeError, ValueError):
            raw = 0.0
        if not math.isfinite(raw):
            raw = 0.0
        scale = self._unit_interval(getattr(self, 'trust', 1.0), default=1.0)
        return max(-1.0, min(1.0, raw * scale))

    @staticmethod
    def _resolve_dim(section, prev_dim):
        """Resolve a Space's nDim sentinel: 0 -> inherit ``prev_dim``.

        Shared between BasicModel.create() and BasicModel.create()
        (and any subclass): both pipelines chain dims through
        InputSpace -> PartSpace -> ConceptualSpace ->
        WholeSpace -> OutputSpace, and an unset / zero-sentinel
        nDim means "inherit the upstream Space's content dim".

        Respect-explicit (Task #11, decision (3)): an EXPLICITLY-set
        (non-zero) <nDim> is ALWAYS returned verbatim -- for the endpoint
        space_roles (InputSpace / OutputSpace) this is the contract that the head /
        input width the config author sized is honoured, never overridden by
        the chained ``prev_dim`` or a canonical band-derived width. Only the
        ``0`` sentinel triggers inheritance (and a KeyError -- a section that
        omits <nDim> entirely -- is treated as the inherit sentinel).
        """
        try:
            raw = TheXMLConfig.space(section, "nDim")
        except KeyError:
            return prev_dim
        return prev_dim if raw == 0 else raw

    @staticmethod
    def _obj_size(section):
        """Per-section objectSize: ``nWhere + nWhen`` from canonical_shape.

        Each Space carries its own positional / temporal encoding widths
        (architectural constants, not config options); the muxed event
        tensor's last-axis size is ``nDim + nWhere + nWhen``.
        """
        nw, nn = canonical_shape(section)
        return nw + nn

    @staticmethod
    def _nvec(section, n_out):
        """Resolve a Space's nVectors sentinel: 0 -> ``n_out``.

        Codebook spaces sample with replacement so nVectors can
        differ from active count; ``0`` means "use the active count
        as the codebook size", which is the safe default for
        non-codebook spaces (where the validator enforces
        ``nVectors == nActive``).
        """
        try:
            raw = TheXMLConfig.space(section, "nVectors")
        except KeyError:
            return n_out
        return n_out if raw == 0 else raw

    # -- Order Partitions (Ramsification) -----------------------------
    # Static helpers shared between BasicModel and the per-stage
    # BasicModel pipeline. Pure-functional except for
    # ``_conceptual_width_mode`` which reads TheXMLConfig.

    @staticmethod
    def _order_partitions(symbol_dim, subsymbolic_order):
        """Compute geometric-decay partition boundaries for symbolSum.

        Each conceptual order writes only to its designated slice,
        so the symbolic space becomes self-describing: the position of
        an activation reveals its conceptual order.

        Partition sizes follow geometric decay -- lower (more fundamental)
        orders occupy larger slices:
            order 0: [0,      D//2)       <- 1/2 of symbol_dim
            order 1: [D//2,   3D//4)      <- 1/4
            order 2: [3D//4,  7D//8)      <- 1/8
            ...
            last order: remainder of D
        """
        partitions, start = [], 0
        for t in range(subsymbolic_order):
            if t == subsymbolic_order - 1:
                end = symbol_dim
            else:
                end = start + max(1, symbol_dim // (2 ** (t + 1)))
            partitions.append((start, end))
            start = end
        return partitions

    @staticmethod
    def _activation_order(activation, partitions):
        """Return the order whose partition has the highest energy."""
        # ``.norm()`` returns zero-dim tensors; stack instead of rewrapping
        # via ``torch.tensor([...])`` to avoid the "copy-construct from a
        # tensor" deprecation warning.
        energies = torch.stack(
            [activation[s:e].norm() for s, e in partitions])
        return int(energies.argmax())

    # -- Hierarchical Epistemic Architecture --------------------------

    @staticmethod
    def _conceptual_width_mode():
        """Read ``architecture.conceptualWidth`` from XML; default
        ``tapered``. Accepts ``tapered`` (geometric halving per
        conceptual order, the historical behavior) or ``uniform``
        (every level keeps the same n_vectors).
        """
        try:
            value = TheXMLConfig.get("architecture.conceptualWidth", "tapered")
            value = str(value).strip().lower() if value is not None else "tapered"
        except Exception:
            value = "tapered"
        if value not in ("tapered", "uniform"):
            value = "tapered"
        return value

    @staticmethod
    def _level_shapes(n_vectors, dim, subsymbolic_order, width_mode="tapered"):
        """Per-level (N_t, D_t) shapes across the conceptual-order stack.

        Two width modes (set via XML ``architecture.conceptualWidth``):

        ``tapered`` (default, historical) -- D stays constant; N halves
            per level. Biological analogue: increasing receptive field
            (V1->V2->V4->IT). Requires ``n_vectors`` to be divisible by
            ``2^subsymbolic_order``.

                percepts:  (N, D)
                level 0:   (N/2, D)
                level 1:   (N/4, D)
                ...
                level k:   (N/(2^(k+1)), D)

        ``uniform`` -- N stays constant at every level. Useful when the
            grammar is the only compositional structure (e.g.
            XOR_grammar) and the per-level geometric reduction would
            otherwise let downstream layers memorize the task without
            using the chart's rule choices. Each level keeps the same
            ``n_vectors`` width:

                percepts:  (N, D)
                level 0:   (N, D)
                level 1:   (N, D)
                ...
                level k:   (N, D)
        """
        if width_mode == "uniform":
            return [(int(n_vectors), int(dim)) for _ in range(subsymbolic_order)]
        # Default: tapered (geometric halving).
        shapes = []
        for t in range(subsymbolic_order):
            n = n_vectors // (2 ** (t + 1))
            assert n > 0, \
                f"Level {t}: n_vectors={n_vectors} not divisible by 2^{t+1}"
            shapes.append((n, dim))
        return shapes

    @staticmethod
    def from_config(config_path=None, model_type=None, data=None):
        """Factory: create the right model type from XML config.

        Defaults to ``data/xor.xml`` for smoke runs; resolves relative
        paths via ``ModelFactory.resolve_xml``. Returns ``(model, cfg)``
        where ``cfg`` is the parsed dict.
        """
        if config_path is None:
            config_path = os.path.join(ProjectPaths.PROJECT_DIR, "data", "xor.xml")
        resolved_path = ModelFactory.resolve_xml(config_path)
        model = BasicModel()
        cfg = model.create_from_config(resolved_path, model_type=model_type, data=data)
        return model, cfg

    def _resolve_artifact_path(self, relpath):
        """Resolve a relative artifact path against the XML config directory."""
        if relpath is None or relpath == "":
            return relpath
        if os.path.isabs(relpath):
            return relpath
        config_path = getattr(self, "_config_path", None)
        config_dir = os.path.dirname(config_path) if config_path else ProjectPaths.PROJECT_DIR
        return os.path.join(config_dir, relpath)

    def create_from_config(self, config_path=None, model_type=None, data=None):
        """Create the model using settings from an XML config file.

        Initializes ``TheXMLConfig`` from ``config_path`` (overlaying
        ``data/model.xml`` defaults), validates the schema, then walks
        through space construction, optimizer setup, and embedding load.
        Mutates ``self`` extensively; returns the parsed config dict.
        """
        self._config_path = config_path
        self._config_data = data

        defaults_path = os.path.join(ProjectPaths.DATA_DIR, "model.xml")
        init_config(path=config_path, defaults_path=defaults_path)
        cfg = TheXMLConfig.data
        self.cfg = cfg

        arch = cfg["architecture"]
        ModelFactory.validate_config(cfg)

        _binding = str(TheXMLConfig.get(
            "architecture.conceptBinding", default="mixing")
        ).strip().lower()
        if _binding not in ("mixing", "aligned"):
            raise ValueError(
                "<architecture><conceptBinding> must be mixing or aligned "
                f"(got {_binding!r}).")
        self.concept_binding = _binding

        _t = TheXMLConfig.training
        _s = TheXMLConfig.space

        # DataLoader prefetch workers. Pulled here so every entry point
        # (ModelFactory.run, BaseModel.from_config, tests) shares the
        # same model.xml-defaulted value. 0 means synchronous in-process
        # batch assembly.
        self._num_workers = int(_t("numWorkers"))

        if model_type is None:
            # The data space_role moved from architecture-level <modelType> into
            # <data> as <dataType> (embedding | numeric) 2026-06-19.
            model_type = TheXMLConfig.data_type()

        # <embeddingPath> retired (2026-05-12): embeddings now ride
        # inside the single .ckpt bundle, not a separate .kv artifact.
        # Global objectSize is the InputSpace's where/when overhead (the muxed
        # event enters at IS); source from canonical_shape, not architecture.* .
        _nWhere, _nWhen = canonical_shape("InputSpace")
        _objectSize = _nWhere + _nWhen
        TheXMLConfig._data.setdefault("architecture", {})["objectSize"] = _objectSize

        # Resolve space output counts with sentinels.
        # nOutput=0 means "same as nInput for this space".
        # nInput=0 means "from data" (InputSpace) or "from previous space" (others).
        # The actual chain derivation happens inside create() using each space's outputShape.
        def _resolve(space, prev):
            raw = _s(space, "nOutput")
            return prev if raw == 0 else raw

        # InputSpace: if nOutput=0, derive from data at create() time (passed as 0 -> handled there)
        nInput    = _s("InputSpace", "nOutput")   # 0 = let create() derive from data
        nPercepts = _resolve("PartSpace", nInput)
        nConcepts = _resolve("ConceptualSpace", nPercepts)
        nSymbols  = _resolve("WholeSpace",   nConcepts)
        nWords    = nSymbols  # SyntacticSpace removed; kept for API compat
        nOutput   = _resolve("OutputSpace",      nSymbols)

        _nObjects = (
            _s("InputSpace", "nVectors")
            + _s("PartSpace", "nVectors")
            + _s("ConceptualSpace", "nVectors")
            + _s("WholeSpace", "nVectors")
            + _s("OutputSpace", "nVectors")
        )
        TheXMLConfig._data.setdefault("architecture", {})["nObjects"] = _nObjects

        self.create(
            nInput=nInput,
            nPercepts=nPercepts,
            nConcepts=nConcepts,
            nSymbols=nSymbols,
            nWords=nWords,
            nOutput=nOutput,
            subsymbolicOrder=arch["subsymbolicOrder"],
            model_type=model_type,
            data=data,
            reconstruction_scale=_t("reconstructionScale"),
            what_scale=_t("whatScale"),
            where_scale=_t("whereScale"),
            when_scale=_t("whenScale"),
        )
        if (self.concept_binding == "aligned"
                and int(getattr(self, "subsymbolicOrder", 0)) < 2):
            raise ValueError(
                "<conceptBinding>aligned</conceptBinding> requires "
                "<subsymbolicOrder> >= 2: order T denotes one base level "
                "plus T-1 folds per PS/WS tower.")

        # IR mask rate: Bernoulli probability that a P-space_role position is
        # replaced by NULL_PERCEPT for masked reconstruction. Default
        # 0.15 (BERT-style).
        _mask_rate = _t("maskRate", 0.15)
        self.mask_rate = float(_mask_rate if _mask_rate is not None else 0.15)
        if not (0.0 <= self.mask_rate <= 1.0):
            raise ValueError(
                f"maskRate must be in [0.0, 1.0], got {self.mask_rate}")

        # serial_mode (AR streaming fast path) retired with
        # <maskedPrediction> 2026-05-14.  Within-sentence training is
        # IR-only; the slide-and-recompute optimisation no longer has
        # a caller.  Keeping the attribute False so legacy reads stay
        # well-defined; remove next release.
        self.serial_mode = False
        if hasattr(self, 'perceptualSpace'):
            self.perceptualSpace.serial_mode = False
        if hasattr(self, 'conceptualSpace'):
            self.conceptualSpace.serial_mode = False
        if getattr(self, 'symbolSpace', None) is not None:
            self.symbolSpace.serial_mode = False

        # ``symbolicOrder`` is a non-negative symbolic / relational loop
        # budget. ``serial`` is the independent mode selector that decides
        # whether the forward body reads the input one word at a time.
        #
        # Back-compat: old configs encoded the mode as symbolicOrder 0/1.
        # When ``<serial>`` is omitted, derive it from ``symbolicOrder > 0``;
        # new configs should set ``<serial>`` explicitly.
        _grammar_default_order = (
            1 if getattr(self, 'useGrammar', 'none') != 'none' else 0)
        _so_raw = TheXMLConfig.get(
            "architecture.symbolicOrder", default=None)
        try:
            _so = (int(_so_raw) if _so_raw is not None
                   else _grammar_default_order)
        except (TypeError, ValueError):
            raise ValueError(
                f"<architecture><symbolicOrder> must be a non-negative "
                f"integer (got {_so_raw!r}).")
        if _so < 0:
            raise ValueError(
                f"<architecture><symbolicOrder> must be >= 0 (got {_so}).")
        self.symbolicOrder = _so

        _serial_raw = TheXMLConfig.get("architecture.serial", default=None)
        if _serial_raw is None:
            _serial = (_so > 0)
        elif isinstance(_serial_raw, bool):
            _serial = _serial_raw
        else:
            _serial_text = str(_serial_raw).strip().lower()
            if _serial_text in ("true", "1", "yes", "on"):
                _serial = True
            elif _serial_text in ("false", "0", "no", "off"):
                _serial = False
            else:
                raise ValueError(
                    f"<architecture><serial> must be boolean "
                    f"(got {_serial_raw!r}).")
        self.serial = bool(_serial)
        if self.serial and self.concept_binding == "mixing":
            # Existing configs that omit conceptBinding retain their historical
            # serial path for checkpoint compatibility.  An explicit request
            # for the mixing matrix in serial mode is invalid: the learned
            # cross-location matrix is a parallel-only ablation.
            _explicit_binding = TheXMLConfig.get(
                "architecture.conceptBinding", default=None)
            if _explicit_binding is not None:
                raise ValueError(
                    "<conceptBinding>mixing</conceptBinding> is parallel-only; "
                    "serial word concepts bind PS and WS at the same location")
        for _cs in (getattr(self, "conceptualSpaces", None) or []):
            object.__setattr__(_cs, "_concept_binding", self.concept_binding)
            object.__setattr__(_cs, "_serial", self.serial)
            # Non-registering owner link: ConceptualSpace can request a
            # value-only aligned active-prefix expansion without making the
            # model its nn.Module child (which would create an ownership
            # cycle/state-dict aliases).
            object.__setattr__(_cs, "_model", self)
        for _ws in (getattr(self, "wholeSpaces", None) or []):
            # Same non-owning expansion seam for WholeSpace.insert_whole.
            object.__setattr__(_ws, "_model", self)

        # syntacticOrder (doc/specs/orders.md, NEW 2026-06-19): the parse-tree
        # DEPTH cap for the serial grammatical reduction. 0 (default) =
        # unbounded (the NULL-seal reduce sweep collapses to a single S, exactly
        # as before -- byte-identical). A positive value caps the sweep to that
        # many forced fold levels; the <= word-count bound holds structurally
        # (a reduce micro-step no-ops once a row's depth reaches 1). Inert in
        # parallel mode (no per-sentence parse tree to bound).
        _syn_raw = TheXMLConfig.get(
            "architecture.syntacticOrder", default=None)
        try:
            _syn = int(_syn_raw) if _syn_raw is not None else 0
        except (TypeError, ValueError):
            raise ValueError(
                f"<architecture><syntacticOrder> must be a non-negative "
                f"integer (got {_syn_raw!r}).")
        if _syn < 0:
            raise ValueError(
                f"<architecture><syntacticOrder> must be >= 0 (got {_syn}).")
        self.syntacticOrder = _syn

        # GrammarOpsPass §6c sentence protocol (author sign-off
        # 2026-06-11): in serial mode, every sentence opens with an
        # independent PARALLEL prelude of ``subsymbolicOrder`` pumps
        # (pump zero: EMA on — the word-learning guarantee; intent-only
        # commit — the gist primes both towers via §5 ``set_intent``
        # and nothing enters the workspace), then the serial per-word
        # task runs with the §6d law in its serial partition; the gist
        # re-pumps on preemption only. ``<architecture>
        # <sentenceProtocol>`` — default OFF until the author's cutover
        # (the meronomy pattern: land dark, cut over deliberately).
        # §6c default cutover (2026-06-18, Alec): ON by default in SERIAL mode
        # (``serial == True``) so the whole-sentence gist (the initial
        # subsymbolic-order prelude pass) CONDITIONS each word's parts/wholes —
        # the context the per-word hard mask drops re-enters via the gist/intent.
        # OFF in parallel (the prelude is only invoked from the serial
        # ``_forward_body_per_word``, so parallel never runs it regardless).
        # Explicit ``<sentenceProtocol>`` in the XML overrides either way.
        _sp_default = self.serial
        self.sentence_protocol = bool(TheXMLConfig.get(
            "architecture.sentenceProtocol", default=_sp_default))
        # Eager diagnostic only. It is pre-declared for stable object shape and
        # deliberately not mutated while Dynamo captures the model.
        self._prelude_pumps = 0

        # TransformChooser kind. Mirrors the router-side read in
        # LanguageLayer._wire_signal_router_grammar_ops so the cutover reaches
        # the model-built structured layers too (the STM reducer). Default
        # "anchordot" -> stateless, basin unchanged.
        self.transform_chooser = str(TheXMLConfig.get(
            "architecture.transformChooser", default="anchordot"))

        # MetaSymbol role-participation Category codebook (doc/Language.md
        # "Participation Categories"). Learn a small role-space VQ during
        # perception: pending MetaSymbols accumulate role evidence, then
        # commit to a single category centroid. The structured grammar layers
        # use that role context for every transform chooser.
        self.category_codebook = bool(TheXMLConfig.get(
            "architecture.categoryCodebook", default=True))

        # Legacy/direct LiftLayer adverb helper flag. The live AdverbLayer
        # grammar op force-builds the same zero-init eigenmodifier; this knob
        # only affects ordinary LiftLayer instances and is mirrored here for
        # introspection. Default off -> plain LiftLayer sigma fold unchanged.
        self.adverb_eig_edit = bool(TheXMLConfig.get(
            "architecture.adverbEigEdit", default=False))

        # Mereological order-raising (doc/specs/mereological-order-raising.md).
        # When on, perception's autobind hook builds a meronymic lattice and
        # raises abstraction order as attention requires. The request + the
        # ramsification-table enable happen in _create_per_stage (order-safe:
        # this self.* read runs AFTER spaces are built, so it is introspection-
        # only -- gating reads the live config at build, like categoryCodebook).
        # Default off -> no table, no raising (byte-identical).
        self.mereology_raise = bool(TheXMLConfig.get(
            "architecture.mereologyRaise", default=False))
        # Experimental local PS/WS `.where` tiling.  It reuses the
        # mereologyRaise pump but replaces its row-wide/batch-collapsed route
        # with fixed-shape per-candidate agreement.  Kept opt-in until the
        # corpus gates in doc/plans/2026-07-13-overlapping-where-tiling-corpus
        # are measured.
        self.overlap_where_tiling = bool(TheXMLConfig.get(
            "architecture.overlapWhereTiling", default=False))
        if self.overlap_where_tiling and not self.mereology_raise:
            raise ValueError(
                "<overlapWhereTiling>true</overlapWhereTiling> requires "
                "<mereologyRaise>true</mereologyRaise>: the tiling is a "
                "refinement policy for the PS/WS<->CS subsymbolic pump.")

        # Reading attention (doc/specs/reading-attention.md "(A) Reading
        # attention"). When on, a learned `.where` producer runs at each t>0
        # subsymbolic pass and writes ``_passback_scope_where`` on the stage-0
        # WholeSpace -- the producer of the scope the <mereologyRaise> handoff
        # already consumes. The producer is a small dimension-agnostic MLP over
        # detached cosine-retrieval + span-geometry features (its widths are
        # not needed, so it is built here, AFTER ``self.create``). Registered
        # as a submodule (rides the state_dict); its readout params are added
        # to the optimizer in ``getOptimizer`` (the spaces walk misses it).
        # Default off -> ``self.reading_attention`` stays None: no module, no
        # scope, no loss (byte-identical).
        self.reading_attention_enabled = bool(TheXMLConfig.get(
            "architecture.readingAttention", default=False))
        self.reading_attention = (ReadingAttention()
                                  if self.reading_attention_enabled else None)

        # Global attention (doc/specs/reading-attention.md "(B) Global
        # attention"). When on, a free, content/relation-driven attention ranges
        # over a TYPED addressable space -- the input window, STM, LTM, and the
        # symbol/whole codebook -- emitting a typed `.where` + a soft-read of the
        # addressed content (parked on ``_global_attention_obs``). It REQUIRES
        # the stochastic element (the two-pass ``exploreTemperature``) to explore
        # -- with no next-word target it cannot break symmetry by itself. Dark by
        # default: no module, no obs, output unchanged (the soft-read is parked,
        # not yet fed back -- the consumer is a later slice). Byte-identical off.
        self.global_attention_enabled = bool(TheXMLConfig.get(
            "architecture.globalAttention", default=False))
        self.global_attention = (GlobalAttention()
                                 if self.global_attention_enabled else None)

        # Global-attention CONSUMER (doc/specs/reading-attention.md "(B)"; Alec:
        # close the loop). When on, the parked soft-read is fed BACK into the
        # head (``Finish``) as a zero-init gated residual, so the answer/output
        # loss trains the retrieval (gradient through the read -> alpha -> the
        # scorer). Requires <globalAttention>. Default off -> the soft-read stays
        # parked (B unchanged) -> byte-identical. This is the retrieval-augmented
        # answer mechanism for training-stages.md stages 4-5; the LTM address
        # space IS the parsed TruthSet (``ltm_store``), so "reading over the
        # TruthSet in LTM" is the SPACE_LTM read fed back here.
        self.global_attention_consume = bool(TheXMLConfig.get(
            "architecture.globalAttentionConsume", default=False))

        # SymbolSpace 3-stream peer bind (2026-06-21 SymbolSpace refactor,
        # Slice B). When on, the CS bind is 3-STREAM: PS (part-percepts), WS
        # (whole-percepts), SS (symbols) -- SymbolSpace becomes the symbol tower
        # with its own codebook + ``forward_symbol`` stream, and WholeSpace goes
        # tall (whole-percepts, not the compact symbol). Default off -> the
        # 2-stream (PS, WS) bind -> BYTE-IDENTICAL. (ConceptualCombine is already
        # n_streams-parametrized, Slice A.)
        # Relevance integration gate (Architecture sec C); default false.
        self.relevance_on = bool(TheXMLConfig.get(
            "architecture.relevance", default=False))
        try:
            self.priming_decay = float(TheXMLConfig.get(
                "architecture.primingDecay", default=0.9))
        except (TypeError, ValueError):
            self.priming_decay = 0.9
        # Energy-dissipating prime diffusion (Alec 2026-07-12): the fraction
        # of a connected row's standing energy that moves to its neighbors
        # per prime event. Live by default; 0 restores pure decay+bump.
        try:
            self.priming_spread = float(TheXMLConfig.get(
                "architecture.primingSpread", default=0.25))
        except (TypeError, ValueError):
            self.priming_spread = 0.25
        # Base grammar-confidence threshold for online STM reduction.  The
        # explicit word-axis controller lowers it with occupancy; legacy
        # serial configs retain their fixed reduce-marginal comparison.
        try:
            self.stm_reduce_tau = float(TheXMLConfig.get(
                "architecture.stmReduceTau", default=0.5))
        except (TypeError, ValueError):
            self.stm_reduce_tau = 0.5
        self.symbol_tower = bool(TheXMLConfig.get(
            "architecture.symbolTower", default=False))

        # Parse-tree-deleted decode (doc/plans/2026-06-19-grammar-inverses-
        # handoff.md "Goal 2"; reading-attention.md "(C) Idea decoding").
        # When on, the reverse path GENERATES a surface from the idea ALONE --
        # the stored ``generate_rules`` (the parse tree) is NOT rebuilt; the
        # decode selection is driven by the primed symbolic space (the
        # ReadingAttention/GlobalAttention distribution) rather than the chart.
        # Step 1 (this slice) gates the chart REBUILD so the reverse path runs
        # chart-free; the attention-driven selection and the symbol-driven
        # relative mask are later slices. Default off -> the chart fire is
        # untouched, the rule-driven reverse is unchanged (byte-identical).
        self.idea_decode = bool(TheXMLConfig.get(
            "architecture.ideaDecode", default=False))

        # Reconstruction from an idea with the forward derivation erased.
        # When on, reverse() clears the grammar/routing traces built during
        # comprehension, then asks SymbolSpace.generate to infer the reverse
        # rule path from the supplied idea snapshot. This differs from
        # <ideaDecode>, which deliberately skips the chart/router rebuild.
        self.reconstruct_from_idea = bool(TheXMLConfig.get(
            "architecture.reconstructFromIdea", default=False))

        # Concept index-read (snap design doc §ontology, Alec 2026-07-15):
        # when on, the serial per-word idea READS THROUGH THE INDEX to the
        # concept's random ``similarity_codebook`` row (signed hypersphere)
        # instead of using only the computed percept-binding event. Percepts
        # are mereological on the bounded unsigned hypercube and contract, so
        # the computed ideas collapse; the concept row is the separable
        # substrate SGNS (step (a)) shapes. Default off -> byte-identical.
        self.concept_index_read = bool(TheXMLConfig.get(
            "architecture.conceptIndexRead", default=False))

        # Truth-grounded reasoning, N-step (doc/plans/2026-06-23-reasoning-
        # live-wiring.md). reasoning_iterations N is the chain depth: N>0 routes
        # a query to the recurrent tool-use loop (the soft policy over the hard
        # isTrue/isPart tools) instead of the generative infer(). Step 3: the
        # default is now 1 (reasoning ON, depth-1, by default) -- a query routes
        # to the reasoner at depth 1 on every config; set
        # <reasoningIterations>0</reasoningIterations> for the old (off) behavior.
        # This is LOSS-identical for training (the answer/predict-next loss
        # weights default 0.0), not behavior-identical (it changes inference/
        # serve-time query routing). The deprecated <queryReasoning> alias: true
        # ⇒ depth 10.
        _ri = TheXMLConfig.get("architecture.reasoningIterations", default=None)
        if _ri is None:
            self.reasoning_iterations = (
                10 if bool(TheXMLConfig.get("architecture.queryReasoning",
                                            default=False)) else 1)
        else:
            self.reasoning_iterations = max(0, int(_ri))
        # Back-compat: code/tests that gate on the boolean still work.
        self.query_reasoning = self.reasoning_iterations > 0

        # The Thinking Kernel (doc/plans/thinking_kernel_spec.md; execution
        # notes doc/plans/2026-07-12-thinking-kernel-execution.md). N = the op
        # budget of a top-level think() frame. Absent/0 ⇒ off (byte-identical);
        # positive ⇒ answer_query additionally attaches the kernel's certified
        # result under the payload's "kernel" key.
        self.thinking_budget = max(0, int(TheXMLConfig.get(
            "architecture.thinkingBudget", default=0) or 0))

        # Serial word-at-a-time object/meta (doc/specs/mereological-order-
        # raising.md "Serial-mode word-at-a-time loop"; Alec 2026-06-17). When
        # on (serial only), the radix stem keeps TWO distinct axes:
        # sentence words (the outer traversal) and the constituent percepts
        # of ONE word (synthesized inside that word's iteration).  Neither
        # axis is the STM: STM remains the bounded, reducing grammatical
        # workspace.  In particular, PartSpace.nOutput is the width of the
        # instantaneous perceptual field, not a raw-character limit, and
        # stmCapacity is not a sentence-length limit.
        self.serial_object_meta = bool(TheXMLConfig.get(
            "architecture.serialObjectMeta", default=False))
        _word_cap = TheXMLConfig.get(
            "architecture.serialWordCapacity", default=None)
        try:
            self.serial_word_capacity = (
                None if _word_cap in (None, "") else max(1, int(_word_cap)))
        except (TypeError, ValueError):
            raise ValueError(
                "<serialWordCapacity> must be a positive integer; got "
                f"{_word_cap!r}")
        _bucket_raw = TheXMLConfig.get(
            "architecture.serialWordBuckets", default=None)
        if _bucket_raw in (None, ""):
            self.serial_word_buckets = (
                (int(self.serial_word_capacity),)
                if self.serial_word_capacity is not None else ())
        else:
            try:
                if isinstance(_bucket_raw, (list, tuple)):
                    _bucket_values = [int(v) for v in _bucket_raw]
                else:
                    _bucket_values = [
                        int(v.strip()) for v in str(_bucket_raw).split(",")
                        if v.strip()]
                self.serial_word_buckets = tuple(sorted(set(_bucket_values)))
            except (TypeError, ValueError):
                raise ValueError(
                    "<serialWordBuckets> must be a comma-separated list of "
                    f"positive integers; got {_bucket_raw!r}")
            if (not self.serial_word_buckets
                    or any(v < 1 for v in self.serial_word_buckets)):
                raise ValueError(
                    "<serialWordBuckets> must contain positive integers; "
                    f"got {_bucket_raw!r}")
            if (self.serial_word_capacity is not None
                    and self.serial_word_buckets[-1]
                    != int(self.serial_word_capacity)):
                raise ValueError(
                    "the largest <serialWordBuckets> width must equal "
                    f"<serialWordCapacity> ({self.serial_word_buckets[-1]} "
                    f"!= {self.serial_word_capacity})")

        # Model-level trust in incoming assertions/testimony. Personal
        # experience remains the model's own evidence; third-party words enter
        # TruthLayer/LTM with their supplied DegreeOfTruth multiplied by this
        # scalar before the existing ``truthCriterion`` gate/factors read them.
        self.trust = self._unit_interval(
            TheXMLConfig.get("architecture.trust", default=1.0),
            default=1.0)

        # LTM consolidation (doc/specs/mereological-order-raising.md "Truth /
        # Ideas processing"; Alec 2026-06-18). When true, the discourse LTM
        # (InterSentenceLayer end-state chain) and the RelativeTruthStore are
        # COMBINED into ONE unified Layers.TernaryTruthStore on SymbolSubSpace
        # (``ltm_store``): rows (NP1, VP, NP2) of full idea vectors + a per-row
        # timestamp + a scalar trust, stored UNSCALED, persisted (rides the
        # state_dict) and surviving Reset. The observe site appends each
        # end-state, ``_route_learned_relation`` appends ineffable relations,
        # and reason/verify_relation read it on the content slice. RTS is
        # constructed only when this gate is OFF. Off -> the legacy two-store
        # path (byte-identical). Row trust values are still scaled by the
        # incoming ``trust`` multiplier when descriptions/testimony are
        # persisted.
        self.ltm_consolidation = bool(TheXMLConfig.get(
            "architecture.ltmConsolidation", default=False))

        # Stateless server (mirrors WikiOracle's server.stateless; the
        # shipped deployment runs --stateless). Default TRUE: the runtime
        # request-body TruthSet (``store_truths``, ORIGIN_USER rows) is
        # request-scoped, so on every state_dict load the consolidated LTM
        # is revived WITHOUT the user rows (config-provisioned + conversation
        # rows persist) and the TruthLayer view is rematerialized -- see
        # ``SymbolSubSpace._revive_ltm_post_load``. Set false for a stateful
        # deployment where a saved checkpoint's user rows are durable state.
        self.stateless = bool(TheXMLConfig.get(
            "architecture.stateless", default=True))

        # Two-pass soft-superposition learning (doc/Language.md
        # "Soft-superposition route"). When ``<learning>`` is true, runEpoch
        # runs each TRAINING batch twice as two independent trials -- pass A
        # at superposition temperature 0 (sharp, recorded) and pass B at
        # ``<exploreTemperature>`` (flatter exploration, trimmed from the
        # batch error). The chooser is in the gradient path directly; there
        # is no advantage/policy term.
        # Default off -> the normal single forward (byte-identical). The
        # explore temperature is the [0, 1] sharp->uniform knob.
        self.two_pass_learning = bool(TheXMLConfig.get(
            "architecture.learning", default=False))
        self.explore_temperature = float(TheXMLConfig.get(
            "architecture.exploreTemperature", default=0.5))

        # Step 1 (2026-06-10 symbolic-iteration plan): mirror the symbolic
        # order and the independent serial mode onto each WholeSpace. The
        # WS forward dispatches SYMBOLIC ITERATIONS (quantize + parallel leg)
        # from the serial flag -- the SyntacticLayer is attached
        # unconditionally (model.xml default grammar), so layer presence
        # cannot distinguish the legs.
        # Space construction (``self.create`` above) runs before this
        # knob is parsed, hence the post-hoc stamp.
        for _ss in (getattr(self, 'wholeSpaces', None) or []):
            object.__setattr__(_ss, '_symbolic_order', self.symbolicOrder)
            object.__setattr__(_ss, '_serial', self.serial)
            object.__setattr__(_ss, '_priming_decay', self.priming_decay)
            object.__setattr__(_ss, '_priming_spread', self.priming_spread)
            # Canonical priming (Alec 2026-07-12): per-stage WS delegate to
            # the terminal's order-indexed codebook surface.
            _ws_list = list(getattr(self, 'wholeSpaces', None) or [])
            if _ws_list and _ss is not _ws_list[-1]:
                object.__setattr__(_ss, '_priming_canonical_ref',
                                   _ws_list[-1])
        # The ConceptualSpaces need the same stamp so the sparse-coding edge
        # population + scatter (gated on ``_symbolic_order > 0`` and parallel)
        # activate together. A plain host-attribute stamp -- byte-identical.
        # (P2 symbolic-only rework: the _n_ps_codes/_n_ws_codes source-layout
        # stamps retired with the percept families -- a_0 comes from the
        # order-0 snap, not PS/WS presence columns.)
        # <sparseReplace> RETIRED (P3 two-phase forward, decision 10): phase
        # separation makes non-replacement STRUCTURAL -- sparse content never
        # substitutes subsymbolic content; the symbolic phase's outputs feed
        # the SS leg, the head-side losses, and the concept table. The knob
        # parses as an inert deprecation warning.
        if TheXMLConfig.get("architecture.sparseReplace",
                            default=None) is not None:
            warnings.warn(
                "<sparseReplace> is retired (two-phase forward): the "
                "symbolic phase never substitutes the subsymbolic advance; "
                "the knob is ignored.", DeprecationWarning)
        for _cs in (getattr(self, 'conceptualSpaces', None) or []):
            object.__setattr__(_cs, '_symbolic_order', self.symbolicOrder)
            object.__setattr__(_cs, '_serial', self.serial)
            object.__setattr__(_cs, '_priming_decay', self.priming_decay)
            object.__setattr__(_cs, '_priming_spread', self.priming_spread)
            # Canonical priming: non-zero CS stages delegate to stage 0
            # (the concept-store owner).
            _cs_list = list(getattr(self, 'conceptualSpaces', None) or [])
            if _cs_list and _cs is not _cs_list[0]:
                object.__setattr__(_cs, '_priming_canonical_ref',
                                   _cs_list[0])
            # Back-ref to the model so the CS can rebuild the optimizer when its
            # per-order sparse weight Parameters grow (mirrors codebook growth).
            object.__setattr__(_cs, '_model', self)

        # Per-word ground-truth cursor enable. Pre-Stage-1.E this was
        # derived directly from ``useGrammar``; post-Stage-1.E it mirrors
        # ``self.serial``.
        # Kept as a back-ref attribute on InputSpace because
        # ``InputSpace.next_word`` and a handful of late-stage per-word
        # loops still consult it; Stage 3 (signal-router parser cleanup)
        # is the appropriate site
        # to retire the boolean entirely.
        if getattr(self, 'inputSpace', None) is not None:
            self.inputSpace._per_word_enabled = bool(self.serial)
            # serialObjectMeta must reach InputSpace.finalize_stem (it builds the
            # capturable word-index / per-word commit tensors only when on).
            object.__setattr__(self.inputSpace, '_serial_object_meta',
                               bool(self.serial_object_meta))
            if self.serial and self.serial_object_meta:
                # Sentence traversal and grammatical workspace are distinct.
                # New configs state the traversal cap explicitly; the STM
                # fallback preserves older serialObjectMeta fixtures without
                # silently changing their loop length.
                _word_capacity = self.serial_word_capacity
                if _word_capacity is None:
                    _word_capacity = int(getattr(
                        getattr(self, 'conceptualSpace', None),
                        'stm_capacity', 0) or 0)
                if _word_capacity < 1:
                    raise ValueError(
                        "serialObjectMeta requires either a positive "
                        "<serialWordCapacity> or ConceptualSpace "
                        "<stmCapacity> >= 1")
                object.__setattr__(self.inputSpace,
                                   '_serial_word_capacity', _word_capacity)
                object.__setattr__(
                    self.inputSpace, '_serial_word_buckets',
                    tuple(self.serial_word_buckets or (_word_capacity,)))
                _ps = getattr(self, 'perceptualSpace', None)
                if _ps is not None:
                    object.__setattr__(_ps, '_serial_object_meta', True)
                    object.__setattr__(
                        _ps, '_serial_aligned_fold_ladder',
                        self.concept_binding == "aligned")
                    object.__setattr__(_ps,
                                       '_serial_word_capacity', _word_capacity)
                    object.__setattr__(
                        _ps, '_serial_word_buckets',
                        tuple(self.serial_word_buckets or (_word_capacity,)))

        # DOCTRINE (Task 4, doc/plans/2026-05-29-stm-serial-parallel-modes.md
        # §"Serial mode = attentional filtering"): serial mode **is** the
        # attentional-filtering regime. The former guard here forced
        # ``conceptualSpace.serial_mode = False`` whenever
        # ``serial_mode and conceptualSpace.hasAttention`` on the theory
        # that attention violates the position-locality serial streaming
        # required. That theory is retired: MentalModel.xml runs serial
        # **with** attention by design (attention narrows the per-word
        # pipeline — a focused beam rather than a position-locality
        # violation). The guard is lifted; serial + attention is now the
        # supported, documented, trained regime. (No downgrade is applied.)

        # Per-word router-fire gating knob ``<architecture><routerWireSerial>``
        # (Task 4 / plan §4). Values:
        #   * ``per-word`` — fire ``symbolSpace.compose`` once per word in
        #     the serial per-word loop (the intra-predictor's routing
        #     context); boundary fire OFF.
        #   * ``boundary`` — fire only at the sentence boundary
        #     (``_chart_compose_at_C`` / ``_chart_generate_from_stm``); the
        #     inter-sentence predictor's routing snapshot.
        #   * ``both`` (DEFAULT) — per-word during training so the
        #     intra-predictor sees routing context, AND boundary still
        #     fires for the inter-sentence predictor. Default ``both``
        #     preserves the pre-existing boundary-fire behaviour.
        #   * ``off`` — neither fires.
        # Read with the same accessor idiom as the sibling
        # ``architecture.serial`` knob above (the
        # ``<routerWireSerial>`` element is a direct child of
        # ``<architecture>``, not under ``<training>``).
        _rws_raw = TheXMLConfig.get(
            "architecture.routerWireSerial", default=None)
        _rws = (str(_rws_raw).strip() if _rws_raw is not None else "both")
        if _rws not in ("per-word", "boundary", "both", "off"):
            raise ValueError(
                f"<architecture><routerWireSerial> must be one of "
                f"'per-word', 'boundary', 'both', 'off' (got {_rws_raw!r}).")
        self.router_wire_serial = _rws

        # InterSentenceLayer ARMA(p, q) loss weight. ``InterSentenceLayer.observe``
        # returns a per-batch MSE which ``runBatch`` weights by
        # ``arma_scale`` before adding to ``TheError``.  Active only
        # when ``<training><sentencePrediction>`` is true.
        self.arma_scale = float(
            TheXMLConfig.training("armaScale", 0.1) or 0.1)
        self.sentence_priming_scale = float(
            TheXMLConfig.training("sentencePrimingScale", 0.05) or 0.05)
        # Inter-sentence end-state prediction loss weight (Task 8, plan §9).
        # Mirrors ``arma_scale`` / ``intra_loss_weight``: ``runBatch`` scales
        # the consumed ``L_inter`` by this. Active only when the discourse
        # layer is present (``<training><sentencePrediction>`` true).
        self.inter_loss_weight = float(
            TheXMLConfig.training("interLossWeight", 0.1) or 0.1)
        # InfoNCE next-idea contrastive term weight + temperature (the discourse
        # layer accumulates it; runBatch consumes + weights it). 0.0 -> MSE-only.
        self.inter_contrastive_weight = float(
            TheXMLConfig.training("interContrastiveWeight", 0.0) or 0.0)
        self.inter_contrastive_temp = float(
            TheXMLConfig.training("interContrastiveTemp", 0.1) or 0.1)
        # Trial-split predictive training: fraction of training BATCHES run as
        # PURE next-idea prediction (recon terms zeroed). 0.0 -> every trial is
        # reconstruct -> byte-identical.
        self.prediction_trial_ratio = float(
            TheXMLConfig.get("architecture.predictionTrialRatio",
                             default=0.0) or 0.0)

        if "trainEmbedding" in arch and not isinstance(arch["trainEmbedding"], dict):
            te = arch["trainEmbedding"]
        elif "trainEmbeddings" in arch and not isinstance(arch["trainEmbeddings"], dict):
            te = arch["trainEmbeddings"]
        else:
            te = _t("trainEmbedding")
        if te is True:
            te = "BOTH"
        elif te is False or te is None:
            te = "NONE"
        self.train_embedding = te.upper()
        self.optimize_embedding = self.train_embedding not in ("NONE", "CBOW", "SBOW")
        # Stage 7 (doc/plans/2026-05-27-perceptstore-meta-taxonomy-
        # reentrancy.md): radix mode reroutes ``vocabulary`` to the
        # PerceptStore, but the orthographic-API Embedding still lives
        # on ``subspace.what`` (constructed by ``_build_what_basis``).
        # Resolve the Embedding via ``subspace.what`` so the back-ref
        # wiring + optimizer hook reach the right object regardless of
        # chunking mode.
        _emb_legacy = getattr(self.perceptualSpace.subspace, "what", None)
        if not isinstance(_emb_legacy, Embedding):
            _emb_legacy = (self.perceptualSpace.vocabulary
                           if isinstance(self.perceptualSpace.vocabulary,
                                         Embedding)
                           else None)
        if self.optimize_embedding and _emb_legacy is not None:
            emb_params = _emb_legacy.embedding_parameters()
            self.perceptualSpace.params = self.perceptualSpace.params + emb_params
        self.loss.embedding_scale = float(_t("embeddingScale") or 0.1)
        self.loss.conceptual_similarity_scale = float(
            _t("conceptualSimilarityScale", 0.0) or 0.0)
        # <definitionSparsityScale> (snap contract sec 5, 2026-07-06): lambda
        # for the rank-ordered soft-L0 that keeps concept definitions compact.
        # 0.0 (default) disables the penalty -- byte-identical.
        self.loss.definition_sparsity_scale = float(
            _t("definitionSparsityScale", 0.0) or 0.0)
        if _emb_legacy is not None:
            _emb_legacy.optimize_embedding = self.optimize_embedding
            object.__setattr__(_emb_legacy, "_model", self)

        self.checkpoint_every_batches = int(os.environ.get(
            "BASIC_CHECKPOINT_EVERY_BATCHES",
            _t("checkpointEveryBatches", 0) or 0,
        ))
        self._training_step_count = 0

        # Separate physical capacity from logical occupancy before autoload.
        # Legacy checkpoints have no active_mask key; VectorQuantize's load
        # hook then preserves this configured prefix instead of exposing the
        # entire preallocated reserve. Future masked checkpoints overwrite it
        # with their persisted active prefix.
        self._configure_aligned_active_codebooks()

        _autoload_env = os.environ.get("BASIC_AUTOLOAD")
        _autoload = bool(_t("autoload"))
        if _autoload_env is not None:
            _autoload = _autoload_env.strip().lower() in (
                "1", "true", "yes", "on")
        if _autoload:
            # Load from the same resolved path used by autosave. Previously an
            # empty <weightsPath> loaded "" while saving used the output-dir
            # fallback, so an autosaved run could never resume.
            wpath = self._checkpoint_path()
            # Checkpoint tensors include lazily registered grammar state.
            # Materialize those destination modules before loading an existing
            # file; doing it afterward would silently discard learned keys.
            # WholeSpace property growth is deliberately not prewarmed here:
            # it is independent of ConceptualSpace capacity and must happen at
            # an explicit optimizer/compiled-graph reset boundary.
            if os.path.exists(wpath):
                self._prewarm_checkpoint_shapes()
            # Single-artifact load: state_dict + vocab_extras +
            # bpe_extras all ride in the .ckpt. The separate .kv
            # embedding artifact was retired (2026-05-12).
            # Autoload: fail fast on a stale/mismatched ckpt (the
            # bivector retirement invalidates pre-refactor weights)
            # instead of silently proceeding on fresh init then crashing.
            self.load_weights(wpath, require_match=True)
        self.max_response_length = arch["maxResponseLength"]

        # LTM consolidation FU (Change 3, 2026-06-18): provisioning the
        # unified LTM from the XML <truthSet> now runs the truth texts through
        # the REAL forward pipeline (so the parse yields real encodings + a
        # real NP/VP/NP split), which requires the dataset to be LOADED. At
        # construction time the data is NOT yet loaded, so provisioning is
        # DEFERRED to the first ``runEpoch`` (lazy, guarded by
        # ``self._ltm_provisioned``); see ``runEpoch``. ``provision_ltm()``
        # stays callable explicitly (tests / serve) after data is loaded.
        self._ltm_provisioned = False

        # Phase C / Step 2: when reasoning trains under the answer-policy loss OR
        # the next-idea-prediction loss, build the InterveningIdeaGenerator +
        # GlobalAttention + NextIdeaScorer NOW -- before getOptimizer runs at
        # train start -- so their params join the optimizer and learn. Guarded:
        # both weights 0 builds nothing (no new params, byte-identical).
        if (float(getattr(self, "answer_loss_weight", 0.0) or 0.0) > 0.0
                or float(getattr(self, "predict_next_loss_weight", 0.0)
                         or 0.0) > 0.0):
            try:
                self._reasoning_tooluser(self._reasoning_spaces())
            except Exception:
                pass

        # Thinking Kernel next-op head (§12.6): built eagerly when its loss
        # weight is positive, for the same reason -- getOptimizer must see the
        # params at train start. Weight 0 builds nothing (byte-identical).
        if float(getattr(self, "thinking_loss_weight", 0.0) or 0.0) > 0.0:
            from thinking import NextOpPolicy
            self._next_op_policy = NextOpPolicy()

        return cfg

    def create(self, **kwargs):
        """Override in subclasses to build model architecture.

        ``BaseModel.create`` is a no-op; ``BasicModel.create`` wires
        the actual space stack.
        """
        pass

    def getOptimizer(self, lr=0.01):
        """Build an Adam optimizer over all trainable parameters.

        Walks ``self.spaces`` collecting params via each space's
        ``getParameters()`` (which excludes alpha-managed params).
        Dedup by tensor ``data_ptr`` so a parameter owned by multiple
        modules is only optimized once.

        When ``trainEmbedding`` is NONE or AR, embedding parameters
        are excluded from the optimizer.
        """
        params = []
        seen = set()
        for s in self.spaces:
            for p in s.getParameters():
                if p.data_ptr() not in seen:
                    seen.add(p.data_ptr())
                    params.append(p)
        # Reading / global attention (doc/specs/reading-attention.md): the
        # learned attention readouts are model-level nn.Modules (NOT Spaces), so
        # the ``self.spaces`` walk above misses their params -- collect them
        # explicitly (deduped). Absent / gate-off => the attr is None => no
        # extra params => byte-identical optimizer state.
        # ``_intervening_generator`` (the reasoning query head) + ``_reasoning_ga``
        # are likewise model-level nn.Modules (Phase C); they must be optimized so
        # the answer-policy loss can actually train the soft route. The data_ptr
        # dedup handles ``_reasoning_ga`` aliasing ``global_attention``; gate-off
        # => the attrs are unset => no extra params (byte-identical).
        for _att_name in ("reading_attention", "global_attention",
                          "_intervening_generator", "_reasoning_ga",
                          "_predict_next_scorer", "_next_op_policy"):
            _att = getattr(self, _att_name, None)
            if _att is not None:
                for p in _att.parameters():
                    if p.requires_grad and p.data_ptr() not in seen:
                        seen.add(p.data_ptr())
                        params.append(p)
        # A4 (2026-06-06 parallel-conceptual-recurrence): the per-stage
        # ConceptualCombine is HELD BY its ConceptualSpace (registered in
        # ``cs.layers`` + ``cs.params`` at build time), so the ``self.spaces``
        # walk above already collects its LDU / butterfly weights -- no
        # separate model-level collection is needed.
        # Identify the perceptual-embedding params and the codebook's
        # sparse-grad preference so we can route them to SparseAdam.
        sparse_ptrs = set()
        embedding_ptrs = set()
        if hasattr(self, 'perceptualSpace') and isinstance(
                self.perceptualSpace.vocabulary, Embedding):
            voc = self.perceptualSpace.vocabulary
            # Mark large embedding codebooks for sparse-grad lookup;
            # _lookup_modality reads voc.sparse_grad.
            try:
                voc.use_sparse_grad()
            except Exception:
                pass
            sparse_grad = bool(getattr(voc, 'sparse_grad', False))
            for p in voc.embedding_parameters():
                embedding_ptrs.add(p.data_ptr())
                if sparse_grad:
                    sparse_ptrs.add(p.data_ptr())
        # Exclude embedding params entirely when trainEmbedding is NONE / AR
        if not getattr(self, 'optimize_embedding', False):
            params = [p for p in params if p.data_ptr() not in embedding_ptrs]
            sparse_ptrs.clear()

        # The shared aligned ConceptualSpace dictionary has a stronger memory
        # contract than an ordinary sparse embedding. Stock SparseAdam still
        # allocates TWO full-size dense moments, which would add ~8 GiB for the
        # 1M x 1032 fp32 table. Its lookup surface emits sparse row gradients,
        # so route that one physical Parameter to compact-prefix RowLocalAdam.
        row_local_ptrs = set()
        seen_codebooks = set()
        for cs in list(getattr(self, "conceptualSpaces", None) or ()):
            cb = getattr(cs, "similarity_codebook", None)
            if cb is None or id(cb) in seen_codebooks:
                continue
            seen_codebooks.add(id(cb))
            W = getattr(cb, "W", None)
            if (bool(getattr(cb, "sparse_lookup_grad", False))
                    and isinstance(W, nn.Parameter) and W.requires_grad):
                row_local_ptrs.add(W.data_ptr())
        if row_local_ptrs & sparse_ptrs:
            raise RuntimeError(
                "a parameter cannot be owned by both SparseAdam and "
                "RowLocalAdam")

        # Split the remaining params into ordinary dense Adam, perceptual
        # SparseAdam, and conceptual RowLocalAdam groups. SparseAdam reduces
        # arithmetic but not state size; RowLocalAdam is reserved for the
        # monotonic aligned concept-row namespace where compact moments are a
        # correctness-enforced memory requirement.
        # CUDA-graph capture (the brick body; test_brick_no_sync)
        # requires the optimizer keep its step counter on-device:
        # stock Adam's _multi_tensor_adam does _get_value(step).item()
        # per param otherwise -- one cudaMemcpyDtoH per param per step.
        # capturable=True keeps step on-device. Gated to actual CUDA
        # params (no-op/overhead on CPU/MPS; SparseAdam has no such flag).
        _cap = any(getattr(p, "is_cuda", False) for p in params)
        # Output-head LR scale: the regression readout is ill-conditioned (the
        # class-flipping feature direction is low-variance) and co-adapts with
        # the upstream features during joint training, so a single shared LR
        # leaves it under-converged at the configured epoch budget (XOR sticks
        # near the feature mean ~0.5 despite the features being separable).
        # <OutputSpace><lrScale> puts the OutputSpace params in their own Adam
        # group at lr*scale; default 1.0 => one group, byte-identical.
        try:
            out_lr_scale = float(
                TheXMLConfig.space("OutputSpace", "lrScale", default=1.0))
        except Exception:
            out_lr_scale = 1.0
        output_ptrs = set()
        if out_lr_scale != 1.0 and hasattr(self, "outputSpace"):
            output_ptrs = {p.data_ptr()
                           for p in self.outputSpace.getParameters()}

        def _dense_arg(dense_params):
            """Plain param list (scale==1.0) or two LR-scaled groups (the
            OutputSpace head at lr*scale, the rest at lr)."""
            if not output_ptrs:
                return dense_params
            head = [p for p in dense_params if p.data_ptr() in output_ptrs]
            rest = [p for p in dense_params if p.data_ptr() not in output_ptrs]
            groups = []
            if rest:
                groups.append({"params": rest, "lr": lr})
            if head:
                groups.append({"params": head, "lr": lr * out_lr_scale})
            return groups

        sparse_params = [p for p in params if p.data_ptr() in sparse_ptrs]
        row_local_params = [
            p for p in params if p.data_ptr() in row_local_ptrs]
        dense_params = [
            p for p in params
            if p.data_ptr() not in sparse_ptrs
            and p.data_ptr() not in row_local_ptrs]
        optimizers = []
        if dense_params:
            optimizers.append(Adam(
                _dense_arg(dense_params), lr=lr, capturable=_cap))
        if row_local_params:
            # Persistent moments are BF16 (same exponent range as fp32), while
            # RowLocalAdam promotes every touched row to fp32 for the actual
            # Adam update. At full 1M x 1032 occupancy this halves the two
            # moments from 8.06 GiB to 4.03 GiB without fp16 underflow.
            optimizers.append(RowLocalAdam(
                row_local_params, lr=lr, moment_dtype=torch.bfloat16))
        if sparse_params:
            optimizers.append(SparseAdam(sparse_params, lr=lr))
        if not optimizers:
            raise RuntimeError("model exposes no trainable optimizer parameters")
        optimizer = (optimizers[0] if len(optimizers) == 1
                     else MultiOptimizer(optimizers))
        pending = getattr(self, "_pending_optimizer_state", None)
        if pending is not None:
            saved_manifest = getattr(
                self, "_pending_optimizer_manifest", None)
            legacy_shapes = getattr(
                self, "_pending_legacy_state_shapes", None) or {}
            require_match = bool(getattr(
                self, "_pending_optimizer_require_match", False))
            reset_wholespace = bool(getattr(
                self, "_pending_optimizer_reset_wholespace", False))
            try:
                live_manifest = build_optimizer_param_manifest(
                    optimizer, self.named_parameters())
                if saved_manifest is None:
                    saved_manifest = infer_legacy_optimizer_param_manifest(
                        live_manifest, legacy_shapes, pending)
                remapped = remap_optimizer_state_by_name(
                    pending, saved_manifest, optimizer.state_dict(),
                    live_manifest,
                    reset_wholespace=reset_wholespace)
                optimizer.load_state_dict(remapped.state)
                self._normalize_optimizer_state_shapes(optimizer)
                TheMessage(
                    f"[{self.name}] {remapped.diagnostics.message()}")
            except ValueError as exc:
                message = (
                    f"[{self.name}] Optimizer checkpoint was not restored: "
                    f"{exc}")
                if require_match:
                    raise ValueError(message) from exc
                TheMessage(message)
            finally:
                self._pending_optimizer_state = None
                self._pending_optimizer_manifest = None
                self._pending_legacy_state_shapes = None
                self._pending_optimizer_reset_wholespace = False
                self._pending_optimizer_require_match = False
        # PartSpace may have to install a previously unseen byte before the
        # first codebook gather of a batch. Keep the current optimizer as a
        # non-module boundary context so that eager growth can migrate Adam
        # state and ownership instead of orphaning the old Parameter.
        part_space = getattr(self, "perceptualSpace", None)
        if part_space is not None:
            object.__setattr__(part_space, "_radix_optimizer", optimizer)
        return optimizer

    def rebuild_optimizer(self):
        """Rebuild the main optimizer after codebook expansion.

        Reads the learning rate from the existing optimizer's first
        param group and constructs a fresh ``getOptimizer`` over the
        current parameter set, picking up any newly added rows.
        """
        if self._optimizer is None:
            return
        lr = self._optimizer.param_groups[0]['lr']
        self._optimizer = self.getOptimizer(lr=lr)

    def _aligned_inventory_capacity(self):
        """Return the serial-aligned concept-reference capacity, or ``None``.

        ConceptualSpace owns this address space.  Downstream symbols are
        references into it; WholeSpace rows are upstream properties and are a
        separate inventory whose size may differ.
        """
        if not (getattr(self, "concept_binding", None) == "aligned"
                and bool(getattr(self, "serial", False))):
            return None
        n_concepts = int(getattr(self, "nConceptCodes", 0) or 0)
        if n_concepts <= 0:
            raise ValueError(
                "aligned checkpoint migration requires a positive "
                f"ConceptualSpace inventory; got CS={n_concepts}")
        return n_concepts

    def _aligned_capacity_codebooks(self):
        """Return each physical full-capacity concept Codebook once.

        Some ConceptualSpace dictionaries are registered through compatibility
        aliases, so deduplicate by module identity rather than by state-dict
        path.  WholeSpace property codebooks and legacy ``analysis_store``
        modules never participate in concept-capacity activation.
        """
        capacity = self._aligned_inventory_capacity()
        if capacity is None:
            return []
        candidates = []
        for cs in list(getattr(self, "conceptualSpaces", None) or ()):
            candidates.append(getattr(cs, "similarity_codebook", None))
        out = []
        seen = set()
        for cb in candidates:
            if not isinstance(cb, Codebook) or id(cb) in seen:
                continue
            W = cb.getW()
            if (not torch.is_tensor(W) or W.ndim != 2
                    or int(W.shape[0]) != capacity):
                continue
            vq = getattr(cb, "vq", None)
            if vq is None or not hasattr(vq, "set_active_rows"):
                continue
            seen.add(id(cb))
            out.append(cb)
        return out

    @staticmethod
    def _active_prefix_rows(vq):
        """Validate a contiguous nonempty active mask and return its length."""
        mask = getattr(vq, "active_mask", None)
        W = getattr(vq, "codebook", None)
        if (not torch.is_tensor(mask) or mask.dtype != torch.bool
                or not torch.is_tensor(W) or mask.ndim != 1
                or int(mask.shape[0]) != int(W.shape[0])):
            raise RuntimeError(
                "aligned VQ active-mask shape/dtype drift: "
                f"mask={None if not torch.is_tensor(mask) else tuple(mask.shape)}, "
                f"codebook={None if not torch.is_tensor(W) else tuple(W.shape)}")
        n = int(mask.sum().item())
        if n < 1:
            raise RuntimeError("aligned VQ must keep at least one active row")
        if (not bool(mask[:n].all().item())
                or (n < int(mask.shape[0])
                    and bool(mask[n:].any().item()))):
            raise RuntimeError(
                "aligned VQ active rows must be one contiguous prefix")
        cached = int(getattr(vq, "_active_rows_count", n))
        if cached != n:
            raise RuntimeError(
                "aligned VQ active-prefix cache drift: "
                f"mask has {n} rows, cached count is {cached}")
        return n

    def _configure_aligned_active_codebooks(self):
        """Install the configured active prefix before checkpoint autoload.

        Physical ``nVectors`` is the optimizer-owned capacity. Logical growth
        only mutates full-shape bool masks, so W, EMA tensors and Adam ownership
        never change identity or shape. VQ reads the active prefix only, so each
        new power-of-two width causes one compiled-graph specialization.
        """
        capacity = self._aligned_inventory_capacity()
        if capacity is None:
            return None
        cs_active = int(TheXMLConfig.space(
            "ConceptualSpace", "activeVectors", default=capacity) or capacity)
        if not (1 <= cs_active <= capacity):
            raise ValueError(
                "ConceptualSpace.activeVectors must be within physical "
                f"concept capacity: active={cs_active}, capacity={capacity}")
        codebooks = self._aligned_capacity_codebooks()
        if not codebooks:
            raise RuntimeError(
                "ConceptualSpace.activeVectors is configured, but no "
                "full-capacity concept VQ Codebooks were found")
        for cb in codebooks:
            cb.vq.set_active_rows(cs_active)
            # This physical table is optimizer-owned at its final configured
            # capacity. Logical growth reveals rows through the full-shape VQ
            # mask below; it must never replace W and orphan RowLocalAdam or a
            # compiled graph. PartSpace's structural radix store remains the
            # intentionally dynamic exception and is not returned here.
            cb.freeze_capacity("aligned ConceptualSpace codebook")
        self._active_inventory_rows = cs_active
        return cs_active

    def _ensure_aligned_active_rows(self, required_rows):
        """Reveal a power-of-two prefix across every aligned VQ in place."""
        capacity = self._aligned_inventory_capacity()
        if capacity is None:
            return None
        required = max(1, int(required_rows))
        if required > capacity:
            raise RuntimeError(
                f"aligned codebook physical capacity exhausted: required "
                f"{required} rows, capacity {capacity}; no mask was changed")
        current = int(getattr(self, "_active_inventory_rows", 0) or 0)
        if current < 1:
            counts = {
                self._active_prefix_rows(cb.vq)
                for cb in self._aligned_capacity_codebooks()
            }
            if len(counts) != 1:
                raise RuntimeError(
                    f"aligned VQ active-prefix mismatch: {sorted(counts)}")
            current = counts.pop()
        if required <= current:
            return current
        codebooks = self._aligned_capacity_codebooks()
        counts = {self._active_prefix_rows(cb.vq) for cb in codebooks}
        if counts != {current}:
            raise RuntimeError(
                "aligned VQ active-prefix mismatch before growth: "
                f"model={current}, codebooks={sorted(counts)}; no mask was "
                "changed")
        target = min(capacity, 1 << (required - 1).bit_length())
        target = max(current, target)
        for cb in codebooks:
            cb.vq.set_active_rows(target)
        self._active_inventory_rows = target
        TheMessage(
            f"[{self.name}] Activated aligned codebook prefix "
            f"{current} -> {target} rows (physical capacity {capacity}; "
            f"Parameter/optimizer shapes unchanged; compiled VQ paths "
            f"specialize on the new active width)")
        return target

    def _resync_aligned_active_codebooks(self):
        """Validate loaded masks and cover every restored concept id."""
        if self._aligned_inventory_capacity() is None:
            return None
        codebooks = self._aligned_capacity_codebooks()
        counts = {self._active_prefix_rows(cb.vq) for cb in codebooks}
        if len(counts) != 1:
            raise ValueError(
                f"checkpoint aligned VQ active-prefix mismatch: "
                f"{sorted(counts)}")
        active = counts.pop()
        self._active_inventory_rows = active
        required = 1
        for cs in list(getattr(self, "conceptualSpaces", None) or ()):
            alloc = getattr(cs, "_concept_allocator", None)
            if alloc is None:
                continue
            # next_id is one past every restored stable concept handle.
            required = max(required, int(getattr(alloc, "next_id", 1) or 1))
            for layer in getattr(alloc, "_layers", {}).values():
                tensor_rows = getattr(layer, "_tensor_rows", None) or {}
                if tensor_rows:
                    required = max(
                        required,
                        max(int(v) for v in tensor_rows.values()) + 1)
        return self._ensure_aligned_active_rows(required)

    def _canonicalize_shared_concept_checkpoint_state(self, state,
                                                       model_state):
        """Make every shared CS Codebook alias load stage 0's state.

        Before the aligned/property-basis architecture shared its conceptual
        dictionary, each subsymbolic stage saved an independent W plus VQ/EMA
        tables.  Stage 0 is authoritative during this migration: it owns the
        word autobind and the late conceptual cutover.  Populate every live
        alias from that one source before shape expansion/load_state_dict so a
        legacy divergent checkpoint loads strictly without letting later
        aliases overwrite the shared physical module.
        """
        if not (bool(getattr(self, "wholePropertyBasis", False))
                and getattr(self, "concept_binding", None) == "aligned"):
            return 0
        spaces = list(getattr(self, "conceptualSpaces", None) or ())
        if not spaces:
            return 0
        shared = getattr(spaces[0], "similarity_codebook", None)
        if (not isinstance(shared, Codebook)
                or any(getattr(cs, "similarity_codebook", None) is not shared
                       for cs in spaces[1:])):
            return 0

        try:
            named = self.named_modules(remove_duplicate=False)
        except TypeError:  # older torch compatibility
            named = self.named_modules()
        prefixes = [
            name for name, module in named
            if module is shared and name
        ]
        if not prefixes:
            return 0
        preferred = "conceptualSpaces.0.similarity_codebook"
        prefixes.sort(key=lambda name: (
            0 if name == preferred else
            1 if name.startswith("conceptualSpaces.0.") else 2,
            name,
        ))

        replaced = 0
        used_non_stage0 = False
        for suffix in shared.state_dict().keys():
            aliases = [
                f"{prefix}.{suffix}" for prefix in prefixes
                if f"{prefix}.{suffix}" in model_state
            ]
            if not aliases:
                continue
            source_key = next((key for key in aliases if key in state), None)
            if source_key is None:
                continue
            if not source_key.startswith("conceptualSpaces.0."):
                used_non_stage0 = True
            source = state[source_key]
            for key in aliases:
                old = state.get(key)
                if old is not None and old is not source:
                    if (not torch.is_tensor(old) or not torch.is_tensor(source)
                            or int(old.data_ptr()) != int(source.data_ptr())):
                        replaced += 1
                state[key] = source

        if replaced or used_non_stage0:
            fallback = (
                " (stage 0 was absent for at least one tensor; the earliest "
                "available alias was used)" if used_non_stage0 else "")
            warnings.warn(
                "Shared ConceptualSpace checkpoint migration: selected the "
                "stage-0 similarity dictionary as canonical and replaced "
                f"{replaced} divergent stage/alias tensor(s){fallback}.",
                stacklevel=2,
            )
        return replaced

    def _expand_aligned_codebook_checkpoint_state(self, state, model_state):
        """Prefix-load a smaller aligned codebook into a larger build.

        This runs during autoload: the model has its final configured shapes,
        but no optimizer or compiled callable exists yet.  Existing rows keep
        their exact ids and values.  New rows retain their freshly constructed
        initialization (including VQ EMA/norm buffers); Adam's corresponding
        moments are padded separately by
        :meth:`_normalize_optimizer_state_shapes`.

        Only row-aligned state belonging to one of the ConceptualSpace
        dictionaries returned by :meth:`_aligned_capacity_codebooks` is
        eligible.  WholeSpace property state and legacy ``analysis_store``
        state are intentionally excluded, even when their row counts happen
        to equal the concept capacity.
        """
        capacity = self._aligned_inventory_capacity()
        if capacity is None:
            return 0

        try:
            named = self.named_modules(remove_duplicate=False)
        except TypeError:  # older torch compatibility
            named = self.named_modules()
        eligible = set()
        concept_codebook_ids = {
            id(cb) for cb in self._aligned_capacity_codebooks()
        }
        row_keys = ("W", "vq.cluster_size", "vq.embed_avg",
                    "vq._b_norms_sq", "vq.active_mask")
        for name, module in named:
            if (not isinstance(module, Codebook)
                    or id(module) not in concept_codebook_ids):
                continue
            W = module.getW()
            if (not torch.is_tensor(W) or W.ndim != 2
                    or int(W.shape[0]) != capacity):
                continue
            prefix = f"{name}." if name else ""
            eligible.update(prefix + suffix for suffix in row_keys)

        # Duplicate registered paths often point at the same live storage.
        # The model was freshly constructed at final capacity, so preserve its
        # initialized tail and copy the learned prefix DIRECTLY into that live
        # storage once.  Point every state-dict alias back at the same detached
        # view.  Cloning a 1M x 512 W and EMA table here would add roughly
        # 4 GiB of avoidable checkpoint-migration peak memory.
        prepared_by_storage = {}
        source_storage_by_target = {}
        expanded = 0
        old_sizes = set()
        for key in sorted(eligible):
            if key not in state or key not in model_state:
                continue
            saved = state[key]
            live = model_state[key]
            if (not torch.is_tensor(saved) or not torch.is_tensor(live)
                    or saved.ndim == 0 or saved.ndim != live.ndim
                    or tuple(saved.shape[1:]) != tuple(live.shape[1:])
                    or int(saved.shape[0]) >= int(live.shape[0])):
                continue
            if int(live.shape[0]) != capacity:
                continue
            storage_key = (
                int(live.data_ptr()), tuple(live.shape), live.dtype,
                live.device.type, live.device.index,
            )
            grown = prepared_by_storage.get(storage_key)
            if grown is None:
                grown = live.detach()
                with torch.no_grad():
                    grown[:saved.shape[0]].copy_(
                        saved.detach().to(device=grown.device,
                                          dtype=grown.dtype))
                prepared_by_storage[storage_key] = grown
                source_storage_by_target[storage_key] = int(saved.data_ptr())
            else:
                # Alias paths must describe the same learned prefix.  A
                # disagreement is checkpoint corruption, not a migration.
                if (int(saved.data_ptr()) !=
                        source_storage_by_target[storage_key]
                        and not torch.equal(
                        grown[:saved.shape[0]],
                        saved.detach().to(device=grown.device,
                                          dtype=grown.dtype))):
                    raise ValueError(
                        f"aligned checkpoint aliases disagree for {key}")
            state[key] = grown
            old_sizes.add(int(saved.shape[0]))
            expanded += 1

        if expanded:
            TheMessage(
                f"[{self.name}] Expanded aligned codebook checkpoint rows "
                f"{sorted(old_sizes)} -> {capacity} across {expanded} state "
                f"keys ({len(prepared_by_storage)} physical tensors); "
                f"learned prefixes copied in place and initialized tails "
                f"preserved")
        return expanded

    def _expand_partspace_codebook_checkpoint_state(self, state, model_state):
        """Prefix-load a smaller radix PS table into a larger initial build.

        Runtime growth happens only at a batch boundary, but construction-time
        capacity increases are safe before optimizer/compiled ownership exists.
        Keep the checkpoint's learned rows exactly and retain the live model's
        initialization for the newly configured tail.
        """
        part_space = getattr(self, "perceptualSpace", None)
        basis = getattr(getattr(part_space, "subspace", None), "what", None)
        store = getattr(part_space, "percept_store", None)
        if (not isinstance(basis, Codebook) or store is None
                or getattr(store, "_basis", None) is not basis):
            return 0
        live_w = basis.getW()
        if not torch.is_tensor(live_w) or live_w.ndim != 2:
            return 0
        capacity = int(live_w.shape[0])

        try:
            named = self.named_modules(remove_duplicate=False)
        except TypeError:
            named = self.named_modules()
        prefixes = [
            f"{name}." if name else ""
            for name, module in named if module is basis
        ]
        if not prefixes:
            return 0
        local_state = basis.state_dict()
        eligible = set()
        for suffix, value in local_state.items():
            if (torch.is_tensor(value) and value.ndim > 0
                    and int(value.shape[0]) == capacity):
                eligible.update(prefix + suffix for prefix in prefixes)

        # As above for CS, the final-capacity live allocation already owns the
        # desired initialized tail.  Prefix-copy in place so migration never
        # clones the entire percept table merely to feed load_state_dict.
        prepared_by_storage = {}
        expanded = 0
        old_sizes = set()
        for key in sorted(eligible):
            if key not in state or key not in model_state:
                continue
            saved = state[key]
            live = model_state[key]
            if (not torch.is_tensor(saved) or not torch.is_tensor(live)
                    or saved.ndim == 0 or saved.ndim != live.ndim
                    or tuple(saved.shape[1:]) != tuple(live.shape[1:])
                    or int(saved.shape[0]) >= int(live.shape[0])
                    or int(live.shape[0]) != capacity):
                continue
            storage_key = (
                int(live.data_ptr()), tuple(live.shape), live.dtype,
                live.device.type, live.device.index,
            )
            grown = prepared_by_storage.get(storage_key)
            if grown is None:
                grown = live.detach()
                with torch.no_grad():
                    grown[:saved.shape[0]].copy_(saved.detach().to(
                        device=grown.device, dtype=grown.dtype))
                prepared_by_storage[storage_key] = grown
            elif not torch.equal(
                    grown[:saved.shape[0]], saved.detach().to(
                        device=grown.device, dtype=grown.dtype)):
                raise ValueError(
                    f"PartSpace checkpoint aliases disagree for {key}")
            state[key] = grown
            old_sizes.add(int(saved.shape[0]))
            expanded += 1
        if expanded:
            TheMessage(
                f"[{self.name}] Expanded PartSpace checkpoint rows "
                f"{sorted(old_sizes)} -> {capacity} across {expanded} "
                "state key(s); learned prefix copied in place and initialized "
                "tail preserved")
        return expanded

    @staticmethod
    def _optimizer_leaves(optimizer):
        """Yield concrete optimizer wrappers from a MultiOptimizer tree."""
        children = getattr(optimizer, "optimizers", None)
        if children is None:
            yield optimizer
            return
        for child in children:
            yield from BaseModel._optimizer_leaves(child)

    def _normalize_optimizer_state_shapes(self, optimizer):
        """Validate or pad loaded row-wise optimizer moments.

        ``Optimizer.load_state_dict`` intentionally maps state by parameter
        order and accepts shape-drifted moment tensors.  The failure otherwise
        appears only on the first Adam step.  Normalize immediately after load:
        preserve the learned prefix, zero every new row, and reject any shape
        change that is not a pure first-axis capacity extension. RowLocalAdam
        is the deliberate exception: its prefix-shaped moments stay compact
        and grow geometrically only when a later sparse gradient touches a new
        row. Expanding them here would recreate the multi-gigabyte state this
        optimizer exists to avoid.
        """
        padded = 0
        for leaf in self._optimizer_leaves(optimizer):
            state_by_param = getattr(leaf, "state", None)
            if state_by_param is None:
                continue
            row_local = bool(getattr(leaf, "row_local_state", False))
            for param, param_state in list(state_by_param.items()):
                if not isinstance(param, nn.Parameter):
                    continue
                target_shape = tuple(param.shape)
                for name, value in list(param_state.items()):
                    if not torch.is_tensor(value) or value.ndim == 0:
                        continue
                    if tuple(value.shape) == target_shape:
                        continue
                    # Adam's step can be a one-element tensor on some torch
                    # versions/backends; it is scalar state, not row state.
                    if str(name) == "step" and value.numel() == 1:
                        continue
                    if (value.ndim != param.ndim
                            or tuple(value.shape[1:]) != target_shape[1:]
                            or int(value.shape[0]) > int(target_shape[0])):
                        raise ValueError(
                            f"optimizer state {name!r} shape "
                            f"{tuple(value.shape)} cannot migrate to parameter "
                            f"shape {target_shape}; only first-axis growth is "
                            f"supported")
                    if row_local:
                        # Valid compact checkpoint state. The optimizer owns
                        # its next-power-of-two growth boundary.
                        continue
                    grown = value.new_zeros(target_shape)
                    grown[:value.shape[0]].copy_(value)
                    param_state[name] = grown
                    padded += 1
        if padded:
            TheMessage(
                f"[{self.name}] Zero-padded {padded} optimizer moment "
                f"tensor(s) for larger configured inventories")
        return padded

    def _checkpoint_path(self, suffix=None):
        """Resolve the configured checkpoint path, optionally adding a suffix.

        ``BASIC_WEIGHTS_PATH`` is an explicit run-local override. Relative
        override paths resolve from the project root (so ``output/foo.ckpt``
        is stable regardless of the XML's directory); XML paths retain their
        historical config-directory semantics when the override is unset.

        Falls back to ``output/<config-stem>.ckpt`` when no ``weightsPath``
        is set in the XML. Config-specific names prevent unrelated BasicModel
        architectures from silently sharing ``output/BasicModel.ckpt``.
        The ``suffix`` is inserted before the extension so emergency /
        autosave variants can coexist with the canonical checkpoint.
        """
        override = os.environ.get("BASIC_WEIGHTS_PATH", "").strip()
        if override:
            expanded = os.path.expanduser(override)
            path = (expanded if os.path.isabs(expanded)
                    else os.path.join(ProjectPaths.PROJECT_DIR, expanded))
            path = os.path.abspath(path)
        else:
            path = TheXMLConfig.get("architecture.weightsPath", None)
            if path:
                path = self._resolve_artifact_path(path)
            else:
                config_path = getattr(self, "_config_path", None)
                stem = (Path(config_path).stem if config_path
                        else str(self.name))
                path = ProjectPaths.output_path(f"{stem}.ckpt")
        if suffix:
            root, ext = os.path.splitext(path)
            path = f"{root}.{suffix}{ext or '.ckpt'}"
        return path

    def save_training_checkpoint(self, reason="checkpoint", suffix=None):
        """Save the model checkpoint (single integrated artifact).

        Resolves the .ckpt path (with optional suffix) and writes the
        full bundle via ``save_weights``: parameters, buffers,
        embeddings, and BPE state. ``reason`` appears in the log only;
        the on-disk filename is unaffected.
        """
        path = self._checkpoint_path(suffix=suffix)
        TheMessage(f"[{self.name}] Saving training checkpoint ({reason})")
        self.save_weights(path)
        return path

    def _maybe_save_periodic_checkpoint(self):
        interval = int(getattr(self, "checkpoint_every_batches", 0) or 0)
        if interval <= 0:
            return
        step = int(getattr(self, "_training_step_count", 0) or 0)
        if step > 0 and step % interval == 0:
            self.save_training_checkpoint(reason=f"batch {step}")

    def _assert_finite_train_state(self, what):
        """Fail at the update site when a parameter or gradient goes non-finite.

        Gated by MODEL_DEBUG (see util.py). The inner checks are ``assert``
        statements so ``python -O`` strips them entirely; MODEL_DEBUG also
        short-circuits the iteration so the per-parameter loop is skipped
        without -O.
        """
        if not _util.MODEL_DEBUG:
            return
        for name, param in self.named_parameters():
            if param is None:
                continue
            pdata = param.detach()
            assert torch.isfinite(pdata).all(), (
                f"Non-finite parameter {name!r} {what}: "
                f"{int((~torch.isfinite(pdata)).sum().item())}/{pdata.numel()} "
                f"entries are nan/inf."
            )
            if param.grad is not None:
                grad = param.grad.detach()
                # ``torch.isfinite`` has no SparseCPU/SparseMPS implementation
                # for a COO tensor. Sparse gradients are finite exactly when
                # their stored values are finite; absent entries are zeros.
                checked_grad = (
                    grad.coalesce().values() if grad.is_sparse else grad)
                assert torch.isfinite(checked_grad).all(), (
                    f"Non-finite gradient for {name!r} {what}: "
                    f"{int((~torch.isfinite(checked_grad)).sum().item())}/"
                    f"{checked_grad.numel()} "
                    f"entries are nan/inf."
                )

    def run(self, numTrials=1, numEpochs=1, batchSize=10, lr=0.001, profile=None):
        """Run multiple independent trials, recreating the model each time.

        Each trial calls create_from_config() to rebuild from scratch so
        results are statistically independent.  If the model was already
        configured by the caller (e.g. manually built models without
        _config_path), trial 0 skips recreation and uses the model as-is.
        """
        acc = np.zeros([numTrials, numEpochs])
        has_config = hasattr(self, '_config_path') and self._config_path is not None
        already_configured = len(list(self.parameters())) > 0
        TheMessage(f"\n\n==== {self.name} ====")
        for trial in range(numTrials):
            TheMessage(f"\nTrial [{trial + 1}/{numTrials}]")
            if has_config and (trial > 0 or not already_configured):
                self.create_from_config(self._config_path, data=self._config_data)
            acc[trial, :] = self.runTrial(
                numEpochs=numEpochs, batchSize=batchSize, lr=lr,
                profile=profile,
            )

        np.savetxt(ProjectPaths.output_path(f"{self.name}.csv"), np.array(acc), delimiter=",")
        return acc

    def paramUpdate(self):
        """Delegate ergodic in-place parameter updates to all spaces.

        Each space's ``paramUpdate`` performs its own annealing /
        renormalization step (e.g. codebook re-projection); not all
        parameter motion is gradient-driven.
        """
        for s in self.spaces:
            s.paramUpdate()
        # A4: each ConceptualCombine is registered in its ``cs.layers``, so the
        # space walk above already cascades ``paramUpdate()`` to it (the dense
        # full/last ergodic path anneals; the butterfly path is a no-op).

    def _advance_codebook_parameter_versions(self):
        """Publish one model-wide parameter epoch after a write barrier.

        Every Space receives the same epoch even when a particular codebook
        had no gradient in this step.  Conservative invalidation is cheap and
        guarantees that a pipeline execution can compare every selected
        carrier against one model-wide ``parameter_version``.
        """
        version = int(getattr(self, '_parameter_epoch', 0)) + 1
        self._parameter_epoch = version
        for space in self.spaces:
            # Compatibility/custom Spaces may still skip Space.__init__.
            # Normal Spaces and SymbolSpace initialize this metadata and
            # advance together at the write barrier.
            if '_codebook_parameter_version' not in vars(space):
                continue
            setter = getattr(space, 'set_codebook_parameter_version', None)
            if callable(setter):
                setter(version)
        return version

    def set_sigma(self, sigma):
        """Propagate exploration meta-parameters to all spaces.

        ``sigma`` controls per-space exploration noise (typically the
        Gaussian sample width in codebook lookup). Spaces that don't
        use exploration may treat this as a no-op.
        """
        for s in self.spaces:
            s.set_sigma(sigma)
        # A4: each ConceptualCombine is registered in its ``cs.layers``, so the
        # space walk above already cascades ``set_sigma()`` to it (no-op on the
        # butterfly path; the dense ergodic path uses it).

    def _get_embedding(self):
        """Return the Embedding instance if this model uses one, else None.

        Stage 7 (doc/plans/2026-05-27-perceptstore-meta-taxonomy-
        reentrancy.md): in radix-mode the ``vocabulary`` property
        returns a PerceptStore, but the orthographic Embedding still
        lives on ``perceptualSpace.subspace.what``; look there first
        so the rest of the orthographic API keeps working in radix mode.
        """
        if not hasattr(self, 'perceptualSpace'):
            return None
        _what = getattr(self.perceptualSpace.subspace, 'what', None)
        if isinstance(_what, Embedding):
            return _what
        _vocab = self.perceptualSpace.vocabulary
        if isinstance(_vocab, Embedding):
            return _vocab
        return None

    # -- Reasoning Methods --------------------------------------------

    def _get_truth_layer(self):
        """Return the TruthLayer if available, else None.

        The TruthLayer now lives on ``SymbolSpace``; when the grammar
        path that builds SymbolSpace is disabled there is no layer.
        """
        return self.symbolSpace.truth_layer if self.symbolSpace is not None else None

    def _get_basis(self):
        """Return the WholeSpace property Basis, if available."""
        ws = getattr(self, 'wholeSpace', None)
        if ws is None:
            return None
        return getattr(getattr(ws, 'subspace', None), 'basis', None)

    @torch.no_grad()
    def _clamp_symbolic_codebook(self):
        """Keep the WholeSpace property codebook inside the [0,1] pair box.

        The compatibility method name is retained for external callers.
        Called after ``optimizer.step()`` when the property basis is monotonic;
        gradient updates can push entries outside the box, so clamp directly
        on the Parameter data before the next forward pass.
        """
        basis = self._get_basis()
        if basis is None or not getattr(basis, 'monotonic', False):
            return
        W = getattr(basis, 'W', None)
        if W is None or not isinstance(W, torch.Tensor):
            return
        W.data.clamp_(0.0, 1.0)

    @torch.no_grad()
    def _normalize_conceptual_codebooks(self):
        """Re-project updated concept rows onto their signed unit sphere.

        Concept atoms are directions; certainty lives in the scalar that
        activates them.  The indexed aligned dictionary therefore needs a
        row-local constraint after Adam updates so ``activation * atom``
        remains in ``[-1, 1]`` without a tanh seam.  Sparse gradients expose
        exactly the rows touched by this sentence, keeping the operation
        proportional to use rather than to the million-row capacity.
        """
        seen = set()
        for cs in list(getattr(self, "conceptualSpaces", None) or ()):
            cb = getattr(cs, "similarity_codebook", None)
            if cb is None or id(cb) in seen:
                continue
            seen.add(id(cb))
            W = getattr(cb, "W", None)
            if not isinstance(W, torch.Tensor) or W.ndim < 2:
                continue
            grad = getattr(W, "grad", None)
            if grad is None:
                continue
            if grad.layout == torch.sparse_coo:
                rows = grad.coalesce().indices()[0].long()
                if rows.numel() == 0:
                    continue
                updated = F.normalize(
                    W.index_select(0, rows).float(), p=2, dim=-1,
                    eps=1e-8).to(dtype=W.dtype)
                W.index_copy_(0, rows, updated)
            else:
                # Dense concept dictionaries occur only in the smaller
                # compatibility models; their full projection is affordable.
                W.copy_(F.normalize(W.float(), p=2, dim=-1,
                                    eps=1e-8).to(dtype=W.dtype))

    @torch.no_grad()
    def isConsistent(self):
        """Analyze the TruthSet for internal consistency.

        Folds all stored truths into a single summary vector via
        ``Ops.disjunction``. Conflicting +/-  assertions cancel
        dimensions. (The 2026-05-01 syntactic-layer refactor removed
        the thin Basis.disjunction wrapper; callers now call the Ops
        kernel directly.)

        Returns:
            dict with keys: consistent (bool), score (float),
            sites (tensor of dim indices below threshold),
            union_vector (tensor).
        """
        # Phase 1 (bivector retirement): the disjunction fold lives on
        # TruthLayer; this is a thin delegator. The basis presence guard
        # stays here (a model-side concept).
        truth_layer = self._get_truth_layer()
        basis = self._get_basis()
        if truth_layer is None or basis is None:
            return {'consistent': True, 'score': 1.0,
                    'sites': torch.tensor([]), 'union_vector': torch.tensor([])}
        return truth_layer.isConsistent()

    @torch.no_grad()
    def ground(self, activation, threshold=0.6):
        """Find the minimal TruthSet subset that entails the query.

        Uses _activation_order() to filter truths by compatible partition
        when partitions are available.

        Returns:
            dict with keys: grounded (bool), basis (list of indices),
            trace (list), confidence (float in [-1, 1]).
        """
        # Phase 1 (bivector retirement): the entailment search is
        # consolidated onto TruthLayer; this is a thin delegator.
        truth_layer = self._get_truth_layer()
        if truth_layer is None:
            return {'grounded': False, 'basis': [], 'trace': [], 'confidence': 0.0}
        return truth_layer.ground(activation, threshold=threshold, model=self)

    @torch.no_grad()
    def isTrue(self, activation):
        """Ground a proposition and return a scalar DoT in [-1, 1].

        Positive = true, negative = false, zero = unknown.
        """
        result = self.ground(activation)
        return 0.0 if not result['grounded'] else result['confidence']

    @torch.no_grad()
    def extrapolate(self, seed_indices=None, max_new=64, attenuation=0.8):
        """Generalize TruthLayer.derive() to all two-argument grammar methods.

        For each pair of stored truths, apply every eligible two-argument
        grammar method. Accept results that preserve or increase luminosity;
        reject those that decrease it.

        Args:
            seed_indices: optional list of truth indices to use as seeds.
            max_new: maximum number of new truths to derive.
            attenuation: DoT scaling for derived truths.

        Returns:
            dict with 'added' (list of new indices) and
            'rejected' (list of (i, j, rule, delta_lum) tuples).
        """
        # Phase 1 (bivector retirement): pairwise grammar extrapolation
        # is consolidated onto TruthLayer; this is a thin delegator.
        truth_layer = self._get_truth_layer()
        if truth_layer is None:
            return {'added': [], 'rejected': []}
        return truth_layer.extrapolate(
            model=self, seed_indices=seed_indices,
            max_new=max_new, attenuation=attenuation)

    # -- Contemplative Awareness Characterizations ---------------------

    # -- Contemplative Awareness Characterizations ---------------------
    # The measure family (Contiguous / Continuous / Peaceful / Area /
    # Luminosity) and its hoc_shape back-projection machinery have been
    # extracted to bin/Mereology.py.  BaseModel inherits from Mereology
    # (mixin order: Mereology before nn.Module) so the methods remain
    # callable as model.Contiguous() / etc.


    def Done(self):
        """Buddhahood (Non-Meditation / Resonance).

        The perfection of Contiguous, Continuous, and Peaceful: the
        elimination of dissonance.

        Characterisation -- non-meditation / resonance:
          * Dissonance manifests as something to learn -- a non-zero
            gradient signal indicating mismatch between model and world.
          * It is *not* the case that knowing everything is required to
            remove dissonance, because the attempt to know often creates
            dissonance (reification, attachment to views).
          * Done() holds when the error function is relatively small in
            all cases: no region of input space produces a large loss
            spike.  The model has nothing more to learn -- not because it
            knows everything, but because it is at peace with what it
            does not know.

        Computationally, Done() should verify that:
          1. Contiguous() holds (one-pointed awareness),
          2. Continuous() holds (smooth awareness),
          3. Peaceful() holds (balanced feelings),
          4. The max loss across a representative sample is below a
             configured resonance threshold.
        """
        raise NotImplementedError

    def save_weights(self, path=None):
        """Persist all model state to a single .ckpt: parameters, buffers,
        embedding vectors, vocabulary mappings, BPE codebook, optimizer,
        RNG state, counters, corpus manifest, and mid-epoch cursor.

        The integrated-weights architecture: one XML config + one
        checkpoint, no separate .kv embedding artifact. The checkpoint
        carries:

        * ``state_dict``: all nn.Parameter + register_buffer state
          (model weights, embeddings, TruthLayer rows, etc.).
        * ``vocab_extras``: Python-side WordVectors mappings that don't
          fit ``state_dict`` (``index_to_key``, ``counts``, ``total_count``).
        * ``bpe_extras``: ChunkLayer merge table and vocabulary (pure
          Python state — merges list, vocab dict, id_to_bytes mapping,
          counters).
        * ``structural_extras``: the non-Module concept allocator and its
          identity/idempotency registries. Without this sidecar a resumed
          model would re-mint word wholes and concepts even though their
          tensor rows survived.
        """
        if path is None:
            path = os.path.join(ProjectPaths.OUTPUT_DIR, "weights.ckpt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        optimizer = getattr(self, "_optimizer", None)
        optimizer_manifest = (
            build_optimizer_param_manifest(
                optimizer, self.named_parameters())
            if optimizer is not None else None)
        bundle = {
            "state_dict": dict(self.state_dict()),
            "vocab_extras": self._collect_vocab_extras(),
            "bpe_extras": self._collect_bpe_extras(),
            "structural_extras": self._collect_structural_extras(),
            "optimizer_state": (
                optimizer.state_dict() if optimizer is not None else None),
            "training_state": {
                "training_step_count": int(getattr(
                    self, "_training_step_count", 0) or 0),
                "train_batches_seen": int(getattr(
                    self, "_train_batches_seen", 0) or 0),
                "epoch_batches_seen": int(getattr(
                    self, "_epoch_batches_seen", 0) or 0),
                "checkpoint_batch_size": getattr(
                    self, "_checkpoint_batch_size", None),
                "torch_rng_state": torch.get_rng_state(),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None),
                "mps_rng_state": _mps_rng_state_or_none(),
                "data_manifest": getattr(
                    getattr(self, "inputSpace", None), "data", None
                ).source_manifest if getattr(
                    getattr(self, "inputSpace", None), "data", None
                ) is not None else None,
            },
        }
        if bool(getattr(self, "wholePropertyBasis", False)):
            bundle = stamp_checkpoint_schema(
                bundle, optimizer_param_names=optimizer_manifest)
        elif optimizer_manifest is not None:
            # Parameter names are safe/useful for legacy resumes too; only the
            # schema-2 property-role marker is gated on explicit opt-in.
            bundle[OPTIMIZER_PARAM_NAMES_KEY] = optimizer_manifest
        util.atomic_torch_save(bundle, path)
        TheMessage(f"[{self.name}] Weights saved to {path}")

    def _collect_vocab_extras(self):
        """Gather lexicon mappings and non-tensor space structure.

        Returns ``None`` only when neither lexicon nor structural/property
        metadata is present.

        In canonical property-basis models, ConceptualSpace owns taxonomy,
        META, and concept identity under ``conceptual_structure``. WholeSpace
        contributes only property tags/provenance under ``whole_properties``.
        Legacy configs retain the old WS envelope until explicitly migrated.
        """
        emb = self._get_embedding()
        whole_space = getattr(self, 'wholeSpace', None)
        concept_space = getattr(self, 'conceptualSpace', None)
        property_basis = bool(getattr(self, "wholePropertyBasis", False))
        conceptual_extras = (
            concept_space.vocab_extras()
            if property_basis and concept_space is not None
            and hasattr(concept_space, "vocab_extras") else None)
        if property_basis and whole_space is not None:
            if hasattr(whole_space, "property_extras"):
                whole_extras = whole_space.property_extras()
            elif hasattr(whole_space, "vocab_extras"):
                # Transitional compatibility: never let an older broad
                # WholeSpace serializer leak concept/taxonomy fields into a
                # schema-2 checkpoint.
                raw_whole = whole_space.vocab_extras() or {}
                whole_extras = {
                    key: value for key, value in raw_whole.items()
                    if (str(key).startswith(("property", "type_"))
                        or str(key) == "ramsification")
                }
            else:
                whole_extras = None
        else:
            whole_extras = (
                whole_space.vocab_extras()
                if whole_space is not None
                and hasattr(whole_space, "vocab_extras") else None)
        # Canonical abstraction-order provenance for the PS percept
        # codebook (the WS tables ride inside ws_extras). None when there
        # is nothing stamped, so pre-feature blobs stay byte-identical.
        ps_cb = getattr(getattr(getattr(self, 'perceptualSpace', None),
                                'subspace', None), 'what', None)
        ps_rams = (ps_cb.ramsification_extras()
                   if ps_cb is not None
                   and hasattr(ps_cb, 'ramsification_extras') else None)
        # The PS percept store's pure-Python state (trie + inverse table +
        # hash map + fallback hits — the WORD surfaces). Emitted ONLY when a
        # store exists and holds entries, so pre-feature / storeless blobs
        # stay byte-identical (2026-07-13 open-fronts plan, Task A).
        ps_store = getattr(getattr(self, 'perceptualSpace', None),
                           'percept_store', None)
        ps_pstore = (ps_store.vocab_extras()
                     if ps_store is not None
                     and hasattr(ps_store, 'vocab_extras')
                     and len(ps_store) > 0 else None)
        if emb is None or getattr(emb, 'wv', None) is None:
            if (conceptual_extras is None and whole_extras is None
                    and ps_rams is None and ps_pstore is None):
                return None
            # Lexicon-less radix mode: only structural/PS state needs to
            # travel; we still wrap it in the standard envelope so
            # ``_restore_vocab_extras`` finds the keys.
            blob = {
                "index_to_key": [],
                "counts": [],
                "total_count": 0,
            }
            if property_basis:
                if conceptual_extras is not None:
                    blob["conceptual_structure"] = conceptual_extras
                if whole_extras is not None:
                    blob["whole_properties"] = whole_extras
            elif whole_extras is not None:
                blob["well_known_atoms"] = whole_extras.get(
                    "well_known_atoms", {})
                blob["ws_taxonomy_extras"] = whole_extras
            if ps_rams is not None:
                blob["ps_ramsification"] = ps_rams
            if ps_pstore is not None:
                blob["ps_percept_extras"] = ps_pstore
            return blob
        wv = emb.wv
        counts = getattr(wv, 'counts', None)
        if counts is not None and hasattr(counts, 'tolist'):
            counts = counts.tolist()
        blob = {
            "index_to_key": list(getattr(wv, 'index_to_key', []) or []),
            "counts": counts or [],
            "total_count": int(getattr(wv, 'total_count', 0) or 0),
        }
        if property_basis:
            if conceptual_extras is not None:
                blob["conceptual_structure"] = conceptual_extras
            if whole_extras is not None:
                blob["whole_properties"] = whole_extras
        elif whole_extras is not None:
            blob["well_known_atoms"] = dict(
                getattr(whole_space, 'well_known_atoms', {}) or {})
            blob["ws_taxonomy_extras"] = whole_extras
        if ps_rams is not None:
            blob["ps_ramsification"] = ps_rams
        if ps_pstore is not None:
            blob["ps_percept_extras"] = ps_pstore
        return blob

    def _collect_bpe_extras(self):
        """Snapshot the ChunkLayer's pure-Python state into a dict
        suitable for ``torch.save``. Returns ``None`` when no BPE
        ChunkLayer is active.
        """
        ps = getattr(self, 'perceptualSpace', None)
        cl = getattr(ps, 'chunk_layer', None) if ps is not None else None
        if cl is None or not getattr(cl, 'bpe', False):
            return None
        return {
            "merges": [list(p) for p in getattr(cl, 'merges', [])],
            "vocab": {",".join(str(x) for x in k): int(v)
                      for k, v in getattr(cl, 'vocab', {}).items()},
            "id_to_bytes": {int(k): list(v)
                            for k, v in getattr(cl, 'id_to_bytes', {}).items()},
            "_next_id": int(getattr(cl, '_next_id', 0) or 0),
            "_max_merge_len": int(getattr(cl, '_max_merge_len', 0) or 0),
            "n_vectors": int(getattr(cl, 'n_vectors', 0) or 0),
            "word_learning": int(
                getattr(cl, 'word_learning', 0) or 0),
        }

    def _collect_structural_extras(self):
        """Snapshot durable host-side concept and mereology state.

        ``ConceptAllocator`` deliberately is not an ``nn.Module``: its ids,
        ordered constituent records, row maps and idempotency dictionaries are
        symbolic structure rather than tensor geometry.  The same is true of
        the word-keyed WholeSpace registries.  They nevertheless form part of
        the learned model; dropping them on reload makes the next observation
        mint duplicate identities.  Keep them in one versioned sidecar.
        """
        alloc_fields = (
            "placement", "raised", "singletons", "retired", "identity",
            "relate_idx", "chain_idx", "word_obj_meta", "joint",
        )
        cs_fields = (
            "_autobound_percept_ids", "_recognized_words",
            "_words_concept_id", "_percept_word_concept",
            "_object_word_concept", "_priming_bridge", "_frozen_concepts",
            "_frozen_named", "_promotion_cache_state",
            "_concept_admission_drops",
        )
        legacy_ws_fields = (
            "_word_whole_ss", "_mereology_raised",
            "_property_class_whole", "_anchored_pids",
            "_pending_words_summary", "_standalone_run_bytes",
            "_lbg_disp_sum", "_lbg_disp_sum_sq", "_lbg_count",
        )
        # Schema-2 WholeSpace sidecars contain perceptual analysis state only.
        # LBG accumulators belong to the retired WS concept/META dictionary and
        # must not be serialized beside the fixed property basis.
        property_ws_fields = ("_standalone_run_bytes",)
        property_basis = bool(getattr(self, "wholePropertyBasis", False))

        conceptual = {}
        for i, cs in enumerate(list(
                getattr(self, "conceptualSpaces", None) or ())):
            entry = {}
            alloc = getattr(cs, "_concept_allocator", None)
            if alloc is not None:
                a = {"next_id": int(getattr(alloc, "next_id", 1))}
                for name in alloc_fields:
                    a[name] = _checkpoint_host_copy(getattr(alloc, name))
                layers = {}
                for order, layer in getattr(alloc, "_layers", {}).items():
                    values = getattr(layer, "values", None)
                    layers[int(order)] = {
                        "nInput": int(layer.nInput),
                        "nOutput": int(layer.nOutput),
                        "constituents": _checkpoint_host_copy(
                            getattr(layer, "_constituents", {})),
                        "tensor_rows": _checkpoint_host_copy(
                            getattr(layer, "_tensor_rows", {})),
                        "row_next": _checkpoint_host_copy(
                            getattr(layer, "_row_next", {})),
                        "rows": [int(v) for v in getattr(layer, "_rows", ())],
                        "cols": [int(v) for v in getattr(layer, "_cols", ())],
                        "init_vals": [
                            float(v) for v in getattr(layer, "_init_vals", ())
                        ],
                        "values": (
                            None if values is None
                            else _checkpoint_host_copy(values)),
                        "values_requires_grad": bool(
                            values is not None and values.requires_grad),
                    }
                a["layers"] = layers
                entry["allocator"] = a
            attrs = {
                name: _checkpoint_host_copy(getattr(cs, name))
                for name in cs_fields
                if hasattr(cs, name) and getattr(cs, name) is not None
            }
            if attrs:
                entry["attributes"] = attrs
            if (property_basis and cs is getattr(
                    self, "conceptualSpace", None)
                    and hasattr(cs, "vocab_extras")):
                snapshot = cs.vocab_extras()
                if snapshot is not None:
                    entry["conceptual_structure"] = (
                        _checkpoint_host_copy(snapshot))
            if entry:
                conceptual[int(i)] = entry

        wholes = {}
        for i, ws in enumerate(list(
                getattr(self, "wholeSpaces", None) or ())):
            entry = {}
            if property_basis and hasattr(ws, "property_extras"):
                vocab_snapshot = ws.property_extras()
            elif hasattr(ws, "vocab_extras"):
                vocab_snapshot = ws.vocab_extras()
                if property_basis:
                    vocab_snapshot = {
                        key: value for key, value in
                        (vocab_snapshot or {}).items()
                        if (str(key).startswith(("property", "type_"))
                            or str(key) == "ramsification")
                    }
            else:
                vocab_snapshot = None
            if vocab_snapshot is not None:
                # In schema 2 this is property metadata/provenance only.
                # Legacy configs retain their broad per-stage WS snapshot.
                entry["vocab_extras"] = _checkpoint_host_copy(vocab_snapshot)
            attrs = {
                name: _checkpoint_host_copy(getattr(ws, name))
                for name in (property_ws_fields if property_basis
                             else legacy_ws_fields)
                if hasattr(ws, name) and getattr(ws, name) is not None
            }
            if attrs:
                entry["attributes"] = attrs
            if entry:
                wholes[int(i)] = entry

        if not conceptual and not wholes:
            return None
        return {
            "version": 2 if property_basis else 1,
            "conceptual_spaces": conceptual,
            ("whole_properties" if property_basis else "whole_spaces"):
                wholes,
        }

    def _restore_allocator_extras(self, cs, saved):
        """Restore one ConceptAllocator without replacing its live aliases."""
        alloc = getattr(cs, "_concept_allocator", None)
        if alloc is None:
            ensure = getattr(cs, "_concept_tables", None)
            if callable(ensure):
                ensure()
            alloc = getattr(cs, "_concept_allocator", None)
        if alloc is None:
            raise ValueError(
                "checkpoint contains ConceptAllocator state, but the live "
                "ConceptualSpace cannot materialize an allocator")

        fields = (
            "placement", "raised", "singletons", "retired", "identity",
            "relate_idx", "chain_idx", "word_obj_meta", "joint",
        )
        for name in fields:
            if name not in saved:
                continue
            incoming = _checkpoint_host_copy(saved[name])
            current = getattr(alloc, name, None)
            # Preserve dict/set identities because ConceptualSpace exposes
            # aliases such as ``_word_obj_meta`` and ``_joint_concepts``.
            if isinstance(current, dict) and isinstance(incoming, dict):
                current.clear()
                current.update(incoming)
            elif isinstance(current, set) and isinstance(incoming, set):
                current.clear()
                current.update(incoming)
            else:
                setattr(alloc, name, incoming)

        next_id = int(saved.get("next_id", 1) or 1)
        known_ids = {0}
        for name in ("placement", "raised", "singletons", "retired",
                     "identity"):
            value = getattr(alloc, name, {})
            known_ids.update(int(v) for v in (
                value.keys() if isinstance(value, dict) else value))
        # A stale counter must never collide with a restored concept. Exact
        # checkpoints already satisfy this; max() is a corruption-safe guard.
        alloc.next_id = max(next_id, max(known_ids) + 1)

        layer_blobs = saved.get("layers") or {}
        if len(layer_blobs) > 1:
            raise ValueError(
                "checkpoint has multiple ConceptAllocator layers; the live "
                "single-store allocator supports exactly one")
        for raw_order, blob in layer_blobs.items():
            order = int(raw_order)
            layer = alloc.layer(order)
            saved_n_input = int(blob.get("nInput", layer.nInput))
            saved_n_output = int(blob.get("nOutput", layer.nOutput))
            live_n_input = int(layer.nInput)
            live_n_output = int(layer.nOutput)
            shape_mismatch = (
                saved_n_input != live_n_input
                or saved_n_output != live_n_output)
            square_expansion = bool(
                shape_mismatch
                and self._aligned_inventory_capacity() is not None
                and saved_n_input == saved_n_output + 1
                and live_n_input == live_n_output + 1
                and saved_n_output < live_n_output)
            if shape_mismatch and not square_expansion:
                raise ValueError(
                    "ConceptAllocator store shape mismatch: checkpoint "
                    f"[{blob.get('nOutput')}, {blob.get('nInput')}] vs live "
                    f"[{layer.nOutput}, {layer.nInput}]")
            rows = [int(v) for v in blob.get("rows", ())]
            cols = [int(v) for v in blob.get("cols", ())]
            if square_expansion:
                # Ordinary row/column ids remain stable under first-axis
                # capacity growth.  The sole exception is the trailing
                # EVERYTHING pole: its coordinate is N, so move old N to new
                # N.  A nonzero region base means the checkpoint partitioned
                # the square into capacity-relative blocks; relocating those
                # rows requires a full coordinated repartition and must not be
                # guessed during load.
                row_next = blob.get("row_next") or {}
                nonzero_bases = [
                    int(base) for base in row_next if int(base) != 0]
                if nonzero_bases:
                    raise ValueError(
                        "ConceptAllocator capacity expansion is unsafe for "
                        "a partitioned row map (nonzero region bases "
                        f"{sorted(nonzero_bases)}). Increase capacity before "
                        "training or migrate the partition explicitly.")
                cols = [
                    live_n_output if c == saved_n_output else c
                    for c in cols
                ]
            if len(rows) != len(cols) or len(set(zip(rows, cols))) != len(rows):
                raise ValueError(
                    "ConceptAllocator sparse COO checkpoint is malformed")
            if any(r < 0 or r >= int(layer.nOutput) for r in rows) or any(
                    c < 0 or c >= int(layer.nInput) for c in cols):
                raise ValueError(
                    "ConceptAllocator sparse COO checkpoint is out of bounds")
            values = blob.get("values")
            if values is not None and int(values.numel()) != len(rows):
                raise ValueError(
                    "ConceptAllocator sparse values/indices length mismatch")
            init_vals = [float(v) for v in blob.get("init_vals", ())]
            if len(init_vals) != len(rows):
                if values is None:
                    raise ValueError(
                        "ConceptAllocator sparse initial-values length mismatch")
                init_vals = [float(v) for v in values.reshape(-1).tolist()]

            layer._constituents = _checkpoint_host_copy(
                blob.get("constituents") or {})
            layer._tensor_rows = _checkpoint_host_copy(
                blob.get("tensor_rows") or {})
            layer._row_next = _checkpoint_host_copy(blob.get("row_next") or {})
            layer._rows = rows
            layer._cols = cols
            layer._init_vals = init_vals
            layer._index = {
                (r, c): i for i, (r, c) in enumerate(zip(rows, cols))
            }
            layer._dev_cache = None
            if values is None:
                layer.values = None
            else:
                target = getattr(layer, "_target_device", None)
                if target is None:
                    target = next(
                        (p.device for p in cs.parameters()), torch.device("cpu"))
                layer.values = nn.Parameter(
                    values.detach().to(target).clone(),
                    requires_grad=bool(blob.get("values_requires_grad", True)),
                )

        object.__setattr__(cs, "_word_obj_meta", alloc.word_obj_meta)
        object.__setattr__(cs, "_joint_concepts", alloc.joint)
        register = getattr(cs, "_sparse_families", None)
        if callable(register) and layer_blobs:
            register(0)
        object.__setattr__(
            cs, "_csw_registered_count",
            sum(int(layer.nnz) for layer in alloc._layers.values()))

    def _restore_structural_extras(
            self, extras, *, legacy_whole_structure=None):
        """Inverse of :meth:`_collect_structural_extras` (old-safe)."""
        if not isinstance(extras, dict):
            return
        version = int(extras.get("version", 0) or 0)
        if version not in (1, 2):
            raise ValueError(
                f"unsupported structural checkpoint version {version}")
        property_basis = bool(getattr(self, "wholePropertyBasis", False))

        conceptual_spaces = list(
            getattr(self, "conceptualSpaces", None) or ())
        for raw_i, entry in (extras.get("conceptual_spaces") or {}).items():
            i = int(raw_i)
            if not 0 <= i < len(conceptual_spaces):
                raise ValueError(
                    f"checkpoint ConceptualSpace index {i} is not present")
            cs = conceptual_spaces[i]
            alloc_blob = entry.get("allocator")
            if isinstance(alloc_blob, dict):
                self._restore_allocator_extras(cs, alloc_blob)
            for name, value in (entry.get("attributes") or {}).items():
                object.__setattr__(cs, str(name), _checkpoint_host_copy(value))
            conceptual_blob = entry.get("conceptual_structure")
            if (isinstance(conceptual_blob, dict)
                    and hasattr(cs, "load_vocab_extras")):
                cs.load_vocab_extras(conceptual_blob)
            refresh = getattr(cs, "_refresh_frozen_values_hook", None)
            if callable(refresh):
                refresh()

        whole_spaces = list(getattr(self, "wholeSpaces", None) or ())
        if property_basis:
            whole_entries = extras.get("whole_properties") or {}
        else:
            whole_entries = extras.get("whole_spaces") or {}
        for raw_i, entry in whole_entries.items():
            i = int(raw_i)
            if not 0 <= i < len(whole_spaces):
                raise ValueError(
                    f"checkpoint WholeSpace index {i} is not present")
            ws = whole_spaces[i]
            ws_vocab = entry.get("vocab_extras")
            if isinstance(ws_vocab, dict):
                if property_basis and hasattr(ws, "load_property_extras"):
                    ws.load_property_extras(ws_vocab)
                elif hasattr(ws, "load_vocab_extras"):
                    ws.load_vocab_extras(ws_vocab)
            cb = getattr(getattr(ws, "subspace", None), "what", None)
            W = cb.getW() if cb is not None and hasattr(cb, "getW") else None
            for name, value in (entry.get("attributes") or {}).items():
                if (property_basis
                        and str(name) != "_standalone_run_bytes"):
                    continue
                restored = _checkpoint_host_copy(value)
                if (not property_basis
                        and str(name).startswith("_lbg_disp_")
                        and torch.is_tensor(W)):
                    restored = {
                        k: (v.to(W.device) if torch.is_tensor(v) else v)
                        for k, v in restored.items()
                    }
                object.__setattr__(ws, str(name), restored)

        if property_basis:
            # Version-1 broad WS state is concept/META structure. Import all
            # stage blobs through terminal CS's explicit merge policy; never
            # replay them onto the eight-row property basis.
            legacy_stages = extras.get("whole_spaces")
            quarantine = (legacy_whole_structure
                          if isinstance(legacy_whole_structure, dict) else {})
            if not isinstance(legacy_stages, dict):
                legacy_stages = quarantine.get("structural_whole_spaces")
            terminal_cs = getattr(self, "conceptualSpace", None)
            if (isinstance(legacy_stages, dict) and legacy_stages
                    and terminal_cs is not None
                    and hasattr(terminal_cs, "load_vocab_extras")):
                terminal_cs.load_vocab_extras({
                    "legacy_whole_spaces": legacy_stages,
                })

    def load_weights(self, path=None, strict=False, require_match=False):
        """Load model state from a single .ckpt bundle.

        The new bundle format carries ``state_dict`` plus
        ``vocab_extras`` (WordVectors mappings) and ``bpe_extras``
        (ChunkLayer merges/vocab) — see ``save_weights``. Old
        embedding-stripped .ckpts (state_dict only, no extras) load
        with a warning; the embeddings stay at their construction-time
        random init.

        ``require_match`` (autoload only): when True, a genuine
        architecture mismatch (shape drift / missing keys / fatal
        unexpected keys) raises ``ValueError`` with the actionable
        diagnostic instead of soft-warning and proceeding on fresh
        weights -> later crash. This is the migration-cliff catcher:
        the bivector retirement invalidates pre-refactor checkpoints
        (e.g. data/MM_5M_bivector.ckpt), so autoloading a stale ckpt
        now fails fast. Non-autoload callers keep ``require_match=False``
        (soft-warn) so vocab-grow / benign-stale-key loads still work.
        """
        if path is None:
            path = os.path.join(ProjectPaths.OUTPUT_DIR, "weights.ckpt")
        if not os.path.exists(path):
            TheMessage(f"[{self.name}] No checkpoint at {path}, starting fresh")
            return False
        # Always inspect/migrate on CPU before any tensor reaches the live
        # accelerator. In particular, a schema-1 checkpoint can contain GiB-
        # scale retired WS concept/analysis tables; mapping it to MPS first
        # would defeat the reset migration's peak-memory guarantee. mmap keeps
        # those soon-to-be-discarded storages file-backed while filtering.
        try:
            saved = torch.load(
                path, map_location="cpu", weights_only=False, mmap=True)
        except (TypeError, RuntimeError) as exc:
            # Older torch lacks the keyword; legacy non-zip serialization can
            # expose it but reject mmap. Do not mask unrelated corruption.
            detail = str(exc).lower()
            if not any(token in detail for token in (
                    "mmap", "memory map", "zip", "keyword")):
                raise
            saved = torch.load(
                path, map_location="cpu", weights_only=False)
        if bool(getattr(self, "wholePropertyBasis", False)):
            migration = migrate_wholespace_checkpoint(
                saved, dict(self.state_dict()))
            saved = migration.checkpoint
            for message in migration.diagnostics.messages():
                TheMessage(f"[{self.name}] {message}")
            legacy_state_shapes = migration.legacy_state_shapes
            reset_wholespace_optimizer = bool(migration.migrated)
        else:
            # Legacy configs have not opted into the ownership migration; do
            # not strip their WS tensors or stamp property-basis semantics.
            state_for_shapes = (
                saved.get("state_dict")
                if isinstance(saved, dict) and "state_dict" in saved
                else saved)
            legacy_state_shapes = {
                str(key): tuple(int(v) for v in value.shape)
                for key, value in (state_for_shapes or {}).items()
                if hasattr(value, "shape")
            }
            reset_wholespace_optimizer = False

        if isinstance(saved, dict) and "state_dict" in saved:
            state = saved["state_dict"]
            vocab_extras = saved.get("vocab_extras")
            bpe_extras = saved.get("bpe_extras")
            structural_extras = saved.get("structural_extras")
            optimizer_state = saved.get("optimizer_state")
            optimizer_manifest = saved.get(OPTIMIZER_PARAM_NAMES_KEY)
            legacy_whole_structure = saved.get(LEGACY_WHOLE_STRUCTURE_KEY)
            training_state = saved.get("training_state") or {}
        else:
            state = saved
            vocab_extras = None
            bpe_extras = None
            structural_extras = None
            optimizer_state = None
            optimizer_manifest = None
            legacy_whole_structure = None
            training_state = {}

        if vocab_extras is None and any("wv._vectors" in k for k in state):
            # Bundled with embeddings but no vocab_extras — unusual
            # but tolerated; the wv._vectors will load through state_dict.
            pass
        elif vocab_extras is None:
            TheMessage(
                f"[{self.name}] Legacy checkpoint at {path} has no "
                f"embedding bundle; embeddings stay at random init.")

        # Resize the live Embedding to match the saved vocab BEFORE the
        # state_dict shape check. Otherwise the wv._vectors row count
        # mismatch (live model is freshly built with a smaller vocab;
        # saved bundle carries the grown vocab) would fail the load.
        if vocab_extras is not None or legacy_whole_structure is not None:
            self._restore_vocab_extras(
                vocab_extras or {},
                legacy_whole_structure=legacy_whole_structure)

        # Pre-check for shape mismatches before attempting to load.
        # This produces an actionable diagnostic instead of a raw PyTorch error.
        model_state = dict(self.state_dict())

        # Space-owned carrier-state migration (2026-07-17): durable Basis and
        # Encoding modules moved from ``Space.subspace`` into the Space's
        # authoritative ``_owned_bases`` / ``_owned_encoders`` containers.
        # Rewrite only when the candidate is an exact live key, so unrelated
        # nested SubSpaces (including SymbolSubSpace) retain their own paths.
        _carrier_key_roles = {
            **{
                role: '_owned_bases'
                for role in ('event', 'what', 'where', 'when', 'activation')
            },
            **{
                role: '_owned_encoders'
                for role in (
                    'activeEncoding',
                    'objectEncoding',
                    'whatEncoding',
                    'whereEncoding',
                    'whenEncoding',
                    'wordEncoding',
                )
            },
        }
        _carrier_renamed = 0
        for _old_key in list(state.keys()):
            _percept_token = '.subspace.percept_store.'
            if _percept_token in _old_key:
                _new_key = _old_key.replace(
                    _percept_token, '._percept_store.', 1
                )
                if _new_key in model_state and _old_key not in model_state:
                    state[_new_key] = state.pop(_old_key)
                    _carrier_renamed += 1
                    continue
            for _role, _container in _carrier_key_roles.items():
                _token = f'.subspace.{_role}.'
                if _token not in _old_key:
                    continue
                _new_key = _old_key.replace(
                    _token, f'.{_container}.{_role}.', 1
                )
                if _new_key in model_state and _old_key not in model_state:
                    state[_new_key] = state.pop(_old_key)
                    _carrier_renamed += 1
                break
        # SymbolSpace migration (2026-06-21 refactor, Stage 3): the
        # SymbolSubSpace coordinator moved UNDER the new SymbolSpace container
        # (``m.symbolSpace`` is now a ``SymbolSpace(Space)`` that owns
        # ``.subspace``), so its state_dict keys shifted from
        # ``symbolSpace.*`` to ``symbolSpace.subspace.*``. Rewrite the old
        # keys in place so pre-refactor checkpoints load. One-time bridge; runs
        # before the bivector pass so renamed keys still get bivector-expanded.
        _ss_old = "symbolSpace."
        _ss_new = "symbolSpace.subspace."
        _ss_renamed = 0
        for k in list(state.keys()):
            if (
                not k.startswith(_ss_old)
                or k.startswith(_ss_new)
                or k in model_state
            ):
                continue
            remainder = k[len(_ss_old):]
            candidates = [_ss_new + remainder]
            head = remainder.split('.', 1)[0]
            if head in ('event', 'what', 'where', 'when', 'activation'):
                candidates.insert(
                    0, f"symbolSpace._owned_bases.{remainder}"
                )
            elif head in (
                'activeEncoding', 'objectEncoding', 'whatEncoding',
                'whereEncoding', 'whenEncoding', 'wordEncoding',
            ):
                candidates.insert(
                    0, f"symbolSpace._owned_encoders.{remainder}"
                )
            target = next(
                (candidate for candidate in candidates if candidate in model_state),
                candidates[-1],
            )
            state[target] = state.pop(k)
            _ss_renamed += 1
        if _ss_renamed:
            warnings.warn(
                f"SymbolSpace migration: rewrote {_ss_renamed} symbolSpace.* "
                f"checkpoint keys to symbolSpace.subspace.* (the "
                f"coordinator moved under the new SymbolSpace container).",
                stacklevel=2,
            )

        # Remove pre-ownership duplicate module paths. Terminal aliases,
        # body-stage ModuleDicts, and Space->SymbolSpace back-references used
        # to register the same modules repeatedly. Canonical ModuleLists and
        # ``model.symbolSpace`` are now their sole structural owners.
        _alias_renamed = 0
        _legacy_ws_grammar_dropped = []
        _legacy_ws_category_dropped = []
        _last_cs = len(getattr(self, 'conceptualSpaces', ())) - 1
        _last_ws = len(getattr(self, 'wholeSpaces', ())) - 1
        for _old_key in list(state.keys()):
            _target = None
            _parts = _old_key.split('.')
            if (
                len(_parts) >= 4
                and _parts[0] == 'wholeSpaces'
                and _parts[1].isdigit()
                and _parts[2] == 'conceptualSpace'
            ):
                # Historical WS->CS module registration duplicated the entire
                # concept tree below each WholeSpace. CS is now a non-owning
                # downstream pointer, so retain the learned value only under
                # its canonical ConceptualSpace stage.
                _target = '.'.join(
                    ['conceptualSpaces', _parts[1], *_parts[3:]])
            elif (
                len(_parts) >= 3
                and _parts[0] == 'wholeSpaces'
                and _parts[1].isdigit()
                and _parts[2] in ('_category_vq', '_category_role')
            ):
                _candidate = '.'.join(
                    ['conceptualSpaces', _parts[1], *_parts[2:]])
                if _candidate in model_state:
                    _target = _candidate
                elif bool(getattr(self, "wholePropertyBasis", False)):
                    _legacy_ws_category_dropped.append(_old_key)
                    del state[_old_key]
                    continue
            elif (_last_cs >= 0
                  and _old_key.startswith('wholeSpace.conceptualSpace.')):
                _target = (
                    f'conceptualSpaces.{_last_cs}.'
                    + _old_key[len('wholeSpace.conceptualSpace.'):])
            elif (_last_cs >= 0 and _old_key.startswith(
                    ('wholeSpace._category_vq.',
                     'wholeSpace._category_role'))):
                _rest = _old_key[len('wholeSpace.'):]
                _candidate = f'conceptualSpaces.{_last_cs}.{_rest}'
                if _candidate in model_state:
                    _target = _candidate
                elif bool(getattr(self, "wholePropertyBasis", False)):
                    _legacy_ws_category_dropped.append(_old_key)
                    del state[_old_key]
                    continue
            elif (
                len(_parts) >= 5
                and _parts[0] == 'body_stages'
                and _parts[1].isdigit()
                and _parts[2] == 'ws'
                and _parts[3] == 'conceptualSpace'
            ):
                _target = '.'.join(
                    ['conceptualSpaces', _parts[1], *_parts[4:]])
            elif (
                len(_parts) >= 4
                and _parts[0] == 'body_stages'
                and _parts[1].isdigit()
                and _parts[2] == 'ws'
                and _parts[3] in ('_category_vq', '_category_role')
            ):
                _candidate = '.'.join(
                    ['conceptualSpaces', _parts[1], *_parts[3:]])
                if _candidate in model_state:
                    _target = _candidate
                elif bool(getattr(self, "wholePropertyBasis", False)):
                    _legacy_ws_category_dropped.append(_old_key)
                    del state[_old_key]
                    continue
            elif (
                len(_parts) >= 4
                and _parts[0] == 'wholeSpaces'
                and _parts[1].isdigit()
                and _parts[2] == 'syntacticLayer'
                and bool(getattr(self, "wholePropertyBasis", False))
            ):
                _rest = '.'.join(_parts[3:])
                _grammar_candidates = (
                    f'symbolSpace.syntacticLayer.{_rest}',
                    f'symbolSpace.subspace.syntacticLayer.{_rest}',
                )
                _target = next((candidate for candidate in _grammar_candidates
                                if candidate in model_state
                                and tuple(model_state[candidate].shape)
                                == tuple(state[_old_key].shape)), None)
                if _target is None:
                    _legacy_ws_grammar_dropped.append(_old_key)
                    del state[_old_key]
                    continue
            elif (bool(getattr(self, "wholePropertyBasis", False))
                  and _old_key.startswith('wholeSpace.syntacticLayer.')
                  and _last_ws >= 0):
                _rest = _old_key[len('wholeSpace.syntacticLayer.'):]
                _grammar_candidates = (
                    f'symbolSpace.syntacticLayer.{_rest}',
                    f'symbolSpace.subspace.syntacticLayer.{_rest}',
                )
                _target = next((candidate for candidate in _grammar_candidates
                                if candidate in model_state
                                and tuple(model_state[candidate].shape)
                                == tuple(state[_old_key].shape)), None)
                if _target is None:
                    _legacy_ws_grammar_dropped.append(_old_key)
                    del state[_old_key]
                    continue
            elif (
                len(_parts) >= 5
                and _parts[0] == 'body_stages'
                and _parts[1].isdigit()
                and _parts[2] == 'ws'
                and _parts[3] == 'syntacticLayer'
                and bool(getattr(self, "wholePropertyBasis", False))
            ):
                _rest = '.'.join(_parts[4:])
                _grammar_candidates = (
                    f'symbolSpace.syntacticLayer.{_rest}',
                    f'symbolSpace.subspace.syntacticLayer.{_rest}',
                )
                _target = next((candidate for candidate in _grammar_candidates
                                if candidate in model_state
                                and tuple(model_state[candidate].shape)
                                == tuple(state[_old_key].shape)), None)
                if _target is None:
                    _legacy_ws_grammar_dropped.append(_old_key)
                    del state[_old_key]
                    continue
            elif _last_cs >= 0 and _old_key.startswith('conceptualSpace.'):
                _target = (
                    f'conceptualSpaces.{_last_cs}.'
                    + _old_key[len('conceptualSpace.'):]
                )
            elif _last_ws >= 0 and _old_key.startswith('wholeSpace.'):
                _target = (
                    f'wholeSpaces.{_last_ws}.'
                    + _old_key[len('wholeSpace.'):]
                )
            else:
                if (
                    len(_parts) >= 4
                    and _parts[0] == 'body_stages'
                    and _parts[1].isdigit()
                    and _parts[2] in ('cs', 'ws')
                ):
                    _owner = (
                        'conceptualSpaces'
                        if _parts[2] == 'cs'
                        else 'wholeSpaces'
                    )
                    _target = '.'.join(
                        [_owner, _parts[1], *_parts[3:]]
                    )
                else:
                    for _token in ('._model_symbolSpace.', '.symbolSpace.'):
                        if _token in _old_key:
                            _target = 'symbolSpace.' + _old_key.split(
                                _token, 1
                            )[1]
                            break
            if _target is None or _target == _old_key:
                continue
            if _target not in state:
                state[_target] = state[_old_key]
            del state[_old_key]
            _alias_renamed += 1

        # Alias canonicalization may have produced canonical legacy
        # ``*.subspace.<slot>.*`` keys. Apply the ownership rewrite once more.
        for _old_key in list(state.keys()):
            _new_key = None
            _percept_token = '.subspace.percept_store.'
            if _percept_token in _old_key:
                _candidate = _old_key.replace(
                    _percept_token, '._percept_store.', 1
                )
                if _candidate in model_state:
                    _new_key = _candidate
            if _new_key is None:
                for _role, _container in _carrier_key_roles.items():
                    _token = f'.subspace.{_role}.'
                    if _token in _old_key:
                        _candidate = _old_key.replace(
                            _token, f'.{_container}.{_role}.', 1
                        )
                        if _candidate in model_state:
                            _new_key = _candidate
                        break
            if _new_key is not None:
                if _new_key not in state:
                    state[_new_key] = state[_old_key]
                del state[_old_key]
                _carrier_renamed += 1

        # The pre-ownership model also registered the signal router's CS binary
        # reducer twice at the model root.  Both spellings are aliases of the
        # canonical SymbolSpace LanguageLayer reducer; retain a value only at
        # that downstream owner, with exact key/shape validation.
        _stm_reducer_aliases = 0
        for _old_key in list(state.keys()):
            _prefix = next((p for p in (
                "_stm_reducer_module.", "_stm_reducer_cached.")
                if _old_key.startswith(p)), None)
            if _prefix is None:
                continue
            _rest = _old_key[len(_prefix):]
            _target = (
                "symbolSpace.subspace.languageLayer._binary_layers.CS."
                + _rest)
            if _target not in model_state:
                # Unknown reducer state remains present and will fail the strict
                # mismatch audit below.
                continue
            if (_target in state
                    and tuple(state[_target].shape)
                    == tuple(model_state[_target].shape)):
                # A canonical, current-shaped copy is already present.  Some
                # historical root aliases were stale even within the same
                # checkpoint (fewer grammar tools); discard that alias rather
                # than overwriting the authoritative learned table.
                del state[_old_key]
                _stm_reducer_aliases += 1
                continue
            if (tuple(model_state[_target].shape)
                    != tuple(state[_old_key].shape)):
                # No authoritative canonical value and the alias cannot fit:
                # leave it for the strict mismatch audit; never resize grammar
                # semantics by guessing rows/feature columns.
                continue
            if _target not in state:
                state[_target] = state[_old_key]
            del state[_old_key]
            _stm_reducer_aliases += 1
        if _alias_renamed:
            warnings.warn(
                f"Space ownership migration: removed {_alias_renamed} "
                f"duplicate alias checkpoint keys.",
                stacklevel=2,
            )
        if _legacy_ws_grammar_dropped:
            warnings.warn(
                "WholeSpace ownership migration dropped "
                f"{len(_legacy_ws_grammar_dropped)} obsolete syntactic-layer "
                "checkpoint keys that have no exact downstream owner.",
                stacklevel=2,
            )
        if _legacy_ws_category_dropped:
            warnings.warn(
                "WholeSpace ownership migration dropped "
                f"{len(_legacy_ws_category_dropped)} category checkpoint "
                "keys because no exact ConceptualSpace owner was present.",
                stacklevel=2,
            )
        if _carrier_renamed:
            warnings.warn(
                f"Space ownership migration: rewrote {_carrier_renamed} "
                f"legacy Space.subspace Basis/Encoding checkpoint keys.",
                stacklevel=2,
            )
        if _stm_reducer_aliases:
            warnings.warn(
                "Space ownership migration: removed "
                f"{_stm_reducer_aliases} duplicate STM reducer alias "
                "checkpoint keys.",
                stacklevel=2,
            )

        # VQ-backed Codebooks formerly registered one Parameter twice as
        # ``W`` and ``vq._codebook``. The parent Basis now owns W exclusively;
        # retain the VQ value only when an unusually stripped artifact lacks W.
        _vq_aliases = 0
        for _old_key in list(state.keys()):
            if not _old_key.endswith('.vq._codebook'):
                continue
            _owner_key = _old_key[:-len('vq._codebook')] + 'W'
            if _owner_key not in model_state:
                continue
            if _owner_key not in state:
                state[_owner_key] = state[_old_key]
            del state[_old_key]
            _vq_aliases += 1
        if _vq_aliases:
            warnings.warn(
                f"Codebook ownership migration: removed {_vq_aliases} "
                f"duplicate VQ Parameter checkpoint keys.",
                stacklevel=2,
            )

        # The live canonical architecture has one physical ConceptualSpace
        # dictionary registered through each stage's compatibility aliases.
        # Resolve old independent stage tables to stage 0 before capacity
        # expansion so every alias receives the same grown tensor.
        self._canonicalize_shared_concept_checkpoint_state(state, model_state)

        # A configured capacity increase is a row-preserving migration, not
        # the historical bivector pole-doubling migration below.  Expand first
        # so (in particular) an exact 2x capacity increase cannot be mistaken
        # for old positive/negative-pole layout.
        self._expand_partspace_codebook_checkpoint_state(state, model_state)
        self._expand_aligned_codebook_checkpoint_state(state, model_state)

        # Bivector migration: pre-bivector checkpoints have a [K, D] symbolic
        # codebook, while the current model expects [2K, D]. Duplicate each
        # row into the positive-pole slot (row 2k), leaving the negative-pole
        # slot (row 2k+1) zero. One-time bridge; no reverse migration.
        # Skip when the saved tensor is empty (saved.shape[0] == 0):
        # ``2 * 0 == 0`` makes the doubling test trivially true and the
        # warning fires for a no-op shape transition.
        for k, saved_v in list(state.items()):
            if k not in model_state:
                continue
            model_v = model_state[k]
            if (saved_v.dim() == 2 and model_v.dim() == 2
                    and saved_v.shape[1] == model_v.shape[1]
                    and saved_v.shape[0] > 0
                    and model_v.shape[0] == 2 * saved_v.shape[0]
                    and "wholeSpace" in k):
                migrated = torch.zeros_like(model_v)
                migrated[0::2] = saved_v
                state[k] = migrated
                warnings.warn(
                    f"Bivector migration: expanded {k} from "
                    f"{list(saved_v.shape)} to {list(model_v.shape)} "
                    f"(positive poles only; negative poles zero).",
                    stacklevel=2,
                )

        mismatches = [
            (k, list(state[k].shape), list(model_state[k].shape))
            for k in state if k in model_state
            and state[k].shape != model_state[k].shape
        ]
        missing = [k for k in model_state if k not in state]
        unexpected = [k for k in state if k not in model_state]
        fatal_unexpected = unexpected if (strict or require_match) else []
        # Under ``require_match`` (autoload path), tolerate missing-only
        # mismatches: the current architecture may have grown new
        # parameters since the checkpoint was saved (e.g., the S → S
        # identity rule or the rule_predictor head's added rule slot).
        # Those new parameters initialize from their fresh-build random
        # values, which is the right outcome for the autoload migration
        # path. Shape mismatches and unexpected keys remain fatal — those
        # signal a real architecture divergence that needs a regenerated
        # checkpoint or a config correction.
        fatal = bool(mismatches or fatal_unexpected)
        if fatal or missing:
            lines = [f"[{self.name}] Weight file mismatch -- cannot load {path}"
                     if fatal
                     else f"[{self.name}] Weight file partial -- {path}"]
            if mismatches:
                lines.append("  Shape mismatches:")
                for key, saved_shape, model_shape in mismatches[:10]:
                    lines.append(f"    {key:<50s}  saved={saved_shape}  model={model_shape}")
                if len(mismatches) > 10:
                    lines.append(f"    ... and {len(mismatches) - 10} more")
            if missing:
                lines.append(f"  Keys in model but missing from file: {len(missing)} (initialized fresh)")
            if fatal_unexpected:
                lines.append(f"  Keys in file not present in model: {len(fatal_unexpected)}")
                for key in fatal_unexpected[:20]:
                    lines.append(f"    {key}")
                if len(fatal_unexpected) > 20:
                    lines.append(
                        f"    ... and {len(fatal_unexpected) - 20} more")
            if fatal:
                lines.append("  The model config likely changed since this checkpoint was saved.")
                lines.append(f"  To fix: correct the model XML to match the saved weights,")
                lines.append(f"          or delete/move {path} to start fresh.")
                if require_match:
                    raise ValueError("\n".join(lines))
                TheMessage("\n".join(lines))
                return False
            TheMessage("\n".join(lines))

        try:
            self.load_state_dict(state, strict=strict)
        except RuntimeError as e:
            TheMessage(f"[{self.name}] Warning: cannot load {path}: {e}")
            return False

        if unexpected:
            TheMessage(
                f"[{self.name}] Ignored {len(unexpected)} stale checkpoint "
                f"keys not present in the current model")

        # Restore Python-side WordVectors mappings (index_to_key etc.).
        if vocab_extras is not None or legacy_whole_structure is not None:
            self._restore_vocab_extras(
                vocab_extras or {},
                legacy_whole_structure=legacy_whole_structure)
        # Restore ChunkLayer BPE state (merges, vocab, id_to_bytes).
        if bpe_extras is not None:
            self._restore_bpe_extras(bpe_extras)
        # Restore symbolic identity before the optimizer is constructed. The
        # sidecar can materialize the ConceptAllocator's sparse values
        # Parameter; getOptimizer must see that live Parameter (and its saved
        # ordering) before loading optimizer_state.
        if (structural_extras is not None
                or legacy_whole_structure is not None):
            self._restore_structural_extras(
                structural_extras or {"version": 2},
                legacy_whole_structure=legacy_whole_structure)

        # Every aligned VQ must expose the same contiguous prefix after load.
        # This also expands (value-only, power-of-two) if a restored allocator
        # cursor already occupies rows beyond a stale saved mask.
        self._resync_aligned_active_codebooks()

        # Model autoload runs before the optimizer exists, so defer optimizer
        # restoration until getOptimizer(). Name manifests prevent retired WS
        # parameters from shifting Adam moments onto unrelated live tensors.
        self._pending_optimizer_state = optimizer_state
        self._pending_optimizer_manifest = optimizer_manifest
        self._pending_legacy_state_shapes = legacy_state_shapes
        self._pending_optimizer_reset_wholespace = (
            reset_wholespace_optimizer)
        self._pending_optimizer_require_match = bool(require_match)
        self._training_step_count = int(
            training_state.get("training_step_count", 0) or 0)
        self._train_batches_seen = int(
            training_state.get("train_batches_seen", 0) or 0)
        self._epoch_batches_seen = int(
            training_state.get("epoch_batches_seen", 0) or 0)
        self._resume_batches_to_skip = self._epoch_batches_seen
        self._checkpoint_batch_size = training_state.get(
            "checkpoint_batch_size")
        saved_manifest = training_state.get("data_manifest")
        live_manifest = getattr(
            getattr(getattr(self, "inputSpace", None), "data", None),
            "source_manifest", None)
        if (saved_manifest is not None and live_manifest is not None
                and saved_manifest != live_manifest):
            message = (
                f"[{self.name}] Checkpoint data manifest does not match "
                f"the loaded corpus: saved={saved_manifest}, "
                f"live={live_manifest}")
            if require_match:
                raise ValueError(message)
            TheMessage(message)
        if training_state.get("torch_rng_state") is not None:
            torch.set_rng_state(training_state["torch_rng_state"].cpu())
        if training_state.get("python_rng_state") is not None:
            random.setstate(training_state["python_rng_state"])
        if training_state.get("numpy_rng_state") is not None:
            np.random.set_state(training_state["numpy_rng_state"])
        if (torch.cuda.is_available()
                and training_state.get("cuda_rng_state_all") is not None):
            torch.cuda.set_rng_state_all(training_state["cuda_rng_state_all"])
        mps_state = training_state.get("mps_rng_state")
        set_mps_state = getattr(
            getattr(torch, "mps", None), "set_rng_state", None)
        if mps_state is not None and callable(set_mps_state):
            try:
                set_mps_state(mps_state.detach().to("cpu"))
            except (RuntimeError, AttributeError):
                # A checkpoint remains loadable on CPU/non-MPS hosts; the
                # state is applied when the active runtime has a Metal RNG.
                pass

        TheMessage(f"[{self.name}] Weights loaded from {path}")
        return True

    def _restore_vocab_extras(
            self, extras, *, legacy_whole_structure=None):
        """Restore the WordVectors mappings AND resize the live
        ``wv._vectors`` parameter to match the saved vocab size.

        Called before the state_dict load so that the shape pre-check
        sees matching dimensions. Vector data itself is populated by
        the subsequent ``load_state_dict`` call; here we just allocate
        the right-sized parameter and rebuild the Python mappings.

        Space-side restores run first, independent of the Embedding: a radix
        envelope may carry them with an empty lexicon. In property-basis mode,
        concept/META structure goes only to terminal ConceptualSpace and WS
        receives property metadata only. Legacy WS structural blobs are handed
        to ConceptualSpace's explicit migration loader, never replayed on WS.
        """
        whole_space = getattr(self, 'wholeSpace', None)
        concept_space = getattr(self, 'conceptualSpace', None)
        property_basis = bool(getattr(self, "wholePropertyBasis", False))
        if property_basis:
            conceptual_blob = extras.get("conceptual_structure")
            if (concept_space is not None
                    and isinstance(conceptual_blob, dict)
                    and hasattr(concept_space, "load_vocab_extras")):
                concept_space.load_vocab_extras(conceptual_blob)
            property_blob = extras.get("whole_properties")
            if whole_space is not None and isinstance(property_blob, dict):
                if hasattr(whole_space, "load_property_extras"):
                    whole_space.load_property_extras(property_blob)
                elif hasattr(whole_space, "load_vocab_extras"):
                    whole_space.load_vocab_extras(property_blob)

            # Raw version-1 fields (for callers that bypassed the checkpoint
            # migrator) and quarantined fields both import downstream into CS.
            legacy_vocab = {}
            raw_ws = extras.get("ws_taxonomy_extras")
            if isinstance(raw_ws, dict):
                legacy_vocab.update(raw_ws)
            raw_atoms = extras.get("well_known_atoms")
            if isinstance(raw_atoms, dict) and raw_atoms:
                legacy_vocab.setdefault("well_known_atoms", raw_atoms)
            quarantine = (legacy_whole_structure
                          if isinstance(legacy_whole_structure, dict) else {})
            quarantined_vocab = quarantine.get("vocab_extras")
            if isinstance(quarantined_vocab, dict):
                old_ws = quarantined_vocab.get("ws_taxonomy_extras")
                if isinstance(old_ws, dict):
                    legacy_vocab.update(old_ws)
                old_atoms = quarantined_vocab.get("well_known_atoms")
                if isinstance(old_atoms, dict) and old_atoms:
                    legacy_vocab.setdefault("well_known_atoms", old_atoms)
            if (legacy_vocab and concept_space is not None
                    and hasattr(concept_space, "load_vocab_extras")):
                concept_space.load_vocab_extras(legacy_vocab)
            legacy_stages = quarantine.get("structural_whole_spaces")
            if (isinstance(legacy_stages, dict)
                    and concept_space is not None
                    and hasattr(concept_space, "load_vocab_extras")):
                concept_space.load_vocab_extras({
                    "legacy_whole_spaces": legacy_stages,
                })
        else:
            # Legacy ownership path, retained only for unmigrated configs.
            well_known = extras.get("well_known_atoms")
            if (whole_space is not None and isinstance(well_known, dict)
                    and well_known):
                whole_space.well_known_atoms = {
                    str(k): int(v) for k, v in well_known.items()
                }
            ws_extras = extras.get("ws_taxonomy_extras")
            if (whole_space is not None and isinstance(ws_extras, dict)
                    and hasattr(whole_space, 'load_vocab_extras')):
                whole_space.load_vocab_extras(ws_extras)
        # The PS percept store's trie/inverse-table state (the WORD
        # surfaces; absent in pre-feature blobs). Runs pre-state-dict per
        # the envelope contract: load_vocab_extras re-allocates the shared
        # codebook capacity first so the shape pre-check sees matching dims.
        ps_pstore = extras.get("ps_percept_extras")
        if isinstance(ps_pstore, dict):
            part_space = getattr(self, 'perceptualSpace', None)
            store = getattr(part_space, 'percept_store', None)
            if store is not None and hasattr(store, 'load_vocab_extras'):
                old_parameter = getattr(
                    getattr(store, "_basis", None), "_parameters", {}
                ).get("W")
                store.load_vocab_extras(ps_pstore)
                new_parameter = getattr(
                    getattr(store, "_basis", None), "_parameters", {}
                ).get("W")
                if new_parameter is not old_parameter:
                    replace = getattr(
                        part_space, "_replace_radix_codebook_parameter", None)
                    if not callable(replace):
                        raise RuntimeError(
                            "checkpoint radix capacity restore replaced W, "
                            "but PartSpace cannot repair Parameter ownership")
                    replace(old_parameter, new_parameter, store.capacity)
        # Canonical abstraction-order provenance for the PS percept codebook
        # (absent in pre-feature blobs). Restore only after the radix sidecar
        # has grown the physical table, otherwise fold rows in the saved tail
        # would be silently discarded against the smaller XML initial prefix.
        ps_rams = extras.get("ps_ramsification")
        if isinstance(ps_rams, dict):
            ps_cb = getattr(getattr(getattr(self, 'perceptualSpace', None),
                                    'subspace', None), 'what', None)
            if ps_cb is not None and hasattr(ps_cb,
                                             'load_ramsification_extras'):
                ps_cb.load_ramsification_extras(ps_rams)
        emb = self._get_embedding()
        if emb is None or getattr(emb, 'wv', None) is None:
            return
        wv = emb.wv
        keys = list(extras.get("index_to_key") or [])
        if not keys:
            return
        counts = extras.get("counts") or []
        total = int(extras.get("total_count") or 0)
        # Resize wv._vectors to match the saved vocab size.
        dim = wv._vectors.shape[1]
        vocab_size = len(keys)
        if wv._vectors.shape[0] != vocab_size:
            # Step 3 (2026-06-10 symbolic-iteration plan): the lexicon is
            # PS-LOCAL permanently -- the tied-storage branch (resize the
            # shared WS codebook so the tied view follows) is retired
            # with ``tie_to_codebook``. Vector DATA is populated by the
            # subsequent load_state_dict; this just allocates the shape.
            new_W = torch.zeros(vocab_size, dim,
                                device=wv._vectors.device,
                                dtype=wv._vectors.dtype)
            wv._vectors = nn.Parameter(new_W, requires_grad=True)
        wv.index_to_key = keys
        wv.key_to_index = {k: i for i, k in enumerate(keys)}
        wv.counts = (np.asarray(counts, dtype=np.int64) if counts
                     else np.zeros(vocab_size, dtype=np.int64))
        wv.total_count = np.int64(total)
        if getattr(emb, 'pretrain', None) is not None:
            emb.pretrain.index_to_key = wv.index_to_key
            emb.pretrain.key_to_index = wv.key_to_index
        wv._normed = None

    def _restore_bpe_extras(self, extras):
        """Write the BPE codebook from a saved bundle back onto the
        live ChunkLayer. Restores ``merges``, ``vocab``, ``id_to_bytes``,
        and the cursors / counters that drive on-the-fly merge growth.
        """
        ps = getattr(self, 'perceptualSpace', None)
        cl = getattr(ps, 'chunk_layer', None) if ps is not None else None
        if cl is None:
            return
        cl.merges = [tuple(p) for p in extras.get("merges") or []]
        cl.vocab = {
            tuple(int(x) for x in k.split(",")) if k else (): int(v)
            for k, v in (extras.get("vocab") or {}).items()
        }
        cl.id_to_bytes = {
            int(k): tuple(int(x) for x in v)
            for k, v in (extras.get("id_to_bytes") or {}).items()
        }
        cl._next_id = int(extras.get("_next_id") or 0)
        cl._max_merge_len = int(extras.get("_max_merge_len") or 0)

    def _get_sentences(self, split):
        """Return raw sentence strings for a data split.

        All splits store raw strings directly in their input lists.
        Runtime maps to train_input (staged by runtime_batch).
        """
        data = self.inputSpace.data
        if split == "train" or split == "runtime":
            result = data.train_input
        elif split == "test":
            result = data.test_input
        elif split == "validation":
            result = data.validation_input
        else:
            return None
        if result and isinstance(result[0], str):
            return result
        return None

    @staticmethod
    def _bytes_to_text(tensor):
        """Decode a byte tensor (or padded int8 tensor) to a string."""
        if isinstance(tensor, str):
            return tensor
        if tensor.dim() > 1:
            tensor = tensor.squeeze()
        chars = [chr(int(b) & 0xFF) for b in tensor.tolist()]
        return "".join(chars).rstrip("\x00")

    @staticmethod
    def _data_len(items):
        if items is None:
            return 0
        if isinstance(items, torch.Tensor):
            return int(items.shape[0])
        return len(items)

    @staticmethod
    def _slice_data(items, n):
        if items is None:
            return []
        if isinstance(items, torch.Tensor):
            return [items[i] for i in range(min(n, int(items.shape[0])))]
        return list(items[:n])

    @staticmethod
    def _display_value(value):
        if isinstance(value, str):
            return value
        if isinstance(value, torch.Tensor):
            x = value.detach().cpu().squeeze()
            if x.numel() == 1:
                return f"{float(x.reshape(-1)[0]):.4f}"
            vals = x.reshape(-1)
            if vals.numel() <= 8:
                return "[" + ", ".join(f"{float(v):.4f}" for v in vals) + "]"
            return f"shape={tuple(x.shape)}"
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(BaseModel._display_value(v) for v in value) + "]"
        return str(value)

    def _input_token_count(self, original):
        if not isinstance(original, str):
            return None
        lex = str(getattr(self, 'lexer', 'word') or 'word').lower()
        if lex in ('word', 'words'):
            try:
                # parse(..., lex='words') yields the inter-word space as
                # its own token ('hello world' -> 3 tokens). Filter to
                # non-whitespace so the count matches actual word slots.
                return len([tok for tok, _ in parse(original, lex='words')
                            if tok and tok.strip()])
            except Exception:
                return len(original.split())
        if lex in ('byte', 'bytes'):
            return len(original.encode('utf-8'))
        if lex == 'sentence':
            return 1
        return len(original.split())

    def _reverse_decode_one(self, vec):
        """Stage 8 structural decode of a single terminal CS vector.

        Thin wrapper around :meth:`RadixLayer.reverse` (Task F): the
        canonical body now lives on the PS-side layer that owns the
        inverse table, so BasicModel no longer reaches across PS, WS,
        and the codebook itself. Returns ``b""`` when the radix path
        is not active (no ``percept_store``) or either space is
        unwired.
        """
        ps_space = getattr(self, "perceptualSpace", None)
        if ps_space is None:
            return b""
        ps_store = getattr(ps_space, "percept_store", None)
        if ps_store is None:
            return b""
        ws = getattr(self, "wholeSpace", None)
        return ps_store.reverse(vec, symbolic_space=ws)

    def _decode_reconstructed_inputs(self, recon, originals):
        if not isinstance(recon, torch.Tensor) or recon.numel() == 0:
            return []
        if getattr(self.inputSpace, 'model_type', None) != "embedding":
            return [self._display_value(recon[i]) for i in range(recon.shape[0])]

        # Stage 8 (doc/plans/2026-05-27-perceptstore-meta-taxonomy-
        # reentrancy.md §Stage 8): when the radix path is active the
        # reverse decode is *structural* -- WS nearest match -> META
        # children -> PS percept id -> inverse_table bytes. No
        # nearest-neighbour against PS vectors at the surface step.
        ps_space = getattr(self, "perceptualSpace", None)
        ps_store = (getattr(ps_space, "percept_store", None)
                    if ps_space is not None else None)
        if ps_store is not None:
            vectors = recon.detach()
            # Normalise to [B, N, D].
            while vectors.ndim > 3 and 1 in vectors.shape[2:-1]:
                for ax in range(2, vectors.ndim - 1):
                    if vectors.shape[ax] == 1:
                        vectors = vectors.squeeze(ax)
                        break
            if vectors.ndim == 2:
                vectors = vectors.unsqueeze(0)
            batch = int(vectors.shape[0]) if vectors.ndim >= 1 else 0
            n_vec = int(vectors.shape[1]) if vectors.ndim >= 2 else 0
            rendered = []
            for b in range(batch):
                words = []
                for v in range(n_vec):
                    raw = self._reverse_decode_one(vectors[b, v])
                    if not raw:
                        # Preserve the slot with an empty placeholder so
                        # the per-row token-count clip aligns with the
                        # original sentence length; dropping it shortens
                        # the rendered output and misaligns words[:N].
                        words.append("")
                        continue
                    try:
                        words.append(raw.decode("utf-8"))
                    except UnicodeDecodeError:
                        words.append(raw.decode("utf-8", errors="replace"))
                expected = (self._input_token_count(originals[b])
                            if b < len(originals) else None)
                if expected is not None:
                    words = words[:expected]
                rendered.append(" ".join(words))
            return rendered

        # Legacy lexicon / BPE / MPHF path: keep the existing decode.
        emb = getattr(getattr(self.perceptualSpace, 'subspace', None),
                      'what', None)
        if not isinstance(emb, Embedding):
            emb = getattr(self.perceptualSpace, 'vocabulary', None)
        if not isinstance(emb, Embedding):
            return [self._display_value(recon[i]) for i in range(recon.shape[0])]

        vectors = recon.detach()
        try:
            codebook = emb.getW()
            if torch.is_tensor(codebook):
                vectors = vectors.to(codebook.device)
            decoded = emb.decode_reverse_meta(
                vectors, subspace=self.perceptualSpace.subspace)
            word_rows = emb.reconstruct_data(decoded, text=False)
        except Exception:
            try:
                word_rows = self.perceptualSpace.reconstruct_data(text=False)
            except Exception:
                return [self._display_value(recon[i])
                        for i in range(recon.shape[0])]

        rendered = []
        for i, words in enumerate(word_rows):
            # 2026-05-28: filter empties, NUL sentinels, AND
            # whitespace-only tokens. The lexer transmits the inter-
            # word space as its own slot (see ``Embedding._token_stream``
            # appending ``\x00`` after ``parse(lex='words')``), so the
            # per-slot decode for "hello world" yields
            # ['hello', ' ', 'world', '\x00', '\x00', ...]. The display
            # contract is "show non-whitespace word tokens"; spaces and
            # nulls are transmitted but not displayed. Without
            # ``w.strip()`` the prior clip ``words[:expected]`` would
            # consume the space token slot and drop the real second
            # word.
            words = [w for w in words
                     if w not in ("", "\x00") and w.strip()]
            expected = (self._input_token_count(originals[i])
                        if i < len(originals) else None)
            if expected is not None:
                words = words[:expected]
            rendered.append(" ".join(words))
        return rendered

    def _reconstructionReport(self):
        """Run a final eval pass and print input reconstruction per row."""
        data = self.inputSpace.data
        split = "test"
        split_input = getattr(data, f"{split}_input", None)
        split_output = getattr(data, f"{split}_output", None)
        if self._data_len(split_input) == 0:
            split = "train"
            split_input = getattr(data, f"{split}_input", None)
            split_output = getattr(data, f"{split}_output", None)
        n_rows = self._data_len(split_input)
        if n_rows == 0:
            return
        try:
            max_rows = int(os.environ.get("BASIC_RECON_REPORT_MAX", "64"))
        except ValueError:
            max_rows = 64
        if max_rows > 0 and n_rows > max_rows:
            TheMessage(f"=== Input Reconstruction ({split}) ===")
            TheMessage(
                f"  skipped: {n_rows} rows exceeds "
                f"BASIC_RECON_REPORT_MAX={max_rows}; set it to 0 to "
                f"print every row.")
            return

        self.set_sigma(0)
        try:
            _, _, allOut, allIn = self.runEpoch(batchSize=n_rows, split=split)
        finally:
            self.set_sigma(0.5)

        if not isinstance(allIn, torch.Tensor) or allIn.numel() == 0:
            TheMessage(f"=== Input Reconstruction ({split}) ===")
            TheMessage("  (no reconstructed input was produced)")
            return

        n = min(n_rows, int(allIn.shape[0]))
        originals = self._slice_data(split_input, n)
        labels = self._slice_data(split_output, n)
        reconstructed = self._decode_reconstructed_inputs(allIn[:n], originals)

        rows = []
        TheMessage(f"=== Input Reconstruction ({split}) ===")
        for i in range(n):
            original = self._display_value(originals[i])
            recon = reconstructed[i] if i < len(reconstructed) else ""
            label = self._display_value(labels[i]) if i < len(labels) else ""
            pred = ""
            if isinstance(allOut, torch.Tensor) and allOut.numel() > 0:
                pred = self._display_value(allOut[i])
            orig_words = original.replace("\x00", " ").split()
            recon_words = recon.replace("\x00", " ").split()
            match = orig_words == recon_words
            status = "OK" if match else "MISMATCH"
            css = "match" if match else "mismatch"
            TheMessage(
                f"  row[{i}] input={original!r} -> "
                f"reconstructed={recon!r} label={label} "
                f"predicted={pred} {status}")
            rows.append([
                original,
                f'<span class="{css}">{recon}</span>',
                label,
                pred,
                f'<span class="{css}">{"Yes" if match else "No"}</span>',
            ])

        TheReport.add_table(
            f"Input vs Reconstructed ({split})",
            ["Input", "Reconstructed", "Label", "Predicted", "Match"],
            rows)
        self.inputSpace.data.reconstructed_input = reconstructed
        if isinstance(allOut, torch.Tensor):
            self.inputSpace.data.reconstructed_output = [
                allOut[i].detach().cpu() for i in range(min(n, allOut.shape[0]))]


class BasicModel(BaseModel):
    """Core model: assembles Spaces into a forward and (optionally) reverse pipeline.

    The forward pass flows:
        InputSpace -> PartSpace -> ConceptualSpace -> WholeSpace -> OutputSpace

    The reverse pass mirrors it:
        OutputSpace -> WholeSpace -> ConceptualSpace -> PartSpace -> InputSpace

    Higher-order processing (subsymbolicOrder) inserts additional
    Percept/Concept/Symbol cycles between the first WholeSpace and OutputSpace,
    concatenating their symbol outputs before the final projection.

    ``create()`` builds the full space hierarchy.  ``create_from_config()`` is the
    XML-driven factory that reads architecture and training parameters from config,
    then delegates to ``create()``.
    """
    name = "BasicModel"

    def create_from_config(self, config_path=None, model_type=None, data=None):
        """Delegate XML-driven construction to BaseModel.

        Thin override that exists only to ensure the BasicModel
        instance handles the call (subclass dispatch); the actual
        construction logic lives in ``BaseModel.create_from_config``.
        """
        return super().create_from_config(config_path, model_type=model_type, data=data)

    def create(self, nInput, nPercepts, nConcepts, nSymbols, nWords=16, nOutput=32,
               subsymbolicOrder=1,
               model_type="numeric", data=None,
               reconstruction_scale=0.5, what_scale=0.7, where_scale=0.2, when_scale=0.1):
        """Build the full space hierarchy from architecture parameters.

        Always dispatches to ``_create_per_stage``: per-stage with
        T=subsymbolicOrder is the single construction path. At T=1 it
        reduces to a single ConceptualSpace + WholeSpace stage,
        producing the same observable output as the legacy flat path.

        Args:
            nInput/nPercepts/nConcepts/nSymbols/nOutput: object counts per space.
            nWords: object count for the SyntacticSpace.
            subsymbolicOrder: number of [Percept->Concept->Symbol] cycles.
            model_type: the data space_role -- "embedding" (text/LM) or "numeric"
                (dense slab). Was the architecture-level modelType.
        """
        return self._create_per_stage(
            nInput, nPercepts, nConcepts, nSymbols, nWords=nWords,
            nOutput=nOutput, subsymbolicOrder=subsymbolicOrder,
            model_type=model_type, data=data,
            reconstruction_scale=reconstruction_scale, what_scale=what_scale,
            where_scale=where_scale, when_scale=when_scale)

    def _make_input_space(self, rawInputShape, spaceShape, inputShape, model_type):
        return InputSpace(rawInputShape, spaceShape, inputShape, model_type=model_type)

    def _make_perceptual_space(self, inputShape, spaceShape, outputShape):
        try:
            demuxed = TheXMLConfig.space("InputSpace", "demuxed")
        except KeyError:
            demuxed = False
        if demuxed:
            return ModalSpace(inputShape, spaceShape, outputShape)
        return PartSpace(inputShape, spaceShape, outputShape)

    def build_pipelines(self):
        """Phase 2: assemble stem/body/head pipelines.

        Always dispatches to ``_build_pipelines_per_stage`` (the
        single per-stage construction path).
        """
        return self._build_pipelines_per_stage()

    # -- Per-stage helpers ---------------------------------------------
    # Pair-merge primitives that previously lived here
    # (``_pair_merge`` / ``_pair_unmerge``) were retired 2026-05-14
    # alongside the reverse pipeline; ``GrammarMergeGlue.forward`` and
    # ``.reverse`` (top of file) carry the live merge/unmerge math.



    def _bound_concept_input(self, x):
        """Keep recurrent concept inputs inside ConceptualSpace's logit domain."""
        if getattr(self.conceptualSpace, "nonlinear", False):
            return x.clamp(min=-1 + epsilon, max=1 - epsilon)
        return x

    def _derive_use_grammar(self):
        """Derive ``useGrammar`` from the configured grammar rules.

        Returns ``"all"`` when the grammar contains any non-default
        rule (anything beyond unary ``pi`` / ``sigma`` substrate
        folds), else ``"none"``. Replaces the retired
        ``<SymbolSpace><useGrammar>`` XML knob — the grammar XML itself
        is now the sole source of truth for whether the chart fires
        and whether per-stage merge glue is wired.
        """
        try:
            from Language import TheGrammar
            TheGrammar._ensure_configured()
            for rule in getattr(TheGrammar, 'rules', []) or []:
                mn = getattr(rule, 'method_name', None)
                arity = int(getattr(rule, 'arity', 1) or 1)
                if mn is None:
                    continue
                if mn in ('pi', 'sigma') and arity == 1:
                    continue
                return "all"
        except Exception:
            pass
        return "none"

    def End(self):
        """Per-batch teardown. Cascades End() to every Space.

        Released after forward + reverse + loss have consumed the cached
        state. Called from runBatch.
        """
        for space in self.spaces:
            if hasattr(space, 'End'):
                space.End()

    def Finish(self, symbols):
        """Project concatenated symbols to task output via OutputSpace.

        Output-range denormalization happens here (not in OutputSpace.forward)
        so the space pipeline stays global-data-free.
        """
        if isinstance(symbols, torch.Tensor):
            self.outputSpace.subspace.set_event(symbols)
            symbols = self.outputSpace.subspace
        # Phase 1.5: ``self.outputs`` was the subsumed back-ref alias
        # name; this Finish-local produce-then-materialize is the only
        # writer/reader, so keep it as a local (no ``self`` attribute,
        # behaviour identical).
        outputs = self.outputSpace.forward(symbols)
        if self.outputSpace.nonlinear_output:
            outputData = outputs.materialize(mode="activation")
        else:
            outputData = outputs.materialize()
        outputData = self.normalizer.denormalize(outputData, which="output")
        if self.plot:
            TheReport.plotActivations(figure=1, symbols=symbols)
        return outputData
    def set_reading(self, desire=True, valence=1.0):
        """READING mode (Architecture sec C, simplified law): desire (or,
        with valence < 0, hate) the frozen hard-coded 'reading' concept;
        while on, each batch's staged word-whole rows are desired too --
        the hard-coded concept -> word-isolating-wholes wiring. The
        surface's seen-decay fades it after set_reading(False)."""
        cs0 = (self.conceptualSpaces[0]
               if getattr(self, 'conceptualSpaces', None) else None)
        if not desire:
            object.__setattr__(self, "_reading_desire", None)
            return None
        v = float(valence)
        object.__setattr__(self, "_reading_desire", v)
        if cs0 is None or not cs0._sparse_active():
            return None
        cid = getattr(self, "_reading_cid", None)
        if cid is None:
            cid = cs0.mint_frozen_concept("reading")
            object.__setattr__(self, "_reading_cid", cid)
        row = cs0._csw_row_of(cid)
        if row is not None:
            cs0.prime_desire(torch.tensor([int(row)]), valence=v)
        return cid

    @torch.no_grad()
    def _primed_reading_step(self):
        """Hard-coded readingAttention (sec C, simplified law): scope =
        the span of the hottest-primed word-whole. Spans come from the
        stem's staging (ws0); the slot->row selections and the heat come
        from the CANONICAL priming surface (the terminal WS codebook, Alec
        2026-07-12) -- only canonical-stamp rows may index it. Writes
        ``wholeSpaces[0]._passback_scope_where`` (the learned producer's
        contract). Silent no-op when spans / stamp / surface are dark or
        the canonical stamp is not slot-aligned with the staged spans."""
        ws0 = (self.wholeSpaces[0]
               if getattr(self, "wholeSpaces", None) else None)
        if ws0 is None:
            return
        spans = getattr(ws0, "_staged_analysis_spans", None)
        ws_c = ws0._priming_target()
        idx = getattr(ws_c, "_stage0_indices", None)
        b = ws_c.priming_weights()
        if (spans is None or idx is None or b is None
                or not torch.is_tensor(spans) or not torch.is_tensor(idx)
                or int(idx.shape[0]) != int(spans.shape[0])):
            return
        B, K = int(spans.shape[0]), int(spans.shape[1])
        k_slots = min(K, int(idx.shape[1]))
        rows = idx[:, :k_slots].long().clamp(0, int(b.shape[0]) - 1)
        heat = b.to(rows.device)[rows]                   # [B, k_slots]
        k_star = heat.argmax(dim=1)                      # [B]
        scope = spans[torch.arange(B, device=spans.device), k_star]
        object.__setattr__(ws0, "_passback_scope_where", scope.float())

    @torch.no_grad()
    def _assemble_relevance_priority(self, cut_cs, stage, last_cs, settled):
        """The simplified relevance law (Architecture sec C): ONE quadratic
        priming surface per space -- SEEN rows primed by being perceived,
        DESIRED/HATED rows by signed intent. The CS surface projects
        directly onto the pyramid's inventory rows as the ranking score
        (boost - 1: neutral 0, desire positive, hate negative). Pure READ:
        the SEEN/DESIRE writes live in ``_prime_seen_step`` (unconditional,
        both paths, once per batch)."""
        b = cut_cs.priming_weights()
        if b is None:
            return None
        return (b - 1.0).unsqueeze(-1)               # [N, 1] signed score

    @torch.no_grad()
    def _prime_seen_step(self):
        """UNCONDITIONAL SEEN priming (Alec 2026-07-12): perception itself
        primes -- once per batch, on both paths, no ``<relevance>`` gate.
        WS: each stage's unity selections (the serial path stamps the
        terminal, the parallel pump stage 0); while READING is desired the
        same rows are desired too (the span -> slot -> row chain IS the
        projection). CS: the pyramid's admitted rows. Consumers stay put
        (the pyramid priority read; the <relevance>-gated reading scope)."""
        _rd = getattr(self, "_reading_desire", None)
        # Canonical-only WS write: rows must index the canonical (terminal)
        # codebook, and the terminal stamps its OWN unity selections on
        # both paths (per-stage on parallel, terminal-only on serial) --
        # the other stages' stamps are private-quantizer rows and dropped.
        _ws_list = getattr(self, "wholeSpaces", None) or []
        ws_c = _ws_list[-1] if len(_ws_list) else None
        if ws_c is not None:
            idx = getattr(ws_c, "_stage0_indices", None)
            if torch.is_tensor(idx):
                ws_c.prime_seen(idx)
                if _rd is not None:
                    ws_c.prime_desire(idx, valence=float(_rd))
        cs0 = (self.conceptualSpaces[0]
               if getattr(self, "conceptualSpaces", None) else None)
        rows = (getattr(cs0, "_cs_level_rows", None)
                if cs0 is not None else None)
        if rows:
            cs0.prime_seen(torch.cat([r.reshape(-1) for r in rows]))
        # CS->PS / CS->WS heat projection (Alec 2026-07-12): the diffused
        # concept heat lands on the word triples' PS pids (primed
        # RECOGNITION) and word-whole WS rows (primed RETRIEVAL -- the
        # surface the reading scope reads). Terminal WS: the relation
        # store's pos->row tables live there (== stage 0 on sO=0 configs).
        if cs0 is not None:
            cs0.project_priming_to_towers(
                getattr(self, "perceptualSpace", None),
                (self.wholeSpaces[-1]
                 if getattr(self, "wholeSpaces", None) else None),
                gain=self.priming_spread)

    def store_truths(self, entries):
        """Store user-supplied truth entries (the request-body TruthSet).

        LTM-BACKED PATH (``<ltmConsolidation>`` on, ``ltm_store`` built and
        attached to the TruthLayer): the canonical home is the unified
        ``TernaryTruthStore``. Each entry text runs through the real
        forward (``_ltm_ingest_truth_texts``, the same path XML
        provisioning uses) so the Change-1 observe-site push appends one
        real parsed end-state row per truth ALONGSIDE the STM-derived
        conversation rows; the landed rows get the effective DoT and the
        ``ORIGIN_USER`` tag + source text. Runtime user rows have
        replace-on-resubmit semantics (the client sends its full TruthSet
        each request), so previous ``ORIGIN_USER`` rows are compacted out
        first -- provisioned and conversation rows persist untouched.
        Finally ``truth_layer.sync_from_ltm()`` rematerializes the
        compatibility view so the flat-field readers (luminosity, falsity
        penalty, consistency, clarifications, assess) read the LTM-backed
        data.

        LEGACY PATH (gate off): encode via a runtime epoch into the
        standalone TruthLayer. Recording is governed by the continuous
        ``truthCriterion`` bar (no binary switch); to capture every
        provided gold truth the bar drops to 0 for the ingestion epoch,
        during which WholeSpace.forward() records the gold activations,
        then restores. Each stored activation is scaled by its effective
        DegreeOfTruth (incoming DoT multiplied by model-level ``<trust>``).

        Both paths end with the same consistency / clarification /
        assessment tail read by the serve layer.

        Args:
            entries: list of dicts with 'content' and 'trust' keys.
        """
        truth_layer = getattr(self.symbolSpace, 'truth_layer', None) if self.symbolSpace is not None else None
        if truth_layer is None:
            return

        # 1. Filter entries with text and trust
        texts, trusts = [], []
        for entry in entries:
            text = entry.get('content', '')
            if not text or not text.strip():
                continue
            trust = entry.get('trust')
            if trust is None:
                continue
            texts.append(text)
            trusts.append(self._effective_incoming_trust(trust))
        if not texts:
            return

        ltm_store = (getattr(self.symbolSpace, 'ltm_store', None)
                     if getattr(self, 'ltm_consolidation', False) else None)
        if ltm_store is not None and truth_layer.ltm_backed is ltm_store:
            # 2a. Consolidated path: user rows land in the unified LTM,
            # then the TruthLayer view is rebuilt from it.
            self._store_truths_into_ltm(ltm_store, truth_layer,
                                        texts, trusts)
        else:
            # 2b. Legacy path: reset the truth store and run a forced
            #    epoch over the gold texts. Recording is governed by the
            #    continuous ``truthCriterion`` bar (no binary arm): the
            #    WholeSpace.forward recording block captures the gold
            #    activations during this epoch exactly as it does during
            #    training. To capture ALL provided gold truths regardless
            #    of the configured bar, drop truthCriterion to 0 for the
            #    ingestion epoch, then restore it.
            truth_layer.clear()
            prev_tc = self.wholeSpace.truth_criterion
            self.wholeSpace.truth_criterion = 0.0
            self.eval()
            self.set_sigma(0)
            try:
                with torch.no_grad(), TheData.runtime_batch(texts):
                    self.runEpoch(batchSize=len(texts), split="runtime")
            finally:
                self.wholeSpace.truth_criterion = prev_tc

            # 3. Apply DoT to each stored activation
            n = min(truth_layer.count.item(), len(trusts))
            for i in range(n):
                truth_layer.truths[i] *= trusts[i]

            # 4. Attach sources/trusts for clarification surfacing (the
            # LTM path gets these from sync_from_ltm).
            stored_count = truth_layer.count.item()
            truth_layer._sources = (
                list(texts[:stored_count])
                + [None] * max(0, stored_count - len(texts))
            )
            truth_layer._trusts = (
                list(trusts[:stored_count])
                + [None] * max(0, stored_count - len(trusts))
            )

        # 5. Run the consistency report and cache any clarification
        # messages on the model for the serve layer to expose.
        basis = getattr(getattr(self.wholeSpace, 'subspace', None),
                        'basis', None)
        try:
            score, contradictions = truth_layer.consistency(
                basis=basis, return_report=True
            )
        except Exception:
            score, contradictions = None, []
        self._last_truth_score = float(score) if score is not None else None
        self._last_clarifications = (
            truth_layer.suggest_clarifications(basis=basis)
            if contradictions else []
        )
        # Phase 5: terminal paraconsistent assessment for the client
        # (support / conflict / ignorance) -- keeps "contested" (the
        # TruthSet splits on p) distinct from "unknown" (silent on p).
        try:
            self._last_truth_assessment = truth_layer.assess(basis=basis)
        except Exception:
            self._last_truth_assessment = None

    @torch.no_grad()
    def _store_truths_into_ltm(self, store, truth_layer, texts, trusts):
        """``store_truths``' consolidated arm: land the user TruthSet in
        the unified LTM and rebuild the TruthLayer view.

        Ordering: the lazy XML provisioning fires FIRST (mirroring the
        runEpoch trigger) so provisioned rows keep the earliest ticks even
        when the first thing a served model does is ingest a user
        TruthSet. Then previous ``ORIGIN_USER`` rows are compacted out
        (replace-on-resubmit), the texts run through the real forward
        (rows land via the observe-site push, NOT gated by
        ``truthCriterion``), each landed row gets its effective DoT +
        user origin + source text, and the view is rematerialized."""
        if not getattr(self, '_ltm_provisioned', True):
            self._ltm_provisioned = True
            # Skip provisioning when the store already carries its XML
            # truthSet: ``_ltm_provisioned`` is a transient attribute (not in
            # the state_dict), so a LOADED checkpoint that was provisioned
            # before saving comes back with the flag reset -- re-provisioning
            # would duplicate those rows (they survive a stateless reload).
            already = len(store.rows_of_origin(store.ORIGIN_PROVISIONED)) > 0
            if not already:
                try:
                    self.provision_ltm()
                except Exception:
                    pass
        store.clear_origin(store.ORIGIN_USER)
        per_text_rows = self._ltm_ingest_truth_texts(store, texts)
        for (start, end), text, trust in zip(per_text_rows, texts, trusts):
            for row in range(start, end):
                store.set_trust(row, trust)
                store.set_origin(row, store.ORIGIN_USER, text=text)
        truth_layer.sync_from_ltm()

    # -- LTM consolidation: XML TruthSet provisioning ------------------
    @torch.no_grad()
    def provision_ltm(self):
        """Append the XML ``<truthSet>`` rows to the consolidated LTM by
        RUNNING THE TRUTH TEXTS THROUGH THE REAL FORWARD PIPELINE (LTM
        consolidation FU1+FU2, Change 3, 2026-06-18; doc/specs/mereological-
        order-raising.md "Truth / Ideas processing").

        No-op (returns 0) unless ``<ltmConsolidation>`` is on AND an
        ``ltm_store`` exists AND there is a ``<truthSet>``. Each truth text is
        run through the REAL forward (``self.eval()`` / ``self.set_sigma(0)``
        then, PER TEXT, ``with no_grad(), TheData.runtime_batch([text]):
        self.forward(inputSpace.prepInput(...))`` -- the same working
        inference forward :meth:`_infer_ir` uses; the documented
        ``runEpoch(split="runtime")`` entry has a no-batch_override raise on
        its inference fast path, so the equivalent ``prepInput`` + ``forward``
        is driven directly). The Change-1 store-append (the observe-site
        conversation push) fires DURING each forward, appending one REAL
        parsed end-state row per truth, in submission order. After all texts
        the rows ``[n_before : len(store)]`` are exactly the provisioned
        truths: each gets its effective XML ``trust`` overwritten
        (``set_trust``; XML DoT multiplied by model-level ``<trust>``), the
        ``ORIGIN_PROVISIONED`` tag + source text (``set_origin``) and,
        when the entry carries a ``kind`` (``partOf`` / ``implies`` /
        ``other``), its ``rel_type`` overridden to the tagged relation.

        This REPLACES the old bounded mean-pool encode + the NP1=VP=NP2
        relation hack: the encoding is now a REAL parse. Returns the number
        of rows appended.

        PARSE-FIDELITY CAVEAT (documented): what the parse yields depends on
        the config's grammar / vocabulary. An IN-GRAMMAR relative sentence
        parses to a real depth-3 ``[predicate, idea1, idea2]`` end-state (a
        relation row); out-of-grammar / out-of-vocab text yields WHATEVER the
        parser produces (often a depth-1 absolute idea) -- but the encoding is
        REAL (no placeholder mean-pool). A ``kind`` tag still forces the
        ``rel_type`` so a relation is stored as a relation even when the
        surface parse collapsed it (the row's slots are then the real parsed
        end-state, not the NP1=VP=NP2 hack)."""
        if not getattr(self, 'ltm_consolidation', False):
            return 0
        ss = getattr(self, 'symbolSpace', None)
        store = getattr(ss, 'ltm_store', None) if ss is not None else None
        if store is None:
            return 0
        ts = TheXMLConfig.get("architecture.truthSet", default=None)
        if not ts:
            return 0
        entries = ts.get('truth') if isinstance(ts, dict) else None
        if entries is None:
            return 0
        if isinstance(entries, dict):           # singleton un-listed by parser
            entries = [entries]
        kind_map = {
            'partof': TernaryTruthStore.REL_PARTOF,
            'implies': TernaryTruthStore.REL_IMPLIES,
            'other': TernaryTruthStore.REL_OTHER,
        }
        # Gather (text, trust, kind) in order; skip blank texts.
        texts, trusts, kinds = [], [], []
        for ent in entries:
            if not isinstance(ent, dict):
                continue
            text = ent.get('text') or ent.get('_') or ''
            text = str(text).strip()
            if not text:
                continue
            try:
                trust = self._effective_incoming_trust(
                    ent.get('trust', 0.0) or 0.0)
            except (TypeError, ValueError):
                trust = 0.0
            kind = str(ent.get('kind', '') or '').strip().lower()
            texts.append(text)
            trusts.append(trust)
            kinds.append(kind)
        if not texts:
            return 0

        n_before = len(store)
        per_text_rows = self._ltm_ingest_truth_texts(store, texts)

        # Overwrite each text's landed rows with the XML trust, tag the
        # provisioned origin + source text, and override the rel_type from
        # the entry's kind (when tagged). Robust to a text landing 0 rows
        # (the range is empty) or >1 (all its rows get the same trust /
        # kind).
        for (start, end), text, trust, kind in zip(
                per_text_rows, texts, trusts, kinds):
            for row in range(start, end):
                store.set_trust(row, trust)
                store.set_origin(
                    row, TernaryTruthStore.ORIGIN_PROVISIONED, text=text)
                if kind in kind_map:
                    store.rel_type[row] = kind_map[kind]
        return max(0, len(store) - n_before)

    def _ltm_ingest_truth_texts(self, store, texts):
        """Run each truth text through the REAL forward (the working
        inference path, as _infer_ir): eval + sigma 0, then PER TEXT stage
        it via runtime_batch and run prepInput + forward. The Change-1
        observe-site store-append fires per sentence, landing one real
        parsed end-state row per truth in submission order. Per-text (B=1)
        keeps the row ordering 1:1 with the truth list (a B>1 batch would
        append B rows in a single boundary, mixing the order). Best-effort
        PER TEXT: a parse failure on one truth must not abort the others
        or crash the caller. Returns the per-text landed row RANGE
        ``[(start, end), ...]`` so trust / origin / rel_type overrides stay
        aligned even if a text lands 0 (parse failure) or >1 rows. Shared
        by XML provisioning (``provision_ltm``) and runtime user ingestion
        (``store_truths``)."""
        self.eval()
        self.set_sigma(0)
        per_text_rows = []                         # list of (start, end)
        for text in texts:
            start = len(store)
            try:
                with torch.no_grad(), TheData.runtime_batch([text]):
                    inp = self.inputSpace.prepInput(
                        list(TheData.train_input))
                    self.forward(inp)
                # Each truth text IS a sentence: fire the sentence-boundary
                # hard Reset the real reading loop fires, so the
                # Reset-driven seams (word autobind, recognition, syntactic
                # anchors) run during provisioning too (2026-07-13; without
                # this the provisioned words never bind).
                for _cs in (list(getattr(self, 'conceptualSpaces', []) or [])
                            or [getattr(self, 'conceptualSpace', None)]):
                    if _cs is not None and hasattr(_cs, 'Reset'):
                        _cs.Reset(hard=True)
            except Exception:
                pass
            end = len(store)
            per_text_rows.append((start, end))
        return per_text_rows

    def infer(self, text, max_length=None, mode='IR'):
        """IR-mode infill inference.

        Lexes the input, embeds it, applies ``mask_rate`` random
        masking to the embedded substrate (NULL_PERCEPT replacement),
        runs one forward pass, and decodes the body's prediction at
        each masked position via nearest-neighbor lookup against the
        lexicon.  Returns ``(slot_index, original_token, predicted_token)``
        triples.

        Sentence-level generation (the legacy AR / ARIR chat loop) is
        now provided by ``generate_sentence`` via ``InterSentenceLayer``
        (ARMA(p, q) over sentence reps).  ``mode`` defaults to ``'IR'``
        and is accepted for back-compat with callers that still pass
        ``mode='IR'``; any other value raises.
        """
        del max_length  # retained for signature back-compat only
        mode = 'IR' if mode is None else mode
        if str(mode).upper() != 'IR':
            raise NotImplementedError(
                f"infer(mode={mode!r}) retired 2026-05-14. Within-sentence "
                "training/inference is IR-only; for sentence-level "
                "generation use BasicModel.generate_sentence (Phase 4)."
            )
        return self._infer_ir(text)

    def _infer_ir(self, text):
        """IR-mode parallel-infill inference.

        Runs one forward pass under no_grad and reads the body's
        prediction at each masked position straight out of the post-
        body perceptual event (the same target ``runBatch`` uses for
        the IR reconstruction loss).  No reverse pipeline involved.
        """
        self.eval()
        self.set_sigma(0)

        with torch.no_grad(), TheData.runtime_batch([text]):
            inputTensor = self.inputSpace.prepInput(list(TheData.train_input))
            forwardInput, _symbols, _predictions, _ = self.forward(inputTensor)
            if forwardInput is None:
                return []
            pred_full = None
            if hasattr(self.perceptualSpace.subspace, 'materialize'):
                pred_full = self.perceptualSpace.subspace.materialize()

        mask_pos = getattr(self, "_ir_mask_positions", None)
        if (mask_pos is None or pred_full is None
                or not bool(mask_pos.any())):
            return []

        # Lex output for the original tokens at each slot. PartSpace
        # owns the codebook (the retired ``_peer_perceptual`` was just a
        # back-ref to it).
        peer = self.perceptualSpace
        codebook = peer.subspace.what
        last_meta = getattr(peer, '_forward_input', None) or {}
        all_tokens = last_meta.get('tokens') or [[]]
        tokens0 = all_tokens[0] if all_tokens else []

        if not hasattr(codebook, 'wv'):
            return []  # Codebook .what: token-infill readout is Embedding-only (5b keeps its mask for training)
        D = codebook.wv.vector_size
        K = pred_full.shape[1]
        indices = mask_pos[0, :K].nonzero(as_tuple=False).squeeze(-1).tolist()
        out = []
        for idx in indices:
            orig = tokens0[idx] if idx < len(tokens0) else ''
            pred_vec = pred_full[0, idx, :D].detach().unsqueeze(0)
            try:
                neighbors = codebook.wv.most_similar(pred_vec, k=1)
                pred = neighbors[0][0] if neighbors else ''
            except Exception:
                pred = ''
            out.append((int(idx), str(orig), str(pred)))
        return out

    def generate_sentence(self, seed_text="", max_chars=128):
        """Chat-loop sentence generation via InterSentenceLayer ARMA.

        Steps (per plan §4, updated for §9):
          1. Ask ``discourse.predict_next_end_state()`` for the predicted
             next-STM-end-state SHAPE ``(depth_hat, payload_hat[depth, D])``
             from the LTM end-state chain (the inter-level
             ``IntraSentenceLayer``). (Was: the single ARMA ``predict_next``
             sentence rep + ``cast`` lift.)
          2. Stage ``payload_hat`` per-slot on ``ConceptualSpace._c_prior``
             (slot-wise) so the body's forward adds it across the first
             ``depth`` STM slots before the codebook lookup.
          3. Run the IR forward over the seed text; sample tokens at
             masked positions by nearest-neighbor lookup in the
             perceptual codebook.
          4. Pool the produced S-space_role root into ``s_t`` and call
             ``discourse.observe(s_t)`` to commit to the AR ring.

        This is a minimal scaffold: a single IR forward + one-shot
        decode of every masked position.  Iterative mask-and-resample
        is left to a later refinement.

        Returns a list of decoded tokens (the predictions at the
        masked positions of the seed).
        """
        del max_chars  # reserved for the iterative variant
        discourse = (self.symbolSpace.discourse
                     if self.symbolSpace is not None else None)
        # Stage the C-prior from the predicted next-end-state SHAPE (Task 8,
        # plan §9): ``_intersentence_seed`` runs the inter-level
        # ``IntraSentenceLayer`` over the LTM end-state chain and returns
        # ``(depth_hat, payload_hat[depth_hat, D])`` already in C-space_role
        # (concept_dim) -- no ``cast`` needed (that lifted the OLD flat ARMA
        # rep; the inter-predictor already emits concept-width slots). Stage
        # ``payload_hat`` per-slot on ``_c_prior`` (slot-wise flag set) so the
        # body's forward adds it across the first ``depth`` STM slots.
        # Cold-start (empty chain -> depth 1 / zeros root): ``_intersentence_
        # seed`` returns None, so staging is skipped and the seed text is the
        # only signal (the degenerate zeros prior would be a no-op add anyway).
        #
        # A6 unification: this is the SAME ``_intersentence_seed`` (one
        # ``predict_next_end_state`` source) the forward's CS_{-1} seed sites
        # use -- the chat-loop prime and the forward seed never diverge into
        # two predictor calls. (generate_sentence primes whenever a predictor
        # exists; the forward additionally gates on ``prediction_mode``.)
        seed = self._intersentence_seed()
        if seed is not None:
            _depth_hat, payload_hat = seed
            self.conceptualSpace._c_prior = (
                payload_hat * float(self.sentence_priming_scale))
            self.conceptualSpace._c_prior_slotwise = True
        # Run the IR forward over the seed text.
        out = self._infer_ir(seed_text)
        # Commit the produced sentence to the ARMA ring.
        if discourse is not None and self._current_discourse_s is not None:
            discourse.observe(self._current_discourse_s)
        return out

    def _warn_zeroed_channel(self, site, detail):
        """Warn-once per site+config when a loss channel degrades to zero (5b fail-loud)."""
        try:
            cfg = str((TheXMLConfig._sources or ["?"])[-1])
        except Exception:
            cfg = "?"
        seen = getattr(self, "_zeroed_channel_warned", None)
        if seen is None:
            seen = set()
            object.__setattr__(self, "_zeroed_channel_warned", seen)
        key = (site, cfg)
        if key not in seen:
            seen.add(key)
            warnings.warn(f"{site} [config {cfg}]: {detail}", RuntimeWarning)

    def _align_output_pred(self, pred, tgt):
        """Reduce the head pred onto the label shape; None (after warn-once) when irreconcilable."""
        _pred = pred
        while _pred.dim() > tgt.dim():
            _pred = _pred.mean(dim=-1)
        # Equal-dim/unequal-shape (e.g. head [B,4,4] vs labels [B,1,1]): mean-reduce pred over label-singleton axes.
        if (_pred.shape != tgt.shape and _pred.dim() == tgt.dim()
                and all(p == t or t == 1
                        for p, t in zip(_pred.shape, tgt.shape))):
            for d, t in enumerate(tgt.shape):
                if t == 1 and _pred.shape[d] != 1:
                    _pred = _pred.mean(dim=d, keepdim=True)
        if _pred.shape == tgt.shape:
            return _pred
        self._warn_zeroed_channel(
            "output_shape_gate",
            f"head pred {tuple(pred.shape)} vs labels {tuple(tgt.shape)} "
            "irreconcilable after reduction; supervised output loss zeroed")
        return None

    def _reverse_event_loss(self, rev_ev, fwd_ev):
        """lossRev seam (nWhere=0 wiring fix, 2026-07-04): clip to the shared
        [K, D] window, then weight with the INPUT event layout's where/when
        band. ModelLoss's constructor band is canonical_shape("OutputSpace")
        == (0, 0) -- right for lossOut, wrong here: the muxed input event's
        2-dim where band was diluted ~512x inside the what-scaled mean-MSE.
        Also serves the D3 serial-path reconstruction (its twin site): the
        D3 compare is the same reverse-vs-input-event layout.
        """
        Kr = min(rev_ev.shape[1], fwd_ev.shape[1])
        Dr = min(rev_ev.shape[-1], fwd_ev.shape[-1])
        sub = getattr(getattr(self, "inputSpace", None), "subspace", None)
        nw = getattr(sub, "nWhere", None)
        nn_ = getattr(sub, "nWhen", None)
        if nw is None or nn_ is None:
            # Fail-loud (5b pattern): an unknowable band must announce itself, never silently take what_scale.
            self._warn_zeroed_channel(
                "lossrev_band_unknown",
                "input event band widths unavailable (no inputSpace."
                "subspace nWhere/nWhen); lossRev treats the whole event "
                "as what")
            nw, nn_ = 0, 0
        elif Dr != rev_ev.shape[-1] or Dr != fwd_ev.shape[-1]:
            # The width clip cuts/misaligns the trailing band; band scales cannot be applied honestly.
            self._warn_zeroed_channel(
                "lossrev_band_clipped",
                f"lossRev width clip (rev {int(rev_ev.shape[-1])} vs fwd "
                f"{int(fwd_ev.shape[-1])}) misaligns the trailing "
                f"where/when band; the clipped event takes what_scale")
            nw, nn_ = 0, 0
        return self.loss.compute(rev_ev[:, :Kr, :Dr],
                                 fwd_ev[:, :Kr, :Dr].detach(),
                                 nWhere=int(nw), nWhen=int(nn_))

    def _masked_event_loss(self, pred, target, mask):
        """lossIn seam (silent-band wiring fix, 2026-07-04): the masked-LM
        compares PERCEPTUAL events (muxed ``[what|where|when]``,
        canonical_shape("PartSpace") == (2, 2)) -- weight the trailing band
        with the live percept layout, not ModelLoss's constructor (0, 0)
        (the where band was diluted ~512x inside the what-scaled mean).
        """
        K = min(pred.shape[1], target.shape[1], mask.shape[1])
        sub = getattr(getattr(self, "perceptualSpace", None), "subspace", None)
        nw = getattr(sub, "nWhere", None)
        nn_ = getattr(sub, "nWhen", None)
        if nw is None or nn_ is None:
            # Fail-loud (5b pattern): an unknowable band must announce itself, never silently take what_scale.
            self._warn_zeroed_channel(
                "lossin_band_unknown",
                "percept event band widths unavailable (no perceptualSpace."
                "subspace nWhere/nWhen); masked lossIn treats the whole "
                "event as what")
            nw, nn_ = 0, 0
        elif pred.shape[-1] != target.shape[-1]:
            # compute_masked's internal width clip would cut/misalign the trailing band; band scales cannot apply honestly.
            self._warn_zeroed_channel(
                "lossin_band_clipped",
                f"masked lossIn width clip (pred {int(pred.shape[-1])} vs "
                f"target {int(target.shape[-1])}) misaligns the trailing "
                f"where/when band; the clipped event takes what_scale")
            nw, nn_ = 0, 0
        return self.loss.compute_masked(pred[:, :K, :], target[:, :K, :],
                                        mask[:, :K],
                                        nWhere=int(nw), nWhen=int(nn_))

    def create_ir_mask(self, percept_subspace):
        """Whole-slab IR mask (BERT-style hide-a-token; UNCHANGED).

        This is the **non-grammar / whole-slab** masking, kept
        byte-identical: ``_per_word_enabled=False`` configs
        (model.xml / idempotent.xml) take ``_forward_body``'s whole-slab
        path which calls this once on pass-0's perceptual event, and
        ``runBatch``'s ``compute_masked`` consumes
        ``_ir_pre_mask_input`` / ``_ir_mask_positions`` exactly as
        before. Rework B's gaussian attentional window
        (:meth:`apply_gaussian_window`) + D3 reconstruction loss replace
        THIS only on the per-word grammar path; the whole-slab path is
        untouched (the spec's "whole-slab byte-identical" invariant).

        Replaces embeddings at random positions with NULL_PERCEPT
        (Embedding ``.what``: the learnable NULL row; Codebook ``.what``,
        the radix/meronomy PS, has no NULL row -- the all-zeros MASK
        vector is injected instead, per the Spaces.py MPHF-table note).
        Captures the pre-mask embedded event as the loss target and
        stashes it on ``self`` along with the per-position bool mask so
        the loss path can compute reconstruction error at masked
        positions only. Mask injection edits only the WHAT slice of the
        muxed event so the body still has WHERE/WHEN positional info at
        masked slots. Padding slots (codebook index 0, byte ``\\x00``)
        are excluded so the model isn't asked to "predict" trailing
        zeros.
        """
        self._ir_mask_positions = None
        self._ir_pre_mask_input = None
        if percept_subspace is None:
            return
        if hasattr(percept_subspace, 'is_empty') and percept_subspace.is_empty():
            return
        event_basis = percept_subspace.event
        if event_basis is None:
            return
        # Spec doc/specs/2026-05-21-subspace-slot-architecture.md: per-batch
        # event reads go through ``materialize(mode='event')``, not the
        # ``_active_payload`` shadow on ``.event.getW()``. The result is
        # the same per-batch ``[B, N, D]`` slab pre-migration; post-
        # migration it's reconstructed from prototype + selection.
        event = percept_subspace.materialize(mode="event")
        if event is None or event.dim() != 3:
            return
        codebook = percept_subspace.what
        cb_w = codebook.getW() if hasattr(codebook, 'getW') else None
        if not torch.is_tensor(cb_w) or cb_w.dim() != 2:
            return  # numeric / rowless .what: cannot stage a mask (runBatch warns on the zeroed channel)
        if (not hasattr(codebook, 'null_percept_idx')
                and not torch.is_grad_enabled()):
            return  # 5b Codebook path is train-step-only: no_grad forwards stay inert (HEAD-parity determinism)

        B, K, D = event.shape
        dev = event.device

        # Sample mask positions [B, K] bool.
        rate = float(self.mask_rate)
        if rate <= 0.0 or rate > 1.0:
            return
        mask = torch.bernoulli(
            torch.full((B, K), rate, device=dev)).bool()

        # Exclude padding slots (codebook index 0 == byte \x00 sentinel)
        # — but only where the what-index column is INFORMATIVE.
        # Byte-staged configs (MM_20M / MM_xor) leave ``_index``'s
        # what-index column uniformly 0 even for REAL content, so the
        # unconditional ``what_idx != 0`` exclusion silently zeroed the
        # whole mask — ``compute_masked`` then returned exactly 0.0
        # every batch, the reconstruction objective never engaged, and
        # nothing stopped the body from collapsing content (the
        # flat-0.5-prediction / blank-reconstruction failure,
        # 2026-06-11). When the column is degenerate (all zero), fall
        # back to NULL-CONTENT padding detection (an all-zero ``.what``
        # slice — what a padding slot actually carries). The selection
        # is a tensor ``torch.where`` on a 0-dim flag — on-device,
        # sync-free — and word-staged configs (informative column,
        # e.g. XOR_exact's verified seed basin) keep the index rule
        # byte-identically.
        active = getattr(percept_subspace, '_index', None)
        if (active is not None and active.dim() == 3
                and active.shape[-1] >= 1):
            what_idx = active[:, :K, 0].long()
            if what_idx.shape == mask.shape:
                nWhat_excl = int(cb_w.shape[-1])
                content_mass = event[..., :nWhat_excl].abs().amax(dim=-1)
                idx_informative = what_idx.amax() > 0      # 0-dim, on-device
                keep = torch.where(idx_informative,
                                   what_idx != 0,
                                   content_mass > 0)
                mask = mask & keep
        elif not hasattr(codebook, 'null_percept_idx'):
            # 5b Codebook path (no _index): exclude zero-content pad slots by content mass.
            mask = mask & (
                event[..., :int(cb_w.shape[-1])].abs().amax(dim=-1) > 0)

        # Snapshot pre-mask embedded event as the loss target.
        # Detach so backward through the loss target doesn't double up
        # on the forward graph (we still get gradient through pred via
        # the brick body + reverse pipeline).
        self._ir_pre_mask_input = event.detach().clone()
        self._ir_mask_positions = mask

        # No `if mask.any()` early-out: an all-False boolean-mask write
        # selects zero rows, so the body below is a content-identical
        # no-op when nothing is masked (new_event == event). Running it
        # unconditionally keeps the brick CUDA-graph-capturable -- the
        # data-dependent `bool(mask.any())` skip was a host sync (see
        # doc/BrickHostSyncStatus.md residual C; same shape as the
        # `_snap_content` fix).
        if hasattr(codebook, 'null_percept_idx'):
            null_vec = cb_w[codebook.null_percept_idx]  # [nWhat] learnable NULL row
        else:
            # 5b: Codebook (radix/meronomy PS) has no NULL row -- MASK is the all-zeros vector (Spaces.py MPHF note).
            null_vec = cb_w.new_zeros(cb_w.shape[-1])
        nWhat = int(null_vec.shape[-1])
        # Replace the WHAT slice at masked positions; preserve WHERE/WHEN.
        # Dense `torch.where` instead of boolean-mask assignment
        # ``new_event[mask, :nWhat] = ...``: advanced boolean indexing is
        # data-dependent (the selected-row count must be read to host) ->
        # an implicit cudaMemcpyDtoH that breaks CUDA-graph capture. The
        # where form is static-shape, fully on-device, and bit-identical
        # (same values written; all-False mask -> no-op). [B,K]->[B,K,1]
        # broadcasts over the nWhat channel slice.
        new_event = event.clone()
        m = mask.unsqueeze(-1)
        nv = null_vec.to(new_event.dtype)
        new_event[..., :nWhat] = torch.where(
            m, nv, new_event[..., :nWhat])
        # Spec doc/specs/2026-05-21-subspace-slot-architecture.md Setter
        # API: per-batch event writes flow through ``set_event`` (which
        # routes appropriately for codebook-bearing vs plain-Tensor
        # ``.event`` slots), NOT through ``basis.setW`` directly. The
        # mocked test path provides ``percept_subspace.set_event``; the
        # production path is the same call.
        if hasattr(percept_subspace, "set_event"):
            percept_subspace.set_event(new_event, compute_activation=False)
        else:
            event_basis.setW(new_event)  # legacy fallback for older mocks

    def gaussian_window_word(self, full_seq, center_k):
        """Rework B (1): GAUSSIAN ATTENTIONAL WINDOW over the WHOLE
        percept sequence, centered at the processed word position
        ``center_k``; returns word k's contextual representation
        (``[B, 1, D]``).

        This **replaces wholesale** the prior BERT-style hide-a-token
        ``create_ir_mask`` **on the per-word grammar path**. It is
        **NOT** target-hiding: the word at the gaussian center
        (``center_k``) is preserved (multiplier ~1) and words *far*
        from ``center_k`` are zeroed by the gaussian tail
        (multiplier ->0).

        ``full_seq`` is the COMPLETE per-sentence input percept slab
        ``[B, T, D]`` (``inputSpace._ar_embedded`` -- every word). For
        the processed word at center ``k`` a single gaussian envelope
        over the whole sequence is built:

            w_i = exp(-(i - k)^2 / (2 * sigma^2))   over percept i in T,
            sigma = maskRate * N_percepts            (D1; maskRate=0.15,
                                                      N_percepts == T)

        normalized so the center ``w_k ~= 1``. The gaussian-windowed
        percepts are mapped into conceptual space and **summed** (the
        weighted sum over the whole sequence) so word ``k``'s
        representation carries a *contextual trace* -- the faint
        gaussian-weighted contribution of nearby words (the local
        context IS the embedding signal). Per-word context washes out
        across the whole-sentence D3 reconstruction (overlapping
        gaussians average). The summed contextual word is what the
        per-word loop feeds PS->CS->STM (one concept per word, the
        existing loop structure preserved; only the per-word INPUT now
        carries the gaussian contextual trace instead of a raw slice).

        Pure static tensor ops (a gaussian over a fixed position arange
        + a single weighted sum), no host sync,
        CUDA-graph-capturable -- the bernoulli/where data-dependent
        path is gone. Returns ``full_seq``'s single-slot fallback
        (``full_seq[:, k:k+1, :]``) unchanged when the envelope is
        inapplicable (degenerate rate / shape) so the caller's existing
        per-word slice contract still holds.
        """
        if (full_seq is None or not torch.is_tensor(full_seq)
                or full_seq.dim() != 3):
            return full_seq
        B, T, D = full_seq.shape
        dev = full_seq.device
        k = int(center_k)
        if k < 0:
            k = 0
        if k > T - 1:
            k = T - 1
        rate = float(self.mask_rate)
        if rate <= 0.0 or rate > 1.0 or T <= 0:
            return full_seq[:, k:k + 1, :]
        # sigma = maskRate * N_percepts (D1; N_percepts == T, the whole
        # sentence's percept-sequence length).
        sigma = max(rate * float(T), 1e-3)
        pos = torch.arange(T, device=dev, dtype=full_seq.dtype)  # [T]
        w = torch.exp(-((pos - float(k)) ** 2) / (2.0 * sigma * sigma))
        # Normalize so the center is exactly ~1 (exp(0)==1 already; the
        # explicit divide keeps the contract robust if sigma underflows).
        w = w / w.max().clamp(min=1e-12)                         # [T]
        # Reported-metric side channel only (the D3 trainable loss reads
        # the COMPLETE UNMASKED _ar_embedded directly, NOT these).
        self._ir_mask_positions = (
            (w <= 0.5).view(1, T).expand(B, T).clone())
        self._ir_pre_mask_input = full_seq.detach().clone()
        # The gaussian-windowed percepts SUMMED over the whole sequence
        # -> word k's contextual representation [B, 1, D]. Grad flows
        # (no detach) so the contextual trace shapes upstream params.
        wcol = w.view(1, T, 1).to(full_seq.dtype)
        windowed = full_seq * wcol                                # [B,T,D]
        return windowed.sum(dim=1, keepdim=True)                  # [B,1,D]

    def word_span_window(self, full_seq, center_k, word_idx):
        """Increment 2 (HARD-MASK-TO-WORD-SPAN; doc/specs/mereological-order-
        raising.md "Serial-mode word-at-a-time loop"). The HARD same-word window
        that replaces :meth:`gaussian_window_word` on the serial
        ``serialObjectMeta`` path: returns word ``center_k``'s representation
        ``[B,1,D]`` as the masked SUM over the slots belonging to the SAME word
        as slot ``center_k`` (so PartSpace processes the ACTIVE WORD ONLY — no
        part with a ``.where`` outside the word). Same ``[B,1,D]`` contract as
        the gaussian, so the caller's shape guard accepts it unchanged.

        ``word_idx`` is a ``[B,T]`` long per-slot word index (slots of one word
        share an id). Pure static tensor ops — DtoH-free, CUDA-graph-capturable
        (``k`` is a host-int constant per unrolled loop position; the mask is a
        fixed-shape elementwise compare). When ``word_idx`` is unavailable
        (None / wrong shape — e.g. byte mode has no word grouping), falls back to
        the single slot ``full_seq[:, k:k+1, :]`` (in fused modes the active word
        IS its own slot; the byte fallback degrades safely)."""
        if (full_seq is None or not torch.is_tensor(full_seq)
                or full_seq.dim() != 3):
            return full_seq
        B, T, D = full_seq.shape
        k = int(center_k)
        if k < 0:
            k = 0
        if k > T - 1:
            k = T - 1
        if (word_idx is None or not torch.is_tensor(word_idx)
                or word_idx.dim() != 2 or int(word_idx.shape[0]) != B
                or int(word_idx.shape[1]) != T):
            return full_seq[:, k:k + 1, :]                # single-slot fallback
        center_id = word_idx[:, k:k + 1]                  # [B,1] (static slice)
        same = (word_idx == center_id)                    # [B,T] same-word mask
        mcol = same.to(full_seq.dtype).unsqueeze(-1)      # [B,T,1] hard 0/1
        return (full_seq * mcol).sum(dim=1, keepdim=True)  # [B,1,D] word rep

    def _prewarm_checkpoint_shapes(self):
        """Materialize lazy state whose final tensors ride in checkpoints.

        Resolve the serial grammar layer's non-owning cache and materialize the
        lazy category codebook. A training checkpoint contains the latter's
        final shapes, so they must exist before ``load_state_dict``. Canonical
        WholeSpace properties are not grown here: property growth is
        independent of concept capacity and requires an explicit
        optimizer/compiled-graph reset boundary. This helper is idempotent and
        shared by autoload and compilation pre-warm.
        """
        if getattr(self, "_stm_reducer_cached", None) is None:
            try:
                self._stm_reducer()
            except Exception:
                # Degenerate/no-grammar configs cache the absence, matching
                # the existing compiled-forward behavior.
                self._stm_reducer_cached = False
        if getattr(self, "_stm_unary_rewriter_cached", None) is None:
            self._stm_unary_rewriter()

        # Compatibility-only: old configs used WholeSpace.nVectors as a
        # terminal symbol-table target and relied on this pre-load resize.
        # propertyBasis=true never takes this path.
        if not bool(getattr(self, "wholePropertyBasis", False)):
            try:
                ws_cb = (self.wholeSpace.subspace.codebook()
                         if getattr(self, "wholeSpace", None) is not None
                         and hasattr(self.wholeSpace.subspace, "codebook")
                         else None)
                budget = int(TheXMLConfig.space(
                    "WholeSpace", "nVectors", default=0) or 0)
                if (ws_cb is not None and budget > 0
                        and int(ws_cb.nVectors) < budget):
                    ws_cb.grow_to(budget)
            except Exception:
                # A legacy config without WS/a codebook has no shape to grow.
                pass

        # Participation-category state is allocated on the first perception
        # forward in a fresh model, but autoload necessarily runs before that
        # forward. At this point SymbolSpace has configured the live grammar,
        # so materialize the requested terminal VQ on the same device as the
        # owning concept/property dictionary. Otherwise its learned
        # prototype/buffer keys in a checkpoint are classified as stale.
        category_owner = (
            getattr(self, "conceptualSpace", None)
            if bool(getattr(self, "wholePropertyBasis", False))
            else getattr(self, "wholeSpace", None))
        if (category_owner is not None
                and getattr(category_owner,
                            "_category_codebook_requested", False)
                and hasattr(category_owner, "category_codebook_enabled")
                and not category_owner.category_codebook_enabled()
                and hasattr(category_owner, "enable_category_codebook")):
            from Language import TheGrammar
            cb = getattr(category_owner, "similarity_codebook", None)
            if cb is None:
                cb = getattr(getattr(
                    category_owner, "subspace", None), "what", None)
            W = cb.getW() if cb is not None and hasattr(cb, "getW") else None
            device = W.device if torch.is_tensor(W) else None
            category_owner.enable_category_codebook(
                TheGrammar, device=device)

    def _aligned_serial_word_mode(self):
        """Return whether the canonical aligned per-word protocol is live."""
        return bool(
            getattr(self, "serial", False)
            and getattr(getattr(self, "inputSpace", None),
                        "_per_word_enabled", False)
            and getattr(self, "serial_object_meta", False)
            and getattr(self, "concept_binding", "mixing") == "aligned")

    def _aligned_serial_sparse_bank_mode(self):
        """Return whether aligned serial execution needs its sparse bank.

        Stage 0 resolves relation identities while the terminal CS consumes
        them.  The staged atoms are therefore sound only when both stages own
        the same physical codebook Parameter, an invariant of the canonical
        model that is checked here rather than assumed at decode time.
        """
        if not self._aligned_serial_word_mode():
            return False
        spaces = list(getattr(self, "conceptualSpaces", None) or ())
        if not spaces:
            return False
        terminal = getattr(self, "conceptualSpace", None)
        if terminal is None:
            terminal = spaces[-1]
        owner_cb = getattr(spaces[0], "similarity_codebook", None)
        terminal_cb = getattr(terminal, "similarity_codebook", None)
        sparse = any(
            bool(getattr(getattr(space, "similarity_codebook", None),
                         "sparse_lookup_grad", False))
            for space in spaces)
        if sparse and owner_cb is not terminal_cb:
            raise RuntimeError(
                "aligned serial sparse staging requires stage-0 and "
                "terminal ConceptualSpace to share one physical codebook")
        return sparse

    def _aligned_part_fold_ladder(self, event):
        """Fullgraph numerical PS ladder for one canonical word tick."""
        passes = tuple(range(int(self.subsymbolicOrder) - 1))
        return self.perceptualSpace.fold_event_ladder(
            event, passes, strict=True)

    def _aligned_whole_fold_ladder(self, event):
        """Fullgraph numerical WS ladder for one canonical word tick."""
        passes = tuple(range(int(self.subsymbolicOrder) - 1))
        return self.wholeSpace.fold_event_ladder(
            event, passes, strict=True)

    def _enable_mps_fullgraph_fold_ladders(self, compile_fn):
        """Compile the complete numerical PS and WS ladders on MPS.

        This is the useful fullgraph boundary for the canonical recurrence:
        each graph contains all three learned folds and their backward path,
        while excluding the stateful carrier, sparse lookup, STM, and host
        grammar machinery that make a whole word tick impractically large for
        the Metal compiler.  It replaces six tiny compile dispatches per word
        with two fullgraph calls.
        """
        os.environ.setdefault("BASICMODEL_MPS_IOBUF", "12")
        os.environ.setdefault("BASICMODEL_MPS_FUSE", "32")
        self._compiled_part_fold_ladder = compile_fn(
            self._aligned_part_fold_ladder, verbose=True, fullgraph=True)
        self._compiled_whole_fold_ladder = compile_fn(
            self._aligned_whole_fold_ladder, verbose=True, fullgraph=True)

    def _configure_lazy_fullgraph_word_loops(self, compile_fn, buckets):
        """Install, but do not lower, one fullgraph sentence loop per W bucket.

        The outer word bucket (W=16/32/64/128) and the masked residual
        constituent layout are both static *inside* that loop.
        Capture the *entire* W loop rather than a word cell replayed W times:
        this gives Inductor visibility of the recurrent carrier chain and
        permits fusion across adjacent word ticks. Lowering a complete
        forward/backward is intentionally deferred until a W bucket is first
        observed; a multi-day run pays that cost once per used W bucket, not
        once per residual spelling, batch, or word.

        Four separately-defined source functions are deliberate.  Dynamo keys
        the Python code object in addition to shapes; a closure factory would
        otherwise turn W=32 into a recompile of the W=16 graph.
        """
        widths = tuple(sorted(set(int(width) for width in buckets)))
        required = (16, 32, 64, 128)
        unsupported = tuple(width for width in widths
                            if width not in required)
        if unsupported:
            raise ValueError(
                "canonical fullgraph word loops support W=16,32,64,128; "
                f"got unsupported {list(unsupported)}")

        def _forward_W16(input_data):
            slab = self.inputSpace._ar_embedded_N
            if slab is None or int(slab.shape[1]) != 16:
                raise RuntimeError("fullgraph W=16 loop received a non-W=16 slab")
            # The complete word loop consumes the eager-staged InputSpace
            # slab, never the raw text tensor.  Keeping that variable-width
            # lexer tensor in the compiled callable's signature gave Dynamo a
            # second (unrelated) shape guard after the W loop had lowered.
            return self.forward(None)

        def _forward_W32(input_data):
            slab = self.inputSpace._ar_embedded_N
            if slab is None or int(slab.shape[1]) != 32:
                raise RuntimeError("fullgraph W=32 loop received a non-W=32 slab")
            return self.forward(None)

        def _forward_W64(input_data):
            slab = self.inputSpace._ar_embedded_N
            if slab is None or int(slab.shape[1]) != 64:
                raise RuntimeError("fullgraph W=64 loop received a non-W=64 slab")
            return self.forward(None)

        def _forward_W128(input_data):
            slab = self.inputSpace._ar_embedded_N
            if slab is None or int(slab.shape[1]) != 128:
                raise RuntimeError("fullgraph W=128 loop received a non-W=128 slab")
            return self.forward(None)

        object.__setattr__(self, "_compiled_word_step_sources", {
            16: _forward_W16,
            32: _forward_W32,
            64: _forward_W64,
            128: _forward_W128,
        })
        self._compiled_word_steps = {}
        self._compiled_word_loop_compile = compile_fn
        self._compiled_word_loop_fullgraph = True

    def _ensure_lazy_fullgraph_word_loop(self, width):
        """Return the one complete fullgraph loop for the active W bucket."""
        width = int(width)
        steps = self._compiled_word_steps
        compiled = steps.get(width)
        if compiled is not None:
            return compiled
        source = getattr(self, "_compiled_word_step_sources", {}).get(width)
        compile_fn = getattr(self, "_compiled_word_loop_compile", None)
        if source is None or compile_fn is None:
            raise RuntimeError(
                f"no lazy fullgraph word-loop source is configured for W={width}")
        TheMessage(
            "Canonical MPS fullgraph lowering: compiling complete "
            f"W={width} recurrent word loop (one-time cost for this bucket)")
        compiled = compile_fn(source, verbose=True, fullgraph=True)
        steps[width] = compiled
        return compiled

    def _stage_fixed_residual_part_capacity(self):
        """Present one fixed, masked constituent layout to a W-loop graph.

        ``P`` is an input layout extent, not the PS live-field width: PS/WS
        still reduce their raw constituents into their configured eight live
        locations.  It must nevertheless be static in the compiled outer
        loop.  Marking P dynamic let Inductor create a second, delayed
        specialization when a later sentence exposed a different radix
        spelling, defeating the one-fullgraph-per-W contract.

        Every word is padded with a valid masked identity percept to the
        fixed capacity (16 by default).  No real constituent is discarded.
        A word that cannot be represented by that radix/trie layout fails at
        the eager boundary with its actual width, rather than being truncated
        or silently requesting another full W-loop compilation.  Such a
        failure is the explicit signal to improve the trie/compaction policy,
        not to grow a hidden compilation bucket.
        """
        isp = self.inputSpace
        ids = getattr(isp, "_ar_word_part_ids", None)
        mask = getattr(isp, "_ar_word_part_mask", None)
        offsets = getattr(isp, "_ar_word_part_offsets", None)
        if (not torch.is_tensor(ids) or ids.dim() != 3
                or not torch.is_tensor(mask)
                or tuple(mask.shape) != tuple(ids.shape)):
            return None
        if offsets is not None and (not torch.is_tensor(offsets)
                                    or tuple(offsets.shape) != tuple(ids.shape)):
            raise RuntimeError(
                "serial residual-part offsets do not align with part ids")

        raw_width = int(ids.shape[-1])
        capacity = int(os.environ.get(
            "BASICMODEL_MPS_RESIDUAL_PARTS", "16"))
        if capacity < 3:
            raise ValueError(
                "BASICMODEL_MPS_RESIDUAL_PARTS must be at least 3; got "
                f"{capacity}")
        if raw_width > capacity:
            raise RuntimeError(
                "a radix word needs "
                f"{raw_width} residual percepts, beyond the fixed "
                f"compiled P={capacity} contract; refusing to truncate it")
        bucket = capacity

        # A valid physical row (zero) is gathered before the false mask
        # removes it.  ``-1`` would alias the final learnable codebook row
        # during the gather and is therefore not a semantic identity.
        pad_width = bucket - raw_width
        pad_shape = tuple(ids.shape[:-1]) + (pad_width,)
        padded_ids = torch.cat((
            ids,
            torch.zeros(pad_shape, dtype=ids.dtype, device=ids.device)),
            dim=-1)
        padded_mask = torch.cat((
            mask,
            torch.zeros(pad_shape, dtype=torch.bool, device=mask.device)),
            dim=-1)
        padded_offsets = None
        if torch.is_tensor(offsets):
            padded_offsets = torch.cat((
                offsets,
                torch.full(pad_shape, -1, dtype=offsets.dtype,
                           device=offsets.device)), dim=-1)
        isp._ar_word_part_ids = padded_ids
        isp._ar_word_part_mask = padded_mask
        isp._ar_word_part_offsets = padded_offsets

        # Keep the eager PartSpace record shape-coherent with the fixed
        # InputSpace presentation.  These fields must use the same physical
        # P as the source ids; otherwise an apparently static word loop gains
        # a hidden shape guard through the PartSpace staging record.
        ps = self.perceptualSpace
        fwd = getattr(ps, "_forward_input", None)
        if isinstance(fwd, dict):
            B, W = (int(ids.shape[0]), int(ids.shape[1]))
            flat_width = W * raw_width
            flat_bucket = W * bucket

            def _pad_flat(name, fill):
                value = fwd.get(name)
                if (not torch.is_tensor(value) or value.dim() < 2
                        or int(value.shape[0]) != B
                        or int(value.shape[1]) != flat_width):
                    return
                view = value.reshape(B, W, raw_width, *value.shape[2:])
                fill_shape = (B, W, pad_width, *value.shape[2:])
                pad = torch.full(
                    fill_shape, fill, dtype=value.dtype, device=value.device)
                fwd[name] = torch.cat((view, pad), dim=2).reshape(
                    B, flat_bucket, *value.shape[2:])

            # ``indices`` is an id tensor and must remain in range; all
            # structural values use their existing masked/padding sentinel.
            _pad_flat("indices", 0)
            _pad_flat("seed_event", 0.0)
            _pad_flat("word_groups", -1)
            _pad_flat("part_spans", 0)
            _pad_flat("percept_where", -1)
            fwd["word_part_indices"] = padded_ids
            fwd["word_part_mask"] = padded_mask
            fwd["word_part_capacity"] = capacity
        return capacity

    def enable_compiled_step(self):
        """Compile the per-batch forward and route runBatch through it.

        O1 fix: ``ModelFactory.run`` used to ``compile(m)`` the whole
        module and then call ``m.run()``, which delegates to the eager
        ``_orig_mod`` -- so the compiled callable was never invoked
        (dynamo traced 0 frames; every "compiled" run was eager).
        Instead we ``torch.compile`` the ``forward`` *callable* and
        invoke that from ``runBatch``; the eager run/runEpoch
        orchestration stays Python (streaming/staging outside the
        compiled region, per the compile-scoped-to-model design).

        Strict gate where the configured forward is host-island-free: the
        recon-then-eliminate program drove that path to **0 graph breaks**
        (``total graph breaks: 0; graphs: 1``). Configurations with an explicit
        eager island relax ``fullgraph`` so the dense regions can still be
        compiled. The canonical aligned serial path remains strict: its sparse
        concept rows are gathered once before the graph, not inside it.
        """
        from util import compile as _compile
        # MPS compiles by default (2026-06-07): torch 2.12's inductor MPS
        # backend traces the per-batch forward fullgraph just like CPU/CUDA
        # (verified ``Model compiled (inductor, ..., fullgraph=True)`` on
        # ``data/MM_20M_legacy.xml``). The old ``BASICMODEL_MPS_COMPILE`` opt-in gate
        # (eager fallback on MPS, from when torch's MPS fake-tensor device
        # propagation was incomplete) is retired -- use ``MODEL_COMPILE=none``
        # (skip) or ``=eager`` (no inductor) to disable/relax compilation on
        # any device.
        # D8 capture-gate (2026-05-19): pre-warm any LAZY-built caches
        # that the captured forward depends on, so Dynamo never traces
        # their build path. ``_stm_reducer`` resolves the already-wired
        # LanguageLayer child and caches that non-owning reference here,
        # eagerly, before the compile wrapper closes over ``self.forward``.
        # Build every lazy, checkpointed shape before capture. Autoload uses
        # the same helper before loading an existing training checkpoint.
        self._prewarm_checkpoint_shapes()
        # Device-coherence sweep before the trace, against the CANONICAL
        # process device (util.TheDevice -- the single source of truth that
        # ``init_device`` keeps in sync with torch's default device). The lazy
        # reducer build (above) and any module minted during ModelFactory's
        # build-on-CPU phase (``init_device("cpu")`` -> build ->
        # ``init_device(target)`` + ``m.to(target)``) can be left on CPU if it
        # was created after the ``m.to``; the
        # compiled forward's fake-tensor propagation then fails loud
        # ("two different devices mps:0, cpu") where eager might limp. The
        # sweep is an idempotent no-op when already coherent. The per-call
        # re-home in ``_stm_reducer`` stays host-only (a ``torch.device``
        # compare is not traceable).
        self.to(str(TheDevice.get()))
        # The strict gate holds where the forward traces end to end. A
        # FULL-ROUTER grammar (anything beyond the default-only
        # pi/sigma rules) routes through the host-side chart fires
        # (``_chart_compose_at_C`` / ``_chart_generate_from_stm``,
        # ``@torch.compiler.disable``'d: their rule-id bookkeeping is
        # data-dependent ``.item()`` branching dynamo cannot guard on —
        # the ``Eq(u0, 2)`` UserError on MM_xor's per-stage path). A
        # disabled call is itself a graph break, which fullgraph=True
        # turns into a hard error — so full-router configs compile
        # with tolerated breaks. Default-only configs (MM_20M's
        # bypass never reaches the router loops) keep the strict gate.
        _ws = getattr(self, 'symbolSpace', None)
        _full_router = (_ws is not None and not getattr(
            _ws, '_grammar_is_default_only', True))
        # Slice B: the symbol tower's ``forward_symbol`` is a
        # ``@torch.compiler.disable`` eager island (host-side order-raising
        # readout -- data-dependent dict iteration + SubSpace construction). Like
        # full-router grammar, its disabled call is itself a graph break, so it
        # needs fullgraph relaxed too. ``_host_islands`` gates both.
        _symbol_tower = bool(getattr(self, 'symbol_tower', False))
        # SERIAL full-router keeps the strict gate (2026-07-07): the per-word
        # path never dispatches the host-side chart fires (the islands above
        # are the PER-STAGE parallel path's), and with the mid-sentence
        # reground removed the serial forward traces fullgraph-clean --
        # verified 0 dynamo breaks / 1 graph on MM_20M_grammar with the
        # sentence protocol ON. The MPS eager-skip below intentionally keeps
        # the BROAD condition: inductor-MPS still miscompiles this forward
        # (NaN/Inf caught by insert_meta; async command-buffer errors) even
        # with the uint16 + 31-buffer workarounds in util.compile.
        _serial_per_word = bool(getattr(
            getattr(self, 'inputSpace', None), '_per_word_enabled', False))
        _sparse_codebook_island = any(
            bool(getattr(
                getattr(_cs, "similarity_codebook", None),
                "sparse_lookup_grad", False))
            for _cs in (getattr(self, "conceptualSpaces", None) or ()))
        _staged_sparse_bank = self._aligned_serial_word_mode()
        _canonical_word_chunk = bool(
            self._aligned_serial_sparse_bank_mode()
            and getattr(_util, "TheCompileBackend", "none") != "none")
        _host_islands = ((_full_router and not _serial_per_word)
                         or _symbol_tower
                         or (_sparse_codebook_island
                             and not _staged_sparse_bank))
        # torch 2.12 inductor-MPS cannot codegen the FULL-ROUTER
        # forward: the generated Metal references undeclared reduction
        # indices ("use of undeclared identifier 'r0_1'" — an upstream
        # backend bug, reproduced on MM_xor 2026-06-11; MM_20M's
        # default-only forward compiles clean on the same device).
        # Until the upstream fix lands those configs run eager on MPS;
        # CUDA / CPU and default-only MPS configs compile as before.
        # MODEL_COMPILE=none stays the manual override everywhere.
        # MPS eager-skip RETIRED (2026-07-07): the "inductor-MPS miscompiles
        # this forward (NaN)" premise was a test-harness artifact (recon_bench
        # _build_model device incoherence), not a compile bug -- the real
        # ModelFactory path compiles fullgraph=True on MPS and runs clean at
        # ~2.6-6x eager once the recompile churn was fixed (symbol-codebook
        # preallocation, Optimizer device-restore spelling, grad-off
        # diagnostics routed eager, prelude-counter/pre-grow warms). The
        # ``MODEL_COMPILE`` policy governs as everywhere else (DEBUG =
        # none/eager to skip the one-time ~6min MPS compile; PRODUCTION =
        # inductor past the amortization break-even). util.compile's MPS
        # block carries the required codegen workarounds (uint16->ushort,
        # max_fusion_unique_io_buffers, assert disables).
        _fullgraph = ENUM_FULLGRAPH and not _host_islands
        if ENUM_FULLGRAPH and _host_islands:
            TheMessage(
                "fullgraph relaxed: host-island config (full-router grammar, "
                "symbolTower forward_symbol, or sparse indexed codebook is "
                "a tolerated graph break)")
        _word_buckets = tuple(
            int(v) for v in getattr(self, "serial_word_buckets", ()) or ())
        self._compiled_word_chunk_step = None
        self._compiled_word_chunk_active = False
        self._compiled_word_chunk_replaying = False
        self._compiled_word_chunk_width = 0
        self._compiled_part_fold_ladder = None
        self._compiled_whole_fold_ladder = None
        self._compiled_word_loop_compile = None
        self._compiled_word_loop_fullgraph = False
        _mps_fullgraph_ladder_boundary = (
            _canonical_word_chunk
            and str(TheDevice.get()).startswith("mps"))
        _mps_word_loop = os.environ.get(
            "BASICMODEL_MPS_WORD_LOOP_FULLGRAPH", "1").strip().lower() not in (
                "0", "false", "no", "off")
        _mps_word_cell = os.environ.get(
            "BASICMODEL_MPS_WORD_FULLGRAPH", "1").strip().lower() not in (
                "0", "false", "no", "off")
        if _mps_fullgraph_ladder_boundary and _mps_word_loop:
            # The primary production boundary: one fullgraph encompasses the
            # complete static W loop, including recurrence and backward. P is
            # a fixed masked layout inside that loop, so it is compiled once
            # for W=16/32/64/128 rather than once per radix constituent width.
            # It is compiled lazily when the first input chooses a W bucket,
            # rather than spending startup time lowering every bucket. A
            # complete W=16 backward has substantially more live inputs than
            # the old one-word cell. These conservative realization/fusion
            # limits keep every generated Metal kernel beneath its 31
            # constant-buffer cap.
            os.environ.setdefault("BASICMODEL_MPS_IOBUF", "8")
            os.environ.setdefault("BASICMODEL_MPS_FUSE", "16")
            os.environ.setdefault("BASICMODEL_MPS_REALIZE", "4")
            self._configure_lazy_fullgraph_word_loops(_compile, _word_buckets)
            self._compiled_step = self.forward
            TheMessage(
                "Canonical MPS fullgraph compile: lazy complete W-loop "
                "capture with one fixed residual-part axis")
        elif _mps_fullgraph_ladder_boundary and _mps_word_cell:
            # Fullgraph training kernel for the actual recurrent word tick.
            # The fixed residual-part bucket below removes the old 3..8192
            # symbolic P range, which was the dominant source of the earlier
            # multi-minute compile. K=1 is the safe default; K=2 is a
            # fullgraph packing option for MPS benchmarking once the common
            # residual buckets are warm.
            _mps_chunk_width = int(os.environ.get(
                "BASICMODEL_MPS_WORD_CHUNK", "1"))
            if _mps_chunk_width not in (1, 2):
                raise ValueError(
                    "BASICMODEL_MPS_WORD_CHUNK must be 1 or 2; got "
                    f"{_mps_chunk_width}")
            os.environ.setdefault("BASICMODEL_MPS_IOBUF", "12")
            os.environ.setdefault("BASICMODEL_MPS_FUSE", "32")
            self._compiled_word_steps = {}
            self._compiled_word_chunk_step = _compile(
                (self._aligned_word_chunk1 if _mps_chunk_width == 1
                 else self._aligned_word_chunk2),
                verbose=True, fullgraph=True)
            self._compiled_word_chunk_active = True
            self._compiled_word_chunk_width = _mps_chunk_width
            self._compiled_step = self.forward
            TheMessage(
                "Canonical MPS fullgraph compile: reusable K="
                f"{_mps_chunk_width} word cell with fixed residual-part "
                "buckets")
        elif _mps_fullgraph_ladder_boundary:
            # Two fullgraph numerical ladders replace the previous six tiny
            # fold calls per word. Capturing the whole K=1/K=2 word cell also
            # captures carrier mutation, sparse lookup, STM, and grammar
            # state; MPS spends minutes lowering that AOT backward before the
            # first batch. The ladders retain the actual learned PS/WS work in
            # fullgraph=True graphs and leave only those stateful boundaries
            # eager.
            self._compiled_word_steps = {}
            self._enable_mps_fullgraph_fold_ladders(_compile)
            self._compiled_step = self.forward
            TheMessage(
                "Canonical MPS compile: PS and WS fold ladders "
                "fullgraph=True with eager state adapter")
        elif _canonical_word_chunk:
            # Compile one reusable, fixed K=2 word cell.  The sentence shell
            # remains eager: it stages two-column views, replays this callable,
            # and installs the returned contributions in sentence order.  In
            # particular this avoids asking Inductor to lower four W-unrolled
            # graphs whose compile time scales with 16/32/64/128 words.
            self._compiled_word_steps = {}
            self._compiled_word_chunk_step = _compile(
                self._aligned_word_chunk2, verbose=True, fullgraph=True)
            self._compiled_word_chunk_active = True
            # A non-None sentinel makes runBatch perform the ordinary eager
            # lexical/sparse-bank staging.  The selected outer callable is the
            # bound eager forward; only the two-word cell above is compiled.
            self._compiled_step = self.forward
            TheMessage(
                "Canonical aligned serial compile: reusable K=2 word cell "
                "with eager sentence adapter")
        elif (_serial_per_word and len(_word_buckets) > 1):
            # Four DISTINCT CODE OBJECTS give Dynamo/Inductor four stable,
            # packed static loop graphs. A closure factory is insufficient:
            # its products share ``__code__``, so Dynamo treats W=32 as a
            # recompile of W=16 even if their function identities/names differ.
            self._compiled_word_steps = {}

            def _forward_W16(input_data):
                slab = self.inputSpace._ar_embedded_N
                if slab is not None and int(slab.shape[1]) != 16:
                    raise RuntimeError(
                        f"compiled W=16 bucket received a W="
                        f"{int(slab.shape[1])} staged slab")
                return self.forward(input_data)

            def _forward_W32(input_data):
                slab = self.inputSpace._ar_embedded_N
                if slab is not None and int(slab.shape[1]) != 32:
                    raise RuntimeError(
                        f"compiled W=32 bucket received a W="
                        f"{int(slab.shape[1])} staged slab")
                return self.forward(input_data)

            def _forward_W64(input_data):
                slab = self.inputSpace._ar_embedded_N
                if slab is not None and int(slab.shape[1]) != 64:
                    raise RuntimeError(
                        f"compiled W=64 bucket received a W="
                        f"{int(slab.shape[1])} staged slab")
                return self.forward(input_data)

            def _forward_W128(input_data):
                slab = self.inputSpace._ar_embedded_N
                if slab is not None and int(slab.shape[1]) != 128:
                    raise RuntimeError(
                        f"compiled W=128 bucket received a W="
                        f"{int(slab.shape[1])} staged slab")
                return self.forward(input_data)

            _fixed_sources = {
                16: _forward_W16,
                32: _forward_W32,
                64: _forward_W64,
                128: _forward_W128,
            }
            unsupported = [
                width for width in _word_buckets
                if width not in _fixed_sources]
            if unsupported:
                raise ValueError(
                    "compiled serialWordBuckets supports exactly widths "
                    f"16,32,64,128; got unsupported {unsupported}")
            object.__setattr__(
                self, "_compiled_word_step_sources", _fixed_sources)
            for _width in _word_buckets:
                self._compiled_word_steps[int(_width)] = _compile(
                    _fixed_sources[int(_width)], verbose=True,
                    fullgraph=_fullgraph)
            self._compiled_step = self._compiled_word_steps[
                max(_word_buckets)]
        else:
            self._compiled_word_steps = {}
            self._compiled_step = _compile(
                self.forward, verbose=True, fullgraph=_fullgraph)

    def _start_spaces_for_forward(self):
        for space in self.spaces:
            if hasattr(space, 'Start'):
                space.Start()
        self._spaces_started_for_forward = True

    def forward(self, inputData):
        """IR-only forward: stem -> body -> head.

        Dispatches to ``_forward_per_stage`` (the single per-stage
        forward path).  Within-sentence training is BERT-style masked-
        LM at the P-space_role; sentence-level AR is delegated to
        ``InterSentenceLayer`` (ARMA(p, q) over sentence reps).
        """
        external_start = bool(getattr(self, '_spaces_started_for_forward',
                                      False))
        if not external_start:
            self._start_spaces_for_forward()
        try:
            return self._forward_per_stage(inputData)
        finally:
            self._spaces_started_for_forward = False

    def runTrial(self, numEpochs=1, batchSize=10, lr=0.01, profile=None):
        """Main training loop: train for numEpochs, evaluate on test set each epoch.

        Alpha (exploration temperature) anneals from 1.0 (full exploration)
        to 0.0 (full exploitation) over the first 5% of training.  This is
        propagated to all Spaces and their layers/bases via set_sigma().

        A single persistent optimizer is used across all epochs so Adam's
        momentum and variance estimates accumulate properly.

        ``BASIC_RUN_TEST`` env (set by ``train.py --test [N]``) controls
        the baseline + post-train test/validation passes:

        * unset: skip both passes (default; avoids the long test-split
          traversal that can dominate wall-clock at large maxDocs).
        * empty string: run both passes uncapped.
        * integer ``N``: run both passes capped at ``N`` batches each.

        Returns a list of per-epoch test accuracies.
        """
        max_seconds_raw = os.environ.get("BASIC_MAX_SECONDS", "").strip()
        self._training_deadline_reached = False
        self._training_deadline_monotonic = None
        if max_seconds_raw:
            try:
                max_seconds = float(max_seconds_raw)
            except ValueError as exc:
                raise ValueError(
                    "BASIC_MAX_SECONDS must be a number of seconds; got "
                    f"{max_seconds_raw!r}") from exc
            if max_seconds > 0.0:
                self._training_deadline_monotonic = (
                    time.monotonic() + max_seconds)
                TheMessage(
                    f"Training wall-clock cap armed: {max_seconds:g}s "
                    "(checked between completed batches).")

        trainLosses       = [[],[]]  # [output_losses, reconstruction_losses]
        validationLosses  = [[],[]]
        testLosses        = [[],[]]
        self.plot         = False
        accuracy          = []
        self._optimizer   = self.getOptimizer(lr=lr)

        # Test gating from BASIC_RUN_TEST. Tri-state: None (skip),
        # 0 (uncapped run), positive int (cap each pass at that many
        # batches). The empty-string env value maps to uncapped.
        _test_env = os.environ.get("BASIC_RUN_TEST")
        if _test_env is None:
            _test_max_batches = None  # signal: skip the test passes
            _run_test = False
        else:
            try:
                _n = int(_test_env) if _test_env != "" else 0
            except ValueError:
                _n = 0
            _run_test = True
            _test_max_batches = _n if _n > 0 else None

        # Enable sigma-driven self-annealing for ergodic layers
        self.set_sigma(0.5)

        # Baseline evaluation before any training (gated on --test).
        if _run_test:
            self.set_sigma(0)
            outErr, inErr, allOut, lastIn = self.runEpoch(
                batchSize=batchSize, split="test",
                max_batches=_test_max_batches)
            outErr = outErr.item() if torch.is_tensor(outErr) else outErr
            inErr = inErr.item() if torch.is_tensor(inErr) else inErr
            self.set_sigma(0.5)
            testLosses[0].append(outErr)
            testLosses[1].append(inErr)
            TheMessage(f"Baseline Test Loss: output={outErr:.4f}, reconstruction={inErr:.4f}")
        else:
            TheMessage(
                "Baseline test pass skipped (--test not specified). "
                "Re-invoke train.py with --test to include baseline + "
                "post-epoch evaluation.")

        for epoch in range(numEpochs):
            TheMessage(f"Epoch [{epoch + 1}/{numEpochs}]")

            outErr, inErr, allOut, lastIn = self.runEpoch(optimizer=self._optimizer, batchSize=batchSize, split="train")
            outErr = outErr.item() if torch.is_tensor(outErr) else outErr
            inErr = inErr.item() if torch.is_tensor(inErr) else inErr
            trainLosses[0].append(outErr)
            trainLosses[1].append(inErr)
            TheMessage(f"Train Loss: output={outErr:.4f}, reconstruction={inErr:.4f}")

            if getattr(self, "_training_deadline_reached", False):
                # Do not start another epoch or an optional test traversal:
                # return through the normal runTrial tail so ModelFactory's
                # final autosave still records the last completed update.
                accuracy += [0.0]
                if len(accuracy) < int(numEpochs):
                    accuracy.extend([0.0] * (int(numEpochs) - len(accuracy)))
                TheMessage(
                    "Training wall-clock cap reached; finishing normally "
                    "for final reporting/checkpoint save.")
                break

            if not _run_test:
                # Per-epoch test pass skipped under --test gating.
                # Record sentinel zeros so the epoch's accuracy slot is
                # well-formed; the post-trial summary still reports the
                # train-side losses.
                accuracy += [0.0]
                continue

            self.set_sigma(0)  # suppress exploration during eval
            outErr, inErr, allOut, lastIn = self.runEpoch(
                batchSize=batchSize, split="test",
                max_batches=_test_max_batches)
            outErr = outErr.item() if torch.is_tensor(outErr) else outErr
            inErr = inErr.item() if torch.is_tensor(inErr) else inErr
            self.set_sigma(0.5)  # re-enable for next training epoch
            testLosses[0].append(outErr)
            testLosses[1].append(inErr)

            if not getattr(self.inputSpace.data,
                           "has_supervised_outputs", True):
                accuracy += [0.0]
                TheMessage(
                    f"Test Loss: reconstruction={inErr:.4f} "
                    "(self-supervised corpus; output accuracy disabled)")
            elif not isinstance(allOut, torch.Tensor) or allOut.numel() == 0:
                # No output predictions (empty dataset or no batches)
                accuracy += [0.0]
                TheMessage(f"Test Loss: output={outErr:.4f}, reconstruction={inErr:.4f} (no predictions)")
            elif allOut.dim() == 1:
                predicted = (allOut > 0.5).long()
                actual = (self.outputSpace.getTestOutput().squeeze() > 0.5).long()
                # test_output now lives on CPU (list-of-tensors kept off the
                # accelerator so DataLoader workers can pickle slices); align
                # to the model's device for comparison.
                actual = actual.to(predicted.device)
                total   = predicted.size(0)
                correct = (predicted == actual).sum().item()
                accuracy += [correct / total]
                TheMessage(f"Test Accuracy: {100 * correct / total:.2f}%")
            else:
                _, predicted = torch.max(allOut, 1)
                _, actual = torch.max(self.outputSpace.getTestOutput(), 1)
                actual = actual.to(predicted.device)
                total   = predicted.size(0)
                correct = (predicted == actual).sum().item()
                accuracy += [correct / total]
                TheMessage(f"Test Accuracy: {100 * correct / total:.2f}%")

            if profile:
                profile.step()

        TheMessage(f"Final Stats:")
        TheReport.plotLoss(self.name, trainLosses, validationLosses, testLosses)
        self.rCorrect = TheReport.mnistReport(self)

        # Post-training: dump the chart's Viterbi-extracted grammar per
        # test row so the user can read the discrete derivation the
        # router committed to. Useful for grammar-from-chart inspection
        # (XOR_grammar.xml et al.).
        try:
            ss = self.symbolSpace
            router = getattr(ss, 'languageLayer', None) if ss is not None else None
            if router is not None:
                rules = ss.current_rules
                gen_rules = ss.generate_rules
                from Language import TheGrammar
                def _decode(rule_id):
                    rid = int(rule_id)
                    if 0 <= rid < len(TheGrammar.rules):
                        rd = TheGrammar.rules[rid]
                        return f"{rid}:{rd.canonical}"
                    return f"{rid}:?"
                TheMessage("=== Signal-router-extracted grammar (Viterbi) ===")
                def _decode_row(row):
                    head, omitted, tail = _grammar_row_preview(row)
                    decoded = [_decode(rid) for rid in head]
                    if omitted:
                        decoded.append(f"... {omitted} rules omitted ...")
                        decoded.extend(_decode(rid) for rid in tail)
                    return decoded
                max_report_rows = int(os.environ.get(
                    "BASIC_GRAMMAR_REPORT_MAX_ROWS", "4"))
                for space_role, rows in (rules or {}).items():
                    TheMessage(f"  compose space_role={space_role!r}:")
                    report_rows = _grammar_rows_for_report(rows)
                    shown_rows = (report_rows if max_report_rows <= 0
                                  else report_rows[:max_report_rows])
                    for b, row in enumerate(shown_rows):
                        decoded = _decode_row(row)
                        TheMessage(f"    row[{b}] = {decoded}")
                    if len(shown_rows) < len(report_rows):
                        TheMessage(
                            f"    ... {len(report_rows) - len(shown_rows)} "
                            "batch rows omitted ...")
                for space_role, rows in (gen_rules or {}).items():
                    TheMessage(f"  generate space_role={space_role!r}:")
                    report_rows = _grammar_rows_for_report(rows)
                    shown_rows = (report_rows if max_report_rows <= 0
                                  else report_rows[:max_report_rows])
                    for b, row in enumerate(shown_rows):
                        decoded = _decode_row(row)
                        TheMessage(f"    row[{b}] = {decoded}")
                    if len(shown_rows) < len(report_rows):
                        TheMessage(
                            f"    ... {len(report_rows) - len(shown_rows)} "
                            "batch rows omitted ...")
                if getattr(router, '_last_root_state', None) is not None:
                    rs = router._last_root_state.detach()
                    TheMessage(f"  S root state shape = {tuple(rs.shape)}")
                    n_root_rows = int(rs.shape[0])
                    shown_root_rows = (n_root_rows if max_report_rows <= 0
                                       else min(n_root_rows, max_report_rows))
                    root_dim_limit = int(os.environ.get(
                        "BASIC_GRAMMAR_REPORT_MAX_DIMS", "32"))
                    for b in range(shown_root_rows):
                        vec = rs[b, 0].tolist()
                        if root_dim_limit > 0:
                            head, omitted, tail = _grammar_row_preview(
                                vec, limit=root_dim_limit)
                        else:
                            head, omitted, tail = vec, 0, []
                        preview = [round(v, 4) for v in head]
                        if omitted:
                            preview.append(f"... {omitted} dims omitted ...")
                            preview.extend(round(v, 4) for v in tail)
                        TheMessage(f"    row[{b}] root = {preview}")
                    if shown_root_rows < n_root_rows:
                        TheMessage(
                            f"    ... {n_root_rows - shown_root_rows} root "
                            "rows omitted ...")
        except Exception as _exc:
            import logging
            logging.getLogger(__name__).warning(
                "post-training grammar dump failed: %s", _exc)

        # Reconstruction report: run final test pass and show input vs reconstructed
        if self.reversible and self.inputSpace.model_type == "embedding":
            self._reconstructionReport()

        self.trainLosses = trainLosses
        self.testLosses  = testLosses
        return accuracy
    
    BatchResult = namedtuple('BatchResult', [
        'outputPred', 'symbols', 'lossOut', 'lossIn', 'inputPred', 'forwardInput',
    ])

    def trainEmbeddings(self, trainMod, index, split):
        """Run one SBOW training step on the sample at (index, split).

        ``trainMod`` is the iterable of training-embedding modes that
        should fire (e.g. {'SBOW'}); the call is a no-op when the
        configured ``train_embedding`` isn't in that set or when the
        lexer is byte-mode (perceptual SBOW replaces InputSpace SBOW).
        Returns the per-step loss or None.
        """
        sbow = None
        te = getattr(self, 'train_embedding', 'NONE')
        if te in trainMod:
            # Skip InputSpace SBOW/CBOW when lexer=byte -- perceptual SBOW
            # replaces it (see perceptual_sbow_loss).
            if getattr(self, 'lexer', None) in ('byte', 'bytes'):
                return None
            emb = self.perceptualSpace.vocabulary
            if isinstance(emb, Embedding):
                sentences = self._get_sentences(split)
                if sentences and index < len(sentences):
                    sentence = sentences[index]
                    words = [t for t, _ in parse(sentence, lex='words')]
                    if te in ('JOINT'):
                        sbow = self.perceptualSpace.sbow_loss(words)
                    elif te in ('CBOW', 'SBOW', 'BOTH'):
                        # CBOW uses padded context; SBOW and BOTH use the faster centroid method
                        method = 'CBOW' if te == 'CBOW' else 'SBOW'
                        self.perceptualSpace.train_embeddings(words, method=method)
        return sbow

    def perceptual_sbow_loss(self):
        """SBOW loss over the percepts USED in the forward, WITH negative
        sampling (the pode/antipode repulsion).

        Each used percept is trained toward the leave-one-out centroid of the
        OTHER used percepts in its row, while K random codebook rows are
        repelled -- ``PretrainModel.sbow_loss_indices`` -> the canonical
        ``_neg_sampling_loss`` in embed.py. Two things make this NOT collapse,
        where the prior centroid-only cosine did (every row decoded to the
        null percept): (1) the positives are exactly the gathered percept ids,
        excluding the null/padding slots AND the untrained reserve -- we train
        ONLY the vectors used in the computation; (2) the negative samples
        supply the antipode repulsion that balances the centroid attractor.

        Returns a scalar loss tensor, or None when the percept-index path is
        unavailable or every row holds < 2 used percepts.
        """
        ps_space = getattr(self, "perceptualSpace", None)
        sub = getattr(ps_space, "subspace", None) if ps_space is not None else None
        cb = getattr(sub, "what", None) if sub is not None else None
        pretrain = getattr(cb, "pretrain", None) if cb is not None else None
        fwd = getattr(ps_space, "_forward_input", None) if ps_space is not None else None
        if (pretrain is None
                or not hasattr(pretrain, "sbow_loss_indices")
                or fwd is None or "indices" not in fwd):
            return None
        pid_grid = fwd["indices"]
        if not torch.is_tensor(pid_grid) or pid_grid.dim() != 2:
            return None
        # Resolve the null/padding percept id so its slots are excluded.
        null_pid = None
        ps_store = fwd.get("percept_store", None)
        if ps_store is not None:
            try:
                null_pid = ps_store.get_id(b"\x00")
            except Exception:
                null_pid = None
        null_pid = int(null_pid) if null_pid is not None else None
        terms = []
        for row in pid_grid.detach().cpu().tolist():
            used = [int(p) for p in row
                    if null_pid is None or int(p) != null_pid]
            if len(used) < 1:
                continue
            t = pretrain.sbow_loss_indices(used)
            if t is not None:
                terms.append(t)
        if not terms:
            return None
        return torch.stack(terms).mean()

    def conceptual_sbow_loss(self):
        """Rotation SBOW (CBOW-NS) over the per-concept similarity codebook.

        Snaps each parked all-siblings concept-slab position to its nearest
        similarity-codebook row by cosine (no grad -- self-supplied concept
        identity), gathers those rows as the SBOW window, and situates them via
        ``embed.conceptual_sbow_loss_codes``. The gather is grad-bearing, so the
        rotation gradient (tangential, radius-preserving) trains the codebook
        rows as the per-concept located codes. Returns a scalar tensor, or
        ``None`` when the codebook / parked slab is unavailable or serial.

        NOTE (2026-06-24, HISTORICAL for the sparse path): the ConceptualSpace
        has NO forward codebook -- its ``.what`` is a *computed* Tensor from
        the percept binding -- so situating the codebook alone could not
        differentiate the conceptual representation. RESOLVED 2026-07-02 (plan
        C1) for sparse-active configs: the parked slab is the LIVE composed
        code (grad-bearing through the SparseLayer families), and the SBOW
        situates the CODES against the codebook pool -- the window is the
        slab, not the snapped rows -- so co-location trains the
        percept->concept binding upstream. The no-grad snap-gather below
        remains the sparse-INACTIVE legacy path.
        """
        cs = getattr(self, "conceptualSpace", None)
        cb = getattr(cs, "similarity_codebook", None) if cs is not None else None
        slab = getattr(self, "_cs_parallel_slab", None)
        cs_src = getattr(self, "_cs_parallel_slab_cs", None)
        object.__setattr__(self, "_cs_parallel_slab", None)  # consume-once
        object.__setattr__(self, "_cs_parallel_slab_cs", None)
        if cs_src is not None:
            # Live sparse path: pool from the dictionary that composed the slab
            # (the parked stage CS), not the terminal codebook.
            cb = getattr(cs_src, "similarity_codebook", None) or cb
        if cb is None or slab is None or self.serial:
            return None
        if not torch.is_tensor(slab) or slab.dim() != 3 or slab.shape[1] < 2:
            return None
        rows = cb.prototype()
        if rows is None or not torch.is_tensor(rows) or rows.shape[0] < 1:
            return None
        eps = 1e-8
        D = min(int(slab.shape[-1]), int(rows.shape[-1]))
        slab = slab[..., :D]
        rows = rows[..., :D]
        from embed import conceptual_sbow_loss_codes
        if slab.requires_grad:
            # Sparse-active live codes (plan C1): the window IS the composed
            # code slab; situating it differentiates the sparse binding.
            return conceptual_sbow_loss_codes(slab, pool=rows, scale=1.0)
        with torch.no_grad():
            s = slab / slab.norm(dim=-1, keepdim=True).clamp_min(eps)
            r = rows / rows.norm(dim=-1, keepdim=True).clamp_min(eps)
            assign = torch.einsum('bnd,vd->bnv', s, r).argmax(dim=-1)   # [B, N]
        window = rows[assign]                                           # [B, N, D]
        return conceptual_sbow_loss_codes(window, pool=rows, scale=1.0)

    def _reconstruction_seed(self):
        """The terminal C-space event seeding the reconstruction reverse.

        PARALLEL mode prefers the LIVE stage carrier (``_combine_last_cs_sub``)
        over the detached STM snapshot: the reverse chain is
        input-differentiable, so a live seed lets ``lossRev`` train the
        forward representation to be reversible. With the detached snapshot
        the loss was a CONSTANT -- no gradient anywhere -- the 2026-07-02
        MM-20M "recon does not converge" root cause. The STM snapshot itself
        stays detached (memory, not a gradient carrier); SERIAL keeps the
        snapshot's reduced-single-idea convention. ``None`` when no usable
        seed exists."""
        serial = bool(getattr(self, 'serial', False))
        if not serial:
            live = getattr(self, "_combine_last_cs_sub", None)
            ev = (live.materialize()
                  if live is not None and hasattr(live, "materialize")
                  else None)
            if (ev is not None and torch.is_tensor(ev) and ev.dim() == 3
                    and ev.shape[1] >= 1):
                return ev
        stm = (self.conceptualSpace.stm
               if self.conceptualSpace is not None else None)
        snap = stm.snapshot() if stm is not None else None
        if (snap is not None and torch.is_tensor(snap)
                and snap.dim() == 3 and snap.shape[1] >= 1):
            return snap if not serial else snap[:, :1, :]
        return None

    def _maybe_compile_brick(self):
        """One-shot Inductor / Dynamo / TF32 setup on first ``runBatch``.

        The actual ``torch.compile`` call lives in ``util.compile`` (the
        canonical compile site, called once at model build in
        ``ModelFactory.run``). This method just configures Dynamo /
        Inductor knobs that are useful regardless of which mode the
        canonical site picked, and enables TF32 for fp32 matmuls.
        Idempotent: only runs on the first invocation.
        """
        if getattr(self, '_brick_compile_attempted', False):
            return
        self._brick_compile_attempted = True
        if not torch.cuda.is_available():
            return
        try:
            dev_type = TheDevice.get().type
        except Exception:
            dev_type = None
        if dev_type != "cuda":
            return
        try:
            import torch._dynamo as _dynamo
            # Bump the cache size limit so warm-up across dtype/shape
            # variants in the cursor path doesn't recompile.
            _dynamo.config.cache_size_limit = max(
                64, int(getattr(_dynamo.config, 'cache_size_limit', 0) or 0))
            # Allow Dynamo to capture ``int(tensor.item())`` patterns
            # rather than break the graph at every site.
            _dynamo.config.capture_scalar_outputs = True
            # Enable TF32 on Ampere+ / Blackwell tensor cores for fp32
            # matmuls -- free perf win on TF32-capable CUDA hardware.
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
            # Raise the CUDAGraph "distinct sizes" warning limit. The
            # default (8) is conservative for models with N-halving
            # stages, where shapes across stages form an ``log2(N)``
            # sequence (each is static per stage). Set to 128 so the
            # warning only fires on genuinely-pathological shape variance.
            try:
                torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = 128
            except Exception:
                pass
            # Skip CUDAGraph capture for graph segments whose Inductor
            # specialization count crosses the dynamic-shape threshold.
            # Keeps Inductor fusion / Triton codegen wins while
            # sidestepping the per-shape CUDAGraph capture cost on
            # architectures with many distinct static shapes.
            try:
                torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
            except Exception:
                pass
        except Exception as e:
            print(
                f"[BasicModel] §7 dynamo / inductor setup failed: "
                f"{type(e).__name__}: {e}. Continuing without these "
                f"hints; util.compile's torch.compile is unaffected.")

    def _invalidate_compiled_step_for_partspace_growth(self):
        """Drop callables that captured the pre-growth PS Parameter.

        Recompilation is deferred until the next batch entry, keeping the
        completed batch boundary free of a multi-minute compile while ensuring
        no old callable is ever replayed against the replacement Parameter.
        """
        had_compiled = bool(
            getattr(self, "_compiled_step", None) is not None
            or getattr(self, "_compiled_word_steps", None))
        self._compiled_step = None
        self._compiled_word_steps = {}
        self._active_compiled_step = None
        # Canonical aligned serial mode can compile one reusable two-word cell
        # instead of asking Inductor to lower a W=16/32/64/128 unrolled
        # sentence graph.  The outer sentence shell remains eager and stages
        # fixed two-column views before each replay.
        self._compiled_word_chunk_step = None
        self._compiled_word_chunk_active = False
        self._compiled_word_chunk_replaying = False
        self._compiled_word_chunk_width = 0
        self._compiled_part_fold_ladder = None
        self._compiled_whole_fold_ladder = None
        self._compiled_word_loop_compile = None
        self._compiled_word_loop_fullgraph = False
        object.__setattr__(self, "_compiled_word_step_sources", {})
        self._compiled_step_needs_rebuild = had_compiled

    def _flush_partspace_promotions(self, optimizer=None):
        """Commit queued radix promotions at the explicit batch boundary."""
        part_space = getattr(self, "perceptualSpace", None)
        flush = getattr(part_space, "flush_pending_promotions", None)
        if not callable(flush):
            return None
        result = flush(optimizer=optimizer)
        if result.get("grew", False):
            self._invalidate_compiled_step_for_partspace_growth()
            TheMessage(
                f"[{self.name}] PartSpace radix codebook grew "
                f"{result['old_capacity']} -> {result['new_capacity']} rows "
                f"at the batch boundary; installed {result['inserted']} "
                "queued promotion(s) and invalidated compiled callables")
        return result

    def _on_partspace_eager_growth(self, result):
        """Handle mandatory byte-row growth during the eager lexical stem."""
        if not result.get("grew", False):
            return
        self._invalidate_compiled_step_for_partspace_growth()
        TheMessage(
            f"[{self.name}] PartSpace radix codebook grew "
            f"{result['old_capacity']} -> {result['new_capacity']} rows "
            f"at the eager lexical boundary; installed "
            f"{result['inserted']} mandatory byte percept(s) and "
            "invalidated compiled callables")

    def _normalize_perceptual_embedding(self):
        """Re-project only the legacy lexicon Embedding after a train step.

        The post-step projection was introduced for ``Embedding`` rows.  A
        radix PartSpace owns a ``Codebook`` in the same ``subspace.what``
        position; calling generic ``Basis.normalize()`` there reads/transforms
        every physical percept row and then rewrites the table.  Radix reads
        are already bounded by the percept UNORM STE and must not take this
        lexicon-only full-capacity path.
        """
        emb = getattr(
            getattr(self.perceptualSpace, "subspace", None), "what", None)
        if isinstance(emb, Embedding):
            emb.normalize()

    def _stage_serial_concept_rows(self):
        """Resolve each staged word to one exact joint CS identity eagerly.

        The compiled word cell must never walk Python relation dictionaries or
        scan the million-row concept inventory. At this boundary we therefore
        resolve ``surface + PS parts + WS properties`` against the durable
        word/location concept ``A`` and park only its codebook row. First-sight
        identities are admitted atomically at this safe pre-graph boundary;
        an admission refused at capacity remains the exact ``-1`` unknown.

        Stage 0 owns the relation identities while every aligned CS stage
        shares the same physical concept dictionary, so a row resolved here is
        valid for the terminal serial cell's indexed decode.
        """
        isp = getattr(self, "inputSpace", None)
        if isp is None:
            return None
        isp._ar_word_concept_rows = None
        isp._ar_word_concept_ids = None
        isp._ar_word_concept_orders = None
        isp._ar_concept_lookup_rows = None
        isp._ar_concept_lookup_atoms = None
        aligned = self._aligned_serial_word_mode()
        if not aligned:
            return None

        part_ids = getattr(isp, "_ar_word_part_ids", None)
        part_mask = getattr(isp, "_ar_word_part_mask", None)
        active = getattr(isp, "_word_active_mask", None)
        ps_fwd = getattr(getattr(self, "perceptualSpace", None),
                         "_forward_input", None)
        word_texts = (ps_fwd.get("word_texts")
                      if isinstance(ps_fwd, dict) else None)
        if (not torch.is_tensor(part_ids) or part_ids.dim() != 3
                or not torch.is_tensor(part_mask)
                or tuple(part_mask.shape) != tuple(part_ids.shape)
                or not torch.is_tensor(active) or active.dim() != 2
                or tuple(active.shape) != tuple(part_ids.shape[:2])
                or word_texts is None):
            return None

        spaces = list(getattr(self, "conceptualSpaces", None) or ())
        owner = spaces[0] if spaces else getattr(self, "conceptualSpace", None)
        ws = getattr(self, "wholeSpace", None)
        alloc = getattr(owner, "_concept_allocator", None)
        if owner is None or ws is None:
            rows = torch.full(
                active.shape, -1, dtype=torch.long, device=part_ids.device)
            isp._ar_word_concept_rows = rows
            isp._ar_word_concept_ids = rows.clone()
            isp._ar_word_concept_orders = rows.clone()
            return rows

        B, W, P = (int(part_ids.shape[0]), int(part_ids.shape[1]),
                   int(part_ids.shape[2]))
        pid_host = part_ids.detach().to("cpu").reshape(B, W, P).tolist()
        mask_host = part_mask.detach().to("cpu").reshape(B, W, P).tolist()
        active_host = active.detach().to("cpu").reshape(B, W).tolist()
        row_host = [[-1] * W for _ in range(B)]
        cid_host = [[-1] * W for _ in range(B)]
        order_host = [[-1] * W for _ in range(B)]
        pending = []
        # One base presentation plus T-1 cumulative native folds. The concept
        # activation draws from all T-1 PS and all T-1 WS fold results, so that
        # final cumulative depth is its actual subsymbolic order.
        actual_order = max(0, int(getattr(self, "subsymbolicOrder", 1)) - 1)
        fold_passes = tuple(range(actual_order))
        fold_support = owner._ordered_fold_support(
            fold_passes, fold_passes)
        wom = getattr(alloc, "word_obj_meta", {}) if alloc is not None else {}

        def _raw_int_refs(refs):
            out = set()
            for ref in refs or ():
                if isinstance(ref, tuple):
                    continue
                try:
                    out.add(int(ref))
                except (TypeError, ValueError):
                    continue
            return out

        for b in range(B):
            texts_b = word_texts[b] if b < len(word_texts) else ()
            for p in range(W):
                if not bool(active_host[b][p]):
                    continue
                value = texts_b[p] if p < len(texts_b) else None
                if value is None:
                    pending.append((b, p, None))
                    continue
                key = str(value)
                triple = wom.get(key)
                current_parts = {
                    int(pid_host[b][p][j]) for j in range(P)
                    if bool(mask_host[b][p][j])
                    and int(pid_host[b][p][j]) >= 0
                }
                current_wholes = set(int(v) for v in
                                     ws.property_rows_for_bytes(key))
                if triple is None:
                    # First sight is admitted here, at the same eager boundary
                    # that already owns radix growth and WS analysis. The
                    # aligned dictionary is physically fixed-capacity, so this
                    # mutates only bounded relation records, the active-prefix
                    # mask, and row metadata before any graph uses them. At
                    # capacity the automatic gate returns None and the exact
                    # -1 unknown path below remains live.
                    triple = owner._automatic_word_object_meta(
                        sorted(current_parts), sorted(current_wholes), key=key)
                    if triple is None:
                        pending.append((b, p, key))
                        continue
                    alloc = getattr(owner, "_concept_allocator", None)
                    wom = (getattr(alloc, "word_obj_meta", {})
                           if alloc is not None else wom)
                A = int(triple[0])
                stored_parts = _raw_int_refs(owner.concept_parts(A))
                stored_wholes = _raw_int_refs(owner.concept_wholes(A))
                # Surface identity alone is not enough: the active PS/WS
                # support must already belong to the durable concept. A trie
                # promotion or newly-discovered property therefore receives an
                # honest unknown tick, then autobind accrues it at Reset.
                if (not current_parts.issubset(stored_parts)
                        or not current_wholes.issubset(stored_wholes)):
                    pending.append((b, p, key))
                    continue
                row = owner._csw_row_of(A)
                if row is None:
                    row = owner._csw_concept_row(
                        owner._concept_source_order(A), A)
                if row is None:
                    pending.append((b, p, key))
                    continue
                row_host[b][p] = int(row)
                cid_host[b][p] = A
                record = owner.record_concept_fold_support(
                    A, fold_support, actual_order)
                order_host[b][p] = int(record["actual_order"])

        rows = torch.tensor(row_host, dtype=torch.long, device=part_ids.device)
        isp._ar_word_concept_rows = rows
        isp._ar_word_concept_ids = torch.tensor(
            cid_host, dtype=torch.long, device=part_ids.device)
        isp._ar_word_concept_orders = torch.tensor(
            order_host, dtype=torch.long, device=part_ids.device)
        # Perform the only sparse-gradient dictionary read once at this eager
        # boundary. The compiled word loop resolves all current/prior lexical
        # references against this small sentence bank. This is mathematically
        # identical to repeated F.embedding calls (their row gradients sum),
        # but avoids W graph breaks and repeated B*N gathers per iteration.
        cb = getattr(owner, "similarity_codebook", None)
        lookup = getattr(cb, "lookup_rows", None)
        if callable(lookup):
            # Keep this bank width invariant across the configured W buckets
            # so the same K=2 graph serves 16/32/64/128-word slabs.  Padding
            # rows gather row zero and are explicitly zero-masked; sparse
            # autograd may retain that structural index, but RowLocalAdam
            # already discards all-zero placeholder rows.
            bank_width = max(
                int(rows.shape[1]),
                int(getattr(self, "serial_word_capacity", 0) or 0),
                max(tuple(int(v) for v in
                          (getattr(self, "serial_word_buckets", ()) or (0,)))))
            bank_rows = torch.full(
                (int(rows.shape[0]), bank_width), -1,
                dtype=torch.long, device=rows.device)
            bank_rows[:, :int(rows.shape[1])] = rows
            lookup_rows = bank_rows.clamp_min(0).long()
            lookup_atoms = lookup(lookup_rows)
            if torch.is_tensor(lookup_atoms) and lookup_atoms.dim() == 3:
                lookup_atoms = lookup_atoms * bank_rows.ge(0).unsqueeze(-1).to(
                    dtype=lookup_atoms.dtype)
                isp._ar_concept_lookup_rows = bank_rows
                isp._ar_concept_lookup_atoms = lookup_atoms
        object.__setattr__(self, "_pending_sparse_concept_support", pending)
        return rows

    # -- per-step lifecycle (model-level) ------------------------------
    #
    # ``runBatch`` orchestrates a step but reaches only through the model,
    # never into model *contents* (``self.symbolSpace.discourse``,
    # ``self.inputSpace`` ...). The model owns the reusable helpers for its
    # lex+embed stem, discourse loss, and post-forward teardown. Per-batch
    # timing is NOT the grammar ``soft_reset`` (that fires only on chart
    # sentence-completion and never per-batch on the IR/MM_20M path, so
    # relocating discourse there would silently stop ARMA).

    def _lex_embed_stem(self, x):
        """Eager stem: lex (InputSpace) -> embed (PartSpace) -> finalize
        bookkeeping (InputSpace), model-orchestrated (2026-06-07).

        Replaces the retired ``InputSpace._peer_perceptual`` coupling: neither
        space holds a reference to the other; PartSpace is passed
        TRANSIENTLY to ``InputSpace.finalize_stem``. Runs in the EAGER pre-
        forward region (the compiled branch of ``runBatch``; inline for the
        eager path) so PS's host-side tokenization never enters the fullgraph
        trace -- the compiled body's ``PartSpace.forward`` then sees
        ``stem_embedded=True`` and skips re-embedding (pi only). Numeric input
        (``InputSpace.forward`` already embeds via the vocab codebook) returns
        ``stem_embedded=True`` and is passed through untouched.

        Dual view (analysis/synthesis plan, rev. 2026-06-09): InputSpace now
        emits ``(percepts_in, concepts_in)``; the ATOM view feeds the
        PS embed (synthesis), the UNITY view is parked on
        ``_staged_concepts_in`` for the symbolic branch (Phase 1: staged,
        UNUSED; Phase 2 consumes it at WS stage 0). This unpack is the
        orchestration-side shim the plan allows -- downstream contracts are
        unchanged.
        """
        # Keep the gated word-major radix mode in sync with the model flag on
        # every forward.  Tests/ablations may toggle ``serialObjectMeta`` on a
        # live model; a construction-time PartSpace stamp must not leave the
        # alternate representation active after the gate is disabled.
        if getattr(self, "perceptualSpace", None) is not None:
            object.__setattr__(
                self.perceptualSpace, "_serial_object_meta",
                bool(getattr(self, "serial_object_meta", False)
                     and getattr(self.inputSpace,
                                 "_serial_object_meta", False)))
        in_sub, concepts_in = self.inputSpace.forward(x)
        if (getattr(self, "serial", False)
                and getattr(self, "serial_object_meta", False)):
            self.inputSpace.select_word_loop_bucket(
                in_sub,
                tuple(self.serial_word_buckets
                      or (int(getattr(
                          self.inputSpace, "_serial_word_capacity", 1)),)),
                perceptual_space=self.perceptualSpace)
        self._staged_concepts_in = concepts_in
        # Dual-towers: an ALL-ZERO unity is NO unity -- validity is decided
        # HERE (host-eager stem, once per batch), never in routing. A None
        # offer routes the carrier body; a real unity routes universe,
        # unconditionally.
        _u_ok = (torch.is_tensor(concepts_in)
                 and bool((concepts_in != 0).any()))
        object.__setattr__(self, "_ws_universe",
                           concepts_in if _u_ok else None)
        # Phase 4b: host-eager analysis cut. The configured WS analysis
        # (word / analyse) divides the unity into parts HERE -- in the
        # eager stem, exactly like the PS host tokenization -- and parks
        # the spans on each WholeSpace for the stage-0 evidence
        # (pure tensor reads inside the compiled body). byte mode parks
        # None (uniform-region pooling needs no spans).
        _ws_list = (list(getattr(self, "wholeSpaces", None) or [])
                    or ([self.wholeSpace]
                        if getattr(self, "wholeSpace", None) is not None
                        else []))
        if _ws_list:
            _spans = (_ws_list[0].stage_analysis_spans(concepts_in)
                      if hasattr(_ws_list[0], "stage_analysis_spans")
                      else None)
            if (getattr(self, "overlap_where_tiling", False)
                    and hasattr(_ws_list[0], "stage_overlapping_spans")):
                _spans = _ws_list[0].stage_overlapping_spans(
                    concepts_in, _spans)
            _kinds = getattr(_ws_list[0], "_staged_analysis_kinds", None)
            for _ss in _ws_list:
                # Preserve the complete eager cut for diagnostics. The live
                # WS kernel has exactly inputShape[0] locations and already
                # consumed only that prefix via ``min(K, N)``. Fit it here so
                # a compiled fixed-W graph does not specialize on the
                # sentence-dependent number of type runs.
                object.__setattr__(_ss, "_staged_analysis_spans_full", _spans)
                object.__setattr__(_ss, "_staged_analysis_kinds_full", _kinds)
                _live_spans = _spans
                _live_kinds = _kinds
                if (getattr(self, "serial_object_meta", False)
                        and torch.is_tensor(_spans)):
                    _n_live = max(1, int(_ss.inputShape[0]))
                    _live_spans = _spans[:, :_n_live, :]
                    if int(_live_spans.shape[1]) < _n_live:
                        _live_spans = torch.cat([
                            _live_spans,
                            _live_spans.new_zeros(
                                int(_live_spans.shape[0]),
                                _n_live - int(_live_spans.shape[1]), 2),
                        ], dim=1)
                    # B=1 makes a sliced tensor report ``is_contiguous`` even
                    # when its size-1 batch stride still reflects full K.
                    # Force a compact allocation so stride guards are stable.
                    _live_spans = _live_spans.clone(
                        memory_format=torch.contiguous_format)
                    if torch.is_tensor(_kinds):
                        _live_kinds = _kinds[:, :_n_live]
                        if int(_live_kinds.shape[1]) < _n_live:
                            _live_kinds = torch.cat([
                                _live_kinds,
                                _live_kinds.new_zeros(
                                    int(_live_kinds.shape[0]),
                                    _n_live - int(_live_kinds.shape[1])),
                            ], dim=1)
                        _live_kinds = _live_kinds.clone(
                            memory_format=torch.contiguous_format)
                _ss._staged_analysis_spans = _live_spans
                _ss._staged_analysis_kinds = _live_kinds
            # Phase 5 adopt-on-first-sight (host-eager): VIRGIN WS
            # codebook rows adopt the stage-0 evidence BEFORE the
            # compiled body snaps, so the STE substitutes data-
            # initialised prototypes rather than frozen random rows
            # (the #13 forward-poison). Data-dependent (``unique``) --
            # must stay out of the compiled graph.
            if (self.training and concepts_in is not None
                    and hasattr(_ws_list[0], "adopt_stage0_evidence")):
                _ws_list[0].adopt_stage0_evidence(concepts_in, _spans)
            # Step 1 (2026-06-10 symbolic-iteration plan): CS-leg
            # adopt-on-first-sight + virgin staging. Stage t's WS adopts
            # from stage t-1's PERSISTENT CS view (one-step-stale: the
            # in-step evidence is produced inside the compiled body,
            # where the data-dependent ``unique`` cannot live), then
            # parks the virgin-row mask the body's symbolic emission
            # reads -- continuous fallback until the winner row has been
            # adopted. Serial/grammar spaces no-op inside the methods.
            _dev = (concepts_in.device
                    if torch.is_tensor(concepts_in) else None)
            _prev_cs = None
            for _stage_k in (getattr(self, "body_stages", None) or []):
                _ws_k = _stage_k["ws"] if "ws" in _stage_k else None
                if _ws_k is not None:
                    if (self.training and _prev_cs is not None
                            and hasattr(_ws_k, "adopt_symbolic_evidence")):
                        _ws_k.adopt_symbolic_evidence(
                            getattr(_prev_cs, "_subspaceForWS", None))
                    if hasattr(_ws_k, "stage_symbolic_virgin_rows"):
                        _ws_k.stage_symbolic_virgin_rows(device=_dev)
                _prev_cs = _stage_k["cs"] if "cs" in _stage_k else None
        if in_sub is None:
            return in_sub
        if hasattr(in_sub, "is_empty") and in_sub.is_empty():
            return in_sub
        if getattr(in_sub, "stem_embedded", True) is False:
            # The callback is transient to avoid a persistent model<->space
            # reference cycle. Atomic byte growth occurs inside embed_stem,
            # before its first W gather; queued word growth remains post-step.
            object.__setattr__(
                self.perceptualSpace, "_radix_growth_callback",
                self._on_partspace_eager_growth)
            try:
                self.perceptualSpace.embed_stem(in_sub)
            finally:
                object.__setattr__(
                    self.perceptualSpace, "_radix_growth_callback", None)
            self.inputSpace.finalize_stem(in_sub, self.perceptualSpace)
        # Resolve sparse concept identities only after the word-major PS stem
        # has exposed its exact residual parts and WS has staged the matching
        # property analysis. This remains wholly eager; the compiled word cell
        # sees fixed tensors and performs no dictionary walk or host sync.
        self._stage_serial_concept_rows()
        # The PS embed now exposes exact per-percept brackets.  Once both
        # towers have staged their candidates, precompute the fixed-T
        # structural schedule in the eager stem; learned content still runs
        # through the ordinary compiled sigma/pi/callosum body.
        if getattr(self, "overlap_where_tiling", False) and _ws_list:
            _pf = getattr(self.perceptualSpace, "_forward_input", None)
            _part_spans = (_pf.get("part_spans")
                           if isinstance(_pf, dict) else None)
            _schedule = _ws_list[0].stage_where_tiling(
                _part_spans, self.subsymbolicOrder)
            for _ss in _ws_list[1:]:
                _ss._where_tiling_schedule = _schedule
                _ss._where_tiling_obs = (_schedule[0]
                                         if _schedule else None)
        return in_sub

    def _stage_intersentence_seed(self):
        """Eagerly compute + park the stage-0 CS_{-1} seed for the upcoming
        compiled forward (Task A6); ``_consume_intersentence_seed`` returns it
        in-trace. Gated to ``<prediction>interSentence`` (parks None
        otherwise). No-op-safe to call repeatedly; ``_end_step`` clears it."""
        seed = None
        if self.prediction_mode == "interSentence":
            seed = self._intersentence_seed()
        self._staged_intersentence_seed = seed
        self._intersentence_seed_staged = True

    def _end_step(self):
        """Per-step teardown: drop the staging parked by ``runBatch``
        (consume-once; eager, post-forward)."""
        # The complete captured W-loop uses only the live tensor depth for
        # capacity demand; it intentionally never reads or mutates the host
        # mirror inside the graph.  Synchronize that mirror once here for
        # eager sentence-boundary consumers (reports, snapshots, and a later
        # eager call) instead of letting its changing Python value fragment
        # the fullgraph specialization.
        if (getattr(self, "_compiled_word_loop_fullgraph", False)
                and self._active_compiled_step is not None):
            stm = getattr(getattr(self, "conceptualSpace", None), "stm", None)
            depth = getattr(stm, "_depth", None) if stm is not None else None
            if torch.is_tensor(depth) and depth.numel():
                stm._max_depth_host = int(depth.max().item())
        self._drain_pending_stm_end_state()
        self._staged_in_sub = None
        self._staged_concepts_in = None
        self._active_compiled_step = None
        for _ss in (getattr(self, "wholeSpaces", None) or []):
            _ss._staged_analysis_spans = None
            _ss._staged_analysis_kinds = None
            _ss._staged_analysis_spans_full = None
            _ss._staged_analysis_kinds_full = None
            _ss._where_tiling_schedule = None
            _ss._where_tiling_obs = None
        self._staged_intersentence_seed = None
        self._intersentence_seed_staged = False
        # Drop the global-attention soft-read (consume-once; the consumer in
        # Finish read it earlier this forward) so a stale obs can never leak
        # into the next forward's head.
        if getattr(self, "global_attention", None) is not None:
            self._global_attention_obs = None
        disc = (self.symbolSpace.discourse
                if self.symbolSpace is not None else None)
        if disc is not None:
            disc.clear_staged_prediction()

    def _drain_pending_stm_end_state(self):
        """Persist a compiled forward's parked STM boundary on the host.

        The forward graph produces only tensors. Ragged payload construction,
        ``tolist`` and persistent-store mutation run here, after the compiled
        callable returns and before the rest of the batch consumes the staged
        discourse prediction. Eager forwards continue using their existing
        in-place boundary path and leave this slot empty.
        """
        pending = getattr(self, "_pending_stm_end_state", None)
        object.__setattr__(self, "_pending_stm_end_state", None)
        if pending is None:
            return
        cs_buf, rel_mask = pending
        discourse = (self.symbolSpace.discourse
                     if getattr(self, "symbolSpace", None) is not None
                     else None)
        ltm_store = getattr(self.symbolSpace, "ltm_store", None)
        ltm_on = bool(
            getattr(self.conceptualSpace, "_ltm_consolidation", False)
            and ltm_store is not None)
        discourse_live = bool(
            discourse is not None
            and hasattr(discourse, "observe_stm_end_state"))
        if not (discourse_live or ltm_on):
            return
        B, cap = int(cs_buf.shape[0]), int(cs_buf.shape[1])
        rel_rows = rel_mask.reshape(-1).to("cpu").tolist()
        depths = [min(3 if bool(rel_rows[b]) else 1, cap)
                  for b in range(B)]
        payloads = [cs_buf[b, :depths[b], :] for b in range(B)]
        tetralemmas = self.conceptualSpace.stm_end_state_trust(
            cs_buf, rel_mask)
        if discourse_live:
            discourse.predict_and_observe_stm_end_state(
                depths, payloads, tetralemmas=tetralemmas)
        if ltm_on:
            for b, payload in enumerate(payloads):
                if payload is None or int(payload.shape[0]) < 1:
                    continue
                d = max(1, min(int(depths[b]), int(payload.shape[0])))
                tet = (tetralemmas[b]
                       if tetralemmas is not None and b < len(tetralemmas)
                       else None)
                trust = float(tet) if tet is not None else 0.0
                if d >= 3:
                    ltm_store.append_relation(
                        payload[d - 2], payload[d - 1], payload[0],
                        rel_type=TernaryTruthStore.REL_OTHER, trust=trust)
                else:
                    ltm_store.append_idea(payload[0], trust=trust)

    def _intersentence_seed(self):
        """The predicted next-end-state SHAPE for the stage-0 CS_{-1} seed,
        or ``None`` (Task A6).

        THE single inter-sentence predictor source: this is the SAME
        ``discourse.predict_next_end_state()`` call ``generate_sentence``
        primes from (``generate_sentence`` now routes through here too), so
        the forward seed and the chat-loop prime share ONE predictor path --
        never two divergent calls.

        Returns ``(depth_hat:int, payload_hat:[depth_hat, D] tensor)`` -- the
        predicted next STM end-state slots in C-space_role (the muxed concept event
        width ``D == discourse._inter_predictor.concept_dim``) -- or ``None``
        when there is nothing real to seed from:

          * no discourse layer / no inter-predictor (``sentencePrediction``
            off, absolute-only configs);
          * a COLD AR ring (empty chain): ``predict_next_end_state`` would
            return the degenerate ``(1, zeros[1, D])``; we treat that as "no
            seed" so a cold start is byte-identical to the empty seed
            (back-compat -- the seed sites keep their zeros);
          * a non-finite predicted payload (defensive -- fail loud is the
            predictor's job, but never seed a corrupt carrier).

        Does NOT gate on ``prediction_mode``: the gate lives at the forward
        seed sites (so ``generate_sentence`` keeps its always-on priming when
        a predictor exists, unchanged). Mirrors ``generate_sentence``'s guard
        exactly (predictor present + non-empty chain + finite payload).
        """
        disc = (self.symbolSpace.discourse
                if self.symbolSpace is not None else None)
        if disc is None or getattr(disc, "_inter_predictor", None) is None:
            return None
        shape = disc.predict_next_end_state()
        # Empty AR ring -> degenerate cold-start prediction; not a real seed.
        if shape is None or not disc.get_stm_chain(n=1):
            return None
        depth_hat, payload_hat = shape
        if payload_hat is None or not torch.isfinite(payload_hat).all():
            return None
        return depth_hat, payload_hat

    def _consume_intersentence_seed(self):
        """Traced-safe accessor for the stage-0 CS_{-1} seed, gated by
        ``<prediction>interSentence`` (Task A6).

        Returns ``(depth_hat, payload_hat)`` or ``None``. Mirrors the
        established ARMA staging idiom (``InterSentenceLayer.predict`` /
        ``stage_prediction``): ``predict_next_end_state`` is
        ``@torch.compiler.disable``'d, so it must NOT run inside the compiled
        forward. ``runBatch``'s eager compiled pre-forward branch parks the
        seed via ``_stage_intersentence_seed`` and sets
        ``_intersentence_seed_staged``; this accessor then returns the parked
        tuple -- a pure attribute read, trace-safe. On the eager/uncompiled
        path (direct ``forward``, unit tests) nothing is staged, so it computes
        the seed live (the disabled predictor is fine there -- not traced).

        The ``prediction_mode != "interSentence"`` gate lives HERE (not in
        ``_intersentence_seed``) so ``generate_sentence`` keeps its always-on
        priming when a predictor exists, while the forward seed only fires
        under interSentence.
        """
        if self.prediction_mode != "interSentence":
            return None
        if getattr(self, "_intersentence_seed_staged", False):
            return self._staged_intersentence_seed
        return self._intersentence_seed()

    def _discourse_arma_loss(self):
        """Inter-sentence ARMA(p, q) loss term for this step, or ``None``.

        Encapsulates ``InterSentenceLayer`` so runBatch sees only a
        model-level loss contribution. Must be called post-body /
        pre-backward: the term trains the ARMA predictor, and
        ``observe`` also commits the sentence rep + residual into the
        per-row rings (vectorized, sync-free).
        """
        disc = (self.symbolSpace.discourse
                if self.symbolSpace is not None else None)
        if disc is None or self._current_discourse_s is None:
            return None
        return disc.observe(self._current_discourse_s)

    def _discourse_inter_loss(self):
        """Inter-sentence end-state prediction loss term (Task 8, plan §9),
        or ``None``.

        The sentence-boundary hook ran ``predict_next_end_state`` (staging a
        predicted root) and ``observe_stm_end_state`` (scoring it against the
        arriving end-state, accumulating ``L_inter`` live on the discourse
        layer). Consume the per-sentence mean here, post-body / pre-backward
        (mirroring the ARMA + intra terms), so the term trains the
        inter-level predictor. Returns ``None`` when the discourse layer is
        absent (absolute-only configs no-op) or nothing was accumulated
        (eval, weight off, or no scored sentence this batch)."""
        disc = (self.symbolSpace.discourse
                if self.symbolSpace is not None else None)
        if disc is None:
            return None
        return disc.consume_inter_loss()

    def present(self) -> int:
        """The absolute model time (the serialized ``when_time`` clock).

        0-initialized; ``runBatch`` advances it by 1 per processed batch on
        BOTH train and inference. It is the authoritative absolute-time
        record (the .when sinusoid only carries local angular resolution)."""
        return int(self.when_time)

    def _advance_when_time(self):
        """Tick the model clock once and propagate ``present()`` to the live
        ``WhenRangeEncoding`` instances so their ``.t`` reference equals the
        absolute time at stamping. Called exactly once per processed batch in
        ``runBatch`` (train AND inference), in eager Python (outside the
        compiled forward), so the encoders the forward stamps with already see
        the advanced time. The clock is a long buffer; the in-place add keeps
        it on-device and serializable. The encoders' ``.t`` is a plain Python
        int (no grad, no device tensor) read by ``encode``/``forward``."""
        # In-place so the registered buffer (and its device) is preserved.
        self.when_time += 1
        t_now = int(self.when_time)
        for sp in getattr(self, "spaces", []) or []:
            sub = getattr(sp, "subspace", None)
            enc = getattr(sub, "whenEncoding", None) if sub is not None else None
            # Only the enabled (nDim > 0) range encoders carry a stampable .t;
            # the disabled (nWhen == 0) carrier is a no-op either way.
            if enc is not None and getattr(enc, "nDim", 0) > 0:
                enc.t = t_now

    def _set_superposition_temperature(self, temperature):
        """Set (``None`` clears) the soft-superposition route temperature on
        every structured grammar layer for a two-pass learning trial.

        ``temperature`` 0 = sharp/deterministic superposition (the exploit
        pass), 1 = flat/uniform (the explore pass); ``None`` restores the
        legacy hard-Viterbi + straight-through forward (the default,
        byte-identical). Reaches the router's unary / binary layers and the
        STM reducer (the structured layers whose forward reads it)."""
        layers = []
        ss = getattr(self, 'symbolSpace', None)
        ll = getattr(ss, 'languageLayer', None) if ss is not None else None
        if ll is not None:
            layers.extend((getattr(ll, '_unary_layers', {}) or {}).values())
            layers.extend((getattr(ll, '_binary_layers', {}) or {}).values())
        stm = getattr(self, '_stm_reducer_module', None)
        if stm is not None:
            layers.append(stm)
        for layer in layers:
            layer.superposition_temperature = temperature
        # Stash model-level too so the attention selection (ReadingAttention /
        # GlobalAttention) honours the SAME two-pass temperature as the grammar
        # chooser: pass A (t=0/None) -> superposition_scale 1 -> sharp/exploit
        # (byte-identical); pass B (t=exploreTemperature) -> flatter -> explore.
        self._superposition_temperature = temperature

    def runBatch(self, train=True, batchNum=0, batchSize=10, split="train",
                 optimizer=None, batch_override=None, progress=None,
                 superposition_temperature=None, exploration_trial=False,
                 trial_mode="reconstruct"):
        """Run a single batch: forward pass, loss, and (if training) backward + step.

        ``superposition_temperature`` (two-pass learning): when not None, the
        structured grammar layers use the pure soft-superposition route at
        this temperature for this trial (0 = sharp/deterministic, 1 = flat);
        None (default) keeps the legacy hard-Viterbi + straight-through
        forward, so an ordinary single-pass step is byte-identical.

        ``exploration_trial`` (two-pass learning, pass B): when True this is
        the non-recorded explore trial over a sentence already processed by
        pass A. It still trains the chooser (forward / loss / backward /
        step), but the per-sentence side effects are SKIPPED so B does not
        double-commit the same sentence: the model clock is not advanced
        (``_advance_when_time``), the discourse ARMA observe and the LTM
        end-state append are not re-run, and ``_training_step_count`` (the
        periodic-checkpoint cadence) is not incremented. Default False is the
        ordinary recorded pass.

        Args:
            train: whether to compute gradients and update parameters.
            batchNum: opaque cursor for the next batch position.
            batchSize: number of examples per batch.
            split: "train", "test", or "validation".
            optimizer: pre-built optimizer (required when train=True).
            batch_override: optional ``(inputTensor, outputTensor)`` pair;
                the primary dispatch path used by the DataLoader streaming
                path in ``runEpoch``.
            progress: optional fraction in ``[0.0, 1.0]`` indicating how
                far through the current split's data the cursor has
                advanced. When set, the per-batch timing line includes
                a percentage so long runs report visible progress.
                ``runEpoch`` populates this from
                ``SentenceStreamDataset.progress()``; callers that drive
                ``runBatch`` directly leave it ``None``.

        Returns:
            (BatchResult, nextBatchNum) on success, or (None, batchNum) when
            the dataset is exhausted.
        """
        # A caller may supply an optimizer built outside ``getOptimizer``.
        # Make that live owner visible to the eager PartSpace byte-growth
        # boundary before lexing this batch.
        if optimizer is not None and getattr(
                self, "perceptualSpace", None) is not None:
            object.__setattr__(
                self.perceptualSpace, "_radix_optimizer", optimizer)
        # A radix growth boundary drops every callable that captured the old
        # PartSpace W Parameter. Rebuild before staging the next batch so the
        # new physical row count receives one safe specialization.
        if getattr(self, "_compiled_step_needs_rebuild", False):
            self.enable_compiled_step()
            self._compiled_step_needs_rebuild = False

        # First-call hook: try to enable §7 torch.compile reduce-overhead
        # mode on CUDA targets. Idempotent; safe on non-CUDA hosts.
        self._maybe_compile_brick()
        sentenceIdx = batchNum  # sentence index before batchNum increments
        if batch_override is None:
            raise RuntimeError(
                "runBatch: no batch_override supplied. Callers must pass "
                "batch_override=(inputTensor, outputTensor) — training "
                "path via the DataLoader in runEpoch, inference path via "
                "InputSpace.prepInput (or, for sentence-level generation, "
                "BasicModel.generate_sentence)."
            )
        batch = batch_override
        inputTensor, outputTensor = batch
        inference_only = not train and split == "runtime"

        # Advance the serialized model clock once per PROCESSED batch (this
        # point is past the no-batch early raise, so it ticks exactly once on
        # BOTH the train and inference paths regardless of which return fires
        # below). Done here, in eager Python before the forward, so the live
        # WhenRangeEncoding(s) the forward stamps already carry the advanced
        # absolute time (``.t == present()``). See ``_advance_when_time``.
        # Skipped on the explore trial (pass B): it re-processes a sentence
        # pass A already clocked, so a second tick would double-advance time.
        if not exploration_trial:
            self._advance_when_time()

        # Pre-allocate per-batch state OUTSIDE the compiled forward.
        # ``SymbolSpace.ensure_microbatch`` allocates ``_stm_fired`` /
        # ``_last_svo`` / ``_svo_valid`` / ``_recent_count`` etc. on
        # first call (and on shape changes); when those allocations
        # happen INSIDE a torch.compile region, CUDAGraph capture
        # takes ownership of the underlying memory, so the next
        # replay's allocation overwrites the previous step's state
        # and the next attribute read raises ``RuntimeError: Error:
        # accessing tensor output of CUDAGraphs that has been
        # overwritten by a subsequent run.`` Hoisting the call up
        # here keeps the resulting tensors Python-owned.
        if self.symbolSpace is not None and not inference_only:
            ss = self.symbolSpace
            try:
                if isinstance(inputTensor, torch.Tensor):
                    B_pre = int(inputTensor.shape[0])
                else:
                    B_pre = int(len(inputTensor))
            except Exception:
                B_pre = None
            if B_pre is not None:
                ss.ensure_microbatch(B_pre, 1)

        if train:
            optimizer.zero_grad()

        # Start a fresh error-term registration window for this batch.
        # TheError accumulates a breakdown of every loss term (category,
        # space, weight, value) alongside the legacy ``totalLoss`` path,
        # so later sites can call ``TheError.breakdown()`` /
        # ``TheError.covariance()`` without having to rewire the backprop
        # source.  See Layers.Error docstring for usage.
        TheError.reset()
        TheError.attach(self.loss)

        # Per-batch Space.Start cascade (moved out of forward() so sliding
        # buffers can persist across forward() calls within a stream).
        self._start_spaces_for_forward()

        # AMP: torch.autocast wrapper from util.amp_context() honors
        # MODEL_AMP env var (hydrated from XML <architecture><amp> in
        # ModelFactory.run). bf16 returns scaler=None; fp16+CUDA returns
        # the process-wide GradScaler used in the backward path below.
        amp_cm, amp_scaler = amp_context()
        with amp_cm:
            # Forward pass returns a 4-tuple.  IR-only contract:
            # ``predictions`` is ``[B, N, predDim]`` (one head emission
            # per P-slot) and ``forwardInput`` is the inputSpace event
            # ``[B, N, D]``.  ``reconstruction`` is always None after
            # the reverse pipeline retirement (2026-05-14); the slot
            # is kept in the tuple for downstream code that pattern-
            # matches on the legacy shape.
            #
            # ``cudagraph_mark_step_begin`` (only meaningful under modes
            # that capture CUDAGraphs -- "reduce-overhead", "max-
            # autotune") tells the runtime to release the previous
            # step's CUDAGraph outputs so the memory pool can be reused
            # for this step.  No-op under "default" mode; idempotent on
            # non-CUDA hosts.
            try:
                torch.compiler.cudagraph_mark_step_begin()
            except (AttributeError, RuntimeError):
                pass
            # O1: route the per-batch compute through the compiled
            # callable when enabled (else eager). runEpoch/runBatch stay
            # eager Python; only this forward+loss+backward unit is
            # torch.compiled. See doc/plans/2026-05-16-compiled-step-
            # boundary-design.md.
            # Keep eager staging immediately beside the compiled invocation:
            # the traced forward only reads these parked tensors.  The eager
            # path intentionally remains unchanged and lexes inline.
            if self._compiled_step is not None:
                if isinstance(inputTensor, torch.Tensor):
                    inputTensor = inputTensor.to(TheDevice.get())
                self._staged_in_sub = self._lex_embed_stem(inputTensor)
                _requires_sparse_bank = self._aligned_serial_sparse_bank_mode()
                if _requires_sparse_bank:
                    _bank_rows = getattr(
                        self.inputSpace, "_ar_concept_lookup_rows", None)
                    _bank_atoms = getattr(
                        self.inputSpace, "_ar_concept_lookup_atoms", None)
                    if (not torch.is_tensor(_bank_rows) or _bank_rows.dim() != 2
                            or not torch.is_tensor(_bank_atoms)
                            or _bank_atoms.dim() != 3
                            or tuple(_bank_rows.shape)
                            != tuple(_bank_atoms.shape[:2])):
                        raise RuntimeError(
                            "compiled aligned serial forward requires the eager "
                            "sparse concept row bank; lexical staging did not "
                            "produce a shape-aligned rows/atoms pair")
                # A full W-loop owns one fixed, identity-masked residual-part
                # layout.  Do not give MPS a symbolic P range or compile a
                # separate full loop for every radix spelling.  The older
                # word-cell path retains its broad dynamic contract below.
                _fullgraph_word_loop = bool(getattr(
                    self, "_compiled_word_loop_fullgraph", False))
                if _fullgraph_word_loop:
                    # Cold-start sentence state must be materialized before
                    # Dynamo observes the word-loop module.  Creating this
                    # attribute inside the captured prelude changes an
                    # ``hasattr`` guard after the first forward and forces a
                    # duplicate W specialization.
                    _ss = getattr(self, "symbolSpace", None)
                    if (_ss is not None
                            and not getattr(_ss, "_per_sentence_initialized", False)):
                        _ss.soft_reset()
                        _ss._per_sentence_initialized = True
                    if _ss is not None:
                        # Establish every scalar/carrier attribute the
                        # captured recurrence writes before its first guard
                        # census.
                        object.__setattr__(
                            _ss, "_target_cursor_length", int(getattr(
                                self.inputSpace, "_active_word_bucket", 0) or 0))
                    object.__setattr__(
                        self, "_prev_cs_for_ps", getattr(
                            self, "_empty_seed_ps", None))
                    object.__setattr__(
                        self, "_prev_cs_for_ss", getattr(
                            self, "_empty_seed_ss", None))
                    # ``_per_word_prelude`` creates a fresh STM working slab
                    # inside the graph.  Establish the same MPS-backed state
                    # before the first guard census; otherwise the
                    # constructor's CPU placeholder is observed on call one
                    # and the MPS replacement forces call two to compile
                    # again.
                    _staged_event = self._staged_in_sub.materialize()
                    _stm = getattr(
                        getattr(self, "conceptualSpace", None), "stm", None)
                    if torch.is_tensor(_staged_event) and _stm is not None:
                        _stm.begin_forward(
                            int(_staged_event.shape[0]),
                            device=_staged_event.device,
                            dtype=(_staged_event.dtype
                                   if _staged_event.is_floating_point() else None))
                    self._stage_fixed_residual_part_capacity()
                else:
                    # W is deliberately static (one of 16/32/64/128); the
                    # number of residual radix constituents inside one
                    # complete word is not.  Mark only that P axis dynamic so
                    # promotion (e.g. five pieces -> one stored percept) does
                    # not compile another W graph. P>=3 excludes the
                    # contradictory Inductor guard ``P - 1 != 1`` while
                    # masked identity padding preserves words with one or two
                    # real constituents.
                    _part_min = 3
                    _part_max = max(_part_min, int(getattr(
                        self.inputSpace, "outputShape", (8192,))[0]))
                    for _part_tensor in (
                            getattr(self.inputSpace, "_ar_word_part_ids", None),
                            getattr(self.inputSpace, "_ar_word_part_mask", None),
                            getattr(self.inputSpace, "_ar_word_part_offsets", None)):
                        if torch.is_tensor(_part_tensor) and _part_tensor.dim() == 3:
                            torch._dynamo.mark_dynamic(
                                _part_tensor, 2, min=_part_min, max=_part_max)
                    _word_width = max(1, int(getattr(
                        self.inputSpace, "_active_word_bucket", 1) or 1))
                    _flat_min = _part_min * _word_width
                    _flat_max = _part_max * _word_width
                    _ps_forward = getattr(
                        self.perceptualSpace, "_forward_input", None)
                    if isinstance(_ps_forward, dict):
                        for _name in ("indices", "seed_event", "word_groups",
                                      "part_spans", "percept_where"):
                            _value = _ps_forward.get(_name)
                            if torch.is_tensor(_value) and _value.dim() >= 2:
                                torch._dynamo.mark_dynamic(
                                    _value, 1, min=_flat_min, max=_flat_max)
                        for _name in ("word_part_indices", "word_part_mask"):
                            _value = _ps_forward.get(_name)
                            if torch.is_tensor(_value) and _value.dim() == 3:
                                torch._dynamo.mark_dynamic(
                                    _value, 2, min=_part_min, max=_part_max)
                for _cs in (getattr(self, "conceptualSpaces", None) or ()):
                    _validate_routing = getattr(_cs, "validate_intra_routing", None)
                    if callable(_validate_routing):
                        _validate_routing()
                _active_bucket = int(getattr(
                    self.inputSpace, "_active_word_bucket", 0) or 0)
                if _fullgraph_word_loop:
                    self._active_compiled_step = (
                        self._ensure_lazy_fullgraph_word_loop(_active_bucket))
                else:
                    self._active_compiled_step = self._compiled_word_steps.get(
                        _active_bucket, self._compiled_step)
                disc = (self.symbolSpace.discourse
                        if self.symbolSpace is not None else None)
                if disc is not None:
                    disc.stage_prediction()
                    # Cast the staged ARMA prediction tuple to the active AMP
                    # dtype so a dtype mismatch inside the compiled forward
                    # doesn't split the graph
                    # (doc/plans/2026-05-20-static-per-word-loop-impl.md §2.7).
                    target_dtype = None
                    if _util.MODEL_AMP == "fp16":
                        target_dtype = torch.float16
                    elif _util.MODEL_AMP == "bf16":
                        target_dtype = torch.bfloat16
                    staged = getattr(disc, "_staged_prediction", None)
                    if target_dtype is not None and staged is not None:
                        pred, conf = staged
                        if (pred is not None
                                and torch.is_tensor(pred)
                                and pred.is_floating_point()
                                and pred.dtype != target_dtype):
                            pred = pred.to(target_dtype)
                        if (conf is not None
                                and torch.is_tensor(conf)
                                and conf.is_floating_point()
                                and conf.dtype != target_dtype):
                            conf = conf.to(target_dtype)
                        disc._staged_prediction = (pred, conf)
                # Park the stage-0 CS_{-1} interSentence seed outside the
                # fullgraph capture; the body consumes the parked tuple.
                self._stage_intersentence_seed()
            # Two-pass learning: switch the structured layers to the pure
            # soft-superposition route at this trial's temperature (eager
            # forward, since the compiled step does not trace the branch).
            if superposition_temperature is not None:
                self._set_superposition_temperature(
                    float(superposition_temperature))
                _fwd = self.forward
            else:
                # Diagnostic passes run under ``torch.no_grad()`` -- training
                # is ALWAYS grad-on (verified 2026-07-07: the only grad-off
                # forward in a training run is runTrial's
                # ``_reconstructionReport``). Entering the compiled step with
                # a different ambient grad mode fails its GLOBAL_STATE guard
                # and forces a full retrace (~100-380s each on MPS), so
                # non-grad passes take the eager forward -- the same
                # precedent as the superposition_temperature override above.
                _fwd = ((self._active_compiled_step
                         or self._compiled_step or self.forward)
                        if torch.is_grad_enabled() else self.forward)
            # Explore trial (pass B): gate the forward-internal per-sentence
            # commit (the LTM end-state append) so it does not fire a second
            # time on the same sentence. Reset in the finally below.
            self._exploration_trial = bool(exploration_trial)
            try:
                forwardInput, symbols, predictions, _ = _fwd(inputTensor)
            finally:
                if superposition_temperature is not None:
                    self._set_superposition_temperature(None)
                self._exploration_trial = False
            self._end_step()
            # Unconditional SEEN priming: the rows this forward fired bump
            # their surfaces (no-grad bookkeeping; consumers read next batch).
            self._prime_seen_step()
            outputDataPred = predictions

            # ε-growing codebook hook (Phase 4 follow-up): when any
            # codebook-bearing space carries ``codebookGrowthEpsilon > 0``
            # in its XSD, invoke ``VectorQuantize.grow_on_novelty`` on
            # the encoder output to insert novel inputs into empty
            # slots before the EMA path locks in assignments.  No-op
            # when ``growth_epsilon`` is 0 (default) or every slot is
            # already populated.
            if train:
                for _sp_attr in ("perceptualSpace", "conceptualSpace",
                                 "wholeSpace"):
                    _sp = getattr(self, _sp_attr, None)
                    _cb = getattr(getattr(_sp, "subspace", None),
                                  "what", None)
                    _vq = getattr(_cb, "vq", None)
                    if _vq is None:
                        continue
                    _eps = float(getattr(_vq, "growth_epsilon", 0.0)
                                 or 0.0)
                    if _eps <= 0.0:
                        continue
                    _ev = (_sp.subspace.materialize()
                           if _sp.subspace is not None else None)
                    if _ev is None:
                        continue
                    try:
                        _vq.grow_on_novelty(_ev, _eps)
                    except Exception:
                        # Growth is best-effort -- never let an
                        # insertion glitch stop the training step.
                        pass

            if inference_only:
                # Inference path: forward only, no loss.
                result = self.BatchResult(
                    outputPred=outputDataPred, symbols=symbols,
                    lossOut=None, lossIn=None,
                    inputPred=None, forwardInput=forwardInput,
                )
                self.End()
                return result, batchNum

            if outputTensor is None:
                raise RuntimeError(
                    f"runBatch: missing output targets for split='{split}'. "
                    "For inference use split='runtime'."
                )

            # 2026-05-28: restore supervised output-head loss for
            # tasks that need it (e.g. XOR_exact with binary labels).
            # The prior IR-only regime hardcoded output_weight=0 and
            # routed all gradient through the masked-LM ``lossIn``;
            # that path is preserved (when ``outputTensor`` is absent
            # or the labels are zero-width, the supervised branch
            # degenerates to the original "side channel" semantics).
            # When ``outputTensor`` IS supplied, compare the head
            # prediction against the labels and apply a non-zero
            # weight so the head receives gradient. Mirrors the
            # ``reconstruction_reverse`` pattern (try-guarded so a
            # shape edge case degrades to zero contribution rather
            # than crashing the step).
            lossOut = torch.zeros((), device=TheDevice.get())
            output_weight = 0.0
            try:
                if (getattr(self.inputSpace.data,
                            "has_supervised_outputs", True)
                        and outputTensor is not None
                        and torch.is_tensor(outputTensor)
                        and outputTensor.numel() > 0
                        and outputDataPred is not None
                        and torch.is_tensor(outputDataPred)
                        and outputDataPred.numel() > 0):
                    # shape reconciliation lives in _align_output_pred; irreconcilable warns once and zeroes the term.
                    _pred = self._align_output_pred(outputDataPred,
                                                    outputTensor)
                    if _pred is not None:
                        lossOut = self.loss.compute(_pred, outputTensor)
                        output_weight = 1.0
            except Exception as _out_exc:
                # Best-effort degrade to zero, but never SILENTLY (5b fail-loud).
                lossOut = torch.zeros((), device=TheDevice.get())
                output_weight = 0.0
                self._warn_zeroed_channel(
                    "output_loss_exception",
                    f"supervised output loss zeroed by "
                    f"{type(_out_exc).__name__}: {_out_exc}")
            TheError.add(
                "output", lossOut,
                weight=output_weight,
                space="OutputSpace", category="prediction",
            )

            # IR masked-LM loss: compare the post-body perceptual event
            # at masked positions against the pre-mask embedding the
            # forward snapshotted in ``_ir_pre_mask_input``.  No reverse
            # pipeline involved -- the body / head ran masked, the
            # masked positions carry the prediction target, the head
            # plays no role in the loss.  This is the BERT-style
            # masked-LM contract.
            inputDataPred = None
            inputPred = None
            mask_pos = getattr(self, "_ir_mask_positions", None)
            pre_mask = getattr(self, "_ir_pre_mask_input", None)
            pred_full = None
            if hasattr(self.perceptualSpace.subspace, 'materialize'):
                pred_full = self.perceptualSpace.subspace.materialize()
            # None checks stay Python (no sync). The masked reconstruction
            # loss is computed densely via `compute_masked` instead of a
            # boolean-mask gather `pred[mask]` -- the gather is
            # data-dependent (its row count is read to host: an implicit
            # cudaMemcpyDtoH that breaks CUDA-graph capture).
            # `compute_masked` is the sync-free, value-equivalent form
            # (masked-sum / masked-count) and returns 0.0 on an empty
            # mask with no NaN (replacing the old nan_to_num gate). See
            # doc/BrickHostSyncStatus.md residual D.
            # Rework B (3): on the PER-WORD grammar path the
            # ``reconstruction`` slot is the D3 reverse(S)->table->
            # vs-complete-unmasked-input continuous reconstruction
            # (differentiable; the trainable per-word IR objective),
            # REPLACING the interim P-space_role ``compute_masked`` masked-LM.
            # The whole-slab / non-grammar (``_per_word_enabled=False``)
            # path keeps ``compute_masked``, band-aware via the
            # ``_masked_event_loss`` seam (silent-band wiring fix,
            # 2026-07-04). ``_d3_active`` / ``_d3_word_metric`` are
            # instrumentation read by the end-to-end objective probe
            # (which loss term is active + the reported 0/1 metric);
            # never on the training-critical path.
            self._d3_active = False
            self._d3_word_metric = None
            _isp = self.inputSpace
            _per_word = (_isp is not None
                         and getattr(_isp, "_per_word_enabled", False))
            d3_loss, d3_metric = (self._d3_reconstruction_loss()
                                  if _per_word else (None, None))
            if d3_loss is not None:
                # D3: the continuous reconstruction is the trainable
                # signal into the existing ``reconstruction`` slot;
                # the word-level 0/1 distance is reported-only.
                lossIn = d3_loss
                self._d3_active = True
                self._d3_word_metric = d3_metric
            elif (mask_pos is not None and pre_mask is not None
                    and pred_full is not None):
                # Band-aware seam: percept-layout where/when scales (silent-band wiring fix, 2026-07-04).
                lossIn = self._masked_event_loss(pred_full, pre_mask,
                                                 mask_pos)
            else:
                lossIn = torch.zeros((), device=TheDevice.get())
                # 5b fail-loud: a dead reconstruction channel must announce itself (warn-once, names the gate).
                _missing = ", ".join(
                    n for n, v in (("mask", mask_pos), ("target", pre_mask),
                                   ("pred", pred_full)) if v is None)
                _cb = getattr(getattr(self.perceptualSpace, 'subspace',
                                      None), 'what', None)
                self._warn_zeroed_channel(
                    "reconstruction_zeroed",
                    f"reconstruction loss zeroed: no D3 (per_word="
                    f"{bool(_per_word)}) and masked-LM inputs missing "
                    f"({_missing}); percept .what={type(_cb).__name__}, "
                    f"pred_full shape="
                    f"{tuple(pred_full.shape) if torch.is_tensor(pred_full) else None}")
            TheError.add(
                "reconstruction", lossIn,
                weight=1.0,
                space="InputSpace", category="reconstruction",
            )

            # Reverse-pass input reconstruction (OS→CS→PS→IS): run the
            # restored reverse pipeline on the head output and compare
            # the reconstructed input against the forward InputSpace
            # event. Blended via ``reconstruction_scale`` (the design's
            # reconRatio model, doc/Architecture.md). Single-input
            # per-space reverse: the round-trip is the local inverse, so
            # this is an approximate reconstruction through the averaged
            # loops -- guarded so a shape edge case in any config
            # degrades to a zero contribution rather than breaking the
            # step.
            lossRev = torch.zeros((), device=TheDevice.get())
            # Dedupe: on D3 lossIn IS the reverse objective; train skips the double count (doc/plans/2026-07-03-reconstruction-fidelity-execution.md); eval totals still include the reverse term.
            _rev_dedupe = bool(self._d3_active) and train
            try:
                if forwardInput is not None and not _rev_dedupe:
                    # C3 (spec sec 7): reconstruction is UNCONDITIONALLY
                    # from concepts. The ``<reconstruct>`` enum
                    # (none/symbols/concepts/both) was retired in A1, so
                    # the old ``rev_mode in ('concepts','both')`` gate is
                    # gone -- whenever reconstruction fires (the
                    # ``reconstruction_scale`` weighting below gates it),
                    # the reverse pass is always concepts-seeded from the
                    # terminal ConceptualSpace ShortTermMemory snapshot.
                    #
                    # Stage 1.F substrate refactor (doc/plans/
                    # 2026-05-26-two-loop-pi-sigma-substrate.md): the
                    # per-stage ``_cs_cache`` forward capture is retired.
                    # The reverse-path seed comes from the canonical
                    # ConceptualSpace ShortTermMemory snapshot.
                    #
                    # 2026-05-28 fix: mode-dependent seed shape.
                    # In PARALLEL the STM is the [B, N, D] slab --
                    # every position is its own slot and the
                    # reverse pipeline should walk them all back.
                    # In SERIAL the STM accumulates per-word ideas and
                    # reduces a simple sentence to a SINGLE S at slot 0
                    # (newest-at-slot-0 convention; _stm_reduce_to_single_S
                    # leaves the root there). ``snap[:, :1, :]`` is that
                    # reduced idea; the prior ``snap[:, -1:, :]`` read the
                    # OLDEST/empty padding slot (slot 0 holds the single S, so
                    # for a depth-1 reduced sentence slot -1 is empty) -> the
                    # reverse decoded an empty seed.
                    rev_sub = None
                    rev_ev = None
                    # 2026-07-05 serial plan Task 2 (Method-1 routing): at
                    # EVAL the SERIAL decode consumes the STORED-derivation
                    # LEAVES replay (_reverse_method1_leaves stages the radix
                    # render thunk on the per-word percept leaves), so
                    # reconstruct_data reads the exact derivation surface, not
                    # the single-slot tensor arm. Method-1 is the exact
                    # TEACHER -- by construction, no training needed. The
                    # tensor arm stays the explicit debug fallback
                    # (serial_tensor_reverse_debug -- the --scaffold analogue);
                    # TRAIN paths (the D3 reverse-from-S student) are untouched.
                    if (not train and bool(getattr(self, 'serial', False))
                            and not getattr(
                                self, 'serial_tensor_reverse_debug', False)):
                        rev_ev = self._reverse_method1_leaves()
                    if rev_ev is None:
                        terminal_idea = self._reconstruction_seed()
                        # Method-2 reverse-reduce (serial plan Task 4): on the
                        # FREE-derivation decode (reconstruct_from_idea, eval),
                        # un-fold the collapsed root back into per-word ideas by
                        # walking the recorded fold steps backward -- each step
                        # the chosen op's basis-threaded reverse, the
                        # codebook-walk recommender (a LOOKUP that reconstitutes
                        # the operand pair, not a subtraction). Falls through to
                        # the single-slot seed when no trace/basis exists.
                        if (terminal_idea is not None and not train
                                and bool(getattr(
                                    self, 'reconstruct_from_idea', False))):
                            # Invalidate any prior batch's render-priority
                            # slab; the un-fold re-stashes it ONLY on the
                            # word-rows path (review finding: an ungated
                            # stash swapped the NON-wordstore ceiling's
                            # render source).
                            _psp = getattr(self, 'perceptualSpace', None)
                            if _psp is not None:
                                object.__setattr__(
                                    _psp, '_unfold_recovered_slab', None)
                            unfolded = self._reverse_reduce_unfold(
                                terminal_idea[:, 0, :])
                            if unfolded is not None:
                                terminal_idea = unfolded   # [B, N_words, D_c]
                        if terminal_idea is not None:
                            cs = self.conceptualSpace
                            cs.subspace.set_event(terminal_idea)
                            rev_sub = self.reverse(
                                cs.subspace)
                        rev_ev = (rev_sub.materialize()
                                  if rev_sub is not None
                                  and hasattr(rev_sub, 'materialize')
                                  else None)
                    if (not train and rev_ev is not None
                            and torch.is_tensor(rev_ev)):
                        inputDataPred = rev_ev.detach()
                    fwd_ev = (forwardInput.materialize()
                              if hasattr(forwardInput, 'materialize')
                              else forwardInput)
                    if (rev_ev is not None and fwd_ev is not None
                            and rev_ev.dim() == fwd_ev.dim()
                            and rev_ev.dim() == 3):
                        # Band-aware seam: the input event's where/when widths, not ModelLoss's (0,0) OutputSpace band.
                        lossRev = self._reverse_event_loss(rev_ev, fwd_ev)
            except Exception:
                # Reverse round-trip is approximate through averaged
                # loops; never let a reconstruction edge case stop the
                # training step.
                lossRev = torch.zeros((), device=TheDevice.get())
            if not _rev_dedupe:
                TheError.add(
                    "reconstruction_reverse", lossRev,
                    weight=float(getattr(self.loss, 'reconstruction_scale',
                                         0.0) or 0.0),
                    space="InputSpace", category="reconstruction",
                )

            # C3 (spec sec 7): the legacy forward-only C/S reconstruction
            # branch (gated on the retired ``<reconstruct>`` enum) was a
            # no-op ``pass`` -- its Stage-3 reinstatement never landed and
            # the enum is gone. Removed. The concepts-seeded reverse pass
            # above (``reconstruction_reverse``) is now the unconditional
            # reconstruction carrier; the IR P-space_role ``reconstruction`` loss
            # remains the active forward gradient source. (No behaviour
            # moved out of the deleted branch -- it never executed any.)

            # JOINT mode: compute SBOW embedding loss
            sbow = None
            if train:
                sbow = self.trainEmbeddings(('JOINT'), sentenceIdx, split)
                # Perceptual SBOW: when lexer=byte, train percept vectors
                # via leave-one-out centroid prediction
                if getattr(self, 'lexer', None) in ('byte', 'bytes'):
                    psbow = self.perceptual_sbow_loss()
                    if psbow is not None:
                        sbow = psbow if sbow is None else sbow + psbow
                if sbow is not None:
                    TheError.add(
                        "embedding_sbow", sbow,
                        weight=self.loss.embedding_scale,
                        space="SymbolSpace", category="embedding",
                    )

            # Inter-sentence ARMA(p, q) loss term -- model-owned
            # (``_discourse_arma_loss``): predicts ``s_hat_t`` from the
            # lagged reps/residuals, returns the per-batch MSE, and
            # commits the new rep + residual into the rings (vectorized,
            # sync-free). Cold-start rows return ``None``. Computed here,
            # post-body / pre-backward, so the term trains the predictor.
            # Skip on the explore trial (pass B): _discourse_arma_loss both
            # trains the inter-predictor AND commits the sentence rep/residual
            # into the persistent ARMA rings (disc.observe). Re-running it on a
            # sentence pass A already observed would double-push the rings and
            # corrupt the lagged history the next batch reads.
            arma_loss = (self._discourse_arma_loss()
                         if (train and not exploration_trial) else None)

            # Intra-sentence prediction loss term (Task 3, STM serial/
            # parallel modes) -- ``ConceptualSpace.forward`` ran the
            # in-STM predictor predict-then-perceive over the per-word
            # steps and accumulated ``L_intra = MSE(prediction, perceived)``
            # live (grad-bearing) on the conceptual space. Consume the
            # per-batch mean here, post-body / pre-backward (mirroring the
            # ARMA term), so the term trains the intra-sentence predictor.
            # ``consume_intra_loss`` resets the accumulator; it returns
            # ``None`` when nothing was accumulated (eval, weight off, or
            # an all-degenerate sentence).
            intra_loss = (self.conceptualSpace.consume_intra_loss()
                          if train else None)

            # Inter-sentence end-state prediction loss term (Task 8, plan
            # §9) -- the sentence-boundary hook ran the inter-level
            # predictor + scored it against the arriving end-state,
            # accumulating ``L_inter`` live on the discourse layer. Consume
            # the per-sentence mean here, post-body / pre-backward (mirroring
            # the ARMA + intra terms). ``None`` when the discourse layer is
            # absent (absolute-only no-op), eval-time, weight off, or no
            # scored sentence this batch.
            inter_loss = self._discourse_inter_loss() if train else None
            # InfoNCE next-idea contrastive term (the discourse layer's second
            # accumulator; populated during the forward boundary observe).
            inter_contrastive = None
            if train and self.symbolSpace is not None:
                _disc = getattr(self.symbolSpace, "discourse", None)
                if _disc is not None and hasattr(
                        _disc, "consume_inter_contrastive_loss"):
                    inter_contrastive = _disc.consume_inter_contrastive_loss()

            # Trial-split: on a PURE next-idea PREDICTION trial, zero the
            # reconstruction + auxiliary terms so the next-idea signal (inter
            # MSE + InfoNCE contrastive) is the SOLE gradient. prediction_trial_
            # ratio == 0 -> trial_mode is always "reconstruct" -> this never
            # fires -> byte-identical. Done post-forward so the forward stays
            # mode-independent (no Dynamo recompiles).
            if train and trial_mode == "predict":
                _z = torch.zeros((), device=TheDevice.get())
                lossOut, lossIn = _z, _z
                sbow = None
                lossRev = None
                arma_loss = None
                intra_loss = None

            totalLoss = self.loss.total(lossOut, lossIn, sbow)
            _cscale = float(getattr(
                self.loss, "conceptual_similarity_scale", 0.0) or 0.0)
            if train and not self.serial and _cscale > 0.0:
                csbow = self.conceptual_sbow_loss()
                if csbow is not None:
                    totalLoss = totalLoss + _cscale * csbow
                    TheError.add(
                        "conceptual_sbow", csbow, weight=_cscale,
                        space="ConceptualSpace", category="embedding",
                    )
            # Rank-ordered soft-L0 that keeps concept definitions compact
            # (snap contract sec 5). Default lambda 0.0 -- byte-identical off.
            _dss = float(getattr(
                self.loss, "definition_sparsity_scale", 0.0) or 0.0)
            if train and _dss > 0.0 and self.conceptualSpace is not None:
                defsp = self.conceptualSpace.definition_sparsity_loss(lam=_dss)
                if defsp is not None:
                    totalLoss = totalLoss + defsp
                    TheError.add(
                        "definition_sparsity", defsp, weight=_dss,
                        space="ConceptualSpace", category="reg")
            if lossRev is not None:
                totalLoss = totalLoss + (
                    float(getattr(self.loss, 'reconstruction_scale', 0.0)
                          or 0.0) * lossRev)
            if arma_loss is not None:
                totalLoss = totalLoss + self.arma_scale * arma_loss
                TheError.add(
                    "arma", arma_loss,
                    weight=self.arma_scale,
                    space="DiscourseSpace", category="discourse",
                )
            if intra_loss is not None:
                totalLoss = (totalLoss
                             + self.conceptualSpace.intra_loss_weight
                             * intra_loss)
                TheError.add(
                    "intra", intra_loss,
                    weight=self.conceptualSpace.intra_loss_weight,
                    space="ConceptualSpace", category="intra",
                )
            if inter_loss is not None:
                totalLoss = (totalLoss
                             + self.inter_loss_weight * inter_loss)
                TheError.add(
                    "inter", inter_loss,
                    weight=self.inter_loss_weight,
                    space="DiscourseSpace", category="inter",
                )
            if inter_contrastive is not None:
                totalLoss = (totalLoss
                             + self.inter_contrastive_weight * inter_contrastive)
                TheError.add(
                    "inter_contrastive", inter_contrastive,
                    weight=self.inter_contrastive_weight,
                    space="DiscourseSpace", category="inter",
                )
            # Phase 3: every stage's WholeSpace.forward wrote its
            # auxiliary terms to ``vspace.errors``; ``copy_context`` shares
            # a single Error instance through the pipeline, so the terminal
            # OutputSpace subspace now carries every term.
            if hasattr(self, 'outputSpace'):
                pipeline_errors = self.outputSpace.subspace.errors
                aux_total = pipeline_errors.total()
                if aux_total is not None:
                    totalLoss = totalLoss + aux_total
                    for name, tensor, weight, space, category in pipeline_errors.terms():
                        TheError.add(name, tensor, weight=weight,
                                     space=space, category=category)
                pipeline_errors.clear()

            # Phase C: truth-grounded answer-policy loss -- trains the soft
            # reasoning route (the InterveningIdeaGenerator query head + the
            # GlobalAttention scorer) against store-derived (A, B, gold)
            # examples. The hard deduction is never differentiated (the bridge
            # mask is detached). Default weight 0.0 -> skipped -> byte-identical;
            # also a no-op when the generator/store is absent.
            if train and float(
                    getattr(self, "answer_loss_weight", 0.0) or 0.0) > 0.0:
                try:
                    a_loss = self._answer_policy_loss()
                except Exception:
                    a_loss = None          # a reasoning hiccup must not abort training
                if a_loss is not None:
                    totalLoss = totalLoss + self.answer_loss_weight * a_loss
                    TheError.add(
                        "answer", a_loss,
                        weight=self.answer_loss_weight,
                        space="SymbolSpace", category="policy")

            # Step 2: the next-idea POLICY loss -- trains the {arma, retrieval,
            # deduction} blend to predict the observed next end-state root.
            # Default weight 0.0 -> skipped -> byte-identical; no-op without a
            # >=2-entry discourse chain.
            # Thinking Kernel next-op policy loss (§12.6): behavior-clones the
            # NextOpPolicy head on grounded kernel traces generated from the
            # reasoning store (the teacher runs with materialize=False, so
            # trace generation never writes LTM). Default weight 0.0 ->
            # skipped -> byte-identical; also a no-op without a head / store /
            # 2-hop chains.
            if train and float(
                    getattr(self, "thinking_loss_weight", 0.0) or 0.0) > 0.0:
                try:
                    tk_loss = self._thinking_policy_loss()
                except Exception:
                    tk_loss = None         # a kernel hiccup must not abort training
                if tk_loss is not None:
                    totalLoss = totalLoss + self.thinking_loss_weight * tk_loss
                    TheError.add(
                        "thinking", tk_loss,
                        weight=self.thinking_loss_weight,
                        space="SymbolSpace", category="policy")

            if train and float(
                    getattr(self, "predict_next_loss_weight", 0.0) or 0.0) > 0.0:
                try:
                    pn_loss = self._predict_next_loss()
                except Exception:
                    pn_loss = None
                if pn_loss is not None:
                    totalLoss = totalLoss + self.predict_next_loss_weight * pn_loss
                    TheError.add(
                        "predict_next", pn_loss,
                        weight=self.predict_next_loss_weight,
                        space="SymbolSpace", category="policy")

            # Method-1 -> Method-2 leaf distillation (snap design doc step
            # 3): the exact-leaf teacher supervises a from-root decoder so
            # the collapsed root stays SEPARABLE per sentence. Default
            # weight 0.0 -> skipped -> byte-identical.
            if train and float(
                    getattr(self, "leaf_distill_weight", 0.0) or 0.0) > 0.0:
                try:
                    ld_loss = self._leaf_distill_loss()
                except Exception:
                    ld_loss = None     # distillation must not abort training
                if ld_loss is not None:
                    totalLoss = totalLoss + self.leaf_distill_weight * ld_loss
                    TheError.add(
                        "leaf_distill", ld_loss,
                        weight=self.leaf_distill_weight,
                        space="ConceptualSpace", category="reconstruction")
                    # The head is built lazily on first use — hand its
                    # params to the LIVE optimizer once (getOptimizer ran
                    # before the head existed).
                    if (optimizer is not None
                            and getattr(self, "_leaf_distill_head_fresh",
                                        False)):
                        self._leaf_distill_head_fresh = False
                        optimizer.add_param_group({"params": list(
                            self._leaf_distill_head_module.parameters())})

            # Truth-modulated loss: delegated to SymbolSpace since the
            # TruthLayer lives there.  SymbolSpace handles the empty-store
            # guard internally; we only gate on ``train``.  The falsity
            # penalty operand is the last cached symbol activation --
            # stored truths are also recorded from symbol space, so both
            # sides of the disjunction live in the basis's native space.
            if train and self.symbolSpace is not None:
                symbol_acts = None
                if hasattr(self, 'symbol_states') and self.symbol_states:
                    symbol_acts = self.symbol_states[-1]
                totalLoss = self.symbolSpace.truth_modulated_loss(
                    totalLoss,
                    symbolic_space=self.wholeSpace,
                    symbol_acts=symbol_acts,
                    universality_score=getattr(self, '_universality_score', None),
                    luminosity_weight=getattr(self, 'luminosity_weight', 0.1),
                    universality_weight=getattr(self, 'universality_weight', 0.1),
                    truth_loss_weight=getattr(self, 'truth_loss_weight', 0.0),
                    allow_excluded_middle=getattr(self, 'allow_excluded_middle', 1),
                    allow_contradiction=getattr(self, 'allow_contradiction', 0),
                    model=self,
                )
                # Gate-L1 sparsity penalty on LiftLayer / LowerLayer
                # raw_gate parameters. Pulls unused singular-component
                # multipliers toward zero so each rule converges to a
                # low-rank slice of its host operator. Default lambda
                # is 0.0 -- no penalty unless a config opts in.
                gate_l1 = self.symbolSpace.gate_l1_loss(
                    lam=getattr(self, 'gate_l1_lambda', 0.0))
                if gate_l1 is not None:
                    totalLoss = totalLoss + gate_l1
                    TheError.add(
                        "gate_l1", gate_l1,
                        weight=getattr(self, 'gate_l1_lambda', 0.0),
                        space="SymbolSpace", category="reg")

                # Stage 3 cleanup: the chart's sparse-MoE load-balance
                # bookkeeping retired with the chart itself. The
                # <loadBalanceWeight> knob stands by for any future
                # signal-router rule load-balancing; no consumer yet.

            # Snapshot the breakdown before the backward pass so later
            # calls to TheError.covariance() can see it in the history
            # even if the step is aborted by a non-finite detector below.
            # snapshot()'s per-term .item() is a cudaMemcpyDtoH (breaks
            # the brick CUDA-graph-capture contract, test_brick_no_sync)
            # and only feeds the diagnostic covariance() API (no
            # training-loop consumer). Gate behind MODEL_DEBUG, like the
            # finite-loss guard just below.
            if _util.MODEL_DEBUG:
                TheError.snapshot()

        # Per-batch finite-loss guard is a GPU sync (.all() materializes).
        # Gate it behind MODEL_DEBUG so production training pays no per-batch
        # sync; failures still surface via NaN gradients downstream.
        if _util.MODEL_DEBUG and not torch.isfinite(totalLoss).all():
            def _loss_value(name, value):
                if value is None:
                    return f"{name}=None"
                if isinstance(value, torch.Tensor):
                    finite = torch.isfinite(value).all().item()
                    if value.numel() == 1:
                        return f"{name}={value.detach().item()} finite={finite}"
                    return f"{name}=shape{tuple(value.shape)} finite={finite}"
                return f"{name}={value}"

            details = ", ".join([
                _loss_value("lossOut", lossOut),
                _loss_value("lossIn", lossIn),
                _loss_value("sbow", sbow),
                _loss_value("arma", arma_loss),
                _loss_value("symbol", symbol_loss),
                _loss_value("total", totalLoss),
            ])
            raise FloatingPointError(
                f"Non-finite total loss in {self.name}.runBatch("
                f"split={split}, batch={batchNum}): {details}"
            )

        # Plain f-string without totalLoss -- interpolating a tensor's
        # __format__ forces a device sync every batch.  The epoch summary
        # still carries per-epoch losses; per-batch loss is available via
        # MODEL_DEBUG if needed.
        #
        # Per-batch wall-clock: time.perf_counter() reads the host
        # monotonic clock (no GPU sync). The first batch (before
        # ``self._last_batch_time`` is set) prints without a delta --
        # that "compile + warm-up" tick is heavily front-loaded and
        # measuring it as "delta from epoch start" would be misleading.
        # Subsequent batches print ``(Δ=X.XXXs)`` where X is the
        # wall-clock elapsed since the last batch's report.
        import time as _time
        now = _time.perf_counter()
        last = getattr(self, '_last_batch_time', None)
        # Optional percent-complete suffix when the cursor populated
        # ``progress`` (set from ``SentenceStreamDataset.progress()``
        # in ``runEpoch``). Direct ``runBatch`` callers (tests,
        # inference) pass progress=None and get the bare timing line.
        pct = ("" if progress is None
               else f", {min(progress, 1.0) * 100.0:.2f}%")
        if last is None:
            TheMessage(f"batch = {batchNum} (warm-up{pct})")
        else:
            TheMessage(f"batch = {batchNum} (Δ={now - last:.3f}s{pct})")
        self._last_batch_time = now

        # Inductor / Dynamo recompile detector. Per-shape recompiles can
        # show up as wall-clock variance on otherwise identical batches;
        # this prints a one-line delta whenever Dynamo records new
        # frame events (compiles, recompiles, graph breaks). Defensive
        # try/except: torch._dynamo counters API has changed across
        # versions, and on MODEL_COMPILE=none there's nothing to count.
        try:
            from torch._dynamo.utils import counters as _dyn_counters
            frames = _dyn_counters.get('frames', {})
            rc_now = sum(int(v) for v in frames.values())
            rc_last = getattr(self, '_last_recompile_count', None)
            if rc_last is None:
                self._last_recompile_count = rc_now
            elif rc_now > rc_last:
                delta = rc_now - rc_last
                # Show the breakdown (e.g. ok=N, recompile=M) so a
                # stable steady-state with occasional recompiles is
                # visible at a glance.
                detail = ", ".join(f"{k}={v}" for k, v in sorted(frames.items()))
                TheMessage(f"  [compile] dynamo +{delta} frame events (total {rc_now}; {detail})")
                self._last_recompile_count = rc_now
        except Exception:
            pass

        if train:
            if amp_scaler is not None:
                # fp16 on CUDA: scale grads to avoid underflow, then unscale
                # inside scaler.step() before the actual optimizer update.
                amp_scaler.scale(totalLoss).backward()
                self._assert_finite_train_state("after backward")
                if self.ergodic:
                    self.paramUpdate()
                # CUDA fp16 GradScaler owns its fused unscale/non-finite
                # check. In auto mode the optimizer wrapper's guard is also
                # disabled, preserving the brick's zero-D2H contract.
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                totalLoss.backward()
                self._assert_finite_train_state("after backward")
                preflight_finite_gradients(
                    optimizer, self.named_parameters(),
                    cache_for_step=hasattr(
                        optimizer, "_step_without_finite_preflight"))
                if self.ergodic:
                    self.paramUpdate()
                optimizer.step()
            self._assert_finite_train_state("after optimizer.step")
            self._flush_partspace_promotions(optimizer=optimizer)
            self._clamp_symbolic_codebook()
            self._normalize_conceptual_codebooks()
            # 2026-05-28: enforce the |W| <= 1 invariant on the
            # Embedding (Lexicon) by re-projecting rows onto the unit
            # ball after each optimizer step. Matches the SBOW
            # pre-training pattern at bin/embed.py:1976. Without this,
            # JOINT training drifts Embedding rows beyond [-1, 1]
            # (measured: |W|.max ~ 1.54 after 200 epochs on XOR_exact),
            # which breaks the nearest-Embedding reverse decode -- the
            # bounded recon vector from pi.reverse cannot reach the
            # unbounded target rows.
            self._normalize_perceptual_embedding()
            self._advance_codebook_parameter_versions()
            # Don't count the explore trial (pass B): the step counter drives
            # the periodic-checkpoint cadence, which should count sentences,
            # not the two trials per sentence.
            if not exploration_trial:
                self._training_step_count = (
                    int(getattr(self, "_training_step_count", 0) or 0) + 1
                )
        else:
            # The eager lexical stem may discover promotions during no-grad
            # inference too. The completed forward is its graph-safe boundary;
            # install now so the next request can use the promoted row.
            self._flush_partspace_promotions(optimizer=None)

        result = self.BatchResult(
            outputPred=outputDataPred,
            symbols=symbols,
            lossOut=lossOut,
            lossIn=lossIn,
            inputPred=inputDataPred,
            forwardInput=forwardInput,
        )
        # Pure compute brick: no Reset, no truth-layer compact, no host
        # sync inside runBatch. The outer doc-streaming loop in runEpoch
        # (or any per-tick driver) is responsible for:
        #   * Hard reset (per-row, on document boundary) via
        #     ``BasicModel.dispatch_per_row_reset(hard_eos_list)``.
        #   * Soft reset (per-row, on grammar sentence completion) via
        #     ``symbolSpace.drain_sentence_completed()`` →
        #     ``symbolSpace.soft_reset(b)``.
        #   * ``truth_layer.compact()`` (one host sync per tick, kept
        #     outside the brick).
        # See doc/plans/2026-04-26-rolling-cursor-doc-streaming-handoff.md.

        # Memory-leak diagnostics (perf-notes/08-*). Three independently
        # gated probes; each is a no-op without its env var.
        if os.environ.get("BASIC_PROFILE_DIAG"):
            try:
                ss_diag = self.symbolSpace
                tl_diag = getattr(ss_diag, 'truth_layer', None) if ss_diag is not None else None
                if tl_diag is not None and hasattr(tl_diag, 'count'):
                    tl_count_diag = int(tl_diag.count.item())
                    tl_pending_diag = int(getattr(tl_diag, '_pending_count', 0))
                else:
                    tl_count_diag = 0
                    tl_pending_diag = 0
                if torch.cuda.is_available():
                    cuda_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    cuda_max_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                else:
                    cuda_alloc_mb = 0.0
                    cuda_max_mb = 0.0
                opt_state_numel = 0
                if optimizer is not None:
                    for st in optimizer.state.values():
                        for v in st.values():
                            if torch.is_tensor(v):
                                opt_state_numel += v.numel()
                word_lens_diag = {}
                for sp in self.spaces:
                    sub = getattr(sp, 'subspace', None)
                    if sub is not None and hasattr(sub, 'word'):
                        word_lens_diag[sp.__class__.__name__] = len(sub.word)
                pair_counts_diag = {}
                for sp in self.spaces:
                    cl = getattr(sp, 'chunkLayer', None) or getattr(sp, 'chunk_layer', None)
                    if cl is not None:
                        pair_counts_diag[sp.__class__.__name__] = (
                            len(getattr(cl, '_pair_counts', {})),
                            len(getattr(cl, '_unigram_counts', {})),
                        )
                TheMessage(
                    f"[diag] batch={batchNum} tl={tl_count_diag}/{tl_pending_diag} "
                    f"cuda_alloc_mb={cuda_alloc_mb:.1f} cuda_max_mb={cuda_max_mb:.1f} "
                    f"opt_state_numel={opt_state_numel} word_lens={word_lens_diag} "
                    f"chunk_pair_uni={pair_counts_diag}"
                )
            except Exception as _diag_exc:
                TheMessage(f"[diag] batch={batchNum} diag_error={_diag_exc!r}")

        if os.environ.get("BASIC_PROFILE_LEAK") and torch.cuda.is_available():
            if batchNum >= 96 and not getattr(self, "_leak_recording", False):
                torch.cuda.memory._record_memory_history(
                    enabled="all", context="all", stacks="python",
                    max_entries=400_000)
                self._leak_recording = True
                TheMessage(f"[leak] start recording at batch {batchNum}")
            elif batchNum >= 160 and getattr(self, "_leak_recording", False):
                _leak_dir = os.path.expanduser("~/WikiOracle/basicmodel/perf-notes")
                os.makedirs(_leak_dir, exist_ok=True)
                out = os.path.join(_leak_dir, "08-leak-snapshot.pkl")
                try:
                    torch.cuda.memory._dump_snapshot(out)
                    TheMessage(f"[leak] dumped snapshot to {out} at batch {batchNum}")
                except Exception as _leak_exc:
                    TheMessage(f"[leak] dump failed: {_leak_exc!r}")
                torch.cuda.memory._record_memory_history(enabled=None)
                self._leak_recording = False

        if os.environ.get("BASIC_PROFILE_TENSORS") and torch.cuda.is_available():
            import gc as _gc, collections as _coll
            counts = _coll.Counter()
            for obj in _gc.get_objects():
                try:
                    if isinstance(obj, torch.Tensor) and obj.device.type == "cuda":
                        counts[(tuple(obj.shape), str(obj.dtype))] += 1
                except Exception:
                    continue
            prev = getattr(self, "_leak_prev_counts", None)
            if prev is not None:
                keys = set(counts) | set(prev)
                deltas = sorted(
                    ((k, counts.get(k, 0) - prev.get(k, 0)) for k in keys
                     if counts.get(k, 0) != prev.get(k, 0)),
                    key=lambda kv: -abs(kv[1]))
                top = ", ".join(f"{k[0]}/{k[1]}:{d:+d}" for k, d in deltas[:8])
                TheMessage(f"[tensors] batch={batchNum} top deltas: {top}")
            self._leak_prev_counts = dict(counts)

        # Clear per-batch IR scratch so the next batch's
        # create_ir_mask starts from a clean slate (no stale mask /
        # pre-mask tensor pinned in GPU memory).
        self._ir_mask_positions = None
        self._ir_pre_mask_input = None

        self.End()
        return result, batchNum

    def dispatch_per_row_reset(self, hard_eos):
        """Fire per-row hard Reset on rows whose ``hard_eos[b]`` is True.

        Called by the outer doc-streaming loop after ``runBatch`` returns.
        ``hard_eos`` is the host-side ``list[bool]`` from
        ``SentenceStreamDataset.next_tick``; rows with True consumed the
        last bytes of their current document this tick and need a full
        per-row state wipe before the next document starts.

        Fast-path: when every row is True (the legacy DataLoader contract,
        and also the common cursor case where all rows finish a doc on
        the same tick), collapse to a single global ``space.Reset()`` per
        space — one call instead of B. That keeps the dispatch overhead
        at parity with the pre-handoff legacy path while preserving the
        per-row capability for the partial-eos cursor case.

        Otherwise, cascades ``space.Reset(batch=b, hard=True)`` over
        every Reset-capable space. Every Reset accepts the per-row
        signature now (the §8d legacy zero-arg fallback was removed).
        """
        if not hard_eos:
            return
        if all(hard_eos):
            # Legacy-parity hot path (L6 in the handoff): a single
            # global Reset per space when every row finishes a doc on
            # the same tick (the common DataLoader case). Keeps the
            # dispatch overhead at parity with the pre-handoff path
            # while preserving the per-row capability for partial-eos.
            for space in self.spaces:
                if hasattr(space, 'Reset'):
                    space.Reset()
        else:
            for b, done in enumerate(hard_eos):
                if not done:
                    continue
                for space in self.spaces:
                    if hasattr(space, 'Reset'):
                        space.Reset(batch=b, hard=True)
        # Clear the per-row eos diagnostic. ``_end_of_stream`` is a
        # host-side ``list[bool]`` post-§8c; resize lazily if the cursor
        # batch grew beyond the AR forward()'s previous sizing.
        eos = self.inputSpace._end_of_stream
        if len(eos) < len(hard_eos):
            eos.extend([False] * (len(hard_eos) - len(eos)))
        for b, done in enumerate(hard_eos):
            if done:
                eos[b] = False

    def dispatch_soft_reset(self):
        """Drain ``symbolSpace._sentence_completed`` and fire per-row soft reset.

        Called by the outer doc-streaming loop after ``runBatch`` returns
        (and *after* ``dispatch_per_row_reset`` so a hard-reset row's soft
        signal is dropped — hard subsumes soft).
        """
        ss = self.symbolSpace
        if ss is None or not hasattr(ss, 'drain_sentence_completed'):
            return
        completed = ss.drain_sentence_completed()
        for b in completed:
            ss.soft_reset(batch=b)

    def post_tick_compact(self):
        """Run post-tick host work: truth-layer compaction.

        Lives outside the compute brick so the brick body remains
        sync-free. Called once per tick by the outer doc-streaming loop
        after ``runBatch`` returns.

        Also severs any cross-batch autograd graph carried by persistent
        state (``SymbolSpace._disc_pred``, ``_last_svo``; Basis transient
        activations in ``_active_payload`` / non-Parameter ``W``; all
        floating-point registered buffers). In-place writes during the
        brick's forward leave persistent attributes wired to that
        batch's autograd graph; once ``backward()`` runs and the saved
        tensors are freed, the next batch's forward re-reads the same
        attributes and the next ``backward()`` walks into freed nodes
        ("Trying to backward through the graph a second time"). This
        hook runs in eager Python every tick after ``backward()``, so
        ``self.modules()`` iteration is safe here (it would graph-break
        if placed inside a traced reset path).
        """
        ss = self.symbolSpace
        if ss is not None:
            tl = getattr(ss, 'truth_layer', None)
            if tl is not None and hasattr(tl, 'compact'):
                # Gold ingestion (store_truths drops truth_criterion to 0
                # to capture ALL provided truths): drop the compact bar the
                # same way -- unity-analysis magnitudes are small by
                # construction, unlike the retired body-snap norms.
                _min_trust = 0.0 if float(getattr(
                    self.wholeSpace, "truth_criterion", 1.0)) == 0.0 else 0.5
                tl.compact(min_trust=_min_trust)
        self._detach_persistent_state()

    def _detach_persistent_state(self):
        """Walk every submodule under this model and detach the floats
        that carry cross-batch autograd edges. See ``post_tick_compact``
        for the full rationale.

        Touches:
          * registered buffers (``self.buffers()``)
          * Basis ``_active_payload`` (transient activations stored when
            ``W`` is a learned Parameter)
          * Basis ``W`` (when ``W`` is a plain tensor, i.e. not an
            ``nn.Parameter``)
          * Model-level cached state tensors that don't live as buffers
            (``SymbolSpace._disc_pred`` / ``_disc_conf``,
            ``InputSpace._ar_embedded``, ``inputSpace._embedded_input``,
            spaces' ``_embedded_input``).
        """
        # 1. All registered FP buffers in the model tree.
        for buf in self.buffers():
            if buf.is_floating_point():
                buf.detach_()
        # 2. Plain-tensor W on every submodule (the ``_active_payload``
        # shadow was retired Stage 4 of
        # doc/plans/2026-05-21-active-payload-retirement.md; per-batch
        # content for codebook-bearing slots now reconstructs via
        # ``SubSpace.materialize`` from prototype + selection).
        for mod in self.modules():
            w = getattr(mod, 'W', None)
            if (w is not None and torch.is_tensor(w)
                    and not isinstance(w, nn.Parameter)
                    and w.is_floating_point()):
                mod.W = w.detach()
        # 3. Known plain-attribute tensor caches that ride across batches.
        ss = self.symbolSpace
        if ss is not None:
            if getattr(ss, '_disc_pred', None) is not None:
                ss._disc_pred = ss._disc_pred.detach()
            if getattr(ss, '_disc_conf', None) is not None:
                ss._disc_conf = ss._disc_conf.detach()
        for sp_attr in ("inputSpace", "perceptualSpace",
                        "conceptualSpace", "wholeSpace", "outputSpace"):
            sp = getattr(self, sp_attr, None)
            if sp is None:
                continue
            for tn in ("_ar_embedded", "_embedded_input",
                       "_cached_embedding"):
                t = getattr(sp, tn, None)
                if t is not None and torch.is_tensor(t) and t.is_floating_point():
                    setattr(sp, tn, t.detach())
        # 4. Serial-arc carriers -- deliberately PLAIN attributes (kept out
        # of buffers/traces), so steps 1-3 miss them; the next batch's
        # forward re-reads each one, chaining the consumed (already
        # optimizer-stepped) graph into its loss ("[1016] at version 1"
        # backward crash, the ladder5 relaunch 2026-07-13).
        def _sever(obj, name):
            t = getattr(obj, name, None)
            if (t is not None and torch.is_tensor(t)
                    and t.is_floating_point() and t.grad_fn is not None):
                object.__setattr__(obj, name, t.detach())

        def _sever_dict(d):
            if isinstance(d, dict):
                for k, v in list(d.items()):
                    if torch.is_tensor(v) and v.grad_fn is not None:
                        d[k] = v.detach()

        for mod in self.modules():
            for tn in ("_live_buffer", "_last_output", "_last_root_state",
                       "_intent_boosts", "_stage0_recon_loss"):
                _sever(mod, tn)
            rs = getattr(mod, "routing_state", None)
            if rs is not None:
                _sever(rs, "rule_probs")
            _sever_dict(getattr(mod, "_bind_context", None))
            # Host-layer registry values are host wrappers, not children.
            reg = getattr(mod, "_host_layer_registry", None)
            if isinstance(reg, dict):
                for hl in reg.values():
                    _sever_dict(getattr(hl, "_bind_context", None))
        _sever(self, "_stm_single_S")
        _sever_dict(getattr(self, "_stm_last_reduce_routing", None))

    def flush_word_buffers(self):
        """Drain the per-subspace tensor word buffers (§6c Path B).

        The chart compose writes per-cell entries into each subspace's
        ``word_records`` / ``word_count`` buffers inside the brick;
        this hook materializes them into the legacy ``subspace.word``
        Python list once per tick so ``decompose``, ``reconstruct``,
        the SVO walker, and derivation-trace tests keep working
        unchanged. One ``.tolist()`` sync per subspace per tick, kept
        outside the brick.
        """
        for space in self.spaces:
            sub = getattr(space, 'subspace', None)
            if sub is None or not hasattr(sub, 'flush_word_buffer'):
                continue
            sub.flush_word_buffer()

    def _stub_outputs(self, B):
        """Per-row sentinel outputs for cursor-mode AR ticks.

        AR training is self-supervised on the input bytes; the
        ``OutputSpace`` only needs a per-row tensor to keep its
        prepOutput signature stable. Returns a list of B zero scalars
        (matches the placeholder created by ``Data.processLM`` when
        labels are absent).
        """
        return [torch.zeros(1) for _ in range(int(B))]

    class _TickPrefetcher:
        """Background-thread prefetch over ``ds.next_tick()``.

        A *single* worker thread calls ``ds.next_tick()`` ahead of the
        consumer and queues the ``(inp_items, out_items, hard_eos)``
        tuple. The main ``runEpoch`` loop pulls from the queue, hiding
        the per-tick CPU cost behind GPU compute (which releases the
        GIL while in CUDA / C++ kernels).

        Single-threaded by design: ``next_tick`` is Python-bound and
        the GIL serializes Python execution, so additional threads
        would contend without speedup. The XML ``<numWorkers>`` knob
        therefore controls *queue depth* (in-flight tick budget),
        not thread count. Queue depth = ``queue_size``: at most one
        tick consumed by main + ``queue_size - 1`` buffered ahead.

        ``next_tick`` is safe to call from this worker because it is a
        pure state machine over ``ds``'s internal cursor: no main-
        thread feedback, no shared mutable state outside ``ds``.
        """

        _SENTINEL = object()

        def __init__(self, ds, queue_size):
            """Spawn the prefetch worker and size the bounded queue.

            ``queue_size`` of N permits up to N-1 in-flight ticks buffered
            ahead of the consumer. The worker is a daemon thread so it
            does not block process exit on abrupt termination.
            """
            from queue import Queue
            from threading import Thread, Event
            self._ds = ds
            self._queue = Queue(maxsize=max(1, int(queue_size)))
            self._stop = Event()
            self._exc = None
            self._thread = Thread(
                target=self._produce, name="TickPrefetcher", daemon=True)
            self._thread.start()

        def _produce(self):
            """Worker loop: call ``ds.next_tick`` and enqueue results.

            Honours the ``_stop`` event, captures any exception into
            ``_exc`` so the consumer can re-raise it, and always enqueues
            ``_SENTINEL`` on exit so the consumer doesn't deadlock.
            """
            try:
                while not self._stop.is_set() and not self._ds.all_done():
                    tick = self._ds.next_tick()
                    while not self._stop.is_set():
                        try:
                            self._queue.put(tick, timeout=0.1)
                            break
                        except Exception:
                            continue
            except BaseException as e:
                self._exc = e
            finally:
                try:
                    self._queue.put(self._SENTINEL, timeout=1.0)
                except Exception:
                    pass

        def next(self):
            """Return next tick, ``None`` when the dataset is exhausted.
            Re-raises any exception that the worker hit."""
            item = self._queue.get()
            if item is self._SENTINEL:
                if self._exc is not None:
                    raise self._exc
                return None
            return item

        def close(self):
            """Signal stop, drain the queue, and join the worker.

            Bounded ``thread.join`` with a 2s timeout so a stuck worker
            can't block process exit indefinitely.
            """
            self._stop.set()
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break
            self._thread.join(timeout=2.0)

    def runEpoch(self, optimizer=None, batchSize=10, split="train",
                 max_batches=None):
        """Run one epoch over the dataset (training if optimizer given, eval if None).

        Drives batching with a ``SentenceStreamDataset`` DataLoader so every
        split is consumed as ``B = batchSize`` contiguous streams. ``B`` is
        capped at ``len(split)`` by ``data_loader``, so small eval sets
        degrade gracefully.

        ``max_batches`` (optional int): cap the outer cursor loop at this
        many ``runBatch`` calls. ``None`` (default) walks the full split.
        Used by ``--test N`` to bound the test-pass cost on large
        corpora; the bound applies to whichever split this call drives.

        Inference (split="runtime", no optimizer) shares the data-driven loop
        with the eval splits (training=False): it drives the data
        ``runtime_batch`` staged into ``train_input`` (via
        ``data_loader(split="runtime")``) through ``runBatch(train=False)``
        with a real ``batch_override``. Training-only work (loss/backward,
        CBOW, two-pass) is gated on ``training`` and skipped.

        Args:
            optimizer: pre-built Adam optimizer (persistent across epochs).
                       Pass None for evaluation mode.
            batchSize: requested batch size (capped by split length).
            split: "train", "test", or "validation"

        Returns (output_loss, reconstruction_loss, all_predictions, last_reconstruction).
        """
        training = optimizer is not None

        # LTM consolidation FU (Change 3, 2026-06-18): provision the unified
        # LTM from the XML <truthSet> LAZILY at the first runEpoch -- by which
        # point the dataset is loaded, so provision_ltm can run the truth
        # texts through the REAL forward. Guard with ``_ltm_provisioned`` set
        # True BEFORE provisioning so the inner runEpoch (split="runtime")
        # provision_ltm fires does NOT recurse. No-op (and the flag flips
        # harmlessly) when the gate is off / no <truthSet> -- byte-identical.
        if not getattr(self, '_ltm_provisioned', True):
            self._ltm_provisioned = True
            if getattr(self, 'ltm_consolidation', False):
                try:
                    self.provision_ltm()
                except Exception:
                    pass

        self.train(training)
        self.outputSpace.clearBatchResults()
        ss = self.symbolSpace
        if ss is not None and getattr(ss, 'discourse', None) is not None:
            ss.discourse.reset()
        ctx = torch.no_grad() if not training else nullcontext()

        # Runtime / inference (split="runtime", no optimizer) shares the
        # normal data-driven loop below with the eval splits (training=False):
        # ``data_loader(split="runtime")`` serves the data ``runtime_batch``
        # staged into ``train_input``, the cursor builds a real per-tick
        # ``batch_override``, and ``runBatch(train=False)`` runs the forward
        # (which records gold truths under ``truthCriterion``). The previous
        # bespoke fast-path called ``runBatch`` WITHOUT a batch_override (the
        # retired ARIR/arir_step entry), so EVERY ``runEpoch(split="runtime")``
        # raised "no batch_override supplied" -- ``store_truths``' documented
        # ingestion path. Falling through reuses the correct cursor + per-row
        # reset + tail dispatch; record()/accumulation is negligible for the
        # small runtime ingestion.

        allOutput = []
        allInput = []
        outputChunks = []
        inputChunks = []
        outErr = 0.0
        inErr = 0.0

        def record(result):
            """Book-keep one runBatch result into the shared accumulators.

            Updates ``outErr`` / ``inErr`` from the result's loss tensors
            (kept 0-dim to avoid a GPU sync). Appends per-batch
            prediction clones into ``outputChunks`` / ``inputChunks``
            only during eval -- training discards them.
            """
            nonlocal outErr, inErr
            # Note: do NOT call outputSpace.putBatch(result).  The list it
            # would append to has no readers, and each BatchResult retains
            # ~one batch of detached tensors -- a per-batch growth of
            # ~17 MB measured on MM_20M, accumulating over the epoch.
            #
            # Keep losses as 0-dim tensors here; .item() is a GPU sync.
            # The epoch-end return path materializes scalars exactly once.
            if result.lossOut is not None:
                outErr = result.lossOut.detach()
            if result.lossIn is not None:
                inErr = result.lossIn.detach()
            # Only the eval path (split in {test, validation, runtime})
            # reads ``allOut``/``allIn`` -- the train caller at runTrial
            # line ~2016 destructures and immediately discards them. So
            # accumulating per-batch outputPred clones during training was
            # pure overhead: ~0.5 MB / runBatch on MM_20M, growing across
            # the entire epoch.
            if not training:
                outputChunks.append(result.outputPred.clone().detach().squeeze())
                if self.reversible and result.inputPred is not None:
                    inputChunks.append(result.inputPred.clone().detach().squeeze())

        # Data splits live on CPU (see Data.toDevice); prepInput/prepOutput
        # move per-batch to device in the main process, so workers just
        # pickle CPU tensors. XML <numWorkers> sets _num_workers via
        # ModelFactory.run; the default lives in data/model.xml.
        num_workers = self._num_workers

        # Cursor universal (§8e). Two cursor modes share one outer loop:
        #
        #   * **Byte cursor** (AR text byte): each row's document is
        #     walked one ``slab_bytes``-wide tick at a time;
        #     ``hard_eos[b]`` flips True when row b crosses a doc
        #     boundary. Sized to ``nIdeas - 1`` so the lex's reserved
        #     EOS-sentinel slot stays byte-faithful.
        #   * **Trial cursor** (numeric / non-AR / non-byte): each tick
        #     yields one batch of trials with ``hard_eos = [True] * B``.
        #     The data cursor aligns with the trial: each tick is one
        #     atomic data unit per row (MNIST image, XOR sample,
        #     non-AR sentence). Per-row Reset fires for every row each
        #     tick, mirroring the pre-handoff DataLoader contract.
        text_input = (
            isinstance(self.inputSpace.data.train_input, list)
            and len(self.inputSpace.data.train_input) > 0
            and isinstance(self.inputSpace.data.train_input[0], str)
        )
        byte_lexer = getattr(self, 'lexer', None) in ('byte', 'bytes')
        use_byte_cursor = (text_input and byte_lexer)

        if use_byte_cursor:
            # InputSpace.outputShape[0] (= ``nIdeas``) is the byte-buffer
            # width the lex emits. Under the §8g/§EOS-removal change
            # the lex no longer reserves a slot for an EOS sentinel
            # (the slot held a null-embedding indistinguishable from
            # the codebook's null padding -- no reader consumed it as
            # a stop signal). Sizing the slab to ``nIdeas`` keeps the
            # cursor byte-faithful through the lex: every byte emitted
            # ends up in a real token, and the assert in
            # ``InputSpace._lex_batch`` (``n_tokens <= nIdeas``) holds
            # exactly under the bytes-mode parse fix that produces one
            # token per input byte.
            slab_bytes = max(1, int(self.inputSpace.outputShape[0]))
        else:
            slab_bytes = None           # trial cursor (one tick = one trial)

        # The runEpoch outer loop drives ``ds.next_tick()`` directly
        # for both modes; the surrounding ``DataLoader`` is built only
        # so existing tests can grab ``loader.dataset``. PyTorch's
        # ``num_workers`` is forced to 0 because the loader is never
        # iterated -- async workers would prefetch tensors no one
        # consumes. Parallel feed is provided by ``_TickPrefetcher``
        # below (see XML ``<numWorkers>``), which calls ``next_tick``
        # on a single background thread.
        loader = self.inputSpace.data.data_loader(
            split=split,
            num_streams=batchSize,
            num_workers=0,
            prefetch_factor=None,
            pin_memory=(TheDevice.get().type == "cuda"),
            slab_bytes=slab_bytes,
        )
        ds = loader.dataset
        B_eff = ds.num_streams

        # A mid-epoch checkpoint contains the update state *after* the saved
        # number of cursor ticks. Recreate the deterministic cursor and skip
        # those ticks before resuming updates. A different stream count would
        # map rows to different ticks, so fail loudly instead of silently
        # replaying or omitting training examples.
        resume_skip = 0
        if training and split == "train":
            resume_skip = int(getattr(
                self, "_resume_batches_to_skip", 0) or 0)
            saved_batch_size = getattr(
                self, "_checkpoint_batch_size", None)
            if (resume_skip and saved_batch_size is not None
                    and int(saved_batch_size) != int(batchSize)):
                raise ValueError(
                    "Cannot resume a mid-epoch checkpoint with a different "
                    f"batch size (saved={saved_batch_size}, "
                    f"requested={batchSize})")

        # Async tick prefetch. ``<numWorkers>`` becomes the in-flight
        # tick budget (1 tick consumed by main + N-1 buffered). 0
        # preserves the legacy synchronous path.
        prefetcher = (BasicModel._TickPrefetcher(ds, queue_size=num_workers)
                      if num_workers > 0 else None)

        # Pre-build the AR stub outputs once for the byte-cursor path
        # (every tick reuses them; the AR target is the input bytes).
        if use_byte_cursor:
            byte_stub_output = self.outputSpace.prepOutput(
                self._stub_outputs(B_eff))
        else:
            byte_stub_output = None

        # BASIC_MAX_BATCHES caps the cumulative training-batch count
        # across all epochs (train.py forwards --batches here).
        # split=="train" only -- test/validation passes use the
        # explicit ``max_batches`` arg from --test.
        global_max = None
        if (training and split == "train"
                and not getattr(self, "_preflight_active", False)):
            # The brick pre-flight runs bounded throw-away epochs; they
            # must NOT consume (or be capped by) the BASIC_MAX_BATCHES
            # ``--batches`` budget, or the warm-up eats it all and the
            # profiled epoch + real training run zero batches.
            try:
                _v = os.environ.get("BASIC_MAX_BATCHES", "").strip()
                global_max = int(_v) if _v else None
            except ValueError:
                global_max = None
            if not hasattr(self, "_train_batches_seen"):
                self._train_batches_seen = 0
        with ctx:
            step = 0
            batches_run = 0
            while True:
                deadline = getattr(
                    self, "_training_deadline_monotonic", None)
                if (training and split == "train"
                        and not getattr(self, "_preflight_active", False)
                        and deadline is not None
                        and time.monotonic() >= float(deadline)):
                    self._training_deadline_reached = True
                    TheMessage(
                        f"runEpoch({split}): hit BASIC_MAX_SECONDS after "
                        f"{batches_run} completed batches this epoch; "
                        "cursor exiting before the next tick.")
                    break
                if max_batches is not None and batches_run >= max_batches:
                    TheMessage(
                        f"runEpoch({split}): hit max_batches={max_batches} "
                        f"cap; cursor exiting early ({batches_run} batches "
                        f"processed, ds.all_done={ds.all_done()}).")
                    break
                if (global_max is not None
                        and self._train_batches_seen >= global_max):
                    TheMessage(
                        f"runEpoch({split}): hit BASIC_MAX_BATCHES="
                        f"{global_max} cumulative cap; cursor exiting "
                        f"early ({batches_run} batches this epoch, "
                        f"{self._train_batches_seen} total).")
                    break
                if prefetcher is not None:
                    tick = prefetcher.next()
                    if tick is None:
                        break
                    inp_items, out_items, hard_eos = tick
                else:
                    if ds.all_done():
                        break
                    inp_items, out_items, hard_eos = ds.next_tick()
                if resume_skip > 0:
                    resume_skip -= 1
                    self._resume_batches_to_skip = resume_skip
                    continue
                if use_byte_cursor:
                    # Byte slab: convert uint8 -> int8 [B, 1, slab_bytes]
                    # to match prepInput's expected shape downstream.
                    # Synchronous H2D: correct + race-free (non_blocking
                    # on an ephemeral pinned buffer corrupts data when
                    # freed pre-transfer -- the NaN source).
                    inputTensor = inp_items.to(
                        device=TheDevice.get(), dtype=torch.int8
                    ).unsqueeze(1)
                    # Hand the *host* slab to the lexer (consumed once in
                    # InputSpace._lex_batch). `inputTensor` is exactly
                    # `inp_items.to(device,int8).unsqueeze(1)`, so the
                    # bytes are identical (_to_text masks `& 0xFF`, so
                    # int8/uint8 agree); lexing the host copy makes
                    # `_to_text`'s `.tolist()` a CPU op instead of a
                    # cudaMemcpyDtoH (residual A, doc/BrickHostSyncStatus
                    # .md). Device path stays the fallback for non-cursor
                    # callers.
                    self.inputSpace._host_input_slab = inp_items
                    outputTensor = byte_stub_output
                    B_step = B_eff
                else:
                    # Trial mode: caller supplies raw inputs/outputs;
                    # prepInput / prepOutput stage them onto device.
                    B_step = (inp_items.shape[0]
                              if isinstance(inp_items, torch.Tensor)
                              else len(inp_items))
                    inputTensor = self.inputSpace.prepInput(inp_items)
                    outputTensor = self.outputSpace.prepOutput(out_items)

                # Unified path: AR modes drive their outer pos loop
                # inside BasicModel.forward() via the sliding-window
                # buffer on InputSpace; non-AR modes run one pass. See
                # basicmodel/doc/specs/2026-04-20-streaming-ar-training-loop-design.md
                #
                # ``progress`` (cursor's doc-fraction completed) lets
                # ``runBatch`` print a percent next to its ``batch =``
                # line so long runs report visible progress. Cap at the
                # ``max_batches`` slice when set so a capped pass shows
                # 100% at its own completion rather than the underlying
                # cursor's fraction.
                if max_batches is not None and max_batches > 0:
                    cap_frac = (batches_run + 1) / float(max_batches)
                    progress_frac = min(1.0, cap_frac)
                else:
                    progress_frac = ds.progress() if hasattr(ds, 'progress') else None
                # Two-pass learning (doc/Language.md soft superposition): run
                # the SAME sentence twice as two separate trials -- pass A at
                # temperature 0 (sharp/deterministic, recorded) and pass B at
                # <exploreTemperature> (flatter, exploration). Default off ->
                # one ordinary pass (superposition_temperature=None).
                _two_pass = bool(training
                                 and getattr(self, 'two_pass_learning', False))
                # Trial-split: a deterministic fraction of TRAINING batches run
                # as pure next-idea PREDICTION trials (recon terms zeroed); the
                # rest reconstruct. Bresenham-style ratio (no RNG -> no test
                # flakiness); ratio 0 -> always "reconstruct" -> byte-identical.
                # Decided ONCE per sentence so both two-pass passes agree.
                _ptr = float(getattr(self, 'prediction_trial_ratio', 0.0) or 0.0)
                _trial_mode = ("predict"
                               if (training and _ptr > 0.0
                                   and int((step + 1) * _ptr) != int(step * _ptr))
                               else "reconstruct")
                result, _ = self.runBatch(
                    train=training, batchNum=step,
                    batchSize=B_step, split=split,
                    optimizer=optimizer,
                    batch_override=(inputTensor, outputTensor),
                    progress=progress_frac,
                    superposition_temperature=(0.0 if _two_pass else None),
                    trial_mode=_trial_mode,
                )
                if result is not None:
                    record(result)
                if _two_pass:
                    # Pass B: explore trial. A separate forward/loss/backward/
                    # step; its result is NOT recorded -- trimmed from the
                    # per-batch error, and the batch count does not advance.
                    # exploration_trial=True gates the per-sentence side
                    # effects (clock / discourse observe / LTM append / step
                    # counter) so B does not double-commit pass A's sentence.
                    # Sever pass A's carried graph first: pass A's optimizer
                    # step already moved the saved tensors' versions, so a
                    # pass-B forward re-reading a carrier would break the
                    # pass-B backward (the post_tick_compact discipline,
                    # applied at the intra-tick pass boundary).
                    self._detach_persistent_state()
                    self.runBatch(
                        train=True, batchNum=step,
                        batchSize=B_step, split=split,
                        optimizer=optimizer,
                        batch_override=(inputTensor, outputTensor),
                        progress=progress_frac,
                        superposition_temperature=float(
                            getattr(self, 'explore_temperature', 0.5)),
                        exploration_trial=True,
                        trial_mode=_trial_mode,
                    )
                # Tail dispatch: word-buffer flush (§6c Path B), per-row
                # hard reset, soft reset, then the truth-layer compact
                # -- all live outside runBatch under the rolling-cursor
                # compute-brick contract. Flush runs first so the
                # materialized ``subspace.word`` is available to any
                # post-tick consumer the Reset path might touch.
                self.flush_word_buffers()
                self.dispatch_per_row_reset(hard_eos)
                self.dispatch_soft_reset()
                self.post_tick_compact()
                step += B_step
                batches_run += 1
                if (training and split == "train"
                        and not getattr(self, "_preflight_active", False)):
                    self._train_batches_seen += 1
                    self._epoch_batches_seen = int(getattr(
                        self, "_epoch_batches_seen", 0) or 0) + 1
                    self._checkpoint_batch_size = int(batchSize)
                    # Save after the outer cursor count advances so a resumed
                    # checkpoint's progress metadata describes the update it
                    # actually contains, not the preceding batch.
                    self._maybe_save_periodic_checkpoint()

                # Per-batch CBOW/SBOW embedding training for text AR
                # modes that don't use the byte cursor (word/sentence
                # lexer). Byte-cursor path has its own embedding update
                # plumbing inside runBatch.
                if (training and not use_byte_cursor
                        and text_input
                        and isinstance(inp_items, list)
                        and inp_items
                        and isinstance(inp_items[0], str)):
                    te = getattr(self, 'train_embedding', 'NONE')
                    if te in ('CBOW', 'SBOW', 'BOTH'):
                        method = 'CBOW' if te == 'CBOW' else 'SBOW'
                        for sentence in inp_items:
                            words = [t for t, _
                                     in parse(sentence, lex='words')]
                            self.perceptualSpace.train_embeddings(
                                words, method=method)

        # Stop the prefetch worker now that the loop is done. The
        # thread is daemon=True so an exception unwinding past this
        # point will not leak a zombie thread; explicit close() lets
        # us reclaim resources promptly on the normal-exit path.
        if prefetcher is not None:
            prefetcher.close()

        if training and split == "train" and ds.all_done():
            self._epoch_batches_seen = 0
            self._resume_batches_to_skip = 0

        if inputChunks:
            if outputChunks[0].dim() == 0:
                allInput = torch.stack(inputChunks, dim=0)
            else:
                allInput = torch.cat(inputChunks, dim=0)
        if outputChunks:
            if outputChunks[0].dim() == 0:
                allOutput = torch.stack(outputChunks, dim=0)
            else:
                allOutput = torch.cat(outputChunks, dim=0)

        # Trim eval-pass outputs to the test-split row count.  The
        # byte-cursor's per-stream pacing keeps emitting one row per
        # tick per stream until every stream's data is drained, so
        # streams that finish early contribute trailing padding rows
        # that have no matching ground-truth label.  Post-epoch
        # consumers (``mnistReport``, accuracy plots) align
        # ``allOutput`` against ``model.outputSpace.getTestOutput()``
        # row-for-row; trim to the shorter length so they stay in
        # sync.  Train pass discards both so the trim is eval-only.
        if not training and isinstance(allOutput, torch.Tensor):
            try:
                ref = self.outputSpace.getTestOutput()
                if isinstance(ref, torch.Tensor) and ref.dim() >= 1:
                    target_n = int(ref.shape[0])
                    if allOutput.shape[0] > target_n:
                        allOutput = allOutput[:target_n]
                        if isinstance(allInput, torch.Tensor) and allInput.shape[0] > target_n:
                            allInput = allInput[:target_n]
            except (AttributeError, RuntimeError):
                pass

        # Return the loss scalars as tensors; the caller (runTrial)
        # materializes them with .item(). That keeps the host sync
        # OUTSIDE the brick's profiled runEpoch -- the pre-flight
        # profiles runEpoch directly, and an in-runEpoch .item() is a
        # cudaMemcpyDtoH that breaks the CUDA-graph-capture contract
        # (test_brick_no_sync). Consumers must materialize before any
        # ``:.4f`` format / list-accumulation (see runTrial).
        return outErr, inErr, allOutput, allInput
    def _create_per_stage(self, nInput, nPercepts, nConcepts, nSymbols, nWords=16, nOutput=32,
               subsymbolicOrder=1,
               model_type="numeric", data=None,
               reconstruction_scale=0.5,
               what_scale=0.7, where_scale=0.2, when_scale=0.1,
               **kwargs):
        """Wire the full per-stage space stack from architecture parameters.

        Builds Input / Perceptual / Conceptual stages (one per
        ``subsymbolicOrder`` step) / Symbolic / Output, plus the optional
        SymbolSpace, subsymbolic, and pipeline modules. Mutates ``self``
        extensively (sets every ``self.*Space`` attribute, ``self.spaces``,
        ``self.symbolSpace``, ``self.reversible``, etc.).
        """
        self.spaces = []
        # Execution-only epoch used by immutable carrier/codebook identities.
        # It intentionally does not ride in state_dict: no carrier or replica
        # survives a model rebuild/checkpoint load.
        self._parameter_epoch = 0
        # Serialized model clock (doc/plans/2026-06-07-model-time-when-
        # encoding.md): a 0-initialized ``long`` that increments once per
        # processed batch (train AND inference) in ``runBatch`` and rides
        # ``state_dict()`` through save/load. It is the AUTHORITATIVE absolute
        # time; ``runBatch`` propagates it to each live ``WhenRangeEncoding``'s
        # ``.t`` so a default-stamped .when carries the absolute model time.
        # Guarded so a re-entrant build does not re-register the buffer.
        if "when_time" not in self._buffers:
            self.register_buffer(
                "when_time", torch.zeros((), dtype=torch.long))
        self.symbolSpace = None  # wired below once the home spaces exist
        self.reversible = True
        self.nInput = nInput
        self.nPercepts = nPercepts
        self.nConcepts = nConcepts
        # The construction argument is the number of simultaneously live WS
        # locations. Keep that geometry distinct from both the WholeSpace
        # property inventory and the downstream concept-reference inventory.
        self.nSymbolSlots = int(nSymbols)
        self.nOutput = nOutput
        self.nWords = nWords
        self.data = data
        self.model_type = model_type
        # Phase 4b (rev. 2026-06-09): <lexer> lives on WholeSpace
        # (lexing is analytic cutting); IS-side <lexer> rejected loudly.
        from Spaces import resolve_lexer as _resolve_lexer
        self.lexer = _resolve_lexer()
        self.ergodic = TheXMLConfig.get("architecture.ergodic")
        self.processSymbols = TheXMLConfig.get("architecture.processSymbols")
        self.certainty = TheXMLConfig.get("architecture.training.certainty")
        # InputSpace.codebook defaults to false; see the matching note in
        # BasicModel.create.
        self.codebook = Space.normalize_codebook_mode(
            TheXMLConfig.space("InputSpace", "codebook", default=False)) != "none"
        # PartSpace is subsymbolic (2026-06-09 asymmetric-VQ plan §7
        # task 7): its <codebook> element was retired from the schema and PS is
        # hardwired to "none" in Space.__init__. The former
        # ``self.perceptCodebook = TheXMLConfig.space("PartSpace",
        # "codebook")`` read is gone -- it was dead (assigned, never consumed)
        # and would now KeyError on the removed element.
        self.conceptCodebook = TheXMLConfig.space("ConceptualSpace", "codebook")
        # Canonical WS role: an upstream property basis, independent of the
        # concept/symbol namespace. Legacy configs omit the knob and retain the
        # pre-migration construction path until they are converted explicitly.
        self.wholePropertyBasis = bool(TheXMLConfig.space(
            "WholeSpace", "propertyBasis", default=False))
        self.subsymbolicOrder = subsymbolicOrder

        # Monotonic SigmaLayer weights (W >= 0). Mirrors PiLayer's monotonic
        # flag; when True, invertible SigmaLayers use NonNegativeInvertibleLinearLayer.
        self.monotonic = bool(
            TheXMLConfig.get("architecture.monotonic", default=False))
        # ``useGrammar`` XML knob retired 2026-05-13: derived from the
        # configured grammar instead. See ``_derive_use_grammar``.
        self.useGrammar = self._derive_use_grammar()
        # ``<reconstruct>`` enum retired (A1: schema element +
        # ``reconstructEnum`` removed). C3 (spec sec 7): reconstruction is
        # now UNCONDITIONALLY concepts-seeded in ``runBatch``, so there is
        # no ``self.reconstruct`` attribute and no gate to parse.
        # Gate-L1 sparsity lambda for LiftLayer / LowerLayer raw_gate
        # parameters. 0.0 (default) disables the penalty; configs that
        # use lift/lower opt in via <gateL1Lambda> in <architecture>.
        self.gate_l1_lambda = float(
            TheXMLConfig.get("architecture.gateL1Lambda", default=0.0) or 0.0)
        # Sparse-MoE load-balance loss weight (Shazeer et al. 2017).
        # Currently INERT: read here but has no consumer -- the chart's
        # sparse-MoE load-balance bookkeeping was retired with the chart
        # itself (see ~line 4254), and ``chartTopK`` is retired. The knob
        # stands by for any future signal-router rule load-balancing.
        self.load_balance_weight = float(
            TheXMLConfig.get(
                "architecture.loadBalanceWeight", default=0.0) or 0.0)
        # thoughtFree is structurally equivalent to subsymbolicOrder=0: no
        # higher-order P/C/S cycles. Reject the nonsense combination early.
        TheXMLConfig.require(
            lambda cfg, _ug=self.useGrammar, _co=self.subsymbolicOrder:
                _ug != "thoughtFree" or _co == 0,
            f"useGrammar='thoughtFree' requires subsymbolicOrder=0 "
            f"(got useGrammar={self.useGrammar!r}, "
            f"subsymbolicOrder={self.subsymbolicOrder})"
        )
        # Truth integration config (optional -- absent in BasicModel.xml)
        self.truth_bias_scale = float(TheXMLConfig.get("architecture.truthBiasScale", default=0.1) or 0.1)
        self.luminosity_weight = float(TheXMLConfig.get("architecture.LuminosityWeight", default=0.1) or 0.1)
        self.universality_weight = float(TheXMLConfig.get("architecture.UniversalityWeight", default=0.1) or 0.1)
        # Quaternary-corner balance knobs (see Philosophy.md for
        # the tetralemma/catuskoti mapping). Defaults permit NEITHER
        # (epistemic uncertainty) but forbid BOTH (classical
        # non-contradiction):
        #   allowExcludedMiddle=1  permit NEITHER
        #   allowContradiction=0   forbid BOTH
        self.allow_excluded_middle = int(
            TheXMLConfig.get("architecture.allowExcludedMiddle", default=1) or 1)
        self.allow_contradiction = int(
            TheXMLConfig.get("architecture.allowContradiction", default=0) or 0)
        self.truth_loss_weight = float(TheXMLConfig.training("TruthLoss", default=0.0) or 0.0)
        # Truth-grounded reasoning answer (policy) loss weight (Phase 5; dark
        # until the <queryReasoning> consumer trains the soft route). Default
        # 0.0 -> no answer-loss term (byte-identical).
        self.answer_loss_weight = float(
            TheXMLConfig.training("answerLossWeight", default=0.0) or 0.0)
        # Step 2: the next-idea POLICY loss weight (trains the {arma, retrieval,
        # deduction} blend). 0.0 -> no term -> byte-identical.
        self.predict_next_loss_weight = float(
            TheXMLConfig.training("predictNextLossWeight", default=0.0) or 0.0)
        # Thinking Kernel next-op policy loss weight (spec §12.6): behavior-
        # clones the NextOpPolicy head on successful kernel traces generated
        # from the reasoning store. 0.0 -> no term -> byte-identical.
        self.thinking_loss_weight = float(
            TheXMLConfig.training("thinkingLossWeight", default=0.0) or 0.0)
        # Method-1 -> Method-2 leaf distillation weight (root separability;
        # snap design doc step 3). 0.0 -> no term -> byte-identical.
        self.leaf_distill_weight = float(
            TheXMLConfig.training("leafDistillWeight", default=0.0) or 0.0)

        # Syntax tree dump — when <writeSyntax>true</writeSyntax> is
        # set in the model XML (under <architecture>), BasicModel.forward
        # writes an XML syntax tree (one per batch row) to syntaxOutPath
        # at the end of each forward pass. See doc/Language.md
        # "POS side-channel" for the format.
        self._write_syntax = bool(TheXMLConfig.get(
            "architecture.writeSyntax", default=False) or False)
        self._syntax_out_path = TheXMLConfig.get(
            "architecture.syntaxOutPath",
            default="output/syntax.xml") or "output/syntax.xml"
        self._syntax_truncated = False

        # --- Dimensional-governance knobs (A1/A2, 2026-06-06) ---
        # ``<reconstruct>`` enum RETIRED (A1, 2026-06-09): the schema element
        # + ``reconstructEnum`` were removed. The retired ``perfect`` mode
        # SKIPPED the per-stage ConceptualCombine (carrier carried unchanged
        # for an exact round-trip). Under ``conceptBinding=mixing`` the
        # surviving learned combine runs unconditionally; ``aligned`` is a
        # separate same-location binder and allocates no ConceptualCombine.
        # No ``self.reconstruct`` / ``self.perfect_reconstruction`` attribute
        # is parsed; binding mode, not reconstruction mode, now selects the
        # appropriate forward/reverse carrier.
        # ``<architecture><prediction>`` (predictionEnum: none|interSentence,
        # default "none"). Stored as the canonical XSD enum string (NOT
        # lowercased) so downstream dispatch matches "interSentence" exactly.
        self.prediction_mode = str(
            TheXMLConfig.get("architecture.prediction", default="none")
            or "none")
        # ``<architecture><sigmaPi>`` (sigmaPiEnum: last|butterfly|full,
        # default "butterfly" at the architecture level per the XSD default).
        # Routed through the same Space.sigma_pi_mode() normalizer used by
        # PartSpace and WholeSpace so the canonical mode string is
        # identical wherever it is read.
        self.sigma_pi_mode = Space.sigma_pi_mode(
            TheXMLConfig.get("architecture.sigmaPi", default="butterfly")
            or "butterfly")

        self.loss = ModelLoss(
            reconstruction_scale=reconstruction_scale,
            what_scale=what_scale,
            where_scale=where_scale,
            when_scale=when_scale,
            nOutput=nOutput,
            subsymbolicOrder=subsymbolicOrder,
            # Loss operates on the output space_role, which carries no where/when.
            nWhere=canonical_shape("OutputSpace")[0],
            nWhen=canonical_shape("OutputSpace")[1],
        )
        # Optional swappable loss head fed STM snapshots during the
        # body forward (Phase 3, 2026-05-12). Installed by
        # ``embed.py embed_pretrain`` for CBOW-over-STM pretraining;
        # AR training leaves it None and keeps using ``outputSpace``
        # for the loss.
        self.loss_head = None

        # "6+2+2": config <nDim> is the EVENT width (it EMBRACES the
        # .where/.when band), not the bare content width. Carve the bare CONTENT
        # width ``*_dim = event - band``; downstream shapes restore the event via
        # ``*_dim + obj_*`` (I/O + per-stage shapes), while a bare ``*_dim`` is the
        # content (codebook / C->P feedback / symbol snap). So nWhat == content
        # == nDim - band and the event width == nDim.
        # Helpers _resolve_dim / _obj_size / _nvec live on BaseModel.
        obj_input   = self._obj_size("InputSpace")
        obj_percept = self._obj_size("PartSpace")
        obj_concept = self._obj_size("ConceptualSpace")
        obj_symbol  = self._obj_size("WholeSpace")
        obj_output  = self._obj_size("OutputSpace")

        # Every interior space carries the canonical (nWhere=4, nWhen=4)
        # band, so ``nDim = nWhat + nWhere + nWhen``. PS/WS native WHAT
        # widths may differ from CS: their sparse codebook activation already
        # emits a conceptual-width event before the CS boundary. ``*_dim`` is
        # therefore the native WHAT width of that space, not evidence of a
        # linear PS->WS->CS chain.
        input_event   = self._resolve_dim("InputSpace",      1)
        percept_event = self._resolve_dim("PartSpace", input_event)
        concept_event = self._resolve_dim("ConceptualSpace", percept_event)
        symbol_event  = self._resolve_dim("WholeSpace",   concept_event)
        # The task head consumes terminal CS directly.  An omitted OutputSpace
        # width therefore inherits the conceptual event, not WholeSpace's
        # native property width.
        output_event  = self._resolve_dim("OutputSpace",    concept_event)

        input_dim   = input_event   - obj_input    # = nWhat (InputSpace)
        percept_dim = percept_event - obj_percept  # = nWhat (PartSpace)
        concept_dim = concept_event - obj_concept  # = nWhat (ConceptualSpace)
        symbol_dim  = symbol_event  - obj_symbol   # = nWhat (WholeSpace)
        output_dim  = output_event  - obj_output   # = nWhat (OutputSpace)

        nvec_input   = self._nvec("InputSpace",      nInput)
        nvec_percept = self._nvec("PartSpace", nPercepts)
        nvec_concept = self._nvec("ConceptualSpace", nConcepts)
        nvec_whole_properties = self._nvec("WholeSpace", nSymbols)
        nvec_output  = self._nvec("OutputSpace",     nOutput)
        self.nConceptCodes = int(nvec_concept)
        # Symbols are downstream references into the conceptual inventory;
        # they do not allocate a second learned prototype table. Preserve the
        # historical attribute name for consumers that ask for reference
        # capacity, while spelling out the independent WS property count.
        self.nSymbols = self.nConceptCodes
        self.nWholeProperties = int(nvec_whole_properties)

        # Stage 2: the loopback is always wired -- ConceptualSpace's
        # per-stage input PiLayer is widened by ``symbolShape[1]`` at
        # construction time. No separate XML validator required.

        # Build I/O shape tuples: [count, dim + objectSize]
        inputShape   = [nInput,    input_dim   + obj_input]
        perceptShape = [nPercepts, percept_dim + obj_percept]
        conceptShape = [nConcepts, concept_dim + obj_concept]
        symbolShape  = [nSymbols,  symbol_dim  + obj_symbol]
        outputShape  = [nOutput,   output_dim  + obj_output]

        # Build codebook (space-internal) shape tuples: [nVectors, nDim]
        spaceShape_input   = [nvec_input,   input_dim]
        spaceShape_percept = [nvec_percept, percept_dim]
        spaceShape_concept = [nvec_concept, concept_dim]
        spaceShape_symbol  = [nvec_whole_properties, symbol_dim]
        spaceShape_output  = [nvec_output,  output_dim]

        rawInputShape = [nInput, input_dim]
        self.inputSpace = self._make_input_space(
            rawInputShape, spaceShape_input, inputShape,
            model_type=model_type)

        # Input -> Percept (uses _make_perceptual_space so demuxed
        # configs route to ModalSpace).
        self.perceptualSpace = self._make_perceptual_space(
            inputShape, spaceShape_percept, perceptShape)
        # 2026-06-07: the ``_peer_perceptual`` back-ref (InputSpace -> PS) is
        # RETIRED. InputSpace is a pure RAW lexer and PartSpace owns all
        # tokenization + codebook work; the forward pipeline drives IS.forward
        # then PS.forward, so no back-reference is wired here.

        conceptInputShape = [nPercepts, percept_dim + obj_percept]

        # ConceptualSpace output shape uses the explicit XML values
        # ``<nOutput>`` (already resolved to ``nConcepts`` upstream)
        # and ``<nOutputDim>`` (when supplied) directly. Earlier
        # versions of this code derived N from a volume-preserving
        # ``input_volume // nOutputDim`` formula; that's wrong because
        # the C-space_role codebook can re-dimension between input and
        # output independent of any volume-preservation contract, and
        # because users who set both ``<nOutput>`` and
        # ``<nOutputDim>`` already accounted for the reshape they
        # want. ``nConcepts`` is ``_resolve('ConceptualSpace',
        # nPercepts)`` -- it returns the XML ``<nOutput>`` when set,
        # else falls through to ``nPercepts``.
        try:
            _c_nOutputDim = TheXMLConfig.space("ConceptualSpace", "nOutputDim")
        except KeyError:
            _c_nOutputDim = 0
        if _c_nOutputDim > 0:
            conceptOutputShape = [nConcepts, _c_nOutputDim]
        else:
            conceptOutputShape = [nConcepts, concept_dim + obj_concept]

        # -- Grammar path: progressive bottleneck per conceptual order --
        if self.useGrammar == "all":
            n_stages = self.subsymbolicOrder
            self._level_shapes_list = self._level_shapes(
                nPercepts, percept_dim + obj_percept, n_stages,
                width_mode=self._conceptual_width_mode())
        else:
            n_stages = self.subsymbolicOrder
            self._level_shapes_list = None

        # -- Per-stage arrays: independent ConceptualSpace / WholeSpace
        # per stage. The pipeline flows stage-by-stage with no shared
        # per-level views; cross-forward autograd retention vanishes.
        # subsymbolicOrder=0 still needs one stage so the pre-seed C->S
        # pass (test_merged_loop.test_unified_loop_conceptualorder_zero_pre_seed_only)
        # has a concreteSpace/wholeSpace to populate. The j-iteration
        # count reported to runBatch is still the configured value.
        T = max(1, int(n_stages))
        self.conceptualSpaces = nn.ModuleList()
        self.wholeSpaces = nn.ModuleList()
        # A canonical aligned/property-basis model has one unpartitioned
        # conceptual namespace.  Build its large physical Codebook once at
        # stage 0 and hand that module to later folds at construction time.
        # Mixing and legacy WholeSpace models retain per-stage dictionaries.
        _share_concept_dictionary = (
            self.wholePropertyBasis
            and getattr(self, "concept_binding", "mixing") == "aligned")
        _serial_meta_raw = TheXMLConfig.get(
            "architecture.serialObjectMeta", default=False)
        _serial_meta = (
            _serial_meta_raw if isinstance(_serial_meta_raw, bool)
            else str(_serial_meta_raw or "").strip().lower()
            in ("true", "1", "yes", "on"))
        # Construction precedes the public ``self.serial`` assignment below,
        # so resolve the selector here with the same legacy fallback. Indexed
        # sparse codebook reads are safe only on the canonical word-serial
        # path; a parallel CS has differentiable dense inventory consumers.
        _serial_raw = TheXMLConfig.get("architecture.serial", default=None)
        if _serial_raw is None:
            _symbolic_raw = TheXMLConfig.get(
                "architecture.symbolicOrder", default=None)
            _grammar_default_order = (
                1 if self.useGrammar != "none" else 0)
            _serial_requested = (
                int(_symbolic_raw) > 0
                if _symbolic_raw is not None
                else _grammar_default_order > 0)
        elif isinstance(_serial_raw, bool):
            _serial_requested = _serial_raw
        else:
            _serial_requested = (
                str(_serial_raw).strip().lower()
                in ("true", "1", "yes", "on"))
        _aligned_codebook_sources = bool(
            _share_concept_dictionary
            and _serial_meta
            and _serial_requested)
        _shared_concept_dictionary = None
        # Fold provenance is keyed by the same stable concept identities as the
        # shared aligned dictionary.  Keep one host-side registry across every
        # conceptual stage as well: stage 0 resolves/adopts identities, while
        # the production checkpoint envelope is collected from the terminal CS.
        # A shared mapping makes both views authoritative and also means loading
        # the terminal checkpoint view immediately restores stage-0 lookups.
        _shared_concept_fold_support = (
            {} if _share_concept_dictionary else None)
        for t in range(T):
            is_last = (t == T - 1)
            if self.useGrammar == "all":
                # Grammar path. Tapered width halves N between stages;
                # uniform width keeps the canonical instantaneous field
                # unchanged at every order. `_level_shapes` has long exposed
                # both modes, but construction historically ignored the
                # uniform result and unconditionally shifted N -- making an
                # eight-wide, order-4 field collapse 8->4->2->1 despite the
                # explicit setting.
                #
                # Width contract (H3, 2026-05-18): the per-stage
                # ConceptualSpace input is the true upstream percept
                # width ``percept_dim + obj_percept`` (``cs_in``), and
                # its output is the conceptual content width
                # ``concept_dim`` (``cs_out``). Pre Stage 1.C the
                # ``sigma_percept`` SigmaLayer did the per-stage
                # percept→concept *lift* here; Stage 1.C retired that
                # atomic fold (see ConceptualSpace docstring) so the
                # per-stage shapes still describe input/output widths
                # for the CS recurrent cell but the lift itself is now
                # the signal-router's responsibility (Stage 3).
                # ``concept_dim`` is the BARE ``<ConceptualSpace><nDim>``
                # content width.  Per the modality re-architecture, CS
                # CARRIES the event where/when (mux at PS->CS): the CS output
                # muxed width is ``concept_dim + obj_concept`` so ``SubSpace``
                # derives ``nWhat == concept_dim`` (= nDim) with where/when as
                # the added tail (``muxedSize == concept_dim + nWhere +
                # nWhen``); WS is demuxed back to the bare ``concept_dim`` (the
                # CS->WS materialize trim, Spaces.py ~14730).  Preserving the
                # bare content width is what the C->P feedback
                # gate (``PartSpace.pi_concept.nInput ==
                # <ConceptualSpace><nDim>``), the Phase-2A.5 symbol
                # snap (``WholeSpace.subspace.what.W`` width ==
                # ``symbol_dim``), and ``WholeSpace.forward``'s
                # ``[B, N, concept_dim]`` pass-through contract
                # (validate_config: ``effective_concept_dim ==
                # symbol_dim``) all require.  For configs where
                # ``concept_dim == percept_dim + obj_percept`` (e.g.
                # MM_xor: 10 == 10) this is identical to the prior
                # width-preserving shapes (no-op); for MM_20M it
                # activates the previously-dropped C->P feedback +
                # snap (1024 vs the old 10).
                _uniform_width = (
                    self._conceptual_width_mode() == "uniform")
                n_t = nPercepts if _uniform_width else (nPercepts >> t)
                # In the aligned serial model PS/WS recurse at their native
                # widths, then each rung's sigma/codebook read emits a
                # conceptual-width activation. CS receives that activation;
                # its boundary itself performs no feature-width change.
                d_in = (concept_dim + obj_concept
                        if _aligned_codebook_sources
                        else percept_dim + obj_percept)
                # CS output muxed width = concept_dim + obj_concept, so the CS
                # SubSpace derives nWhat == concept_dim (= nDim) and the event
                # carries where/when. The per-stage codebook content
                # (stage_space_concept) stays bare nDim -- where/when ride as
                # muxed traces, not codebook rows.
                d_out = concept_dim + obj_concept
                cs_in = [n_t, d_in]
                cs_out = ([n_t, d_out]
                          if is_last or _uniform_width
                          else [n_t >> 1, d_out])
                # WS is BARE concept_dim (the demux target; canonical WS=(0,0)).
                # WS slot count is the PRE-merge stage input ``n_t`` -- NOT the
                # post-merge ``cs_out[0]`` (= ``n_t >> 1`` for non-last stages).
                # The merge runs at the END of a stage and halves the CS that
                # becomes the NEXT stage's input; so the WS at stage ``t``
                # consumes the prior stage's post-merge output, whose slot count
                # is ``nPercepts >> t == n_t`` (this stage's ``cs_in[0]``), and
                # the stage-``t`` merge then takes that same ``n_t`` down to
                # ``n_t >> 1`` for stage ``t+1``.  Sizing the WS (and its square
                # ``sigma`` butterfly, N == ws_in[0] * sigma_dim) at the halved
                # ``cs_out[0]`` made it one power-of-two too small on every
                # non-last stage: a runtime ``n_t``-slot event overflowed the
                # ``n_t>>1``-slot cascade (``butterfly_flatten`` M_total
                # mismatch) the moment the WS sigma fold was live. Last stage:
                # ``cs_out[0] == n_t`` already, so this is unchanged there.
                ws_in = [n_t, concept_dim]
                ws_out = [n_t, concept_dim]
            else:
                # Plain path: all stages share the legacy conceptInputShape /
                # conceptOutputShape. No N-halving.
                cs_in = list(conceptInputShape)
                cs_out = list(conceptOutputShape)
                ws_in = list(conceptOutputShape)
                ws_out = list(symbolShape)

            # dimensional-governance (2026-06-06): WS may RESHAPE the deep CS
            # idea into WIDE symbols -- honor an explicit
            # <WholeSpace><nOutputDim> as the OUTPUT width (e.g. deep
            # [8,1024] -> wide [1024,8] with a small symbol code), DECOUPLED
            # from the codebook/processing width (nDim). Without this the
            # construction sizes the WS output at concept width, ballooning a
            # wide-symbol config. Applies to both grammar and plain paths.
            try:
                _ws_od = int(TheXMLConfig.space("WholeSpace", "nOutputDim"))
            except (KeyError, TypeError, ValueError):
                _ws_od = 0
            if _ws_od > 0:
                ws_out = [ws_out[0], _ws_od]

            # Non-codebook spaces require nVectors (spaceShape[0]) ==
            # nActive (outputShape[0]); resize the per-stage codebook shape
            # to match the halved N. Dual-towers rev 2: the sparse-PARALLEL
            # path instead honors the config <nVectors> INVENTORY (the taper
            # blocks + inert tail live inside it; symbolicOrder parses after
            # create, so read the XML directly).
            _so_raw = TheXMLConfig.get("architecture.symbolicOrder",
                                       default=None)
            try:
                _so_v = int(_so_raw) if _so_raw is not None else 0
            except (TypeError, ValueError):
                _so_v = 0
            _ser_raw = TheXMLConfig.get("architecture.serial", default=None)
            if _ser_raw is None:
                _ser_v = _so_v > 0
            elif isinstance(_ser_raw, bool):
                _ser_v = _ser_raw
            else:
                _ser_v = str(_ser_raw).strip().lower() in (
                    "true", "1", "yes", "on")
            if (self.wholePropertyBasis
                    and getattr(self, "concept_binding", "mixing")
                    == "aligned"):
                # Canonical aligned models address concepts by CS inventory
                # row. Every stage therefore owns the full preallocated CS
                # dictionary; WholeSpace's much smaller property basis is not
                # a fallback source of concept rows.
                stage_space_concept = [spaceShape_concept[0],
                                       spaceShape_concept[1]]
            elif _so_v > 0 and not _ser_v:
                stage_space_concept = [spaceShape_concept[0],
                                       spaceShape_concept[1]]
            else:
                stage_space_concept = [cs_out[0], spaceShape_concept[1]]
                # Concept index-read (snap design doc §ontology): the
                # concept inventory must seat the word/object concepts
                # (≈2× distinct words at order 0 + the METAs in the pool)
                # — vocabulary-scaled, not tile-scaled. The serial arm's
                # tile-count sizing starved it (measured: 4 words onto
                # caps0=2 rows COLLIDED and the read hurt). Honor an
                # EXPLICIT <ConceptualSpace><nVectors> as a floor, gated
                # on <conceptIndexRead> so every existing config keeps
                # byte-identical shapes.
                _explicit_word_traversal = TheXMLConfig.get(
                    "architecture.serialWordCapacity", default=None)
                if (bool(TheXMLConfig.get(
                        "architecture.conceptIndexRead", default=False))
                        or _explicit_word_traversal not in (None, "")):
                    _nv_cs = self._nvec("ConceptualSpace", 0)
                    if _nv_cs > 0:
                        stage_space_concept[0] = max(
                            int(stage_space_concept[0]), int(_nv_cs))
            # In the canonical property-basis mode every quantized WholeSpace
            # stage draws from the configured property inventory. It is
            # independent of live slot geometry and ConceptualSpace capacity.
            # This is the initial physical seed only; later dynamic property
            # growth must rebuild optimizer/compiled ownership at an explicit
            # reset boundary. Legacy configs retain their prior terminal-only
            # inventory sizing until explicitly migrated.
            _ws_codebook_mode = Space.normalize_codebook_mode(
                TheXMLConfig.space(
                    "WholeSpace", "codebook", default="quantize"))
            _whole_property_codebook = (
                self.wholePropertyBasis
                and _ws_codebook_mode == "quantize")
            _legacy_terminal_dictionary = (
                not self.wholePropertyBasis
                and is_last and _ws_codebook_mode == "quantize")
            stage_space_symbol = [
                (spaceShape_symbol[0]
                 if (_whole_property_codebook
                     or _legacy_terminal_dictionary) else ws_out[0]),
                spaceShape_symbol[1],
            ]
            # Right-half loopback widening retired (see ConceptualSpace
            # docstring): per-order input sourcing replaces the concat,
            # so the C-space_role PiLayer input width is just nInputDim.
            cs = ConceptualSpace(
                cs_in, stage_space_concept, cs_out,
                stage_idx=t, is_last=is_last,
                shared_similarity_codebook=(
                    _shared_concept_dictionary
                    if _share_concept_dictionary else None),
                indexed_similarity_codebook=_aligned_codebook_sources,
            )
            if _shared_concept_fold_support is not None:
                object.__setattr__(
                    cs, "_concept_fold_support",
                    _shared_concept_fold_support)
            if _share_concept_dictionary and _shared_concept_dictionary is None:
                _shared_concept_dictionary = cs.similarity_codebook
            if _aligned_codebook_sources:
                # The serial aligned path reaches the million-row dictionary
                # only through indexed concept identities.  Preserve that
                # sparsity in backward so the dedicated row-local optimizer
                # never allocates a dense 1M x ConceptDim gradient/moment set.
                cs.similarity_codebook.sparse_lookup_grad = True
            ws = WholeSpace(ws_in, stage_space_symbol, ws_out,
                               conceptualSpace=cs)
            # Non-owning back-ref CS->WS (mirrors the perceptualSpace_ref
            # idiom below): object.__setattr__ so it is NOT registered as
            # an nn.Module child of cs. Read-only structural pairing.
            object.__setattr__(cs, 'wholeSpace_ref', ws)
            # Per-stage flags consumed by build_pipelines / forward.
            ws.is_last = is_last
            ws.quantize = not is_last
            self.conceptualSpaces.append(cs)
            self.wholeSpaces.append(ws)

        # Backwards-compat aliases: read-only callers (e.g.
        # symbolSpace.truth_layer = self.wholeSpace) see the terminal stage.
        # Terminal convenience aliases are routing references; the ModuleLists
        # above are the sole structural owners.
        object.__setattr__(self, 'conceptualSpace', self.conceptualSpaces[-1])
        object.__setattr__(self, 'wholeSpace', self.wholeSpaces[-1])

        # Per-tower fold-width law (2026-07-16): every sigma/pi slot fold is
        # sized at its OWN space's CONTENT width (nDim). Dense "full" slab
        # folds (nInput == sigma_pi_slab) are the one exemption: their content
        # law is the flattened N*content slab. PS and WS are peer perceptual
        # towers and therefore retain one shared native width. Their sparse
        # sigma/codebook activations are already conceptual-width when they
        # reach CS; neither peer is widened here and there is no PS->WS edge.
        _law_folds = []
        if getattr(self, 'perceptualSpace', None) is not None:
            _law_folds.append(("PartSpace.sigma", self.perceptualSpace,
                               getattr(self.perceptualSpace, 'sigma', None)))
        for _t, _lw in enumerate(self.wholeSpaces):
            _law_folds.append((f"WholeSpace[{_t}].pi", _lw,
                               getattr(_lw, 'pi', None)))
        _law_widths = {}
        for _nm, _sp, _fold in _law_folds:
            _fw = int(getattr(_fold, 'nInput', 0) or 0)
            _nd = int(getattr(_sp, 'nDim', 0) or 0)
            if _fw <= 0 or _nd <= 0:
                continue
            if _fw == int(getattr(_sp, 'sigma_pi_slab', 0) or 0):
                continue
            assert _fw == _nd, (
                f"unified fold-width law violated: {_nm}.nInput={_fw} != "
                f"nDim={_nd} of its space; slot folds are sized at the "
                f"space's content width (one parameter).")
            _law_widths[_nm] = _fw
        assert len(set(_law_widths.values())) <= 1, (
            f"unified fold-width law violated across spaces: {_law_widths}; "
            "PS/WS slot folds must share one native content width. The "
            "conceptual dimension increase belongs to each source's sparse "
            "sigma/codebook activation, not to either perceptual tower.")

        # §6d reference-partitioned codebook update law (GrammarOpsPass;
        # author 2026-06-11): percepts are shaped by the parallel pass,
        # references by the serial pass. Install on both towers with a
        # LAZY table getter (the binding table is created on first gate
        # use; the mask stays None — legacy behavior — until then or
        # with the meronomy off). PS/extent tower: bound OBJECT ids are
        # references; WS/intent tower: bound WORD ids (WS also
        # self-installs at table creation; this is idempotent).
        _law_ss = self.wholeSpace
        _law_get = lambda: getattr(_law_ss, 'reference_table', None)
        if getattr(self, 'perceptualSpace', None) is not None:
            self.perceptualSpace.install_reference_update_law(
                _law_get, side='object')
        _law_ss.install_reference_update_law(_law_get, side='word')

        # §5 intent priming (GrammarOpsPass; author 2026-06-11): wire
        # both codebook towers' row selection to the single current
        # intent (``set_intent``). Dark by construction: with no intent
        # set the producer returns None and recognition is
        # byte-identical to the unprimed path.
        if getattr(self, 'perceptualSpace', None) is not None:
            self.perceptualSpace.install_intent_priming()
        _law_ss.install_intent_priming()

        # VQ-VAE EMA / growing-codebook knob overrides per space.
        # Each Space's ``subspace.what`` may carry an internal
        # ``VectorQuantize`` (``.vq``); when XSD knobs are set, override
        # the defaults baked in at ``Codebook.addVectors`` time so
        # configs can tune EMA decay and dead-code retirement without
        # code edits.  ``codebookGrowthEpsilon`` is stashed as
        # ``.growth_epsilon`` on the VQ for runBatch to consult.
        for _sect, _sp in (
            ("PartSpace", self.perceptualSpace),
            ("ConceptualSpace", self.conceptualSpace),
            ("WholeSpace", self.wholeSpace),
        ):
            _cb = getattr(getattr(_sp, 'subspace', None), 'what', None)
            _vq = getattr(_cb, 'vq', None) if _cb is not None else None
            if _vq is None:
                continue
            _decay = TheXMLConfig.space(_sect, "commitmentDecay",
                                        default=None)
            if _decay is not None:
                _vq.decay = float(_decay)
            _retire = TheXMLConfig.space(_sect, "codebookRetire",
                                         default=None)
            if _retire is not None:
                _vq.codebook_retire = bool(_retire)
            _thr = TheXMLConfig.space(_sect, "codebookEmaDeadThreshold",
                                      default=None)
            if _thr is not None:
                _vq.threshold_ema_dead_code = int(_thr)
            _eps = TheXMLConfig.space(_sect, "codebookGrowthEpsilon",
                                      default=None)
            _vq.growth_epsilon = (float(_eps) if _eps is not None
                                  else 0.0)

        # Whole-property VQ training: the canonical property-basis mode has
        # exactly one WS dictionary (``subspace.what``). It drops EMA and
        # learns selected properties from the reconstruction gradient. Legacy
        # configs temporarily retain their separate ``analysis_store`` update
        # path for checkpoint compatibility; it is never involved in aligned
        # concept-capacity growth.
        for _ws_stage in self.wholeSpaces:
            _ws_codebooks = [getattr(
                getattr(_ws_stage, 'subspace', None), 'what', None)]
            if not self.wholePropertyBasis:
                _ws_codebooks.append(getattr(
                    _ws_stage, 'analysis_store', None))
            for _cb in _ws_codebooks:
                _vq = getattr(_cb, 'vq', None) if _cb is not None else None
                if self.wholePropertyBasis and _vq is None:
                    # Canonical properties are a direct learned Parameter,
                    # without VQ/EMA shadow tensors. WholeSpace constructs and
                    # registers W; enlist that one physical owner explicitly.
                    _W = (_cb.getW() if _cb is not None
                          and hasattr(_cb, 'getW') else None)
                    if (isinstance(_W, nn.Parameter)
                            and all(_W is not p for p in _ws_stage.params)):
                        _ws_stage.params += [_W]
                    continue
                if _vq is None:
                    continue
                _vq.ema_update = False
                _vq.commitment_weight = 0.0
                # The recon leg (input -> codebook, asymmetric-vq sec.4):
                # the WS property codebook trains by the EXACT reconstruction
                # gradient -- promote the VQ codebook to an nn.Parameter
                # (auto-registered on the VQ module) and enlist it in the
                # space's optimised params. The stage-0 unity path emits
                # the recon term (plain differentiable gather vs the
                # DETACHED pre-snap evidence: gradient lands on the
                # selected rows only; the argmin blocks the encoder leg);
                # the t>0 symbolic iteration emits the CS-leg twin.
                if not isinstance(_vq.codebook, nn.Parameter):
                    _vq.codebook = nn.Parameter(
                        _vq.codebook.detach().clone())
                    with torch.no_grad():
                        _vq._b_norms_sq.copy_(
                            (_vq.codebook.detach() ** 2).sum(dim=-1))
                _ws_stage.params += [_vq.codebook]

        # Cross-space forward inputs (perceptual + symbolic loop into C,
        # C→P feedback into P) are now passed as explicit ``forward``
        # arguments by the recurrent cell in ``_forward_body`` -- no
        # post-construction ``wholeSpace_ref`` / ``perceptualSpace_ref``
        # / ``conceptualSpace_ref`` plumbing. The WholeSpace lexicon
        # ref below is structural (vocabulary ownership), not forward
        # input, and is kept.
        # Lexicon ownership (post-lexicon-migration): wire every
        # WholeSpace stage to PartSpace so ``S.vocabulary``
        # and the orthographic-API methods reach the physical Embedding
        # that lives on PartSpace for input-pipeline reasons.
        for ws in self.wholeSpaces:
            object.__setattr__(ws, 'perceptualSpace_ref',
                               self.perceptualSpace)

        # Task G auto-META on word learning: the auto-bind moved from
        # PartSpace to ConceptualSpace. Each ``cs`` already has
        # ``wholeSpace_ref`` (wired above per stage); add the
        # matching ``perceptualSpace_ref`` so the stage-0 cs.forward
        # can read the pid grid stashed on
        # ``perceptualSpace_ref._forward_input['indices']``. The
        # back-ref points at the canonical PartSpace; the autobind
        # gate (``if int(self.stage_idx) == 0``) restricts firing to
        # the first stage where the pid grid still aligns with the
        # incoming subspace event.
        #
        # The autobind grows the META taxonomy, which is owned by the
        # TERMINAL WholeSpace (``self.wholeSpace`` ==
        # ``wholeSpaces[-1]``). Per-stage WS codebooks are slot-fixed
        # by the where-space registry and cannot grow without overrunning
        # a downstream slice -- so wire a separate
        # ``terminalSymbolSpace_ref`` distinct from the per-stage
        # ``wholeSpace_ref`` used by other CS consumers.
        for cs in self.conceptualSpaces:
            object.__setattr__(cs, 'perceptualSpace_ref',
                               self.perceptualSpace)
            object.__setattr__(cs, 'terminalSymbolSpace_ref',
                               self.wholeSpace)
            # serialObjectMeta reaches CS here (a NEW stamp site — _mereology_
            # raise never reaches CS; _maybe_autobind_meta reads it off self).
            # Read the live config directly: _create_per_stage runs DURING
            # construction, before self.serial_object_meta is parsed (same
            # order-safety idiom as the mereologyRaise request below).
            object.__setattr__(cs, '_serial_object_meta', bool(
                TheXMLConfig.get("architecture.serialObjectMeta", default=False)))
            # Model-level incoming assertion trust reaches CS here too for
            # probes/tests that need the architecture value during construction.
            object.__setattr__(cs, '_trust', self._unit_interval(
                TheXMLConfig.get("architecture.trust", default=1.0),
                default=1.0))
            # ltmConsolidation reaches CS here too (same NEW stamp site). When
            # on, _route_learned_relation's ineffable branch appends to the
            # unified ltm_store instead of the (retired) RelativeTruthStore, and
            # reason/verify_relation select the ltm_store. Read the live config
            # directly (order-safe during construction, before
            # self.ltm_consolidation is parsed).
            object.__setattr__(cs, '_ltm_consolidation', bool(
                TheXMLConfig.get("architecture.ltmConsolidation",
                                 default=False)))

        # S3 relocation (2026-06-17): ConceptualSpace is the OWNER of the
        # relation-only symbol table / taxonomy (doc/specs/mereological-order-
        # raising.md "relation-only completion"). Like the META taxonomy, the
        # relation table is a SINGLE TERMINAL owner -- the terminal CS
        # (``conceptualSpaces[-1]`` == ``self.conceptualSpace``) -- so wire a
        # ``terminalConceptualSpace_ref`` onto every CS and onto legacy
        # symbol-dictionary WS instances, mirroring the
        # ``terminalSymbolSpace_ref`` fan-out, so any symbolic stage can reach
        # the one canonical relation owner without fragmenting it into N
        # tables. A property-basis WS is strictly upstream and receives no
        # downstream ConceptualSpace pointer.
        # (Fix #1 ownership-by-reference: the physical position-keyed dicts are
        # still served from the terminal WholeSpace, which mints the codebook
        # rows atomically; CS owns the relation INTERFACE. A full physical move
        # is blocked on retiring the meta-vector seed -- a gradient-path change,
        # deferred -- and is NOT attempted here.)
        for sp in self.conceptualSpaces:
            object.__setattr__(sp, 'terminalConceptualSpace_ref',
                               self.conceptualSpace)
        for sp in self.wholeSpaces:
            if not bool(getattr(sp, 'property_basis', False)):
                object.__setattr__(sp, 'terminalConceptualSpace_ref',
                                   self.conceptualSpace)

        # Wire the back-ref WS <- Embedding (terminal stage:
        # ``wholeSpaces[-1]`` == ``self.wholeSpace``) so the
        # vocabulary/orthographic API delegation keeps working. The
        # Stage-1.B paired-row contract this ref used to serve
        # (``insert_paired_word`` on PS-side OOV inserts) was RETIRED in
        # Step 3 of the 2026-06-10 symbolic-iteration plan.
        terminal_ss = (self.wholeSpaces[-1]
                       if self.wholeSpaces else None)
        emb = getattr(self.perceptualSpace, 'vocabulary', None)
        if terminal_ss is not None and isinstance(emb, Embedding):
            object.__setattr__(emb, 'wholeSpace_ref', terminal_ss)
            # Step 3 (2026-06-10 symbolic-iteration plan): the tied-
            # storage migration (``_tie_lexicon_to_codebook`` -- the
            # PS->WS reach-across that wrote an orth row + random
            # semantic partner per lexicon word and remapped
            # ``key_to_index`` onto WS rows) is RETIRED. The Step-1
            # symbol codebook on the CS leg captures the code-as-written
            # vs code-for-the-concept correspondence in place; the
            # lexicon keeps PS-LOCAL storage permanently (untied), and
            # the decode resolves row->word through the inverse of
            # ``key_to_index`` (identity when untied).

        # No SyntacticSpace -- syntax is handled by Grammar centrally.
        self.syntacticSpace = None

        # Output: primary path is IS→PS→CS→OS, so OutputSpace consumes
        # the terminal ConceptualSpace output (WholeSpace is the
        # symbolic recurrent loop leg, off the head path). Size the head
        # from the terminal C stage's ACTUAL ``outputShape`` (the
        # source of truth) rather than a recomputed
        # ``concept_dim + obj_concept``. The bivector regime was retired
        # (2026-05): the terminal C stage emits a single signed scalar
        # per prototype, so ``.nOutputDim == .outputShape[1]`` for every
        # config and the old stale-width mismatch ("Bug #1") is
        # structurally gone -- this site needs no special-casing.
        _term_cs_shape = list(self.conceptualSpaces[-1].outputShape)
        output_n = int(_term_cs_shape[0])
        outputInputShape = [output_n, int(_term_cs_shape[1])]
        self.outputSpace = OutputSpace(outputInputShape, spaceShape_output, outputShape,
                                       vectors=self.perceptualSpace.vocabulary)

        self._symbol_shape = [nPercepts, percept_dim + obj_percept]

        # Build SymbolSpace -- the unified container for grammar
        # infrastructure (SymbolSubSpace, three SyntacticLayers, the
        # TruthLayer, and conditionally the DiscourseSpace substrate).
        # Its ``__init__`` configures the grammar, sizes the word
        # buffer from ConceptualSpace's column layout, builds each space_role's
        # SyntacticLayer, and back-wires the home spaces so
        # compose/decompose routes through ``self.symbolSpace``.
        self.symbolSpace = SymbolSpace(
            perceptualSpace=self.perceptualSpace,
            conceptualSpace=self.conceptualSpace,
            wholeSpace=self.wholeSpace,
            nPercepts=nPercepts,
            nConcepts=nPercepts,
            nSymbols=nSymbols,
            concept_dim=concept_dim + obj_concept,
            # Symbol identities/operators live in CS WHAT coordinates.  The
            # 128-D WholeSpace property basis is upstream and never sizes SS.
            symbol_dim=concept_dim,
        )

        # MetaSymbol Category codebook (doc/Language.md
        # "Participation Categories"). Enabled by default via
        # <categoryCodebook>: in property-basis mode request the small
        # role-space VQ from terminal ConceptualSpace, which owns concepts and
        # META structure. Legacy configs keep the prior WholeSpace host until
        # they opt into the migration. Role columns are enumerated from the
        # now-configured grammar (SymbolSubSpace.__init__ ran
        # _ensure_configured above).
        # Read the flag from config HERE (not self.category_codebook): the
        # architecture flags are assigned in create_from_config AFTER the
        # spaces are built, so self.category_codebook is not yet set when
        # _create_per_stage runs -- reading the live config is the order-safe
        # source (the config is loaded before any space is created).
        _category_codebook = bool(TheXMLConfig.get(
            "architecture.categoryCodebook", default=True))
        category_owner = (
            self.conceptualSpace
            if self.wholePropertyBasis else terminal_ss)
        if _category_codebook and category_owner is not None:
            # REQUEST allocation; the actual enable runs LAZILY from the
            # autobind hook on the first perception forward, when TheGrammar is
            # fully configured with the (role-collapsed) operator roles. At
            # build the role rules may not be loaded yet, so a build-time
            # enable would see 0 roles and no-op.
            category_owner._category_codebook_requested = True

        # Mereological order-raising (doc/specs/mereological-order-raising.md).
        # Opt-in via <mereologyRaise>: build a meronymic lattice in perception
        # and raise abstraction order as attention requires. Read the LIVE
        # config (the self.* arch flags are not yet assigned when
        # _create_per_stage runs). Stamp the flag onto the PartSpace +
        # terminal WholeSpace so the perception hot path reads it off
        # ``self`` (the _symbolic_order stamping convention).
        # NOTE (todo "make abstraction order canonical"): the ramsification
        # table is NO LONGER gated here -- Codebook.create allocates it for
        # every built codebook with max_order = max(1, subsymbolicOrder),
        # and the sigma/pi processing sites stamp it live. This flag now
        # gates ONLY the lattice behaviors (word-whole autobind, order
        # raising, the top-down handoff).
        _mereology_raise = bool(TheXMLConfig.get(
            "architecture.mereologyRaise", default=False))
        if _mereology_raise and terminal_ss is not None:
            object.__setattr__(terminal_ss, '_mereology_raise', True)
            object.__setattr__(self.perceptualSpace, '_mereology_raise', True)
            # Stamp the STAGE-0 WholeSpace too so it computes the read-only
            # run-structure observation (``_mereology_ratio_obs``) the top-down
            # attention handoff dispatches on -- at sO>1 the stage-0 ws is NOT
            # the terminal, and only the stage that runs ``_stage0_unity_forward``
            # parks the obs. wholeSpaces[0] runs ONLY the unity branch (never the
            # recurrent / autobind mutating path), so this enables the obs alone
            # (byte-identical). Idempotent when stage 0 IS the terminal (sO=1).
            _ws0 = self.wholeSpaces[0] if len(self.wholeSpaces) else None
            if _ws0 is not None:
                object.__setattr__(_ws0, '_mereology_raise', True)
        # <radialStmReduce> (Alec 2026-07-13, "min should be a radial min
        # to deal with signed activations"): the STM idea folds use the
        # RADIAL kernels (radmin/radmax) and their reverses use the radial
        # recommender filters. Stamped on the terminal WS grammar hosts
        # the bounded reducer adapts; identical to the lattice pair on the
        # non-negative SS activations, signed-safe on CS ideas. Default
        # off = byte-identical.
        if bool(TheXMLConfig.get("architecture.radialStmReduce",
                                 default=False)) and terminal_ss is not None:
            _sl = getattr(terminal_ss, 'syntacticLayer', None)
            for _h in (getattr(_sl, '_by_name', {}) or {}).values():
                if getattr(_h, 'rule_name', '') in ('conjunction',
                                                    'disjunction'):
                    object.__setattr__(_h, 'radial', True)

        # The conceptual basis honours the ``architecture.monotonic``
        # knob rather than being unconditionally bitonic: monotone
        # (W>=0) is order-preserving, which the parthood predicate
        # (``Ops.part``) requires for the ramsified symbolic codebook
        # to match a symbol mapped through the subsymbolic loop across
        # orders. Default False -> unchanged (bitonic) behavior.
        self.conceptualSpace.subspace.basis.monotonic = self.monotonic

        self.spaces.extend([self.inputSpace, self.perceptualSpace])
        self.spaces.extend(list(self.conceptualSpaces))
        self.spaces.extend(list(self.wholeSpaces))
        self.spaces.extend([self.outputSpace])
        self.spaces.append(self.symbolSpace)

        # Bind stable, unique owner paths before any carrier reader is issued.
        # The terminal aliases (``conceptualSpace`` / ``wholeSpace``) are not
        # used because the ModuleLists are the structural owners.
        _carrier_owners = [
            ('inputSpace', self.inputSpace),
            ('perceptualSpace', self.perceptualSpace),
            *[
                (f'conceptualSpaces.{i}', space)
                for i, space in enumerate(self.conceptualSpaces)
            ],
            *[
                (f'wholeSpaces.{i}', space)
                for i, space in enumerate(self.wholeSpaces)
            ],
            ('outputSpace', self.outputSpace),
            ('symbolSpace', self.symbolSpace),
        ]
        for _owner_path, _space in _carrier_owners:
            binder = getattr(_space, 'bind_codebook_owner_path', None)
            if callable(binder):
                binder(_owner_path)

        self.inputSpace.outputSpace = self.outputSpace
        # Seed the pipeline context: InputSpace stamps every outgoing
        # subspace's ``symbolSpace`` with this reference so downstream stages
        # read ``vspace.symbolSpace`` instead of reaching back through a
        # Model back-channel.
        self.inputSpace.set_word_space(self.symbolSpace)

        # Phase 1: wire a Normalizer onto every space so spaces can call
        # self.normalizer.{normalize,denormalize} instead of the TheData global.
        # Phase G of doc/specs/2026-05-21-wordsubspace-stm-layer-refactor.md
        # retired the per-SubSpace ``symbolSpace`` back-pointer; the
        # SymbolSubSpace reference lives on each ``Space`` via the routing
        # pointer set by ``Space.attach_symbolSpace``. SymbolSubSpace's
        # constructor wires P/C/S spaces; here we mirror that wiring onto
        # every other space (InputSpace / OutputSpace / ModalSpace) so
        # ``space.symbolSpace`` is non-None project-wide.
        self.normalizer = Normalizer(TheData)
        for space in self.spaces:
            space.normalizer = self.normalizer
            sub = getattr(space, 'subspace', None)
            if sub is not None:
                sub.normalizer = self.normalizer
            if (space is not self.symbolSpace
                    and not (self.wholePropertyBasis
                             and isinstance(space, WholeSpace))
                    and getattr(space, 'symbolSpace', None) is None
                    and hasattr(space, 'attach_symbolSpace')):
                space.attach_symbolSpace(self.symbolSpace)

        # Slice C: the per-stage ConceptualSpace cells (which run bind_streams)
        # need the SymbolSpace ref so the CS-mediated symbol leg
        # (ConceptualSpace._build_symbol_leg) can sync the order-raising codes
        # onto SS.subspace.what. GATED on symbol_tower so off-path per-stage CS
        # keep _model_symbolSpace=None -> byte-identical (the autobind's
        # _model_symbolSpace read at Reset is unaffected for non-tower configs).
        if bool(TheXMLConfig.get("architecture.symbolTower", default=False)):
            for _cs in getattr(self, 'conceptualSpaces', ()):
                if getattr(_cs, '_model_symbolSpace', None) is None:
                    object.__setattr__(_cs, '_model_symbolSpace', self.symbolSpace)

        # Precompute partition boundaries for partitioned symbolSum
        self._partitions = self._order_partitions(concept_dim,
                                                   self.subsymbolicOrder)
        self.symbol_states = []
        # Per-step lifecycle slots, initialized explicitly so every
        # read is a plain attribute access (no getattr-with-default):
        #   _staged_in_sub      -- lex+embed stem subspace parked by
        #                          runBatch for the compiled forward
        #                          (None == not staged: eager/uncompiled
        #                          path lexes inline).
        #   _compiled_step      -- the torch.compiled callable, or None
        #                          (set by enable_compiled_step; None
        #                          here so the eager path is valid even
        #                          if that is never called).
        #   _current_discourse_s -- S-space_role sentence rep stashed by the
        #                          forward for the post-body ARMA term.
        #   _staged_concepts_in -- the UNITY view [B, 1, N] parked by
        #                          _lex_embed_stem for the symbolic
        #                          branch (analysis/synthesis dual-input
        #                          plan; Phase 2 consumes it at WS
        #                          stage 0).
        self._staged_in_sub = None
        self._staged_concepts_in = None
        self._compiled_step = None
        self._compiled_word_steps = {}
        self._active_compiled_step = None
        self._compiled_word_chunk_step = None
        self._compiled_word_chunk_active = False
        self._compiled_word_chunk_replaying = False
        self._compiled_word_chunk_width = 0
        self._compiled_part_fold_ladder = None
        self._compiled_whole_fold_ladder = None
        self._pending_stm_end_state = None
        self._current_discourse_s = None
        # A6 stage-0 CS_{-1} interSentence seed staging (mirrors
        # _staged_in_sub): the predicted next-end-state tuple parked by
        # runBatch for the compiled forward (predict_next_end_state is
        # @torch.compiler.disable'd, so it cannot run in-trace). The flag
        # distinguishes "staged None" (cold / none-mode) from "not staged"
        # (eager path computes live). Both reset by _end_step.
        self._staged_intersentence_seed = None
        self._intersentence_seed_staged = False
        # Forward-local verification handle (Task A6 test): the seed payload
        # actually used on the last forward (None == empty seed).
        self._intersentence_seed_payload = None
        # MPHF instrumentation handles -- always initialised so the
        # per-word body can use direct attribute reads (no getattr
        # fallback, which is a Dynamo graph break).
        self._mphf_last_idx = None
        self._mphf_call_count = 0
        # Persistent empty CS seeds for the recurrent cell's pass-0
        # (prevCS_forPS / prevCS_forSS). Built ONCE here instead of
        # constructing a fresh SubSpace every _forward_body call: the
        # per-forward object churn changes traced object-identity and
        # is the prime suspect for splitting the compiled forward into
        # 2 guarded graphs. These are zero-length ``is_empty()``
        # sentinels; the recurrent cell only ``.is_empty()`` /
        # ``.materialize()``s them (consumers write their own subspace,
        # never the seed), so a shared persistent instance is
        # behaviour-identical to the prior fresh-per-forward seed.
        self._empty_seed_ps = self._empty_subspace()
        self._empty_seed_ss = self._empty_subspace()

        # Phase 2: Sequential pipeline is the only path.
        self.build_pipelines()

        self.to(TheDevice.get())
        TheXMLConfig.validate()

    # -- Phase 2: Sequential pipeline ----------------------------------
    #
    # ``_flatten_k`` / ``_restore_k`` retired 2026-05-13 together with
    # the AR cursor unfold. The body and head now operate on
    # B-shaped tensors throughout; per-cursor predictions are produced
    # by the serial K-loop in ``_forward_per_stage_no_unfold``.

    def _build_pipelines_per_stage(self):
        """Phase 2: stem/body/head pipelines for BasicModel.

        Pipeline shape (microbatch AR):
            stem  : nn.Sequential(inputSpace, FlattenK(perceptualSpace))
                    + per-word P->C->S->C round trip filling
                    ConceptualSpace.stm
            body  : explicit for-loop over ``self.body_stages``
                    (ModuleList of ModuleDicts), one dict per stage:
                      cs -> chart-at-C -> [merge?] -> ws
                    K-axis flatten/restore is hoisted into _forward_body.
            head  : FlattenK(outputSpace)

        T = len(self.conceptualSpaces). Per-stage WholeSpace and
        ConceptualSpace outputs are persisted directly on the stage
        spaces' ``.subspace`` — the per-stage forward capture lists
        ``_ws_cache`` / ``_cs_cache`` were retired by Stage 1.F of the
        two-loop pi/sigma substrate refactor (doc/plans/
        2026-05-26-two-loop-pi-sigma-substrate.md). Reverse-pass
        consumers read the terminal C-space_role idea from
        ``self.conceptualSpace.stm.snapshot()``; ``symbol_states`` is
        rebuilt by iterating ``self.wholeSpaces`` directly.
        ``self.symbol_cache`` is a property returning the terminal
        WholeSpace's ``.subspace`` — same role as before, no longer
        a CachePoint module and no longer a per-stage list lookup.
        """
        T = len(self.conceptualSpaces)
        use_grammar_merge = (self.useGrammar == "all")

        # PartSpace quantize: matches legacy's percept_quantize logic.
        # Reversible+invertible configurations skip codebook quantization to
        # preserve exact-invertibility of the perceptual step.
        self.perceptualSpace.quantize = not (
            self.reversible and getattr(self.perceptualSpace, "invertible", False))

        # Compatibility-only ownership: legacy WholeSpace doubled as the
        # symbolic stage. Canonical property-basis WS is strictly upstream and
        # must not retain a downstream SymbolSpace pointer.
        if not self.wholePropertyBasis:
            self.wholeSpace.attach_symbolSpace(self.symbolSpace)

        # Stage 1.F substrate refactor (doc/plans/2026-05-26-two-loop-
        # pi-sigma-substrate.md): the per-stage forward capture lists
        # ``_ws_cache`` / ``_cs_cache`` are retired entirely. The
        # terminal C-space_role idea built up over the sentence lives on
        # ``self.conceptualSpace.stm`` (ShortTermMemory snapshot); the
        # terminal symbolic subspace lives directly on
        # ``self.wholeSpace.subspace`` (each stage's
        # ``ws.forward(...)`` writes there in place). Both
        # ``symbol_cache`` and the reverse-pass reconstruction loss
        # consumer were migrated to read from those canonical owners.

        # Determine initial_n per stage for the N-halving GrammarMergeGlue.
        try:
            base_n = int(self.perceptualSpace.subspace.inputShape[0])
        except Exception:
            base_n = int(getattr(self, "nPercepts", 0))

        # Per-stage body: ModuleList of ModuleDicts, driven by an
        # explicit for-loop in ``_forward_body`` / ``_reverse_body``.
        # Replaces ``_body_inner = nn.Sequential(*body_modules)``; the
        # adapter classes (FlattenKWrapper, ReverseAdapter, CachePoint)
        # that existed only to fit Sequential's one-arg contract go
        # away. Each stage's dict contains:
        #   "cs":      ConceptualSpace   (required)
        #   "merge":   GrammarMergeGlue  (optional, useGrammar=="all")
        #   "ws":      WholeSpace     (required)
        # The legacy ``"reparse": ChartCompose`` entry was retired
        # 2026-05-12 alongside chart-at-stem -- the chart now fires
        # uniformly at C-space_role inside ``_forward_body`` for every stage.
        self.body_stages = nn.ModuleList()
        for t in range(T):
            merge = None
            if use_grammar_merge:
                stage_n = base_n // (2 ** t)
                merge = GrammarMergeGlue(
                    stage_idx=t, initial_n=stage_n,
                    is_last=(t == T - 1))
            self.body_stages.append(
                _BodyStage(
                    self.conceptualSpaces[t],
                    self.wholeSpaces[t],
                    merge=merge,
                )
            )

        # --- A4 (2026-06-06 parallel-conceptual-recurrence): per-stage
        # ConceptualCombine modules. One SQUARE augment-threaded invertible
        # combine per stage replaces the per-stage ``cs.forward`` content
        # fold + the prior WS.sigma (pi/sigma) alternation in the PARALLEL
        # ``_forward_body`` (doc/plans/2026-06-06-parallel-conceptual-
        # recurrence.md sec. 3.3). Built ONCE here and registered on its
        # ConceptualSpace (``cs.layers`` / ``cs.params``) so the LDU /
        # butterfly weights are optimised with the rest of the model -- they
        # must NOT be constructed inside ``forward`` (that re-inits the
        # weights every call and loses learning).
        #
        # ``content_dim`` D is the per-stage FULL MUXED EVENT width
        # ``cs.muxedSize`` (== ``nWhat + nWhere + nWhen``), NOT the demux'd
        # content. Option B (doc/specs/2026-06-06-muxed-events-and-positional-
        # bands.md): the combine transforms the WHOLE muxed event -- content
        # AND the .where/.when band TOGETHER -- so the positional tail
        # PARTICIPATES in the conceptual advance instead of riding along
        # untouched. (That is the whole point of muxing where/when in.)
        #
        # DIMENSIONAL-GOVERNANCE NOTE (MM_20M today): PS and CS events are
        # both ``muxedSize = 1024`` (``[B, 8, 1024]``, 1020 what + 2 where +
        # 2 when), but the WholeSpace OUTPUT is a COMPRESSED symbol code
        # (``ws.nOutputDim = 8``) -- the WS event width does NOT match the
        # CS/PS event width. This is the wide<->deep symbol handoff: WS emits
        # an 8-wide code that the combine (like ``ConceptualSpace.forward``'s
        # STM) zero-pads up to D. The combine therefore sizes D to the
        # CONCEPTUAL event width (``cs.muxedSize``) -- the dominant stream and
        # the carrier the head consumes -- and the WS stream is fit (zero-
        # padded) up to D before the combine (see ``_combine_demux`` /
        # ``_combine_fit`` in ``_forward_body``).
        # The flat-slab invariant holds for the PS/CS carriers (both 1024);
        # the WS-narrower-than-D case is handled by the pad rather than
        # blocking the integration.
        # ``naive=False`` is the design's REQUIRED reverse path (exact
        # structured ``_solve_ldu`` / closed-form butterfly inverse, NOT
        # pinv). Read the architecture knob only to honour an explicit
        # debug override; default False.
        #
        # OWNERSHIP (2026-06-06): the combine is the C-space_role conceptual-advance
        # operator, so each ``ConceptualSpace`` HOLDS its own (cf. PS owns
        # ``pi``, WS owns ``sigma``). Built here -- where the per-stage config
        # (sigmaPi span, naive/ergodic, merge gating) is resolved -- then
        # registered ON the cs: appended to ``cs.layers`` (so ``paramUpdate``/
        # ``set_sigma`` cascade through ``Space``) and its params added to
        # ``cs.params`` (so the ``self.spaces`` getParameters() walk optimises
        # it). ``cs.combine`` is the access handle; there is NO model-level
        # ``conceptual_combines`` list -- the combine learns through its cs,
        # like every other Space layer.
        _combine_naive = bool(TheXMLConfig.get("architecture.naive", False))
        _combine_binding = str(TheXMLConfig.get(
            "architecture.conceptBinding", default="mixing")
        ).strip().lower()
        for t in range(T):
            cs = self.conceptualSpaces[t]
            if _combine_binding != "mixing":
                # Location-aligned binding is parameter-free and retains its
                # source carrier directly on CS.  Do not allocate a dormant
                # ConceptualCombine: the mixing matrix exists only for the
                # explicitly selected parallel ablation.
                object.__setattr__(cs, "combine", None)
                continue
            # Slot-stack bind (geometry corrected 2026-06-10): the streams
            # stack along the VECTOR axis (N slots each, 2N total) and ONE
            # cascade runs over the flattened 2N*D slab -- 16*nDim for the
            # production parallel config (= 2^14 exactly, zero pad):
            # cross-slot reach. D is the FULL muxed event width
            # (cs.muxedSize == the config nDim: content + where/when band)
            # so the band PARTICIPATES in the bind (option B, as in the
            # 3-stream design) -- the spec's 16*nDim arithmetic is the
            # event width.
            D_t = int(getattr(cs, "muxedSize", 0))
            if D_t < 1:
                # Defensive: fall back to the concept buffer / content width
                # so the combine is always constructible (D>=1 is the bar).
                D_t = max(1, int(getattr(cs, "concept_dim",
                                         getattr(cs, "nWhat", 1))))
            # The combine binds the DEEP conceptual hub (n_vectors slots of
            # the muxed event width D_t), NOT the wide per-stage position
            # grid. For flat-slab wide<->deep configs (e.g. MM_20M_grammar:
            # wide PS [1024, 8] reshapes to deep CS [8, 1024]) the grammar
            # N-halving path inflated ``cs.outputShape[0]`` to the WIDE
            # position count (nPercepts>>t = 512/256), so the cascade ballooned
            # to next_pow2(2 * 512 * 1028) = 2^21 (~339M params, ~114s build)
            # binding phantom slots that never carry deep content. Cap the slot
            # count at the deep conceptual hub ``nConcepts``: bind_streams'
            # ``_bind_regroup`` already reshapes the wide legs into n_vectors
            # deep slots, so the runtime data is unchanged -- only the cascade
            # is sized to the real 8-slot hub (MM_20M's 2^15-scale combine).
            N_t = max(1, int(cs.outputShape[0]))
            _deep_n = int(getattr(self, "nConcepts", 0))
            if _deep_n > 0 and _deep_n < N_t:
                N_t = _deep_n
            combine = ConceptualCombine(
                content_dim=D_t,
                n_vectors=N_t,
                naive=_combine_naive,
                sigma_pi_mode=self.sigma_pi_mode,
                ergodic=self.ergodic,
                # Slice B: 3-stream when symbolTower is on. Read the config
                # DIRECTLY (not self.symbol_tower) -- the per-stage pipelines are
                # built by _create_per_stage BEFORE create_from_config sets
                # self.symbol_tower (line ~794), so the attr isn't live yet here.
                # combine.n_streams is what the bind keys on; the forward-time
                # loop gate (self.symbol_tower) is set by the time forward runs.
                n_streams=(3 if bool(TheXMLConfig.get(
                    "architecture.symbolTower", default=False)) else 2),
                # Concepts live on the unit ball: when ConceptualSpace
                # <nonlinear> is set, the per-stage combine saturates the bound
                # concept with tanh (forward) / atanh (reverse) -- the joint
                # nonlinearity a non-separable readout (XOR) needs. Read the
                # config directly (cs.nonlinear may not be live at per-stage
                # build time, matching the symbolTower read above).
                nonlinear=bool(TheXMLConfig.space(
                    "ConceptualSpace", "nonlinear")))
            # Held by the ConceptualSpace: register for paramUpdate/set_sigma
            # (cs.layers) and optimisation (cs.params); expose as cs.combine
            # WITHOUT a second nn.Module registration (object.__setattr__,
            # mirroring the wholeSpace_ref idiom -- cs.layers is the real
            # child registration).
            cs.layers.append(combine)
            cs.params += combine.getParameters()
            object.__setattr__(cs, "combine", combine)

        all_spaces = ([self.inputSpace, self.perceptualSpace]
                      + list(self.conceptualSpaces)
                      + list(self.wholeSpaces)
                      + [self.outputSpace])
        self.any_invertible = any(
            s.invertible if hasattr(s, "invertible") else False
            for s in all_spaces)

        # Forward midpoint cache. None for invertible (Case A); for
        # non-invertible (Case B) the forward result is stored here so
        # the round-trip path can rebuild from it. Plain attribute --
        # used to be a CachePoint module.
        self.midpoint_cache = None

    # BasicModel.End and BasicModel.Finish were dropped 2026-05-05:
    # End() was identical to BasicModel.End; Finish() differed only in
    # skipping the optional plotActivations call, which is gated on
    # ``self.plot`` (False by default) so inheriting BasicModel.Finish
    # is a no-op behavior change for the per-stage path.

    @property
    def symbol_cache(self):
        """Terminal stage's symbolic subspace, or None if not yet set.

        Post Stage 1.F of the two-loop pi/sigma substrate refactor
        (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md): the
        per-stage ``_ws_cache`` capture list is retired. The terminal
        WholeSpace's ``.subspace`` is the canonical owner of the
        terminal symbolic state — each stage's ``ws.forward(...)``
        writes there in place — so this property resolves directly to
        ``self.wholeSpace.subspace``. Read-only consumer surface;
        same role as before, no per-stage list lookup.
        """
        ws = getattr(self, 'wholeSpace', None)
        if ws is None:
            return None
        return getattr(ws, 'subspace', None)





    def _forward_head(self, sub):
        """Head: outputSpace forward."""
        return self.outputSpace(sub)

    @staticmethod
    def _empty_subspace(d=1):
        """A zero-length SubSpace seed: ``is_empty()`` is True so the
        recurrent cell's pass-0 inputs short-circuit (PS uses pi_input
        alone; CS uses the perceptual primary alone; WS returns it
        unchanged)."""
        return SubSpace(inputShape=(0, d), outputShape=(0, d),
                        nInputDim=d, nOutputDim=d)

    @staticmethod
    def _zero_symbol_subspace(ws, ctx_sub):
        """"Nothing to quantize -> pass zeros": a correctly-batched
        zero WholeSpace output.

        When WS is fed an empty seed (e.g. subsymbolicOrder=1, where the
        only pass's ``prevCS_forSS`` is the empty seed) it returns the
        activation-less empty subspace, breaking the symbol-state
        contract (``end_state`` / discourse / ARMA / head). The cell
        knows the real batch (from ``ctx_sub`` = the percept-side input),
        so it emits a ``[B, N_S, D_S]`` zero symbol here. ``set_event``
        installs a (non-None) activation, satisfying the contract;
        zeros are an additive no-op into any downstream consumer.
        """
        sample = ctx_sub.materialize()
        if sample is not None and sample.dim() >= 1:
            B = int(sample.shape[0])
            dev, dt = sample.device, sample.dtype
        else:
            B = 1
            dev, dt = TheDevice.get(), torch.get_default_dtype()
        N = int(ws.outputShape[0])
        D = int(ws.subspace.muxedSize)
        # Write onto WS's OWN subspace (mirrors the held_at_zero idiom)
        # so ``model.wholeSpace.subspace`` carries the zero event +
        # activation -- the symbol-state contract inspects that object,
        # not just the cached return. Place the zeros on the model's
        # device/dtype (held_at_zero uses ``device=sample.device,
        # dtype=sample.dtype``); a bare ``torch.zeros`` is CPU and
        # mismatches a CUDA model (``cuda:0 vs cpu`` on metalbaby).
        ws.subspace.copy_context(ctx_sub)
        ws.subspace.set_event(torch.zeros(B, N, D, device=dev, dtype=dt))
        return ws.subspace

    def _forward_body(self, in_sub):
        """Recurrent cell: IS→PS→CS→OS with CS→PS and CS→WS loops.

        Per pass t over ``self.body_stages`` (T = subsymbolicOrder):

          * PS and WS run **in parallel** (no intra-pass dependency --
            WS still consumes the prior pass's CS view; PS is single-
            arg post-Stage-1.A substrate refactor):
              ``PS_sub = perceptualSpace.forward(in_sub)``
              ``WS_sub = ws.forward(prevCS_forSS)``
          * ConceptualSpace combines them (CS is the per-pass terminal):
              ``CS_sub = cs.forward(PS_sub, WS_sub)``
          * ``cs._subspaceForPS`` / ``cs._subspaceForWS`` (read
            post-merge) feed the next pass's subsymbolic / symbolic
            loops.

        Pass 0 seeds ``prevCS_*`` with empty subspaces so the cell
        degrades to PS-only / primary-only (matches the old
        ``ref is None`` cold start). The head consumes the terminal CS
        (``return last_cs``); WholeSpace is the symbolic loop leg,
        off the head path. ``in_sub`` is the stable InputSpace subspace
        (InputSpace ran once in the stem).

        When ``self.loss_head`` is set the post-body STM snapshot feeds
        the head and the loss is stashed on ``self._loss_head_loss``.
        """
        # Stage 1.E: explicit two-mode dispatch on
        # ``self.serial`` (XML knob ``<architecture><serial>``, parsed in
        # ``create_from_config``).
        #
        #   * ``True`` (= SERIAL / GRAMMATICAL) -- per-word IR-reconstruction
        #     loop via :meth:`_forward_body_per_word`. ONE forward = ONE
        #     sentence: each ground-truth word ``[B,1,D]`` is pumped
        #     through the SAME per-stage PS->CS->WS computation and the
        #     resulting per-word concept is SHIFTed onto
        #     ConceptualSpace.stm; the NULL seal (next_word -> None)
        #     ends the loop. The accumulated STM then feeds the EXISTING
        #     compose-to-S / chart / head / reverse() / IR-loss TAIL
        #     entirely unchanged.
        #
        #   * ``False`` (= PARALLEL) -- T iterations of the per-stage body
        #     (whole-slab path, T = ``<subsymbolicOrder>``), the legacy
        #     non-grammar path. Falls through to the loop below.
        #
        # Pre-Stage-1.E this dispatch was implicit (driven by the
        # InputSpace-side ``_per_word_enabled`` boolean derived from
        # ``useGrammar``); ``_per_word_enabled`` is now a back-ref
        # mirrored from ``self.serial`` (see
        # ``create_from_config``) and retained for the remaining InputSpace
        # / per-word-loop late-stage consumers.
        if self.serial:
            return self._forward_body_per_word(in_sub)

        # A5 fullgraph fix (doc/plans/2026-06-06-parallel-conceptual-
        # recurrence.md sec 2 & 4): seed a FRESH per-forward STM working
        # buffer as a trace INTERMEDIATE before any stage touches the STM.
        # The STM accumulates the per-stage slabs WITHIN this forward
        # (parallel mode writes all N slots per stage via
        # ``_stm_set_all_slots``); seeding fresh here is identical to the
        # sentence-boundary clear it already starts from, but makes the
        # buffer a graph intermediate (not a mutated/output-aliased graph
        # input), which is the load-bearing fullgraph fix -- nothing with an
        # oscillating ``requires_grad`` survives across forwards. The live
        # grad threads through the ``folded`` slab the combine/loss hold.
        stm = (self.conceptualSpace.stm
               if self.conceptualSpace is not None else None)
        if stm is not None:
            in_ev = in_sub.materialize() if in_sub is not None else None
            B_stm = (int(in_ev.shape[0])
                     if in_ev is not None and torch.is_tensor(in_ev) else 1)
            dev_stm = (in_ev.device
                       if in_ev is not None and torch.is_tensor(in_ev)
                       else None)
            stm.begin_forward(B_stm, device=dev_stm)

        # Stage 10 (doc/plans/2026-05-27-perceptstore-meta-taxonomy-
        # reentrancy.md): PARALLEL mode walks the per-stage CS pipeline:
        #
        #   * Stage 0 ingests ``PS.pi(IS)`` as its contribution; the
        #     stage's owned ``sigma_in[0]`` folds it (inside
        #     ``cs.forward``).
        #   * Stage k > 0 ingests the PRIOR stage's CS output as its
        #     contribution; the stage's owned ``sigma_in[k]`` folds it.
        #     Additionally, ``sigma_cs[k]`` lifts the prior CS state
        #     (residual lift) and is added to the new contribution per
        #     the unified equation:
        #
        #       ``CS_t1[k] = sigma_in[k](contribution) + SS_t1[k]
        #                   + sigma_cs[k](CS_t0[k])``
        #
        #     PARALLEL: ``SS_t1[k] = 0`` per the plan's mode table; the
        #     legacy ``ws.forward(prevCS_forSS)`` call is preserved
        #     (the WS state contract still updates) but the WS
        #     contribution at C-space_role is the empty seed when the seed
        #     was empty.
        #
        # The merge step (when ``useGrammar=='all'``) halves N
        # between stages; the next stage's sigma_in is sized for that
        # halved shape at construction time.
        prevCS_forSS = self._empty_seed_ss
        last_cs = None
        # Run PS ONCE for stage-0 ingestion (subsymbolic is single-
        # pass per the locked decision; PS.pi(IS) is the canonical
        # contribution at stage 0).
        # Gate PartSpace's serial warm-path on pass 0.
        if self.symbolSpace is not None:
            self.symbolSpace.recur_pass = 0
        else:
            self.perceptualSpace._recurrent_pass_idx = 0
        PS_sub_stage0 = self.perceptualSpace.forward(in_sub)
        self.create_ir_mask(PS_sub_stage0)
        # ``contribution`` is the incoming subspace for each stage's
        # ``cs.forward``. Stage 0 -> PS output; stage k > 0 -> prior
        # stage's post-merge CS output.
        contribution = PS_sub_stage0
        # A4 (2026-06-06 parallel-conceptual-recurrence): per-stage augment
        # list, threaded as a LOCAL through the T stages and into the SAME
        # forward's reverse. NOT stored on any space/layer (the design's
        # data-flow rule; A5 makes the per-batch threading fully clean).
        # ``prev_cs_content`` is the conceptual carrier content CS_t fed to
        # the combine -- CS_{-1} is the empty seed (zeros) UNLESS
        # ``<prediction>interSentence`` is set AND the discourse AR ring is
        # warm, in which case A6 seeds CS_{-1} from the SAME predictor
        # ``generate_sentence`` primes from (``_intersentence_seed`` -- one
        # source). The predicted ``payload_hat[depth, Dp]`` is broadcast into
        # the ``[B, N, D]`` slab at the t=0 bind site (where ``cs_content``
        # supplies the leading shape) and ADDED INTO WS_t -- the symbolic
        # prior (the CS stream is retired by the 2-stream bind, C-10).
        # ``prediction_mode == "none"`` (default) and a cold ring both keep
        # ``payload`` None -> no addition (byte-identical empty seed).
        carriers = []            # per-stage exact bind output (the FULL mix)
        prev_cs_stage = None     # P3: prior stage's cs (demux-feedback reads)
        _pump_qe = []            # P4: per-stage per-percept settle signal
        _seed = self._consume_intersentence_seed()
        seed_payload = _seed[1] if _seed is not None else None
        # Verification handle (Task A6 test): the predicted CS_{-1} seed
        # payload actually used this forward (None when empty / none-mode /
        # cold). Forward-local, not an nn.Module buffer.
        object.__setattr__(self, "_intersentence_seed_payload", seed_payload)
        for t, stage in enumerate(self.body_stages):
            cs = stage["cs"]
            ws = stage["ws"]
            # Preserve the recur_pass back-ref for any consumer that
            # still reads it (the AR-streaming warm-cache gate, sparse
            # state tests).
            if self.symbolSpace is not None:
                self.symbolSpace.recur_pass = int(t)
            else:
                self.perceptualSpace._recurrent_pass_idx = t
            # P4 stack selection: WS reads the pump pass for its per-pass pi
            # (plain host stamp -- byte-identical when the stack is off).
            object.__setattr__(ws, "_pump_pass_idx", int(t))
            # Dual-input plan sec.2 (rev. 2026-06-09): stage 0 reads the
            # UNITY view (parked by _lex_embed_stem) as direct symbolic
            # evidence -- the top-down analysis branch's input. Later
            # stages read only the recurrent CS (input once, mirroring
            # PS). Pure attr read, so the compiled/export paths trace it
            # exactly like _staged_in_sub.
            # Dual-towers rev 2: the universe bootstraps stage 0 ALWAYS and
            # drives every stage on the PARALLEL path (glue contract);
            # non-parallel t>0 stays carrier-driven (the live recurrent leg).
            _par = self.symbolicOrder > 0 and not self.serial
            # Under trace the unity is a lifted constant that the export
            # config leaves unused (lift_constants_pass StopIteration);
            # the traced path takes the carrier -- numerically what the
            # pre-delivery export computed (its unity was all-zero).
            _u_ok = not (torch.compiler.is_compiling()
                         or torch.compiler.is_exporting())
            WS_sub = ws.forward(
                (self._staged_concepts_in
                 if ((_par or t == 0) and _u_ok) else None),
                cs_out=prevCS_forSS)
            # ``cs.forward`` does the STM push + the C->P / C->S handoff
            # bookkeeping and produces this stage's perception event CS_0
            # (STM bookkeeping, no parameterised fold). PRESERVED intact --
            # the combine only replaces the CONTENT advance below.
            CS_sub = cs.forward(contribution, WS_sub)
            # Park the all-siblings concept slab at the parallel pass (stage 0)
            # for the post-forward conceptual SBOW. Used only for the (no-grad)
            # nearest-row assignment, so a detached materialize suffices. Gated
            # on conceptualSimilarityScale > 0 so a config that does not train
            # the similarity codebook keeps a byte-identical forward.
            if (t == 0 and not self.serial and not cs._sparse_active()
                    and float(getattr(getattr(self, "loss", None),
                                      "conceptual_similarity_scale", 0.0) or 0.0) > 0.0):
                # Sparse-inactive legacy park: detached t=0 slab (terminal-
                # codebook pool). Sparse-active parks LIVE at the POST-PUMP
                # cutover instead (P3): the SBOW situates the SETTLED
                # symbolic content, so the substitutability gradient reaches
                # the sparse family values and the snap (2026-07-02 plan C1).
                object.__setattr__(self, "_cs_parallel_slab",
                                   CS_sub.materialize().detach())
            if t == 0:
                # Asymmetric recon leg (input -> codebook): lift the
                # stage-0 WS codebook reconstruction term -- threaded by
                # ``_stage0_unity_forward`` as a forward-local -- onto the
                # pipeline-chained error container (CS_sub's errors object
                # is shared down to the OutputSpace via copy_context; the
                # fresh stage-0 WS subspace's is not).
                _ws_recon = getattr(ws, "_stage0_recon_loss", None)
                if (_ws_recon is not None and torch.is_tensor(_ws_recon)
                        and _ws_recon.requires_grad):
                    CS_sub.errors.add(
                        "ws_codebook_recon", _ws_recon, weight=1.0,
                        space="WholeSpace", category="symbol")
                object.__setattr__(ws, "_stage0_recon_loss", None)
                # Task 5 (C-13): lift the semantic-arrangement term the
                # same way (off unless <semanticArrangement> is set).
                _ws_sem = getattr(ws, "_stage0_semantic_loss", None)
                if (_ws_sem is not None and torch.is_tensor(_ws_sem)
                        and _ws_sem.requires_grad):
                    CS_sub.errors.add(
                        "ws_semantic_arrangement", _ws_sem, weight=1.0,
                        space="WholeSpace", category="symbol")
                object.__setattr__(ws, "_stage0_semantic_loss", None)
            else:
                # Step 1 (2026-06-10 symbolic-iteration plan): lift the
                # CS-leg recon gather the same way the stage-0 gather is
                # lifted at t==0 -- the winner row trains toward the
                # concept code that selected it. Same error name: Error
                # sums same-name terms, so multi-stage runs accumulate.
                _ws_recon = getattr(ws, "_csleg_recon_loss", None)
                if (_ws_recon is not None and torch.is_tensor(_ws_recon)
                        and _ws_recon.requires_grad):
                    CS_sub.errors.add(
                        "ws_codebook_recon", _ws_recon, weight=1.0,
                        space="WholeSpace", category="symbol")
                object.__setattr__(ws, "_csleg_recon_loss", None)
            # A4 conceptual combine: ONE square invertible 2-stream BIND
            # per stage (held by the cs) replaces the prior
            # ``cs.forward`` content fold + the WS.sigma (pi/sigma)
            # alternation (doc/plans/2026-06-06-parallel-conceptual-
            # recurrence.md sec. 3.3). Gated to the plain (non-grammar)
            # parallel path, matching the retired sigma gate -- the
            # useGrammar="all" cascade keeps its own N-halving ``merge``.
            #
            #   next_cs, aug_t = combine_t(PS_t, WS_t, CS_t)
            #
            # Option B: each stream is the FULL muxed event (sized at
            # ``cs.muxedSize``), so .where/.when PARTICIPATE in the combine.
            # PS_t is live ONLY at t=0 (alpha_ps reads the input once), WS_t
            # the WS event zero-padded to D, and CS_t the prior carrier event
            # (zeros at t=0). The advanced carrier ``next_cs`` (the full muxed
            # event) is written straight back into CS_sub -- the band is INSIDE
            # the transformed event now, not a tail riding along. The reverse
            # consumes the structured zero-pad (dropped) to invert the same
            # square map. (The retired ``reconstruct=perfect`` mode skipped the
            # combine -- only one stream live per stage, carrier carried
            # unchanged for an exact round-trip; when mixing is selected the
            # combine now always runs. Aligned binding takes the sibling path.)
            combine = (getattr(cs, "combine", None)
                       if self.subsymbolicOrder >= 1
                       and "merge" not in stage else None)
            _aligned_binding = bool(
                getattr(self, "concept_binding", "mixing") == "aligned"
                and self.subsymbolicOrder >= 1
                and "merge" not in stage)
            if combine is not None or _aligned_binding:
                _tiling_schedule = getattr(
                    self.wholeSpaces[0], "_where_tiling_schedule", None)
                _tiling_obs = (
                    _tiling_schedule[min(t, len(_tiling_schedule) - 1)]
                    if _tiling_schedule else None)
                object.__setattr__(cs, "_where_tiling_obs", _tiling_obs)
                # Processing contract (2026-06-10): the bind CALCULATION
                # lives on ConceptualSpace (``bind_streams``: demux / fit /
                # slot-stack / cascade / corpus-callosum glue / event
                # write); this loop only orchestrates. PS_t is the stage-0
                # percept re-fed at EVERY stage; the A6 interSentence seed
                # primes the SYMBOLIC stream at t=0; the FULL bind carrier
                # rides ON the stage's SubSpace (``_bind_carrier``) for the
                # reverse, with ``carriers`` kept as a test/verification
                # handle.
                ps_t = PS_sub_stage0
                if (getattr(self, "relevance_on", False) and t > 0
                        and getattr(self, "reading_attention", None) is None):
                    # Hard-coded readingAttention over the priming surface
                    # (sec C, simplified law): the reading scope is the span
                    # of the hottest-primed word-whole. Same output contract
                    # as the learned producer; no-op without spans/surface.
                    self._primed_reading_step()
                if (getattr(self, "reading_attention", None) is not None
                        and t > 0):
                    # Reading attention (doc/specs/reading-attention.md
                    # "(A) Reading attention"): the learned `.where` producer
                    # writes ``wholeSpaces[0]._passback_scope_where`` for the
                    # handoff below to consume + adds the next-word CE loss to
                    # CS_sub.errors. Runs BEFORE _passback_scope_ps so the
                    # "scoped" branch picks up the freshly-produced scope.
                    self._reading_attention_step(
                        t, prevCS_forSS, PS_sub_stage0, CS_sub)
                if (getattr(self, "global_attention", None) is not None
                        and t > 0):
                    # Global attention (doc/specs/reading-attention.md "(B)"):
                    # free attention over the typed addressable space (input /
                    # STM / LTM / codebook); parks the typed `.where` + soft-read
                    # on _global_attention_obs. Dark (parked, not fed back).
                    self._global_attention_step(prevCS_forSS, PS_sub_stage0)
                if (getattr(self, "mereology_raise", False)
                        and t > 0 and not prevCS_forSS.is_empty()):
                    # Top-down attention handoff (doc/specs/mereological-order-
                    # raising.md "the top-down attention handoff"): the stage-0
                    # WholeSpace passes back a scoped chunk/.where that scopes
                    # PartSpace's analysis on this t>0 pass (t==0 stays
                    # wide-open -- the first-pass-wide gate IS this t>0 guard).
                    # Dark unless a scope or words-category attention is engaged
                    # -> the pass-back action is "noop" -> byte-identical.
                    _prev_ps_fb = (getattr(prev_cs_stage, "_subspaceForPS", None)
                                   if prev_cs_stage is not None else None)
                    ps_t = self._passback_scope_ps(
                        t, PS_sub_stage0, prevCS_forSS,
                        prevPS_forPS=_prev_ps_fb)
                if (t > 0 and cs._sparse_active()
                        and ps_t is PS_sub_stage0
                        and prev_cs_stage is not None):
                    # P3 demux feedback (decision 7): the pump's PS bind leg
                    # at t>0 is the prior stage's UNBOUND part-stream (the
                    # C->P handoff carries the un-mix down for further σ
                    # synthesis) -- not the stage-0 percept re-fed. Scope
                    # attention, when it fired above, takes precedence.
                    # P4: the pass-t stack sigma applies to the feedback
                    # (identity at a no-op slot).
                    _ps_fb = getattr(prev_cs_stage, "_subspaceForPS", None)
                    if (_ps_fb is not None and hasattr(_ps_fb, "is_empty")
                            and not _ps_fb.is_empty()):
                        ps_t = self.perceptualSpace.synthesize_feedback(
                            _ps_fb, t)
                # Slice C: the SS (symbol) bind leg comes from
                # ``SymbolSpace.forward_concept_to_symbol(CS_sub)`` -- the
                # ``.forward()``-mediated CS->SS transform (the dataflow rule:
                # cross-space interaction goes ONLY through ``forward``; the old
                # CS-mediated ``_build_symbol_leg`` reach into WholeSpace +
                # ``_model_symbolSpace`` is retired). Computed only under the
                # symbol tower in parallel mode -- the leg the 3-stream bind
                # consumes. Off-path (2-stream / symbol tower off) leaves it
                # None, so ``bind_streams`` never enters the SS branch ->
                # byte-identical. P3 two-phase forward: the SPARSE pump is
                # 2-stream -- NO symbol leg inside the loop (quantization only
                # at the post-pump cutover below); the symbolicOrder=0
                # symbolTower scaffold path keeps its in-loop leg untouched.
                SS_sub = (self.symbolSpace.forward_concept_to_symbol(CS_sub)
                          if (getattr(self, "symbol_tower", False)
                              and not self.serial
                              and self.symbolSpace is not None
                              and not cs._sparse_active())
                          else None)
                if _aligned_binding:
                    full_t = cs.bind_aligned_streams(ps_t, WS_sub, CS_sub)
                else:
                    full_t = cs.bind_streams(
                        ps_t, WS_sub, CS_sub, SS_sub=SS_sub,
                        seed_payload=(seed_payload if t == 0 else None))
                if full_t is not None and t == 0:
                    # Snapshot the stage-0 bind so round-trip tests can
                    # compare it against the threaded carrier.
                    object.__setattr__(
                        self, "_combine_fwd_cs0", full_t.detach())
                carriers.append(full_t)
                # P4 settle signal (report-only, no control flow): each
                # pump stage's per-percept residual against the order-0
                # block -- the QE snap-error read as a SIGNAL for the later
                # adaptive-exit work. Runs whenever the sparse pump is live
                # (the per-pass stack is now canonical).
                if cs._sparse_active():
                    _qe_t = cs.snap_settle_qe(CS_sub.materialize())
                    if _qe_t is not None:
                        _pump_qe.append(_qe_t)
            # Stage 1.F: ``_cs_cache[t] = CS_sub`` retired. The
            # terminal C-space_role idea lives on ``conceptualSpace.stm``
            # (the bookkeeping push happens inside cs.forward); the
            # reverse path reads ``stm.snapshot()[:, -1, :]``.
            self._chart_compose_at_C(stage_idx=t)
            if "merge" in stage:
                CS_sub = stage["merge"](CS_sub)
            # Read the two C views after the optional merge (merge
            # mutates the CS subspace in place; the C->S view aliases
            # it so it reflects post-merge N). The cross-pass handoff
            # lives directly on the persistent ``cs._subspaceForPS`` /
            # ``cs._subspaceForWS`` SubSpaces (mutated in place by
            # ConceptualSpace.forward).
            prevCS_forSS = cs._subspaceForWS
            # "Nothing to quantize -> pass zeros": WS is inert when its
            # input was the empty seed (subsymbolicOrder=1, or pass 0).
            # CS already consumed WS_sub as-is above (pass-0 math
            # unchanged); the correctly-batched zero symbol is
            # produced here so the symbol-state contract holds for
            # the head / discourse.
            if WS_sub.is_empty():
                WS_sub = self._zero_symbol_subspace(ws, in_sub)
            # Stage 1.F: ``_ws_cache[t] = WS_sub`` retired. The
            # terminal symbolic state lives on
            # ``self.wholeSpace.subspace`` (written by
            # ``ws.forward(...)``); ``symbol_cache`` resolves there.
            last_cs = CS_sub
            # Cascade: this stage's (symbolically generalized) output
            # becomes the next stage's contribution -- the MIX goes UP
            # (the demux feedback above sends the un-mix DOWN per tower).
            contribution = CS_sub
            prev_cs_stage = cs
        # P3 LATE CUTOVER (two-phase forward): the pump above stayed purely
        # continuous/subsymbolic; NOW -- once, at the bandwidth seam -- snap
        # the settled field to the order-0 codebook block and run the
        # symbolic phase (``cs_symbolic_phase``). The activations stamp the
        # terminal CS (the SS leg + head-side losses read them); the settled
        # symbolic content feeds the conceptual SBOW (C1, live). The cutover
        # cs is STAGE 0's (the relation store the autobind populates lives
        # there); symbolicOrder=0 (incl. the symbolTower scaffold) never
        # enters -- byte-identical.
        _cut_cs = (self.body_stages[0]["cs"]
                   if getattr(self, "body_stages", None) else None)
        if (last_cs is not None and _cut_cs is not None
                and _cut_cs._sparse_active()):
            _settled = last_cs.materialize()
            # Relevance integration (sec C, UNCONDITIONAL 2026-07-12): the
            # priming surface is always warm (``_prime_seen_step``), so the
            # pyramid's top-K always reads the boost-1 ranking bias -- zero
            # (neutral) until primes accumulate. no_grad -- relevance
            # selects, never trains.
            _prio = self._assemble_relevance_priority(
                _cut_cs, prev_cs_stage, last_cs, _settled)
            object.__setattr__(_cut_cs, "_relevance_priority", _prio)
            _content, _acts = _cut_cs.cs_symbolic_phase(_settled)
            # SEEN write moved to ``_prime_seen_step`` (unconditional, once
            # per batch, both paths) -- awareness primes (simplified law).
            # Fail loud on a mis-shaped pyramid output (rows-first [N, B]):
            # a transposed acts would silently misuse batch rows as symbols.
            assert _acts is None or (
                _acts.dim() == 2
                and int(_acts.shape[0]) == int(_cut_cs.nVectors)), (
                f"symbolic phase acts must be [nVectors, B]; got "
                f"{None if _acts is None else tuple(_acts.shape)}")
            object.__setattr__(last_cs, "_concept_activations", _acts)
            if _acts is not None:
                # The SS leg, ONCE: syncs the SS codebook to the settled
                # concept codes; the leg's gradient rides the activations.
                if (getattr(self, "symbol_tower", False)
                        and self.symbolSpace is not None):
                    self.symbolSpace.forward_concept_to_symbol(last_cs)
                # SBOW C1 on the SETTLED symbolic slab (grad-bearing; the
                # pool is the SAME dictionary that composed it).
                if float(getattr(getattr(self, "loss", None),
                                 "conceptual_similarity_scale", 0.0)
                         or 0.0) > 0.0:
                    object.__setattr__(self, "_cs_parallel_slab", _content)
                    object.__setattr__(self, "_cs_parallel_slab_cs", _cut_cs)
        # A4: thread the per-stage augments to the reverse as a transient
        # local on ``self`` that the SAME forward's reverse reads (the
        # design's accepted near-term threading; A5 makes per-batch data
        # threading fully clean). ``object.__setattr__`` matches the CS
        # handoff idiom -- avoids mutating ``self._modules`` under the
        # torch.compile guards. ``_combine_last_cs_sub`` lets a caller
        # (e.g. the perfect-reconstruction test) reproduce CS_0 by driving
        # ``_reverse_body`` from the terminal carrier.
        # 2-stream bind (C-10): the augment thread is GONE -- the carrier
        # IS the whole bind, so the reverse needs nothing alongside it.
        object.__setattr__(self, "_combine_carriers", carriers)
        object.__setattr__(self, "_combine_last_cs_sub", last_cs)
        # P4 settle signal: per-stage per-percept residuals (report-only).
        object.__setattr__(self, "_pump_settle_qe", _pump_qe)
        # Reset so standalone PartSpace.forward calls (and the
        # next forward's pass 0) see the AR-streaming serial warm path.
        if self.symbolSpace is not None:
            self.symbolSpace.recur_pass = 0
        else:
            self.perceptualSpace._recurrent_pass_idx = 0
        if self.loss_head is not None:
            # Prefer the grad-bearing per-word stack stashed by
            # ``_forward_stem_per_word``; fall back to the (detached)
            # STM snapshot when the stack is missing (e.g. body-only
            # forward path that bypasses the per-word stem).
            grad_input = self._loss_head_input
            if grad_input is None:
                grad_input = self.conceptualSpace.stm.snapshot()
            if grad_input is not None:
                self._loss_head_loss = self.loss_head(grad_input)
            else:
                self._loss_head_loss = None
        return last_cs

    # ------------------------------------------------------------------
    # 2b-2-i: bounded binary grammar producer (STM -> single S).
    # ------------------------------------------------------------------
    def _stm_reducer(self):
        """Lazily resolve + cache the bounded-reduce scorer/combiner.

        The bounded soft REDUCE reuses the EXISTING reduction math
        verbatim -- it is NOT re-authored here (two-loop spec
        Phase-1-D §3 "STM-7": "score top-r of STM with the existing
        soft reducer ... the ``BinaryStructuredReductionLayer``
        anchors, Language.py:1608"; "parent = Σ_op weight_op ·
        op(left,right)  # fixed op axis, one weighted reduce"). One
        ``BinaryStructuredReductionLayer`` already wired on this model's
        SymbolSpace is reused directly.  Its op table and rule ids are an
        immutable consequence of the grammar active when this model was
        constructed; resolving them again through process-global
        ``TheGrammar`` would let a later model's XML silently change this
        model's lazy reducer. The layer's
        own forward computes ``chosen_reduced = Σ_op softmax(reduce_
        score)_op · op(left,right)`` over the fixed op axis (one
        weighted reduce, no shared in-place accumulator -- the proven
        ``_superposed_op`` pattern), which IS the spec's ``parent``.

        Returns ``None`` when no model-owned arity-2 host op is available
        (degenerate grammar) so the caller can skip the REDUCE
        (SHIFT-only, depth still bounded by back-pressure -- a forced
        no-op reduce simply pops the older slot).
        """
        cached = getattr(self, "_stm_reducer_cached", None)
        if cached is not None:
            if cached is not False and not torch.compiler.is_compiling():
                # Re-home to the live model device. The MPS build path
                # (ModelFactory._run_hydrated) constructs the model on CPU and
                # only then ``m.to(mps)``; this is normally handled through the
                # LanguageLayer's registered ownership, and the check keeps
                # hand-built/test models coherent with the STM it reads
                # ("weight is on cpu but expected on mps"). ``.to`` is a no-op
                # once the devices already match. HOST-SIDE ONLY: a
                # ``torch.device`` ``!=`` compare is not traceable (its
                # ``__eq__`` is a builtin method-wrapper -> Dynamo
                # "Polyfill handler ... does not have a traceable function"),
                # and inside the compiled forward the device is already fixed,
                # so ``is_compiling()`` skips it under trace.
                # Canonical device (util.TheDevice), not an incidental
                # tensor's -- the single source of truth init_device keeps.
                _dev = torch.device(str(TheDevice.get()))
                if next(cached.parameters(), None) is not None:
                    if next(cached.parameters()).device != _dev:
                        cached = cached.to(_dev)
            return cached if cached is not False else None
        grammar_owner = getattr(self, "symbolSpace", None)
        language = getattr(grammar_owner, "languageLayer", None)
        layers = getattr(language, "_binary_layers", None)
        layer = layers["CS"] if layers is not None and "CS" in layers else None
        # This is a non-owning cache. LanguageLayer remains the sole nn.Module
        # owner so optimizer/checkpoint traversal cannot register the same
        # reducer twice under model-root aliases.
        object.__setattr__(
            self, "_stm_reducer_cached",
            layer if layer is not None else False)
        return layer

    def _stm_unary_rewriter(self):
        """Return the already-wired conceptual unary grammar layer.

        Unlike the bounded binary scorer, a unary structured layer already
        operates independently at each position, so the signal router's CS
        layer can be reused directly. No parameters or alternate unary math
        are minted in the word-loop controller.
        """
        cached = getattr(self, "_stm_unary_rewriter_cached", None)
        if cached is not None:
            if cached is False:
                return None
            if not torch.compiler.is_compiling():
                dev = self.conceptualSpace.stm._buffer.device
                param = next(cached.parameters(), None)
                if param is not None and param.device != dev:
                    cached = cached.to(dev)
                    object.__setattr__(
                        self, "_stm_unary_rewriter_cached", cached)
            return cached
        ws = getattr(self, "wholeSpace", None)
        property_basis = bool(
            getattr(self, "wholePropertyBasis", False)
            or getattr(ws, "property_basis", False))
        grammar_owner = (getattr(self, "symbolSpace", None)
                         if property_basis else ws)
        language = getattr(grammar_owner, "languageLayer", None)
        layers = getattr(language, "_unary_layers", None)
        layer = layers["CS"] if layers is not None and "CS" in layers else None
        object.__setattr__(
            self, "_stm_unary_rewriter_cached",
            layer if layer is not None else False)
        return layer

    def _stm_bounded_unary_step(self, row_gate=None):
        """Permit at most one arity-1 grammatical operation per word.

        The unary router may choose COPY; only an actual APPLY mutates slot 0.
        STM depth is unchanged. Concept order is preserved, while grammatical
        derivation depth increases by one, keeping the two notions of order
        explicit and non-interchangeable.
        """
        stm = self.conceptualSpace.stm
        layer = self._stm_unary_rewriter()
        buf = stm._buffer
        B = int(buf.shape[0])
        can = stm._depth >= 1
        if row_gate is not None:
            can = can & row_gate.to(
                device=buf.device, dtype=torch.bool).reshape(B)
        if layer is None:
            return torch.zeros_like(can)
        hard, soft, routing = layer(buf[:, :1, :])
        candidate = (soft + (hard - soft).detach())[:, 0, :]
        snapped = self._stm_symbolic_roundtrip(candidate.unsqueeze(1))
        if snapped is not None and snapped.dim() == 3:
            candidate = snapped[:, 0, :]
        apply_mask = routing.get("apply_mask")
        if not torch.is_tensor(apply_mask) or apply_mask.shape[1] < 1:
            applied = torch.zeros_like(can)
        else:
            applied = apply_mask[:, 0, :].bool().any(dim=-1) & can
        object.__setattr__(self, "_stm_last_unary_routing", routing)

        order_slab, grammar_slab = stm.ensure_order_state()
        concept_rows, concept_activations = stm.ensure_reference_state()
        buf_new = torch.where(
            applied.view(B, 1), candidate, buf[:, 0, :])
        stm._buffer = torch.cat([buf_new.unsqueeze(1), buf[:, 1:, :]], dim=1)
        # ``_buffer`` replacement invalidates metadata by design; restore the
        # transformed parallel slabs explicitly.
        stm._orders = order_slab
        top_grammar = grammar_slab[:, 0]
        raised_grammar = torch.where(
            top_grammar >= 0, top_grammar + 1, top_grammar)
        grammar_new = grammar_slab.clone()
        grammar_new[:, 0] = torch.where(
            applied, raised_grammar, top_grammar)
        stm._grammar_orders = grammar_new
        reference_rows_new = concept_rows.clone()
        reference_activations_new = concept_activations.clone()
        reference_rows_new[:, 0] = torch.where(
            applied, torch.full_like(concept_rows[:, 0], -1),
            concept_rows[:, 0])
        reference_activations_new[:, 0] = torch.where(
            applied, torch.zeros_like(concept_activations[:, 0]),
            concept_activations[:, 0])
        stm._concept_rows = reference_rows_new
        stm._concept_activations = reference_activations_new
        routing["stm_apply_mask"] = applied
        return applied

    @staticmethod
    def _stm_grammar_reduce_confidence(routing):
        """Return a rule-count-neutral P(REDUCE) for a two-slot window.

        ``routing['reduce_marginal']`` sums probability over every binary
        grammar operator.  Consequently an untrained grammar with two
        otherwise-equivalent REDUCE operators starts near 2/3 rather than
        1/2 and appears to license a reduction after virtually every word.
        The STM controller needs a hierarchical decision instead: first KEEP
        versus BINARY-APPLY, then (conditional on APPLY) which operator.

        Log-mean-exp over each action's operator axis removes that rule-count
        prior.  The two COPY positions form one route, so their normalized
        action scores add; the REDUCE route consumes the pair in one action.
        The result remains differentiable with respect to chooser scores.
        """
        if not isinstance(routing, dict):
            return None
        copy_score = routing.get("copy_score")
        reduce_score = routing.get("reduce_score")
        if (not torch.is_tensor(copy_score)
                or not torch.is_tensor(reduce_score)
                or copy_score.dim() != 3
                or reduce_score.dim() != 3
                or copy_score.shape[1] < 2
                or reduce_score.shape[1] < 1
                or copy_score.shape[-1] < 1
                or reduce_score.shape[-1] < 1):
            return None
        n_copy = float(copy_score.shape[-1])
        n_reduce = float(reduce_score.shape[-1])
        copy_action = (
            torch.logsumexp(copy_score[:, :2, :], dim=-1)
            - math.log(n_copy))
        reduce_action = (
            torch.logsumexp(reduce_score[:, 0, :], dim=-1)
            - math.log(n_reduce))
        return torch.sigmoid(reduce_action - copy_action.sum(dim=1))

    @staticmethod
    def _stm_occupancy_threshold(depth, capacity, base_tau):
        """Interpolate grammar threshold from soft pressure to demand.

        At depth two (the first reducible state) the grammar must clear the
        configured ``base_tau`` on its own.  As the finite STM fills, the
        threshold falls linearly.  At capacity the separate demand mask in
        :meth:`_stm_bounded_reduce_step` makes the best grammatical operator
        mandatory, including the exact-zero-confidence edge case.
        """
        cap = max(1, int(capacity))
        tau = torch.as_tensor(
            base_tau, device=depth.device, dtype=torch.float32).clamp(0.0, 1.0)
        if tau.dim() == 0:
            tau = tau.expand_as(depth)
        else:
            tau = tau.reshape(depth.shape).to(device=depth.device)
        if cap <= 2:
            pressure = torch.zeros_like(depth, dtype=tau.dtype)
        else:
            pressure = (
                (depth.to(tau.dtype) - 2.0) / float(cap - 2)
            ).clamp(0.0, 1.0)
        return tau * (1.0 - pressure), pressure

    def _stm_bounded_reduce_step(self, protect_depth=None, gate_tau=None,
                                 row_gate=None, occupancy_pressure=False,
                                 demand=False):
        """ONE statically-unrolled, masked grammatical REDUCE micro-step.

        Operates on the live STM ``[B, cap, D]`` buffer + the tensor
        depth (``stm._depth``). The reduction itself is pure ``torch.where`` /
        gather / scatter over a fixed ``[B, cap, D]`` slab + a tensor
        depth, with no data-dependent trip count (spec
        STM-7: "Pop/push are masked roll/scatter/where over [B,7,D]
        + a tensor depth -- never pop().item()"). Eager word-axis execution
        performs one post-commit scalar re-pin of the host snapshot-width
        mirror; compiled execution skips that bookkeeping read.

        Mechanism (spec STM-7 controller, newest-at-slot-0 convention):
          * Take the top-2 STM constituents -- the two NEWEST, at slots
            ``0`` (newest) and ``1`` (2nd-newest). Keep ``left`` = the
            OLDER of the pair (slot 1) and ``right`` = the NEWER (slot 0)
            so the (asymmetric) reduce op sees the SAME operand order it
            saw under the old oldest-first convention.
          * ``parent = Σ_op weight_op · op(left,right)`` via the cached
            ``BinaryStructuredReductionLayer`` (its ``soft`` slab over a
            length-2 window IS the single weighted reduce over the
            fixed op axis -- reused, not re-authored).
          * Snap the parent C<->S via ``_stm_symbolic_roundtrip`` (the
            existing idempotent primitive; passthrough for non-
            ProjectionBasis ``.what`` -- MM_20M/MM_xor ``quantize``).
          * The folded ``parent`` becomes the new NEWEST: write it to
            slot 0, shift the surviving older constituents (old slots
            ``2..d-1``) DOWN into slots ``1..d-2``, clear the vacated last
            slot, ``d <- d - 1``. This keeps the occupied window anchored
            at ``[0, depth)`` with slot 0 = newest, and preserves the EXACT
            fold associativity of the old scheme (so the collapsed root is
            byte-identical).

        Three controller modes share this primitive:

          * legacy/final-seal ``gate_tau is None`` folds every eligible row;
          * ``occupancy_pressure=True`` compares a rule-count-neutral grammar
            confidence with a threshold that falls as STM fills;
          * ``demand=True`` forces the best *grammatical* operator on selected
            full rows before admitting another word.

        The latter two never use the degenerate mean-fold: a missing grammar
        is a no-op under soft pressure and an explicit capacity failure under
        demand. Rows with depth < 2 are no-ops. Returns the per-row boolean
        mask that ACTUALLY reduced, and mutates the STM buffer/depth via
        out-of-place masked ops. The return makes diagnostics count stack
        changes rather than mistaking a reducer call for a reduction.

        ``protect_depth`` (Task 6a, §7): an optional per-row ``[B]``
        long tensor giving each row's MINIMUM retained depth -- a row
        folds only while ``depth > protect_depth``. ``None`` (the
        default) means the historical floor of 1 (collapse-to-single-S);
        when supplied, RELATIVE rows pass ``protect_depth == 3`` so they
        stop at the depth-3 end-state ``[predicate, idea1, idea2]`` while
        ABSOLUTE rows keep ``protect_depth == 1``. The gate stays a pure
        per-row tensor mask (no data-dependent Python branch on a tensor
        value), so the static-unroll / CUDA-graph capture contract is
        preserved. With ``protect_depth == 1`` the added ``depth > 1``
        term is implied by the pre-existing ``depth >= 2`` term (integer
        depths), so the gate -- and the whole step -- is BYTE-IDENTICAL
        to the no-arg call: the absolute-only path is unchanged.
        """
        stm = self.conceptualSpace.stm
        reducer = self._stm_reducer()
        buf = stm._buffer                                  # [B, cap, D]
        B, cap, D = buf.shape
        depth = stm._depth                                 # [B] long
        device = buf.device
        order_slab, grammar_slab = stm.ensure_order_state()
        concept_rows, concept_activations = stm.ensure_reference_state()
        # Rows that CAN reduce: at least 2 constituents on the stack.
        can = (depth >= 2)                                 # [B] bool
        if row_gate is not None:
            can = can & row_gate.to(
                device=device, dtype=torch.bool).reshape(B)
        # Task 6a per-row depth floor. ``protect_depth`` (a [B] long, or
        # None == floor 1) keeps RELATIVE rows from folding below their
        # depth-3 end-state. The extra ``depth > protect_depth`` term is
        # a pure tensor mask; for the floor-1 default it is implied by
        # ``depth >= 2`` so ``can`` is bit-for-bit unchanged.
        if protect_depth is not None:
            can = can & (depth > protect_depth)            # [B] bool
        grammar_controlled = bool(
            gate_tau is not None or occupancy_pressure or demand)
        if reducer is None and grammar_controlled:
            # Soft pressure without a binary grammar cannot license a fold.
            # A capacity demand, however, must fail loudly rather than let
            # ``push_step_masked`` saturate and silently discard the oldest
            # constituent.  The host sync exists only on this exceptional,
            # grammar-free path.
            if demand and not torch.compiler.is_compiling():
                if bool(can.detach().any().item()):
                    raise RuntimeError(
                        "STM capacity reached but no binary grammatical "
                        "operator can reduce the occupied constituents")
            return torch.zeros_like(depth, dtype=torch.bool)
        # Gather the top-2 (the two NEWEST): right = newest = slot 0,
        # left = 2nd-newest = slot 1. The (older, newer) = (left, right)
        # operand assignment is preserved from the old convention so the
        # asymmetric reduce op output is byte-identical. Slot 1 is bogus
        # for the depth<2 rows but they are masked out by ``can`` below.
        rows = torch.arange(B, device=device)
        idx_newer = torch.zeros(B, dtype=depth.dtype, device=device)   # 0
        idx_older = torch.ones(B, dtype=depth.dtype, device=device)    # 1
        right = buf[rows, idx_newer]                        # [B, D] newest
        left = buf[rows, idx_older]                         # [B, D] 2nd-newest
        left_order = order_slab[:, 1] if cap > 1 else order_slab[:, 0]
        right_order = order_slab[:, 0]
        known_concept_order = (left_order >= 0) & (right_order >= 0)
        parent_order = torch.where(
            known_concept_order,
            torch.maximum(left_order, right_order),
            torch.full_like(left_order, -1))
        left_grammar = (grammar_slab[:, 1]
                        if cap > 1 else grammar_slab[:, 0])
        right_grammar = grammar_slab[:, 0]
        known_grammar_order = (left_grammar >= 0) & (right_grammar >= 0)
        parent_grammar = torch.where(
            known_grammar_order,
            torch.maximum(left_grammar, right_grammar) + 1,
            torch.full_like(left_grammar, -1))
        if reducer is not None:
            window = torch.stack([left, right], dim=1)      # [B, 2, D]
            hard, soft, routing = reducer(window)
            # Discrete chosen op (Alec 2026-06-22): a symbol reduce must be a
            # SINGLE grammatical op so it is INTERPRETABLE -- not the soft
            # Σ weight·op blend. Forward = the argmax-chosen op (``hard``,
            # slot 0 = the folded root, LanguageLayer.compose root-state
            # convention); gradient flows via the soft DP-marginal blend
            # (straight-through), so the reduce scorer still trains. ``routing``
            # names which op fired -- parked for interpretable readback.
            parent = (soft + (hard - soft).detach())[:, 0, :]   # [B, D]
            object.__setattr__(self, "_stm_last_reduce_routing", routing)
            if occupancy_pressure or demand:
                # Conditional parent from an actual grammar operator.  This
                # remains correct when memory pressure overrides a COPY route.
                chosen = routing.get("chosen_reduced")
                if torch.is_tensor(chosen) and chosen.shape[1] >= 1:
                    parent = chosen[:, 0, :]
            if occupancy_pressure:
                confidence = self._stm_grammar_reduce_confidence(routing)
                if confidence is None:
                    can = torch.zeros_like(can)
                else:
                    threshold, pressure = self._stm_occupancy_threshold(
                        depth, cap, self.stm_reduce_tau
                        if gate_tau is None else gate_tau)
                    # A full but inactive/protected batch row is not part of
                    # this word's demand.  Mask with the eligibility computed
                    # above so telemetry and the controller agree per row.
                    demand_rows = (depth >= cap) & can
                    can = can & (demand_rows | (confidence > threshold))
                    routing["grammar_reduce_confidence"] = confidence
                    routing["stm_reduce_threshold"] = threshold
                    routing["stm_occupancy_pressure"] = pressure
                    routing["stm_demand_mask"] = demand_rows
            elif gate_tau is not None and not demand:
                # 2b-2 scored gate: the DP's reduce-vs-copy marginal on the
                # 2-window -- a row folds only where the grammar licenses
                # the reduce above tau. Pure tensor mask, no host sync.
                can = can & (
                    routing["reduce_marginal"][:, 0] > float(gate_tau))
            # Method-2 reverse-reduce trace (serial plan Task 4): append this
            # step's chosen-op posteriors + the rows that actually folded, so
            # the reverse can walk the fold BACKWARD calling each op's
            # basis-threaded reverse (the codebook-walk recommender) to
            # reconstitute the operand pair. Host list of detached tensors,
            # reset per sweep by _stm_reduce_to_single_S; eager-only (a Python
            # list mutation would graph-break under compile).
            if not torch.compiler.is_compiling():
                # Slot-kind provenance (open-fronts Task B): tag this fold's
                # operands (left = slot 1, right = slot 0) as word/other and
                # fold the kinds (word ∧ word -> word). Runs at EVERY eager
                # reduce so the kind stacks track the buffer; the trace
                # rides the tags when it is recording. len < 2 under can =
                # bookkeeping mismatch -> True/True (legacy unfiltered row).
                ks = getattr(stm, "_slot_kinds", None)
                lw = rw = None
                if ks is not None and len(ks) == B:
                    # STRICT mismatch default (dilution fix): with the tag
                    # sites now span-verified, an out-of-sync stack means
                    # UNKNOWN — and unknown is not a word.
                    lw = torch.zeros(B, dtype=torch.bool)
                    rw = torch.zeros(B, dtype=torch.bool)
                    for b, c in enumerate(
                            can.detach().to("cpu").tolist()):
                        if not c:
                            continue
                        k = ks[b]
                        if len(k) >= 2:
                            lwb = (k[1] == "word")
                            rwb = (k[0] == "word")
                            lw[b] = lwb
                            rw[b] = rwb
                            # ATOMIC-word semantics (snap design doc,
                            # 2026-07-14): a fold parent is a COMPOSITE,
                            # never a word — the old word∧word→word rule
                            # re-tagged composites-of-words as words, so
                            # later folds' word-tagged operands made the
                            # un-fold snap emit duplicate words (measured
                            # 5 emissions on 2-word sentences).
                            k[0:2] = ["other"]
                trace = getattr(self, "_stm_reduce_op_trace", None)
                if trace is not None:
                    if lw is not None:
                        trace.append((
                            routing["reduce_marginal_op"].detach(),
                            can.detach(), lw, rw))
                    else:
                        trace.append((
                            routing["reduce_marginal_op"].detach(),  # [B,1,R]
                            can.detach()))                           # [B] bool
        else:
            # Legacy degenerate grammar: the historical forced structural
            # mean-fold remains for checkpoint compatibility. Grammar-
            # controlled pressure/demand modes returned above and can never
            # reach this unlicensed fallback.
            parent = 0.5 * (left + right)
        # A hard grammar COPY returns one unchanged operand and therefore
        # retains that operand's exact lexical reference. Every actual binary
        # transform (including pressure/demand's chosen reduction and the
        # degenerate mean fallback) has no single exact row identity.
        parent_concept_row = torch.full(
            (B,), -1, dtype=torch.long, device=device)
        parent_concept_activation = torch.zeros(
            (B,), dtype=buf.dtype, device=device)
        if reducer is not None and not (occupancy_pressure or demand):
            action_kind = routing.get("action_kind")
            src_left = routing.get("src_left")
            if (torch.is_tensor(action_kind) and action_kind.dim() == 2
                    and int(action_kind.shape[1]) >= 1
                    and torch.is_tensor(src_left) and src_left.dim() == 2
                    and int(src_left.shape[1]) >= 1):
                # Reducer input is [left=old slot 1, right=old slot 0].
                reference_window_rows = torch.stack(
                    [concept_rows[:, 1], concept_rows[:, 0]], dim=1)
                reference_window_activations = torch.stack(
                    [concept_activations[:, 1],
                     concept_activations[:, 0]], dim=1)
                source = src_left[:, 0].clamp(0, 1).long()
                copy_selected = (
                    (action_kind[:, 0] == 0) & (src_left[:, 0] >= 0))
                copied_row = reference_window_rows.gather(
                    1, source.unsqueeze(1)).squeeze(1)
                copied_activation = reference_window_activations.gather(
                    1, source.unsqueeze(1)).squeeze(1)
                parent_concept_row = torch.where(
                    copy_selected, copied_row, parent_concept_row)
                parent_concept_activation = torch.where(
                    copy_selected, copied_activation,
                    parent_concept_activation)
        # C<->S idempotent snap of the parent (reused primitive;
        # passthrough for non-ProjectionBasis ``.what``).
        snapped = self._stm_symbolic_roundtrip(
            parent.unsqueeze(1))                            # [B, 1, D]
        if snapped is not None and snapped.dim() == 3:
            parent = snapped[:, 0, :]
        # Masked fold: ``can`` incorporates the legacy fixed gate, the new
        # occupancy-pressure decision, or a hard demand. A short/protected
        # row is always a pure no-op.
        # Build the post-fold layout for the can-rows: the folded
        # ``parent`` becomes the new newest at slot 0, the surviving older
        # constituents (old slots ``2..cap-1``) shift DOWN into slots
        # ``1..cap-2``, and the now-vacated last slot is zeroed. This is a
        # pure gather/scatter over the fixed [B, cap, D] slab (no
        # ``.item()``, no data-dependent trip count) so the static-unroll /
        # CUDA-graph capture contract is preserved.
        zero_tail = torch.zeros(B, 1, D, dtype=buf.dtype, device=device)
        if cap >= 3:
            tail = buf[:, 2:cap, :]                         # [B, cap-2, D]
            shifted = torch.cat(
                [parent.unsqueeze(1), tail, zero_tail], dim=1)  # [B, cap, D]
        else:
            # cap < 3: nothing above slot 1 to keep; the folded parent is
            # the sole survivor (slot 0), slot 1 (if present) clears.
            if cap == 2:
                shifted = torch.cat(
                    [parent.unsqueeze(1), zero_tail], dim=1)    # [B, 2, D]
            else:  # cap == 1 (degenerate): can is never True (needs d>=2)
                shifted = parent.unsqueeze(1)                   # [B, 1, D]
        # Masked commit: only the can-rows adopt the shifted layout; the
        # rest keep their buffer unchanged.
        buf_new = torch.where(can.view(B, 1, 1), shifted, buf)
        stm._buffer = buf_new
        order_tail = torch.full(
            (B, 1), -1, dtype=torch.long, device=device)
        if cap >= 3:
            shifted_orders = torch.cat(
                [parent_order.unsqueeze(1), order_slab[:, 2:cap],
                 order_tail], dim=1)
            shifted_grammar = torch.cat(
                [parent_grammar.unsqueeze(1), grammar_slab[:, 2:cap],
                 order_tail], dim=1)
        elif cap == 2:
            shifted_orders = torch.cat(
                [parent_order.unsqueeze(1), order_tail], dim=1)
            shifted_grammar = torch.cat(
                [parent_grammar.unsqueeze(1), order_tail], dim=1)
        else:
            shifted_orders = parent_order.unsqueeze(1)
            shifted_grammar = parent_grammar.unsqueeze(1)
        stm._orders = torch.where(
            can.view(B, 1), shifted_orders, order_slab)
        stm._grammar_orders = torch.where(
            can.view(B, 1), shifted_grammar, grammar_slab)
        reference_tail_rows = torch.full(
            (B, 1), -1, dtype=torch.long, device=device)
        reference_tail_activations = torch.zeros(
            (B, 1), dtype=buf.dtype, device=device)
        if cap >= 3:
            shifted_concept_rows = torch.cat(
                [parent_concept_row.unsqueeze(1),
                 concept_rows[:, 2:cap], reference_tail_rows], dim=1)
            shifted_concept_activations = torch.cat(
                [parent_concept_activation.unsqueeze(1),
                 concept_activations[:, 2:cap],
                 reference_tail_activations], dim=1)
        elif cap == 2:
            shifted_concept_rows = torch.cat(
                [parent_concept_row.unsqueeze(1), reference_tail_rows],
                dim=1)
            shifted_concept_activations = torch.cat(
                [parent_concept_activation.unsqueeze(1),
                 reference_tail_activations], dim=1)
        else:
            shifted_concept_rows = parent_concept_row.unsqueeze(1)
            shifted_concept_activations = (
                parent_concept_activation.unsqueeze(1))
        stm._concept_rows = torch.where(
            can.view(B, 1), shifted_concept_rows, concept_rows)
        stm._concept_activations = torch.where(
            can.view(B, 1), shifted_concept_activations,
            concept_activations)
        # d <- d - 1 for rows that reduced (g==1 there); tensor op.
        dec = can.to(depth.dtype)                          # [B] {0,1}
        stm._depth = depth - dec
        # Eager parser correctness: snapshot() and the next capacity check
        # need the CURRENT maximum depth, not a historical high-water mark.
        # Keeping the latter made every post-word snapshot include cleared
        # tail slots and triggered an "emergency" reducer call on every word
        # after word 8 even when licensed grammar had already reduced depth
        # back to 1. Compiled mode keeps the tensor-only path; its static loop
        # does not consult this host value for numerical gating.
        if (not torch.compiler.is_compiling()
                and getattr(self, "serial_word_capacity", None) is not None):
            stm._max_depth_host = int(stm._depth.max().item())
        return can

    def _sentence_relative_mask(self, B, device=None):
        """Per-row ``[B]`` bool: True where the current sentence is a
        RELATIVE truth (the ``part`` / ``isEqual`` predicate family).

        Task 6a (doc/plans/2026-05-29-stm-serial-parallel-modes.md §7).
        Conservative by construction -- the reduce site uses this to
        PRESERVE a depth-3 relative end-state, and a FALSE POSITIVE
        (absolute sentence wrongly flagged relative) would stop its
        collapse and break the dominant absolute path + the IR loss that
        consumes the single ``S``. So this returns False on ANY
        uncertainty.

        SIGNAL (host-side, grammar-driven -- does NOT depend on the
        unreliable post-reduce STM category metadata): scan
        ``symbolSpace.current_rules``' SS AND CS rule_id lists (the
        relative producers are CS-role rules; the CS scan is
        anchor-gated -- see ``Language.sentence_relative_mask``) for any
        rule_id that ``TheGrammar.is_relative_rule`` flags (lhs == a
        relative start role state, or an ``isEqual`` / ``isPart`` op).
        The read is a host dict lookup
        BEFORE the captured sweep, so it never enters the CUDA-graph.

        SHAPE HANDLING (``current_rules[space_role]`` is ``list[list[int]]``):
          * full-router path (``LanguageLayer.compose``) -> one inner
            list PER BATCH ROW (``len == B``): per-row detection.
          * default-only path -> a single batch-SHARED inner list
            (``len == 1``): scalar result broadcast to ``[B]``.
          * any other length, or missing / empty rules, or a grammar
            with no relative rule at all -> all-False (collapse as
            today). The absolute path is byte-identical in every
            fall-through case.
        """
        from Language import sentence_relative_mask
        return sentence_relative_mask(
            getattr(self, 'symbolSpace', None), B, device=device)

    def _stm_reduce_to_single_S(self):
        """NULL-seal finalize: bounded reduce sweep over STM -> single S.

        Statically unrolled to ``cap - 1`` forced micro-steps (spec
        STM-7: "REDUCE micro-steps, bounded to K-1, STATICALLY
        UNROLLED"; "on NULL seal: final bounded reduce sweep to
        root"). Each step folds the top-2 into one, so ``cap-1``
        steps drive any depth in ``[1, cap]`` down to exactly 1. The
        trip count is the static buffer capacity (NOT data-dependent
        -- CUDA-graph-capturable), masked when a row already has
        depth < 2.

        Returns the single root idea ``S`` ``[B, D]`` plus the post-sweep
        depth tensor for verification. Newest-at-slot-0 convention: after
        a full ABSOLUTE collapse the root is the sole survivor at slot 0;
        a RELATIVE row stops at depth 3 with the predicate at the OLDEST
        slot (``depth-1``). ``S`` is read per-row at slot ``depth-1`` so it
        is the collapsed root for absolute rows AND the predicate for
        relative rows -- IDENTICAL semantics to the old oldest-first
        ``buf[:, 0, :]`` read (where slot 0 was the oldest survivor).

        Task 6a (§7) RELATIVE preservation: ABSOLUTE rows collapse to
        depth 1 as before; RELATIVE rows (detected host-side via
        ``_sentence_relative_mask``) stop at the depth-3 end-state
        ``[predicate, idea1, idea2]``. This is implemented by a per-row
        ``protect_depth`` floor threaded into each masked micro-step
        (floor 3 for relative rows, floor 1 -- the historical default --
        for absolute rows). When no relative sentence is detected the
        floor is 1 everywhere and every step is byte-identical to the
        pre-Task-6a sweep. For a relative row, slot 0 of the returned
        ``S`` is the predicate and the returned depth is 3.
        """
        stm = self.conceptualSpace.stm
        cap = int(stm.capacity)
        buf = stm._buffer
        B = buf.shape[0]
        device = buf.device
        # Method-2 reverse-reduce trace: reset per sweep so the trace covers
        # exactly the folds that build THIS root S (back-pressure folds before
        # the sweep are not traced -- those rows fall back to the CS reverse).
        # Host attr, eager-only (compile keeps the pre-existing behavior).
        # SENTENCE-scoped override (open-fronts Task B): on wordStore configs
        # the serial loop already reset the trace at sentence start so the
        # mid-reading 2b-2 folds are covered -- the sweep reset stands down.
        if not torch.compiler.is_compiling():
            if not getattr(self, "_stm_trace_sentence_scope", False):
                object.__setattr__(self, "_stm_reduce_op_trace", [])
        # Host-side relative detection BEFORE the captured sweep (reads
        # the host ``current_rules`` dict; never enters the graph).
        rel = self._sentence_relative_mask(B, device=device)    # [B] bool
        depth_dtype = stm._depth.dtype
        # Per-row depth floor: 3 for relative rows, 1 otherwise. Computed
        # UNCONDITIONALLY as a pure tensor -- the old ``if bool(rel.any())``
        # host branch forced a tensor->Python sync and failed Dynamo with
        # ``Could not guard on data-dependent expression Eq(u0, 1)``. When
        # no row is relative this is all-ones, and
        # ``_stm_bounded_reduce_step``'s ``depth > protect_depth`` term is
        # then implied by its ``depth >= 2`` gate (integer depths) -- so the
        # sweep stays BYTE-IDENTICAL to the former ``protect_depth=None``
        # fast path while tracing under fullgraph=True.
        protect_depth = torch.where(
            rel,
            torch.full((B,), 3, dtype=depth_dtype, device=device),
            torch.full((B,), 1, dtype=depth_dtype, device=device),
        )                                                        # [B] long
        # syntacticOrder (doc/specs/orders.md) caps the parse-tree DEPTH: a
        # positive value runs at most that many fold levels (statically
        # clamped to cap-1, keeping the CUDA-graph trip count static -- it
        # never depends on the runtime word count). 0 = unbounded (the full
        # cap-1 sweep -> collapse to a single S, byte-identical). The <= W
        # bound holds structurally: a reduce micro-step is a no-op once a
        # row's depth reaches 1, so capping below cap-1 simply hands on a
        # partially-composed forest.
        _syn = int(getattr(self, "syntacticOrder", 0) or 0)
        _n_levels = max(0, cap - 1)
        if _syn > 0:
            _n_levels = min(_n_levels, _syn)
        _grammar_boundary_demand = bool(
            getattr(self, "serial_object_meta", False)
            and getattr(self, "serial_word_capacity", None) is not None)
        for _ in range(_n_levels):
            if _grammar_boundary_demand:
                self._stm_bounded_reduce_step(
                    protect_depth=protect_depth, demand=True)
            else:
                self._stm_bounded_reduce_step(
                    protect_depth=protect_depth)
        # After cap-1 forced folds every ABSOLUTE row's stack is
        # collapsed to a single slot (slot 0, the newest accumulator):
        # that slot is the sentence idea S (the start-symbol root -- this
        # producer reduces toward ``Grammar.start_symbol`` by
        # construction; the only S-space_role reduce ops the grammar exposes all
        # have lhs == start_symbol for MM_xor/MM_20M, so the folded root's
        # category tracks start_symbol). RELATIVE rows stop at depth 3 with
        # the predicate at the OLDEST slot (``depth-1``). Reading per-row
        # at slot ``depth-1`` returns the collapsed root for absolute rows
        # (depth 1 -> slot 0) AND the predicate for relative rows (depth 3
        # -> slot 2), matching the old oldest-first ``buf[:, 0, :]``.
        post_depth = stm._depth
        B = int(stm._buffer.shape[0])
        rows = torch.arange(B, device=stm._buffer.device)
        root_idx = (post_depth - 1).clamp(min=0)            # [B]
        S = stm._buffer[rows, root_idx]                     # [B, D]
        return S, post_depth

    def _maybe_concept_index_read(self, idea_bd):
        """Replace ``idea_bd``'s content slice with the word's already-known
        OBJECT-concept row (snap design doc §ontology; Alec: the forward
        folds object concepts — Method-2 un-folds into object concepts and
        TRANSLATES them back to word concepts, the exact reverse). Sources
        the promoted percept id per batch row from the per-word PS forward's
        index grid (the same PS codes ``create_word_object_meta`` tied at
        mint), resolves each to its object-concept's ``similarity_codebook``
        row, and writes the re-normalized row into the content dims —
        keeping the ``.where`` / ``.when`` bands (placement is untouched).
        Untied rows keep the computed idea (mask). Best-effort: any failure
        returns ``idea_bd`` unchanged (never break the forward)."""
        try:
            # Resolve against the CS that HOLDS the reverse index — the
            # autobind runs on a per-stage/terminal CS instance that may
            # differ from ``self.conceptualSpace`` (multi-stage towers;
            # same pattern as recognized_word_rows in the un-fold).
            cs = self.conceptualSpace
            for _c in (list(getattr(self, "conceptualSpaces", []) or [])
                       or [cs]):
                if getattr(_c, "_percept_word_concept", None):
                    cs = _c
                    break
            ps = self.perceptualSpace
            fi = getattr(ps, "_forward_input", None) or {}
            idxg = fi.get("indices")
            if not torch.is_tensor(idxg) or idxg.dim() < 2:
                return idea_bd
            pid_b = idxg[:, 0].reshape(-1)          # promoted percept / row
            if int(pid_b.shape[0]) != int(idea_bd.shape[0]):
                return idea_bd
            content, mask = cs.concept_row_content(pid_b)
            if content is None or mask is None or not bool(mask.any()):
                return idea_bd
            d = min(int(content.shape[-1]), int(idea_bd.shape[-1]))
            sel = mask.to(idea_bd.device).unsqueeze(-1)
            out = idea_bd.clone()
            out[:, :d] = torch.where(sel, content[:, :d].to(idea_bd.dtype),
                                     idea_bd[:, :d])
            return out
        except Exception:
            return idea_bd

    def _mphf_route_word(self, word_slice, cursor_pos):
        """Rework A: route the per-word percept -> concept through
        ``idx = MPHF(percept_bytes)`` -> the D2 table row.

        ``word_slice`` is the ``[B,1,D]`` muxed per-word slice the
        InputSpace cursor returned (``_ar_embedded[:, p:p+1, :]``);
        ``cursor_pos`` is its per-word slot index ``p`` (the same
        ``[B,nIdeas,M]`` per-word slot axis ``_embed_bpe``'s
        ``set_forward_content`` writes the per-word frozen lexicon row
        onto -- ``PartSpace.subspace._index[:,p,0]`` (==
        ``per_word_first``, the byte-derived O(1) frozen ``key_to_index``
        resolution that IS the MPHF index for the in-vocab percept; the
        ``_ar_embedded`` per-word cursor axis shares this exact slot
        axis -- verified: both ``[B,1024,*]`` for MM_20M). The standalone
        static byte->row ``PartSpace.mphf_index`` is the explicit
        formalization, exposed + gated, with the per-word route using
        this already-computed equivalent so the percept is resolved
        exactly ONCE, no host sync, no improvised byte-span.

        Substitutes the table's concept-activation row
        (``wv._vectors[idx]`` -- the REUSED Phase-1A.1 learnable lexicon
        param, NOT a second embedding; gradient flows, no detach) into
        the WHAT slab of ``word_slice``, preserving WHERE/WHEN (mirrors
        ``create_ir_mask``'s WHAT-slice-only edit). Returns
        ``(routed_slice, idx_or_None)``; a no-op pass-through (returns
        the input unchanged) when the MPHF table is inapplicable
        (numeric / non-Embedding codebook -- the existing path is then
        byte-identical).
        """
        ps = self.perceptualSpace
        isp = self.inputSpace
        if ps is None or isp is None or word_slice is None:
            return word_slice, None
        if ps._mphf_codebook() is None:
            return word_slice, None
        # The per-word frozen lexicon row (== the MPHF index for the
        # in-vocab percept) is written by ``_embed_bpe`` /
        # ``set_forward_content`` onto the PERCEPTUAL space's subspace
        # ``_index`` (``[B,nIdeas,M]``; ``[...,0]`` == ``per_word_first``,
        # the byte-derived frozen ``key_to_index`` row). It shares the
        # ``_ar_embedded`` per-word cursor slot axis (both ``[B,nIdeas,*]``).
        sub = getattr(ps, "subspace", None)
        active = getattr(sub, "_index", None) if sub is not None else None
        if (active is None or active.dim() != 3
                or cursor_pos >= active.shape[1]):
            return word_slice, None
        # The percept's frozen lexicon row at this cursor slot == the
        # MPHF index (byte-derived, O(1), frozen, NON-invertible). The
        # static tables are PRE-WARMED by the caller
        # (``_forward_body_per_word`` setup before the per-word loop,
        # or the test fixture before ``torch.compile``) so the captured
        # region never traces the build path -- ``build_mphf_table``
        # iterates ``bytes`` which is Dynamo-untraceable. ``ps.mphf_table_rows``
        # below uses the already-resolved row index (no hash lookup);
        # the table is only consulted via the codebook ``getW()`` gather,
        # which is pure-tensor and capture-safe.
        if getattr(ps, "_mphf_static_tables", None) is None:
            # Availability fallback: caller forgot to pre-warm and we
            # can't safely build inside a captured region. Pass-through.
            return word_slice, None
        idx = active[:, cursor_pos, 0].long()              # [B] MPHF row
        table_rows = ps.mphf_table_rows(idx)               # [B, D_what]
        nWhat = table_rows.shape[-1]
        routed = word_slice.clone()
        if routed.shape[-1] >= nWhat:
            routed[:, 0, :nWhat] = table_rows
        else:
            routed = table_rows.unsqueeze(1)
        return routed, idx

    def _per_word_prelude(self, in_sub):
        """D8 boundary-side setup that MUST run before the per-word
        loop. Hoisted out of ``_forward_body_per_word`` so the gate
        test (which calls ``_per_word_body_step`` in isolation) can
        replay the same contract.

        Side effects:
          * ``conceptualSpace.stm.begin_forward(B)`` -- seeds a FRESH
            per-forward STM working buffer (a ``torch.zeros`` graph
            INTERMEDIATE) sized to the live batch and starting empty
            (A5 fullgraph fix; replaces the retired ``ensure_batch`` +
            in-place ``clear`` of a persisted registered buffer).
          * ``perceptualSpace._mphf_tables()`` -- lazy build-once
            cache is primed (build path iterates ``bytes`` which
            Dynamo refuses to trace).
          * ``symbolSpace.recur_pass = 0`` -- invariant across the
            per-word loop (pass-index 0 throughout).
          * ``self._prev_cs_for_ps`` / ``self._prev_cs_for_ss`` pre-
            seeded to the persistent empty seeds so iteration 0's
            PS/WS forwards see a non-None feedback subspace (avoids a
            None-vs-SubSpace branch inside the captured body that
            would force a recompile after iteration 1). The body
            switches these to ``cs._subspaceForPS`` /
            ``cs._subspaceForWS`` (the persistent CS-space_role storage that
            CS.forward mutates in place) right after iteration 0's
            cs.forward, so iterations 1+ pick up the in-place updates
            without any further pointer churn.

        Stage 1.F of the two-loop pi/sigma substrate refactor
        (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md) retired
        the per-stage ``_cs_cache`` / ``_ws_cache`` capture lists —
        no per-forward reallocation here either; the terminal C-space_role
        idea lives on ``stm`` (cleared above), and the terminal
        symbolic subspace is the persistent ``self.wholeSpace
        .subspace`` (overwritten in place by each ``ws.forward``).

        Returns ``(stm, N_target, word_carrier, in_event)`` -- the
        values ``_forward_body_per_word`` needs from the prelude.
        ``in_event`` is the materialized whole-sentence event the
        post-loop step uses to restore the InputSpace subspace.
        """
        stm = self.conceptualSpace.stm
        in_event = in_sub.materialize() if in_sub is not None else None
        if stm is not None:
            B_in = int(in_event.shape[0]) if in_event is not None else 1
            # A5 fullgraph fix (doc/plans/2026-06-06-parallel-conceptual-
            # recurrence.md sec 2 & 4): seed a FRESH per-forward STM working
            # buffer as a trace INTERMEDIATE (was: ``ensure_batch`` + in-place
            # ``clear`` of a persisted registered buffer, which the compiled
            # forward then mutated/output-aliased -> the recompile guard flip
            # + AOT alias-regeneration crash). The per-word loop SHIFTs onto
            # this fresh buffer; the live grad to the head threads through the
            # stacked ``_per_word_contributions`` list, not the buffer.
            dev_in = (in_event.device
                      if in_event is not None and torch.is_tensor(in_event)
                      else None)
            stm.begin_forward(B_in, device=dev_in)

        # MPHF pre-warm (Dynamo-unfriendly build path).
        ps = self.perceptualSpace
        if (ps.synthesis_mode == "mphf"
                and ps._mphf_static_tables is None):
            try:
                ps._mphf_tables()
            except Exception:
                pass
        # STM bounded-reducer pre-warm: hoisted to ``enable_compiled_step``
        # (the true eager boundary BEFORE the compile wrapper closes
        # over ``self.forward``). The prelude itself runs INSIDE the
        # compiled forward on the production path, so a lazy build
        # here would still be traced.

        # SymbolSpace per-sentence state + per-forward pre-seed. The
        # cursor / recur_pass are allocated by ``symbolSpace.soft_reset``
        # (triggered by ``post_tick_compact`` when a sentence
        # completes); the first forward of the first sentence runs
        # BEFORE any soft_reset has fired, so cold-start allocation
        # falls to this prelude. ``self._prev_cs_for_ps/ws`` are
        # UNCONDITIONALLY re-seeded to the empty seeds each forward so
        # the per-word loop's iteration 0 always sees the canonical
        # empty starting context -- never a stale SubSpace from a prior
        # sentence's per-word loop. The body switches these to
        # ``cs._subspaceForPS`` / ``cs._subspaceForWS`` after the first
        # iteration's cs.forward; subsequent iterations read the same
        # persistent objects (in-place updates via set_event keep the
        # references stable).
        # Cold-start: the per-sentence state (STM arms, SVO clear, etc.)
        # is normally re-initialized by ``post_tick_compact`` when a
        # sentence completes. The first forward of the first sentence
        # runs BEFORE any compaction has fired, so we run soft_reset
        # once at construction-time. ``_per_sentence_initialized`` is a
        # plain Python bool sentinel — Dynamo-friendly (specializes once
        # then becomes a no-op).
        if not getattr(self.symbolSpace, '_per_sentence_initialized', False):
            self.symbolSpace.soft_reset()
            self.symbolSpace._per_sentence_initialized = True
        self.symbolSpace.recur_pass = 0
        # These are transient carrier pointers, not owned model children.
        # Registering them through ``nn.Module.__setattr__`` on the first
        # captured forward mutates ``self._modules`` and invalidates the just
        # compiled W-loop guard on the next sentence.
        object.__setattr__(self, "_prev_cs_for_ps", self._empty_seed_ps)
        object.__setattr__(self, "_prev_cs_for_ss", self._empty_seed_ss)
        # A6: the serial per-word path has no combine ``prev_cs_content`` leg
        # (that is the parallel-only carrier). Its CS_{-1} analog is the
        # ``_c_prior`` priming ``ConceptualSpace.forward`` adds across the
        # first ``depth`` STM slots -- the SAME mechanism
        # ``generate_sentence`` primes from. The seed comes from
        # ``_consume_intersentence_seed`` (parked eagerly by ``runBatch`` on
        # the compiled path so the disabled predictor never runs in the trace;
        # computed live on the eager path), gated to ``<prediction>
        # interSentence`` + a warm AR ring. ``prediction_mode == "none"``
        # (default) yields None, leaving ``_c_prior`` None -> byte-identical to
        # today. Scaled by ``sentence_priming_scale`` to match the chat-loop
        # prime exactly.
        seed_payload = None
        _seed = self._consume_intersentence_seed()
        if _seed is not None:
            _, payload_hat = _seed
            if payload_hat.shape[-1] == int(
                    self.conceptualSpace.stm.concept_dim):
                seed_payload = (
                    payload_hat * float(self.sentence_priming_scale))
                self.conceptualSpace._c_prior = seed_payload
                self.conceptualSpace._c_prior_slotwise = True
        object.__setattr__(self, "_intersentence_seed_payload", seed_payload)

        N_target = (int(in_event.shape[1])
                    if in_event is not None and in_event.dim() == 3
                    else None)
        word_carrier = in_sub

        # Static per-word loop scaffolding
        # (doc/plans/2026-05-20-static-per-word-loop-impl.md §2).
        # ``_target_cursor_length`` tells SymbolSpace.compose to pad the
        # S-space_role rule cursor to N with ``id_SS`` (forward-only). The
        # per-iteration commit-time gate is sourced from
        # ``inputSpace._word_active_mask`` (tensor-only, compile-stable);
        # the rule cursor itself stays aligned for downstream consumers
        # but does NOT drive control flow inside the captured body.
        _word_slab = getattr(self.inputSpace, "_ar_embedded_N", None)
        N_static = (int(_word_slab.shape[1])
                    if torch.is_tensor(_word_slab)
                    and _word_slab.dim() == 3
                    else int(getattr(
                        self.inputSpace, "_serial_word_capacity",
                        self.inputSpace.outputShape[0])))
        if self.symbolSpace is not None:
            object.__setattr__(
                self.symbolSpace, "_target_cursor_length", N_static)
        # ``_per_word_contributions`` holds the per-position ``[B, D_c]``
        # contributions (zero at inactive batch rows / padding columns) for
        # ``torch.stack`` after the loop; a python list + stack preserves
        # autograd flow through the per-position concept outputs (an in-place
        # write into a non-grad buffer would silently detach). Seeded empty
        # here; ``_forward_body_per_word`` reallocates it to a FIXED-LENGTH
        # ``N_static`` list (index = word position) so the accumulator length
        # no longer varies with the sentence word-count (a Dynamo recompile
        # driver -- see the loop site).
        self._per_word_contributions: list = []
        return stm, N_target, word_carrier, in_event

    # -- §6c sentence protocol (GrammarOpsPass; author 2026-06-11) ------
    # (The §6 preemption knobs PREEMPT_THRESHOLD/HYSTERESIS left with the
    # mid-sentence reground's removal, 2026-07-07; the layer-level
    # ``preemption_signal`` keeps its own defaults for the reasoning path.)

    def _set_serial_pump(self, flag):
        """Stamp the per-pump mode for the §6d update law (mode is a
        PER-PUMP property within a serial sentence — §6b/§6c): the
        parallel prelude runs with ``serial_pump=False`` (percepts
        move), the per-word serial ticks with ``True`` (references
        move). ``None`` un-stamps (the law falls back to the legacy
        ``serial_mode`` read)."""
        for sp in (getattr(self, 'perceptualSpace', None),
                   getattr(self, 'wholeSpace', None)):
            if sp is not None:
                sp.serial_pump = flag

    def _sentence_prelude(self, in_event, word_carrier):
        """Pump zero (§6c): one independent PARALLEL prelude per serial
        sentence — ``subsymbolicOrder`` whole-slab pumps up the space_role
        ladder (the π carving with the σ fusion: the Gelug first
        moment, direct and non-conceptual), seeding BOTH codebook
        towers (EMA on — the word-learning guarantee).

        Commit is INTENT-ONLY (sign-off): the terminal CS event's
        pooled content is the gist, fed to §5 ``set_intent`` (one
        intent priming both towers); NOTHING enters the workspace —
        the empty-seed feedback pointers are restored after the pumps,
        the STM is untouched (no push), and ``_per_word_prelude``'s
        sentence state is re-read by the serial loop exactly as if the
        prelude had not run. The §6d law sees these pumps as PARALLEL
        (``serial_pump=False``); the per-word ticks that follow run
        with ``serial_pump=True``. The mutex is honored throughout —
        one commit per pump, interference avoided by temporal
        separation.

        Returns the gist (or ``None``). Eager-path only for now (the
        protocol is config-gated OFF by default; the captured-graph
        per-word path never sees it).
        """
        T = max(1, int(getattr(self, 'subsymbolicOrder', 1) or 1))
        cs = self.conceptualSpace
        ws = self.wholeSpace
        prev_ps, prev_ss = self._prev_cs_for_ps, self._prev_cs_for_ss
        self._set_serial_pump(False)
        # Intent-only means exactly that: the parallel perceptual prelude may
        # form a gist, but its eight-slot fields must not prefill the reducing
        # STM before surface-word ingestion begins.
        _defer_gist_push = bool(getattr(self, "serial_object_meta", False))
        object.__setattr__(cs, "_serial_defer_push", _defer_gist_push)
        try:
            if in_event is not None:
                word_carrier.set_event(in_event)
                word_carrier.stem_embedded = True
            gist = None
            # Dual view (serialObjectMeta; Alec 2026-06-18): "PS/WS look at
            # INPUT for the first step, then process subsymbolically." Both
            # towers see the input at pump 0 -- PS the atom/input view
            # (``word_carrier``) and WS the UNITY view
            # (``_staged_concepts_in``, ``[B,1,N]``). Pump 0's recurrent CS is
            # the EMPTY seed, so passing IS_concepts routes WS through
            # ``_stage0_unity_forward`` (the legal stage-0 path -- no repeated-
            # injection NotImplementedError). Later pumps read the recurrent CS
            # (subsymbolic). Gated so other serial configs' prelude is unchanged.
            _dual = bool(getattr(self, 'serial_object_meta', False))
            _unity = getattr(self, '_staged_concepts_in', None)
            for _i in range(T):
                PS_sub = self.perceptualSpace.forward(word_carrier)
                WS_sub = ws.forward(getattr(self, "_ws_universe", None),
                                    cs_out=self._prev_cs_for_ss)
                CS_sub = cs.forward(PS_sub, WS_sub)
                object.__setattr__(self, "_prev_cs_for_ps", cs._subspaceForPS)
                object.__setattr__(self, "_prev_cs_for_ss", cs._subspaceForWS)
                if CS_sub is not None:
                    idea = CS_sub.materialize()
                    if idea is not None and idea.dim() == 3:
                        gist = idea.mean(dim=(0, 1))
            self._last_gist = (gist.detach() if gist is not None
                               else None)
            self.set_intent(self._last_gist)
            if not torch.compiler.is_compiling():
                self._prelude_pumps = int(self._prelude_pumps) + 1
        finally:
            # Intent-only: the star never becomes a workspace slot.
            object.__setattr__(cs, "_serial_defer_push", False)
            object.__setattr__(self, "_prev_cs_for_ps", prev_ps)
            object.__setattr__(self, "_prev_cs_for_ss", prev_ss)
            self._set_serial_pump(True)
        return self._last_gist

    def _push_kind_word_col(self, p, B):
        """Per-row WORD tag for the trip-``p`` STM push (slot-kind
        provenance, todo §1 dilution fix): a push is a word only when the
        slot's percept FILLS its word-tile span — the same fill test the
        1:1 recognition uses — not merely when the slot belongs to a word
        group (early byte-run slots do) or when the slab column is active
        (padding columns are, on muxed events). Missing records tag
        'other' (strict; no words until the towers say so)."""
        ps_fwd = (getattr(self.perceptualSpace, "_forward_input", None)
                  or {})
        word_active = ps_fwd.get("word_active_mask")
        word_truncated = ps_fwd.get("word_truncated_mask")
        if (torch.is_tensor(word_active) and word_active.dim() == 2
                and p < int(word_active.shape[1])):
            active_col = word_active[:, p].detach().to("cpu").tolist()
            if (torch.is_tensor(word_truncated)
                    and tuple(word_truncated.shape)
                    == tuple(word_active.shape)):
                trunc_col = word_truncated[:, p].detach().to("cpu").tolist()
            else:
                trunc_col = [False] * len(active_col)
            return [bool(a) and not bool(t)
                    for a, t in zip(active_col, trunc_col)]
        pid = ps_fwd.get("indices")
        spans = ps_fwd.get("tile_spans")
        store = ps_fwd.get("percept_store")
        if (not torch.is_tensor(pid) or pid.dim() != 2
                or p >= int(pid.shape[1]) or spans is None
                or store is None or not hasattr(store, "bytes_for")):
            return [False] * B
        col = pid[:, p].detach().to("cpu").tolist()
        out = []
        for b in range(B):
            ok = False
            if b < len(spans) and p < len(spans[b]) and int(col[b]) >= 0:
                s, e = spans[b][p]
                if e > s:
                    try:
                        ok = len(store.bytes_for(int(col[b])) or b"") \
                            == int(e) - int(s)
                    except (IndexError, TypeError, ValueError):
                        ok = False
            out.append(ok)
        return out

    def _aligned_word_chunk2(self, words_b_2_d, gates_b_2):
        """Run two canonical aligned word ticks as one reusable graph.

        The eager driver presents a two-column view through InputSpace's
        ordinary staged attributes, so the existing word cell can use local
        positions 0 and 1.  No sentence-global ``p`` value or growing output
        list enters this callable.  Contributions are returned as tensors and
        installed in the sentence-wide lists by the driver after replay.
        """
        if (words_b_2_d.dim() != 3 or int(words_b_2_d.shape[1]) != 2
                or gates_b_2.dim() != 2 or int(gates_b_2.shape[1]) != 2):
            raise RuntimeError(
                "aligned word chunk requires words [B,2,D] and gates [B,2]")

        cs = self.conceptualSpace
        # Keep the Python accumulator structure identical at every replay.
        # The fixed-capacity predictor appends one term per local word,
        # including a zero-weight first-word term when STM is empty.
        cs._intra_loss_accum = None
        cs._intra_loss_weight_accum = None
        cs._intra_loss_count = 0
        concept_slots = [None, None]
        percept_slots = [None, None]
        self._per_word_percept_contributions = percept_slots

        for local_p in range(2):
            self._per_word_body_step(
                words_b_2_d[:, local_p:local_p + 1, :],
                local_p,
                gates_b_2[:, local_p:local_p + 1],
                concept_slots,
                active_host=True)

        if (concept_slots[0] is None or concept_slots[1] is None
                or percept_slots[0] is None or percept_slots[1] is None):
            raise RuntimeError(
                "canonical aligned chunk did not emit both concept and "
                "percept contributions")
        return (
            torch.stack(concept_slots, dim=1),
            torch.stack(percept_slots, dim=1),
        )

    def _aligned_word_chunk1(self, words_b_1_d, gates_b_1):
        """Run one canonical word tick in a static fullgraph kernel.

        K=1 avoids the compilation blow-up of the two-tick stateful AOT
        backward while still containing the complete numerical PS/WS/CS/STM
        recurrence for one word.  The outer adapter establishes sentence order
        and passes a fixed residual-part bucket on every replay.
        """
        if (words_b_1_d.dim() != 3 or int(words_b_1_d.shape[1]) != 1
                or gates_b_1.dim() != 2 or int(gates_b_1.shape[1]) != 1):
            raise RuntimeError(
                "aligned word chunk requires words [B,1,D] and gates [B,1]")
        cs = self.conceptualSpace
        cs._intra_loss_accum = None
        cs._intra_loss_weight_accum = None
        cs._intra_loss_count = 0
        concept_slots = [None]
        percept_slots = [None]
        self._per_word_percept_contributions = percept_slots
        self._per_word_body_step(
            words_b_1_d, 0, gates_b_1, concept_slots, active_host=True)
        if concept_slots[0] is None or percept_slots[0] is None:
            raise RuntimeError(
                "canonical aligned chunk did not emit concept and percept "
                "contributions")
        return (
            concept_slots[0].unsqueeze(1),
            percept_slots[0].unsqueeze(1),
        )

    @torch.compiler.disable
    def _run_aligned_word_chunk_loop(self, out_slot, n_trips):
        """Eager sentence driver around the reusable two-word graph.

        Only fixed-shape, canonical-layout tensors cross into the compiled
        callable. The full sparse concept bank remains installed and grad-bearing for the
        whole sentence; the temporary InputSpace swaps affect only surfaces
        whose second axis the legacy word cell indexes by ``p``.
        """
        step = self._compiled_word_chunk_step
        chunk_width = int(getattr(self, "_compiled_word_chunk_width", 2))
        isp = self.inputSpace
        slab = getattr(isp, "_ar_embedded_N", None)
        if (step is None or not torch.is_tensor(slab)
                or chunk_width < 1 or int(n_trips) < chunk_width
                or int(n_trips) % chunk_width != 0):
            return None
        n_trips = int(n_trips)
        if int(slab.shape[1]) < n_trips:
            return None

        active = getattr(isp, "_word_active_mask", None)
        if active is None:
            active = torch.ones(
                int(slab.shape[0]), n_trips, dtype=torch.bool,
                device=slab.device)

        staged_names = (
            "_ar_word_part_ids",
            "_ar_word_part_mask",
            "_ar_word_part_offsets",
            "_ar_word_concept_rows",
            "_ar_word_concept_orders",
            "_word_last_slot_mask",
        )
        dynamic_part_names = {
            "_ar_word_part_ids",
            "_ar_word_part_mask",
            "_ar_word_part_offsets",
        }
        full = {name: getattr(isp, name, None) for name in staged_names}
        required = tuple(full[name] for name in staged_names)
        if not all(torch.is_tensor(value) for value in required):
            return None

        # The residual-part axis is semantic but normally tiny.  Marking it
        # dynamic over InputSpace's 8192-byte ceiling made MPS Inductor lower
        # a very broad symbolic graph for every K=1/K=2 word cell.  Instead,
        # stage an exact padded bucket (8/16/32/...) once per sentence.  The
        # sentinel IDs and false mask make the added lanes identity elements;
        # nothing is truncated.  A finite bucket set gives a bounded number of
        # fullgraph specializations while retaining arbitrary long residual
        # words through the next power-of-two fallback.
        part_width = int(full["_ar_word_part_ids"].shape[-1])
        part_bucket = next(
            (width for width in (8, 16, 32, 64, 128)
             if part_width <= width),
            1 << max(0, part_width - 1).bit_length())
        staged = dict(full)
        if part_width < part_bucket:
            for name in dynamic_part_names:
                value = full[name]
                pad_shape = tuple(value.shape[:-1]) + (
                    part_bucket - part_width,)
                fill = False if name == "_ar_word_part_mask" else -1
                pad = torch.full(
                    pad_shape, fill, dtype=value.dtype, device=value.device)
                staged[name] = torch.cat((value, pad), dim=-1)

        ps_out_slot = self._per_word_percept_contributions
        cs = self.conceptualSpace
        loss_terms = list(getattr(cs, "_intra_loss_accum", None) or [])
        loss_weights = list(
            getattr(cs, "_intra_loss_weight_accum", None) or [])
        last_cs = None
        was_replaying = bool(getattr(
            self, "_compiled_word_chunk_replaying", False))
        self._compiled_word_chunk_replaying = True
        try:
            for start in range(0, n_trips, chunk_width):
                stop = start + chunk_width
                for name in staged_names:
                    value = staged[name]
                    # A view retains its sentence slab's W-dependent leading
                    # stride, even though its visible shape is always K=2.
                    # Force a fresh canonical layout so W=16/32/64/128 all
                    # satisfy the same Dynamo guards (``contiguous()`` alone
                    # may return the original view when B=1).
                    chunk_value = value[:, start:stop].clone(
                        memory_format=torch.contiguous_format)
                    object.__setattr__(isp, name, chunk_value)
                word_chunk = slab[:, start:stop, :].clone(
                    memory_format=torch.contiguous_format)
                gate_chunk = active[:, start:stop].clone(
                    memory_format=torch.contiguous_format)
                concept_chunk, percept_chunk = step(
                    word_chunk, gate_chunk)
                for local_p in range(chunk_width):
                    out_slot[start + local_p] = concept_chunk[:, local_p, :]
                    if isinstance(ps_out_slot, list):
                        ps_out_slot[start + local_p] = (
                            percept_chunk[:, local_p, :])
                loss_terms.extend(
                    list(getattr(cs, "_intra_loss_accum", None) or []))
                loss_weights.extend(list(
                    getattr(cs, "_intra_loss_weight_accum", None) or []))
                last_cs = cs.subspace
        finally:
            for name, value in full.items():
                object.__setattr__(isp, name, value)
            self._per_word_percept_contributions = ps_out_slot
            self._compiled_word_chunk_replaying = was_replaying

        cs._intra_loss_accum = loss_terms or None
        cs._intra_loss_weight_accum = loss_weights or None
        cs._intra_loss_count = len(loss_terms)
        stm = cs.stm
        if stm is not None and torch.is_tensor(getattr(stm, "_depth", None)):
            # One boundary sync, outside every compiled replay.  Snapshot and
            # the NULL-seal reducer need the current logical width; no word
            # cell reads this Python mirror in chunk mode.
            stm._max_depth_host = int(stm._depth.max().item())
        return last_cs

    def _per_word_body_step(self, w, p, gate_b_1, out_slot, active_host=True):
        """One per-word iteration of the static loop (replaces the
        legacy data-dependent ``while next_word()`` body).

        Static-N variant of the constant-shape PS/WS -> CS sub-graph
        replayed ``N = InputSpace.outputShape[0]`` times per forward.
        Per-iteration side-effects (carrier event commits, STM push,
        concept-buffer scatter) are masked by ``gate_b_1`` so padding
        columns leave recurrent state bit-identical to the active
        prefix. See
        doc/plans/2026-05-20-static-per-word-loop-impl.md §2.4.

        Args:
          w: ``[B, 1, D]`` word slice for position ``p``
             (``InputSpace.word_at(p)``); zero at inactive batch rows.
          p: Python int loop position; constant per unrolled iteration.
          gate_b_1: ``[B, 1]`` bool active mask from
             ``inputSpace._word_active_mask[:, p:p+1]``. ``torch.where``
             gate for masked commits.
          out_slot: fixed-length ``N_static`` python list (per-position
             concept accumulator); this call writes slot ``p`` by its
             python-constant index (``None`` in unwritten slots).

        Returns ``(CS_sub, idea_bd)`` — last_cs tracking and idea
        broadcast handle. Both are produced unconditionally (PS/WS/CS
        forwards always fire); only their downstream effects are gated.

        Capture-gate contract: still the ONLY callable that runs inside
        the middle captured graph; every helper it calls remains
        DtoH-free. ``test/test_per_word_capture_gate.py`` is the active
        gate.
        """
        isp = self.inputSpace
        ps = self.perceptualSpace
        cs = self.conceptualSpace
        ws = self.wholeSpace
        stm = cs.stm
        word_carrier = isp.subspace

        # Word-major radix: InputSpace presents this word's discrete local
        # part state; PartSpace consumes and synthesizes it inside PS.forward.
        # The sentence loop advances over W words while the PS-local part axis
        # is reset for each presentation.
        local_part_ids = (isp.word_part_ids_at(p)
                          if hasattr(isp, "word_part_ids_at") else None)
        local_part_mask = (isp.word_part_mask_at(p)
                           if hasattr(isp, "word_part_mask_at") else None)
        local_part_offsets = (isp.word_part_offsets_at(p)
                              if hasattr(isp, "word_part_offsets_at")
                              else None)
        word_major = local_part_ids is not None and local_part_mask is not None

        # Gaussian window + MPHF route. ``p`` is the static loop position;
        # the legacy ``_per_word_cursor - 1`` mirror is retired with
        # ``next_word()``.
        _full_seq = isp._ar_embedded
        # serialObjectMeta: HARD-MASK-TO-WORD-SPAN — replace the soft gaussian
        # window with the hard same-word window so PS processes the ACTIVE WORD
        # ONLY (the word's block; no slot outside it). Whole-sentence context
        # re-enters via the §6c prelude's gist/intent, not this window. Flag-off
        # keeps the gaussian path byte-identical.
        if word_major:
            _ctx_w = None
        elif getattr(self, 'serial_object_meta', False):
            _ctx_w = self.word_span_window(
                _full_seq, p, getattr(isp, '_word_index_N', None))
        else:
            _ctx_w = self.gaussian_window_word(_full_seq, p)
        if (not word_major and _ctx_w is not None
                and torch.is_tensor(_ctx_w)
                and _ctx_w.dim() == 3
                and _ctx_w.shape[1] == 1
                and w is not None and torch.is_tensor(w)
                and _ctx_w.shape[0] == w.shape[0]
                and _ctx_w.shape[-1] == w.shape[-1]):
            w = _ctx_w

        w, _mphf_idx = self._mphf_route_word(w, p)
        if _mphf_idx is not None:
            self._mphf_last_idx = _mphf_idx
            self._mphf_call_count = self._mphf_call_count + 1

        word_carrier.set_event(w)
        word_carrier.stem_embedded = True
        object.__setattr__(word_carrier, "_word_local_parts", word_major)
        object.__setattr__(word_carrier, "_word_part_ids", local_part_ids)
        object.__setattr__(word_carrier, "_word_part_mask", local_part_mask)
        object.__setattr__(word_carrier, "_word_part_offsets", local_part_offsets)
        word_sub = word_carrier

        # The canonical word path commits once at the word boundary. Its
        # aligned concept option interprets subsymbolicOrder=T as one base
        # field plus T-1 cumulative folds on EACH tower, all inside this word
        # iteration. Existing configs default to the learned mixing path.
        word_commit_mode = bool(
            getattr(self, 'serial_object_meta', False)
            and getattr(isp, '_word_last_slot_mask', None) is not None)
        aligned_fold_binding = bool(
            word_commit_mode
            and getattr(self, "concept_binding", "mixing") == "aligned")
        fold_passes = (
            tuple(range(int(self.subsymbolicOrder) - 1))
            if aligned_fold_binding else ())

        # Read the cross-iteration C→P / C→S feedback off
        # ``self._prev_cs_for_ps/ws``. The prelude seeds these to the
        # empty seeds for iteration 0; the tail of this method switches
        # them to the persistent ``cs._subspaceForPS/WS`` storage that
        # cs.forward mutates in place, so iterations 1+ pick up the
        # latest event without any explicit pointer update.
        prevCS_forPS = self._prev_cs_for_ps
        prevCS_forSS = self._prev_cs_for_ss
        # Snapshot the carriers' event tensors BEFORE cs.forward writes
        # new ones (set_event mutates in place; without snapshots the
        # masked-blend below would compare new-vs-new).
        prev_ps_event_snap = (
            prevCS_forPS._event
            if (prevCS_forPS is not None
                and getattr(prevCS_forPS, '_event', None) is not None)
            else None)
        prev_ws_event_snap = (
            prevCS_forSS._event
            if (prevCS_forSS is not None
                and getattr(prevCS_forSS, '_event', None) is not None)
            else None)

        # Stage 1.A substrate refactor: PartSpace.forward is
        # single-arg now (``pi(x) + sigma(x)`` on the same input —
        # no CS-feedback path entering PS at this level).
        PS_base = ps.forward(word_sub)
        part_folds = None
        if aligned_fold_binding:
            part_event = PS_base.materialize()
            compiled_part_ladder = getattr(
                self, "_compiled_part_fold_ladder", None)
            if (compiled_part_ladder is not None
                    and not torch.compiler.is_compiling()):
                part_folds = compiled_part_ladder(part_event)
            else:
                part_folds = ps.fold_event_ladder(
                    part_event, fold_passes, strict=True)
            PS_base.set_event(part_folds[-1])
            PS_sub = PS_base
        else:
            PS_sub = PS_base
        # Retain one PartSpace result per word so post-loop loss/reverse
        # readers see a word-aligned slab instead of only the last iteration.
        if word_major and PS_sub is not None:
            _ps_event = PS_sub.materialize()
            _ps_slots = getattr(self, "_per_word_percept_contributions", None)
            if (torch.is_tensor(_ps_event) and _ps_event.dim() == 3
                    and int(_ps_event.shape[1]) >= 1
                    and isinstance(_ps_slots, list)
                    and 0 <= p < len(_ps_slots)):
                _ps_slots[p] = torch.where(
                    gate_b_1, _ps_event[:, 0, :],
                    torch.zeros_like(_ps_event[:, 0, :]))
        # Universe every pump (the carrier arrives as cs_out feedback);
        # the earlier bootstrap-only law reacted to a misdiagnosed
        # flatline (valid_mask collapse -- exec notes item 36).
        WS_base = ws.forward(getattr(self, "_ws_universe", None),
                             cs_out=prevCS_forSS)
        # Preserve H0 before the cumulative pi ladder replaces WS_base's live
        # event with H3. H0 is the whole-side peer of the parameter-free P0
        # word union and contributes its own bounded RMS evidence to CS.
        whole_base_event = (
            WS_base.materialize() if aligned_fold_binding else None)
        whole_folds = None
        if aligned_fold_binding:
            whole_event = WS_base.materialize()
            compiled_whole_ladder = getattr(
                self, "_compiled_whole_fold_ladder", None)
            if (compiled_whole_ladder is not None
                    and not torch.compiler.is_compiling()):
                whole_folds = compiled_whole_ladder(whole_event)
            else:
                whole_folds = ws.fold_event_ladder(
                    whole_event, fold_passes, strict=True)
            WS_base.set_event(whole_folds[-1])
            WS_sub = WS_base
        else:
            WS_sub = WS_base
        # ``serialObjectMeta`` makes the grammar loop word-grained even when
        # the radix store still spells an unfamiliar word with several
        # percepts.  Predict/perceive only at the word's final percept: the
        # hard ``word_span_window`` above has already assembled the complete
        # word event by then.  Earlier percepts are representation-building
        # steps, not extra language-model targets.
        #
        # The legacy flag-off path remains slot-grained.  In word mode the
        # outer block below owns the single physical STM push (it also owns
        # capacity back-pressure and grammar dispatch), so ConceptualSpace
        # computes/stashes the prediction but defers its push to that block.
        commit_b_1 = (
            isp._word_last_slot_mask[:, p:p + 1]
            if word_commit_mode else gate_b_1)

        decoded_fold_sources = None
        concept_orders_full = None
        prior_symbol_source = None
        prior_symbol_validity = None
        word_concept_row = None
        word_concept_activation = None
        base_part_count = 0
        base_whole_count = 0
        if aligned_fold_binding:
            # Each tower completes its native recursion first. Only the scalar
            # activation normalized by that source crosses into concept-index
            # space; this normalization is not another learned sigma/pi fold.
            # The indexed CS dictionary read below is the dimensional increase.
            # There is no PS->WS edge and no coordinate adapter owned by CS.
            n_locations = int(cs.outputShape[0])

            def _fit_locations(event, label):
                if not torch.is_tensor(event) or event.dim() != 3:
                    raise RuntimeError(
                        f"{label} native fold must be a [B,N,D] tensor")
                have = int(event.shape[1])
                if have > n_locations:
                    raise RuntimeError(
                        f"{label} native fold has {have} locations but CS "
                        f"accepts {n_locations}; locations cannot be truncated")
                if have == n_locations:
                    return event
                return torch.cat([
                    event,
                    event.new_zeros(
                        int(event.shape[0]), n_locations - have,
                        int(event.shape[2])),
                ], dim=1)

            # P0 is the complete coordinatewise union of this word's residual
            # parts. It is a distinct, parameter-free PS contribution: its
            # activation is RMS(P0), i.e. RMS(union), before any learned sigma
            # order raise. Keep P1..P3 as independent contributions as well.
            # The word-local carrier is mutated to P0 inside PS.forward;
            # PS_base is already the learned/quantized tower result and must
            # not be substituted for this base state.
            base_parts = ()
            if word_major:
                base_event = word_sub.materialize()
                if torch.is_tensor(base_event):
                    base_parts = (_fit_locations(base_event, "PS base P0"),)
            base_part_count = len(base_parts)
            native_parts = tuple(
                _fit_locations(event, f"PS fold {i + 1}")
                for i, event in enumerate(part_folds))
            base_wholes = ()
            if torch.is_tensor(whole_base_event):
                base_wholes = (_fit_locations(whole_base_event, "WS base H0"),)
            base_whole_count = len(base_wholes)
            native_wholes = tuple(
                _fit_locations(event, f"WS fold {i + 1}")
                for i, event in enumerate(whole_folds))
            native_part_sources = base_parts + native_parts
            native_whole_sources = base_wholes + native_wholes
            native_sources = native_part_sources + native_whole_sources
            if not native_sources:
                raise RuntimeError(
                    "aligned concept activation requires native fold sources")

            row_b = (isp.word_concept_rows_at(p)
                     if hasattr(isp, "word_concept_rows_at") else None)
            order_b = (isp.word_concept_orders_at(p)
                       if hasattr(isp, "word_concept_orders_at") else None)
            batch = int(native_sources[0].shape[0])
            if row_b is None:
                row_b = torch.full(
                    (batch,), -1, dtype=torch.long,
                    device=native_sources[0].device)
            else:
                row_b = row_b.to(
                    device=native_sources[0].device, dtype=torch.long)
            if order_b is None:
                order_b = torch.full_like(row_b, -1)
            else:
                order_b = order_b.to(
                    device=native_sources[0].device, dtype=torch.long)
            active_rows = gate_b_1.reshape(-1).to(
                device=row_b.device, dtype=torch.bool)
            row_b = torch.where(active_rows, row_b,
                                torch.full_like(row_b, -1))
            order_b = torch.where(active_rows, order_b,
                                  torch.full_like(order_b, -1))
            concept_rows = row_b.unsqueeze(1).expand(-1, n_locations)
            concept_orders_full = order_b.unsqueeze(1).expand(
                -1, n_locations)

            part_what = int(ps.nWhat)
            whole_what = int(ws.nWhat)
            activations = torch.stack([
                ps.native_fold_activation(event, part_what)
                for event in native_part_sources
            ] + [
                ws.native_fold_activation(event, whole_what)
                for event in native_whole_sources
            ], dim=1)
            source_bands = torch.stack([
                event[..., part_what:] for event in native_part_sources
            ] + [
                event[..., whole_what:] for event in native_whole_sources
            ], dim=1)
            decoded_fold_sources = cs.decode_sparse_concept_rows(
                concept_rows, activations, source_bands,
                staged_rows=getattr(
                    isp, "_ar_concept_lookup_rows", None),
                staged_atoms=getattr(
                    isp, "_ar_concept_lookup_atoms", None))

            # Canonical SS peer: read the complete *prior* STM slab before
            # this word enters CS. Slot i maps directly to location i. The
            # indexed CS lookup is SS's sigma fold / dimensional increase;
            # there is no inventory scan and no current-tick CS->SS->CS edge.
            prior_symbol_source, prior_symbol_validity = (
                cs.decode_prior_stm_peer(
                    stm, active_rows,
                    staged_rows=getattr(
                        isp, "_ar_concept_lookup_rows", None),
                    staged_atoms=getattr(
                        isp, "_ar_concept_lookup_atoms", None)))

            # BasicModel contributes P0..P3 and H0..H3. General aligned
            # fixtures retain their configured fold count; SS increments the
            # denominator only at locations where it exists.
            native_count = int(decoded_fold_sources.shape[1])
            denominator = (
                torch.full_like(
                    prior_symbol_validity, float(native_count),
                    dtype=decoded_fold_sources.dtype)
                + prior_symbol_validity.to(decoded_fold_sources.dtype))
            concept_event = (
                decoded_fold_sources.sum(dim=1) + prior_symbol_source
            ) / denominator.unsqueeze(-1)
            word_concept_row = row_b
            word_concept_activation = (
                activations[:, :, 0].sum(dim=1) / denominator[:, 0])
            word_concept_activation = torch.where(
                word_concept_row >= 0, word_concept_activation,
                torch.zeros_like(word_concept_activation))
            cs.subspace.copy_context(PS_sub)
            cs.subspace.set_event(concept_event)
            cs_input = cs.subspace
            cs_symbol_input = None
        else:
            cs_input = PS_sub
            cs_symbol_input = WS_sub

        object.__setattr__(cs, "_serial_row_gate", commit_b_1)
        object.__setattr__(cs, "_serial_defer_push", word_commit_mode)
        try:
            CS_sub = cs.forward(cs_input, cs_symbol_input)
        finally:
            object.__setattr__(cs, "_serial_row_gate", None)
            object.__setattr__(cs, "_serial_defer_push", False)

        # The live PS and WS fields meet here, inside the current word
        # iteration. CS.forward above is explicitly in defer mode, so field
        # positions are not mistaken for separate workspace pushes.
        #
        # ``aligned``: preserve location indices and aggregate P0..P3 plus
        # H0..H3 (BasicModel T=4 -> eight sources). ``mixing`` retains the
        # historical learned matrix.
        if word_commit_mode and CS_sub is not None:
            if aligned_fold_binding:
                n_part = len(part_folds)
                cs.bind_fold_streams(
                    tuple(decoded_fold_sources[:, base_part_count + i]
                          for i in range(n_part)),
                    tuple(decoded_fold_sources[
                        :, (base_part_count + n_part
                            + base_whole_count + i)]
                          for i in range(len(whole_folds))),
                    CS_sub,
                    part_passes=fold_passes,
                    whole_passes=fold_passes,
                    concept_orders=concept_orders_full,
                    # ConceptualSpace.forward has already applied and consumed
                    # any inter-sentence/chat ``_c_prior``.  This second call is
                    # provenance attachment only; replacing the target event
                    # here would silently erase that prior before the STM push.
                    preserve_target_event=True,
                    symbol_source=prior_symbol_source,
                    symbol_validity=prior_symbol_validity)
                support = getattr(CS_sub, "_fold_support", None)
                object.__setattr__(self, "_last_concept_fold_support", support)
                # META minting is intentionally eager at the sentence
                # boundary. Stage the tensor-free ordered support on its WS
                # owner so those durable concepts retain the same paths.
                object.__setattr__(ws, "_active_fold_support", support)
            else:
                cs.bind_streams(PS_sub, WS_sub, CS_sub)

        # Masked-blend the persistent CS carriers' new events with the
        # snapshots so padding columns preserve the recurrent state
        # from the prior iteration (no ConceptualSpace averaging leak
        # from WS at zero input). Shapes must match — on iteration 0
        # the prev is the empty seed (length 0) and the blend is
        # skipped; the cs.forward result propagates as-is.
        self._maybe_blend_event(
            cs._subspaceForPS, prev_ps_event_snap, gate_b_1)
        self._maybe_blend_event(
            cs._subspaceForWS, prev_ws_event_snap, gate_b_1)

        # Switch the prev pointer to the persistent CS storage so the
        # next iteration's first reads see CS.forward's in-place
        # writes. After this line ``self._prev_cs_for_ps`` aliases
        # ``cs._subspaceForPS``; subsequent iterations read the same
        # object whose ``_event`` cs.forward keeps fresh.
        object.__setattr__(self, "_prev_cs_for_ps", cs._subspaceForPS)
        object.__setattr__(self, "_prev_cs_for_ss", cs._subspaceForWS)

        # Stage 1.F substrate refactor (doc/plans/2026-05-26-two-loop-
        # pi-sigma-substrate.md): the per-stage ``_cs_cache`` /
        # ``_ws_cache`` capture lists are retired. The terminal C-space_role
        # idea is owned by ``cs.stm`` (the bookkeeping push fires
        # below via ``stm.push_step_masked``, which is host-gated to
        # active iterations); the terminal symbolic subspace is owned
        # by ``self.wholeSpace.subspace`` (overwritten in place by
        # ``ws.forward`` above on every iteration). On padding
        # iterations the WS write is harmless (the input was empty)
        # and downstream readers gate on the active-host mask
        # / STM depth, so no extra guard is needed here.
        if WS_sub.is_empty():
            WS_sub = self._zero_symbol_subspace(ws, word_sub)

        idea_bd = None
        if CS_sub is not None:
            idea = CS_sub.materialize()
            if (idea is not None and idea.dim() == 3
                    and idea.shape[1] >= 1):
                idea_bd = idea[:, 0, :]                 # [B, D_c]
                # Concept index-read (snap design doc §ontology, Alec
                # 2026-07-15): light up the word's ALREADY-KNOWN concept
                # (minted by the parallel path) — replace the idea's
                # content slice with the concept's random signed-hypersphere
                # row so the fold consumes a separable concept, not the
                # collapsing percept-binding. Default-off -> byte-identical;
                # eager-only (host resolve). No-op where the percept is
                # untied (untrained / not-yet-minted).
                if (getattr(self, 'concept_index_read', False)
                        and not torch.compiler.is_compiling()
                        and idea_bd is not None):
                    idea_bd = self._maybe_concept_index_read(idea_bd)
                if stm is not None and active_host:
                    # serialObjectMeta: fire the STM push ONCE PER WORD (the
                    # last-slot-of-word commit gate) so a multi-slot
                    # (radix-spelled) word pushes ONE idea, not one per byte.
                    # Flag-off / no word-index -> commit_b_1 IS gate_b_1
                    # (per-slot, byte-identical). The host depth mirror stays
                    # per-iteration -- a conservative UPPER bound that may
                    # trigger an early (but safe) reduce in radix multi-slot
                    # words; raise stmCapacity in the serial config if a long
                    # sentence would trip it.
                    # ``commit_b_1`` was resolved before CS.forward so the
                    # predictor target and the physical STM commit share the
                    # exact same word boundary.
                    # Capacity demand fires here, immediately before a word
                    # that would overflow a full row.  It may only select a
                    # binary grammatical operator; silently rolling off the
                    # oldest constituent would hide a failed parse.
                    # ``push_step_masked`` is gated on ``active_host``
                    # so padding iterations never index past depth and
                    # never trip a capacity OOB.
                    #
                    # Recompile-churn fix (doc/plans/2026-07-04): the loop now
                    # runs the full static per-word slab width, so a SHORT
                    # sentence's padding columns (``commit_b_1`` all-False)
                    # reach here. ``push_step_masked`` is a masked rolling roll
                    # -> a byte-identical no-op there, and the mirror
                    # over-advance those columns cause is corrected by the
                    # single post-loop re-pin in ``_forward_body_per_word``
                    # (mirror only sizes ``snapshot()``; back-pressure stays
                    # dormant below capacity, so a below-cap sentence's padding
                    # never trips the reduce -- the pre-filled configs that DO
                    # reach cap fill every column, i.e. have no padding).
                    # Relative protection (depth-3 campaign, 2026-07-13 +
                    # review finding same day): computed ONCE per word from
                    # the rules the PREVIOUS fires populated, and threaded
                    # into the capacity demand and the online pressure
                    # decision below — the batch-global _max_depth_host mirror
                    # otherwise force-folds a finished relative row below its
                    # protected depth-3 end-state. Eager-only, like the
                    # per-word fire that feeds it (compile has no per-word
                    # rules to read). Depth <= 3 rows are never the rows
                    # actually at capacity, so protection cannot block a
                    # genuinely forced fold.
                    _protect = None
                    if (not torch.compiler.is_compiling()
                            and stm._buffer is not None):
                        _Bp = int(stm._buffer.shape[0])
                        _rel = self._sentence_relative_mask(
                            _Bp, device=stm._buffer.device)
                        _protect = torch.where(
                            _rel,
                            torch.full((_Bp,), 3, dtype=stm._depth.dtype,
                                       device=_rel.device),
                            torch.ones(_Bp, dtype=stm._depth.dtype,
                                       device=_rel.device))
                    # Predict from the retained reduced context against the
                    # freshly bound word concept. Its actual subsymbolic order
                    # is carried explicitly from the aligned fold support;
                    # grammatical depth starts separately at zero.
                    _compiled_recurrent = bool(
                        self._compiled_word_chunk_replaying
                        or (torch.compiler.is_compiling()
                            and getattr(self,
                                        "_compiled_word_loop_fullgraph",
                                        False)))
                    if _compiled_recurrent:
                        cs._stm_predict_then_perceive_serial_fixed(
                            idea_bd, row_gate=commit_b_1)
                    else:
                        cs._stm_predict_then_perceive_serial(
                            idea_bd, row_gate=commit_b_1)
                    _concept_orders = getattr(
                        CS_sub, "_concept_orders", None)
                    if (torch.is_tensor(_concept_orders)
                            and _concept_orders.dim() == 2
                            and int(_concept_orders.shape[0])
                            == int(idea_bd.shape[0])):
                        _word_concept_order = _concept_orders[:, 0].to(
                            device=idea_bd.device, dtype=torch.long)
                    else:
                        _word_concept_order = torch.full(
                            (int(idea_bd.shape[0]),), -1,
                            dtype=torch.long, device=idea_bd.device)
                    _word_grammar_order = torch.zeros_like(
                        _word_concept_order)
                    _pressure_controller = bool(
                        getattr(self, "serial_object_meta", False)
                        and getattr(self, "serial_word_capacity", None)
                        is not None)
                    _binary_used = torch.zeros(
                        int(idea_bd.shape[0]), dtype=torch.bool,
                        device=idea_bd.device)
                    _legacy_pre_binary = False
                    if _compiled_recurrent:
                        # A captured recurrence cannot branch on the Python
                        # high-water mirror.  Compute the ordinary demand
                        # gate from the live tensor depth; rows below capacity
                        # are exact masked no-ops.  The post-insert reducer
                        # below still receives ``~_binary_used``, preserving
                        # the one-binary-op budget when a demand fold did fire.
                        _full_active = (
                            (stm._depth >= int(stm.capacity))
                            & commit_b_1.reshape(-1).to(
                                device=stm._depth.device,
                                dtype=torch.bool))
                        _demand_reduced = self._stm_bounded_reduce_step(
                            protect_depth=_protect,
                            row_gate=_full_active,
                            demand=True)
                        _binary_used = _binary_used | _demand_reduced
                    elif stm._max_depth_host >= stm.capacity:
                        if _pressure_controller:
                            _full_active = (
                                (stm._depth >= int(stm.capacity))
                                & commit_b_1.reshape(-1).to(
                                    device=stm._depth.device,
                                    dtype=torch.bool))
                            _demand_reduced = self._stm_bounded_reduce_step(
                                protect_depth=_protect,
                                row_gate=_full_active,
                                demand=True)
                            _binary_used = _binary_used | _demand_reduced
                        else:
                            # Legacy serial configurations retain their
                            # unconditional structural back-pressure fold.
                            _demand_reduced = self._stm_bounded_reduce_step(
                                protect_depth=_protect,
                                gate_tau=self.stm_reduce_tau)
                            _binary_used = _binary_used | _demand_reduced
                            stm._max_depth_host = stm.capacity - 1
                            _legacy_pre_binary = True
                    stm.push_step_masked(
                        idea_bd, commit_b_1,
                        orders=_word_concept_order,
                        grammar_orders=_word_grammar_order,
                        concept_row=word_concept_row,
                        concept_activation=word_concept_activation)
                    # Slot-kind provenance (open-fronts Task B; dilution fix
                    # same day): a push is a WORD only when PS's word grid
                    # says column p carries a real word (pad groups are -1)
                    # — the commit gate alone is the always-true slab
                    # activity on muxed events, which tagged 6 padding
                    # pushes per row as words. Padding pushes still note
                    # 'other' (the kind stacks must mirror the buffer).
                    if (not torch.compiler.is_compiling()
                            and getattr(stm, "_slot_kinds", None) is not None):
                        gate_rows = commit_b_1.view(-1).detach().to(
                            "cpu").tolist()
                        wcol = self._push_kind_word_col(p, len(gate_rows))
                        stm.note_push_masked(
                            [g and w for g, w in zip(gate_rows, wcol)],
                            "word")
                        stm.note_push_masked(
                            [g and not w for g, w in zip(gate_rows, wcol)],
                            "other")
                    if not _compiled_recurrent:
                        stm._max_depth_host = stm._max_depth_host + 1
                    # Per-word router fire (Alec 2026-07-13): parse as the
                    # word arrives — the STM cannot hold the sentence
                    # until a boundary-only parse. Eager-only.
                    if not torch.compiler.is_compiling():
                        self._chart_compose_per_word()
                    # Normal grammatical parse DURING reading: one controller
                    # decision follows each word insertion. At low occupancy the
                    # rule-count-neutral grammar confidence must clear
                    # <stmReduceTau> by itself.  The threshold falls as the
                    # configured STM fills and becomes a hard demand at
                    # capacity.  This replaces the arbitrary two reduction
                    # attempts per word, which collapsed random adjacent pairs
                    # simply because two binary operators gave the summed DP
                    # marginal a 2/3 prior.
                    # Relative protection (depth-3 campaign, 2026-07-13):
                    # the per-word fire above just populated THIS sentence's
                    # rules, so the mid-read folds honor the same depth-3
                    # floor as the sweep -- without it a relative sentence
                    # is already collapsed below depth 3 before the
                    # protected sweep runs. Eager-only, like the fire that
                    # feeds it (compile has no per-word rules to read).
                    _protect = None
                    if (not torch.compiler.is_compiling()
                            and stm._buffer is not None):
                        _Bp = int(stm._buffer.shape[0])
                        _rel = self._sentence_relative_mask(
                            _Bp, device=stm._buffer.device)
                        _protect = torch.where(
                            _rel,
                            torch.full((_Bp,), 3, dtype=stm._depth.dtype,
                                       device=_rel.device),
                            torch.ones(_Bp, dtype=stm._depth.dtype,
                                       device=_rel.device))
                    # The word budget is strict: at most one binary grammar
                    # application, even when a capacity-demand reduction was
                    # needed before insertion. One unary decision may then apply
                    # to the surviving top concept. This admits the useful
                    # binary+unary pair without an unbounded per-word cascade.
                    _post_binary_gate = (
                        commit_b_1.reshape(-1).to(
                            device=_binary_used.device, dtype=torch.bool)
                        & ~_binary_used)
                    if _pressure_controller:
                        self._stm_bounded_reduce_step(
                            protect_depth=_protect,
                            gate_tau=self.stm_reduce_tau,
                            row_gate=_post_binary_gate,
                            occupancy_pressure=True)
                    else:
                        # Older serial configurations share the same bounded
                        # performance contract: one binary opportunity.
                        if not _legacy_pre_binary:
                            self._stm_bounded_reduce_step(
                                protect_depth=_protect,
                                gate_tau=self.stm_reduce_tau)
                    self._stm_bounded_unary_step(row_gate=commit_b_1)
                # Masked contribution: at inactive batch rows / padding
                # columns the contribution is zero so it doesn't push
                # bogus state into downstream concept reads. ``out_slot``
                # is the caller's FIXED-LENGTH per-position buffer (a list
                # preallocated to ``N_static``); write column ``p`` by its
                # python-constant index (was ``append`` -> a length that grew
                # with the trip count, a Dynamo recompile driver). The caller
                # ``torch.stack``s all columns post-loop so autograd flows
                # through the per-position contributions.
                contribution = torch.where(
                    gate_b_1, idea_bd, torch.zeros_like(idea_bd))
                if (out_slot is not None and isinstance(out_slot, list)
                        and 0 <= p < len(out_slot)):
                    out_slot[p] = contribution
        return CS_sub, idea_bd

    @staticmethod
    def _maybe_blend_event(carrier, prev_event, gate_b_1):
        # Masked-blend ``carrier._event`` against ``prev_event`` using
        # ``gate_b_1`` ([B, 1] bool). Skips when the carrier is missing,
        # the new event is missing, or shapes don't match (first
        # iteration after the empty seed). Same-shape blends are pure
        # tensor ops — no DtoH, no graph break.
        #
        # ``prev_event`` is detached before use so the blend never
        # chains into the previous iteration's autograd graph at a
        # padding column — without this, repeated forward/backward
        # cycles can hit "Trying to backward through the graph a
        # second time" when an inactive batch row's carrier reuses a
        # saved intermediate that has already been freed.
        if carrier is None or prev_event is None:
            return
        new_ev = getattr(carrier, '_event', None)
        if new_ev is None or new_ev.shape != prev_event.shape:
            return
        gate_3d = gate_b_1.unsqueeze(-1)  # [B, 1, 1] broadcasts over D
        carrier._event = torch.where(gate_3d, new_ev, prev_event.detach())

    def _forward_body_per_word(self, in_sub):
        """Per-word IR-reconstruction body (2b-1 capstone).

        The FRONT of the forward when the InputSpace per-word cursor is
        enabled (grammar configs). Replaces the whole-slab
        representation-build of :meth:`_forward_body` with an internal
        per-word loop -- **ONE forward = ONE sentence = ONE IR loss =
        ONE backward**:

          ``inputSpace.run_word_loop(per_word_cell, W)``:
            * the ``[B,1,D]`` ground-truth word ``w`` is run through the
              SAME per-stage PS->CS->WS recurrent cell as the whole-slab
              path (``subsymbolicOrder`` passes, with the C->P / C->S
              feedback identical to ``_forward_body``), but with a
              single-word ``in_sub`` instead of the whole slab;
            * the resulting per-word terminal concept (the last pass's
              CS event, ``[B,1,D_c]``) is inserted into
              ``conceptualSpace.stm`` via the vectorised masked push;
          the statically masked W-position loop ends the sentence traversal.

        Then the accumulated STM feeds the **EXISTING** compose-to-S
        chart (``_chart_compose_at_C``) and the method returns the
        terminal CS subspace so the existing ``_forward_head`` +
        ``runBatch`` P-space_role masked-LM IR-loss + ``reverse()`` TAIL run
        **unchanged** (guiding principle: minimise new training-critical
        surface -- reuse the existing IR machinery verbatim; only the
        representation-build FRONT changes from whole-slab to per-word
        accumulation into STM).

        IR-loss faithfulness: the IR mask is created exactly as in the
        whole-slab path -- via the unchanged ``self.create_ir_mask`` --
        on the **first word's** pass-0 perceptual event (the faithful
        per-word analogue of "pass 0's perceptual event").
        ``runBatch``'s masked-LM loss then reads the post-body
        PartSpace event vs the snapshotted pre-mask embedding at
        the mask positions, byte-identically to today (no new loss
        surface, no compose/reverse loss rewiring).

        STM is sentence-scoped: ``ConceptualSpace.stm`` is sized to
        ``<stmCapacity>`` (sentence length -- the CKY+resize-
        equivalent baseline per the two-loop spec's Phase-1-D §3); the
        bounded soft REDUCE-to-<=7 over STM is the SEPARATE 2b-2
        increment (out of scope here). ``push_step`` requires depth to
        start at 0, so the STM is cleared once at loop entry (the
        sentence-boundary clear that ``ConceptualSpace.Reset(hard=True)``
        also performs).
        """
        # Prelude: STM resize+clear, MPHF pre-warm, SymbolSubSpace
        # per-sentence invariants + CS-feedback pre-seed
        # (``self._prev_cs_for_ps/ws``), fresh _cs/_ss caches. Hoisted
        # into ``_per_word_prelude`` so the capture-gate test can
        # replay the same boundary-side contract. Now also: allocate /
        # zero the preallocated ``_per_word_concept_buf`` [B, N, D_c]
        # and set ``symbolSpace._target_cursor_length = N`` so compose
        # pads the S-space_role rule cursor to N with id_SS.
        stm, N_target, word_carrier, in_event = self._per_word_prelude(in_sub)

        # §6c sentence protocol (config-gated; default OFF until the
        # author's cutover): pump zero — the independent parallel
        # prelude seeding both towers and producing the §5 intent —
        # runs BEFORE the serial task; the per-word ticks then carry
        # the §6d serial partition (``serial_pump=True``, set by the
        # prelude's epilogue). Gist refresh below is on preemption
        # only (sign-off: no clause-boundary refresh, no sandwich).
        protocol_on = bool(getattr(self, 'sentence_protocol', False))
        if protocol_on:
            self._sentence_prelude(in_event, word_carrier)

        # The per-word loop IS the recurrence: it replaces the
        # whole-slab cell's ``subsymbolicOrder`` pass loop with a
        # word-indexed loop. Per the ratified design each word is ONE
        # PS->CS->WS step (Pre Stage 1.C this was a single
        # ``sigma_percept`` Σ-lift; post 1.C it is a single STM push of
        # the materialised PS/WS combine onto ``cs.stm``), and the
        # C->P / C->S feedback carries word-to-word
        # (the cross-step carrier), mirroring exactly how
        # ``_forward_body`` carries ``prevCS_*`` across its passes:
        # initialise the feedback ONCE before the loop, update it each
        # step. (Resetting the feedback per word would cold-start WS
        # every word -> WS always sees the empty seed -> the symbol
        # subspace zeroes out; the whole-slab path explicitly carries
        # this feedback across steps.)
        #
        # The canonical cell is the **TERMINAL-stage** PS/CS/WS
        # (``self.perceptualSpace`` / ``self.conceptualSpace`` /
        # ``self.wholeSpace`` == ``*Spaces[-1]``). The whole-slab
        # path chains the ``subsymbolicOrder`` per-stage spaces (each a
        # DISTINCT instance with a progressively N-halved shape, the
        # terminal one keeping full N); but the per-word loop replaces
        # that stage-chaining N-bottleneck with the per-word STM
        # accumulation, so it runs the single terminal cell -- the one
        # whose shape/codebooks the ENTIRE reused tail reads
        # (``_chart_compose_at_C`` -> ``self.conceptualSpace.stm``,
        # ``symbol_cache`` -> ``self.wholeSpace``, the head sized
        # from ``conceptualSpaces[-1].outputShape``, ``runBatch``'s
        # P-space_role IR loss -> ``self.perceptualSpace.subspace``). Using a
        # non-terminal stage would leave ``self.wholeSpace.subspace``
        # / ``self.conceptualSpace.stm`` unwritten (the bug an earlier
        # ``body_stages[0]`` cut hit). The word loop -- not a stage
        # loop -- provides the recurrence depth.
        # NOTE: cs/ws handles, prevCS_forPS/WS empty-seed init, MPHF
        # pre-warm, recur_pass reset, and STM resize+clear all folded
        # into ``_per_word_prelude`` (called above). The captured body
        # reads ``self._prev_cs_for_ps/ws`` uniformly (prelude seeds
        # them to the empty seeds; the body switches to
        # ``cs._subspaceForPS/WS`` once cs.forward has run for the
        # first time) -- no per-iteration None branch.

        # Static per-word loop (replaces the data-dependent
        # ``while next_word() is not None`` boundary).
        #
        # Recompile-churn fix (doc/plans/2026-07-04): the trip count is the
        # STATIC per-word SLAB WIDTH ``N_words`` (the padded ``_ar_embedded_N``
        # column count == PartSpace.nOutput, a per-config constant), NOT the
        # per-sentence ``min(N_static, K_host)`` the loop used before. That old
        # bound == ``K_host`` (since ``InputSpace.outputShape[0]`` is the huge
        # raw-byte width, e.g. 8192 >> the word count), so it varied with the
        # sentence word-count (1..N_words) -> Dynamo unrolled a fresh graph per
        # length and blew ``cache_size_limit``. ``out_slot`` is a FIXED-LENGTH
        # ``N_words`` list written by python-constant index ``p`` (was
        # ``append`` -> a length that grew with the trip count, a second
        # recompile driver -- the ``len(out_slot) == k`` guard the census hit).
        # Padding columns (``p >= K_host``) are byte-identical no-ops: the
        # ``word_active`` gate zeroes their contribution and masks the STM push
        # (``push_step_masked`` with an all-False gate leaves ``_buffer`` /
        # ``_depth`` untouched); the CS/idea forwards fired anyway. ``K_host``
        # now only reconstructs the host depth mirror post-loop -- it never
        # enters the graph.
        # See doc/plans/2026-05-20-static-per-word-loop-impl.md §2.
        _slab = self.inputSpace._ar_embedded_N
        N_words = (int(_slab.shape[1]) if _slab is not None
                   else (int(N_target) if N_target is not None
                         else int(self.inputSpace.outputShape[0])))
        word_active = self.inputSpace._word_active_mask
        _compiled_static_loop = bool(
            torch.compiler.is_compiling()
            and getattr(self, "_compiled_word_loop_fullgraph", False))
        if _compiled_static_loop:
            # The tensor word mask already makes padding columns exact
            # no-ops.  Do not read the eager host summary here: every distinct
            # sentence length otherwise becomes a Python guard and requests a
            # new W-loop compilation despite the static W recurrence.
            N_loop = N_words
        else:
            K_host = int(self.inputSpace._valid_len_host)
            N_loop = min(N_words, K_host)
        out_slot = [None] * N_words
        self._per_word_contributions = out_slot
        _word_major = getattr(self.inputSpace, "_ar_word_part_ids", None)
        self._per_word_percept_contributions = (
            [None] * N_words if torch.is_tensor(_word_major) else None)

        # Skip-padding (eager only): run the body only up to the batch's
        # LONGEST sentence (``N_loop == K_host`` is the last column with any
        # active row across the batch, so columns ``[N_loop, N_words)`` are
        # empty for EVERY row). Byte-identical -- a skipped column leaves
        # ``out_slot[p]`` None, which the post-loop fills with the SAME zeros
        # the gate-masked body would have produced, and its STM push / mirror
        # advance were no-ops. Under ``torch.compile`` the trip count stays the
        # static ``N_words`` (a data-dependent count recompiles per length --
        # the 2026-07-04 static-loop contract); eager (CPU / host-island MPS,
        # where MM_20M actually runs) has no such constraint, so the loop cost
        # tracks the real sentence length instead of the padded slab width.
        _n_trips = N_words
        if not torch.compiler.is_compiling():
            _n_trips = max(1, min(N_words, N_loop))

        last_cs = None
        # Mid-sentence reground REMOVED (Alec 2026-07-07). The §6 preattention
        # rung (per-word conflict-mass check -> re-pump the gist on the rising
        # edge) was the forward's ONLY fullgraph=True blocker (6 data-dependent
        # ``.item()`` reads), and it was structurally dormant in training: the
        # absolute truth store's writers (``record``/``record_batch``) live on
        # the reasoning path only, so the store is empty during runEpoch and
        # the conflict mass is identically 0 -- the rung never fired. The
        # sentence-START prelude (the §6c pump-zero gist/intent) is unchanged;
        # the intent is now sentence-scoped with no mid-sentence refresh. If a
        # truth-grounded curriculum later wires store writes into the forward,
        # reintroduce the reground MASKLESS (always-pump + ``torch.where`` on
        # the intent, latch as a tensor) so fullgraph survives; the old
        # branched form is in git history at this site.
        # Slot-kind provenance recording (open-fronts Task B): enabled only
        # on <PartSpace><wordStore> configs, eager-only. Carried STM content
        # (a previous sentence's root) is tagged 'other'; the per-word
        # commit pushes below tag 'word'.
        if not torch.compiler.is_compiling():
            # Drain the deferred WORDS summary-row writes (stashed at the
            # sentence-boundary registration; a Parameter write there
            # would invalidate the pending backward). Pre-graph = safe.
            _ws_t = getattr(self, 'wholeSpace', None)
            if getattr(_ws_t, '_pending_words_summary', None):
                self.conceptualSpace.apply_pending_words_summary(_ws_t)
        if (not torch.compiler.is_compiling() and stm is not None
                and getattr(self.perceptualSpace, "word_store_reverse",
                            False)):
            _B_stm = int(stm._buffer.shape[0]) if stm._buffer is not None \
                else 0
            ks = getattr(stm, "_slot_kinds", None)
            if _B_stm and (ks is None or len(ks) != _B_stm):
                stm.kinds_enable(
                    _B_stm,
                    depths=stm._depth.detach().to("cpu").tolist(),
                    kind="other")
            # SENTENCE-scoped reverse-reduce trace (open-fronts Task B,
            # second pass): online grammatical reductions fold most of
            # the sentence DURING reading, so a sweep-local trace sees
            # only composite survivors and the un-fold cannot reach the
            # per-word operands. Reset HERE (sentence start) and mark the
            # scope so the sweep's own reset stands down; the backward
            # walk then unwinds ALL of this sentence's folds.
            object.__setattr__(self, "_stm_reduce_op_trace", [])
            object.__setattr__(self, "_stm_trace_sentence_scope", True)
        _chunk_replayed = False
        if (self._compiled_word_chunk_active
                and self._compiled_word_chunk_step is not None
                and torch.is_grad_enabled()):
            # The eager outer forward would otherwise choose the active-prefix
            # trip count.  Replaying the static even bucket keeps K=2 shapes
            # stable; inactive tail columns remain exact masked no-ops.
            _n_trips = N_words
            last_cs = self._run_aligned_word_chunk_loop(
                out_slot, _n_trips)
            _chunk_replayed = last_cs is not None
        else:
            last_cs = None
        if last_cs is None:
            last_cs = self.inputSpace.run_word_loop(
                self._per_word_body_step, out_slot, _n_trips,
                active_host=True)
        if protocol_on:
            # Sentence end: un-stamp the per-pump mode (the §6d law
            # falls back to the legacy read between sentences).
            self._set_serial_pump(None)
        word_count = N_loop
        # STM host depth mirror correction (2026-07-04 recompile-churn fix).
        # The per-word body advances ``_max_depth_host`` once per iteration,
        # and the loop now runs the full static per-word slab width
        # ``N_words`` instead of the active prefix ``N_loop``. So a SHORT
        # sentence's ``N_words - N_loop`` padding columns each added ONE
        # spurious advance (their masked push is a state no-op, but the host
        # ``+= 1`` still ran). SUBTRACT exactly that over-count to restore the
        # value the varying-trip-count loop left. This is a pure host-int
        # correction (never enters the graph). It must NOT clobber the mark to
        # the per-word count: ``_max_depth_host`` is a high-water mark advanced
        # by EVERY STM push site (e.g. ConceptualSpace.forward's
        # ``_stm_shift_and_push`` also bumps it), so the true value can exceed
        # ``N_loop`` (test_bounded_stm_fold measures N_total=24 >> the 8 words).
        # Padding columns never trip the ``>= capacity`` back-pressure (a
        # below-cap sentence's mirror stays < cap there; the configs that DO
        # reach cap fill every column, so ``N_words == N_loop`` and the
        # correction is zero), so the subtraction exactly undoes the padding
        # advances. Pins: test_bounded_stm_fold::test_cap_equivalence (the
        # high-water mark survives) + test_per_word_ss_padding_noop (the
        # per-forward advance == active-prefix count, not N).
        if (stm is not None and not _chunk_replayed
                and not _compiled_static_loop):
            # Only the columns that ACTUALLY ran advanced the mirror, so the
            # over-count is ``_n_trips - N_loop`` (== ``N_words - N_loop`` on
            # the static/compiled path; 0 on the eager skip-padding path, which
            # never ran the padding columns).  The reusable chunk path never
            # increments this host mirror inside a padded word cell and repins
            # it from the live tensor depth once after replay, so applying the
            # legacy subtraction there would double-correct short sentences.
            _pad_overcount = max(0, _n_trips - N_loop)
            if _pad_overcount:
                stm._max_depth_host = max(
                    0, int(stm._max_depth_host) - _pad_overcount)

        # Restore the full whole-sentence event onto the InputSpace
        # subspace so post-body readers (``_forward_per_stage``'s
        # ``forwardInput``/``input_state``, the guarded ``lossRev``
        # reverse term) see the same ``[B,T,D]`` surface the whole-slab
        # path leaves behind. ``_ar_embedded`` itself was never touched
        # (the cursor source), so this only re-stamps the subspace event
        # the per-word loop borrowed.
        if in_event is not None:
            word_carrier.set_event(in_event)

        # Reset the recurrent-pass index so standalone
        # PartSpace.forward calls (and the next forward's pass 0)
        # see the AR-streaming serial warm path -- mirrors the
        # whole-slab path's tail.
        if self.symbolSpace is not None:
            self.symbolSpace.recur_pass = 0
        else:
            self.perceptualSpace._recurrent_pass_idx = 0

        # Stack the per-iteration contributions (gradient-preserving)
        # and pad / truncate to ``N_target``. Inactive batch rows /
        # padding columns contributed zero so the tail is naturally
        # zero-padded. ``copy_context`` from the last per-word CS
        # subspace keeps the pipeline symbolSpace/errors/stem-route
        # contract intact.
        # See doc/plans/2026-05-20-static-per-word-loop-impl.md §2.5.
        # ``per_word_contribs`` is now the FIXED-LENGTH ``N_words`` list
        # (index = word position). Non-``None`` slots are the ``[B, D_c]``
        # per-position contributions the body wrote (padding columns and any
        # empty-CS iteration are gate-masked zeros / left ``None``); a
        # ``None`` slot is an iteration whose CS/idea never materialised --
        # the same slots the old ``append`` path simply skipped. Fill those
        # with a zero column so the stack is rectangular, then stack ALL
        # ``N_words`` columns. Padding columns already carry zeros, so the
        # stacked ``[B, N_words, D_c]`` equals the old
        # ``stack(active-prefix) + zero-pad-to-N_target`` byte-for-byte when
        # ``N_words == N_target`` (the serial-config invariant).
        per_word_contribs = self._per_word_contributions
        _real = next((c for c in per_word_contribs if c is not None), None)
        if last_cs is not None and _real is not None:
            _zc = torch.zeros_like(_real)                     # [B, D_c]
            filled = [c if c is not None else _zc
                      for c in per_word_contribs]
            stacked = torch.stack(filled, dim=1)              # [B, N_words, D_c]
            Bc, n_w, Dc = stacked.shape
            if N_target is not None and n_w < N_target:
                pad = torch.zeros(
                    Bc, N_target - n_w, Dc,
                    device=stacked.device, dtype=stacked.dtype)
                stacked = torch.cat([stacked, pad], dim=1)
            elif N_target is not None and n_w > N_target:
                stacked = stacked[:, :N_target, :]
            cs_sub = self.conceptualSpace.subspace
            cs_sub.copy_context(last_cs)
            cs_sub.set_event(stacked)
            last_cs = cs_sub
        # Restore a word-aligned PartSpace slab for the existing P-space IR /
        # reverse consumers.  During the loop PartSpace correctly held only
        # the active word; without this stack, those readers would see the
        # final word repeated as the whole sentence representation.
        ps_word_contribs = self._per_word_percept_contributions
        if isinstance(ps_word_contribs, list):
            ps_real = next(
                (c for c in ps_word_contribs if c is not None), None)
            if ps_real is not None:
                ps_zero = torch.zeros_like(ps_real)
                ps_stacked = torch.stack([
                    c if c is not None else ps_zero
                    for c in ps_word_contribs
                ], dim=1)
                self.perceptualSpace.subspace.set_event(ps_stacked)
                object.__setattr__(
                    self.perceptualSpace, "_word_aligned_output", ps_stacked)
        self._per_word_percept_contributions = None
        # S2 operand PROVENANCE (serial plan Task 2, 2026-07-05): stash the
        # serial derivation's LEAVES -- the per-word PERCEPT events the
        # bottom-up parse started from (``_ar_embedded_N``, ``[B, N, D]``,
        # word order: position 0 = first word, padding positions zero) -- so
        # the Method-1 reverse (``_reverse_method1_leaves``) can replay them
        # to reconstruct the surface EXACTLY. Percept leaves (not the per-word
        # CS ideas) because a percept's vector position IS its identity: the
        # percept-store nearest-neighbour decode recovers each word by
        # construction, untrained, whereas the CS reverse of the lattice-fold
        # is only exact once trained (doc/Spaces.md#percept-guarantees). Captured
        # per forward and batch-scoped exactly like ``_stm_single_S`` below
        # (``reconstruct_data`` reads the last staged batch); a ``.clone()``
        # decouples it from the next batch's slab.
        _leaf_slab = getattr(self.inputSpace, "_ar_embedded_N", None)
        if _leaf_slab is not None and torch.is_tensor(_leaf_slab):
            self._stm_pre_reduce_slab = _leaf_slab.detach().clone()
        self._per_word_contributions = []

        # 2b-2-i: NULL-seal finalize -- the BOUNDED soft REDUCE sweep
        # composes the accumulated STM down to a SINGLE idea S (the
        # start-symbol root). The legacy ``_chart_compose_at_C`` CKY
        # chart stays dormant/dead (it was a 100% runtime no-op on
        # empty STM and its math was never run); per the two-loop spec
        # ("Phase 2 rebuilds the producer, it does not tune it") it is
        # REPLACED -- not resurrected -- by this bounded soft
        # shift-reduce / selector-in-S producer. The per-word SHIFT
        # loop above filled STM (bounded <= cap by back-pressure); the
        # NULL seal (``next_word`` -> None ended the loop) now fires
        # the final bounded reduce sweep to the root.
        #
        # SCOPE FENCE (2b-2-i): the IR loss is UNCHANGED this
        # increment. ``last_cs`` (the position-aligned ``[B,N,D_c]``
        # representation, exactly as 2b-1) is what the existing
        # ``_forward_head`` + ``runBatch`` P-space_role masked-LM IR tail
        # consumes -- the training signal is byte-identical to 2b-1.
        # The single S is PRODUCED and verified here (depth -> 1,
        # category tracks ``Grammar.start_symbol``) but NOT yet
        # consumed by the loss; 2b-2-ii rewires the loss to
        # ``reverse(S)``-vs-unmasked using exactly this S.
        if (stm is not None and word_count > 0
                and getattr(self.inputSpace, "_per_word_enabled", False)):
            # (The S2 Method-1 leaf slab -- the per-word percept events --
            # was stashed above, BEFORE this reduce collapses the STM, so the
            # exact teacher replay never sees the folded state.)
            S, post_depth = self._stm_reduce_to_single_S()
            # Verification handles for the end-to-end probe / future
            # 2b-2-ii consumer: the single sentence idea S [B, D_c]
            # and the post-sweep STM depth (must be 1 across rows).
            self._stm_single_S = S
            self._stm_post_depth = post_depth
            # ``cs_buf`` / ``rel_mask`` feed the boundary discourse hook
            # (``observe_stm_end_state``) below. Relative-relation LEARNING
            # (``learn_relations_from_stm``) is HOISTED out of this captured
            # forward into ``ConceptualSpace.Reset`` (host-side sentence
            # boundary): it does ``mask.tolist()`` + taxonomy/codebook
            # mutation, untraceable under ``fullgraph``. See that Reset.
            cs_buf = self.conceptualSpace.stm._buffer
            rel_mask = self._sentence_relative_mask(
                int(cs_buf.shape[0]), device=cs_buf.device)

            # Task 7 (§8): LTM — record EVERY sentence's STM end-state on
            # the InterSentenceLayer's per-row chain (the AR sequence for
            # inter-sentence prediction). This is ADDITIVE to the ARMA /
            # reduce / IR-loss path: it reads the already-computed STM
            # end-state buffer (``cs_buf``) and the same host-side
            # ``rel_mask`` the reduce used, then hands them to the
            # boundary-only ``observe_stm_end_state`` (which is
            # ``@torch.compiler.disable``'d — host-side, outside the
            # captured per-word graph). It NO-OPS gracefully when there
            # is no discourse layer (``sentencePrediction`` off /
            # absolute-only configs), so MM_xor stays byte-identical.
            # Note: LTM is NOT gated by ``truthCriterion`` — every
            # end-state lands here; ``truthCriterion`` only gates the
            # separate WS-codebook insertion above.
            discourse = (self.symbolSpace.discourse
                         if getattr(self, 'symbolSpace', None) is not None
                         else None)
            # LTM consolidation FU (Change 1, 2026-06-18): the persistent
            # ``ltm_store`` (conversation push) is now an INDEPENDENT sink
            # from the discourse AR deque -- a config without a predictor
            # (no ``sentencePrediction``) must still push conversation into
            # the unified store. So compute ``depths`` / ``payloads`` /
            # ``tetralemmas`` ONCE here (they only need ``cs_buf`` /
            # ``rel_mask`` / ``B`` / ``cap``, all available regardless of a
            # discourse) and drive TWO independent sinks below:
            #   (a) the persistent store-append (the SINGLE conversation
            #       push), gated on ``ltmConsolidation`` + an ``ltm_store``;
            #   (b) the discourse AR predict+observe, when a discourse
            #       exists -- unchanged.
            # The whole block stays byte-identical with both gates off / no
            # discourse: the compute is skipped entirely unless at least one
            # sink is live.
            ltm_store = getattr(self.symbolSpace, 'ltm_store', None)
            ltm_consolidation_on = (
                getattr(self.conceptualSpace, '_ltm_consolidation', False)
                and ltm_store is not None)
            discourse_live = (discourse is not None and hasattr(
                discourse, 'observe_stm_end_state'))
            if ((discourse_live or ltm_consolidation_on)
                    and torch.compiler.is_compiling()):
                # A compiled graph may only park fixed tensors. The eager step
                # teardown reconstructs ragged rows and performs both sinks.
                object.__setattr__(
                    self, "_pending_stm_end_state",
                    (cs_buf.detach(), rel_mask.detach()))
                discourse_live = False
                ltm_consolidation_on = False
            if discourse_live or ltm_consolidation_on:
                B = int(cs_buf.shape[0])
                cap = int(cs_buf.shape[1])
                # Per-row end-state depth: 3 for a relative row (the
                # depth-3 ``[predicate, idea1, idea2]`` preserve), else 1
                # (the collapsed absolute root). Single host hop on the
                # boundary mask — outside any captured region. Clamp to
                # the buffer capacity so the recorded depth always equals
                # the stored payload's row count (the reduce site only
                # preserves depth 3 when cap >= 3, so this is defensive).
                rel_rows = rel_mask.reshape(-1).tolist()
                depths = [min(3 if bool(rel_rows[b]) else 1, cap)
                          for b in range(B)]
                # Ragged per-row payloads: the ``depth`` leading slots of
                # this row's STM buffer. Newest-at-slot-0 convention: slot
                # 0 is the newest (absolute: the collapsed root; relative:
                # the folded-rest idea2), and the OLDEST slot (``depth-1``)
                # is the root/predicate. ``_reduce_end_state_to_root``
                # reads that last slot to recover the root.
                payloads = [cs_buf[b, :depths[b], :] for b in range(B)]
                # Attach per-row scalar trust to the LTM row ("LTM is
                # persisted STM"). Absolute rows carry the model's trust that
                # the description refers to an actual event; relative rows
                # carry that trust applied to the relation's t - f scalar.
                tetralemmas = self.conceptualSpace.stm_end_state_trust(
                    cs_buf, rel_mask)
                # Skip on the explore trial (pass B): both sinks append the
                # sentence's end-state. Pass A already committed this
                # sentence; a second append would duplicate it (in the AR
                # deque AND the persistent store).
                if not getattr(self, '_exploration_trial', False):
                    # SINK (b) -- the discourse AR predict+observe (Task 7/8)
                    # runs FIRST so its prediction is staged from the chain
                    # state BEFORE this boundary's end-state lands in the
                    # store. Use the combined predict+observe so the inter-
                    # predictor STAGES a next-end-state prediction (from the
                    # chain BEFORE this boundary's end-state is appended) and
                    # then SCORES it against the arriving end-state —
                    # accumulating ``L_inter`` during training. ``predict_and_
                    # observe_stm_end_state`` degenerates to a bare observe
                    # when there is no inter-predictor (absolute-only no-op)
                    # or on the cold first sentence (nothing to predict from
                    # yet). When consolidated (Change 2 / FU3) ``observe``
                    # reads the unified store via ``get_stm_chain`` for AR
                    # context and does NOT append to the deque -- the store-
                    # append (a) below is the single source, so it MUST run
                    # after this predict staging (else the predictor would
                    # see this end-state as part of its own history).
                    if discourse_live:
                        discourse.predict_and_observe_stm_end_state(
                            depths, payloads, tetralemmas=tetralemmas)
                    # SINK (a) -- the SINGLE conversation push into the
                    # persistent unified ``ltm_store`` (LTM consolidation,
                    # gated ``ltmConsolidation``). Independent of any
                    # discourse: a predictor-less config still records
                    # conversation here. Slot mapping is newest-at-slot-0
                    # (matching ``learn_relations_from_stm``): for a relative
                    # (depth>=3) row the predicate is the OLDEST slot
                    # (depth-1), idea1 the 2nd-oldest (depth-2) and idea2 the
                    # newest folded rest (slot 0); stored as NP1=idea1,
                    # VP=predicate, NP2=idea2. Host-side (this whole block is
                    # @torch.compiler.disable'd) and fully guarded so the
                    # off-path stays byte-identical.
                    if ltm_consolidation_on:
                        for b in range(B):
                            payload = payloads[b]
                            if payload is None or payload.shape[0] < 1:
                                continue
                            d = int(depths[b]) if b < len(depths) else int(
                                payload.shape[0])
                            d = max(1, min(d, int(payload.shape[0])))
                            tet = (tetralemmas[b]
                                   if (tetralemmas is not None
                                       and b < len(tetralemmas))
                                   else None)
                            trust = float(tet) if tet is not None else 0.0
                            if d >= 3:
                                ltm_store.append_relation(
                                    payload[d - 2], payload[d - 1], payload[0],
                                    rel_type=TernaryTruthStore.REL_OTHER,
                                    trust=trust)
                            else:
                                ltm_store.append_idea(payload[0], trust=trust)

        # Existing loss_head plumbing (dormant: ``loss_head`` is always
        # None today) -- kept identical to the whole-slab tail so the
        # two paths stay structurally symmetric.
        if self.loss_head is not None:
            grad_input = self._loss_head_input
            if grad_input is None:
                grad_input = self.conceptualSpace.stm.snapshot()
            if grad_input is not None:
                self._loss_head_loss = self.loss_head(grad_input)
            else:
                self._loss_head_loss = None
        return last_cs

    def _chart_compose_per_word(self):
        """Per-word router fire (Alec 2026-07-13: "the router should be
        firing as every word is added, since the STM is not long enough to
        preserve the sentence before parsing is initiated").

        The same signal-router fire as the boundary chart, run over the
        CURRENT STM contents right after each word push, so
        ``current_rules`` carries THIS sentence's rules while it is still
        being read — the reduce sweep's relative mask (``protect_depth``
        3) can then key on them. This implements the documented
        ``<routerWireSerial>`` semantics ('per-word' and the 'both'
        default) which were previously VALIDATION-ONLY — no fire site
        existed. Host-eager (the full-router island is
        ``@torch.compiler.disable``'d); skipped under compile like the
        sibling provenance bookkeeping.
        """
        if getattr(self, 'router_wire_serial', 'both') not in (
                'per-word', 'both'):
            return
        snap = self.conceptualSpace.stm.snapshot()
        # Parsing initiates once TWO constituents exist — a 1-item stack
        # has nothing to reduce, and the binary layer's degenerate N<=1
        # path returns a routing dict without the rule-id keys.
        if snap is None or snap.dim() != 3 or int(snap.shape[1]) < 2:
            return
        # DETACHED CLONE: snapshot() is a VIEW of the live STM buffer, and
        # the router's compose mutates its slab — an in-place write there
        # bumps the buffer version the pending backward saved (measured:
        # "[1016] at version 1" backward crash). The per-word fire exists
        # for the RULE BOOKKEEPING (current_rules); the boundary fire
        # keeps the gradient role.
        self.symbolSpace.forward(snap.detach().clone())

    def _chart_compose_at_C(self, stage_idx=0):
        """Fire the signal router at C-space_role over
        ``conceptualSpace.stm`` contents.

        Populates ``symbolSpace.current_rules`` for downstream WS
        dispatch. Uses :meth:`ShortTermMemory.snapshot` to obtain a
        single uniform ``[B, max_depth, D_c]`` slab (rows with shorter
        sentences carry zero-padding at the tail).

        FULL-ROUTER fires run as an eager island
        (``_ss_compose_eager``, ``@torch.compiler.disable``'d — the
        house pattern, cf. ``observe_stm_end_state``): the router's
        rule-id bookkeeping (``languageLayer.compose``'s per-row
        ``int(kind[b, j].item())`` branching) is data-dependent host
        control flow dynamo cannot guard on (``Could not guard on
        data-dependent expression Eq(u0, 2)`` — MM_xor on the compiled
        per-stage path). Its products are host dicts
        (``current_rules`` / cursors) read BEFORE captured regions, so
        the island changes no numerics; full-router configs compile
        with fullgraph=False (see ``enable_compiled_step``). The
        DEFAULT-ONLY bypass never reaches the router loops, stays
        traceable inline, and keeps the strict fullgraph gate
        (a disabled call inside a fullgraph=True trace would itself
        hard-error — MM_20M).

        Method name preserved across the Stage 3 chart retirement; it
        now drives ``SymbolSubSpace.compose`` -> ``languageLayer.compose``.

        Gated by ``<routerWireSerial>`` (Task 4 / plan §4): the boundary
        fire runs iff ``self.router_wire_serial in ('boundary', 'both')``.
        Default ``both`` keeps this firing — pre-existing behaviour is
        preserved. This is the final routing snapshot the inter-sentence
        predictor consumes (retained per plan §4).
        """
        if getattr(self, 'router_wire_serial', 'both') not in (
                'boundary', 'both'):
            return
        snap = self.conceptualSpace.stm.snapshot()
        if snap is None:
            return
        # CS couples to SymbolSpace only via forward/reverse (the dispatch +
        # eager-island handling now live on SymbolSpace; 2026-06-21 refactor).
        self.symbolSpace.forward(snap)

    def _reverse_seed_snapshot(self, seed):
        """Return a ``[B, N, D]`` idea snapshot from reverse's seed, if any."""
        if seed is None:
            return None
        try:
            snap = seed.materialize() if hasattr(seed, "materialize") else seed
        except Exception:
            return None
        if not torch.is_tensor(snap):
            return None
        if snap.dim() == 2:
            return snap.unsqueeze(1)
        if snap.dim() == 3:
            return snap
        return None

    def _chart_generate_from_stm(self, seed=None):
        """Fire ``symbolSpace.generate`` over the C-space_role STM snapshot.

        Reverse-path mirror of ``_chart_compose_at_C`` (full-router
        fires run via the ``_ss_generate_eager`` island for the same
        reason — see there): populates ``symbolSpace.generate_rules``
        so each stage's reverse dispatch can pop them via its
        SyntacticLayer cursor.

        Method name preserved across the Stage 3 chart retirement; it
        now drives ``SymbolSubSpace.generate`` -> ``languageLayer.generate``.

        Gated by ``<routerWireSerial>`` (Task 4 / plan §4): the boundary
        reverse fire runs iff
        ``self.router_wire_serial in ('boundary', 'both')``. Default
        ``both`` keeps this firing — pre-existing behaviour is preserved.

        REVERSE-LEG NOTE (Task 4): there is no per-word reverse loop
        analogous to ``_forward_body_per_word`` — the reverse path
        (``reverse`` / ``_reverse_from_S``) fires ``generate`` only here, at the
        boundary, over the whole STM snapshot. So per-word reverse has
        no natural site; the boundary reverse fire is the reverse-leg
        deliverable and is retained (gated identically to the forward
        boundary fire).

        IDEA-DECODE (``<ideaDecode>``, Goal 2 step 1): when on by itself, the
        parse tree is DELETED -- so there is no ``generate_rules`` to rebuild
        here. Skip the chart fire entirely; the reverse path then runs with an
        empty rule set, driven by the primed symbolic space rather than the
        chart. ``<reconstructFromIdea>`` is the opposite experiment: erase the
        old derivation, then rebuild ``generate_rules`` from the supplied idea
        seed (falling back to STM when no seed is usable).
        """
        from_idea = bool(getattr(self, 'reconstruct_from_idea', False))
        if getattr(self, 'idea_decode', False) and not from_idea:
            return
        if getattr(self, 'router_wire_serial', 'both') not in (
                'boundary', 'both'):
            return
        ss = self.symbolSpace
        if ss is None:
            return
        if from_idea and hasattr(ss, "clear_grammar_cache"):
            ss.clear_grammar_cache()
        stm = self.conceptualSpace.stm
        if stm is None:
            return
        snap = self._reverse_seed_snapshot(seed) if from_idea else None
        if snap is None:
            snap = stm.snapshot()
        if snap is None:
            return
        ss.reverse(snap)

    def _stm_symbolic_roundtrip(self, slab):
        """Idempotent C→S→C round-trip over a full ``[B, cap, D_c]`` STM
        slab (the widened symbolic-space loop -- H2).

        This is the retired ``_forward_stem_per_word`` C→S→C round-trip
        (``a8737da~1``) -- ``cb.forward(c)`` then ``cb.reverse(snap, V=1)``
        then ``idea_back.sum(dim=1)`` per single word -- generalized from
        the single-vector ``V=1`` regime to ``V = STM capacity`` so the
        whole ``[B, cap, concept_dim]`` STM slab passes through symbolic
        space and back in one shot, idempotently and **per slot** (all
        ``cap`` slots preserved -- the legacy single-slot
        ``idea_back.sum(dim=1)`` collapse is dropped, NOT replaced by a
        V-axis mean collapse).

        Mechanism (faithful generalization of the retired vectorised
        path's ``c_flat = c_all.reshape(B_flat * N, 1, D_c)`` trick):
        fold the ``cap`` axis into the batch dim so each STM slot is its
        own row through ``ProjectionBasis.forward`` / ``.reverse(V=1)``,
        then unfold. ``ProjectionBasis.forward`` collapses the V axis
        (mean-over-V); calling it with ``cb.forward(slab)`` directly +
        ``cb.reverse(snap, V=cap)`` would therefore replicate ONE summary
        across all ``cap`` slots (``x_summary.unsqueeze(1).expand(-1, v,
        -1)``) -- destroying STM as a per-slot working memory. The
        per-slot fold preserves each slot's own reverse-lift, exactly as
        the retired loop did, while remaining idempotent (forward then
        reverse projects each slot onto span(W); a re-snapped slot is a
        fixed point -- the property ``test_idempotent_loop`` pins for
        ``V=1``, here per-slot over ``cap``).

        Returns the round-tripped ``[B, cap, D_c]`` slab, or the input
        unchanged when no ``ProjectionBasis`` codebook is available
        (matches the retired loop's ``ideas = c_all`` fall-through for
        non-ProjectionBasis ``.what``: ``quantize`` / ``none`` configs).

        NOTE (H2 Part B / Phase-2B): this is the rebuilt subsymbolic
        producer's round-trip primitive. It is intentionally NOT yet
        wired into the live forward as the STM read/write -- the STM
        producer lifecycle (when to push, when ``snapshot`` is read,
        the bounded soft shift-reduce controller, the NULL-seal
        finalize, per-word vs per-sentence) is the still-pending,
        owner-perf-gated Phase-2B architecture
        (``doc/plans/2026-05-18-two-loop-pipeline-architecture.md``
        §"STM-7" / Phase-2B; CKY+resize is the documented fallback).
        """
        if slab is None or not torch.is_tensor(slab) or slab.dim() != 3:
            return slab
        cb = getattr(self.wholeSpace.subspace, 'what', None)
        cb_is_projection = (cb is not None
                            and type(cb).__name__ == 'ProjectionBasis')
        if not cb_is_projection:
            # Non-ProjectionBasis ``.what`` (``quantize`` / ``none``):
            # no idempotent projection surface -- pass the slab through
            # unchanged, exactly as the retired ``_forward_stem_per_word``
            # did (``ideas = c_all``).
            return slab
        B, cap, D_c = slab.shape
        # Fold ``cap`` into the batch dim: one STM slot per row so
        # ``ProjectionBasis.forward``'s mean-over-V is a no-op (V=1 per
        # row) and ``reverse(V=1)`` lifts each slot independently. This
        # is the retired vectorised path's ``reshape(B_flat * N, 1,
        # D_c)`` -> ``cb.forward`` -> ``cb.reverse(snap_flat, V=1)``
        # generalized to the whole slab.
        c_flat = slab.reshape(B * cap, 1, D_c)            # [B*cap, 1, D_c]
        snap_flat = cb.forward(c_flat)                    # [B*cap, N, 1]
        back_flat = cb.reverse(snap_flat, V=1)            # [B*cap, 1, D_c]
        if back_flat is None:
            return slab
        if back_flat.dim() == 3:
            # [B*cap, 1, D_c] -> [B, cap, D_c]. The legacy single-slot
            # ``idea_back.sum(dim=1)`` is dropped (V=1 here, so the sum
            # was a no-op squeeze): all ``cap`` slots are kept.
            return back_flat.reshape(B, cap, D_c)
        return back_flat.reshape(B, cap, D_c)

    # ------------------------------------------------------------------
    # A4 (2026-06-06 parallel-conceptual-recurrence) demux/fit helpers.
    # Option B (doc/specs/2026-06-06-muxed-events-and-positional-bands.md):
    # the per-stage ConceptualCombine operates on the FULL muxed event
    # (content + .where/.when together; sized at ``cs.muxedSize``), so a demux
    # AT the event width returns the whole event with an EMPTY band -- the
    # band PARTICIPATES in the combine rather than riding along. These helpers
    # also serve the narrower-stream fit (the compressed WS symbol code
    # zero-padded up to D).
    # ------------------------------------------------------------------
    @staticmethod
    def _combine_demux(sub, content_dim):
        """Delegates to :meth:`ConceptualSpace._bind_demux` -- the
        calculation lives in the Space (processing contract,
        2026-06-10); this alias serves remaining Models-side callers."""
        return ConceptualSpace._bind_demux(sub, content_dim)

    @staticmethod
    def _combine_fit(content, D, like):
        """Delegates to :meth:`ConceptualSpace._bind_fit` -- the
        calculation lives in the Space (processing contract,
        2026-06-10); this alias serves remaining Models-side callers."""
        return ConceptualSpace._bind_fit(content, D, like)

    # ``_symbolic_sigma_step`` (the content-demux WS.sigma advance the
    # ConceptualCombine replaced) was removed 2026-06-06 (Action C): it was
    # dead in ``bin/`` -- the parallel body advances the carrier through the
    # combine, and ``ws.sigma`` round-trips are exercised directly. The
    # generalization step now lives IN the combine (option B, on the full
    # muxed event), not in a separate content-only operator.

    @torch.compiler.disable
    def _reading_attention_step(self, t, prevCS_forSS, ps_stage0, cs_sub):
        """Run the reading-attention producer for a ``t>0`` subsymbolic pass
        (doc/specs/reading-attention.md "(A) Reading attention").

        Writes the next reading scope to ``wholeSpaces[0]._passback_scope_where``
        (the producer of the ``.where`` the ``<mereologyRaise>`` handoff
        consumes) and -- in text mode (``dataType`` embedding) training -- adds
        the next-word cross-entropy term to ``cs_sub.errors``. Teacher forcing:
        the WRITTEN scope is the TRUE next span during training (so PartSpace
        reads the right word while the producer learns to predict it) and the
        producer's own soft prediction at inference (free-run). The handoff
        collapses the batch to one span, so the write is the mean over the rows
        whose next-word target is a real (non-pad) span.

        Gated: caller only calls this when ``self.reading_attention`` is present
        (``<readingAttention>``) and ``t>0`` (pass 0 is wide-open). Byte mode
        stages no spans -> no-op. ``@torch.compiler.disable``'d (it materializes
        + reads shapes host-side, like ``_ss_compose_eager``): under a compiled
        config with reading attention on it graph-breaks cleanly at the call
        rather than failing inside ``materialize()``; default-off configs never
        reach it, so the compiled default path is byte-identical."""
        ra = self.reading_attention
        wss = getattr(self, "wholeSpaces", None)
        ws0 = wss[0] if wss else None
        if ra is None or ws0 is None or int(t) <= 0:
            return
        spans = getattr(ws0, "_staged_analysis_spans", None)
        if spans is None or not torch.is_tensor(spans):
            return
        percept_ev = (ps_stage0.materialize()
                      if ps_stage0 is not None else None)
        if (percept_ev is None or not torch.is_tensor(percept_ev)
                or percept_ev.dim() != 3 or int(percept_ev.shape[1]) == 0):
            return
        N = int(percept_ev.shape[1])
        # query: the prior pass's concept (subsymbolic / mereological
        # retrieval term) + the active STM symbols (the symbolic term).
        # Both DETACHED inside ReadingAttention -- the grad stops here.
        concept_q = ReadingAttention._pool(prevCS_forSS)
        stm = (self.conceptualSpace.stm
               if self.conceptualSpace is not None else None)
        stm_snap = stm.snapshot(detach=True) if stm is not None else None
        symbol_q = ReadingAttention._pool(stm_snap)
        # Subsymbolic retrieval index: the PERCEPT codebook (PartSpace) -- the
        # span content snaps to its prototypes, weighted by the intent boosts
        # (the literal intent_boosts / selection_boost_fn path). The boosts are
        # the tower's primed-intent state when set (e.g. by the sentence
        # protocol), else derived from the concept inside the module. None when
        # the space carries no codebook -> the concept-content cosine fallback.
        cb_rows, cb_boosts = None, None
        ps_space = getattr(self, "perceptualSpace", None)
        ps_sub = getattr(ps_space, "subspace", None) if ps_space else None
        cb = (ps_sub.codebook()
              if ps_sub is not None and hasattr(ps_sub, "codebook") else None)
        getW = getattr(cb, "getW", None) if cb is not None else None
        if callable(getW):
            cb_rows = getW()
            if cb_rows is not None:
                cb_boosts = ps_space.intent_boosts()
        read_idx = int(t) - 1     # pass t reads word (t-1), left to right
        # Stochastic element: the two-pass superposition temperature (None/0 on
        # an ordinary pass -> byte-identical; exploreTemperature on pass B).
        temp = getattr(self, "_superposition_temperature", None) or 0.0
        next_where, ce_loss, _ = ra(
            concept_q=concept_q, symbol_q=symbol_q, percept_ev=percept_ev,
            spans=spans, read_idx=read_idx, N=N, training=bool(self.training),
            codebook_rows=cb_rows, intent_boosts=cb_boosts, temperature=temp)
        # R2: the reading-attention scope is CS-owned (CS is the home of the
        # .where-producer over the towers); the model passes it to the WS->PS
        # passback. Falls back to ws0 when CS is absent (standalone tests).
        _scope_owner = (self.conceptualSpace
                        if self.conceptualSpace is not None else ws0)
        if next_where is None:
            object.__setattr__(_scope_owner, "_passback_scope_where", None)
            return
        # Scope write (consumed by the <mereologyRaise> handoff). Fresh each
        # pass; None when the read cursor has run past the last word (no next
        # word to scope) so the handoff falls back to its route_hint/noop.
        K = int(spans.shape[1])
        Nf = float(max(N, 1))
        scope = None
        if 0 <= read_idx < K:
            spans_f = spans.to(percept_ev.dtype)
            real = (spans_f[:, read_idx, 1]
                    - spans_f[:, read_idx, 0]).clamp_min(0.0) > 0    # [B]
            if bool(real.any()):
                scope_bk = (spans_f[:, read_idx, :] / Nf
                            if self.training else next_where)        # [B, 2]
                w = real.to(scope_bk.dtype).unsqueeze(-1)            # [B, 1]
                scope = ((scope_bk * w).sum(dim=0)
                         / w.sum().clamp_min(1.0)).reshape(-1).detach()
        object.__setattr__(_scope_owner, "_passback_scope_where", scope)
        # Text-mode (embedding) next-word CE. Trains the producer readout
        # only (the codebooks stay EMA-only -- the loss graph touches no VQ).
        if (ce_loss is not None and self.training
                and getattr(self, "model_type", None) == "embedding"
                and cs_sub is not None and hasattr(cs_sub, "errors")):
            cs_sub.errors.add(
                "reading_attention", ce_loss, weight=1.0,
                space="ConceptualSpace", category="symbol")

    @torch.compiler.disable
    def _addressable_spaces(self, prevCS_forSS, ps_stage0):
        """Gather the TYPED addressable spaces global attention ranges over
        (doc/specs/reading-attention.md "(B)"; Alec: input window / STM / LTM /
        symbolic codebook). Each entry is a ``GlobalAttention`` space dict with
        DETACHED keys -- per-batch ``[B, M, D]`` (input window, STM) or shared
        ``[M, D]`` (LTM, codebook). Absent spaces are simply omitted."""
        GA = GlobalAttention
        spaces = []
        # INPUT window: the staged word brackets pooled to per-span percept
        # content (the same keys reading uses); else the per-slot percept.
        ws0 = (self.wholeSpaces[0]
               if getattr(self, "wholeSpaces", None) else None)
        percept_ev = (ps_stage0.materialize()
                      if ps_stage0 is not None else None)
        if (percept_ev is not None and torch.is_tensor(percept_ev)
                and percept_ev.dim() == 3 and int(percept_ev.shape[1]) > 0):
            N = int(percept_ev.shape[1])
            spans = getattr(ws0, "_staged_analysis_spans", None) if ws0 else None
            if spans is not None and torch.is_tensor(spans):
                keys = ReadingAttention._span_keys(percept_ev, spans)  # [B,K,D]
                spans_f = spans.to(percept_ev.dtype)
                where = (spans_f / float(max(N, 1)))[..., :2]          # [B,K,2]
                valid = (spans_f[..., 1] - spans_f[..., 0]).clamp_min(0.0) > 0
                spaces.append({"id": GA.SPACE_INPUT, "keys": keys.detach(),
                               "where": where.detach(), "valid": valid})
            else:
                spaces.append({"id": GA.SPACE_INPUT,
                               "keys": percept_ev.detach()})
        # STM: the live short-term-memory rows.
        stm = (self.conceptualSpace.stm
               if self.conceptualSpace is not None else None)
        stm_snap = stm.snapshot(detach=True) if stm is not None else None
        if (stm_snap is not None and torch.is_tensor(stm_snap)
                and stm_snap.dim() == 3 and int(stm_snap.shape[1]) > 0):
            spaces.append({"id": GA.SPACE_STM, "keys": stm_snap})
        # LTM: the consolidated TernaryTruthStore rows (present only under
        # <ltmConsolidation>); pool the three idea slots to one content key.
        ss = getattr(self, "symbolSpace", None)
        ltm = getattr(ss, "ltm_store", None) if ss is not None else None
        if ltm is not None and hasattr(ltm, "slots"):
            cnt = int(getattr(ltm, "count", torch.tensor(0)).item()) \
                if torch.is_tensor(getattr(ltm, "count", None)) else 0
            if cnt > 0:
                rows = ltm.slots[:cnt].detach().mean(dim=1)            # [M, D]
                spaces.append({"id": GA.SPACE_LTM, "keys": rows})
        # The three tower codebooks, emitted as distinct address spaces (PART,
        # WHOLE, SYMBOL). Was previously one lumped SPACE_CODEBOOK (ws0-or-PS).
        concept_q = ReadingAttention._pool(prevCS_forSS)

        def _codebook_rows(_sp):
            sub = getattr(_sp, "subspace", None) if _sp is not None else None
            cb = (sub.codebook()
                  if sub is not None and hasattr(sub, "codebook") else None)
            getW = getattr(cb, "getW", None) if cb is not None else None
            rows = getW() if callable(getW) else None
            if (rows is not None and torch.is_tensor(rows)
                    and rows.dim() == 2 and int(rows.shape[0]) > 0):
                return rows
            return None

        # PART codebook (PartSpace part-percepts).
        ps_rows = _codebook_rows(getattr(self, "perceptualSpace", None))
        if ps_rows is not None:
            spaces.append({"id": GA.SPACE_PART, "keys": ps_rows.detach()})
        # WHOLE codebook (WholeSpace) -- carries the intent-priming boosts.
        ws_rows = _codebook_rows(ws0)
        if ws_rows is not None:
            ws_boosts = ws0.intent_boosts() if ws0 is not None else None
            if ws_boosts is None and concept_q is not None:
                from Spaces import intent_priming_weights
                ws_boosts = intent_priming_weights(concept_q.detach(), ws_rows)
            spaces.append({"id": GA.SPACE_WHOLE, "keys": ws_rows.detach(),
                           "boosts": ws_boosts})
        # SYMBOL codebook (SS.subspace.what). Present only under <symbolTower>
        # (else the SS ``.what`` is an empty Basis with no ``getW``).
        ss = getattr(self, "symbolSpace", None)
        ss_sub = getattr(ss, "subspace", None) if ss is not None else None
        ss_cb = getattr(ss_sub, "what", None) if ss_sub is not None else None
        ss_getW = getattr(ss_cb, "getW", None) if ss_cb is not None else None
        if callable(ss_getW):
            sym_rows = ss_getW()
            if (sym_rows is not None and torch.is_tensor(sym_rows)
                    and sym_rows.dim() == 2 and int(sym_rows.shape[0]) > 0):
                # CLONE (not just detach): _build_symbol_leg writes this
                # codebook Parameter in-place (no_grad) LATER in the same
                # forward, which would bump the version of a detached view the
                # attention scorer saved for backward. A clone decouples the
                # attention key from the live Parameter. (The other spaces'
                # codebooks are written only at the Reset boundary, so a plain
                # detach is safe for them.)
                spaces.append({"id": GA.SPACE_SYMBOL,
                               "keys": sym_rows.detach().clone()})
        return spaces, concept_q

    @torch.compiler.disable
    def _global_attention_step(self, prevCS_forSS, ps_stage0):
        """Run free global attention over the typed addressable space
        (doc/specs/reading-attention.md "(B) Global attention") and PARK the
        result on ``self._global_attention_obs`` (a typed ``.where`` + the
        soft-read content). Dark: the read is parked, NOT fed back into the
        output -- the consumer is a later slice -- so the forward is
        byte-identical with the flag on. Gated <globalAttention> (caller checks
        the module is present). ``@torch.compiler.disable``'d (host-side gather,
        like reading)."""
        ga = self.global_attention
        if ga is None:
            return
        spaces, concept_q = self._addressable_spaces(prevCS_forSS, ps_stage0)
        if not spaces:
            object.__setattr__(self, "_global_attention_obs", None)
            return
        stm = (self.conceptualSpace.stm
               if self.conceptualSpace is not None else None)
        symbol_q = ReadingAttention._pool(
            stm.snapshot(detach=True) if stm is not None else None)
        temp = getattr(self, "_superposition_temperature", None) or 0.0
        obs = ga(concept_q=concept_q, symbol_q=symbol_q, spaces=spaces,
                 temperature=temp)
        object.__setattr__(self, "_global_attention_obs", obs)

    @torch.compiler.disable
    def _reasoning_spaces(self):
        """Gather the typed truth-space the reasoner ranges over -- STM, LTM, and
        the PART/WHOLE/SYMBOL codebooks -- via the SAME registry the forward's
        global attention uses (``_addressable_spaces``), minus the live input
        window. Built from PERSISTENT state, so it is callable outside a forward
        pass (at query/serve time)."""
        try:
            spaces, _ = self._addressable_spaces(None, None)
        except Exception:
            spaces = []
        return spaces or []

    def _reasoning_tooluser(self, spaces):
        """Lazily build + cache the reasoning policy's soft components: the
        InterveningIdeaGenerator (an MLP query head) + a GlobalAttention, sized
        to the truth-space content width. Registered as submodules so they ride
        the state_dict and train under the answer loss (Phase C). Returns
        ``(generator, ga)`` -- ``(None, None)`` when the truth-space is empty."""
        gen = getattr(self, "_intervening_generator", None)
        if gen is not None:
            return gen, getattr(self, "_reasoning_ga", None)
        Dc = None
        for s in (spaces or []):
            k = s.get("keys")
            if k is not None and torch.is_tensor(k) and k.dim() in (2, 3):
                w = int(k.shape[-1])
                Dc = w if Dc is None else min(Dc, w)
        if not Dc:
            return None, None
        # Place the soft components on the model device (the spaces' keys carry
        # it) so answer-loss training does not hit a CPU/MPS/CUDA mismatch.
        dev = None
        for s in (spaces or []):
            k = s.get("keys")
            if k is not None and torch.is_tensor(k):
                dev = k.device
                break
        from reasoning import InterveningIdeaGenerator, NextIdeaScorer
        gen = InterveningIdeaGenerator(dim=int(Dc))
        self._intervening_generator = gen if dev is None else gen.to(dev)
        if self.global_attention is not None:
            self._reasoning_ga = self.global_attention
        else:
            ga = GlobalAttention()
            self._reasoning_ga = ga if dev is None else ga.to(dev)
        # Step 2: the next-idea blend scorer (the learned per-tool prior over
        # {arma, retrieval, deduction}); built alongside the generator so it
        # joins the optimizer + state_dict.
        sc = NextIdeaScorer(dim=int(Dc))
        self._predict_next_scorer = sc if dev is None else sc.to(dev)
        return self._intervening_generator, self._reasoning_ga

    @torch.no_grad()
    def reason_about(self, query_spec, *, spaces=None, beam=8):
        """Run the N-step truth-grounded reasoning policy on a ``QuerySpec``
        (doc/plans/2026-06-23-reasoning-live-wiring.md). N = reasoning_iterations
        -- the chain depth = the number of intervening ideas. Returns a
        ``ReasoningResult`` (posture + N relevance-ranked ideas + chain), or
        ``None`` when reasoning is off (N == 0), so the caller falls back to the
        ordinary generative path (byte-identical). Inference-only; the Phase-C
        answer-loss training drives the soft route through its own hook."""
        N = int(getattr(self, "reasoning_iterations", 0) or 0)
        if N <= 0:
            return None
        from reasoning import TruthGroundedReasoner, NeuralToolUser
        reasoner = TruthGroundedReasoner(self)
        if spaces is None:
            spaces = self._reasoning_spaces()
        gen, ga = self._reasoning_tooluser(spaces)
        tool = NeuralToolUser(
            reasoner, generator=gen, ga=ga, spaces=spaces, iterations=N,
            beam=beam,
            # Model-level flag is ltm_consolidation (:912); the underscored
            # name lives on ConceptualSpace only (fixed 2026-07-16).
            materialize=bool(getattr(self, "ltm_consolidation", False)))
        return tool.run(query_spec)

    @torch.no_grad()
    def think_about(self, query_spec, *, spaces=None):
        """Run the Thinking Kernel on a ``QuerySpec`` (doc/plans/
        thinking_kernel_spec.md): the runtime-enforced lookup/part/think/query/
        answer loop over a budgeted STM frame stack. Returns the kernel's
        ``ChildResult`` (value + truth interval + trust + trace), or ``None``
        when the kernel is off (``thinking_budget == 0``) — byte-identical.
        Shares the reasoner + the soft ordering half (generator/GA/spaces) with
        ``reason_about``; LTM lemma write-back is gated by
        ``<ltmConsolidation>`` exactly as there."""
        budget = int(getattr(self, "thinking_budget", 0) or 0)
        if budget <= 0:
            return None
        from reasoning import TruthGroundedReasoner
        from thinking import ThinkingKernel, KernelPolicy
        reasoner = TruthGroundedReasoner(self)
        if spaces is None:
            spaces = self._reasoning_spaces()
        gen, ga = self._reasoning_tooluser(spaces)
        kernel = ThinkingKernel(
            reasoner, budget=budget, generator=gen, ga=ga, spaces=spaces,
            policy=KernelPolicy(
                next_op=getattr(self, "_next_op_policy", None)),
            # Same ltm_consolidation naming fix as reason_about above.
            materialize=bool(getattr(self, "ltm_consolidation", False)))
        return kernel.run(query_spec)

    def _thinking_policy_loss(self):
        """§12.6: the next-op behavior-cloning loss. Generates grounded kernel
        traces from store-derived 2-hop isPart targets (the deterministic
        teacher, materialize=False -- no LTM writes in the hot loop) and
        cross-entropy-trains the NextOpPolicy head on their (state, op) pairs.
        ``None`` when the head is not built / no store / no chains -- so the
        caller adds nothing (byte-identical). Gated by
        ``thinking_loss_weight > 0`` at the call site."""
        head = getattr(self, "_next_op_policy", None)
        if head is None:
            return None
        from reasoning import TruthGroundedReasoner
        from thinking import ThinkingKernel, traces_from_store, next_op_loss
        reasoner = TruthGroundedReasoner(self)
        budget = int(getattr(self, "thinking_budget", 0) or 0) or 16
        kernel = ThinkingKernel(reasoner, budget=budget, materialize=False)
        examples = traces_from_store(kernel)
        if not examples:
            return None
        return next_op_loss(head, examples)

    def reason_predict_next(self, state_idea, *, spaces=None):
        """Step 2: the differentiable next-idea blend over {arma, retrieval,
        deduction}. Returns ``(e_hat, weights)`` or ``None`` when reasoning is
        off (reasoning_iterations == 0). Grad-enabled (NOT @no_grad): the
        training hook calls this so the generator query head + the blend scorer
        learn which tool predicts the next idea."""
        N = int(getattr(self, "reasoning_iterations", 0) or 0)
        if N <= 0:
            return None
        from reasoning import TruthGroundedReasoner, NeuralToolUser
        reasoner = TruthGroundedReasoner(self)
        if spaces is None:
            spaces = self._reasoning_spaces()
        gen, ga = self._reasoning_tooluser(spaces)
        if gen is None:
            return None
        tool = NeuralToolUser(reasoner, generator=gen, ga=ga, spaces=spaces,
                              iterations=N, beam=8)
        return tool.reason_predict_next(
            state_idea, spaces=spaces,
            scorer=getattr(self, "_predict_next_scorer", None))

    def _predict_next_loss(self):
        """Step 2: train the next-idea POLICY. ``e_gold`` = the observed next
        end-state root (the SAME chain ``arma`` reads, DETACHED); the state = the
        prior root. ``L = 1 - cos(e_hat, e_gold)``. ``None`` when there is no
        >=2-entry chain / reasoning off / no predictor -- so the caller adds
        nothing (byte-identical). Gated by ``predict_next_loss_weight > 0``."""
        if int(getattr(self, "reasoning_iterations", 0) or 0) <= 0:
            return None
        disc = getattr(getattr(self, "symbolSpace", None), "discourse", None)
        if disc is None or getattr(disc, "_inter_predictor", None) is None:
            return None
        chain = disc.get_stm_chain(n=2, b=0)
        if len(chain) < 2:
            return None
        state = disc._reduce_end_state_to_root(chain[-2][1]).detach().reshape(-1)
        e_gold = disc._reduce_end_state_to_root(chain[-1][1]).detach().reshape(-1)
        out = self.reason_predict_next(state)
        if out is None or out[0] is None:
            return None
        from reasoning import _fit_dim
        e_hat = _fit_dim(out[0], int(e_gold.numel()))
        eh = e_hat / e_hat.norm().clamp_min(1e-12)
        eg = e_gold / e_gold.norm().clamp_min(1e-12)
        return 1.0 - (eh * eg).sum()        # 1 - cosine; grad via the blend

    def _answer_policy_loss(self):
        """Phase C: the differentiable answer-policy loss. Trains the soft
        reasoning route (the InterveningIdeaGenerator query head + the
        GlobalAttention scorer) on (A, B, gold) examples derived from the
        reasoning store; the hard deduction (the bridge mask) is detached, so
        gradient never flows through a verdict (§0). ``None`` when the generator
        is not built / no store / no chains -- so the caller adds nothing
        (byte-identical). Gated by ``answer_loss_weight > 0`` at the call site."""
        gen = getattr(self, "_intervening_generator", None)
        if gen is None:
            return None
        from reasoning import (TruthGroundedReasoner,
                               policy_examples_from_store, policy_answer_loss)
        reasoner = TruthGroundedReasoner(self)
        examples = policy_examples_from_store(reasoner)
        if not examples:
            return None
        return policy_answer_loss(gen, self._reasoning_spaces(),
                                  reasoner, examples)

    def _realize_vec(self, vec):
        """Tier-1 realize (Phase D): snap an idea vector to its nearest lexicon
        word via the PERCEPTUAL codebook -- the same nearest-codebook path
        ``_infer_ir`` uses. Returns '' on a zero-norm vector or a codebook miss.
        Tier-2 (the grammatical ``decode_to_concept``) is inert on compact
        configs (identity unless symbol_dim == concept_dim) and is deliberately
        NOT used here -- a silent identity passthrough must not masquerade as a
        decode (it lands once the decode round-trip ships on a tall-WS config)."""
        sub = getattr(self.perceptualSpace, "subspace", None)
        cb = getattr(sub, "what", None) if sub is not None else None
        if cb is None or not hasattr(cb, "wv"):
            return ""
        v = vec.detach().reshape(-1).float()
        if float(v.norm()) == 0.0:
            return ""
        try:
            D = cb.wv.vector_size
            nbrs = cb.wv.most_similar(v[:D], topn=1)
        except Exception:
            return ""
        return str(nbrs[0][0]) if nbrs else ""

    def _realize_idea(self, idea):
        """Realize ONE reasoning idea dict to a surface string: the intervening
        idea is a bridge concept vector, realized to its nearest lexicon word.
        (Multi-word 'np1 vp np2' relation rendering lands once an idea carries a
        stored-relation ``row`` -- the chain hops in ``result.chain`` do; the
        ``result.ideas`` the N-sentence output draws from are bare concepts.)"""
        vec = idea.get("idea")
        if vec is None or not torch.is_tensor(vec):
            return ""
        return self._realize_vec(vec)

    def _realize_ideas(self, result, *, n=None):
        """Phase D: render ``result.ideas`` (already relevance-sorted + N-capped
        in ``NeuralToolUser.run``) to surface sentences, top-``n`` by relevance,
        dropping empties. Empty for isTrue/isEqual leaves (no intervening ideas)
        -- the caller falls back to posture + trace."""
        ideas = getattr(result, "ideas", None) or []
        if n is not None:
            ideas = ideas[:int(n)]
        out = []
        for idea in ideas:
            s = self._realize_idea(idea)
            if s:
                out.append(s)
        return out

    def _detect_query(self, user_msg):
        """Phase E: detect that ``user_msg`` is a query and extract its operands.
        Two-part signal (no live is_query flag exists): (1) the grammar declares
        a query rule; (2) the surface is interrogative. Operands are resolved to
        VECTORS via the perceptual codebook (QuerySpec wants tensors). Returns
        ``(surface_name, A_vec, B_vec)`` or ``(None, None, None)``."""
        try:
            from Language import TheGrammar
            if hasattr(TheGrammar, "_ensure_configured"):
                TheGrammar._ensure_configured()
            # The query capability signal: a <Queries> declaration (the
            # post-relocation home -- complete.grammar carries NO query="true"
            # rules since 2026-07-05) OR a legacy query="true" rule.
            has_query = (bool(getattr(TheGrammar, "query_ops", []))
                         or any(getattr(r, "query", False)
                                for r in getattr(TheGrammar, "rules", [])))
        except Exception:
            has_query = False
        if not has_query:
            return None, None, None
        text = (user_msg or "").strip().lower()
        if not (text.endswith("?")
                or text.startswith(("is ", "are ", "does ", "do "))):
            return None, None, None
        sub = getattr(self.perceptualSpace, "subspace", None)
        cb = getattr(sub, "what", None) if sub is not None else None

        def lookup(tok):
            if cb is None or not hasattr(cb, "wv"):
                return None
            try:
                return cb.wv[tok]
            except Exception:
                return None

        toks = [t.strip("?.,") for t in text.split() if t.strip("?.,")]
        surface, A, B = "queryPart", None, None
        if "part" in toks:
            i = toks.index("part")
            left = [t for t in toks[:i] if t not in ("is", "are")]
            right = [t for t in toks[i + 1:] if t != "of"]
            if left:
                A = lookup(left[-1])
            if right:
                B = lookup(right[-1])
        else:
            content = [t for t in toks if t not in
                       ("is", "are", "does", "do", "a", "the", "of")]
            if content and content[-1] in ("exist", "exists"):
                content = content[:-1]               # "does X exist?" -> isTrue(X)
                if content:
                    surface, A = "exist", lookup(content[0])
            elif len(content) >= 2:
                A, B = lookup(content[0]), lookup(content[-1])
            elif len(content) == 1:
                surface, A = "exist", lookup(content[0])
        if A is None:
            return None, None, None
        A = torch.as_tensor(A, dtype=torch.float32)
        if B is not None:
            B = torch.as_tensor(B, dtype=torch.float32)
        return surface, A, B

    def answer_query(self, user_msg, *, beam=8):
        """Phase E: when reasoning is on AND ``user_msg`` is a query, run the
        truth-grounded reasoner and return a payload dict (posture + the N
        relevance-ranked sentences + trace); else ``None`` so the caller falls
        back to the generative ``infer()``. Byte-identical off: returns ``None``
        immediately when ``reasoning_iterations == 0``, before any detection."""
        if int(getattr(self, "reasoning_iterations", 0) or 0) <= 0:
            return None
        surface, A, B = self._detect_query(user_msg)
        if surface is None:
            return None
        from reasoning import QuerySpec, KIND_IS_TRUE
        try:
            spec = QuerySpec.from_surface(surface, A, B)
        except ValueError:
            return None
        # isPart/isEqual need BOTH operands; a missing/OOV right operand cannot
        # form a binary query -> fall back to the generative path.
        if spec.predicate != KIND_IS_TRUE and spec.right is None:
            return None
        result = self.reason_about(spec, beam=beam)
        if result is None:
            return None
        payload = {
            "posture": result.posture,
            "confidence": float(result.confidence),
            "support_true": float(result.support_true),
            "support_false": float(result.support_false),
            "sentences": self._realize_ideas(result, n=self.reasoning_iterations),
            "trace": result.trace,
        }
        # The Thinking Kernel rides ALONGSIDE the reasoner payload (off ⇒ no
        # key ⇒ byte-identical). A kernel error must not break the answer.
        if int(getattr(self, "thinking_budget", 0) or 0) > 0:
            try:
                kres = self.think_about(spec)
            except Exception:
                kres = None
            if kres is not None:
                payload["kernel"] = {
                    "value": str(kres.value),
                    "interval": [float(kres.interval.lower),
                                 float(kres.interval.upper)],
                    "trust": float(kres.trust),
                    "luminosity": float(kres.interval.luminosity),
                    "ops": [str(e.get("op")) for e in kres.trace],
                }
        return payload

    def _passback_scope_ps(self, pass_idx, ps_default, prevCS_forSS,
                           prevPS_forPS=None):
        """Apply the WS->PS top-down pass-back to choose PartSpace's input for a
        ``t>0`` subsymbolic pass (doc/specs/mereological-order-raising.md "the
        top-down attention handoff").

        Reads the 4-case action off the STAGE-0 WholeSpace
        (``wholeSpaces[0].passback_action``) and returns the scoped PS input:

          * ``"noop"``   -> ``ps_default`` (the stage-0 percept re-fed) --
            byte-identical; the default when no scope, no words-category
            attention, or no parked run-structure observation;
          * ``"refine"`` / ``"chunk"`` -> hand the prior symbols
            (``prevCS_forSS``) back to PartSpace so its SigmaLayer (re)analyses
            them (the wide<->deep CS-symbol regroup runs inside
            ``PartSpace.forward``);
          * ``"scoped"`` -> additionally thread the nth word's ``.where`` as a
            read-only PartSpace forward-local (the
            ``.where``-on-the-second-argument scope) before the re-feed.

        Gated ``<mereologyRaise>`` (only reached under that flag). The pass-back
        sits on the multi-stage carrier the sO=3 combine fix restored."""
        wss = getattr(self, "wholeSpaces", None)
        ws0 = wss[0] if wss is not None and len(wss) > 0 else None
        if ws0 is None or not hasattr(ws0, "passback_action"):
            return ps_default
        # R2: read the CS-owned reading scope and hand it to the WS->PS passback
        # (CS is the home of the .where-producer; WS still owns the run-structure
        # route_hint that passback_action also consults).
        cs_scope = (getattr(self.conceptualSpace, "_passback_scope_where", None)
                    if self.conceptualSpace is not None else None)
        action, where = ws0.passback_action(pass_idx, scope=cs_scope)
        if action == "scoped" and where is not None:
            # null-content + the nth word's `.where`: re-analyse the prior
            # symbols, then FOCUS the percept to that span -- zero the slots
            # outside the decoded normalized [start, end] bracket (the
            # `.where`-on-the-second-argument scope). Read-only; falls back to
            # the unfocused re-feed when the span is degenerate.
            ps = self.perceptualSpace.forward(prevCS_forSS)
            ev = ps.materialize() if ps is not None else None
            if ev is not None and torch.is_tensor(ev) and ev.dim() == 3:
                # Support one shared [2] bracket and row-local [B,2]
                # brackets.  The latter is required by local tiling/reading;
                # the old reshape(-1) accidentally used row 0 for the batch.
                w = where.to(ev.device, torch.float32)
                if w.dim() == 1:
                    w = w[:2].view(1, 2).expand(int(ev.shape[0]), -1)
                elif w.dim() >= 2:
                    w = w.reshape(-1, 2)
                    if int(w.shape[0]) == 1 and int(ev.shape[0]) > 1:
                        w = w.expand(int(ev.shape[0]), -1)
                if (w.dim() == 2 and int(w.shape[0]) == int(ev.shape[0])
                        and int(w.shape[1]) >= 2):
                    start = w[:, 0].clamp(0.0, 1.0)
                    end = w[:, 1].clamp(0.0, 1.0)
                    N = int(ev.shape[1])
                    pos = ((torch.arange(N, device=ev.device).float() + 0.5)
                           / max(N, 1)).view(1, N)
                    keep = ((pos >= start.unsqueeze(-1))
                            & (pos <= end.unsqueeze(-1))
                            & (end >= start).unsqueeze(-1)).unsqueeze(-1)
                    ps.set_event(torch.where(keep, ev, torch.zeros_like(ev)))
            return ps
        # Experimental overlap path: every local family routes in parallel.
        # The PS part-stream gets σ only at slots marked by the observation;
        # WS.forward has already applied the pass-t π stack to its peer stream.
        # ReadingAttention's explicit scope above remains the higher-priority
        # serial override.
        tiling = (ws0.where_tiling_for_pass(pass_idx)
                  if hasattr(ws0, "where_tiling_for_pass") else None)
        if isinstance(tiling, dict):
            source = prevPS_forPS
            if (source is None or not hasattr(source, "is_empty")
                    or source.is_empty()):
                source = prevCS_forSS
            return self.perceptualSpace.synthesize_feedback_where(
                source, pass_idx, tiling.get("sigma_part"),
                default=ps_default)
        if action in (None, "noop"):
            return ps_default
        if action in ("refine", "chunk"):
            return self.perceptualSpace.forward(prevCS_forSS)
        return ps_default

    def _reverse_body(self, sub):
        """Per-stage body reverse, mirroring ``_forward_body`` order.

        Inverts the primary IS→PS→CS→OS path's body: walk stages in
        reverse, undo the optional N-halving ``merge`` then
        ``ConceptualSpace.reverse`` (C → percept space_role). WholeSpace is
        the symbolic recurrent loop leg, off the OS→CS→PS→IS
        reconstruction path; its loop contribution was averaged into C
        in the forward and cannot be decomposed by a single-input
        reverse (round-trip holds for the local per-space inverse).
        B-shaped throughout (the legacy K-axis flatten/restore was
        retired with the AR cursor unfold).

        The merge's forward-cached ``_merge_diff`` is overwritten on
        each of the T recurrent passes, so its N-halving is only
        exactly invertible for the last pass; per-stage reverse is
        guarded so a shape mismatch in an earlier stage degrades the
        reconstruction (approximate through the averaged loops, per the
        single-input-reverse contract) instead of aborting.
        """
        carriers = getattr(self, "_combine_carriers", None)
        _concepts_recon = None
        for t in reversed(range(len(self.body_stages))):
            stage = self.body_stages[t]
            # A4 (2026-06-06 parallel-conceptual-recurrence): invert the
            # per-stage ConceptualCombine before the CS bookkeeping reverse
            # -- exact mirror of the forward order ``cs.forward -> combine``.
            #
            # The combine reverse is driven by the EXACT forward carrier
            # ``carriers[t]`` (the stage-t combine output, threaded as a
            # forward-local), NOT by ``sub``'s content. Reason: the STM push
            # happens INSIDE ``cs.forward`` -- BEFORE the combine writes the
            # advanced carrier via ``set_event`` -- so the STM (and therefore
            # ``sub``, which the reconstruction path sources from the STM
            # snapshot) holds the PRE-combine perception event, not the combine
            # output ``next_cs`` that ``aug_t`` is paired with. Feeding ``sub``
            # into ``combine.reverse`` would break the exact ``next_cs <-> aug``
            # pairing. CONSEQUENCE (a deliberate deviation from design sec
            # 3.2/3.3): this is NOT a propagated reverse walk -- each stage
            # re-inverts against the stashed forward carrier rather than a
            # carrier propagated from the next stage's reverse. Acceptable
            # near-term (the STM is the only state the production reverse path
            # can source); because the round-trip's exactness comes from the
            # stashing rather than a true chained walk,
            # ``test_mm5m_combine_carrier_roundtrip`` asserts the chained
            # consistency (``cs_rec[t] == carriers[t-1]``) so a future break of
            # the forward ``prev_cs`` threading is still caught.
            # ``combine.reverse`` recovers the three input
            # streams ``(PS_t, WS_t, CS_t)``:
            #
            #   * t > 0: the prior carrier is just ``carriers[t-1]`` (already
            #     threaded), so the per-stage reverse only needs to surface
            #     the recovered slab for ``cs.reverse``'s bookkeeping.
            #   * t == 0 (FINAL reverse stage): the PS-stream output PS_0 is
            #     the pi-ENCODED input content (alpha_ps live only at t=0) --
            #     the slab the downstream ``cs.reverse`` / ``_reverse_perceptual``
            #     (pi.reverse) decode back to the input. Surfacing PS_0 here
            #     closes the input-reconstruction round-trip (e.g. XOR_exact).
            #
            # The combine ALWAYS ran on the forward (the ``reconstruct=perfect``
            # skip-combine mode was retired with the enum, A1), so the reverse
            # is UNCONDITIONALLY the dropped reverse: the structured zero-pad
            # inverts the same square map (exact only on the rank-D subspace
            # that survived). Gated to the plain (non-grammar) parallel path,
            # matching the forward gate. Falls back to ``sub``'s own content
            # when the forward carrier is unavailable (e.g. a reverse not
            # preceded by a body forward in this object).
            # The combine-reverse demuxes ``sub``, recovers the content, and
            # writes it back via ``set_event`` so the downstream ``cs.reverse``
            # reads the canonical event.
            if self.subsymbolicOrder >= 1 and "merge" not in stage:
                try:
                    combine = getattr(stage["cs"], "combine", None)
                    if combine is None:
                        raise RuntimeError("stage cs has no combine")
                    D = int(combine.content_dim)
                    content, band, event = self._combine_demux(sub, D)
                    if event is not None:
                        # 2-stream EXACT inverse (C-10), in-Space per the
                        # processing contract: ``cs.unbind`` reads the
                        # bind carrier RIDING ON the stage's SubSpace
                        # (``_bind_carrier``) and applies ILL^{-1} --
                        # (PS_t, WS_t) exactly, nothing threaded
                        # alongside. The PS stream is the percept leg
                        # (re-fed at every stage), which the downstream
                        # ``cs.reverse`` / ``_reverse_perceptual`` decode
                        # back to the input: reconstruction ends at the
                        # percept store (PS owns reconstruction). The
                        # model-level ``carriers`` handle remains a
                        # fallback for reverses driven from a foreign
                        # subspace.
                        rec = stage["cs"].unbind(stage["cs"].subspace)
                        if (rec is None and carriers is not None
                                and t < len(carriers)
                                and carriers[t] is not None):
                            rec = combine.reverse(carriers[t])
                        if rec is not None:
                            ps_rec, _ws_rec = rec
                            recovered = ps_rec
                            if t == 0:
                                # Phase 7 (painting reverse): the stage-0
                                # WS stream of the unbind IS the conceptual
                                # reconstruction branch. Captured here and
                                # stamped onto the RETURNED sub below (the
                                # per-stage cs.reverse may swap subspace
                                # objects); it rides the SubSpace down to
                                # InputSpace.reverse (processing contract:
                                # data in SubSpaces; reverse stays
                                # single-arg), where the Universal view
                                # paints the background and the Atomic
                                # view is averaged in.
                                _concepts_recon = _ws_rec.detach()
                        else:
                            # No carrier available (a reverse not preceded
                            # by a body forward): surface the sub's own
                            # content for cs.reverse's bookkeeping.
                            recovered = content
                        # Write the recovered slab back, band riding along
                        # unchanged.
                        if band is not None and band.shape[-1] > 0:
                            sub.set_event(
                                torch.cat([recovered, band], dim=-1))
                        else:
                            sub.set_event(recovered)
                except (RuntimeError, AssertionError, ValueError):
                    pass
            if "merge" in stage:
                try:
                    sub = stage["merge"].reverse(sub)
                except (RuntimeError, AssertionError, ValueError):
                    pass
            try:
                sub = stage["cs"].reverse(sub)
            except (RuntimeError, AssertionError, ValueError):
                pass
        if _concepts_recon is not None and sub is not None:
            # Stamp the conceptual reconstruction branch onto the RETURNED
            # sub (the per-stage reverses may have swapped objects); the
            # model reverse() re-stamps it across _reverse_perceptual's
            # handoff so it reaches InputSpace.reverse for the painting.
            object.__setattr__(sub, "_concepts_recon", _concepts_recon)
        return sub

    def _reverse_perceptual(self, sub):
        """Reverse the perceptual boundary (percept space_role → input)."""
        return self.perceptualSpace.reverse(sub)

    def reverse(self, x):
        """Reverse pipeline from the terminal ConceptualSpace state -- i.e.
        GENERATE the surface (a sentence) for an idea.

        This is NOT merely "undo the forward". When there is no reconstruction
        information for an idea -- e.g. an idea placed into STM TOP-DOWN
        (generated or recalled, never perceived) -- we still must produce a
        sentence for it. That is exactly this pass: starting from the idea (the
        terminal C-space_role state), run the reverse pipeline to emit its surface.

        The FIRST steps on that path are the high-level GRAMMATICAL operations
        (``_chart_generate_from_stm`` -> ``_reverse_body``): guess the HEAD of
        the sentence's NP (helped by the activated codebook), determine the VP
        from that head, and recurse down to all the surface words. Only then do
        the lower space_roles (``_reverse_perceptual`` -> ``inputSpace.reverse``)
        render the chosen words back to bytes.

        C3 (spec sec 7): this is the UNCONDITIONAL reconstruction carrier.
        After the ``<reconstruct>`` enum was retired (A1), the symbolic/output
        reverse modes are gone and ``runBatch`` always seeds the reverse pass
        from concepts: the output head may be intentionally lower-dimensional,
        while the terminal C-space_role state still carries the full reversible
        surface needed for reconstruction. (The retired head-seeded primitive
        ``_run_pipeline_rev`` -- which started from ``outputSpace.reverse`` /
        the lower-dim OS output, hence the wrong-size seed -- was removed
        2026-06-07; concepts-seeding is precisely the fix for that size gap.)
        """
        if x is None:
            return None
        self._chart_generate_from_stm(x)
        # Idea-decode (Stage D3 CONSUMER, doc/old/2026-06-20-idea-decoder.md):
        # when <ideaDecode> is on, RUN the grammar <generate> reverse on the
        # syntactic WholeSpace (idea -> symbol expansion -> concept) and DRIVE the
        # reverse seed from it, so the surface words come from the grammar
        # (generation = dual of comprehension). Shape-guarded: drives on an exact
        # match (the symbol_dim==concept_dim design invariant); compact-symbol
        # configs fall back unchanged (they need a learned symbol->concept
        # expander). Default off -> skipped -> byte-identical.
        if getattr(self, "idea_decode", False):
            x = self._idea_decode_drive(x)
        x = self._reverse_body(x)
        # Phase 7 (painting reverse): carry the conceptual reconstruction
        # branch across the PS handoff (the perceptual reverse may swap
        # subspace objects) so it reaches InputSpace.reverse riding the
        # SubSpace.
        _concepts = getattr(x, "_concepts_recon", None) if x is not None else None
        x = self._reverse_perceptual(x)
        if _concepts is not None and x is not None:
            object.__setattr__(x, "_concepts_recon", _concepts)
        x = self.inputSpace.reverse(x)
        return x

    @torch.compiler.disable
    def _idea_decode_ws(self):
        """The WholeSpace that owns the grammar ``<generate>`` machinery.

        It is ``symbolSpace.wholeSpace`` -- the S-space_role host where
        ``_attach_per_space_syntactic_layer`` installs the SyntacticLayer
        (Language.py) and whose subspace is populated by the forward. This is the
        LAST per-stage WholeSpace (``wholeSpaces[-1]``), NOT ``wholeSpaces[0]``
        (which has no syntacticLayer and an empty subspace). Falls back to the
        last per-stage WS."""
        ws = getattr(getattr(self, "symbolSpace", None), "wholeSpace", None)
        if ws is None and getattr(self, "wholeSpaces", None):
            ws = self.wholeSpaces[-1]
        return ws

    @torch.compiler.disable
    def _run_idea_decode_generate(self):
        """Run the grammar ``<generate>`` reverse (idea -> symbol expansion ->
        concept) on the SYNTACTIC WholeSpace and return its result (Stage D3).
        Runs ``WholeSpace.reverse`` -> ``SyntacticLayer.reverse`` (the generate
        rules) -> ``reverseSymbols`` -> ``reverseEnd``. Best-effort: any failure
        returns ``None`` so the reconstruction path is never broken."""
        ws = self._idea_decode_ws()
        if ws is None or getattr(ws, "subspace", None) is None:
            return None
        try:
            return ws.reverse(ws.subspace)
        except (RuntimeError, AssertionError, ValueError, TypeError, IndexError):
            return None

    @torch.compiler.disable
    def _idea_decode_drive(self, x):
        """Stage D3 CONSUMER: drive the reverse SEED from the grammar
        ``<generate>`` decode, so the surface words come from the grammar
        (generation = dual of comprehension), not the carried reconstruction.

        The grammar decode (``_run_idea_decode_generate`` on the syntactic WS)
        yields the idea's symbol expansion; ``decode_to_concept`` maps it to
        concept space. When that matches the reverse seed ``x``'s event shape
        EXACTLY, we set it as the seed and let the existing ``_reverse_body ->
        _reverse_perceptual -> _reverse_text`` chain render it to words (the
        injection point verified by the D3 map: the seed event drives the words).

        SHAPE-GUARDED: drives only on an exact shape match. The
        ``symbol_dim == concept_dim`` design invariant makes ``decode_to_concept``
        an exact (identity) lift; the COMPACT-symbol configs (``nOutputDim`` 8 <<
        concept width, e.g. MM_20M) need a LEARNED ``symbol_dim -> concept_dim``
        expander (a lossy bottleneck, a training follow-on) -- there the widths
        differ and this falls back to ``x`` unchanged. Best-effort + gated by
        ``idea_decode`` (default off -> never called -> byte-identical). The
        grammar decode is also parked on ``_idea_decode_parked`` for observability."""
        self._idea_decode_parked = None
        gen = self._run_idea_decode_generate()
        if gen is None or x is None:
            return x
        try:
            gev = gen.materialize()
        except Exception:
            return x
        if gev is None or not torch.is_tensor(gev):
            return x
        self._idea_decode_parked = gev.detach().clone()
        try:
            ws = self._idea_decode_ws()
            gev = ws.decode_to_concept(gev) if ws is not None else gev
            cur = x.materialize()
        except Exception:
            return x
        if (cur is None or not torch.is_tensor(cur)
                or gev.dim() != 3 or cur.dim() != 3
                or tuple(gev.shape) != tuple(cur.shape)):
            # Width/N mismatch: compact-symbol config (needs the learned
            # symbol->concept expander) or a non-single-S seed. Fall back to the
            # carried reconstruction unchanged.
            return x
        try:
            x.set_event(gev)
        except Exception:
            return x
        return x

    def _reverse_method1_leaves(self):
        """Method-1 EXACT decode (serial plan Task 2): render the STORED
        per-word percept LEAVES straight through the percept store.

        The serial derivation records its leaves -- the per-word percept
        events the bottom-up parse started from -- on the forward
        (``_stm_pre_reduce_slab``, ``[B, N, D]``, word order, batch-scoped
        like ``_stm_single_S``). ``reverse`` replays them by staging the
        radix render thunk directly on those leaves: the percept-store
        nearest-neighbour decode recovers each word EXACTLY, by construction
        -- it needs no training and no per-op inverse, because a percept's
        vector position IS its identity (doc/Spaces.md#percept-guarantees). This
        is the design's TEACHER (the exact reference Method-2's free
        derivation is scored against); the collapsed-idea CS reverse
        (``_reverse_from_S``) stays the trained STUDENT path.

        Why not the CS reverse: the reduce folds per-word ideas into one S
        through lattice ops whose inverse is not exact on an untrained model
        (the CS-reverse of the collapsed root decodes one dominant word, and
        of the per-word ideas decodes nearest-cone junk) -- so Method-1's
        by-construction exactness has to ride the STORED leaves, not an
        algebraic un-fold.

        Returns the ``[B, N, D]`` leaf slab (also the reverse event the
        eval reconstruction loss reads), or ``None`` when no leaves were
        stashed (parallel mode / a non-radix percept store) so the caller
        falls back to the tensor arm.
        """
        slab = getattr(self, '_stm_pre_reduce_slab', None)
        if slab is None or not torch.is_tensor(slab) or slab.dim() != 3:
            return None
        psp = getattr(self, 'perceptualSpace', None)
        radix = getattr(psp, 'vocabulary', None) if psp is not None else None
        try:
            from Layers import RadixLayer
        except ImportError:
            return None
        if not isinstance(radix, RadixLayer):
            return None
        # Stage the render thunk on the leaves (mirrors PartSpace.reverse's
        # radix staging); ``reconstruct_data`` reads it for the decode +
        # where-recovery. Reset the memoised decode so the fresh leaves win
        # (including over any stale Method-2 un-fold slab).
        object.__setattr__(psp, '_recovered_input', None)
        object.__setattr__(psp, '_unfold_recovered_slab', None)
        object.__setattr__(
            psp, '_recovered_input_thunk',
            ("radix", radix, slab.detach(), psp.subspace))
        return slab

    def _grammar_reverse_ops(self):
        """Enumerate the arity-2 REVERSE ops from the grammar's ``<generate>``
        section (Alec 2026-07-14, doc/plans/2026-07-14-signed-space-snap-
        design.md) — symmetric to ``_stm_reducer``'s ``<compose>`` enumeration.

        A generate rule (``op_I1, op_I2 = op.reverse(op_O1)``) carries
        ``.reverse`` in its canonical string; its ``method_name`` resolves the
        SAME host layer the forward uses (``syntacticLayer._by_name``). Keep
        the arity-2 hosts whose ``reverse`` accepts ``left_rows`` (the
        snap/recommender family — union / intersection / …). Returns
        ``[(name, host), …]``; the reverse operation set the derivation
        chooses from, DETERMINED FROM THE GRAMMAR FILE, not the forward
        trace."""
        ws = getattr(self, "wholeSpace", None)
        sl = getattr(ws, "syntacticLayer", None)
        if sl is None or not hasattr(sl, "_by_name"):
            return []
        try:
            from Language import TheGrammar
        except ImportError:
            return []
        ops, seen = [], set()
        for rdef in TheGrammar.rules:
            if ".reverse" not in (getattr(rdef, "canonical", "") or ""):
                continue                        # a <compose> (forward) rule
            mn = getattr(rdef, "method_name", None)
            if not mn or mn in seen:
                continue
            host = sl._by_name.get(mn)
            if host is None or int(getattr(host, "arity", 1)) != 2:
                continue
            rv = getattr(host, "reverse", None)
            try:
                names = rv.__code__.co_varnames
            except AttributeError:
                continue
            if "left_rows" not in names:
                continue                        # not snap/recommender-capable
            seen.add(mn)
            ops.append((mn, host))
        return ops

    def _reverse_choose_op(self, parent, reverse_ops, basis, word_rows):
        """Reverse chooser (Alec 2026-07-14): the reverse derivation's own
        op selection, with NO forward record. For each candidate grammar
        reverse op, snap-reverse the parent to an operand pair, RE-FOLD it
        forward, and score the ROUND-TRIP fit (cosine to the parent). The op
        whose forward fold best reconstructs the parent is the one that
        explains it — the trace-free analogue of the forward anchor-dot
        chooser. Returns ``((name, host), (x1, x2), fit)`` or ``None``."""
        import torch.nn.functional as _F
        p = parent.reshape(-1)
        d_what = int(p.shape[0])
        best = None
        for name, host in reverse_ops:
            try:
                x1, x2 = host.reverse(
                    parent.unsqueeze(0), basis=basis,
                    left_rows=word_rows, right_rows=word_rows, snap=True)
                a = x1.reshape(-1)[:d_what]
                b = x2.reshape(-1)[:d_what]
                recon = host.compose(
                    a.unsqueeze(0), b.unsqueeze(0)).reshape(-1)[:d_what]
                fit = float(_F.cosine_similarity(
                    recon.detach().unsqueeze(0),
                    p.detach().unsqueeze(0), dim=-1))
            except Exception:
                continue                        # op not applicable here
            if best is None or fit > best[2]:
                best = ((name, host), (x1, x2), fit)
        return best

    def _leaf_distill_loss(self):
        """Method-1 -> Method-2 distillation (Alec 2026-07-14, snap design
        doc step 3): train the leaf-decoder head to regenerate the EXACT
        per-word leaves (``_stm_pre_reduce_slab``, the Method-1 teacher)
        from the collapsed root (``_stm_single_S``). Gradient flows through
        the root into the fold, so the root cannot satisfy this while
        collapsing distinct sentences into one attractor (measured pairwise
        contraction 0.0117 $\\to$ 0.00017 — the root-separability gap).
        Masked MSE over the teacher's non-zero leaf slots; ``None`` when a
        stash is missing or the root carries no graph."""
        S = getattr(self, "_stm_single_S", None)
        slab = getattr(self, "_stm_pre_reduce_slab", None)
        if (S is None or slab is None or not torch.is_tensor(S)
                or not torch.is_tensor(slab) or S.dim() != 2
                or slab.dim() != 3 or S.grad_fn is None
                or int(S.shape[0]) != int(slab.shape[0])):
            return None
        _B, N, D = slab.shape
        head = getattr(self, "_leaf_distill_head_module", None)
        if (head is None or head.d_in != int(S.shape[1])
                or head.n_slots != int(N) or head.d_out != int(D)):
            from Layers import LeafDecoderHead
            head = LeafDecoderHead(int(S.shape[1]), int(N), int(D))
            head = head.to(S.device)
            self.add_module("_leaf_distill_head_module", head)
            self._leaf_distill_head_fresh = True
        pred = head(S)                                       # [B, N, D]
        target = slab.detach().to(pred.device)
        mask = (target.abs().sum(-1, keepdim=True) > 0).to(pred.dtype)
        if float(mask.sum()) == 0.0:
            return None
        num = (mask * (pred - target) ** 2).sum()
        return num / (mask.sum() * float(D))

    def _nearest_word(self, node, basis, word_rows):
        """Nearest store word row to ``node`` (``[d_what]``) by cosine, and
        that cosine. The leaf test of the trace-free derivation: a node that
        already IS a store row (cos ~1) is a word, not a composite to
        un-fold further."""
        import torch.nn.functional as _F
        W = basis.getW()
        Wr = W.index_select(0, word_rows.to(W.device))
        sims = _F.cosine_similarity(node.reshape(1, -1), Wr, dim=-1)
        j = int(sims.argmax())
        return Wr[j], float(sims[j])

    def _reverse_derive_words(self, S, d_what, basis, word_rows, reverse_ops):
        """Trace-free reverse derivation (Alec 2026-07-14): un-fold each root
        into word leaves by CHOOSING the reverse op per step
        (``_reverse_choose_op`` — round-trip fit over the grammar's
        ``<generate>`` ops), with NO forward record. The chosen op's snap
        returns store word rows, so a node that snaps to a row (cos ~1) is a
        leaf; the flat/2-word corpus bottoms out in one un-fold. Deeper trees
        recurse (bounded by the STM cap) but the collapsed root cannot yet
        yield separable composite children — gated on root separability
        (step 3). Returns ``[[word_vec, …], …]`` per batch row."""
        B = S.shape[0]
        cap = int(getattr(self.conceptualSpace.stm, "capacity", 8) or 8)
        LEAF = 0.999
        words_per_row = [[] for _ in range(B)]
        for b in range(B):
            leaves, frontier, steps = [], [S[b, :d_what]], 0
            while frontier and len(leaves) < cap and steps < 2 * cap:
                steps += 1
                node = frontier.pop(0)          # FIFO -> left-first reading
                if float(node.abs().sum()) == 0.0:
                    continue
                w, wf = self._nearest_word(node, basis, word_rows)
                if wf > LEAF:
                    leaves.append(w)             # node IS a word: leaf
                    continue
                choice = self._reverse_choose_op(
                    node, reverse_ops, basis, word_rows)
                if choice is None:
                    leaves.append(w)             # no op inverts: snap + stop
                    continue
                (_name, _host), (x1, x2), _fit = choice
                frontier.append(x1.reshape(-1)[:d_what])
                frontier.append(x2.reshape(-1)[:d_what])
            words_per_row[b] = leaves
        return words_per_row

    def _reverse_reduce_unfold(self, S):
        """Method-2 reverse-reduce (serial plan Task 4): un-fold the collapsed
        root ``S`` ``[B, D]`` back into per-word ideas ``[B, N, D]`` by walking
        the forward's recorded fold steps BACKWARD.

        Each backward step calls the CHOSEN op's basis-threaded ``reverse``
        (e.g. ``UnionLayer.reverse(parent, basis)`` -> ``Ops.disjunctionReverse``,
        the CODEBOOK-WALK recommender: it picks an operand pair ``(x1, x2)``
        from the codebook with ``op(x1, x2) ~= parent`` -- since neither word
        is a part of the other, the join keeps enough of each word's edge to
        reconstitute the residual word; this is a LOOKUP, not a subtraction).
        Per the newest-at-slot-0 fold convention (left = older), the backward
        walk emits ``x1`` (left) as the next word and carries ``x2`` (right)
        into the next step; the final carry is the last word.

        Trace: ``_stm_reduce_op_trace`` -- per sweep step
        ``(reduce_marginal_op [B, 1, R], can [B])`` appended by
        ``_stm_bounded_reduce_step`` (reset per ``_stm_reduce_to_single_S``
        sweep). Rows masked out of a step (can=False) emit nothing there.
        Returns ``None`` (caller falls back to the single-slot CS reverse)
        when there is no trace, no reducer, or no dimension-matched basis.
        Eval/eager only (the free-derivation decode path).
        """
        if S is None or not torch.is_tensor(S) or S.dim() != 2:
            return None
        B, D = S.shape
        # The STM idea is the MUXED event [what | where | when]; the codebook
        # rows are nWhat-wide (content only). Un-fold on the .what slice; the
        # band tail rides zeroed (scaffold placement comes from the forward
        # record; blind placement from the percept band, not these ideas).
        sub = self.conceptualSpace.subspace
        nw = int(getattr(sub, "nWhere", 0) or 0)
        nn_ = int(getattr(sub, "nWhen", 0) or 0)
        d_what = D - nw - nn_
        if d_what <= 0:
            return None
        # Basis for the recommender: gated <PartSpace><wordStore>, the PS
        # percept store's WORD collection (type="words") — the words are
        # rows of PS ``subspace.what`` (percept id == row), dim-matched to
        # the idea .what by construction (content-width rows), restricted
        # via the recommender's left_rows/right_rows masks
        # (doc/plans/2026-07-12-word-store-typed-reverse.md). Knob off /
        # no word rows / dim mismatch -> the first dim-matched codebook,
        # unchanged (docstring contract: "typically WholeSpace.subspace.what").
        basis = None
        word_rows = None
        ps_space = getattr(self, "perceptualSpace", None)
        if getattr(ps_space, "word_store_reverse", False):
            store = getattr(ps_space, "percept_store", None)
            what = getattr(getattr(ps_space, "subspace", None), "what", None)
            W = what.getW() if hasattr(what, "getW") else None
            if (store is not None and hasattr(store, "word_ids")
                    and torch.is_tensor(W) and W.dim() == 2
                    and int(W.shape[1]) == int(d_what)):
                # AUTHORITATIVE source (plan v3): the IN-MODEL label — words
                # recognized 1:1 and registered under the WORDS concept.
                rows = None
                _cs_list = (list(getattr(self, "conceptualSpaces", []) or [])
                            or [getattr(self, "conceptualSpace", None)])
                for _cs in _cs_list:
                    fn = getattr(_cs, "recognized_word_rows", None)
                    r = fn() if callable(fn) else None
                    if r is not None and int(r.numel()) > 0:
                        rows = r
                        break
                # Fallback: the store's promoted collection (synthesis-side
                # recurrence evidence) — e.g. configs without <mereologyRaise>
                # never run the recognition seam.
                if rows is None:
                    _ws_list = getattr(self, "wholeSpaces", None)
                    _sb = getattr(_ws_list[0] if _ws_list else None,
                                  "_standalone_run_bytes", None)
                    r = store.word_ids(standalone_bytes=_sb)
                    rows = r if int(r.numel()) > 0 else None
                if rows is not None:
                    basis = what
                    word_rows = rows.to(S.device)
        if basis is None:
            for space in (getattr(self, "wholeSpace", None),
                          getattr(self, "conceptualSpace", None)):
                what = getattr(getattr(space, "subspace", None), "what", None)
                W = what.getW() if hasattr(what, "getW") else None
                if (torch.is_tensor(W) and W.dim() == 2
                        and int(W.shape[1]) == int(d_what)):
                    basis = what
                    break
        if basis is None:
            return None
        # TRACE-FREE grammar-driven derivation (Alec 2026-07-14, doc/plans/
        # 2026-07-14-signed-space-snap-design.md): when word rows exist and
        # the grammar declares reverse ops, the reverse finds its OWN
        # derivation — choosing the op per un-fold step by round-trip fit
        # over the <generate> ops, with NO forward record. Legacy configs
        # (no word rows) fall through to the recorded-trace walk below.
        reverse_ops = (self._grammar_reverse_ops()
                       if word_rows is not None else [])
        if reverse_ops:
            words_per_row = self._reverse_derive_words(
                S, d_what, basis, word_rows, reverse_ops)
            n = max(1, max((len(w) for w in words_per_row), default=1))
            out = S.new_zeros(B, n, D)
            for b, ws in enumerate(words_per_row):
                for i, w in enumerate(ws):
                    out[b, i, :d_what] = w
            self._stamp_unfold_where(out, d_what, basis, word_rows)
            psp = getattr(self, 'perceptualSpace', None)
            if psp is not None:
                object.__setattr__(psp, '_unfold_recovered_slab',
                                   out.detach())
            return out
        # LEGACY trace-walk: replay the recorded forward fold trace backward
        # (no word rows / no grammar reverse ops).
        trace = getattr(self, "_stm_reduce_op_trace", None)
        if not trace:
            return None
        reducer = self._stm_reducer()
        if reducer is None or not len(getattr(reducer, "ops", [])):
            return None
        words_per_row = [[] for _ in range(B)]      # emitted, earliest-first
        # Snap-path words, by which SIDE of the fold carried them (review
        # finding, 2026-07-14): an rw step peels the LAST word of the
        # remaining span (the serial chain folds fold(composite, NEWEST
        # word)) — those collect latest-first and reverse at the end; an
        # lw-only step peels the FIRST word (the seal sweep's shape —
        # after its first fold the parent sits at slot 0 and every later
        # fold is fold(older word, composite)) — those are already
        # earliest-first in walk order; the pair (both words) sits
        # between the peeled heads and tails.
        head_per_row = [[] for _ in range(B)]       # lw-only, walk order
        pair_per_row = [[] for _ in range(B)]       # word∧word pair
        tail_per_row = [[] for _ in range(B)]       # rw, LATEST-first
        carry = [S[b, :d_what] for b in range(B)]   # [d_what] per row
        # Word-bearing filtering (open-fronts Task B): 4-tuple trace steps
        # carry the fold's operand kinds — emit x1 only where the LEFT was a
        # word; the final carry is a word only if the FIRST forward fold's
        # RIGHT was (backward-order overwrite lands exactly that). Legacy
        # 2-tuples = unfiltered.
        carry_word = [True] * B
        # Word-bearing folds un-fold through the RECOMMENDER family (todo
        # §1, fold-op choice): the untrained DP chooser routes folds
        # through relation hosts whose reverses return non-codebook
        # vectors — those can never bind word content or placement. On
        # the gated path, a step whose operands include a word dispatches
        # the snap (union-preferred recommender over the word rows)
        # instead of the chosen op's reverse; non-word steps keep the
        # chosen op (status quo).
        rec_gl = None
        if word_rows is not None:
            for _ad in reducer.ops:
                _g = getattr(_ad, "gl", None)
                _rv = getattr(_g, "reverse", None)
                if _rv is not None and \
                        "left_rows" in _rv.__code__.co_varnames:
                    if rec_gl is None:
                        rec_gl = _g
                    # Prefer the MAX-fold family: its recommender keeps
                    # word rows feasible (largest row <= parent), where the
                    # min-fold's >= filter yields only sentinels against a
                    # composite parent.
                    if getattr(_g, "rule_name", "") in ("union",
                                                        "disjunction"):
                        rec_gl = _g
                        break
        for step in reversed(trace):
            if len(step) == 4:
                marg, can, lw, rw = step
            else:
                marg, can = step
                lw = rw = None
            op_idx = marg[:, 0, :].argmax(dim=-1)   # [B] chosen op per row
            for b in range(B):
                if not bool(can[b]):
                    continue                        # row did not fold this step
                gl = getattr(reducer.ops[int(op_idx[b])], "gl", None)
                if gl is None or not hasattr(gl, "reverse"):
                    return None
                if (rec_gl is not None and lw is not None
                        and (bool(lw[b]) or bool(rw[b]))):
                    gl = rec_gl                     # the word-fold snap
                parent = carry[b].unsqueeze(0)      # [1, d_what]
                if (word_rows is not None and lw is not None
                        and (bool(lw[b]) or bool(rw[b]))):
                    # Signed-space snap (Alec's design call, 2026-07-14;
                    # doc/plans/2026-07-14-signed-space-snap-design.md):
                    # dot-metric snap over the word rows replaces the
                    # order-filter recommender at idea grain — the radial
                    # feasibility filter admitted no real row against
                    # trained composites and returned the ⊤ sentinel for
                    # every free-derivation operand (measured). Words
                    # route to head/pair/tail by fold side (see the list
                    # declarations above); the old emit-x1-if-left-was-
                    # word contract dropped every right-side word —
                    # measured empty decodes at trained budgets.
                    if float(parent.abs().sum()) == 0.0:
                        continue    # span exhausted by an earlier pair
                    Wm = basis.getW()
                    if bool(lw[b]) and bool(rw[b]):
                        a, bb = Ops.word_pair_snap(parent, Wm, word_rows)
                        pair_per_row[b] = [a.reshape(-1), bb.reshape(-1)]
                        # Both operands recovered: the single-carry walk
                        # cannot branch further into this fold's span.
                        carry[b] = torch.zeros_like(carry[b])
                        carry_word[b] = False
                    else:
                        w_star, resid = Ops.word_side_snap(
                            parent, Wm, word_rows)
                        if bool(rw[b]):
                            tail_per_row[b].append(w_star.reshape(-1))
                        else:
                            head_per_row[b].append(w_star.reshape(-1))
                        carry[b] = resid.reshape(-1)
                        carry_word[b] = False
                    continue
                else:
                    try:
                        if word_rows is not None:
                            pair = gl.reverse(parent, basis=basis,
                                              left_rows=word_rows,
                                              right_rows=word_rows)
                        else:
                            pair = gl.reverse(parent, basis=basis)
                    except TypeError:
                        try:
                            pair = gl.reverse(parent, basis=basis)
                        except TypeError:
                            pair = gl.reverse(parent)  # unary-signature op
                        except NotImplementedError:
                            return None         # no faithful inverse: fall back
                    except NotImplementedError:
                        return None             # no faithful inverse: fall back
                if not (isinstance(pair, tuple) and len(pair) == 2):
                    return None
                x1, x2 = pair
                if lw is None or bool(lw[b]):
                    words_per_row[b].append(x1.reshape(-1))  # left = older word
                carry[b] = x2.reshape(-1)                    # right rides on
                carry_word[b] = True if rw is None else bool(rw[b])
        for b in range(B):
            if carry_word[b]:
                words_per_row[b].append(carry[b])   # final carry = last word
            # Assemble the reading order the where-stamp assumes: the
            # left-peeled heads (walk order = earliest-first), the pair,
            # then the right-peeled tail reversed to earliest-first.
            words_per_row[b].extend(head_per_row[b])
            words_per_row[b].extend(pair_per_row[b])
            words_per_row[b].extend(reversed(tail_per_row[b]))
        n = max(1, max(len(w) for w in words_per_row))
        out = S.new_zeros(B, n, D)                  # band tail default zero
        for b, ws in enumerate(words_per_row):
            for i, w in enumerate(ws):
                out[b, i, :d_what] = w
        # Placement (Alec 2026-07-13): the fold order IS the position —
        # emissions are earliest-first and each operand snaps to a stored
        # word row whose surface length the store knows, so sequential
        # byte offsets re-derive EXACTLY and are stamped like the forward
        # stamps them. The previously all-zero band was WHY the
        # free-derivation where_recovery read 0.0 (never written on this
        # path — not a lossy decode, not a training gap).
        if word_rows is not None:
            self._stamp_unfold_where(out, d_what, basis, word_rows)
            # §13 increment (2026-07-14 snap design doc): stash the stamped
            # slab as the RENDER-priority source — the trained reverse
            # transport collapses the multi-slot event back to the root's
            # single slot and re-stages its own thunk downstream, so the
            # render must read the slab directly (consumed once in
            # _materialize_recovered_input). Word-rows-gated: non-wordstore
            # configs keep their transported render source byte-identical.
            psp = getattr(self, 'perceptualSpace', None)
            if psp is not None:
                object.__setattr__(
                    psp, '_unfold_recovered_slab', out.detach())
        return out

    def _stamp_unfold_where(self, out, d_what, basis, word_rows):
        """Write sequential word offsets into the un-fold output's ``.where``
        band. Each emitted slot is matched to its word row (the recommender
        returns basis rows verbatim); offset = cumulative surface length + 1
        separator, encoded through the CS ``whereEncoding`` (the forward's
        stamp idiom). Slots that are not word rows (sentinels / zeros tail)
        stop the row's stamp — the band stays unwritten there, exactly the
        pre-existing below-noise-floor contract."""
        cs_sub = self.conceptualSpace.subspace
        w_enc = getattr(cs_sub, "whereEncoding", None)
        if w_enc is None or int(getattr(w_enc, "nDim", 0) or 0) <= 0:
            return
        D = int(out.shape[-1])
        idx = [int(i) for i in w_enc.resolve(D)]
        if not idx or idx[0] < d_what or idx[-1] >= D:
            return
        store = getattr(getattr(self, "perceptualSpace", None),
                        "percept_store", None)
        W = basis.getW() if hasattr(basis, "getW") else None
        if store is None or W is None:
            return
        Wr = W.index_select(0, word_rows.to(W.device)).detach()  # [K, d_what]
        B, n = int(out.shape[0]), int(out.shape[1])
        offsets = torch.full((B, n), -1.0, device=out.device)
        for b in range(B):
            cum = 0
            for i in range(n):
                v = out[b, i, :d_what].detach()
                if float(v.abs().sum()) == 0.0:
                    break                        # zeros tail: row done
                d2 = ((Wr - v.unsqueeze(0)) ** 2).sum(dim=-1)
                j = int(d2.argmin())
                if float(d2[j]) > 1e-6:
                    break                        # sentinel / non-row operand
                pid = int(word_rows[j])
                word = store.bytes_for(pid)
                if not word:
                    break
                offsets[b, i] = float(cum)
                cum += len(word) + 1             # word + separator
        mask = offsets >= 0.0
        if not bool(mask.any()):
            return
        stamp = w_enc.encode(offsets.clamp(min=0.0))
        stamp = stamp * mask.unsqueeze(-1).to(stamp.dtype)
        out[..., idx] = stamp.to(out.dtype)

    def _reverse_from_S(self, S):
        """Rework B (2): ``reverse(S)`` -- replay SymbolSpace's STORED
        forward derivations from the single non-NULL sentence idea
        ``S`` (``_stm_single_S``, ``[B, D_c]``) to reconstitute the
        per-percept surface representation.

        Drives the EXISTING reverse-trace machinery (NOT new per-op
        reverse math): stamp ``S`` as a ``[B, 1, D_c]`` event onto the
        ConceptualSpace subspace, then run the existing body/percept
        reverse chain. This is the TRAINED reverse (the D3 reconstruction
        objective) -- the LEARNED student that Method-2 refines. The
        EXACT Method-1 teacher decode lives in ``_reverse_method1_leaves``
        (serial plan Task 2): it renders the STORED per-word percept leaves
        directly through the percept store, exact by construction, and is
        what the serial EVAL decode consumes. The owner's not-yet-written
        per-op ``reverse()`` methods stay identity stubs (weak,
        non-corrupting). Returns the reconstructed ``[B, N, D]`` per-percept
        subspace materialization (or ``None``).
        """
        if S is None or not torch.is_tensor(S):
            return None
        cs = self.conceptualSpace
        if cs is None or cs.subspace is None:
            return None
        S3 = S.unsqueeze(1) if S.dim() == 2 else S         # [B, 1, D_c]
        cs.subspace.set_event(S3)
        # D4: replay SymbolSpace's STORED forward derivations. The
        # ``generate_rules`` the reverse dispatch pops from were
        # ALREADY populated by the forward (``_default_generate_rules``
        # at construction; refreshed by the compose path). We do NOT
        # re-fire ``ss.generate()`` here -- re-running the chart's
        # outside pass on the post-reduce single-S STM snapshot would
        # mis-shape the seed (the chart expects the pre-reduce percept
        # slab, not the [B,1,D_c] reduced root). The per-op
        # ``SyntacticLayer.reverse`` consumes the stored rules; the
        # owner's not-yet-written per-op reverses are identity stubs
        # (weak, non-corrupting).
        x = self._reverse_body(cs.subspace)
        x = self._reverse_perceptual(x)
        x = self.inputSpace.reverse(x)
        if x is None:
            return None
        recon = (x.materialize()
                 if hasattr(x, 'materialize') else x)
        # Asymmetric forward/reverse: forward pads the rule cursor to
        # N with S→S; reverse decodes only real rules then left-shifts
        # the per-position surface so it begins with non-NULL content
        # (doc/plans/2026-05-20-static-per-word-loop-impl.md §2R).
        # The forward's ``_word_active_mask`` is the per-position source
        # of truth — left-shift the reverse output to that layout so
        # downstream consumers (D3 reconstruction loss, reverse_map
        # eval) see real content packed at the start and zeros at the
        # tail.
        if recon is not None and torch.is_tensor(recon) and recon.dim() == 3:
            real_pos = getattr(self.inputSpace, '_word_active_mask', None)
            if (real_pos is not None
                    and real_pos.dim() == 2
                    and real_pos.shape[0] == recon.shape[0]
                    and real_pos.shape[1] == recon.shape[1]):
                recon = self._left_shift_by_mask(recon, real_pos)
        return recon

    @staticmethod
    def _left_shift_by_mask(gen, real_pos):
        # Pack positions marked True in ``real_pos`` [B, N] to the start
        # of ``gen`` [B, N, D]; positions past the real count become
        # zeros. Tensor-only (no Python loop, no ``.item()``) so it
        # stays inside the compiled region without inducing recompiles.
        B, N, D = gen.shape
        device = gen.device
        position_index = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        # Composite sort key: real positions get index in [0, N), padding
        # positions get index in [N, 2N). Stable sort preserves original
        # order within each group.
        key = torch.where(
            real_pos,
            position_index,
            position_index + N)
        sorted_indices = torch.argsort(key, dim=1, stable=True)
        out = torch.gather(
            gen, 1,
            sorted_indices.unsqueeze(-1).expand(-1, -1, D))
        n_real = real_pos.long().sum(dim=1, keepdim=True)
        dest_pos = position_index
        keep = (dest_pos < n_real).unsqueeze(-1).to(out.dtype)
        return out * keep

    def _d3_reconstruction_loss(self):
        """Rework B (3): the D3 reconstruction loss -- the trainable
        per-word IR objective that REPLACES the interim P-space_role
        ``compute_masked`` masked-LM on the per-word grammar path.

        Reconstruction = the per-percept predictions from
        ``reverse(S)`` (driven from ``_stm_single_S``), mapped back
        through the Rework-A MPHF->table, compared vs the COMPLETE
        UNMASKED input sentence (``inputSpace._ar_embedded`` -- every
        word, the full pre-mask muxed slab the per-word loop never
        touched).

        Returns ``(loss, metric)``:
          * ``loss`` -- the CONTINUOUS percept/concept-vector
            reconstruction (differentiable, MSE-style via the band-aware
            ``_reverse_event_loss`` seam); this is the trainable signal that
            goes into the ``reconstruction`` error slot, replacing
            ``compute_masked`` for this path. Per-word gaussian context
            washes out across this whole-sentence comparison.
          * ``metric`` -- the word-level 0/1 input distance (the
            fraction of percept positions whose nearest table row
            (``reverse_map_concept``) differs from the forward percept
            row). REPORTED only -- NON-differentiable; never
            backpropped (the lexer-discrete word-level distance cannot
            be finer-grained than 0/1 and is not the gradient source).

        Returns ``(None, None)`` when the per-word S / unmasked target
        is unavailable (caller falls back to the existing path).
        """
        S = getattr(self, '_stm_single_S', None)
        if S is None:
            return None, None
        # COMPLETE UNMASKED target: every word, full pre-mask slab.
        target = getattr(self.inputSpace, '_ar_embedded', None)
        if target is None or not torch.is_tensor(target) or target.dim() != 3:
            return None, None
        recon = self._reverse_from_S(S)
        if recon is None or not torch.is_tensor(recon) or recon.dim() != 3:
            return None, None
        # Trainable signal: continuous reconstruction vs the complete
        # unmasked input, through the band-aware ``_reverse_event_loss``
        # seam (input-event layout; detaches the target so backward
        # flows through the prediction -- the per-word encoder / S /
        # the bounded reduce -- not the fixed input).
        Kr = min(recon.shape[1], target.shape[1])
        Dr = min(recon.shape[-1], target.shape[-1])
        loss = self._reverse_event_loss(recon, target)

        # REPORTED-ONLY metric (NON-differentiable; never backpropped --
        # the lexer-discrete word-level distance is not the gradient
        # source). PREFER the word-level 0/1 input distance via the
        # Rework-A table (forward ``surface[idx]`` where the percept
        # index is KNOWN -- the reverse-map concern -- over the
        # nearest-vector map). The reverse(S) trace currently runs the
        # owner's identity-stub per-op reverses, so its output width can
        # differ from the codebook width; when the table-index 0/1 is
        # not materializable at compatible widths, fall back to a
        # reported continuous WHAT-distance proxy (still NEVER
        # backpropped) rather than silently dropping the report. Fully
        # wrapped: a metric edge case must never perturb the step.
        metric = None
        try:
            ps = self.perceptualSpace
            cb = ps._mphf_codebook() if ps is not None else None
            with torch.no_grad():
                cb_w = (cb.getW().shape[-1]
                        if cb is not None else None)
                # Word-level 0/1 (preferred) -- only when the recon
                # width matches the codebook so the nearest-row lookup
                # is meaningful.
                if (cb is not None and cb_w is not None
                        and recon.shape[-1] == cb_w):
                    pred_idx = ps.reverse_map_concept(
                        recon[:, :Kr, :], return_surface=False)
                    # Forward percept rows where known: the per-word
                    # frozen lexicon row on PartSpace
                    # ``_index[:, :, 0]`` IS the MPHF index (preferred
                    # over a nearest-vector remap of the target).
                    sub = getattr(ps, 'subspace', None)
                    active = (getattr(sub, '_index', None)
                              if sub is not None else None)
                    if (active is not None and active.dim() == 3
                            and active.shape[-1] >= 1):
                        tgt_idx = active[:, :Kr, 0].long()
                    else:
                        tgt_idx = ps.reverse_map_concept(
                            target[:, :Kr, :cb_w], return_surface=False)
                    Km = min(pred_idx.shape[-1], tgt_idx.shape[-1])
                    metric = (pred_idx[..., :Km]
                              != tgt_idx[..., :Km]).float().mean()
                else:
                    # Reported continuous WHAT-distance proxy (the
                    # identity-stub reverse yields a non-codebook-width
                    # output; the discrete 0/1 needs the owner's
                    # per-op reverses filled). NEVER backpropped.
                    metric = F.mse_loss(
                        recon[:, :Kr, :Dr],
                        target[:, :Kr, :Dr]).detach()
        except Exception:
            metric = None
        return loss, metric







    def _forward_per_stage(self, inputData, in_sub_override=None):
        """IR-only forward via stem/body/head pipeline.

        Shape:
          stem -> ``[B, N, D]`` (InputSpace.forward + PartSpace
                  .forward; no K-axis).
          body -> per-stage CS/WS chain on B-shaped tensors; each
                  WholeSpace output lives on that stage's
                  ``.subspace`` (the per-stage ``_ws_cache`` /
                  ``_cs_cache`` capture lists were retired by Stage
                  1.F of doc/plans/2026-05-26-two-loop-pi-sigma-
                  substrate.md — terminal STM owns the C-space_role idea
                  and ``self.wholeSpaces[*]`` own the per-stage
                  symbolic state).
          head -> ``outputSpace``; result event ``[B, N, predDim]``.

        Within-sentence training is IR-only (BERT-style masked-LM at
        the P-space_role).  Sentence-level AR moved to ``InterSentenceLayer``
        (ARMA(p, q) over sentence reps).  ``<maskedPrediction>`` and
        the per-cursor K-loop retired 2026-05-14; legacy reverse
        pipeline retired in the same change.
        """
        if isinstance(inputData, torch.Tensor):
            # Device placement is the eager producer's job on the
            # compiled path: runBatch moves inputTensor onto the compute
            # device *before* the traced step (same principle as the
            # Phase-3 lex+embed stem). Calling ``TheDevice.get()``
            # *inside* the trace hands dynamo a ``DeviceHandle`` it
            # cannot proxy ("Failed to convert args/kwargs to proxy")
            # -> hard error under ``fullgraph=True``. Skip it when staged
            # (compiled path; producer already placed it); keep it for
            # eager / uncompiled callers (tests, MPS) where no producer
            # ran -- behaviour there is unchanged.
            if self._staged_in_sub is None:
                inputData = inputData.to(TheDevice.get())
        self._ar_valid_pos = None  # IR has no per-cursor axis.

        # Detach per-batch event content carried over from the prior
        # forward to break the autograd graph. Post-Stage-4 of
        # doc/plans/2026-05-21-active-payload-retirement.md the
        # per-batch event lives:
        #   * on ``event.W`` for pure-event (plain Tensor) slots; can
        #     be detached in place via setW.
        #   * reconstructed from prototype + selection for muxed
        #     codebook slots — the Parameter handles grad-tracking
        #     directly; no cached event to detach.
        # Skip the detach for codebook-bearing slots (``getW`` returns
        # the 2-D Parameter, which detach would clobber).
        for sp in (self.perceptualSpace, self.conceptualSpace,
                   self.wholeSpace):
            if sp is None:
                continue
            sub = sp.subspace
            if sub is None or sub.event is None:
                continue
            if getattr(sub, "muxed", False):
                # Per-batch event reconstructs from prototype +
                # selection; nothing to detach on ``event`` itself.
                continue
            w = sub.event.getW()
            # Host housekeeping only: under export the traced detach of a
            # stem-parked host event lifts a DEAD constant (StopIteration
            # in torch.export's lift_constants_pass), so skip it there.
            if (w is not None and torch.is_tensor(w)
                    and not isinstance(w, nn.Parameter)
                    and w.ndim >= 3 and w.requires_grad
                    and not torch.compiler.is_exporting()):
                sub.event.setW(w.detach())

        # Per-run scratch.
        self.symbol_states = []
        self._unified_j_iterations = 0

        # Task D1: with an export-driven ``in_sub_override`` there is no
        # ``inputData`` tensor (it is None); derive B/device from the
        # override's materialized staged slab so the symbolic seed batches
        # correctly. Otherwise read them off ``inputData`` (unchanged).
        if isinstance(inputData, torch.Tensor):
            B = inputData.shape[0]
            device = inputData.device
        elif in_sub_override is not None:
            _ov_ev = in_sub_override.materialize()
            if _ov_ev is not None and torch.is_tensor(_ov_ev):
                B = int(_ov_ev.shape[0])
                device = _ov_ev.device
            else:
                B = 1
                device = TheDevice.get()
        else:
            # Complete MPS W-loop callables deliberately receive no raw
            # lexer tensor: that surface has already been staged eagerly and
            # its varying byte extent must not become a Dynamo guard.  Recover
            # the true batch/device from the staged carrier instead of
            # assuming B=1, so the boundary remains correct for configured
            # microbatches other than the current default.
            _staged_ev = (
                self._staged_in_sub.materialize()
                if self._staged_in_sub is not None else None)
            if torch.is_tensor(_staged_ev):
                B = int(_staged_ev.shape[0])
                device = _staged_ev.device
            else:
                B = 1
                device = TheDevice.get()
        self.symbolic_state = self.wholeSpace.empty_state(batch=B).to(device)

        # Inter-sentence prior (read-only at forward time; the actual
        # ARMA observe + loss happen in ``runBatch`` after the body).
        self._predicted_snapshot = None
        self._predicted_confidence = None
        discourse_for_prime = (
            self.symbolSpace.discourse
            if self.symbolSpace is not None else None)
        if discourse_for_prime is not None:
            d_pred, d_conf = discourse_for_prime.predict()
            self._predicted_snapshot = d_pred
            self._predicted_confidence = d_conf

        # Stem: InputSpace forward only (``[B, N, D]``). PartSpace
        # now runs inside the recurrent cell (it takes the C→P feedback
        # as an explicit arg), so it is no longer a once-per-sentence
        # stem step.
        #
        # Phase 3: when runBatch staged the lex+embed eagerly (compiled
        # path), read the pre-populated persistent subspace instead of
        # lexing here -- keeps host tokenisation/OOV out of the traced
        # region (graph breaks #1-4). Read-only (no attr write in the
        # traced path). Eager/uncompiled callers don't stage, so this
        # falls back to inline lexing -- behaviour unchanged.
        #
        # Task D1 (torch.export tensor core): ``forward_core`` drives the
        # body from a SubSpace whose materialized event IS the export
        # ARGUMENT (not ``self._staged_in_sub``). When ``in_sub_override``
        # is supplied we read it instead, so the traced tensor pipeline
        # roots in the passed-in staged tensor -- the normal forward keeps
        # reading the parked ``self._staged_in_sub`` (no regression).
        if in_sub_override is not None:
            in_sub = in_sub_override
        else:
            in_sub = self._staged_in_sub
        if in_sub is None:
            in_sub = self._lex_embed_stem(inputData)
        if in_sub is None or in_sub.is_empty():
            # Phase 1.5: the five inputs/percepts/concepts/symbols/outputs
            # back-ref aliases were subsumed -- every reader now goes
            # through the owning Space directly
            # (``self.perceptualSpace.subspace`` etc.), so there is no
            # alias to (re)stamp here. Empty-input early return unchanged.
            return None, None, None, None

        # Body: recurrent cell over T stages on B rows. PartSpace
        # runs per pass with the C→P feedback; the IR masked-LM mask is
        # applied once inside the cell on pass 0's perceptual event
        # (``_ir_pre_mask_input`` / ``_ir_mask_positions`` for
        # ``runBatch``). Chart-at-C reads ``ConceptualSpace.stm``.
        body_sub = self._forward_body(in_sub)

        # Global-attention CONSUMER (doc/specs/reading-attention.md "(B)"; Alec):
        # feed the parked soft-read back into the HEAD input as a zero-init gated
        # residual, so the answer/output loss trains the retrieval (gradient
        # through the read -> alpha -> the scorer). Gated <globalAttentionConsume>.
        # The head reads ``body_sub`` -- which the REVERSE also reads (it is
        # ``_combine_last_cs_sub``) -- so apply the consume on a SWAPPED event
        # only for the head, then restore the original so reconstruction is
        # untouched. Off / no parked read -> body_sub unchanged (byte-identical).
        _restore_ev = None
        if (getattr(self, "global_attention_consume", False)
                and getattr(self, "global_attention", None) is not None
                and body_sub is not None and hasattr(body_sub, "materialize")):
            obs = getattr(self, "_global_attention_obs", None)
            ev = body_sub.materialize()
            if (obs is not None and obs.get("content") is not None
                    and ev is not None and torch.is_tensor(ev)):
                consumed = self.global_attention.consume(ev, obs["content"])
                if consumed is not ev:
                    # ``consume`` CLONES (does not mutate ``ev``), so the
                    # ORIGINAL muxed event is ``ev`` itself -- save it and
                    # restore via ``set_event`` (the event lives on
                    # ``body_sub.event.W``, not a ``_event`` attribute, so a
                    # getattr-based snapshot would be dead code and leave the
                    # consumed event for the reverse to read).
                    _restore_ev = ev
                    body_sub.set_event(consumed)

        # Head: outputSpace -> ``[B, N, predDim]``. The restore is in a finally
        # so the pre-consume carrier is ALWAYS put back for the reverse /
        # reconstruction, even if the head raises.
        try:
            head_sub = self._forward_head(body_sub)
        finally:
            if _restore_ev is not None:
                body_sub.set_event(_restore_ev)
        pred = head_sub.materialize() if head_sub is not None else None
        if pred is not None:
            pred = self.normalizer.denormalize(pred, which="output")

        # Capture symbol_states by iterating per-stage WholeSpaces
        # directly. Each stage's ``ws.forward(...)`` in
        # ``_forward_body`` writes its result onto that stage's
        # ``.subspace``; reading per-stage straight off
        # ``self.wholeSpaces`` is the canonical replacement for the
        # retired ``self._ws_cache`` per-forward capture list (Stage
        # 1.F of doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md).
        # Per-word path only fires the terminal stage's ws.forward, so
        # match the legacy ``_ws_cache`` populating pattern by walking
        # only the terminal stage there — preserves the legacy
        # symbol_states cardinality (1 entry in per-word path; T
        # entries in the whole-slab path).
        isp = getattr(self, 'inputSpace', None)
        per_word_path = (isp is not None
                         and getattr(isp, '_per_word_enabled', False))
        if per_word_path:
            stages_iter = (
                [self.wholeSpaces[-1]] if self.wholeSpaces
                else [])
        else:
            stages_iter = list(self.wholeSpaces)
        captured_states = []
        for ws in stages_iter:
            sub = getattr(ws, 'subspace', None)
            if sub is None:
                continue
            sv = sub.materialize()
            if sv is None:
                continue
            captured_states.append(sv.clone())
        self.symbol_states = captured_states
        self._unified_j_iterations = min(
            self.subsymbolicOrder, len(captured_states))

        # Phase 1.5: the five inputs/percepts/concepts/symbols/outputs
        # back-ref aliases (formerly stamped here via object.__setattr__)
        # were subsumed -- they were read-only handles to each Space's
        # terminal subspace, not cross-round state. Every reader now goes
        # through the owning Space directly (``self.perceptualSpace
        # .subspace`` etc.), so nothing is stamped on ``self`` here and
        # ``self._modules`` stays constant in the traced forward without
        # the band-aid.

        # forwardInput contract for runBatch: ``[B, N, D]`` embedded input.
        input_state = self.inputSpace._ar_embedded
        if input_state is None:
            input_state = self.inputSpace.subspace.materialize()
        if input_state is None:
            input_state = self.perceptualSpace._embedded_input

        # Symbol vectors at last stage.
        sym_sub = self.symbol_cache
        if sym_sub is not None:
            sym_vectors = sym_sub.materialize()
        else:
            sym_vectors = self.wholeSpace.subspace.materialize()

        # Inter-sentence snapshot: pass the S-space_role event to runBatch,
        # which calls ``InterSentenceLayer.observe`` to capture the
        # sentence rep, compute the ARMA loss, and update the rings.
        # ``_current_discourse_s`` carries the unmodified S-space_role event;
        # the layer's ``_pool_sentence_rep`` does the flattening.
        self._current_discourse_s = (
            sym_vectors.detach() if sym_vectors is not None else None)

        self._universality_score = None

        # Downward head emission (S -> C).
        self._predicted_head = None
        try:
            gen_on = bool(TheXMLConfig.get('SymbolSpace.downwardGeneration'))
        except KeyError:
            gen_on = False
        if (gen_on and self.symbolSpace is not None
                and sym_vectors is not None and sym_vectors.ndim >= 3):
            final_state = sym_vectors[:, 0, :]
            codebook_space = (self.perceptualSpace
                              if self.inputSpace.model_type == "embedding"
                              else self.inputSpace)
            head_result = self.symbolSpace.reconstruct(final_state, codebook_space)
            self._predicted_head = head_result['heads']

        # Optional syntax tree dump.
        if self._write_syntax:
            try:
                self.write_syntax_tree(self._syntax_out_path)
            except Exception as e:
                import sys
                print(f"[writeSyntax] error: {e}", file=sys.stderr)

        return input_state, sym_vectors, pred, None




    def write_syntax_tree(self, path):
        """Append an XML syntax tree per batch row to ``path``.

        Reads the chart's derivation trace
        (``Chart._derivation_trace``, a list-per-batch of
        ``(gid, i, k, j, A)`` tuples — rule global id, left edge, split
        point, right edge, merged-cell category index) and the
        symbolic codebook's per-atom category tags
        (``WholeSpace.subspace.what.category_ids``).  Decodes input
        atoms to surface tokens via ``InputSpace.wv.index_to_key``.

        Format (one ``<forward>`` element per call, with a ``<batch>``
        per row).  Category names (``cat=...``), rule canonicals
        (``rule=...``), and POS tags (``pos=...``) come straight from
        the live grammar's ``Grammar.categories`` /
        ``RuleDef.canonical``; the dumper has no built-in category
        vocabulary.  Example output for an XOR_grammar parse where the
        only non-terminal is ``S``::

            <forward tick="42">
              <batch n="0">
                <node cat="S" rule="S=conjunction(S,S)" i="0" j="2">
                  <leaf token="1" pos="S" i="0"/>
                  <leaf token="1" pos="S" i="1"/>
                </node>
                <rules>1</rules>
              </batch>
            </forward>

        A grammar declaring richer typed categories (e.g.
        ``NP = intersection(ADJ, N)``) produces trees with those
        category names instead.

        The output file is a stream of ``<forward>`` fragments — one
        per call.  Wrap with a ``<syntaxLog>`` root element (or use a
        streaming parser) if a single fully-valid XML document is
        needed.

        Side effects: appends to ``path`` (creating parent dirs as
        needed). The first call within a process truncates ``path``
        so old content from prior runs is cleared.
        """
        import os
        import xml.etree.ElementTree as ET
        symbolSpace = self.symbolSpace
        if symbolSpace is None:
            return
        # Chart's per-leaf POS category names and per-leaf atom_idx
        # aren't populated by the signal router yet; until they are,
        # this branch emits ``<noTrace/>``.
        chart = None
        traces = None
        cat_names = ['?']

        # Resolve atom_idx -> surface token via InputSpace.wv (word
        # vectors); fall back to a placeholder when wv isn't wired.
        # `pos` here is a position within the parsed sentence (0..N-1),
        # NOT a codebook index — surface-decoding requires looking up
        # the actual codebook row of the atom at that position.
        index_to_key = None
        try:
            wv = getattr(self.inputSpace, 'wv', None)
            if wv is not None:
                index_to_key = getattr(wv, 'index_to_key', None)
        except Exception:
            index_to_key = None

        # Per-leaf atom indices: prefer the chart's stashed nearest-match
        # row from `_apply_codebook_pos_seed`. Shape [B*N], may be None.
        # Reshape to [B, N] so we can look up by (batch, position).
        per_leaf_atom = None
        try:
            stash = getattr(chart, '_last_seed_atom_idx', None)
            if stash is not None and traces is not None:
                B_t = len(traces)
                if B_t > 0 and stash.numel() % B_t == 0:
                    N_t = stash.numel() // B_t
                    per_leaf_atom = stash.reshape(B_t, N_t).cpu()
        except Exception:
            per_leaf_atom = None

        # Direct surface-form recovery: re-tokenize the last batch's
        # input strings (stored on the InputSpace as `_last_input_texts`
        # if available, otherwise pulled from TheData splits when the
        # input subspace exposes its byte buffer). Falls through to the
        # atom-index path when no source text is reachable.
        per_leaf_text = None
        try:
            sym_sub = self.wholeSpace.subspace
            in_sub = self.inputSpace.subspace
            # Spec doc/specs/2026-05-21-subspace-slot-architecture.md
            # Reader API: ``materialize(mode='what')`` is the per-batch
            # what-content read. InputSpace's ``.what`` is a plain
            # ``Tensor`` (no Parameter), so this returns the same
            # ``[B, N, nWhat]`` slab as the legacy ``what.getW()``;
            # the migrated call is contract-stable AND survives the
            # Stage-4 strict assertion on Parameter-bearing slots.
            wbuf = (in_sub.materialize(mode="what")
                    if in_sub is not None
                    and hasattr(in_sub, "materialize") else None)
            if wbuf is not None and torch.is_tensor(wbuf) and wbuf.dim() == 3:
                B_t = wbuf.shape[0]
                from util import parse as _parse
                texts_per_batch = []
                for bi in range(B_t):
                    # what_buf row is [N, nWhat] of UTF-8 bytes
                    # (null-terminated). Decode each token slot.
                    row = wbuf[bi]
                    toks = []
                    for sj in range(row.shape[0]):
                        bs = row[sj].to(torch.uint8).tolist()
                        # Strip trailing nulls; everything else is the token.
                        while bs and bs[-1] == 0:
                            bs.pop()
                        if bs:
                            try:
                                toks.append(bytes(bs).decode('utf-8', errors='replace'))
                            except Exception:
                                toks.append(f"slot_{sj}")
                        else:
                            toks.append("")
                    texts_per_batch.append(toks)
                per_leaf_text = texts_per_batch
        except Exception:
            per_leaf_text = None

        def _decode_token(b, pos):
            """Return the surface text for batch row ``b`` at slot ``pos``.

            Prefers the decoded UTF-8 text from ``per_leaf_text``, falls
            back to the codebook key via ``per_leaf_atom`` / index_to_key,
            and finally to a ``slot_<pos>`` placeholder.
            """
            if (per_leaf_text is not None and b < len(per_leaf_text)
                    and pos < len(per_leaf_text[b])
                    and per_leaf_text[b][pos]):
                return per_leaf_text[b][pos]
            atom_idx = None
            if per_leaf_atom is not None and b < per_leaf_atom.shape[0] \
                    and pos < per_leaf_atom.shape[1]:
                atom_idx = int(per_leaf_atom[b, pos].item())
            if (index_to_key is not None and atom_idx is not None
                    and 0 <= atom_idx < len(index_to_key)):
                return str(index_to_key[atom_idx])
            if atom_idx is not None:
                return f"atom_{atom_idx}"
            return f"slot_{pos}"

        # POS lookup: prefer the chart's per-leaf POS distribution
        # (chart._chart_pos[b, i, i+1, :] argmax) over the stale
        # codebook category_ids buffer. Fall back to category_ids when
        # chart_pos isn't available.
        chart_pos = None
        try:
            chart_pos = getattr(chart, '_chart_pos', None)
        except Exception:
            chart_pos = None

        cat_ids = None
        try:
            sym_sub = self.wholeSpace.subspace
            what = getattr(sym_sub, 'what', None)
            cat_ids = getattr(what, 'category_ids', None) if what else None
        except Exception:
            cat_ids = None

        def _pos_at(b, pos):
            """Return the POS tag name for batch row ``b`` at slot ``pos``.

            Prefers the chart's per-leaf POS distribution
            (``chart_pos[b, pos, pos+1, :].argmax``); falls back to the
            possibly-stale codebook ``category_ids`` and finally ``'?'``.
            """
            # Prefer chart_pos[b, pos, pos+1, :].argmax() — what the
            # chart actually believed at this leaf during this parse.
            if chart_pos is not None:
                try:
                    if (b < chart_pos.shape[0]
                            and pos + 1 < chart_pos.shape[1]):
                        cid = int(chart_pos[b, pos, pos + 1, :].argmax().item())
                        if 0 <= cid < len(cat_names):
                            return cat_names[cid]
                except Exception:
                    pass
            # Fallback: stale codebook tag.
            if cat_ids is not None and 0 <= pos < cat_ids.shape[0]:
                cid = int(cat_ids[pos].item())
                if 0 <= cid < len(cat_names):
                    return cat_names[cid]
            return '?'

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        mode = 'w' if not getattr(self, '_syntax_truncated', False) else 'a'
        self._syntax_truncated = True

        try:
            from Language import TheGrammar
        except ImportError:
            TheGrammar = None

        def _entry_str(entry):
            gid, i, k, j, A = entry
            try:
                canon = TheGrammar.rules[gid].canonical if TheGrammar else str(gid)
            except Exception:
                canon = str(gid)
            cat = (cat_names[int(A)] if 0 <= int(A) < len(cat_names)
                   else '?')
            return cat, canon

        def _rule_canon(rule_id):
            try:
                if TheGrammar:
                    return TheGrammar.rules[int(rule_id)].canonical
                return str(rule_id)
            except Exception:
                return str(rule_id)

        forward_el = ET.Element(
            'forward',
            {'tick': str(getattr(self, '_current_tick', '?'))})
        if traces is None:
            ET.SubElement(forward_el, 'noTrace')
        else:
            for b, trace in enumerate(traces):
                batch_el = ET.SubElement(
                    forward_el, 'batch', {'n': str(b)})
                if not trace:
                    ET.SubElement(batch_el, 'noTrace')
                    continue
                span_to_entry = {}
                for entry in trace:
                    _, i_e, _, j_e, _ = entry
                    span_to_entry.setdefault((i_e, j_e), entry)

                def _build(span, parent_el, _b=b):
                    """Recursively emit XML nodes/leaves for chart ``span``.

                    Looks up the chart entry at ``(si, sj)``; emits a
                    ``<node>`` element with category + rule then recurses
                    into the left/right children. Spans without an entry
                    become a run of ``<leaf>`` elements.
                    """
                    entry = span_to_entry.get(span)
                    si, sj = span
                    if entry is None:
                        for pos in range(si, sj):
                            ET.SubElement(parent_el, 'leaf', {
                                'token': _decode_token(_b, pos),
                                'pos': _pos_at(_b, pos),
                                'i': str(pos),
                            })
                        return
                    cat, canon = _entry_str(entry)
                    _, _, k_e, _, _ = entry
                    node_el = ET.SubElement(parent_el, 'node', {
                        'cat': cat,
                        'rule': canon,
                        'i': str(si),
                        'j': str(sj),
                    })
                    _build((si, k_e), node_el)
                    _build((k_e, sj), node_el)

                root = trace[0]
                _, root_i, _, root_j, _ = root
                _build((root_i, root_j), batch_el)
                rules_el = ET.SubElement(batch_el, 'rules')
                rules_el.text = ",".join(str(int(e[0])) for e in trace)

        # Pretty-print and write the fragment.
        try:
            ET.indent(forward_el, space="  ")
        except AttributeError:
            # ET.indent landed in Python 3.9; older versions get
            # the unindented form.
            pass
        with open(path, mode, encoding='utf-8') as fh:
            fh.write(ET.tostring(forward_el, encoding='unicode'))
            fh.write("\n")


    # -- Intent priming (GrammarOpsPass §5; author 2026-06-11) ---------

    def set_intent(self, intent, *, gain=1.0):
        """ONE current-intent code priming BOTH codebook towers
        simultaneously: primed RECOGNITION on the PS/extent tower
        alongside primed RETRIEVAL on the WS/intent tower, weighting
        the analytical and synthetic superpositions toward the same
        context. Priming is attention over symbols — codebook focus
        only, never rule dispatch.

        The intent is the product of the parallel parse (the §6c
        pump-zero gist); it is sentence-scoped and refreshed on
        preemption. ``intent=None`` clears the tower boosts (the
        byte-identical off state); the taxonomy buffer's merged boost
        dissipates through its own sentence-scoped decay/reset.

        Returns the WS-tower boosts (or ``None``).
        """
        ps = getattr(self, 'perceptualSpace', None)
        ws = getattr(self, 'wholeSpace', None)
        if ps is not None and hasattr(ps, 'set_intent'):
            ps.set_intent(intent, gain=gain)
        ws_boosts = None
        if ws is not None and hasattr(ws, 'set_intent'):
            ws_boosts = ws.set_intent(intent, gain=gain)
        # WS retrieval plumbing: merge into the word-space taxonomy
        # priming buffer (consumed by the inverse recommender through
        # priming_kwargs_for_slots). Best-effort and dark: no taxonomy
        # or no buffer => recognition priming alone.
        if ws_boosts is not None:
            tax = getattr(getattr(self, 'symbolSpace', None),
                          'taxonomy', None)
            if tax is not None and hasattr(tax, 'prime_with_weights'):
                try:
                    tax.prime_with_weights(ws_boosts)
                except Exception:
                    pass
        return ws_boosts

    # -- Bidirectional Reasoning Loop (Phase 3) ------------------------

    @torch.no_grad()

    def reason(self, givens, target=None, direction='forward', max_steps=8):
        """Bidirectional reasoning loop.

        Forward (givens -> conclusion):
            Encode givens, extrapolate new truths, check isTrue(target)
            at each step. Stop when target DoT exceeds threshold or
            max_steps reached.

        Reverse (target -> grounding):
            Encode target, ground() to find minimal basis, extrapolate
            if insufficient.

        Args:
            givens: tensor or list of tensors to encode as premises.
            target: optional target activation to prove/ground.
            direction: 'forward' or 'reverse'.
            max_steps: maximum reasoning steps.

        Returns:
            dict with 'proved' (bool), 'confidence' (float),
            'steps' (int), 'trace' (list of step results).
        """
        truth_layer = self._get_truth_layer()
        if truth_layer is None:
            return {'proved': False, 'confidence': 0.0,
                    'steps': 0, 'trace': []}

        trace = []

        if direction == 'forward':
            basis = self._get_basis()
            # Encode givens into TruthSet
            if isinstance(givens, (list, tuple)):
                for g in givens:
                    if g.norm() > 1e-6:
                        truth_layer.record(g.detach(), degree=1.0, basis=basis)
            elif givens.norm() > 1e-6:
                truth_layer.record(givens.detach(), degree=1.0, basis=basis)

            for step in range(max_steps):
                lum_before = float(self.Luminosity(truth_layer=truth_layer))

                # Extrapolate new truths
                result = self.extrapolate(max_new=16,
                                           attenuation=0.8 ** (step + 1))
                trace.append({
                    'step': step,
                    'added': len(result['added']),
                    'rejected': len(result['rejected']),
                    'luminosity': lum_before,
                })

                # Check target if provided
                if target is not None:
                    dot = self.isTrue(target)
                    trace[-1]['target_dot'] = dot
                    if abs(dot) > 0.5:
                        return {'proved': True, 'confidence': dot,
                                'steps': step + 1, 'trace': trace}

                # Stop if no progress
                if len(result['added']) == 0:
                    break

            confidence = self.isTrue(target) if target is not None else 0.0
            return {'proved': abs(confidence) > 0.5,
                    'confidence': confidence,
                    'steps': len(trace), 'trace': trace}

        else:  # reverse
            if target is None:
                return {'proved': False, 'confidence': 0.0,
                        'steps': 0, 'trace': []}

            for step in range(max_steps):
                result = self.ground(target)
                trace.append({
                    'step': step,
                    'grounded': result['grounded'],
                    'basis_size': len(result['basis']),
                    'confidence': result['confidence'],
                })

                if result['grounded']:
                    return {'proved': True,
                            'confidence': result['confidence'],
                            'steps': step + 1, 'trace': trace}

                # Extrapolate and retry
                ext = self.extrapolate(max_new=16)
                if len(ext['added']) == 0:
                    break

            return {'proved': False, 'confidence': 0.0,
                    'steps': len(trace), 'trace': trace}

TheBasicModel = BasicModel()

class ModelFactory:
    """Create, train, and evaluate models from an XML config file.

    Dispatches to the right model class based on <data><dataType>:
      - dataType=embedding -> BasicModel (embedding/language model path)
      - dataType=numeric   -> BasicModel (dense-slab path; e.g. MNIST)
      - Otherwise           -> SimpleModel parameterized by:
            ergodic, certainty, codebook, normed, reverse, invert
    """

    @staticmethod
    def model_name(ergodic, certainty, codebook, normed=False, reverse=False, invert=False):
        """Generate a human-readable model name from its flags.

        Returns ``"SimpleModel"`` when no flags are set; otherwise
        concatenates the matching flag names with ``" + "``. Reverse /
        invert are mutually exclusive in the rendered name (invert wins).
        """
        if not ergodic and not certainty and not codebook:
            return "SimpleModel"
        parts = []
        if ergodic:
            parts.append("Ergodic")
        if certainty:
            parts.append("Certainty")
        if codebook:
            parts.append("Codebook")
        if normed:
            parts.append("Normed")
        if invert:
            parts.append("Invertible")
        elif reverse:
            parts.append("Reversible")
        return " + ".join(parts) if parts else "SimpleModel"

    @staticmethod
    def get_space_param(cfg, space_name, key):
        """Look up key in space section, fall back to architecture section.

        Resolution order: cfg[space_name][key] -> cfg["architecture"][key]
        All parameters must be in model.xml; raises KeyError if missing.
        """
        space = cfg.get(space_name, {})
        if key in space:
            return space[key]
        arch = cfg.get("architecture", {})
        if key in arch:
            return arch[key]
        raise KeyError(f"Required parameter '{key}' not found in <{space_name}> or <architecture>")

    @staticmethod
    def validate_config(cfg):
        """Check merged config for known inconsistencies and raise on error.

        Called after defaults have been merged so all keys are present.
        Uses get_space_param() to read from space-scoped sections.
        """
        gsp = ModelFactory.get_space_param
        arch = cfg.get("architecture", {})
        errors = []
        # Clear any requirements left over from a previous validate_config
        # call (e.g. one that bailed via ``errors`` before reaching
        # ``TheXMLConfig.validate()``). Otherwise stale ``require()``
        # closures from one test leak into the next.
        TheXMLConfig._requirements.clear()

        # The old ``hasAttention``-vs-``nInputDim``/``flatten`` reshape guard
        # (and its ``_has_reshape`` helper) was retired together with the
        # ``QKVAttentionLayer`` enlistment in PartSpace/ConceptualSpace
        # (plan 2026-06-06-symbolic-heat-retrieval.md §Handoff addendum). It
        # only protected the constraint that transformer self-attention needs
        # 3D multi-vector input; ``<attention>`` is now a symbolic-retrieval
        # mode (not tensor self-attention), so ``hasAttention=True`` together
        # with a reshape/flatten is no longer an error.

        # ``<maskedPrediction>`` retired 2026-05-14: within-sentence
        # training is IR-only and there's no ARIR mode to validate.
        # ``<reconstruct>`` enum retired (A1: schema + ``reconstructEnum``
        # removed). C3 (spec sec 7): reconstruction is unconditionally
        # concepts-seeded in ``runBatch`` -- there is no mode to validate.

        def _resolve_dim(space_name, prev_dim):
            try:
                raw = gsp(cfg, space_name, "nDim")
            except KeyError:
                return prev_dim
            return prev_dim if raw == 0 else raw

        def _resolve_count(space_name, prev_count):
            try:
                raw = gsp(cfg, space_name, "nOutput")
            except KeyError:
                return prev_count
            return prev_count if raw == 0 else raw

        # Aligned binding uses ConceptualSpace's concept-reference inventory.
        # WholeSpace.nVectors is an independent upstream property inventory;
        # live CS/WS slot counts are separate geometry as well.
        if str(arch.get("conceptBinding", "mixing")).strip().lower() == "aligned":
            try:
                _cs_codes = int(gsp(cfg, "ConceptualSpace", "nVectors"))
                try:
                    _cs_active = int(gsp(
                        cfg, "ConceptualSpace", "activeVectors"))
                except KeyError:
                    _cs_active = _cs_codes
                if not (0 < _cs_active <= _cs_codes):
                    errors.append(
                        "ConceptualSpace.activeVectors must be positive and "
                        f"no larger than its nVectors; got active={_cs_active}, "
                        f"capacity={_cs_codes}")
            except (KeyError, TypeError, ValueError):
                errors.append(
                    "aligned binding requires explicit positive nVectors in "
                    "ConceptualSpace")

        from architecture import canonical_shape as _cshape
        input_dim = _resolve_dim("InputSpace", 1)
        percept_dim = _resolve_dim("PartSpace", input_dim)
        concept_dim = _resolve_dim("ConceptualSpace", percept_dim)
        # 2026-06-06 uniform-band convention: every space_role carries the same
        # (nWhere, nWhen) band, so nDim = nWhat + band uniformly. An omitted WS
        # width still inherits for legacy configs; canonical serialObjectMeta
        # sets the native WS property width explicitly because WS is a peer,
        # not a downstream copy of CS.
        symbol_dim = _resolve_dim("WholeSpace", concept_dim)

        # Bivector / projection configs let ConceptualSpace output a
        # narrower activation via ``<nOutputDim>``. Compare nWhat-to-nWhat
        # across CS and WS (subtract each space_role's own band).
        try:
            cs_out_dim = int(gsp(cfg, "ConceptualSpace", "nOutputDim"))
        except KeyError:
            cs_out_dim = 0
        _cs_band = sum(_cshape("ConceptualSpace"))
        _ws_band = sum(_cshape("WholeSpace"))
        _cs_event = cs_out_dim if cs_out_dim > 0 else concept_dim
        effective_concept_dim = _cs_event - _cs_band          # CS nWhat
        symbol_nwhat = symbol_dim - _ws_band                  # WS nWhat

        # WholeSpace owns no legacy C->S width-changing bridge. Parallel legacy
        # configurations that treat WS as a pass-through still require equal
        # widths; the serial peer architecture below does not.
        #
        # Skip only the explicit ``passthrough`` bypass mode. The CS->WS
        # dimensional pass-through (WS owns no Sigma) is real for both the
        # simple and embedding pipelines -- ``test_symbol_dim_must_match_
        # concept_dim`` exercises it in default/simple mode -- so this is
        # NOT gated as narrowly as the embedding-only flat-slab invariant
        # below. ``passthrough`` carries arbitrary placeholder per-space
        # <nDim> (e.g. the config-scoping nOutput-reading fixture, all
        # nDim=1) and intentionally bypasses dimensional validation; the
        # "6+2+2" band subtraction would otherwise drive its CS content
        # negative and trip this check spuriously.
        # dimensional-governance (doc/specs/2026-06-05-dimensional-
        # governance.md sec.4/sec.6): SERIAL mode treats WS as an independent
        # native-width property tower whose sparse codebook activation enters
        # CS at conceptual width. Therefore ``symbol_dim == concept_dim`` is
        # relaxed for serial configs. Parallel legacy configs (square Pi/Sigma
        # over the constant slab) still require the pass-through match.
        _serial_raw = arch.get("serial", None)
        if _serial_raw is None:
            try:
                _sym_order = int(arch.get("symbolicOrder", 0) or 0)
            except (TypeError, ValueError):
                _sym_order = 0
            _ws_fold_bridged = (_sym_order > 0)
        elif isinstance(_serial_raw, bool):
            _ws_fold_bridged = _serial_raw
        else:
            _ws_fold_bridged = (
                str(_serial_raw).strip().lower()
                in ("true", "1", "yes", "on"))
        # This remains only a legacy PASS-THROUGH assumption. An explicit
        # nInputDim != nOutputDim likewise says WS's recurrent input and native
        # peer output are distinct interfaces, so output-width equality with
        # CS must not be imposed.
        def _ws_dim(key, fallback):
            try:
                v = int(gsp(cfg, "WholeSpace", key))
                return v if v > 0 else int(fallback)
            except (KeyError, TypeError, ValueError):
                return int(fallback)
        _ws_reshapes = (_ws_dim("nInputDim", symbol_dim)
                        != _ws_dim("nOutputDim", symbol_dim))
        # The data space_role (was architecture-level <modelType>) now lives under
        # <data> as <dataType> (embedding | numeric). The flat-slab
        # WS.nWhat==CS.nWhat invariant below is embedding-only: the numeric
        # (dense-slab) path carries no lexicon, so it may re-dimension across
        # space_roles (e.g. MNIST IS=784 -> CS=20). Read from the PASSED cfg (this
        # validator runs on a merged dict that may not be the global config),
        # canonical <data><dataType> first then an architecture-level dataType
        # fallback (dict-based test overrides set it there).
        _data_sec = arch.get("data") if isinstance(arch.get("data"), dict) else {}
        data_type = str(_data_sec.get("dataType")
                        or arch.get("dataType") or "numeric").strip().lower()
        is_embedding_mode = (data_type == "embedding")
        if (is_embedding_mode
                and not _ws_fold_bridged and not _ws_reshapes):
            TheXMLConfig.require(
                lambda cfg, _c=effective_concept_dim, _s=symbol_nwhat: _c == _s,
                f"WholeSpace requires WS.nWhat == CS.nWhat "
                f"(got CS.nWhat={effective_concept_dim}, "
                f"WS.nWhat={symbol_nwhat}). Fix: set <WholeSpace><nDim> "
                f"to match <ConceptualSpace><nOutputDim> if present, else "
                f"<ConceptualSpace><nDim>."
            )

        # ---- Concept-activation boundary ----------------------------
        # Stage 1.D refactor (doc/plans/2026-05-26-two-loop-pi-sigma-
        # substrate.md): legacy PS->CS tensor handoffs require equal flat
        # content slabs (nOutput * content width). The canonical aligned
        # serial-word architecture has a different boundary: PS, WS, and SS
        # remain independent peers, and each source's sparse sigma/codebook
        # activation reads a conceptual-width row before CS. Thus CS receives
        # conceptual-width events and performs no feature-width conversion.
        #
        # Embedding-mode only: the numeric (dense-slab) path doesn't carry a
        # lexicon, so the "PS per-word is CS-space" rationale doesn't apply --
        # the slab can re-dimension across space_roles (e.g. MNIST IS=784 pixels,
        # CS=20 features). The invariant fires only when
        # ``<data><dataType>embedding`` (``is_embedding_mode``, resolved above).

        def _effective_out_dim(space_name, fallback_dim):
            """Return nOutputDim if set, else nDim, else the chained
            ``fallback_dim`` from the previous space_role."""
            try:
                raw = gsp(cfg, space_name, "nOutputDim")
                if raw and int(raw) > 0:
                    return int(raw)
            except KeyError:
                pass
            try:
                raw = gsp(cfg, space_name, "nDim")
                if raw and int(raw) > 0:
                    return int(raw)
            except KeyError:
                pass
            return int(fallback_dim)

        def _effective_out_count(space_name, fallback_count):
            try:
                raw = gsp(cfg, space_name, "nOutput")
                if raw and int(raw) > 0:
                    return int(raw)
            except KeyError:
                pass
            return int(fallback_count)

        # "6+2+2": nDim/nOutputDim are EVENT widths; chain on the event width
        # but the flat-slab reshape invariant holds on the CONTENT slab
        # (nDim - band) -- the .where/.when band is per-position overhead, not
        # part of the reshaped content. Subtract each space_role's band for the slab.
        from architecture import canonical_shape as _cshape
        is_dim_e = _effective_out_dim("InputSpace", input_dim)
        is_n = _effective_out_count("InputSpace", 1)
        ps_dim_e = _effective_out_dim("PartSpace", is_dim_e)
        ps_n = _effective_out_count("PartSpace", is_n)
        cs_dim_e = _effective_out_dim("ConceptualSpace", ps_dim_e)
        cs_n = _effective_out_count("ConceptualSpace", ps_n)
        _is_band_shape = tuple(_cshape("InputSpace"))
        _ps_band_shape = tuple(_cshape("PartSpace"))
        _cs_band_shape = tuple(_cshape("ConceptualSpace"))
        is_dim = is_dim_e - sum(_is_band_shape)
        ps_dim = ps_dim_e - sum(_ps_band_shape)
        cs_dim = cs_dim_e - sum(_cs_band_shape)

        is_slab = is_n * is_dim
        ps_slab = ps_n * ps_dim
        cs_slab = cs_n * cs_dim
        # Handoff-consistency (dimensional-governance, 2026-06-06): IS is the
        # raw input and may be BIGGER than perception -- PS scopes it down via
        # chunking/embedding ("more input than output"), so the char input is
        # NOT constrained to the PS/CS content slab. Historically PS->CS then
        # had to be a pure reshape. The one narrow exception is the canonical
        # aligned serial-word model: its native PS/WS peers share locations
        # and width, then each sparse sigma/codebook activation reads the
        # high-dimensional concept row. Mixing/parallel/legacy paths retain
        # the flat-slab equality.
        def _config_true(raw):
            if isinstance(raw, bool):
                return raw
            return str(raw or "").strip().lower() in (
                "true", "1", "yes", "on")

        _serial_word_major = (
            _config_true(arch.get("serial", False))
            and _config_true(arch.get("serialObjectMeta", False)))
        _aligned_serial_word = (
            _serial_word_major
            and str(arch.get("conceptBinding", "mixing")).strip().lower()
            == "aligned")
        try:
            _cs_input_dim_raw = int(gsp(
                cfg, "ConceptualSpace", "nInputDim"))
        except (KeyError, TypeError, ValueError):
            _cs_input_dim_raw = 0
        _cs_input_dim = (
            _cs_input_dim_raw if _cs_input_dim_raw > 0 else ps_dim_e)
        try:
            _cs_input_n_raw = int(gsp(cfg, "ConceptualSpace", "nInput"))
        except (KeyError, TypeError, ValueError):
            _cs_input_n_raw = 0
        _cs_input_n = _cs_input_n_raw if _cs_input_n_raw > 0 else ps_n
        _ws_band_shape = tuple(_cshape("WholeSpace"))
        ws_dim_e = _effective_out_dim("WholeSpace", cs_dim_e)
        ws_n = _effective_out_count("WholeSpace", cs_n)
        ws_dim = ws_dim_e - sum(_ws_band_shape)
        _sparse_activation_boundary = (
            _aligned_serial_word
            and ps_n == ws_n == cs_n == _cs_input_n
            and ps_dim_e == ws_dim_e
            and _ps_band_shape == _ws_band_shape == _cs_band_shape
            and _cs_input_dim == cs_dim_e
            and ps_dim == ws_dim
            and 0 < ps_dim < cs_dim)
        if (is_embedding_mode and not (ps_slab == cs_slab)
                and not _sparse_activation_boundary):
            errors.append(
                f"flat-slab invariant violated: PS.nOutput*content "
                f"({ps_n}*{ps_dim}={ps_slab}) must equal CS.nOutput*content "
                f"({cs_n}*{cs_dim}={cs_slab}), except when aligned "
                "serialObjectMeta uses equal native PS/WS peers followed by "
                "sparse sigma/codebook activation. That activation must "
                "already emit CS-width events. The configured geometry is "
                f"PS {ps_n}x{ps_dim_e}, WS {ws_n}x{ws_dim_e}, and CS input "
                f"{_cs_input_n}x{_cs_input_dim} -> output {cs_n}x{cs_dim_e}; "
                "IS may be larger (PS scopes the input down)."
            )
        if is_embedding_mode and _aligned_serial_word:
            if (ps_n != ws_n or ps_dim_e != ws_dim_e
                    or _ps_band_shape != _ws_band_shape):
                errors.append(
                    "aligned serialObjectMeta requires PS and WS to be "
                    "independent peers with identical native event geometry; "
                    f"got PS={ps_n}x{ps_dim_e} with band {_ps_band_shape} and "
                    f"WS={ws_n}x{ws_dim_e} with band {_ws_band_shape}. The "
                    "conceptual codebook activation widens both sources only "
                    "after their native folds; PS must not feed WS.")
            if (_cs_input_n != cs_n or _cs_input_dim != cs_dim_e):
                errors.append(
                    "aligned serialObjectMeta requires ConceptualSpace input "
                    "to match its conceptual output geometry because source "
                    "sigma/codebook activation owns the dimension increase; "
                    f"got CS input={_cs_input_n}x{_cs_input_dim} and output="
                    f"{cs_n}x{cs_dim_e}.")
        if is_embedding_mode and _serial_word_major:
            # The outer word traversal is configured independently by
            # <serialWordCapacity>.  PS/CS remain the canonical instantaneous
            # field and STM workspace, so their slot counts stay aligned with
            # STM capacity rather than expanding to the surface-word cap.
            try:
                _stm_cap = int(gsp(cfg, "ConceptualSpace", "stmCapacity"))
            except (KeyError, TypeError, ValueError):
                _stm_cap = 8
            if ps_n != _stm_cap or cs_n != _stm_cap:
                errors.append(
                    "serialObjectMeta field/workspace mismatch: "
                    f"PartSpace.nOutput={ps_n} and "
                    f"ConceptualSpace.nOutput={cs_n} must both equal "
                    f"stmCapacity={_stm_cap}. The independent outer sentence "
                    "limit belongs in <serialWordCapacity>.")

        # ---- Recurrent WS input and direct CS->OS head (fail-loud) ---
        # dimensional-governance Task C1 (doc/specs/2026-06-05-dimensional-
        # governance.md sec.4/sec.6; doc/plans/2026-06-06-dimensional-
        # governance-completion.md). BasicModel is not a linear
        # IS->PS->WS->CS->OS chain: PS, WS, and SS are peers feeding CS. Two
        # explicit shape contracts remain around that peer loop:
        #
        #   CS->WS : WS's recurrent INPUT accepts the prior conceptual event.
        #            WS still emits its native property-width peer state; this
        #            input compatibility is not a PS->WS dependency.
        #   CS->OS : the terminal output head consumes terminal CS directly.
        #            WholeSpace is an upstream property peer and is not an
        #            intermediate terminal producer.
        #
        # An UNSET consumer dim means "inherit the producer's width" (the
        # resolvers fall back to the producer value), so an omitted
        # <nInputDim> can never trip these -- only an EXPLICIT mismatching
        # value RAISES. Gated on embedding mode + non-passthrough to stay
        # consistent with the flat-slab and nWhat checks (legacy
        # SimpleModel / numeric / passthrough configs are untouched).
        def _effective_in_dim(space_name, fallback_dim):
            """nInputDim if explicitly set (>0), else the chained
            ``fallback_dim`` (the producer's output width)."""
            try:
                raw = gsp(cfg, space_name, "nInputDim")
                if raw and int(raw) > 0:
                    return int(raw)
            except KeyError:
                pass
            return int(fallback_dim)

        def _effective_in_count(space_name, fallback_count):
            try:
                raw = gsp(cfg, space_name, "nInput")
                if raw and int(raw) > 0:
                    return int(raw)
            except KeyError:
                pass
            return int(fallback_count)

        if is_embedding_mode:
            # CS recurrent feedback accepted by WS (event width). cs_dim_e is
            # CS's effective output event width resolved above.
            ws_in_dim = _effective_in_dim("WholeSpace", cs_dim_e)
            if ws_in_dim != cs_dim_e:
                errors.append(
                    f"CS->WS recurrent input inconsistent: "
                    f"WS.nInputDim={ws_in_dim} must equal "
                    f"CS.nOutputDim={cs_dim_e}. WS may then emit its native "
                    f"property-width peer state. Fix: set "
                    f"<WholeSpace><nInputDim> to "
                    f"{cs_dim_e} (or omit it to inherit CS.nOutputDim)."
                )
            # The output head is constructed from terminal CS, not WS. Keep
            # the direct event geometry exact; a coincidentally equal flattened
            # product with swapped count/dimension is not the same interface.
            os_in_dim = _effective_in_dim("OutputSpace", cs_dim_e)
            os_in_n = _effective_in_count("OutputSpace", cs_n)
            if os_in_n != cs_n or os_in_dim != cs_dim_e:
                errors.append(
                    f"CS->OS handoff inconsistent: terminal CS emits "
                    f"{cs_n}x{cs_dim_e}, but OutputSpace expects "
                    f"{os_in_n}x{os_in_dim}. The output head consumes CS "
                    "directly; WholeSpace is not an intermediate producer."
                )

        # Monotonicity ⟹ no ConceptualSpace projection codebook.
        # ``architecture.monotonic`` makes the subsymbolic loop
        # order-preserving (W>=0) so the ramsified codebook can match a
        # symbol mapped across orders via ``Ops.part``. A ConceptualSpace
        # projection codebook is a basis *expansion* to a wide prototype
        # space and is not reliably order-preserving, so it breaks that
        # invariant. Require the projection bypassed when monotonic.
        if bool(arch.get("monotonic", False)) and Space.normalize_codebook_mode(
                gsp(cfg, "ConceptualSpace", "codebook")) != "none":
            errors.append(
                "architecture.monotonic=True requires "
                "<ConceptualSpace><codebook>none</codebook>: a basis "
                "expansion (quantize/project codebook) is not order-"
                "preserving and breaks the monotone-loop invariant the "
                "ramsified symbolic match (Ops.part) depends on. Set "
                "<ConceptualSpace><codebook>none</codebook> or "
                "<architecture><monotonic>false</monotonic>.")

        # ---- Respect-explicit endpoint dims (fail-loud) -------------------
        # Task #11 (architecture-backlog §1, decision (3)). The endpoint space_roles
        # (InputSpace, OutputSpace) carry author-facing task widths: the input
        # cardinality and the OUTPUT TARGET cardinality the head must
        # represent. An EXPLICITLY-set <nDim> on these must be honoured
        # verbatim by ``_resolve_dim`` -- never silently replaced by the
        # chained upstream width or a canonical band-derived value. Assert the
        # resolver returns the explicit value (it does today; this pins the
        # contract so a future canonical-override regression fails loud here
        # instead of silently collapsing the head's target cardinality).
        for _endpoint, _prev in (("InputSpace", 1),
                                 ("OutputSpace", symbol_dim)):
            try:
                _raw = gsp(cfg, _endpoint, "nDim")
            except KeyError:
                continue  # omitted -> inherit sentinel, nothing to honour
            if _raw and int(_raw) > 0:
                _resolved = _resolve_dim(_endpoint, _prev)
                if int(_resolved) != int(_raw):
                    errors.append(
                        f"{_endpoint} explicit <nDim>={int(_raw)} was "
                        f"overridden to {int(_resolved)}: endpoint widths are "
                        f"honoured verbatim (respect-explicit, Task #11). The "
                        f"config-author-sized {_endpoint} width must not be "
                        f"silently canonical/chained-overridden.")

        # Invertible PartSpace shape constraints are registered inside
        # PartSpace._register_requirements() (not here) to keep them self-contained.
        percept_inv = gsp(cfg, "PartSpace", "invertible")

        # Warn only for the legacy naive reverse path. Non-naive inversion
        # uses the LDU/triangular-solve path and does not use pinv.
        naive = bool(arch.get("naive", False))
        if naive and percept_inv:
            warnings.warn(
                "PartSpace: architecture.naive=True materializes dense "
                "inverse weights on the reverse path. This is slower and less "
                "memory efficient than the non-naive LDU solve path. Consider "
                "setting <naive>false</naive> unless debugging the dense path.",
                stacklevel=2)

        if errors:
            raise ValueError(
                "XML config inconsistencies:\n  - " + "\n  - ".join(errors))

        # Fire any requirements registered above at validate_config time, so
        # they surface as config errors *before* model construction, alongside
        # the errors.append path.  Any remaining requirements registered later
        # (inside Space._register_requirements during __init__) will fire
        # via the second TheXMLConfig.validate() call at the end of
        # BasicModel.__init__.
        TheXMLConfig.validate()

    @staticmethod
    def resolve_xml(path):
        """Resolve an XML config path relative to the project directory.

        Tries the absolute path first, then ``PROJECT_DIR/path``, then
        ``PROJECT_DIR/data/path``. Returns the input unchanged if none
        of the candidates exist so the eventual open() surfaces the
        original argument in its error.
        """
        if os.path.isabs(path):
            return path
        # Try relative to project root first (handles "data/simple.xml")
        candidate = os.path.join(ProjectPaths.PROJECT_DIR, path)
        if os.path.exists(candidate):
            return candidate
        # Try inside data/ (handles bare "simple.xml")
        candidate = os.path.join(ProjectPaths.PROJECT_DIR, "data", path)
        if os.path.exists(candidate):
            return candidate
        return path

    @staticmethod
    def _brick_preflight(m, batch_size, lr):
        """CUDA-graph-capture pre-flight gate (all training entry).

        Profiles ONE short ``runEpoch`` and hard-aborts if the brick
        body issues any ``cudaMemcpyDtoH``: a host sync defeats
        CUDA-graph capture and silently wastes GPU on a non-capturable
        run, so fail fast *before* substantial training rather than
        after hours. CUDA-only -- on CPU/MPS ``torch.profiler`` records
        no device memcpy and the capture contract is CUDA-specific, so
        this is a no-op there (the MPS dev path is unaffected).

        Runs two short *bounded* epochs (warm-up + profiled) at the
        **real configured ``batchSize``**, ``max_batches`` each. The
        batch size must match training: a CUDA graph is captured for a
        specific shape, so a brick that is sync-free at ``bs=2`` says
        nothing about ``bs=64`` (the only shape that matters is the one
        training runs). Bounding the *number* of batches -- not their
        size -- keeps this cheap (a handful of real-shape steps), so the
        gate validates exactly what training will execute. These do NOT
        count against ``BASIC_MAX_BATCHES`` (the ``--batches`` budget):
        the pre-flight is diagnostic, so ``_preflight_active`` exempts
        it from the cumulative cap in ``runEpoch``. Without both the
        bound and the exemption, on a large corpus the warm-up consumes
        the whole ``--batches`` budget, leaving the profiled epoch (and
        the real training) with zero batches -- the gate then "passes"
        having profiled nothing, and no training happens.

        **MODEL_DEBUG-gated.** ``enable_compiled_step`` now compiles the
        forward with ``fullgraph=True`` -- a *static* no-graph-break /
        no-host-sync guarantee enforced at compile time (a regression
        *raises* during compile, not after hours of wasted GPU). That
        supersedes this runtime profiler-based DtoH gate, which costs an
        extra warm-up compile plus a Kineto-profiled compiled epoch on
        every production run. Keep it only as an opt-in cross-check
        under ``MODEL_DEBUG``; skip it (faster compiles, no double
        warm-up) otherwise.
        """
        if not _util.MODEL_DEBUG:
            return
        if not torch.cuda.is_available():
            return
        # Profile at the real training batch size (CUDA-graph capture is
        # shape-specialized -- validating a different shape is
        # meaningless). Cheapness comes from the max_batches bound, not
        # from shrinking the batch.
        bs = max(1, int(batch_size))
        # Bounded so a large corpus (MM_20M: 358k sentences) doesn't run
        # a full epoch here; enough to clear first-touch lazy-init
        # before the profiled pass.
        pf_warmup, pf_profile = 4, 2
        opt = m.getOptimizer(lr=lr)
        # O1: ModelFactory.run now calls m.enable_compiled_step() before
        # this, so runBatch routes through the torch.compiled callable.
        # This pre-flight therefore profiles the COMPILED step. (Prior
        # 58/430 cudaMemcpyDtoH figures were measured fully eager --
        # torch.compile was a no-op then -- and are moot. Re-measured
        # after the non-grammar break-elimination program.)
        m._preflight_active = True
        try:
            # Warm-up so first-touch lazy-init syncs aren't counted as
            # steady state (mirrors test_brick_no_sync).
            m.runEpoch(optimizer=opt, batchSize=bs, split="train",
                       max_batches=pf_warmup)
            with torch_profile(activities=[ProfilerActivity.CPU,
                                           ProfilerActivity.CUDA]) as prof:
                m.runEpoch(optimizer=opt, batchSize=bs, split="train",
                           max_batches=pf_profile)
        finally:
            m._preflight_active = False
        dtoh = [e for e in prof.events()
                if "memcpy" in e.name.lower() and "dtoh" in e.name.lower()]
        if dtoh:
            raise RuntimeError(
                f"Brick pre-flight FAILED: runEpoch issued {len(dtoh)} "
                f"cudaMemcpyDtoH event(s) -- the brick body is not "
                f"CUDA-graph-capture-ready. Substantial training is "
                f"aborted to avoid wasting GPU on a non-capturable run. "
                f"Example events: {[e.name for e in dtoh[:5]]}. Profile "
                f"and eliminate the host syncs (see "
                f"test/test_brick_no_sync.py and "
                f"doc/plans/2026-04-27-brick-vectorization-and-legacy-"
                f"removal-handoff.md §6), or train on CPU/MPS.")
        TheMessage("[brick] pre-flight OK: 0 cudaMemcpyDtoH in runEpoch")

    @staticmethod
    def run(config_path):
        """Main entry point -- create, train, and evaluate a model from XML config.

        Loads the dataset via ``TheData.load``, constructs the model
        through ``BaseModel.from_config``, compiles it, then runs
        training (optionally under cProfile). Returns
        ``[(name, rCorrect, model)]``. Honors a handful of ``BASIC_*``
        env vars for runtime overrides.
        """
        # Pre-read config for dataset loading (needed before create_from_config)
        defaults_path = os.path.join(ProjectPaths.DATA_DIR, "model.xml")
        init_config(path=config_path, defaults_path=defaults_path)
        cfg = TheXMLConfig.data
        arch = cfg.get("architecture", {})
        dat = arch.get("data", {})
        trn = arch.get("training", {})

        # Device hydration: env var wins (the MODEL_AMP/MODEL_COMPILE
        # precedent); XML <architecture><device> is a checked-in default
        # applied only when BASICMODEL_DEVICE is unset. The XOR_exact
        # gate fixtures pin <device>cpu</device>: MPS kernels are
        # nondeterministic, so a seeded gate is only reproducible on CPU.
        # The process-wide device is restored on exit so in-process
        # callers (tests, probes) are not left on the fixture's device.
        prev_device = None
        if not os.environ.get("BASICMODEL_DEVICE"):
            xml_device = arch.get("device")
            if xml_device:
                prev_device = str(TheDevice.get())
                init_device(str(xml_device).strip().lower())

        # Determinism: <training><seed> pins torch/python/numpy RNG for
        # the whole run (construction init + data order + training).
        # The XOR_exact gates are read as single CLI runs; unseeded,
        # their recon column is run-to-run flaky (predictions stable).
        seed = os.environ.get("BASIC_SEED", trn.get("seed"))
        if seed is not None:
            torch.manual_seed(int(seed))
            random.seed(int(seed))
            np.random.seed(int(seed))

        try:
            return ModelFactory._run_hydrated(config_path, arch, dat, trn)
        finally:
            if prev_device is not None:
                init_device(prev_device)

    @staticmethod
    def _run_hydrated(config_path, arch, dat, trn):
        """Post-hydration body of :meth:`run` (config parsed, device and
        seed applied). Split out so the device override restores in a
        ``finally`` regardless of how the run exits."""
        # AMP hydration: env var wins (matches MODEL_COMPILE precedent).
        # XML <architecture><amp> is a checked-in default applied only when
        # the env var is unset.
        if not os.environ.get("MODEL_AMP"):
            xml_amp = arch.get("amp")
            if xml_amp:
                os.environ["MODEL_AMP"] = str(xml_amp)
                init_model_amp()

        dataset = os.environ.get("BASIC_DATASET", dat.get("dataset"))
        # Environment overrides for num_shards/max_docs (set by train.py)
        num_shards = int(os.environ.get("BASIC_NUM_SHARDS", dat.get("numShards", 1)))
        max_docs = int(os.environ.get("BASIC_MAX_DOCS", dat.get("maxDocs", 10000)))
        max_tokens_env = os.environ.get("BASIC_MAX_TOKENS")
        max_tokens = int(max_tokens_env) if max_tokens_env else None
        random_shards = os.environ.get("BASIC_RANDOM_SHARDS", "0") == "1"

        TheData.load(dataset,
                     num_shards=num_shards,
                     max_docs=max_docs,
                     shard_dir=dat.get("shardDir"),
                     dat=dat,
                     random_shards=random_shards,
                     max_tokens=max_tokens)

        target_device = TheDevice.get()
        if target_device.type == "mps":
            # MPS construction can corrupt setup-time parameter tensors in
            # large radix/byte configs (finite CPU construction of the same
            # config is clean, and a subsequent module move to MPS is clean).
            # Build the module tree on CPU, then move registered parameters
            # and buffers to the requested MPS device before training. Runtime
            # forward/backward still executes on MPS because TheDevice is
            # restored immediately after construction.
            init_device("cpu")
            try:
                m, _ = BaseModel.from_config(config_path, data=TheData)
            finally:
                init_device(target_device)
            m.to(target_device)
        else:
            m, _ = BaseModel.from_config(config_path, data=TheData)
        TheMessage(f"Device: {TheDevice}")

        # O1: `compile(m)` on the module is a no-op here -- `m.run()`
        # delegates to the eager `_orig_mod`, so the compiled callable
        # never ran. Compile the per-batch forward callable and invoke
        # it from runBatch instead (eager streaming stays outside).
        m.enable_compiled_step()

        def _t(key, default=None):
            return trn.get(key, default)

        def _d(key, default=None):
            return dat.get(key, default)

        num_epochs = int(os.environ.get("BASIC_NUM_EPOCHS", _t("numEpochs", 3)))
        batch_size = int(os.environ.get(
            "BASIC_BATCH_SIZE", _t("batchSize", 10)))

        # CUDA-graph-capture pre-flight: gates ALL training entry (the
        # profiled path and the normal path) for every model run via
        # train.py. Hard-aborts before substantial training if the
        # brick body still issues host syncs. CUDA-only no-op elsewhere.
        ModelFactory._brick_preflight(
            m, batch_size, _t("learningRate", 0.01))

        do_profile = os.environ.get("BASIC_PROFILE", "").lower() in ("1", "true") or _t("profile", False)
        if do_profile:
            with torch_profile(
                activities=[ProfilerActivity.CPU],
                schedule=profiler_schedule(wait=1, warmup=1, active=3, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                m.run(_t("numTrials", 1),
                            num_epochs,
                            batch_size,
                            lr=_t("learningRate", 0.01), profile=prof)

            # Print summary table
            TheMessage(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

            # Export Chrome trace
            trace_path = ProjectPaths.output_path("profile_trace.json")
            prof.export_chrome_trace(trace_path)
            TheMessage(f"Chrome trace saved to {trace_path}")
            return [(m.name, m.rCorrect, m)]

        m.run(_t("numTrials", 1),
                    num_epochs,
                    batch_size,
                    lr=_t("learningRate", 0.01))

        _autosave_env = os.environ.get("BASIC_AUTOSAVE")
        _autosave = bool(_t("autosave", False))
        if _autosave_env is not None:
            _autosave = _autosave_env.strip().lower() in (
                "1", "true", "yes", "on")
        if _autosave:
            m.save_training_checkpoint(reason="final autosave")

        return [(m.name, m.rCorrect, m)]
BasicModelFactory = ModelFactory

def test():
    """Smoke test: verify encodings and run the XOR config end-to-end.

    Touches ``WhereEncoding`` / ``WhenEncoding`` self-tests, then runs
    the XOR XML config to exercise the full model factory path.
    """
    WhereEncoding.test()
    WhenEncoding.test()
    ModelFactory.run(os.path.join(ProjectPaths.PROJECT_DIR, "data", "xor.xml"))


# --- CLI entry point ---
# Usage: python BasicModel.py [config.xml]
#        python BasicModel.py --compare config1.xml config2.xml
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        prog="BasicModel.py",
        description=(
            "Train and evaluate a BasicModel from an XML config file.\n\n"
            "Examples:\n"
            "  python BasicModel.py data/xor.xml\n"
            "  python BasicModel.py data/XOR_spaces.xml\n"
            "  python BasicModel.py --compare data/xor.xml data/XOR_exact.xml\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        metavar="CONFIG",
        help=(
            "Path to the XML config file (relative to data/ or absolute). "
            "Defaults to data/xor.xml when omitted."
        ),
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("CONFIG1", "CONFIG2"),
        help=(
            "Run two configs side by side and plot per-digit accuracy, "
            "combined accuracy, and combined loss comparisons."
        ),
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Generate figures and HTML report at the end of the run.",
    )
    parser.add_argument(
        "--compile",
        default=None,
        metavar="BACKEND",
        help=(
            "Compilation backend: none, inductor, eager, aot_eager. "
            "Overrides MODEL_COMPILE env var."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Profile the model's training step using torch.profiler (overrides XML <training><profile>).",
    )
    args = parser.parse_args()

    if args.compile is not None:
        init_compile_backend(args.compile)

    if args.profile:
        os.environ["BASIC_PROFILE"] = "true"

    TheReport.enabled = args.report

    try:
        if args.compare:
            # Compare mode: run two XML configs and plot per-digit accuracy side by side
            xml1 = ModelFactory.resolve_xml(args.compare[0])
            xml2 = ModelFactory.resolve_xml(args.compare[1])
            TheReport.add_xml(xml1)
            TheReport.add_xml(xml2)
            results = ModelFactory.run(xml1) + ModelFactory.run(xml2)
            if len(results) >= 2:
                TheReport.plotComparison([(name, rc) for name, rc, _ in results])
                TheReport.plotCombinedAccuracy([(name, rc) for name, rc, _ in results])
                TheReport.plotCombinedLoss([m for _, _, m in results])
        else:
            # Single run mode
            xml = ModelFactory.resolve_xml(args.config) if args.config else os.path.join(ProjectPaths.PROJECT_DIR, "data", "xor.xml")
            TheReport.add_xml(xml)
            results = ModelFactory.run(xml)
    finally:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    TheReport.write_html()
