"""Top-level model assembly, data loading, and experiment reporting.

``BasicModel`` composes the custom layers from ``Model.py`` into a set of
spaces that move between raw inputs, percepts, concepts, symbols, syntax,
and outputs.  The same module also carries the project utilities used to
load datasets, resolve config paths, plot results, and save reports.
"""

import logging
import math, os, warnings
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
from util import ProjectPaths, XMLConfig, compile, TheXMLConfig, init_config, init_compile_backend, amp_context, init_model_amp
import util as _util
from embed import WordVectors, PretrainModel, _random_unit_ball
from data import Data, TheData

from Layers import Layer, PiLayer, SigmaLayer  # Import custom layers from Model.py
from Layers import LinearLayer, AttentionLayer
from Layers import LiftingLayer, CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon
from Layers import Error, TheError
from Layers import Ops, GRAMMAR_LAYER_CLASSES, CONTIGUITY_PRESERVING_OPS
from Mereology import Mereology
from dataclasses import dataclass, field
from typing import List

from Spaces import ActiveEncoding, WhereEncoding, WhenEncoding, WhatEncoding, EventEncoding
from Spaces import Basis, Tensor, Codebook, Embedding
from Spaces import SubSpace, Space, InputSpace, PerceptualSpace, ModalSpace, ConceptualSpace, SymbolicSpace, OutputSpace, ShortTermMemory
# ``normalize_codebook_mode`` moved onto ``Space`` as a staticmethod
# (2026-05-21) so the parsing logic stays namespaced; callers below
# read it as ``Space.normalize_codebook_mode(...)``.
from Language import WordSubSpace
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


def _log_advisory_exception(stage: str, exc: BaseException) -> None:
    """Log a caught pipeline-stage exception once with traceback,
    then deduped counts. Surfacing a silent fall-through to the
    fallback path is important: the chart's contribution becomes a
    no-op and you stop training what the config says you train.
    """
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    file = tb.tb_frame.f_code.co_filename if tb else "?"
    line = tb.tb_lineno if tb else 0
    key = (type(exc).__name__, file, line)
    log = logging.getLogger(__name__)
    seen = _PIPELINE_EXC_SEEN.get(key, 0)
    _PIPELINE_EXC_SEEN[key] = seen + 1
    if seen == 0:
        log.warning(
            "%s failed (%s: %s) at %s:%d -- "
            "the chart's contribution is now a no-op. "
            "Subsequent occurrences of this exact failure will be deduped.",
            stage, type(exc).__name__, exc, file, line,
            exc_info=True,
        )
    elif (seen + 1) % _DEDUPE_FLUSH_EVERY == 0:
        log.warning(
            "%s: %s at %s:%d has now fired %d times.",
            stage, type(exc).__name__, file, line, seen + 1,
        )


class ReverseAdapter(nn.Module):
    """Wrap a module so its reverse() method is called via forward().

    Retained for backwards-compat (tests reference it); the live
    BasicModel reverse path no longer uses it -- explicit
    ``module.reverse(x)`` calls inside ``_run_pipeline_rev`` do the job.
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


class CachePoint(nn.Module):
    """Identity module that caches the last subspace it saw.

    Retained for backwards-compat (tests reference it); the live
    BasicModel midpoint cache is now a plain attribute populated
    inside ``_run_pipeline_rt``.
    """

    def __init__(self):
        super().__init__()
        self.last = None

    def forward(self, subspace):
        self.last = subspace
        return subspace

    def reverse(self, subspace):
        return subspace


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


# FlattenKWrapper retired 2026-05-11: K-axis flatten/restore is now
# handled inline by ``BasicModel._flatten_k`` / ``_restore_k`` around
# each stem/body/head boundary in the pipeline methods.

# -- End inlined-from-Pipeline section -------------------------------


class _MultiOptimizer:
    """Composite optimizer: forwards step / zero_grad / state_dict to a
    list of underlying torch.optim.Optimizer instances.

    Used to combine SparseAdam (for the perceptual embedding's sparse-
    grad parameter) with Adam (for the dense parameters) without
    requiring callers to know about the split.  Mirrors the subset of
    torch.optim.Optimizer's API that BaseModel.runBatch and the
    autosave path actually use.
    """

    def __init__(self, optimizers):
        """Wrap the optimizer list and expose a flattened param_groups view.

        The flat ``param_groups`` lets callers that read or set
        ``param_groups[i]['lr']`` (rebuild_optimizer, LR scheduling)
        see every group across the underlying optimizers.
        """
        self.optimizers = list(optimizers)
        # Synthesize a flat param_groups view so callers reading
        # `optimizer.param_groups[0]['lr']` (e.g. rebuild_optimizer)
        # see all groups across the underlying optimizers.
        pg = []
        for o in self.optimizers:
            pg.extend(o.param_groups)
        self.param_groups = pg

    def step(self, closure=None):
        """Call ``step`` on every underlying optimizer; return their results.

        Forwards ``closure`` only when not None so opts without closure
        support don't choke on the extra keyword.
        """
        results = []
        for o in self.optimizers:
            results.append(o.step(closure=closure) if closure is not None else o.step())
        return results

    def zero_grad(self, set_to_none=True):
        """Forward ``zero_grad`` to every underlying optimizer."""
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return a dict with one entry per underlying optimizer."""
        return {'optimizers': [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state):
        """Restore each underlying optimizer from the matching slot in ``state``."""
        for o, s in zip(self.optimizers, state.get('optimizers', [])):
            o.load_state_dict(s)


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
    # Scale applied to the DiscourseSpace contrastive loss. The
    # inter-sentence DiscourseSpace lives on ``self.wordSpace``
    # (``self.wordSpace.discourse``) rather than directly on the
    # model. Callers that need it should read through
    # ``wordSpace``; ``<training><sentencePrediction>false`` in
    # config leaves ``wordSpace.discourse`` as ``None``.
    # Class-level defaults for ARMA loss weight + priming scale.
    # Pre-2026-05-14 contrastive sentence-loss knobs retired alongside
    # <maskedPrediction>; the single ARMA MSE replaces the contrastive
    # + predictive cosine pair.
    arma_scale = 0.1
    sentence_priming_scale = 0.05

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
    def _resolve_dim(section, prev_dim):
        """Resolve a Space's nDim sentinel: 0 -> inherit ``prev_dim``.

        Shared between BasicModel.create() and BasicModel.create()
        (and any subclass): both pipelines chain dims through
        InputSpace -> PerceptualSpace -> ConceptualSpace ->
        SymbolicSpace -> OutputSpace, and an unset / zero-sentinel
        nDim means "inherit the upstream Space's content dim".
        """
        try:
            raw = TheXMLConfig.space(section, "nDim")
        except KeyError:
            return prev_dim
        return prev_dim if raw == 0 else raw

    @staticmethod
    def _obj_size(section):
        """Per-section objectSize: ``nWhere + nWhen``, defaulting to 0.

        Each Space carries its own positional / temporal encoding
        widths; the muxed event tensor's last-axis size is
        ``nDim + nWhere + nWhen``. Missing keys default to 0 so
        sections that don't declare them inherit the architecture
        default (also 0).
        """
        try:
            nw = TheXMLConfig.space(section, "nWhere")
        except KeyError:
            nw = 0
        try:
            nn = TheXMLConfig.space(section, "nWhen")
        except KeyError:
            nn = 0
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
    def _order_partitions(symbol_dim, conceptual_order):
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
        for t in range(conceptual_order):
            if t == conceptual_order - 1:
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
    def _level_shapes(n_vectors, dim, conceptual_order, width_mode="tapered"):
        """Per-level (N_t, D_t) shapes across the conceptual-order stack.

        Two width modes (set via XML ``architecture.conceptualWidth``):

        ``tapered`` (default, historical) -- D stays constant; N halves
            per level. Biological analogue: increasing receptive field
            (V1->V2->V4->IT). Requires ``n_vectors`` to be divisible by
            ``2^conceptual_order``.

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
            return [(int(n_vectors), int(dim)) for _ in range(conceptual_order)]
        # Default: tapered (geometric halving).
        shapes = []
        for t in range(conceptual_order):
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

        _t = TheXMLConfig.training
        _s = TheXMLConfig.space

        # DataLoader prefetch workers. Pulled here so every entry point
        # (ModelFactory.run, BaseModel.from_config, tests) shares the
        # same model.xml-defaulted value. 0 means synchronous in-process
        # batch assembly.
        self._num_workers = int(_t("numWorkers"))

        if model_type is None:
            model_type = arch["modelType"]

        # <embeddingPath> retired (2026-05-12): embeddings now ride
        # inside the single .ckpt bundle, not a separate .kv artifact.
        _nWhere = TheXMLConfig.get("architecture.nWhere")
        _nWhen = TheXMLConfig.get("architecture.nWhen")
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
        nPercepts = _resolve("PerceptualSpace", nInput)
        nConcepts = _resolve("ConceptualSpace", nPercepts)
        nSymbols  = _resolve("SymbolicSpace",   nConcepts)
        nWords    = nSymbols  # SyntacticSpace removed; kept for API compat
        nOutput   = _resolve("OutputSpace",      nSymbols)

        _nObjects = (
            _s("InputSpace", "nVectors")
            + _s("PerceptualSpace", "nVectors")
            + _s("ConceptualSpace", "nVectors")
            + _s("SymbolicSpace", "nVectors")
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
            conceptualOrder=arch["conceptualOrder"],
            model_type=model_type,
            data=data,
            reconstruction_scale=_t("reconstructionScale"),
            what_scale=_t("whatScale"),
            where_scale=_t("whereScale"),
            when_scale=_t("whenScale"),
        )

        # IR mask rate: Bernoulli probability that a P-tier position is
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
        if getattr(self, 'wordSpace', None) is not None:
            self.wordSpace.serial_mode = False

        # Per-word ground-truth cursor enable. Mirrors the retired
        # serial-boolean plumbing above (model computes a bool from its
        # own determination, then pushes it onto a Space) -- InputSpace
        # was not in the serial push list, so it is added here. The
        # predicate is "grammar enabled": ``self.useGrammar`` is derived
        # by ``_derive_use_grammar`` (set in create() above) and is
        # exactly "all" when the grammar XML carries a real, non-substrate
        # rule (chart fires / per-stage merge glue wired) else "none".
        # ``!= "none"`` is therefore precisely "there is grammar
        # enabled". INERT this increment: ``InputSpace.next_word`` is the
        # sole reader of ``_per_word_enabled`` and has no live caller, so
        # setting it True for grammar configs is byte-identical (the
        # live whole-slab ``forward`` is unchanged). Eager build/init
        # (here, post-create) -- never in the traced forward.
        if getattr(self, 'inputSpace', None) is not None:
            self.inputSpace._per_word_enabled = bool(
                getattr(self, 'useGrammar', 'none') != 'none')

        # Attention on ConceptualSpace violates position locality required
        # by serial_mode; downgrade conceptualSpace only. PerceptualSpace
        # is unaffected.
        if (self.serial_mode
                and getattr(self, 'conceptualSpace', None) is not None
                and getattr(self.conceptualSpace, 'hasAttention', False)):
            import warnings
            warnings.warn(
                "ConceptualSpace.hasAttention=True violates the position-"
                "locality constraint required by serial_mode; forcing "
                "conceptualSpace.serial_mode=False. PerceptualSpace "
                "serial_mode is unaffected.",
                RuntimeWarning,
            )
            self.conceptualSpace.serial_mode = False

        # InterSentenceLayer ARMA(p, q) loss weight. ``InterSentenceLayer.observe``
        # returns a per-batch MSE which ``runBatch`` weights by
        # ``arma_scale`` before adding to ``TheError``.  Active only
        # when ``<training><sentencePrediction>`` is true.
        self.arma_scale = float(
            TheXMLConfig.training("armaScale", 0.1) or 0.1)
        self.sentence_priming_scale = float(
            TheXMLConfig.training("sentencePrimingScale", 0.05) or 0.05)

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
        if self.optimize_embedding and isinstance(self.perceptualSpace.vocabulary, Embedding):
            emb_params = self.perceptualSpace.vocabulary.embedding_parameters()
            self.perceptualSpace.params = self.perceptualSpace.params + emb_params
        self.loss.embedding_scale = float(_t("embeddingScale") or 0.1)
        if isinstance(self.perceptualSpace.vocabulary, Embedding):
            self.perceptualSpace.vocabulary.optimize_embedding = self.optimize_embedding
            object.__setattr__(self.perceptualSpace.vocabulary, "_model", self)

        self.checkpoint_every_batches = int(os.environ.get(
            "BASIC_CHECKPOINT_EVERY_BATCHES",
            _t("checkpointEveryBatches", 0) or 0,
        ))
        self._training_step_count = 0

        if _t("autoload"):
            wpath = TheXMLConfig.get("architecture.weightsPath")
            wpath = self._resolve_artifact_path(wpath)
            # Single-artifact load: state_dict + vocab_extras +
            # bpe_extras all ride in the .ckpt. The separate .kv
            # embedding artifact was retired (2026-05-12).
            # Autoload: fail fast on a stale/mismatched ckpt (the
            # bivector retirement invalidates pre-refactor weights)
            # instead of silently proceeding on fresh init then crashing.
            self.load_weights(wpath, require_match=True)
        self.max_response_length = arch["maxResponseLength"]
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
        # Split the remaining params into a SparseAdam group (sparse
        # grads only) and an Adam group (everything else).  SparseAdam
        # touches only the rows that received gradients per step --
        # critical for V=1M-style perceptual codebooks where dense Adam
        # would update O(V*D) optimizer state on every step.
        # CUDA-graph capture (the brick body; test_brick_no_sync)
        # requires the optimizer keep its step counter on-device:
        # stock Adam's _multi_tensor_adam does _get_value(step).item()
        # per param otherwise -- one cudaMemcpyDtoH per param per step.
        # capturable=True keeps step on-device. Gated to actual CUDA
        # params (no-op/overhead on CPU/MPS; SparseAdam has no such flag).
        _cap = any(getattr(p, "is_cuda", False) for p in params)
        if sparse_ptrs:
            sparse_params = [p for p in params if p.data_ptr() in sparse_ptrs]
            dense_params = [p for p in params if p.data_ptr() not in sparse_ptrs]
            opt_sparse = optim.SparseAdam(sparse_params, lr=lr)
            opt_dense = optim.Adam(dense_params, lr=lr, capturable=_cap)
            return _MultiOptimizer([opt_dense, opt_sparse])
        return optim.Adam(params, lr=lr, capturable=_cap)

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

    def _checkpoint_path(self, suffix=None):
        """Resolve the configured checkpoint path, optionally adding a suffix.

        Falls back to ``output/<name>.ckpt`` when no ``weightsPath`` is
        set in the XML. The ``suffix`` is inserted before the extension
        so emergency / autosave variants can coexist with the canonical
        checkpoint.
        """
        path = TheXMLConfig.get("architecture.weightsPath", None)
        if path:
            path = self._resolve_artifact_path(path)
        else:
            path = ProjectPaths.output_path(f"{self.name}.ckpt")
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
                assert torch.isfinite(grad).all(), (
                    f"Non-finite gradient for {name!r} {what}: "
                    f"{int((~torch.isfinite(grad)).sum().item())}/{grad.numel()} "
                    f"entries are nan/inf."
                )

    def _debug_tensor_stats(self, name, value):
        """Print compact tensor stats when MODEL_DEBUG is enabled.

        Cannot use ``assert`` (printing has side effects), so only the
        MODEL_DEBUG gate applies.
        """
        if not _util.MODEL_DEBUG:
            return
        if value is None or not torch.is_tensor(value):
            TheMessage(f"[recon-debug] {name}: {value}")
            return
        x = value.detach()
        finite = torch.isfinite(x)
        count = x.numel()
        bad = int((~finite).sum().item())
        if finite.any():
            xf = x[finite]
            xmin = xf.min().item()
            xmax = xf.max().item()
            xmean = xf.float().mean().item()
        else:
            xmin = xmax = xmean = float("nan")
        TheMessage(
            f"[recon-debug] {name}: shape={tuple(x.shape)} "
            f"bad={bad}/{count} range=[{xmin:.6g}, {xmax:.6g}] "
            f"mean={xmean:.6g}"
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

    def set_sigma(self, sigma):
        """Propagate exploration meta-parameters to all spaces.

        ``sigma`` controls per-space exploration noise (typically the
        Gaussian sample width in codebook lookup). Spaces that don't
        use exploration may treat this as a no-op.
        """
        for s in self.spaces:
            s.set_sigma(sigma)

    def _get_embedding(self):
        """Return the Embedding instance if this model uses one, else None."""
        if hasattr(self, 'perceptualSpace') and isinstance(self.perceptualSpace.vocabulary, Embedding):
            return self.perceptualSpace.vocabulary
        return None

    # -- Reasoning Methods --------------------------------------------

    def _get_truth_layer(self):
        """Return the TruthLayer if available, else None.

        The TruthLayer now lives on ``WordSpace``; when the grammar
        path that builds WordSpace is disabled there is no layer.
        """
        return self.wordSpace.truth_layer if self.wordSpace is not None else None

    def _get_basis(self):
        """Return the Basis from symbolicSpace's subspace, else None."""
        ss = getattr(self, 'symbolicSpace', None)
        if ss is None:
            return None
        return getattr(getattr(ss, 'subspace', None), 'basis', None)

    @torch.no_grad()
    def _clamp_symbolic_codebook(self):
        """Keep the SymbolicSpace bivector codebook inside the [0,1] pair box.

        Called after ``optimizer.step()``. The SymbolicSpace codebook is
        monotonic (paired-index pos/neg poles), so each scalar lives in
        [0,1]. Gradient updates can push entries outside this box; clamp
        directly on the Parameter data so the invariant holds going into
        the next forward pass.
        """
        basis = self._get_basis()
        if basis is None or not getattr(basis, 'monotonic', False):
            return
        W = getattr(basis, 'W', None)
        if W is None or not isinstance(W, torch.Tensor):
            return
        W.data.clamp_(0.0, 1.0)

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

    @staticmethod
    def _is_one_component(adj):
        """BFS over a small boolean adjacency matrix; True if the graph is
        a single connected component. ``adj`` must be square and contain
        the identity (self-loops) so isolated rows are detected.
        """
        K = adj.shape[0]
        if K == 0:
            return True
        visited = torch.zeros(K, dtype=torch.bool, device=adj.device)
        frontier = torch.zeros(K, dtype=torch.bool, device=adj.device)
        frontier[0] = True
        while frontier.any():
            visited = visited | frontier
            neighbors = adj[frontier].any(dim=0)
            frontier = neighbors & ~visited
        return bool(visited.all().item())

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
        embedding vectors, vocabulary mappings, and BPE codebook.

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
        """
        if path is None:
            path = os.path.join(ProjectPaths.OUTPUT_DIR, "weights.ckpt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        bundle = {
            "state_dict": dict(self.state_dict()),
            "vocab_extras": self._collect_vocab_extras(),
            "bpe_extras": self._collect_bpe_extras(),
        }
        util.atomic_torch_save(bundle, path)
        TheMessage(f"[{self.name}] Weights saved to {path}")

    def _collect_vocab_extras(self):
        """Gather WordVectors mappings + SymbolicSpace's well-known
        atoms dict that don't ride in state_dict.

        Returns ``None`` when no Embedding is present.
        """
        emb = self._get_embedding()
        if emb is None or getattr(emb, 'wv', None) is None:
            return None
        wv = emb.wv
        counts = getattr(wv, 'counts', None)
        if counts is not None and hasattr(counts, 'tolist'):
            counts = counts.tolist()
        sym_space = getattr(self, 'symbolicSpace', None)
        well_known = dict(getattr(sym_space, 'well_known_atoms', {}) or {})
        return {
            "index_to_key": list(getattr(wv, 'index_to_key', []) or []),
            "counts": counts or [],
            "total_count": int(getattr(wv, 'total_count', 0) or 0),
            "well_known_atoms": well_known,
        }

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

    @staticmethod
    def print_weights_info(path):
        """Print a human-readable summary of a .ckpt weights artifact.

        Does not require a model to be loaded.  Useful for diagnosing
        mismatches between a saved checkpoint and a changed XML config.
        """
        if not os.path.exists(path):
            TheMessage(f"Weights file not found: {path}")
            return
        saved = torch.load(path, map_location="cpu", weights_only=False)
        state = saved["state_dict"] if isinstance(saved, dict) and "state_dict" in saved else saved
        total = sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor))
        TheMessage(f"Weights file    : {path}")
        TheMessage(f"  Total params  : {total:,}")
        TheMessage(f"  Layers ({len(state)}):")
        for key, tensor in state.items():
            if isinstance(tensor, torch.Tensor):
                TheMessage(f"    {key:<50s}  {list(tensor.shape)}")

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
        saved = torch.load(path, map_location=TheDevice.get(), weights_only=False)

        if isinstance(saved, dict) and "state_dict" in saved:
            state = saved["state_dict"]
            vocab_extras = saved.get("vocab_extras")
            bpe_extras = saved.get("bpe_extras")
        else:
            state = saved
            vocab_extras = None
            bpe_extras = None

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
        if vocab_extras is not None:
            self._restore_vocab_extras(vocab_extras)

        # Pre-check for shape mismatches before attempting to load.
        # This produces an actionable diagnostic instead of a raw PyTorch error.
        model_state = dict(self.state_dict())

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
                    and "symbolicSpace" in k):
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
        fatal_unexpected = unexpected if strict else []
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
        if vocab_extras is not None:
            self._restore_vocab_extras(vocab_extras)
        # Restore ChunkLayer BPE state (merges, vocab, id_to_bytes).
        if bpe_extras is not None:
            self._restore_bpe_extras(bpe_extras)

        TheMessage(f"[{self.name}] Weights loaded from {path}")
        return True

    def _restore_vocab_extras(self, extras):
        """Restore the WordVectors mappings AND resize the live
        ``wv._vectors`` parameter to match the saved vocab size.

        Called before the state_dict load so that the shape pre-check
        sees matching dimensions. Vector data itself is populated by
        the subsequent ``load_state_dict`` call; here we just allocate
        the right-sized parameter and rebuild the Python mappings.
        """
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
            wv._vectors = nn.Parameter(
                torch.zeros(vocab_size, dim,
                            device=wv._vectors.device,
                            dtype=wv._vectors.dtype),
                requires_grad=True)
        wv.index_to_key = keys
        wv.key_to_index = {k: i for i, k in enumerate(keys)}
        wv.counts = (np.asarray(counts, dtype=np.int64) if counts
                     else np.zeros(vocab_size, dtype=np.int64))
        wv.total_count = np.int64(total)
        if getattr(emb, 'pretrain', None) is not None:
            emb.pretrain.index_to_key = wv.index_to_key
            emb.pretrain.key_to_index = wv.key_to_index
        wv._normed = None
        # Restore the well-known atoms dict on SymbolicSpace so the
        # "words" meronomy parent (and any future named parents) lines
        # up with the rows in the saved codebook.
        sym_space = getattr(self, 'symbolicSpace', None)
        well_known = extras.get("well_known_atoms")
        if sym_space is not None and isinstance(well_known, dict) and well_known:
            sym_space.well_known_atoms = {
                str(k): int(v) for k, v in well_known.items()
            }

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

    def _restore_vocab(self, emb, saved_vocab,
                       counts=None, total_count=0, pending_counts=None):
        """Resize Embedding to match saved vocabulary exactly.

        After resizing, refresh ``emb.null_percept_idx`` (used by IR
        mode's mask injection) so it points to a valid row of the new
        codebook. If the saved vocab already contains
        ``PerceptualSpace.NULL_PERCEPT_KEY`` (a kv saved after IR-mode
        setup), we reuse its index. Otherwise we append a fresh
        NULL_PERCEPT slot at the tail and grow the codebook by 1,
        mirroring what ``Embedding.create`` does on a fresh build.
        """
        from Spaces import PerceptualSpace
        null_key = PerceptualSpace.NULL_PERCEPT_KEY
        dim = emb.wv._vectors.shape[1]
        saved_vocab = list(saved_vocab)
        had_null = null_key in saved_vocab
        if not had_null:
            saved_vocab.append(null_key)
            if counts is not None:
                counts = list(counts) + [0]
        vocab_size = len(saved_vocab)

        # Rebuild word mappings (shared between wv and pretrain)
        emb.wv.index_to_key = list(saved_vocab)
        emb.wv.key_to_index = {w: i for i, w in enumerate(saved_vocab)}
        emb.pretrain.index_to_key = emb.wv.index_to_key
        emb.pretrain.key_to_index = emb.wv.key_to_index
        emb.wv._vectors = nn.Parameter(
            torch.zeros(vocab_size, dim, device=TheDevice.get()), requires_grad=True)
        emb.wv.counts = (np.asarray(counts, dtype=np.int64) if counts is not None
                         else np.zeros(vocab_size, dtype=np.int64))
        emb.wv.total_count = np.int64(total_count)
        emb._pending_counts = dict(pending_counts) if pending_counts else {}
        emb.wv._normed = None

        # Re-anchor null_percept_idx to the canonical slot in the
        # restored codebook. The slot was either present in the saved
        # vocab (reuse its index) or just appended above (last row).
        emb.null_percept_idx = emb.pretrain.key_to_index[null_key]

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

    def _reconstructionReport(self):
        """Stub retained for API parity with the legacy reverse pipeline.

        The full input-reconstruction report relied on
        ``BasicModel.reverse`` and the input-space round trip, both of
        which were retired together with ``<reconstruct>output</...>``
        on 2026-05-14.  Within-sentence training is now IR-only; the
        masked-LM accuracy is reported via the per-batch
        ``reconstruction`` term in ``TheError`` and does not need a
        separate text-level dump.
        """
        return

        if not isinstance(allOut, torch.Tensor) or allOut.numel() == 0:
            return  # no predictions to report

        rows = []
        # Use reconstruct_data() for lex-based models (embedding vectors, not bytes)
        use_lex_recon = (self.inputSpace.model_type == "embedding" and
                         self.perceptualSpace.get_recovered_word(0, 0) is not None)
        if use_lex_recon:
            recon_text_list = self.perceptualSpace.reconstruct_data(text=True)
        for i in range(len(test_input)):
            original = self._bytes_to_text(test_input[i])
            if use_lex_recon:
                recon = recon_text_list[i]
            elif hasattr(self.inputSpace, 'reconstructed'):
                recon = self._bytes_to_text(self.inputSpace.reconstructed[i])
            else:
                recon = "(no reconstruction)"
            # Strip \x00 padding from both sides before comparing words
            orig_words = original.replace("\x00", " ").split()
            recon_words = recon.replace("\x00", " ").split()
            match = orig_words == recon_words
            css = "match" if match else "mismatch"
            label = test_output[i]
            if isinstance(label, torch.Tensor):
                label = label.squeeze().tolist()
            pred_val = allOut[i]
            if pred_val.numel() == 1:
                pred_str = f'{pred_val.item():.4f}'
            else:
                pred_str = f'[{pred_val.shape}]'
            rows.append([
                f'{original}',
                f'<span class="{css}">{recon}</span>',
                f'{label}',
                pred_str,
                f'<span class="{css}">{"Yes" if match else "No"}</span>',
            ])
            TheMessage(f"  Input: {original:30s} -> Reconstructed: {recon:30s} Predicted: {pred_str} {'OK' if match else 'MISMATCH'}")

        TheReport.add_table(
            "Input vs Reconstructed",
            ["Input", "Reconstructed", "Label", "Predicted", "Match"],
            rows)

        # Buffer reconstruction via nWhere byte offsets (non-differentiable display).
        # _materialize_recovered_input() runs the deferred word-recovery
        # decode (reverse() defers it -- it is report-only and its
        # per-word .item() would break the brick CUDA-graph contract).
        recovered_meta = self.perceptualSpace._materialize_recovered_input()
        if use_lex_recon and recovered_meta is not None:
            buf_size = max(len(test_input[0].tolist()) if isinstance(test_input[0], torch.Tensor) else 64, 64)
            buffer_strings = self.perceptualSpace.reconstruct_to_buffer(buf_size=buf_size)
            buf_rows = []
            total_chars = 0
            matching_chars = 0
            for i in range(len(test_input)):
                original = self._bytes_to_text(test_input[i])
                buf_recon = buffer_strings[i] if i < len(buffer_strings) else ""
                orig_stripped = original.rstrip('\x00')
                n = max(len(orig_stripped), len(buf_recon))
                chars_match = sum(
                    a == b for a, b in zip(orig_stripped.ljust(n, '\x00'),
                                           buf_recon.ljust(n, '\x00')))
                total_chars += n
                matching_chars += chars_match
                acc = chars_match / max(n, 1) * 100
                css = "match" if acc > 90 else "mismatch"
                buf_rows.append([
                    f'{orig_stripped}',
                    f'{buf_recon}',
                    f'<span class="{css}">{acc:.0f}%</span>',
                ])
                TheMessage(f"  Buffer: {orig_stripped:30s} -> {buf_recon:30s} ({acc:.0f}% char accuracy)")
            overall_acc = matching_chars / max(total_chars, 1) * 100
            buf_rows.append(["<strong>Overall</strong>", "", f"<strong>{overall_acc:.1f}%</strong>"])
            TheReport.add_table(
                "Buffer Reconstruction (nWhere placement)",
                ["Original", "Buffer", "Char Accuracy"],
                buf_rows)

            # Piecewise token-level metrics (no buffer shadowing)
            meta = recovered_meta if isinstance(recovered_meta, dict) else {}
            orig_tokens = meta.get('tokens', None)
            if orig_tokens is not None:
                pw_rows = []
                total_tok = 0
                match_tok = 0
                for i in range(min(len(test_input), len(orig_tokens))):
                    original = self._bytes_to_text(test_input[i]).rstrip('\x00')
                    tokens = orig_tokens[i]
                    n_tok = len(tokens)
                    n_match = 0
                    for word, offset in tokens:
                        if word in ("", "\x00"):
                            continue
                        total_tok += 1
                        # Check if this word appears at this offset in original
                        if offset is not None and 0 <= offset <= len(original) - len(word):
                            if original[offset:offset + len(word)] == word:
                                n_match += 1
                                match_tok += 1
                    tok_acc = n_match / max(n_tok, 1) * 100
                    TheMessage(f"  Piecewise: {original:30s} -> {n_match}/{n_tok} tokens matched ({tok_acc:.0f}%)")
                pw_overall = match_tok / max(total_tok, 1) * 100
                TheMessage(f"  Piecewise overall: {match_tok}/{total_tok} ({pw_overall:.0f}%)")

            # Push reconstructed data to TheData
            self.inputSpace.data.reconstructed_input = buffer_strings

        # Push reconstructed output predictions to TheData
        if allOut is not None:
            self.inputSpace.data.reconstructed_output = [
                allOut[i].detach().cpu() for i in range(allOut.shape[0])]

class BasicModel(BaseModel):
    """Core model: assembles Spaces into a forward and (optionally) reverse pipeline.

    The forward pass flows:
        InputSpace -> PerceptualSpace -> ConceptualSpace -> SymbolicSpace -> OutputSpace

    The reverse pass mirrors it:
        OutputSpace -> SymbolicSpace -> ConceptualSpace -> PerceptualSpace -> InputSpace

    Higher-order processing (conceptualOrder) inserts additional
    Percept/Concept/Symbol cycles between the first SymbolicSpace and OutputSpace,
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
               conceptualOrder=1,
               model_type="simple", data=None,
               reconstruction_scale=0.5, what_scale=0.7, where_scale=0.2, when_scale=0.1):
        """Build the full space hierarchy from architecture parameters.

        Always dispatches to ``_create_per_stage``: per-stage with
        T=conceptualOrder is the single construction path. At T=1 it
        reduces to a single ConceptualSpace + SymbolicSpace stage,
        producing the same observable output as the legacy flat path.

        Args:
            nInput/nPercepts/nConcepts/nSymbols/nOutput: object counts per space.
            nWords: object count for the SyntacticSpace.
            conceptualOrder: number of [Percept->Concept->Symbol] cycles.
            model_type: "simple", "embedding", "passthrough", or "vq".
        """
        return self._create_per_stage(
            nInput, nPercepts, nConcepts, nSymbols, nWords=nWords,
            nOutput=nOutput, conceptualOrder=conceptualOrder,
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
        return PerceptualSpace(inputShape, spaceShape, outputShape)

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

    def _symbol_feedback_from_vectors(self, sym_vectors, n_feedback, feedback_dim):
        """Project symbolic state back to the percept-shaped feedback tensor.

        Truncates / right-pads ``sym_vectors`` to ``n_feedback`` rows.
        If dims already match ``feedback_dim``, returns the bounded
        tensor; otherwise expands per-row norms to fill the feedback
        dim so a scalar symbolic signal still drives every channel.
        """
        feedback = sym_vectors
        if feedback.shape[1] >= n_feedback:
            feedback = feedback[:, -n_feedback:, :]
        else:
            pad = torch.zeros(
                feedback.shape[0],
                n_feedback - feedback.shape[1],
                feedback.shape[2],
                device=feedback.device,
                dtype=feedback.dtype,
            )
            feedback = torch.cat([feedback, pad], dim=1)

        if feedback.shape[-1] == feedback_dim:
            return self._bound_concept_input(feedback)

        norms = feedback.norm(dim=-1, keepdim=True)
        return self._bound_concept_input(norms.expand(-1, -1, feedback_dim))


    def _derive_use_grammar(self):
        """Derive ``useGrammar`` from the configured grammar rules.

        Returns ``"all"`` when the grammar contains any non-default
        rule (anything beyond unary ``pi`` / ``sigma`` substrate
        folds), else ``"none"``. Replaces the retired
        ``<WordSpace><useGrammar>`` XML knob — the grammar XML itself
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

    def _extract_prediction_sequential(self, fwd_out):
        """Materialize OutputSpace's subspace and denormalize to task range."""
        if fwd_out is None:
            return None
        if self.outputSpace.nonlinear_output:
            outputData = fwd_out.materialize(mode="activation")
        else:
            outputData = fwd_out.materialize()
        if outputData is None:
            return None
        return self.normalizer.denormalize(outputData, which="output")

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
    def store_truths(self, entries):
        """Encode truth entries via runEpoch and store in WordSpace.truth_layer.

        Truths are processed through the full pipeline by running a
        standard inference epoch.  SymbolicSpace.forward() records raw
        activations into the TruthLayer via ``self.wordSpace.truth_layer``.
        After the epoch completes, each stored activation is scaled by
        its DegreeOfTruth.

        Args:
            entries: list of dicts with 'content' and 'trust' keys.
        """
        truth_layer = getattr(self.wordSpace, 'truth_layer', None) if self.wordSpace is not None else None
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
            trusts.append(float(trust))
        if not texts:
            return

        # 2. Reset truth store, enable accumulation, run full pipeline
        truth_layer.clear()
        prev_accum = self.symbolicSpace.accumulateTruth
        self.symbolicSpace.accumulateTruth = 1.0
        self.eval()
        self.set_sigma(0)
        try:
            with torch.no_grad(), TheData.runtime_batch(texts):
                self.runEpoch(batchSize=len(texts), split="runtime")
        finally:
            self.symbolicSpace.accumulateTruth = prev_accum

        # 3. Apply DoT to each stored activation
        n = min(truth_layer.count.item(), len(trusts))
        for i in range(n):
            truth_layer.truths[i] *= trusts[i]

        # 4. Attach sources/trusts for clarification surfacing, run the
        # consistency report, and cache any clarification messages on
        # the model for the serve layer to expose.
        stored_count = truth_layer.count.item()
        truth_layer._sources = (
            list(texts[:stored_count])
            + [None] * max(0, stored_count - len(texts))
        )
        truth_layer._trusts = (
            list(trusts[:stored_count])
            + [None] * max(0, stored_count - len(trusts))
        )
        basis = getattr(getattr(self.symbolicSpace, 'subspace', None),
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

    def infer(self, text, max_length=None, mode=None):
        """IR-mode infill inference.

        Lexes the input, embeds it, applies ``mask_rate`` random
        masking to the embedded substrate (NULL_PERCEPT replacement),
        runs one forward pass, and decodes the body's prediction at
        each masked position via nearest-neighbor lookup against the
        lexicon.  Returns ``(slot_index, original_token, predicted_token)``
        triples.

        Sentence-level generation (the legacy AR / ARIR chat loop) is
        now provided by ``generate_sentence`` via ``InterSentenceLayer``
        (ARMA(p, q) over sentence reps).  ``mode`` is accepted for
        back-compat with callers that still pass ``mode='IR'``; any
        other value raises.
        """
        del max_length  # retained for signature back-compat only
        if mode is not None and str(mode).upper() != 'IR':
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

        mask_pos = self._ir_mask_positions
        if (mask_pos is None or pred_full is None
                or not bool(mask_pos.any())):
            return []

        # Lex output for the original tokens at each slot.
        peer = self._peer_perceptual if hasattr(self, '_peer_perceptual') else None
        if peer is None:
            peer = self.perceptualSpace
        codebook = peer.subspace.what
        last_meta = getattr(peer, '_forward_input', None) or {}
        all_tokens = last_meta.get('tokens') or [[]]
        tokens0 = all_tokens[0] if all_tokens else []

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

        Steps (per plan §4):
          1. Ask ``self.wordSpace.discourse.predict_next()`` for the
             ARMA-predicted next sentence rep ``s_hat_{t+1}``.
          2. Lift it through ``InterSentenceLayer.cast`` into
             ``concept_dim`` and stage on ``ConceptualSpace._c_prior``
             so the body's forward picks it up as a sentence-level
             prior before the codebook lookup.
          3. Run the IR forward over the seed text; sample tokens at
             masked positions by nearest-neighbor lookup in the
             perceptual codebook.
          4. Pool the produced S-tier root into ``s_t`` and call
             ``discourse.observe(s_t)`` to commit to the AR ring.

        This is a minimal scaffold: a single IR forward + one-shot
        decode of every masked position.  Iterative mask-and-resample
        is left to a later refinement.

        Returns a list of decoded tokens (the predictions at the
        masked positions of the seed).
        """
        del max_chars  # reserved for the iterative variant
        discourse = (self.wordSpace.discourse
                     if self.wordSpace is not None else None)
        # Stage the C-prior when the discourse layer has primed
        # history (cold-start: no prior, the seed text is the only
        # signal).
        if discourse is not None and discourse.predictor is not None:
            s_hat = discourse.predict_next()
            if s_hat is not None and discourse.cast is not None:
                prior = discourse.cast(
                    s_hat if s_hat.dim() == 2 else s_hat.unsqueeze(0))
                self.conceptualSpace._c_prior = (
                    prior * float(self.sentence_priming_scale))
        # Run the IR forward over the seed text.
        out = self._infer_ir(seed_text)
        # Commit the produced sentence to the ARMA ring.
        if discourse is not None and self._current_discourse_s is not None:
            discourse.observe(self._current_discourse_s)
        return out

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

        Replaces embeddings at random positions with NULL_PERCEPT.
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
        if not hasattr(codebook, 'null_percept_idx'):
            return  # numeric mode / non-Embedding codebook

        B, K, D = event.shape
        dev = event.device

        # Sample mask positions [B, K] bool.
        rate = float(self.mask_rate)
        if rate <= 0.0 or rate > 1.0:
            return
        mask = torch.bernoulli(
            torch.full((B, K), rate, device=dev)).bool()

        # Exclude padding slots (codebook index 0 == byte \x00 sentinel).
        active = getattr(percept_subspace, '_active', None)
        if (active is not None and active.dim() == 3
                and active.shape[-1] >= 1):
            what_idx = active[:, :K, 0].long()
            if what_idx.shape == mask.shape:
                mask = mask & (what_idx != 0)

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
        null_vec = codebook.getW()[codebook.null_percept_idx]  # [nWhat]
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

        Strict gate, **all configs**: the recon-then-eliminate program
        drove the forward to **0 graph breaks** (non-grammar *and*
        grammar; ``total graph breaks: 0; graphs: 1``), so we compile
        with ``fullgraph=True`` unconditionally -- any regression that
        reintroduces a host sync / dynamic-shape op / data-dependent
        control-flow break now *raises* at compile time instead of
        silently degrading to eager (or wasting GPU hours on a
        non-capturable run). The grammar path is no longer
        ``@torch.compiler.disable``'d (a disabled call is itself a
        break under ``fullgraph=True``); its breaks were eliminated in
        the same recon loop.
        """
        from util import compile as _compile
        # torch.compile's fake-tensor tracing has incomplete MPS device
        # propagation (raises "Unhandled FakeTensor Device Propagation"
        # on ops like mul across mps/implicit-cpu tensors) -- the same
        # class of gap as MPS `_cdist_backward`. Compile on CUDA (the
        # capture target) and CPU (CI/correctness); fall back to eager
        # on MPS so the local dev path stays usable.
        if TheDevice.get().type == "mps":
            TheMessage("[compiled-step] MPS: torch.compile fake-tensor "
                       "device propagation is incomplete -- running "
                       "eager (compile active on CUDA/CPU only).")
            self._compiled_step = None
            return
        # D8 capture-gate (2026-05-19): pre-warm any LAZY-built caches
        # that the captured forward depends on, so Dynamo never traces
        # their build path. ``_stm_reducer`` constructs a
        # ``BinaryStructuredReductionLayer`` (calls ``nn.Parameter()``
        # which Dynamo refuses to trace -- "Attempted to use
        # torch.nn.Parameter() constructor with Dynamo"). Build it
        # here, eagerly, before the compile wrapper closes over
        # ``self.forward``. The build is one-shot and idempotent
        # (cached on ``_stm_reducer_cached``).
        if getattr(self, "_stm_reducer_cached", None) is None:
            try:
                self._stm_reducer()
            except Exception:
                # Build failure (degenerate grammar etc.) caches False;
                # the per-word body's reducer call returns None
                # gracefully and the SHIFT-only depth-bound stays via
                # back-pressure. Not a compile blocker.
                self._stm_reducer_cached = False
        self._compiled_step = _compile(
            self.forward, verbose=True, fullgraph=True)

    def _start_spaces_for_forward(self):
        for space in self.spaces:
            if hasattr(space, 'Start'):
                space.Start()
        self._spaces_started_for_forward = True

    def forward(self, inputData):
        """IR-only forward: stem -> body -> head.

        Dispatches to ``_forward_per_stage`` (the single per-stage
        forward path).  Within-sentence training is BERT-style masked-
        LM at the P-tier; sentence-level AR is delegated to
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

            if not isinstance(allOut, torch.Tensor) or allOut.numel() == 0:
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
            ws = self.wordSpace
            if (ws is not None and ws.chart is not None
                    and ws.chart.router_kind == "signal"):
                rules = ws.current_rules
                gen_rules = ws.generate_rules
                from Language import TheGrammar
                def _decode(rule_id):
                    rid = int(rule_id)
                    if 0 <= rid < len(TheGrammar.rules):
                        rd = TheGrammar.rules[rid]
                        return f"{rid}:{rd.canonical}"
                    return f"{rid}:?"
                TheMessage("=== Chart-extracted grammar (signal/Viterbi) ===")
                for tier, rows in (rules or {}).items():
                    TheMessage(f"  compose tier={tier!r}:")
                    for b, row in enumerate(rows):
                        decoded = [_decode(rid) for rid in row]
                        TheMessage(f"    row[{b}] = {decoded}")
                for tier, rows in (gen_rules or {}).items():
                    TheMessage(f"  generate tier={tier!r}:")
                    for b, row in enumerate(rows):
                        decoded = [_decode(rid) for rid in row]
                        TheMessage(f"    row[{b}] = {decoded}")
                router = ws.chart._signal_router
                if router is not None and getattr(router, '_last_root_state', None) is not None:
                    rs = router._last_root_state.detach()
                    TheMessage(f"  S root state shape = {tuple(rs.shape)}")
                    for b in range(rs.shape[0]):
                        vec = rs[b, 0].tolist()
                        TheMessage(f"    row[{b}] root = {[round(v, 4) for v in vec]}")
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
        """SBOW loss on percept vectors: leave-one-out centroid prediction.

        For byte lexer mode, percepts replace word embeddings as the
        unit of distributional similarity.  Each percept in a sentence
        should be predictable from the centroid of the others.

        Returns a scalar loss tensor, or None if < 2 percepts.
        """
        # Phase 1.5: the ``self.percepts`` alias was subsumed -- read the
        # PerceptualSpace terminal subspace directly. The old
        # ``hasattr(self,'percepts')`` guard meant "no forward has stamped
        # the alias yet"; the faithful equivalent is "the subspace has no
        # materializable event yet" (``materialize()`` -> None), so guard
        # on ``vecs is None`` below.
        ps_sub = self.perceptualSpace.subspace
        if ps_sub is None:
            return None
        vecs = ps_sub.materialize()                 # [B, nPercepts, dim]
        if vecs is None:
            return None
        B, N, D = vecs.shape
        if N < 2:
            return None

        # Leave-one-out centroids per sentence: [B, N, D]
        total = vecs.sum(dim=1, keepdim=True)       # [B, 1, D]
        centroids = (total - vecs) / (N - 1)        # [B, N, D]

        # Cosine similarity loss: each percept should match its centroid
        c_norm = F.normalize(centroids, dim=-1)
        v_norm = F.normalize(vecs, dim=-1)
        cos_sim = (c_norm * v_norm).sum(dim=-1)     # [B, N]

        # Non-negative cosine loss: 0 means each percept matches the
        # leave-one-out centroid exactly.
        return (1 - cos_sim).mean()

    def accumulate_output_symbol_residual(self, outputTensor, outputDataPred):
        """Use supervised output targets to form a symbol-space residual.

        The target output is reversed through OutputSpace into the symbol
        domain, then compared with the output-facing symbols from the
        forward pass.  This provides a primary symbol residual even for
        continuous paths that intentionally avoid codebook quantization.
        """
        if not self.reversible or not hasattr(self, 'symbolicSpace'):
            return
        residual_scale = getattr(
            self.symbolicSpace, 'output_symbol_residual_scale', 0.0)
        if residual_scale <= 0.0:
            return
        try:
            pred_symbols = self.symbolicSpace.subspace.materialize()
        except Exception:
            return
        if pred_symbols is None or pred_symbols.numel() == 0:
            return
        n = min(self.nSymbols, pred_symbols.shape[1])
        pred_symbols = pred_symbols[:, :n, :]

        target_event = outputTensor.to(outputDataPred.device)
        while target_event.dim() < outputDataPred.dim():
            target_event = target_event.unsqueeze(-1)
        target_event = target_event.expand_as(outputDataPred)

        target_event_norm = self.normalizer.normalize(target_event, which="output")
        self.outputSpace.subspace.set_event(target_event_norm)
        try:
            target_space = self.outputSpace.reverse(self.outputSpace.subspace)
            target_symbols = target_space.materialize()
        finally:
            self.outputSpace.subspace.set_event(outputDataPred)
        if target_symbols is None or target_symbols.numel() == 0:
            return
        n = min(n, target_symbols.shape[1])
        if n <= 0:
            return
        terms = self.symbolicSpace._compute_symbol_terms(
            pred_symbols[:, :n, :],
            target=target_symbols[:, :n, :],
            residual_scale=residual_scale,
        )
        self.symbolicSpace._emit_symbol_terms(
            self.outputSpace.subspace, terms)

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

    # -- per-step lifecycle (model-level) ------------------------------
    #
    # ``runBatch`` orchestrates a step but must talk only to the *model*,
    # never reach into model *contents* (``self.wordSpace.discourse``,
    # ``self.inputSpace`` ...). These three methods are that boundary:
    # the model owns its discourse / input-staging lifecycle; runBatch
    # just calls ``self._begin_step`` / ``self._discourse_arma_loss`` /
    # ``self._end_step``. Per-batch timing (identical to the prior
    # inline code) -- NOT the grammar ``soft_reset`` (that fires only on
    # chart sentence-completion and never per-batch on the IR/MM_5M
    # path, so relocating discourse there would silently stop ARMA).

    def _begin_step(self, inputTensor):
        """Eager pre-forward staging for the compiled step.

        Compiled path only: move the input onto the compute device
        (outside the trace -- ``TheDevice.get()`` is unproxyable
        inside it), park the lex+embed stem subspace, and stage the
        inter-sentence ARMA prediction so the traced forward just reads
        parked tensors. No-op on the eager/uncompiled path (the forward
        lexes inline, behaviour unchanged). Returns the (possibly
        device-moved) input.
        """
        if self._compiled_step is None:
            return inputTensor
        if isinstance(inputTensor, torch.Tensor):
            inputTensor = inputTensor.to(TheDevice.get())
        self._staged_in_sub = self.inputSpace.forward(inputTensor)
        disc = (self.wordSpace.discourse
                if self.wordSpace is not None else None)
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
        return inputTensor

    def _end_step(self):
        """Per-step teardown: drop the staging parked by ``_begin_step``
        (consume-once; eager, post-forward)."""
        self._staged_in_sub = None
        disc = (self.wordSpace.discourse
                if self.wordSpace is not None else None)
        if disc is not None:
            disc.clear_staged_prediction()

    def _discourse_arma_loss(self):
        """Inter-sentence ARMA(p, q) loss term for this step, or ``None``.

        Encapsulates ``InterSentenceLayer`` so runBatch sees only a
        model-level loss contribution. Must be called post-body /
        pre-backward: the term trains the ARMA predictor, and
        ``observe`` also commits the sentence rep + residual into the
        per-row rings (vectorized, sync-free).
        """
        disc = (self.wordSpace.discourse
                if self.wordSpace is not None else None)
        if disc is None or self._current_discourse_s is None:
            return None
        return disc.observe(self._current_discourse_s)

    def runBatch(self, train=True, batchNum=0, batchSize=10, split="train",
                 optimizer=None, batch_override=None, progress=None):
        """Run a single batch: forward pass, loss, and (if training) backward + step.

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

        # Pre-allocate per-batch state OUTSIDE the compiled forward.
        # ``WordSpace.ensure_microbatch`` allocates ``_stm_fired`` /
        # ``_last_svo`` / ``_svo_valid`` / ``_recent_count`` etc. on
        # first call (and on shape changes); when those allocations
        # happen INSIDE a torch.compile region, CUDAGraph capture
        # takes ownership of the underlying memory, so the next
        # replay's allocation overwrites the previous step's state
        # and the next attribute read raises ``RuntimeError: Error:
        # accessing tensor output of CUDAGraphs that has been
        # overwritten by a subsequent run.`` Hoisting the call up
        # here keeps the resulting tensors Python-owned.
        if self.wordSpace is not None and not inference_only:
            ws = self.wordSpace
            try:
                if isinstance(inputTensor, torch.Tensor):
                    B_pre = int(inputTensor.shape[0])
                else:
                    B_pre = int(len(inputTensor))
            except Exception:
                B_pre = None
            if B_pre is not None:
                ws.ensure_microbatch(B_pre, 1)

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
            # Per-step lifecycle is model-owned (``_begin_step`` /
            # ``_end_step``): runBatch stays out of model *contents*.
            # ``_begin_step`` does the eager pre-forward staging on the
            # compiled path -- device move (outside the trace), lex+embed
            # stem park, ARMA prediction stage -- so the traced forward
            # only reads parked tensors; no-op on the eager/uncompiled
            # path (forward lexes inline, behaviour unchanged).
            inputTensor = self._begin_step(inputTensor)
            _fwd = self._compiled_step or self.forward
            forwardInput, symbols, predictions, _ = _fwd(inputTensor)
            self._end_step()
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
                                 "symbolicSpace"):
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

            # IR head is purely a side channel — the prediction lives in
            # the masked P-tier slots, not in the head.  ``lossOut`` is
            # kept wired (with zero weight) for code-path symmetry with
            # downstream Error tracking; the gradient signal flows
            # through ``lossIn`` only.
            lossOut = torch.zeros((), device=TheDevice.get())
            output_weight = 0.0
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
            mask_pos = self._ir_mask_positions
            pre_mask = self._ir_pre_mask_input
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
            # REPLACING the interim P-tier ``compute_masked`` masked-LM.
            # The whole-slab / non-grammar (``_per_word_enabled=False``)
            # path is UNCHANGED -- it keeps ``compute_masked`` exactly,
            # byte-identical. ``_d3_active`` / ``_d3_word_metric`` are
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
                K = min(pred_full.shape[1], mask_pos.shape[1],
                        pre_mask.shape[1])
                lossIn = self.loss.compute_masked(
                    pred_full[:, :K, :], pre_mask[:, :K, :],
                    mask_pos[:, :K])
            else:
                lossIn = torch.zeros((), device=TheDevice.get())
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
            try:
                if forwardInput is not None:
                    rev_mode = (getattr(self, 'reconstruct', 'none')
                                or 'none').lower()
                    rev_sub = None
                    if rev_mode in ('concepts', 'both'):
                        for cs_sub in reversed(getattr(self, '_cs_cache', [])):
                            if cs_sub is not None:
                                rev_sub = self._run_pipeline_rev_from_concepts(cs_sub)
                                break
                    if rev_sub is None:
                        rev_sub = self._run_pipeline_rev(self.outputSpace.subspace)
                    rev_ev = (rev_sub.materialize()
                              if rev_sub is not None
                              and hasattr(rev_sub, 'materialize')
                              else None)
                    fwd_ev = (forwardInput.materialize()
                              if hasattr(forwardInput, 'materialize')
                              else forwardInput)
                    if (rev_ev is not None and fwd_ev is not None
                            and rev_ev.dim() == fwd_ev.dim()
                            and rev_ev.dim() == 3):
                        Kr = min(rev_ev.shape[1], fwd_ev.shape[1])
                        Dr = min(rev_ev.shape[-1], fwd_ev.shape[-1])
                        lossRev = self.loss.compute(
                            rev_ev[:, :Kr, :Dr],
                            fwd_ev[:, :Kr, :Dr].detach())
            except Exception:
                # Reverse round-trip is approximate through averaged
                # loops; never let a reconstruction edge case stop the
                # training step.
                lossRev = torch.zeros((), device=TheDevice.get())
            TheError.add(
                "reconstruction_reverse", lossRev,
                weight=float(getattr(self.loss, 'reconstruction_scale', 0.0)
                             or 0.0),
                space="InputSpace", category="reconstruction",
            )

            # Forward-only C/S reconstruction loss
            # (``<reconstruct>concepts|symbols|both</...>``).  Targets
            # are derived from the pre-mask P-tier event lifted on the
            # fly through ``sigma_percept`` (plan §"Reconstruction-loss
            # target shape" Option B); the model's own sigma_percept
            # supplies the lift so no second body forward is needed.
            # The C/S losses fire alongside the IR P-tier loss above
            # and are blended via ``self.loss.reconstruction_scale``.
            recon_mode = (getattr(self, 'reconstruct', 'none')
                          or 'none').lower()
            # `recon_mode in (...)` and the None checks are Python (no
            # sync); the `bool(mask_pos.any())` skip was a host sync.
            # Run unconditionally and sanitize the empty-mask NaN per
            # term below (sync-free; 0.0 == not-added for the summed
            # TheError.total). See doc/BrickHostSyncStatus.md residual D.
            if (recon_mode in ('concepts', 'symbols', 'both')
                    and mask_pos is not None and pre_mask is not None):
                # Lift pre_mask through sigma_percept once; this is
                # the C-tier target both ``concepts`` and ``symbols``
                # consume (``symbols`` could route through SS for a
                # tighter target — left to a later refinement; the
                # plan accepts the C-tier lift as the Option-B
                # approximation for both).
                sigma_p = getattr(self.conceptualSpace,
                                  'sigma_percept', None)
                target_c = (sigma_p(pre_mask) if sigma_p is not None
                            else None)
                recon_weight = self.loss.reconstruction_scale
                if (recon_mode in ('concepts', 'both')
                        and target_c is not None and self._cs_cache):
                    cs_sub = self._cs_cache[-1]
                    pred_c = (cs_sub.materialize()
                              if cs_sub is not None
                              and hasattr(cs_sub, 'materialize')
                              else None)
                    if pred_c is not None and pred_c.dim() == 3:
                        K_c = min(pred_c.shape[1],
                                  target_c.shape[1],
                                  mask_pos.shape[1])
                        loss_c = self.loss.compute_masked(
                            pred_c[:, :K_c, :],
                            target_c[:, :K_c, :].detach(),
                            mask_pos[:, :K_c])
                        TheError.add(
                            "reconstruction_c", loss_c,
                            weight=recon_weight,
                            space="ConceptualSpace",
                            category="reconstruction",
                        )
                if (recon_mode in ('symbols', 'both')
                        and target_c is not None and self._ss_cache):
                    ss_sub = self._ss_cache[-1]
                    pred_s = (ss_sub.materialize()
                              if ss_sub is not None
                              and hasattr(ss_sub, 'materialize')
                              else None)
                    if pred_s is not None and pred_s.dim() == 3:
                        # Match dims: target_c is at C-tier width;
                        # SS may have a different muxed width.  Use
                        # the leading shared dims.
                        K_s = min(pred_s.shape[1],
                                  target_c.shape[1],
                                  mask_pos.shape[1])
                        loss_s = self.loss.compute_masked(
                            pred_s[:, :K_s, :],
                            target_c[:, :K_s, :].detach(),
                            mask_pos[:, :K_s])
                        TheError.add(
                            "reconstruction_s", loss_s,
                            weight=recon_weight,
                            space="SymbolicSpace",
                            category="reconstruction",
                        )

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
                        space="WordSpace", category="embedding",
                    )

            # Inter-sentence ARMA(p, q) loss term -- model-owned
            # (``_discourse_arma_loss``): predicts ``s_hat_t`` from the
            # lagged reps/residuals, returns the per-batch MSE, and
            # commits the new rep + residual into the rings (vectorized,
            # sync-free). Cold-start rows return ``None``. Computed here,
            # post-body / pre-backward, so the term trains the predictor.
            arma_loss = self._discourse_arma_loss() if train else None

            totalLoss = self.loss.total(lossOut, lossIn, sbow)
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
            # Phase 3: every stage's SymbolicSpace.forward wrote its
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

            # Truth-modulated loss: delegated to WordSpace since the
            # TruthLayer lives there.  WordSpace handles the empty-store
            # guard internally; we only gate on ``train``.  The falsity
            # penalty operand is the last cached symbol activation --
            # stored truths are also recorded from symbol space, so both
            # sides of the disjunction live in the basis's native space.
            if train and self.wordSpace is not None:
                symbol_acts = None
                if hasattr(self, 'symbol_states') and self.symbol_states:
                    symbol_acts = self.symbol_states[-1]
                totalLoss = self.wordSpace.truth_modulated_loss(
                    totalLoss,
                    symbolic_space=self.symbolicSpace,
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
                gate_l1 = self.wordSpace.gate_l1_loss(
                    lam=getattr(self, 'gate_l1_lambda', 0.0))
                if gate_l1 is not None:
                    totalLoss = totalLoss + gate_l1
                    TheError.add(
                        "gate_l1", gate_l1,
                        weight=getattr(self, 'gate_l1_lambda', 0.0),
                        space="WordSpace", category="reg")

                # Sparse-MoE load-balance penalty (Shazeer et al. 2017).
                # Penalises CV² of per-rule activation counts so the
                # noisy top-K gating doesn't collapse onto 1-2 rules.
                # Active only when chartTopK > 0 AND loadBalanceWeight > 0.
                lb_w = getattr(self, 'load_balance_weight', 0.0)
                if (lb_w > 0.0 and self.wordSpace is not None
                        and getattr(self.wordSpace, 'chart', None)
                        is not None):
                    lb_loss = self.wordSpace.chart.load_balance_loss(
                        weight=lb_w)
                    if isinstance(lb_loss, torch.Tensor):
                        totalLoss = totalLoss + lb_loss
                        TheError.add(
                            "load_balance", lb_loss,
                            weight=lb_w,
                            space="WordSpace", category="reg")
                    # Reset between batches so the next batch's count
                    # reflects only its own gating distribution.
                    self.wordSpace.chart.reset_load_count()

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
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                totalLoss.backward()
                self._assert_finite_train_state("after backward")
                if self.ergodic:
                    self.paramUpdate()
                optimizer.step()
            self._assert_finite_train_state("after optimizer.step")
            self._clamp_symbolic_codebook()
            self._training_step_count = (
                int(getattr(self, "_training_step_count", 0) or 0) + 1
            )
            self._maybe_save_periodic_checkpoint()

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
        #     ``wordSpace.drain_sentence_completed()`` →
        #     ``wordSpace.soft_reset(b)``.
        #   * ``truth_layer.compact()`` (one host sync per tick, kept
        #     outside the brick).
        # See doc/plans/2026-04-26-rolling-cursor-doc-streaming-handoff.md.

        # Memory-leak diagnostics (perf-notes/08-*). Three independently
        # gated probes; each is a no-op without its env var.
        if os.environ.get("BASIC_PROFILE_DIAG"):
            try:
                ws_diag = self.wordSpace
                tl_diag = getattr(ws_diag, 'truth_layer', None) if ws_diag is not None else None
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
        """Drain ``wordSpace._sentence_completed`` and fire per-row soft reset.

        Called by the outer doc-streaming loop after ``runBatch`` returns
        (and *after* ``dispatch_per_row_reset`` so a hard-reset row's soft
        signal is dropped — hard subsumes soft).
        """
        ws = self.wordSpace
        if ws is None or not hasattr(ws, 'drain_sentence_completed'):
            return
        completed = ws.drain_sentence_completed()
        for b in completed:
            ws.soft_reset(batch=b)

    def post_tick_compact(self):
        """Run post-tick host work: truth-layer compaction.

        Lives outside the compute brick so the brick body remains
        sync-free. Called once per tick by the outer doc-streaming loop
        after ``runBatch`` returns.

        Also severs any cross-batch autograd graph carried by persistent
        state (``WordSpace._disc_pred``, ``_last_svo``; Basis transient
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
        ws = self.wordSpace
        if ws is not None:
            tl = getattr(ws, 'truth_layer', None)
            if tl is not None and hasattr(tl, 'compact'):
                tl.compact(min_trust=0.5)
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
            (``WordSpace._disc_pred`` / ``_disc_conf``,
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
        ws = self.wordSpace
        if ws is not None:
            if getattr(ws, '_disc_pred', None) is not None:
                ws._disc_pred = ws._disc_pred.detach()
            if getattr(ws, '_disc_conf', None) is not None:
                ws._disc_conf = ws._disc_conf.detach()
        for sp_attr in ("inputSpace", "perceptualSpace",
                        "conceptualSpace", "symbolicSpace", "outputSpace"):
            sp = getattr(self, sp_attr, None)
            if sp is None:
                continue
            for tn in ("_ar_embedded", "_embedded_input",
                       "_cached_embedding"):
                t = getattr(sp, tn, None)
                if t is not None and torch.is_tensor(t) and t.is_floating_point():
                    setattr(sp, tn, t.detach())

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

        In inference mode (split="runtime", no optimizer): skips loss
        construction, output accumulation, progress printing, and CBOW
        updates. Routes ARIR runtime mode through ``arir_step`` directly.

        Args:
            optimizer: pre-built Adam optimizer (persistent across epochs).
                       Pass None for evaluation mode.
            batchSize: requested batch size (capped by split length).
            split: "train", "test", or "validation"

        Returns (output_loss, reconstruction_loss, all_predictions, last_reconstruction).
        For inference mode, returns (0, 0, [], []).
        """
        training = optimizer is not None
        inference = split == "runtime" and not training
        self.train(training)
        self.outputSpace.clearBatchResults()
        ws = self.wordSpace
        if ws is not None and getattr(ws, 'discourse', None) is not None:
            ws.discourse.reset()
        ctx = torch.no_grad() if not training else nullcontext()

        # Inference fast path: skip loss construction and accumulation.
        # Runtime ARIR / inference route through arir_step (stateful across calls).
        if inference:
            with ctx:
                batchNum = 0
                while True:
                    result, batchNum = self.runBatch(
                        train=False, batchNum=batchNum, batchSize=batchSize,
                        split=split,
                    )
                    if result is None:
                        break
                    # Inference output is consumed at the call site if
                    # needed; do NOT retain results here -- nothing reads
                    # outputSpace._batch_results, and the BatchResult
                    # tuples each carry ~one batch of detached tensors.
            return 0, 0, [], []

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
            # ~17 MB measured on MM_5M, accumulating over the epoch.
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
            # pure overhead: ~0.5 MB / runBatch on MM_5M, growing across
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
        #     boundary. Sized to ``nObj - 1`` so the lex's reserved
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
            # InputSpace.outputShape[0] (= ``nObj``) is the byte-buffer
            # width the lex emits. Under the §8g/§EOS-removal change
            # the lex no longer reserves a slot for an EOS sentinel
            # (the slot held a null-embedding indistinguishable from
            # the codebook's null padding -- no reader consumed it as
            # a stop signal). Sizing the slab to ``nObj`` keeps the
            # cursor byte-faithful through the lex: every byte emitted
            # ends up in a real token, and the assert in
            # ``InputSpace._lex_batch`` (``n_tokens <= nObj``) holds
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
                result, _ = self.runBatch(
                    train=training, batchNum=step,
                    batchSize=B_step, split=split,
                    optimizer=optimizer,
                    batch_override=(inputTensor, outputTensor),
                    progress=progress_frac,
                )
                if result is not None:
                    record(result)
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
                if global_max is not None:
                    self._train_batches_seen += 1

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
               conceptualOrder=1,
               model_type="simple", data=None,
               reconstruction_scale=0.5,
               what_scale=0.7, where_scale=0.2, when_scale=0.1,
               **kwargs):
        """Wire the full per-stage space stack from architecture parameters.

        Builds Input / Perceptual / Conceptual stages (one per
        ``conceptualOrder`` step) / Symbolic / Output, plus the optional
        WordSpace, subsymbolic, and pipeline modules. Mutates ``self``
        extensively (sets every ``self.*Space`` attribute, ``self.spaces``,
        ``self.wordSpace``, ``self.reversible``, etc.).
        """
        self.spaces = []
        self.wordSpace = None  # wired below once the home spaces exist
        self.reversible = True
        self.nInput = nInput
        self.nPercepts = nPercepts
        self.nConcepts = nConcepts
        self.nSymbols = nSymbols
        self.nOutput = nOutput
        self.nWords = nWords
        self.data = data
        self.model_type = model_type
        self.lexer = TheXMLConfig.space("InputSpace", "lexer")
        self.ergodic = TheXMLConfig.get("architecture.ergodic")
        self.processSymbols = TheXMLConfig.get("architecture.processSymbols")
        self.certainty = TheXMLConfig.get("architecture.training.certainty")
        # InputSpace.codebook defaults to false; see the matching note in
        # BasicModel.create.
        self.codebook = Space.normalize_codebook_mode(
            TheXMLConfig.space("InputSpace", "codebook", default=False)) != "none"
        self.perceptCodebook = TheXMLConfig.space("PerceptualSpace", "codebook")
        self.conceptCodebook = TheXMLConfig.space("ConceptualSpace", "codebook")
        self.conceptualOrder = conceptualOrder

        # Monotonic SigmaLayer weights (W >= 0). Mirrors PiLayer's monotonic
        # flag; when True, invertible SigmaLayers use NonNegativeInvertibleLinearLayer.
        self.monotonic = bool(
            TheXMLConfig.get("architecture.monotonic", default=False))
        # ``useGrammar`` XML knob retired 2026-05-13: derived from the
        # configured grammar instead. See ``_derive_use_grammar``.
        self.useGrammar = self._derive_use_grammar()
        # ``<architecture><reconstruct>`` selects the forward-only
        # reconstruction-loss target (``none`` / ``symbols`` /
        # ``concepts`` / ``both``).  ``output`` was retired
        # 2026-05-14 with the reverse pipeline.  ``runBatch`` reads
        # this attribute to gate the C/S reconstruction term.
        self.reconstruct = str(
            TheXMLConfig.get("architecture.reconstruct", default="none")
            or "none").lower()
        # Gate-L1 sparsity lambda for LiftLayer / LowerLayer raw_gate
        # parameters. 0.0 (default) disables the penalty; configs that
        # use lift/lower opt in via <gateL1Lambda> in <architecture>.
        self.gate_l1_lambda = float(
            TheXMLConfig.get("architecture.gateL1Lambda", default=0.0) or 0.0)
        # Sparse-MoE load-balance loss weight (Shazeer et al. 2017).
        # Penalises CV² of per-rule activation counts so the chart's
        # noisy top-K gating spreads mass across rules.  No-op unless
        # both <loadBalanceWeight> > 0 AND <chartTopK> > 0.
        self.load_balance_weight = float(
            TheXMLConfig.get(
                "architecture.loadBalanceWeight", default=0.0) or 0.0)
        # thoughtFree is structurally equivalent to conceptualOrder=0: no
        # higher-order P/C/S cycles. Reject the nonsense combination early.
        TheXMLConfig.require(
            lambda cfg, _ug=self.useGrammar, _co=self.conceptualOrder:
                _ug != "thoughtFree" or _co == 0,
            f"useGrammar='thoughtFree' requires conceptualOrder=0 "
            f"(got useGrammar={self.useGrammar!r}, "
            f"conceptualOrder={self.conceptualOrder})"
        )
        # Truth integration config (optional -- absent in BasicModel.xml)
        self.truth_bias_scale = float(TheXMLConfig.get("architecture.truthBiasScale", default=0.1) or 0.1)
        self.luminosity_weight = float(TheXMLConfig.get("architecture.LuminosityWeight", default=0.1) or 0.1)
        self.universality_weight = float(TheXMLConfig.get("architecture.UniversalityWeight", default=0.1) or 0.1)
        # Quaternary-corner balance knobs (see BuddhistParallels.md for
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

        self.loss = ModelLoss(
            reconstruction_scale=reconstruction_scale,
            what_scale=what_scale,
            where_scale=where_scale,
            when_scale=when_scale,
            nOutput=nOutput,
            conceptualOrder=conceptualOrder,
            nWhere=TheXMLConfig.get("architecture.nWhere"),
            nWhen=TheXMLConfig.get("architecture.nWhen"),
        )
        # Optional swappable loss head fed STM snapshots during the
        # body forward (Phase 3, 2026-05-12). Installed by
        # ``embed.py embed_pretrain`` for CBOW-over-STM pretraining;
        # AR training leaves it None and keeps using ``outputSpace``
        # for the loss.
        self.loss_head = None

        # Resolve dims, chaining through the pipeline (nDim=0 -> same as input dim)
        # Helpers _resolve_dim / _obj_size / _nvec live on BaseModel.
        input_dim   = self._resolve_dim("InputSpace",      1)
        percept_dim = self._resolve_dim("PerceptualSpace", input_dim)
        concept_dim = self._resolve_dim("ConceptualSpace", percept_dim)
        symbol_dim  = self._resolve_dim("SymbolicSpace",   concept_dim)
        output_dim  = self._resolve_dim("OutputSpace",     symbol_dim)

        obj_input   = self._obj_size("InputSpace")
        obj_percept = self._obj_size("PerceptualSpace")
        obj_concept = self._obj_size("ConceptualSpace")
        obj_symbol  = self._obj_size("SymbolicSpace")
        obj_output  = self._obj_size("OutputSpace")

        nvec_input   = self._nvec("InputSpace",      nInput)
        nvec_percept = self._nvec("PerceptualSpace", nPercepts)
        nvec_concept = self._nvec("ConceptualSpace", nConcepts)
        nvec_symbol  = self._nvec("SymbolicSpace",   nSymbols)
        nvec_output  = self._nvec("OutputSpace",     nOutput)

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
        spaceShape_symbol  = [nvec_symbol,  symbol_dim]
        spaceShape_output  = [nvec_output,  output_dim]

        rawInputShape = [nInput, input_dim]
        self.inputSpace = self._make_input_space(
            rawInputShape, spaceShape_input, inputShape,
            model_type=model_type)

        # Input -> Percept (uses _make_perceptual_space so demuxed
        # configs route to ModalSpace).
        self.perceptualSpace = self._make_perceptual_space(
            inputShape, spaceShape_percept, perceptShape)
        if isinstance(self.perceptualSpace.vocabulary, Embedding):
            object.__setattr__(self.inputSpace, '_peer_perceptual',
                               self.perceptualSpace)

        conceptInputShape = [nPercepts, percept_dim + obj_percept]

        # ConceptualSpace output shape uses the explicit XML values
        # ``<nOutput>`` (already resolved to ``nConcepts`` upstream)
        # and ``<nOutputDim>`` (when supplied) directly. Earlier
        # versions of this code derived N from a volume-preserving
        # ``input_volume // nOutputDim`` formula; that's wrong because
        # the C-tier codebook can re-dimension between input and
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
            n_stages = self.conceptualOrder
            self._level_shapes_list = self._level_shapes(
                nPercepts, percept_dim + obj_percept, n_stages,
                width_mode=self._conceptual_width_mode())
        else:
            n_stages = self.conceptualOrder
            self._level_shapes_list = None

        # -- Per-stage arrays: independent ConceptualSpace / SymbolicSpace
        # per stage. The pipeline flows stage-by-stage with no shared
        # per-level views; cross-forward autograd retention vanishes.
        # conceptualOrder=0 still needs one stage so the pre-seed C->S
        # pass (test_merged_loop.test_unified_loop_conceptualorder_zero_pre_seed_only)
        # has a concreteSpace/symbolicSpace to populate. The j-iteration
        # count reported to runBatch is still the configured value.
        T = max(1, int(n_stages))
        self.conceptualSpaces = nn.ModuleList()
        self.symbolicSpaces = nn.ModuleList()
        for t in range(T):
            is_last = (t == T - 1)
            if self.useGrammar == "all":
                # Grammar path: each stage halves N (except last). Shapes
                # follow _level_shapes but the per-stage ConceptualSpace /
                # SymbolicSpace are plain.
                #
                # Width contract (H3, 2026-05-18): the per-stage
                # ConceptualSpace input is the true upstream percept
                # width ``percept_dim + obj_percept`` (``cs_in``), and
                # its output is the conceptual content width
                # ``concept_dim`` (``cs_out``).  ``sigma_percept`` is
                # therefore the percept->concept *lift*
                # (``percept_dim+obj_percept -> concept_dim``), not a
                # width-preserving map.  ``concept_dim`` is the BARE
                # ``<ConceptualSpace><nDim>`` -- where/when ride muxed
                # *inside* it (``SubSpace`` derives
                # ``nWhat = concept_dim - nWhere - nWhen``), exactly as
                # the passing grammar reference MM_xor does
                # (``cs_out[1] == concept_dim``; obj_concept is NOT
                # added on this path).  This is what the C->P feedback
                # gate (``PerceptualSpace.pi_concept.nInput ==
                # <ConceptualSpace><nDim>``), the Phase-2A.5 symbol
                # snap (``SymbolicSpace.subspace.what.W`` width ==
                # ``symbol_dim``), and ``SymbolicSpace.forward``'s
                # ``[B, N, concept_dim]`` pass-through contract
                # (validate_config: ``effective_concept_dim ==
                # symbol_dim``) all require.  For configs where
                # ``concept_dim == percept_dim + obj_percept`` (e.g.
                # MM_xor: 10 == 10) this is identical to the prior
                # width-preserving shapes (no-op); for MM_5M it
                # activates the previously-dropped C->P feedback +
                # snap (1024 vs the old 10).
                n_t = nPercepts >> t
                d_in = percept_dim + obj_percept
                d_out = concept_dim
                cs_in = [n_t, d_in]
                cs_out = [n_t, d_out] if is_last else [n_t >> 1, d_out]
                ss_in = cs_out[:]
                ss_out = cs_out[:]
            else:
                # Plain path: all stages share the legacy conceptInputShape /
                # conceptOutputShape. No N-halving.
                cs_in = list(conceptInputShape)
                cs_out = list(conceptOutputShape)
                ss_in = list(conceptOutputShape)
                ss_out = list(symbolShape)

            # Non-codebook spaces require nVectors (spaceShape[0]) ==
            # nActive (outputShape[0]); resize the per-stage codebook shape
            # to match the halved N.
            stage_space_concept = [cs_out[0], spaceShape_concept[1]]
            stage_space_symbol = [ss_out[0], spaceShape_symbol[1]]
            # Right-half loopback widening retired (see ConceptualSpace
            # docstring): per-order input sourcing replaces the concat,
            # so the C-tier PiLayer input width is just nInputDim.
            cs = ConceptualSpace(cs_in, stage_space_concept, cs_out,
                                 stage_idx=t,
                                 is_last=is_last)
            ss = SymbolicSpace(ss_in, stage_space_symbol, ss_out,
                               conceptualSpace=cs)
            # Non-owning back-ref CS->SS (mirrors the perceptualSpace_ref
            # idiom below): object.__setattr__ so it is NOT registered as
            # an nn.Module child of cs. Read-only structural pairing.
            object.__setattr__(cs, 'symbolicSpace_ref', ss)
            # Per-stage flags consumed by build_pipelines / forward.
            ss.is_last = is_last
            ss.quantize = not is_last
            self.conceptualSpaces.append(cs)
            self.symbolicSpaces.append(ss)

        # Backwards-compat aliases: read-only callers (e.g.
        # wordSpace.truth_layer = self.symbolicSpace) see the terminal stage.
        self.conceptualSpace = self.conceptualSpaces[-1]
        self.symbolicSpace = self.symbolicSpaces[-1]

        # VQ-VAE EMA / growing-codebook knob overrides per space.
        # Each Space's ``subspace.what`` may carry an internal
        # ``VectorQuantize`` (``.vq``); when XSD knobs are set, override
        # the defaults baked in at ``Codebook.addVectors`` time so
        # configs can tune EMA decay and dead-code retirement without
        # code edits.  ``codebookGrowthEpsilon`` is stashed as
        # ``.growth_epsilon`` on the VQ for runBatch to consult.
        for _sect, _sp in (
            ("PerceptualSpace", self.perceptualSpace),
            ("ConceptualSpace", self.conceptualSpace),
            ("SymbolicSpace", self.symbolicSpace),
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

        # Cross-space forward inputs (perceptual + symbolic loop into C,
        # C→P feedback into P) are now passed as explicit ``forward``
        # arguments by the recurrent cell in ``_forward_body`` -- no
        # post-construction ``symbolicSpace_ref`` / ``perceptualSpace_ref``
        # / ``conceptualSpace_ref`` plumbing. The SymbolicSpace lexicon
        # ref below is structural (vocabulary ownership), not forward
        # input, and is kept.
        # Lexicon ownership (post-lexicon-migration): wire every
        # SymbolicSpace stage to PerceptualSpace so ``S.vocabulary``
        # and the orthographic-API methods reach the physical Embedding
        # that lives on PerceptualSpace for input-pipeline reasons.
        for ss in self.symbolicSpaces:
            object.__setattr__(ss, 'perceptualSpace_ref',
                               self.perceptualSpace)

        # No SyntacticSpace -- syntax is handled by Grammar centrally.
        self.syntacticSpace = None

        # Output: primary path is IS→PS→CS→OS, so OutputSpace consumes
        # the terminal ConceptualSpace output (SymbolicSpace is the
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

        # Build WordSpace -- the unified container for grammar
        # infrastructure (WordSubSpace, three SyntacticLayers, the
        # TruthLayer, and conditionally the DiscourseSpace substrate).
        # Its ``__init__`` configures the grammar, sizes the word
        # buffer from SymbolicSpace's column layout, builds each tier's
        # SyntacticLayer, and back-wires the home spaces so
        # compose/decompose routes through ``self.wordSpace``.
        self.wordSpace = WordSubSpace(
            perceptualSpace=self.perceptualSpace,
            conceptualSpace=self.conceptualSpace,
            symbolicSpace=self.symbolicSpace,
            nPercepts=nPercepts,
            nConcepts=nPercepts,
            nSymbols=nSymbols,
            concept_dim=concept_dim + obj_concept,
            symbol_dim=symbol_dim + obj_symbol,
        )
        # The conceptual basis honours the ``architecture.monotonic``
        # knob rather than being unconditionally bitonic: monotone
        # (W>=0) is order-preserving, which the parthood predicate
        # (``Ops.part``) requires for the ramsified symbolic codebook
        # to match a symbol mapped through the subsymbolic loop across
        # orders. Default False -> unchanged (bitonic) behavior.
        self.conceptualSpace.subspace.basis.monotonic = self.monotonic

        self.spaces.extend([self.inputSpace, self.perceptualSpace])
        self.spaces.extend(list(self.conceptualSpaces))
        self.spaces.extend(list(self.symbolicSpaces))
        self.spaces.extend([self.outputSpace])
        self.spaces.append(self.wordSpace)

        self.inputSpace.outputSpace = self.outputSpace
        # Seed the pipeline context: InputSpace stamps every outgoing
        # subspace's ``wordSpace`` with this reference so downstream stages
        # read ``vspace.wordSpace`` instead of reaching back through a
        # Model back-channel.
        self.inputSpace.set_word_space(self.wordSpace)

        # Phase 1: wire a Normalizer onto every space so spaces can call
        # self.normalizer.{normalize,denormalize} instead of the TheData global.
        self.normalizer = Normalizer(TheData)
        for space in self.spaces:
            space.normalizer = self.normalizer
            sub = getattr(space, 'subspace', None)
            if sub is not None:
                sub.normalizer = self.normalizer
                # Stamp the model's (build-stable) WordSpace onto every
                # persistent subspace ONCE, eagerly. The traced forward's
                # copy_context no longer re-assigns it (assigning a
                # Module trips nn.Module.__setattr__ -> fullgraph break,
                # and registering it as a submodule infinitely recurses
                # model.to()); since wordSpace never changes after build,
                # this eager stamp keeps every downstream
                # ``vspace.wordSpace`` read valid with no in-trace work.
                sub.attach_wordSpace(self.wordSpace)

        # Precompute partition boundaries for partitioned symbolSum
        self._partitions = self._order_partitions(symbol_dim + obj_symbol,
                                                   self.conceptualOrder)
        self.symbol_states = []
        # Per-step lifecycle slots, initialized explicitly so every
        # read is a plain attribute access (no getattr-with-default):
        #   _staged_in_sub      -- lex+embed stem subspace parked by
        #                          _begin_step for the compiled forward
        #                          (None == not staged: eager/uncompiled
        #                          path lexes inline).
        #   _compiled_step      -- the torch.compiled callable, or None
        #                          (set by enable_compiled_step; None
        #                          here so the eager path is valid even
        #                          if that is never called).
        #   _current_discourse_s -- S-tier sentence rep stashed by the
        #                          forward for the post-body ARMA term.
        self._staged_in_sub = None
        self._compiled_step = None
        self._current_discourse_s = None
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
                      cs -> chart-at-C -> [merge?] -> ss
                    K-axis flatten/restore is hoisted into _forward_body.
            head  : FlattenK(outputSpace)

        T = len(self.conceptualSpaces). Each per-stage SymbolicSpace
        output is captured into ``self._ss_cache[t]`` (plain Python list)
        by ``_forward_body``. ``_forward_per_stage`` reads them to
        rebuild ``symbol_states`` (un-flattened back to [B,K,N,D]).
        ``self.symbol_cache`` is a property returning the terminal
        entry — same role as before, no longer a CachePoint module.
        """
        T = len(self.conceptualSpaces)
        use_grammar_merge = (self.useGrammar == "all")

        # PerceptualSpace quantize: matches legacy's percept_quantize logic.
        # Reversible+invertible configurations skip codebook quantization to
        # preserve exact-invertibility of the perceptual step.
        self.perceptualSpace.quantize = not (
            self.reversible and getattr(self.perceptualSpace, "invertible", False))

        # WordSpace ownership hangs off the terminal symbolic stage
        # (Rule 3): ``self.symbolicSpace`` is the alias for
        # ``self.symbolicSpaces[-1]``.
        self.symbolicSpace.wordSpace = self.wordSpace

        # Per-stage symbol output capture. Plain Python list populated
        # by ``_forward_body`` each call; ``symbol_cache`` property
        # resolves to the terminal entry. Replaces the prior CachePoint
        # ModuleList (CachePoints existed only to fit nn.Sequential).
        self._ss_cache = [None] * T
        # Per-stage concept output capture (parallel to ``_ss_cache``)
        # for the ``<reconstruct>concepts</...>`` / ``both`` forward-
        # only reconstruction loss in ``runBatch``.
        self._cs_cache = [None] * T

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
        #   "ss":      SymbolicSpace     (required)
        # The legacy ``"reparse": ChartCompose`` entry was retired
        # 2026-05-12 alongside chart-at-stem -- the chart now fires
        # uniformly at C-tier inside ``_forward_body`` for every stage.
        self.body_stages = nn.ModuleList()
        for t in range(T):
            stage = nn.ModuleDict()
            stage["cs"] = self.conceptualSpaces[t]
            if use_grammar_merge:
                stage_n = base_n // (2 ** t)
                stage["merge"] = GrammarMergeGlue(
                    stage_idx=t, initial_n=stage_n,
                    is_last=(t == T - 1))
            stage["ss"] = self.symbolicSpaces[t]
            self.body_stages.append(stage)

        all_spaces = ([self.inputSpace, self.perceptualSpace]
                      + list(self.conceptualSpaces)
                      + list(self.symbolicSpaces)
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
        """Terminal stage's captured symbolic subspace, or None if not yet set.

        Replaces the prior ``self.symbol_cache = self.ss_caches[T-1]``
        CachePoint. Read-only — populated by ``_forward_body``.
        """
        return self._ss_cache[-1] if self._ss_cache else None





    def _forward_head(self, sub):
        """Head: outputSpace forward."""
        return self.outputSpace(sub)

    @staticmethod
    def _empty_subspace(d=1):
        """A zero-length SubSpace seed: ``is_empty()`` is True so the
        recurrent cell's pass-0 inputs short-circuit (PS uses pi_input
        alone; CS uses the perceptual primary alone; SS returns it
        unchanged)."""
        return SubSpace(inputShape=(0, d), outputShape=(0, d),
                        nInputDim=d, nOutputDim=d)

    @staticmethod
    def _zero_symbol_subspace(ss, ctx_sub):
        """"Nothing to quantize -> pass zeros": a correctly-batched
        zero SymbolicSpace output.

        When SS is fed an empty seed (e.g. conceptualOrder=1, where the
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
        N = int(ss.outputShape[0])
        D = int(ss.subspace.muxedSize)
        # Write onto SS's OWN subspace (mirrors the held_at_zero idiom)
        # so ``model.symbolicSpace.subspace`` carries the zero event +
        # activation -- the symbol-state contract inspects that object,
        # not just the cached return. Place the zeros on the model's
        # device/dtype (held_at_zero uses ``device=sample.device,
        # dtype=sample.dtype``); a bare ``torch.zeros`` is CPU and
        # mismatches a CUDA model (``cuda:0 vs cpu`` on metalbaby).
        ss.subspace.copy_context(ctx_sub)
        ss.subspace.set_event(torch.zeros(B, N, D, device=dev, dtype=dt))
        return ss.subspace

    def _forward_body(self, in_sub):
        """Recurrent cell: IS→PS→CS→OS with CS→PS and CS→SS loops.

        Per pass t over ``self.body_stages`` (T = conceptualOrder):

          * PS and SS run **in parallel** (no intra-pass dependency --
            both consume the prior pass's CS views):
              ``PS_sub = perceptualSpace.forward(in_sub, prevCS_forPS)``
              ``SS_sub = ss.forward(prevCS_forSS)``
          * ConceptualSpace combines them (CS is the per-pass terminal):
              ``CS_sub = cs.forward(PS_sub, SS_sub)``
          * ``cs._subspaceForPS`` / ``cs._subspaceForSS`` (read
            post-merge) feed the next pass's subsymbolic / symbolic
            loops.

        Pass 0 seeds ``prevCS_*`` with empty subspaces so the cell
        degrades to PS-only / primary-only (matches the old
        ``ref is None`` cold start). The head consumes the terminal CS
        (``return last_cs``); SymbolicSpace is the symbolic loop leg,
        off the head path. ``in_sub`` is the stable InputSpace subspace
        (InputSpace ran once in the stem).

        When ``self.loss_head`` is set the post-body STM snapshot feeds
        the head and the loss is stashed on ``self._loss_head_loss``.
        """
        # 2b-1 capstone: per-word IR-reconstruction loop. When the
        # InputSpace per-word cursor is enabled (grammar configs;
        # ``_per_word_enabled = (useGrammar != 'none')``, wired in 2a),
        # the FRONT of the forward changes from whole-slab to an
        # internal per-word loop -- ONE forward = ONE sentence. Each
        # ground-truth word ``[B,1,D]`` is pumped through the SAME
        # per-stage PS->CS->SS computation and the resulting per-word
        # concept is SHIFTed onto ConceptualSpace.stm; the NULL seal
        # (next_word -> None) ends the loop. The accumulated STM then
        # feeds the EXISTING compose-to-S / chart / head / reverse() /
        # IR-loss TAIL entirely unchanged (the guiding principle:
        # minimize new training-critical surface -- reuse the existing
        # IR machinery verbatim, change only how the representation is
        # built). ``_per_word_enabled=False`` (non-grammar, model.xml)
        # falls through to the whole-slab path below, byte-identical.
        isp = self.inputSpace
        if isp is not None and getattr(isp, "_per_word_enabled", False):
            return self._forward_body_per_word(in_sub)

        prevCS_forPS = self._empty_seed_ps
        prevCS_forSS = self._empty_seed_ss
        last_cs = None
        for t, stage in enumerate(self.body_stages):
            cs = stage["cs"]
            ss = stage["ss"]
            # Gate PerceptualSpace's serial warm-path: it is an
            # AR-streaming optimisation across forward calls and must
            # not splice on in-forward recurrent passes >0 (each pass
            # carries distinct C→P feedback). ``recur_pass`` lives on
            # WordSpace; PerceptualSpace.forward reads it via the
            # ``subspace.wordSpace`` back-reference.
            if self.wordSpace is not None:
                self.wordSpace.recur_pass = int(t)
            else:
                self.perceptualSpace._recurrent_pass_idx = t
            PS_sub = self.perceptualSpace.forward(in_sub, prevCS_forPS)
            if t == 0:
                # IR masked-LM mask applied once, on the first pass's
                # perceptual event, before CS consumes it
                # (``_ir_pre_mask_input`` / ``_ir_mask_positions`` are
                # read by ``runBatch``).
                self.create_ir_mask(PS_sub)
            SS_sub = ss.forward(prevCS_forSS)
            CS_sub = cs.forward(PS_sub, SS_sub)
            self._cs_cache[t] = CS_sub
            self._chart_compose_at_C(stage_idx=t)
            if "merge" in stage:
                CS_sub = stage["merge"](CS_sub)
            # Read the two C views after the optional merge (merge
            # mutates the CS subspace in place; the C->S view aliases
            # it so it reflects post-merge N). The cross-pass handoff
            # lives directly on the persistent ``cs._subspaceForPS`` /
            # ``cs._subspaceForSS`` SubSpaces (mutated in place by
            # ConceptualSpace.forward); no SentenceState carrier — the
            # reverse path generates its own estimates of these
            # intermediate views (reconstruction contract).
            prevCS_forPS = cs._subspaceForPS
            prevCS_forSS = cs._subspaceForSS
            # "Nothing to quantize -> pass zeros": SS is inert when its
            # input was the empty seed (conceptualOrder=1, or pass 0).
            # CS already consumed SS_sub as-is above (pass-0 math
            # unchanged); cache a correctly-batched zero symbol so the
            # symbol-state contract holds for the head / discourse.
            if SS_sub.is_empty():
                SS_sub = self._zero_symbol_subspace(ss, in_sub)
            self._ss_cache[t] = SS_sub
            last_cs = CS_sub
        # Reset so standalone PerceptualSpace.forward calls (and the
        # next forward's pass 0) see the AR-streaming serial warm path.
        if self.wordSpace is not None:
            self.wordSpace.recur_pass = 0
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
    # 2b-2-i: bounded soft shift-reduce producer (STM -> single S).
    # ------------------------------------------------------------------
    def _stm_reducer(self):
        """Lazily build + cache the bounded-reduce scorer/combiner.

        The bounded soft REDUCE reuses the EXISTING reduction math
        verbatim -- it is NOT re-authored here (two-loop spec
        Phase-1-D §3 "STM-7": "score top-r of STM with the existing
        soft reducer ... the ``BinaryStructuredReductionLayer``
        anchors, Language.py:1608"; "parent = Σ_op weight_op ·
        op(left,right)  # fixed op axis, one weighted reduce"). One
        ``BinaryStructuredReductionLayer`` is constructed from the
        grammar's S-tier arity-2 reduce ops, each wrapped with the
        SAME ``_BinaryGrammarOpAdapter`` ``_wire_signal_router_grammar_ops``
        uses (so ``op(left,right)`` dispatches into the registered
        host fold -- IntersectionLayer/UnionLayer/...). The layer's
        own forward computes ``chosen_reduced = Σ_op softmax(reduce_
        score)_op · op(left,right)`` over the fixed op axis (one
        weighted reduce, no shared in-place accumulator -- the proven
        ``_superposed_op`` pattern), which IS the spec's ``parent``.

        Returns ``None`` when no S-tier arity-2 host op is available
        (degenerate grammar) so the caller can skip the REDUCE
        (SHIFT-only, depth still bounded by back-pressure -- a forced
        no-op reduce simply pops the older slot).
        """
        cached = getattr(self, "_stm_reducer_cached", None)
        if cached is not None:
            return cached if cached is not False else None
        ss = self.symbolicSpace
        sl = getattr(ss, "syntacticLayer", None)
        if sl is None or not hasattr(sl, "_by_name"):
            self._stm_reducer_cached = False
            return None
        try:
            from Language import (TheGrammar,
                                  BinaryStructuredReductionLayer,
                                  _BinaryGrammarOpAdapter)
        except ImportError:
            self._stm_reducer_cached = False
            return None
        ops = []
        for rid in range(len(TheGrammar.rules)):
            rdef = TheGrammar.rules[rid]
            if int(getattr(rdef, "arity", 1)) != 2:
                continue
            mn = getattr(rdef, "method_name", None)
            host = sl._by_name.get(mn) if mn else None
            if host is None or not hasattr(host, "compose"):
                continue
            ops.append(_BinaryGrammarOpAdapter(host))
        if not ops:
            self._stm_reducer_cached = False
            return None
        D_c = int(self.conceptualSpace.stm.concept_dim)
        layer = BinaryStructuredReductionLayer(
            d_model=D_c, ops=ops, r_copy=1, temperature=1.0)
        layer = layer.to(self.conceptualSpace.stm._buffer.device)
        # Register as a child so its anchors/op params are optimised
        # with the rest of the model (grad must shape the reducer).
        self.add_module("_stm_reducer_module", layer)
        self._stm_reducer_cached = layer
        return layer

    def _stm_bounded_reduce_step(self):
        """ONE statically-unrolled, masked REDUCE micro-step (forced).

        Operates on the live STM ``[B, cap, D]`` buffer + the tensor
        depth (``stm._depth``). De-sync-safe: pure ``torch.where`` /
        gather / scatter over a fixed ``[B, cap, D]`` slab + a tensor
        depth -- NO ``.item()``, NO data-dependent trip count (spec
        STM-7: "Pop/push are masked roll/scatter/where over [B,7,D]
        + a tensor depth -- never pop().item()").

        Mechanism (spec STM-7 controller):
          * Take the top-2 STM constituents (slots ``d-2`` and ``d-1``).
          * ``parent = Σ_op weight_op · op(left,right)`` via the cached
            ``BinaryStructuredReductionLayer`` (its ``soft`` slab over a
            length-2 window IS the single weighted reduce over the
            fixed op axis -- reused, not re-authored).
          * Snap the parent C<->S via ``_stm_symbolic_roundtrip`` (the
            existing idempotent primitive; passthrough for non-
            ProjectionBasis ``.what`` -- MM_5M/MM_xor ``quantize``).
          * Replace slot ``d-2`` with ``parent``, clear slot ``d-1``,
            ``d <- d - 1``.

        Both call sites (back-pressure at capacity AND the NULL-seal
        sweep) want an unconditional fold whenever a row has >=2
        constituents -- the FORCED reduce that replaces the legacy
        ``ShortTermMemory.push`` capacity RuntimeError (spec STM-7:
        "at d==7 the best reduce is FORCED"). A scored soft gate
        ``g in (0,1)`` (opportunistic mid-sentence reduce) is the
        2b-2 refinement; 2b-2-i's bound is the forced fold. Rows with
        depth < 2 are no-ops (masked to gate 0). Returns nothing;
        mutates the STM buffer/depth via out-of-place masked ops.
        """
        stm = self.conceptualSpace.stm
        reducer = self._stm_reducer()
        buf = stm._buffer                                  # [B, cap, D]
        B, cap, D = buf.shape
        depth = stm._depth                                 # [B] long
        device = buf.device
        # Rows that CAN reduce: at least 2 constituents on the stack.
        can = (depth >= 2)                                 # [B] bool
        # Gather the top-2 (left = older = slot d-2, right = top =
        # slot d-1). Clamp indices for the depth<2 rows (masked out
        # by ``can`` below so the bogus gather never takes effect).
        idx_r = (depth - 1).clamp(min=0)                   # [B]
        idx_l = (depth - 2).clamp(min=0)                   # [B]
        rows = torch.arange(B, device=device)
        left = buf[rows, idx_l]                            # [B, D]
        right = buf[rows, idx_r]                            # [B, D]
        if reducer is not None:
            window = torch.stack([left, right], dim=1)      # [B, 2, D]
            _hard, soft, _routing = reducer(window)
            # soft[:, 0, :] is the folded root of the length-2 window
            # (the recursive-reduction leading position -- the
            # LanguageLayer.compose root-state convention,
            # Language.py:1100-1104). This is the spec's ``parent``.
            parent = soft[:, 0, :]                           # [B, D]
        else:
            # Degenerate grammar (no S-tier arity-2 op): the FORCED
            # reduce is a structural pop of the top onto its parent
            # (mean of the two constituents -- the minimal lattice
            # join), keeping depth bounded without a host op.
            parent = 0.5 * (left + right)
        # C<->S idempotent snap of the parent (reused primitive;
        # passthrough for non-ProjectionBasis ``.what``).
        snapped = self._stm_symbolic_roundtrip(
            parent.unsqueeze(1))                            # [B, 1, D]
        if snapped is not None and snapped.dim() == 3:
            parent = snapped[:, 0, :]
        # Forced fold gate: 1.0 where the row can reduce (>=2
        # constituents), 0.0 otherwise. The ``can`` mask is what makes
        # the static unroll safe -- a short row is a pure no-op for
        # this step. (A scored soft gate g in (0,1) is the 2b-2
        # opportunistic-reduce refinement; 2b-2-i forces the fold.)
        g_col = can.to(buf.dtype).view(B, 1)               # [B, 1]
        # STM[d-2] <- g·parent + (1-g)·STM[d-2]  (masked: only the
        # can-rows' slot d-2 changes).
        new_left = g_col * parent + (1.0 - g_col) * left   # [B, D]
        buf_new = buf.clone()
        buf_new[rows, idx_l] = torch.where(
            can.view(B, 1), new_left, left)
        # Clear the consumed top slot (d-1) for the can-rows.
        cleared = torch.where(
            can.view(B, 1), torch.zeros_like(right), right)
        buf_new[rows, idx_r] = cleared
        stm._buffer = buf_new
        # d <- d - 1 for rows that reduced (g==1 there); tensor op,
        # no sync. ``_max_depth_host`` is the host mirror snapshot()
        # reads; it tracks the max depth -- a reduce never increases
        # it, and the post-sweep depth is read explicitly by the
        # caller, so leave the mirror as the high-water mark (the
        # snapshot slab is zero-padded past the live depth anyway).
        dec = can.to(depth.dtype)                          # [B] {0,1}
        stm._depth = depth - dec

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

        Returns the single root idea ``S`` ``[B, D]`` (slot 0 of every
        row after the sweep -- the start-symbol root constituent) plus
        the post-sweep depth tensor for verification.
        """
        stm = self.conceptualSpace.stm
        cap = int(stm.capacity)
        for _ in range(max(0, cap - 1)):
            self._stm_bounded_reduce_step()
        # After cap-1 forced folds every row's stack is collapsed to
        # its bottom slot: that single slot is the sentence idea S
        # (the start-symbol root -- this producer reduces toward
        # ``Grammar.start_symbol`` by construction; the only S-tier
        # reduce ops the grammar exposes all have lhs == start_symbol
        # for MM_xor/MM_5M, so the folded root's category tracks
        # start_symbol).
        S = stm._buffer[:, 0, :]                            # [B, D]
        return S, stm._depth

    def _mphf_route_word(self, word_slice, cursor_pos):
        """Rework A: route the per-word percept -> concept through
        ``idx = MPHF(percept_bytes)`` -> the D2 table row.

        ``word_slice`` is the ``[B,1,D]`` muxed per-word slice the
        InputSpace cursor returned (``_ar_embedded[:, p:p+1, :]``);
        ``cursor_pos`` is its per-word slot index ``p`` (the same
        ``[B,nObj,M]`` per-word slot axis ``_embed_bpe``'s
        ``set_forward_content`` writes the per-word frozen lexicon row
        onto -- ``PerceptualSpace.subspace._active[:,p,0]`` (==
        ``per_word_first``, the byte-derived O(1) frozen ``key_to_index``
        resolution that IS the MPHF index for the in-vocab percept; the
        ``_ar_embedded`` per-word cursor axis shares this exact slot
        axis -- verified: both ``[B,1024,*]`` for MM_5M). The standalone
        static byte->row ``PerceptualSpace.mphf_index`` is the explicit
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
        # ``_active`` (``[B,nObj,M]``; ``[...,0]`` == ``per_word_first``,
        # the byte-derived frozen ``key_to_index`` row). It shares the
        # ``_ar_embedded`` per-word cursor slot axis (both ``[B,nObj,*]``).
        sub = getattr(ps, "subspace", None)
        active = getattr(sub, "_active", None) if sub is not None else None
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
          * ``conceptualSpace.stm.ensure_batch + clear`` -- the
            bounded stack is sized to the live batch and emptied.
          * ``perceptualSpace._mphf_tables()`` -- lazy build-once
            cache is primed (build path iterates ``bytes`` which
            Dynamo refuses to trace).
          * ``wordSpace.recur_pass = 0`` -- invariant across the
            per-word loop (pass-index 0 throughout).
          * ``self._prev_cs_for_ps`` / ``self._prev_cs_for_ss`` pre-
            seeded to the persistent empty seeds so iteration 0's
            PS/SS forwards see a non-None feedback subspace (avoids a
            None-vs-SubSpace branch inside the captured body that
            would force a recompile after iteration 1). The body
            switches these to ``cs._subspaceForPS`` /
            ``cs._subspaceForSS`` (the persistent CS-tier storage that
            CS.forward mutates in place) right after iteration 0's
            cs.forward, so iterations 1+ pick up the in-place updates
            without any further pointer churn.
          * Fresh ``_cs_cache`` / ``_ss_cache`` lists.

        Returns ``(stm, N_target, word_carrier, in_event)`` -- the
        values ``_forward_body_per_word`` needs from the prelude.
        ``in_event`` is the materialized whole-sentence event the
        post-loop step uses to restore the InputSpace subspace.
        """
        stm = self.conceptualSpace.stm
        in_event = in_sub.materialize() if in_sub is not None else None
        if stm is not None:
            B_in = int(in_event.shape[0]) if in_event is not None else 1
            stm.ensure_batch(B_in)
            stm.clear()

        # Fresh per-forward caches (mirrors the whole-slab path).
        T_cache = len(self._ss_cache)
        self._cs_cache = [None] * T_cache
        self._ss_cache = [None] * T_cache

        # MPHF pre-warm (Dynamo-unfriendly build path).
        ps = self.perceptualSpace
        if (ps.chunking_mode == "mphf"
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

        # WordSpace per-sentence state + per-forward pre-seed. The
        # cursor / recur_pass are allocated by ``wordSpace.soft_reset``
        # (triggered by ``post_tick_compact`` when a sentence
        # completes); the first forward of the first sentence runs
        # BEFORE any soft_reset has fired, so cold-start allocation
        # falls to this prelude. ``self._prev_cs_for_ps/ss`` are
        # UNCONDITIONALLY re-seeded to the empty seeds each forward so
        # the per-word loop's iteration 0 always sees the canonical
        # empty starting context -- never a stale SubSpace from a prior
        # sentence's per-word loop. The body switches these to
        # ``cs._subspaceForPS`` / ``cs._subspaceForSS`` after the first
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
        if not getattr(self.wordSpace, '_per_sentence_initialized', False):
            self.wordSpace.soft_reset()
            self.wordSpace._per_sentence_initialized = True
        self.wordSpace.recur_pass = 0
        self._prev_cs_for_ps = self._empty_seed_ps
        self._prev_cs_for_ss = self._empty_seed_ss

        N_target = (int(in_event.shape[1])
                    if in_event is not None and in_event.dim() == 3
                    else None)
        word_carrier = in_sub

        # Static per-word loop scaffolding
        # (doc/plans/2026-05-20-static-per-word-loop-impl.md §2).
        # ``_target_cursor_length`` tells WordSpace.compose to pad the
        # S-tier rule cursor to N with ``id_SS`` (forward-only). The
        # per-iteration commit-time gate is sourced from
        # ``inputSpace._word_active_mask`` (tensor-only, compile-stable);
        # the rule cursor itself stays aligned for downstream consumers
        # but does NOT drive control flow inside the captured body.
        N_static = int(self.inputSpace.outputShape[0])
        if self.wordSpace is not None:
            self.wordSpace._target_cursor_length = N_static
        # ``_per_word_contributions`` accumulates the per-iteration
        # ``[B, D_c]`` contributions (zero at inactive batch rows /
        # padding columns) for ``torch.stack`` after the loop. Using a
        # list + stack preserves autograd flow through the per-position
        # concept outputs, where an in-place write into a non-grad
        # buffer would silently detach.
        self._per_word_contributions: list = []
        return stm, N_target, word_carrier, in_event

    def _per_word_body_step(self, w, p, gate_b_1, out_slot, active_host=True):
        """One per-word iteration of the static loop (replaces the
        legacy data-dependent ``while next_word()`` body).

        Static-N variant of the constant-shape PS/SS -> CS sub-graph
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
          out_slot: preallocated ``[B, N, D_c]`` concept buffer; this
             call writes column ``p``.

        Returns ``(CS_sub, idea_bd)`` — last_cs tracking and idea
        broadcast handle. Both are produced unconditionally (PS/SS/CS
        forwards always fire); only their downstream effects are gated.

        Capture-gate contract: still the ONLY callable that runs inside
        the middle captured graph; every helper it calls remains
        DtoH-free. ``test/test_per_word_capture_gate.py`` is the active
        gate.
        """
        isp = self.inputSpace
        cs = self.conceptualSpace
        ss = self.symbolicSpace
        stm = cs.stm
        word_carrier = isp.subspace

        # Gaussian window + MPHF route — unchanged. ``p`` is the static
        # loop position; the legacy ``_per_word_cursor - 1`` mirror is
        # retired with ``next_word()``.
        _full_seq = isp._ar_embedded
        _ctx_w = self.gaussian_window_word(_full_seq, p)
        if (_ctx_w is not None and torch.is_tensor(_ctx_w)
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
        word_sub = word_carrier

        # Read the cross-iteration C→P / C→S feedback off
        # ``self._prev_cs_for_ps/ss``. The prelude seeds these to the
        # empty seeds for iteration 0; the tail of this method switches
        # them to the persistent ``cs._subspaceForPS/SS`` storage that
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
        prev_ss_event_snap = (
            prevCS_forSS._event
            if (prevCS_forSS is not None
                and getattr(prevCS_forSS, '_event', None) is not None)
            else None)

        PS_sub = self.perceptualSpace.forward(word_sub, prevCS_forPS)
        SS_sub = ss.forward(prevCS_forSS)
        CS_sub = cs.forward(PS_sub, SS_sub)

        # Masked-blend the persistent CS carriers' new events with the
        # snapshots so padding columns preserve the recurrent state
        # from the prior iteration (no ConceptualSpace averaging leak
        # from SS at zero input). Shapes must match — on iteration 0
        # the prev is the empty seed (length 0) and the blend is
        # skipped; the cs.forward result propagates as-is.
        self._maybe_blend_event(
            cs._subspaceForPS, prev_ps_event_snap, gate_b_1)
        self._maybe_blend_event(
            cs._subspaceForSS, prev_ss_event_snap, gate_b_1)

        # Switch the prev pointer to the persistent CS storage so the
        # next iteration's first reads see CS.forward's in-place
        # writes. After this line ``self._prev_cs_for_ps`` aliases
        # ``cs._subspaceForPS``; subsequent iterations read the same
        # object whose ``_event`` cs.forward keeps fresh.
        self._prev_cs_for_ps = cs._subspaceForPS
        self._prev_cs_for_ss = cs._subspaceForSS

        # Terminal-stage caches — only update at active host positions so
        # padding iterations don't overwrite the last-real-word state
        # that downstream reconstruction loss reads. ``active_host`` is
        # a Python bool derived host-side from
        # ``inputSpace._valid_len_host``; the body itself stays loop-
        # body sentence-length-free, but the cache writes here are
        # gated. Dynamo specializes on the host int.
        if SS_sub.is_empty():
            SS_sub = self._zero_symbol_subspace(ss, word_sub)
        if active_host:
            self._cs_cache[-1] = CS_sub
            self._ss_cache[-1] = SS_sub

        idea_bd = None
        if CS_sub is not None:
            idea = CS_sub.materialize()
            if (idea is not None and idea.dim() == 3
                    and idea.shape[1] >= 1):
                idea_bd = idea[:, 0, :]                 # [B, D_c]
                if stm is not None and active_host:
                    # Capacity back-pressure check fires here; the host
                    # mirror is also advanced per real push so the
                    # check is accurate inside the loop.
                    # ``push_step_masked`` is gated on ``active_host``
                    # so padding iterations never index past depth and
                    # never trip a capacity OOB.
                    if stm._max_depth_host >= stm.capacity:
                        self._stm_bounded_reduce_step()
                        stm._max_depth_host = stm.capacity - 1
                    stm.push_step_masked(idea_bd, gate_b_1)
                    stm._max_depth_host = stm._max_depth_host + 1
                # Masked contribution: at inactive batch rows / padding
                # columns the contribution is zero so it doesn't push
                # bogus state into downstream concept reads. ``out_slot``
                # is the caller's per-iteration accumulator (a list);
                # the caller is responsible for ``torch.stack``ing it
                # back to ``[B, N, D_c]`` post-loop so autograd flows
                # through the per-position contributions.
                contribution = torch.where(
                    gate_b_1, idea_bd, torch.zeros_like(idea_bd))
                if out_slot is not None and isinstance(out_slot, list):
                    out_slot.append(contribution)
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

          ``while (w := inputSpace.next_word()) is not None:``
            * the ``[B,1,D]`` ground-truth word ``w`` is run through the
              SAME per-stage PS->CS->SS recurrent cell as the whole-slab
              path (``conceptualOrder`` passes, with the C->P / C->S
              feedback identical to ``_forward_body``), but with a
              single-word ``in_sub`` instead of the whole slab;
            * the resulting per-word terminal concept (the last pass's
              CS event, ``[B,1,D_c]``) is SHIFTed onto
              ``conceptualSpace.stm`` via ``push_step`` (the existing
              vectorised single-step push);
          the NULL seal (``next_word`` returns ``None``) ends the loop.

        Then the accumulated STM feeds the **EXISTING** compose-to-S
        chart (``_chart_compose_at_C``) and the method returns the
        terminal CS subspace so the existing ``_forward_head`` +
        ``runBatch`` P-tier masked-LM IR-loss + ``reverse()`` TAIL run
        **unchanged** (guiding principle: minimise new training-critical
        surface -- reuse the existing IR machinery verbatim; only the
        representation-build FRONT changes from whole-slab to per-word
        accumulation into STM).

        IR-loss faithfulness: the IR mask is created exactly as in the
        whole-slab path -- via the unchanged ``self.create_ir_mask`` --
        on the **first word's** pass-0 perceptual event (the faithful
        per-word analogue of "pass 0's perceptual event").
        ``runBatch``'s masked-LM loss then reads the post-body
        PerceptualSpace event vs the snapshotted pre-mask embedding at
        the mask positions, byte-identically to today (no new loss
        surface, no compose/reverse loss rewiring).

        STM is sentence-scoped: ``ConceptualSpace.stm`` is sized to
        ``<WordSpace><wMax>`` (sentence length -- the CKY+resize-
        equivalent baseline per the two-loop spec's Phase-1-D §3); the
        bounded soft REDUCE-to-<=7 over STM is the SEPARATE 2b-2
        increment (out of scope here). ``push_step`` requires depth to
        start at 0, so the STM is cleared once at loop entry (the
        sentence-boundary clear that ``ConceptualSpace.Reset(hard=True)``
        also performs).
        """
        # Prelude: STM resize+clear, MPHF pre-warm, WordSubSpace
        # per-sentence invariants + CS-feedback pre-seed
        # (``self._prev_cs_for_ps/ss``), fresh _cs/_ss caches. Hoisted
        # into ``_per_word_prelude`` so the capture-gate test can
        # replay the same boundary-side contract. Now also: allocate /
        # zero the preallocated ``_per_word_concept_buf`` [B, N, D_c]
        # and set ``wordSpace._target_cursor_length = N`` so compose
        # pads the S-tier rule cursor to N with id_SS.
        stm, N_target, word_carrier, in_event = self._per_word_prelude(in_sub)

        # The per-word loop IS the recurrence: it replaces the
        # whole-slab cell's ``conceptualOrder`` pass loop with a
        # word-indexed loop. Per the ratified design each word is ONE
        # PS->CS->SS step (a single ``sigma_percept`` Σ-lift -> one
        # concept), and the C->P / C->S feedback carries word-to-word
        # (the cross-step carrier), mirroring exactly how
        # ``_forward_body`` carries ``prevCS_*`` across its passes:
        # initialise the feedback ONCE before the loop, update it each
        # step. (Resetting the feedback per word would cold-start SS
        # every word -> SS always sees the empty seed -> the symbol
        # subspace zeroes out; the whole-slab path explicitly carries
        # this feedback across steps.)
        #
        # The canonical cell is the **TERMINAL-stage** PS/CS/SS
        # (``self.perceptualSpace`` / ``self.conceptualSpace`` /
        # ``self.symbolicSpace`` == ``*Spaces[-1]``). The whole-slab
        # path chains the ``conceptualOrder`` per-stage spaces (each a
        # DISTINCT instance with a progressively N-halved shape, the
        # terminal one keeping full N); but the per-word loop replaces
        # that stage-chaining N-bottleneck with the per-word STM
        # accumulation, so it runs the single terminal cell -- the one
        # whose shape/codebooks the ENTIRE reused tail reads
        # (``_chart_compose_at_C`` -> ``self.conceptualSpace.stm``,
        # ``symbol_cache`` -> ``self.symbolicSpace``, the head sized
        # from ``conceptualSpaces[-1].outputShape``, ``runBatch``'s
        # P-tier IR loss -> ``self.perceptualSpace.subspace``). Using a
        # non-terminal stage would leave ``self.symbolicSpace.subspace``
        # / ``self.conceptualSpace.stm`` unwritten (the bug an earlier
        # ``body_stages[0]`` cut hit). The word loop -- not a stage
        # loop -- provides the recurrence depth.
        # NOTE: cs/ss handles, prevCS_forPS/SS empty-seed init, MPHF
        # pre-warm, recur_pass reset, and STM resize+clear all folded
        # into ``_per_word_prelude`` (called above). The captured body
        # reads ``self._prev_cs_for_ps/ss`` uniformly (prelude seeds
        # them to the empty seeds; the body switches to
        # ``cs._subspaceForPS/SS`` once cs.forward has run for the
        # first time) -- no per-iteration None branch.

        # Static per-word loop (replaces the data-dependent
        # ``while next_word() is not None`` boundary). Trip count is the
        # Python int constant ``N = InputSpace.outputShape[0]``, so
        # Dynamo unrolls / specializes once and length variation no
        # longer drives recompiles. The per-iteration gate
        # ``inputSpace._word_active_mask[:, p:p+1]`` (a [B, 1] bool
        # tensor, built tensor-only in InputSpace.forward) sourcing
        # ``gate_b_1`` masks all per-iteration commits in
        # ``_per_word_body_step``.
        # See doc/plans/2026-05-20-static-per-word-loop-impl.md §2.
        N_static = int(self.inputSpace.outputShape[0])
        word_active = self.inputSpace._word_active_mask
        out_slot = self._per_word_contributions
        # ``K_host`` is the active prefix length (host int from the
        # eager-computed ``_valid_len_host``). The static-N loop runs
        # ``N_static`` iterations regardless, but cache writes and the
        # ``last_cs`` tracker gate on ``p < K_host`` so padding
        # iterations don't overwrite last-real-iter state that
        # downstream reconstruction reads.
        K_host = int(self.inputSpace._valid_len_host)

        last_cs = None
        for p in range(N_static):
            w = self.inputSpace.word_at(p)
            if w is None:
                break
            if word_active is not None:
                gate_b_1 = word_active[:, p:p + 1]
            else:
                gate_b_1 = torch.ones(
                    w.shape[0], 1, dtype=torch.bool, device=w.device)
            active_host = p < K_host
            CS_sub, idea_bd = self._per_word_body_step(
                w, p, gate_b_1, out_slot, active_host=active_host)
            if active_host and CS_sub is not None:
                last_cs = CS_sub
        word_count = K_host
        # STM host mirror is advanced inside the body's ``active_host``
        # branch per real push; no post-loop fixup needed.

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
        # PerceptualSpace.forward calls (and the next forward's pass 0)
        # see the AR-streaming serial warm path -- mirrors the
        # whole-slab path's tail.
        if self.wordSpace is not None:
            self.wordSpace.recur_pass = 0
        else:
            self.perceptualSpace._recurrent_pass_idx = 0

        # Stack the per-iteration contributions (gradient-preserving)
        # and pad / truncate to ``N_target``. Inactive batch rows /
        # padding columns contributed zero so the tail is naturally
        # zero-padded. ``copy_context`` from the last per-word CS
        # subspace keeps the pipeline wordSpace/errors/stem-route
        # contract intact.
        # See doc/plans/2026-05-20-static-per-word-loop-impl.md §2.5.
        per_word_contribs = self._per_word_contributions
        if last_cs is not None and per_word_contribs:
            stacked = torch.stack(per_word_contribs, dim=1)  # [B, n_w, D_c]
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
        # ``_forward_head`` + ``runBatch`` P-tier masked-LM IR tail
        # consumes -- the training signal is byte-identical to 2b-1.
        # The single S is PRODUCED and verified here (depth -> 1,
        # category tracks ``Grammar.start_symbol``) but NOT yet
        # consumed by the loss; 2b-2-ii rewires the loss to
        # ``reverse(S)``-vs-unmasked using exactly this S.
        if (stm is not None and word_count > 0
                and getattr(self.inputSpace, "_per_word_enabled", False)):
            S, post_depth = self._stm_reduce_to_single_S()
            # Verification handles for the end-to-end probe / future
            # 2b-2-ii consumer: the single sentence idea S [B, D_c]
            # and the post-sweep STM depth (must be 1 across rows).
            self._stm_single_S = S
            self._stm_post_depth = post_depth

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

    def _chart_compose_at_C(self, stage_idx=0):
        """Fire the chart at C-tier over ``conceptualSpace.stm`` contents.

        Populates ``wordSpace.current_rules`` for downstream SS dispatch.
        Uses :meth:`ShortTermMemory.snapshot` to obtain a single uniform
        ``[B, max_depth, D_c]`` slab (rows with shorter sentences carry
        zero-padding at the tail; handled by the chart's ``valid_mask``
        machinery).
        """
        snap = self.conceptualSpace.stm.snapshot()
        if snap is None:
            return
        self.wordSpace.compose(snap)

    def _chart_generate_from_stm(self):
        """Fire ``wordSpace.generate`` over the C-tier STM snapshot.

        Reverse-path mirror of ``_chart_compose_at_C``: populates
        ``wordSpace.generate_rules`` so each stage's reverse dispatch can
        pop them via its SyntacticLayer cursor.
        """
        ws = self.wordSpace
        if ws is None:
            return
        stm = self.conceptualSpace.stm
        if stm is None:
            return
        snap = stm.snapshot()
        if snap is None:
            return
        ws.generate(snap)

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
        cb = getattr(self.symbolicSpace.subspace, 'what', None)
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

    def _reverse_head(self, sub):
        """Reverse the head: ``outputSpace.reverse`` (OS → C tier)."""
        return self.outputSpace.reverse(sub)

    def _reverse_body(self, sub):
        """Per-stage body reverse, mirroring ``_forward_body`` order.

        Inverts the primary IS→PS→CS→OS path's body: walk stages in
        reverse, undo the optional N-halving ``merge`` then
        ``ConceptualSpace.reverse`` (C → percept tier). SymbolicSpace is
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
        for t in reversed(range(len(self.body_stages))):
            stage = self.body_stages[t]
            if "merge" in stage:
                try:
                    sub = stage["merge"].reverse(sub)
                except (RuntimeError, AssertionError, ValueError):
                    pass
            try:
                sub = stage["cs"].reverse(sub)
            except (RuntimeError, AssertionError, ValueError):
                pass
        return sub

    def _reverse_perceptual(self, sub):
        """Reverse the perceptual boundary (percept tier → input)."""
        return self.perceptualSpace.reverse(sub)

    def _run_pipeline_rev(self, x):
        """Reverse pipeline: reconstruct input from the head output.

        Mirror of the forward path ``IS→PS→CS→OS`` run backwards:
        ``OS.reverse → chart-generate (C-tier STM) → body.reverse →
        perceptual.reverse → inputSpace.reverse``. Returns the
        reconstructed input subspace (or ``None``).
        """
        if x is None:
            return None
        x = self._reverse_head(x)
        self._chart_generate_from_stm()
        x = self._reverse_body(x)
        x = self._reverse_perceptual(x)
        x = self.inputSpace.reverse(x)
        return x

    def _run_pipeline_rev_from_concepts(self, x):
        """Reverse pipeline from the terminal ConceptualSpace state.

        This is the surviving reconstruction carrier after the symbolic/output
        reconstruction modes were retired: the output head may be intentionally
        lower-dimensional, while the terminal C-tier state still carries the
        full reversible surface needed for input reconstruction.
        """
        if x is None:
            return None
        self._chart_generate_from_stm()
        x = self._reverse_body(x)
        x = self._reverse_perceptual(x)
        x = self.inputSpace.reverse(x)
        return x

    def _reverse_from_S(self, S):
        """Rework B (2): ``reverse(S)`` -- replay WordSpace's STORED
        forward derivations from the single non-NULL sentence idea
        ``S`` (``_stm_single_S``, ``[B, D_c]``) to reconstitute the
        per-percept surface representation.

        Drives the EXISTING reverse-trace machinery (NOT new per-op
        reverse math): stamp ``S`` as a ``[B, 1, D_c]`` event onto the
        ConceptualSpace subspace, then replay the stored grammatical
        derivations via ``_chart_generate_from_stm`` (repopulates
        ``WordSpace.generate_rules`` from the STM snapshot the forward
        left) and run the existing body/percept reverse chain. The
        owner's not-yet-written per-op ``reverse()`` methods stay
        IDENTITY STUBS (``SyntacticLayer.reverse`` returns the subspace
        unchanged for any layer where ``not invertible`` --
        ``Language.py:4170``); we do NOT author per-op reverse math.
        Returns the reconstructed ``[B, N, D]`` per-percept subspace
        materialization (or ``None``).
        """
        if S is None or not torch.is_tensor(S):
            return None
        cs = self.conceptualSpace
        if cs is None or cs.subspace is None:
            return None
        S3 = S.unsqueeze(1) if S.dim() == 2 else S         # [B, 1, D_c]
        cs.subspace.set_event(S3)
        # D4: replay WordSpace's STORED forward derivations. The
        # ``generate_rules`` the reverse dispatch pops from were
        # ALREADY populated by the forward (``_default_generate_rules``
        # at construction; refreshed by the compose path). We do NOT
        # re-fire ``ws.generate()`` here -- re-running the chart's
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
        per-word IR objective that REPLACES the interim P-tier
        ``compute_masked`` masked-LM on the per-word grammar path.

        Reconstruction = the per-percept predictions from
        ``reverse(S)`` (driven from ``_stm_single_S``), mapped back
        through the Rework-A MPHF->table, compared vs the COMPLETE
        UNMASKED input sentence (``inputSpace._ar_embedded`` -- every
        word, the full pre-mask muxed slab the per-word loop never
        touched).

        Returns ``(loss, metric)``:
          * ``loss`` -- the CONTINUOUS percept/concept-vector
            reconstruction (differentiable, MSE-style via the existing
            ``self.loss.compute``); this is the trainable signal that
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
        # unmasked input (differentiable MSE via the existing
        # ``self.loss.compute``; the target is detached so backward
        # flows through the prediction -- the per-word encoder / S /
        # the bounded reduce -- not the fixed input).
        Kr = min(recon.shape[1], target.shape[1])
        Dr = min(recon.shape[-1], target.shape[-1])
        loss = self.loss.compute(
            recon[:, :Kr, :Dr],
            target[:, :Kr, :Dr].detach())

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
                    # frozen lexicon row on PerceptualSpace
                    # ``_active[:, :, 0]`` IS the MPHF index (preferred
                    # over a nearest-vector remap of the target).
                    sub = getattr(ps, 'subspace', None)
                    active = (getattr(sub, '_active', None)
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







    def _forward_per_stage(self, inputData):
        """IR-only forward via stem/body/head pipeline.

        Shape:
          stem -> ``[B, N, D]`` (InputSpace.forward + PerceptualSpace
                  .forward; no K-axis).
          body -> per-stage CS/SS chain on B-shaped tensors; each
                  SymbolicSpace output captured into ``self._ss_cache``.
          head -> ``outputSpace``; result event ``[B, N, predDim]``.

        Within-sentence training is IR-only (BERT-style masked-LM at
        the P-tier).  Sentence-level AR moved to ``InterSentenceLayer``
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
                   self.symbolicSpace):
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
            if (w is not None and torch.is_tensor(w)
                    and not isinstance(w, nn.Parameter)
                    and w.ndim >= 3 and w.requires_grad):
                sub.event.setW(w.detach())

        # Per-run scratch.
        self.symbol_states = []
        self._unified_j_iterations = 0

        B = inputData.shape[0] if isinstance(inputData, torch.Tensor) else 1
        device = (inputData.device if isinstance(inputData, torch.Tensor)
                  else TheDevice.get())
        self.symbolic_state = self.symbolicSpace.empty_state(batch=B).to(device)

        # Inter-sentence prior (read-only at forward time; the actual
        # ARMA observe + loss happen in ``runBatch`` after the body).
        self._predicted_snapshot = None
        self._predicted_confidence = None
        discourse_for_prime = (
            self.wordSpace.discourse
            if self.wordSpace is not None else None)
        if discourse_for_prime is not None:
            d_pred, d_conf = discourse_for_prime.predict()
            self._predicted_snapshot = d_pred
            self._predicted_confidence = d_conf

        # Stem: InputSpace forward only (``[B, N, D]``). PerceptualSpace
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
        in_sub = self._staged_in_sub
        if in_sub is None:
            in_sub = self.inputSpace.forward(inputData)
        if in_sub is None or in_sub.is_empty():
            # Phase 1.5: the five inputs/percepts/concepts/symbols/outputs
            # back-ref aliases were subsumed -- every reader now goes
            # through the owning Space directly
            # (``self.perceptualSpace.subspace`` etc.), so there is no
            # alias to (re)stamp here. Empty-input early return unchanged.
            return None, None, None, None

        # Body: recurrent cell over T stages on B rows. PerceptualSpace
        # runs per pass with the C→P feedback; the IR masked-LM mask is
        # applied once inside the cell on pass 0's perceptual event
        # (``_ir_pre_mask_input`` / ``_ir_mask_positions`` for
        # ``runBatch``). Chart-at-C reads ``ConceptualSpace.stm``.
        body_sub = self._forward_body(in_sub)

        # Head: outputSpace -> ``[B, N, predDim]``.
        head_sub = self._forward_head(body_sub)
        pred = head_sub.materialize() if head_sub is not None else None
        if pred is not None:
            pred = self.normalizer.denormalize(pred, which="output")

        # Capture symbol_states from the per-stage cache list (plain
        # Python ``self._ss_cache`` populated by ``_forward_body``).
        captured_states = []
        for sub in self._ss_cache:
            if sub is None:
                continue
            sv = sub.materialize()
            if sv is None:
                continue
            captured_states.append(sv.clone())
        self.symbol_states = captured_states
        self._unified_j_iterations = min(
            self.conceptualOrder, len(captured_states))

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
            sym_vectors = self.symbolicSpace.subspace.materialize()

        # Inter-sentence snapshot: pass the S-tier event to runBatch,
        # which calls ``InterSentenceLayer.observe`` to capture the
        # sentence rep, compute the ARMA loss, and update the rings.
        # ``_current_discourse_s`` carries the unmodified S-tier event;
        # the layer's ``_pool_sentence_rep`` does the flattening.
        self._current_discourse_s = (
            sym_vectors.detach() if sym_vectors is not None else None)

        self._universality_score = None

        # Downward head emission (S -> C).
        self._predicted_head = None
        try:
            gen_on = bool(TheXMLConfig.get('WordSpace.downwardGeneration'))
        except KeyError:
            gen_on = False
        if (gen_on and self.wordSpace is not None
                and sym_vectors is not None and sym_vectors.ndim >= 3):
            final_state = sym_vectors[:, 0, :]
            codebook_space = (self.perceptualSpace
                              if self.inputSpace.model_type == "embedding"
                              else self.inputSpace)
            head_result = self.wordSpace.reconstruct(final_state, codebook_space)
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
        (``SymbolicSpace.subspace.what.category_ids``).  Decodes input
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
        wordSpace = self.wordSpace
        if wordSpace is None:
            return
        parse_state = getattr(wordSpace, 'parse_state', None)
        chart = getattr(wordSpace, 'chart', None)
        if chart is None and parse_state is None:
            return
        traces = getattr(chart, '_derivation_trace', None) if chart is not None else None
        cat_names = getattr(chart, '_category_names', None) if chart is not None else None
        cat_names = cat_names or ['?']

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
            sym_sub = self.symbolicSpace.subspace
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
            sym_sub = self.symbolicSpace.subspace
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
        if parse_state is not None and getattr(parse_state, 'actions', None):
            row_traces = getattr(parse_state, 'row_traces', None) or {
                0: getattr(parse_state, 'trace', None) or parse_state.actions
            }
            for b, actions in sorted(row_traces.items()):
                batch_el = ET.SubElement(
                    forward_el, 'batch', {'n': str(b)})
                if not actions:
                    ET.SubElement(batch_el, 'noTrace')
                    continue
                action_by_parent = {a.parent_index: a for a in actions}

                def _build_frame(frame_idx, parent_el, _b=b):
                    frame = parse_state.frames[int(frame_idx)]
                    action = action_by_parent.get(int(frame_idx))
                    if action is None:
                        pos = max(0, int(frame.span_start))
                        ET.SubElement(parent_el, 'leaf', {
                            'token': _decode_token(_b, pos),
                            'pos': str(frame.category),
                            'i': str(pos),
                        })
                        return
                    node_el = ET.SubElement(parent_el, 'node', {
                        'cat': str(frame.category),
                        'rule': _rule_canon(action.rule_id),
                        'i': str(frame.span_start),
                        'j': str(frame.span_end),
                    })
                    for child_idx in action.operand_indices:
                        _build_frame(child_idx, node_el, _b)

                root = actions[-1]
                _build_frame(root.parent_index, batch_el)
                rules_el = ET.SubElement(batch_el, 'rules')
                rules_el.text = ",".join(str(int(a.rule_id)) for a in actions)
        elif traces is None:
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

    Dispatches to the right model class based on <architecture> flags:
      - modelType=embedding   -> BasicModel (embedding/language model path)
      - modelType=passthrough -> BasicModel (passthrough path)
      - modelType=vq         -> BasicModel (vector-quantized path)
      - Otherwise             -> SimpleModel parameterized by:
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

        # Attention is incompatible with reshape that changes vector count
        # (attention expects multi-vector 3D, reshape to 1 vector collapses it).
        def _has_reshape(space_name):
            """True if ``space_name`` reshapes its input (nInputDim != 0 or flatten).

            Used to detect the attention-vs-reshape incompatibility:
            attention expects 3D multi-vector input, reshape collapses
            to a single vector.
            """
            try:
                nid = gsp(cfg, space_name, "nInputDim")
            except KeyError:
                nid = 0
            try:
                fl = gsp(cfg, space_name, "flatten")
            except KeyError:
                fl = False
            return nid != 0 or fl
        if _has_reshape("PerceptualSpace") and gsp(cfg, "PerceptualSpace", "hasAttention"):
            errors.append(
                "PerceptualSpace hasAttention=True is incompatible with nInputDim reshape. "
                "Set <hasAttention>false</hasAttention> in <PerceptualSpace>.")

        # ``<maskedPrediction>`` retired 2026-05-14: within-sentence
        # training is IR-only and there's no ARIR mode to validate.
        # ``<reconstruct>`` is independent of the (gone) AR mode --
        # the surviving values (``none`` / ``symbols`` / ``concepts``
        # / ``both``) all fire forward-only loss terms in ``runBatch``.
        if _has_reshape("ConceptualSpace") and gsp(cfg, "ConceptualSpace", "hasAttention"):
            errors.append(
                "ConceptualSpace hasAttention=True is incompatible with nInputDim reshape. "
                "Set <hasAttention>false</hasAttention> in <ConceptualSpace>.")

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

        input_dim = _resolve_dim("InputSpace", 1)
        percept_dim = _resolve_dim("PerceptualSpace", input_dim)
        concept_dim = _resolve_dim("ConceptualSpace", percept_dim)
        symbol_dim = _resolve_dim("SymbolicSpace", concept_dim)

        # Bivector / projection configs let ConceptualSpace output a
        # narrower activation via ``<nOutputDim>``. The width that
        # actually flows into SymbolicSpace is that nOutputDim (when
        # set), not nDim. Compare against the effective width.
        try:
            cs_out_dim = int(gsp(cfg, "ConceptualSpace", "nOutputDim"))
        except KeyError:
            cs_out_dim = 0
        effective_concept_dim = cs_out_dim if cs_out_dim > 0 else concept_dim

        # SymbolicSpace owns no SigmaLayer/PiLayer; the C->S transform
        # was previously a learned ``SigmaLayer(concept_dim, symbol_dim)``
        # inside SS. With SS.sigma retired, the path is dimensionally a
        # pass-through and the configured dims must match.
        TheXMLConfig.require(
            lambda cfg, _c=effective_concept_dim, _s=symbol_dim: _c == _s,
            f"SymbolicSpace requires symbol_dim == "
            f"ConceptualSpace effective output dim "
            f"(got concept_out_dim={effective_concept_dim}, "
            f"symbol_dim={symbol_dim}). Fix: set <SymbolicSpace><nDim> "
            f"to match <ConceptualSpace><nOutputDim> if present, else "
            f"<ConceptualSpace><nDim>."
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

        # Invertible PerceptualSpace shape constraints are registered inside
        # PerceptualSpace._register_requirements() (not here) to keep them self-contained.
        percept_inv = gsp(cfg, "PerceptualSpace", "invertible")

        # Warn only for the legacy naive reverse path. Non-naive inversion
        # uses the LDU/triangular-solve path and does not use pinv.
        naive = bool(arch.get("naive", False))
        if naive and percept_inv:
            warnings.warn(
                "PerceptualSpace: architecture.naive=True materializes dense "
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
        # Bounded so a large corpus (MM_5M: 358k sentences) doesn't run
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

        TheData.load(dataset,
                     num_shards=num_shards,
                     max_docs=max_docs,
                     shard_dir=dat.get("shardDir"),
                     dat=dat)

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

        # CUDA-graph-capture pre-flight: gates ALL training entry (the
        # profiled path and the normal path) for every model run via
        # train.py. Hard-aborts before substantial training if the
        # brick body still issues host syncs. CUDA-only no-op elsewhere.
        ModelFactory._brick_preflight(
            m, _t("batchSize", 10), _t("learningRate", 0.01))

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
                            _t("batchSize", 10),
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
                    _t("batchSize", 10),
                    lr=_t("learningRate", 0.01))

        if _t("autosave", False):
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
