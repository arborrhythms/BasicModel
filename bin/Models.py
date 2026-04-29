"""Top-level model assembly, data loading, and experiment reporting.

``BasicModel`` composes the custom layers from ``Model.py`` into a set of
spaces that move between raw inputs, percepts, concepts, symbols, syntax,
and outputs.  The same module also carries the project utilities used to
load datasets, resolve config paths, plot results, and save reports.
"""

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
from embed import WordVectors, PretrainModel
from data import Data, TheData

from Layers import Layer, PiLayer, SigmaLayer  # Import custom layers from Model.py
from Layers import LinearLayer, AttentionLayer
from Layers import ColumnUsageTracker, LiftingLayer, CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon
from Layers import Error, TheError

from Spaces import ActiveEncoding, WhereEncoding, WhenEncoding, WhatEncoding, EventEncoding
from Spaces import Basis, Tensor, Codebook, Embedding
from Spaces import SubSpace, Space, InputSpace, PerceptualSpace, ModalSpace, ConceptualSpace, SymbolicSpace, OutputSpace
from Language import WordSpace
from util import parse
from Pipeline import (
    CachePoint, GrammarMergeGlue, ReverseAdapter, FlattenKWrapper,
)


class Normalizer:
    """Thin wrapper over TheData's min/max scaling.

    Spaces hold a reference to an instance of this class (set during
    model construction) and call ``self.normalizer.{normalize,denormalize}``
    instead of reaching into the TheData global. Keeps the forward/reverse
    contract free of module-level data coupling.
    """

    def __init__(self, source):
        self._source = source

    def normalize(self, x, which="input"):
        return self._source.normalize(x, which=which)

    def denormalize(self, x, which="input"):
        return self._source.denormalize(x, which=which)


class BaseModel(nn.Module):
    """Shared training, plotting, and persistence infrastructure for all models."""
    name           = "BaseModel"
    spaces         = []
    reversible    = False
    plot           = False
    _optimizer     = None
    checkpoint_every_batches = 0
    _training_step_count = 0
    _autosave_on_exception = False
    # Scale applied to the DiscourseSpace contrastive loss. The
    # inter-sentence DiscourseSpace lives on ``self.wordSpace``
    # (``self.wordSpace.discourse``) rather than directly on the
    # model. Callers that need it should read through
    # ``wordSpace``; ``<training><sentencePrediction>false`` in
    # config leaves ``wordSpace.discourse`` as ``None``.
    sentence_prediction_scale = 0.1
    sentence_contrastive_scale = 0.1
    sentence_predictive_scale = 0.1
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
    def from_config(config_path=None, model_type=None, data=None):
        """Factory: create the right model type from XML config."""
        if config_path is None:
            config_path = os.path.join(ProjectPaths.PROJECT_DIR, "data", "xor.xml")
        resolved_path = ModelFactory.resolve_xml(config_path)
        raw_cfg = BaseModel.load_config(resolved_path)
        model_kind = XMLConfig.infer_model_kind(raw_cfg)
        model_cls = {"mental": MentalModel}.get(model_kind, BasicModel)
        model = model_cls()
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
        """Create the model using settings from an XML config file."""
        self._config_path = config_path
        self._config_data = data

        defaults_path = os.path.join(ProjectPaths.DATA_DIR, "model.xml")
        init_config(path=config_path, defaults_path=defaults_path)
        cfg = TheXMLConfig.data

        arch = cfg["architecture"]
        model_family = TheXMLConfig.model_kind
        ModelFactory.validate_config(cfg, model_family=model_family)

        _t = TheXMLConfig.training
        _s = TheXMLConfig.space

        # DataLoader prefetch workers. Pulled here so every entry point
        # (ModelFactory.run, BaseModel.from_config, tests) shares the
        # same model.xml-defaulted value. 0 means synchronous in-process
        # batch assembly.
        self._num_workers = int(_t("numWorkers"))

        if model_type is None:
            model_type = arch["modelType"]

        embedding_path = TheXMLConfig.get("architecture.embeddingPath", None) or None
        if embedding_path is not None:
            embedding_path = self._resolve_artifact_path(embedding_path)
            TheXMLConfig._data["architecture"]["embeddingPath"] = embedding_path

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
            embedding_path=embedding_path,
            reverse_scale=_t("reverseScale"),
            what_scale=_t("whatScale"),
            where_scale=_t("whereScale"),
            when_scale=_t("whenScale"),
            masked_prediction=str(_t("maskedPrediction", "NONE") or "NONE").upper(),
        )

        # Propagate masked_prediction to InputSpace so its forward() can
        # gate the AR streaming state machine (cursor, sliding buffer,
        # null-byte sentinel). Without this, InputSpace.forward silently
        # takes the non-AR branch every call and the model.forward
        # while-loop in AR mode never terminates.
        if hasattr(self, 'inputSpace'):
            self.inputSpace.masked_prediction = self.masked_prediction

        # serial_mode: true when streaming AR is active — enables the
        # slide-and-recompute fast path in PerceptualSpace/ConceptualSpace.
        is_runtime_arir = (
            data is not None
            and getattr(data, '_runtime_mode', None) == 'ARIR')
        self.serial_mode = (
            self.masked_prediction in ('ARLM', 'ARUS', 'ARIR')
            or is_runtime_arir
        )
        if hasattr(self, 'perceptualSpace'):
            self.perceptualSpace.serial_mode = self.serial_mode
        if hasattr(self, 'conceptualSpace'):
            self.conceptualSpace.serial_mode = self.serial_mode
        if getattr(self, 'wordSpace', None) is not None:
            self.wordSpace.serial_mode = self.serial_mode

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

        # Inter-sentence contrastive loss weight. DiscourseSpace (owned
        # by WordSpace) contributes a dual-force cosine loss to
        # ``runBatch`` with this scale when ``<training><sentencePrediction>``
        # is true. Gated inside Language.WordSpace construction.
        self.sentence_prediction_scale = float(
            TheXMLConfig.training("sentencePredictionScale", 0.1) or 0.1)
        # Contrastive scale falls back to the legacy
        # sentencePredictionScale so existing configs keep behaving
        # as before for the contrastive term.
        self.sentence_contrastive_scale = float(
            TheXMLConfig.training(
                "sentenceContrastiveScale",
                self.sentence_prediction_scale) or self.sentence_prediction_scale)
        self.sentence_predictive_scale = float(
            TheXMLConfig.training("sentencePredictiveScale", 0.1) or 0.1)
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
        self._autosave_on_exception = bool(_t("autosave", False)) or \
            self.checkpoint_every_batches > 0

        if _t("autoload"):
            wpath = TheXMLConfig.get("architecture.weightsPath")
            wpath = self._resolve_artifact_path(wpath)
            self.load_weights(wpath)
            # The .kv artifact's vocab + word/chunk vectors are already
            # restored by PerceptualSpace.vocabulary during model build
            # (see Embedding.create -> _load_embeddings). load_embeddings()
            # additionally restores the LTM truth-layer rows that the
            # vocabulary path doesn't touch, so call it here under the
            # same <autoload> gate. The vocab side is idempotent under
            # re-load; load_embeddings short-circuits if the file is
            # absent.
            epath = TheXMLConfig.get("architecture.embeddingPath")
            if epath:
                epath = self._resolve_artifact_path(epath)
                self.load_embeddings(epath)
        self.max_response_length = arch["maxResponseLength"]
        return cfg

    def create(self, **kwargs):
        """Override in subclasses to build model architecture."""
        pass

    def getOptimizer(self, lr=0.01):
        """Build an Adam optimizer over all trainable parameters.

        Walks ``self.spaces`` collecting params via each space's
        ``getParameters()`` (which excludes alpha-managed params).
        Dedup by tensor ``data_ptr`` so a parameter owned by multiple
        modules is only optimized once.

        When ``trainEmbedding`` is NONE or ARLM, embedding parameters
        are excluded from the optimizer.
        """
        params = []
        seen = set()
        for s in self.spaces:
            for p in s.getParameters():
                if p.data_ptr() not in seen:
                    seen.add(p.data_ptr())
                    params.append(p)
        # Exclude embedding params when trainEmbedding is NONE or ARLM
        if not getattr(self, 'optimize_embedding', False):
            exclude = set()
            if hasattr(self, 'perceptualSpace') and isinstance(self.perceptualSpace.vocabulary, Embedding):
                for p in self.perceptualSpace.vocabulary.embedding_parameters():
                    exclude.add(p.data_ptr())
            if exclude:
                params = [p for p in params if p.data_ptr() not in exclude]
        return optim.Adam(params, lr=lr)

    def rebuild_optimizer(self):
        """Rebuild the main optimizer after codebook expansion."""
        if self._optimizer is None:
            return
        lr = self._optimizer.param_groups[0]['lr']
        self._optimizer = self.getOptimizer(lr=lr)

    def _checkpoint_path(self, suffix=None):
        """Resolve the configured checkpoint path, optionally adding a suffix."""
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
        """Save model weights and embeddings during or after training."""
        path = self._checkpoint_path(suffix=suffix)
        TheMessage(f"[{self.name}] Saving training checkpoint ({reason})")
        self.save_weights(path)
        self.save_embeddings()
        return path

    def _maybe_save_periodic_checkpoint(self):
        interval = int(getattr(self, "checkpoint_every_batches", 0) or 0)
        if interval <= 0:
            return
        step = int(getattr(self, "_training_step_count", 0) or 0)
        if step > 0 and step % interval == 0:
            self.save_training_checkpoint(reason=f"batch {step}")

    def _save_exception_checkpoint(self, exc):
        if not getattr(self, "_autosave_on_exception", False):
            return
        try:
            path = self.save_training_checkpoint(
                reason=f"exception: {type(exc).__name__}",
                suffix="emergency",
            )
            TheMessage(f"[{self.name}] Emergency checkpoint saved to {path}")
        except Exception as save_exc:
            TheMessage(
                f"[{self.name}] Emergency checkpoint failed after "
                f"{type(exc).__name__}: {save_exc}"
            )

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
            try:
                acc[trial, :] = self.runTrial(
                    numEpochs=numEpochs, batchSize=batchSize, lr=lr,
                    profile=profile,
                )
            except Exception as exc:
                self._save_exception_checkpoint(exc)
                raise

        np.savetxt(ProjectPaths.output_path(f"{self.name}.csv"), np.array(acc), delimiter=",")
        return acc

    def paramUpdate(self):
        """Delegate ergodic in-place parameter updates to all spaces."""
        for s in self.spaces:
            s.paramUpdate()

    def set_sigma(self, sigma):
        """Propagate exploration meta-parameters to all spaces."""
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

        Uses Basis.disjunction() to fold all stored truths into a single
        summary vector. Conflicting +/-  assertions cancel dimensions.

        Returns:
            dict with keys: consistent (bool), score (float),
            sites (tensor of dim indices below threshold),
            union_vector (tensor).
        """
        truth_layer = self._get_truth_layer()
        basis = self._get_basis()
        if truth_layer is None or basis is None:
            return {'consistent': True, 'score': 1.0,
                    'sites': torch.tensor([]), 'union_vector': torch.tensor([])}

        n = truth_layer.count.item()
        if n == 0:
            return {'consistent': True, 'score': 1.0,
                    'sites': torch.tensor([]), 'union_vector': torch.tensor([])}

        stored = truth_layer.truths[:n]  # (n, D)

        # Fold via successive disjunction
        union = stored[0].clone()
        for i in range(1, n):
            union = basis.disjunction(union, stored[i])

        # Consistency score = mean absolute value of result
        score = union.abs().mean().item()

        # Inconsistency sites = dimensions with magnitude below threshold
        threshold = 0.1
        weak_dims = (union.abs() < threshold).nonzero(as_tuple=True)[0]

        consistent = len(weak_dims) == 0 or score > 0.5

        return {
            'consistent': consistent,
            'score': score,
            'sites': weak_dims,
            'union_vector': union,
        }

    @torch.no_grad()
    def ground(self, activation, threshold=0.6):
        """Find the minimal TruthSet subset that entails the query.

        Uses _activation_order() to filter truths by compatible partition
        when partitions are available.

        Returns:
            dict with keys: grounded (bool), basis (list of indices),
            trace (list), confidence (float in [-1, 1]).
        """
        truth_layer = self._get_truth_layer()
        if truth_layer is None:
            return {'grounded': False, 'basis': [], 'trace': [], 'confidence': 0.0}

        n = truth_layer.count.item()
        if n == 0:
            return {'grounded': False, 'basis': [], 'trace': [], 'confidence': 0.0}

        stored = truth_layer.truths[:n]  # (n, D)
        partitions = getattr(self, '_partitions', None)

        # Determine query order from partition energy
        query_order = None
        if partitions is not None and len(partitions) > 1:
            query_order = self._activation_order(activation, partitions)

        # Pin the query to the stored truths' device so the cosine
        # matmul below doesn't trip a cross-device gather. Tests often
        # construct activations on the default device while the
        # truth_layer's buffers live on the model's device.
        activation = activation.to(stored.device)

        # Normalize for cosine similarity
        a_norm = F.normalize(activation.unsqueeze(0), dim=-1)  # (1, D)
        s_norm = F.normalize(stored, dim=-1)                    # (n, D)
        sims = (a_norm @ s_norm.T).squeeze(0)                   # (n,)

        # Filter by compatible order if partitions exist
        if query_order is not None and partitions is not None:
            for i in range(n):
                truth_order = self._activation_order(stored[i], partitions)
                if truth_order != query_order:
                    sims[i] = 0.0  # exclude incompatible orders

        # Direct groundings: similarity > threshold
        mask = sims.abs() > threshold
        basis_indices = mask.nonzero(as_tuple=True)[0].tolist()

        if not basis_indices:
            # Try derivation via TruthLayer.derive() (depth capped).
            # After ownership transfer, the S-tier SyntacticLayer lives
            # on WordSpace, not on SymbolicSpace.
            ws = self.wordSpace
            sl = getattr(ws, 'syntacticLayer', None) if ws is not None else None
            part_fn = getattr(sl, 'partForward', None) if sl is not None else None
            max_depth = 3
            for depth in range(max_depth):
                if part_fn is None:
                    break
                derived = truth_layer.derive(part_fn, threshold=threshold)
                if derived == 0:
                    break
                # Re-check with expanded truth set
                n_new = truth_layer.count.item()
                stored_new = truth_layer.truths[:n_new]
                s_norm_new = F.normalize(stored_new, dim=-1)
                sims_new = (a_norm @ s_norm_new.T).squeeze(0)
                mask_new = sims_new.abs() > threshold
                basis_indices = mask_new.nonzero(as_tuple=True)[0].tolist()
                if basis_indices:
                    break

        if not basis_indices:
            return {'grounded': False, 'basis': [], 'trace': [], 'confidence': 0.0}

        # Confidence: DoT-weighted similarity of basis entries
        basis_sims = sims[:truth_layer.count.item()][basis_indices] if basis_indices else torch.tensor([0.0])
        basis_norms = stored[:truth_layer.count.item()][basis_indices].norm(dim=-1)
        confidence = (basis_sims * basis_norms).sum().item() / max(len(basis_indices), 1)
        confidence = max(-1.0, min(1.0, confidence))

        trace = [{'index': idx, 'similarity': sims[idx].item() if idx < len(sims) else 0.0}
                 for idx in basis_indices]

        return {
            'grounded': True,
            'basis': basis_indices,
            'trace': trace,
            'confidence': confidence,
        }

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
        truth_layer = self._get_truth_layer()
        if truth_layer is None:
            return {'added': [], 'rejected': []}

        n = truth_layer.count.item()
        if n < 2:
            return {'added': [], 'rejected': []}

        stored = truth_layer.truths[:n]
        partitions = getattr(self, '_partitions', None)

        # Two-argument grammar methods eligible for extrapolation
        two_arg_rules = ['union', 'intersection', 'equals', 'part']

        indices = seed_indices if seed_indices is not None else list(range(n))
        added = []
        rejected = []

        ss = getattr(self, 'symbolicSpace', None)
        cs = getattr(self, 'conceptualSpace', None)
        pi_layer = ss.sigma if ss is not None else None
        # Get the actual SubSpace for grammar methods (needs .basis)
        subspace = getattr(ss, 'subspace', None) if ss is not None else None
        # *Forward lives on the S-tier SyntacticLayer, which WordSpace
        # now owns post-ownership-transfer.
        ws = self.wordSpace
        syntactic_layer = (getattr(ws, 'syntacticLayer', None)
                           if ws is not None else None)

        for i in indices:
            if stored[i].norm() < 1e-6:
                continue
            for j in indices:
                if i == j or stored[j].norm() < 1e-6:
                    continue
                if len(added) >= max_new:
                    return {'added': added, 'rejected': rejected}

                # Determine source order for partition-aware writing
                if partitions is not None and len(partitions) > 1:
                    order_i = self._activation_order(stored[i], partitions)
                    order_j = self._activation_order(stored[j], partitions)
                    if order_i != order_j:
                        continue  # skip cross-order pairs

                for rule_name in two_arg_rules:
                    method_name = f'{rule_name}Forward'
                    method = getattr(syntactic_layer, method_name, None)
                    if method is None:
                        continue

                    candidate = method(stored[i].unsqueeze(0),
                                       stored[j].unsqueeze(0), subspace)
                    if candidate is None:
                        continue
                    candidate = candidate.squeeze(0)

                    if candidate.norm() < 1e-6:
                        continue

                    # Luminosity non-decrease check
                    lum_before = truth_layer.luminosity(pi_layer)
                    saved_count = truth_layer.count.item()

                    # DoT for derived truth
                    dot_i = stored[i].norm().item()
                    dot_j = stored[j].norm().item()
                    degree = attenuation * min(dot_i, dot_j)

                    direction = F.normalize(candidate.unsqueeze(0), dim=-1).squeeze(0)
                    truth_layer.record(direction, degree, basis=self._get_basis())
                    lum_after = truth_layer.luminosity(pi_layer)

                    delta = (lum_after - lum_before).item()

                    if delta >= 0:
                        # Accept
                        added.append(truth_layer.count.item() - 1)
                    else:
                        # Reject: rollback
                        truth_layer.count.fill_(saved_count)
                        truth_layer.truths[saved_count:] = 0
                        rejected.append((i, j, rule_name, delta))

        return {'added': added, 'rejected': rejected}

    # -- Contemplative Awareness Characterizations ---------------------

    def _active_codebook_rows(self):
        """Indices of currently-active rows in the SymbolicSpace codebook.

        A row counts as active when its L2 norm exceeds the
        ``truthMinMagnitude`` activity floor configured on
        SymbolicSpace. Returns a 1-D ``LongTensor`` of indices into the
        codebook ``W`` matrix.
        """
        sym = self.symbolicSpace
        W = sym.subspace.what.getW()
        if W is None or W.ndim < 2:
            return torch.empty(0, dtype=torch.long)
        norms = W.norm(dim=-1)
        return (norms > sym._truth_min_magnitude).nonzero(as_tuple=True)[0]

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

    def Contiguous(self) -> bool:
        """One-Pointedness (Shamatha / Focused Attention).

        True iff the currently-active SymbolicSpace codebook rows form a
        single connected mereological component -- no disjoint islands
        under ``Basis.part`` parthood (the same metric ImpenetrableLayer
        uses).

        Connectivity test: build a [K, K] adjacency from the
        ``ImpenetrableLayer._classify`` output -- two rows share an edge
        iff their pair is anything other than ``disjoint`` (so
        ``part_ij``, ``part_ji``, ``equal``, and ``overlap`` all count
        as connected). Self-loops are added so a single active row is
        trivially one component. BFS reports whether the graph spans
        every active node.

        Returns True when:
          - 0 or 1 active rows (degenerate one-pointedness)
          - All active rows lie in one connected component
        """
        sym = self.symbolicSpace
        what = sym.subspace.what
        codebook = what.getW()
        if codebook is None:
            return True
        active_idx = self._active_codebook_rows()
        if active_idx.numel() <= 1:
            return True
        sub = codebook[active_idx]
        P = sym._impenetrable._pairwise_parthood(sub, what)
        relations = sym._impenetrable._classify(P)
        edges = ~relations["disjoint"]
        eye = torch.eye(edges.shape[0], device=edges.device, dtype=torch.bool)
        edges = edges | eye
        return self._is_one_component(edges)

    def Continuous(self):
        """Simplicity (Continuity / Open Awareness).

        Developing a continuous N-dimensional awareness within space.
        Requires continuity: small shifts in perceptual and symbolic
        space must produce proportionally small shifts in conceptual
        space.

        Characterisation -- OA (Open Awareness):
          * The mapping PerceptualSpace -> ConceptualSpace is Lipschitz-
            continuous: ||f(x) - f(y)|| <= K ||x - y|| for a bounded K.
          * Equivalently, the Jacobian of the forward pass through
            PiLayer and SigmaLayer has bounded spectral norm.
          * In symbolic space, adjacent codebook entries map to nearby
            concept vectors (smooth codebook topology).

        Computationally, Continuous() should estimate the local Lipschitz
        constant of the perception-to-concept mapping and verify that it
        remains below a configured threshold, ensuring that awareness can
        shift smoothly rather than jumping between attractors.
        """
        raise NotImplementedError

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
            either real (grounded in PerceptualSpace with trust > 0) or
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
        """Persist model weights (excluding embeddings) to disk.

        Embedding weights live in a separate artifact (the .kv/.pt file
        specified by <embeddingPath> in the XML config).  The three files
        -- XML config, embedding artifact, weights checkpoint -- partition
        the model's behaviour and are managed independently.
        """
        if path is None:
            path = os.path.join(ProjectPaths.OUTPUT_DIR, "weights.ckpt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Filter out embedding parameters -- they belong to the .kv artifact
        state = {k: v for k, v in self.state_dict().items()
                 if "wv._vectors" not in k}
        util.atomic_torch_save({"state_dict": state}, path)
        TheMessage(f"[{self.name}] Weights saved to {path}")

    def save_embeddings(self, path=None):
        """Snapshot current nn.Embedding weights and save the .pt artifact.

        Also persists LTM (TruthLayer) data alongside the embeddings so
        truths travel with the vocabulary and survive architecture
        changes. When PerceptualSpace runs in BPE-chunking mode, the
        ChunkLayer's merge table is co-saved into the same ``.kv`` file
        under the artifact's ``bpe`` section -- the resulting ``.kv``
        carries both Lexicon and BPE under ``kind="both"`` so it serves
        either path. See :mod:`embed`.
        """
        if path is None:
            path = getattr(self, 'embedding_path', None)
        if path is None:
            return
        emb = self._get_embedding()
        if emb is None:
            return

        # Collect truth data from SymbolicSpace for co-storage
        truth_data = None
        truth_layer = getattr(getattr(self, 'symbolicSpace', None), 'truth', None)
        if truth_layer is not None and truth_layer.count.item() > 0:
            n = truth_layer.count.item()
            truth_data = {
                "truths": truth_layer.truths[:n].cpu().clone(),
                "count": n,
            }

        # Co-save the BPE codebook when PerceptualSpace owns one in
        # BPE mode. The ChunkLayer's merge table lives outside the
        # nn.Embedding artifact in legacy code; the unified
        # vocab-artifact format lets us bundle them.
        bpe_section = None
        ps = getattr(self, 'perceptualSpace', None)
        cl = getattr(ps, 'chunk_layer', None) if ps is not None else None
        if cl is not None and getattr(cl, 'bpe', False):
            try:
                from embed import bpe_section_from_chunk_layer
                bpe_section = bpe_section_from_chunk_layer(cl)
            except Exception as e:
                TheMessage(
                    f"[{self.name}] BPE co-save skipped "
                    f"({type(e).__name__}: {e}); saving Lexicon only.")
                bpe_section = None

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        emb.save_embeddings(path, truth_data=truth_data,
                            bpe_section=bpe_section)
        suffix_parts = []
        if truth_data:
            suffix_parts.append(f"{n} truths")
        if bpe_section is not None:
            suffix_parts.append(
                f"BPE {len(bpe_section.get('vocab', {}))} entries")
        suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
        TheMessage(f"[{self.name}] Embeddings saved to {path}{suffix}")

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

    def load_embeddings(self, path=None):
        """Load embedding weights and vocab from a .pt artifact."""
        if path is None:
            path = getattr(self, 'embedding_path', None)
        if path is None:
            return False
        if not os.path.exists(path):
            return False
        emb = self._get_embedding()
        if emb is None:
            return False
        wv = WordVectors.load(path)
        # Check that the saved embedding dimensionality matches the model.
        expected_dim = emb.wv.vector_size
        if wv.vector_size != expected_dim:
            TheMessage(
                f"[{self.name}] Embedding dimension mismatch -- cannot load {path}\n"
                f"  File has {wv.vector_size}-dim vectors, model expects {expected_dim}-dim.\n"
                f"  To fix: correct <nDim> in the model XML to match the saved embeddings,\n"
                f"          or delete/move {path} to start fresh."
            )
            return False
        self._restore_vocab(emb, list(wv.index_to_key),
                            counts=wv.counts.tolist(),
                            total_count=int(wv.total_count))
        # Copy loaded weights into the live parameter
        with torch.no_grad():
            emb.wv._vectors.data.copy_(wv._vectors.to(emb.wv._vectors.device))
        TheMessage(f"[{self.name}] Embeddings loaded from {path}")

        # Restore LTM truths if present in the embedding artifact
        truth_data = getattr(wv, 'truth_data', None)
        if truth_data is not None:
            truth_layer = getattr(getattr(self, 'symbolicSpace', None), 'truth', None)
            if truth_layer is not None:
                n = truth_data["count"]
                with torch.no_grad():
                    truth_layer.truths[:n] = truth_data["truths"].to(
                        truth_layer.truths.device)
                    truth_layer.count.fill_(n)
                TheMessage(f"[{self.name}] Restored {n} truths from {path}")
        return True

    def load_weights(self, path=None, strict=False):
        """Load model weights from disk (excluding embeddings).

        Embedding weights are loaded separately from the .kv artifact
        specified by <embeddingPath>.  This method only restores layer
        weights, attention parameters, etc.

        Supports both new format {"state_dict": ...} and legacy format
        (bare state_dict).
        """
        if path is None:
            path = os.path.join(ProjectPaths.OUTPUT_DIR, "weights.ckpt")
        if not os.path.exists(path):
            TheMessage(f"[{self.name}] No checkpoint at {path}, starting fresh")
            return False
        saved = torch.load(path, map_location=TheDevice.get(), weights_only=False)

        if isinstance(saved, dict) and "state_dict" in saved:
            state = saved["state_dict"]
        else:
            state = saved

        # Pre-check for shape mismatches before attempting to load.
        # This produces an actionable diagnostic instead of a raw PyTorch error.
        model_state = {k: v for k, v in self.state_dict().items()
                       if "wv._vectors" not in k}

        # Bivector migration: pre-bivector checkpoints have a [K, D] symbolic
        # codebook, while the current model expects [2K, D]. Duplicate each
        # row into the positive-pole slot (row 2k), leaving the negative-pole
        # slot (row 2k+1) zero. One-time bridge; no reverse migration.
        for k, saved_v in list(state.items()):
            if k not in model_state:
                continue
            model_v = model_state[k]
            if (saved_v.dim() == 2 and model_v.dim() == 2
                    and saved_v.shape[1] == model_v.shape[1]
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
        if mismatches or missing or unexpected:
            lines = [f"[{self.name}] Weight file mismatch -- cannot load {path}"]
            if mismatches:
                lines.append("  Shape mismatches:")
                for key, saved_shape, model_shape in mismatches[:10]:
                    lines.append(f"    {key:<50s}  saved={saved_shape}  model={model_shape}")
                if len(mismatches) > 10:
                    lines.append(f"    ... and {len(mismatches) - 10} more")
            if missing:
                lines.append(f"  Keys in model but missing from file: {len(missing)}")
            if unexpected:
                lines.append(f"  Keys in file not present in model: {len(unexpected)}")
            lines.append("  The model config likely changed since this checkpoint was saved.")
            lines.append(f"  To fix: correct the model XML to match the saved weights,")
            lines.append(f"          or delete/move {path} to start fresh.")
            TheMessage("\n".join(lines))
            return False

        try:
            self.load_state_dict(state, strict=strict)
        except RuntimeError as e:
            TheMessage(f"[{self.name}] Warning: cannot load {path}: {e}")
            return False

        TheMessage(f"[{self.name}] Weights loaded from {path}")
        return True

    def _restore_vocab(self, emb, saved_vocab,
                       counts=None, total_count=0, pending_counts=None):
        """Resize Embedding to match saved vocabulary exactly."""
        dim = emb.wv._vectors.shape[1]
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
        """Run a test pass with reverse and report input vs reconstructed text."""
        if hasattr(self, 'masked_prediction') and self.masked_prediction != 'NONE':
            return  # masked prediction has variable batch sizes; skip reconstruction report
        self.set_sigma(0)  # suppress exploration for evaluation
        test_input, test_output = self.inputSpace.getTestData()
        _, _, allOut, _ = self.runEpoch(batchSize=len(test_input), split="test")

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

        # Buffer reconstruction via nWhere byte offsets (non-differentiable display)
        recovered_meta = self.perceptualSpace._recovered_input
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
        """Delegate XML-driven construction to BaseModel."""
        return super().create_from_config(config_path, model_type=model_type, data=data)

    def create(self, nInput, nPercepts, nConcepts, nSymbols, nWords=16, nOutput=32,
               conceptualOrder=1,
               model_type="simple", data=None, embedding_path=None,
               reverse_scale=0.5, what_scale=0.7, where_scale=0.2, when_scale=0.1,
               masked_prediction='NONE'):
        """Build the full space hierarchy from architecture parameters.

        Config-derivable flags (reshape, ergodic, codebook, etc.) are read
        from TheXMLConfig by each Space constructor.  Only runtime/pipeline
        params are passed here.

        Args:
            nInput/nPercepts/nConcepts/nSymbols/nOutput: object counts per space.
            nWords: object count for the SyntacticSpace.
            conceptualOrder: number of extra Percept->Concept->Symbol cycles.
            model_type: "simple", "embedding", "passthrough", or "vq".
        """
        self.spaces = []  # reset -- prevent stale accumulation from prior create() calls
        self.wordSpace = None  # wired below once the home spaces exist
        TheXMLConfig._requirements.clear()  # clear stale requirements from prior create()/tests
        self.reversible      = True
        self.ergodic          = TheXMLConfig.get("architecture.ergodic")
        self.processSymbols   = TheXMLConfig.get("architecture.processSymbols")
        self.certainty        = TheXMLConfig.get("architecture.training.certainty")
        self.syntax           = False  # BasicModel: no syntax
        TheXMLConfig._data.setdefault("architecture", {})["syntax"] = False
        self.lexer            = TheXMLConfig.space("InputSpace", "lexer")
        # The lexicon lives on PerceptualSpace via its chunking_mode.
        # InputSpace.codebook defaults to false; configs that set it
        # true opt into the InputSpace lexical path.
        self.codebook         = TheXMLConfig.space("InputSpace", "codebook", default=False)
        self.perceptCodebook  = TheXMLConfig.space("PerceptualSpace", "codebook")
        self.conceptCodebook  = TheXMLConfig.space("ConceptualSpace", "codebook")
        self.perceptPassThrough = TheXMLConfig.space("PerceptualSpace", "passThrough")
        self.symbolPassThrough  = TheXMLConfig.space("SymbolicSpace", "passThrough")
        self.invertible       = TheXMLConfig.space("PerceptualSpace", "invertible")
        self.perceptHasAttention = TheXMLConfig.space("PerceptualSpace", "hasAttention")
        self.conceptHasAttention = TheXMLConfig.space("ConceptualSpace", "hasAttention")
        self.perceptPrototypes  = TheXMLConfig.space("PerceptualSpace", "nVectors")
        self.conceptPrototypes  = TheXMLConfig.space("ConceptualSpace", "nVectors")
        self.min_frequency    = float(TheXMLConfig.data_param("minFrequency", 0.0))
        self.neg_samples      = int(TheXMLConfig.training("negSamples", 64))
        # Runtime params
        self.nInput           = nInput
        self.nOutput          = nOutput
        self.nPercepts        = nPercepts
        self.nConcepts        = nConcepts
        self.nSymbols         = nSymbols
        TheXMLConfig.require(
            lambda cfg, _ns=nSymbols, _no=nOutput: _ns >= _no,
            f"nSymbols ({nSymbols}) must be >= nOutput ({nOutput}): "
            f"the symbolic bottleneck must have at least as many symbols as outputs"
        )
        self.nWords           = nWords
        self.data             = data
        self.model_type       = model_type
        self.embedding_path   = embedding_path
        self.conceptualOrder  = conceptualOrder
        self.loss = ModelLoss(reverse_scale=reverse_scale,
                         what_scale=what_scale,
                         where_scale=where_scale,
                         when_scale=when_scale,
                         certainty=self.certainty,
                         nOutput=nOutput,
                         conceptualOrder=conceptualOrder,
                         nWhere=TheXMLConfig.get("architecture.nWhere"),
                         nWhen=TheXMLConfig.get("architecture.nWhen"))
        self.masked_prediction = masked_prediction
        if data is not None and hasattr(data, 'masked_prediction') and data.masked_prediction != 'NONE':
            data.masked_prediction = masked_prediction
        # Resolve nInput=0 sentinel (derive from data when InputSpace.nOutput was 0 in XML)
        if nInput == 0:
            _d = data if data is not None else TheData
            nInput = getattr(_d, 'nInput', 0)
            self.nInput = nInput

        # Resolve dims, chaining through the pipeline.
        # nDim=0 for a space means "same as the previous space's output dim".
        # InputSpace output dim is its configured nDim (embedding/feature size).
        def _resolve_dim(section, prev_dim):
            try:
                raw = TheXMLConfig.space(section, "nDim")
            except KeyError:
                return prev_dim
            return prev_dim if raw == 0 else raw

        input_dim   = _resolve_dim("InputSpace",    1)
        percept_dim = _resolve_dim("PerceptualSpace",  input_dim)
        concept_dim = _resolve_dim("ConceptualSpace",  percept_dim)
        symbol_dim  = _resolve_dim("SymbolicSpace",    concept_dim)
        output_dim  = _resolve_dim("OutputSpace",      symbol_dim)

        # Per-space objectSize: nWhere + nWhen (falls back to architecture, then 0)
        def _obj_size(section):
            try:
                nw = TheXMLConfig.space(section, "nWhere")
            except KeyError:
                nw = 0
            try:
                nn = TheXMLConfig.space(section, "nWhen")
            except KeyError:
                nn = 0
            return nw + nn

        obj_input   = _obj_size("InputSpace")
        obj_percept = _obj_size("PerceptualSpace")
        obj_concept = _obj_size("ConceptualSpace")
        obj_symbol  = _obj_size("SymbolicSpace")
        obj_output  = _obj_size("OutputSpace")

        # Resolve nVectors sentinels (0 -> same as output count for that space)
        def _nvec(section, n_out):
            try:
                raw = TheXMLConfig.space(section, "nVectors")
            except KeyError:
                return n_out
            return n_out if raw == 0 else raw

        nvec_input   = _nvec("InputSpace",    nInput)
        nvec_percept = _nvec("PerceptualSpace", nPercepts)
        nvec_concept = _nvec("ConceptualSpace", nConcepts)
        nvec_symbol  = _nvec("SymbolicSpace",  nSymbols)
        nvec_output  = _nvec("OutputSpace",    nOutput)

        # Build I/O shape tuples: [count, dim + objectSize]
        # Each space's shape includes its own objectSize.
        inputShape   = [nInput,    input_dim   + obj_input]
        perceptShape = [nPercepts, percept_dim + obj_percept]
        conceptShape = [nConcepts, concept_dim + obj_concept]
        symbolShape  = [nSymbols,  symbol_dim  + obj_symbol]
        outputShape  = [nOutput,   output_dim  + obj_output]

        # Build codebook (space-internal) shape tuples: [nVectors, nDim]
        # spaceShape uses raw content dim -- codebook vectors don't include objectSize.
        spaceShape_input   = [nvec_input,   input_dim]
        spaceShape_percept = [nvec_percept, percept_dim]
        spaceShape_concept = [nvec_concept, concept_dim]
        spaceShape_symbol  = [nvec_symbol,  symbol_dim]
        spaceShape_output  = [nvec_output,  output_dim]

        # InputSpace receives raw data (no encoding) as input but produces encoded vectors.
        rawInputShape = [nInput, input_dim]
        self.inputSpace      = self._make_input_space(rawInputShape, spaceShape_input, inputShape,
                                                      model_type=model_type)
        self.perceptualSpace = self._make_perceptual_space(inputShape, spaceShape_percept, perceptShape)
        if isinstance(self.perceptualSpace.vocabulary, Embedding):
            # object.__setattr__ bypasses nn.Module submodule registration so
            # the PerceptualSpace reference is not double-counted in state_dict.
            # Text-mode InputSpace.forward calls _embed (lexicon lives there).
            object.__setattr__(self.inputSpace, '_peer_perceptual',
                               self.perceptualSpace)
        # Convert masked-word string labels to embedding vectors now that
        # the Embedding vocabulary is available on PerceptualSpace.
        if data is not None and hasattr(data, '_lm_labels') and data._lm_labels is not None:
            embedding = self.perceptualSpace.vocabulary
            if embedding is not None and hasattr(embedding, 'pretrain'):
                data.prepare_lm_targets(embedding)
                # Move new targets to device
                data.toDevice()
        self.conceptualSpace = ConceptualSpace(perceptShape, spaceShape_concept, conceptShape)
        self.symbolicSpace   = SymbolicSpace(conceptShape, spaceShape_symbol, symbolShape,
                                             conceptualSpace=self.conceptualSpace)
        self.spaces.extend([self.inputSpace, self.perceptualSpace, self.conceptualSpace, self.symbolicSpace])
        self.syntacticSpace = None

        self.outputSpace     = OutputSpace([nSymbols, symbol_dim + obj_symbol], spaceShape_output, outputShape,
                                           masked_prediction=(masked_prediction != 'NONE'),
                                           vectors=self.perceptualSpace.vocabulary)
        self.spaces.extend([self.outputSpace])
        self.inputSpace.outputSpace = self.outputSpace

        # The output dimensionality of the input layer must be equal to the output dimensionality of the perceptual layer, since the conceptual layer operates on both.
        #assert self.inputSpace.outputShape[1] == self.perceptualSpace2.outputShape[1] # inputDim == perceptDim
        # The input dimensionality of the symbolic layer must be equal to the input dimensionality of the perceptual layer, since they both operate on the output of the conceptual layer.
        #assert self.symbolicSpace.inputShape[1] == self.perceptualSpace2.inputShape[1] == self.conceptualSpace.outputShape[1]#  conceptDim = conceptDim
        # The output shape of the symbolic space is equal to the input shape of the output space
        #assert self.symbolicSpace.outputShape[1] == self.outputSpace.inputShape[1] # these are in conceptual space, or symbolic space if symbols emit objectSize symbols (processSymbols == True)

        # Initialize Grammar, WordSpace, and all three SyntacticLayers
        # (only in autoregressive modes). ``WordSpace.__init__`` owns
        # grammar config, word-stream buffer sizing, SyntacticLayer
        # construction, the TruthLayer, and (conditionally) the
        # DiscourseSpace substrate. See plan: "Architectural
        # addition -- WordSpace".
        if str(masked_prediction).upper() in ('ARLM', 'ARUS', 'RARLM'):
            self.wordSpace = WordSpace(
                perceptualSpace=self.perceptualSpace,
                conceptualSpace=self.conceptualSpace,
                symbolicSpace=self.symbolicSpace,
                nPercepts=nPercepts,
                nConcepts=nConcepts,
                nSymbols=nSymbols,
                concept_dim=concept_dim + obj_concept,
                symbol_dim=symbol_dim + obj_symbol,
            )
            # Register WordSpace with the Space-walking training
            # contract so ``getOptimizer`` collects its parameters
            # (including the truth layer and, if present, the
            # discourse predictor) via the uniform ``getParameters``
            # path.
            self.spaces.append(self.wordSpace)
            # Seed the pipeline context (see InputSpace.set_word_space docstring).
            self.inputSpace.set_word_space(self.wordSpace)

        # Phase 1: wire a Normalizer onto every space so spaces can call
        # self.normalizer.{normalize,denormalize} instead of the TheData global.
        self.normalizer = Normalizer(TheData)
        for space in self.spaces:
            space.normalizer = self.normalizer
            if hasattr(space, 'subspace'):
                space.subspace.normalizer = self.normalizer

        # Phase 2: Sequential pipeline is the only path.
        self.build_pipelines()

        self.to(TheDevice.get())
        TheXMLConfig.validate()

    # --- Factory methods (override in subclasses to swap Space types) ---
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
        """Phase 2: assemble stem/body/head pipelines for BasicModel.

        Stem (inputSpace + perceptualSpace) emits a subspace whose event
        has shape [B, K, N, D] when k_axis=True (microbatch AR path) or
        [B, N, D] when k_axis=False (non-AR / inference). Body
        (conceptualSpace + symbolicSpace + symbol_cache) and head
        (outputSpace) wrap their stages in FlattenKWrapper, which is a
        no-op when k_axis=False and folds K into the batch dim when
        k_axis=True.
        """
        # Cache the subspace flowing out of symbolicSpace so forward()
        # can recover it after pipeline_fwd. Needed because passthrough
        # symbolicSpace doesn't populate its own .subspace.
        self.symbol_cache = CachePoint()

        # Set per-stage attributes used by the single-arg forward() wrappers.
        self.symbolicSpace.is_last = True
        self.symbolicSpace.quantize = False
        if self.wordSpace is not None:
            self.symbolicSpace.wordSpace = self.wordSpace

        # Stem: inputSpace produces the K-extended subspace (in AR mode);
        # perceptualSpace's per-window work runs flat via FlattenKWrapper.
        self.pipeline_stem = nn.Sequential(
            self.inputSpace,
            FlattenKWrapper(self.perceptualSpace),
        )

        # Body: pure transforms wrapped to flatten K.
        self._body_inner = nn.Sequential(
            self.conceptualSpace,
            self.symbolicSpace,
            self.symbol_cache,
        )
        self.pipeline_body = FlattenKWrapper(self._body_inner)

        # Head: output projection wrapped to flatten K.
        self.pipeline_head = FlattenKWrapper(self.outputSpace)

        self.pipeline_fwd = nn.Sequential(
            self.pipeline_stem, self.pipeline_body, self.pipeline_head,
        )

        all_spaces = [self.inputSpace, self.perceptualSpace,
                      self.conceptualSpace, self.symbolicSpace, self.outputSpace]
        any_invertible = any(getattr(s, "invertible", False) for s in all_spaces)

        if any_invertible:
            self.pipeline_rev = nn.Sequential(
                FlattenKWrapper(ReverseAdapter(self.outputSpace)),
                FlattenKWrapper(nn.Sequential(
                    ReverseAdapter(self.symbol_cache),
                    ReverseAdapter(self.symbolicSpace),
                    ReverseAdapter(self.conceptualSpace),
                )),
                FlattenKWrapper(ReverseAdapter(self.perceptualSpace)),
                ReverseAdapter(self.inputSpace),
            )
            self.pipeline_rt = None
            self.midpoint_cache = None
        else:
            self.midpoint_cache = CachePoint()
            self.pipeline_rt = nn.Sequential(
                self.pipeline_stem, self.pipeline_body, self.pipeline_head,
                self.midpoint_cache,
                FlattenKWrapper(ReverseAdapter(self.outputSpace)),
                FlattenKWrapper(nn.Sequential(
                    ReverseAdapter(self.symbol_cache),
                    ReverseAdapter(self.symbolicSpace),
                    ReverseAdapter(self.conceptualSpace),
                )),
                FlattenKWrapper(ReverseAdapter(self.perceptualSpace)),
                ReverseAdapter(self.inputSpace),
            )
            self.pipeline_rev = None

    def End(self):
        """Per-batch teardown. Cascades End() to every Space.

        Released after forward + reverse + loss have consumed the cached
        state. Called from runBatch.
        """
        for space in self.spaces:
            if hasattr(space, 'End'):
                space.End()

    def StartReverse(self, symbols):
        """Reverse pass: Symbol -> Concept -> Percept -> Input (reconstruction)."""
        if isinstance(symbols, torch.Tensor):
            self.symbolicSpace.subspace.set_event(symbols)
            symbols = self.symbolicSpace.subspace
        concepts_state = self.symbolicSpace.reverse(symbols)
        self._debug_tensor_stats(
            "reverse.concepts_state", concepts_state.materialize())
        percepts_state = self.conceptualSpace.reverse(concepts_state)
        self._debug_tensor_stats(
            "reverse.percepts_state", percepts_state.materialize())
        input_state = self.perceptualSpace.reverse(percepts_state)
        self._debug_tensor_stats(
            "reverse.input_state", input_state.materialize())
        self.inputs = self.inputSpace.reverse(input_state)
        self._debug_tensor_stats(
            "reverse.inputs", self.inputs.materialize())
        input = input_state.materialize()
        inputData  = self.inputs.materialize()
        return inputData, input
    def Finish(self, symbols):
        """Project concatenated symbols to task output via OutputSpace.

        Output-range denormalization happens here (not in OutputSpace.forward)
        so the space pipeline stays global-data-free.
        """
        if isinstance(symbols, torch.Tensor):
            self.outputSpace.subspace.set_event(symbols)
            symbols = self.outputSpace.subspace
        self.outputs = self.outputSpace.forward(symbols)
        if self.outputSpace.nonlinear_output:
            outputData = self.outputs.materialize(mode="activation")
        else:
            outputData = self.outputs.materialize()
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
        truth_layer.count.zero_()
        truth_layer._sources = []
        truth_layer._trusts = []
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

    def infer(self, text, max_length=None, mode=None):
        """Autoregressive inference via the standard batch pipeline.

        Two modes:

        ``ARLM`` (append-and-rerun): stages seed text, runs forward,
        decodes the output token, appends it to the input via
        ``pushInput()``, and repeats.  Each iteration re-lexes and
        re-embeds the full (growing) input.

        ``ARIR`` (autoregressive input reconstruction, default): TODO --
        reconstructs a degraded input in-place, reusing the lexing and
        codebook lookup from the initial forward pass.  See design plan
        in ``docs/plans/``.

        Stops when: EOF is predicted, ``max_length`` characters have
        been produced, or the InputSpace output buffer is full.

        Args:
            text: input string (seed text)
            max_length: max characters to generate
            mode: 'ARLM' for traditional append-and-rerun,
                  'ARIR' for input reconstruction (default).
                  Also accepts traditional=True/False for backwards compat
                  via keyword: ``infer(text, traditional=True)`` is
                  equivalent to ``infer(text, mode='ARLM')``.

        Returns:
            list of predicted tokens (words or characters)
        """
        if mode is None:
            mode = getattr(self, 'masked_prediction', 'ARIR')
        mode = mode.upper()
        if max_length is None:
            max_length = getattr(self, 'max_response_length', 256)

        if mode not in {'ARLM', 'ARIR'}:
            raise ValueError(f"infer: unknown mode '{mode}'. Use 'ARLM' or 'ARIR'.")

        tokens = None
        if mode == 'ARIR':
            if not self.reversible:
                raise ValueError("infer(mode='ARIR') requires reversible=True.")
            self.eval()
            self.set_sigma(0)

            with torch.no_grad(), TheData.runtime_batch([text], [[0]], mode='ARIR'):
                self.inputSpace._arir_reset()
                self.inputSpace._arir_max_chars = max_length
                self.runEpoch(batchSize=1, split="runtime")

            tokens = self.inputSpace.get_predicted_tokens()
        else: # 'ARLM'
            self.eval()
            self.set_sigma(0)
            nOutput = self.inputSpace.outputShape[0]
            tokens = []
            total_chars = 0

            with torch.no_grad(), TheData.runtime_batch([text]):
                batchNum=0
                while True:
                    inputTensor = self.inputSpace.prepInput(
                        list(TheData.train_input))
                    result, batchNum = self.runBatch(
                        train=False, batchNum=batchNum, batchSize=1, split="runtime",
                        batch_override=(inputTensor, None),
                    )
                    if result is None:
                        break

                    decoded = self.perceptualSpace.vocabulary.predict(result.outputPred)
                    word = decoded[0]

                    if word is None or word == '' or word == '\x00':
                        break

                    tokens.append(word)
                    total_chars += len(word)

                    if total_chars >= max_length:
                        break

                    if len(tokens) >= nOutput:
                        break

                    TheData.pushInput(word)
        return tokens

    def forward(self, inputData):
        """Microbatch AR forward: stem -> body -> head straight-line.

        Stem (inputSpace + perceptualSpace) emits [B, K, N, D] for AR
        and [B, N, D] for non-AR. The body's FlattenKWrapper folds K
        into the batch dim so conceptual/symbolic operate on [B*K, N, D]
        in one shot. Head restores the K axis and produces
        [B, K, N, predDim] (AR) or [B, N, predDim] (non-AR). The K axis
        is the AR microbatch -- one prediction per cursor position.
        """
        if isinstance(inputData, torch.Tensor):
            inputData = inputData.to(TheDevice.get())
        self._ar_valid_pos = None

        # Keep InputSpace's AR gate in sync with the model's masked_prediction.
        self.inputSpace.masked_prediction = self.masked_prediction

        is_runtime_arir = (
            self.inputSpace.data is not None
            and self.inputSpace.data._runtime_mode == 'ARIR'
        )
        is_ar_mode = (
            self.masked_prediction in ('ARLM', 'ARUS', 'ARIR')
            and not is_runtime_arir
        )

        result = self.pipeline_fwd(inputData)
        # Empty/sentinel passthrough: nothing to predict.
        if result is None or (hasattr(result, 'is_empty') and result.is_empty()):
            self.inputs = self.inputSpace.subspace
            return None, None, None, None

        self.inputs = self.inputSpace.subspace

        if self.outputSpace.nonlinear_output:
            pred = result.materialize(mode="activation")
        else:
            pred = result.materialize()
        if pred is not None:
            pred = self.normalizer.denormalize(pred, which="output")

        # AR path: head emits [B, K, N, predDim]; runBatch flattens K.
        # Non-AR path: head emits [B, N, predDim] -- pass through.
        sym_sub = self.symbol_cache.last
        symbols = sym_sub.materialize() if sym_sub is not None else None

        # forwardInput contract for runBatch:
        #   * AR modes: [B, T, D] embedded input (T == K cursors).
        #   * Non-AR:   [B, N, D] inputSpace event.
        # In AR mode the inputSpace.subspace event was flattened to
        # [B*K, N, D] by the body's FlattenKWrapper; expose the original
        # embedding instead so the AR loss in runBatch can compare
        # per-cursor predictions against per-cursor targets.
        if is_ar_mode:
            last_input_state = self.inputSpace._ar_embedded
            if last_input_state is None:
                last_input_state = self.perceptualSpace._embedded_input
        else:
            last_input_state = self.inputSpace.subspace.materialize()
            if last_input_state is None:
                last_input_state = self.perceptualSpace._embedded_input

        # Expose the [B, K] valid mask for runBatch.
        if is_ar_mode and self.inputSpace.subspace.valid_mask is not None:
            self._ar_valid_pos = self.inputSpace.subspace.valid_mask

        # nWhere bookkeeping: advance positional counter once per cursor
        # (K times per call) to match legacy serial-AR semantics where each
        # cursor iteration was a separate forward call.
        if last_input_state is not None:
            batch = last_input_state.shape[0]
            K = pred.shape[1] if (is_ar_mode and pred is not None and pred.dim() == 4) else 1
            self.inputSpace.subspace.whenEncoding.increment(batch * K)

        return last_input_state, symbols, pred, None

    def reverse(self, symbols, outputData):
        """Full reverse pass: symbols -> concepts -> percepts -> input."""
        inputData, input = self.StartReverse(symbols)
        return inputData, input

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
            self.set_sigma(0.5)  # re-enable for next training epoch
            testLosses[0].append(outErr)
            testLosses[1].append(inErr)

            if hasattr(self, 'masked_prediction') and self.masked_prediction != 'NONE':
                # Masked prediction: report loss only (no classification accuracy)
                accuracy += [0.0]
                TheMessage(f"Test Loss: output={outErr:.4f}, reconstruction={inErr:.4f}")
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
        if not hasattr(self, 'percepts') or self.percepts is None:
            return None
        vecs = self.percepts.materialize()          # [B, nPercepts, dim]
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
            # default (8) is conservative for models with butterfly /
            # N-halving stages, where shapes across stages form an
            # ``log2(N)`` sequence (each is static per stage). Set to
            # 128 so the warning only fires on genuinely-pathological
            # shape variance.
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
        if batch_override is not None:
            batch = batch_override
        elif (split == "runtime"
              and getattr(self.inputSpace.data, '_runtime_mode', None) == 'ARIR'):
            inputData = self.inputSpace.data.train_input
            result, batchNum = self.inputSpace.arir_step(inputData, batchNum)
            if result is None:
                return None, batchNum
            batch = result
        else:
            raise RuntimeError(
                "runBatch: no batch_override supplied. Callers must pass "
                "batch_override=(inputTensor, outputTensor) -- ARLM/ARUS "
                "infer() prep via InputSpace.prepInput, training path "
                "via the DataLoader in runEpoch. ARIR is the only runtime "
                "mode with a dedicated state machine (arir_step)."
            )

        inputTensor, outputTensor = batch
        inference_only = not train and split == "runtime"
        arir_mode = (split == "runtime"
                     and getattr(self.inputSpace.data, '_runtime_mode', None) == 'ARIR')

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
        #
        # ``K`` is deterministic for AR training under the cursor
        # contract: ``slab_bytes = nObj`` (constant), parse(lex='bytes')
        # produces exactly ``nObj`` tokens (byte-exact post-§8g),
        # ``_embed`` / ``_embed_bpe`` materialize ``[B, nObj, nDim]``
        # constants -- so K = T = nObj. For non-AR / numeric trial-
        # cursor data, K = 1. ARIR inference (where K depends on the
        # runtime buffer length) keeps the in-forward call: under
        # inference there's no compile wrapper and the dynamic K is
        # safe in eager.
        if not arir_mode and self.wordSpace is not None and not inference_only:
            ws = self.wordSpace
            try:
                if isinstance(inputTensor, torch.Tensor):
                    B_pre = int(inputTensor.shape[0])
                else:
                    B_pre = int(len(inputTensor))
            except Exception:
                B_pre = None
            if B_pre is not None:
                is_ar_outer = getattr(self, 'masked_prediction', 'NONE') in (
                    'ARLM', 'ARUS', 'ARIR')
                K_pre = (int(self.inputSpace.outputShape[0])
                         if is_ar_outer else 1)
                ws.ensure_microbatch(B_pre, K_pre)

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
        for space in self.spaces:
            if hasattr(space, 'Start'):
                space.Start()

        # AMP: DEBUG DISABLED
        amp_scaler = None
        from contextlib import nullcontext as _nc
        with _nc():
            # Forward pass returns a 4-tuple. AR modes: ``predictions``
            # is a [B, K, N_window, predDim] tensor (one column per
            # cursor position) and ``forwardInput`` is the embedded
            # source [B, T, D]. Non-AR: ``predictions`` is [B, N, predDim]
            # and ``forwardInput`` is the inputSpace event [B, N, D].
            # ``reconstruction`` is the ARIR reverse output (None elsewhere).
            #
            # ``cudagraph_mark_step_begin`` (only meaningful under modes
            # that capture CUDAGraphs -- "reduce-overhead", "max-
            # autotune") tells the runtime to release the previous
            # step's CUDAGraph outputs so the memory pool can be reused
            # for this step. No-op under "default" mode (kernel fusion
            # only); idempotent on non-CUDA hosts.
            try:
                torch.compiler.cudagraph_mark_step_begin()
            except (AttributeError, RuntimeError):
                pass
            forwardInput, symbols, predictions, reconstruction = self.forward(inputTensor)
            is_ar_mode = (
                self.masked_prediction in ('ARLM', 'ARUS', 'ARIR')
                and not arir_mode
            )
            outputDataPred = predictions

            if arir_mode:
                # ARIR inference: no output loss, but the forward pass (AR path)
                # already produced a terminal reconstruction; pass it through.
                inputPred = reconstruction
                if inputPred is None and self.reversible:
                    # Fallback for non-AR ARIR runtime mode.
                    _, inputPred = self.reverse(symbols, outputDataPred)
                result = self.BatchResult(
                    outputPred=outputDataPred, symbols=symbols,
                    lossOut=None, lossIn=None,
                    inputPred=inputPred, forwardInput=forwardInput,
                )
                self.End()
                return result, batchNum

            if inference_only:
                # Inference path: forward only, no loss, no reverse.
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
                    "For inference use split='runtime', or stage runtime_batch(..., outputs=...) "
                    "if targets are required."
                )

            if is_ar_mode:
                # Microbatch AR: ``predictions`` is [B, K, N_window, predDim].
                # The "predicted next token" at cursor k is the rightmost
                # slot of window k -- collapse N_window down to that slot
                # to recover the legacy [B, K, predDim] per-cursor view.
                #
                # OutputSpace and InputSpace can have different muxed widths:
                # InputSpace carries [what + where + when] (e.g. 100+2+2=104),
                # OutputSpace typically carries only [what] (100). We compare
                # only the leading ``pred`` dims, treating the where/when of
                # the target as side info that is not predicted.
                pred_stack = outputDataPred[:, :, -1, :]  # [B, K, predDim]
                B_, K, predDim = pred_stack.shape
                target_stack = forwardInput[:, :K, :predDim]  # [B, K, predDim]
                # Per-cursor valid mask from forward(): rows shorter than K
                # have NULL targets at the tail; training on those teaches
                # the model to predict padding. Mask them out.
                valid_pos = getattr(self, "_ar_valid_pos", None)
                if valid_pos is not None:
                    # valid_pos is [B, K] (window-level validity).
                    mask = valid_pos[:, :K].unsqueeze(-1).expand_as(pred_stack)
                    pred_stack = pred_stack[mask].view(-1, predDim)
                    target_stack = target_stack[mask].view(-1, predDim)
                else:
                    pred_stack = pred_stack.reshape(-1, predDim)
                    target_stack = target_stack.reshape(-1, predDim)
                if self.masked_prediction == 'ARUS':
                    lossOut = torch.tensor(0.0, device=TheDevice.get())
                    output_weight = 0.0
                elif pred_stack.numel() == 0:
                    # All rows were padding (zero-length batch). No signal.
                    lossOut = torch.tensor(0.0, device=TheDevice.get())
                    output_weight = 0.0
                else:
                    lossOut = self.loss.output(pred_stack, target_stack)
                    # ARIR blends output with reconstruction via reverse_scale;
                    # ARLM has no reconstruction term so the output gets full weight.
                    output_weight = ((1 - self.loss.reverse_scale)
                                     if self.masked_prediction == 'ARIR' else 1.0)
            else:
                # Non-AR: today's behavior.
                outputPred = outputDataPred.squeeze()
                output     = outputTensor.squeeze()
                lossOut    = self.loss.output(outputPred, output)
                self.accumulate_output_symbol_residual(outputTensor, outputDataPred)
                output_weight = 1.0 - self.loss.reverse_scale

            TheError.add(
                "output", lossOut,
                weight=output_weight,
                space="OutputSpace", category="prediction",
            )

            # Reconstruction term.
            if is_ar_mode:
                # AR modes: reconstruction is non-None only under ARIR.
                if reconstruction is not None:
                    pred_sq = reconstruction
                    target_sq = forwardInput
                    # Align sequence dim.
                    if pred_sq.shape[1] != target_sq.shape[1]:
                        M = min(pred_sq.shape[1], target_sq.shape[1])
                        pred_sq = pred_sq[:, :M, :]
                        target_sq = target_sq[:, :M, :]
                    lossIn = self.loss.compute(pred_sq, target_sq)
                    TheError.add(
                        "reconstruction", lossIn,
                        weight=self.loss.reverse_scale,
                        space="InputSpace", category="reconstruction",
                    )
                    inputDataPred = reconstruction
                else:
                    inputDataPred = None
                    lossIn = None
            elif self.reversible and self.loss.reverse_scale > 0:
                # Non-AR reversible: today's reverse branch.
                inputDataPred, inputPred = self.reverse(symbols, outputDataPred)
                pred_sq = inputDataPred
                target_sq = forwardInput.squeeze()
                if self.loss.nWhere > 0:
                    lossIn = self.loss.compute_piecewise(pred_sq, target_sq)
                else:
                    lossIn = self.loss.compute(pred_sq, target_sq)
                TheError.add(
                    "reconstruction", lossIn,
                    weight=self.loss.reverse_scale,
                    space="InputSpace", category="reconstruction",
                )
            else:
                inputDataPred = None
                lossIn = None

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

            # Inter-sentence discourse losses. DiscourseSpace produces
            # two complementary terms over the flattened ``[S | W]``
            # snapshot vector:
            #   * contrastive -- dual-force cosine, attractive toward
            #     the recent context centroid, repulsive from older
            #     centroids.  Expands the codebook.
            #   * predictive  -- cosine distance between the AR
            #     sentence predictor's output and the actual current
            #     snapshot.  Collapses the codebook.
            # Both share gradient through the live ``(s_tensor, w_tensor)``
            # pair; stored history is detached.  First-sentence / empty
            # buffer cases return ``None`` and we just snapshot.
            discourse_contrastive = None
            discourse_predictive = None
            discourse = getattr(self.wordSpace, 'discourse', None) if self.wordSpace is not None else None
            if train and discourse is not None:
                s_tensor = getattr(self, '_current_discourse_s', None)
                w_tensor = getattr(self, '_current_discourse_w', None)
                if s_tensor is not None and w_tensor is not None:
                    discourse_contrastive = discourse.contrastive_loss(
                        s_tensor, w_tensor)
                    predicted = getattr(self, '_predicted_snapshot', None)
                    if predicted is not None:
                        discourse_predictive = discourse.predictive_loss(
                            s_tensor, w_tensor, predicted)
                    discourse.snapshot(s_tensor, w_tensor)

            totalLoss = self.loss.total(lossOut, lossIn, sbow)
            if discourse_contrastive is not None:
                totalLoss = totalLoss + self.sentence_contrastive_scale * discourse_contrastive
                TheError.add(
                    "discourse_contrastive", discourse_contrastive,
                    weight=self.sentence_contrastive_scale,
                    space="DiscourseSpace", category="discourse",
                )
            if discourse_predictive is not None:
                totalLoss = totalLoss + self.sentence_predictive_scale * discourse_predictive
                TheError.add(
                    "discourse_predictive", discourse_predictive,
                    weight=self.sentence_predictive_scale,
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
                )

            # Snapshot the breakdown before the backward pass so later
            # calls to TheError.covariance() can see it in the history
            # even if the step is aborted by a non-finite detector below.
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
                _loss_value("discourse_contrastive", discourse_contrastive),
                _loss_value("discourse_predictive", discourse_predictive),
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
                ws_diag = getattr(self, 'wordSpace', None)
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
        ws = getattr(self, 'wordSpace', None)
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
        """
        ws = getattr(self, 'wordSpace', None)
        if ws is None:
            return
        tl = getattr(ws, 'truth_layer', None)
        if tl is not None and hasattr(tl, 'compact'):
            tl.compact(min_trust=0.5)

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
        ws = getattr(self, 'wordSpace', None)
        for space in self.spaces:
            sub = getattr(space, 'subspace', None)
            if sub is None or not hasattr(sub, 'flush_word_buffer'):
                continue
            if ws is not None and getattr(ws, 'syntacticLayer', None) is not None:
                ws.syntacticLayer.flush_word_buffer(sub)
            else:
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
            """Book-keep one runBatch result into the shared accumulators."""
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
        is_ar_mode_outer = (
            hasattr(self, 'masked_prediction')
            and self.masked_prediction in ('ARLM', 'ARUS', 'ARIR')
        )
        text_input = (
            isinstance(self.inputSpace.data.train_input, list)
            and len(self.inputSpace.data.train_input) > 0
            and isinstance(self.inputSpace.data.train_input[0], str)
        )
        byte_lexer = getattr(self, 'lexer', None) in ('byte', 'bytes')
        use_byte_cursor = (is_ar_mode_outer and text_input and byte_lexer)

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
        # so existing tests can grab ``loader.dataset``. ``num_workers``
        # is forced to 0 because the loader is never iterated -- async
        # workers would prefetch tensors no one consumes.
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
        if training and split == "train":
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
            while not ds.all_done():
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
                inp_items, out_items, hard_eos = ds.next_tick()
                if use_byte_cursor:
                    # Byte slab: convert uint8 -> int8 [B, 1, slab_bytes]
                    # to match prepInput's expected shape downstream.
                    inputTensor = inp_items.to(
                        device=TheDevice.get(), dtype=torch.int8
                    ).unsqueeze(1)
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
                # inside MentalModel.forward() via the sliding-window
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
                        and is_ar_mode_outer
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

        # Materialize loss scalars exactly once at epoch end -- avoid
        # per-batch .item() syncs that drain the GPU pipeline.
        if torch.is_tensor(outErr):
            outErr = outErr.item()
        if torch.is_tensor(inErr):
            inErr = inErr.item()
        return outErr, inErr, allOutput, allInput
TheBasicModel = BasicModel()

class MentalModel(BaseModel):
    name = "MentalModel"

    BatchResult      = BasicModel.BatchResult
    runBatch         = BasicModel.runBatch
    runEpoch         = BasicModel.runEpoch
    runTrial         = BasicModel.runTrial
    trainEmbeddings  = BasicModel.trainEmbeddings
    perceptual_sbow_loss = BasicModel.perceptual_sbow_loss
    accumulate_output_symbol_residual = BasicModel.accumulate_output_symbol_residual
    infer            = BasicModel.infer
    # Outer doc-streaming helpers (rolling-cursor handoff). MentalModel
    # inherits the runBatch/runEpoch class attrs explicitly above, so
    # the post-tick dispatch / compact helpers must be re-mapped too;
    # otherwise runEpoch's tail dispatch raises AttributeError.
    dispatch_per_row_reset = BasicModel.dispatch_per_row_reset
    dispatch_soft_reset    = BasicModel.dispatch_soft_reset
    post_tick_compact      = BasicModel.post_tick_compact
    flush_word_buffers     = BasicModel.flush_word_buffers
    _stub_outputs          = BasicModel._stub_outputs
    _maybe_compile_brick   = BasicModel._maybe_compile_brick

    def create(self, nInput, nPercepts, nConcepts, nSymbols, nWords=16, nOutput=32,
               conceptualOrder=1,
               model_type="simple", data=None, embedding_path=None,
               reverse_scale=0.5,
               what_scale=0.7, where_scale=0.2, when_scale=0.1,
               masked_prediction='NONE', **kwargs):

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
        self.embedding_path = embedding_path
        self.lexer = TheXMLConfig.space("InputSpace", "lexer")
        self.ergodic = TheXMLConfig.get("architecture.ergodic")
        self.processSymbols = TheXMLConfig.get("architecture.processSymbols")
        self.certainty = TheXMLConfig.get("architecture.training.certainty")
        # InputSpace.codebook defaults to false; see the matching note in
        # BasicModel.create.
        self.codebook = TheXMLConfig.space("InputSpace", "codebook", default=False)
        self.perceptCodebook = TheXMLConfig.space("PerceptualSpace", "codebook")
        self.conceptCodebook = TheXMLConfig.space("ConceptualSpace", "codebook")
        self.conceptualOrder = conceptualOrder

        # Orthogonal architecture flags.  useButterflies and useGrammar
        # are mutually exclusive -- butterfly permutations fight
        # constituency structure.
        self.useButterflies = bool(
            TheXMLConfig.get("architecture.useButterflies", default=False))
        # Monotonic SigmaLayer weights (W >= 0). Mirrors PiLayer's monotonic
        # flag; when True, invertible SigmaLayers use NonNegativeInvertibleLinearLayer.
        self.monotonic = bool(
            TheXMLConfig.get("architecture.monotonic", default=False))
        try:
            from basicmodel.bin.util import parse_use_grammar
        except ModuleNotFoundError:
            from util import parse_use_grammar
        _raw_use_grammar = TheXMLConfig.get(
            "WordSpace.useGrammar", default="none"
        )
        self.useGrammar = parse_use_grammar(_raw_use_grammar)
        # thoughtFree is structurally equivalent to conceptualOrder=0: no
        # higher-order P/C/S cycles. Reject the nonsense combination early.
        TheXMLConfig.require(
            lambda cfg, _ug=self.useGrammar, _co=self.conceptualOrder:
                _ug != "thoughtFree" or _co == 0,
            f"useGrammar='thoughtFree' requires conceptualOrder=0 "
            f"(got useGrammar={self.useGrammar!r}, "
            f"conceptualOrder={self.conceptualOrder})"
        )
        # Butterflies still conflict with full constituency grammar.
        if self.useButterflies and self.useGrammar == "all":
            raise ValueError(
                "useButterflies=true + useGrammar=\"all\" is excluded: "
                "butterfly permutations fight constituency structure")

        # Butterfly-path state cache (populated in the useButterflies branch
        # below). Mirrors the per-stage SigmaLayer/PiLayer butterfly mode.
        self._butterfly_state_vectors = None
        self._butterfly_state_dim = None
        self._butterfly_symbol_width = None
        self._butterfly_symbol_factor = None

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
        self.masked_prediction = masked_prediction

        self.loss = ModelLoss(
            reverse_scale=reverse_scale,
            what_scale=what_scale,
            where_scale=where_scale,
            when_scale=when_scale,
            nOutput=nOutput,
            conceptualOrder=conceptualOrder,
            nWhere=TheXMLConfig.get("architecture.nWhere"),
            nWhen=TheXMLConfig.get("architecture.nWhen"),
        )

        # Resolve dims, chaining through the pipeline (nDim=0 -> same as input dim)
        def _resolve_dim(section, prev_dim):
            try:
                raw = TheXMLConfig.space(section, "nDim")
            except KeyError:
                return prev_dim
            return prev_dim if raw == 0 else raw

        input_dim   = _resolve_dim("InputSpace",       1)
        percept_dim = _resolve_dim("PerceptualSpace",  input_dim)
        concept_dim = _resolve_dim("ConceptualSpace",  percept_dim)
        symbol_dim  = _resolve_dim("SymbolicSpace",    concept_dim)
        output_dim  = _resolve_dim("OutputSpace",      symbol_dim)

        # Per-space objectSize: nWhere + nWhen (falls back to architecture, then 0)
        def _obj_size(section):
            try:
                nw = TheXMLConfig.space(section, "nWhere")
            except KeyError:
                nw = 0
            try:
                nn = TheXMLConfig.space(section, "nWhen")
            except KeyError:
                nn = 0
            return nw + nn

        obj_input   = _obj_size("InputSpace")
        obj_percept = _obj_size("PerceptualSpace")
        obj_concept = _obj_size("ConceptualSpace")
        obj_symbol  = _obj_size("SymbolicSpace")
        obj_output  = _obj_size("OutputSpace")

        # Resolve nVectors sentinels (0 -> same as output count for that space)
        def _nvec(section, n_out):
            try:
                raw = TheXMLConfig.space(section, "nVectors")
            except KeyError:
                return n_out
            return n_out if raw == 0 else raw

        nvec_input   = _nvec("InputSpace",    nInput)
        nvec_percept = _nvec("PerceptualSpace", nPercepts)
        nvec_concept = _nvec("ConceptualSpace", nConcepts)
        nvec_symbol  = _nvec("SymbolicSpace",  nSymbols)
        nvec_output  = _nvec("OutputSpace",    nOutput)

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
        self.inputSpace = InputSpace(rawInputShape, spaceShape_input, inputShape,
                                     model_type=model_type)

        # Input -> Percept
        self.perceptualSpace = PerceptualSpace(inputShape, spaceShape_percept, perceptShape)
        if isinstance(self.perceptualSpace.vocabulary, Embedding):
            object.__setattr__(self.inputSpace, '_peer_perceptual',
                               self.perceptualSpace)

        conceptInputShape = [nPercepts, percept_dim + obj_percept]

        # Effective conceptOutputShape: if ConceptualSpace specifies
        # nOutputDim, the forwardEnd reshape changes the output layout.
        # Use the volume-preserving shape [N', nOutputDim] where
        # N' = input_volume / nOutputDim (sigma is square/invertible).
        try:
            _c_nOutputDim = TheXMLConfig.space("ConceptualSpace", "nOutputDim")
        except KeyError:
            _c_nOutputDim = 0
        if _c_nOutputDim > 0:
            _c_input_volume = conceptInputShape[0] * conceptInputShape[1]
            conceptOutputShape = [_c_input_volume // _c_nOutputDim, _c_nOutputDim]
        else:
            conceptOutputShape = [nPercepts, concept_dim + obj_concept]

        # -- Butterfly path: pairwise sigma/pi with N-halving --
        if self.useButterflies:
            state_vectors = nPercepts
            state_dim = percept_dim + obj_percept
            symbol_width = symbol_dim + obj_symbol
            n_stages = min(self.conceptualOrder, int(math.log2(state_vectors)))
            # Butterfly power-of-two / reconstruct=symbols / volume equality /
            # state_dim divisibility requirements are registered in
            # ModelFactory.validate_config (see Models.py:3669+) and fire there
            # before model construction.
            self._butterfly_state_vectors = state_vectors
            self._butterfly_state_dim = state_dim
            self._butterfly_symbol_width = symbol_width
            self._butterfly_symbol_factor = state_dim // symbol_width if symbol_width > 0 else 1
            self._level_shapes_list = self._level_shapes(
                nPercepts, state_dim, n_stages)
        # -- Grammar path: progressive bottleneck per conceptual order --
        elif self.useGrammar == "all":
            n_stages = self.conceptualOrder
            self._level_shapes_list = self._level_shapes(
                nPercepts, percept_dim + obj_percept, n_stages)
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
        naive = TheXMLConfig.get("architecture.naive")
        symbol_nonlinear = TheXMLConfig.space("SymbolicSpace", "nonlinear")
        self.conceptualSpaces = nn.ModuleList()
        self.symbolicSpaces = nn.ModuleList()
        for t in range(T):
            is_last = (t == T - 1)
            if self.useButterflies:
                # Butterfly: ConceptualSpace's PiLayer operates on packed
                # pairs [B, N_t/2, 2*state_dim] internally. The butterfly
                # access pattern (permute / pack / unpack / merge) is
                # built into PiLayer/SigmaLayer themselves -- pass
                # stage_idx/initial_n/is_last and the layer's forward
                # becomes the butterfly-aware version with pre-cached
                # permutation buffers.
                pair_dim = 2 * state_dim
                # Conceptual sees N = state_vectors >> t at its input.
                # Symbolic sees N = state_vectors >> (t+1) when Conceptual
                # halved (not is_last), or N = state_vectors >> t when
                # Conceptual was is_last (no halve).
                cs_n_t = state_vectors >> t
                ss_n_t = cs_n_t if is_last else (state_vectors >> (t + 1))
                cs_layer = PiLayer(
                    pair_dim, pair_dim,
                    naive=naive, ergodic=self.ergodic,
                    invertible=True, nonlinear=True,
                    monotonic=self.monotonic,
                    stage_idx=t, n_t=cs_n_t,
                    is_last=is_last)
                cs_layer.saturate = False
                # SymbolicSpace sees the post-conceptual N. Its sigma
                # operates on pairs packed from that stream and skips
                # further merge (is_last=True).
                ss_layer = SigmaLayer(
                    pair_dim, pair_dim, invertible=True,
                    monotonic=self.monotonic, nonlinear=symbol_nonlinear,
                    stage_idx=t, n_t=ss_n_t,
                    is_last=True)
                cs_in = [cs_n_t, state_dim]
                cs_out = cs_in[:] if is_last else [state_vectors >> (t + 1), state_dim]
                ss_in = cs_out[:]
                ss_out = cs_out[:]
            elif self.useGrammar == "all":
                # Grammar path: each stage halves N (except last). Shapes
                # follow _level_shapes but the per-stage ConceptualSpace /
                # SymbolicSpace are plain (no butterfly wrapping).
                n_t = nPercepts >> t
                d_t = percept_dim + obj_percept
                cs_in = [n_t, d_t]
                cs_out = [n_t, d_t] if is_last else [n_t >> 1, d_t]
                ss_in = cs_out[:]
                ss_out = cs_out[:]
                cs_layer = None
                ss_layer = None
            else:
                # Plain path: all stages share the legacy conceptInputShape /
                # conceptOutputShape. No N-halving.
                cs_in = list(conceptInputShape)
                cs_out = list(conceptOutputShape)
                ss_in = list(conceptOutputShape)
                ss_out = list(symbolShape)
                cs_layer = None
                ss_layer = None

            # Non-codebook spaces require nVectors (spaceShape[0]) ==
            # nActive (outputShape[0]); resize the per-stage codebook shape
            # to match the halved N.
            stage_space_concept = [cs_out[0], spaceShape_concept[1]]
            stage_space_symbol = [ss_out[0], spaceShape_symbol[1]]
            cs = ConceptualSpace(cs_in, stage_space_concept, cs_out,
                                 layer=cs_layer)
            ss = SymbolicSpace(ss_in, stage_space_symbol, ss_out,
                               conceptualSpace=cs, layer=ss_layer)
            # Per-stage flags consumed by build_pipelines / forward.
            ss.is_last = is_last
            ss.quantize = True if self.useButterflies else (not is_last)
            self.conceptualSpaces.append(cs)
            self.symbolicSpaces.append(ss)

        # Backwards-compat aliases: read-only callers (e.g.
        # wordSpace.truth_layer = self.symbolicSpace) see the terminal stage.
        self.conceptualSpace = self.conceptualSpaces[-1]
        self.symbolicSpace = self.symbolicSpaces[-1]

        # No SyntacticSpace -- syntax is handled by Grammar centrally.
        self.syntacticSpace = None

        # Output: receives the actual final symbol stream from the pipeline.
        # Per-stage ConceptualSpace/SymbolicSpace already encode any
        # N-halving in their outputShape, so the terminal symbolic stage
        # dictates what enters OutputSpace.
        output_n = int(self.symbolicSpaces[-1].outputShape[0])
        outputInputShape = [output_n, symbol_dim + obj_symbol]
        self.outputSpace = OutputSpace(outputInputShape, spaceShape_output, outputShape,
                                       masked_prediction=(masked_prediction != 'NONE'),
                                       vectors=self.perceptualSpace.vocabulary)

        self._symbol_shape = [nPercepts, percept_dim + obj_percept]

        # Build WordSpace -- the unified container for grammar
        # infrastructure (WordSubSpace, three SyntacticLayers, the
        # TruthLayer, and conditionally the DiscourseSpace substrate).
        # Its ``__init__`` configures the grammar, sizes the word
        # buffer from SymbolicSpace's column layout, builds each tier's
        # SyntacticLayer, and back-wires the home spaces so
        # compose/decompose routes through ``self.wordSpace``.
        self.wordSpace = WordSpace(
            perceptualSpace=self.perceptualSpace,
            conceptualSpace=self.conceptualSpace,
            symbolicSpace=self.symbolicSpace,
            nPercepts=nPercepts,
            nConcepts=nPercepts,
            nSymbols=nSymbols,
            concept_dim=concept_dim + obj_concept,
            symbol_dim=symbol_dim + obj_symbol,
        )
        # Post S-tier merge: compositional rules live on the single
        # unified SyntacticLayer, which does not need a back-reference
        # to SymbolicSpace (the older ternary-lift path used by C-tier
        # compose has been removed).
        self.conceptualSpace.subspace.basis.monotonic = False

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
            if hasattr(space, 'subspace'):
                space.subspace.normalizer = self.normalizer

        # Precompute partition boundaries for partitioned symbolSum
        self._partitions = self._order_partitions(symbol_dim + obj_symbol,
                                                   self.conceptualOrder)
        self.symbol_states = []

        # Phase 2: Sequential pipeline is the only path.
        self.build_pipelines()

        self.to(TheDevice.get())
        TheXMLConfig.validate()

    # -- Phase 2: Sequential pipeline ----------------------------------

    def build_pipelines(self):
        """Phase 2: stem/body/head pipelines for MentalModel.

        Pipeline shape (microbatch AR):
            stem  : inputSpace -> FlattenK(perceptualSpace)
            body  : FlattenK( (conceptualSpaces[t] -> [GrammarMergeGlue?]
                              -> symbolicSpaces[t] -> ss_cache[t])×T
                              -> symbol_cache )
            head  : FlattenK(outputSpace)

        T = len(self.conceptualSpaces). Each per-stage SymbolicSpace
        gets its own CachePoint (``self.ss_caches[t]``) inserted
        immediately after it inside the body. ``forward()`` reads them
        to rebuild ``symbol_states`` (un-flattened back to [B,K,N,D]).
        ``self.symbol_cache`` is the terminal cache (after the last
        SymbolicSpace) — same role as in BasicModel.
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

        # Per-stage symbol_states capture lives in CachePoints inside
        # the body. The terminal one doubles as ``self.symbol_cache``
        # for the BasicModel-style head/symbols read.
        self.ss_caches = nn.ModuleList(
            [CachePoint() for _ in range(T)]
        )
        self.symbol_cache = self.ss_caches[T - 1] if T > 0 else CachePoint()

        # Determine initial_n per stage for the N-halving GrammarMergeGlue.
        try:
            base_n = int(self.perceptualSpace.subspace.inputShape[0])
        except Exception:
            base_n = int(getattr(self, "nPercepts", 0))

        body_modules = []
        for t in range(T):
            body_modules.append(self.conceptualSpaces[t])
            if use_grammar_merge:
                stage_n = base_n // (2 ** t)
                body_modules.append(
                    GrammarMergeGlue(stage_idx=t, initial_n=stage_n,
                                     is_last=(t == T - 1))
                )
            body_modules.append(self.symbolicSpaces[t])
            body_modules.append(self.ss_caches[t])

        self._body_inner = nn.Sequential(*body_modules)

        # Stem / body / head wrappers — same shape as BasicModel.
        self.pipeline_stem = nn.Sequential(
            self.inputSpace,
            FlattenKWrapper(self.perceptualSpace),
        )
        self.pipeline_body = FlattenKWrapper(self._body_inner)
        self.pipeline_head = FlattenKWrapper(self.outputSpace)
        self.pipeline_fwd = nn.Sequential(
            self.pipeline_stem, self.pipeline_body, self.pipeline_head,
        )

        all_spaces = ([self.inputSpace, self.perceptualSpace]
                      + list(self.conceptualSpaces)
                      + list(self.symbolicSpaces)
                      + [self.outputSpace])
        any_invertible = any(getattr(s, "invertible", False) for s in all_spaces)

        # Reverse adapters for each forward module, in reverse order.
        rev_body_inner = nn.Sequential(
            *[ReverseAdapter(m) for m in reversed(body_modules)]
        )
        rev_body = FlattenKWrapper(rev_body_inner)
        rev_head = FlattenKWrapper(ReverseAdapter(self.outputSpace))
        rev_perceptual = FlattenKWrapper(ReverseAdapter(self.perceptualSpace))
        rev_input = ReverseAdapter(self.inputSpace)

        if any_invertible:
            self.pipeline_rev = nn.Sequential(
                rev_head, rev_body, rev_perceptual, rev_input,
            )
            self.pipeline_rt = None
            self.midpoint_cache = None
        else:
            self.midpoint_cache = CachePoint()
            self.pipeline_rt = nn.Sequential(
                self.pipeline_stem, self.pipeline_body, self.pipeline_head,
                self.midpoint_cache,
                rev_head, rev_body, rev_perceptual, rev_input,
            )
            self.pipeline_rev = None

    # -- Order Partitions (Ramsification) -----------------------------

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
    def _level_shapes(n_vectors, dim, conceptual_order):
        """Per-level (N_t, D_t) for averaging merge.

        D stays constant; only N halves per level.  Averaging keeps norms
        bounded so tanh never saturates.  Differences are cached for exact
        inversion.

            percepts:  (N, D)
            level 0:   (N/2, D)
            level 1:   (N/4, D)
            ...
            level k:   (N/2^(k+1), D)

        Biological analogue: increasing receptive field (V1->V2->V4->IT).
        """
        shapes = []
        for t in range(conceptual_order):
            n = n_vectors // (2 ** (t + 1))
            assert n > 0, \
                f"Level {t}: n_vectors={n_vectors} not divisible by 2^{t+1}"
            shapes.append((n, dim))
        return shapes

    def _butterfly_merge(self, x):
        """Average-merge: [B, N, D] -> [B, N/2, D].

        Averages adjacent vector pairs, keeping D constant and norms bounded.
        Caches (left - right) differences in self._merge_diffs for exact
        inversion in the reverse pass.
        """
        B, N, D = x.shape
        assert N % 2 == 0, f"butterfly_merge requires even N, got {N}"
        left = x[:, 0::2, :]    # [B, N/2, D]
        right = x[:, 1::2, :]   # [B, N/2, D]
        self._merge_diffs.append(left - right)
        return (left + right) / 2

    def _butterfly_unmerge(self, x):
        """Exact inverse of average-merge: [B, N/2, D] -> [B, N, D].

        Uses cached difference to recover both original vectors. When the
        cached diff is None (is_last GrammarMergeGlue stage, which is a
        forward-pass-through), this is a no-op to keep the reverse loop
        count aligned with T without reshaping.
        """
        diff = self._merge_diffs.pop()  # left - right
        if diff is None:
            return x
        left = x + diff / 2
        right = x - diff / 2
        B, N_half, D = left.shape
        # Interleave left and right back to original ordering
        out = torch.zeros(B, N_half * 2, D, device=x.device)
        out[:, 0::2, :] = left
        out[:, 1::2, :] = right
        return out

    def _bound_concept_input(self, x):
        """Keep recurrent concept inputs inside ConceptualSpace's logit domain."""
        if getattr(self.conceptualSpace, "nonlinear", False):
            return x.clamp(min=-1 + epsilon, max=1 - epsilon)
        return x

    def _symbol_feedback_from_vectors(self, sym_vectors, n_feedback, feedback_dim):
        """Project symbolic state back to the percept-shaped feedback tensor."""
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

    # -- Helpers -------------------------------------------------------

    def Finish(self, data):
        """Project through OutputSpace.

        Output-range denormalization happens here (not in OutputSpace.forward)
        so the space pipeline stays global-data-free.
        """
        if isinstance(data, torch.Tensor):
            self.outputSpace.subspace.set_event(data)
            data = self.outputSpace.subspace
        self.outputs = self.outputSpace.forward(data)
        if self.outputSpace.nonlinear_output:
            outputData = self.outputs.materialize(mode="activation")
        else:
            outputData = self.outputs.materialize()
        return self.normalizer.denormalize(outputData, which="output")

    def End(self):
        """Per-batch teardown. Counterpart to Start().

        Cascades End() to every Space so cached per-batch state
        (subspace event tensors, per-layer scratch) is released before
        the next batch begins. Called from runBatch after
        forward + reverse + loss have consumed the cached state.
        """
        for space in self.spaces:
            if hasattr(space, 'End'):
                space.End()

    # -- Forward / reverse ----------------------------------------------

    # ----- Sequential path helpers -----

    def _is_ar_mode(self):
        """True when the current config is AR (ARLM/ARUS/ARIR at training time)."""
        is_runtime_arir = (
            self.inputSpace.data is not None
            and getattr(self.inputSpace.data, '_runtime_mode', None) == 'ARIR'
        )
        return (self.masked_prediction in ('ARLM', 'ARUS', 'ARIR')
                and not is_runtime_arir)

    def _extract_prediction_sequential(self, fwd_out):
        """Materialize OutputSpace's subspace and denormalize to task range.

        In AR mode (predictions collected into a list that runBatch
        torch.stacks on dim=1), the nOutput=1 middle axis is squeezed
        here so stack yields [B, N, D] and not [B, N, 1, D]. Non-AR
        callers (returning predictions[0] directly) keep the 3D shape
        so existing loss contracts hold.
        """
        if fwd_out is None:
            return None
        if self.outputSpace.nonlinear_output:
            outputData = fwd_out.materialize(mode="activation")
        else:
            outputData = fwd_out.materialize()
        if outputData is None:
            return None
        outputData = self.normalizer.denormalize(outputData, which="output")
        is_ar = self.masked_prediction in ('ARLM', 'ARUS', 'ARIR')
        if is_ar and outputData.dim() == 3 and outputData.shape[1] == 1:
            outputData = outputData.squeeze(1)
        return outputData

    def _should_reconstruct(self):
        """True when reverse reconstruction should run after the forward loop.

        Only ARIR triggers an automatic reverse pass (input reconstruction).
        Non-AR and ARLM/ARUS callers invoke `model.reverse()` explicitly.
        """
        return self.masked_prediction == 'ARIR'

    def _run_reverse_sequential(self, last_forward_result):
        """Case A reconstruction: run pipeline_rev once after the forward loop."""
        if self.pipeline_rev is None or last_forward_result is None:
            return None
        return self.pipeline_rev(last_forward_result)

    def forward(self, inputData):
        """Microbatch AR forward via stem/body/head pipeline.

        Shape:
          stem  -> [B, K, N, D] (K = T-N+1 progressive-prefix windows
                   for AR; K = 1 for non-AR / inference).
          body  -> per-stage CS/SS chain on [B*K, N, D], with each
                   SymbolicSpace's output captured into ``ss_caches``.
          head  -> outputSpace; result event [B, K, N, predDim].

        Replaces the legacy while-loop: InputSpace produces all K
        windows in one call; the body processes them in parallel via
        FlattenKWrapper. K=1 is the inference (ARIR) and non-AR case.
        """
        if isinstance(inputData, torch.Tensor):
            inputData = inputData.to(TheDevice.get())

        self.inputSpace.masked_prediction = self.masked_prediction
        self._ar_valid_pos = None

        # Detach cached events carried over from prior forward to break
        # the autograd graph; in-place writes below restore live grads.
        for sp in (self.perceptualSpace, self.conceptualSpace,
                   self.symbolicSpace):
            if sp is None or not hasattr(sp, 'subspace'):
                continue
            ev = getattr(sp.subspace, 'event', None)
            if ev is None or not hasattr(ev, 'getW') or not hasattr(ev, 'setW'):
                continue
            w = ev.getW()
            if w is not None and torch.is_tensor(w) and w.requires_grad:
                ev.setW(w.detach())

        # Per-run scratch.
        self.symbol_states = []
        self._nonrams_sym_feedbacks = []
        if self.useGrammar == "all":
            self._sym_feedbacks = []
            self._merge_diffs = []
        self._unified_j_iterations = 0

        B = inputData.shape[0] if isinstance(inputData, torch.Tensor) else 1
        device = (inputData.device if isinstance(inputData, torch.Tensor)
                  else TheDevice.get())
        # WordSpace state cascade is owned by InputSpace.forward, which
        # calls ensure_microbatch(B, K) for both AR (K=T) and non-AR
        # (K=1) paths.  No standalone resize needed here.
        self.symbolic_state = self.symbolicSpace.empty_state(batch=B).to(device)

        # Discourse priming prediction (pre-pipeline).
        self._predicted_snapshot = None
        self._predicted_confidence = None
        discourse_for_prime = (
            self.wordSpace.discourse
            if self.wordSpace is not None else None)
        if discourse_for_prime is not None:
            d_pred, d_conf = discourse_for_prime.predict()
            self._predicted_snapshot = d_pred
            self._predicted_confidence = d_conf

        is_runtime_arir = (
            self.inputSpace.data is not None
            and self.inputSpace.data._runtime_mode == 'ARIR'
        )
        is_ar_mode = (
            self.masked_prediction in ('ARLM', 'ARUS', 'ARIR')
            and not is_runtime_arir
        )

        # Single-call pipeline. InputSpace produces [B, K, N, D] (or
        # [B, N, D] for non-AR with k_axis=False). FlattenKWrapper
        # passes through transparently when k_axis is False.
        result = self.pipeline_fwd(inputData)

        # Empty-sentinel: input exhausted.
        if result is None or (hasattr(result, 'is_empty') and result.is_empty()):
            self.inputs = self.inputSpace.subspace
            self.percepts = self.perceptualSpace.subspace
            self.concepts = self.conceptualSpace.subspace
            self.symbols = self.symbolicSpace.subspace
            self.outputs = self.outputSpace.subspace
            return None, None, None, None

        # Harvest per-stage state the legacy reverse() for useGrammar=="all"
        # expects. GrammarMergeGlue caches its pairwise diff on forward
        # (None for is_last stages, which pass through). The pipeline does
        # not apply per-stage symbol feedback, so _sym_feedbacks is None for
        # every stage; reverse's `if fb is not None` branch then skips the
        # subtraction and _butterfly_unmerge handles the None diff as a
        # no-op, matching the is_last pass-through.
        if self.useGrammar == "all":
            for m in self._body_inner:
                if isinstance(m, GrammarMergeGlue):
                    self._merge_diffs.append(m._merge_diff)
                    self._sym_feedbacks.append(None)

        # Discover K from the result subspace (head's FlattenKWrapper
        # leaves k_axis=True with event [B, K, N, predDim]). The stem
        # subspace has k_axis=False after the body's FlattenKWrapper
        # flattens it in place, so it can't be the K source.
        stem_sub = self.inputSpace.subspace
        K = None
        if result.k_axis:
            result_event = result.materialize()
            if result_event is not None and result_event.dim() == 4:
                K = result_event.shape[1]
        elif (stem_sub.valid_mask is not None
              and stem_sub.valid_mask.dim() == 2):
            K = stem_sub.valid_mask.shape[1]

        # Capture symbol_states from per-stage CachePoints. Inside the
        # body each cached event is [B*K, N, D]; un-flatten back to
        # [B, K, N, D] when K is set.
        captured_states = []
        for cache in self.ss_caches:
            sub = cache.last
            if sub is None:
                continue
            sv = sub.materialize() if hasattr(sub, 'materialize') else None
            if sv is None:
                continue
            if K is not None and sv.dim() == 3 and sv.shape[0] == B * K:
                sv = sv.view(B, K, sv.shape[1], sv.shape[2])
            captured_states.append(sv.clone())
        self.symbol_states = captured_states
        self._unified_j_iterations = min(
            self.conceptualOrder, len(captured_states))

        # Reverse reconstruction (ARIR triggers it).
        reconstruction = None
        if self._should_reconstruct():
            reconstruction = self._run_reverse_sequential(result)

        # Predictions tensor: [B, K, N, predDim] microbatch / [B, N, predDim] non-AR.
        if self.outputSpace.nonlinear_output:
            pred = result.materialize(mode="activation")
        else:
            pred = result.materialize()
        if pred is not None:
            pred = self.normalizer.denormalize(pred, which="output")

        # Reshape sym/percept events back to [B, K, ...] for downstream consumers.
        sym_vectors = self.symbolicSpace.subspace.materialize()
        if (K is not None and sym_vectors is not None
                and sym_vectors.dim() == 3
                and sym_vectors.shape[0] == B * K):
            sym_vectors = sym_vectors.view(
                B, K, sym_vectors.shape[1], sym_vectors.shape[2])

        percepts_t = self.perceptualSpace.subspace.materialize()
        if (K is not None and percepts_t is not None
                and percepts_t.dim() == 3
                and percepts_t.shape[0] == B * K):
            percepts_t = percepts_t.view(
                B, K, percepts_t.shape[1], percepts_t.shape[2])

        symbols = sym_vectors
        if sym_vectors is not None and percepts_t is not None and percepts_t.ndim >= 3:
            symbols = sym_vectors.norm(dim=-1).unsqueeze(-1).expand(
                *sym_vectors.shape[:-1], percepts_t.shape[-1])

        # forwardInput contract for runBatch (mirrors BasicModel.forward):
        #   * AR modes: [B, T, D] embedded input (T == K cursors).
        #   * Non-AR:   [B, N, D] inputSpace event.
        # The body's FlattenKWrapper flattens the stem event to
        # [B*K, N, D] in place, so we expose the original embedding
        # instead of materializing the modified subspace.
        if is_ar_mode:
            input_state = self.inputSpace._ar_embedded
            if input_state is None:
                input_state = self.perceptualSpace._embedded_input
        else:
            input_state = self.inputSpace.subspace.materialize()
            if input_state is None:
                input_state = self.perceptualSpace._embedded_input

        self.inputs = self.inputSpace.subspace
        self.percepts = self.perceptualSpace.subspace
        self.concepts = self.conceptualSpace.subspace
        self.symbols = self.symbolicSpace.subspace
        self.outputs = self.outputSpace.subspace

        # Per-cursor validity mask for the AR loss in runBatch: [B, K].
        if is_ar_mode and stem_sub.valid_mask is not None:
            self._ar_valid_pos = stem_sub.valid_mask

        # Discourse snapshot — last window's symbols match legacy
        # last-cursor semantics.
        self._current_discourse_s = None
        self._current_discourse_w = None
        discourse = (self.wordSpace.discourse
                     if self.wordSpace is not None else None)
        if discourse is not None:
            if sym_vectors is not None and sym_vectors.dim() == 4:
                s_state = sym_vectors[:, -1]
            else:
                s_state = sym_vectors
            try:
                w_state = self.wordSpace.read()
            except Exception:
                w_state = None
            # WordSpace was ensure_batch'd to B*K in the body; collapse the
            # K axis to mirror s_state's last-window semantics: [B*K, M, D]
            # -> [B, K, M, D] -> [B, M, D].
            if (K is not None and w_state is not None
                    and w_state.dim() == 3 and w_state.shape[0] == B * K):
                w_state = w_state.view(
                    B, K, w_state.shape[1], w_state.shape[2])[:, -1]
            if w_state is None and s_state is not None:
                w_state = torch.zeros(
                    B, discourse.max_depth, discourse.n_dim,
                    device=s_state.device, dtype=s_state.dtype)
            if s_state is not None:
                self._current_discourse_s = s_state.detach()
                self._current_discourse_w = w_state.detach()

        # Universality (Golden Rule) score.
        self._universality_score = None
        truth_layer = (self.wordSpace.truth_layer
                       if self.wordSpace is not None else None)
        syntactic_layer = (self.wordSpace.syntacticLayer
                           if self.wordSpace is not None else None)
        svo = syntactic_layer.last_svo if syntactic_layer is not None else None
        lifting_layer = (syntactic_layer.lifting_layer
                         if syntactic_layer is not None else None)
        if (truth_layer is not None and len(truth_layer) > 0
                and svo is not None and lifting_layer is not None):
            s, v, o = svo
            self._universality_score = truth_layer.universality(
                s, v, o, lifting_layer, self.symbolicSpace)

        # Downward head emission (S -> C). Use last window for K-axis.
        self._predicted_head = None
        try:
            gen_on = bool(TheXMLConfig.get('WordSpace.downwardGeneration'))
        except KeyError:
            gen_on = False
        sv_for_head = sym_vectors
        if sv_for_head is not None and sv_for_head.dim() == 4:
            sv_for_head = sv_for_head[:, -1]
        if (gen_on and self.wordSpace is not None
                and sv_for_head is not None and sv_for_head.ndim >= 3):
            final_state = sv_for_head[:, 0, :]
            codebook_space = (self.perceptualSpace
                              if self.inputSpace.model_type == "embedding"
                              else self.inputSpace)
            head_result = self.wordSpace.reconstruct(final_state, codebook_space)
            self._predicted_head = head_result['heads']

        return input_state, symbols, pred, reconstruction

    def reverse(self, symbols, outputData):
        sym_vec = self.symbolicSpace.subspace.materialize()
        concepts_state = self.concepts

        T = len(self.symbolicSpaces)

        if self.useButterflies:
            x = sym_vec
            for t in reversed(range(T)):
                self.symbolicSpaces[t].subspace.set_event(x)
                x = self.symbolicSpaces[t].reverse(
                    self.symbolicSpaces[t].subspace).materialize()
                self.conceptualSpaces[t].subspace.set_event(x)
                x = self.conceptualSpaces[t].reverse(
                    self.conceptualSpaces[t].subspace).materialize()
            self.perceptualSpace.subspace.set_event(x)
            input_state = self.perceptualSpace.reverse(self.perceptualSpace.subspace)
            self.inputs = self.inputSpace.reverse(input_state)
            input_latent = input_state.materialize()
            input_data = self.inputs.materialize()
            return input_data, input_latent

        if self.useGrammar == "all":
            # Progressive-bottleneck: invert the pipeline order
            # (conceptualSpaces[t] -> GrammarMergeGlue[t] -> symbolicSpaces[t])
            # at each stage, bridging stages with symbolicSpaces[t-1].reverse.
            self.symbols.set_event(sym_vec)
            x = self.symbolicSpaces[T - 1].reverse(
                self.symbols).materialize()
            for t in reversed(range(T)):
                fb = self._sym_feedbacks.pop()
                if fb is not None:
                    x = x - fb
                x = self._butterfly_unmerge(x)
                self.symbols.set_event(x)
                concept_input_state = self.conceptualSpaces[t].reverse(self.symbols)
                x = concept_input_state.materialize()
                if t > 0:
                    self.symbolicSpaces[t - 1].subspace.set_event(x)
                    x = self.symbolicSpaces[t - 1].reverse(
                        self.symbolicSpaces[t - 1].subspace).materialize()
            concept_input_state.set_event(x)
        else:
            # Flat recurrent path: reverse sigma, subtract cached feedback.
            concept_input_state = self.conceptualSpace.reverse(concepts_state)
            if getattr(self, '_nonrams_sym_feedbacks', None):
                fb = self._nonrams_sym_feedbacks[-1]
                if fb is not None:
                    recovered = concept_input_state.materialize() - fb
                    concept_input_state.set_event(recovered)

        # -- Shared tail: percept/input reverse --
        concept_input = concept_input_state.materialize()
        percepts_portion = concept_input[:, :self.nPercepts, :]

        concept_input_state.set_event(percepts_portion)
        input_state = self.perceptualSpace.reverse(concept_input_state)
        self.inputs = self.inputSpace.reverse(input_state)
        input_latent = input_state.materialize()
        input_data = self.inputs.materialize()
        return input_data, input_latent

    # -- Grammar Learning (Phase 2) ------------------------------------

    def grammar_learning_step(self, inputTensor, optimizer):
        """Single grammar learning step: symbolic reconstruction loss.

        1. Forward: sentence -> symbolSum (normal useGrammar forward)
        2. Reverse over partition slices with soft rule superposition
        3. Re-encode reconstruction -> symbolSum_hat
        4. Loss = ||symbolSum_hat - symbolSum||^2 (symbolic level)
        5. Optional luminosity validity penalty

        Args:
            inputTensor: input batch tensor.
            optimizer: optimizer for grammar weights.

        Returns:
            dict with 'recon_loss' and 'validity_loss' scalars.
        """
        optimizer.zero_grad()

        # AMP wraps forward / reverse / re-forward.  The luminosity block
        # below uses .item() against truth-layer state and must see fp32,
        # so it stays outside the autocast region.
        amp_cm, amp_scaler = amp_context()
        with amp_cm:
            # Forward pass to get symbolSum
            input_state, symbols, outputData, _ = self.forward(inputTensor)

            # Get the current symbolSum from the symbolic space
            symbolSum = self.symbolicSpace.subspace.event.clone()  # [B, nC, symbol_dim]

            # Reverse pass to reconstruct
            inputPred, _ = self.reverse(symbols, outputData)

            # Re-encode through forward to get symbolSum_hat
            _, _, _, _ = self.forward(inputPred)
            symbolSum_hat = self.symbolicSpace.subspace.event  # [B, nC, symbol_dim]

            # Reconstruction loss at symbolic level
            recon_loss = F.mse_loss(symbolSum_hat, symbolSum.detach())

        # Optional luminosity validity penalty
        truth_layer = self._get_truth_layer()
        validity_loss = torch.tensor(0.0, device=recon_loss.device)
        if truth_layer is not None and len(truth_layer) > 0:
            pi_layer = self.symbolicSpace.sigma
            lum_before = truth_layer.luminosity(pi_layer)
            # Check if reconstruction preserves luminosity
            # (temporarily store reconstructed symbols)
            saved_count = truth_layer.count.item()
            mean_sym = symbolSum_hat.mean(dim=(0, 1)).detach()
            if mean_sym.norm() > 1e-6:
                truth_layer.record(mean_sym, degree=1.0, basis=self._get_basis())
                lum_after = truth_layer.luminosity(pi_layer)
                validity_loss = torch.relu(lum_before - lum_after)
                truth_layer.count.fill_(saved_count)
                truth_layer.truths[saved_count:] = 0

        total = recon_loss + 0.1 * validity_loss
        if amp_scaler is not None:
            amp_scaler.scale(total).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            total.backward()
            optimizer.step()
        self._clamp_symbolic_codebook()

        return {
            'recon_loss': recon_loss.item(),
            'validity_loss': validity_loss.item() if isinstance(validity_loss, torch.Tensor) else validity_loss,
        }

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

        pi_layer = getattr(self.symbolicSpace, 'layer', None)
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
                lum_before = truth_layer.luminosity(pi_layer)

                # Extrapolate new truths
                result = self.extrapolate(max_new=16,
                                           attenuation=0.8 ** (step + 1))
                trace.append({
                    'step': step,
                    'added': len(result['added']),
                    'rejected': len(result['rejected']),
                    'luminosity': lum_before.item(),
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
TheMentalModel = MentalModel()

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
        """Generate a human-readable model name from its flags."""
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
    def validate_config(cfg, model_family=None):
        """Check merged config for known inconsistencies and raise on error.

        Called after defaults have been merged so all keys are present.
        Uses get_space_param() to read from space-scoped sections.
        """
        gsp = ModelFactory.get_space_param
        arch = cfg.get("architecture", {})
        errors = []
        if model_family is None:
            model_family = XMLConfig.infer_model_kind(cfg)

        # Attention is incompatible with reshape that changes vector count
        # (attention expects multi-vector 3D, reshape to 1 vector collapses it).
        def _has_reshape(space_name):
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

        # ARIR mode auto-runs the reverse pass to produce a reconstruction.
        # That makes no sense if reconstruct is disabled.
        training = arch.get("training", {})
        mp = str(training.get("maskedPrediction", "NONE")).upper()
        rc = str(arch.get("reconstruct", "")).upper()
        if mp == "ARIR" and rc == "NONE":
            errors.append(
                "maskedPrediction=ARIR requires <reconstruct> to be set "
                "(not 'NONE'); ARIR runs the reverse path and needs a "
                "reconstruction target. Set <reconstruct>symbols</reconstruct> "
                "(or another non-NONE value) under <architecture>.")
        if _has_reshape("ConceptualSpace") and gsp(cfg, "ConceptualSpace", "hasAttention"):
            errors.append(
                "ConceptualSpace hasAttention=True is incompatible with nInputDim reshape. "
                "Set <hasAttention>false</hasAttention> in <ConceptualSpace>.")

        use_butterflies = bool(arch.get("useButterflies", False))
        if model_family == "mental" and use_butterflies:
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

            def _obj_size(space_name):
                total = 0
                for key in ("nWhere", "nWhen"):
                    try:
                        total += gsp(cfg, space_name, key)
                    except KeyError:
                        pass
                return total

            n_input = _resolve_count("InputSpace", 0)
            n_percepts = _resolve_count("PerceptualSpace", n_input)
            n_concepts = _resolve_count("ConceptualSpace", n_percepts)
            n_symbols = _resolve_count("SymbolicSpace", n_concepts)

            input_dim = _resolve_dim("InputSpace", 1)
            percept_dim = _resolve_dim("PerceptualSpace", input_dim)
            concept_dim = _resolve_dim("ConceptualSpace", percept_dim)
            symbol_dim = _resolve_dim("SymbolicSpace", concept_dim)

            state_dim = percept_dim + _obj_size("PerceptualSpace")
            symbol_width = symbol_dim + _obj_size("SymbolicSpace")

            if n_percepts > 0 and n_symbols > 0 and state_dim > 0 and symbol_width > 0:
                state_volume = n_percepts * state_dim
                symbol_volume = n_symbols * symbol_width
                TheXMLConfig.require(
                    lambda cfg, _sv=state_volume, _yv=symbol_volume: _sv == _yv,
                    f"useButterflies=true requires latent/symbol volume equality: "
                    f"nPercepts*state_dim ({state_volume}) must equal "
                    f"nSymbols*symbol_width ({symbol_volume}). Fix: adjust "
                    f"PerceptualSpace.nOutput/nDim/nWhere/nWhen or "
                    f"SymbolicSpace.nOutput/nDim/nWhere/nWhen so the two "
                    f"volumes match."
                )
                TheXMLConfig.require(
                    lambda cfg, _d=state_dim, _s=symbol_width: _s > 0 and _d % _s == 0,
                    f"useButterflies=true requires state_dim to be divisible by "
                    f"symbol_width so n = D/S is integral (got D={state_dim}, "
                    f"S={symbol_width}). Fix: set PerceptualSpace muxed width "
                    f"(nDim+nWhere+nWhen) to a multiple of SymbolicSpace muxed "
                    f"width (nDim+nWhere+nWhen)."
                )
                TheXMLConfig.require(
                    lambda cfg, _np=n_percepts: _np > 0 and (_np & (_np - 1)) == 0,
                    f"useButterflies=true butterfly schedule requires nPercepts "
                    f"to be a positive power of two (got nPercepts={n_percepts}). "
                    f"Fix: set PerceptualSpace.nOutput to 2^k (e.g. 512, 1024, 2048)."
                )

        # SymbolicSpace passThrough requires shape consistency with ConceptualSpace
        sym_pt = gsp(cfg, "SymbolicSpace", "passThrough")
        if sym_pt:
            symDim = gsp(cfg, "SymbolicSpace", "nDim")
            conDim = gsp(cfg, "ConceptualSpace", "nDim")
            if symDim != 0 and symDim != conDim:
                errors.append(
                    f"SymbolicSpace passThrough=True requires symbolDim == conceptDim "
                    f"(got symbolDim={symDim}, conceptDim={conDim}). "
                    f"Set <nDim>{conDim}</nDim> in <SymbolicSpace> or use <nDim>0</nDim>.")

            # passThrough emits one symbol per active ConceptualSpace output slot,
            # so nVectors must equal ConceptualSpace.nOutput (0 = sentinel, skip check).
            sym_nvec = gsp(cfg, "SymbolicSpace", "nVectors")
            con_nout = gsp(cfg, "ConceptualSpace", "nOutput")
            if sym_nvec != 0 and con_nout != 0 and sym_nvec != con_nout:
                errors.append(
                    f"SymbolicSpace passThrough=True requires nVectors == ConceptualSpace.nOutput "
                    f"(got SymbolicSpace.nVectors={sym_nvec}, ConceptualSpace.nOutput={con_nout}). "
                    f"Set <nVectors>{con_nout}</nVectors> in <SymbolicSpace> or use <nVectors>0</nVectors>.")
            # nOutput == nVectors for passThrough is enforced by _register_requirements().

        # Invertible PerceptualSpace shape constraints are registered inside
        # PerceptualSpace._register_requirements() (not here) to keep them self-contained.
        percept_inv = gsp(cfg, "PerceptualSpace", "invertible")
        percept_pt = gsp(cfg, "PerceptualSpace", "passThrough")
        # Note: invertible PerceptualSpace shape constraints (nOutput == 2*nInput or
        # 4*nInput*inputDim == nOutput*outputDim for reshape) are registered as
        # requirements inside PerceptualSpace._register_requirements(), not here.

        # Warn only for the legacy naive reverse path. Non-naive inversion
        # uses the LDU/triangular-solve path and does not use pinv.
        naive = bool(arch.get("naive", False))
        if naive and percept_inv and not percept_pt:
            warnings.warn(
                "PerceptualSpace: architecture.naive=True materializes dense "
                "inverse weights on the reverse path. This is slower and less "
                "memory efficient than the non-naive LDU solve path. Consider "
                "setting <naive>false</naive> unless debugging the dense path.",
                stacklevel=2)

        if errors:
            raise ValueError(
                "XML config inconsistencies:\n  - " + "\n  - ".join(errors))

        # Fire any requirements registered above (butterfly volume/divisibility/
        # power-of-two/reconstruct, etc.) at validate_config time, so they
        # surface as config errors *before* model construction, alongside the
        # errors.append path.  Any remaining requirements registered later
        # (inside Space._register_requirements during __init__) will fire
        # via the second TheXMLConfig.validate() call at the end of
        # MentalModel.__init__.
        TheXMLConfig.validate()

    @staticmethod
    def resolve_xml(path):
        """Resolve an XML config path relative to the project directory."""
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
    def run(config_path):
        """Main entry point -- create, train, and evaluate a model from XML config."""
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

        m = compile(m)

        def _t(key, default=None):
            return trn.get(key, default)

        def _d(key, default=None):
            return dat.get(key, default)

        num_epochs = int(os.environ.get("BASIC_NUM_EPOCHS", _t("numEpochs", 3)))
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
    """Smoke test: verify encodings and run the XOR config end-to-end."""
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
