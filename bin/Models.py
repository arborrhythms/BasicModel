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
from util import ProjectPaths, XMLConfig, compile, TheXMLConfig, init_config, init_compile_backend
from embed import WordVectors, PretrainModel
from data import Data, TheData

from Layers import Layer, PiLayer, SigmaLayer # Import custom layers from Model.py
from Layers import LinearLayer, AttentionLayer
from Layers import ColumnUsageTracker, LiftingLayer, CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon
from Layers import Error, TheError

from Spaces import ActiveEncoding, WhereEncoding, WhenEncoding, WhatEncoding, EventEncoding
from Spaces import Basis, Tensor, Codebook, Embedding
from Spaces import SubSpace, Space, InputSpace, PerceptualSpace, ModalSpace, ConceptualSpace, SymbolicSpace, OutputSpace, WordSpace
from parse import quick_parser

class BaseModel(nn.Module):
    """Shared training, plotting, and persistence infrastructure for all models."""
    name           = "BaseModel"
    spaces         = []
    reversible    = False
    plot           = False
    _optimizer     = None
    # Scale applied to the DiscourseSpace contrastive loss. The
    # inter-sentence DiscourseSpace lives on ``self.wordSpace``
    # (``self.wordSpace.discourse``) rather than directly on the
    # model. Callers that need it should read through
    # ``wordSpace``; ``<training><sentencePrediction>false`` in
    # config leaves ``wordSpace.discourse`` as ``None``.
    sentence_prediction_scale = 0.1

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
            reconstruct=arch["reconstruct"],
        )

        # Inter-sentence contrastive loss weight. DiscourseSpace (owned
        # by WordSpace) contributes a dual-force cosine loss to
        # ``runBatch`` with this scale when ``<training><sentencePrediction>``
        # is true. Gated inside Spaces.WordSpace construction.
        self.sentence_prediction_scale = float(
            TheXMLConfig.training("sentencePredictionScale", 0.1) or 0.1)

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
        if self.optimize_embedding and isinstance(self.inputSpace.vocabulary, Embedding):
            emb_params = self.inputSpace.vocabulary.embedding_parameters()
            self.inputSpace.params = self.inputSpace.params + emb_params
        self.loss.embedding_scale = float(_t("embeddingScale") or 0.1)
        if isinstance(self.inputSpace.vocabulary, Embedding):
            self.inputSpace.vocabulary.optimize_embedding = self.optimize_embedding
            object.__setattr__(self.inputSpace.vocabulary, "_model", self)

        if _t("autoload"):
            wpath = TheXMLConfig.get("architecture.weightsPath")
            wpath = self._resolve_artifact_path(wpath)
            self.load_weights(wpath)
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
            if hasattr(self, 'inputSpace') and isinstance(self.inputSpace.vocabulary, Embedding):
                for p in self.inputSpace.vocabulary.embedding_parameters():
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
            acc[trial, :] = self.runTrial(numEpochs=numEpochs, batchSize=batchSize, lr=lr, profile=profile)

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
        if hasattr(self, 'inputSpace') and isinstance(self.inputSpace.vocabulary, Embedding):
            return self.inputSpace.vocabulary
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
            sl = getattr(ws, 'symbolicSyntacticLayer', None) if ws is not None else None
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
        pi_layer = ss.layer if ss is not None else None
        # Get the actual SubSpace for grammar methods (needs .basis)
        subspace = getattr(ss, 'subspace', None) if ss is not None else None
        # *Forward lives on the S-tier SyntacticLayer, which WordSpace
        # now owns post-ownership-transfer.
        ws = self.wordSpace
        syntactic_layer = (getattr(ws, 'symbolicSyntacticLayer', None)
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

    def Contiguous(self):
        """One-Pointedness (Shamatha / Focused Attention).

        Maintaining awareness of a given convex region in 5D Perceptual
        Space.  Requires stillness: the model holds a single locus of
        attention without wandering.

        Characterisation -- ShamathaSpeech mode:
          * The symbolic grammar is restricted to a single S derivation
            rule (S -> C), so no Equals, no swap, no compound sentences.
          * Perceptually, the active region decodes to a *contiguous*
            subspace -- no disjoint islands of activation.
          * Symbolically, the active symbols form a contiguous block in
            the codebook ordering (no gaps).

        Computationally, Contiguous() should verify that the current
        model state (symbol_states / STM) occupies a single connected,
        convex region in PerceptualSpace and a contiguous span in
        SymbolicSpace.  When thought_free mode is active the grammar
        already enforces one-pointedness; this method characterises the
        resulting spatial property.
        """
        raise NotImplementedError

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
        torch.save({"state_dict": state}, path)
        TheMessage(f"[{self.name}] Weights saved to {path}")

    def save_embeddings(self, path=None):
        """Snapshot current nn.Embedding weights and save the .pt artifact.

        Also persists LTM (TruthLayer) data alongside the embeddings so
        truths travel with the vocabulary and survive architecture changes.
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

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        emb.save_embeddings(path, truth_data=truth_data)
        TheMessage(f"[{self.name}] Embeddings saved to {path}"
                   + (f" ({n} truths)" if truth_data else ""))

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
                         self.inputSpace.get_recovered_word(0, 0) is not None)
        if use_lex_recon:
            recon_text_list = self.inputSpace.reconstruct_data(text=True)
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
        recovered_meta = getattr(self.inputSpace, '_recovered_input', None)
        if use_lex_recon and recovered_meta is not None:
            buf_size = max(len(test_input[0].tolist()) if isinstance(test_input[0], torch.Tensor) else 64, 64)
            buffer_strings = self.inputSpace.reconstruct_to_buffer(buf_size=buf_size)
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
               masked_prediction='NONE', reconstruct='NONE'):
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
        # Read config-derivable flags
        self.reconstruct     = reconstruct.lower()
        self.reversible      = str(TheXMLConfig.get("architecture.reconstruct")).upper() != "NONE"
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
        self.nOutputSymbols   = nOutput
        self.nReconSymbols    = max(0, nSymbols - nOutput)
        self.recon_symbols    = None
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

        nOutputSymbols = self.nOutputSymbols
        # InputSpace receives raw data (no encoding) as input but produces encoded vectors.
        rawInputShape = [nInput, input_dim]
        self.inputSpace      = self._make_input_space(rawInputShape, spaceShape_input, inputShape,
                                                      model_type=model_type)
        # Convert masked-word string labels to embedding vectors now that
        # the Embedding vocabulary is available.
        if data is not None and hasattr(data, '_lm_labels') and data._lm_labels is not None:
            embedding = self.inputSpace.vocabulary if self.inputSpace.subspace.what is not None else None
            if embedding is not None and hasattr(embedding, 'pretrain'):
                data.prepare_lm_targets(embedding)
                # Move new targets to device
                data.toDevice()
        self.perceptualSpace = self._make_perceptual_space(inputShape, spaceShape_percept, perceptShape)
        self.conceptualSpace = ConceptualSpace(perceptShape, spaceShape_concept, conceptShape)
        self.symbolicSpace   = SymbolicSpace(conceptShape, spaceShape_symbol, symbolShape,
                                             conceptualSpace=self.conceptualSpace)
        self.spaces.extend([self.inputSpace, self.perceptualSpace, self.conceptualSpace, self.symbolicSpace])
        self.syntacticSpace = None

        self.nTotalOutputSymbols = nOutputSymbols
        self.outputSpace     = OutputSpace([nOutputSymbols, symbol_dim + obj_symbol], spaceShape_output, outputShape,
                                           masked_prediction=(masked_prediction != 'NONE'),
                                           vectors=self.inputSpace.vocabulary)
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

    def Start(self, inputData):
        """Forward pass through the core pipeline: Input -> Percept -> Concept -> Symbol."""
        ws = self.wordSpace
        # Per-sentence lifecycle: reset the word-stream buffer.
        if ws is not None:
            ws.clear_sentence()
        if hasattr(self, 'symbolicSpace'):
            self.symbolicSpace.reset_symbol_objective()
        self.inputs   = self.inputSpace.forward(inputData)
        self.percepts = self.perceptualSpace.forward(self.inputs, wordSpace=ws)
        self.concepts = self.conceptualSpace.forward(self.percepts, wordSpace=ws)
        self.symbols  = self.symbolicSpace.forward(self.concepts, wordSpace=ws)
        input = self.inputs.materialize()
        concepts = self.concepts.materialize()
        symbols = self.symbols.materialize()
        if self.plot:
            TheReport.plotActivations(figure=1, concepts=concepts)
        return input, concepts, symbols
    def StartReverse(self, symbols):
        """Reverse pass: Symbol -> Concept -> Percept -> Input (reconstruction)."""
        ws = self.wordSpace
        if isinstance(symbols, torch.Tensor):
            self.symbolicSpace.subspace.set_event(symbols)
            symbols = self.symbolicSpace.subspace
        concepts_state = self.symbolicSpace.reverse(symbols, wordSpace=ws)
        percepts_state = self.conceptualSpace.reverse(concepts_state, wordSpace=ws)
        input_state = self.perceptualSpace.reverse(percepts_state, wordSpace=ws)
        self.inputs = self.inputSpace.reverse(input_state)
        input = input_state.materialize()
        inputData  = self.inputs.materialize()
        return inputData, input
    def Finish(self, symbols):
        """Project concatenated symbols to task output via OutputSpace."""
        if isinstance(symbols, torch.Tensor):
            self.outputSpace.subspace.set_event(symbols)
            symbols = self.outputSpace.subspace
        self.outputs = self.outputSpace.forward(symbols)
        outputData = self.outputs.materialize()
        if self.plot:
            TheReport.plotActivations(figure=1, symbols=symbols)
        return outputData
    def FinishReverse(self, outputData):
        """Reconstruct the symbol tensor from output for the reverse pass.

        reconstruct="symbols" (default): use cached forward symbols only.
        reconstruct="output": use outputSpace.reverse(outputData) only.
        reconstruct="both": reversed output + cached recon_symbols.
        """
        if isinstance(outputData, torch.Tensor):
            self.outputSpace.subspace.set_event(outputData)
            outputData = self.outputSpace.subspace
        mode = getattr(self, 'reconstruct', 'symbols')
        if mode == 'output':
            return self.outputSpace.reverse(outputData).materialize()
        elif mode == 'both':
            output_symbols = self.outputSpace.reverse(outputData).materialize()
        else:  # 'symbols'
            output_symbols = self.output_symbols
        if self.recon_symbols is not None and self.nReconSymbols > 0:
            return torch.cat([output_symbols, self.recon_symbols], dim=1)
        return output_symbols

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
                    result, batchNum = self.runBatch(
                        train=False, batchNum=batchNum, batchSize=1, split="runtime",
                    )
                    if result is None:
                        break

                    decoded = self.inputSpace.predict(result.outputPred)
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
        """Full forward pass: core pipeline + higher-order cycles + output projection.

        Returns (output_prediction, perceptual_state).
        Symbols from each processing stage are concatenated before OutputSpace.
        """
        if isinstance(inputData, torch.Tensor):
            inputData = inputData.to(TheDevice.get())
        input, concepts, symbols = self.Start(inputData)
        if self.nReconSymbols > 0:
            self.output_symbols = symbols[:, :self.nTotalOutputSymbols, :]
            self.recon_symbols = symbols[:, self.nTotalOutputSymbols:, :]
        else:
            self.output_symbols = symbols
            self.recon_symbols = None
        outputData = self.Finish(self.output_symbols)
        batch = input.shape[0]
        self.inputSpace.subspace.whenEncoding.increment(batch)
        return input, symbols, outputData
    def reverse(self, symbols, outputData):
        """Full reverse pass: core reconstruction."""
        symbols = self.FinishReverse(outputData)
        inputData, input = self.StartReverse(symbols)
        return inputData, input

    def runTrial(self, numEpochs=1, batchSize=10, lr=0.01, profile=None):
        """Main training loop: train for numEpochs, evaluate on test set each epoch.

        Alpha (exploration temperature) anneals from 1.0 (full exploration)
        to 0.0 (full exploitation) over the first 5% of training.  This is
        propagated to all Spaces and their layers/bases via set_sigma().

        A single persistent optimizer is used across all epochs so Adam's
        momentum and variance estimates accumulate properly.

        Returns a list of per-epoch test accuracies.
        """
        trainLosses       = [[],[]]  # [output_losses, reconstruction_losses]
        validationLosses  = [[],[]]
        testLosses        = [[],[]]
        self.plot         = False
        accuracy          = []
        self._optimizer   = self.getOptimizer(lr=lr)

        # Enable sigma-driven self-annealing for ergodic layers
        self.set_sigma(0.5)

        # Baseline evaluation before any training
        self.set_sigma(0)
        outErr, inErr, allOut, lastIn = self.runEpoch(batchSize=batchSize, split="test")
        self.set_sigma(0.5)
        testLosses[0].append(outErr)
        testLosses[1].append(inErr)
        TheMessage(f"Baseline Test Loss: output={outErr:.4f}, reconstruction={inErr:.4f}")

        for epoch in range(numEpochs):
            TheMessage(f"Epoch [{epoch + 1}/{numEpochs}]")

            outErr, inErr, allOut, lastIn = self.runEpoch(optimizer=self._optimizer, batchSize=batchSize, split="train")
            trainLosses[0].append(outErr)
            trainLosses[1].append(inErr)
            TheMessage(f"Train Loss: output={outErr:.4f}, reconstruction={inErr:.4f}")

            self.set_sigma(0)  # suppress exploration during eval
            outErr, inErr, allOut, lastIn = self.runEpoch(batchSize=batchSize, split="test")
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
            emb = self.inputSpace.vocabulary
            if isinstance(emb, Embedding):
                sentences = self._get_sentences(split)
                if sentences and index < len(sentences):
                    sentence = sentences[index]
                    words = [t for t, _ in quick_parser(sentence)]
                    if te in ('JOINT'):
                        sbow = self.inputSpace.sbow_loss(words)
                    elif te in ('CBOW', 'SBOW', 'BOTH'):
                        # CBOW uses padded context; SBOW and BOTH use the faster centroid method
                        method = 'CBOW' if te == 'CBOW' else 'SBOW'
                        self.inputSpace.train_embeddings(words, method=method)
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
        n = min(getattr(self, 'nOutputSymbols', pred_symbols.shape[1]),
                pred_symbols.shape[1])
        pred_symbols = pred_symbols[:, :n, :]

        target_event = outputTensor.to(outputDataPred.device)
        while target_event.dim() < outputDataPred.dim():
            target_event = target_event.unsqueeze(-1)
        target_event = target_event.expand_as(outputDataPred)

        self.outputSpace.subspace.set_event(target_event)
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
        self.symbolicSpace.accumulate_symbol_objective(
            pred_symbols[:, :n, :],
            target_symbols[:, :n, :],
            residual_scale=residual_scale,
        )

    def runBatch(self, train=True, batchNum=0, batchSize=10, split="train",
                 optimizer=None, batch_override=None):
        """Run a single batch: forward pass, loss, and (if training) backward + step.

        Args:
            train: whether to compute gradients and update parameters.
            batchNum: opaque cursor returned by getBatch for the next batch.
            batchSize: number of examples per batch.
            split: "train", "test", or "validation".
            optimizer: pre-built optimizer (required when train=True).
            batch_override: optional ``(inputTensor, outputTensor)`` pair that
                bypasses ``InputSpace.getBatch`` -- used by the streaming path
                where the DataLoader has already produced the batch and
                ``InputSpace._cached_embedding`` carries the masked embedding.

        Returns:
            (BatchResult, nextBatchNum) on success, or (None, batchNum) when
            the dataset is exhausted.
        """
        sentenceIdx = batchNum  # sentence index before getBatch increments
        if batch_override is not None:
            batch = batch_override
        else:
            batch, batchNum = self.inputSpace.getBatch(batchNum, batchSize, split)
            if batch is None:
                return None, batchNum

        inputTensor, outputTensor = batch
        masked_pred = hasattr(self, 'masked_prediction') and self.masked_prediction != 'NONE'
        inference_only = not train and split == "runtime"
        arir_mode = (split == "runtime"
                     and getattr(self.inputSpace.data, '_runtime_mode', None) == 'ARIR')

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

        # Forward pass (masking, if any, is applied inside InputSpace.forward())
        forwardInput, symbols, outputDataPred = self.forward(inputTensor)

        if arir_mode:
            # ARIR inference: no output loss, but always run reverse pass
            # so that reconstructed vectors and _recovered_words are available
            # for the next getBatch() call.
            inputPred = None
            if self.reversible:
                _, inputPred = self.reverse(symbols, outputDataPred)
            return self.BatchResult(
                outputPred=outputDataPred, symbols=symbols,
                lossOut=None, lossIn=None,
                inputPred=inputPred, forwardInput=forwardInput,
            ), batchNum

        if inference_only:
            # Inference path: forward only, no loss, no reverse.
            return self.BatchResult(
                outputPred=outputDataPred, symbols=symbols,
                lossOut=None, lossIn=None,
                inputPred=None, forwardInput=forwardInput,
            ), batchNum

        if outputTensor is None:
            raise RuntimeError(
                f"runBatch: missing output targets for split='{split}'. "
                "For inference use split='runtime', or stage runtime_batch(..., outputs=...) "
                "if targets are required."
            )

        outputPred = outputDataPred.squeeze()
        output     = outputTensor.squeeze()
        lossOut    = self.loss.output(outputPred, output)
        self.accumulate_output_symbol_residual(outputTensor, outputDataPred)

        # ARUS: suppress output loss (unsupervised -- no target signal)
        if hasattr(self, 'masked_prediction') and self.masked_prediction == 'ARUS':
            lossOut = torch.tensor(0.0, device=TheDevice.get())

        # Register output term: (1 - reverse_scale) is the effective
        # weight in ModelLoss.forward, matching the legacy composition
        # so TheError.total() shadows the legacy totalLoss.
        TheError.add(
            "output", lossOut,
            weight=1.0 - self.loss.reverse_scale,
            space="OutputSpace", category="prediction",
        )

        use_recon = self.reversible and self.loss.reverse_scale > 0
        if use_recon:
            inputDataPred, inputPred = self.reverse(symbols, outputDataPred)
            pred_sq = inputDataPred
            masked_pred = hasattr(self, 'masked_prediction') and self.masked_prediction != 'NONE'

            # Use pre-masked, post-encoding target when available
            recon_target, recon_mask = self.inputSpace.get_reconstruction_target()
            if recon_target is not None:
                target_sq = recon_target.squeeze()
            else:
                target_sq = forwardInput.squeeze()

            if masked_pred and recon_mask is not None and pred_sq.dim() >= 2:
                # Masked prediction: compute loss only at masked positions
                mask = recon_mask
                if pred_sq.dim() == 3:
                    mask = mask.unsqueeze(-1).expand_as(pred_sq)
                lossIn = self.loss.compute(pred_sq[mask], target_sq[mask])
            elif self.loss.nWhere > 0:
                # Piecewise (Chamfer) loss: per-token matching + coverage.
                # Avoids buffer-level error shadowing when tokens overlap.
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

        # Inter-sentence contrastive loss. DiscourseSpace computes a
        # dual-force cosine loss over the full flattened ``[S | W]``
        # snapshot vector: attractive toward the recent context
        # centroid, repulsive from older centroids. Gradient flows
        # through the live ``(s_tensor, w_tensor)`` pair while the
        # stored history is detached. First-sentence case (empty
        # recent buffer) returns ``None`` and we just snapshot.
        discourse_loss = None
        discourse = getattr(self.wordSpace, 'discourse', None) if self.wordSpace is not None else None
        if train and discourse is not None:
            s_tensor = getattr(self, '_current_discourse_s', None)
            w_tensor = getattr(self, '_current_discourse_w', None)
            if s_tensor is not None and w_tensor is not None:
                discourse_loss = discourse.loss(s_tensor, w_tensor)
                discourse.snapshot(s_tensor, w_tensor)

        totalLoss = self.loss.total(lossOut, lossIn, sbow)
        if discourse_loss is not None:
            totalLoss = totalLoss + self.sentence_prediction_scale * discourse_loss
            TheError.add(
                "discourse", discourse_loss,
                weight=self.sentence_prediction_scale,
                space="DiscourseSpace", category="discourse",
            )
        symbol_loss = None
        if hasattr(self, 'symbolicSpace'):
            symbol_loss = self.symbolicSpace.symbol_objective_loss()
            if symbol_loss is not None:
                totalLoss = totalLoss + symbol_loss
                # Per-term symbol objective registration.  The weights
                # are already baked into each term by accumulate_symbol_objective,
                # so we register with weight=1.0 and category="symbol".
                for term_name, term_value in \
                        self.symbolicSpace.symbol_objective_terms().items():
                    TheError.add(
                        term_name, term_value, weight=1.0,
                        space="SymbolicSpace", category="symbol",
                    )

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

        if not torch.isfinite(totalLoss).all():
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
                _loss_value("discourse", discourse_loss),
                _loss_value("symbol", symbol_loss),
                _loss_value("total", totalLoss),
            ])
            raise FloatingPointError(
                f"Non-finite total loss in {self.name}.runBatch("
                f"split={split}, batch={batchNum}): {details}"
            )

        TheMessage(f"batch = {batchNum}, loss = {totalLoss} ")

        if train:
            totalLoss.backward()
            if self.ergodic:
                self.paramUpdate()
            optimizer.step()
            self._clamp_symbolic_codebook()

        result = self.BatchResult(
            outputPred=outputDataPred,
            symbols=symbols,
            lossOut=lossOut,
            lossIn=lossIn,
            inputPred=inputDataPred,
            forwardInput=forwardInput,
        )
        return result, batchNum

    def runEpoch(self, optimizer=None, batchSize=10, split="train"):
        """Run one epoch over the dataset (training if optimizer given, eval if None).

        Drives batching with a ``SentenceStreamDataset`` DataLoader so every
        split is consumed as ``B = batchSize`` contiguous streams. ``B`` is
        capped at ``len(split)`` by ``data_loader``, so small eval sets
        degrade gracefully.

        In inference mode (split="runtime", no optimizer): skips loss
        construction, output accumulation, progress printing, and CBOW
        updates. Uses ``InputSpace.getBatch`` directly for ARIR state
        machine compatibility.

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
        # Runtime ARIR / inference use getBatch directly because its state
        # machine is stateful across calls.
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
                    self.outputSpace.putBatch(result)
            return 0, 0, [], []

        allOutput = []
        allInput = []
        outputChunks = []
        inputChunks = []
        outErr = 0.0
        inErr = 0.0
        masked_pred = (hasattr(self, 'masked_prediction')
                       and self.masked_prediction != 'NONE')

        def record(result):
            """Book-keep one runBatch result into the shared accumulators."""
            nonlocal outErr, inErr
            self.outputSpace.putBatch(result)
            if result.lossOut is not None:
                outErr = result.lossOut.item()
            if result.lossIn is not None:
                inErr = result.lossIn.item()
            outputChunks.append(result.outputPred.clone().detach().squeeze())
            if self.reversible and result.inputPred is not None:
                inputChunks.append(result.inputPred.clone().detach().squeeze())

        num_workers = getattr(self, '_num_workers', 0)
        loader = self.inputSpace.data.data_loader(
            split=split,
            num_streams=batchSize,
            num_workers=num_workers,
            prefetch_factor=(2 if num_workers > 0 else None),
            pin_memory=(TheDevice.get().type == "cuda"),
        )

        with ctx:
            step = 0
            for inp_items, out_items in loader:
                B = (inp_items.shape[0] if isinstance(inp_items, torch.Tensor)
                     else len(inp_items))
                inputTensor = self.inputSpace.prepInput(inp_items)
                outputTensor = self.outputSpace.prepOutput(out_items)

                text_batch = (isinstance(inp_items, list)
                              and inp_items
                              and isinstance(inp_items[0], str))

                if masked_pred and text_batch:
                    # Per-position masking over the B-wide sentence batch.
                    # Embed once per fetched batch and reuse across mask
                    # positions so we do not re-tokenize / re-lookup.
                    with torch.no_grad():
                        embedded = (self.inputSpace
                                    .forward(inputTensor)
                                    .materialize())
                    # Reset the positional counter the forward() call
                    # incremented -- the cached-embedding bypass does not
                    # advance .p itself.
                    self.inputSpace.subspace.whereEncoding.p = 0

                    nVec = embedded.shape[1]
                    N = min(max(len(s.split()) for s in inp_items), nVec)
                    for pos in range(N):
                        masked, targets, mask_positions = (
                            self.inputSpace.expand_masked_batched(
                                embedded, inp_items,
                                self.masked_prediction, pos,
                            ))
                        self.inputSpace._cached_embedding = masked
                        self.inputSpace._unmasked_embedding = (
                            embedded.detach())
                        self.inputSpace._mask_positions = mask_positions
                        result, _ = self.runBatch(
                            train=training, batchNum=step,
                            batchSize=B, split=split,
                            optimizer=optimizer,
                            batch_override=(inputTensor,
                                            targets.unsqueeze(1)),
                        )
                        if result is not None:
                            record(result)
                        step += 1

                    # Embedding training once per fetched batch -- feed
                    # each sentence's words directly (no index into
                    # train_input needed).
                    if (training
                            and getattr(self, 'lexer', None)
                            not in ('byte', 'bytes')):
                        te = getattr(self, 'train_embedding', 'NONE')
                        if te in ('CBOW', 'SBOW', 'BOTH'):
                            method = 'CBOW' if te == 'CBOW' else 'SBOW'
                            for sentence in inp_items:
                                words = [t for t, _
                                         in quick_parser(sentence)]
                                self.inputSpace.train_embeddings(
                                    words, method=method)
                else:
                    # Single forward/backward per B-wide batch.
                    result, _ = self.runBatch(
                        train=training, batchNum=step,
                        batchSize=B, split=split,
                        optimizer=optimizer,
                        batch_override=(inputTensor, outputTensor),
                    )
                    if result is not None:
                        record(result)
                    step += 1

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

    def create(self, nInput, nPercepts, nConcepts, nSymbols, nWords=16, nOutput=32,
               conceptualOrder=1,
               model_type="simple", data=None, embedding_path=None,
               reverse_scale=0.5,
               what_scale=0.7, where_scale=0.2, when_scale=0.1,
               masked_prediction='NONE', reconstruct='NONE', **kwargs):

        self.spaces = []
        self.wordSpace = None  # wired below once the home spaces exist
        self.reversible = str(TheXMLConfig.get("architecture.reconstruct")).upper() != "NONE"
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
        # below).  Named after ButterflyStage.
        self._butterfly_state_vectors = None
        self._butterfly_state_dim = None
        self._butterfly_symbol_width = None
        self._butterfly_symbol_factor = None

        # Truth integration config (optional -- absent in BasicModel.xml)
        self.truth_bias_scale = float(TheXMLConfig.get("architecture.truthBiasScale", default=0.1) or 0.1)
        self.luminosity_weight = float(TheXMLConfig.get("architecture.LuminosityWeight", default=0.1) or 0.1)
        self.universality_weight = float(TheXMLConfig.get("architecture.UniversalityWeight", default=0.1) or 0.1)
        # Belnap-Dunn balance knobs. Defaults mirror Kleene 3-valued logic:
        #   allowExcludedMiddle=1  permit NEITHER (epistemic uncertainty)
        #   allowContradiction=0   forbid BOTH (classical non-contradiction)
        self.allow_excluded_middle = int(
            TheXMLConfig.get("architecture.allowExcludedMiddle", default=1) or 1)
        self.allow_contradiction = int(
            TheXMLConfig.get("architecture.allowContradiction", default=0) or 0)
        self.truth_loss_weight = float(TheXMLConfig.training("TruthLoss", default=0.0) or 0.0)
        self.reconstruct = reconstruct.lower()
        self.masked_prediction = masked_prediction
        self.nOutputSymbols = nOutput
        self.nReconSymbols = max(0, nSymbols - nOutput)
        self.nTotalOutputSymbols = nOutput
        self.recon_symbols = None

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

        butterfly_config = None

        # -- Butterfly path: pairwise sigma/pi with N-halving --
        if self.useButterflies:
            state_vectors = nPercepts
            state_dim = percept_dim + obj_percept
            symbol_width = symbol_dim + obj_symbol
            n_stages = min(self.conceptualOrder, int(math.log2(state_vectors)))
            TheXMLConfig.require(
                lambda cfg, _n=state_vectors: _n > 0 and (_n & (_n - 1)) == 0,
                f"butterfly path requires nPercepts ({state_vectors}) "
                f"to be a positive power of two"
            )
            TheXMLConfig.require(
                lambda cfg, _r=str(TheXMLConfig.get('architecture.reconstruct')).lower(): _r == "symbols",
                "butterfly reverse requires reconstruct=symbols so the full symbol state "
                "remains available for exact inversion"
            )
            self._butterfly_state_vectors = state_vectors
            self._butterfly_state_dim = state_dim
            self._butterfly_symbol_width = symbol_width
            self._butterfly_symbol_factor = state_dim // symbol_width if symbol_width > 0 else 1
            butterfly_config = {
                "conceptual_order": n_stages,
                "state_vectors": state_vectors,
                "state_dim": state_dim,
                "symbol_width": symbol_width,
                "symbol_factor": state_dim // symbol_width if symbol_width > 0 else 1,
                "ergodic": self.ergodic,
                "naive": TheXMLConfig.get("architecture.naive"),
            }
            self._level_shapes_list = self._level_shapes(
                nPercepts, state_dim, n_stages)
        # -- Grammar path: progressive bottleneck per conceptual order --
        elif self.useGrammar == "all":
            self._level_shapes_list = self._level_shapes(
                nPercepts, percept_dim + obj_percept, self.conceptualOrder)
        else:
            self._level_shapes_list = None

        self.conceptualSpace = ConceptualSpace(conceptInputShape, spaceShape_concept,
                                               conceptOutputShape,
                                               level_shapes=self._level_shapes_list,
                                               butterfly_config=butterfly_config)

        self.symbolicSpace = SymbolicSpace(conceptOutputShape, spaceShape_symbol, symbolShape,
                                           conceptualSpace=self.conceptualSpace,
                                           level_shapes=self._level_shapes_list,
                                           butterfly_config=butterfly_config)

        # No SyntacticSpace -- syntax is handled by Grammar centrally.
        self.syntacticSpace = None

        # Output: from first nOutputSymbols symbol vectors (matches BasicModel pattern)
        outputInputShape = [self.nOutputSymbols, symbol_dim + obj_symbol]
        self.outputSpace = OutputSpace(outputInputShape, spaceShape_output, outputShape,
                                       masked_prediction=(masked_prediction != 'NONE'),
                                       vectors=self.inputSpace.vocabulary)

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

        # Store SymbolicSpace ref for ternary lift (concept<->symbol projection).
        # The C-tier SyntacticLayer now lives on WordSpace under the
        # ownership-transfer refactor. Use ``object.__setattr__`` to
        # avoid nn.Module circular submodule registration
        # (SymbolicSpace -> ConceptualSpace -> SyntacticLayer -> SymbolicSpace).
        object.__setattr__(self.wordSpace.conceptualSyntacticLayer,
                           '_symbolic_space', self.symbolicSpace)
        self.conceptualSpace.subspace.basis.monotonic = False

        self.spaces.extend([
            self.inputSpace,
            self.perceptualSpace,
            self.conceptualSpace,
            self.symbolicSpace,
            self.outputSpace,
        ])
        self.spaces.append(self.wordSpace)

        self.inputSpace.outputSpace = self.outputSpace

        # Precompute partition boundaries for partitioned symbolSum
        self._partitions = self._order_partitions(symbol_dim + obj_symbol,
                                                   self.conceptualOrder)
        self.symbol_states = []

        self.to(TheDevice.get())
        TheXMLConfig.validate()

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
        energies = [activation[s:e].norm() for s, e in partitions]
        return int(torch.tensor(energies).argmax())

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

        Uses cached difference to recover both original vectors.
        """
        diff = self._merge_diffs.pop()  # left - right
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
        """Project through OutputSpace."""
        if isinstance(data, torch.Tensor):
            self.outputSpace.subspace.set_event(data)
            data = self.outputSpace.subspace
        self.outputs = self.outputSpace.forward(data)
        return self.outputs.materialize()

    # -- Forward / reverse ----------------------------------------------

    def forward(self, inputData):
        if isinstance(inputData, torch.Tensor):
            inputData = inputData.to(TheDevice.get())

        ws = self.wordSpace

        # 1. Input -> Percept
        self.inputs = self.inputSpace.forward(inputData)
        input_state = self.inputs.materialize()
        percept_quantize = not (
            self.reversible and getattr(self.perceptualSpace, "invertible", False)
        )
        self.percepts = self.perceptualSpace.forward(
            self.inputs, wordSpace=ws, quantize=percept_quantize)
        percepts = self.percepts.materialize()  # [B, nPercepts, dim]

        B = percepts.shape[0]

        # Per-sentence lifecycle: reset the word-stream buffer so each
        # forward pass starts with a clean derivation history. The
        # buffer's capacity is fixed -- rule applications inside
        # compose() push records onto it during this pass.
        if self.wordSpace is not None:
            self.wordSpace.ensure_batch(B)
            self.wordSpace.clear_sentence()

        # -- Pre-loop init --
        self.symbol_states = []
        self.symbolicSpace.reset_symbol_objective()

        if self.useButterflies:
            # ButterflyStage merges internally; no external merge stack.
            x = percepts                     # [B, N_percept, D_percept]
            sym_feedback = None
        elif self.useGrammar == "all":
            # Progressive-bottleneck path: external pair-average merge.
            x = percepts
            sym_feedback = None
            self._merge_diffs = []
            self._sym_feedbacks = []
        else:
            # Flat: feedback is carried via the symbolic state ``ss``
            # that threads between iterations. The reverse path reads
            # _nonrams_sym_feedbacks to unwind the recurrent addition.
            ss = self.symbolicSpace.empty_state(batch=B).to(percepts.device)
            self._nonrams_sym_feedbacks = []

        sym_vectors = None
        self._unified_j_iterations = 0

        # -- Unified Sigma-Pi loop --
        # One ``for t in range(conceptualOrder)`` covers all three modes.
        # Mode-specific logic:
        #   * concept_input construction (merge step vs percept+ss feedback)
        #   * target_count hint to ConceptualSpace.compose
        #   * quantize flag passed to SymbolicSpace.forward
        #   * feedback update at the end of each iteration
        for t in range(self.conceptualOrder):
            # 1. Input construction
            if self.useButterflies:
                concept_input = x
            elif self.useGrammar == "all":
                x = self._butterfly_merge(x)
                if sym_feedback is not None:
                    sym_feedback = (sym_feedback[:, 0::2, :] + sym_feedback[:, 1::2, :]) / 2
                    self._sym_feedbacks.append(sym_feedback)
                    x = x + sym_feedback
                else:
                    self._sym_feedbacks.append(None)
                concept_input = x
            else:
                ss_feedback = self._symbol_feedback_from_vectors(
                    ss, self._symbol_shape[0], percepts.shape[-1])
                self._nonrams_sym_feedbacks.append(ss_feedback)
                concept_input = self._bound_concept_input(percepts + ss_feedback)

            # 2. Sigma (conceptual transformation)
            self.percepts.set_event(concept_input)
            c_target = (None if (self.useButterflies or self.useGrammar == "all")
                        else self.nOutputSymbols)
            self.concepts = self.conceptualSpace[t].forward(
                self.percepts, wordSpace=ws, target_count=c_target)
            concept_vectors = self.concepts.materialize()
            if self.useButterflies or self.useGrammar == "all":
                x = concept_vectors          # carry forward for next merge

            # 3. Pi (symbolic projection)
            is_last_step = (t == self.conceptualOrder - 1)
            quantize_sym = bool(self.useButterflies)
            self.symbols = self.symbolicSpace[t].forward(
                self.concepts, wordSpace=ws,
                quantize=quantize_sym, is_last=is_last_step)
            sym_vectors = self.symbols.materialize()

            # 4. Feedback for next iteration
            if self.useGrammar == "all" and t < self.conceptualOrder - 1:
                _N_t, D_t = self._level_shapes_list[t]
                sym_norms = sym_vectors.norm(dim=-1, keepdim=True)
                sym_feedback = sym_norms.expand(-1, -1, D_t)
            elif not (self.useButterflies or self.useGrammar == "all"):
                ss = sym_vectors

            # 5. Cache symbol states for truth penalty.
            self.symbol_states.append(sym_vectors.clone())
            self._unified_j_iterations += 1

        # thoughtFree / none modes enforce conceptualOrder==0. The loop
        # ran zero times, so do a single pre-seed C->S pass (the spec's
        # "implicit j=-1") so OutputSpace has concept/symbol state.
        if (self.conceptualOrder == 0
                and not self.useButterflies
                and self.useGrammar != "all"):
            self._nonrams_sym_feedbacks.append(ss)
            self.percepts.set_event(self._bound_concept_input(percepts))
            self.concepts = self.conceptualSpace[0].forward(
                self.percepts, wordSpace=ws, target_count=self.nOutputSymbols)
            self.symbols = self.symbolicSpace[0].forward(
                self.concepts, wordSpace=ws, quantize=False, is_last=True)
            sym_vectors = self.symbols.materialize()

        # -- Post-loop finalization (butterfly only) --
        if self.useButterflies:
            self.conceptualSpace.subspace.set_event(x)
            self.concepts = self.conceptualSpace.subspace
            self.symbolicSpace.subspace.set_event(sym_vectors)
            self.symbols = self.symbolicSpace.subspace

        # Discourse snapshot -- stash the final (S, W) pair so runBatch
        # can compute a next-sentence prediction loss against it.
        # S is the symbolic materialization (already committed).
        # W is the WordSubSpace's read() of the current per-sentence
        # stack -- captured now, before ``Start()`` clears the stack for
        # the next sentence. We stash raw tensors; DiscourseSpace's
        # ``snapshot()`` mean-pools batch and pads/truncates to fit.
        self._current_discourse_s = None
        self._current_discourse_w = None
        discourse = getattr(self.wordSpace, 'discourse', None) if self.wordSpace is not None else None
        if discourse is not None:
            # S: the symbolic state at the end of the Sigma-Pi loop.
            try:
                s_state = self.symbolicSpace.subspace.materialize()
            except Exception:
                s_state = sym_vectors
            # W: the word-buffer read. Shape is
            # [batch, max_depth, muxedSize]; empty when no words were
            # pushed (zero tensor per the sentinel convention).
            try:
                w_state = self.wordSpace.read()
            except Exception:
                w_state = None
            if w_state is None:
                w_state = torch.zeros(
                    discourse.max_depth,
                    discourse.n_dim,
                    device=s_state.device, dtype=s_state.dtype)
            self._current_discourse_s = s_state.detach()
            self._current_discourse_w = w_state.detach()

        # Universality evaluation
        self._universality_score = None
        truth_layer = getattr(self.wordSpace, 'truth_layer', None) if self.wordSpace is not None else None
        if truth_layer is not None and len(truth_layer) > 0:
            svo = self.conceptualSpace.last_svo
            if svo is not None:
                s, v, o = svo
                c_sl = self.wordSpace.conceptualSyntacticLayer
                self._universality_score = truth_layer.universality(
                    s, v, o, c_sl.lifting_layer, self.symbolicSpace)

        # Output from first nOutputSymbols of symbol vectors
        if self.useButterflies or self.useGrammar == "all":
            self.symbolicSpace.subspace.set_event(sym_vectors)
        output_syms = sym_vectors[:, :self.nOutputSymbols, :].clone()
        outputData = self.Finish(output_syms)
        symbols = sym_vectors.norm(dim=-1).unsqueeze(-1).expand(
            -1, -1, percepts.shape[-1])

        return input_state, symbols, outputData

    def reverse(self, symbols, outputData):
        ws = self.wordSpace
        if isinstance(outputData, torch.Tensor):
            self.outputSpace.subspace.set_event(outputData)
            outputData = self.outputSpace.subspace

        use_cached = (self.reconstruct == 'symbols')

        if use_cached:
            sym_vec = self.symbolicSpace.subspace.materialize()
            concepts_state = self.concepts
        else:
            output_state = self.outputSpace.reverse(outputData)
            sym_vec = output_state.materialize()
            concepts_state = output_state

        if self.useButterflies:
            # ButterflyStage inverts each stage; reconstruct='symbols'
            # gives us the full symbol state at the final level.
            if not use_cached:
                raise ValueError(
                    "useButterflies requires reconstruct='symbols' "
                    "so the full symbol state is available for exact inversion"
                )
            self.symbolicSpace.subspace.set_event(sym_vec)
            x = self.symbolicSpace[self.conceptualOrder - 1].reverse(
                self.symbolicSpace.subspace, wordSpace=ws).materialize()
            for t in reversed(range(self.conceptualOrder)):
                self.conceptualSpace.subspace.set_event(x)
                x = self.conceptualSpace[t].reverse(
                    self.conceptualSpace.subspace, wordSpace=ws).materialize()
            self.perceptualSpace.subspace.set_event(x)
            input_state = self.perceptualSpace.reverse(self.perceptualSpace.subspace, wordSpace=ws)
            self.inputs = self.inputSpace.reverse(input_state)
            input_latent = input_state.materialize()
            input_data = self.inputs.materialize()
            return input_data, input_latent

        if self.useGrammar == "all":
            # Progressive-bottleneck: peel off last pi, then per-level
            # reverse with external unmerge and feedback subtraction.
            self.symbols.set_event(sym_vec)
            x = self.symbolicSpace[self.conceptualOrder - 1].reverse(
                self.symbols, wordSpace=ws).materialize()
            for t in reversed(range(self.conceptualOrder)):
                self.symbols.set_event(x)
                concept_input_state = self.conceptualSpace[t].reverse(
                    self.symbols, wordSpace=ws)
                x = concept_input_state.materialize()
                fb = self._sym_feedbacks.pop()
                if fb is not None:
                    x = x - fb
                x = self._butterfly_unmerge(x)
            concept_input_state.set_event(x)
        else:
            # Flat recurrent path: reverse sigma, subtract cached feedback.
            concept_input_state = self.conceptualSpace.reverse(
                concepts_state, wordSpace=ws)
            if getattr(self, '_nonrams_sym_feedbacks', None):
                fb = self._nonrams_sym_feedbacks[-1]
                if fb is not None:
                    recovered = concept_input_state.materialize() - fb
                    concept_input_state.set_event(recovered)

        # -- Shared tail: percept/input reverse --
        concept_input = concept_input_state.materialize()
        percepts_portion = concept_input[:, :self.nPercepts, :]

        concept_input_state.set_event(percepts_portion)
        input_state = self.perceptualSpace.reverse(concept_input_state, wordSpace=ws)
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

        # Forward pass to get symbolSum
        input_state, symbols, outputData = self.forward(inputTensor)

        # Get the current symbolSum from the symbolic space
        symbolSum = self.symbolicSpace.subspace.event.clone()  # [B, nC, symbol_dim]

        # Reverse pass to reconstruct
        inputPred, _ = self.reverse(symbols, outputData)

        # Re-encode through forward to get symbolSum_hat
        _, _, _ = self.forward(inputPred)
        symbolSum_hat = self.symbolicSpace.subspace.event  # [B, nC, symbol_dim]

        # Reconstruction loss at symbolic level
        recon_loss = F.mse_loss(symbolSum_hat, symbolSum.detach())

        # Optional luminosity validity penalty
        truth_layer = self._get_truth_layer()
        validity_loss = torch.tensor(0.0, device=recon_loss.device)
        if truth_layer is not None and len(truth_layer) > 0:
            pi_layer = self.symbolicSpace.layer
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
                if state_volume != symbol_volume:
                    errors.append(
                        "useButterflies=true requires latent/symbol volume equality: "
                        f"nPercepts*state_dim ({state_volume}) must equal "
                        f"nSymbols*symbol_width ({symbol_volume})."
                    )
                if state_dim % symbol_width != 0:
                    errors.append(
                        "useButterflies=true requires state_dim to be divisible by "
                        f"symbol_width so n = D/S is integral (got D={state_dim}, S={symbol_width})."
                    )
                if n_percepts & (n_percepts - 1):
                    errors.append(
                        "useButterflies=true butterfly schedule requires nPercepts to be a "
                        f"power of two (got nPercepts={n_percepts})."
                    )
                if str(arch.get("reconstruct", "NONE")).lower() != "symbols":
                    errors.append(
                        "useButterflies=true reverse requires reconstruct=symbols so the full "
                        "symbol state remains available for exact inversion."
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
        reversible = arch.get("reconstruct", "NONE").upper() != "NONE"
        masked_prediction = str(arch.get("training", {}).get("maskedPrediction", "NONE") or "NONE").upper()
        if masked_prediction == "ARIR" and not reversible:
            errors.append(
                "maskedPrediction=ARIR requires reconstruct != NONE. "
                "ARIR is autoregressive input reconstruction, so the model must "
                "have a reverse path."
            )
        percept_inv = gsp(cfg, "PerceptualSpace", "invertible")
        percept_pt = gsp(cfg, "PerceptualSpace", "passThrough")
        # Note: invertible PerceptualSpace shape constraints (nOutput == 2*nInput or
        # 4*nInput*inputDim == nOutput*outputDim for reshape) are registered as
        # requirements inside PerceptualSpace._register_requirements(), not here.

        # Warn about reversible + not invertible: uses pinv which may be numerically unstable
        if reversible and not percept_inv and not percept_pt:
            warnings.warn(
                "PerceptualSpace: reversible=True with invertible=False uses two "
                "InvertiblePiLayers with separate weights. The reverse path involves "
                "a matrix pseudoinverse (pinv) which may be numerically unstable. "
                "Consider setting <invertible>true</invertible> for shared-weight "
                "inversion, or be aware of potential SVD convergence failures.",
                stacklevel=2)

        if errors:
            raise ValueError(
                "XML config inconsistencies:\n  - " + "\n  - ".join(errors))

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

        dataset = os.environ.get("BASIC_DATASET", dat.get("dataset"))
        # Environment overrides for num_shards/max_docs (set by train.py)
        num_shards = int(os.environ.get("BASIC_NUM_SHARDS", dat.get("numShards", 1)))
        max_docs = int(os.environ.get("BASIC_MAX_DOCS", dat.get("maxDocs", 10000)))

        # DataLoader prefetch workers: 0 means synchronous in-process
        # batch assembly (default for small / local runs).
        _num_workers = int(trn.get("numWorkers", 0))

        TheData.load(dataset,
                     num_shards=num_shards,
                     max_docs=max_docs,
                     shard_dir=dat.get("shardDir"),
                     dat=dat)

        m, _ = BaseModel.from_config(config_path, data=TheData)
        m._num_workers = _num_workers
        TheMessage(f"Device: {TheDevice}")

        m = compile(m)
        # compile() may return a wrapper; preserve the flag there too.
        m._num_workers = _num_workers

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
            wpath = TheXMLConfig.get("architecture.weightsPath", "weights.ckpt")
            wpath = m._resolve_artifact_path(wpath)
            m.save_weights(wpath)
            m.save_embeddings()

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

    TheReport.write_html()
