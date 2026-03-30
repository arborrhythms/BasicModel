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
from vector_quantize_pytorch import ResidualVQ, VectorQuantize
import torch.optim as optim
from torch.profiler import profile as torch_profile, ProfilerActivity, schedule as profiler_schedule
from functools import partial
from datetime import datetime

import util
TheDevice = util.TheDevice
TheMessage = util.TheMessage

from visualize import Report, TheReport
from util import ProjectPaths, compile, TheXMLConfig, init_config, init_compile_backend
from embed import WordVectors, PretrainModel
from data import TheData, Data

from Model import Layer, PiLayer, SigmaLayer # Import custom layers from Model.py
from Model import VQLayer, NormLayer, LinearLayer, AttentionLayer
from Model import ColumnUsageTracker, LiftingLayer, CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon

from Space import ActiveEncoding, WhereEncoding, WhenEncoding, WhatEncoding, EventEncoding
from Space import Basis, Tensor, Codebook, Embedding
from Space import SubSpace, Space, InputSpace, PerceptualSpace, ModalSpace, ConceptualSpace, SymbolicSpace, SyntacticSpace, OutputSpace

class BaseModel(nn.Module):
    """Shared training, plotting, and persistence infrastructure for all models."""
    name           = "BaseModel"
    spaces         = []
    reversible    = False
    plot           = False
    _optimizer     = None

    @staticmethod
    def load_config(config_path=None):
        """Load model settings from an XML config file.

        Delegates to XMLConfig._parse_xml().  Returns a dict of dicts;
        missing fields are filled by create_from_config() using model.xml.
        """
        if config_path is None:
            config_path = os.path.join(ProjectPaths.PROJECT_DIR, "model.xml")
        from util import XMLConfig
        return XMLConfig._parse_xml(config_path)

    @staticmethod
    def from_config(config_path=None, model_type=None, data=None):
        """Factory: create the right model type from XML config."""
        if config_path is None:
            config_path = os.path.join(ProjectPaths.PROJECT_DIR, "data", "xor.xml")
        resolved_path = ModelFactory.resolve_xml(config_path)
        raw_cfg = BaseModel.load_config(resolved_path)
        arch = raw_cfg.get("architecture", {})
        model_kind = str(arch.get("type", "basic") or "basic").strip().lower()
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
        model_family = str(arch.get("type", "basic") or "basic").strip().lower()
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

        # InputSpace: if nOutput=0, derive from data at create() time (passed as 0 → handled there)
        nInput    = _s("InputSpace", "nOutput")   # 0 = let create() derive from data
        nPercepts = _resolve("PerceptualSpace", nInput)
        nConcepts = _resolve("ConceptualSpace", nPercepts)
        nSymbols  = _resolve("SymbolicSpace",   nConcepts)
        nWords    = _resolve("SyntacticSpace",   nSymbols)
        nOutput   = _resolve("OutputSpace",      nSymbols)

        _nObjects = (
            _s("InputSpace", "nVectors")
            + _s("PerceptualSpace", "nVectors")
            + _s("ConceptualSpace", "nVectors")
            + _s("SymbolicSpace", "nVectors")
            + _s("SyntacticSpace", "nVectors")
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
            symbolicOrder=arch["symbolicOrder"],
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
        if self.optimize_embedding and isinstance(self.inputSpace.get_vectors(), Embedding):
            emb_params = self.inputSpace.get_vectors().embedding_parameters()
            self.inputSpace.params = self.inputSpace.params + emb_params
        self.loss.embedding_scale = float(_t("embeddingScale") or 0.1)
        if isinstance(self.inputSpace.get_vectors(), Embedding):
            self.inputSpace.get_vectors().optimize_embedding = self.optimize_embedding
            object.__setattr__(self.inputSpace.get_vectors(), "_model", self)

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
        """Build an Adam optimizer over all space parameters.

        Uses getParameters() from each Space (the universal training contract),
        which excludes temperature params managed by alpha_update.
        Falls back to standard PyTorch parameters() when not in ergodic mode.

        When trainEmbedding is NONE or ARLM, embedding parameters are excluded
        from the optimizer.
        """
        if getattr(self, 'ergodic', True):
            params = []
            seen = set()
            for s in self.spaces:
                for p in s.getParameters():
                    if p.data_ptr() not in seen:
                        seen.add(p.data_ptr())
                        params.append(p)
        else:
            params = list(self.parameters())
        # Exclude embedding params when trainEmbedding is NONE or ARLM
        if not getattr(self, 'optimize_embedding', False):
            exclude = set()
            if hasattr(self, 'inputSpace') and isinstance(self.inputSpace.get_vectors(), Embedding):
                for p in self.inputSpace.get_vectors().embedding_parameters():
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
        if hasattr(self, 'inputSpace') and isinstance(self.inputSpace.get_vectors(), Embedding):
            return self.inputSpace.get_vectors()
        return None

    def save_weights(self, path=None):
        """Persist model weights (excluding embeddings) to disk.

        Embedding weights live in a separate artifact (the .kv/.pt file
        specified by <embeddingPath> in the XML config).  The three files
        — XML config, embedding artifact, weights checkpoint — partition
        the model's behaviour and are managed independently.
        """
        if path is None:
            path = os.path.join(ProjectPaths.OUTPUT_DIR, "weights.ckpt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Filter out embedding parameters — they belong to the .kv artifact
        state = {k: v for k, v in self.state_dict().items()
                 if "wv._vectors" not in k}
        torch.save({"state_dict": state}, path)
        TheMessage(f"[{self.name}] Weights saved to {path}")

    def save_embeddings(self, path=None):
        """Snapshot current nn.Embedding weights and save the .pt artifact."""
        if path is None:
            path = getattr(self, 'embedding_path', None)
        if path is None:
            return
        emb = self._get_embedding()
        if emb is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        emb.save_embeddings(path)
        TheMessage(f"[{self.name}] Embeddings saved to {path}")

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
                f"[{self.name}] Embedding dimension mismatch — cannot load {path}\n"
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
        mismatches = [
            (k, list(state[k].shape), list(model_state[k].shape))
            for k in state if k in model_state
            and state[k].shape != model_state[k].shape
        ]
        missing = [k for k in model_state if k not in state]
        unexpected = [k for k in state if k not in model_state]
        if mismatches or missing or unexpected:
            lines = [f"[{self.name}] Weight file mismatch — cannot load {path}"]
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
                # Character-level accuracy: strip nulls only for display;
                # pad with '\x00' (not space) to match the actual input encoding
                # so the model is scored on correctly predicting null padding.
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

    Higher-order processing (conceptualOrder, symbolicOrder) inserts additional
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
               conceptualOrder=1, symbolicOrder=1,
               model_type="simple", data=None, embedding_path=None,
               reverse_scale=0.5, what_scale=0.7, where_scale=0.2, when_scale=0.1,
               masked_prediction='NONE', reconstruct='NONE'):
        """Build the full space hierarchy from architecture parameters.

        Config-derivable flags (reshape, ergodic, quantized, etc.) are read
        from TheXMLConfig by each Space constructor.  Only runtime/pipeline
        params are passed here.

        Args:
            nInput/nPercepts/nConcepts/nSymbols/nOutput: object counts per space.
            nWords: object count for the SyntacticSpace (used when symbolicOrder >= 1).
            conceptualOrder: number of extra Percept->Concept->Symbol cycles.
            symbolicOrder: number of extra Syntax->Symbol cycles.
            model_type: "simple", "embedding", "passthrough", or "vq".
        """
        self.spaces = []  # reset — prevent stale accumulation from prior create() calls
        TheXMLConfig._requirements.clear()  # clear stale requirements from prior create()/tests
        # Read config-derivable flags
        self.reconstruct     = reconstruct.lower()
        self.reversible      = str(TheXMLConfig.get("architecture.reconstruct")).upper() != "NONE"
        self.ergodic          = TheXMLConfig.get("architecture.ergodic")
        self.processSymbols   = TheXMLConfig.get("architecture.processSymbols")
        self.certainty        = TheXMLConfig.get("architecture.certainty")
        self.syntax           = TheXMLConfig.get("architecture.syntax", False)
        self.lexer            = TheXMLConfig.space("InputSpace", "lexer")
        self.quantized        = TheXMLConfig.space("InputSpace", "quantized")
        self.perceptQuantized = TheXMLConfig.space("PerceptualSpace", "quantized")
        self.conceptQuantized = TheXMLConfig.space("ConceptualSpace", "quantized")
        self.perceptPassThrough = TheXMLConfig.space("PerceptualSpace", "passThrough")
        self.symbolPassThrough  = TheXMLConfig.space("SymbolicSpace", "passThrough")
        self.invertible       = TheXMLConfig.space("PerceptualSpace", "invertible")
        self.hasNorm          = TheXMLConfig.space("ConceptualSpace", "hasNorm")
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
        self.symbolicOrder    = symbolicOrder
        self.loss = ModelLoss(reverse_scale=reverse_scale,
                         what_scale=what_scale,
                         where_scale=where_scale,
                         when_scale=when_scale,
                         certainty=self.certainty,
                         nOutput=nOutput,
                         conceptualOrder=conceptualOrder,
                         symbolicOrder=symbolicOrder,
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
            raw = TheXMLConfig.space(section, "nDim")
            return prev_dim if raw == 0 else raw

        input_dim   = _resolve_dim("InputSpace",    1)
        percept_dim = _resolve_dim("PerceptualSpace",  input_dim)
        concept_dim = _resolve_dim("ConceptualSpace",  percept_dim)
        symbol_dim  = _resolve_dim("SymbolicSpace",    concept_dim)
        syntax_dim  = _resolve_dim("SyntacticSpace",   symbol_dim)
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
        obj_syntax  = _obj_size("SyntacticSpace")
        obj_output  = _obj_size("OutputSpace")

        # Resolve nVectors sentinels (0 → same as output count for that space)
        def _nvec(section, n_out):
            raw = TheXMLConfig.space(section, "nVectors")
            return n_out if raw == 0 else raw

        nvec_input   = _nvec("InputSpace",    nInput)
        nvec_percept = _nvec("PerceptualSpace", nPercepts)
        nvec_concept = _nvec("ConceptualSpace", nConcepts)
        nvec_symbol  = _nvec("SymbolicSpace",  nSymbols)
        nvec_syntax  = _nvec("SyntacticSpace", nSymbols)
        nvec_output  = _nvec("OutputSpace",    nOutput)

        # Build I/O shape tuples: [count, dim + objectSize]
        # Each space's shape includes its own objectSize.
        inputShape   = [nInput,    input_dim   + obj_input]
        perceptShape = [nPercepts, percept_dim + obj_percept]
        conceptShape = [nConcepts, concept_dim + obj_concept]
        symbolShape  = [nSymbols,  symbol_dim  + obj_symbol]
        outputShape  = [nOutput,   output_dim  + obj_output]

        # Build codebook (space-internal) shape tuples: [nVectors, nDim]
        # spaceShape uses raw content dim — codebook vectors don't include objectSize.
        spaceShape_input   = [nvec_input,   input_dim]
        spaceShape_percept = [nvec_percept, percept_dim]
        spaceShape_concept = [nvec_concept, concept_dim]
        spaceShape_symbol  = [nvec_symbol,  symbol_dim]
        spaceShape_syntax  = [nvec_syntax,  syntax_dim]
        spaceShape_output  = [nvec_output,  output_dim]

        # nOutputSymbols tracks total symbol count fed to OutputSpace.
        # Starts with only the output-destined symbols (not reconstruction symbols).
        # It grows as higher-order cycles (conceptualOrder, symbolicOrder) append symbols.
        nOutputSymbols = self.nOutputSymbols
        # InputSpace receives raw data (no encoding) as input but produces encoded vectors.
        rawInputShape = [nInput, input_dim]
        self.inputSpace      = self._make_input_space(rawInputShape, spaceShape_input, inputShape,
                                                      model_type=model_type)
        # Convert masked-word string labels to embedding vectors now that
        # the Embedding vocabulary is available.
        if data is not None and hasattr(data, '_lm_labels') and data._lm_labels is not None:
            embedding = self.inputSpace.get_vectors() if self.inputSpace.subspace.event is not None else None
            if embedding is not None and hasattr(embedding, 'pretrain'):
                data.prepare_lm_targets(embedding)
                # Move new targets to device
                data.toDevice()
        self.perceptualSpace = self._make_perceptual_space(inputShape, spaceShape_percept, perceptShape)
        self.conceptualSpace = ConceptualSpace(perceptShape, spaceShape_concept, conceptShape)
        self.symbolicSpace   = SymbolicSpace(conceptShape, spaceShape_symbol, symbolShape,
                                             conceptualSpace=self.conceptualSpace)
        self.spaces.extend([self.inputSpace, self.perceptualSpace, self.conceptualSpace, self.symbolicSpace])

        if self.conceptualOrder == 2:
            self.perceptualSpace2 = PerceptualSpace(conceptShape, spaceShape_percept, perceptShape)
            self.conceptualSpace2 = ConceptualSpace(perceptShape, spaceShape_concept, conceptShape)
            self.symbolicSpace2   = SymbolicSpace(conceptShape, spaceShape_symbol, symbolShape,
                                                  conceptualSpace=self.conceptualSpace2)
            nOutputSymbols += (self.conceptualOrder - 1) * self.nSymbols
            self.spaces.extend([self.perceptualSpace2, self.conceptualSpace2, self.symbolicSpace2])

        if self.symbolicOrder == 2:
            # SyntacticSpace3 receives the full symbol tensor (nSymbols objects)
            self.syntacticSpace3 = SyntacticSpace(symbolShape, spaceShape_syntax, symbolShape)
            self.symbolicSpace3  = SymbolicSpace(symbolShape, spaceShape_symbol, symbolShape)
            nOutputSymbols += (self.symbolicOrder - 1) * self.nSymbols
            self.spaces.extend([self.syntacticSpace3, self.symbolicSpace3])

        self.nTotalOutputSymbols = nOutputSymbols
        self.outputSpace     = OutputSpace([nOutputSymbols, symbol_dim + obj_symbol], spaceShape_output, outputShape,
                                           masked_prediction=(masked_prediction != 'NONE'),
                                           vectors=self.inputSpace.get_vectors())
        self.spaces.extend([self.outputSpace])
        self.inputSpace.outputSpace = self.outputSpace

        # The output dimensionality of the input layer must be equal to the output dimensionality of the perceptual layer, since the conceptual layer operates on both.
        #assert self.inputSpace.outputShape[1] == self.perceptualSpace2.outputShape[1] # inputDim == perceptDim
        # The input dimensionality of the symbolic layer must be equal to the input dimensionality of the perceptual layer, since they both operate on the output of the conceptual layer.
        #assert self.symbolicSpace.inputShape[1] == self.perceptualSpace2.inputShape[1] == self.conceptualSpace.outputShape[1]#  conceptDim = conceptDim
        # The output shape of the symbolic space is equal to the input shape of the output space
        #assert self.symbolicSpace.outputShape[1] == self.outputSpace.inputShape[1] # these are in conceptual space, or symbolic space if symbols emit objectSize symbols (processSymbols == True)

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
        self.inputs = self.inputSpace.forward(inputData)
        self.percepts = self.perceptualSpace.forward(self.inputs)
        self.concepts = self.conceptualSpace.forward(self.percepts)
        self.symbols = self.symbolicSpace.forward(self.concepts)
        input = self.inputs.materialize()
        concepts = self.concepts.materialize()
        symbols = self.symbols.materialize()
        if self.plot:
            TheReport.plotActivations(figure=1, concepts=concepts)
        return input, concepts, symbols
    def StartReverse(self, symbols):
        """Reverse pass: Symbol -> Concept -> Percept -> Input (reconstruction)."""
        if isinstance(symbols, torch.Tensor):
            self.symbolicSpace.subspace.set_vectors(symbols)
            symbols = self.symbolicSpace.subspace
        concepts_state = self.symbolicSpace.reverse(symbols)
        percepts_state = self.conceptualSpace.reverse(concepts_state)
        input_state = self.perceptualSpace.reverse(percepts_state)
        self.inputs = self.inputSpace.reverse(input_state)
        input = input_state.materialize()
        inputData  = self.inputs.materialize()
        return inputData, input
    def SubsymbolicThought(self, data):
        """Extra Percept->Concept->Symbol cycle (conceptualOrder >= 1)."""
        if isinstance(data, torch.Tensor):
            self.perceptualSpace2.subspace.set_vectors(data)
            data = self.perceptualSpace2.subspace
        percepts_state = self.perceptualSpace2.forward(data)
        concepts_state = self.conceptualSpace2.forward(percepts_state)
        symbols_state  = self.symbolicSpace2.forward(concepts_state)
        percepts = percepts_state.materialize()
        concepts = concepts_state.materialize()
        symbols = symbols_state.materialize()
        if self.plot:
            TheReport.plotActivations(figure=1, percepts=percepts, concepts=concepts)
        return concepts, symbols
    def SubsymbolicThoughtReverse(self, concepts, symbols):
        """Reverse of SubsymbolicThought."""
        if isinstance(symbols, torch.Tensor):
            self.symbolicSpace2.subspace.set_vectors(symbols)
            symbols = self.symbolicSpace2.subspace
        concepts_state = self.symbolicSpace2.reverse(symbols)
        percepts_state = self.conceptualSpace2.reverse(concepts_state)
        percepts = percepts_state.materialize()
        return percepts
    def SyntacticDerivation(self):
        """Run per-space syntactic derivation: predict rules, then execute projections.

        Two phases at each cognitive space:
          1. **Predict:** syntactic_layer.forward(activation) → rule distributions + word tuples
          2. **Project:** space.projectXxx(rule, ...) → composed representation

        Representation types by space:
          - Symbolic: [B, N] scalar activations (association/attention)
          - Conceptual: [B, N, D] embedded vectors (mereological composition)
          - Perceptual: SubSpace (word embedding recovery)

        Transition rules (S→C at rule 6, C→P at rule 10) signal hand-off.

        Word tuples from all three layers are collected into self.all_words
        (using global Grammar rule IDs) for XML parse tree composition.

        Returns: list of (batch, vector, rule) word tuples across all spaces.
        """
        all_words = []

        # ── Symbolic derivation (rules 1-5, transition 6) ────────
        # Predicts rules AND executes soft-weighted projections on [B, N] activations
        sym_act = self.symbols.get_activation()
        if sym_act is not None and hasattr(self.symbolicSpace, 'composeSyntax'):
            sym_out = self.symbolicSpace.composeSyntax(sym_act)
            all_words.extend(sym_out["words"])
            self._symbolic_syntax_out = sym_out

        # ── Conceptual derivation (rules 7-9, transition 10) ─────
        # Predicts rules AND executes soft-weighted projections on [B, N, D] vectors
        con_act = self.concepts.get_activation()
        con_vec = self.concepts.materialize()
        if con_act is not None and hasattr(self.conceptualSpace, 'composeSyntax'):
            con_out = self.conceptualSpace.composeSyntax(con_act, con_vec)
            all_words.extend(con_out["words"])
            self._conceptual_syntax_out = con_out

        # ── Perceptual derivation (rule 11: terminal) ────────────
        # Terminal rule — only predicts, no composition needed
        per_act = self.percepts.get_activation()
        if per_act is not None and hasattr(self.perceptualSpace, 'syntactic_layer') \
                and self.perceptualSpace.syntactic_layer is not None:
            per_out = self.perceptualSpace.syntactic_layer.forward(per_act)
            all_words.extend(per_out["words"])
            self._perceptual_syntax_out = per_out

        self.all_words = all_words
        return all_words

    def get_parse_tree(self, batch_index=0):
        """Return the XML parse tree for a batch element after forward pass.

        Must be called after forward() (which runs SyntacticDerivation).
        Uses the vocabulary from InputSpace to map vector indices to words.

        Args:
            batch_index: which batch element to emit (default 0).

        Returns:
            XML string, or empty string if no derivation was produced.
        """
        from parse import derivation_to_xml
        from Model import Grammar
        words = [(b, v, r) for b, v, r in getattr(self, 'all_words', [])
                 if b == batch_index]
        if not words:
            return ""
        vocab = self.inputSpace.get_vocabulary() if hasattr(self.inputSpace, 'get_vocabulary') else None
        return derivation_to_xml(words, Grammar(), vocab)

    def SymbolicThought(self, data):
        """Extra Syntax->Symbol cycle (symbolicOrder >= 1)."""
        if isinstance(data, torch.Tensor):
            self.syntacticSpace3.subspace.set_vectors(data)
            data = self.syntacticSpace3.subspace
        words_state = self.syntacticSpace3.forward(data)
        symbols_state = self.symbolicSpace3.forward(words_state)
        words = words_state.materialize()
        symbols = symbols_state.materialize()
        if self.plot:
            TheReport.plotActivations(figure=1, symbols=symbols)
        return symbols, words
    def SymbolicThoughtReverse(self, symbols, words):
        """Reverse of SymbolicThought."""
        if isinstance(words, torch.Tensor):
            self.syntacticSpace3.subspace.set_vectors(words)
            words = self.syntacticSpace3.subspace
        symbols_state = self.syntacticSpace3.reverse(words)
        data_state = self.symbolicSpace3.reverse(symbols_state)
        data = data_state.materialize()
        return data
    def Finish(self, symbols):
        """Project concatenated symbols to task output via OutputSpace."""
        if isinstance(symbols, torch.Tensor):
            self.outputSpace.subspace.set_vectors(symbols)
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
            self.outputSpace.subspace.set_vectors(outputData)
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

    def infer(self, text, max_length=None, mode=None):
        """Autoregressive inference via the standard batch pipeline.

        Two modes:

        ``ARLM`` (append-and-rerun): stages seed text, runs forward,
        decodes the output token, appends it to the input via
        ``pushInput()``, and repeats.  Each iteration re-lexes and
        re-embeds the full (growing) input.

        ``ARIR`` (autoregressive input reconstruction, default): TODO —
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
        # Run per-space syntactic derivation (grammar rules on activations)
        if self.syntax:
            self.SyntacticDerivation()
        # Higher-order subsymbolic cycles (conceptualOrder extra passes)
        for n in range(1,self.conceptualOrder):
            NA, symbols1 = self.SubsymbolicThought(concepts)
            symbols = torch.cat((symbols, symbols1), dim=1)
        # Higher-order symbolic cycles (symbolicOrder extra passes)
        for n in range(1,self.symbolicOrder):
            NA, symbols2 = self.SymbolicThought(symbols)
            symbols = torch.cat((symbols, symbols2), dim=1)
        # Split AFTER higher-order cycles: output symbols for prediction,
        # recon symbols for reconstruction
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
        """Full reverse pass: unwind higher-order cycles then core reconstruction.

        Slices the concatenated symbol tensor to route each chunk to its
        corresponding reverse stage, in reverse order of the forward pass.
        """
        symbols = self.FinishReverse(outputData)
        nSym = round(self.nSymbols)
        symbolIndex = 0
        for n in range(1, self.symbolicOrder):
            symbols1 = symbols[:, symbolIndex*nSym:(symbolIndex+1)*nSym]
            symbolIndex += 1
            symbols = self.SymbolicThoughtReverse(symbols, symbols1)
        for n in range(1, self.conceptualOrder):
            symbols1 = symbols[:, symbolIndex*nSym:(symbolIndex+1)*nSym]
            symbolIndex += 1
            symbols = self.SubsymbolicThoughtReverse(symbols, symbols1)
        # Final chunk goes to the core reverse pipeline
        symbols = symbols[:, symbolIndex * nSym:(symbolIndex + 1) * nSym]
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
                total   = predicted.size(0)
                correct = (predicted == actual).sum().item()
                accuracy += [correct / total]
                TheMessage(f"Test Accuracy: {100 * correct / total:.2f}%")
            else:
                _, predicted = torch.max(allOut, 1)
                _, actual = torch.max(self.outputSpace.getTestOutput(), 1)
                total   = predicted.size(0)
                correct = (predicted == actual).sum().item()
                accuracy += [correct / total]
                TheMessage(f"Test Accuracy: {100 * correct / total:.2f}%")

            self.inputSpace.shuffle()

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
            emb = self.inputSpace.get_vectors()
            if isinstance(emb, Embedding):
                sentences = self._get_sentences(split)
                if sentences and index < len(sentences):
                    sentence = sentences[index]
                    from parse import quick_parser
                    words = [t for t, _ in quick_parser(sentence)]
                    if te in ('JOINT'):
                        sbow = self.inputSpace.sbow_loss(words)
                    elif te in ('CBOW', 'SBOW', 'BOTH'):
                        # CBOW uses padded context; SBOW and BOTH use the faster centroid method
                        method = 'CBOW' if te == 'CBOW' else 'SBOW'
                        self.inputSpace.train_embeddings(words, method=method)
        return sbow

    def runBatch(self, train=True, batchNum=0, batchSize=10, split="train",
                 optimizer=None):
        """Run a single batch: forward pass, loss, and (if training) backward + step.

        Args:
            train: whether to compute gradients and update parameters.
            batchNum: opaque cursor returned by getBatch for the next batch.
            batchSize: number of examples per batch.
            split: "train", "test", or "validation".
            optimizer: pre-built optimizer (required when train=True).

        Returns:
            (BatchResult, nextBatchNum) on success, or (None, batchNum) when
            the dataset is exhausted.
        """
        sentenceIdx = batchNum  # sentence index before getBatch increments
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

        # ARUS: suppress output loss (unsupervised — no target signal)
        if hasattr(self, 'masked_prediction') and self.masked_prediction == 'ARUS':
            lossOut = torch.tensor(0.0, device=TheDevice.get())

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
            else:
                lossIn = self.loss.compute(pred_sq, target_sq)
        else:
            inputDataPred = None
            lossIn = None

        # JOINT mode: compute SBOW embedding loss
        sbow = None
        if train:
            sbow = self.trainEmbeddings(('JOINT'), sentenceIdx, split)

        totalLoss = self.loss.total(lossOut, lossIn, sbow)

        TheMessage(f"batch = {batchNum}, loss = {totalLoss} ")

        if train:
            totalLoss.backward()
            if self.ergodic:
                self.paramUpdate()
            optimizer.step()

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

        Uses getBatch() stream interface for flexible batch iteration.
        Delegates per-batch work to ``runBatch()``.

        In inference mode (split="runtime", no optimizer): skips loss
        construction, output accumulation, progress printing, and CBOW
        updates.  Returns immediately after the getBatch/runBatch loop.

        Args:
            optimizer: pre-built Adam optimizer (persistent across epochs).
                       Pass None for evaluation mode.
            batchSize: number of examples per batch (standard mode only)
            split: "train", "test", or "validation"

        Returns (output_loss, reconstruction_loss, all_predictions, last_reconstruction).
        For inference mode, returns (0, 0, [], []).
        """
        training = optimizer is not None
        inference = split == "runtime" and not training
        self.train(training)
        self.outputSpace.clearBatchResults()
        ctx = torch.no_grad() if not training else nullcontext()


        # Inference fast path: skip loss construction and accumulation
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

        # Training / evaluation path
        allOutput = []
        outputChunks = []
        allInput = []
        inputChunks = []
        outErr = 0
        inErr = 0
        masked_pred = hasattr(self, 'masked_prediction') and self.masked_prediction != 'NONE'

        with ctx:
            batchNum = 0
            while True:
                result, batchNum = self.runBatch(
                    train=training, batchNum=batchNum, batchSize=batchSize,
                    split=split, optimizer=optimizer,
                )
                if result is None:
                    break

                self.outputSpace.putBatch(result)

                # Embedding training (post-batch, needs batchNum for sentence lookup)
                if training and masked_pred:
                    self.trainEmbeddings(('CBOW', 'SBOW', 'BOTH'), batchNum, split)

                outErr = result.lossOut.item()
                inErr = result.lossIn.item()

                outputDataPred = result.outputPred.clone().detach().squeeze()
                outputChunks.append(outputDataPred)

                if self.reversible and result.inputPred is not None:
                    inputDataPred = result.inputPred.clone().detach().squeeze()
                    inputChunks.append(inputDataPred)

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

    BatchResult = BasicModel.BatchResult
    runBatch    = BasicModel.runBatch
    runEpoch    = BasicModel.runEpoch
    runTrial    = BasicModel.runTrial
    infer       = BasicModel.infer

    def create(self, nInput, nPercepts, nConcepts, nSymbols, nWords=16, nOutput=32,
               conceptualOrder=1, symbolicOrder=1,
               model_type="simple", data=None, embedding_path=None,
               reverse_scale=0.5,
               what_scale=0.7, where_scale=0.2, when_scale=0.1,
               masked_prediction='NONE', reconstruct='NONE', **kwargs):

        self.spaces = []
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
        self.certainty = TheXMLConfig.get("architecture.certainty")
        self.quantized = TheXMLConfig.space("InputSpace", "quantized")
        self.perceptQuantized = TheXMLConfig.space("PerceptualSpace", "quantized")
        self.conceptQuantized = TheXMLConfig.space("ConceptualSpace", "quantized")
        self.conceptualOrder = conceptualOrder
        self.symbolicOrder = symbolicOrder
        self.reconstruct = reconstruct.lower()
        self.masked_prediction = masked_prediction
        self.syntax = TheXMLConfig.get("architecture.syntax", False)
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
            symbolicOrder=symbolicOrder,
            nWhere=TheXMLConfig.get("architecture.nWhere"),
            nWhen=TheXMLConfig.get("architecture.nWhen"),
        )

        # Resolve dims, chaining through the pipeline (nDim=0 → same as input dim)
        def _resolve_dim(section, prev_dim):
            raw = TheXMLConfig.space(section, "nDim")
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

        # Resolve nVectors sentinels (0 → same as output count for that space)
        def _nvec(section, n_out):
            raw = TheXMLConfig.space(section, "nVectors")
            return n_out if raw == 0 else raw

        nvec_input   = _nvec("InputSpace",    nInput)
        nvec_percept = _nvec("PerceptualSpace", nPercepts)
        nvec_concept = _nvec("ConceptualSpace", nConcepts)
        nvec_symbol  = _nvec("SymbolicSpace",  nSymbols)
        nvec_output  = _nvec("OutputSpace",    nOutput)

        # Build I/O shape tuples: [count, dim + objectSize]
        # Shapes include the full vector width (content + spatial encoding).
        # Each space's shape includes its own objectSize.
        inputShape   = [nInput,            input_dim   + obj_input]
        perceptShape = [nPercepts,          percept_dim + obj_percept]
        conceptShape = [nConcepts,          concept_dim + obj_concept]
        symbolShape  = [nSymbols,           symbol_dim  + obj_symbol]
        outputShape  = [nOutput,            output_dim  + obj_output]
        # MentalModel joins percepts + concepts before symbolicSpace
        # When PerceptualSpace is passThrough, only concepts feed SymbolicSpace
        _perceptPassThrough = TheXMLConfig.space("PerceptualSpace", "passThrough")
        if _perceptPassThrough:
            joinShape = [nConcepts, concept_dim + obj_concept]
        else:
            joinShape = [nPercepts + nConcepts, concept_dim + obj_concept]

        # Build codebook (space-internal) shape tuples: [nVectors, nDim]
        # spaceShape uses raw content dim — codebook vectors don't include objectSize.
        spaceShape_input   = [nvec_input,   input_dim]
        spaceShape_percept = [nvec_percept, percept_dim]
        spaceShape_concept = [nvec_concept, concept_dim]
        spaceShape_symbol  = [nvec_symbol,  symbol_dim]
        spaceShape_output  = [nvec_output,  output_dim]

        rawInputShape = [nInput, input_dim]
        self.inputSpace = InputSpace(rawInputShape, spaceShape_input, inputShape,
                                     model_type=model_type)

        # Branch 1: Input -> Percepts
        self.perceptualSpace = PerceptualSpace(inputShape, spaceShape_percept, perceptShape)

        # Branch 2: Input -> Concepts
        self.conceptualSpace = ConceptualSpace(inputShape, spaceShape_concept, conceptShape)

        # Join: [Percepts, Concepts] -> Symbols
        self.symbolicSpace = SymbolicSpace(joinShape, spaceShape_symbol, symbolShape,
                                           conceptualSpace=self.conceptualSpace)

        # SyntacticSpace operates on recon_symbols (all symbols when nOutput=0)
        self.syntacticSpace = None
        if self.syntax:
            syntax_dim = _resolve_dim("SyntacticSpace", symbol_dim)
            spaceShape_syntax = [nvec_symbol, syntax_dim]
            self.syntacticSpace = SyntacticSpace(symbolShape, spaceShape_syntax, symbolShape)

        self.outputSpace = OutputSpace([self.nOutputSymbols, symbol_dim + obj_symbol], spaceShape_output, outputShape,
                                       masked_prediction=(masked_prediction != 'NONE'),
                                       vectors=self.inputSpace.get_vectors())

        self.spaces.extend([
            self.inputSpace,
            self.perceptualSpace,
            self.conceptualSpace,
            self.symbolicSpace,
        ])
        if self.syntacticSpace is not None:
            self.spaces.append(self.syntacticSpace)
        self.spaces.append(self.outputSpace)

        self.inputSpace.outputSpace = self.outputSpace
        self.to(TheDevice.get())
        TheXMLConfig.validate()

    def Start(self, inputData):
        self.inputs = self.inputSpace.forward(inputData)
        self.concepts = self.conceptualSpace.forward(self.inputs)
        input_state = self.inputs.materialize()
        concepts = self.concepts.materialize()
        if self.perceptualSpace and not self.perceptualSpace.passThrough:
            self.percepts = self.perceptualSpace.forward(self.inputs)
            percepts = self.percepts.materialize()
            merged = torch.cat([percepts, concepts], dim=1)
        else:
            self.percepts = None
            merged = concepts
        self.symbols = self.symbolicSpace.forward(self._wrap_reverse(self.symbolicSpace, merged))
        symbols = self.symbols.materialize()
        return input_state, concepts, symbols

    def Finish(self, symbols):
        if isinstance(symbols, torch.Tensor):
            symbols = self._wrap_reverse(self.outputSpace, symbols)
        self.outputs = self.outputSpace.forward(symbols)
        return self.outputs.materialize()

    def forward(self, inputData):
        if isinstance(inputData, torch.Tensor):
            inputData = inputData.to(TheDevice.get())
        input_state, concepts, symbols = self.Start(inputData)
        # Split symbols: output_symbols → OutputSpace, recon_symbols → SyntacticSpace
        if self.nReconSymbols > 0:
            self.output_symbols = symbols[:, :self.nTotalOutputSymbols, :]
            self.recon_symbols = symbols  # all symbols when nOutput=0
        else:
            self.output_symbols = symbols
            self.recon_symbols = None
        # Route recon_symbols through SyntacticSpace (grammar derivation)
        self.syntax_state = None
        if self.recon_symbols is not None and self.syntacticSpace is not None:
            self.syntax_state = self.syntacticSpace.forward(
                self._wrap_reverse(self.syntacticSpace, self.recon_symbols))
        outputData = self.Finish(self.output_symbols)
        return input_state, symbols, outputData

    def _wrap_reverse(self, space, tensor):
        """Wrap a raw tensor in the space's subspace for reverse()."""
        space.subspace.set_vectors(tensor)
        return space.subspace

    def reverse(self, symbols, outputData):
        # Pass SubSpaces through the chain (like BasicModel.StartReverse)
        if isinstance(outputData, torch.Tensor):
            self.outputSpace.subspace.set_vectors(outputData)
            outputData = self.outputSpace.subspace
        symbols_state = self.outputSpace.reverse(outputData)

        # SyntacticSpace reverse: recover activation from word tuples
        if hasattr(self, 'syntax_state') and self.syntax_state is not None:
            symbols_state = self.syntacticSpace.reverse(self.syntax_state)

        concepts_state = self.symbolicSpace.reverse(symbols_state)

        if self.percepts is not None:
            merged = concepts_state.materialize()
            percepts = merged[:, :self.nPercepts, :]
            concepts = merged[:, self.nPercepts:self.nPercepts + self.nConcepts, :]
            input_from_percepts = self.perceptualSpace.reverse(self._wrap_reverse(self.perceptualSpace, percepts)).materialize()
            input_from_concepts = self.conceptualSpace.reverse(self._wrap_reverse(self.conceptualSpace, concepts)).materialize()
            input_latent = 0.5 * (input_from_percepts + input_from_concepts)
            input_state = self._wrap_reverse(self.inputSpace, input_latent)
        else:
            input_state = self.conceptualSpace.reverse(concepts_state)

        self.inputs = self.inputSpace.reverse(input_state)
        input_latent = input_state.materialize()
        input_data = self.inputs.materialize()

        return input_data, input_latent
TheMentalModel = MentalModel()

class ModelFactory:
    """Create, train, and evaluate models from an XML config file.

    Dispatches to the right model class based on <architecture> flags:
      - modelType=embedding   → BasicModel (embedding/language model path)
      - modelType=passthrough → BasicModel (passthrough path)
      - modelType=vq         → BasicModel (vector-quantized path)
      - Otherwise             → SimpleModel parameterized by:
            ergodic, certainty, quantized, normed, reverse, invert
    """

    @staticmethod
    def model_name(ergodic, certainty, quantized, normed=False, reverse=False, invert=False):
        """Generate a human-readable model name from its flags."""
        if not ergodic and not certainty and not quantized:
            return "SimpleModel"
        parts = []
        if ergodic:
            parts.append("Ergodic")
        if certainty:
            parts.append("Certainty")
        if quantized:
            parts.append("Quantized")
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
            model_family = str(arch.get("type", "basic") or "basic").strip().lower()

        # Attention is incompatible with flatten (attention expects 3D, flatten flattens to 2D)
        if gsp(cfg, "PerceptualSpace", "flatten") and gsp(cfg, "PerceptualSpace", "hasAttention"):
            errors.append(
                "PerceptualSpace hasAttention=True is incompatible with flatten=True. "
                "Set <hasAttention>false</hasAttention> in <PerceptualSpace>.")
        if gsp(cfg, "ConceptualSpace", "flatten") and gsp(cfg, "ConceptualSpace", "hasAttention"):
            errors.append(
                "ConceptualSpace hasAttention=True is incompatible with flatten=True. "
                "Set <hasAttention>false</hasAttention> in <ConceptualSpace>.")

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
        """Main entry point — create, train, and evaluate a model from XML config."""
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
                            _t("numEpochs", 3),
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
                    _t("numEpochs", 3),
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
            "Overrides BASICMODEL_COMPILE env var."
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
