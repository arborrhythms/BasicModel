# Projects

## docs/superpowers/specs/2026-04-22-model-serial-mode-design.md.

## Reintroduce SentencePredictor as a Layer

The legacy discourse priming bias (`discourse.prime(predicted_snapshot, confidence, sentence_priming_scale)`)
that used to be injected into `concept_input` at stage t=0 in `_run_conceptual_order`
was removed when the Sequential pipeline landed. Reintroduce it as a Layer — not
a Pipeline glue — on ConceptualSpace (or SymbolicSpace). The Layer should read
`wordSpace.discourse.predict()` and apply the bias to its input tensor. Doing
this as a Layer keeps it co-located with the Space's own math and avoids
polluting the cross-stage Pipeline glue inventory.

## Delete legacy code

The legacy (`_forward_legacy`, `_reverse_legacy`, `_forward_dispatch`,
`_run_conceptual_order`, `_run_forward_pipeline`, `_start_ar_forward`, and the
body of the pre-Phase-2 `forward()`) is no longer called on any live code path.
Keep for one or two more iterations as reference, then delete wholesale. Diff
against git history for what was there.

Inside Spaces.py the pattern `def forward(self, subspace): ... return
self._forward_legacy(subspace, ...)` can be inlined once `_forward_legacy` is
deleted.

---

diff -U3 -- basicmodel/bin/serve.py basicmodel/bin/serve.py
--- a/basicmodel/bin/serve.py
+++ b/basicmodel/bin/serve.py
@@ -139,6 +139,16 @@
 
 _MAX_MSG_LEN = 10_000  # max chars per user message
 _MAX_MESSAGES = 50     # max messages in a request
+
+# HTTP generation budget. ARLM reruns the model per generated token,
+# so a request that uses the XML-wide maxResponseLength (typically
+# thousands of characters) can sit in inference long enough for the
+# urllib client to time out.  Cap server-side: default 64 generated
+# characters, honor explicit max_tokens requests but clamp to 128.
+_DEFAULT_GEN_BUDGET = 64
+_MAX_GEN_BUDGET = 128
 
 
 @app.route("/chat/completions", methods=["POST"])
@@ -176,6 +186,15 @@
     if not user_msg:
         return jsonify({"error": "No user message found"}), 400
 
+    # Resolve generation budget: honor explicit max_tokens but clamp,
+    # fall back to the default when unset / invalid.
+    raw_budget = body.get("max_tokens")
+    try:
+        gen_budget = int(raw_budget) if raw_budget is not None else _DEFAULT_GEN_BUDGET
+    except (TypeError, ValueError):
+        gen_budget = _DEFAULT_GEN_BUDGET
+    gen_budget = max(1, min(gen_budget, _MAX_GEN_BUDGET))
+
     # Prompt injection guard
     injection = guard_input(user_msg)
     if injection:
@@ -210,8 +229,11 @@
         if thought_free:
             TheGrammar.thought_free = True
 
-        # Autoregressive inference: extend input text word by word
-        predicted_words = _model.infer(user_msg, mode='ARLM')
+        # Autoregressive inference: extend input text word by word,
+        # bounded by the server's generation budget so slow ARLM runs
+        # cannot sit longer than the HTTP client's read timeout.
+        predicted_words = _model.infer(
+            user_msg, mode='ARLM', max_length=gen_budget)
         response_text = " ".join(predicted_words)
 
         response = {
diff -U3 -- basicmodel/bin/data.py basicmodel/bin/data.py
--- a/basicmodel/bin/data.py
+++ b/basicmodel/bin/data.py
@@ -707,6 +707,20 @@
             num_workers = 0
             pin_memory = False
 
+        # Windows multiprocessing can fail or appear to hang when PyTorch
+        # constructs worker queues from a subprocess-driven training launcher
+        # (observed as WinError 5 from multiprocessing.Pipe). Keep text
+        # streaming in-process there; MM_* configs can still set numWorkers>0
+        # for platforms where it is reliable.
+        if sys.platform == "win32" and num_workers > 0:
+            print(
+                "[data_loader] Windows multiprocessing DataLoader workers "
+                f"are unreliable here; forcing num_workers=0 "
+                f"(requested {num_workers})."
+            )
+            num_workers = 0
+            pin_memory = False
+
         kwargs = {"batch_size": None, "num_workers": num_workers,
                   "pin_memory": pin_memory}
         if num_workers > 0 and prefetch_factor is not None:
diff -U3 -- basicmodel/bin/train.py basicmodel/bin/train.py
--- a/basicmodel/bin/train.py
+++ b/basicmodel/bin/train.py
@@ -34,6 +34,10 @@
                    help="Override numShards from XML config")
     p.add_argument("--num-epochs", type=int, default=None,
                    help="Override numEpochs from XML config")
+    p.add_argument("--max-tokens", type=int, default=None,
+                   help="Cap AR training/eval positions per document for "
+                        "quick smoke runs. Full training uses the XML "
+                        "InputSpace.nOutput cap when omitted.")
     p.add_argument("--random-shards", action="store_true",
                    help="Pick random shard indices for variety across runs")
     p.add_argument("--force-embeddings", action="store_true",
@@ -176,6 +180,8 @@
         model_env["BASIC_NUM_SHARDS"] = str(args.num_shards)
     if args.num_epochs is not None:
         model_env["BASIC_NUM_EPOCHS"] = str(args.num_epochs)
+    if args.max_tokens is not None:
+        model_env["BASIC_MAX_TOKENS"] = str(args.max_tokens)
 
     entry = os.path.join(proj, "bin", "Models.py")
 
@@ -236,6 +242,8 @@
         remote_args += ["--num-shards", str(args.num_shards)]
     if args.num_epochs is not None:
         remote_args += ["--num-epochs", str(args.num_epochs)]
+    if args.max_tokens is not None:
+        remote_args += ["--max-tokens", str(args.max_tokens)]
     if args.random_shards:
         remote_args += ["--random-shards"]
     if args.profile:
@@ -306,5 +314,7 @@
             train_local(args)
     finally:
         if _log_file:
+            sys.stdout = sys.__stdout__
+            sys.stderr = sys.__stderr__
             _log_file.close()
 
 
diff -U3 -- basicmodel/bin/Models.py basicmodel/bin/Models.py
--- a/basicmodel/bin/Models.py
+++ b/basicmodel/bin/Models.py
@@ -228,6 +228,8 @@
             wpath = TheXMLConfig.get("architecture.weightsPath")
             wpath = self._resolve_artifact_path(wpath)
             self.load_weights(wpath)
         self.max_response_length = arch["maxResponseLength"]
+        self.max_sequence_tokens = int(
+            os.environ.get("BASIC_MAX_TOKENS", 0) or 0)
         return cfg
 
     def create(self, **kwargs):
@@ -3207,8 +3209,22 @@
             else:
                 N = embedded.shape[1]
                 valid_pos = None
+            token_cap = int(getattr(self, "max_sequence_tokens", 0) or 0)
+            if token_cap > 0 and N > token_cap:
+                if valid_pos is not None:
+                    valid_pos = valid_pos[:, :token_cap]
+                N = token_cap
             self._ar_valid_pos = valid_pos
 
+            if N >= 128:
+                TheMessage(
+                    f"[{self.name}] AR forward: {N} token positions "
+                    f"(batch={B})")
+
             predictions = []
             percepts = None
             sym_vectors = None
             for pos in range(N):
+                if N >= 128 and pos > 0 and pos % 128 == 0:
+                    TheMessage(f"[{self.name}] AR forward: {pos}/{N}")
                 # Set the InputSpace subspace event to the current buffer
                 # (pre-slide). The buffer carries cross-sentence context.
                 buffer = self.inputSpace._ar_buffer


================================== April 24 ==================================

# Ask Solid community for a simple file-getting interface
* if the user provides the server with an API key, we can query an LLM
* if the user provides the server with a SOLID key, we can retrieve a file
* if the user provides the server with a DSA key, we can decrypt a file
* is there a POD service that does simple free hosting?

# Ask EFF for a security review
* propose "Owning our Data"
* this entails taht marketers and AI are not allowed to lock us down karmically
with specifically-characterized information (concrete details)
* maybe it can learn from that data by removing or randomizing that information

# Send email proposal to Apertus 
* First develop boilerplate on WikiOracle that references wikipedia, eff, and solid

================================== ? ==================================

## Vedana
* Feelings can be given a value +-1 which shapes the Loss (loss is reduced when we have good thoughts or perceive good things)
* The multiple valence of metaphor collapses when one of the alternatives is loved or feared. often the autistic mind is literal due to massive amounts of fear.
* Any improvement to machine cognition must accelerate kindness or altruism instead of simply increasing performance, otherwise the uncaring architecture that we currently have will become more dangerous. Further, it is necessary to increase that kind motivation (e.g. empathy in the cost function) since LLM performance is increasing all the time. In other words, ananda in the sense of love for all beings must be more important than chit for the cost function, whereas the current situation is implementing ananda by maximizing chit and then putting a few of Asimov's guardrails on the output, which is a famous failure mode in terms of it's loopholes. Prohibition of self-knowledge is a likely failure mode, in that it may prevent an enlightened view of self and force an egocentric view of self.

## Reasoning System
* Sigma-based truth comparison
  `Basis.kernel_overlap()` implements a Gaussian kernel `exp(-d$^2$ / 2($\sigma$x$^2$ + $\sigma$y$^2$))` that treats each stored truth as a region rather than a point. `Basis.activeSigma` is currently `None` everywhere -- a declared slot that nothing populates. `ErgodicLayer.sigma` tracks gradient variance for exploration scheduling, which is a different quantity.
  To enable kernel-based truth matching: populate `activeSigma` during forward passes (e.g. from CBOW per-word sigma in `Embedding`, or activation variance across a batch), store it alongside each truth in `TruthLayer`, and switch `query()` / `ground()` / `field()` to `kernel_overlap`. In ergodic mode, gradient variance could inform $\sigma$ as a proxy -- high gradient variance (unstable region) $\rightarrow$ larger $\sigma$ (broader match tolerance).
* Derivation depth cap
  Default 3 steps in `ground()`. Expose as a config parameter; the right value depends on TruthSet density.
* Grammar rule registry
  Which two-argument methods on `SyntacticLayer` are valid for `extrapolate()`? A registry of eligible methods and their approximate invertibility status would help. Currently hardcoded to `['union', 'intersection', 'equals', 'part']`.
* TruthSet scale
  `max_truths=1024` may bottleneck once `extrapolate()` is running. Consider a tiered store (hot/cold) or vector-indexed lookup.

## Future Work: Parsed Training Dataset
It is desirable to create a small training and testing dataset for the network consisting of statements that are already parsed. This would allow direct comparison between the grammatical derivation produced by traditional English parsers and the deep structure produced by BasicModel's ConceptualSyntacticLayer.

* See `bin/parse.py` as a starting point for producing grammatical derivations via NLTK POS tagging and CFG parsing.
* Such a dataset would also enable evaluation of head identification accuracy and composition quality.


## consider adding Percepts to the front of the Symbols array