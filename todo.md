
Some of the Grammatical operations have been recently integrated with SigmaLayer and PiLayer of the NN. This trend should be continued; for example, the Sigma and Pi layers positive and monotonic, so we need a not() operator that performs negation on the symbol, and emits a word in the grammatical derivation.

This can be extended: the necessary precondition of xAI (HI) is that every operation has a (grammatical) interpretation. Several of these implmentations will not happen at the symbolic layer:

true/false/non: Sym
not: Sym (or learnable NOT layer just after Sigma Layer?)
conjunction/disjunction: Sym
interspection/union: SigmaPi
equals: Def (definition: introduction of a new Sym into WordSpace)
part: Def (definition: asserts a part/whole relation over WordSpace)
slot (where, when, why): Sym
query: mereological
swap: Sym
lift/lower: SigmaPi
chunk: Per (not currently in the grammar; similar to Sigma layer)

LEGEND:
Sym -- implemented at the symbolic layer
SigmaPi -- implemented by SigmaPi (symbol-conceptual round trip)
Per -- implemented at the perceptual layer ()
Def -- equals and part express relations between existing symbols, and will define an explicit mereology over the symbolic codebook  

================================== Butterfly Performance ==================================

Five more honest sources, in order of how much I'd attribute to each:

1. Param count vs the GPU it's running on
This is the one I should have led with. 5M params × bf16 = 10MB. That fits inside the L2 cache on essentially any modern GPU. There is not enough memory traffic to give the SMs a reason to wake up.

NanoChat's smallest model is 561M params (~1.1GB at bf16) — it actually overflows L2 and forces streaming from HBM, which is what GPUs are optimized to do well. A 5M model running on Blackwell is basically running on whatever fraction of one SM the dispatcher feels like assigning that microsecond. Most of the chip is idle by physics, not by inefficiency.

This is the deepest reason and it's not really fixable without making the model bigger. NanoChat's "small" is bigger than our "big".

2. Number of distinct passes per batch
NanoChat per batch:

1 forward, 1 backward, 1 optimizer step. That's it.
MM_5M per batch:

Surface forward + reverse (the invertible pass)
Deep forward + reverse (composed prediction at conceptualOrder=3)
TruthLayer LIFT (S, V, O → ternary)
STM update (concept_states snapshot)
LTM bridge via Pi+L1
Span table construction
(Until now) BPE growth step
Each is its own graph traversal with its own dispatch chain. Even with everything else equal, you're paying ~5–10× the per-batch dispatch cost.

3. Tokens per step (effective batch)
NanoChat d12: roughly 524,288 tokens per step (device-batch × seq-len × grad-accum × GPUs).

MM_5M: 32 sentences × ~30 BPE tokens ≈ 1,000 tokens per step.

That's ~500× more arithmetic to amortize the same per-step overhead. The GPU loves big contiguous batches because they convert dispatch overhead into a vanishing fraction of step time. Small batches expose every microsecond of Python/CUDA-launch overhead.

4. dtype + Tensor Cores
NanoChat is bf16 throughout, fp8 where H100 supports it. Tensor cores give ~10× the throughput of FP32 dense matmul.

MM_5M: I'd want to confirm, but I believe we're FP32 across the model. On Blackwell that means the tensor cores are mostly cold. Switching to bf16 (or at least running matmuls in bf16 with FP32 accumulation) is one of the higher-ROI changes we haven't made.

5. Kernel granularity
A NanoChat transformer layer = roughly 5 kernels (Q/K/V projection, FlashAttention, output projection, two FFN GEMMs). FlashAttention fuses ~7 operations into one launch.

MM_5M layer = many more small kernels. log2(1024)=10 butterfly stages × conceptualOrder=3 × forward/reverse = ~60 stages, each its own kernel in eager mode. Plus span-table updates, TruthLayer ops, STM/LTM bookkeeping. Easily hundreds of kernel launches per forward pass, none fused.

This is what MODEL_COMPILE=auto (with frozen BPE so shapes are stable) is supposed to claw back: Inductor can fuse adjacent butterfly stages into single kernels if the shapes don't change underneath it. But until BPE is frozen, it can't.

What to take from this
The unfixable items: #1 (param count vs GPU) and parts of #2 (the cognitive architecture has more passes by design).

The fixable items, in rough ROI order:

bf16 / mixed precision (~5–10×, easy)
Frozen BPE + torch.compile + kernel fusion (~3–5×, just landed the prerequisite)
Larger effective batch via grad accumulation (~2–3×, easy if memory allows)
Skipping the nObj=1024 padding cost on short sentences (~variable, architectural)
Stack those together and we're plausibly at 30–150× our current throughput — not enough to reach NanoChat-on-H100 numbers, but enough to make the 6-day plan see ~30× more tokens.

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

