
================================ 2026-07-01 (MM_20M_xor / XOR pipeline ================================

* **Reverted to clean slate (2026-07-01):** the in-progress `.where`-minting + reconstruction-loss
  attempts for the three items below were reverted to HEAD (they broke 6 mereology tests and left
  dark debug scaffolding). KEPT as deliberate keepers: (a) the sparse CS matrix stays LIVE -- the
  cross-tower autobind still calls `_populate_concept_weights(loc_sym)`, gated dark unless
  `_sparse_active()` (symbolicOrder>=1), so the CS MLP still operates over percepts+symbols;
  (b) the MPS COO fix in `_build_csw` (MPS has no `_to_sparse_csr`; `torch.sparse.mm` takes COO);
  (c) `ConceptualCombine(nonlinear=...)` tanh/atanh, default-off / byte-identical; (d) the
  MM_20M.xml -> MM_20M_legacy.xml rename. The 7 originally-failing tests are green again; the three
  items below are the un-started work, not half-done state.

* **Output @ symbolicOrder=1 (sparse forward):** the ramsified sparse forward (`cs_forward_content`) produces a
  degenerate concept code (the `'h h'` collapse), so XOR output is wrong at symbolicOrder=1. symbolicOrder=0 (the
  dense butterfly path) solves XOR cleanly. The word concepts are minted but the joint/sentence concept binding
  BOTH words isn't formed. Likely needs SBOW (to avoid the mean-collapse) + butterfly=True in the joint mixing.
  See doc/Architecture.md "Symbolic weights".
  STATUS 2026-07-02 (repro + fixes; doc/plans/2026-07-02-sparse-layer-conceptual-embedding.md
  Continuation): (a) FIXED a backward crash on the live path (SS identity rows must be CLONED --
  each stage's no_grad codebook sync bumped a saved view's version); (b) ROOT-CAUSED + FIXED presence
  saturation: raw dot-SUM presences were the all-ones CONSTANT over all 65536 PS codes (std-over-batch
  exactly 0 -- input-blind); per Alec the readout is the NORMALIZED SUM (slot-mean projection onto
  the unit code direction / sqrt(D)) -- event MAGNITUDE preserved (objects in the unit hypercube
  differentiate by magnitude), NOT a cosine; (c) the JOINT/sentence concept now MINTS
  (`create_joint_concept`: first-order, parts = the row's word A-symbols, whole = EVERYTHING ->
  the bias edge; keyed per sentence type; Hebbian on re-occurrence) -- the SYMBOLIC joint mixing;
  (d) the SUBSYMBOLIC butterfly mixing is PRESERVED as an option: `<sparseReplace>false</sparseReplace>`
  keeps the truly-invertible butterfly content advance while the sparse activations still feed the
  symbolic loop (SS leg / concept table) -- the loops run side by side. OPEN: REPLACE mode on this
  config is capacity/slot-coupled -- the sparse content substitutes only when CS nVectors == the
  STM slot width, so an inventory bigger than the slab needs the SELECTION step (Architecture.md:
  "attention picks the salient subset") before replace-mode XOR is testable end-to-end. SBOW
  situates the LIVE codes when conceptualSimilarityScale > 0 -- this config leaves it unset.
  STATUS 2026-07-02 EVENING (two-phase rework, doc/plans/2026-07-02-two-phase-loops-sparse-relation.md
  P1-P5 executed): the forward is now TWO PHASES -- a purely subsymbolic pump (2-stream, demux
  feedback via `combine.views`: the per-tower windows of the MIXED carrier, NOT `combine.reverse`,
  whose exact inversion returns each tower's own input and froze the pump -- root-caused) and ONE
  late cutover (snap -> ramsified composition -> SS leg once -> SBOW on the settled slab).
  `sparseReplace` RETIRED (non-replacement structural). The 'h h' COLLAPSE IS GONE: sO=1 runs
  clean end-to-end; XOR at sO=1 now plateaus UNDECIDED (~0.5 all rows, output loss flat at 0.175
  over 300 epochs) while the SAME config at sO=0 solves XOR to 0.000. ROOT-CAUSED (probe scripts,
  gradients + batch-variance): (a) early-stage combine grads are ~100x weaker at sO=1 (the
  demux-fed recursion re-binds already-tanh'd carrier windows -- variance compression), so the
  escape drift that frees sO=0 after ~50 epochs is ~100x slower; (b) the snap's slot-mean readout
  is input-blind at init on the tiny 8-row CS (activation batch-std ~5e-6) until the EMA identity
  trace differentiates the rows. Both belong to the deferred training-dynamics pass -- and both
  are superseded by the ITERATED SINGLE-LAYER redesign (Alec, same evening:
  doc/plans/2026-07-02-iterated-symbolic-loop.md, DRAFT -- untyped square SparseLayer, wave
  activation, no self-edges, Kripke groundedness as the cycle diagnostic).

* **Reconstruction @ symbolicOrder=0:** XOR output is correct but the input reconstruction renders blank/junk
  (`' '` / `'h h'`), not words. Only the LEXICON reverse-decode reconstructs (XOR_exact renders `'hello world'`);
  the bpe (MM_20M_legacy) and radix/meronomy (MM_20M_xor) decode paths both come back empty — the reverse-decode
  for those paths isn't wired to render. Fastest way in: diff XOR_exact's working lexicon decode vs the radix
  decode side-by-side. (600 epochs @ reconstructionScale=0.5 did NOT converge -> structural, not under-training.)
  STATUS 2026-07-02 (two fixes landed, remainder root-caused; test_radix_recon_render.py): (a) the
  RENDER is now WIRED -- PartSpace.reverse's numeric branch stages a radix thunk (RadixLayer.reverse
  structural decode -> inverse_table bytes; shared token-stamper hoisted from Embedding); it used to
  raise "called before reverse()" (staging was Embedding-branch-only). (b) the recon loss was
  GRAD-DEAD (a constant): seeded from the DETACHED STM snapshot through an input-differentiable but
  PARAMETER-detached reverse chain -- `_reconstruction_seed()` now prefers the LIVE terminal carrier
  (`_combine_last_cs_sub`, per that stash's documented intent), gradient verified end-to-end; XOR
  output now trains to ~0.000 loss alongside. (c) STILL blank -- the remaining structural gaps are
  the reverted 2026-07-01 ground: recon loss magnitude ~5e-4 (both sides small-normed; no shaping
  pressure), recovered `.where` band ~zero -> decodes to ~maxVal aliasing junk (61k-65k ~ nObjects)
  -> every token out-of-range, content decodes to sentence-chunks not word chunks. Needs the
  reverse-fidelity design pass (where-band supervision + content-recovery normalization), a design
  decision, not a patch.

* **DONE 2026-07-02 (as SparseLayer, superseding the SigmaLayer-option idea):** the PS/WS -> CS and
  SS -> CS symbolic weights now map through a dedicated `SparseLayer` (bin/Layers.py) -- forward
  tanh(W x), reverse tanh(W^T y) (transpose autoencoder; no LDU), export-safe scatter-add kernel,
  edge add/remove for pruning rounds. Two DISTINCT per-order families (percept [PS|WS], symbol
  [SS_<k]) summed pre-tanh, sized by the dyadic `order_capacities`. A SigmaLayer option was
  deliberately rejected: SigmaLayer's atanh entry expects logit-domain codes, these maps consume
  presences. Also landed: grad-preserving 0-D symbol leg, closest-links pruning rounds, .when tie
  check, assert_concept_relation (B-pole refinement), Hebbian C-tie strengthening. See
  doc/Architecture.md sec A + doc/plans/2026-07-02-sparse-layer-conceptual-embedding.md.

* The "Codebook.property_basis" is a hack that needs to be removed. Please summarize the WholeSpace property mechanism. You said properties "are" WholeSpace.what. But that codebook currently holds the symbol/truth prototypes wired into the codebook-snap machinery; making properties the live .what semantics would rip that out and move the basin. So I built the property capability as opt-in/additive (Codebook.property_basis) alongside the existing symbol codebook, not as a wholesale replacement. If you intended the live cutover, that's a separate deliberate step.

* Pelase make sure that the folllowing is wired into training: the model/train-level two-pass driver that runs the forward twice and applies two_pass_loss; the optional MLPTransformChooser; the soft-codebook option.

* The TruthLayer had been a container for user truth, and to some degree it has been replaced by the LTM.
For simplicity, it would be better to add user Truth to the LTM, and have TruthLayer be an interface to that tensor.

* Make sure the chart parser predicts masked words, so that we can train it predictions. To do so, it will be useful to consider the cateogry of the word and attention over all concepts.

* Learning a new concept is a parallel symbolic operation that may not be active at the same time as grammatical processing (serial mode). 
Ensure that we spend enough tim learning the definitons of words (I think we do Subsymbolic before Symbolic order, can we do symbolic order in addition to serial processing?).

* The "persisted episodic store of perceptual events" is an LTM store that corresponds to the words instead of the ideas composed of objects. Right now we treat words symbolically; that is their primary use case. However, the parser needs to extract word context to become a fluent symbol manipulator.
So the encoding of the words themselves, qua words, is not being done except in the mereology of the partSpace and wholeSpace.
So we need to store information about the context of words; that is where all of the power of word2vec comes from, which is a significant source of knowing that should not be ignored.
How might we well-capture that in this architecture? There is no context learning happening above symbolic encoding; might we regard the conceptual taxonomy, currently storing pairwise associations, as a bit of a markoff chain? Is there a way to do that which maintains pairwise relations, but which allows a nonlinearity at each association to add information to higher network levels?

The gap, stated precisely. A word here currently has three encodings, all hard/structural: its symbolic identity (codebook index), its orthographic mereology (PartSpace bytes/radix → WholeSpace word-as-whole), and its grammatical/taxonomic relations (the pairwise edges). What's missing is the fourth, soft/distributional one — meaning-from-company — which is the whole of word2vec's contribution. And you're right that it's a major source of knowing: the taxonomy tells you men ⊑ mortal; the distributional geometry tells you king − man + woman ≈ queen — graded analogical structure that no pairwise edge-set encodes.

* Formal Concept Analysis, DisCoCat

* A code's basis is expressed at one of several orders, and therefore the codebooks must be ramsified. Please analyse this. 

* The process of naming/identifying may require traversing multiple orders before a symbol can be matched with its meaning.

* Full LLM expressivity requires multiple layers of DISTINCT Sigma/Pi matrices. Lets enable that. 

* Ensure that we are store explicit Taxonymic knowledge when processing a definiton.
This should only happen when some measure of sentence confidence is high.

* Any improvement to machine cognition must accelerate kindness or altruism instead of simply increasing performance, otherwise the uncaring architecture that we currently have will become more dangerous. Further, it is necessary to increase that kind motivation (e.g. empathy in the cost function) since LLM performance is increasing all the time. In other words, ananda in the sense of love for all beings must be more important than chit for the cost function, whereas the current situation is implementing ananda by maximizing chit and then putting a few of Asimov's guardrails on the output, which is a famous failure mode in terms of it's loopholes. Prohibition of self-knowledge is a likely failure mode, in that it may prevent an enlightened view of self and force an egocentric view of self.

================================== April 24 ==================================

### Ask Solid community for a simple file-getting interface
* if the user provides the server with an API key, we can query an LLM
* if the user provides the server with a SOLID key, we can retrieve a file
* if the user provides the server with a DSA key, we can decrypt a file
* is there a POD service that does simple free hosting?

### Ask EFF for a security review
* propose "Owning our Data"
* this entails taht marketers and AI are not allowed to lock us down karmically
with specifically-characterized information (concrete details)
* maybe it can learn from that data by removing or randomizing that information

### Send email proposal to Apertus 
* First develop boilerplate on WikiOracle that references wikipedia, eff, and solid

