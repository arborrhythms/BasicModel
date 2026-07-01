
=============================== BINDING ALGORITHM (for review, 2026-07-01) ===============================

How PS percepts + WS wholes mint the relation-only CS symbols (the "concepts"). Lives in bin/Spaces.py
(ConceptualSpace); authoritative spec: doc/specs/mereological-order-raising.md. Called from
cs.forward(PS_sub, WS_sub) -- the ONE layer that sees BOTH towers. Host-side, at Reset. The WHOLE thing
is gated <mereologyRaise> (dark by default => byte-identical), and no-ops when .where / analysis spans
are unavailable (byte-mode analysis stages no spans), so it stays inert until a config stages words.

ENTRY -- _maybe_autobind_meta(pid_2d, vec_tensor, word_groups, tokens, word_texts, percept_where):
    pid_2d        [B,N] int    promoted PS percept ids (-1 = unpromoted / pad), from _embed_radix
    vec_tensor    [B,N,D]      matching event vectors (seed a fresh WS row on first sight of a pid)
    percept_where [B,N] int    each percept's .where = RAW integer offset, read from PS subspace.where
    ws._staged_analysis_spans  [B,K,2] = per-word (start,end) = the WS-side .where (the LOCATIONS)

  When <mereologyRaise> + word_groups are present, THREE bindings run side-by-side (additive; the
  taxonomy lattice keeps every edge, no single-parent conflict):
    1. _autobind_word_wholes   -- per-TEXT sigma bind: a word's spell-out pids all bind to ONE shared
       word-whole; the whole then holds >1 part so maybe_raise_order fires (legacy WS taxonomy).
    2. _autobind_cross_tower   -- the .where-GATED meronomy: bind each PS percept-TYPE to the generic
       WS word-TYPE (_WORD_CLASS) when the percept .where NESTS in a WS whole .where (span).
       "Is letter A part of *word*? -- decided by the .where." (record_cross_tower_meronomy)
    3. create_word_object_meta -- per word, mint A (word: parts ⊑ word-whole), B (object: ATOM ⊑
       UNIVERSE, later refined), C = word≡object META = reify(A,B). Keyed by surface text (one
       stable triple per word across presentations).

CS SYMBOL TABLE (relation-only; the migration target / new home for the legacy WS meta-taxonomy):
  A SYMBOL has NO vector, NO codebook row. It is TWO independent multi-valued sets -- Parts(S) and
  Wholes(S) -- tying PS part-codes <-> WS whole-codes (read from their codebooks).

  _populate_cs_symbols(pid_2d, percept_where, spans): for each analysis SPAN [s,e) (a LOCATION):
      loc_sym = new_concept()
      for each percept n with   s <= percept_where[n] < e:   add_part(loc_sym, pid)   # .where nests
      add_whole(loc_sym, _WORD_CLASS)
      _populate_concept_weights(loc_sym)          # <-- sparse-CS keeper (gated)
  i.e. "knit the parts covering a location to the wholes covering it."

SPARSE CS WEIGHTS (the CS matrix; gated dark unless _sparse_active() / symbolicOrder>=1, else byte-id):
  _populate_concept_weights(concept_id) decomposes the symbol into per-order sparse weight EDGES
  add_concept_weight(order, concept_local, source_global) over the [PS | WS | SS] source blocks:
  raw PS parts -> PS block, raw WS wholes -> WS block, sub-symbol refs -> the SS block of their
  (lower) order. MIN-SUPPORT >= 2 constituents (the ATOM/UNIVERSE pole-pair is skipped). Order from
  the ramsified _concept_source_order; per-order caps are dyadic (N/2, N/4, N/8, ... order_capacities).

LIFECYCLE (retire-on-trigger, drives BOTH towers; runs at the tail of _autobind_cross_tower):
  resolve_identities():  a symbol with EXACTLY 1 part + 1 whole is the identity-of-indiscernibles tie
      (sigma-up meets pi-down at the object) -- subsumed by the codebooks, so record (part,whole) on
      _concept_identity, CLEAR both sets (they vanish), drop it from the active set. EXCEPTION: an
      ATOM/UNIVERSE pole-pair is the maximally-general placeholder -> stays active for refinement.
  refine_over_collected(k=4): for each still-active OVER-collected symbol:
      too many PARTS  (>4) -> sigma-SYNTHESIS APPLIED: synthesize_higher_order(codes) groups the parts
                              under a higher-order symbol H (H's constituents ARE its definition ->
                              never re-refined).
      too many WHOLES (>4) -> pi-ANALYSE REQUEST only (deferred: which finer wholes is underspecified).
      then RETIRE the trigger symbol (transient -- the restructuring supersedes it).
      returns send-back requests = the convergence-loop signal.

INVARIANTS / NOTES:
  - Tightest relation = LARGEST part <-> SMALLEST whole (impenetrability); .where nesting is the test.
  - .where is CONTENT-addressed (QuadratureEncoding, period = 1/2 * InputSpace); PS .where = raw offset,
    WS .where = the analysis span [start,end).
  - Additive throughout: the legacy WS meta-vector taxonomy runs ALONGSIDE and migrates onto the CS
    symbol table in a later stage.

OPEN QUESTIONS TO REVIEW:
  - Membership test is  s <= wp < e  -- ANY covering span. Should it instead pick the TIGHTEST whole
    (smallest covering span) to honor the largest-part<->smallest-whole invariant stated above?
  - MIN-SUPPORT >= 2 drops singleton concepts -- right for words, but does it starve order-0 atoms?
  - k_many = 4 over-collection threshold: tuned, or placeholder?
  - sigma-synthesis is APPLIED but pi-analyse is only REQUESTED -- the finer-whole split is still open.

================================ 2026-07-01 (MM_20M_xor / XOR pipeline) ================================

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

* **Reconstruction @ symbolicOrder=0:** XOR output is correct but the input reconstruction renders blank/junk
  (`' '` / `'h h'`), not words. Only the LEXICON reverse-decode reconstructs (XOR_exact renders `'hello world'`);
  the bpe (MM_20M_legacy) and radix/meronomy (MM_20M_xor) decode paths both come back empty — the reverse-decode
  for those paths isn't wired to render. Fastest way in: diff XOR_exact's working lexicon decode vs the radix
  decode side-by-side. (600 epochs @ reconstructionScale=0.5 did NOT converge -> structural, not under-training.)

* **SigmaLayer sparse=False option (NOT yet implemented, verified 2026-07-01):** the PS/WS -> CS and SS -> CS
  symbolic weights currently map through raw `torch.sparse` COO matrices (`cs_sparse_encode`). Replace with a
  SigmaLayer carrying a COO-sparse inner layer: add a `sparse=False` `__init__` option (default dense =
  byte-identical). Because CS/SS are ramsified, per-order sizes follow the dyadic capacities N/2, N/4, N/8, ...
  (`order_capacities`).

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

