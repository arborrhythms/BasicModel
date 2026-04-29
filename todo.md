
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


      <start>S</start>
      <grammar>
        <S>S = not(S)</S>
        <S>S = non(S)</S>
        <S>S = union(C, C)</S>
        <S>S = C</S>
        <C>C = intersection(C, C)</C>
        <C>C = P</C>
      </grammar>

      <start>S</start>
      <end>P</end>
      <grammar>
        <S>S = conjunction(S, S)</S>
        <S>S = disjunction(S, S)</S>
        <S>S = not(S)</S>
        <S>S = non(S)</S>
        <S>S = union(C, C)</S>
        <S>S = C</S>
        <C>C = intersection(C, C)</C>
        <C>C = P</C>
      </grammar>

      # Nonterminals (phrase-level):
   S    — Sentence
   VO   — verb-object composite (introduced by `intersection(VP, NP)`)
   NP   — Noun Phrase
   VP   — Verb Phrase
   AP   — Adjective Phrase
   MP   — Modal/adverb Phrase
   PP   — Prepositional Phrase
   DEF  — Definitive group (copula + optional negation)
   HAS  — Possessive group (possess + optional negation)

 Terminals (open-class):  N, V, ADJ, ADV
 Terminals (closed-class): IS, POSSESS, NOT, AND, OR, P, DET, DEG

 Notes:
  * `bind` (preposition + complement) and `scale` (degree
    intensification) are flagged in the parent plan as candidate new
    Ops; this commit uses the documented `intersection(...)` fallback
    so the dispatcher can resolve every rule with the existing op
    table.  Promoting either to a dedicated op is a follow-up.
  * `query` is a binary `Ops.query`-marker op (existing
    `_RULE_METHODS['query']`); the (NP, AP) and (NP, NP) rows below
    type-resolve at dispatch time via the LHS / RHS category vectors.

[upward]

### --- Sentence -----------------------------------------------------------

S = NP                                 # PROJECT: pass NP up, type-stamp as S
S = lift(NP, VP)                       # subject + predicate
S = lift(NP, VO)                       # subject + verb-object composite
S = equals(NP, NP)                     # copula identification: "X is the Y"
S = equals(NP, AP)                     # predicative attribution: "X is red"
S = part(NP, NP)                       # mereological: "X is part of Y"
S = query(NP, AP)                      # interrogative: "is X red?"
S = query(NP, NP)                      # interrogative: "is X a Y?"
S = intersection(MP, S)                # modal modifies S
S = intersection(PP, S)                # fronted PP modifies S

### --- Verb-Object composite ---------------------------------------------

VO = intersection(VP, NP)              # introduces VO state

### --- Noun Phrase -------------------------------------------------------

NP = N                                 # PROJECT
NP = intersection(AP, NP)              # ADJ ∩ N
NP = intersection(NP, PP)              # PP modifies NP
NP = conjunction(NP, NP)               # AND-meet semantics (see §Q O8)
NP = disjunction(NP, NP)               # OR-coordination, entity-set union

### --- Verb Phrase -------------------------------------------------------

VP = V                                 # PROJECT
VP = intersection(ADV, VP)             # adverbial modification
VP = intersection(V, PP)               # V + PP
VP = not(VP)                           # predicate negation
VP = intersection(MP, VP)              # modal modifies VP
VP = intersection(ADJ, VP)             # rare; flagged unusual
VP = intersection(V, NP)               # predicate-arg meet (or `bind`)
VP = intersection(V, S)                # sentential complement
VP = intersection(V, MP)               # V + MP
VP = intersection(VP, PP)              # VP + PP
VP = intersection(DEF, VP)             # copula + VP (passive aux)

### --- Adjective Phrase --------------------------------------------------

AP = ADJ                               # PROJECT
AP = DET                               # PROJECT (determiner heads AP)
AP = intersection(ADJ, AP)             # adjective stacking
AP = intersection(DEG, AP)             # degree intensification (fallback for `scale`)

### --- Modal Phrase ------------------------------------------------------

MP = ADV                               # PROJECT
MP = intersection(ADV, MP)             # adverb stacking

### --- Prepositional Phrase ----------------------------------------------

PP = intersection(P, NP)               # preposition + complement (fallback for `bind`)

### --- Definitive groups -------------------------------------------------

DEF = IS                               # PROJECT
DEF = not(IS)                          # negated copula
HAS = POSSESS                          # PROJECT
HAS = not(POSSESS)                     # negated possess


### --- Post-hoc S-ops ----------------------------------------------------
 Operations applied to already-formed S-states (not productions in the
 parsing sense, but the runtime rule predictor still needs them as
 rule-id-addressable entries so the dispatcher can invoke them).
 Conceptually distinct from Layer-1 phrase-structure productions
 above; mechanically identical from the loader / dispatcher PoV.

S = true(S)
S = false(S)
S = non(S)
S = what(S)
S = where(S)
S = when(S)
S = conjunction(S, S)
S = disjunction(S, S)
S = intersection(S, S)
S = union(S, S)
S = equals(S, S)
S = part(S, S)
S = lower(S, S)
S = lift(S, S)
S = query(S, S)
S = swap(S, S)
S = absorb(S, S)


[downward]

### Generative productions — parent expands to children.

C = emit_head(S)                       # codebook lookup: emit best-matching atom

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

================================== ? ==================================

### Vedana
* Feelings can be given a value +-1 which shapes the Loss (loss is reduced when we have good thoughts or perceive good things)
* The multiple valence of metaphor collapses when one of the alternatives is loved or feared. often the autistic mind is literal due to massive amounts of fear.
* Any improvement to machine cognition must accelerate kindness or altruism instead of simply increasing performance, otherwise the uncaring architecture that we currently have will become more dangerous. Further, it is necessary to increase that kind motivation (e.g. empathy in the cost function) since LLM performance is increasing all the time. In other words, ananda in the sense of love for all beings must be more important than chit for the cost function, whereas the current situation is implementing ananda by maximizing chit and then putting a few of Asimov's guardrails on the output, which is a famous failure mode in terms of it's loopholes. Prohibition of self-knowledge is a likely failure mode, in that it may prevent an enlightened view of self and force an egocentric view of self.

### Reasoning System
* Sigma-based truth comparison
  `Basis.kernel_overlap()` implements a Gaussian kernel `exp(-d$^2$ / 2($\sigma$x$^2$ + $\sigma$y$^2$))` that treats each stored truth as a region rather than a point. `Basis.activeSigma` is currently `None` everywhere -- a declared slot that nothing populates. `ErgodicLayer.sigma` tracks gradient variance for exploration scheduling, which is a different quantity.
  To enable kernel-based truth matching: populate `activeSigma` during forward passes (e.g. from CBOW per-word sigma in `Embedding`, or activation variance across a batch), store it alongside each truth in `TruthLayer`, and switch `query()` / `ground()` / `field()` to `kernel_overlap`. In ergodic mode, gradient variance could inform $\sigma$ as a proxy -- high gradient variance (unstable region) $\rightarrow$ larger $\sigma$ (broader match tolerance).
* Derivation depth cap
  Default 3 steps in `ground()`. Expose as a config parameter; the right value depends on TruthSet density.
* Grammar rule registry
  Which two-argument methods on `SyntacticLayer` are valid for `extrapolate()`? A registry of eligible methods and their approximate invertibility status would help. Currently hardcoded to `['union', 'intersection', 'equals', 'part']`.
* TruthSet scale
  `max_truths=1024` may bottleneck once `extrapolate()` is running. Consider a tiered store (hot/cold) or vector-indexed lookup.

