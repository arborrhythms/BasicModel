# Decoder, operators and 6.8 — §13 candidate, design clarification pending

2026-10-04. Working HEAD remains `802abb1acc95e1bddc8cb237b13230a336681c49`.
One working tree; nothing committed. **The §13 thirty-training measurement has
not started.** The §12 measurements stand without retry: class 0/10,
reconstruction 7/10, joint 0/10, MM_xor 10/10, sum control 10/10.
Their complete receipt is preserved in [README-before-review13.md](README-before-review13.md).
The accepted 6.9 baseline remains class MSE .1147481948, reconstruction 0/4,
zero ownership conflicts; prior comparison counts remain class 9/10 and
reconstruction 5/10.

## Design change during implementation

The direct request and review-start §13.4 require a lookup of learned
percept-concept locations, with signed evidence weights and detached occurrence
wholes. A concurrent amendment then specified `c = (n*M + F)/(n+1)`. That
amendment was implemented and tested before any campaign began. It is saved
in [the first plan diff](review13-plan-amendment.diff) with
[its source hash](review13-plan-amendment.json).

The plan changed again to:

> A concept vector is [perceptual coordinates | conceptual coordinates]; nothing
> is scaled or weighted.

Its new closing instruction specifies native perceptual coordinates and
`XOR_grammar` nDim 14. This conflicts with the direct request's explicit
percept-concept lookup. The [second plan diff](review13-plan-subspace-amendment.diff)
and [source hash](review13-plan-subspace-amendment.json) preserve that change.
Codex has not edited the plan. A clarification is pending on which design to
implement and measure once. No gate result has been obtained or selected.

## Current candidate

- `MereologicalCodes` owns a lookup reserve for native percept concepts. There
  is no independent word parameter. The dictionary `W` is a detached buffer,
  refreshed from the derivation at forward entry and on each lookup; live
  reads retain gradients to percept locations and signed feature evidence.
- A word's form is the evidence-weighted sum of its native percept-concept
  locations. Group literals share those locations. A recurrent percept gains
  its own location when canonicalization replaces the earlier part group.
- The currently implemented intermediate amendment combines that live form
  with detached occurrence context as `(n*M+F)/(n+1)`. `n` counts existing
  occurrence rows, and `M` divides the recency-weighted root sum by its total
  weight. Recency is `1/(1+age)` on the store's logical timestamps. The same
  pre-forward context is used by both walk paths. References witness
  membership independently of sentence truth; truth poles remain unchanged.
- Both native slot references and the existing inverted leaf-code postings
  supply occurrence membership. No context concepts or extra LTM rows are
  minted by the new code. Serial context does not substitute upstream WS
  property tags for occurrence wholes.
- Priming conducts through existing rows to their constituent words, using
  the existing spread coefficient. It now uses the grammar's actual concept
  owner in unshared serial models. The real XOR mechanism check finds words
  activated outside the current sentence, with no cross-batch transfer.
- The audit adds pairwise root cosines and centered singular values, alongside
  code geometry, percept-concept identities, occurrence counts and form
  weights. Read-back decisions distinguish a unique code winner, a winner
  selected by priming, and an unresolved exact tie. These observations use
  the same emitted leaves, without another decode or training step.

**Declared current capacity changes:** ConceptualSpace inventory 6 → 262 in
XOR_grammar and 8 → 264 in MM_grammar, adding the configured 256-percept
reserve. Word-slot counts and dimensions are unchanged in this candidate.
The latest plan's nDim 14 change is not implemented pending the design choice.

The measured gates, assertions, seed calls, optimizer choices, learning rates,
budgets and guards remain unchanged. The full old/new changed-test sources
and these checks are in [the contract report](review13-contracts-design-hold.json).
The two old test fixture ports keep their assertions: the direct-parameter VQ
contract now uses the field dictionary; the output ownership tests use words
and explicitly legal actions to separate credit ownership from a fresh
model's support choices. Production support-mask tests remain unchanged.

## Saved probes

Every probe includes its pre-run source archive, log and guarded process report.

| Probe | Result |
|---|---|
| [derived-before](probes/review13-derived-before/run.log) | 6 expected failures: no derived-code implementation |
| [derived-first](probes/review13-derived-first/run.log) | 5 pass, 1 failure: VQ construction still allocated EMA buffers; saved before repair |
| [derived-second](probes/review13-derived-second/run.log) | 13 pass, 7 existing slow skips |
| [integration](probes/review13-integration/run.log) | 115 pass, 7 fail, 1 skip; old parameter/data fixtures and missing recurrent-percept location |
| [integration-repaired](probes/review13-integration-repaired/run.log) | 37 pass, 4 ownership-fixture failures, 1 skip |
| [policy-port](probes/review13-policy-port/run.log) | All 4 unchanged assertion sets pass with explicit legal-action fixtures |
| [observer](probes/review13-observer/run.log) | One ordinary training batch plus evaluation validates margins, named derivations and zero ownership conflicts |
| [occurrence-before](probes/review13-occurrence-before/run.log) | Fails on the actual model's wrong priming owner; saved before repair |
| [occurrence-owner](probes/review13-occurrence-owner/run.log) | 12 pass, including activated competitors on actual existing XOR occurrences |
| [context-amendment-before](probes/review13-context-amendment-before/run.log) | Fails against the new occurrence/form blend; saved before implementation |
| [final-mechanisms](probes/review13-final-mechanisms/run.log) | **123 pass, 1 existing skip**, 117.5 s guarded wall time, peak 3,292,287,200 bytes |

These are mechanism checks, not extra gate trainings. No fresh BasicModel
NanoChat scoring, attribution training, full sweep or native run was added.
The existing frozen evaluator's small manifest check and acceptance probe
remain; the trained gate still waits for item 4's checkpoint.

## Remaining after clarification

Implement the settled design, complete its relevant mechanism checks, and
freeze source. Run sum ×10 and require 10/10 before XOR ×10 and MM_xor ×10.
Each XOR training supplies both unchanged bars, the bands, joint result,
final operators and priming-versus-code annotation; the tenth also supplies
ownership, walk/margin and start/end geometry audits. Update GradientFlow,
Architecture, the catalogue's distributional item and todo 6.8 from that
same result, and stop for Claude's review before any commit.
