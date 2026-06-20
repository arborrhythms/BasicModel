# Doc-consistency pass + legacy flags — 2026-06-19

*Results of a general doc/code-consistency sweep (parallel agent audit over the
living docs, the specs, the doc-gen scripts, and the doc tree). The codebase docs
are in good shape post-rename — only two drift items in the living docs. Nothing
here is deleted: legacy content is **flagged** for Alec to act on (deletion is
hard to reverse; prefer archive when unsure).*

## Applied (safe fixes)

- **`doc/Architecture.md:165`** — STM depth `~7` → `~8` to match
  `ShortTermMemory.DEFAULT_CAPACITY = 8` (`Layers.py:11255`); `doc/Spaces.md`
  already says `~8`. (One-character cosmetic alignment.)

## Consistency flags (NOT auto-fixed — judgement needed)

- **`doc/build_architecture_decomposition.py`** (the doc-gen script) — stale prose
  in docstrings/templates: `"SymbolicSpace/SymbolicSubSpace sidecar"` at lines
  ~460/472/706/721, while lines ~484/721 already use the correct "SymbolicSpace:
  grammar and truth host." Because this *generates* docs, a blind string swap
  risks producing wrong output; fix deliberately against the current taxonomy
  (this was already noted in the 2026-06-19 rename handoff as a remaining
  doc-prose follow-up). Medium.
- **`doc/specs/2026-05-21-wordsubspace-stm-layer-refactor.md`** — carries
  pre-rename terms (`WordSubSpace`, `WordSpace = WordSubSpace` → now
  `SymbolicSubSpace`). It is a **dated/historical** spec, so the right action is
  *archive*, not rewrite (see below).

### Audit false positive (left as-is, deliberately)

- **`doc/Params.md:85`** `[Ethics.md](../../doc/Ethics.md)` was flagged as a
  broken link with a suggested change to `../doc/Ethics.md`. **The original is
  correct** — from `basicmodel/doc/`, `../../doc/Ethics.md` resolves to
  `WikiOracle/doc/Ethics.md`, which exists; the suggested "fix" would have broken
  it. No change made. (Recorded for transparency — verify suggested link fixes
  against the actual tree.)

## Legacy — delete candidates (high confidence)

| path | rationale |
|---|---|
| `doc/Spaces.md.orig` | Outdated backup (25 KB, Jun 8) of `Spaces.md`; unreferenced; the live file is larger/current. |
| `doc/BivectorOutputSpaceRebasePlan.md` | Explicitly superseded by `BivectorRetirementPlan.md` ("rewritten here as as-built record"). |
| `doc/old/invertible_tanh_product.md` | Standalone math note; no current references; the architecture uses LDU factorisation, not this. |
| `todo.md` (repo root) | Stale task list whose first entries date to April; superseded by the dated handoffs + `handoff-truth-ideas.md`. |

## Legacy — archive candidates (move to `doc/archive/`, keep as record)

| path | rationale |
|---|---|
| `doc/BivectorRetirementPlan.md` | Marked IMPLEMENTED (2026-06-18); a completion record. (Also: line 17 `../doc/Truth.md` should be `../../doc/Truth.md` — fix only if kept live.) |
| `doc/BrickHostSyncStatus.md` | Status doc from 2026-05; the residual is explicitly deferred to the per-word IR path. |
| `doc/handoff-truth-ideas.md` | EXECUTED 2026-06-18 (stages 1–5 landed behind `<truthIdeas>`); only stage 6 (episodic LTM) remains future. |
| `doc/old/*` (`CognitiveScienceSociety.md`, `Grammar.md`) | `doc/old/` *is* the archive; `Grammar.md` is pre-symbolic-refactor (superseded by `Language.md` + `plans/GrammarOpsPass.md`); `CognitiveScienceSociety.md` is a pre-impl essay (could move to `doc/research/`). |
| `doc/specs/2026-04-*.md`, `2026-05-*.md` (the dated design specs) | Superseded by the current authoritative specs (`orders.md`, `reading-attention.md`, `mereological-order-raising.md`, `symbol_firewall.md`). Several self-note their absorption into the 2026-05-26 two-loop plan. |

## `doc/plans/` archive policy (94 files)

The 94 `doc/plans/*.md` are **historical handoff records** — do not rewrite or
individually triage. Recommended: create `doc/plans/archive/` and move everything
older than the current working set into it, keeping live the recent handoffs
(`2026-06-15-handoff.md`, `2026-06-19-handoff.md`,
`2026-06-19-reading-attention-A.md`, this file, and the grammar-inverse handoff)
plus the named living plans (`MeronomyPlan.md`, `MeronomySpec.md`,
`GrammarOpsPass.md`, `NeuralToolUser.md`). This is a mechanical `git mv`, left for
Alec.

## Keep (explicitly not legacy)

- `doc/research/*` — external reference papers + notes (Gumbel-softmax, capsules,
  attention, Hopfield, PCFG, EBM, catastrophic forgetting, …). Reference, not
  legacy.
- The current authoritative specs and living docs
  (`Architecture.md`, `Language.md`, `Training.md`, `Spaces.md`, `STM.md`,
  `Reasoning.md`, `Mereology.md`, `Logic.md`, `BasicModel.md`, `Params.md`,
  `SymbolFirewall.md`, `doc/diagrams/gen_diagrams.py`) — no drift detected beyond
  the two items above.
- New this session: `doc/specs/training-stages.md`,
  `doc/plans/2026-06-19-grammar-inverses-handoff.md`.
