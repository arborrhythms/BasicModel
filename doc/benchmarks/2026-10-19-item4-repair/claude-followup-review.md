# Re-review of BasicModel candidate `7285b38` (`f7e74af04..7285b38`, read only)

**Disposition: not approved for the landing sweep, and Item 4 is not complete.**

- B3 is still waiting on Alec's decision. I have not invented a decision for it or waived it.
- B3 is not the only remaining blocker. I found two more concrete source defects: one introduced by this round's wiring (N1) and one carried from the baseline (N2).
- All the other prior findings are resolved in source, apart from one documented limitation on H3.

## New actionable findings

**N1 — Medium: the S gain reads incoming-word lanes that P has already attenuated, while the S residual reads raw lanes.**
- **Where:**
  - `bin/Models.py:20729` swaps `payload` for the output of `_attention_sentence_payload`.
  - `bin/Models.py:19433-19434` multiplies its last element by `_percept_word_gain`.
  - That P-gained pair then becomes `incoming_lanes=payload[11]` in `symbol_field.read(...)`, right after `bin/Models.py:20738`. `bin/SymbolField.py:68-72` writes it into `SymbolRead.observed`.
  - Meanwhile, `SymbolField.arrive` scores the residual and loss on the raw placement lanes, around `bin/Models.py:20746` (`field.placements()['lanes']`).
- **Impact:**
  - When both P and S gains are nonzero, the compose consumer's `s` compares the P-dimmed arrival with ê. That inflates the apparent omission and under-attenuates. The arrival is no longer the raw `o` of spec §4.1, and the consumer and the residual see different observations.
  - At the shipped zero gains this does nothing, which is why the 0.5-gain compiled run and the tests don't expose it. None of them combine P>0 with a fixed S prior.
- **Repair:** Pass the raw poles as `incoming_lanes`, e.g. `read_poles(...)`/`_attention_poles` gathered at `index`, or the same raw placement lanes. Add a regression with P>0 and S>0 that asserts the S gain is the same with and without P.

**N2 — Medium: thought departures keep on R but are credited on R+A+work.**
- **Where:** `bin/WalkTrials.py:175` is `wins = departure >= 0 and second[0] < first[0]`, while `costs` at `:192` (used for credit at `:199`) is `sum(components) + work`.
- **Impact:** This contradicts Alec's decision that "the keep rule reads R + A, the same judge as the chooser's credit". A thought departure can be credited as better while losing the keep, or the reverse.
- **History:** this line is unchanged since the baseline (`9207f5e03`). I missed it in my first review; it is not a regression from this round.
- **Repair:** Keep on the same R+A judge (plus the episode's metered work, if Alec wants work in the thought judge). Update `keep_costs` and the audit, and add a regression where R and A conflict.

**Low (no action needed to sweep):** κ is now explicit at the row and document closings (`max(σ(lane logits))`). The S compose read (`bin/Expectation.py:315-316`) and the word read still use κ=1 implicitly. Either record that as intended or make it uniform.

## Prior findings
| Finding | Status |
|---|---|
| **B1** | **Resolved.** κ is per bank symbol (`bin/Layers.py:10453-10456`, `bin/Meaning.py:69-71`). Role-shaped presence is rejected. The test covers permuted three-role closings. |
| **B2 / M4** | **Resolved.** Support is the soft union `max(o, ê)` (`bin/Expectation.py:82-92`). Both writers go through one withheld-aware path (`bin/Layers.py:10515-10526`, `bin/Layers.py:9666-9684`), and the test checks the two agree. |
| **B3** | **Open: waiting on Alec.** Blocks the sweep. |
| **H1** | **Resolved.** Identification goes through native handles: S uses codebook addresses (`bin/SymbolField.py:29-30,42-43`); R/D use concept IDs (`bin/Models.py:19591-19603`, `bin/Layers.py:10666-10670`). Each domain is consistent within itself, and an unknown handle never falls back to its nearest neighbour (`bin/Expectation.py:267-287`). |
| **H2** | **Resolved.** The compose read uses the factual two-lane source plus the incoming lanes (but see N1 for where those lanes come from). |
| **H3** | **Resolved, with a documented limit.** Only realized-support keys are retained (`bin/Layers.py:10465-10495`). IDs and logits are still O(N) per estimate row; that needs measuring at Item 0 scale. |
| **M1 / M2 / M3 / M5** | **Resolved.** The scope knob, unused arguments, structured gate and stubs are gone. Migrated IDs and orders are −1 (`bin/Layers.py:9168`). The free view uses z=1 with no envelope "need", and the width migration is correct for pre-Item-4 layouts. The schema test is corrected and lxml is present. |
| **L1–L5** | **Resolved.** The inverse CDF is normalized and edge-safe. The omission threshold is configured and in the schema. Dead state is removed. Config and docs are cleaned (including `GradientFlow`). L5 is documented. |

## Missing §9 evidence (not source defects; none waived)
- **§9.1:** parity with expectation enabled is not measured. That is tied to B3.
- **Math-chain certificates:** the copular checkpoint is still missing, and the carried initial-binding distribution check still fails.
- **Learning gates not established:** per-level residual and collapse, all-level XOR and math chain, global derivation optimality (gate false), and a learned fully-expected cat.
- **Unchanged:** the original failed native run stays failed and is not regraded. Item 0 and 5.5 stay open. There has been no sweep, push or WikiOracle bump, so clean-tree completion is still pending.

**Next step:** fix N1 and N2, get Alec's B3 decision and implement it, then do one short check before the single full sweep.
