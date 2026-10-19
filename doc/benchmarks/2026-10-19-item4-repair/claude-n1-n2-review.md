# Short check of `282fcf918` (`7285b38..282fcf918`, read only)

**N1 and N2 are repaired in source, with native regressions that test what they claim.** No new code defect. Remaining before the sweep: B3 (still waiting on Alec), one decision to record and one stale-doc fix (both under N2).

## N1 — the S gain now reads raw arrival lanes: confirmed
- **Source:**
  - `bin/Models.py:20728-20735` reads the raw poles through the declared consumer before `_attention_sentence_payload` applies P. `'SymbolField.read'` is added to the consumer list at `bin/ModelAttention.py:10`.
  - `bin/Models.py:20748` passes `incoming_lanes=raw_lanes` to `SymbolField.read`.
  - The closing path keeps its own `payload[11]`, which P never touches. When `reading is None`, the payload isn't P-scaled either, so the fallback is equally raw.
  - Content still receives P through `payload[11]` → `leaf_evidence`.
  - The S read and `SymbolField.arrive` (raw placement lanes) now see the same observation.
- **Regression** (`test/test_item4_review.py:145-199`):
  - It runs the native sentence pipeline at gP=0 and gP=0.75, with gS=0.5 and a fixed perfect S prior.
  - It asserts:
    - the S read receives the raw lanes;
    - the content payload equals raw × (1−gP);
    - the S gain is 0.5 at both settings;
    - at least one row was actually exercised (`assert records`).
  - Because the gain is computed by handle (`identities=ids`), it can't silently fall back to matching on dimmed values.

## N2 — one R+A judge for thought departures: confirmed in code; one decision and one doc fix outstanding
- **Source:**
  - The episode scorer's components are `(reconstruction, answer)` (`bin/WalkTrials.py:149-155`), so `sum(first)` is R+A.
  - Keep (`bin/WalkTrials.py:175-176`), credit (`:197`) and the audit (`:190-200`) now all use that same `costs`.
  - A tie keeps greedy. Work is reported as `work_spent` and stays bounded by the meter.
- **Regressions:**
  - `test/test_walk_trials.py:85-106` covers the conflict both ways (R favours one path, A the other) and asserts that keep, audit and credit agree, plus the sign of the surrogate.
  - `test/test_item6_2_thinking.py:273-276` now keeps the answered chain.
  - `test/test_unified_thought_controller.py:132-139` sets `WHAT_STEP_COST=1000` and checks that work cannot change the judge.
  - The `test_walk_trials` pending-list port is correct.
- **Decision to record:** removing the per-step work price from the thought episode's judge changes behaviour the thinking spec describes. `doc/specs/2026-10-07-thinking.md:997-999` says the thought walk "already pays `WHAT_STEP_COST` … between *its* two trials". That line now describes retired behaviour. My last review left work in the thought judge as Alec's choice. Applying R+A literally is a defensible reading of his decision, but it should be his nod. If he agrees, annotate that spec line as superseded.
- **Stale live contracts (low, docs only):**
  - `doc/GradientFlow.md:28-31` still says keep on strictly lower R, with advantage from `C = R + E + A`.
  - `doc/Architecture.md:2629-2632` still says "strict reconstruction keep … full total supplies the chooser's advantage".
  - Both contradict the candidate's R+A judge for sentences (since `f7e74af04`) and now for thoughts, and contradict `doc/Architecture.md:2654` and `doc/ExpectationRetention.md:111`.
  - Fix these before the sweep commit.
- **Low note:** κ=1 at word/S and per-symbol κ at R/D are now recorded in ExpectationRetention. I accept that as an explicit implementation choice.

## Still open (not source defects; none waived)
- **B3:** Alec's choice between a prime weight defaulting to 0 with φ=1, or accepting the changed defaults and measuring parity with expectation enabled. Until he answers, there is no approval for the landing sweep.
- **§9 evidence:**
  - Parity with expectation enabled is not measured.
  - The copular-identity checkpoint is still missing, so the forced ordinary-answer certificate can't execute.
  - The carried initial-binding distribution check still fails.
  - Per-level residual and collapse are not established.
  - All-level XOR and the math chain are not established.
  - Global derivation optimality: gate false.
  - The fully expected cat has been shown only as a mechanism with a controlled head, not as learned prediction.
- **Unchanged:** the original failed learning run stays failed and is not regraded. Item 0 and 5.5 are unchanged. No sweep, push or WikiOracle bump has happened, so clean-tree completion is still pending.

**Next step:** once Alec decides B3 (and nods on the thought-judge work price), implement it and fix the two stale contract lines. I don't need another source review for N1 or N2 before the single full sweep.
