"""Write the held §15 receipt from completed saved probes; never run training."""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROBE = HERE/'probes/review15-final-focused'
HELD = 'test/test_grammar_word_learning.py::test_normal_text_reconstruction_updates_the_grammar_chooser'


def outcomes(folder):
    result = {}
    for path in sorted(folder.glob('worker-*.json')):
        for row in json.loads(path.read_text()).get('reports', []):
            node, outcome = row['nodeid'], row['outcome']
            if node not in result or outcome == 'failed' or (result[node] == 'passed' and outcome != 'passed'):
                result[node] = outcome
    return result


def main():
    result = json.loads((PROBE/'result.json').read_text())
    assert not result.get('active_workers')
    assert len(result['completed']) == len(result['selected'])
    final = outcomes(PROBE)
    failures = [node for node, outcome in final.items() if outcome == 'failed']
    assert failures == [HELD], failures
    counts = dict(collections.Counter(final.values()))
    old = json.loads((HERE/'review14-sweep-summary.json').read_text())
    output_files = ('test_prepared_answer_boundary.py', 'test_trial_policy_ownership.py',
                    'test_generation_catalog.py', 'test_output_path_supervised.py', 'test_arithmetic_isolation.py')
    fixed_output = {r['nodeid']: final.get(r['nodeid']) for r in old['failure_reports']
                    if r['nodeid'].split('::')[0].split('/')[-1] in output_files}
    assert len(fixed_output) == 16 and set(fixed_output.values()) == {'passed'}, fixed_output
    summary = dict(counts=counts, failures=failures, selected=len(result['selected']),
                   completed=len(result['completed']), elapsed_seconds=result['elapsed_seconds'],
                   original_output_regressions=fixed_output, gates_started=False,
                   full_sweep_green=False, source='review15-final-focused-source/source.zip',
                   focused_files=(HERE/'review15-focused-files.txt').read_text().splitlines())
    with (HERE/'review15-final-focused-summary.json').open('x') as f:
        json.dump(summary, f, indent=2); f.write('\n')
    from review15_classify import main as classify
    classify()
    count_text = ', '.join(f'{v} {k}' for k, v in counts.items())
    receipt = f'''# Decoder, operators and 6.8 — §15 candidate held before measurement

2026-10-04. HEAD remains `802abb1acc95e1bddc8cb237b13230a336681c49`. One working tree; nothing committed. Concept capacities are **6 (XOR_grammar) / 8 (MM_grammar)**. The [§14 receipt and addendum](README-before-review15.md) and all earlier measurements stand.

**Status:** the implementation and regression repairs are present, but §15 is not complete or accepted. The final **51-file focused run** completed **{len(final)} cases: {count_text}**. All **16 output-gradient regressions pass with their assertions unchanged**. The unchanged chooser-learning test remains red. The required green full sweep has therefore not been achieved, and **zero §15 gate trainings for sum, XOR_grammar or MM_xor have run**.

## The held contract

§15.4 requires a preference update only when exploration has strictly lower reconstruction cost, and also requires `test_normal_text_reconstruction_updates_the_grammar_chooser` to pass as written. Its fixed seed is still 613; the entire file is unchanged from the incoming §15 candidate. The [saved diagnostic](review15-chooser-after1.jsonl), on the hard-inverse source, recorded exact greedy/explore ties:

| Batch row | Greedy reconstruction | Explore reconstruction | Strict win |
|---|---:|---:|---|
| 0 | 0.48405393958091736 | 0.48405393958091736 | no |
| 1 | 0.18007595837116241 | 0.18007595837116241 | no |

The forced departures select STOP/morphology or another numerically identical unary, so the comparison supplies no preference signal. The preference term is absent and chooser weights stay unchanged. The test unconditionally requires `operand_order.weight` and an MLP weight to change. It still fails in the final focused run. A controlled win/tie mechanism check verifies that a strict win moves the selected logit and a tie registers no term or update; it does not replace this mandatory test.

**Decision pending:** whether the fixture may be made to exercise a strictly better departure while retaining its seed and every assertion. No such setup change has been made. Rewarding a tie, adding another chooser objective, changing the seed, or weakening an assertion would contradict the requested round. This is a held test/rule conflict, not a new class-gate regression finding.

## Implemented source

The form is `L = max(c_p for d_p > 0)`, content coordinates only. **`d = relu(e_for − e_against)`** is net 11b evidence per native part/property address; repeated addresses within one feature group are deduplicated. A positive part edge selects its code at full presence, without multiplying by `d`. Wholes do not enter the form. The midpoint and weighted centroid are gone. Perception prototypes and evidence stay detached from the sentence path; there is no free word Parameter. Certainty belongs to the signed leaf activation, whose projection onto code `c` is `(leaf·c)/(c·c)`. The code's length is not certainty.

The upper bound remains `U = min_w(1-d_w*(1-c_w))`, or one without wholes. The deterministic room pass raises the minimal whole by the full positive violation `relu(L-U+m)`, then clamps it to [0,1]; parts never move in this pass. `latticeMargin` remains zero by default. Fractional whole evidence, another tied minimum, clipping, or a missing adjustable whole can leave residual violations; the report retains before/after counts and maxima without a tolerance or hidden iteration.

XOR and the sum control use the same raw-root, output-owned affine reader. The unit-norm transform and its XML setting are removed. Pair search returns its hard pick from detached candidate codes; its residual remains relative to the parent's mean square. The straight-through pair blend is removed. The generator's straight-through walk is unchanged.

Both trials are costed before training. A strictly better explore trial adds one reconstruction-owned preference term for its forced composition departure, in the existing explore optimizer step. Losing trials and ties add none. Its owners are the compose chooser and, for the stateless AnchorDot scorer, its containing operation anchors. No learning rate, optimizer, budget, seed or bar changed.

## Regression repairs and test ports

The output path inferred occupied roles from nonzero numerical values. A valid zero-valued answer could therefore lose both its conditioned slot and its gradient. It also sent native numerical answer inverses through lexical-bank eligibility and hard candidate substitution. Declared role masks now preserve zero-valued answers, and the native path uses its operator inverse. The lexical decoder keeps its existing support checks. The materialization API still returns no generation targets. All sixteen original output failures are individually checked in the [final focused summary](review15-final-focused-summary.json).

Other regression repairs preserve their assertions: exploration draw dimensions and the `torch.full` fixture interaction; the singleton-axis reduction order; the consistency probe's original mean via catalogued `sum`; and separator staging, where an existing symbol identity now supplies the same resolved physical row to forward composition and the decoder bank, without minting. Generativity fixtures admit their forms before reading derived codes. The explicit answer-expansion fixture uses a captured live word form. The separator fixture supplies the same forward operator as its already-fixed inverse; it still checks two decoded words, both missing-row cases, and trailing separators.

Ports replace superseded free-dictionary-Parameter, competitor-code-gradient, soft inverse, shared-read `data_ptr`, and midpoint expectations. The closing fixture includes the new log-probability journal column; the typed-answer stub accepts occupancy. The sentence-closing test counts top-level writes while retaining valid nested rows. Complete old/new bodies are saved in the [delivery](review15-delivery/test-ports.json) against published HEAD and in the [§15 contracts](review15-contracts-final.json) against the incoming candidate. The [per-failure classification](review15-failure-classification.json) preserves historical failure messages and labels each treatment as a port or regression; the chooser conflict remains unresolved.

## Verification and provenance

The first full default sweep collected and completed **5,181 cases with ten workers**: **4,885 passed, 286 skipped, one XPASS, nine failed**, in **704.8 seconds**, with no runner retries. Its [report](review15-sweep/report.html), [failures](review15-sweep-summary.json), and [source](review15-pre-sweep/source.zip) are preserved. Repairs followed this diagnostic sweep. **Those full-sweep counts do not describe the final repaired source.** A final green full sweep remains required before measurement.

The [final focused run](probes/review15-final-focused/report.html) used all [51 files](review15-focused-files.txt): every original [49-file selection](review15-focused-files-before-sweep-repairs.txt), plus `test_sentence_end_state.py` and `test_thought_answer_adapters.py`. No file was dropped. All narrower diagnostic file/selector lists are in their saved requests or process commands. The 87-case sweep-repair probe finished with 70 passed, 14 skipped and three failed; its two separator failures were saved before the shared-row repair, after which all eight separator tests passed. The initial interrupted diagnostic, the 21-failure focused probe, and subsequent failing probes remain under [probes](probes/), with source archives saved before repairs.

The [one ordinary four-item batch](probes/review15-final-focused/audit-wiring-batch/probe-context.json) validates observation plumbing on the final source. Its [audit](probes/review15-final-focused/audit-wiring-batch/run-audit.json) records **zero sentence-path gradient at native perception prototypes/evidence**, one reader epoch, and one chooser preference-step record; its ownership audit has **zero conflicts**. This is a mechanism check, not a 400-epoch gate training. The preserved original metadata's inherited tenth-gate label is corrected in its accompanying `complete.json` and explicitly explained in `probe-context.json`.

The gate observer is prepared to record each run's `d` ranges at start/end; pairwise word cosines and mean cos(L); support; centered root singular values; `||unit(r_hw)-unit(r_ht)-unit(r_lw)+unit(r_lt)||`; room reports; reader weight trajectories for XOR and sum; read-back decisions per word; final named operators; and the tenth run's ownership, decoder margin/gradient, kept-path stability and chooser-logit movement. **No §15 gate ranges, geometry, bands or convergence counts are claimed.** The [campaign](review15_campaign.py) refuses to start without a complete green sweep on matching frozen source, then requires sum 10/10 before XOR, and MM afterward.

## Historical comparisons — no remeasurement

| Record | XOR / MM_grammar capacity | XOR class | XOR reconstruction | Joint | MM_xor | Sum |
|---|---|---|---|---|---|---|
| 6.9 closing | accepted source | MSE .1147481948 | 0/4 sentences | — | red through §17 | — |
| §12 | 6 / 8 | 0/10 | 7/10 | 0/10 | 10/10 | 10/10 |
| §13 | 262 / 264 | 0/10 | 0/10 | 0/10 | 10/10 | 10/10 |
| §14 before addendum | 262 / 264 | 1/10 | 8/10 | 1/10 | 10/10 | 10/10 |
| §15 | 6 / 8 | not run | not run | not run | not run | not run |

The accepted baseline retains zero ownership conflicts; prior class **9/10** and reconstruction **5/10** remain historical. **1,533 saved measurement/sweep files** were checked against earlier delivery hashes and are unchanged. The XOR table measures perception's composition of forms, its inverse, the affine reader and ownership. Empty same-context meanings are correct; this table is not a grammar gate or a request to separate those meanings. MM_xor's field path measures connectives where meanings exist.

Architecture, GradientFlow, Philosophy, Spaces, the accessible-mind spec, operator catalogue, FutureWork and the incoming 6.8 plan are unchanged from the start of §15. Only todo 6.8, the 6.9 §20.3 rows, and this receipt are updated. Unit-sphere codes remain retired, magnitude/certainty returns in cube form, and the antipode row stays removed; the distributional row is unchanged.

Carried work remains in FutureWork: Kleene connectives, the form fold and anagrams, complement bootstrap, the `not` items, the negative image on the concept face, and percept-presence density. No native benchmark or fresh BasicModel scoring ran; frozen evaluation admits nothing and the trained NanoChat gate waits for item 4's checkpoint. **Nothing committed; held for Claude's review and resolution of the chooser fixture before the green sweep and gates.**
'''
    assert (HERE/'README.md').read_text() == (HERE/'README-before-review15.md').read_text()
    (HERE/'README.md').write_text(receipt)
    todo = ROOT/'todo.md'
    text = todo.read_text()
    start = text.index('   **October 4 §14 round, held for Claude:**')
    end = text.index('   **Item-7 reading residue:**', start)
    replacement = f'''   **October 4 §15 candidate, held before measurement:** the XOR table is the
   **composition mechanism gate** (§12.1). Form codes are the coordinate-wise
   max of parts with positive net evidence `d = relu(e_for-e_against)`, at full
   presence. Wholes only bound the form; the room projection raises the minimal
   whole by the full violation and never shrinks parts. Both class gates use the
   same raw-root output-owned affine head. Pair search is hard and detached;
   a strict explore reconstruction win trains the compose operation preference
   in the existing step. Ties add no term. The generator's straight-through path
   and all seeds, bars, budgets and guards are unchanged.
   The symbol/concept pairing remains two spaces, one index, `[form | meaning]`;
   CS capacities are **6 / 8**, with no conceptual rows or reserve for letters.
   Perception owns prototypes/evidence; sentence gradients there are zero.
   Order-zero meaning is detached context only, empty in these gate fixtures.
   The table measures forms, inverse, affine read and ownership; MM_xor's field
   path measures meanings. Same-context concept coincidence is correct.
   **Verification:** first ten-worker default sweep: 5181 completed,
   4885 passed / 286 skipped / 1 XPASS / 9 failed, saved before further repairs.
   Final focused probe retains all 49 original files and adds the two discovered
   fixture files: **{len(final)} cases, {count_text}**. All sixteen original
   output-gradient regressions pass without assertion changes. Complete old/new
   ports and per-failure classifications are in the one
   [receipt](doc/benchmarks/2026-10-03-operators-attention/README.md).
   **Hold:** the unchanged seed-613 chooser-learning fixture has exact trial
   ties (.48405394 and .18007596) but requires weights to move. The strict-win
   rule permits no preference update. Fixture setup authorization is pending;
   no seed or assertion was changed, and no full-sweep-green claim is made.
   **Zero §15 gate trainings.** After this is resolved: green full sweep, sum ×10
   (10/10 required), ten shared XOR trainings with both bars and all requested
   audits, then MM_xor ×10, once on frozen source. No tuning or gate retries.
   Historical results stand: §12 class/reconstruction/joint **0/7/0** of ten;
   §13 **0/0/0**; §14 **1/8/1** on pre-addendum CS 262/264. Each had MM and sum
   **10/10**. The accepted MSE **.114748**, reconstruction **0/4**, zero conflicts,
   and prior **9/10** and **5/10** remain historical, without remeasurement.
   The geometry, evidence-range, room, reader, margin and chooser observers are
   wired and validated on one ordinary batch, with zero ownership conflicts.
   Protected architecture/design documents are unchanged. The 6.9 §20.3 catalogue
   returns magnitude/certainty in cube form; unit sphere stays retired and the
   antipode row stays removed. Carried items remain in FutureWork's §13–§15 list.
   Frozen evaluation admits nothing; the trained NanoChat gate waits for item 4.
   No fresh BasicModel scoring. **Nothing committed; stop for Claude's review.**
'''
    todo.write_text(text[:start] + replacement + text[end:])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
