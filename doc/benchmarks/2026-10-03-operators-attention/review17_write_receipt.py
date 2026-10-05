"""Render the single §17 receipt from saved results; no model execution."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent


def read(name):
    return json.loads((H / name).read_text())


def f(value):
    return '—' if value is None else f'{value:.7g}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def form(audit, phase):
    return next(book for book in audit[phase]['codes'] if book['rows'])


def room(audit, phase, side):
    rows = [row[side] for row in audit['room'][phase] if row]
    return f"{sum(row['count'] for row in rows)} / {f(max(row['largest'] for row in rows))}"


def main():
    s = read('review17-measurements/summary.json')
    a = read('review17-measurements/audit-summary.json')
    sweep = read('review17-final-sweep/summary.json')
    ordinary = read('review17-ordinary-batch.json')
    c = s['counts']
    counts = sweep['counts']
    assert sweep['exit_code'] == 0 and s['complete']['completed']
    text = [f'''# Decoder, operators and 6.8 — §17 measured candidate

2026-10-05. The frozen §16.3 candidate was committed on `main` as **`eb1fbefb5f4a33a22cbb4a590cc0927d6a60761d`**, tagged **`6.8-s16.3-candidate`**, before any §17 change. Its 693 runtime files match the §16.3 source manifest; the requested current plan, FutureWork, other documentation and historical receipts are included. See the [commit proof](review17-candidate-commit.json) and [incoming receipt](README-before-review17.md). The parent WikiOracle index still points to `802abb1a`. **The §17 work is uncommitted; nothing was pushed. Review is required before the next commit or any push.** One working tree; conceptual capacities remain **6 / 8**.

The final ten-worker sweep is green: **{sweep['selected']:,} completed, {counts.get('passed',0):,} passed, {counts.get('skipped',0)} skipped, {counts.get('xpassed',0)} non-strict XPASS, {counts.get('xfailed',0)} XFAIL, zero failed**, in **{sweep['elapsed_seconds']:.1f} seconds**. The once-only campaign then measured **sum {c['sum_pass']}/10**, **XOR class {c['class_pass']}/10, reconstruction {c['reconstruction_pass']}/10, joint {c['joint']}/10**, and **MM_xor {c['mm_pass']}/10**. Sum was read first. There were no gate retries, source changes during measurement, or tuning.

## What changed

The forward chooser's reconstruction-owned surrogate is `K · R · p(a_dep | shared prefix) · (C_explore − C_greedy)`, with both costs detached and the existing active-row mean reduction. K is the number of value-distinct eligible alternatives at the sampled round; R is the number of eligible rounds in that sentence. K·R cancels the uniform proposal's probability, giving the sum of baseline-subtracted gradients over those eligible alternatives and rounds. The proposal, greedy argmax, eligibility exclusions and strictly-lower-cost keep rule are unchanged. A tie adds no term.

The selected undo now has its own five-weight decomposition scorer over the existing shortlist: negative relative residual, left/right activation and left/right priming. Softmax supplies cross-entropy; hard argmax supplies the pair. Initialization `[1, 0, 0, 0, 0]` reproduces the old residual argmin and consumes no RNG. Teacher targets are the resolved input-word identities at the composition's operand positions. Codes, parent features and context features are detached; only the separate decomposition weights receive this teacher gradient. Targets absent from the shortlist, including compound operands with no word-bank identity, are counted and omitted from CE. CaseSelection retains its separate non-word candidate algebra.

Both trial teacher graphs are built before either update. Their CE enters reconstruction's owner step **after** the existing byte-cost comparison and keep decision, so target labels never enter free decoding or choose the understanding. The walk policy, §11.6 eligibility mask and byte scorer are unchanged. Perception owns prototypes and evidence. No capacity, seed, learning rate, optimizer, budget, gate bar or resource limit changed.

The campaign guard now follows pytest: a **non-strict XPASS is a pass**; a strict XPASS still fails. The .8 overlap assertion and `xfail(strict=False)` on `test_topk_recovered_words_overlap_input` are untouched. Claude's §17 observation was **5 of 8 unseeded passes**, with the other three at overlap .5 because two of four words coincide under the join in that small content width. Those eight runs are Claude's observation; this round ran the overlap test only within its full sweeps, with no standalone reruns to reproduce that estimate. The limitation remains with the operators update.

## Once-only measurements

Both XOR bars consume the same model from one training per run. The unchanged class bar requires all four answers correct and MSE < .05; reconstruction requires all four sentence word-multisets recovered without an unavailable decode. Bands are §20.5: “at 0” means MSE < .05; “at ¼” means |MSE−.25| ≤ .02; the remaining values are between or above ¼. The sum criterion is unchanged: |checkerboard contrast| ≤ 1e-4 and the class bar not met. MM_xor keeps best MSE < .20.
''']
    rows = []
    for x in s['xor']:
        sequences = [' → '.join(names) for names in x['operator_names']]
        operators = sequences[0] if len(set(sequences)) == 1 else '; '.join(
            f'{sentence}: {sequence}' for sentence, sequence in zip(x['inputs'], sequences))
        norms = x['run_audit']['reader_weights']
        rows.append([x['run'], f(x['mse']), x['band'], f"{x['correct']}/4",
                     f"{x['recovered']}/4", 'yes' if x['joint'] else 'no', operators,
                     f"{f(norms[0]['parameters']['answer_record_reader.weight'])} → {f(norms[-1]['parameters']['answer_record_reader.weight'])}"])
    text.append(table(['Run','MSE','Band','Answers','Read-back','Joint','Final greedy operators','Affine reader norm, epoch 1 → 400'], rows))
    text.append('\nBands: ' + ', '.join(f"**{name}: {value}**" for name,value in c['xor_bands'].items()) + '.\n')
    text.append('The advance forecast of reconstruction 10/10 is ' +
                ('met' if c['reconstruction_pass'] == 10 else 'not met') +
                '; the forecast of a majority of class runs at 0 is ' +
                ('met' if c['xor_bands']['at 0'] > 5 else 'not met') +
                '. Quarter-error runs retain their geometry and reader trajectories below; no convergence cause is inferred from a scalar weight norm alone.\n')
    text.append('All ten affine reader norms changed during training; some flattened late while others continued to grow. All **80 final word read-backs were code-decided**, with none decided by priming. The forms themselves had **zero coordinate change** in all ten runs; mean cos(L) ranged from **.7505434 to .9486687**. These are [checks of the saved observations](review17-results-validation.json), with no additional training.\n')
    text.append('''[Complete per-run results](review17-measurements/summary.json) preserve all four named greedy derivations, predictions and **each word's read-back annotation** (code versus priming, winner IDs, ties and scores). Per-epoch reader norms are in each `xor-NN/run-audit.json`, including the actual affine head and the other output parameters separately. The [reader plot](review17-measurements/reader-weight-trajectories.png) shows only `answer_record_reader.weight`; a flat aggregate over all output parameters is not evidence of a stationary reader.

![Affine reader trajectories](review17-measurements/reader-weight-trajectories.png)

Historical comparisons remain observations of their stated sources and capacities:
''')
    text.append(table(['Record','CS XOR / MM','Class','Reconstruction','Joint','MM_xor','Sum'], [
        ['6.9 closing','accepted source','MSE .1147481948','0/4 sentences','—','red through 6.9 §17','—'],
        ['§12','6 / 8','0/10','7/10','0/10','10/10','10/10'],
        ['§13','262 / 264','0/10','0/10','0/10','10/10','10/10'],
        ['§14 before addendum','262 / 264','1/10','8/10','1/10','10/10','10/10'],
        ['§15 and §16.3','6 / 8','not measured','not measured','—','—','—'],
        ['§17','6 / 8',f"{c['class_pass']}/10",f"{c['reconstruction_pass']}/10",f"{c['joint']}/10",f"{c['mm_pass']}/10",f"{c['sum_pass']}/10"]]))
    text.append('''
The accepted closing record's zero ownership conflicts and prior **9/10 class / 5/10 reconstruction** are preserved. Under two spaces and one index, this XOR table measures composition of **forms**, the inverse, the raw-root affine read and ownership. The gate contexts have empty order-zero meanings; their coincidence is correct. MM_xor's field path measures the connectives where meanings exist.
''')
    below = s['below_comparison']
    text.append('Compared with §14, ' + ('the following counts are lower: ' + ', '.join(
        f"{row['count']} {row['prior']}/10 → {row['current']}/10" for row in below) + '.'
        if below else 'no measured count is lower.') + ' These results stand without retries or tuning.\n')
    text.append('## Geometry, evidence and room\n')
    rows = []
    for x in s['xor']:
        audit = x['run_audit']
        start, end = form(audit,'start'), form(audit,'end')
        sv = lambda phase: '[' + ', '.join(f(v) for v in audit[phase]['roots']['centered_singular_values']) + ']'
        rows.append([x['run'], f"{f(start['forms']['mean_pairwise_cosine'])} → {f(end['forms']['mean_pairwise_cosine'])}",
                     sv('start') + ' → ' + sv('end'),
                     f"{f(audit['start']['unit_root_xor_interaction'])} → {f(audit['end']['unit_root_xor_interaction'])}"])
    text.append(table(['Run','Mean cos(L), start → end','Centered root singular values, start → end','Unit-root XOR interaction, start → end'], rows))
    text.append('''
The interaction is `‖r_hw − r_ht − r_lw + r_lt‖` after normalizing each root for this **audit only**. The class reader remains raw and affine. Each run audit saves complete pairwise word cosine matrices, code and root values, singular values and support per word (fraction nonzero and minimum absolute coordinate), at start and end.

`d = relu(e⁺ − e⁻)` is net 11b evidence for a native part or whole address. `d > 0` selects a part at full presence; it does not scale its code. The next table gives all nonempty stages' evidence ranges and room-clamp counts / largest violations. Room remains `L + m ≤ U`, m = 0: only the minimal whole moves upward, with the existing [0,1] clamp. Any remaining violation is recorded as measured.
''')
    rows = []
    for x in s['xor']:
        audit = x['run_audit']
        def evidence(phase):
            d = form(audit,phase)['net_evidence']
            return '; '.join(f"{kind} [{f(row['minimum'])}, {f(row['maximum'])}] (n={row['count']})" for kind,row in d.items())
        rows.append([x['run'], evidence('start') + ' → ' + evidence('end'),
                     room(audit,'start','before') + ' → ' + room(audit,'start','after'),
                     room(audit,'end','before') + ' → ' + room(audit,'end','after')])
    text.append(table(['Run','d range, start → end','First clamp: before → after','Last clamp: before → after'],rows))
    text.append('## Tenth-run audits\n')
    chooser = a['chooser']
    ranges = chooser['per_epoch_ranges'].values()
    text.append(f'''The tenth training supplies **{chooser['sentences']} sentence/step records**, **{chooser['departures']} departures** and **{chooser['nonzero_advantage_sentences']} nonzero advantages**. [Its run audit](review17-measurements/xor-10/run-audit.json) records departure/action, both costs, advantage, K, R, probability before/after and all 400 epoch logit ranges. Finite logits span **[{f(min(r['minimum'] for r in ranges))}, {f(max(r['maximum'] for r in ranges))}]**. Maximum analytical gradient error is **{f(chooser['maximum_gradient_error'])}** and finite-difference error **{f(chooser['maximum_finite_difference_error'])}**.
''')
    decomp = a['decomposition_chooser']
    text.append(table(['Feature','Initial weight','Final weight'],[
        [name,f(before),f(after)] for name,before,after in zip(decomp['start']['feature_names'],
            decomp['start']['weights'],decomp['end']['weights'])]))
    text.append('\n' + table(['Decomposition audit','Start','End'],[
        ['True pair in shortlist',f"{decomp['start']['true_pair_in_shortlist']}/{decomp['start']['undos']}",f"{decomp['end']['true_pair_in_shortlist']}/{decomp['end']['undos']}"],
        ['Pick equals true ordered pair',f"{decomp['start']['pick_equals_true']}/{decomp['start']['undos']}",f"{decomp['end']['pick_equals_true']}/{decomp['end']['undos']}"],
        ['Absent target count',decomp['start']['absent_targets'],decomp['end']['absent_targets']]]))
    gradients = s['xor'][-1]['run_audit']['sentence_gradients']
    text.append(f'''
The target metric uses **ordered input identities**; the unchanged reconstruction gate compares word multisets. Sentence-path gradient at prototypes and evidence is **{max(row['maximum'] for row in gradients.values()):.7g}**, with **{sum(row['nonzero'] for row in gradients.values())} nonzero gradient observations**. Ownership conflicts: **{a['ownership']['conflicts']}**. [Ownership](review17-measurements/xor-10/ownership/ownership.json), [sentence gradients](review17-measurements/xor-10/ownership/sentence-path-ownership.json) and [audit summary](review17-measurements/audit-summary.json) retain the parameter-level evidence.

The same training supplies {a['first_step_records']} decoder first-step records and {a['optimizer_steps']} owner steps. STOP-over-undo logit margins, gradients and actual fixed-parent changes are saved by epoch and named operation; the [raw events](review17-measurements/xor-10/ownership/events.jsonl) and audit summary preserve them. Pooled across both compose trials, kept-path modal fractions by input are {', '.join(f(row['modal_fraction']) for row in a['kept_paths'])}; distinct kept paths are {', '.join(str(row['distinct']) for row in a['kept_paths'])}. The [same saved records separated by compose trial](review17-results-validation.json) give modal fractions **1, 1, 1, 1** for greedy compose and **1, .965, .9425, 1** for explore compose (400 observations per input per trial). Strictly lower reconstruction is checked on every kept-path decision.

## Mechanism checks and sweep repairs

The [ordinary batch](review17-ordinary-batch.json) retains seed 613, batch `["a b c d e", "f g h i j"]`, its original optimizer, one training batch and **no cost override**. Its two `non` departures have K = 1, R = 31. The gradients agree with `K·R·ΔC·∇p` after the existing two-row reduction, and probabilities move in the expected directions:
''')
    text.append(table(['Row','ΔC','p before → after','Max gradient error','Finite-difference error'],[
        [row['batch_row'],f(row['advantage']),f"{f(row['p_before'])} → {f(row['p_after'])}",
         f(row['gradient_max_error']),f(row['finite_difference']['error'])]
        for row in ordinary['compose_score_function_steps']]))
    text.append('''
The [decomposition tests](../../../test/test_decomposition_chooser.py) verify exact argmin initialization without RNG, supervised recovery of the true shortlisted pair in a misleading-context fixture, no absent-target gradient, K×R counts, and use of resolved physical word IDs when the legacy WORD lane is absent. The existing controlled tie/win/dearer chooser tests retain their seed, batch and assertions. The [one-batch XOR wiring audit](review17-mechanism/probe-context.json) is a mechanism check, not a gate run.

Failures were saved before repair, with complete old/new bodies:

- [First full sweep](review17-sweep/summary.json): 4,893 passed, 286 skipped, one XPASS and 14 failures. **13 ports** supplied the new scorer in minimal decoder fixtures or updated the exact independent-parameter inventory. **One regression** omitted the scorer from the explicit optimizer parameter list; it was fixed in production, retaining the ownership assertion.
- [Repaired focused probe](probes/review17-repaired/process.json): 144 passed, one skipped, one failure. The strengthened teacher-coverage audit exposed a **regression**: legacy program word rows were absent. Training now uses reconstruction's resolved physical input rows, matching its candidate bank. [Old/new repair](review17-teacher-row-repair.json); the coverage assertion stays.
- [Same-list repaired probe](probes/review17-row-repaired/process.json): **146 passed, one skipped** across the same twelve files, including the new row-identity regression case.
- [Next full sweep](review17-green-sweep/summary.json): **4,907 passed, 286 skipped, one XPASS, one failure**. This was a **port** of the assertion that reconstruction's registry contains only byte loss; §17 adds decomposition CE. [Complete old/new file](review17-registry-port-proposed.json). Its checks that target leaves do not change decoding or the byte cost are unchanged.
- [Final full sweep](review17-final-sweep/summary.json) and [HTML report](review17-final-sweep/report.html): green. All sixteen original output-gradient regressions pass with assertions intact. All [59 focused files](review17-focused-files.txt), including every file in each repair probe, are included; none dropped. The separate ordinary-batch probe names only its measurement file because it checks the requested estimator identity, not repair coverage.

The [frozen source](review17-final-source/source.zip), [round contracts](review17-contracts-final.json) and [delivery](review17-delivery/source.zip) preserve hashes, complete test ports and zero changed seed calls. Historical receipts are [hash-checked](review17-historical-preservation.json). Resource guards are unchanged; the sole runner semantic change is the authorized non-strict XPASS handling. Collection and every failing probe remain available. The final sweep's skips are the existing default-suite selection, not new exclusions; the 30 explicit gate trainings run separately.

Only todo 6.8, the 6.9 §20.3 status, GradientFlow's chooser descriptions and this receipt are updated here from §17. The plan's concurrent clarification that teacher CE stays outside the byte-cost comparison is [preserved separately](review17-external-docs.json); the implementation already keeps that boundary. Architecture, Philosophy, Spaces, the accessible-mind spec, the operator catalogue and FutureWork remain as committed. Unit-sphere codes stay retired; magnitude/certainty stays returned in cube form; the antipode row stays removed; the distributional row is unchanged. Carried work remains deferred: Kleene connectives, the fold composing forms, complement bootstrap, the not items, the negative image on the concept face, form-density questions and any dense control variate. No native benchmark or fresh bulk BasicModel scoring ran. Frozen evaluation admits nothing; the trained NanoChat gate waits for item 4's checkpoint. **Stop here for review: §17 uncommitted, candidate local, nothing pushed.**
''')
    (H / 'README.md').write_text('\n\n'.join(text))
    print('Wrote one receipt from saved results.')


if __name__ == '__main__':
    main()
