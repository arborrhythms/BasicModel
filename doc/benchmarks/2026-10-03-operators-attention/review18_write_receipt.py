"""Write the §18 receipt from saved measurements; no model execution."""
from collections import Counter
import json
from pathlib import Path
from review17_write_receipt import table, f, form, room

H=Path(__file__).resolve().parent

def read(name):return json.loads((H/name).read_text())
def span(row):return f"{f(row['minimum'])}–{f(row['maximum'])} (mean {f(row['mean'])})"
def seq(values):return '['+', '.join(f(v) for v in values)+']'

def main():
    s=read('review18-measurements/summary.json'); a=read('review18-measurements/audit-summary.json')
    v=read('review18-results-validation.json'); n=read('review18-measurements/start-norms.json')
    sweep=read('review18-sweep/summary.json'); c=s['counts']; q=sweep['counts']
    assert sweep['exit_code']==0 and s['complete']['completed']
    parts=[f'''# Decoder, operators and 6.8 — §18 full-presence initialization

2026-10-05. One production change from the measured §17 source: initialize `perceptualSpace._percept_store.byte_fallback.byte_codebook` by per-row max-absolute normalization and the `[0, 1]` presence-cube clamp. **New work is uncommitted; nothing was pushed. Stop for review before any commit.** `main` remains at the local §16.3 candidate `eb1fbefb5f4a33a22cbb4a590cc0927d6a60761d`, tag `6.8-s16.3-candidate`; the parent WikiOracle index remains `802abb1a`. One working tree; conceptual capacities **6 / 8**.

The once-only ten-worker sweep is green: **{sweep['selected']:,} completed, {q.get('passed',0):,} passed, {q.get('skipped',0)} skipped, {q.get('xpassed',0)} non-strict XPASS, {q.get('xfailed',0)} XFAIL, zero failures**, in **{sweep['elapsed_seconds']:.1f} seconds**. The same thirty trainings then measured **sum {c['sum_pass']}/10**, **XOR class {c['class_pass']}/10, reconstruction {c['reconstruction_pass']}/10, joint {c['joint']}/10**, and **MM_xor {c['mm_pass']}/10**. Sum was read first. No retries, tuning or changes to seeds, bars, budgets, assertions or guards.

## Sole production change

```python
init = torch.randn(256, self.dim)
init = init / init.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
self.byte_codebook = nn.Parameter(init.clamp(0.0, 1.0))
```

This replaces `torch.randn(256, self.dim) * 0.02`. It consumes the same single random draw, preserves the Parameter registration and changes only initialization. The clamp clips negative coordinates to zero; it does not impose a unit L2 norm. No training writer, loss, chooser, inverse, reader, projection, capacity or data changes accompany it. The [complete initializer before/after and numerical check](review18-change-verification.json) and [one-function diff](review18-only-change.patch) verify that every other runtime file and function matches §17. No test was ported this round; the earlier complete old/new bodies remain in the [§17 delivery](review17-delivery/test-ports.json).

The first initialization-check harness used a CPU replay generator against the platform's default MPS device and stopped before checking the table. Its [saved failure](review18-init-check-first-attempt.json) precedes the CPU-environment correction in the [probe](review18-init-check.py); no model trained and no production repair followed. The corrected check verifies the exact normalization/clamp, identical RNG consumption and unchanged parameter ownership.

## Why the observed forms remain small

The requested fallback table is initialized at full presence, but **the gate's word forms do not read that table**. The frozen source's `ensure_atomic_bytes()` calls `RadixLayer.insert(chunk)` without an initializer; `insert()` overwrites the admitted what-basis row with `normal_(mean=0, std=0.02)`. `MereologicalCodes._native(0)` returns the perceptual what-basis, and `_interval()` looks up those admitted row IDs. Its part join therefore still sees small random rows. The [source trace](review18-form-source-trace.json) preserves the exact functions and hashes. This corrects §18.1's identification of the source of letter codes; the static trace is corroborated by the measured form norms below. No second initializer or derivation path was changed, and no training was retried.

## Starting form and root norms

These are **Euclidean (L2) norms**, computed from already saved raw vectors at the first reconstruction trial of the first training batch, before an owner update. Forms use their content coordinates; roots use the full recorded vector. No reader normalization, extra forward pass or fresh-model scoring was added. [Per-word and per-sentence norms](review18-measurements/start-norms.json) also include maximum absolute coordinates and vector widths for all twenty lexical trainings (XOR and sum); MM_xor retains its field-path gate.
''']
    rows=[]
    for round_ in ('review17','review18'):
        for kind in ('xor','sum'):
            t=n['rounds'][round_]['aggregate'][kind]
            rows.append([round_.replace('review','§'),kind,span(t['forms']),span(t['roots'])])
    parts.append(table(['Source','Corpus','Form L2 range (mean)','Root L2 range (mean)'],rows))
    rows=[]
    for row in n['rounds']['review18']['runs']:
        if row['kind']!='xor':continue
        fs={item['word']:item['l2'] for item in row['forms']}
        rs={item['sentence']:item['l2'] for item in row['roots']}
        rows.append([row['run'],seq([fs[w] for w in ('hello','world','there','loving')]),
                     seq([rs[w] for w in ('hello world','hello there','loving world','loving there')])])
    parts.append(table(['XOR run','Form norms: hello, world, there, loving','Root norms: hw, ht, lw, lt'],rows))
    parts.append('''Norm comparisons are unpaired observations of the two frozen sources, not reseeded trials. Root norms also depend on the selected operator. The starting and final pairwise word cosines, root singular values, unit-root XOR interaction, form support and room reports remain in the unchanged run audits.

## Once-only gates

The unchanged XOR bars share one trained model per run: class requires all four labels correct and MSE < .05; reconstruction requires all four sentence word-multisets recovered without an unavailable decode. §20.5 bands: “at 0” is MSE < .05; “at ¼” is |MSE−.25| ≤ .02; the remaining values are between or above ¼. The sum bar is |checkerboard contrast| ≤ 1e-4 and the class bar not met; MM_xor keeps best MSE < .20. Budgets remain 400 epochs for XOR/sum and at most 200 for MM_xor.
''')
    rows=[]
    for x in s['xor']:
        ss=[' → '.join(op) for op in x['operator_names']]
        op=ss[0] if len(set(ss))==1 else '; '.join(f'{text}: {o}' for text,o in zip(x['inputs'],ss))
        w=x['run_audit']['reader_weights']
        rows.append([x['run'],f(x['mse']),x['band'],f"{x['correct']}/4",f"{x['recovered']}/4",'yes' if x['joint'] else 'no',op,
                     f"{f(w[0]['parameters']['answer_record_reader.weight'])} → {f(w[-1]['parameters']['answer_record_reader.weight'])}"])
    parts.append(table(['Run','MSE','Band','Answers','Read-back','Joint','Final greedy operators','Reader norm, epoch 1 → 400'],rows))
    parts.append('Bands: '+', '.join(f"**{k}: {val}**" for k,val in c['xor_bands'].items())+'.')
    rb=Counter()
    for x in s['xor']:rb.update(x['readback_counts'])
    parts.append(f"Final word read-back annotations: **{dict(rb)}**, total {sum(rb.values())}. [Complete results](review18-measurements/summary.json) retain every word's code/priming decision, winner and scores, and every named derivation. Maximum code-coordinate displacement across the ten XOR trainings: **{f(max(r['code_maximum_change'] for r in v['runs']))}**. All 400 reader-weight observations per run are preserved, with the actual affine head separated from other output parameters.")
    parts.append('![Affine reader trajectories](review18-measurements/reader-weight-trajectories.png)')
    parts.append(table(['Record','CS XOR / MM','Class','Reconstruction','Joint','MM_xor','Sum'],[
        ['6.9 closing','accepted source','MSE .1147481948','0/4 sentences','—','red through 6.9 §17','—'],
        ['§12','6 / 8','0/10','7/10','0/10','10/10','10/10'],
        ['§13','262 / 264','0/10','0/10','0/10','10/10','10/10'],
        ['§14 before addendum','262 / 264','1/10','8/10','1/10','10/10','10/10'],
        ['§17','6 / 8','1/10','10/10','1/10','10/10','10/10'],
        ['§18','6 / 8',f"{c['class_pass']}/10",f"{c['reconstruction_pass']}/10",f"{c['joint']}/10",f"{c['mm_pass']}/10",f"{c['sum_pass']}/10"]]))
    below=s['below_comparison']
    parts.append('Against §17, '+('no gate count is lower.' if not below else '; '.join(f"{r['count']}: {r['prior']}/10 → {r['current']}/10" for r in below)+'.')+' These are the measured counts; no failed gate is retried or tuned.')
    parts.append('''The accepted 6.9 closing record remains MSE .1147481948, reconstruction 0/4, zero conflicts, with prior class 9/10 and reconstruction 5/10. The XOR table measures perception's composition of forms, its inverse, the affine reader and ownership. Its context-only order-zero meanings are empty; same-context coincidence is correct. MM_xor measures the field path where meanings exist.

## Geometry, evidence and room
''')
    rows=[];erows=[]
    for x in s['xor']:
        audit=x['run_audit']; b=form(audit,'start'); e=form(audit,'end')
        rows.append([x['run'],f"{f(b['forms']['mean_pairwise_cosine'])} → {f(e['forms']['mean_pairwise_cosine'])}",
            seq(audit['start']['roots']['centered_singular_values'])+' → '+seq(audit['end']['roots']['centered_singular_values']),
            f"{f(audit['start']['unit_root_xor_interaction'])} → {f(audit['end']['unit_root_xor_interaction'])}"])
        def evidence(book):return '; '.join(f"{k} [{f(d['minimum'])}, {f(d['maximum'])}] (n={d['count']})" for k,d in book['net_evidence'].items())
        erows.append([x['run'],evidence(b)+' → '+evidence(e),room(audit,'start','before')+' → '+room(audit,'start','after'),room(audit,'end','before')+' → '+room(audit,'end','after')])
    parts.append(table(['Run','Mean cos(L), start → end','Centered root singular values, start → end','Unit-root XOR interaction, start → end'],rows))
    parts.append('The interaction is `‖r_hw − r_ht − r_lw + r_lt‖` after normalization for this audit only. The reader remains raw and affine. Complete cosine matrices and coordinate support (fraction nonzero and minimum absolute value) are saved for every word at start and end.')
    parts.append('`d = relu(e⁺ − e⁻)` is the net 11b evidence per native address. `d > 0` selects a part at full presence; it does not scale it. Room remains `L + m ≤ U`, m = 0, with only the minimal whole moving upward. The room columns give count / largest violation before and after the first and last clamps.')
    parts.append(table(['Run','d ranges, start → end','First clamp: before → after','Last clamp: before → after'],erows))
    parts.append('## Tenth-run audits')
    ch=a['chooser'];r=ch['per_epoch_ranges'].values()
    parts.append(f"The tenth run records **{ch['sentences']} sentence/step observations, {ch['departures']} departures and {ch['nonzero_advantage_sentences']} nonzero advantages**. Finite chooser logits range from **{f(min(v['minimum'] for v in r))} to {f(max(v['maximum'] for v in r))}**. Maximum error against `K·R·ΔC·∇p` is **{f(ch['maximum_gradient_error'])}**; maximum finite-difference error is **{f(ch['maximum_finite_difference_error'])}**. [All records](review18-measurements/xor-10/run-audit.json) preserve actions, both costs, advantages, probabilities before/after, K/R and every epoch's range.")
    d=a['decomposition_chooser']
    parts.append(table(['Decomposition feature','Initial weight','Final weight'],[[name,f(b),f(e)] for name,b,e in zip(d['start']['feature_names'],d['start']['weights'],d['end']['weights'])]))
    parts.append(table(['Decomposition measure','Start','End'],[[key,f"{d['start'][key]}/{d['start']['undos']}",f"{d['end'][key]}/{d['end']['undos']}"] for key in ('true_pair_in_shortlist','pick_equals_true','absent_targets')]))
    grads=s['xor'][-1]['run_audit']['sentence_gradients']
    parts.append(f"Sentence-path prototype/evidence gradient: maximum **{f(max(r['maximum'] for r in grads.values()))}**, **{sum(r['nonzero'] for r in grads.values())}** nonzero observations. Ownership conflicts: **{a['ownership']['conflicts']}**. [Audit summary](review18-measurements/audit-summary.json), [ownership](review18-measurements/xor-10/ownership/ownership.json) and [sentence gradients](review18-measurements/xor-10/ownership/sentence-path-ownership.json) retain the parameter-level checks.")
    parts.append(f"The decoder audit contains **{a['first_step_records']} first-step logit records and {a['optimizer_steps']} owner steps**. STOP-over-undo margins, gradients, fixed-parent changes and named paths remain in the [raw events](review18-measurements/xor-10/ownership/events.jsonl). Kept-path modal fractions conditioned on compose trial: "+'; '.join(trial+' '+seq([r['modal_fraction'] for r in rows]) for trial,rows in v['kept_stability_by_compose_trial'].items())+'. All keep decisions are checked for strict improvement; ties stay greedy.')
    parts.append('''## Verification and review hold

The [full sweep summary](review18-sweep/summary.json), [HTML report](review18-sweep/report.html), collection, worker requests and logs are saved. The sweep covers the same complete test list as §17, plus the documentation-link case for `README-before-review18.md`; nothing was dropped and no assertion was ported. Non-strict XPASS retains pytest's passing semantics. All sixteen original output-gradient regression tests remain unchanged and pass.

The [source archive](review18-source/source.zip) and hashes match collection, the green sweep and all thirty trainings. [Campaign changes](review18-campaign-changes.patch) are only receipt paths and the historical comparison row; all §17 training observers are reused byte-for-byte. [Result validation](review18-results-validation.json) checks source/observer hashes, both XOR bars consuming the same training, zero retries, read-back annotations, code displacement, gradient ownership and kept-path decisions. [Preservation](review18-preservation.json) hashes the prior evidence; the incoming receipt is [archived intact](README-before-review18.md). The [delivery manifest](review18-delivery/bridge.json) binds the exact source and this round's complete evidence.

Architecture and design documents, todo, tests and XML configurations are unchanged by this round. Claude's existing §18 plan review was preserved. Carried operators work remains deferred: the form fold, Kleene connectives over meanings, complement bootstrap, not items, concept-face negative image, form density, and any dense control variate. No native run or bulk fresh-model scoring was added. Frozen evaluation admits nothing; the trained NanoChat gate still waits for item 4. **Stop for review: all post-candidate work remains uncommitted, with nothing pushed.**
''')
    (H/'README.md').write_text('\n\n'.join(parts))
    print('Wrote §18 receipt from saved results.')

if __name__=='__main__':main()
