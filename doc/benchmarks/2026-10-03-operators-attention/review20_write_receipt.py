"""Render the §20 receipt only from preserved diagnostics and measurements."""
from collections import Counter
from pathlib import Path
import json
from review17_write_receipt import table,f,form,room
H=Path(__file__).resolve().parent
read=lambda n:json.loads((H/n).read_text())
seq=lambda xs:'['+', '.join(f(x) for x in xs)+']'

def main():
 s=read('review20-measurements/summary.json');a=read('review20-measurements/audit-summary.json')
 v=read('review20-results-validation.json');n=read('review20-measurements/start-norms.json')
 sweep=read('review20-sweep/summary.json');c=s['counts'];q=sweep['counts']
 assert sweep['exit_code']==0 and s['complete']['completed']
 parts=[f'''# Decoder, operators and 6.8 — §20 activation magnitude

2026-10-05. **Review hold: all new work is uncommitted, nothing pushed.** Main remains at the local §16.3 candidate `eb1fbefb5f4a33a22cbb4a590cc0927d6a60761d`, tagged `6.8-s16.3-candidate`. The parent WikiOracle index remains `802abb1a`. One working tree, conceptual capacities **6 / 8**.

The complete ten-worker sweep is green: **{sweep['selected']:,} completed; {q.get('passed',0):,} passed, {q.get('skipped',0)} skipped, {q.get('xpassed',0)} non-strict XPASS, {q.get('xfailed',0)} XFAIL, zero failures**, in {sweep['elapsed_seconds']:.1f} seconds. The subsequent once-only campaign measured **sum {c['sum_pass']}/10**, **XOR class {c['class_pass']}/10, reconstruction {c['reconstruction_pass']}/10, joint {c['joint']}/10**, and **MM_xor {c['mm_pass']}/10**. Sum was read first. No gate retries or tuning; seeds, bars, budgets, optimizers, capacities, XML configurations and guards unchanged.

## Causes reported before ports

1. **Tied chooser test: production regression.** The score-function term was zero. `reconstruction.free_bytes` nevertheless reached the shared MLP/tool embedding through input attention's straight-through credit, the cached perception graph and its pullback. The greedy trial's gradient norm at `mlp.0.weight` was .0065753693; there was no grammar-lesson or local preference term. Detaching attention credit at the sentence handoff closes that unintended path. The original no-movement tie assertion is retained.
2. **Distinct case's generator assertion: port.** Every live decoder step had one legal action, hence an exactly zero softmax derivative. Six-action records were an already-empty batch row with parent norm zero. The test still requires movement whenever a live choice exists and requires no movement otherwise. The fixture's seed and batch are unchanged.
3. **MM_grammar row 4: port, not a letter or chunk-promotion regression.** In the original sweep order, the untrained grammar chose a `part` relation. `ClauseTaxonomyPlan` created the native order-one concept `('pool', 9)` at row 4, symbolizing object 4 with part 2. There is no magnitude threshold in that admission. The test now asserts the exact four-word inventory immediately after word admission and checks that any later rows are higher-order native concepts with conceptual parts. It continues to reject concept rows for letters. No admission rule, capacity or promotion threshold was changed.

[Diagnosis and classifications](review20-diagnosis.json), [per-objective gradients and optimizer movements](review20-diagnosis-before-worker-reproduction/test_normal_text_reconstruction_updates_the_grammar_chooser[tie].json), [inventory allocation trace](review20-diagnosis/test_small_inventory_pairs_words_without_allocating_letter_rows[MM_grammar-8].json), and [original-order failing probe](probes/review20-inventory-worker-order/run.log) preserve the evidence. Two observer setup errors (wrong row-owner class, then unsupported optimizer hooks) are saved separately; they changed neither production code nor tests. The expanded observer includes the perception pullback, which a direct-only gradient query misses.

Claude's §21 note arrived after the starting snapshot and is [preserved unchanged](review20-external-plan.txt), with [hashes and provenance](review20-external-docs.json). The delivered repair detaches the sentence handoff; it does **not** add §21.1's proposed attention score-function objective. Our saved seed-613 trace differs from §21.2: its initial legal counts are `[1,1]`, followed by `[6,1]` with zero parent in the six-action row; no live row in that trace has zero legal actions. Both one-code and recomposed-pair eligibility residuals are mean-square errors against the same parent, so dividing both by that parent's positive mean square leaves their comparison unchanged. No eligibility threshold was changed. The §21.3 allocation question is resolved by the call stack: `ClauseTaxonomyPlan` tests a selected `part` relation, positive native references outside truth rows, distinct objects, order compatibility and capacity. It symbolizes the parent when no matching symbol exists; it compares no code norm, similarity or co-activation threshold. The random grammar's selection may respond to changed inputs, but that is different from a scale threshold in admission. These differences remain explicit for review.

## Kernels and ownership

Both binding kernels separate **code direction** from **activation magnitude**. Write `u=unit(x)`, `v=unit(y)` and let a,b be the operand activations: conjunction is `a*b*unit(u*v)`; disjunction is `(a+b-a*b)*unit(u+v-u*v)`. The tensor API treats a supplied nonzero bare code as present; explicit activation arguments override presence. A zero code remains zero. Signed input directions retain polarity; magnitudes use the absolute activation. A repeated conjunction reference retains its direction and activation. The serial chooser supplies native leaves' projection coefficients; a composed kernel root already carries its activation as its norm. The minimal two-tensor chooser API remains supported at full presence.

The sum control still computes its mean and the affine reader still reads the raw root. There is no reader normalization, new trainable parameter or new loss. Pair search remains hard, relative-residual and detached at candidate codes; the byte bank remains detached. The score-function loss, K·R scale, sampled exploration and strict keep rule are unchanged. Perception retains the sole writer of prototypes and evidence. Both full-presence initializers from §§18–19 remain in place.

The first zero-step observation, before any backward or optimizer step, gave forms `[.624036, 2.058787, 1.285894, 2.329995]` in hello/world/there/loving order and four root norms within 1.2e-7 of one. [Raw observation](review20-start-norms.json). This is an unseeded mechanism observation, not an additional training or acceptance gate.

## Verification and ports

The first frozen sweep attempt was stopped after the saved failures were known: 4912/5203 completed. Three additional tests encoded norm-as-certainty; one minimal-state fixture exposed an indexing regression, fixed without changing that test. The interrupt also stopped a nested training-device test; that consequence was classified separately, with no repair. [Preserved initial sweep](review20-sweep-initial/result.json), [classification](review20-sweep-classification.json).

All final focused probes retained the earlier file lists and added the affected contracts: **170 passed, 18 skipped** over the [complete 24-file list](review20-focused-files.json). [Final focused output](probes/review20-final-repaired-focused/run.log). The full delivered-source sweep then completed all tests. All sixteen original output-gradient regression assertions pass. The overlap fixture retains `xfail(strict=False)` and the .8 assertion; its historical unseeded result was about 5/8 passes, otherwise overlap .5 from small-width join collisions. Non-strict XPASS counts as pass; strict XPASS still fails.

[Complete old/new test files](review20-source/test-ports.json) preserve every port, including the binding formula expectations and the decoder credit fixture whose old norm threshold could no longer separate roots from form codes. [Seed audit](review20-source/seed-port-audit.json), [changes from §19](review20-source/changes-from-review19.patch), [full sweep](review20-sweep/summary.json), [HTML report](review20-sweep/report.html).

## Once-only gates

Both XOR bars consume the same training in each run. Class requires four correct answers and MSE < .05; reconstruction requires all four input word-multisets recovered without unavailable decoding. Bands: at 0 means MSE < .05; at ¼ means |MSE−.25| ≤ .02; remaining errors lie between or above. Sum keeps |checkerboard contrast| ≤ 1e-4 and the class bar unmet. MM_xor keeps best MSE < .20. XOR/sum budgets remain 400 epochs; MM_xor at most 200.
''']
 rows=[]
 for x in s['xor']:
  ops=[' → '.join(o) for o in x['operator_names']]
  op=ops[0] if len(set(ops))==1 else '; '.join(f'{t}: {o}' for t,o in zip(x['inputs'],ops))
  w=x['run_audit']['reader_weights']
  rows.append([x['run'],f(x['mse']),x['band'],str(x['correct'])+'/4',str(x['recovered'])+'/4','yes' if x['joint'] else 'no',op,f(w[0]['parameters']['answer_record_reader.weight'])+' → '+f(w[-1]['parameters']['answer_record_reader.weight'])])
 parts.append(table(['Run','MSE','Band','Answers','Read-back','Joint','Final greedy operators','Reader norm, epochs 1 → 400'],rows))
 parts.append('Bands: '+', '.join(f'**{k}: {value}**' for k,value in c['xor_bands'].items())+'.')
 parts.append(table(['Record','CS XOR / MM','Class','Reconstruction','Joint','MM_xor','Sum'],[
 ['6.9 closing','accepted source','MSE .1147481948','0/4 sentences','—','red through 6.9 §17','—'],
 ['§12','6 / 8','0/10','7/10','0/10','10/10','10/10'],['§13','262 / 264','0/10','0/10','0/10','10/10','10/10'],
 ['§14 pre-addendum','262 / 264','1/10','8/10','1/10','10/10','10/10'],['§17','6 / 8','1/10','10/10','1/10','10/10','10/10'],
 ['§18','6 / 8','0/10','10/10','0/10','10/10','10/10'],['§19','6 / 8','not run','not run','not run','not run','not run'],
 ['§20','6 / 8',*[str(c[k])+'/10' for k in ('class_pass','reconstruction_pass','joint','mm_pass','sum_pass')]]]))
 parts.append('The accepted closing record remains .1147481948, reconstruction 0/4 and zero ownership conflicts; prior measurements remain class 9/10 and reconstruction 5/10. This XOR table measures composition of perceptual forms, its inverse, the raw affine read and ownership. Order-zero meanings are context-only and empty here; their coincidence is correct. MM_xor measures the field path where meanings exist. Counts are unpaired measurements, not causal attributions.')
 if s['below_comparison']:
  parts.append('**Counts below the preceding measured §18 campaign:** '+', '.join(f"{r['count']}: {r['current']}/10 versus {r['prior']}/10" for r in s['below_comparison'])+'. These results stand; the green sweep does not imply that every gate passes. No source repair or retraining followed these observations.')
 failures=[]
 for x,checked in zip(s['xor'],v['runs'],strict=True):
  if x['reconstruction_pass']:continue
  collisions=checked['exact_form_collisions']
  failures.append([x['run'],'; '.join(f"{sentence} → {back!r} (unavailable={unavailable})" for sentence,back,unavailable in zip(x['inputs'],x['readbacks'],x['readback_unavailable'],strict=True)),str(collisions['start'])+' → '+str(collisions['end'])])
 if failures:
  parts.append(table(['Reconstruction failure','Saved read-backs','Exact form collisions, start → end'],failures))
  parts.append('These collision comparisons read the saved vectors exactly, without a tolerance, forward call or training. They concern perceptual forms; coinciding empty meanings remain correct. They do not establish a causal comparison with earlier random runs.')
 rb=Counter()
 for x in s['xor']:rb.update(x['readback_counts'])
 parts.append(f"Final read-back annotations: **{dict(rb)}**, {sum(rb.values())} word decisions. Maximum code displacement across XOR runs: **{f(max(r['code_maximum_change'] for r in v['runs']))}**. [Complete per-word annotations, named derivations and 400 reader observations per run](review20-measurements/summary.json).")
 parts.append('Five word positions in run 3 produced no read-back decision; their unavailable/partial sentence outputs are shown above. No final winner was decided by priming. The forecast of 10/10 reconstruction and a majority of class runs at zero was not met. The reader trajectories are retained without tuning or assigning a common cause to the class failures.')
 parts.append('![Affine reader trajectories](review20-measurements/reader-weight-trajectories.png)')
 parts.append('## Starting norms and geometry')
 rows=[]
 for x in n['rounds']['review20']['runs']:
  if x['kind']!='xor':continue
  fs={r['word']:r['l2'] for r in x['forms']};rs={r['sentence']:r['l2'] for r in x['roots']}
  rows.append([x['run'],seq([fs[w] for w in ('hello','world','there','loving')]),seq([rs[w] for w in ('hello world','hello there','loving world','loving there')])])
 parts.append(table(['XOR run','Form L2: hello, world, there, loving','Root L2: hw, ht, lw, lt'],rows))
 parts.append('[All start norms](review20-measurements/start-norms.json), including sum runs and unpaired §§17–18 history, are computed from saved first-trial vectors before owner updates. No extra forwards or trainings were added.')
 rows=[];evidence=[]
 for x in s['xor']:
  au=x['run_audit'];b=form(au,'start');e=form(au,'end')
  rows.append([x['run'],f(b['forms']['mean_pairwise_cosine'])+' → '+f(e['forms']['mean_pairwise_cosine']),seq(au['start']['roots']['centered_singular_values'])+' → '+seq(au['end']['roots']['centered_singular_values']),f(au['start']['unit_root_xor_interaction'])+' → '+f(au['end']['unit_root_xor_interaction'])])
  ev=lambda book:'; '.join(f"{k} [{f(d['minimum'])}, {f(d['maximum'])}] (n={d['count']})" for k,d in book['net_evidence'].items())
  evidence.append([x['run'],ev(b)+' → '+ev(e),room(au,'start','before')+' → '+room(au,'start','after'),room(au,'end','before')+' → '+room(au,'end','after')])
 parts.append(table(['Run','Mean cos(L), start → end','Centered root singular values, start → end','Unit-root XOR interaction, start → end'],rows))
 parts.append('Each run audit also retains full pairwise word/root cosines and per-word support (fraction nonzero, minimum absolute coordinate). The interaction is `‖r_hw−r_ht−r_lw+r_lt‖` on unit roots for observation only. `d=relu(e⁺−e⁻)` is net 11b evidence: positivity selects a part at full presence, not a scale. Room remains L+m≤U, m=0; only the minimal whole moves upward. Room entries below are count / maximum violation.')
 parts.append(table(['Run','d range, start → end','First clamp, before → after','Last clamp, before → after'],evidence))
 parts.append('## Tenth-run audits')
 ch=a['chooser'];ranges=ch['per_epoch_ranges'].values();dc=a['decomposition_chooser'];grads=s['xor'][-1]['run_audit']['sentence_gradients']
 parts.append(f"**{ch['sentences']} sentence/step records; {ch['departures']} departures; {ch['nonzero_advantage_sentences']} nonzero advantages.** Logits range from {f(min(r['minimum'] for r in ranges))} to {f(max(r['maximum'] for r in ranges))}. Maximum error against K·R·ΔC·∇p: {f(ch['maximum_gradient_error'])}; finite-difference error: {f(ch['maximum_finite_difference_error'])}. [Raw costs/actions/advantages/p before-after and per-epoch ranges](review20-measurements/xor-10/run-audit.json).")
 parts.append('The nonzero-advantage count is not near zero: it is 860/1600 (53.75%) in this measurement. The comparison continued to train the compose chooser; all reported logit ranges are finite.')
 parts.append(table(['Decomposition feature','Start weight','End weight'],[[name,f(b),f(e)] for name,b,e in zip(dc['start']['feature_names'],dc['start']['weights'],dc['end']['weights'])]))
 parts.append(table(['Decomposition measure','Start','End'],[[k,f"{dc['start'][k]}/{dc['start']['undos']}",f"{dc['end'][k]}/{dc['end']['undos']}"] for k in ('true_pair_in_shortlist','pick_equals_true','absent_targets')]))
 parts.append(f"Sentence-path prototype/evidence gradient maximum **{f(max(r['maximum'] for r in grads.values()))}**, with **{sum(r['nonzero'] for r in grads.values())}** nonzero observations. Ownership conflicts **{a['ownership']['conflicts']}**. [Ownership](review20-measurements/xor-10/ownership/ownership.json) and [audit summary](review20-measurements/audit-summary.json).")
 parts.append(f"The decoder audit has {a['first_step_records']} first-step records and {a['optimizer_steps']} owner steps, preserving STOP-over-undo margins and gradients, paths and derivation stability in the [events](review20-measurements/xor-10/ownership/events.jsonl). Kept-path modal fractions by compose trial: "+'; '.join(trial+' '+seq([r['modal_fraction'] for r in rows]) for trial,rows in v['kept_stability_by_compose_trial'].items())+'. Every keep decision is checked against strict improvement; ties stay greedy.')
 parts.append('''## Review package

[Source and hashes](review20-source/source.json), [source archive](review20-source/source.zip), [result validation](review20-results-validation.json), [historical preservation](review20-preservation.json) and [delivery manifest](review20-delivery/bridge.json). The incoming receipt is [preserved intact](README-before-review20.md). The campaign reuses the prior observation helpers unchanged and measures the same frozen source that passed collection and the full sweep.

No fresh bulk BasicModel scoring or native benchmark. Frozen evaluation admits nothing; the trained NanoChat gate waits for item 4. Complement bootstrap learning, form fold, Kleene connectives, not items, the concept-face negative image, form density, and the dense control variate remain deferred. **Stop for Claude's review before any commit. Nothing has been pushed.**
''')
 (H/'README.md').write_text('\n\n'.join(parts).rstrip()+'\n')
 print('Wrote §20 receipt from saved results.')
if __name__=='__main__':main()
