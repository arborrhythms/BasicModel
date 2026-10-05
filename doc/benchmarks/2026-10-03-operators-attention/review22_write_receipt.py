"""Render the §22 receipt only from preserved diagnostics and measurements."""
from collections import Counter
from pathlib import Path
import json
from review17_write_receipt import table,f,form,room
H=Path(__file__).resolve().parent
read=lambda n:json.loads((H/n).read_text())
seq=lambda xs:'['+', '.join(f(x) for x in xs)+']'

def main():
 s=read('review22-measurements/summary.json');a=read('review22-measurements/audit-summary.json')
 v=read('review22-results-validation.json');n=read('review22-measurements/start-norms.json')
 sweep=read('review22-final-sweep/summary.json');c=s['counts'];q=sweep['counts']
 assert sweep['exit_code']==0 and s['complete']['completed']
 parts=[f'''# Decoder, operators and 6.8 — §22 initialization and content capacity

2026-10-05. **Review hold: all new work is uncommitted, nothing pushed.** Main remains at the local §16.3 candidate `eb1fbefb5f4a33a22cbb4a590cc0927d6a60761d`, tagged `6.8-s16.3-candidate`. The parent WikiOracle index remains `802abb1a`. One working tree.

The complete ten-worker sweep is green: **{sweep['selected']:,} completed; {q.get('passed',0):,} passed, {q.get('skipped',0)} skipped, {q.get('xpassed',0)} non-strict XPASS, {q.get('xfailed',0)} XFAIL, zero failures**, in {sweep['elapsed_seconds']:.1f} seconds. The subsequent once-only campaign measured **sum {c['sum_pass']}/10**, **XOR class {c['class_pass']}/10, reconstruction {c['reconstruction_pass']}/10, joint {c['joint']}/10**, and **MM_xor {c['mm_pass']}/10**. Sum was read first. No gate retries or tuning; seeds, bars, budgets, optimizers and guards unchanged.

## The two declared changes

1. `RadixLayer.insert()` draws the same Gaussian row, divides by its L2 norm, then clamps to [0,1]. The normalization precedes the clamp; there is no second normalization afterward. Explicit initializers and duplicate admission retain their behavior and parameter identity. The §18 byte-fallback initializer is unchanged. [Exact initializer/RNG check and XML-value comparison](review22-change-verification.json).
2. **Declared capacity increase:** `nDim` is 14 → 22 in InputSpace, PartSpace, ConceptualSpace and WholeSpace of both `XOR_grammar.xml` and `MM_grammar.xml`. Each processing event now has **14 content + 4 where + 4 when coordinates**, previously 6 + 4 + 4. Concept-row capacities remain **6 / 8**. The separate MM_grammar WholeSpace output-width override remains 14; all XML values other than the eight declared `nDim` values are unchanged. Fixture comments name the current content width. `MM_xor.xml` is unchanged.

The kernels retain §20's activation magnitudes and code directions. Forms remain joins of positively evidenced parts at full presence. Attention credit stays detached at the sentence handoff; the attention chooser's estimator is the operators update's. The forward score-function K·R term, decomposition chooser, hard pair search, byte scorer, room rule, raw-root affine reader and §20 ports are unchanged. Perception retains the sole writer of its codes and evidence. Architecture, GradientFlow, FutureWork and the 6.8 plan are preserved byte-for-byte from the starting state.

## Verification and ports

The initializer check used no seed override, forward, backward or training. It verified the exact L2-then-clamp row, the same single Gaussian draw, unchanged other rows, parameter identity, duplicate admission and explicit initializers, at widths 6 and 14. Its first XML comparison mistakenly included indentation affected by comment edits; the [failed checker](review22-init-check-before-whitespace-fix.py) and [output](review22-init-check-first-attempt.log) are preserved. The repaired checker compares XML values, with no production repair.

The initial full sweep was interrupted after **5030/5204** cases to port two old six-coordinate assertions. The failures were `test_actual_serial_code_uses_six_native_coordinates_and_no_context_bootstrap` and the nested audit-wiring probe's support-dimension assertion. Both now expect the declared fourteen-coordinate content band; the first test's name and address-band slice follow that width. The historical §17 probe remains intact; the wrapper selects a §22 copy. No regression assertion, gate bar, ownership assertion or seed changed. [Initial source](review22-source/source.zip), [initial sweep](review22-sweep/result.json), [failure classification](review22-sweep-classification.json), [complete test-file ports](review22-delivered-source/test-ports.json), [complete nested-probe port](review22-observer-port.json).

The final focused probe retained its first file and added the second affected file: **21/21 passed**, over [the declared file list](review22-focused-files.json). [Focused result](probes/review22-final-focused/result.json). The subsequent full sweep completed all tests on the delivered source, including all sixteen original output-gradient regression assertions. Non-strict XPASS counts as a pass; the overlap fixture retains `xfail(strict=False)` and its .8 assertion (historically about 5/8 unseeded passes, otherwise overlap .5 from small-width join collisions). [Full sweep](review22-final-sweep/summary.json), [HTML report](review22-final-sweep/report.html), [seed audit](review22-delivered-source/seed-port-audit.json), [changes from §20](review22-delivered-source/changes-from-review20.patch).

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
 ['§14 pre-addendum','262 / 264','1/10','8/10','1/10','10/10','10/10'],['§15–§16.3','6 / 8','not measured','not measured','not measured','not measured','not measured'],
 ['§17','6 / 8','1/10','10/10','1/10','10/10','10/10'],
 ['§18','6 / 8','0/10','10/10','0/10','10/10','10/10'],['§19','6 / 8','not run','not run','not run','not run','not run'],
 ['§20','6 / 8','2/10','8/10','2/10','10/10','10/10'],
 ['§22 (content 14)','6 / 8',*[str(c[k])+'/10' for k in ('class_pass','reconstruction_pass','joint','mm_pass','sum_pass')]]]))
 parts.append('The accepted closing record remains .1147481948, reconstruction 0/4 and zero ownership conflicts; prior measurements remain class 9/10 and reconstruction 5/10. This XOR table measures composition of perceptual forms, its inverse, the raw affine read and ownership. Order-zero meanings are context-only and empty here; their coincidence is correct. MM_xor measures the field path where meanings exist. Counts are unpaired measurements, not causal attributions.')
 if s['below_comparison']:
  parts.append('**Counts below the preceding measured §20 campaign:** '+', '.join(f"{r['count']}: {r['current']}/10 versus {r['prior']}/10" for r in s['below_comparison'])+'. These results stand; the green sweep does not imply that every gate passes. No source repair or retraining followed these observations.')
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
 parts.append(f"Final read-back annotations: **{dict(rb)}**, {sum(rb.values())} word decisions. Maximum code displacement across XOR runs: **{f(max(r['code_maximum_change'] for r in v['runs']))}**. [Complete per-word annotations, named derivations and 400 reader observations per run](review22-measurements/summary.json).")
 missing=80-sum(rb.values())
 parts.append(f"Missing final read-back decisions: **{missing}/80** word positions. Priming-decided winners and ties are retained in the annotations; the full [per-position annotation file](review22-measurements/readback-annotations.json) includes unavailable positions. The forecast of 10/10 reconstruction is {'met' if c['reconstruction_pass']==10 else 'not met'}; a majority of class runs at zero is {'met' if c['xor_bands']['at 0']>5 else 'not met'}.")
 parts.append('![Affine reader trajectories](review22-measurements/reader-weight-trajectories.png)')
 parts.append('## Starting norms and geometry')
 rows=[]
 for x in n['rounds']['review22']['runs']:
  if x['kind']!='xor':continue
  fs={r['word']:r['l2'] for r in x['forms']};rs={r['sentence']:r['l2'] for r in x['roots']}
  rows.append([x['run'],seq([fs[w] for w in ('hello','world','there','loving')]),seq([rs[w] for w in ('hello world','hello there','loving world','loving there')])])
 parts.append(table(['XOR run','Form L2: hello, world, there, loving','Root L2: hw, ht, lw, lt'],rows))
 parts.append('[All start norms](review22-measurements/start-norms.json), including sum runs and unpaired §§17–18 and §20 history, are computed from saved first-trial vectors before owner updates. No extra forwards or trainings were added.')
 rows=[];evidence=[]
 for x in s['xor']:
  au=x['run_audit'];b=form(au,'start');e=form(au,'end')
  rows.append([x['run'],f(b['forms']['mean_pairwise_cosine'])+' → '+f(e['forms']['mean_pairwise_cosine']),seq(au['start']['roots']['centered_singular_values'])+' → '+seq(au['end']['roots']['centered_singular_values']),f(au['start']['unit_root_xor_interaction'])+' → '+f(au['end']['unit_root_xor_interaction'])])
  ev=lambda book:'; '.join(f"{k} [{f(d['minimum'])}, {f(d['maximum'])}] (n={d['count']})" for k,d in book['net_evidence'].items())
  evidence.append([x['run'],ev(b)+' → '+ev(e),room(au,'start','before')+' → '+room(au,'start','after'),room(au,'end','before')+' → '+room(au,'end','after')])
 parts.append(table(['Run','Mean cos(L), start → end','Centered root singular values, start → end','Unit-root XOR interaction, start → end'],rows))
 parts.append('Each run audit also retains full pairwise word/root cosines and per-word support (fraction nonzero, minimum absolute coordinate). The interaction is `‖r_hw−r_ht−r_lw+r_lt‖` on unit roots for observation only. `d=relu(e⁺−e⁻)` is net 11b evidence: positivity selects a part at full presence, not a scale. Room remains L+m≤U, m=0; only the minimal whole moves upward. Room entries below are count / maximum violation.')
 parts.append(table(['Run','d range, start → end','First clamp, before → after','Last clamp, before → after'],evidence))
 parts.append(table(['Run','Exact form collisions at start','Exact form collisions at end'],[[row['run'],str(row['exact_form_collisions']['start']),str(row['exact_form_collisions']['end'])] for row in v['runs']]))
 parts.append('## Tenth-run audits')
 ch=a['chooser'];ranges=ch['per_epoch_ranges'].values();dc=a['decomposition_chooser'];grads=s['xor'][-1]['run_audit']['sentence_gradients']
 parts.append(f"**{ch['sentences']} sentence/step records; {ch['departures']} departures; {ch['nonzero_advantage_sentences']} nonzero advantages.** Logits range from {f(min(r['minimum'] for r in ranges))} to {f(max(r['maximum'] for r in ranges))}. Maximum error against K·R·ΔC·∇p: {f(ch['maximum_gradient_error'])}; finite-difference error: {f(ch['maximum_finite_difference_error'])}. [Raw costs/actions/advantages/p before-after and per-epoch ranges](review22-measurements/xor-10/run-audit.json).")
 parts.append(f"Nonzero advantages account for {100*ch['nonzero_advantage_sentences']/ch['sentences']:.2f}% of the sentence/step records; finite logit ranges: {v['finite_chooser_logits']}.")
 parts.append(table(['Decomposition feature','Start weight','End weight'],[[name,f(b),f(e)] for name,b,e in zip(dc['start']['feature_names'],dc['start']['weights'],dc['end']['weights'])]))
 parts.append(table(['Decomposition measure','Start','End'],[[k,f"{dc['start'][k]}/{dc['start']['undos']}",f"{dc['end'][k]}/{dc['end']['undos']}"] for k in ('true_pair_in_shortlist','pick_equals_true','absent_targets')]))
 parts.append(f"Sentence-path prototype/evidence gradient maximum **{f(max(r['maximum'] for r in grads.values()))}**, with **{sum(r['nonzero'] for r in grads.values())}** nonzero observations. Ownership conflicts **{a['ownership']['conflicts']}**. [Ownership](review22-measurements/xor-10/ownership/ownership.json) and [audit summary](review22-measurements/audit-summary.json).")
 parts.append(f"The decoder audit has {a['first_step_records']} first-step records and {a['optimizer_steps']} owner steps, preserving STOP-over-undo margins and gradients, paths and derivation stability in the [events](review22-measurements/xor-10/ownership/events.jsonl). Kept-path modal fractions by compose trial: "+'; '.join(trial+' '+seq([r['modal_fraction'] for r in rows]) for trial,rows in v['kept_stability_by_compose_trial'].items())+'. Every keep decision is checked against strict improvement; ties stay greedy.')
 parts.append('''## Review package

[Source and hashes](review22-delivered-source/source.json), [source archive](review22-delivered-source/source.zip), [result validation](review22-results-validation.json), [historical preservation](review22-preservation.json) and [delivery manifest](review22-delivery/bridge.json). The incoming receipt is [preserved intact](README-before-review22.md). The campaign reuses the prior training observers unchanged; the separate audit-wiring probe is ported for the declared width. It measures the same frozen source that passed collection and the full sweep.

The comparison is unpaired and includes a declared content-capacity change; it does not isolate the effects of initialization and width. No fresh bulk BasicModel scoring or native benchmark. Frozen evaluation admits nothing; the trained NanoChat gate waits for item 4. Complement bootstrap learning, form fold, Kleene connectives, not items, the concept-face negative image, form density, and the dense control variate remain deferred. **Stop for Claude's review before any commit. Nothing has been pushed.**
''')
 (H/'README.md').write_text('\n\n'.join(parts).rstrip()+'\n')
 print('Wrote §22 receipt from saved results.')
if __name__=='__main__':main()
