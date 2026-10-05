"""Build the one §14 receipt exclusively from saved measurements."""
from collections import Counter, defaultdict
import json
from pathlib import Path
H=Path(__file__).resolve().parent
O=H/'review14-measurements'
R=H.parents[2]
def read(p): return json.loads(p.read_text())
def fmt(x):return f'{x:.8g}'
def wordcount(text):return Counter((text or '').replace(chr(0),' ').split())
def yes(x):return 'pass' if x else 'fail'
s=read(O/'summary.json');a=read(O/'audit-summary.json');counts=s['counts']
sweep=read(H/'review14-sweep/result.json')
assert len(sweep['completed'])==len(sweep['selected'])
reports=[r for w in sweep['workers'] for r in w.get('reports',[])]
by_node=defaultdict(list)
for r in reports:by_node[r['nodeid']].append(r)
case_counts=Counter()
for node in sweep['selected']:
 outcomes={r['outcome'] for r in by_node[node]}
 outcome=next((x for x in ('failed','xpassed','passed','xfailed','skipped') if x in outcomes), 'missing')
 case_counts[outcome]+=1
failures=[r for r in reports if r['outcome'] in ('failed','xpassed')]
(H/'review14-sweep-summary.json').write_text(json.dumps(dict(cases=dict(case_counts),
 selected=len(sweep['selected']),completed=len(sweep['completed']),failure_reports=failures,
 retries=sweep['compile_cache_retries'],limits=sweep['limits'],elapsed=sweep['elapsed_seconds']),indent=2)+'\n')
readbacks=[];initial=[]
for row in s['xor']:
 audit=row['run_audit'];mapping={r['row']:r['word'] for book in audit['end']['codes'] for r in book['support']}
 for decision in row['readback_decisions']:
  readbacks.append(dict(run=row['run'],expected_sentence=row['inputs'][decision['batch_row']],
    emitted_word=mapping.get(decision['winner_row']),code_only_word=mapping.get(decision['code_winner_row']),**decision))
 initial.append(sum(wordcount(x)==wordcount(y) and y is not None and not truncated for x,y,truncated in
   zip(row['inputs'],audit['before_learning_readback']['texts'],audit['before_learning_readback']['truncated'])))
(H/'review14-word-readbacks.json').write_text(json.dumps(readbacks,indent=2)+'\n')
trial_paths=defaultdict(Counter)
for line in (O/'xor-10/ownership/events.jsonl').read_text().splitlines():
 event=json.loads(line)
 if event['kind']=='decoder_comparison':
  for row,(win,g,e) in enumerate(zip(event['wins'],event['greedy'],event['explore'])):
   trial_paths[event['trial'],row][tuple(x for x in (e if win else g) if x>=0)]+=1
trial_stability=[dict(compose_trial=trial,batch_row=row,observations=sum(paths.values()),
 modal_fraction=paths.most_common(1)[0][1]/sum(paths.values()),distinct=len(paths),
 paths=[dict(actions=actions,count=n) for actions,n in paths.most_common()])
 for (trial,row),paths in sorted(trial_paths.items())]
(H/'review14-decoder-stability-by-trial.json').write_text(json.dumps(trial_stability,indent=2)+'\n')
lines=['# Decoder, operators and 6.8 — §14 measured candidate, held for Claude','',
'2026-10-04. HEAD remains `802abb1acc95e1bddc8cb237b13230a336681c49`. One working tree; nothing committed. '+
'The [§13 receipt](README-before-review14.md) and every earlier measurement stand without retry.','',
f'**Measured result:** XOR class {counts["class_pass"]}/10, reconstruction {counts["reconstruction_pass"]}/10, joint {counts["joint"]}/10; sum {counts["sum_pass"]}/10 and MM_xor {counts["mm_pass"]}/10. '+
f'The full sweep has {case_counts["failed"]} failed tests. This is a measured candidate held for review, not an acceptance claim.','',
'## Implementation','',
'Codes are perception’s. Conceptual derivation detaches native PS/WS prototypes and 11b evidence; sentence pair search, byte scoring and the affine answer cannot train them. The free word row stays removed. Perception reconstruction retains the native parameters.','',
'With `d = relu(e_for − e_against)`, the content bounds are `L = max_parts(d*c)` and `U = min_WS_property_wholes(1-d*(1-c))`, defaulting to one without wholes. The code is `(W_P*L + W_W*U)/(W_P+W_W)`, or `L` without wholes. The both corner is attention’s. Repeated native part addresses do not multiply an evidence edge. Type location/time coordinates stay zero; occurrence rows remain the detached context mean on the complement, not upper-bound wholes.','',
'The deterministic room pass moves the maximal part down and minimal property whole up by half each positive `L-U+m`, clamped to [0,1]. `ConceptualSpace.latticeMargin=0`. Missing towers retain their fixed bounds. Fractional evidence, clipping and floating-point arithmetic can leave residual violations; exact counts and maxima are reported before and after. No objective was added.','',
'Pair-search candidate codes are detached in residuals and the soft blend; the parent remains live. Residuals use the parent mean square before the unchanged `.01` temperature. Exactly zero parents use the existing zero-target penalty convention (divisor one); positive scales have no floor. Byte scoring detaches the bank and keeps the recovered leaf live.','',
'XOR_grammar reads the root at unit norm through a fixed transform, with both affine reader paths still output-owned. The sum control retains its existing unnormalized affine reader: normalizing an additive root would change the control’s zero-contrast property. This scope choice was stated before freezing; the receipt-local XML diff records it. Learning rates, budgets and optimizers are unchanged.','',
'The antipode function, reporting key, model helper and exactly two antipode tests are deleted. The retained 256-percept reserve supplies a shared-index address for each native percept-concept, whose code refers to its PS row; it adds no free word-code parameter. No new capacity change: XOR_grammar remains 262 rows, MM_grammar 264, with the §13 widths.','',
'**Deferred:** Kleene connectives; the form-band fold; complement bootstrap from wholes’ locations and its co-activation learning; the catalogue §3.8 / plan §13.4 `not` items. These belong to the operators update. The gate’s context complement is empty, so it establishes no learned context result.','',
'## One frozen measurement','',
'The final focused probe passed **141 tests, 1 skipped**. Every §14 focused probe used the same [15-file list](review14-focused-files.txt); no file was dropped. The audit wiring check uses one ordinary four-item training batch in its own guarded process. Failing probe sources, commands and logs were saved before each repair.','',
f'Collection selected **{len(sweep["selected"])} tests**. The full default sweep ran once with **10 workers**, preserving the published slow guards: **'+', '.join(f'{n} {name}' for name,n in case_counts.items())+f'**, {sweep["elapsed_seconds"]:.1f} seconds. Counts are per test, with failures taking precedence over a passed call and failed teardown. '+
'Complete reports are in [the sweep](review14-sweep/report.html) and [failure details](review14-sweep-summary.json). PCH reuse was disabled from the outset to avoid compiler-cache recovery reruns. '+f'The outer sweep made **{len(sweep["compile_cache_retries"])} retries**. Source was not repaired after freezing.','',
'Sum was read first and required 10/10, followed by ten XOR trainings, then ten MM_xor trainings. Each XOR training supplied both unchanged bars. There were **30 gate trainings, zero retries or tuning**. The sweep and gates share the frozen source.','',
'| Gate | 6.9 closing | §12 | §13 | §14 |','|---|---|---:|---:|---:|',
f'| XOR class | single-run MSE .1147481948 | 0/10 | 0/10 | {counts["class_pass"]}/10 |',
f'| XOR reconstruction | 0/4 sentences | 7/10 | 0/10 | {counts["reconstruction_pass"]}/10 |',
f'| XOR joint | — | 0/10 | 0/10 | {counts["joint"]}/10 |',
f'| MM_xor | red through 6.9 §17 | 10/10 | 10/10 | {counts["mm_pass"]}/10 |',
f'| Sum control | — | 10/10 | 10/10 | {counts["sum_pass"]}/10 |','',
'The accepted closing record retains **zero ownership conflicts**. Earlier **class 9/10 and reconstruction 5/10** remain historical comparisons, not rerun measurements. The XOR table is the composition mechanism gate (§12.1), not a grammatical-learning gate.','',
'Bands (§20.5): at 0 means MSE < .05; at ¼ means within .02 of .25; remaining values are between or above ¼. Counts: '+', '.join(f'**{n} {name}**' for name,n in counts['xor_bands'].items())+'.','',
'## Per-run results','',
'Operator abbreviations: C = conjunction, D = disjunction, N = not. Each entry follows the saved sentence order: hello world / hello there / loving world / loving there. Full named derivations and raw measurements are in [summary.json](review14-measurements/summary.json).','',
'| Run | Class MSE | Band | Class | Reconstructed | Joint | Final operators | Code / priming / tie | Before learning |',
'|---|---:|---|---|---:|---|---|---|---:|']
for row,n in zip(s['xor'],initial):
 ops=' / '.join('+'.join({'conjunction':'C','disjunction':'D','not':'N'}.get(x,x) for x in seq) for seq in row['operator_names'])
 rb=' / '.join(str(row['readback_counts'].get(x,0)) for x in ('code','priming','tie'))
 lines.append(f'| {row["run"]} | {fmt(row["mse"])} | {row["band"]} | {yes(row["class_pass"])} | {row["recovered"]}/4 | {yes(row["joint"])} | {ops} | {rb} | {n}/4 |')
lines += ['', '“Before learning” reads the first already-costed training trial, before any optimizer step; it adds no forward or gate training. The forecast of reconstruction without learning is '+('met in all ten recorded first trials.' if all(n==4 for n in initial) else 'not met in all ten recorded first trials.')+' The unchanged final reconstruction bar passes '+str(counts['reconstruction_pass'])+'/10. Each emitted word has its own [read-back annotation](review14-word-readbacks.json), including the code-only winner and whether priming changed it. A missing emitted word has no winner and still fails reconstruction.', '',
'The forecast that every class run ends at 0 or ¼ '+('holds for these ten final measurements.' if not counts['xor_bands']['between'] and not counts['xor_bands']['above 1/4'] else 'does not hold for these ten final measurements.')+' The saved reader trajectories show weight movement; the final error band alone is not a measurement of a plateau. No cause is assigned from an additional run.', '',
'MM_xor best errors (unchanged < .20 bar): '+', '.join(fmt(r['best']) for r in s['mm'])+'.','',
'Sum contrasts (unchanged absolute ≤ 1e-4 and no class-bar pass): '+', '.join(fmt(r['contrast']) for r in s['sum'])+'.','',
'## Reader, geometry and room','',
'![Saved reader weight trajectories](review14-measurements/reader-weight-trajectories.png)','',
'Each [run audit](review14-measurements/) retains all 400 reader-weight norms, start/end code and root cosine matrices, centered singular values, exact per-word support, and first/final room projections. Weight norms below combine the output-owned weight matrices; individual matrices are retained.','',
'| Run | Reader norm after epoch 1 | Epoch 400 | Norm change, epoch 350 → 400 | Root third centered singular value, start → end | Minimum word support, start → end | Room violations, first before/after → final before/after |','|---|---:|---:|---:|---|---|---|']
for r in s['xor']:
 audit=r['run_audit'];w=audit['reader_weights']
 sigma=[audit[p]['roots']['centered_singular_values'][2] for p in ('start','end')]
 support=[min(x['nonzero_fraction'] for b in audit[p]['codes'] for x in b['support']) for p in ('start','end')]
 def room(phase):
  reports=audit['room'][phase]
  return ' / '.join(f'{sum(r[t]["count"] for r in reports)} (max {fmt(max(r[t]["largest"] for r in reports))})' for t in ('before','after'))
 lines.append(f'| {r["run"]} | {fmt(w[0]["norm"])} | {fmt(w[-1]["norm"])} | {fmt(w[-1]["norm"]-w[349]["norm"])} | {fmt(sigma[0])} → {fmt(sigma[1])} | {fmt(support[0])} → {fmt(support[1])} | {room("start")} → {room("end")} |')
lines += ['', 'Room “start” is the first post-step projection; geometry start is the first trial before training. Counts aggregate the reported conceptual stages. No tolerance is used to hide positive room residuals or nonzero code coordinates. Root distinctness is measured, not presumed.', '',
'## Tenth-run audit','',
'![Decoder margins and gradients](review14-measurements/decoder-margin.png)','',f'**{a["ownership"]["conflicts"]} ownership conflicts**. The [sentence-path audit](review14-measurements/xor-10/ownership/sentence-path-ownership.json) includes the full saved perception pullback and records zero gradient at native prototypes and 11b evidence. '+
'Absent and zero gradients are distinguished. The complete [owner list](review14-measurements/xor-10/ownership/ownership.json) retains inactive parameters.','',
f'The audit has **{a["optimizer_steps"]} optimizer steps**, **{a["first_step_records"]} first-step logit records** and **{a["steps_reaching_decoder"]} steps with decoder observations**. '+
'The saved [events](review14-measurements/xor-10/ownership/events.jsonl) retain STOP-minus-undo margins, actual logit gradients and fixed-parent margin changes. [Numerical summaries](review14-measurements/audit-summary.json) include legal masks and kept-path stability. A masked STOP is read as masked, not as a slow-learning policy.','',
'| Undo action | Nonzero margin-gradient differences | Nonzero fixed-parent margin changes |','|---|---:|---:|']
for r in a['binary'].values():lines.append(f'| {r["rule_name"]} | {r["nonzero_gradient"]} | {r["nonzero_change"]} |')
lines += ['', 'The table below pools the decoder calls for both compose trials. Each trial forces a different binary undo. The [same saved events grouped by compose trial](review14-decoder-stability-by-trial.json) show one decoder path for each sentence in each trial, stable through all 400 epochs. The [kept compose derivation](review14-measurements/xor-10/ownership/derivation-stability.json) is disjunction for all four sentences throughout. The pooled ½ modal share and zero adjacent stability in the raw walk counter therefore reflect alternating compose trials.', '',
 '| Sentence | Pooled kept-path modal share | Distinct kept paths | Pooled greedy modal share | Distinct greedy paths |', '|---|---:|---:|---:|---:|']
for sentence,k,g in zip(s['xor'][-1]['inputs'],a['kept_paths'],a['greedy_paths']):
 lines.append(f'| {sentence} | {fmt(k["modal_fraction"])} | {k["distinct"]} | {fmt(g["modal_fraction"])} | {g["distinct"]} |')
lines += ['',f'First-step eligibility: `{a["first_step_eligibility"]}`. Activated-competitor observations: **{a["activated_competitors"]}**, of which **{a["activated_outranks_own"]}** outrank own words. These are observation counts, not unique word counts. Walk departures remain sampled among eligible actions; both paths share pre-update parameters and only strictly lower owner cost keeps exploration.', '',
 '| Walk | Comparisons | Explorable | Explore kept | Strict-cost violations |', '|---|---:|---:|---:|---:|']
for name,row in a['walks'].items():
 lines.append(f'| {name} | {row["walks"]} | {row["explorable"]} | {row["explore_wins"]} | {row["strict_violations"]} |')
lines += ['',
'## Evidence and review','',
'- [Frozen source and complete test ports](review14-source/): 710 files, 169 complete old/new test ports against accepted HEAD; zero seed-call differences. The outgoing §13 candidate is in [review14-before](review14-before/).',
'- [Contract verification](review14-contracts-before-freeze.json) preserves all gate assertions and resource guards. The declared mechanism assertion ports replace live-code/signed-fold and retired antipode expectations. The two inverse fixtures use close candidates for the relative residual, retaining their original gradient assertions.',
'- [Focused file list](review14-focused-files.txt), [saved failing and repaired probes](probes/), and [final focused run](probes/review14-final2/process.json).',
'- [Measurement provenance verification](review14-verification.json), [sum read first](review14-measurements/sum-read-first.json), and [all gate processes](review14-measurements/complete.json).','',
'- [Final review delivery](review14-delivery/): the same measured source with updated documents, complete old/new ports, and a supplement containing this receipt, the saved §14 probes and measurements. [Final contracts](review14-contracts-final.json) retain the gate and guard checks.','',
'No source repair or additional attribution training followed the frozen measurement. No standalone native run or fresh BasicModel scoring was performed. Frozen evaluation still admits no definitions; the trained NanoChat gate waits for item 4’s checkpoint.','',
'**Nothing committed. Held for Claude’s review.** The full-sweep failures and every gate result stand as measured.','']
(H/'README.md').write_text('\n'.join(lines))
p=R/'todo.md';text=p.read_text();start=text.index('   **October 4 §14 round, held for Claude:**' if '   **October 4 §14 round, held for Claude:**' in text else '   **October 4 §13 round, held for Claude:**');end=text.index('   **Item-7 reading residue:**',start)
text=text[:start]+f'''   **October 4 §14 round, held for Claude:** the XOR table remains the
   **composition mechanism gate** (§12.1). Codes are perception's: the
   conceptual lattice derivation detaches native prototypes and 11b evidence;
   pair-search and byte-scoring banks are detached, with the root live.
   The content code is the weighted midpoint of part/WS-property-whole
   bounds. The deterministic room margin defaults to zero. XOR's affine
   class reader reads a unit root; the sum control retains its affine
   unnormalized reader. The antipode API, reporting key and two tests are gone.
   No new capacity change; the native percept-concept shared-index reserve remains.
   **Measured once on frozen source:** class **{counts['class_pass']}/10**,
   reconstruction **{counts['reconstruction_pass']}/10**, joint **{counts['joint']}/10**;
   MM_xor **{counts['mm_pass']}/10**, sum **{counts['sum_pass']}/10**, read first.
   Bands: {counts['xor_bands']}. Each XOR training supplies both unchanged bars.
   The single ten-worker default sweep completed {len(sweep['selected'])} cases:
   {dict(case_counts)}. Its failures stand; no retry or source repair followed.
   Final focused probe: 141 passed, 1 skipped; all 15 files retained.
   Per-run read-back annotations, root/code geometry, support, room reports
   and 400-epoch reader norms accompany the tenth run's ownership, walk/margin
   and stability audits in the one [receipt](doc/benchmarks/2026-10-03-operators-attention/README.md).
   Sentence-path gradients at native prototypes and evidence are zero.
   §12 remains 0/10 class, 7/10 reconstruction, 0/10 joint; §13 remains
   0/10 on all three; both had MM and sum 10/10. The accepted .114748 /
   0-of-4 / zero-conflict baseline and prior 9/10 and 5/10 remain historical.
   **Deferred to the operators update:** Kleene connectives, the form-band
   fold, complement bootstrap/co-activation learning and the catalogued `not`
   items. The gate complement is empty. Frozen evaluation admits nothing;
   the trained NanoChat gate waits for item 4. No fresh BasicModel scoring.
   **Nothing committed; stop for Claude's review.** Seeds, gate assertions,
   bars, budgets and guards are unchanged; complete old/new ports are saved.
''' +text[end:];p.write_text(text)
print(json.dumps(dict(receipt=str(H/'README.md'),counts=counts,sweep=dict(case_counts),before_learning=initial)))
