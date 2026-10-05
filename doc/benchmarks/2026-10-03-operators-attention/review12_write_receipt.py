"""Write the one §12 receipt from saved data; no model or additional training."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE/'review12-measurements'
def read(path): return json.loads(path.read_text())
def mark(value): return 'pass' if value else 'fail'
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])
def operators(row):
    paths=[' → '.join(step['rule_name'] for step in sentence['sequence']) for sentence in row['final_greedy_compose']]
    return paths[0]+' ×4' if len(set(paths))==1 else ' / '.join(paths)

s=read(OUT/'summary.json'); a=read(OUT/'audit-summary.json'); c=s['counts']
retro=read(HERE/'review12-retrospective.json'); controls=read(OUT/'sum-read-first.json')
manifest=read(HERE/'review12-source/manifest.json')
assert controls['passed']==controls['total']==10 and controls['before_xor_or_mm']
assert s['complete']['completed'] and s['complete']['source_matched']
below=', '.join(f"{r['count']} {r['current']}/10 versus {r['prior']}/10" for r in s['below_comparison']) or 'none'
comparison=table(['Measure','§11','§12'], [[label,f'{old}/10',f'{c[key]}/10'] for label,key,old in [
    ('XOR class','class_pass',2),('XOR reconstruction','reconstruction_pass',6),
    ('Both in the same training','joint',2),('MM_xor','mm_pass',10),('Sum control','sum_pass',10)]])
xor=table(['Run','MSE','Band','Labels /4','Read-back /4','Class','Reconstruction','Joint','Final greedy compose'],
    [[r['run'],f"{r['mse']:.9f}",r['band'],r['correct'],r['recovered'],mark(r['class_pass']),
      mark(r['reconstruction_pass']),mark(r['joint']),operators(r)] for r in s['xor']])
old=table(['§11 run','Saved MSE','Band','Saved final compose'],
    [[r['run'],f"{r['mse']:.9f}",r['band'],'not recorded' if r['final_greedy_compose'] is None
      else 'disjunction (then mean), rule 2, all four sentences'] for r in retro['rows']])
mm_sum=table(['Run','MM best MSE','MM calls','MM bar','Sum MSE','Sum contrast','Sum bar'],
    [[m['run'],f"{m['best']:.9f}",m['calls'],mark(m['passed']),f"{v['mse']:.9f}",
      f"{v['contrast']:.9g}",mark(v['sum_bar'])] for m,v in zip(s['mm'],s['sum'],strict=True)])
margin=[]
for binary,v in a['binary'].items():
    epochs=[r for r in a['epochs'] if r['binary']==int(binary)]; first,last=epochs[0],epochs[-1]
    margin.append([f"{v['rule_name']} (rule {v['rule_id']})",f"{first['margin']['mean']:.6f} → {last['margin']['mean']:.6f}",
        f"{last['margin']['min']:.6f}–{last['margin']['max']:.6f}",
        f"{v['nonzero_gradient']}/{v['gradient_difference']['n']}",f"{v['nonzero_change']}/{v['fixed_parent_change']['n']}",
        f"{v['fixed_parent_change']['mean']:.9g}"])
margin=table(['Undo','Raw margin mean, epoch 1 → 400','Final range','Nonzero gradient differences',
              'Nonzero fixed-parent changes','Mean fixed-parent change'],margin)
walks=table(['Walk','Comparisons','Explore kept','Strict violations','Kept-path stability'],
    [[name,r['walks'],r['explore_wins'],r['strict_violations'],
      f"{r['stable_pairs']}/{r['stability_pairs']} = {r['derivation_stability']:.6f}"] for name,r in a['walks'].items()])
maximum=lambda field:max(max(abs(v[field]['min']),abs(v[field]['max'])) for v in a['binary'].values())
peak=max(r['process']['peak_memory_bytes'] for r in s['complete']['jobs'])
text=f'''# Decoder, operators and 6.8 — §12 composition mechanism results

This is the one receipt for the uncommitted candidate from published HEAD
`802abb1acc95e1bddc8cb237b13230a336681c49`, in the existing working tree.
The controlling review and reading are [6.8 §§12–12.1](../../plans/2026-09-27-item-6-8-one-attention.md).
**Nothing is committed or pushed. Work stops for Claude's review.**

One measurement on frozen source, thirty trainings and no retries:
**XOR_grammar class {c['class_pass']}/10, reconstruction {c['reconstruction_pass']}/10,
joint {c['joint']}/10; MM_xor {c['mm_pass']}/10; sum {c['sum_pass']}/10.**
All ten controls completed and were read as 10/10 before XOR or MM started.
Below-comparison counts: **{below}**. Every result is retained.

{comparison}

The table is the **composition mechanism gate**: nonlinear composition into
one understanding, free decoding, affine answer reading, and one objective
owner per parameter. It does not test which operation a grammatical
construction means. Supervised grammatical learning is measured by
`MM_grammar_wording`'s compose lesson and bounded wording gate; the reviewed
item-9 evaluation measures the unsupervised form on item 0's future checkpoint.
This round adds no grammar-learning training or benchmark.

The accepted 6.9 baseline is preserved: **MSE .1147481948, reconstruction 0/4,
zero ownership conflicts**. §22's class **9/10** and reconstruction **5/10**
remain historical measurements. MM_xor was red through 6.9 §17, then measured
10/10 with the affine head in §11. The [§11 results](README-before-review12.md)
stand unchanged and were not retried. [§10](README-before-review11.md) and the
[initial round](README-before-review10.md) remain preserved. The original
single-run class comparison remains **not a regression finding**. No conference freeze.

## Implementation and audit

Disjunction computes `(norm(x)+norm(y)-norm(x)*norm(y))*unit(x+y-x*y)`.
The zero direction yields zero; no norm clamp, parameter, capacity or learning
setting is added. Reverse and generate inherit conjunction's free bounded pair
search through the new kernel. `complete.grammar`, XOR_grammar and MM_grammar
(through `default.grammar`) already select this name and now use its new
implementation. The three XML edits change comments only; parsed elements match.

`sum` retains arithmetic mean and its three numerical faces. The direct balanced
inverse recomposes the parent; the free decoder searches both operands in its
primed bank. It remains the additive control. Both binary choices in XOR_grammar
are now nonlinear, restoring the intended condition of 6.9 §3.11. A fixed-example
rank probe checks affine XOR readability; it does not guarantee convergence or
prevent degenerate learned codes. The unchanged bars remain the criteria.

Compose derivation records contain rule ID, operator name, surface alias, arity
and position. Decoder derivations and margin captures name their rules too.
Names come from the model's held catalogue. Every new XOR run captures final
greedy compose derivations at its existing evaluation boundary, with no extra
forward. Raw placement indices in walk stability are explicitly distinguished
from grammar IDs. Diagnostic strings stay outside model state and gradients.

Sampled departures, same-parameter comparisons, strict owner-cost selection,
ties to greedy, affine numeric reading and support-governed decoding remain.
Frozen evaluation admits no definitions or reservations. The small three-item
evaluator mechanism check passed among the focused tests; the existing
[acceptance probe](nanochat_acceptance_probe.py) remains. **Fresh BasicModel
scoring stays stopped; the trained NanoChat gate waits for item 4's checkpoint.**
Item 1 retains its measured 2.6× forward slowdown.

## §11 retrospective — saved observations only

[Extraction with artifact hashes](review12-retrospective.json),
[reader](review12_retrospective.py). No training or forward was run. Names are
resolved from the frozen §11 grammar and implementation.

{old}

Run 10 supports the mean hypothesis. Runs 1–9 lack saved final compose
derivations, including four of the five quarter-band runs: the claim that all
five settled on mean is **unconfirmed**. Missing operators are not inferred
from MSE. Run 3's .232321 is in the unchanged quarter band, not exactly at the
additive floor. Run 10's final greedy evaluation used mean on all four inputs;
its training audit also contains mixed derivations.

## One frozen-source measurement

[Plan](review12-measurements/plan.json), [completion](review12-measurements/complete.json),
[controls read before gates](review12-measurements/sum-read-first.json),
[per-run summaries](review12-measurements/summary.json), [source](review12-source/manifest.json).

Each XOR model trains once for 400 epochs and supplies both unchanged tests;
the observer checks the same model identity for both consumers. Class requires
four correct labels and MSE < .05. Reconstruction requires all four word
multisets and no unavailable inverse; the old selector containing `50_pct` is
unchanged. §20.5 bands: at 0, MSE < .05; at ¼, abs(MSE−.25) ≤ .02; remaining
errors between or above ¼. Bands do not replace bars.

Compose names are in input order **hello world / hello there / loving world /
loving there**; `×4` means the same whole sequence for all four. Per-run JSON
retains each rule ID, name, arity, position and word row.

{xor}

Bands: **{json.dumps(c['xor_bands'])}**. Joint: **{c['joint']}/10**.

MM uses its unchanged best-MSE < .20 convergence test, up to 200 calls.
Sum uses the original harness, substituting only `sum.forward(S,S)` for the
three compose rules. Its unchanged control is absolute checkerboard contrast
≤ 1e-4 and failure of the class bar. All ten final sum derivations are saved too.

{mm_sum}

## Tenth-run ownership, margins and walk stability

[Full audit](review12-measurements/xor-10/ownership/),
[summary](review12-measurements/audit-summary.json), [plot](review12-measurements/decoder-margin.png).
Ownership conflicts: **{a['ownership']['conflicts']}**; {a['ownership']['active']}
active and {a['ownership']['inactive']} inactive parameters over
{a['ownership']['backward_steps']} backwards. There are {a['first_step_records']}
batched first-logit captures and {a['optimizer_steps']} optimizer steps;
{a['steps_reaching_decoder']} steps reach decoder graphs.
First-step row/path eligibility: **{json.dumps(a['first_step_eligibility'])}**.

{margin}

Gradient difference means `dL/dSTOP − dL/dundo`. Largest absolute STOP gradient:
**{maximum('stop_gradient'):.9g}**; gradient difference: **{maximum('gradient_difference'):.9g}**;
fixed-parent margin change: **{maximum('fixed_parent_change'):.9g}**. Changes come
from the actual optimizer update with the parent held fixed. Raw epoch margins
can also change with the parent; discarded paths may have zero gradient.
Masked actions remain in the audit.

{walks}

The same tenth training supplies both bars and every audit. No attribution
training, full sweep or native run was added. Observed associations are not
assigned as causes of a failed class or decoding bar.

## Verification and complete ports

[Review-start archive](review12-before/manifest.json), [contracts](review12-before-measurement-contracts.json),
[probe/source bridge](review12-source/probe-source-bridge.json). There are
**{manifest['whole_old_new_test_ports']} complete old/new published-HEAD test ports** and **zero changed seed calls**.
Existing assertions in this round's ports are unchanged. Protected XOR and MM
tests, guards, Makefile, pytest.ini and NanoChat manifest match HEAD byte for byte.

| Saved probe | Result and disposition |
|---|---|
| [before-operator-and-audit](probes/review12-before-operator-and-audit/run.log) | 9 fail, 1 pass: old semantics, missing audit helper, and a probe that had not followed MM_grammar's grammar-file reference |
| [formula-and-old-port](probes/review12-formula-and-old-port/run.log) | 18 pass; old mean expectation fails, saved before its parameter-data port |
| [focused](probes/review12-focused/run.log) | 141 pass, 1 fail, 1 skip; synthetic margin fixture lacked new rule metadata |
| [metadata-port](probes/review12-metadata-port/run.log) | 5 pass, including unchanged margin assertions and compiled exploration fixtures |
| [observer](probes/review12-observer/run.log) | Failed before updates: the trial record is local before commit; saved before passing it explicitly |
| [observer-repaired](probes/review12-observer-repaired/run.log) | Failed diagnostic serialization after one batch: placement indices mistaken for rule IDs; saved before annotation repair |
| [observer-final](probes/review12-observer-final/run.log) | One ordinary batch and evaluation pass: four margin captures, three owned steps, zero conflicts, four named final derivations |

Final focused coverage is **142 distinct passing checks and one existing skip**,
plus the observer mechanism check. Bounded mechanism probes are separate from
gate trainings. Every failing probe and source archive precedes its repair.
Seeds, bars, existing assertions, optimizers, learning rates, budgets and guards
are unchanged. Earlier failures and complete ports remain in archived receipts.

All thirty measured processes retain output, exit, guards and final arrays.
Hashes are checked throughout; no measured model was repaired or retried.
Elapsed: **{s['complete']['seconds']:.3f} seconds**; largest worker:
**{peak:,} bytes**. Guards: **8 GiB / 1,800 seconds per worker**, at most three
one-thread workers reserving **24 GiB**, within the standing **28 GiB** ceiling
and CPU headroom. One actual working tree; no commits or pushes.

[Final delivered source and ports](review12-final/manifest.json),
[final source bridge](review12-final-bridge.json), [final contracts](review12-contracts-final.json).
'''
(HERE/'README.md').write_text(text)
print(json.dumps(dict(counts=c,below_comparison=s['below_comparison'],receipt=str(HERE/'README.md'))))
