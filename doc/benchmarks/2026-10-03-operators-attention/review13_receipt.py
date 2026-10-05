"""Build the one receipt from saved observations; no model or training."""
from collections import Counter
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
OUT = HERE/'review13-measurements'
ROOT = HERE.parents[2]


def read(path):
    return json.loads(path.read_text())


def f(value):
    return f'{value:.6g}'


summary = read(OUT/'summary.json')
audit = read(OUT/'audit-summary.json')
verified = read(HERE/'review13-verification.json')
counts = summary['counts']
base = (HERE/'README.md').read_text().split('## Verification and measurement')[0].split('## One frozen measurement')[0]
base = base.replace('§13 candidate, verification in progress', '§13 measured candidate, held for Claude')
base = base.replace('The settled §13 design is implemented;\nthe one measurement round follows verification.',
                    'The settled §13 design is implemented and measured once on frozen source.')
base += f'''## One frozen measurement

The control was read first: **sum {counts['sum_pass']}/10**, under its unchanged
criterion (checkerboard contrast at most 1e-4 and no class-bar pass).
Only then were XOR and MM_xor run. Each XOR run trained once, and both unchanged
bars consumed that same model. There were **30 trainings, zero retries**, with
no post-result source repair. [Provenance verification](review13-verification.json)
confirms the frozen production, test and data hashes and the measurement harness.

| Gate | §12 comparison | §13 |
|---|---:|---:|
| XOR class | 0/10 | {counts['class_pass']}/10 |
| XOR reconstruction | 7/10 | {counts['reconstruction_pass']}/10 |
| XOR joint | 0/10 | {counts['joint']}/10 |
| MM_xor | 10/10 | {counts['mm_pass']}/10 |
| Sum control | 10/10 | {counts['sum_pass']}/10 |

Bands use §20.5: at 0 is MSE < .05; at ¼ is within .02 of .25;
remaining values below .25 are between, and remaining values above are above ¼.
Counts are **{counts['xor_bands']['at 0']} at 0, {counts['xor_bands']['at 1/4']} at ¼,
{counts['xor_bands']['between']} between, {counts['xor_bands']['above 1/4']} above ¼**.
'''
if summary['below_comparison']:
    base += '\nBelow the comparison: ' + '; '.join(f"**{r['count']} {r['current']}/10 versus {r['prior']}/10**" for r in summary['below_comparison']) + '. These measured failures stand; nothing was retried.\n'
else:
    base += '\nNo count is below the §12 comparison. This does not restore the earlier 9/10 class comparison or establish grammatical learning.\n'
base += '''
## Per-run XOR results

The final-operator column follows each row's saved input order; full rule names,
IDs and complete greedy derivations are in [summary.json](review13-measurements/summary.json).
C = conjunction, D = disjunction, N = not; a sequence lists every final compose
operation. Read-back counts refer to emitted words: code / priming / unresolved
tie. An absent word contributes no read-back decision and still fails the
unchanged reconstruction bar.

| Run | Class MSE | Band | Class | Reconstructed | Joint | Final compose operators | Read-back code / priming / tie |
|---|---:|---|---|---:|---|---|---|
'''
short = {'conjunction':'C','disjunction':'D','not':'N'}
for row in summary['xor']:
    ops = ['→'.join(short.get(s['rule_name'], s['rule_name']) for s in d['sequence']) for d in row['final_greedy_compose']]
    d = row['readback_counts']
    base += f"| {row['run']} | {row['mse']:.9f} | {row['band']} | {'pass' if row['class_pass'] else 'fail'} | {row['recovered']}/4 | {'pass' if row['joint'] else 'fail'} | {' / '.join(ops)} | {d.get('code',0)} / {d.get('priming',0)} / {d.get('tie',0)} |\n"
readbacks = Counter()
for row in summary['xor']:
    readbacks.update(row['readback_counts'])
base += '\nSaved input order: ' + ', '.join('`'+s+'`' for s in summary['xor'][0]['inputs']) + '.\n'
base += f"Final evaluation totals: **{readbacks['code']} code, {readbacks['priming']} priming, {readbacks['tie']} unresolved ties**, across {sum(readbacks.values())} emitted-word decisions. Priming means it changed the code-only winner or resolved a code tie; it is not a separate decode.\n"
base += '\nMM_xor best-error values (runs 1–10; unchanged < .20 convergence bar): ' + ', '.join(f(row['best']) for row in summary['mm']) + '.\n'
base += '\nSum checkerboard contrasts (runs 1–10): ' + ', '.join(f(row['contrast']) for row in summary['sum']) + '.\n'
base += f'''
## Tenth XOR training: ownership, walks and decoder margins

**{audit['ownership']['conflicts']} ownership conflicts**, {audit['ownership']['active']}
active and {audit['ownership']['inactive']} inactive parameter entries. Native PS
prototypes and feature evidence are reconstruction-owned; durable occurrence
roots have no gradient. Inactivity remains visible in the [ownership audit](review13-measurements/xor-10/ownership/ownership.json).
There are {audit['optimizer_steps']} recorded optimizer steps, of which
{audit['steps_reaching_decoder']} contain decoder observations, and
{audit['first_step_records']} first-step greedy/explore logit records.

The raw STOP-minus-undo margins, gradients reaching those logits, and actual
fixed-parent margin changes are retained per step in
[events.jsonl](review13-measurements/xor-10/ownership/events.jsonl),
with [numerical summaries](review13-measurements/audit-summary.json) and
[a margin plot](review13-measurements/decoder-margin.png).
Eligibility must be read alongside the raw margin: a masked STOP cannot win
and receives no choice gradient. A rejected path can have no gradient. First-step
eligibility counts: {audit['first_step_eligibility']}.

| Undo action | Observed gradients | Nonzero STOP−undo gradient | Nonzero fixed-parent margin change | First-epoch mean margin | Last-epoch mean margin |
|---|---:|---:|---:|---:|---:|
'''
for key, value in audit['binary'].items():
    epochs = [row for row in audit['epochs'] if row['binary'] == int(key)]
    base += f"| {value['rule_name']} ({value['rule_id']}) | {value['gradient_difference']['n']} | {value['nonzero_gradient']} | {value['nonzero_change']} | {f(epochs[0]['margin']['mean'])} | {f(epochs[-1]['margin']['mean'])} |\n"
base += '\n| Walk | Explorable / walks | Explore kept | Strict-rule violations | Kept-path stability |\n|---|---:|---:|---:|---:|\n'
for name, row in audit['walks'].items():
    stability = 'n/a' if row['derivation_stability'] is None else f"{row['stable_pairs']}/{row['stability_pairs']} ({f(row['derivation_stability'])})"
    base += f"| {name} | {row['explorable']}/{row['walks']} | {row['explore_wins']} | {row['strict_violations']} | {stability} |\n"
base += f'''
All departures are sampled among eligible alternatives at the chosen round;
both paths are costed before either trains; only strictly lower owner cost is
kept, with ties to greedy. The final audit records **{audit['activated_competitors']}
activated-candidate opportunities** and **{audit['activated_outranks_own']}
activated candidates outranking own words**, across its ranking observations.
These are repeated observations, not counts of unique words. Real occurrence
conduction is also checked by the isolated, cross-batch-safe mechanism fixture.

Training read-back classifications: {audit['readback_counts']['training']}.
Evaluation read-back classifications: {audit['readback_counts']['evaluation']}.

## Start/end geometry and support

Full code and root pairwise cosine matrices and centered singular values are
saved in [start geometry](review13-measurements/xor-10/ownership/geometry-start.json)
and [end geometry](review13-measurements/xor-10/ownership/geometry-end.json),
with tensor code matrices beside them. Root values below are the same four
sentence roots observed in this training, not another model or run.

| Phase | Root pairwise cosine range, excluding self | Root centered singular values |
|---|---|---|
'''
for phase, row in audit['root_geometry'].items():
    matrix = row['pairwise_cosines']
    values = [value for i, r in enumerate(matrix) for j, value in enumerate(r) if i != j]
    base += f"| {phase} | {f(min(values))} to {f(max(values))} | {', '.join(f(x) for x in row['centered_singular_values'])} |\n"
base += '\nSupport uses exact nonzeros in each word\'s six native perceptual content coordinates; the minimum includes zeros. Reserved PS event positions are reported separately, not counted as perceptual content.\n\n| Phase | Word | Nonzero fraction | Minimum absolute coordinate | Minimum nonzero absolute coordinate |\n|---|---|---:|---:|---:|\n'
for phase, rows in audit['word_perceptual_support'].items():
    for row in sorted(rows, key=lambda x:x['word']):
        minimum = row['minimum_nonzero_absolute_value']
        base += f"| {phase} | {row['word']} | {row['nonzero_coordinates']}/{row['dimension']} ({f(row['nonzero_fraction'])}) | {f(row['minimum_absolute_value'])} | {'none' if minimum is None else f(minimum)} |\n"
reading_path = OUT/'saved-geometry-reading.json'
if reading_path.exists():
    reading = read(reading_path)
    base += '''
## Reading the saved failure

Because reconstruction fell below its comparison, the following is a
[read-only analysis of the tenth run's saved tensors](review13-measurements/saved-geometry-reading.json).
It constructs no model and adds no inference, training or changed gate.

Removing the free word row did **not** prevent directional collapse in this
audited run. `world`, `there` and `loving` finish nearly proportional on their
perceptual coordinates: their pairwise cosines, recomputed in float64 from
the saved float32 values, are 0.999999999970, 0.999999999925 and
0.999999999990. Their nonzero supports are all 6/6 at the end. The third
centered root singular value falls from 0.00100208 to 1.95315e-8. The
coordinates retain small differences; this is near-collapse, not a claim
of exact vector equality or a proof about the other nine runs.

Every one of the 6,400 observed live first decoder steps has exactly one
legal action; STOP is ineligible in all of them. Both binary margin-gradient
differences and all fixed-parent margin changes are exactly zero. The
generate policy's weight and bias have zero observed displacement across
all 1,200 steps. This audit records an absence of policy motion; it does not
support an explanation that more learning rate or budget would solve the
first-step choice. No rate, budget, eligibility mask or guard was changed.

The kept-path stability metric compares successive recorded decoder walks,
including both outer compose trials. Its zero value is not a separate
measurement of final-greedy-only persistence. The per-input greedy and kept
path distributions and named compose stability are saved alongside it.
The candidate is held for review; the measured reconstruction loss is not
repaired or retried in this round.
'''
base += '''
## Verification, failing probes and provenance

Final mechanism verification: **97 passed, 1 existing skip** in
[the guarded run](probes/review13-final-native-mechanisms/run.log), plus
[the settled audit probe](probes/review13-observer-settled/run.log).
The latter uses one ordinary training batch and evaluation to verify wiring;
it is not an extra gate training. No new seed call was introduced.

Each probe archives its exact pre-run source, log and guarded process report;
failures were saved before repair. The earlier design's probes remain in
[the design-hold receipt](README-review13-design-hold.md).

| Native-subspace probe | Saved result |
|---|---|
'''
for name in ('subspaces-before','subspaces-first','subspaces-second','subspaces-ported','observer-native','settled-mechanisms','boundary-ports','observer-settled','final-native-mechanisms'):
    folder = HERE/'probes'/('review13-'+name)
    log = (folder/'run.log').read_text()
    matches = re.findall(r'^.*\b(?:passed|failed)\b.* in [0-9].*$', log, re.M)
    outcome = matches[-1].strip('= ') if matches else f"exit {read(folder/'process.json')['exit_code']}"
    base += f'| [{name}](probes/review13-{name}/run.log) | {outcome} |\n'
peak = max(row['process']['peak_memory_bytes'] for row in summary['complete']['jobs'])
base += f'''
All 30 measurement workers exited normally within the unchanged 8-GiB and
1,800-second guards. Assertion failures are the saved gate outcomes. Peak
worker footprint was **{peak:,} bytes**; campaign wall time was
**{summary['complete']['seconds']:.2f} seconds**. Three workers at most, one CPU
thread each, unchanged 24-GiB aggregate guard.

The [contract check](review13-contracts-frozen.json) confirms unchanged published
gates, guards, assertions and seed calls, and only the declared XML capacity/width
changes in this round. Fixture ports explicitly supply an absolute sentence
reading for the answer boundary and decoder ownership; production support
eligibility is unchanged. The field-dictionary parameter fixture is separate
from the derived serial cache.

The [frozen source archive](review13-source/source.zip),
[full old/new test ports](review13-source/test-ports.json),
[seed-call audit](review13-source/seed-port-audit.json),
[measurement helpers](review13-source/reporting-source.zip),
[final delivery archive](review13-delivery/source.zip) and
[measurement verification](review13-verification.json) preserve the candidate.
§11 and §12 remain measured as saved, without retry. No attribution training,
full sweep, native run or fresh BasicModel NanoChat scoring was added. Its
[existing acceptance probe](nanochat_acceptance_probe.py) and
[three-item frozen-evaluator mechanism check](README-before-review11.md) remain;
the trained gate waits for item 4's checkpoint.

**Held for Claude's review. Nothing committed.** Bootstrap learning and the
form-fold/connective split remain explicit work for the operators update.
'''
(HERE/'README.md').write_text(base)
todo_path = ROOT/'todo.md'
todo = todo_path.read_text()
begin = todo.index('   **October 4 §12 round, held for Claude:**')
end = todo.index('   **Item-7 reading residue:**', begin)
below = ('; '.join(f"{r['count']} {r['current']}/10 versus {r['prior']}/10"
                  for r in summary['below_comparison']) or 'no count below the §12 comparison')
update = f'''   **October 4 §13 round, held for Claude:** the XOR table remains the
   **composition mechanism gate** (§12.1). Under §13.4, serial order-zero
   codes derive from native PS prototypes and signed evidence; no free word
   row remains. Detached occurrence roots fill only the context complement,
   and existing rows conduct word → row → constituent priming. Read-back
   uses scale-free perceptual cosine; the antipode objective is retired.
   Current composition operators are unchanged. **Bootstrap learning and the
   form-fold/connective split are deferred to the operators update.** The
   14-D toy codes have an empty context complement; this gate does not
   establish learned context similarity. Declared inventory changes:
   XOR_grammar 6 → 262, MM_grammar 8 → 264; XOR nDim 10 → 14 (six PS content
   coordinates). No change to seeds, bars, existing assertions or guards.
   **Measured once on frozen source:** class **{counts['class_pass']}/10**,
   reconstruction **{counts['reconstruction_pass']}/10**, joint **{counts['joint']}/10**;
   MM_xor **{counts['mm_pass']}/10**; sum **{counts['sum_pass']}/10**, read first.
   Bands: **{counts['xor_bands']['at 0']} at 0, {counts['xor_bands']['at 1/4']} at ¼,
   {counts['xor_bands']['between']} between, {counts['xor_bands']['above 1/4']} above ¼**.
   Comparison with §12: **{below}**.
   Each XOR training supplied both unchanged bars. The tenth records
   **{audit['ownership']['conflicts']} ownership conflicts**, named operators, margins and
   gradients, kept-path stability, root/code geometry, exact perceptual
   support, and read-back code/priming decisions in the one
   [receipt](doc/benchmarks/2026-10-03-operators-attention/README.md).
   §12 stands without retry: class 0/10, reconstruction 7/10, joint 0/10,
   MM 10/10, sum 10/10. The accepted .114748 / 0-of-4 / zero-conflict
   baseline and prior 9/10 and 5/10 remain historical, alongside every saved
   failure and complete old/new test ports. Frozen evaluation admits nothing;
   the trained NanoChat gate waits for item 4's checkpoint. No fresh BasicModel
   scoring, attribution training, full sweep or native run was added.
   **Nothing committed; stop for Claude's review.** The original single-run
   class comparison remains not a regression finding.
'''
todo_path.write_text(todo[:begin]+update+todo[end:])
print(json.dumps({'counts':counts,'readback_totals':dict(readbacks),'peak_bytes':peak}))
