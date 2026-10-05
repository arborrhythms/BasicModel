"""Update the requested result entries only; leave the design documents intact."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parents[2]
read = lambda name: json.loads((H / name).read_text())


def main():
    s = read('review22-measurements/summary.json')
    a = read('review22-measurements/audit-summary.json')
    v = read('review22-results-validation.json')
    sweep = read('review22-final-sweep/summary.json')
    c, q, ch = s['counts'], sweep['counts'], a['chooser']
    assert s['complete']['completed'] and sweep['exit_code'] == 0
    max_gradient = max(row['maximum'] for row in s['xor'][-1]['run_audit']['sentence_gradients'].values())
    collisions = sum(bool(row['exact_form_collisions']['start'] or row['exact_form_collisions']['end']) for row in v['runs'])
    b = c['xor_bands']
    path = ROOT / 'doc/plans/2026-09-29-item-6-9-xor-grammar.md'
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith('| **Unit-sphere codes**'):
            old = 'Neither face has a unit-sphere constraint.'
            assert old in line
            lines[i] = line.replace(old, old + ' 6.8 §22 normalizes a Gaussian admitted percept row to unit L2 only at initialization, before the [0,1] clamp; this is not a maintained norm constraint.')
        elif line.startswith('| **Magnitude = certainty**'):
            old = 'No constraint on code norms, reader normalization or antipode loss.'
            assert old in line
            lines[i] = line.replace(old, '§22 changes admitted-row initialization and widens the two grammar fixtures from six to fourteen content coordinates (event width 14 → 22); activation ownership and the kernels stay as §20. No maintained code-norm constraint, reader normalization or antipode loss.')
        elif line.startswith('| **Decoder exploration**'):
            lines[i] = (
                '| **Decoder exploration** (§26.3; implemented, §22 measured candidate) | '
                'The generate walk infers operations from the root (§25.1). | '
                'Greedy argmax alone did not leave STOP when the missing-word penalty supplied no untaken-transition credit. | '
                'Greedy and sampled one-departure walks are both costed before learning; only strictly lower reconstruction keeps explore. '
                'The generate straight-through path and eligibility mask remain. The forward compose chooser uses detached-cost '
                '`K·R·p(a_dep)·(C_explore−C_greedy)`; attention credit stays detached at the sentence handoff and its estimator is deferred to the operators update. '
                'The separate decomposition chooser learns the true shortlisted pair by cross-entropy outside that trial-cost comparison, with codes detached. '
                f'The green §22 sweep precedes one campaign: XOR class {c["class_pass"]}/10, reconstruction {c["reconstruction_pass"]}/10, joint {c["joint"]}/10; '
                f'MM_xor {c["mm_pass"]}/10 and sum {c["sum_pass"]}/10. '
                f'The tenth-run audit records {a["ownership"]["conflicts"]} ownership conflicts, sentence-path prototype/evidence gradient maximum {max_gradient:.6g}, '
                f'and {ch["nonzero_advantage_sentences"]} nonzero advantages out of {ch["sentences"]}. '
                'The §16.3 candidate is committed/tagged; subsequent work is uncommitted and under review, nothing pushed. '
                'See the [receipt](../benchmarks/2026-10-03-operators-attention/README.md). |')
    path.write_text('\n'.join(lines) + '\n')

    path = ROOT / 'todo.md'
    text = path.read_text()
    start = text.index('   **October 5 §20 measured candidate; review pending:**')
    end = text.index('   **Item-7 reading residue:**', start)
    replacement = f'''   **October 5 §22 measured candidate; review pending:** main remains at
   `eb1fbefb`, tagged `6.8-s16.3-candidate`. Subsequent work is **uncommitted;
   nothing pushed; no parent submodule bump**. One working tree.
   **Two declared changes:** admitted percept rows use a Gaussian row,
   unit-L2 normalization, then the [0,1] clamp. The two grammar fixtures have
   a declared capacity increase: **nDim 14 → 22**, content **6 → 14** after
   the eight address coordinates, in IS/PS/CS/WS. Concept rows stay **6 / 8**;
   MM_xor's configuration is unchanged. The byte-fallback initializer stays
   as §18. The §20 kernels still take magnitude from activation; the reader
   remains affine on the raw root. Attention credit stays detached at the
   sentence handoff; its estimator is the operators update's. The forward
   score-function K·R term, decomposition chooser, pair search and room rule
   are unchanged.
   **Green full sweep:** **{sweep['selected']} cases; {q.get('passed',0)} passed,
   {q.get('skipped',0)} skipped, {q.get('xpassed',0)} non-strict XPASS,
   {q.get('xfailed',0)} XFAIL, zero failed**, ten workers. Two assertions for
   the old six-coordinate content width were ported, with complete old/new
   bodies and failures saved; no runtime regression repair was needed.
   The final focused probe retained both affected files (21/21 passed).
   Seeds, bars, budgets, optimizers and guards are unchanged.
   **Once-only frozen campaign:** sum **{c['sum_pass']}/10**, read first; XOR
   class **{c['class_pass']}/10**, reconstruction **{c['reconstruction_pass']}/10**,
   joint **{c['joint']}/10** from ten shared trainings; MM_xor **{c['mm_pass']}/10**.
   Bands: **{b['at 0']} at 0, {b['at 1/4']} at ¼, {b['between']} between,
   {b['above 1/4']} above ¼**. No retries or tuning; acceptance waits for review.
   Exact form collisions occurred in **{collisions}/10** runs at start or end.
   Historical class/reconstruction/joint counts: §12 **0/7/0**, §13 **0/0/0**,
   §14 **1/8/1** (pre-addendum CS 262/264), §17 **1/10/1**, §18 **0/10/0**,
   §20 **2/8/2**; MM and sum were **10/10**. §§15–16.3 and §19 supplied no
   gate measurements. Accepted MSE **.114748**, reconstruction **0/4**, zero
   conflicts and prior **9/10 / 5/10** remain historical. The declared
   capacity difference and independent random runs preclude attributing a
   count change to one of this round's two changes.
   **Audits:** {ch['sentences']} tenth-run sentence/step records,
   {ch['departures']} departures, {ch['nonzero_advantage_sentences']} nonzero advantages;
   K·R·ΔC·∇p error {ch['maximum_gradient_error']:.6g}, finite-difference error
   {ch['maximum_finite_difference_error']:.6g}. Prototype/evidence sentence gradient
   maximum {max_gradient:.6g}; ownership conflicts {a['ownership']['conflicts']}.
   The [receipt](doc/benchmarks/2026-10-03-operators-attention/README.md) retains
   starting form/root norms, named operators, per-word read-backs, geometry,
   exact collisions, support, d ranges, room, reader trajectories, decomposition
   weights/pick rates, decoder margins and kept paths. Empty same-context
   meanings correctly coincide. Complement bootstrap and the carried operators
   work remain deferred. Frozen evaluation admits nothing; the trained NanoChat
   gate waits for item 4. No fresh bulk scoring or native benchmark.
   **Claude reviews before any further commit; any push waits for Alec's word.**
'''
    path.write_text(text[:start] + replacement + text[end:])
    print('Updated todo 6.8 and the 6.9 §20.3 rows from saved §22 results.')


if __name__ == '__main__':
    main()
