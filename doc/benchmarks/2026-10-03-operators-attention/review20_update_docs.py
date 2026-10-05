"""Update only the requested result entries from the saved §20 campaign."""
from pathlib import Path
import json

H = Path(__file__).resolve().parent
ROOT = H.parents[2]


def read(name):
    return json.loads((H / name).read_text())


def main():
    summary = read('review20-measurements/summary.json')
    audit = read('review20-measurements/audit-summary.json')
    sweep = read('review20-sweep/summary.json')
    validation = read('review20-results-validation.json')
    assert summary['complete']['completed'] and sweep['exit_code'] == 0
    c, q = summary['counts'], sweep['counts']
    chooser = audit['chooser']
    gradients = summary['xor'][-1]['run_audit']['sentence_gradients']
    maximum_gradient = max(row['maximum'] for row in gradients.values())
    bands = c['xor_bands']
    status = 'Class remains red.' if c['class_pass'] < 10 else 'Both XOR bars passed all ten runs.'
    if c['reconstruction_pass'] < 10:
        status += ' Reconstruction is below the 10/10 measured in §§17–18.'

    path = ROOT / 'doc/GradientFlow.md'
    text = path.read_text()
    old = '''Disjunction now combines magnitudes by `a+b-a*b` and identity by
`unit(x+y-x*y)`; its free inverse searches through that same kernel. The mean
remains `sum`, the additive control.'''
    new = '''The binding kernels take magnitude from the operand activations, not the
lengths of their form codes (§20). With `u=unit(x)`, `v=unit(y)` and activation
magnitudes `a,b`, conjunction returns `a*b*unit(u*v)` and disjunction returns
`(a+b-a*b)*unit(u+v-u*v)`. A present native word has activation one regardless
of its form norm; a composed root carries activation in its norm. The free
inverse searches through the same kernels. The mean remains `sum`, the additive
control; both gates retain the raw-root affine reader.'''
    assert text.count(old) == 1
    text = text.replace(old, new)
    anchor = '''- **The compose chooser's gradient is the score-function estimator (Alec,'''
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''- **The sentence handoff detaches input attention's credit (§20).** The
  attention and compose walks share a scorer. Byte reconstruction previously
  reached it through the attention straight-through value, cached perception
  and the perception pullback even when the two compose costs tied. Detaching
  that credit closes this additional sentence-path writer; the original
  no-movement tie assertion passes. This repair does not add the separate
  attention score-function objective proposed in the later §21 review note.
''' + anchor)
    path.write_text(text)

    path = ROOT / 'doc/plans/2026-09-29-item-6-9-xor-grammar.md'
    text = path.read_text()
    old = 'Returned as the projection coefficient `(leaf·c)/(c·c)` onto the full-presence code `c`; the leaf is activation times that code. No unit-sphere constraint or antipode loss.'
    new = 'Returned as the projection coefficient `(leaf·c)/(c·c)` onto the full-presence code `c`; the leaf is activation times that code. In 6.8 §20 the binding kernels take these activations separately and compose code directions: root magnitude carries the resulting activation, not form length. No constraint on code norms, reader normalization or antipode loss.'
    assert text.count(old) == 1
    text = text.replace(old, new)
    lines = text.splitlines()
    matches = [i for i, line in enumerate(lines) if line.startswith('| **Decoder exploration**')]
    assert len(matches) == 1
    lines[matches[0]] = (
        '| **Decoder exploration** (§26.3; implemented, §20 measured candidate) | '
        'The generate walk infers operations from the root (§25.1). | '
        'Greedy argmax alone did not leave STOP when the missing-word penalty supplied no untaken-transition credit. | '
        'Greedy and sampled one-departure walks are both costed before learning; only strictly lower reconstruction keeps explore. '
        'The generate straight-through path and eligibility mask remain. The forward compose chooser uses detached-cost '
        '`K·R·p(a_dep)·(C_explore−C_greedy)`; §20 closes the additional path through input-attention credit. '
        'The separate decomposition chooser learns the true shortlisted pair by cross-entropy outside that trial-cost comparison, with codes detached. '
        f'The green §20 sweep precedes one campaign: XOR class {c["class_pass"]}/10, reconstruction {c["reconstruction_pass"]}/10, '
        f'joint {c["joint"]}/10; MM_xor {c["mm_pass"]}/10 and sum {c["sum_pass"]}/10. '
        f'The tenth-run audit records {audit["ownership"]["conflicts"]} ownership conflicts, sentence-path prototype/evidence gradient maximum '
        f'{maximum_gradient:.6g}, and {chooser["nonzero_advantage_sentences"]} nonzero advantages out of {chooser["sentences"]}. '
        'The §16.3 candidate is committed/tagged; subsequent work is uncommitted and under review, nothing pushed. '
        'See the [receipt](../benchmarks/2026-10-03-operators-attention/README.md). |')
    path.write_text('\n'.join(lines) + '\n')

    path = ROOT / 'todo.md'
    text = path.read_text()
    start = text.index('   **October 5 §17 measured candidate; review pending:**')
    end = text.index('   **Item-7 reading residue:**', start)
    replacement = f'''   **October 5 §20 measured candidate; review pending:** main remains at
   `eb1fbefb`, tagged `6.8-s16.3-candidate`. Subsequent work is **uncommitted;
   nothing pushed; no parent submodule bump**. One working tree, CS **6 / 8**.
   Both full-presence initializers (§§18–19) remain. The binding kernels now
   take magnitude from activation and identity from code direction. Native
   leaves supply their projection coefficients; composed roots carry the
   resulting activation as their norm. The raw-root affine reader and sum
   control are unchanged; the XOR table is the composition mechanism gate.
   **Three failures diagnosed before ports:** byte reconstruction reached the
   shared compose scorer through attention credit and the perception pullback;
   detaching the sentence handoff fixes it and retains the tie assertion.
   The saved distinct-case trace offered one action per live decoder step, so
   the movement assertion now applies where a live choice exists. The extra
   MM row was a native order-one clause-taxonomy symbol, not a letter or
   promoted chunk; the inventory check now distinguishes word admission from
   later higher-order rows. Complete old/new test files and failed probes are
   in the one [receipt](doc/benchmarks/2026-10-03-operators-attention/README.md).
   Claude's concurrent §21 note is preserved, with the diagnostic differences
   and the unimplemented attention score-function proposal stated explicitly.
   **Green full sweep:** {sweep['selected']} cases, **{q.get('passed',0)} passed,
   {q.get('skipped',0)} skipped, {q.get('xpassed',0)} non-strict XPASS,
   {q.get('xfailed',0)} XFAIL, zero failed**, ten workers. All sixteen original
   output-gradient assertions pass. No seeds, bars, budgets, optimizers,
   configurations or guards changed. The final focused list drops no files.
   **Once-only frozen campaign:** sum **{c['sum_pass']}/10**, read first; XOR
   class **{c['class_pass']}/10**, reconstruction **{c['reconstruction_pass']}/10**,
   joint **{c['joint']}/10** from ten shared trainings; MM_xor **{c['mm_pass']}/10**.
   Bands: **{bands['at 0']} at 0, {bands['at 1/4']} at ¼, {bands['between']} between,
   {bands['above 1/4']} above ¼**. {status} No retries or tuning.
   Historical class/reconstruction/joint counts: §12 **0/7/0**, §13 **0/0/0**,
   §14 **1/8/1** (pre-addendum CS 262/264), §17 **1/10/1**, §18 **0/10/0**;
   all had MM and sum **10/10**. §19 ran no gates after its red sweep.
   Accepted MSE **.114748**, reconstruction **0/4**, zero conflicts and prior
   **9/10 / 5/10** remain historical, without causal attribution.
   **Audits:** {chooser['sentences']} tenth-run sentence/step records,
   {chooser['departures']} departures, {chooser['nonzero_advantage_sentences']} nonzero
   advantages; K·R·ΔC·∇p error {chooser['maximum_gradient_error']:.6g}, finite-difference
   error {chooser['maximum_finite_difference_error']:.6g}; finite chooser logits:
   {validation['finite_chooser_logits']}. Prototype/evidence sentence gradient
   maximum {maximum_gradient:.6g}; ownership conflicts {audit['ownership']['conflicts']}.
   Per-run starting form/root norms, operators, per-word read-backs, geometry,
   support, d ranges, room, reader weights, margins and kept paths are saved.
   Empty same-context meanings correctly coincide. Complement bootstrap and
   the carried operators work remain deferred. Frozen evaluation admits
   nothing; the trained NanoChat gate waits for item 4. No fresh bulk BasicModel
   scoring or native benchmark. **Claude reviews before any further commit;
   any push waits for Alec's word.**
'''
    path.write_text(text[:start] + replacement + text[end:])
    print('Updated GradientFlow, the 6.9 catalogue rows and todo 6.8 from saved §20 results.')


if __name__ == '__main__':
    main()
