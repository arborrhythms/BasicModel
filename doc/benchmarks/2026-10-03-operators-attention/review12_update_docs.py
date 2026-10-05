"""Reconcile completed §12 measurements into the existing repository documents."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
s=json.loads((HERE/'review12-measurements/summary.json').read_text())
a=json.loads((HERE/'review12-measurements/audit-summary.json').read_text())
c=s['counts']; bands=c['xor_bands']; stability=a['walks']['generate.decoder']['derivation_stability']
below=', '.join(f"{r['count']} {r['current']}/10 versus {r['prior']}/10" for r in s['below_comparison']) or 'none'
paragraph=f'''   **October 4 §12 round, held for Claude:** the XOR table is the
   **composition mechanism gate**, under plan §12.1. Disjunction now combines
   magnitudes by the probabilistic sum and normalizes `x+y-x*y`; mean stays
   catalogued as `sum`, the control. Rule names accompany IDs in derivation
   audits, and every XOR run saves its final greedy compose derivations.
   **Measured once on frozen source:** class **{c['class_pass']}/10**,
   reconstruction **{c['reconstruction_pass']}/10**, joint **{c['joint']}/10**;
   MM_xor **{c['mm_pass']}/10**; sum **{c['sum_pass']}/10**, read before the gates.
   Bands: **{bands['at 0']} at 0, {bands['at 1/4']} at ¼, {bands['between']} between,
   {bands['above 1/4']} above ¼**. Below §11 comparison: **{below}**.
   The final XOR training records **{a['ownership']['conflicts']} ownership conflicts**
   and decoder kept-path stability **{stability:.6f}**, with full raw margins,
   gradients and fixed-parent changes in the one
   [receipt](doc/benchmarks/2026-10-03-operators-attention/README.md).
   **§11's results stand without retry:** class 2/10, reconstruction 6/10,
   joint 2/10, MM 10/10, sum 10/10. From its saved observations, only run 10's
   final operator is recoverable: mean on all four inputs, at .250079. The
   claim that all five quarter-band runs chose mean remains unconfirmed;
   the other nine runs lack saved final compose derivations. Run 3's .232321
   is in the quarter band, not exactly at the floor.
   Affine numeric reading, supported-undo eligibility, sampled exploration
   and one owner per weight remain. Frozen evaluation admits nothing; the
   trained NanoChat gate waits for item 4's checkpoint. The accepted
   .114748 / 0-of-4 / zero-conflict baseline and prior 9/10 and 5/10 counts
   remain historical, alongside all failures and complete old/new ports.
   No attribution training, full sweep or native run was added. Seeds, bars,
   existing assertions and guards are unchanged. **Nothing committed.**
   The original single-run class comparison remains not a regression finding.
'''
path=ROOT/'todo.md'; text=path.read_text()
start=text.index('   **October 4 §11 review:**')
end=text.index('   **Item-7 reading residue:**',start)
path.write_text(text[:start]+paragraph+text[end:])

path=ROOT/'doc/GradientFlow.md'; text=path.read_text()
anchor='a cause to the class failures. The candidate is uncommitted for Claude\'s review.'
assert text.count(anchor)==1
max_change=max(max(abs(v['fixed_parent_change']['min']),abs(v['fixed_parent_change']['max'])) for v in a['binary'].values())
update=f'''\n\nThe [§12 composition mechanism receipt](benchmarks/2026-10-03-operators-attention/README.md)
measures class **{c['class_pass']}/10**, reconstruction **{c['reconstruction_pass']}/10**,
joint **{c['joint']}/10**, MM_xor **{c['mm_pass']}/10**, and sum **{c['sum_pass']}/10**.
All ten sum controls were read before the gates started. Comparison is the
saved §11 round (2/10, 6/10, 2/10, 10/10, 10/10); below-comparison counts:
**{below}**. The final run records **{a['ownership']['conflicts']} ownership conflicts**
and decoder kept-path stability **{stability:.6f}**. Its first-step eligibility
counts are `{json.dumps(a['first_step_eligibility'])}`, and the largest absolute
fixed-parent margin change is **{max_change:.9g}**. Full named paths, margins and
gradients remain in the receipt. No attribution training or retry was added.
§11 remains as measured: only its tenth run saved final compose operators,
all mean disjunction; the other runs' operators cannot be recovered from their
scores. The five-run mean hypothesis is unconfirmed. This candidate remains
uncommitted for Claude's review.
'''
path.write_text(text.replace(anchor,anchor+update))
print('Updated todo 6.8 and GradientFlow from the completed §12 data.')
