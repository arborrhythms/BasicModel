"""Receipt for the exact §19 initializer with a failing unchanged sweep."""
import json
from pathlib import Path
from review17_write_receipt import table,f
H=Path(__file__).resolve().parent

def read(name):return json.loads((H/name).read_text())
def main():
    s=read('review19-sweep/summary.json');n=read('review19-start-norms.json');c=s['counts']
    assert s['exit_code']!=0 and s['selected']==s['completed']
    text=[f'''# Decoder, operators and 6.8 — §19 admission initialization, held at the sweep

2026-10-05. The requested change is made: **only `RadixLayer.insert()`'s default admitted-row initializer** changes from `normal_(0, .02)` to a Gaussian row normalized by its maximum absolute coordinate and clamped to `[0, 1]`. Explicit supplied initializers are untouched. The §18 fallback-table change remains as delivered. **New work is uncommitted; nothing was pushed.** `main` remains at `eb1fbefb5f4a33a22cbb4a590cc0927d6a60761d`, tag `6.8-s16.3-candidate`; the parent WikiOracle index remains `802abb1a`. One working tree; capacities remain 6 / 8.

The single full sweep finished: **{s['selected']:,} completed, {c.get('passed',0):,} passed, {c.get('failed',0)} failed, {c.get('skipped',0)} skipped, {c.get('xpassed',0)} non-strict XPASS, {c.get('xfailed',0)} XFAIL**, in **{s['elapsed_seconds']:.1f} seconds**. **It is not green. The thirty-training campaign was not started**, because the campaign requires a green sweep of the exact delivered source. The user's “no other change” constraint is preserved: no follow-on production repair, seed change, assertion port, weakened bar or retry was used to clear this hold.

## Exact change and checks

```python
row = master.data[new_id, :].normal_(mean=0.0, std=1.0)
row.div_(row.abs().amax().clamp(min=1e-8)).clamp_(0.0, 1.0)
```

The [one-function diff](review19-only-change.patch) and [complete before/after body and checks](review19-change-verification.json) verify that every other runtime file and function matches the delivered §18 source. The check uses widths 6 and 8, replays the ambient RNG with a private generator without setting a seed, and verifies the exact rule, the same one-row random draw, unchanged other rows, unchanged Parameter identity/ownership, duplicate insertion as a no-op, and byte-identical handling of an explicit initializer. It makes no model or optimizer step. No test file changed.

## Starting form and root norms

The requested scale is now present in the admitted form codes. These numbers come from **one forward-only XOR_grammar probe**, at the first ordinary greedy reconstruction boundary, before any reconstruction cost, backward or optimizer step. The probe stops at that boundary. It is **not a gate training or a population estimate**; there were zero training runs and zero optimizer steps. No seed was specified. Norms are Euclidean (L2), over form content coordinates and the full recorded root respectively.
''']
    text.append(table(['Word','Starting form L2','Maximum absolute coordinate'],[[r['word'],f(r['l2']),f(r['max_abs'])] for r in n['forms']]))
    text.append(table(['Sentence','Starting root L2','Maximum absolute coordinate'],[[r['sentence'],f(r['l2']),f(r['max_abs'])] for r in n['roots']]))
    text.append(f"Form norms range **{f(min(r['l2'] for r in n['forms']))}–{f(max(r['l2'] for r in n['forms']))}**; root norms **{f(min(r['l2'] for r in n['roots']))}–{f(max(r['l2'] for r in n['roots']))}**. The [saved initial geometry](review19-start-norms.json) includes raw vectors, pairwise cosines, support, root singular values, unit-root XOR interaction, evidence ranges and room violations. The [probe request](review19-norm-probe/request.json), [log](review19-norm-probe/run.log), [bounded process result](review19-norm-probe/process.json) and [observer source](review19_start_norms.py) make the zero-step boundary reviewable. This probe's file list contains only that observer; it does not replace any test in the full sweep.")
    text.append('''## Sweep failures, preserved without repairs

The following failures are against unchanged assertions. None is an assertion of the old `.02` row magnitude, so none has been declared a superseded-design port. The observed failures are recorded here; the exact gradient path and the identity of the extra conceptual row require a separate diagnosis before deciding a repair or any justified test port.
''')
    for r in s['failures']:
        text.append(f"### `{r['nodeid']}`\n\n```text\n{r['message']}\n```")
    text.append('''The tie fixture explicitly replaces the two reconstruction costs with equal values, yet compose-chooser parameters change. The distinct fixture supplies an explore advantage of −.5; its chooser-direction checks pass, but the generate-policy parameter-movement assertion fails. The MM_grammar inventory case finds an extra conceptual row beyond the four input-word rows; capacity itself remains 8. The failure alone does not establish whether that extra row is a letter, a word or another concept.

The [full sweep summary](review19-sweep/summary.json), [HTML report](review19-sweep/report.html), collection, all worker requests, logs and source hashes are retained. [Failure observations before any repair](review19-observed-failures-before-any-repair.json) and the [complete affected test files](review19-failed-test-bodies.json) are saved. No failing test was retried to obtain a pass. The full sweep includes every §18 selector plus the link check for the archived incoming receipt; nothing was dropped. All sixteen original output-gradient regression assertions retain their original bodies; their outcomes are in the summary.

## Measurement status and prior observations

The same [campaign](review19_campaign.py) is prepared with unchanged training observers, bars, budgets, seeds and guards, but **not executed**. Its only changes from §18 are receipt paths and the comparison row; the [diff](review19-campaign-changes.patch) is saved. Thus §19 has no sum, XOR or MM_xor training result, no tenth-run training audit and no class-gate conclusion. The required green-sweep precondition is the blocker. Earlier observations stand as measured:
''')
    text.append(table(['Record','CS XOR / MM','Class','Reconstruction','Joint','MM_xor','Sum'],[
        ['6.9 closing','accepted source','MSE .1147481948','0/4 sentences','—','red through 6.9 §17','—'],
        ['§12','6 / 8','0/10','7/10','0/10','10/10','10/10'],
        ['§13','262 / 264','0/10','0/10','0/10','10/10','10/10'],
        ['§14 before addendum','262 / 264','1/10','8/10','1/10','10/10','10/10'],
        ['§17','6 / 8','1/10','10/10','1/10','10/10','10/10'],
        ['§18','6 / 8','0/10','10/10','0/10','10/10','10/10'],
        ['§19','6 / 8','not run','not run','not run','not run','not run']]))
    text.append('''The accepted closing record also retains zero ownership conflicts and the prior class 9/10 and reconstruction 5/10. These prior measurements are not attributed to this changed source.

## Review hold

The [frozen source](review19-source/source.zip), [measurement-helper hashes](review19-source/measurement-helpers.json), [preservation check](review19-preservation.json) and [delivery manifest](review19-delivery/bridge.json) bind this exact initializer to the failing sweep and the initial-norm observation. The [incoming §18 receipt](README-before-review19.md) and all earlier evidence remain intact. Architecture and plan edits already present when this round began are preserved; no design documents, todo, XMLs or tests are changed by this round.

**Stopped for review with the initializer uncommitted, the sweep red and the campaign unrun. Nothing pushed.** Completing the requested green sweep and thirty trainings requires a decision on the newly exposed failures; this receipt does not treat them as passed or waive the campaign guard.
''')
    (H/'README.md').write_text('\n'.join(line.rstrip() for line in '\n\n'.join(text).split('\n')))
    print('Wrote the held §19 receipt without gate claims.')

if __name__=='__main__':main()
