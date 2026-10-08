"""Summarize saved repair outcomes without running a model or drawing RNG."""
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PRIOR=HERE.parent/'2026-10-07-item6-2'
sys.path[:0]=[str(HERE),str(ROOT/'test')]
import bounded_tests as bounded


def read(path):return json.loads(path.read_text())


def outcomes(result):
    reports={node:[] for node in result['selected']}
    for worker in result['workers']:
        for row in worker.get('reports',[]):
            if row['nodeid'] in reports and (row['phase']=='call' or row['outcome'] in ('failed','skipped')):
                reports[row['nodeid']].append(row['outcome'])
    assert all(reports.values()), 'incomplete selected-node reports'
    return {node:('failed' if 'failed' in values else values[-1]) for node,values in reports.items()}


def main():
    from verification import validate
    source=bounded.source_snapshot(ROOT);full=validate(source)
    helpers=read(HERE/'measured-source/measurement-helpers.json')
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in helpers.items())
    first=read(HERE/'first-receipt-hashes.json')
    now={str(p.relative_to(PRIOR)):hashlib.sha256(p.read_bytes()).hexdigest() for p in PRIOR.rglob('*') if p.is_file()}
    assert first==now, 'first receipt changed'
    thinking=read(HERE/'thinking-gate/result.json')
    assert sorted(thinking['completed'])==sorted(thinking['selected'])
    assert read(HERE/'thinking-gate/source-manifest.json')['validated_source']==source
    standing=read(HERE/'measurements/summary.json')
    assert standing['complete']['completed'] and standing['complete']['source_matched']
    assert read(HERE/'measurements/source.json')==source
    mm=read(HERE/'mm-query-configured/outcome.json')
    assert read(HERE/'mm-query-configured/plan.json')['source']==source
    cases=outcomes(thinking)
    groups={name:dict(Counter(value for node,value in cases.items() if node.startswith(prefix)))
            for name,prefix in (
                ('11c',('test/test_dual_towers.py','test/test_frozen_concepts.py','test/test_priming_energy.py')),
                ('mm',('test/test_reasoning_cde_model.py',)),
                ('per_row',('test/test_unified_thought_controller.py',)),
                ('original_certificates',('test/test_item6_2_thinking.py',)),
                ('ownership_certificates',('test/test_item6_2_repair.py',)))}
    count=standing['counts'];episodes=sum(row['open_episodes'] for row in standing['thinking'])
    checks_green=(thinking['exit_code']==0 and mm['outcome']=='completed'
                  and mm.get('training_epochs')==300 and episodes==0
                  and all(count[name]==10 for name in ('class_pass','reconstruction_pass','joint','sum_pass','mm_pass')))
    moved=mm.get('chooser_initial_to_final_l2',0.)>0
    credit=mm.get('credit_summary',{})
    status=('repair measurements green; awaiting Claude review' if checks_green
            else 'repair candidate held; measurement failures require Claude review')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert head=='631d9e8e44b7c8034263e22b36dc74fa5df4eb75'
    summary=dict(status=status,base_commit=head,first_receipt_unchanged=True,
        full=dict(selected=len(full['selected']),completed=len(full['completed']),outcomes=dict(Counter(outcomes(full).values()))),
        thinking=dict(selected=len(cases),outcomes=dict(Counter(cases.values())),groups=groups),
        standing=count,standing_episodes=episodes,configured_mm=mm,
        shared_chooser_moved=moved,checks_green=checks_green,
        seed=None,retries=0,replacements=0,source_matched=True,helpers_matched=True,
        learning_claim=False,commit=False)
    (HERE/'review-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    old=read(HERE/'baseline-source.json')
    with zipfile.ZipFile(HERE/'baseline-source.zip') as archive:
        diff=[]
        for name in sorted(set(old)|set(source)):
            if old.get(name)==source.get(name):continue
            before=archive.read(name).decode().splitlines(True) if name in old else []
            after=(ROOT/name).read_text().splitlines(True) if name in source else []
            diff.extend(difflib.unified_diff(before,after,fromfile='measured-6.2/'+name,tofile='repair/'+name))
    (HERE/'repair-source.patch').write_text(''.join(diff))
    counts=summary['full']['outcomes'];thought=summary['thinking']['outcomes']
    movement=mm.get('chooser_initial_to_final_l2')
    final=f'''# Item 6.2 repair pass — October 7, 2026

Status: **{status}**. Stop before any commit, push or parent bump.

This starts from the [held 6.2 candidate](../2026-10-07-item6-2/README.md),
source `d6d00f6640af051a4c84599ef5a5a235cb25faa9691b873b855699621cf80b62`,
following [Claude's review, spec §9](../../specs/2026-10-07-thinking.md).
The first receipt remains byte-for-byte intact ({len(first)} files checked).

## Repairs

Fill now carries a supplied meaning's constituent graph into the goal's
local table and rebases its transferred addresses, including binding by a
stored occurrence. Each nested child retains its own local table; shared
children remain shared. Fill validates the goal, supplied graph and result,
including local references in semantic metadata. Malformed or dangling
addresses raise at fill, before indexing or writing. Tensor gradients survive.

The exact saved construction fails before the change and succeeds after it,
writing the child and conclusion: [before](construction-before.json),
[after](construction-after.json). Twelve ownership cases cover empty and
nonempty goals, nested graphs, shared children, occurrence binding, malformed
addresses, nested metadata and gradients.

Three 11c tests now use the resolved 104-wide identity event, current
universe/neutral-property routing and native percept IDs in the live top-K
consumption window. Two tests were retired because their word-row reading
projection and `_primed_reading_step` contracts were removed. Their exact old
texts and reasons are in [the retirement record](test-retirements.json);
[all port descriptions](test-ports.json) and complete old test files are saved.
The retained 11c helper no longer sets seeds. All seventeen surviving nodes
remain in the explicit gate.

The MM routing fix loads the configured grammar before constructing its
predictors. Their routing width now matches this model's rule vocabulary,
including after another grammar was loaded. The optimizer smoke keeps its
training call; the configuration check verifies the widths. The thinking
launcher reads `result.json` from the returned report path.

## Measurements on one frozen source

| Check | Result |
| --- | --- |
| Default complete sweep | {len(full['completed'])}/{len(full['selected'])}; {counts.get('passed',0)} passed, {counts.get('skipped',0)} existing skips, {counts.get('xpassed',0)} non-strict XPASS |
| Explicit thinking gate | {thought.get('passed',0)} passed, {thought.get('failed',0)} failed; {len(cases)} selected |
| Original 25 certificates | {groups['original_certificates']} |
| Twelve ownership certificates | {groups['ownership_certificates']} |
| Ported 11c nodes | {groups['11c']} |
| MM configuration / optimizer smoke | {groups['mm']} |
| Per-row answer integration | {groups['per_row']} |
| XOR class / reconstruction / both, shared ten trainings | {count['class_pass']}/10 / {count['reconstruction_pass']}/10 / {count['joint']}/10 |
| Sum control / MM_xor, ten each | {count['sum_pass']}/10 / {count['mm_pass']}/10 |
| Thought episodes in the standing thirty | {episodes} |
| Unforced configured MM_query_reasoning | {mm['outcome']}; {mm.get('training_epochs',0)}/300 epochs |

The configured run has {credit.get('observations',0)} recorded credit
observations, {credit.get('nonzero_chooser_gradients',0)} with a nonzero raw
gradient into the shared operation scorer, and {credit.get('exact_cost_ties',0)}
exact cost ties. Its scorer's initial-to-final parameter change has L2 norm
`{movement}`. [The full outcome](mm-query-configured/outcome.json) reports
operation chains, cost pairs, credit sources, gradient norms and per-epoch
parameter movement; initial and final scorer tensors are saved beside it.
{('The recorded failure is `'+mm.get('error','unknown')+'`.' if mm['outcome']!='completed' else '')}

The movement is measured during the actual run, without reseeding, replaying
training or forcing a reading. Other owner losses also update the shared
scorer, so total movement alone does not establish learned chaining. The
focused answer and expectation certificates test its credit direction.

The four standing gate/config files match 6.5 exactly. All thirty trainings
were attempted once; no failed outcome was replaced. The sum control is read
before XOR starts. Resource limits and the independent MM/thinking/standing
schedule are in [the predeclared protocol](protocol.json).

## Review evidence

- [Machine-readable review summary](review-summary.json), [standing measurements](measurements/summary.json), [thinking result](thinking-gate/result.json) and [full sweep](final-sweep/result.json).
- [Only the repair's source changes](repair-source.patch); complete source and measurement-helper archives under `measured-source/`.
- [Frozen hashes](measured-source/freeze.json), [initial first-receipt hashes](first-receipt-hashes.json) and [thinking selection](thinking-gate-plan.json).
- `affected-01` records the development fixture's two-row/four-row mismatch; `affected-02` passes its corrected native read. `affected-03` passes all twelve ownership checks.
- `development-sweep-1` preserves the first green source before the final occurrence-binding case. The final sweep matches the delivered source. No configured training or measured gate had run before that change.
- [Premeasurement launcher correction](launcher-development.json); the measured thinking launcher has no post-report formatting failure.

The 6.5 learning gates — held-out anaphora, verb reuse, prediction control,
shuffled order, renamed vocabulary and determiner control, seeds 0/1/2 —
remain pending the million-sentence checkpoint and are not claimed.
'''
    (HERE/'README.md').write_text(final)
    p=ROOT/'todo.md';text=p.read_text().replace(
        'is an uncommitted candidate on review hold under',
        'is an uncommitted candidate after the [repair pass](doc/benchmarks/2026-10-07-item6-2-repair/README.md), awaiting Claude\'s review under')
    p.write_text(text)
    p=ROOT/'doc/ThoughtHistory.md';text=p.read_text().replace(
        "The candidate's source-matched measurements are in the [6.2 receipt](benchmarks/2026-10-07-item6-2/README.md); Claude's review precedes any commit.",
        "The [held 6.2 receipt](benchmarks/2026-10-07-item6-2/README.md) remains intact. The [repair receipt](benchmarks/2026-10-07-item6-2-repair/README.md) records constituent ownership, ported contracts and new measurements; Claude's review precedes any commit.")
    p.write_text(text)
    docs=[ROOT/'todo.md',ROOT/'doc/ThoughtHistory.md',ROOT/'doc/specs/2026-10-07-thinking.md',
          ROOT/'doc/plans/2026-10-07-thought-loop.md',HERE/'README.md',HERE/'review-summary.json',
          HERE/'test-retirements.json',HERE/'test-ports.json',HERE/'repair-source.patch',HERE/'finalize.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in docs}
    (HERE/'measured-source/review-documents.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(HERE/'measured-source/review-documents.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in docs:archive.write(p,p.relative_to(ROOT))
    print(json.dumps({key:summary[key] for key in ('status','full','thinking','standing','standing_episodes','shared_chooser_moved')}))


if __name__=='__main__':main()
