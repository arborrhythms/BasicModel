"""Post-measurement receipt corrections only; no model, RNG or test execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as bounded


def read(path):return json.loads(path.read_text())


def node_outcomes(result):
    reports={node:[] for node in result['selected']}
    for worker in result['workers']:
        for row in worker.get('reports',[]):
            if row['nodeid'] in reports and (row['phase']=='call' or row['outcome'] in ('failed','skipped')):
                reports[row['nodeid']].append(row['outcome'])
    assert all(reports.values())
    return {node:values[-1] for node,values in reports.items()}


def main():
    summary=read(HERE/'review-summary.json')
    full=read(HERE/'final-sweep/result.json')
    thought=read(HERE/'thinking-gate/result.json')
    source=bounded.source_snapshot(ROOT)
    assert source==read(HERE/'measured-source/source.json')
    assert source==read(HERE/'final-sweep/source-manifest.json')['validated_source']
    assert source==read(HERE/'thinking-gate/source-manifest.json')['validated_source']
    assert source==read(HERE/'measurements/source.json')
    assert read(HERE/'measurements/complete.json')['source_matched']
    old_counts=summary['full']['outcomes']
    summary['full']['outcomes']=dict(Counter(node_outcomes(full).values()))
    cases=node_outcomes(thought)
    groups={}
    for name,predicate in (
        ('new_62',lambda node:node.startswith('test/test_item6_2_thinking.py')),
        ('native_answer',lambda node:node.startswith('test/test_unified_thought_controller.py')),
        ('mm_smoke',lambda node:node.startswith('test/test_reasoning_cde_model.py')),
        ('11c',lambda node:node.startswith(('test/test_dual_towers.py','test/test_frozen_concepts.py','test/test_priming_energy.py')))):
        groups[name]=dict(Counter(value for node,value in cases.items() if predicate(node)))
    summary['thinking']['groups']=groups
    summary['status']='review hold: thinking gate failed; not ready to land; no commit'
    summary['blocking_findings']='review-blockers.md'
    summary['postprocessing']={
        'unique_node_counts':'Repeated report entries from worker fixture lifecycle are collapsed by the selected node ID; no test is rerun.',
        'thinking_launcher':'The frozen launcher raised a formatting TypeError after all 47 cases completed; result.json remains authoritative.',
        'source_changed_after_measurement':False,'training_retries':0}
    (HERE/'review-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    receipt=(HERE/'README.md').read_text()
    receipt=receipt.replace('Status: delivered for Claude\'s review. **No commit, push or parent bump.**',
        'Status: **review hold; not ready to land**. The default sweep is green, but the explicit thinking gate failed. **No commit, push or parent bump.**\n\nThe [review blockers](review-blockers.md) give the exact failures and the\nconstruction-only diagnosis. No failed training or gate was retried.')
    counts=summary['full']['outcomes']
    receipt=receipt.replace(str(old_counts),f"{counts.get('passed',0):,} passed, {counts.get('skipped',0)} existing skips, {counts.get('xpassed',0)} non-strict XPASS")
    receipt=receipt.replace('## Review artifacts',
        'The explicit thinking result breaks down as 25/25 new certificates, 1/1\nnative answer integration, 1/2 MM configuration/smoke checks, and 14/19\n11c checks. These failures are retained in [the review blockers](review-blockers.md).\n\n## Review artifacts')
    (HERE/'README.md').write_text(receipt)
    p=HERE/'review-blockers.md';text=p.read_text().replace('none with a nonzero policy-gradient term before','none carrying a policy gradient before')
    text=text.replace('The thirty standing trainings are completing under their original assertions\nand configuration files. Their complete outcomes, including any misses, belong\nto the main receipt.',
        'The thirty standing trainings have completed under their original assertions\nand configuration files. Their complete outcomes, including any misses, are\nin the main receipt.')
    p.write_text(text)
    p=ROOT/'todo.md';text=p.read_text().replace('is an uncommitted implementation candidate under',
        'is an uncommitted candidate on review hold under');p.write_text(text)
    docmap=read(HERE/'measured-source/review-documents.json')
    extra=['review-blockers.md','diagnose_constituent.py','constituent-diagnostic.json',
        'thinking-failures.json','finalize_review_hold.py','review-summary.json']
    docs=sorted({ROOT/name for name in docmap} | {HERE/name for name in extra})
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in docs}
    (HERE/'measured-source/review-documents.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(HERE/'measured-source/review-documents.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in docs:z.write(p,p.relative_to(ROOT))
    (HERE/'post-measurement-diagnostics.json').write_text(json.dumps({name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in extra},indent=2)+'\n')
    print(json.dumps({key:summary[key] for key in ('status','full','standing','standing_episodes','thinking')}))

if __name__=='__main__':main()
