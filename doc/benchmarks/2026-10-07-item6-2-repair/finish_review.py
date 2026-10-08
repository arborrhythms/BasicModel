"""Attach post-measurement findings; never change measured source or rerun work."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def read(path):return json.loads(path.read_text())


def main():
    summary=read(HERE/'review-summary.json')
    gate=read(HERE/'thinking-gate/result.json')
    failures=[row for worker in gate['workers'] for row in worker.get('reports',[])
              if row['outcome']=='failed']
    (HERE/'thinking-failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    summary['review_notes']='review-notes.md'
    summary['per_row_unseeded_integration_completed']=False
    summary['configured_thought_credit_moved_chooser']=False
    summary['measurement_limitations']=[
        'The per-row integration fixture attempted a nested fixed seed. The guard rejected it before model construction; its unseeded credit assertion was not reached. No retry.',
        'The unforced 300-epoch run had one episode and one exact cost tie, with zero nonzero thought-credit gradients and no two-query chain. Shared reduce/apply anchors moved; the thought anchor did not.']
    (HERE/'review-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    path=HERE/'README.md';text=path.read_text()
    marker='## Repairs'
    note='''The [remaining review issues](review-notes.md) distinguish a fixture failure
from the measured training outcome. The thinking gate is **56/57**, because a
nested helper's seed call was rejected before the per-row integration could
construct its model. The configured run completed **300/300 epochs**, but its
single thought-credit comparison was an exact tie. It demonstrates completion,
not learned chaining. The repairs and all other selected checks passed.

'''
    assert marker in text
    if note not in text:text=text.replace(marker,note+marker,1)
    path.write_text(text)
    mapping=read(HERE/'measured-source/review-documents.json')
    extra=['review-notes.md','thinking-failures.json','finish_review.py',
           'mm-query-configured/movement-analysis.json']
    paths=sorted({ROOT/name for name in mapping}|{HERE/name for name in extra})
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (HERE/'measured-source/review-documents.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(HERE/'measured-source/review-documents.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in paths:archive.write(p,p.relative_to(ROOT))
    (HERE/'post-measurement-findings.json').write_text(json.dumps(
        {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in extra},indent=2)+'\n')
    print(json.dumps({'status':summary['status'],'thinking':summary['thinking'],
                      'MM_epochs':summary['configured_mm']['training_epochs']}))


if __name__=='__main__':main()
