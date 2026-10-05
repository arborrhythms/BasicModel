"""Validate and summarize saved observations only; no model calls or RNG."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

H=Path(__file__).resolve().parent; ROOT=H.parents[2]; OUT=H/'review20-measurements'
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as bounded


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def describe(values):
    return dict(n=len(values),minimum=min(values),maximum=max(values),mean=statistics.mean(values))


def norm_rows(geometry):
    forms=[]
    for stage,book in enumerate(geometry['codes']):
        names={r['row']:r['word'] for r in book['support']}
        for row,values in zip(book['rows'],book['forms']['values'],strict=True):
            forms.append(dict(stage=stage,row=row,word=names[row],l2=math.hypot(*values),
                              max_abs=max(map(abs,values)),content_width=len(values)))
    roots=[dict(sentence=sentence,l2=math.hypot(*values),max_abs=max(map(abs,values)),width=len(values))
           for sentence,values in zip(geometry['root_inputs'],geometry['roots']['values'],strict=True)]
    return dict(forms=forms,roots=roots)


def norms():
    output=dict(definition='Euclidean (L2) norm of the saved raw form and root vectors; max-abs is also reported. No reader normalization and no extra forward.',
                observation='The first reconstruction trial of the first training batch, before an owner update. Forms use content coordinates; roots use the full recorded vector.',rounds={})
    for round_ in ('review17','review18','review20'):
        entries=[]
        for kind in ('xor','sum'):
            for run in range(1,11):
                a=read(H/f'{round_}-measurements/{kind}-{run:02}/run-audit.json')
                entries.append(dict(kind=kind,run=run,**norm_rows(a['start'])))
        totals={kind:{field:describe([v['l2'] for e in entries if e['kind']==kind for v in e[field]])
                      for field in ('forms','roots')} for kind in ('xor','sum')}
        output['rounds'][round_]=dict(runs=entries,aggregate=totals)
    write(OUT/'start-norms.json',output)
    return output


def main():
    src=read(H/'review20-source/source.json')
    assert src==bounded.source_snapshot(ROOT)==read(OUT/'source.json')
    assert src==read(H/'review20-sweep/source-manifest.json')['validated_source']
    helpers=read(H/'review20-source/measurement-helpers.json')
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in helpers.items())
    s=read(OUT/'summary.json'); a=read(OUT/'audit-summary.json')
    assert s['complete']['completed'] and len(s['complete']['jobs'])==30
    assert s['complete']['retries']==0
    runs=[]
    def collisions(book):
        names={row['row']:row['word'] for row in book['support']}
        vectors=list(zip(book['rows'],book['forms']['values'],strict=True))
        return [[names[left],names[right]]
                for i,(left,x) in enumerate(vectors)
                for right,y in vectors[i+1:] if x==y]
    for x in s['xor']:
        audit=x['run_audit']
        start=next(b for b in audit['start']['codes'] if b['rows'])
        end=next(b for b in audit['end']['codes'] if b['rows'])
        assert start['rows']==end['rows']
        delta=max(abs(v-w) for r,t in zip(start['values'],end['values'],strict=True) for v,w in zip(r,t,strict=True))
        runs.append(dict(run=x['run'],code_maximum_change=delta,
            mean_form_cosines=[start['forms']['mean_pairwise_cosine'],end['forms']['mean_pairwise_cosine']],
            support_words=[len(start['support']),len(end['support'])],
            exact_form_collisions=dict(start=collisions(start),end=collisions(end)),
            zero_sentence_path_gradient=all(g['maximum']==0 and g['nonzero']==0 for g in audit['sentence_gradients'].values()),
            reader_epochs=len(audit['reader_weights']),final_word_annotations=len(x['readback_decisions'])))
    ranges=a['chooser']['per_epoch_ranges']
    finite=all(math.isfinite(r['minimum']) and math.isfinite(r['maximum']) for r in ranges.values())
    groups=defaultdict(lambda:[Counter() for _ in range(4)])
    with (OUT/'xor-10/ownership/events.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            if row['kind']!='decoder_comparison':continue
            for i,(cost,win,dep,g,e) in enumerate(zip(row['costs'],row['wins'],row['departure'],row['greedy'],row['explore'],strict=True)):
                assert win==(dep>=0 and cost[1]<cost[0])
                groups[row['trial']][i][tuple(v for v in (e if win else g) if v>=0)]+=1
    stability={trial:[dict(modal_fraction=c.most_common(1)[0][1]/sum(c.values()),distinct=len(c),
        observations=sum(c.values()),paths=[dict(actions=k,count=v) for k,v in c.most_common()]) for c in rows]
        for trial,rows in groups.items()}
    result=dict(source_matched=True,helpers_matched=True,gate_trainings=30,retries=0,
        counts=s['counts'],runs=runs,zero_ownership_conflicts=a['ownership']['conflicts']==0,
        finite_chooser_logits=finite,kept_stability_by_compose_trial=stability)
    n=norms();result['start_norm_ranges']=n['rounds']['review20']['aggregate']
    prior=read(H/'review18-sweep/result.json');current=read(H/'review20-sweep/result.json')
    result['collection_delta']=dict(added=sorted(set(current['selected'])-set(prior['selected'])),removed=sorted(set(prior['selected'])-set(current['selected'])))
    write(H/'review20-results-validation.json',result)
    print(json.dumps(dict(counts=s['counts'],start_norm_ranges=result['start_norm_ranges'])))


if __name__=='__main__':main()
