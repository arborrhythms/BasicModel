"""Summarize saved observations only. Never constructs or trains a model."""
from collections import Counter, defaultdict
from pathlib import Path
import json, math, statistics
HERE=Path(__file__).resolve().parent

def read(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default

def write(p, value): p.write_text(json.dumps(value, indent=2)+'\n')
def finite(v): return isinstance(v,(int,float)) and math.isfinite(v)
def distribution(values):
    v=[x for x in values if finite(x)]
    return dict(n=len(v),min=min(v),median=statistics.median(v),mean=statistics.mean(v),max=max(v)) if v else dict(n=0)
def fmt(x):
    if x is None:return '—'
    if isinstance(x,float):return f'{x:.6g}'
    return str(x).replace('|','\\|').replace('\n',' ')
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(fmt(x)for x in row)+' |'for row in rows])+'\n'

def objective_summary(folder):
    log=folder/'events.jsonl'
    if not log.exists():return None
    events=[]
    for line in log.read_text().splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:pass
    terms=defaultdict(lambda:defaultdict(list)); selections=[]; rankings=defaultdict(Counter)
    for row in events:
        if row['kind']in ('trial','batch'):
            scope=('train'if row['train'] else 'evaluation')+'.'+row['kind']
            for name,t in row['terms'].items():
                values=terms[(scope,name)]
                for k in ('value','raw','baseline','weighted','active_entries'):
                    if finite(t.get(k)):values[k].append(t[k])
                for k in ('weight','context_weight','kind','objective','category','trained'):
                    values[k].append(t.get(k))
        if row['kind']=='selection':
            for i,active in enumerate(row['active']):
                if not active:continue
                chosen,other=('explore','exploit') if row['wins'][i] else ('exploit','explore')
                record=dict(pair=row['pair'],row=i,kept=chosen)
                for term in ('reconstruction','expectation','supplied_answer','total'):
                    a,b=row[chosen][term][i],row[other][term][i]
                    record[term]=None if a is None or b is None else dict(kept=a,other=b,difference=a-b)
                selections.append(record)
        if row['kind']=='activated_word_ranking':
            scope='train'if row['train']else'evaluation'
            for k in ('words','with_activated_candidate','activated_outranks_own'):
                rankings[scope][k]+=row[k]
    term_rows=[]
    for (scope,name),t in sorted(terms.items()):
        term_rows.append(dict(scope=scope,name=name,
            **{k:distribution(v)for k,v in t.items()if k in ('value','raw','baseline','weighted','active_entries')},
            **{k:sorted(set(v),key=str)for k,v in t.items()if k not in ('value','raw','baseline','weighted','active_entries')}))
    audit={}
    for term in ('reconstruction','expectation','supplied_answer','total'):
        valid=[r[term]['difference']for r in selections if r[term]is not None]
        worse=[x for x in valid if x>0]
        audit[term]=dict(compared=len(valid),kept_worse=len(worse),positive_difference=distribution(worse),all_difference=distribution(valid))
    explored=[r for r in selections if r['kept']=='explore']
    selection_rule=dict(explore_kept=len(explored),
        reconstruction_violations=sum(r['reconstruction']is not None and r['reconstruction']['difference']>0 for r in explored),
        total_violations=sum(r['total']is not None and r['total']['difference']>=0 for r in explored))
    ownership=read(folder/'ownership.json',{})
    owner_counts=defaultdict(lambda:Counter(parameters=0,elements=0,reached_parameters=0,reached_elements=0))
    for r in ownership.get('parameters',[]):
        c=owner_counts[r['owner']];c['parameters']+=1;c['elements']+=r['elements']
        if r['writers']:c['reached_parameters']+=1;c['reached_elements']+=r['elements']
    groups=[]
    for g in read(folder/'gradients.json',[]):
        groups.append({k:v for k,v in g.items()if k not in ('versions','per_parameter')})
    outcome=read(folder/'outcome.json',{})
    summary=dict(complete=read(folder/'complete.json'),ownership_counts=owner_counts,
        ownership_conflicts=ownership.get('conflicts'),backward_steps=ownership.get('backward_steps'),
        terms=term_rows,selection=audit,selection_rule=selection_rule,selection_rows=selections,activated_word_ranking=rankings,
        gradient_snapshots=groups,endpoint={k:v for k,v in outcome.items()if k in ('training_batches','evaluation_batches','last_training_trials','last_batch','receipt_added_endpoint')})
    write(folder/'summary.json',summary)
    rows=[]
    for r in term_rows:
        med=lambda k:r.get(k,{}).get('median')
        rows.append([r['scope'],r['name'],','.join(map(str,r['weight'])),','.join(map(str,r['context_weight'])),','.join(map(str,r['trained'])),med('raw'),med('baseline'),med('value'),med('weighted')])
    md='# Saved objective measurements\n\nComplete raw observations are in `events.jsonl`; values below are medians conditional on a term being recorded, not new forwards. Each sample count and range is in `summary.json`. Missing terms are not imputed as observations.\n\n'
    configured=read(folder/'configured-training.json')
    if configured is None:
        configured=read(folder/'plan.json',{}).get('effective',{}).get('architecture',{}).get('training',{})
    md+='## Configured priorities\n\nThese include disabled or unexercised objectives; the term table separately shows what was actually costed. Term purpose and ownership are specified in the main receipt and GradientFlow.\n\n'
    md+=table(['Configuration field','Value'],[[k,v]for k,v in configured.items()if any(x in k.lower()for x in ('weight','scale','loss','lambda','prediction','expectation','embedding'))])+'\n'
    md+='## Term magnitudes and weights\n\n'+table(['Scope','Term','Weight','Context','Trained','Raw','Baseline','Relative / penalty','Weighted'],rows)
    md+='\n## Actual optimizer ownership\n\n'+table(['Owner','Parameters','Elements','Reached parameters','Reached elements'],[[k,*[v[x]for x in ('parameters','elements','reached_parameters','reached_elements')]]for k,v in owner_counts.items()])
    md+='\nAn empty writer list is inactive, not a second owner. Full parameter names and writer lists are in `ownership.json`.\n\n'
    md+='## Selection audit\n\n'+table(['Objective','Comparisons','Kept worse','Mean excess when worse','Maximum excess'],[[k,v['compared'],v['kept_worse'],v['positive_difference'].get('mean'),v['positive_difference'].get('max')]for k,v in audit.items()])
    md+='\nThe rule constrains accepting explore; greedy can remain when explore improves only one side of the condition. Thus a kept-worse count alone is not a rule violation. Explore acceptance audit: '+str(selection_rule)+'.\n'
    md+='\n## Activated candidates\n\n'+table(['Scope','Own-word occurrences','With activated competitor','Activated outranks own'],[[k,*[v[x]for x in ('words','with_activated_candidate','activated_outranks_own')]]for k,v in rankings.items()])
    md+='\n## Same-state gradient snapshots\n\nNorms are restricted to permitted optimizer writers. `None` cosines mean a zero/absent vector, not an observed angle. These diagnostic groups can overlap: an answer-only `perceptualSpace.synthesis_layer` adapter appears under both the perception container and the generate/reader groups. That does not make an input-perception parameter answer-owned; the per-parameter names and actual owner are recorded separately.\n\n'
    for g in groups:
        md+=f'### {g["scope"]}, {g["trial"]}, pair {g["pair"]}\n\nParameter version digest: `{g["version_sha256"]}`. RNG unchanged: {g["rng_unchanged"]}.\n\n'
        rows=[]
        for name,values in g['groups'].items():
            rows.append([name,*[values['norms'].get(k)for k in ('reconstruction','expectation','supplied_answer')],*[values['cosines'].get(k)for k in ('reconstruction__expectation','reconstruction__supplied_answer','expectation__supplied_answer')]])
        md+=table(['Group','R norm','E norm','A norm','R:E cosine','R:A cosine','E:A cosine'],rows)+'\n'
    md+='## Endpoint costs\n\n`outcome.json` retains the last training trials and evaluation batch; `summary.json` preserves both. The answer-free total below omits the answer at the same recorded state; it is not a second training arm.\n\n'
    batches=[r for r in events if r['kind']=='batch']
    md+=table(['Phase','R','E','A','R+E (without A)','Trained total'],[[label,*[r['objective_costs'].get(k)for k in ('reconstruction','expectation','output')],r['without_answer'],r['raw']['totalLoss']]for label,selected in [('last training',[r for r in batches if r['train']]),('last evaluation',[r for r in batches if not r['train']])]for r in selected[-1:]])
    trials=[r for r in events if r['kind']=='trial']
    rows=[]
    for train in (True,False):
        for trial in ('exploit','explore'):
            selected=[r for r in trials if r['train']==train and r['trial']==trial]
            if not selected:continue
            r=selected[-1];cost={k:statistics.mean(v for v,a in zip(values,r['active'])if a and v is not None)if any(a and v is not None for v,a in zip(values,r['active']))else None for k,values in r['weighted'].items()}
            rows.append(['training'if train else'evaluation',trial,cost.get('reconstruction'),cost.get('expectation'),cost.get('supplied_answer'),cost.get('total'),(cost.get('reconstruction')or 0)+(cost.get('expectation')or 0),r.get('evaluation_supplied_answer')])
    md+='\nThe final column is the observer’s optional evaluation answer cost for each row, not an answer prediction. Training rows have no such extra evaluation read.\n\n'+table(['Phase','Trial','R','E','A','Selection total R+A','R+E without A','Observer evaluation answer costs per row'],rows)
    (folder/'summary.md').write_text(md)
    return summary

def gate_rows():
    rows=[]
    for p in sorted((HERE/'measurements').glob('gate-0[56]-trial-*/observations.jsonl')):
        for line in p.read_text().splitlines():
            r=json.loads(line)
            if r['kind']!='grammar':continue
            answers,targets=r['predictions'],r['targets'];y=dict(zip(r['inputs'],answers))
            mse=statistics.mean((a-b)**2 for a,b in zip(answers,targets));correct=sum((a>.5)==(b>.5)for a,b in zip(answers,targets))
            n=sum(Counter(a.split())==Counter((b or '').replace(chr(0),' ').split())for a,b in zip(r['inputs'],r['gate_reconstructions']))
            rows.append(dict(run=p.parent.name,kind='class'if r['gate']==5 else'reconstruction',answers=answers,targets=targets,inputs=r['inputs'],mse=mse,correct=correct,class_bar=correct==4 and mse<.05,read_backs=r['gate_reconstructions'],unavailable=r['grammar_reconstruction_unavailable'],reconstructed=n,reconstruction_bar=n==4 and not any(r['grammar_reconstruction_unavailable']),contrast=y['hello world']+y['loving there']-y['hello there']-y['loving world']))
    for p in sorted((HERE/'measurements').glob('sum-*/measurement.json')):rows.append(dict(read(p),run=p.parent.name))
    write(HERE/'gate-results.json',rows)
    md='# Closing gates: all saved runs\n\nOrder: hello world (0), hello there (1), loving world (1), loving there (0). All values come from the final evaluation already run by the gates.\n\n'
    md+=table(['Run','Answers','MSE','Correct','Read-backs','Words right / 4','Contrast','Class bar','Reconstruction bar','Sum bar'],[[r['run'],', '.join(fmt(x)for x in r['answers']),r['mse'],r['correct'],' / '.join(str(x)for x in r['read_backs']),r['reconstructed'],r['contrast'],r['class_bar'],r['reconstruction_bar'],r.get('sum_bar')]for r in rows])
    (HERE/'gate-results.md').write_text(md)
    return rows

if __name__=='__main__':
    for folder in (HERE/'xor-ownership',HERE/'native-stage1/BasicModel_answers_tied_benchmark-ownership'):
        if folder.exists():objective_summary(folder)
    gate_rows()
